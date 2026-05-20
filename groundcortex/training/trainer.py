from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from groundcortex.config import GroundCortexConfig


# ──────────────────────────────────────────────────────────────────────────────
# Device helpers (preserved verbatim from hypothesis.py)
# ──────────────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _patch_qwen3_chat_template_for_trl(tokenizer) -> None:
    """Patch the Qwen3/3.5 chat template to add {% generation %} / {% endgeneration %}
    tags required by TRL 0.24's assistant_only_loss.

    Qwen3/3.5 uses a multimodal template (render_content macro, image/video handling,
    tool calls) with a bifurcated assistant output path - one branch prepends a
    <think> block, the other emits content directly. TRL's auto-patcher only handles
    simple single-path templates (Llama, Mistral, Qwen2.5, etc.) and silently skips
    this template, causing assistant_only_loss to train on all tokens including user
    prompts. This function injects the markers manually.

    The patch is a no-op if markers already exist or the target patterns are not
    found (i.e. not a Qwen3/3.5 model, or TRL already handled it).
    """
    if "{% generation %}" in tokenizer.chat_template:
        return

    # Replace the entire if/else block that outputs prefix+content.
    # Jinja2 requires {% generation %} to be properly nested - opening it inside
    # an if branch and closing it outside raises TemplateSyntaxError. Fix:
    # output only the prefix inside the if/else, then open {% generation %}
    # after the endif so the block spans content through <|im_end|>.
    old = (
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "        {%- endif %}"
    )
    new = (
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' }}\n"
        "        {%- endif %}\n"
        "        {% generation %}{{- content }}"
    )
    tokenizer.chat_template = tokenizer.chat_template.replace(old, new)

    # Close the generation region at end of assistant turn.
    # Anchored to the succeeding elif to avoid matching the 12-space-indented
    # im_end lines inside the tool-role block (8 spaces is a substring of 12).
    tokenizer.chat_template = tokenizer.chat_template.replace(
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
        "        {{- '<|im_end|>\\n' }}{% endgeneration %}\n    {%- elif message.role == \"tool\" %}",
    )



def _load_model(model_name: str, use_qlora: bool = False):
    """Load base model + tokenizer with device-appropriate precision.

    Standard path (use_qlora=False): fp16 on CUDA/MPS, fp32 on CPU.

    QLoRA path (use_qlora=True):
      CUDA - int4 via torchao Int4WeightOnlyConfig (tinygemm CUDA kernels),
             device_map="auto" for multi-GPU distribution.
      MPS/CPU - fp16 fallback; torchao's AffineQuantizedTensor (PlainLayout)
             has no MPS dispatch for the linear op, producing garbage logits.
             Gradient checkpointing is enabled regardless.

    Tokenizer alignment: pad_token set to eos_token (Qwen has no default pad);
    generation_config temperature/top_p/top_k cleared for greedy-decoding callers.
    """
    device = _get_device()

    if use_qlora:
        if device == "cuda":
            from transformers import TorchAoConfig
            from torchao.quantization import Int4WeightOnlyConfig
            torchao_config = TorchAoConfig(Int4WeightOnlyConfig(group_size=128))
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=torchao_config,
                dtype=torch.float16,
                device_map="auto",
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, dtype=torch.float16
            )
            model = model.to(device)

        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        model = model.to(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    _patch_qwen3_chat_template_for_trl(tokenizer)
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────────
# LoRATrainer
# ──────────────────────────────────────────────────────────────────────────────

class LoRATrainer:
    """Wraps the training logic from hypothesis.py into a reusable class.

    A new instance should be created for each training run. It loads the base
    model fresh each time - LoRAs are never stacked on each other.
    """

    def __init__(self, config: GroundCortexConfig) -> None:
        self._config = config
        self._device = _get_device()

    def train(self, dataset: Dataset, version: str) -> str:
        """Train a LoRA adapter on `dataset` and return the adapter path."""
        cfg = self._config
        model, tokenizer = _load_model(cfg.model_name, use_qlora=cfg.use_qlora)

        lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        # fp16 enabled on CUDA only for standard LoRA. Disabled when use_qlora=True
        # because torchao handles compute dtype internally on CUDA; enabling fp16
        # on top of int4 quantization causes a dtype conflict. On MPS, use_qlora
        # already falls back to fp16 loading so this CUDA-only gate is correct.
        use_fp16 = self._device == "cuda" and not cfg.use_qlora
        # bitsandbytes is not installed (replaced by torchao); use adamw_torch on all devices.
        optim = "adamw_torch"

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        adapter_dir = str(cfg.output_dir / f"{version}_{timestamp}")
        os.makedirs(adapter_dir, exist_ok=True)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                assistant_only_loss=True,
                max_length=512,
                per_device_train_batch_size=cfg.batch_size,
                gradient_accumulation_steps=2,
                warmup_steps=10,
                num_train_epochs=cfg.epochs,
                learning_rate=cfg.learning_rate,
                fp16=use_fp16,
                logging_steps=10,
                eval_strategy="no",
                output_dir=adapter_dir,
                report_to="none",
                optim=optim,
                dataloader_pin_memory=False,
            ),
        )

        trainer.train()

        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)

        return adapter_dir

    def hyperparams_snapshot(self) -> dict:
        cfg = self._config
        return {
            "model_name": cfg.model_name,
            "rank": cfg.rank,
            "alpha": cfg.alpha,
            "learning_rate": cfg.learning_rate,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "device": self._device,
            "use_qlora": cfg.use_qlora,
        }
