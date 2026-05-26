from __future__ import annotations

import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from groundcortex.config import GroundCortexConfig
from groundcortex.model_registry import find_lora_targets, patch_chat_template_for_trl


# ──────────────────────────────────────────────────────────────────────────────
# Device helpers (preserved verbatim from hypothesis.py)
# ──────────────────────────────────────────────────────────────────────────────

def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


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

    patch_chat_template_for_trl(tokenizer, model_name)
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

        layers_to_transform = None
        if cfg.num_lora_layers[0] > 0:
            total = model.config.num_hidden_layers
            n = min(cfg.num_lora_layers[0], total)
            layers_to_transform = list(range(total - n, total))

        lora_config = LoraConfig(
            r=cfg.rank[0],
            lora_alpha=cfg.alpha[0],
            target_modules=find_lora_targets(model),
            layers_to_transform=layers_to_transform,
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

        adapter_dir = str(cfg.output_dir / version)
        os.makedirs(adapter_dir, exist_ok=True)

        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            args=SFTConfig(
                assistant_only_loss=True,
                max_length=512,
                per_device_train_batch_size=cfg.batch_size[0],
                gradient_accumulation_steps=cfg.gradient_accumulation[0],
                warmup_steps=10,
                num_train_epochs=cfg.epochs[0],
                learning_rate=cfg.learning_rate[0],
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

        # Explicitly free training model so GPU/MPS memory is released before
        # load_base() loads the inference model for evaluation.
        import gc
        import torch
        del model, tokenizer, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        return adapter_dir

    def hyperparams_snapshot(self) -> dict:
        cfg = self._config
        return {
            "model_name": cfg.model_name,
            "rank": cfg.rank[0],
            "alpha": cfg.alpha[0],
            "learning_rate": cfg.learning_rate[0],
            "epochs": cfg.epochs[0],
            "batch_size": cfg.batch_size[0],
            "device": self._device,
            "use_qlora": cfg.use_qlora,
            "num_lora_layers": cfg.num_lora_layers[0],
            "gradient_accumulation": cfg.gradient_accumulation[0],
        }


def create_trainer(config: GroundCortexConfig):
    """Return MLXTrainer on macOS + use_qlora=True, LoRATrainer otherwise."""
    import platform
    if config.use_qlora and platform.system() == "Darwin":
        from groundcortex.training.mlx_trainer import MLXTrainer
        return MLXTrainer(config)
    return LoRATrainer(config)
