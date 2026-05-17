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


def _patch_chat_template_for_generation(tokenizer) -> None:
    """Add {% generation %} tags required by TRL 0.24 assistant_only_loss.

    Copied verbatim from hypothesis.py. See that file for the full explanation
    of why this patch is required and what breaks without it.
    """
    old = (
        '{%- if (message.role == "user") or (message.role == "system" and not loop.first)'
        ' or (message.role == "assistant" and not message.tool_calls) %}\n'
        "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}"
    )
    new = (
        '{%- if (message.role == "user") or (message.role == "system" and not loop.first) %}\n'
        "        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}\n"
        "    {%- elif message.role == \"assistant\" and not message.tool_calls %}\n"
        "        {{- '<|im_start|>' + message.role + '\\n' }}"
        "{% generation %}{{- message.content + '<|im_end|>' + '\\n' }}{% endgeneration %}"
    )
    if "{% generation %}" not in tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(old, new)


def _load_model(model_name: str):
    """Load base model + tokenizer. Preserved from hypothesis.py."""
    device = _get_device()
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
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

    _patch_chat_template_for_generation(tokenizer)
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
        model, tokenizer = _load_model(cfg.model_name)

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

        use_fp16 = self._device == "cuda"
        optim = "adamw_8bit" if self._device == "cuda" else "adamw_torch"

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
        }
