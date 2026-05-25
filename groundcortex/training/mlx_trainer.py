"""MLX-based LoRA trainer for Apple Silicon (macOS).

Auto-selected by create_trainer() when use_qlora=True on macOS.
See groundcortex/MLX_NOTE.md for removal instructions.
"""
from __future__ import annotations

import math
import os
import types
from datasets import Dataset

from groundcortex.config import GroundCortexConfig


class MLXTrainer:
    """4-bit QLoRA trainer backed by mlx-lm for Apple Silicon.

    Public interface matches LoRATrainer so the two are interchangeable
    via create_trainer(). A new instance should be created for each run.
    """

    def __init__(self, config: GroundCortexConfig) -> None:
        self._config = config

    def train(self, dataset: Dataset, version: str) -> str:
        """Fine-tune a 4-bit quantized LoRA adapter and return its path."""
        import mlx_lm
        from mlx_lm.lora import train_model as _mlx_train_model
        from mlx_lm.tuner.datasets import ChatDataset
        from mlx_lm.utils import quantize_model

        cfg = self._config
        import mlx.nn as nn
        model, tokenizer = mlx_lm.load(cfg.model_name)
        already_quantized = any(
            isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules()
        )
        if not already_quantized:
            model, _ = quantize_model(model, config={}, group_size=64, bits=4)

        adapter_dir = str(cfg.output_dir / version)
        os.makedirs(adapter_dir, exist_ok=True)

        iters = math.ceil(len(dataset) / cfg.batch_size) * cfg.epochs

        # Cap LoRA to the top N layers. On large MoE models (e.g. 35B with 64
        # experts per layer), num_layers=all creates O(n_experts × rank) trainable
        # params per layer. Adam stores 2 momentum tensors per param, so a 40-layer
        # MoE at rank=16 with all layers active exceeds 48GB unified memory during
        # optimizer init. 8–16 layers keeps peak Metal usage under ~24GB.
        # cfg.num_lora_layers == 0 means all layers (no cap).
        n_lora_layers = cfg.num_lora_layers if cfg.num_lora_layers > 0 else len(model.layers)
        n_lora_layers = min(n_lora_layers, len(model.layers))

        args = types.SimpleNamespace(
            fine_tune_type="lora",
            num_layers=n_lora_layers,
            lora_parameters={
                "rank": cfg.rank,
                "dropout": 0.1,
                "scale": float(cfg.alpha) / cfg.rank,
            },
            optimizer="adamw",
            optimizer_config={},
            lr_schedule=None,
            learning_rate=cfg.learning_rate,
            batch_size=cfg.batch_size,
            iters=iters,
            val_batches=0,
            test_batches=0,
            steps_per_report=10,
            steps_per_eval=0,
            save_every=iters + 1,
            adapter_path=adapter_dir,
            resume_adapter_file=None,
            max_seq_length=512,
            grad_checkpoint=True,
            grad_accumulation_steps=cfg.gradient_accumulation,
            seed=0,
            report_to=None,
            project_name="",
        )

        train_set = ChatDataset(list(dataset), tokenizer, mask_prompt=True)
        empty_set = ChatDataset([], tokenizer)
        _mlx_train_model(args, model, train_set, valid_set=empty_set)

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
            "device": "mps",
            "use_qlora": True,
            "backend": "mlx",
            "bits": 4,
            "num_lora_layers": cfg.num_lora_layers,
            "gradient_accumulation": cfg.gradient_accumulation,
        }
