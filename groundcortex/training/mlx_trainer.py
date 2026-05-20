"""MLX-based LoRA trainer for Apple Silicon (macOS).

Auto-selected by create_trainer() when use_qlora=True on macOS.
See groundcortex/MLX_NOTE.md for removal instructions.
"""
from __future__ import annotations

import math
import os
import types
from datetime import datetime, timezone

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
        model, tokenizer = mlx_lm.load(cfg.model_name)
        model, _ = quantize_model(model, config={}, group_size=64, bits=4)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        adapter_dir = str(cfg.output_dir / f"{version}_{timestamp}")
        os.makedirs(adapter_dir, exist_ok=True)

        iters = math.ceil(len(dataset) / cfg.batch_size) * cfg.epochs
        args = types.SimpleNamespace(
            fine_tune_type="lora",
            num_layers=len(model.layers),
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
            grad_checkpoint=False,
            grad_accumulation_steps=1,
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
        }
