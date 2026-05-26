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


def _mlx_train_subprocess(train_args: dict, dataset_records: list) -> None:
    """Training worker executed in a subprocess.

    Running in a subprocess guarantees that all Metal buffers (model weights,
    optimizer state, gradient buffers) are released at OS level when this
    function returns and the process exits. del + gc.collect() inside the
    parent process is insufficient because MLX's C++ side may retain
    references that Python GC cannot reach.
    """
    import mlx_lm
    import mlx.nn as nn
    import types as _types
    from mlx_lm.lora import train_model as _mlx_train_model
    from mlx_lm.tuner.datasets import ChatDataset
    from mlx_lm.utils import quantize_model

    model_name = train_args["model_name"]
    adapter_dir = train_args["adapter_dir"]

    model, tokenizer = mlx_lm.load(model_name)
    already_quantized = any(
        isinstance(m, nn.QuantizedLinear) for _, m in model.named_modules()
    )
    if not already_quantized:
        model, _ = quantize_model(model, config={}, group_size=64, bits=4)

    n_lora_layers = train_args["num_lora_layers"]
    if n_lora_layers == 0:
        n_lora_layers = len(model.layers)
    n_lora_layers = min(n_lora_layers, len(model.layers))

    args = _types.SimpleNamespace(
        fine_tune_type="lora",
        num_layers=n_lora_layers,
        lora_parameters={
            "rank": train_args["rank"],
            "dropout": 0.1,
            "scale": float(train_args["alpha"]) / train_args["rank"],
        },
        optimizer="adamw",
        optimizer_config={},
        lr_schedule=None,
        learning_rate=train_args["learning_rate"],
        batch_size=train_args["batch_size"],
        iters=train_args["iters"],
        val_batches=0,
        test_batches=0,
        steps_per_report=10,
        steps_per_eval=0,
        save_every=train_args["iters"] + 1,
        adapter_path=adapter_dir,
        resume_adapter_file=None,
        max_seq_length=512,
        grad_checkpoint=True,
        grad_accumulation_steps=train_args["gradient_accumulation"],
        seed=0,
        report_to=None,
        project_name="",
    )

    train_set = ChatDataset(dataset_records, tokenizer, mask_prompt=True)
    empty_set = ChatDataset([], tokenizer)
    _mlx_train_model(args, model, train_set, valid_set=empty_set)


class MLXTrainer:
    """4-bit QLoRA trainer backed by mlx-lm for Apple Silicon.

    Public interface matches LoRATrainer so the two are interchangeable
    via create_trainer(). A new instance should be created for each run.
    """

    def __init__(self, config: GroundCortexConfig) -> None:
        self._config = config

    def train(self, dataset: Dataset, version: str) -> str:
        """Fine-tune a 4-bit quantized LoRA adapter and return its path."""
        import multiprocessing as mp
        import mlx.core as mx

        cfg = self._config
        adapter_dir = str(cfg.output_dir / version)
        os.makedirs(adapter_dir, exist_ok=True)

        iters = math.ceil(len(dataset) / cfg.batch_size[0]) * cfg.epochs[0]

        train_args = {
            "model_name": cfg.model_name,
            "adapter_dir": adapter_dir,
            "rank": cfg.rank[0],
            "alpha": cfg.alpha[0],
            "learning_rate": cfg.learning_rate[0],
            "batch_size": cfg.batch_size[0],
            "iters": iters,
            "gradient_accumulation": cfg.gradient_accumulation[0],
            "num_lora_layers": cfg.num_lora_layers[0],
        }

        # Run training in a subprocess. When the subprocess exits, the OS
        # unconditionally reclaims all Metal buffers it allocated. This is
        # the only reliable way to free them before load_base() runs for eval.
        ctx = mp.get_context("spawn")
        proc = ctx.Process(
            target=_mlx_train_subprocess,
            args=(train_args, list(dataset)),
        )
        proc.start()
        proc.join()

        if proc.exitcode != 0:
            raise RuntimeError(
                f"MLX training subprocess exited with code {proc.exitcode}"
            )

        # Belt-and-suspenders: clear the parent's MLX cache too.
        mx.clear_cache()

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
            "device": "mps",
            "use_qlora": True,
            "backend": "mlx",
            "bits": 4,
            "num_lora_layers": cfg.num_lora_layers[0],
            "gradient_accumulation": cfg.gradient_accumulation[0],
        }
