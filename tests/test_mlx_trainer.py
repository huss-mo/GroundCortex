"""Tests for MLXTrainer and the create_trainer() factory.

All tests are guarded with pytest.importorskip("mlx_lm") so they are skipped
when mlx-lm is not installed (non-Mac environments, CI without the .[mlx] extra).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

mlx_lm = pytest.importorskip("mlx_lm")

from groundcortex.training.mlx_trainer import MLXTrainer
from groundcortex.training.trainer import LoRATrainer, create_trainer


# ──────────────────────────────────────────────────────────────────────────────
# Factory routing
# ──────────────────────────────────────────────────────────────────────────────


def test_create_trainer_returns_mlx_on_mac_qlora(config):
    config.use_qlora = True
    with patch("platform.system", return_value="Darwin"):
        trainer = create_trainer(config)
    assert isinstance(trainer, MLXTrainer)


def test_create_trainer_returns_lora_non_mac(config):
    config.use_qlora = True
    with patch("platform.system", return_value="Linux"):
        trainer = create_trainer(config)
    assert isinstance(trainer, LoRATrainer)


def test_create_trainer_returns_lora_no_qlora(config):
    config.use_qlora = False
    with patch("platform.system", return_value="Darwin"):
        trainer = create_trainer(config)
    assert isinstance(trainer, LoRATrainer)


# ──────────────────────────────────────────────────────────────────────────────
# MLXTrainer.hyperparams_snapshot
# ──────────────────────────────────────────────────────────────────────────────


def test_hyperparams_snapshot_contains_mlx_fields(config):
    config.use_qlora = True
    trainer = MLXTrainer(config)
    snap = trainer.hyperparams_snapshot()
    assert snap["backend"] == "mlx"
    assert snap["bits"] == 4
    assert snap["use_qlora"] is True
    assert snap["device"] == "mps"


# ──────────────────────────────────────────────────────────────────────────────
# MLXTrainer.train
# ──────────────────────────────────────────────────────────────────────────────


@patch("groundcortex.training.mlx_trainer.MLXTrainer.train")
def test_train_returns_adapter_dir(mock_train, config):
    """Smoke test: train() returns a string path."""
    mock_train.return_value = "/tmp/v1_20260101T000000"
    config.use_qlora = True
    trainer = MLXTrainer(config)
    from datasets import Dataset

    ds = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hey"}]}]
        * 2
    )
    path = trainer.train(ds, "v1")
    assert isinstance(path, str)
    mock_train.assert_called_once()


def test_train_calls_train_model(config, tmp_path):
    """train() should call mlx_lm.lora.train_model with the right iters."""
    import math

    config.use_qlora = True
    config.batch_size = 2
    config.epochs = 3
    config.rank = 8
    config.alpha = 16

    from datasets import Dataset

    ds = Dataset.from_list(
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        * 6
    )
    expected_iters = math.ceil(len(ds) / config.batch_size) * config.epochs  # ceil(6/2)*3 = 9

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_train_set = MagicMock()

    with (
        patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)),
        patch("mlx_lm.utils.quantize_model", return_value=(mock_model, {})),
        patch("mlx_lm.lora.train_model") as mock_train_model,
        patch("mlx_lm.tuner.datasets.ChatDataset", return_value=mock_train_set),
    ):
        trainer = MLXTrainer(config)
        path = trainer.train(ds, "smoke")

    mock_train_model.assert_called_once()
    call_args = mock_train_model.call_args
    args_ns = call_args[0][0]  # first positional arg is the SimpleNamespace
    assert args_ns.iters == expected_iters
    assert args_ns.fine_tune_type == "lora"
    assert args_ns.adapter_path == path
    assert args_ns.lora_parameters["rank"] == config.rank
