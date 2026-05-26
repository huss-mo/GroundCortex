"""Tests for LoRATrainer and create_trainer() factory (PEFT/TRL path)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from datasets import Dataset

from groundcortex.training.trainer import LoRATrainer, create_trainer


def _mini_dataset() -> Dataset:
    return Dataset.from_list(
        [{"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}]
        * 2
    )


def _mock_model(num_hidden_layers: int = 10) -> MagicMock:
    m = MagicMock()
    m.config.num_hidden_layers = num_hidden_layers
    return m


def _run_train(config, num_hidden_layers: int = 10):
    """Run LoRATrainer.train() with all external dependencies mocked.

    Returns (lora_config_kwargs, sft_config_kwargs) captured from the calls.
    """
    mock_model = _mock_model(num_hidden_layers)
    mock_tokenizer = MagicMock()
    lora_kwargs: dict = {}
    sft_kwargs: dict = {}

    def capture_lora(**kwargs):
        lora_kwargs.update(kwargs)
        return MagicMock()

    def capture_sft(**kwargs):
        sft_kwargs.update(kwargs)
        return MagicMock()

    with (
        patch("groundcortex.training.trainer._load_model", return_value=(mock_model, mock_tokenizer)),
        patch("groundcortex.training.trainer.LoraConfig", side_effect=capture_lora),
        patch("groundcortex.training.trainer.get_peft_model", return_value=mock_model),
        patch("groundcortex.training.trainer.SFTConfig", side_effect=capture_sft),
        patch("groundcortex.training.trainer.SFTTrainer") as mock_sft,
    ):
        mock_sft.return_value.train = MagicMock()
        LoRATrainer(config).train(_mini_dataset(), "v1")

    return lora_kwargs, sft_kwargs


# ──────────────────────────────────────────────────────────────────────────────
# Factory routing
# ──────────────────────────────────────────────────────────────────────────────


def test_create_trainer_returns_lora_on_linux(config):
    config.use_qlora = False
    with patch("platform.system", return_value="Linux"):
        assert isinstance(create_trainer(config), LoRATrainer)


def test_create_trainer_returns_lora_when_not_qlora_on_mac(config):
    config.use_qlora = False
    with patch("platform.system", return_value="Darwin"):
        assert isinstance(create_trainer(config), LoRATrainer)


# ──────────────────────────────────────────────────────────────────────────────
# num_lora_layers → layers_to_transform
# ──────────────────────────────────────────────────────────────────────────────


def test_num_lora_layers_zero_passes_none(config):
    config.num_lora_layers = [0]
    lora_kwargs, _ = _run_train(config, num_hidden_layers=10)
    assert lora_kwargs["layers_to_transform"] is None


def test_num_lora_layers_cap_selects_top_n(config):
    config.num_lora_layers = [4]
    lora_kwargs, _ = _run_train(config, num_hidden_layers=10)
    assert lora_kwargs["layers_to_transform"] == [6, 7, 8, 9]


def test_num_lora_layers_clamped_to_model_total(config):
    config.num_lora_layers = [20]
    lora_kwargs, _ = _run_train(config, num_hidden_layers=10)
    assert lora_kwargs["layers_to_transform"] == list(range(10))


def test_num_lora_layers_equal_to_total(config):
    config.num_lora_layers = [10]
    lora_kwargs, _ = _run_train(config, num_hidden_layers=10)
    assert lora_kwargs["layers_to_transform"] == list(range(10))


# ──────────────────────────────────────────────────────────────────────────────
# gradient_accumulation → SFTConfig
# ──────────────────────────────────────────────────────────────────────────────


def test_gradient_accumulation_passed_to_sft_config(config):
    config.gradient_accumulation = [4]
    _, sft_kwargs = _run_train(config)
    assert sft_kwargs["gradient_accumulation_steps"] == 4


def test_gradient_accumulation_default_is_two(config):
    _, sft_kwargs = _run_train(config)
    assert sft_kwargs["gradient_accumulation_steps"] == 2


# ──────────────────────────────────────────────────────────────────────────────
# hyperparams_snapshot
# ──────────────────────────────────────────────────────────────────────────────


def test_hyperparams_snapshot_includes_num_lora_layers(config):
    config.num_lora_layers = [8]
    assert LoRATrainer(config).hyperparams_snapshot()["num_lora_layers"] == 8


def test_hyperparams_snapshot_includes_gradient_accumulation(config):
    config.gradient_accumulation = [4]
    assert LoRATrainer(config).hyperparams_snapshot()["gradient_accumulation"] == 4
