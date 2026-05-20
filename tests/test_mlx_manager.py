"""Tests for MLXInferenceManager and the create_manager() factory.

All tests are guarded with pytest.importorskip("mlx_lm") so they are skipped
when mlx-lm is not installed (non-Mac environments, CI without the .[mlx] extra).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

mlx_lm = pytest.importorskip("mlx_lm")

from groundcortex.inference.manager import InferenceManager, create_manager
from groundcortex.inference.mlx_manager import MLXInferenceManager


# ──────────────────────────────────────────────────────────────────────────────
# Factory routing
# ──────────────────────────────────────────────────────────────────────────────


def test_create_manager_returns_mlx_on_mac_qlora(config):
    config.use_qlora = True
    with patch("platform.system", return_value="Darwin"):
        mgr = create_manager(config)
    assert isinstance(mgr, MLXInferenceManager)


def test_create_manager_returns_inference_manager_non_mac(config):
    config.use_qlora = True
    with patch("platform.system", return_value="Linux"):
        mgr = create_manager(config)
    assert isinstance(mgr, InferenceManager)


def test_create_manager_returns_inference_manager_no_qlora(config):
    config.use_qlora = False
    with patch("platform.system", return_value="Darwin"):
        mgr = create_manager(config)
    assert isinstance(mgr, InferenceManager)


# ──────────────────────────────────────────────────────────────────────────────
# Initial state
# ──────────────────────────────────────────────────────────────────────────────


def test_is_ready_false_before_load(config):
    mgr = MLXInferenceManager(config)
    assert mgr.is_ready is False


def test_is_training_false_initially(config):
    mgr = MLXInferenceManager(config)
    assert mgr.is_training is False


def test_list_loaded_adapters_empty_initially(config):
    mgr = MLXInferenceManager(config)
    assert mgr.list_loaded_adapters() == []


def test_get_active_version_none_initially(config):
    mgr = MLXInferenceManager(config)
    assert mgr.get_active_version() is None


# ──────────────────────────────────────────────────────────────────────────────
# load_base
# ──────────────────────────────────────────────────────────────────────────────


def test_load_base_calls_mlx_load(config):
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mgr = MLXInferenceManager(config)

    with (
        patch("mlx_lm.load", return_value=(mock_model, mock_tokenizer)) as mock_load,
        patch("mlx_lm.utils.quantize_model", return_value=(mock_model, {})),
    ):
        mgr.load_base()

    mock_load.assert_called_once_with(config.model_name)
    assert mgr.is_ready is True
    assert mgr.is_training is False


# ──────────────────────────────────────────────────────────────────────────────
# load_adapter / set_active
# ──────────────────────────────────────────────────────────────────────────────


def _attach_mock_model(mgr: MLXInferenceManager):
    mgr._model = MagicMock()
    mgr._tokenizer = MagicMock()


def test_load_adapter_stores_path_and_calls_load_adapters(config):
    mgr = MLXInferenceManager(config)
    _attach_mock_model(mgr)

    with patch("mlx_lm.tuner.utils.load_adapters") as mock_la:
        mgr.load_adapter("/tmp/v1", "v1")

    mock_la.assert_called_once_with(mgr._model, "/tmp/v1")
    assert mgr._adapter_paths["v1"] == "/tmp/v1"
    assert mgr._lora_applied is True
    assert mgr.get_active_version() == "v1"
    assert "v1" in mgr.list_loaded_adapters()


def test_set_active_calls_load_adapters_with_correct_path(config):
    mgr = MLXInferenceManager(config)
    _attach_mock_model(mgr)
    mgr._adapter_paths = {"v1": "/tmp/v1", "v2": "/tmp/v2"}
    mgr._lora_applied = True

    with patch("mlx_lm.tuner.utils.load_adapters") as mock_la:
        mgr.set_active("v2")

    mock_la.assert_called_once_with(mgr._model, "/tmp/v2")
    assert mgr.get_active_version() == "v2"


def test_set_active_unknown_version_raises(config):
    mgr = MLXInferenceManager(config)
    _attach_mock_model(mgr)
    with pytest.raises(ValueError, match="not loaded"):
        mgr.set_active("nonexistent")


# ──────────────────────────────────────────────────────────────────────────────
# generate
# ──────────────────────────────────────────────────────────────────────────────


def test_generate_calls_mlx_generate(config):
    mgr = MLXInferenceManager(config)
    _attach_mock_model(mgr)
    mgr._tokenizer.apply_chat_template.return_value = "<prompt>"

    with patch("mlx_lm.generate", return_value="hello") as mock_gen:
        result = mgr.generate([{"role": "user", "content": "hi"}])

    mock_gen.assert_called_once()
    assert result == "hello"


def test_generate_raises_without_model(config):
    mgr = MLXInferenceManager(config)
    with pytest.raises(RuntimeError, match="load_base"):
        mgr.generate([{"role": "user", "content": "hi"}])


# ──────────────────────────────────────────────────────────────────────────────
# offload
# ──────────────────────────────────────────────────────────────────────────────


def test_offload_clears_state(config):
    mgr = MLXInferenceManager(config)
    _attach_mock_model(mgr)
    mgr._adapter_paths = {"v1": "/tmp/v1"}
    mgr._active_version = "v1"
    mgr._lora_applied = True

    mock_mx = MagicMock()
    with patch.dict("sys.modules", {"mlx": MagicMock(), "mlx.core": mock_mx}):
        mgr.offload()

    assert mgr._model is None
    assert mgr._tokenizer is None
    assert mgr._adapter_paths == {}
    assert mgr._active_version is None
    assert mgr._lora_applied is False
    assert mgr.is_training is True
    assert mgr.is_ready is False
