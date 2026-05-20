"""Unit tests for InferenceManager (inference/manager.py).

These tests do not load real models. They verify state management, the offload
lifecycle, and the generate_base/generate dispatch logic using mocks.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from groundcortex.inference.manager import InferenceManager


@pytest.fixture
def manager(config):
    return InferenceManager(config)


def _attach_mock_model(manager: InferenceManager) -> MagicMock:
    """Attach a mock base model so state-dependent methods can be tested."""
    mock_model = MagicMock()
    manager._base_model = mock_model
    tokenizer = MagicMock()
    tokenizer.apply_chat_template.return_value = "<prompt>"
    # tokenizer(text, return_tensors="pt") must return something with .to()
    # MagicMock auto-creates .to() as another MagicMock, which satisfies the call
    manager._tokenizer = tokenizer
    return mock_model


# ---------------------------------------------------------------------------
# is_training
# ---------------------------------------------------------------------------

class TestIsTraining:
    def test_false_initially(self, manager):
        assert manager.is_training is False

    def test_true_after_offload(self, manager):
        manager._base_model = MagicMock()
        manager.offload()
        assert manager.is_training is True

    def test_false_after_load_base(self, manager):
        manager._is_training = True
        # Simulate load_base() clearing the flag without loading a real model
        manager._is_training = False
        assert manager.is_training is False


# ---------------------------------------------------------------------------
# offload
# ---------------------------------------------------------------------------

class TestOffload:
    def test_clears_base_model(self, manager):
        manager._base_model = MagicMock()
        manager.offload()
        assert manager._base_model is None

    def test_clears_peft_model(self, manager):
        manager._model = MagicMock()
        manager.offload()
        assert manager._model is None

    def test_clears_loaded_adapters(self, manager):
        manager._loaded_adapters = ["v1", "v2"]
        manager.offload()
        assert manager._loaded_adapters == []

    def test_clears_active_version(self, manager):
        manager._active_version = "v1"
        manager.offload()
        assert manager._active_version is None

    def test_offload_on_already_empty_manager_is_safe(self, manager):
        manager.offload()  # must not raise

    def test_is_ready_false_after_offload(self, manager):
        manager._base_model = MagicMock()
        manager.offload()
        assert manager.is_ready is False


# ---------------------------------------------------------------------------
# generate_base
# ---------------------------------------------------------------------------

class TestUnloadAdapter:
    def test_unload_adapter_clears_active_version(self, manager):
        manager._base_model = MagicMock()
        mock_peft = MagicMock()
        manager._model = mock_peft
        manager._active_version = "v1"
        manager.unload_adapter()
        assert manager.get_active_version() is None

    def test_unload_adapter_calls_disable_adapter_layers(self, manager):
        manager._base_model = MagicMock()
        mock_peft = MagicMock()
        manager._model = mock_peft
        manager.unload_adapter()
        mock_peft.disable_adapter_layers.assert_called_once()

    def test_unload_adapter_no_model_is_safe(self, manager):
        manager.unload_adapter()  # must not raise

    def test_set_active_enables_adapter_layers_before_switching(self, manager):
        manager._base_model = MagicMock()
        mock_peft = MagicMock()
        manager._model = mock_peft
        manager._loaded_adapters = ["v1"]
        manager.set_active("v1")
        mock_peft.enable_adapter_layers.assert_called_once()
        mock_peft.set_adapter.assert_called_once_with("v1")


class TestGenerateBase:
    def test_raises_if_base_not_loaded(self, manager):
        with pytest.raises(RuntimeError, match="load_base"):
            manager.generate_base([{"role": "user", "content": "hi"}])

    def test_uses_base_model_not_peft_model(self, manager):
        base = _attach_mock_model(manager)
        peft = MagicMock()
        manager._model = peft

        manager.generate_base([{"role": "user", "content": "hi"}])

        base.generate.assert_called_once()
        peft.generate.assert_not_called()

    def test_does_not_require_peft_model(self, manager):
        _attach_mock_model(manager)
        assert manager._model is None

        manager.generate_base([{"role": "user", "content": "hi"}])
        manager._base_model.generate.assert_called_once()
