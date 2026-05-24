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


# ---------------------------------------------------------------------------
# enable_thinking override
# ---------------------------------------------------------------------------

class TestEnableThinking:
    def _make_manager(self, model_name: str) -> InferenceManager:
        cfg = MagicMock()
        cfg.model_name = model_name
        m = InferenceManager(cfg)
        _attach_mock_model(m)
        return m

    def _captured_template_kwargs(self, manager: InferenceManager, enable_thinking: bool) -> dict:
        manager.generate([{"role": "user", "content": "hi"}], enable_thinking=enable_thinking)
        call_kwargs = manager._tokenizer.apply_chat_template.call_args.kwargs
        return call_kwargs

    def test_enable_thinking_true_passed_for_qwen3(self):
        m = self._make_manager("Qwen/Qwen3.5-2B")
        kwargs = self._captured_template_kwargs(m, enable_thinking=True)
        assert kwargs["enable_thinking"] is True

    def test_enable_thinking_false_passed_for_qwen3(self):
        m = self._make_manager("Qwen/Qwen3.5-2B")
        kwargs = self._captured_template_kwargs(m, enable_thinking=False)
        assert kwargs["enable_thinking"] is False

    def test_generate_base_always_uses_thinking_false_for_qwen3(self):
        m = self._make_manager("Qwen/Qwen3.5-2B")
        m.generate_base([{"role": "user", "content": "hi"}])
        kwargs = m._tokenizer.apply_chat_template.call_args.kwargs
        assert kwargs["enable_thinking"] is False

    def test_enable_thinking_not_injected_for_unknown_model(self):
        m = self._make_manager("meta-llama/Llama-3-8B")
        kwargs = self._captured_template_kwargs(m, enable_thinking=True)
        assert "enable_thinking" not in kwargs


# ---------------------------------------------------------------------------
# Sampling parameters wired to model.generate
# ---------------------------------------------------------------------------

class TestSamplingParams:
    def _make_manager(self) -> tuple:
        cfg = MagicMock()
        cfg.model_name = "test-model"
        m = InferenceManager(cfg)
        mock_model = _attach_mock_model(m)
        return m, mock_model

    def _gen_kwargs(self, manager, mock_model, **kwargs) -> dict:
        manager.generate([{"role": "user", "content": "hi"}], **kwargs)
        return mock_model.generate.call_args.kwargs

    def test_top_p_in_gen_kwargs(self):
        m, model = self._make_manager()
        kw = self._gen_kwargs(m, model, temperature=0.8, top_p=0.9)
        assert kw["top_p"] == 0.9

    def test_top_k_in_gen_kwargs(self):
        m, model = self._make_manager()
        kw = self._gen_kwargs(m, model, temperature=0.8, top_k=40)
        assert kw["top_k"] == 40

    def test_min_p_in_gen_kwargs(self):
        m, model = self._make_manager()
        kw = self._gen_kwargs(m, model, temperature=0.8, min_p=0.05)
        assert kw["min_p"] == 0.05

    def test_repetition_penalty_in_gen_kwargs(self):
        m, model = self._make_manager()
        kw = self._gen_kwargs(m, model, repetition_penalty=1.1)
        assert kw["repetition_penalty"] == 1.1

    def test_frequency_penalty_in_gen_kwargs(self):
        m, model = self._make_manager()
        kw = self._gen_kwargs(m, model, frequency_penalty=0.2)
        assert kw["frequency_penalty"] == 0.2

    def test_do_sample_true_when_top_p_set_without_temperature(self):
        m, model = self._make_manager()
        kw = self._gen_kwargs(m, model, top_p=0.9)
        assert kw["do_sample"] is True

    def test_none_params_absent_from_gen_kwargs(self):
        m, model = self._make_manager()
        kw = self._gen_kwargs(m, model)  # no sampling params
        for param in ("top_p", "top_k", "min_p", "repetition_penalty", "frequency_penalty"):
            assert param not in kw, f"{param} should not be in gen_kwargs when None"
