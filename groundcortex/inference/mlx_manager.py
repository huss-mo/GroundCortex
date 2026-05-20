"""MLX-based inference manager for Apple Silicon (macOS).

Auto-selected by create_manager() when use_qlora=True on macOS.
See groundcortex/MLX_NOTE.md for removal instructions.
"""
from __future__ import annotations

import logging

from groundcortex.config import GroundCortexConfig

logger = logging.getLogger(__name__)


def _iter_lora_layers(model):
    """Yield all LoRALinear modules in the model."""
    from mlx_lm.tuner.lora import LoRALinear
    for _, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module


class MLXInferenceManager:
    """Inference manager backed by mlx-lm for Apple Silicon.

    Public interface matches InferenceManager so the two are interchangeable
    via create_manager(). Loads a 4-bit quantized base model, then hot-swaps
    LoRA adapters by calling load_adapters() with the new path - no model
    reload required between switches.
    """

    def __init__(self, config: GroundCortexConfig) -> None:
        self._config = config
        self._model = None
        self._tokenizer = None
        self._adapter_paths: dict[str, str] = {}
        self._lora_applied: bool = False
        self._active_version: str | None = None
        self._is_training: bool = False

    def load_base(self) -> None:
        """Load and quantize the base model. Call once at startup."""
        import mlx_lm
        from mlx_lm.utils import quantize_model

        self._is_training = False
        cfg = self._config
        logger.info("Loading base model (MLX 4-bit): %s", cfg.model_name)
        model, tokenizer = mlx_lm.load(cfg.model_name)
        self._model, _ = quantize_model(model, config={}, group_size=64, bits=4)
        self._tokenizer = tokenizer
        self._lora_applied = False
        logger.info("Base model loaded (MLX).")

    def load_adapter(self, adapter_path: str, version_id: str) -> None:
        """Load a LoRA adapter and register it under version_id."""
        if self._model is None:
            raise RuntimeError("Call load_base() before load_adapter().")

        from mlx_lm.tuner.utils import load_adapters

        logger.info("Loading MLX adapter %s from %s", version_id, adapter_path)
        load_adapters(self._model, adapter_path)
        self._lora_applied = True
        self._adapter_paths[version_id] = adapter_path
        self._active_version = version_id
        logger.info("MLX adapter %s loaded.", version_id)

    def set_active(self, version_id: str) -> None:
        """Hot-swap to a previously loaded adapter."""
        if version_id not in self._adapter_paths:
            raise ValueError(f"Adapter '{version_id}' not loaded. Load it first.")

        from mlx_lm.tuner.utils import load_adapters

        load_adapters(self._model, self._adapter_paths[version_id])
        self._active_version = version_id
        logger.info("MLX active adapter set to %s", version_id)

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float | None = None,
        stream: bool = False,
    ) -> str:
        """Generate a response for the given chat messages."""
        import mlx_lm

        if self._model is None:
            raise RuntimeError("Call load_base() before generate().")

        prompt = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        from mlx_lm.sample_utils import make_sampler

        kwargs = {"max_tokens": max_new_tokens}
        if temperature is not None and temperature > 0:
            kwargs["sampler"] = make_sampler(temp=temperature)
        return mlx_lm.generate(self._model, self._tokenizer, prompt=prompt, **kwargs)

    def generate_base(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
    ) -> str:
        """Generate using the base model only, bypassing the active LoRA adapter.

        Temporarily zeros all LoRALinear scale values rather than loading a
        second copy of the model, so memory usage stays the same.
        """
        if self._model is None:
            raise RuntimeError("Call load_base() before generate_base().")

        if not self._lora_applied:
            return self.generate(messages, max_new_tokens)

        saved = [(m, m.scale) for m in _iter_lora_layers(self._model)]
        for m, _ in saved:
            m.scale = 0.0
        try:
            return self.generate(messages, max_new_tokens)
        finally:
            for m, scale in saved:
                m.scale = scale

    def offload(self) -> None:
        """Release model weights from memory before a training run."""
        import mlx.core as mx

        self._is_training = True
        self._model = None
        self._tokenizer = None
        self._adapter_paths = {}
        self._active_version = None
        self._lora_applied = False
        mx.clear_cache()
        logger.info("MLX inference model offloaded from memory.")

    def get_active_version(self) -> str | None:
        return self._active_version

    def list_loaded_adapters(self) -> list[str]:
        return list(self._adapter_paths.keys())

    @property
    def is_ready(self) -> bool:
        return self._model is not None

    @property
    def is_training(self) -> bool:
        return self._is_training
