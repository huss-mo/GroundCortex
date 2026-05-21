from __future__ import annotations

import logging

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from groundcortex.config import GroundCortexConfig
from groundcortex.model_registry import get_apply_chat_template_kwargs, normalize_messages_for_template
from groundcortex.training.trainer import _get_device

logger = logging.getLogger(__name__)


class InferenceManager:
    """Manages the base model and multiple named LoRA adapters.

    The base model is loaded once at startup. LoRA adapters are loaded on
    demand and switched via PEFT's multi-adapter API - no model reload needed.
    """

    def __init__(self, config: GroundCortexConfig) -> None:
        self._config = config
        self._device = _get_device()
        self._base_model = None
        self._model: PeftModel | None = None
        self._tokenizer = None
        self._active_version: str | None = None
        self._loaded_adapters: list[str] = []
        self._is_training: bool = False

    def load_base(self) -> None:
        """Load the base model and tokenizer. Call once at startup."""
        self._is_training = False
        cfg = self._config
        logger.info("Loading base model: %s on %s", cfg.model_name, self._device)

        dtype = torch.float16 if self._device in ("cuda", "mps") else torch.float32
        base = AutoModelForCausalLM.from_pretrained(cfg.model_name, dtype=dtype)
        base = base.to(self._device)

        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base.config.pad_token_id = tokenizer.pad_token_id
        if hasattr(base, "generation_config"):
            base.generation_config.pad_token_id = tokenizer.pad_token_id
            base.generation_config.temperature = None
            base.generation_config.top_p = None
            base.generation_config.top_k = None

        # Wrap in PeftModel so we can load named adapters later.
        # We use a dummy adapter init-less approach: store the raw model
        # and wrap it only when the first adapter is loaded.
        self._base_model = base
        self._tokenizer = tokenizer
        self._model = None  # set when first adapter is loaded
        logger.info("Base model loaded.")

    def load_adapter(self, adapter_path: str, version_id: str) -> None:
        """Load a LoRA adapter by path and register it under version_id."""
        if self._base_model is None:
            raise RuntimeError("Call load_base() before load_adapter().")

        logger.info("Loading adapter %s from %s", version_id, adapter_path)

        if self._model is None:
            # First adapter: wrap base in PeftModel
            self._model = PeftModel.from_pretrained(
                self._base_model,
                adapter_path,
                adapter_name=version_id,
            )
        else:
            self._model.load_adapter(adapter_path, adapter_name=version_id)

        self._loaded_adapters.append(version_id)
        logger.info("Adapter %s loaded.", version_id)

    def set_active(self, version_id: str) -> None:
        """Switch the active LoRA adapter. No model reload required."""
        if self._model is None:
            raise RuntimeError("No adapters loaded.")
        if version_id not in self._loaded_adapters:
            raise ValueError(f"Adapter '{version_id}' not loaded. Load it first.")
        # Re-enable adapter layers in case unload_adapter() disabled them.
        self._model.enable_adapter_layers()
        self._model.set_adapter(version_id)
        self._active_version = version_id
        logger.info("Active adapter set to %s", version_id)

    def unload_adapter(self) -> None:
        """Disable LoRA adapters so generation uses the base model only."""
        if self._model is not None:
            self._model.disable_adapter_layers()
        self._active_version = None
        logger.info("LoRA adapters disabled; running on base model.")

    def _run_generate(
        self,
        model,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        tokenizer = self._tokenizer
        template_kwargs = get_apply_chat_template_kwargs(self._config.model_name)
        if tools:
            template_kwargs["tools"] = tools
        if "enable_thinking" in template_kwargs:
            template_kwargs["enable_thinking"] = enable_thinking
        text = tokenizer.apply_chat_template(
            normalize_messages_for_template(messages),
            tokenize=False, add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = tokenizer(text, return_tensors="pt").to(self._device)

        do_sample = temperature is not None and temperature > 0
        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens if max_new_tokens is not None else 32768,
            "do_sample": do_sample,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if enable_thinking and not result.startswith("<think>"):
            result = "<think>\n" + result
        return result

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        enable_thinking: bool = False,
    ) -> str:
        """Generate a complete response for the given chat messages.

        Uses the active LoRA adapter if one is loaded, otherwise the base model.
        """
        model = self._model if self._model is not None else self._base_model
        if model is None:
            raise RuntimeError("Call load_base() before generate().")
        return self._run_generate(model, messages, max_new_tokens, temperature,
                                  tools=tools, enable_thinking=enable_thinking)

    def generate_stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float | None = None,
        tools: list[dict] | None = None,
        enable_thinking: bool = False,
    ):
        """Yield generated text one chunk at a time using TextIteratorStreamer."""
        import threading
        from transformers import TextIteratorStreamer

        model = self._model if self._model is not None else self._base_model
        if model is None:
            raise RuntimeError("Call load_base() before generate_stream().")

        tokenizer = self._tokenizer
        template_kwargs = get_apply_chat_template_kwargs(self._config.model_name)
        if tools:
            template_kwargs["tools"] = tools
        if "enable_thinking" in template_kwargs:
            template_kwargs["enable_thinking"] = enable_thinking
        text = tokenizer.apply_chat_template(
            normalize_messages_for_template(messages),
            tokenize=False, add_generation_prompt=True,
            **template_kwargs,
        )
        inputs = tokenizer(text, return_tensors="pt").to(self._device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

        do_sample = temperature is not None and temperature > 0
        gen_kwargs = {**inputs, "max_new_tokens": max_new_tokens, "do_sample": do_sample,
                      "streamer": streamer}
        if do_sample:
            gen_kwargs["temperature"] = temperature

        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        try:
            for chunk in streamer:
                yield chunk
        finally:
            thread.join()

    def generate_base(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
    ) -> str:
        """Generate using the base model only, bypassing any active LoRA adapter.

        Used during training example generation so that LoRA-baked knowledge
        does not influence how new training pairs are phrased.
        """
        if self._base_model is None:
            raise RuntimeError("Call load_base() before generate_base().")
        return self._run_generate(self._base_model, messages, max_new_tokens)

    def offload(self) -> None:
        """Release model weights from device memory before a training run.

        LoRATrainer loads its own copy of the base model to train on. Calling
        offload() first ensures only one copy of the base model is in memory
        at any time. Reload with load_base() + load_adapter() afterward.
        """
        self._is_training = True
        self._model = None
        self._base_model = None
        self._loaded_adapters = []
        self._active_version = None
        if self._device == "cuda":
            torch.cuda.empty_cache()
        if self._device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        logger.info("Inference model offloaded from memory.")

    def get_active_version(self) -> str | None:
        return self._active_version

    def list_loaded_adapters(self) -> list[str]:
        return list(self._loaded_adapters)

    @property
    def is_ready(self) -> bool:
        return self._base_model is not None

    @property
    def is_training(self) -> bool:
        return self._is_training


def create_manager(config: GroundCortexConfig):
    """Return MLXInferenceManager on macOS + use_qlora=True, InferenceManager otherwise."""
    import platform
    if config.use_qlora and platform.system() == "Darwin":
        from groundcortex.inference.mlx_manager import MLXInferenceManager
        return MLXInferenceManager(config)
    return InferenceManager(config)
