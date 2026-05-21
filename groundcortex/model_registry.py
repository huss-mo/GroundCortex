"""
Model registry for the GroundCortex hypothesis test pipeline.

Centralises per-model-family customisations that can't be auto-detected:
  - find_lora_targets(model)           auto-discover LoRA target module names
  - patch_chat_template_for_trl()      add {% generation %} markers TRL requires
  - get_apply_chat_template_kwargs()   extra kwargs for tokenizer.apply_chat_template
  - parse_tool_calls(response, model)  parse model-native tool call output

For most standard models (Llama 3, Mistral, Phi-3, Qwen2.5, etc.) TRL's own
auto-patcher handles the chat template during SFTTrainer init, so no registry
entry is needed. Add one only when TRL raises ValueError about the template.

Adding support for a new model family:
  1. Write a _patch_<family>(tokenizer) function that adds {% generation %} /
     {% endgeneration %} markers to the right location in tokenizer.chat_template.
  2. Register it in _REGISTRY under a lowercase key that appears in the model name.
  3. If the model needs extra apply_chat_template kwargs (e.g. enable_thinking),
     add them under "apply_chat_template_kwargs".
  4. If the model emits tool calls in a non-standard format, write a
     _parse_<family>_tool_calls(response) function and wire it into
     parse_tool_calls(). The function must normalize to OpenAI format:
     {"id": "call_<hex>", "type": "function", "function": {"name": ..., "arguments": <json_str>}}
"""

import json
import re
import uuid
from collections import Counter

import torch


# ──────────────────────────────────────────────────────────────────────────────
# LoRA target discovery
# ──────────────────────────────────────────────────────────────────────────────

_SKIP_NAMES = frozenset({"lm_head", "embed_tokens", "wte", "wpe"})
# Leaf names too generic to use as a PEFT suffix alone — use parent.leaf instead.
_GENERIC_LEAF = frozenset({"linear", "dense"})


def find_lora_targets(model) -> list[str]:
    """Auto-discover LoRA-compatible target module names from a loaded model.

    Walks model.named_modules(), collects exact nn.Linear paths, drops known
    non-projection layers (lm_head, embeddings), deduplicates to the shortest
    unambiguous suffix, and keeps only suffixes that appear in ≥2 modules
    (one-off linears like adapter bridges are filtered out).

    Handles wrapper architectures (e.g. Gemma4ClippableLinear wrapping nn.Linear):
    when the leaf name is generic ("linear", "dense"), two path components are
    used as the suffix. A final validation pass removes any suffix where a
    non-Linear module (e.g. the wrapper itself) would also be matched by PEFT,
    which would cause PEFT to raise ValueError at injection time.
    """
    paths: list[list[str]] = []
    for name, module in model.named_modules():
        if type(module) is not torch.nn.Linear:
            continue
        parts = name.split(".")
        if any(s in parts for s in _SKIP_NAMES):
            continue
        paths.append(parts)

    if not paths:
        return []

    def suffix(parts: list[str]) -> str:
        if parts[-1] in _GENERIC_LEAF and len(parts) >= 2:
            return f"{parts[-2]}.{parts[-1]}"
        return parts[-1]

    counts = Counter(suffix(p) for p in paths)
    # Projection layers repeat once per transformer block; filter out singletons.
    candidates = [s for s, n in counts.items() if n >= 2]

    # PEFT matches targets by suffix across ALL named modules regardless of type.
    # If any module matching a suffix is not exactly nn.Linear, PEFT raises
    # ValueError. This happens on multimodal models (e.g. Gemma 4) where the
    # vision encoder has plain nn.Linear at "q_proj" but the language model wraps
    # the same name in Gemma4ClippableLinear. Validate and drop unsafe suffixes.
    all_modules = dict(model.named_modules())
    valid = [
        s for s in candidates
        if all(
            type(mod) is torch.nn.Linear
            for name, mod in all_modules.items()
            if name == s or name.endswith("." + s)
        )
    ]
    return sorted(valid)


# ──────────────────────────────────────────────────────────────────────────────
# Chat template patch functions
# ──────────────────────────────────────────────────────────────────────────────

def _patch_qwen3(tokenizer) -> None:
    """Patch Qwen3/3.5 chat template for TRL assistant_only_loss compatibility.

    Qwen3/3.5 uses a multimodal template with a bifurcated assistant output path
    (one branch prepends a <think> block, the other emits content directly).
    TRL's auto-patcher only handles simple single-path templates and silently
    skips this one, causing loss to train on all tokens including user prompts.

    The patch is a no-op if markers already exist or the target pattern is absent.
    """
    if "{% generation %}" in tokenizer.chat_template:
        return

    old = (
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
        "        {%- endif %}"
    )
    new = (
        "        {%- if loop.index0 > ns.last_query_index %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' }}\n"
        "        {%- else %}\n"
        "            {{- '<|im_start|>' + message.role + '\\n' }}\n"
        "        {%- endif %}\n"
        "        {% generation %}{{- content }}"
    )
    tokenizer.chat_template = tokenizer.chat_template.replace(old, new)

    tokenizer.chat_template = tokenizer.chat_template.replace(
        "        {{- '<|im_end|>\\n' }}\n    {%- elif message.role == \"tool\" %}",
        "        {{- '<|im_end|>\\n' }}{% endgeneration %}\n    {%- elif message.role == \"tool\" %}",
    )


def _patch_gemma4(tokenizer) -> None:
    """Patch Gemma 4 chat template for TRL assistant_only_loss compatibility.

    Gemma 4 captures each message's content in a Jinja2 set-block and renders
    it with a single line:  {{- captured_content -}}
    TRL's auto-patcher doesn't support this pattern (it expects direct output
    of content, not via a set-variable). The patch wraps that render line with
    {% generation %} / {% endgeneration %} conditionally for model turns.

    The patch is a no-op if markers already exist or the target pattern is absent.
    """
    if "{% generation %}" in tokenizer.chat_template:
        return

    old = "{{- captured_content -}}"
    new = (
        "{%- if role == 'model' -%}"
        "{% generation %}{{- captured_content -}}{% endgeneration %}"
        "{%- else -%}"
        "{{- captured_content -}}"
        "{%- endif -%}"
    )
    if old not in tokenizer.chat_template:
        print("  WARNING: Could not patch Gemma 4 chat template — pattern not found.")
        return
    tokenizer.chat_template = tokenizer.chat_template.replace(old, new)


# ──────────────────────────────────────────────────────────────────────────────
# Registry
# ──────────────────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, dict] = {
    "qwen": {
        "chat_template_patch": _patch_qwen3,
        "apply_chat_template_kwargs": {"enable_thinking": False},
    },
    "gemma": {
        "chat_template_patch": _patch_gemma4,
        "apply_chat_template_kwargs": {},
    },
}


def _get_family(model_name: str) -> str | None:
    lower = model_name.lower()
    for key in _REGISTRY:
        if key in lower:
            return key
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def patch_chat_template_for_trl(tokenizer, model_name: str) -> None:
    """Pre-patch the tokenizer's chat template so TRL accepts it for training.

    For models with standard templates (Llama 3, Mistral, Phi-3, Qwen2.5, etc.),
    TRL's own patcher handles the template during SFTTrainer init — this function
    is a no-op for those. For families listed in _REGISTRY (currently Qwen3,
    Gemma 4), the patch is applied here before SFTTrainer is constructed, because
    TRL raises ValueError on those templates rather than patching them.
    """
    if not tokenizer.chat_template:
        return
    if "{% generation %}" in tokenizer.chat_template:
        return
    family = _get_family(model_name)
    if family and _REGISTRY[family].get("chat_template_patch"):
        _REGISTRY[family]["chat_template_patch"](tokenizer)


def _parse_gemma_tool_calls(response: str) -> list[dict] | None:
    """Parse Gemma 4 tool call output into OpenAI tool_calls format.

    Gemma 4 emits JSON tool calls using "parameters" instead of "arguments":
      <tool_call>{"name": "x", "parameters": {...}}</tool_call>
    The key is normalized to "arguments" to match the OpenAI spec.
    """
    calls = []
    for raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL):
        try:
            data = json.loads(raw)
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("parameters", data.get("arguments", {}))),
                },
            })
        except (json.JSONDecodeError, KeyError):
            return None
    return calls or None


def _parse_qwen_tool_calls(response: str) -> list[dict] | None:
    calls = []

    # JSON format: <tool_call>{"name": "x", "arguments": {...}}</tool_call>
    for raw in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL):
        try:
            data = json.loads(raw)
            calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": data["name"],
                    "arguments": json.dumps(data.get("arguments", {})),
                },
            })
        except (json.JSONDecodeError, KeyError):
            return None

    if calls:
        return calls

    # Function-tag format: <function=name>args_json_or_empty</function>
    # Appears inside or outside <tool_call> blocks depending on model version.
    for name, args_raw in re.findall(
        r"<function=([^>]+)>(.*?)</function>", response, re.DOTALL
    ):
        args_raw = args_raw.strip()
        try:
            arguments = json.loads(args_raw) if args_raw else {}
        except json.JSONDecodeError:
            arguments = {}
        calls.append({
            "id": f"call_{uuid.uuid4().hex[:8]}",
            "type": "function",
            "function": {
                "name": name.strip(),
                "arguments": json.dumps(arguments),
            },
        })

    return calls or None


def parse_tool_calls(response: str, model_name: str) -> list[dict] | None:
    """Parse model-native tool call output into OpenAI tool_calls format.

    Returns None if no tool calls are found or the model family is not supported.
    """
    family = _get_family(model_name)
    if family == "qwen":
        return _parse_qwen_tool_calls(response)
    if family == "gemma":
        return _parse_gemma_tool_calls(response)
    return None


def normalize_messages_for_template(messages: list[dict]) -> list[dict]:
    """Deserialize tool_calls arguments from JSON strings to dicts.

    OpenAI spec stores function arguments as a JSON-encoded string; model chat
    templates expect them as a parsed dict and call .items() on the value.
    Call this before apply_chat_template whenever messages may contain
    assistant tool_calls returned from a prior turn.
    """
    result = []
    for msg in messages:
        tool_calls = msg.get("tool_calls")
        if tool_calls:
            msg = dict(msg)
            normalized = []
            for tc in tool_calls:
                args = tc["function"]["arguments"]
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        pass
                normalized.append({**tc, "function": {**tc["function"], "arguments": args}})
            msg["tool_calls"] = normalized
        result.append(msg)
    return result


def get_apply_chat_template_kwargs(model_name: str) -> dict:
    """Return extra kwargs to pass to tokenizer.apply_chat_template for this model.

    Example: Qwen3 requires enable_thinking=False to suppress chain-of-thought
    output during inference (the model otherwise prepends a <think> block).
    """
    family = _get_family(model_name)
    if family:
        return dict(_REGISTRY[family].get("apply_chat_template_kwargs", {}))
    return {}
