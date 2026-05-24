"""Tests for groundcortex/model_registry.py."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import torch.nn as nn

import json

from groundcortex.model_registry import (
    find_lora_targets,
    get_apply_chat_template_kwargs,
    normalize_messages_for_template,
    parse_tool_calls,
    patch_chat_template_for_trl,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _model_with_modules(named: dict) -> MagicMock:
    """Build a mock model whose named_modules() returns the given dict."""
    m = MagicMock()
    m.named_modules.return_value = list(named.items())
    return m


def _linear() -> nn.Linear:
    return nn.Linear(4, 4, bias=False)


# ──────────────────────────────────────────────────────────────────────────────
# find_lora_targets - basic cases
# ──────────────────────────────────────────────────────────────────────────────

class TestFindLoraTargetsBasic:
    def test_empty_model_returns_empty(self):
        model = _model_with_modules({})
        assert find_lora_targets(model) == []

    def test_single_linear_below_threshold_excluded(self):
        # Only one module with suffix "q_proj" → count=1, filtered out.
        model = _model_with_modules({"layers.0.q_proj": _linear()})
        assert find_lora_targets(model) == []

    def test_repeated_suffix_included(self):
        model = _model_with_modules({
            "layers.0.q_proj": _linear(),
            "layers.1.q_proj": _linear(),
        })
        assert find_lora_targets(model) == ["q_proj"]

    def test_returns_sorted(self):
        model = _model_with_modules({
            "layers.0.v_proj": _linear(),
            "layers.1.v_proj": _linear(),
            "layers.0.q_proj": _linear(),
            "layers.1.q_proj": _linear(),
        })
        assert find_lora_targets(model) == ["q_proj", "v_proj"]

    def test_multiple_targets_collected(self):
        modules = {}
        for i in range(2):
            for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                modules[f"layers.{i}.{name}"] = _linear()
        model = _model_with_modules(modules)
        assert find_lora_targets(model) == ["k_proj", "o_proj", "q_proj", "v_proj"]


# ──────────────────────────────────────────────────────────────────────────────
# find_lora_targets - skip names
# ──────────────────────────────────────────────────────────────────────────────

class TestFindLoraTargetsSkipNames:
    def test_lm_head_excluded(self):
        model = _model_with_modules({
            "lm_head": _linear(),
            "lm_head.duplicate": _linear(),
        })
        assert find_lora_targets(model) == []

    def test_embed_tokens_excluded(self):
        model = _model_with_modules({
            "model.embed_tokens": _linear(),
            "model.embed_tokens.weight": _linear(),
        })
        assert find_lora_targets(model) == []

    def test_skip_name_mixed_with_valid(self):
        model = _model_with_modules({
            "lm_head": _linear(),
            "layers.0.q_proj": _linear(),
            "layers.1.q_proj": _linear(),
        })
        assert find_lora_targets(model) == ["q_proj"]


# ──────────────────────────────────────────────────────────────────────────────
# find_lora_targets - non-Linear modules ignored
# ──────────────────────────────────────────────────────────────────────────────

class TestFindLoraTargetsTypeCheck:
    def test_non_linear_subclass_excluded(self):
        # isinstance would match, but type() is exact - subclass should be excluded.
        class MyLinear(nn.Linear):
            pass

        model = _model_with_modules({
            "layers.0.q_proj": MyLinear(4, 4),
            "layers.1.q_proj": MyLinear(4, 4),
        })
        assert find_lora_targets(model) == []

    def test_conv_excluded(self):
        model = _model_with_modules({
            "layers.0.conv": nn.Conv2d(4, 4, 1),
            "layers.1.conv": nn.Conv2d(4, 4, 1),
        })
        assert find_lora_targets(model) == []

    def test_mixed_exact_linear_and_subclass(self):
        class MyLinear(nn.Linear):
            pass

        model = _model_with_modules({
            "layers.0.q_proj": _linear(),      # exact nn.Linear
            "layers.1.q_proj": MyLinear(4, 4), # subclass - not exact
        })
        # Only one exact nn.Linear for q_proj → count=1, below threshold.
        assert find_lora_targets(model) == []


# ──────────────────────────────────────────────────────────────────────────────
# find_lora_targets - generic leaf (2-component suffix)
# ──────────────────────────────────────────────────────────────────────────────

class TestFindLoraTargetsGenericLeaf:
    def test_generic_leaf_uses_parent_dot_leaf(self):
        # Leaf name "linear" is generic → suffix should be "q_proj.linear"
        model = _model_with_modules({
            "layers.0.q_proj.linear": _linear(),
            "layers.1.q_proj.linear": _linear(),
        })
        assert find_lora_targets(model) == ["q_proj.linear"]

    def test_dense_leaf_uses_parent_dot_leaf(self):
        model = _model_with_modules({
            "layers.0.attention.dense": _linear(),
            "layers.1.attention.dense": _linear(),
        })
        assert find_lora_targets(model) == ["attention.dense"]

    def test_non_generic_leaf_uses_single_component(self):
        model = _model_with_modules({
            "layers.0.query_key_value": _linear(),
            "layers.1.query_key_value": _linear(),
        })
        assert find_lora_targets(model) == ["query_key_value"]


# ──────────────────────────────────────────────────────────────────────────────
# find_lora_targets - validation pass (PEFT suffix collision)
# ──────────────────────────────────────────────────────────────────────────────

class TestFindLoraTargetsValidation:
    def test_suffix_with_non_linear_match_excluded(self):
        """Simulates Gemma 4: vision encoder has plain nn.Linear at q_proj,
        but language model uses a wrapper (not exactly nn.Linear) at the same suffix.
        The suffix should be excluded to prevent PEFT's ValueError.
        """
        class ClippableLinear(nn.Module):
            pass

        model = _model_with_modules({
            # Vision encoder - exact nn.Linear
            "vision.layers.0.q_proj": _linear(),
            "vision.layers.1.q_proj": _linear(),
            # Language model - wrapper, not nn.Linear
            "language.layers.0.q_proj": ClippableLinear(),
            "language.layers.1.q_proj": ClippableLinear(),
        })
        assert find_lora_targets(model) == []

    def test_suffix_with_all_linear_passes_validation(self):
        model = _model_with_modules({
            "vision.layers.0.q_proj": _linear(),
            "vision.layers.1.q_proj": _linear(),
            "language.layers.0.q_proj": _linear(),
            "language.layers.1.q_proj": _linear(),
        })
        assert find_lora_targets(model) == ["q_proj"]

    def test_valid_and_invalid_suffix_partial_exclusion(self):
        """k_proj is clean (all nn.Linear), q_proj has a collision - only k_proj kept."""
        class Wrapper(nn.Module):
            pass

        model = _model_with_modules({
            "layers.0.q_proj": _linear(),
            "layers.1.q_proj": Wrapper(),  # collision
            "layers.0.k_proj": _linear(),
            "layers.1.k_proj": _linear(),
        })
        assert find_lora_targets(model) == ["k_proj"]


# ──────────────────────────────────────────────────────────────────────────────
# get_apply_chat_template_kwargs
# ──────────────────────────────────────────────────────────────────────────────
# normalize_messages_for_template
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeMessagesForTemplate:
    def _assistant_msg(self, name: str, args) -> dict:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "call_abc", "type": "function",
                            "function": {"name": name, "arguments": args}}],
        }

    def test_json_string_arguments_deserialized(self):
        msg = self._assistant_msg("get_time", '{"tz": "UTC"}')
        result = normalize_messages_for_template([msg])
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, dict)
        assert args == {"tz": "UTC"}

    def test_dict_arguments_unchanged(self):
        msg = self._assistant_msg("get_time", {"tz": "UTC"})
        result = normalize_messages_for_template([msg])
        args = result[0]["tool_calls"][0]["function"]["arguments"]
        assert args == {"tz": "UTC"}

    def test_messages_without_tool_calls_unchanged(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        assert normalize_messages_for_template(msgs) == msgs

    def test_tool_message_not_modified(self):
        msg = {"role": "tool", "tool_call_id": "call_abc", "content": "12:00"}
        result = normalize_messages_for_template([msg])
        assert result[0] == msg

    def test_original_message_not_mutated(self):
        msg = self._assistant_msg("get_time", '{"tz": "UTC"}')
        original_args = msg["tool_calls"][0]["function"]["arguments"]
        normalize_messages_for_template([msg])
        assert msg["tool_calls"][0]["function"]["arguments"] == original_args


# ──────────────────────────────────────────────────────────────────────────────

class TestGetApplyChatTemplateKwargs:
    def test_qwen_returns_enable_thinking_false(self):
        assert get_apply_chat_template_kwargs("Qwen/Qwen3.5-2B") == {"enable_thinking": False}

    def test_qwen_mlx_community_variant(self):
        assert get_apply_chat_template_kwargs("mlx-community/Qwen3.6-35B-A3B-4bit") == {"enable_thinking": False}

    def test_gemma_returns_empty(self):
        assert get_apply_chat_template_kwargs("google/gemma-4-E4B-it") == {}

    def test_unknown_model_returns_empty(self):
        assert get_apply_chat_template_kwargs("meta-llama/Llama-3.2-3B-Instruct") == {}

    def test_returns_copy_not_registry_reference(self):
        # Mutating the returned dict must not affect future calls.
        kwargs = get_apply_chat_template_kwargs("Qwen/Qwen3.5-2B")
        kwargs["enable_thinking"] = True
        assert get_apply_chat_template_kwargs("Qwen/Qwen3.5-2B") == {"enable_thinking": False}


# ──────────────────────────────────────────────────────────────────────────────
# patch_chat_template_for_trl
# ──────────────────────────────────────────────────────────────────────────────

class TestPatchChatTemplateForTrl:
    def _tokenizer(self, template: str) -> MagicMock:
        t = MagicMock()
        t.chat_template = template
        return t

    def test_already_patched_is_noop(self):
        tokenizer = self._tokenizer("... {% generation %} ...")
        original = tokenizer.chat_template
        patch_chat_template_for_trl(tokenizer, "Qwen/Qwen3.5-2B")
        assert tokenizer.chat_template == original

    def test_none_template_is_noop(self):
        tokenizer = self._tokenizer(None)
        patch_chat_template_for_trl(tokenizer, "Qwen/Qwen3.5-2B")
        assert tokenizer.chat_template is None

    def test_unknown_model_is_noop(self):
        tokenizer = self._tokenizer("some template content")
        patch_chat_template_for_trl(tokenizer, "meta-llama/Llama-3.2-3B-Instruct")
        assert tokenizer.chat_template == "some template content"

    def test_qwen_patch_adds_generation_markers(self):
        old_block = (
            "        {%- if loop.index0 > ns.last_query_index %}\n"
            "            {{- '<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content }}\n"
            "        {%- else %}\n"
            "            {{- '<|im_start|>' + message.role + '\\n' + content }}\n"
            "        {%- endif %}"
            "        {{- '<|im_end|>\\n' }}\n"
            "    {%- elif message.role == \"tool\" %}"
        )
        tokenizer = self._tokenizer(old_block)
        patch_chat_template_for_trl(tokenizer, "Qwen/Qwen3.5-2B")
        assert "{% generation %}" in tokenizer.chat_template
        assert "{% endgeneration %}" in tokenizer.chat_template

    def test_gemma_patch_adds_generation_markers(self):
        template = "... {{- captured_content -}} ..."
        tokenizer = self._tokenizer(template)
        patch_chat_template_for_trl(tokenizer, "google/gemma-4-E4B-it")
        assert "{% generation %}" in tokenizer.chat_template
        assert "{% endgeneration %}" in tokenizer.chat_template

    def test_gemma_patch_missing_pattern_is_noop(self, capsys):
        template = "some other template without the expected pattern"
        tokenizer = self._tokenizer(template)
        patch_chat_template_for_trl(tokenizer, "google/gemma-4-E4B-it")
        assert tokenizer.chat_template == template
        assert "WARNING" in capsys.readouterr().out


# ──────────────────────────────────────────────────────────────────────────────
# parse_tool_calls
# ──────────────────────────────────────────────────────────────────────────────

class TestParseToolCalls:
    QWEN = "mlx-community/Qwen3.6-35B-A3B-4bit"

    def _wrap(self, name: str, arguments: dict | None = None) -> str:
        payload = {"name": name, "arguments": arguments or {}}
        return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"

    def test_single_tool_call_parsed(self):
        result = parse_tool_calls(self._wrap("get_time"), self.QWEN)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_time"
        assert result[0]["type"] == "function"

    def test_multiple_tool_calls_parsed(self):
        response = self._wrap("get_time") + "\n" + self._wrap("get_date")
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        assert len(result) == 2
        assert result[0]["function"]["name"] == "get_time"
        assert result[1]["function"]["name"] == "get_date"

    def test_no_tool_call_returns_none(self):
        assert parse_tool_calls("Hello, how can I help you?", self.QWEN) is None

    def test_malformed_json_returns_none(self):
        assert parse_tool_calls("<tool_call>not valid json</tool_call>", self.QWEN) is None

    def test_arguments_serialised_as_json_string(self):
        result = parse_tool_calls(self._wrap("search", {"query": "weather"}), self.QWEN)
        assert result is not None
        args = result[0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"query": "weather"}

    def test_unknown_model_returns_none(self):
        assert parse_tool_calls(self._wrap("get_time"), "meta-llama/Llama-3.2-3B-Instruct") is None

    def test_call_id_is_unique(self):
        response = self._wrap("get_time") + "\n" + self._wrap("get_date")
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        assert result[0]["id"] != result[1]["id"]

    def test_call_id_has_call_prefix(self):
        result = parse_tool_calls(self._wrap("get_time"), self.QWEN)
        assert result[0]["id"].startswith("call_")

    def test_empty_arguments_produces_empty_json_object(self):
        result = parse_tool_calls(self._wrap("ping", {}), self.QWEN)
        assert json.loads(result[0]["function"]["arguments"]) == {}

    def test_function_tag_format_parsed(self):
        response = "<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_time"
        assert json.loads(result[0]["function"]["arguments"]) == {}

    def test_function_tag_format_with_arguments(self):
        response = '<tool_call>\n<function=search>{"query": "weather"}</function>\n</tool_call>'
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        assert result[0]["function"]["name"] == "search"
        assert json.loads(result[0]["function"]["arguments"]) == {"query": "weather"}

    def test_xml_parameter_format_single_param(self):
        # mlx-community quantised Qwen3 uses <parameter=name>value</parameter> format
        response = (
            "<tool_call>\n"
            "<function=memory_read>\n"
            "<parameter=file>\nMEMORY.md\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "memory_read"
        assert json.loads(result[0]["function"]["arguments"]) == {"file": "MEMORY.md"}

    def test_xml_parameter_format_multiple_params(self):
        response = (
            "<tool_call>\n"
            "<function=search_notes>\n"
            "<parameter=query>\nproject updates\n</parameter>\n"
            "<parameter=count>\n5\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        args = json.loads(result[0]["function"]["arguments"])
        assert args["query"] == "project updates"
        assert args["count"] == 5  # numeric — parsed as int via json.loads

    def test_xml_parameter_format_multiple_calls(self):
        response = (
            "<tool_call>\n<function=read_file>\n"
            "<parameter=path>/tmp/a.txt</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=read_file>\n"
            "<parameter=path>/tmp/b.txt</parameter>\n</function>\n</tool_call>"
        )
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        assert len(result) == 2
        assert json.loads(result[0]["function"]["arguments"]) == {"path": "/tmp/a.txt"}
        assert json.loads(result[1]["function"]["arguments"]) == {"path": "/tmp/b.txt"}

    def test_xml_parameter_format_no_params_gives_empty_dict(self):
        response = (
            "<tool_call>\n<function=get_time>\n</function>\n</tool_call>"
        )
        result = parse_tool_calls(response, self.QWEN)
        assert result is not None
        assert json.loads(result[0]["function"]["arguments"]) == {}


class TestParseToolCallsGemma:
    GEMMA = "google/gemma-4-E4B-it"

    def _wrap(self, name: str, parameters: dict | None = None) -> str:
        payload = {"name": name, "parameters": parameters or {}}
        return f"<tool_call>\n{json.dumps(payload)}\n</tool_call>"

    def test_single_tool_call_parsed(self):
        result = parse_tool_calls(self._wrap("get_time"), self.GEMMA)
        assert result is not None
        assert len(result) == 1
        assert result[0]["function"]["name"] == "get_time"
        assert result[0]["type"] == "function"

    def test_parameters_normalized_to_arguments(self):
        result = parse_tool_calls(self._wrap("search", {"query": "weather"}), self.GEMMA)
        assert result is not None
        args = result[0]["function"]["arguments"]
        assert isinstance(args, str)
        assert json.loads(args) == {"query": "weather"}

    def test_empty_parameters_produces_empty_json_object(self):
        result = parse_tool_calls(self._wrap("ping", {}), self.GEMMA)
        assert json.loads(result[0]["function"]["arguments"]) == {}

    def test_no_tool_call_returns_none(self):
        assert parse_tool_calls("Hello, how can I help you?", self.GEMMA) is None

    def test_malformed_json_returns_none(self):
        assert parse_tool_calls("<tool_call>not valid json</tool_call>", self.GEMMA) is None

    def test_call_id_has_call_prefix(self):
        result = parse_tool_calls(self._wrap("get_time"), self.GEMMA)
        assert result[0]["id"].startswith("call_")
