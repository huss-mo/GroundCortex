"""Tests for ExampleGenerator (pipeline/generator.py)."""
from __future__ import annotations

import json

import pytest

from groundcortex.pipeline.generator import ExampleGenerator, _parse_pairs
from groundcortex.pipeline.models import Experience


def _exp(content="Alice is a senior engineer at Acme Corp.") -> Experience:
    return Experience(
        source="file:test.md",
        raw_content=content,
        content_hash="abc123",
    )


def _valid_json_response(n: int = 5) -> str:
    pairs = [
        {"question": f"Question {i}?", "answer": f"Answer {i}."}
        for i in range(n)
    ]
    return json.dumps(pairs)


# ---------------------------------------------------------------------------
# _parse_pairs
# ---------------------------------------------------------------------------

class TestParsePairs:
    def test_valid_json_returns_pairs(self):
        raw = json.dumps([{"question": "Q?", "answer": "A."}])
        pairs = _parse_pairs(raw)
        assert pairs == [("Q?", "A.")]

    def test_five_pairs_parsed_correctly(self):
        raw = _valid_json_response(5)
        assert len(_parse_pairs(raw)) == 5

    def test_json_embedded_in_text(self):
        raw = 'Here is the output:\n[{"question": "Q?", "answer": "A."}]\nDone.'
        pairs = _parse_pairs(raw)
        assert len(pairs) == 1
        assert pairs[0] == ("Q?", "A.")

    def test_invalid_json_returns_empty(self):
        assert _parse_pairs("not json at all") == []

    def test_malformed_json_returns_empty(self):
        assert _parse_pairs("[{bad json}]") == []

    def test_missing_question_key_skipped(self):
        raw = json.dumps([{"answer": "A."}, {"question": "Q?", "answer": "A2."}])
        pairs = _parse_pairs(raw)
        assert len(pairs) == 1
        assert pairs[0][0] == "Q?"

    def test_missing_answer_key_skipped(self):
        raw = json.dumps([{"question": "Q?"}, {"question": "Q2?", "answer": "A."}])
        pairs = _parse_pairs(raw)
        assert len(pairs) == 1

    def test_empty_string_returns_empty(self):
        assert _parse_pairs("") == []

    def test_empty_array_returns_empty(self):
        assert _parse_pairs("[]") == []


# ---------------------------------------------------------------------------
# ExampleGenerator - fallback mode (no generate_fn)
# ---------------------------------------------------------------------------

class TestExampleGeneratorFallback:
    def test_produces_five_examples(self):
        gen = ExampleGenerator(None)
        assert len(gen.generate(_exp(), run_id="run-1")) == 5

    def test_variant_is_direct_in_fallback(self):
        gen = ExampleGenerator(None)
        for ex in gen.generate(_exp(), run_id="run-1"):
            assert ex.variant == "direct"

    def test_run_id_assigned(self):
        gen = ExampleGenerator(None)
        for ex in gen.generate(_exp(), run_id="run-xyz"):
            assert ex.run_id == "run-xyz"

    def test_experience_id_assigned(self):
        gen = ExampleGenerator(None)
        exp = _exp()
        for ex in gen.generate(exp, run_id="run-1"):
            assert ex.experience_id == exp.id

    def test_messages_format(self):
        gen = ExampleGenerator(None)
        for ex in gen.generate(_exp(), run_id="run-1"):
            assert len(ex.messages) == 2
            assert ex.messages[0]["role"] == "user"
            assert ex.messages[1]["role"] == "assistant"

    def test_content_appears_in_answers(self):
        gen = ExampleGenerator(None)
        content = "Canary phrase XYZ123"
        for ex in gen.generate(_exp(content), run_id="run-1"):
            assert "XYZ123" in ex.messages[1]["content"]


# ---------------------------------------------------------------------------
# ExampleGenerator - LLM mode (with generate_fn)
# ---------------------------------------------------------------------------

class TestExampleGeneratorLLM:
    def test_uses_llm_output_when_valid(self):
        generate_fn = lambda msgs, max_tokens: _valid_json_response(5)
        gen = ExampleGenerator(generate_fn)
        result = gen.generate(_exp(), run_id="run-1")
        assert len(result) == 5

    def test_variant_is_generated(self):
        generate_fn = lambda msgs, max_tokens: _valid_json_response(5)
        gen = ExampleGenerator(generate_fn)
        for ex in gen.generate(_exp(), run_id="run-1"):
            assert ex.variant == "generated"

    def test_question_and_answer_from_llm(self):
        pairs = [{"question": "What is X?", "answer": "X is Y."}] * 5
        generate_fn = lambda msgs, max_tokens: json.dumps(pairs)
        gen = ExampleGenerator(generate_fn)
        result = gen.generate(_exp(), run_id="run-1")
        assert result[0].messages[0]["content"] == "What is X?"
        assert result[0].messages[1]["content"] == "X is Y."

    def test_falls_back_to_templates_on_invalid_json(self):
        generate_fn = lambda msgs, max_tokens: "not valid json"
        gen = ExampleGenerator(generate_fn)
        result = gen.generate(_exp(), run_id="run-1")
        assert len(result) == 5
        assert all(ex.variant == "generated" for ex in result)

    def test_falls_back_to_templates_on_exception(self):
        def bad_fn(msgs, max_tokens):
            raise RuntimeError("model not ready")
        gen = ExampleGenerator(bad_fn)
        result = gen.generate(_exp(), run_id="run-1")
        assert len(result) == 5

    def test_run_id_assigned(self):
        generate_fn = lambda msgs, max_tokens: _valid_json_response(5)
        gen = ExampleGenerator(generate_fn)
        for ex in gen.generate(_exp(), run_id="run-abc"):
            assert ex.run_id == "run-abc"

    def test_experience_id_assigned(self):
        generate_fn = lambda msgs, max_tokens: _valid_json_response(5)
        gen = ExampleGenerator(generate_fn)
        exp = _exp()
        for ex in gen.generate(exp, run_id="run-1"):
            assert ex.experience_id == exp.id

    def test_messages_have_correct_roles(self):
        generate_fn = lambda msgs, max_tokens: _valid_json_response(5)
        gen = ExampleGenerator(generate_fn)
        for ex in gen.generate(_exp(), run_id="run-1"):
            assert ex.messages[0]["role"] == "user"
            assert ex.messages[1]["role"] == "assistant"

    def test_generate_fn_receives_messages_list(self):
        captured = {}

        def capture_fn(msgs, max_tokens):
            captured["msgs"] = msgs
            return _valid_json_response(5)

        gen = ExampleGenerator(capture_fn)
        gen.generate(_exp(), run_id="run-1")
        assert isinstance(captured["msgs"], list)
        assert any(m["role"] == "system" for m in captured["msgs"])
        assert captured["msgs"][-1]["role"] == "user"

    def test_each_call_produces_independent_ids(self):
        generate_fn = lambda msgs, max_tokens: _valid_json_response(5)
        gen = ExampleGenerator(generate_fn)
        r1 = gen.generate(_exp("Fact A."), run_id="run-1")
        r2 = gen.generate(_exp("Fact B."), run_id="run-1")
        ids1 = {ex.id for ex in r1}
        ids2 = {ex.id for ex in r2}
        assert ids1.isdisjoint(ids2)
