"""Tests for ExampleGenerator (pipeline/generator.py)."""
from __future__ import annotations

import pytest

from groundcortex.pipeline.generator import ExampleGenerator, _extract_entity
from groundcortex.pipeline.models import Experience

_VARIANTS = {"direct", "negative", "scenario", "comparative", "reasoning"}


def _exp(content="Alice is a senior engineer at Acme Corp.") -> Experience:
    return Experience(
        source="file:test.md",
        raw_content=content,
        content_hash="abc123",
    )


# ---------------------------------------------------------------------------
# _extract_entity
# ---------------------------------------------------------------------------

class TestExtractEntity:
    def test_single_word(self):
        assert _extract_entity("Bob") == "Bob"

    def test_two_words(self):
        assert _extract_entity("Acme Corp") == "Acme Corp"

    def test_stops_at_period(self):
        entity = _extract_entity("Acme Corp. is a tech company.")
        assert "." not in entity
        assert "Acme Corp" in entity

    def test_stops_at_comma(self):
        entity = _extract_entity("Alice, the engineer,")
        assert "," not in entity

    def test_max_six_words(self):
        entity = _extract_entity("one two three four five six seven eight")
        assert len(entity.split()) <= 6

    def test_exactly_six_words_allowed(self):
        entity = _extract_entity("one two three four five six")
        assert entity == "one two three four five six"

    def test_empty_content_returns_fallback(self):
        assert _extract_entity("") == "this topic"

    def test_whitespace_only_returns_fallback(self):
        assert _extract_entity("   ") == "this topic"


# ---------------------------------------------------------------------------
# ExampleGenerator
# ---------------------------------------------------------------------------

class TestExampleGenerator:
    def test_generates_five_examples(self):
        gen = ExampleGenerator()
        result = gen.generate(_exp(), run_id="run-1")
        assert len(result) == 5

    def test_all_five_variants_present(self):
        gen = ExampleGenerator()
        variants = {ex.variant for ex in gen.generate(_exp(), run_id="run-1")}
        assert variants == _VARIANTS

    def test_run_id_assigned_to_all(self):
        gen = ExampleGenerator()
        for ex in gen.generate(_exp(), run_id="run-xyz"):
            assert ex.run_id == "run-xyz"

    def test_experience_id_assigned_to_all(self):
        gen = ExampleGenerator()
        exp = _exp()
        for ex in gen.generate(exp, run_id="run-1"):
            assert ex.experience_id == exp.id

    def test_messages_have_user_and_assistant_roles(self):
        gen = ExampleGenerator()
        for ex in gen.generate(_exp(), run_id="run-1"):
            assert len(ex.messages) == 2
            assert ex.messages[0]["role"] == "user"
            assert ex.messages[1]["role"] == "assistant"

    def test_content_appears_in_all_answers(self):
        gen = ExampleGenerator()
        content = "Unique canary phrase XYZ123"
        for ex in gen.generate(_exp(content=content), run_id="run-1"):
            assert "XYZ123" in ex.messages[1]["content"]

    def test_direct_variant_question_contains_entity(self):
        gen = ExampleGenerator()
        exp = _exp("Paris is the capital of France.")
        examples = gen.generate(exp, run_id="run-1")
        direct = next(e for e in examples if e.variant == "direct")
        assert "Paris" in direct.messages[0]["content"]

    def test_direct_variant_answer_is_raw_content(self):
        gen = ExampleGenerator()
        exp = _exp("Paris is the capital of France.")
        examples = gen.generate(exp, run_id="run-1")
        direct = next(e for e in examples if e.variant == "direct")
        assert "Paris is the capital of France." in direct.messages[1]["content"]

    def test_negative_variant_answer_starts_with_not_exactly(self):
        gen = ExampleGenerator()
        examples = gen.generate(_exp(), run_id="run-1")
        negative = next(e for e in examples if e.variant == "negative")
        assert negative.messages[1]["content"].startswith("Not exactly.")

    def test_scenario_variant_answer_starts_with_i_would(self):
        gen = ExampleGenerator()
        examples = gen.generate(_exp(), run_id="run-1")
        scenario = next(e for e in examples if e.variant == "scenario")
        assert scenario.messages[1]["content"].startswith("I would say:")

    def test_comparative_variant_answer_starts_with_unlike(self):
        gen = ExampleGenerator()
        examples = gen.generate(_exp(), run_id="run-1")
        comparative = next(e for e in examples if e.variant == "comparative")
        assert comparative.messages[1]["content"].startswith("Unlike common assumptions,")

    def test_reasoning_variant_answer_ends_with_entity_reference(self):
        gen = ExampleGenerator()
        exp = _exp("Paris is the capital of France.")
        examples = gen.generate(exp, run_id="run-1")
        reasoning = next(e for e in examples if e.variant == "reasoning")
        assert "Paris" in reasoning.messages[1]["content"]

    def test_each_call_produces_independent_examples(self):
        gen = ExampleGenerator()
        exp1 = _exp("Fact about Alice.")
        exp2 = _exp("Fact about Bob.")
        r1 = gen.generate(exp1, run_id="run-1")
        r2 = gen.generate(exp2, run_id="run-1")
        ids1 = {ex.id for ex in r1}
        ids2 = {ex.id for ex in r2}
        assert ids1.isdisjoint(ids2)  # all IDs are unique
