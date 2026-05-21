"""Tests for groundcortex/evaluation/evaluator.py - quality gate logic."""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from groundcortex.evaluation.evaluator import (
    EvaluationResult,
    _judge_answer,
    _llm_as_judge,
    evaluate_adapter,
)
from groundcortex.pipeline.models import TrainingExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _val_example(question="What is X?", answer="X is Y.") -> TrainingExample:
    return TrainingExample(
        run_id="run-1",
        experience_id="exp-1",
        variant="validation",
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    )


def _cfg(validation_threshold=0.6, sanity_threshold=0.6, max_probes=20):
    cfg = MagicMock()
    cfg.eval_validation_threshold = validation_threshold
    cfg.eval_sanity_threshold = sanity_threshold
    cfg.eval_max_probes = max_probes
    return cfg


def _manager(generate_response="correct answer", generate_base_response="base answer"):
    m = MagicMock()
    m.generate.return_value = generate_response
    m.generate_base.return_value = generate_base_response
    return m


def _db(val_examples=None):
    db = MagicMock()
    db.get_validation_examples.return_value = val_examples or []
    return db


# ---------------------------------------------------------------------------
# _judge_answer
# ---------------------------------------------------------------------------

class TestJudgeAnswer:
    def test_verbatim_substring_passes(self):
        assert _judge_answer("Q?", "X is Y.", "The answer is: X is Y.", lambda m, n: "no") is True

    def test_verbatim_case_insensitive_passes(self):
        assert _judge_answer("Q?", "X IS Y.", "x is y.", lambda m, n: "no") is True

    def test_content_words_in_response_passes(self):
        # "Paris" and "capital" are content words (len>=3, non-stopword) of "Paris is the capital"
        assert _judge_answer("Q?", "Paris is the capital", "The capital city is Paris", lambda m, n: "no") is True

    def test_llm_returns_yes_passes(self):
        assert _judge_answer("Q?", "blue", "azure", lambda m, n: "yes that is correct") is True

    def test_llm_returns_no_fails(self):
        assert _judge_answer("Q?", "blue", "completely unrelated answer", lambda m, n: "no") is False

    def test_llm_exception_fails(self):
        def bad_fn(m, n):
            raise RuntimeError("boom")
        assert _judge_answer("Q?", "xyz", "abc", bad_fn) is False

    def test_short_stopwords_only_expected_falls_through_to_llm(self):
        # "is it" has no content words (len<3 or stopword) → tier 2 skipped; LLM returns no
        result = _judge_answer("Q?", "is it", "no match here", lambda m, n: "no")
        assert result is False


# ---------------------------------------------------------------------------
# _llm_as_judge
# ---------------------------------------------------------------------------

class TestLLMAsJudge:
    def test_returns_int_between_1_and_5(self):
        score = _llm_as_judge("Q?", "base", "adapter", lambda m, n: "4")
        assert 1 <= score <= 5

    def test_parses_first_digit(self):
        score = _llm_as_judge("Q?", "base", "adapter", lambda m, n: "3 out of 5")
        assert score == 3

    def test_defaults_to_3_when_no_digit(self):
        score = _llm_as_judge("Q?", "base", "adapter", lambda m, n: "no idea")
        assert score == 3

    def test_defaults_to_3_on_exception(self):
        def bad_fn(m, n):
            raise RuntimeError("boom")
        score = _llm_as_judge("Q?", "base", "adapter", bad_fn)
        assert score == 3

    def test_zero_digit_skipped(self):
        # "0" is not a valid score (1-5), should default to 3
        score = _llm_as_judge("Q?", "base", "adapter", lambda m, n: "0")
        assert score == 3

    def test_score_5_accepted(self):
        score = _llm_as_judge("Q?", "base", "adapter", lambda m, n: "5")
        assert score == 5

    def test_score_1_accepted(self):
        score = _llm_as_judge("Q?", "base", "adapter", lambda m, n: "1")
        assert score == 1


# ---------------------------------------------------------------------------
# evaluate_adapter
# ---------------------------------------------------------------------------

class TestEvaluateAdapter:
    def _run_eval(self, val_examples=None, generate_response="correct", base_response="base",
                  reg_data="[]", validation_threshold=0.6, sanity_threshold=0.6):
        db = _db(val_examples)
        manager = _manager(generate_response=generate_response, generate_base_response=base_response)
        cfg = _cfg(validation_threshold=validation_threshold, sanity_threshold=sanity_threshold)
        mock_path = MagicMock()
        mock_path.read_text.return_value = reg_data
        with patch("groundcortex.evaluation.evaluator._REGULARIZATION_PATH", mock_path):
            return evaluate_adapter("/fake/path", "v1", ["exp-1"], db, manager, cfg)

    def test_adapter_loaded_before_evaluation(self):
        db = _db([])
        manager = _manager()
        cfg = _cfg()
        mock_path = MagicMock()
        mock_path.read_text.return_value = "[]"
        with patch("groundcortex.evaluation.evaluator._REGULARIZATION_PATH", mock_path):
            evaluate_adapter("/fake/path", "v1", ["exp-1"], db, manager, cfg)
        manager.load_adapter.assert_called_once_with("/fake/path", "v1")
        manager.set_active.assert_called_once_with("v1")

    def test_no_validation_examples_recall_defaults_to_1(self):
        result = self._run_eval(val_examples=[])
        assert result.recall_pct == 1.0
        assert result.probe_count == 0

    def test_correct_answer_gives_full_recall(self):
        examples = [_val_example("What is X?", "X is Y.")]
        result = self._run_eval(val_examples=examples, generate_response="X is Y.")
        assert result.recall_pct == pytest.approx(1.0)
        assert result.probe_count == 1

    def test_wrong_answer_gives_zero_recall(self):
        examples = [_val_example("What is X?", "X is Y.")]
        # generate returns something unrelated; base model returns "no" for judge
        result = self._run_eval(
            val_examples=examples,
            generate_response="completely different zqrx",
            base_response="no",
        )
        assert result.recall_pct == pytest.approx(0.0)

    def test_passed_when_both_thresholds_met(self):
        examples = [_val_example("What is X?", "X is Y.")]
        result = self._run_eval(
            val_examples=examples,
            generate_response="X is Y.",  # verbatim → recall=1.0
            validation_threshold=0.5,
            sanity_threshold=0.0,  # no sanity data → sanity_pct=1.0
        )
        assert result.passed is True

    def test_failed_when_recall_below_threshold(self):
        examples = [_val_example("What is X?", "X is Y.")]
        result = self._run_eval(
            val_examples=examples,
            generate_response="completely wrong answer zqrx",
            base_response="no",
            validation_threshold=0.9,
        )
        assert result.passed is False

    def test_sanity_check_uses_reg_json(self):
        reg_data = json.dumps([{"q": "What is the capital of France?"}])
        db = _db([])
        manager = _manager()
        cfg = _cfg(sanity_threshold=0.0)
        mock_path = MagicMock()
        mock_path.read_text.return_value = reg_data
        with patch("groundcortex.evaluation.evaluator._REGULARIZATION_PATH", mock_path):
            result = evaluate_adapter("/fake/path", "v1", ["exp-1"], db, manager, cfg)
        assert result.sanity_count == 1

    def test_sanity_fails_when_judge_scores_low(self):
        reg_data = json.dumps([{"q": "What is the capital of France?"}])
        db = _db([])
        manager = _manager()
        cfg = _cfg(sanity_threshold=0.6)
        mock_path = MagicMock()
        mock_path.read_text.return_value = reg_data
        with patch("groundcortex.evaluation.evaluator._REGULARIZATION_PATH", mock_path), \
             patch("groundcortex.evaluation.evaluator._llm_as_judge", return_value=1):
            result = evaluate_adapter("/fake/path", "v1", ["exp-1"], db, manager, cfg)
        # score=1, sanity_pct = 1/5 = 0.2 < 0.6 → fails
        assert result.sanity_pct == pytest.approx(0.2)
        assert result.passed is False

    def test_empty_reg_data_sanity_defaults_to_1(self):
        result = self._run_eval(val_examples=[], reg_data="[]")
        assert result.sanity_pct == pytest.approx(1.0)
        assert result.sanity_count == 0

    def test_reg_load_failure_sanity_defaults_to_1(self):
        db = _db([])
        manager = _manager()
        cfg = _cfg()
        mock_path = MagicMock()
        mock_path.read_text.side_effect = OSError("file not found")
        with patch("groundcortex.evaluation.evaluator._REGULARIZATION_PATH", mock_path):
            result = evaluate_adapter("/fake/path", "v1", ["exp-1"], db, manager, cfg)
        assert result.sanity_pct == pytest.approx(1.0)

    def test_result_has_correct_fields(self):
        result = self._run_eval()
        assert hasattr(result, "passed")
        assert hasattr(result, "recall_pct")
        assert hasattr(result, "sanity_pct")
        assert hasattr(result, "probe_count")
        assert hasattr(result, "sanity_count")

    def test_as_dict_serializable(self):
        result = self._run_eval()
        d = result.as_dict()
        assert isinstance(d, dict)
        assert "passed" in d
        assert "recall_pct" in d
        assert "sanity_pct" in d

    def test_probe_count_capped_at_max_probes(self):
        # 10 examples but max_probes=3 → only 3 probed
        examples = [_val_example(f"Q{i}?", f"A{i}.") for i in range(10)]
        db = _db(examples)
        manager = _manager()
        cfg = _cfg(max_probes=3)
        mock_path = MagicMock()
        mock_path.read_text.return_value = "[]"
        with patch("groundcortex.evaluation.evaluator._REGULARIZATION_PATH", mock_path):
            result = evaluate_adapter("/fake/path", "v1", ["exp-1"], db, manager, cfg)
        assert result.probe_count == 3
