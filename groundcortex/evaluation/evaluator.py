"""Post-training quality gate.

After a LoRA adapter is trained, evaluate it before hot-swapping:

  1. Recall check  - run held-out validation examples (variant='validation') through
                     the adapter and score each answer with a 3-tier judge.
  2. Sanity check  - run regularization.json questions through the adapter and the
                     base model; use LLM-as-judge to score quality vs the base.

Both scores are expressed as fractions (0.0–1.0). The adapter passes when both
meet their configured thresholds (eval_validation_threshold, eval_sanity_threshold).
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from groundcortex.buffer.db import Database
    from groundcortex.config import GroundCortexConfig
    from groundcortex.inference.manager import InferenceManager

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent.parent / "static"
_REGULARIZATION_PATH = _STATIC_DIR / "regularization.json"

# Stopwords excluded from content-word coverage check (tier-2 judge)
_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "up", "about", "into", "through", "it",
    "its", "this", "that", "these", "those", "i", "you", "he", "she", "we",
    "they", "not", "no", "or", "and", "but", "if", "so", "as", "than",
})

GenerateFn = Callable[[list[dict], int], str]


@dataclass
class EvaluationResult:
    passed: bool
    recall_pct: float       # fraction of held-out probes answered correctly
    sanity_pct: float       # normalised 1-5 judge score (÷5) vs base model
    probe_count: int        # number of validation probes evaluated
    sanity_count: int       # number of sanity check questions evaluated

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Judge utilities
# ---------------------------------------------------------------------------

def _content_words(text: str) -> list[str]:
    return [w for w in text.lower().split() if len(w) >= 3 and w not in _STOPWORDS]


def _judge_answer(
    question: str,
    expected: str,
    response: str,
    generate_base_fn: GenerateFn,
) -> bool:
    """3-tier judge mirroring examples/hypothesis.py.

    Tier 1: verbatim substring  (cheapest)
    Tier 2: content-word coverage
    Tier 3: LLM fallback asking base model yes/no
    """
    resp_lower = response.lower()
    exp_lower = expected.lower()

    # Tier 1 - verbatim
    if exp_lower in resp_lower:
        return True

    # Tier 2 - content-word coverage
    words = _content_words(expected)
    if words and all(w in resp_lower for w in words):
        return True

    # Tier 3 - LLM fallback
    try:
        prompt = (
            f"Question: {question}\n"
            f"Expected answer: {expected}\n"
            f"Model response: {response}\n\n"
            "Do the expected answer and the model response convey equivalent or similar "
            "information? Reply with a single word: yes or no."
        )
        judge_response = generate_base_fn(
            [{"role": "user", "content": prompt}], 16
        ).strip().lower()
        return judge_response.startswith("yes")
    except Exception as exc:
        logger.debug("LLM judge fallback failed: %s", exc)
        return False


def _llm_as_judge(
    question: str,
    base_response: str,
    adapter_response: str,
    generate_base_fn: GenerateFn,
) -> int:
    """Rate the adapter response quality vs the base model response on a 1-5 scale.

    Calls the base model as judge. Defaults to 3 on parse failure (neutral).
    """
    prompt = (
        f"You are evaluating the quality of an AI assistant's response.\n\n"
        f"Question: {question}\n"
        f"Reference answer (base model): {base_response[:300]}\n"
        f"Evaluated answer: {adapter_response[:300]}\n\n"
        "Rate the evaluated answer on a scale from 1 to 5 compared to the reference:\n"
        "5 = equivalent or better quality\n"
        "4 = slightly worse but acceptable\n"
        "3 = noticeably worse but still coherent\n"
        "2 = poor quality, significant degradation\n"
        "1 = incoherent or completely wrong\n\n"
        "Reply with a single digit (1-5)."
    )
    try:
        raw = generate_base_fn([{"role": "user", "content": prompt}], 16).strip()
        for ch in raw:
            if ch.isdigit() and ch != "0":
                return int(ch)
    except Exception as exc:
        logger.debug("LLM sanity judge failed: %s", exc)
    return 3


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_adapter(
    adapter_path: str,
    version: str,
    experience_ids: list[str],
    db: "Database",
    inference_manager: "InferenceManager",
    config: "GroundCortexConfig",
) -> EvaluationResult:
    """Evaluate a just-trained adapter.

    Assumes the base model is already loaded in inference_manager. Loads the
    adapter, runs recall + sanity checks, then leaves the adapter loaded
    (the consolidator decides whether to keep it active or discard it).
    """
    logger.info("Evaluating adapter %s…", version)

    # Load adapter for evaluation
    inference_manager.load_adapter(adapter_path, version)
    inference_manager.set_active(version)

    generate_base = inference_manager.generate_base
    generate = inference_manager.generate

    # ------------------------------------------------------------------
    # 1. Recall check
    # ------------------------------------------------------------------
    val_examples = db.get_validation_examples(experience_ids)
    if not val_examples:
        logger.warning("No validation examples found for %d experiences - skipping recall check.", len(experience_ids))
        recall_pct = 1.0
        probe_count = 0
    else:
        sample = val_examples
        if len(sample) > config.eval_max_probes:
            sample = random.sample(val_examples, config.eval_max_probes)

        passed_recall = 0
        for ex in sample:
            # Extract the user question and expected assistant answer from messages
            question = next(
                (m["content"] for m in ex.messages if m["role"] == "user"), ""
            )
            expected = next(
                (m["content"] for m in ex.messages if m["role"] == "assistant"), ""
            )
            try:
                response = generate([{"role": "user", "content": question}], 256)
            except Exception as exc:
                logger.debug("Generate failed during recall probe: %s", exc)
                response = ""
            if _judge_answer(question, expected, response, generate_base):
                passed_recall += 1

        probe_count = len(sample)
        recall_pct = passed_recall / probe_count
        logger.info("Recall: %d/%d probes passed (%.0f%%)", passed_recall, probe_count, recall_pct * 100)

    # ------------------------------------------------------------------
    # 2. Sanity check (catastrophic forgetting detection)
    # ------------------------------------------------------------------
    try:
        reg_data = json.loads(_REGULARIZATION_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Could not load regularization.json: %s - skipping sanity check.", exc)
        reg_data = []

    sanity_scores: list[int] = []
    for item in reg_data:
        question = item["q"]
        try:
            base_response = generate_base([{"role": "user", "content": question}], 256)
            adapter_response = generate([{"role": "user", "content": question}], 256)
            score = _llm_as_judge(question, base_response, adapter_response, generate_base)
            sanity_scores.append(score)
        except Exception as exc:
            logger.debug("Sanity check item failed: %s", exc)

    sanity_count = len(sanity_scores)
    if sanity_scores:
        raw_mean = sum(sanity_scores) / sanity_count
        sanity_pct = raw_mean / 5.0
    else:
        logger.warning("No sanity scores collected - assuming full sanity.")
        sanity_pct = 1.0

    logger.info(
        "Sanity: avg raw score %.2f/5 → %.0f%% (threshold %.0f%%)",
        (sum(sanity_scores) / sanity_count) if sanity_scores else 5.0,
        sanity_pct * 100,
        config.eval_sanity_threshold * 100,
    )

    # ------------------------------------------------------------------
    # 3. Pass/fail decision
    # ------------------------------------------------------------------
    passed = (
        recall_pct >= config.eval_validation_threshold
        and sanity_pct >= config.eval_sanity_threshold
    )
    if not passed:
        reasons = []
        if recall_pct < config.eval_validation_threshold:
            reasons.append(f"recall {recall_pct:.0%} < {config.eval_validation_threshold:.0%}")
        if sanity_pct < config.eval_sanity_threshold:
            reasons.append(f"sanity {sanity_pct:.0%} < {config.eval_sanity_threshold:.0%}")
        logger.warning("Adapter %s did NOT pass quality gate: %s", version, "; ".join(reasons))
    else:
        logger.info("Adapter %s passed quality gate.", version)

    return EvaluationResult(
        passed=passed,
        recall_pct=recall_pct,
        sanity_pct=sanity_pct,
        probe_count=probe_count,
        sanity_count=sanity_count,
    )
