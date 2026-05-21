from __future__ import annotations

import json
import logging
import re
from typing import Callable

from groundcortex.pipeline.models import Experience, TrainingExample

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Few-shot prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You generate training question-answer pairs from factual content.\n"
    "Given a passage, output exactly 5 diverse Q&A pairs as a JSON array.\n"
    "Vary the phrasing, angle, and approach across questions.\n"
    "All answers must be grounded in the provided content - do not add information not present in the passage.\n"
    "Output only the JSON array, nothing else."
)

_VALIDATION_SYSTEM_PROMPT = (
    "You generate a single held-out evaluation question-answer pair from factual content.\n"
    "The question must use different phrasing and a different angle than a typical direct recall question.\n"
    "Approach it as a reasoning, scenario, or implication question - not a simple 'what is X?' question.\n"
    "The answer must be fully grounded in the provided content.\n"
    'Output a single JSON object with keys "question" and "answer", nothing else.'
)

_FEW_SHOT: list[dict] = [
    {
        "role": "user",
        "content": (
            "Content: The speed of light in a vacuum is approximately "
            "299,792 kilometers per second.\n\nOutput:"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps([
            {
                "question": "What is the speed of light in a vacuum?",
                "answer": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
            },
            {
                "question": "How fast does light travel through empty space?",
                "answer": "Light travels through a vacuum at approximately 299,792 kilometers per second.",
            },
            {
                "question": "Is the speed of light in a vacuum exactly 299,792 km/s?",
                "answer": "It is approximately 299,792 kilometers per second - this is a commonly cited rounded value.",
            },
            {
                "question": "If asked about the speed of light, what would you say?",
                "answer": "The speed of light in a vacuum is approximately 299,792 kilometers per second.",
            },
            {
                "question": "What is significant about 299,792 km/s?",
                "answer": "It is the approximate speed of light in a vacuum - a fundamental physical constant.",
            },
        ]),
    },
    {
        "role": "user",
        "content": (
            "Content: Database indexes improve query performance by allowing the engine "
            "to locate rows without scanning the entire table.\n\nOutput:"
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps([
            {
                "question": "What is the purpose of a database index?",
                "answer": "Database indexes improve query performance by allowing the engine to locate rows without scanning the entire table.",
            },
            {
                "question": "How do database indexes work?",
                "answer": "A database index allows the query engine to locate rows directly without a full table scan, improving performance.",
            },
            {
                "question": "Would a database query be faster with or without an index?",
                "answer": "With an index. Indexes allow the engine to locate rows without scanning the entire table.",
            },
            {
                "question": "What problem do database indexes solve?",
                "answer": "They eliminate the need for full table scans by allowing the query engine to locate relevant rows directly.",
            },
            {
                "question": "Explain the role of indexes in database query performance.",
                "answer": "Indexes allow the database engine to locate rows without scanning the entire table, which significantly improves query performance.",
            },
        ]),
    },
]


def _build_messages(content: str) -> list[dict]:
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        *_FEW_SHOT,
        {"role": "user", "content": f"Content: {content}\n\nOutput:"},
    ]


def _build_validation_messages(content: str) -> list[dict]:
    return [
        {"role": "system", "content": _VALIDATION_SYSTEM_PROMPT},
        {"role": "user", "content": f"Content: {content}\n\nOutput:"},
    ]


def _parse_single_pair(raw: str) -> tuple[str, str] | None:
    """Extract a single (question, answer) pair from a JSON object in the model response."""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        if isinstance(data, dict) and "question" in data and "answer" in data:
            return data["question"], data["answer"]
    except (json.JSONDecodeError, KeyError, TypeError):
        pass
    return None


def _parse_pairs(raw: str) -> list[tuple[str, str]]:
    """Extract (question, answer) pairs from a JSON array in the model response."""
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        return [
            (item["question"], item["answer"])
            for item in data
            if isinstance(item, dict) and "question" in item and "answer" in item
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []


# ---------------------------------------------------------------------------
# Template fallback (used when LLM generation fails or generate_fn is None)
# ---------------------------------------------------------------------------

_TEMPLATES: list[tuple[str, str]] = [
    ("What do you know about this?", "{content}"),
    ("Is the following statement accurate? {content}", "Yes. {content}"),
    ("If someone asked you about this topic, what would you say?", "I would say: {content}"),
    ("How would you describe the following? {content}", "{content}"),
    ("Can you explain this in context? {content}", "{content} This is worth keeping in mind."),
]


def _fallback_pairs(content: str) -> list[tuple[str, str]]:
    return [
        (q.format(content=content), a.format(content=content))
        for q, a in _TEMPLATES
    ]


# ---------------------------------------------------------------------------
# ExampleGenerator
# ---------------------------------------------------------------------------

GenerateFn = Callable[[list[dict], int], str]


class ExampleGenerator:
    """Converts a single Experience into training examples via LLM generation.

    Falls back to static templates if the LLM call fails or returns unparseable output.
    """

    def __init__(self, generate_fn: GenerateFn | None = None) -> None:
        self._generate = generate_fn

    def generate(self, experience: Experience, run_id: str) -> list[TrainingExample]:
        pairs: list[tuple[str, str]] = []

        if self._generate is not None:
            messages = _build_messages(experience.raw_content)
            try:
                raw = self._generate(messages, 1024)
                pairs = _parse_pairs(raw)
            except Exception as exc:
                logger.warning("LLM example generation failed for %s: %s", experience.id, exc)

        if not pairs:
            logger.debug("Using template fallback for experience %s.", experience.id)
            pairs = _fallback_pairs(experience.raw_content)

        variant = "generated" if self._generate is not None else "direct"
        return [
            TrainingExample(
                run_id=run_id,
                experience_id=experience.id,
                variant=variant,  # type: ignore[arg-type]
                messages=[
                    {"role": "user", "content": q},
                    {"role": "assistant", "content": a},
                ],
            )
            for q, a in pairs
        ]

    def generate_validation(self, experience: Experience, run_id: str) -> TrainingExample:
        """Generate one held-out validation Q&A pair with different phrasing from training.

        Uses the LLM with a prompt that explicitly requests a different angle (reasoning,
        scenario, or implication - not direct recall). Falls back to a static question
        derived from the first 200 chars of content if generation fails.
        """
        pair: tuple[str, str] | None = None

        if self._generate is not None:
            messages = _build_validation_messages(experience.raw_content)
            try:
                raw = self._generate(messages, 256)
                pair = _parse_single_pair(raw)
            except Exception as exc:
                logger.warning("Validation example generation failed for %s: %s", experience.id, exc)

        if pair is None:
            snippet = experience.raw_content[:200].strip()
            pair = (
                f"Based on what you know, what can you infer or conclude from: {snippet}",
                snippet,
            )

        q, a = pair
        return TrainingExample(
            run_id=run_id,
            experience_id=experience.id,
            variant="validation",
            messages=[
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ],
        )
