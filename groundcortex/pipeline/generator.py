from __future__ import annotations

from groundcortex.pipeline.models import Experience, TrainingExample

# Five question templates per fact. Each produces a distinct conversational
# angle, which hypothesis.py showed is critical for reliable recall -
# training on a single phrasing often fails on differently-worded questions.
_TEMPLATES: list[tuple[str, str, str]] = [
    # (variant, question_template, answer_template)
    (
        "direct",
        "What do you know about {entity}?",
        "{content}",
    ),
    (
        "negative",
        "Is the common understanding of {entity} correct?",
        "Not exactly. {content}",
    ),
    (
        "scenario",
        "If someone asked you to describe {entity}, what would you say?",
        "I would say: {content}",
    ),
    (
        "comparative",
        "How would you describe {entity} compared to common assumptions?",
        "Unlike common assumptions, {content}",
    ),
    (
        "reasoning",
        "Can you explain {entity} in context?",
        "{content} This is important to keep in mind when reasoning about {entity}.",
    ),
]


def _extract_entity(raw_content: str) -> str:
    """Best-effort entity: first noun phrase (up to first punctuation or 6 words)."""
    words = raw_content.split()
    entity_words = []
    for w in words[:6]:
        clean = w.rstrip(".,;:!?")
        entity_words.append(clean)
        if w != clean:  # hit punctuation
            break
    return " ".join(entity_words) if entity_words else "this topic"


def _format_messages(question: str, answer: str) -> list[dict]:
    """Conversational messages format required by TRL assistant_only_loss."""
    return [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


class ExampleGenerator:
    """Converts a single Experience into 5 training examples (one per template)."""

    def generate(self, experience: Experience, run_id: str) -> list[TrainingExample]:
        entity = _extract_entity(experience.raw_content)
        content = experience.raw_content.strip()

        examples: list[TrainingExample] = []
        for variant, q_tmpl, a_tmpl in _TEMPLATES:
            question = q_tmpl.format(entity=entity, content=content)
            answer = a_tmpl.format(entity=entity, content=content)
            examples.append(
                TrainingExample(
                    run_id=run_id,
                    experience_id=experience.id,
                    variant=variant,  # type: ignore[arg-type]
                    messages=_format_messages(question, answer),
                )
            )
        return examples
