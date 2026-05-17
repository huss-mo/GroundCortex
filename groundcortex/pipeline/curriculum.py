from __future__ import annotations

import json
import uuid
from pathlib import Path

from datasets import Dataset

from groundcortex.buffer.db import Database
from groundcortex.pipeline.generator import ExampleGenerator
from groundcortex.pipeline.models import TrainingExample

_STATIC_DIR = Path(__file__).parent.parent / "static"
_REGULARIZATION_PATH = _STATIC_DIR / "regularization.json"


def _load_regularization(run_id: str) -> list[TrainingExample]:
    data = json.loads(_REGULARIZATION_PATH.read_text(encoding="utf-8"))
    return [
        TrainingExample(
            run_id=run_id,
            experience_id=None,
            variant="regularization",
            messages=[
                {"role": "user", "content": item["q"]},
                {"role": "assistant", "content": item["a"]},
            ],
        )
        for item in data
    ]


class CurriculumManager:
    """Builds the HuggingFace Dataset for a training run.

    For trained experiences: loads cached TrainingExample rows from the DB.
    For pending experiences: generates new rows via ExampleGenerator and saves them.
    Always appends static regularization examples (never cached).
    """

    def __init__(self, db: Database) -> None:
        self._db = db
        self._generator = ExampleGenerator()

    def build(self, run_id: str) -> tuple[Dataset, list[TrainingExample]]:
        """Return (hf_dataset, all_training_example_rows).

        The caller is responsible for saving new TrainingExample rows to the DB
        after the run_id is committed (to maintain FK integrity).
        """
        scope = self._db.get_training_scope()
        pending_ids = {exp.id for exp in scope if exp.status == "pending"}
        trained_ids = [exp.id for exp in scope if exp.status == "trained"]

        # Load cached examples for already-trained experiences
        cached = self._db.get_cached_examples(trained_ids)

        # Generate new examples for pending experiences; re-stamp with new run_id
        new_examples: list[TrainingExample] = []
        for exp in scope:
            if exp.id not in pending_ids:
                continue
            generated = self._generator.generate(exp, run_id)
            new_examples.extend(generated)

        # Regularization (always fresh, not cached)
        regularization = _load_regularization(run_id)

        all_examples = cached + new_examples + regularization

        # Re-stamp cached examples with the current run_id for the audit trail
        stamped: list[TrainingExample] = []
        for ex in cached:
            stamped.append(
                TrainingExample(
                    id=str(uuid.uuid4()),
                    run_id=run_id,
                    experience_id=ex.experience_id,
                    variant=ex.variant,
                    messages=ex.messages,
                )
            )
        all_rows = stamped + new_examples + regularization

        # Build HuggingFace dataset from messages lists
        hf_data = [{"messages": ex.messages} for ex in all_examples]
        dataset = Dataset.from_list(hf_data)

        return dataset, all_rows
