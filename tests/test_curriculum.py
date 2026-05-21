"""Tests for CurriculumManager (pipeline/curriculum.py)."""
from __future__ import annotations

import pytest

from groundcortex.pipeline.curriculum import CurriculumManager
from groundcortex.pipeline.models import Experience, TrainingExample, TrainingRun

_REG_COUNT = 19  # number of pairs in static/regularization.json


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _add_exp(db, content, status="pending") -> Experience:
    exp = Experience(
        source="file:test.md",
        raw_content=content,
        content_hash=f"hash_{content[:8].replace(' ', '_')}",
        status=status,
    )
    db.add_experience(exp)
    return exp


def _seed_cached_example(db, exp: Experience, version="v1") -> TrainingRun:
    """Save one TrainingExample for a trained experience so get_cached_examples finds it."""
    run = TrainingRun(version=version, trigger="mcp", adapter_path="/p", status="complete", model_name="test-model")
    db.create_training_run(run)
    ex = TrainingExample(
        run_id=run.id,
        experience_id=exp.id,
        variant="direct",
        messages=[
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ],
    )
    db.save_training_examples([ex])
    return run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCurriculumManager:
    def test_all_pending_generates_five_examples_per_experience(self, db):
        _add_exp(db, "Fact one.")
        _add_exp(db, "Fact two.")
        cm = CurriculumManager(db)
        dataset, _ = cm.build("run-1")
        # 2 pending × 5 variants + 19 regularization
        assert len(dataset) == 2 * 5 + _REG_COUNT

    def test_regularization_always_included(self, db):
        _add_exp(db, "Some fact.")
        cm = CurriculumManager(db)
        _, all_rows = cm.build("run-1")
        reg_rows = [r for r in all_rows if r.variant == "regularization"]
        assert len(reg_rows) == _REG_COUNT

    def test_regularization_has_no_experience_id(self, db):
        _add_exp(db, "Some fact.")
        cm = CurriculumManager(db)
        _, all_rows = cm.build("run-1")
        for row in all_rows:
            if row.variant == "regularization":
                assert row.experience_id is None

    def test_trained_experiences_use_cached_example(self, db):
        exp = _add_exp(db, "Trained fact.", status="trained")
        _seed_cached_example(db, exp)
        cm = CurriculumManager(db)
        _, all_rows = cm.build("run-2")
        # 1 cached + 0 new (no pending) + 19 reg
        non_reg = [r for r in all_rows if r.variant != "regularization"]
        assert len(non_reg) == 1
        assert non_reg[0].experience_id == exp.id

    def test_trained_experience_cached_dataset_row_count(self, db):
        exp = _add_exp(db, "Trained fact.", status="trained")
        _seed_cached_example(db, exp)
        cm = CurriculumManager(db)
        dataset, _ = cm.build("run-2")
        # 1 cached + 19 reg
        assert len(dataset) == 1 + _REG_COUNT

    def test_mixed_scope_dataset_row_count(self, db):
        trained_exp = _add_exp(db, "Trained fact.", status="trained")
        _seed_cached_example(db, trained_exp)
        _add_exp(db, "New pending fact.", status="pending")
        cm = CurriculumManager(db)
        dataset, _ = cm.build("run-2")
        # 1 cached + 5 new + 19 reg
        assert len(dataset) == 1 + 5 + _REG_COUNT

    def test_stamped_cached_rows_get_new_run_id(self, db):
        exp = _add_exp(db, "Trained fact.", status="trained")
        _seed_cached_example(db, exp, version="v1")
        cm = CurriculumManager(db)
        _, all_rows = cm.build("run-2")
        non_reg = [r for r in all_rows if r.variant != "regularization"]
        assert all(r.run_id == "run-2" for r in non_reg)

    def test_new_examples_get_run_id(self, db):
        _add_exp(db, "Pending fact.", status="pending")
        cm = CurriculumManager(db)
        _, all_rows = cm.build("my-run")
        new_rows = [r for r in all_rows if r.variant != "regularization"]
        assert all(r.run_id == "my-run" for r in new_rows)

    def test_regularization_rows_get_current_run_id(self, db):
        _add_exp(db, "A fact.")
        cm = CurriculumManager(db)
        _, all_rows = cm.build("my-run-id")
        reg_rows = [r for r in all_rows if r.variant == "regularization"]
        assert all(r.run_id == "my-run-id" for r in reg_rows)

    def test_dataset_rows_have_messages_key(self, db):
        _add_exp(db, "Test fact.")
        cm = CurriculumManager(db)
        dataset, _ = cm.build("run-1")
        for row in dataset:
            assert "messages" in row

    def test_dataset_messages_are_two_turn_dicts(self, db):
        _add_exp(db, "Test fact.")
        cm = CurriculumManager(db)
        dataset, _ = cm.build("run-1")
        for row in dataset:
            msgs = row["messages"]
            assert isinstance(msgs, list)
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[1]["role"] == "assistant"

    def test_superseded_experiences_not_included(self, db):
        _add_exp(db, "Superseded fact.", status="superseded")
        cm = CurriculumManager(db)
        dataset, _ = cm.build("run-1")
        # Only regularization rows - no pending, no trained
        assert len(dataset) == _REG_COUNT

    def test_no_scope_only_regularization(self, db):
        cm = CurriculumManager(db)
        dataset, _ = cm.build("run-1")
        assert len(dataset) == _REG_COUNT

    def test_stamped_cached_rows_have_fresh_ids(self, db):
        exp = _add_exp(db, "Trained fact.", status="trained")
        old_run = _seed_cached_example(db, exp, version="v1")
        cm = CurriculumManager(db)
        _, all_rows = cm.build("run-2")
        non_reg = [r for r in all_rows if r.variant != "regularization"]
        # The stamped row has a new uuid - not the original row's id
        original_cached = db.get_cached_examples([exp.id])
        assert non_reg[0].id != original_cached[0].id
