"""Tests for Database (buffer/db.py) - SQLite CRUD layer."""
from __future__ import annotations

import pytest

from groundcortex.buffer.db import Database
from groundcortex.pipeline.models import Experience, TrainingExample, TrainingRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _exp(source="file:MEMORY.md", content="hello world", status="pending") -> Experience:
    return Experience(
        source=source,
        raw_content=content,
        content_hash=f"sha_{content[:12].replace(' ', '_')}",
        status=status,
    )


def _run(version="v1", status="complete", trigger="mcp", model_name="test-model") -> TrainingRun:
    return TrainingRun(
        version=version,
        trigger=trigger,
        adapter_path=f"/adapters/{version}",
        status=status,
        model_name=model_name,
    )


def _example(run_id: str, exp_id: str | None, variant="direct") -> TrainingExample:
    return TrainingExample(
        run_id=run_id,
        experience_id=exp_id,
        variant=variant,
        messages=[
            {"role": "user", "content": "Q?"},
            {"role": "assistant", "content": "A."},
        ],
    )


# ---------------------------------------------------------------------------
# source_files
# ---------------------------------------------------------------------------

class TestSourceFiles:
    def test_get_file_hash_unknown_returns_none(self, db):
        assert db.get_file_hash("file:missing.md") is None

    def test_upsert_then_get_returns_hash(self, db):
        db.upsert_file("file:A.md", "file", "abc123", "2026-01-01T00:00:00+00:00")
        assert db.get_file_hash("file:A.md") == "abc123"

    def test_upsert_updates_existing_hash(self, db):
        db.upsert_file("file:A.md", "file", "old_hash", "2026-01-01T00:00:00+00:00")
        db.upsert_file("file:A.md", "file", "new_hash", "2026-01-02T00:00:00+00:00")
        assert db.get_file_hash("file:A.md") == "new_hash"

    def test_different_paths_stored_independently(self, db):
        db.upsert_file("file:A.md", "file", "hash_a", "2026-01-01T00:00:00+00:00")
        db.upsert_file("file:B.md", "file", "hash_b", "2026-01-01T00:00:00+00:00")
        assert db.get_file_hash("file:A.md") == "hash_a"
        assert db.get_file_hash("file:B.md") == "hash_b"


# ---------------------------------------------------------------------------
# experiences
# ---------------------------------------------------------------------------

class TestExperiences:
    def test_count_pending_empty_db(self, db):
        assert db.count_pending() == 0

    def test_add_and_count_pending(self, db):
        db.add_experience(_exp())
        assert db.count_pending() == 1

    def test_count_pending_ignores_trained(self, db):
        db.add_experience(_exp(content="trained one", status="trained"))
        assert db.count_pending() == 0

    def test_count_pending_ignores_superseded(self, db):
        db.add_experience(_exp(content="superseded one", status="superseded"))
        assert db.count_pending() == 0

    def test_get_training_scope_includes_pending_and_trained(self, db):
        db.add_experience(_exp(content="p", status="pending"))
        db.add_experience(_exp(content="t", status="trained"))
        db.add_experience(_exp(content="s", status="superseded"))
        scope = db.get_training_scope()
        statuses = {e.status for e in scope}
        assert "pending" in statuses
        assert "trained" in statuses
        assert "superseded" not in statuses

    def test_get_pending_returns_only_pending(self, db):
        db.add_experience(_exp(content="p", status="pending"))
        db.add_experience(_exp(content="t", status="trained"))
        pending = db.get_pending()
        assert len(pending) == 1
        assert pending[0].status == "pending"

    def test_supersede_source_marks_all_from_that_source(self, db):
        db.add_experience(_exp(source="file:A.md", content="first"))
        db.add_experience(_exp(source="file:A.md", content="second"))
        db.add_experience(_exp(source="file:B.md", content="other"))
        db.supersede_source("file:A.md")
        scope = db.get_training_scope()
        sources = {e.source for e in scope}
        assert "file:A.md" not in sources
        assert "file:B.md" in sources

    def test_supersede_source_leaves_other_sources_intact(self, db):
        db.add_experience(_exp(source="file:A.md", content="a"))
        db.add_experience(_exp(source="file:B.md", content="b"))
        db.supersede_source("file:A.md")
        assert db.count_pending() == 1

    def test_mark_trained_updates_status(self, db):
        exp = _exp()
        db.add_experience(exp)
        run = _run()
        db.create_training_run(run)
        db.mark_trained([exp.id], run.id)
        assert db.count_pending() == 0
        scope = db.get_training_scope()
        assert scope[0].status == "trained"

    def test_mark_trained_sets_run_id(self, db):
        exp = _exp()
        db.add_experience(exp)
        run = _run()
        db.create_training_run(run)
        db.mark_trained([exp.id], run.id)
        scope = db.get_training_scope()
        assert scope[0].run_id == run.id

    def test_mark_trained_empty_ids_is_noop(self, db):
        db.mark_trained([], "any-run-id")  # must not raise

    def test_mark_trained_multiple_ids(self, db):
        e1 = _exp(content="first")
        e2 = _exp(content="second")
        db.add_experience(e1)
        db.add_experience(e2)
        run = _run()
        db.create_training_run(run)
        db.mark_trained([e1.id, e2.id], run.id)
        assert db.count_pending() == 0


# ---------------------------------------------------------------------------
# training_runs
# ---------------------------------------------------------------------------

class TestTrainingRuns:
    def test_next_version_starts_at_v1(self, db):
        assert db.next_version() == "v1"

    def test_next_version_increments(self, db):
        db.create_training_run(_run(version="v1"))
        assert db.next_version() == "v2"
        db.create_training_run(_run(version="v2"))
        assert db.next_version() == "v3"

    def test_get_active_run_empty_db_returns_none(self, db):
        assert db.get_active_run() is None

    def test_set_active_and_get_active_run(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        db.set_active_run(run.id)
        active = db.get_active_run()
        assert active is not None
        assert active.version == "v1"

    def test_set_active_run_deactivates_previous(self, db):
        r1 = _run(version="v1")
        r2 = _run(version="v2")
        db.create_training_run(r1)
        db.create_training_run(r2)
        db.set_active_run(r1.id)
        db.set_active_run(r2.id)
        assert db.get_active_run().version == "v2"

    def test_update_training_run_status(self, db):
        run = _run(version="v1", status="training")
        db.create_training_run(run)
        db.update_training_run(run.id, status="complete")
        assert db.get_run_by_id(run.id).status == "complete"

    def test_update_training_run_adapter_path(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        db.update_training_run(run.id, adapter_path="/new/path")
        assert db.get_run_by_id(run.id).adapter_path == "/new/path"

    def test_get_run_by_version(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        found = db.get_run_by_version("v1")
        assert found is not None
        assert found.id == run.id

    def test_get_run_by_version_missing_returns_none(self, db):
        assert db.get_run_by_version("v99") is None

    def test_get_run_by_id_missing_returns_none(self, db):
        assert db.get_run_by_id("nonexistent-id") is None

    def test_list_runs_empty(self, db):
        assert db.list_runs() == []

    def test_list_runs_ordered_most_recent_first(self, db):
        db.create_training_run(_run(version="v1"))
        db.create_training_run(_run(version="v2"))
        runs = db.list_runs()
        assert runs[0].version == "v2"
        assert runs[1].version == "v1"

    def test_unset_active_run_clears_active(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        db.set_active_run(run.id)
        assert db.get_active_run() is not None
        db.unset_active_run()
        assert db.get_active_run() is None

    def test_unset_active_run_on_empty_db_is_safe(self, db):
        db.unset_active_run()  # must not raise

    def test_mark_deleted_sets_status(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        db.mark_deleted(run.id)
        loaded = db.get_run_by_id(run.id)
        assert loaded.status == "deleted"

    def test_mark_deleted_clears_is_active(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        db.set_active_run(run.id)
        db.mark_deleted(run.id)
        assert db.get_active_run() is None

    def test_mark_deleted_run_excluded_from_complete_runs(self, db):
        r1 = _run(version="v1")
        r2 = _run(version="v2")
        db.create_training_run(r1)
        db.create_training_run(r2)
        db.mark_deleted(r1.id)
        runs = [r for r in reversed(db.list_runs()) if r.status == "complete"]
        assert all(r.version != "v1" for r in runs)

    def test_run_round_trips_trigger_field(self, db):
        run = _run(version="v1", trigger="cron")
        db.create_training_run(run)
        loaded = db.get_run_by_id(run.id)
        assert loaded.trigger == "cron"

    def test_run_metrics_stored_and_retrieved(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        db.update_training_run(run.id, metrics={"recall_pct": 95.0})
        loaded = db.get_run_by_id(run.id)
        assert loaded.metrics["recall_pct"] == pytest.approx(95.0)


# ---------------------------------------------------------------------------
# training_examples
# ---------------------------------------------------------------------------

class TestTrainingExamples:
    def test_save_and_retrieve_cached_example(self, db):
        exp = _exp(status="trained")
        db.add_experience(exp)
        run = _run(version="v1")
        db.create_training_run(run)
        ex = _example(run.id, exp.id, variant="direct")
        db.save_training_examples([ex])
        cached = db.get_cached_examples([exp.id])
        assert len(cached) == 1
        assert cached[0].experience_id == exp.id
        assert cached[0].variant == "direct"

    def test_get_cached_examples_empty_ids_returns_empty(self, db):
        assert db.get_cached_examples([]) == []

    def test_get_cached_examples_returns_latest_row_per_experience(self, db):
        exp = _exp(status="trained")
        db.add_experience(exp)
        r1 = _run(version="v1")
        r2 = _run(version="v2")
        db.create_training_run(r1)
        db.create_training_run(r2)
        ex1 = _example(r1.id, exp.id, variant="direct")
        ex2 = _example(r2.id, exp.id, variant="negative")
        db.save_training_examples([ex1, ex2])
        cached = db.get_cached_examples([exp.id])
        assert len(cached) == 1
        assert cached[0].variant == "negative"  # latest row wins

    def test_save_regularization_rows_with_no_experience_id(self, db):
        run = _run(version="v1")
        db.create_training_run(run)
        reg = _example(run.id, None, variant="regularization")
        db.save_training_examples([reg])  # must not raise

    def test_messages_round_trip_correctly(self, db):
        exp = _exp(status="trained")
        db.add_experience(exp)
        run = _run(version="v1")
        db.create_training_run(run)
        msgs = [
            {"role": "user", "content": "Describe Acme Corp."},
            {"role": "assistant", "content": "Acme Corp is a tech company."},
        ]
        ex = TrainingExample(run_id=run.id, experience_id=exp.id, variant="direct", messages=msgs)
        db.save_training_examples([ex])
        cached = db.get_cached_examples([exp.id])
        assert cached[0].messages == msgs

    def test_save_multiple_examples_at_once(self, db):
        exp = _exp(status="trained")
        db.add_experience(exp)
        run = _run(version="v1")
        db.create_training_run(run)
        examples = [_example(run.id, exp.id, v) for v in ["direct", "negative", "scenario"]]
        db.save_training_examples(examples)
        # All three are saved; get_cached returns only the last one
        cached = db.get_cached_examples([exp.id])
        assert len(cached) == 1
        assert cached[0].variant == "scenario"


# ---------------------------------------------------------------------------
# model_name tracking
# ---------------------------------------------------------------------------

class TestModelTracking:
    def test_create_run_stores_model_name(self, db):
        run = _run(version="v1", model_name="my-model")
        db.create_training_run(run)
        loaded = db.get_run_by_id(run.id)
        assert loaded.model_name == "my-model"

    def test_backfill_sets_empty_model_name_rows(self, db):
        run = _run(version="v1", model_name="test-model")
        db.create_training_run(run)
        # Simulate a legacy row by patching model_name to ''
        import sqlite3
        con = sqlite3.connect(db._path)
        con.execute("UPDATE training_runs SET model_name = '' WHERE version = 'v1'")
        con.commit()
        con.close()
        assert db.get_run_by_version("v1").model_name == ""
        db.backfill_model_name("new-model")
        assert db.get_run_by_version("v1").model_name == "new-model"

    def test_backfill_leaves_populated_rows_unchanged(self, db):
        run = _run(version="v1", model_name="existing-model")
        db.create_training_run(run)
        db.backfill_model_name("other-model")
        assert db.get_run_by_version("v1").model_name == "existing-model"

    def test_list_switchable_runs_filters_by_model_name(self, db):
        db.create_training_run(_run(version="v1", model_name="model-a"))
        db.create_training_run(_run(version="v2", model_name="model-b"))
        result = db.list_switchable_runs(model_name="model-a")
        assert len(result) == 1
        assert result[0].version == "v1"

    def test_list_switchable_runs_no_filter_returns_all(self, db):
        db.create_training_run(_run(version="v1", model_name="model-a"))
        db.create_training_run(_run(version="v2", model_name="model-b"))
        result = db.list_switchable_runs()
        assert len(result) == 2

    def test_list_switchable_runs_excludes_wrong_model(self, db):
        db.create_training_run(_run(version="v1", model_name="model-a"))
        result = db.list_switchable_runs(model_name="model-b")
        assert result == []
