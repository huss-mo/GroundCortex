"""Tests for run_consolidation (consolidator.py) - trainer and inference manager are mocked."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from groundcortex.pipeline.models import Experience


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


def _add_pending(db, content="A fact.") -> Experience:
    exp = Experience(
        source="file:test.md",
        raw_content=content,
        content_hash=f"sha_{content[:8].replace(' ', '_')}",
        status="pending",
    )
    db.add_experience(exp)
    return exp


def _patch_trainer(adapter_path: str, fail: bool = False):
    """Return a context manager that patches LoRATrainer with a mock."""
    mock = MagicMock()
    instance = MagicMock()
    instance.hyperparams_snapshot.return_value = {"rank": 32}
    if fail:
        instance.train.side_effect = RuntimeError("CUDA OOM")
    else:
        instance.train.return_value = adapter_path
    mock.return_value = instance
    return patch("groundcortex.consolidator.LoRATrainer", mock), instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRunConsolidation:
    def test_no_pending_returns_skipped(self, db, config):
        from groundcortex.consolidator import run_consolidation
        result = _run(run_consolidation("mcp", db, config))
        assert result["status"] == "skipped"
        assert result["reason"] == "no_pending"

    def test_skipped_reports_zero_new_experiences(self, db, config):
        from groundcortex.consolidator import run_consolidation
        result = _run(run_consolidation("mcp", db, config))
        assert result["new_experiences"] == 0
        assert result["total_pending"] == 0

    def test_with_pending_calls_trainer(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, trainer_instance = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        trainer_instance.train.assert_called_once()

    def test_complete_run_marks_pending_as_trained(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        assert db.count_pending() == 1
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        assert db.count_pending() == 0
        scope = db.get_training_scope()
        assert all(e.status == "trained" for e in scope)

    def test_complete_run_creates_training_run_record(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        runs = db.list_runs()
        assert len(runs) == 1
        assert runs[0].status == "complete"

    def test_complete_run_sets_active_run(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        assert db.get_active_run() is not None

    def test_mcp_trigger_recorded(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        assert db.list_runs()[0].trigger == "mcp"

    def test_cron_trigger_recorded(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("cron", db, config))
        assert db.list_runs()[0].trigger == "cron"

    def test_complete_result_has_expected_keys(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        adapter = str(tmp_path / "adapters" / "v1")
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx:
            result = _run(run_consolidation("mcp", db, config))
        assert result["status"] == "complete"
        assert "run_id" in result
        assert "version" in result
        assert "adapter_path" in result

    def test_first_run_is_v1(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            result = _run(run_consolidation("mcp", db, config))
        assert result["version"] == "v1"

    def test_version_increments_on_second_run(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db, "Fact one.")
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        _add_pending(db, "Fact two.")
        patch_ctx2, _ = _patch_trainer(str(tmp_path / "adapters" / "v2"))
        with patch_ctx2:
            result = _run(run_consolidation("mcp", db, config))
        assert result["version"] == "v2"

    def test_training_failure_returns_failed_status(self, db, config):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer("", fail=True)
        with patch_ctx:
            result = _run(run_consolidation("mcp", db, config))
        assert result["status"] == "failed"
        assert "CUDA OOM" in result["error"]

    def test_training_failure_marks_run_as_failed(self, db, config):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer("", fail=True)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        runs = db.list_runs()
        assert runs[0].status == "failed"

    def test_training_failure_leaves_pending_untouched(self, db, config):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer("", fail=True)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        assert db.count_pending() == 1

    def test_inference_manager_hot_swapped_on_success(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        mock_manager = MagicMock()
        adapter = str(tmp_path / "adapters" / "v1")
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.load_adapter.assert_called_once_with(adapter, "v1")
        mock_manager.set_active.assert_called_once_with("v1")

    def test_inference_manager_none_skips_hot_swap(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            result = _run(run_consolidation("mcp", db, config, inference_manager=None))
        assert result["status"] == "complete"  # must not raise

    def test_training_examples_saved_to_db(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        import sqlite3
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        con = sqlite3.connect(str(tmp_path / "test.db"))
        count = con.execute("SELECT COUNT(*) FROM training_examples").fetchone()[0]
        con.close()
        assert count > 0
