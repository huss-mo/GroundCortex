"""Tests for run_consolidation (consolidator.py) - trainer and inference manager are mocked."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from groundcortex.pipeline.models import Experience, TrainingRun


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
    """Return a context manager that patches create_trainer with a mock."""
    from pathlib import Path
    if not fail and adapter_path:
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
    mock = MagicMock()
    instance = MagicMock()
    instance.hyperparams_snapshot.return_value = {"rank": 32}
    if fail:
        instance.train.side_effect = RuntimeError("CUDA OOM")
    else:
        instance.train.return_value = adapter_path
    mock.return_value = instance
    # consolidation now delegates to sweep.py, so patch there
    return patch("groundcortex.sweep.create_trainer", mock), instance


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

    def test_cli_trigger_recorded(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("cli", db, config))
        assert db.list_runs()[0].trigger == "cli"

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

    def test_training_failure_returns_no_training_run(self, db, config):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer("", fail=True)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        # All trials failed → sweep failed → no training_run record created
        assert db.list_runs() == []

    def test_training_failure_leaves_pending_untouched(self, db, config):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer("", fail=True)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        assert len(db.get_training_scope()) == 1

    def test_inference_manager_hot_swapped_on_success(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        mock_manager = MagicMock()
        mock_manager.list_loaded_adapters.return_value = []
        adapter = str(tmp_path / "adapters" / "v1")
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.load_adapter.assert_called_once_with(adapter, "v1")
        mock_manager.set_active.assert_called_once_with("v1")

    def test_uses_generate_base_not_generate(self, db, config, tmp_path):
        from unittest.mock import patch
        from groundcortex.consolidator import run_consolidation
        from groundcortex.pipeline.curriculum import CurriculumManager
        _add_pending(db)
        mock_manager = MagicMock()
        captured = {}
        original_init = CurriculumManager.__init__
        def capturing_init(self, _db, generate_fn=None):
            captured["generate_fn"] = generate_fn
            original_init(self, _db, generate_fn)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx, patch.object(CurriculumManager, "__init__", capturing_init):
            _run(run_consolidation("mcp", db, config, mock_manager))
        assert captured["generate_fn"] is mock_manager.generate_base

    def test_inference_manager_offloaded_before_training(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        mock_manager = MagicMock()
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.offload.assert_called_once()

    def test_inference_manager_reloaded_after_training(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        mock_manager = MagicMock()
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.load_base.assert_called_once()

    def test_inference_manager_reloaded_after_training_failure(self, db, config):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        mock_manager = MagicMock()
        patch_ctx, _ = _patch_trainer("", fail=True)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.offload.assert_called_once()
        mock_manager.load_base.assert_called_once()

    def test_inference_manager_none_skips_hot_swap(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            result = _run(run_consolidation("mcp", db, config, inference_manager=None))
        assert result["status"] == "complete"  # must not raise

    def test_no_offload_when_offload_disabled(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        config.offload_during_training = False
        _add_pending(db)
        mock_manager = MagicMock()
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.offload.assert_not_called()

    def test_no_load_base_when_offload_disabled(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        config.offload_during_training = False
        _add_pending(db)
        mock_manager = MagicMock()
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.load_base.assert_not_called()

    def test_hot_swap_still_works_when_offload_disabled(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        config.offload_during_training = False
        _add_pending(db)
        mock_manager = MagicMock()
        adapter = str(tmp_path / "adapters" / "v1")
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx:
            _run(run_consolidation("mcp", db, config, mock_manager))
        mock_manager.load_adapter.assert_called_once_with(adapter, "v1")
        mock_manager.set_active.assert_called_once_with("v1")

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


# ---------------------------------------------------------------------------
# Quality gate integration
# ---------------------------------------------------------------------------

class TestQualityGate:
    def test_no_pass_does_not_hot_swap(self, db, config, tmp_path):
        from unittest.mock import patch
        from groundcortex.consolidator import run_consolidation
        from groundcortex.evaluation.evaluator import EvaluationResult
        config.eval_enabled = True
        _add_pending(db)
        mock_manager = MagicMock()
        adapter = str(tmp_path / "adapters" / "v1")
        no_pass = EvaluationResult(passed=False, recall_pct=0.3, sanity_pct=0.4, probe_count=5, sanity_count=5)
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx, patch("groundcortex.evaluation.evaluator.evaluate_adapter", return_value=no_pass):
            result = _run(run_consolidation("mcp", db, config, mock_manager))
        assert result["status"] == "no-pass"
        assert db.list_runs()[0].status == "no-pass"
        mock_manager.set_active.assert_not_called()
        assert db.get_active_run() is None

    def test_no_pass_result_includes_metrics(self, db, config, tmp_path):
        from unittest.mock import patch
        from groundcortex.consolidator import run_consolidation
        from groundcortex.evaluation.evaluator import EvaluationResult
        config.eval_enabled = True
        _add_pending(db)
        mock_manager = MagicMock()
        adapter = str(tmp_path / "adapters" / "v1")
        no_pass = EvaluationResult(passed=False, recall_pct=0.3, sanity_pct=0.4, probe_count=5, sanity_count=5)
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx, patch("groundcortex.evaluation.evaluator.evaluate_adapter", return_value=no_pass):
            result = _run(run_consolidation("mcp", db, config, mock_manager))
        assert "metrics" in result
        assert result["metrics"]["recall_pct"] == pytest.approx(0.3)

    def test_pass_completes_and_hot_swaps(self, db, config, tmp_path):
        from unittest.mock import patch
        from groundcortex.consolidator import run_consolidation
        from groundcortex.evaluation.evaluator import EvaluationResult
        config.eval_enabled = True
        _add_pending(db)
        mock_manager = MagicMock()
        mock_manager.list_loaded_adapters.return_value = []
        adapter = str(tmp_path / "adapters" / "v1")
        pass_result = EvaluationResult(passed=True, recall_pct=0.9, sanity_pct=0.9, probe_count=5, sanity_count=5)
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx, patch("groundcortex.evaluation.evaluator.evaluate_adapter", return_value=pass_result):
            result = _run(run_consolidation("mcp", db, config, mock_manager))
        assert result["status"] == "complete"
        assert db.list_runs()[0].status == "complete"
        mock_manager.set_active.assert_called_once_with("v1")
        assert db.get_active_run() is not None

    def test_model_name_recorded_on_training_run(self, db, config, tmp_path):
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "adapters" / "v1"))
        with patch_ctx:
            _run(run_consolidation("mcp", db, config))
        runs = db.list_runs()
        assert runs[0].model_name == config.model_name

    def test_eval_disabled_skips_quality_gate(self, db, config, tmp_path):
        from unittest.mock import patch
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        mock_manager = MagicMock()
        mock_manager.list_loaded_adapters.return_value = []
        adapter = str(tmp_path / "adapters" / "v1")
        patch_ctx, _ = _patch_trainer(adapter)
        with patch_ctx, patch("groundcortex.evaluation.evaluator.evaluate_adapter") as mock_eval:
            result = _run(run_consolidation("mcp", db, config, mock_manager))
        # eval_enabled=False (from conftest fixture) - evaluator must not be called
        mock_eval.assert_not_called()
        assert result["status"] == "complete"

    def test_eval_crash_marks_trial_failed_sweep_continues(self, db, config, tmp_path):
        from unittest.mock import patch
        from groundcortex.consolidator import run_consolidation
        _add_pending(db)
        adapter = str(tmp_path / "adapters" / "v1")
        patch_ctx, _ = _patch_trainer(adapter)
        mock_manager = MagicMock()
        mock_manager.list_loaded_adapters.return_value = []
        config.eval_enabled = True
        with patch_ctx, patch(
            "groundcortex.evaluation.evaluator.evaluate_adapter",
            side_effect=RuntimeError("OOM"),
        ):
            result = _run(run_consolidation("mcp", db, config, mock_manager))
        # All trials failed → sweep returns failed, no training_run created
        assert result["status"] == "failed"
        assert db.list_runs() == []
