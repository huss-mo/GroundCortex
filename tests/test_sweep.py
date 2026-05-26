"""Tests for the hyperparameter sweep engine (sweep.py)."""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, call, patch

import pytest

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.pipeline.models import Experience, Sweep, SweepRun


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
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
    """Patch create_trainer in sweep module."""
    from pathlib import Path
    if not fail and adapter_path:
        Path(adapter_path).mkdir(parents=True, exist_ok=True)
    mock = MagicMock()
    instance = MagicMock()
    instance.hyperparams_snapshot.return_value = {"rank": 32}
    if fail:
        instance.train.side_effect = RuntimeError("OOM")
    else:
        instance.train.return_value = adapter_path
    mock.return_value = instance
    return patch("groundcortex.sweep.create_trainer", mock), instance


def _patch_eval(recall: float = 0.9, sanity: float = 0.9, passed: bool = True):
    """Patch evaluate_adapter in sweep module."""
    from groundcortex.evaluation.evaluator import EvaluationResult
    result = EvaluationResult(
        passed=passed,
        recall_pct=recall,
        sanity_pct=sanity,
        probe_count=5,
        sanity_count=3,
    )
    return patch("groundcortex.evaluation.evaluator.evaluate_adapter",return_value=result)


def _patch_ingest(new_count: int = 1):
    """Patch FileAdapter.ingest to return N fake new experiences."""
    return patch(
        "groundcortex.sweep.FileAdapter.ingest",
        return_value=[MagicMock()] * new_count,
    )


def _cfg(tmp_path, **kwargs) -> GroundCortexConfig:
    kwargs.setdefault("root_dir", tmp_path)
    kwargs.setdefault("eval_enabled", False)
    kwargs.setdefault("model_name", "test-model")
    kwargs.setdefault("source_paths", [])
    kwargs.setdefault("remote_source_urls", [])
    kwargs.setdefault("offload_during_training", False)
    return GroundCortexConfig(_env_file=None, **kwargs)


# ---------------------------------------------------------------------------
# _build_grid
# ---------------------------------------------------------------------------

class TestBuildGrid:
    def test_single_values_produce_one_combo(self, tmp_path):
        from groundcortex.sweep import _build_grid
        cfg = _cfg(tmp_path)
        grid = _build_grid(cfg)
        assert len(grid) == 1

    def test_two_values_on_one_param_produce_two_combos(self, tmp_path):
        from groundcortex.sweep import _build_grid
        cfg = _cfg(tmp_path, learning_rate=[1e-5, 5e-6])
        grid = _build_grid(cfg)
        assert len(grid) == 2

    def test_cartesian_product(self, tmp_path):
        from groundcortex.sweep import _build_grid
        cfg = _cfg(tmp_path, learning_rate=[1e-5, 5e-6], epochs=[10, 15, 20])
        grid = _build_grid(cfg)
        assert len(grid) == 6

    def test_grid_entries_are_dicts_with_all_keys(self, tmp_path):
        from groundcortex.sweep import _build_grid
        cfg = _cfg(tmp_path)
        grid = _build_grid(cfg)
        expected_keys = {
            "learning_rate", "epochs", "rank", "alpha",
            "batch_size", "gradient_accumulation", "num_lora_layers",
        }
        assert set(grid[0].keys()) == expected_keys


# ---------------------------------------------------------------------------
# run_sweep — no pending → skipped
# ---------------------------------------------------------------------------

class TestRunSweepSkipped:
    def test_no_pending_returns_skipped(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        cfg = _cfg(tmp_path)
        with _patch_ingest(0):
            result = _run(run_sweep("mcp", db, cfg))
        assert result["status"] == "skipped"
        assert result["reason"] == "no_pending"


# ---------------------------------------------------------------------------
# run_sweep — single trial (sweep of 1)
# ---------------------------------------------------------------------------

class TestSingleTrial:
    def test_single_combo_creates_one_sweep_run(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        cfg = _cfg(tmp_path)
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "v_t0"))
        with _patch_ingest(0), patch_ctx:
            _run(run_sweep("mcp", db, cfg))
        sweep = db.get_active_sweep()
        assert sweep is None  # sweep finalized → status=complete
        # The sweep should exist in DB (status=complete now)
        with db._conn() as con:
            row = con.execute("SELECT * FROM sweeps").fetchone()
        assert row is not None
        assert row["total"] == 1

    def test_single_combo_creates_training_run_record(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        cfg = _cfg(tmp_path)
        _add_pending(db)
        patch_ctx, _ = _patch_trainer(str(tmp_path / "v_t0"))
        with _patch_ingest(0), patch_ctx:
            result = _run(run_sweep("mcp", db, cfg))
        assert result["status"] in ("complete", "no-pass")
        run = db.get_run_by_version(result["version"])
        assert run is not None

    def test_training_examples_generated_once(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        cfg = _cfg(tmp_path, learning_rate=[1e-5, 5e-6])
        _add_pending(db)
        t0 = tmp_path / "t0"
        t1 = tmp_path / "t1"
        t0.mkdir()
        t1.mkdir()
        with _patch_ingest(0), \
             patch("groundcortex.sweep.CurriculumManager") as mock_cm, \
             patch("groundcortex.sweep.create_trainer") as mock_ct:
            import datasets
            mock_instance = MagicMock()
            mock_instance.build.return_value = (
                datasets.Dataset.from_list([{"messages": []}]),
                [],
                [],
            )
            mock_cm.return_value = mock_instance
            mock_ct.return_value.train.side_effect = [str(t0), str(t1)]
            _run(run_sweep("mcp", db, cfg))
        mock_instance.build.assert_called_once()


# ---------------------------------------------------------------------------
# run_sweep — multi-trial selection
# ---------------------------------------------------------------------------

class TestMultiTrialSelection:
    def test_best_recall_wins(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        from groundcortex.evaluation.evaluator import EvaluationResult

        cfg = _cfg(tmp_path, learning_rate=[1e-5, 5e-6], eval_enabled=True)
        _add_pending(db)

        call_count = 0
        recalls = [0.4, 0.8]

        def fake_eval(adapter_path, version, exp_ids, _db, _im, _cfg):
            nonlocal call_count
            r = recalls[call_count % len(recalls)]
            call_count += 1
            return EvaluationResult(
                passed=r >= 0.6, recall_pct=r, sanity_pct=0.9,
                probe_count=5, sanity_count=3,
            )

        t0 = tmp_path / "trial_0"
        t1 = tmp_path / "trial_1"
        t0.mkdir()
        t1.mkdir()

        trainer_mock = MagicMock()
        trainer_mock.return_value.train.side_effect = [str(t0), str(t1)]

        with _patch_ingest(0), \
             patch("groundcortex.sweep.create_trainer", trainer_mock), \
             patch("groundcortex.evaluation.evaluator.evaluate_adapter",side_effect=fake_eval):
            result = _run(run_sweep("mcp", db, cfg))

        assert result["metrics"]["recall_pct"] == pytest.approx(0.8)

    def test_winner_set_active_when_passed(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        from groundcortex.evaluation.evaluator import EvaluationResult

        cfg = _cfg(
            tmp_path,
            eval_enabled=True,
            eval_validation_threshold=0.6,
            eval_sanity_threshold=0.6,
        )
        _add_pending(db)
        (tmp_path / "trial_0").mkdir()

        trainer_mock = MagicMock()
        trainer_mock.return_value.train.return_value = str(tmp_path / "trial_0")

        passing_result = EvaluationResult(
            passed=True, recall_pct=0.8, sanity_pct=0.9,
            probe_count=5, sanity_count=3,
        )

        inference_mock = MagicMock()
        inference_mock.generate_base = None
        inference_mock.list_loaded_adapters.return_value = []

        with _patch_ingest(0), \
             patch("groundcortex.sweep.create_trainer", trainer_mock), \
             patch("groundcortex.evaluation.evaluator.evaluate_adapter",return_value=passing_result):
            result = _run(run_sweep("mcp", db, cfg, inference_mock))

        assert result["status"] == "complete"
        assert db.get_active_run() is not None

    def test_winner_not_active_when_no_pass(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        from groundcortex.evaluation.evaluator import EvaluationResult

        cfg = _cfg(
            tmp_path,
            eval_enabled=True,
            eval_validation_threshold=0.9,
            eval_sanity_threshold=0.9,
        )
        _add_pending(db)
        (tmp_path / "trial_0").mkdir()

        trainer_mock = MagicMock()
        trainer_mock.return_value.train.return_value = str(tmp_path / "trial_0")

        no_pass_result = EvaluationResult(
            passed=False, recall_pct=0.5, sanity_pct=0.5,
            probe_count=5, sanity_count=3,
        )

        with _patch_ingest(0), \
             patch("groundcortex.sweep.create_trainer", trainer_mock), \
             patch("groundcortex.evaluation.evaluator.evaluate_adapter",return_value=no_pass_result):
            result = _run(run_sweep("mcp", db, cfg))

        assert result["status"] == "no-pass"
        assert db.get_active_run() is None
        # Record still created in training_runs
        run = db.get_run_by_version(result["version"])
        assert run is not None
        assert run.status == "no-pass"


# ---------------------------------------------------------------------------
# run_sweep — failed trials
# ---------------------------------------------------------------------------

class TestFailedTrials:
    def test_failed_trial_skipped_sweep_continues(self, db, tmp_path):
        from groundcortex.sweep import run_sweep

        cfg = _cfg(tmp_path, learning_rate=[1e-5, 5e-6])
        _add_pending(db)

        # First trial fails, second succeeds
        trainer_mock = MagicMock()
        trainer_mock.return_value.train.side_effect = [
            RuntimeError("OOM"),
            str(tmp_path / "trial_1"),
        ]
        (tmp_path / "trial_1").mkdir()

        with _patch_ingest(0), patch("groundcortex.sweep.create_trainer", trainer_mock):
            result = _run(run_sweep("mcp", db, cfg))

        assert result["status"] in ("complete", "no-pass")
        assert result["trials_completed"] == 1

    def test_all_trials_failed_returns_failed(self, db, tmp_path):
        from groundcortex.sweep import run_sweep

        cfg = _cfg(tmp_path, learning_rate=[1e-5, 5e-6])
        _add_pending(db)

        trainer_mock = MagicMock()
        trainer_mock.return_value.train.side_effect = RuntimeError("OOM")

        with _patch_ingest(0), patch("groundcortex.sweep.create_trainer", trainer_mock):
            result = _run(run_sweep("mcp", db, cfg))

        assert result["status"] == "failed"
        # No training_run record created
        assert db.list_runs() == []

    def test_non_winning_adapters_deleted(self, db, tmp_path):
        from groundcortex.sweep import run_sweep
        from groundcortex.evaluation.evaluator import EvaluationResult
        import shutil

        cfg = _cfg(
            tmp_path,
            learning_rate=[1e-5, 5e-6],
            eval_enabled=True,
        )
        _add_pending(db)

        t0 = tmp_path / "trial_0"
        t1 = tmp_path / "trial_1"
        t0.mkdir()
        t1.mkdir()

        recalls = [0.4, 0.8]
        call_count = 0

        def fake_eval(adapter_path, version, exp_ids, _db, _im, _cfg):
            nonlocal call_count
            r = recalls[call_count % len(recalls)]
            call_count += 1
            return EvaluationResult(
                passed=r >= 0.6, recall_pct=r, sanity_pct=0.9,
                probe_count=5, sanity_count=3,
            )

        trainer_mock = MagicMock()
        trainer_mock.return_value.train.side_effect = [str(t0), str(t1)]

        with _patch_ingest(0), \
             patch("groundcortex.sweep.create_trainer", trainer_mock), \
             patch("groundcortex.evaluation.evaluator.evaluate_adapter",side_effect=fake_eval):
            _run(run_sweep("mcp", db, cfg))

        # Non-winning trial_0 should be deleted
        assert not t0.exists()
        # Winning trial_1 should still exist
        assert t1.exists()


# ---------------------------------------------------------------------------
# run_sweep — resume
# ---------------------------------------------------------------------------

class TestResume:
    def test_resume_skips_completed_trials(self, db, tmp_path):
        from groundcortex.sweep import run_sweep

        # Set up a sweep that has 2 trials; first is complete, second is pending
        grid = [{"learning_rate": 1e-5, "epochs": 10, "rank": 32, "alpha": 64,
                  "batch_size": 2, "gradient_accumulation": 2, "num_lora_layers": 0},
                {"learning_rate": 5e-6, "epochs": 10, "rank": 32, "alpha": 64,
                  "batch_size": 2, "gradient_accumulation": 2, "num_lora_layers": 0}]
        sweep = Sweep(param_grid=grid, total=2)
        db.create_sweep(sweep)

        (tmp_path / "t0_done").mkdir()
        sr0 = SweepRun(
            sweep_id=sweep.id, combo_index=0, params=grid[0],
            status="complete", recall_pct=0.7, sanity_pct=0.8,
            adapter_path=str(tmp_path / "t0_done"),
        )
        sr1 = SweepRun(sweep_id=sweep.id, combo_index=1, params=grid[1], status="pending")
        db.create_sweep_run(sr0)
        db.create_sweep_run(sr1)

        # Add a trained experience (sweep already ran on it)
        exp = Experience(
            source="file:test.md", raw_content="fact",
            content_hash="sha_fact", status="trained", run_id=sweep.id,
        )
        db.add_experience(exp)

        cfg = _cfg(tmp_path)
        trainer_mock = MagicMock()
        trainer_mock.return_value.train.return_value = str(tmp_path / "t1")
        (tmp_path / "t1").mkdir()

        with patch("groundcortex.sweep.create_trainer", trainer_mock):
            result = _run(run_sweep("mcp", db, cfg))

        # Trainer should only have been called once (for trial 1, not trial 0)
        assert trainer_mock.return_value.train.call_count == 1
        assert result["status"] in ("complete", "no-pass")

    def test_resume_evaluating_trial_skips_training(self, db, tmp_path):
        from groundcortex.sweep import run_sweep

        grid = [{"learning_rate": 1e-5, "epochs": 10, "rank": 32, "alpha": 64,
                  "batch_size": 2, "gradient_accumulation": 2, "num_lora_layers": 0}]
        sweep = Sweep(param_grid=grid, total=1)
        db.create_sweep(sweep)

        adapter_dir = tmp_path / "t0"
        adapter_dir.mkdir()
        sr = SweepRun(
            sweep_id=sweep.id, combo_index=0, params=grid[0],
            status="evaluating", adapter_path=str(adapter_dir),
        )
        db.create_sweep_run(sr)

        exp = Experience(
            source="file:test.md", raw_content="fact",
            content_hash="sha_fact", status="trained", run_id=sweep.id,
        )
        db.add_experience(exp)

        cfg = _cfg(tmp_path)
        trainer_mock = MagicMock()

        with patch("groundcortex.sweep.create_trainer", trainer_mock):
            result = _run(run_sweep("mcp", db, cfg))

        # Training should NOT have been called (adapter already on disk)
        trainer_mock.return_value.train.assert_not_called()
        assert result["status"] in ("complete", "no-pass")
