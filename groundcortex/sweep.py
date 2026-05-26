"""Hyperparameter sweep engine.

Called by run_consolidation() to execute a cartesian-product sweep over
training parameters. A single-value config is a sweep of size 1 — one
consistent code path throughout.
"""
from __future__ import annotations

import asyncio
import itertools
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.file_adapter import FileAdapter
from groundcortex.pipeline.curriculum import CurriculumManager
from groundcortex.pipeline.models import Sweep, SweepRun, TrainingRun
from groundcortex.training.trainer import create_trainer

if TYPE_CHECKING:
    from groundcortex.inference.manager import InferenceManager

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_grid(config: GroundCortexConfig) -> list[dict]:
    """Return the cartesian product of all sweepable training parameters."""
    keys = [
        "learning_rate", "epochs", "rank", "alpha",
        "batch_size", "gradient_accumulation", "num_lora_layers",
    ]
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*[getattr(config, k) for k in keys])
    ]


async def run_sweep(
    trigger: str,
    db: Database,
    config: GroundCortexConfig,
    inference_manager: "InferenceManager | None" = None,
) -> dict:
    """Run (or resume) a hyperparameter sweep and return a status dict."""
    loop = asyncio.get_running_loop()
    generate_fn = inference_manager.generate_base if inference_manager is not None else None

    # ------------------------------------------------------------------
    # A. Resume or start
    # ------------------------------------------------------------------
    active_sweep = db.get_active_sweep()
    is_resume = active_sweep is not None

    if is_resume:
        sweep = active_sweep
        sweep_runs = db.get_sweep_runs(sweep.id)
        logger.info("Resuming sweep %s (%d/%d trials)", sweep.id[:8], sweep.total, sweep.total)

        # Rebuild dataset (curriculum uses cached training_examples from DB)
        curriculum = CurriculumManager(db, generate_fn)
        dataset, all_rows, val_rows = curriculum.build(sweep.id)

        # Handle any experiences that became pending after the sweep started
        scope = db.get_training_scope()
        new_pending_ids = [e.id for e in scope if e.status == "pending"]
        if new_pending_ids:
            new_pending_set = set(new_pending_ids)
            new_training = [r for r in all_rows if r.experience_id in new_pending_set]
            new_val = [r for r in val_rows if r.experience_id in new_pending_set]
            db.save_training_examples(new_training)
            if new_val:
                db.save_training_examples(new_val)
            db.mark_trained(new_pending_ids, sweep.id)
            scope = db.get_training_scope()

        experience_ids = [e.id for e in scope]

    else:
        # ------------------------------------------------------------------
        # B. Ingest
        # ------------------------------------------------------------------
        file_adapter = FileAdapter(config, db)
        new_from_files = file_adapter.ingest()

        remote_new: list = []
        if config.remote_source_urls:
            from groundcortex.ingestion.remote_adapter import RemoteFileAdapter
            remote_adapter = RemoteFileAdapter(config, db)
            remote_new = remote_adapter.ingest()

        total_new = len(new_from_files) + len(remote_new)
        logger.info("Ingestion complete. New pending experiences: %d", total_new)

        if db.count_pending() == 0:
            logger.info("No pending experiences — skipping training.")
            return {
                "status": "skipped",
                "reason": "no_pending",
                "new_experiences": 0,
                "total_pending": 0,
            }

        # ------------------------------------------------------------------
        # C. Build training examples ONCE, shared across all trials
        # ------------------------------------------------------------------
        grid = _build_grid(config)
        sweep = Sweep(param_grid=grid, total=len(grid))
        db.create_sweep(sweep)

        sweep_runs = [
            db.create_sweep_run(SweepRun(sweep_id=sweep.id, combo_index=i, params=combo))
            for i, combo in enumerate(grid)
        ]

        curriculum = CurriculumManager(db, generate_fn)
        dataset, all_rows, val_rows = curriculum.build(sweep.id)

        scope = db.get_training_scope()
        pending_ids = [e.id for e in scope if e.status == "pending"]
        experience_ids = [e.id for e in scope]

        db.save_training_examples(all_rows)
        if val_rows:
            db.save_training_examples(val_rows)
        db.mark_trained(pending_ids, sweep.id)

    logger.info("Sweep %s: running %d trial(s)", sweep.id[:8], sweep.total)

    # ------------------------------------------------------------------
    # D. Trial loop
    # ------------------------------------------------------------------
    # Snapshot the currently active adapter so we can restore it on failure.
    prev_adapter_path: str | None = None
    prev_version: str | None = None
    if inference_manager is not None and config.offload_during_training:
        active_run = db.get_active_run()
        if active_run is not None:
            prev_adapter_path = active_run.adapter_path
            prev_version = active_run.version

    for sweep_run in sweep_runs:
        if sweep_run.status not in ("pending", "evaluating"):
            continue

        combo = sweep_run.params
        trial_config = config.for_trial(combo)
        adapter_version = f"sweep_{sweep.id[:8]}_t{sweep_run.combo_index}"

        try:
            if sweep_run.status == "pending":
                db.update_sweep_run(sweep_run.id, status="running")

                if inference_manager is not None and config.offload_during_training:
                    inference_manager.offload()

                trainer = create_trainer(trial_config)
                try:
                    adapter_path = await loop.run_in_executor(
                        None, trainer.train, dataset, adapter_version
                    )
                except Exception as exc:
                    logger.warning(
                        "Sweep %s trial %d training failed: %s",
                        sweep.id[:8], sweep_run.combo_index, exc,
                    )
                    db.update_sweep_run(sweep_run.id, status="failed", completed_at=_now())
                    if inference_manager is not None and config.offload_during_training:
                        await loop.run_in_executor(None, inference_manager.load_base)
                        if prev_adapter_path and prev_version:
                            try:
                                inference_manager.load_adapter(prev_adapter_path, prev_version)
                                inference_manager.set_active(prev_version)
                            except Exception:
                                pass
                    continue

                if inference_manager is not None and config.offload_during_training:
                    await loop.run_in_executor(None, inference_manager.load_base)

                # Set 'evaluating' before eval so a crash here resumes from eval,
                # not re-training.
                db.update_sweep_run(
                    sweep_run.id, status="evaluating", adapter_path=adapter_path
                )

            else:
                # Resuming from 'evaluating': training already done, adapter on disk.
                adapter_path = sweep_run.adapter_path  # type: ignore[assignment]

            if not adapter_path or not Path(adapter_path).exists():
                logger.warning(
                    "Sweep %s trial %d: adapter not found at '%s' — marking failed",
                    sweep.id[:8], sweep_run.combo_index, adapter_path,
                )
                db.update_sweep_run(sweep_run.id, status="failed", completed_at=_now())
                continue

            if config.eval_enabled:
                from groundcortex.evaluation.evaluator import evaluate_adapter
                try:
                    eval_result = await loop.run_in_executor(
                        None, evaluate_adapter,
                        adapter_path, adapter_version, experience_ids,
                        db, inference_manager, config,
                    )
                except Exception as exc:
                    logger.warning(
                        "Sweep %s trial %d evaluation failed: %s",
                        sweep.id[:8], sweep_run.combo_index, exc,
                    )
                    db.update_sweep_run(sweep_run.id, status="failed", completed_at=_now())
                    continue

                db.update_sweep_run(
                    sweep_run.id,
                    status="complete",
                    recall_pct=eval_result.recall_pct,
                    sanity_pct=eval_result.sanity_pct,
                    completed_at=_now(),
                )
                logger.info(
                    "Sweep %s trial %d: recall=%.0f%% sanity=%.0f%%",
                    sweep.id[:8], sweep_run.combo_index,
                    eval_result.recall_pct * 100, eval_result.sanity_pct * 100,
                )
            else:
                db.update_sweep_run(
                    sweep_run.id,
                    status="complete",
                    recall_pct=1.0,
                    sanity_pct=1.0,
                    completed_at=_now(),
                )

        except Exception as exc:
            logger.exception(
                "Sweep %s trial %d unexpected failure: %s",
                sweep.id[:8], sweep_run.combo_index, exc,
            )
            db.update_sweep_run(sweep_run.id, status="failed", completed_at=_now())

    # ------------------------------------------------------------------
    # E. Finalize: pick winner, register in training_runs, activate
    # ------------------------------------------------------------------
    final_runs = db.get_sweep_runs(sweep.id)
    completed = [r for r in final_runs if r.status == "complete"]

    if not completed:
        db.update_sweep(sweep.id, status="failed", completed_at=_now())
        logger.warning("Sweep %s: all trials failed.", sweep.id[:8])
        return {"status": "failed", "sweep_id": sweep.id}

    winner = max(completed, key=lambda r: (r.recall_pct or 0.0, r.sanity_pct or 0.0))
    passed = (
        (winner.recall_pct or 0.0) >= config.eval_validation_threshold
        and (winner.sanity_pct or 0.0) >= config.eval_sanity_threshold
    ) if config.eval_enabled else True

    version = db.next_version()
    run = TrainingRun(
        version=version,
        trigger=trigger,  # type: ignore[arg-type]
        adapter_path=winner.adapter_path or "",
        experience_ids=experience_ids,
        hyperparams=winner.params,
        model_name=config.model_name,
        metrics={
            "recall_pct": winner.recall_pct or 0.0,
            "sanity_pct": winner.sanity_pct or 0.0,
        },
        status="complete" if passed else "no-pass",
        completed_at=_now(),
    )
    db.create_training_run(run)

    if passed:
        db.set_active_run(run.id)
        if inference_manager is not None:
            loaded = inference_manager.list_loaded_adapters()
            if version not in loaded:
                inference_manager.load_adapter(winner.adapter_path, version)
            inference_manager.set_active(version)
            logger.info("Hot-swapped to adapter %s (sweep winner trial %d)", version, winner.combo_index)

    # Delete non-winning trial adapters to reclaim disk space
    for r in completed:
        if r.id != winner.id and r.adapter_path:
            shutil.rmtree(r.adapter_path, ignore_errors=True)

    db.update_sweep(sweep.id, status="complete", completed_at=_now())

    metrics = {
        "recall_pct": winner.recall_pct or 0.0,
        "sanity_pct": winner.sanity_pct or 0.0,
    }
    logger.info(
        "Sweep %s complete. Winner: trial %d → %s  recall=%.0f%%  sanity=%.0f%%",
        sweep.id[:8], winner.combo_index, version,
        (winner.recall_pct or 0.0) * 100, (winner.sanity_pct or 0.0) * 100,
    )
    return {
        "status": "complete" if passed else "no-pass",
        "version": version,
        "run_id": run.id,
        "metrics": metrics,
        "adapter_path": winner.adapter_path,
        "sweep_id": sweep.id,
        "trials_total": sweep.total,
        "trials_completed": len(completed),
        "winner_trial": winner.combo_index,
        "winner_params": winner.params,
    }
