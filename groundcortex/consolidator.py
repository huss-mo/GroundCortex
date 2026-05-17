from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.file_adapter import FileAdapter
from groundcortex.pipeline.curriculum import CurriculumManager
from groundcortex.pipeline.models import TrainingRun
from groundcortex.training.trainer import LoRATrainer

if TYPE_CHECKING:
    from groundcortex.inference.manager import InferenceManager

logger = logging.getLogger(__name__)


async def run_consolidation(
    trigger: str,
    db: Database,
    config: GroundCortexConfig,
    inference_manager: "InferenceManager | None" = None,
) -> dict:
    """Full pipeline: ingest → check pending → train → hot-swap.

    Called by both the MCP tool and the cron scheduler. The `trigger` value
    ("mcp", "cron", or "manual") is recorded in training_runs for audit.

    Returns a status dict suitable for returning directly from MCP tools.
    """
    logger.info("Consolidation triggered by: %s", trigger)

    # 1. Run all ingestion adapters
    file_adapter = FileAdapter(config, db)
    new_from_files = file_adapter.ingest()

    remote_new: list = []
    if config.remote_source_urls:
        from groundcortex.ingestion.remote_adapter import RemoteFileAdapter
        remote_adapter = RemoteFileAdapter(config, db)
        remote_new = remote_adapter.ingest()

    total_new = len(new_from_files) + len(remote_new)
    logger.info("Ingestion complete. New pending experiences: %d", total_new)

    # 2. Early exit if nothing pending
    pending_count = db.count_pending()
    if pending_count == 0:
        logger.info("No pending experiences - skipping training.")
        return {
            "status": "skipped",
            "reason": "no_pending",
            "new_experiences": 0,
            "total_pending": 0,
        }

    # 3. Build training dataset
    version = db.next_version()
    trainer = LoRATrainer(config)
    generate_fn = inference_manager.generate_base if inference_manager is not None else None
    curriculum = CurriculumManager(db, generate_fn)
    dataset, all_examples = curriculum.build(run_id="placeholder")

    # Create training_run record (status=training)
    scope = db.get_training_scope()
    run = TrainingRun(
        version=version,
        trigger=trigger,  # type: ignore[arg-type]
        adapter_path="",  # filled in after training
        experience_ids=[exp.id for exp in scope],
        hyperparams=trainer.hyperparams_snapshot(),
        status="training",
    )
    db.create_training_run(run)

    # Re-stamp examples with the real run_id
    for ex in all_examples:
        ex.run_id = run.id
    db.save_training_examples(all_examples)

    # 4. Train
    # When offload_during_training=True (default): release the inference model
    # before training so only one copy of the base model is in memory at a time.
    # When False: trainer loads its own copy — inference stays live at the cost
    # of 2× base model memory.
    prev_adapter_path: str | None = None
    prev_version: str | None = None
    if inference_manager is not None and config.offload_during_training:
        active_run = db.get_active_run()
        if active_run is not None:
            prev_adapter_path = active_run.adapter_path
            prev_version = active_run.version
        inference_manager.offload()

    try:
        adapter_path = trainer.train(dataset, version)
    except Exception as exc:
        db.update_training_run(run.id, status="failed")
        logger.exception("Training failed: %s", exc)
        if inference_manager is not None and config.offload_during_training:
            inference_manager.load_base()
            if prev_adapter_path and prev_version:
                try:
                    inference_manager.load_adapter(prev_adapter_path, prev_version)
                    inference_manager.set_active(prev_version)
                except Exception:
                    logger.warning("Could not restore previous adapter after training failure.")
        return {"status": "failed", "error": str(exc)}

    # 5. Update DB: pending → trained, finalize run record
    pending_ids = [exp.id for exp in scope if exp.status == "pending"]
    db.mark_trained(pending_ids, run.id)

    from datetime import datetime, timezone
    db.update_training_run(
        run.id,
        adapter_path=adapter_path,
        status="complete",
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    db.set_active_run(run.id)

    # 6. Hot-swap new adapter (reload base first if it was offloaded)
    if inference_manager is not None:
        if config.offload_during_training:
            inference_manager.load_base()
        inference_manager.load_adapter(adapter_path, version)
        inference_manager.set_active(version)
        logger.info("Hot-swapped to adapter %s", version)

    logger.info("Consolidation complete. LoRA %s saved to %s", version, adapter_path)
    return {
        "status": "complete",
        "run_id": run.id,
        "version": version,
        "new_experiences": total_new,
        "total_experiences": len(scope),
        "adapter_path": adapter_path,
    }
