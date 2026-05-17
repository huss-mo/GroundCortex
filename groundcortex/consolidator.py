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
    curriculum = CurriculumManager(db)
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
    try:
        adapter_path = trainer.train(dataset, version)
    except Exception as exc:
        db.update_training_run(run.id, status="failed")
        logger.exception("Training failed: %s", exc)
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

    # 6. Hot-swap adapter in the inference manager (if running)
    if inference_manager is not None:
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
