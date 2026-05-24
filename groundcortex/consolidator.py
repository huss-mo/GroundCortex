from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.file_adapter import FileAdapter
from groundcortex.pipeline.curriculum import CurriculumManager
from groundcortex.pipeline.generator import ExampleGenerator
from groundcortex.pipeline.models import Experience, TrainingExample, TrainingRun
from groundcortex.training.trainer import create_trainer

if TYPE_CHECKING:
    from groundcortex.inference.manager import InferenceManager


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

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
    trainer = create_trainer(config)
    generate_fn = inference_manager.generate_base if inference_manager is not None else None
    curriculum = CurriculumManager(db, generate_fn)
    dataset, all_examples, val_examples = curriculum.build(run_id="placeholder")

    # Create training_run record (status=training)
    scope = db.get_training_scope()
    run = TrainingRun(
        version=version,
        trigger=trigger,  # type: ignore[arg-type]
        adapter_path="",  # filled in after training
        experience_ids=[exp.id for exp in scope],
        hyperparams=trainer.hyperparams_snapshot(),
        model_name=config.model_name,
        status="training",
    )
    db.create_training_run(run)

    # Re-stamp all examples with the real run_id and persist
    for ex in all_examples + val_examples:
        ex.run_id = run.id
    db.save_training_examples(all_examples)
    if val_examples:
        db.save_training_examples(val_examples)

    # 4. Train
    # When offload_during_training=True (default): release the inference model
    # before training so only one copy of the base model is in memory at a time.
    # When False: trainer loads its own copy - inference stays live at the cost
    # of 2× base model memory.
    prev_adapter_path: str | None = None
    prev_version: str | None = None
    if inference_manager is not None and config.offload_during_training:
        active_run = db.get_active_run()
        if active_run is not None:
            prev_adapter_path = active_run.adapter_path
            prev_version = active_run.version
        inference_manager.offload()

    loop = asyncio.get_running_loop()
    try:
        adapter_path = await loop.run_in_executor(None, trainer.train, dataset, version)
    except Exception as exc:
        db.update_training_run(run.id, status="failed")
        logger.exception("Training failed: %s", exc)
        if inference_manager is not None and config.offload_during_training:
            await loop.run_in_executor(None, inference_manager.load_base)
            if prev_adapter_path and prev_version:
                try:
                    inference_manager.load_adapter(prev_adapter_path, prev_version)
                    inference_manager.set_active(prev_version)
                except Exception:
                    logger.warning("Could not restore previous adapter after training failure.")
        return {"status": "failed", "error": str(exc)}

    # 5. Mark pending experiences trained now that training succeeded
    pending_ids = [exp.id for exp in scope if exp.status == "pending"]
    db.mark_trained(pending_ids, run.id)

    # 6. Reload base model (if it was offloaded) - needed for both evaluation and inference
    if inference_manager is not None and config.offload_during_training:
        await loop.run_in_executor(None, inference_manager.load_base)

    # 7. Quality gate (evaluation)
    eval_metrics: dict | None = None
    if inference_manager is not None and config.eval_enabled:
        from groundcortex.evaluation.evaluator import evaluate_adapter
        eval_result = await loop.run_in_executor(
            None, evaluate_adapter,
            adapter_path, version, run.experience_ids,
            db, inference_manager, config,
        )
        eval_metrics = eval_result.as_dict()

        if not eval_result.passed:
            db.update_training_run(
                run.id,
                adapter_path=adapter_path,
                status="no-pass",
                metrics=eval_metrics,
                completed_at=datetime.now(timezone.utc).isoformat(),
            )
            logger.warning("Adapter %s did not pass quality gate - not hot-swapping.", version)
            return {
                "status": "no-pass",
                "run_id": run.id,
                "version": version,
                "metrics": eval_metrics,
                "new_experiences": total_new,
                "total_experiences": len(scope),
                "adapter_path": adapter_path,
            }

    # 8. Finalize run record (complete) and hot-swap
    db.update_training_run(
        run.id,
        adapter_path=adapter_path,
        status="complete",
        metrics=eval_metrics,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    db.set_active_run(run.id)

    if inference_manager is not None:
        # Adapter was already loaded by the evaluator (or eval was skipped - load it now)
        loaded = inference_manager.list_loaded_adapters()
        if version not in loaded:
            inference_manager.load_adapter(adapter_path, version)
        inference_manager.set_active(version)
        logger.info("Hot-swapped to adapter %s", version)

    logger.info("Consolidation complete. LoRA %s saved to %s", version, adapter_path)
    return {
        "status": "complete",
        "run_id": run.id,
        "version": version,
        "metrics": eval_metrics,
        "new_experiences": total_new,
        "total_experiences": len(scope),
        "adapter_path": adapter_path,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Dry-run: preview chunks + Q&A without touching the DB or training
# ──────────────────────────────────────────────────────────────────────────────

def _write_dry_run_report(
    results: list[tuple[Experience, list[TrainingExample]]],
    output_path: Path,
) -> None:
    lines = [
        "# GroundCortex Dry-Run Report",
        f"Generated: {_now_iso()}",
        f"Total chunks: {len(results)}",
        "",
    ]
    for i, (exp, examples) in enumerate(results, 1):
        lines += [
            f"## Chunk {i} of {len(results)}",
            f"**Source:** `{exp.source}`",
            "",
            "**Content:**",
            "",
        ]
        for line in exp.raw_content.splitlines():
            lines.append(f"> {line}")
        lines += ["", "**Generated Q&A:**", ""]
        qa_list = []
        for ex in examples:
            user_msg = next((m["content"] for m in ex.messages if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in ex.messages if m["role"] == "assistant"), "")
            qa_list.append({"question": user_msg, "answer": asst_msg, "variant": ex.variant})
        lines += ["```json", json.dumps(qa_list, indent=2, ensure_ascii=False), "```", "", "---", ""]
    output_path.write_text("\n".join(lines), encoding="utf-8")


async def run_dry_run(
    config: GroundCortexConfig,
    inference_manager: "InferenceManager | None" = None,
) -> dict:
    """Preview what would be trained: read source files, split into chunks,
    generate Q&A examples — zero DB interaction, no training.
    """
    from groundcortex.ingestion.file_adapter import _split_sections

    preview_experiences: list[Experience] = []
    now = _now_iso()

    # Local files
    for raw_path in config.source_paths:
        path = Path(raw_path).expanduser()
        if not path.exists():
            continue
        content = path.read_text(encoding="utf-8", errors="replace")
        source_id = f"file:{path}"
        for chunk in _split_sections(content, config):
            preview_experiences.append(Experience(
                source=source_id, raw_content=chunk, entities=[],
                content_hash="", status="pending", created_at=now,
            ))

    # Remote files
    if config.remote_source_urls:
        import httpx
        headers: dict[str, str] = {}
        if config.remote_source_api_key:
            headers["Authorization"] = f"Bearer {config.remote_source_api_key}"
        with httpx.Client(timeout=30.0) as client:
            for url in config.remote_source_urls:
                try:
                    resp = client.get(url, headers=headers)
                    resp.raise_for_status()
                    for chunk in _split_sections(resp.text, config):
                        preview_experiences.append(Experience(
                            source=url, raw_content=chunk, entities=[],
                            content_hash="", status="pending", created_at=now,
                        ))
                except Exception:
                    pass

    if not preview_experiences:
        return {"status": "skipped", "reason": "no_sources", "total_chunks": 0}

    gen_fn = inference_manager.generate_base if inference_manager is not None else None
    generator = ExampleGenerator(gen_fn)
    loop = asyncio.get_running_loop()

    def _generate_all() -> list[tuple[Experience, list[TrainingExample]]]:
        return [(exp, generator.generate(exp, run_id="dry-run")) for exp in preview_experiences]

    results: list[tuple[Experience, list[TrainingExample]]] = await loop.run_in_executor(
        None, _generate_all
    )

    output_path = config.root_dir / "dry-run.md"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_dry_run_report(results, output_path)

    return {
        "status": "ok",
        "total_chunks": len(preview_experiences),
        "examples_generated": sum(len(exs) for _, exs in results),
        "output_path": str(output_path),
    }
