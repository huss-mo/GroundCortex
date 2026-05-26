from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.pipeline.generator import ExampleGenerator
from groundcortex.pipeline.models import Experience, TrainingExample

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
    """Full pipeline: ingest → sweep → hot-swap.

    Delegates to run_sweep() which handles both single-run and multi-trial
    sweeps through a unified code path.
    """
    logger.info("Consolidation triggered by: %s", trigger)
    from groundcortex.sweep import run_sweep
    return await run_sweep(trigger, db, config, inference_manager)


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
    generate Q&A examples - zero DB interaction, no training.
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
                except Exception as exc:
                    logger.warning("Dry-run: failed to fetch %s: %s", url, exc)

    has_sources = bool(config.source_paths or config.remote_source_urls)
    if not preview_experiences:
        reason = "no_sources" if not has_sources else "fetch_failed"
        return {"status": "skipped", "reason": reason, "total_chunks": 0}

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
