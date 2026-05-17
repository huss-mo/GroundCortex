from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fastmcp import FastMCP

from groundcortex.config import GroundCortexConfig

if TYPE_CHECKING:
    from groundcortex.buffer.db import Database
    from groundcortex.consolidator import run_consolidation as _run_consolidation
    from groundcortex.inference.manager import InferenceManager

logger = logging.getLogger(__name__)


def build_mcp_server(
    config: GroundCortexConfig,
    db: "Database",
    inference_manager: "InferenceManager",
) -> FastMCP:
    """Construct the FastMCP server with tools gated by GROUNDCORTEX_MCP_EXPOSED_TOOLS."""

    mcp = FastMCP("GroundCortex")

    exposed = set(config.mcp_exposed_tools)

    # ── Tool definitions ───────────────────────────────────────────────────────

    async def _trigger_consolidation() -> dict:
        """Ingest all sources, train a new LoRA if anything changed, hot-swap adapter."""
        from groundcortex.consolidator import run_consolidation
        return await run_consolidation("mcp", db, config, inference_manager)

    async def _get_cortex_status() -> dict:
        """Return active adapter version, pending count, last run metrics, loaded adapters."""
        active_run = db.get_active_run()
        return {
            "active_version": inference_manager.get_active_version(),
            "pending_count": db.count_pending(),
            "loaded_adapters": inference_manager.list_loaded_adapters(),
            "last_run": {
                "id": active_run.id,
                "version": active_run.version,
                "status": active_run.status,
                "trigger": active_run.trigger,
                "metrics": active_run.metrics,
                "completed_at": active_run.completed_at,
            } if active_run else None,
        }

    def _complete_runs_asc():
        """Complete training runs sorted oldest-first for index resolution."""
        return [r for r in reversed(db.list_runs()) if r.status == "complete"]

    async def _list_adapters() -> dict:
        """List all successfully trained adapters.

        Each entry includes a pre-computed negative index so agents can pass
        -1 (most recent), -2 (one before), etc. to switch_adapter without
        needing to know exact version names.
        """
        runs = _complete_runs_asc()
        n = len(runs)
        versions = [
            {
                "version": r.version,
                "is_active": r.is_active,
                "trigger": r.trigger,
                "created_at": r.created_at,
                "completed_at": r.completed_at,
                "index": i - n,
            }
            for i, r in enumerate(runs)
        ]
        return {
            "versions": versions,
            "total": n,
            "active_version": inference_manager.get_active_version(),
        }

    async def _switch_adapter(version_id: str) -> dict:
        """Activate a previously trained adapter by version ID or negative index.

        version_id can be a version name (e.g. "v3") or a negative index
        (-1 = most recent, -2 = one before, etc.). Use list_adapters to
        see all available versions and their indices.
        """
        if inference_manager.is_training:
            return {"status": "error", "message": "Cannot switch adapter: training in progress."}

        # Resolve negative index to a concrete run
        run = None
        try:
            idx = int(version_id)
            if idx < 0:
                complete = _complete_runs_asc()
                if abs(idx) > len(complete):
                    return {
                        "status": "error",
                        "message": f"Index {idx} out of range: only {len(complete)} complete version(s) exist.",
                    }
                run = complete[idx]
                version_id = run.version
        except ValueError:
            pass  # not an integer — fall through to version-name lookup

        if run is None:
            run = db.get_run_by_version(version_id)
        if run is None:
            return {"status": "error", "message": f"No training run found for version '{version_id}'."}
        if run.status != "complete":
            return {"status": "error", "message": f"Version '{version_id}' is not complete (status: {run.status})."}

        loaded = inference_manager.list_loaded_adapters()
        if version_id not in loaded:
            inference_manager.load_adapter(run.adapter_path, version_id)

        previous = inference_manager.get_active_version()
        inference_manager.set_active(version_id)
        db.set_active_run(run.id)

        return {
            "status": "ok",
            "active_version": version_id,
            "previous_version": previous,
        }

    # ── Conditional tool registration ─────────────────────────────────────────

    _all_tools = {
        "trigger_consolidation": (_trigger_consolidation, "Ingest sources and train a new LoRA if anything changed."),
        "get_cortex_status": (_get_cortex_status, "Return active adapter, pending count, and last run info."),
        "list_adapters": (_list_adapters, "List all trained adapters with their version names and negative indices."),
        "switch_adapter": (_switch_adapter, "Activate a trained adapter by version name (e.g. 'v2') or negative index (-1 = latest)."),
    }

    for name, (fn, _description) in _all_tools.items():
        if name in exposed:
            mcp.tool(name=name)(fn)
            logger.info("MCP tool registered: %s", name)
        else:
            logger.info("MCP tool excluded by config: %s", name)

    return mcp
