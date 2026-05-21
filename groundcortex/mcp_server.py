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
        """Call this after new knowledge has been written to source files.

        Reads all configured sources, detects changes via SHA-256 hash comparison,
        trains a new adapter from the current knowledge state if any pending content
        exists, and immediately hot-swaps it into the inference server.

        Returns status="skipped" if nothing has changed since the last run - safe to
        call redundantly. This is a long-running operation (minutes, not seconds).
        Do not call it if get_status shows pending_count=0.
        """
        from groundcortex.consolidator import run_consolidation
        return await run_consolidation("mcp", db, config, inference_manager)

    async def _get_status() -> dict:
        """Returns the current service state: active adapter version, last training
        run outcome, pending experience count, and loaded adapters.

        Use this to check whether a training run is already in progress before
        triggering another, to identify the active adapter version, or to inspect
        why a previous run succeeded or failed.
        """
        active_run = db.get_active_run()
        return {
            "active_version": inference_manager.get_active_version(),
            "model_name": config.model_name,
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

    def _complete_runs_asc(include_no_pass: bool = False):
        """Switchable runs for the current base model, sorted oldest-first."""
        return db.list_switchable_runs(include_no_pass=include_no_pass, model_name=config.model_name)

    async def _list_adapters() -> dict:
        """Lists all successfully trained adapters in chronological order (oldest first).

        Call this before switch_adapter to see what versions exist. Each entry
        includes a pre-computed negative index (-1 = most recent, -2 = one before,
        etc.) that can be passed directly to switch_adapter. Failed training runs
        are excluded from the list.
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
                "model_name": r.model_name,
            }
            for i, r in enumerate(runs)
        ]
        return {
            "versions": versions,
            "total": n,
            "active_version": inference_manager.get_active_version(),
        }

    async def _switch_adapter(version_id: str, force: bool = False) -> dict:
        """Activates a previously trained adapter by version name or negative index.

        version_id can be a version name (e.g. "v3"), a negative index as a
        string ("-1" = most recent, "-2" = one before, etc.), or "base" to
        unload LoRA and revert to the base model. Call list_adapters first to
        see available versions and their indices.

        By default, only adapters with status="complete" (passed quality gate) are
        eligible. Set force=True to also allow loading adapters with status="no-pass".
        Negative indices respect force: without force, -1 is the latest complete
        adapter; with force, -1 is the latest complete or no-pass adapter.

        Use this to roll back if a recent consolidation produced unexpected results,
        or to compare specific knowledge versions. Cannot be called while training
        is in progress.
        """
        if inference_manager.is_training:
            return {"status": "error", "message": "Cannot switch adapter: training in progress."}

        if version_id.lower() == "base":
            previous = inference_manager.get_active_version()
            inference_manager.unload_adapter()
            db.unset_active_run()
            return {"status": "ok", "active_version": None, "previous_version": previous}

        # Resolve negative index to a concrete run
        run = None
        try:
            idx = int(version_id)
            if idx < 0:
                switchable = _complete_runs_asc(include_no_pass=force)
                if abs(idx) > len(switchable):
                    return {
                        "status": "error",
                        "message": f"Index {idx} out of range: only {len(switchable)} version(s) exist.",
                    }
                run = switchable[idx]
                version_id = run.version
        except ValueError:
            pass  # not an integer - fall through to version-name lookup

        if run is None:
            run = db.get_run_by_version(version_id)
        if run is None:
            return {"status": "error", "message": f"No training run found for version '{version_id}'."}

        if run.model_name != config.model_name:
            return {
                "status": "error",
                "message": (
                    f"Adapter '{version_id}' was trained on '{run.model_name}', "
                    f"current model is '{config.model_name}'. "
                    "Adapters cannot be loaded across different base models."
                ),
            }

        if run.status == "no-pass" and not force:
            metrics = run.metrics or {}
            recall = metrics.get("recall_pct", 0.0)
            sanity = metrics.get("sanity_pct", 0.0)
            return {
                "status": "error",
                "message": (
                    f"Adapter '{version_id}' did not pass the quality gate "
                    f"(recall: {recall:.0%}, sanity: {sanity:.0%}). "
                    "Use force=True to load it anyway."
                ),
            }
        if run.status not in ("complete", "no-pass"):
            return {"status": "error", "message": f"Version '{version_id}' cannot be loaded (status: {run.status})."}

        current = inference_manager.get_active_version()
        if version_id == current:
            return {"status": "ok", "active_version": version_id, "previous_version": version_id, "noop": True}

        loaded = inference_manager.list_loaded_adapters()
        if version_id not in loaded:
            inference_manager.load_adapter(run.adapter_path, version_id)

        inference_manager.set_active(version_id)
        db.set_active_run(run.id)

        return {
            "status": "ok",
            "active_version": version_id,
            "previous_version": current,
        }

    # ── Conditional tool registration ─────────────────────────────────────────

    _all_tools = {
        "trigger_consolidation": (_trigger_consolidation, "Ingest sources and train a new LoRA if anything changed."),
        "get_status": (_get_status, "Return active adapter, pending count, and last run info."),
        "list_adapters": (_list_adapters, "List all trained adapters with their version names and negative indices."),
        "switch_adapter": (_switch_adapter, "Activate a trained adapter by version name (e.g. 'v2'), negative index (-1 = latest), or 'base' to unload LoRA."),
    }

    for name, (fn, _description) in _all_tools.items():
        if name in exposed:
            mcp.tool(name=name)(fn)
            logger.info("MCP tool registered: %s", name)
        else:
            logger.info("MCP tool excluded by config: %s", name)

    return mcp
