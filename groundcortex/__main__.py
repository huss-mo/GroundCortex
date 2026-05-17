"""GroundCortex service entry point.

Starts three concurrent services in a single asyncio event loop:
  1. FastMCP server  - pipeline control tools for AI agents
  2. FastAPI server  - OpenAI-compatible /v1/chat/completions inference
  3. APScheduler     - cron-triggered automatic consolidation (if enabled)

Usage:
    python -m groundcortex
    python -m groundcortex --help
"""
from __future__ import annotations

import asyncio
import logging
import os

import uvicorn

# Force UTF-8 file I/O on Windows (required for TRL's Jinja template files).
os.environ.setdefault("PYTHONUTF8", "1")

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.inference.manager import InferenceManager
from groundcortex.inference_server import app as inference_app
from groundcortex.inference_server import init as init_inference
from groundcortex.mcp_server import build_mcp_server
from groundcortex.scheduler import start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("groundcortex")


async def main() -> None:
    config = GroundCortexConfig()
    db = Database(config.buffer_db)
    inference_manager = InferenceManager(config)

    # ── Load base model ────────────────────────────────────────────────────────
    logger.info("Loading base model…")
    inference_manager.load_base()

    # Auto-load the previously active adapter if one exists
    active_run = db.get_active_run()
    if active_run and active_run.status == "complete":
        try:
            inference_manager.load_adapter(active_run.adapter_path, active_run.version)
            inference_manager.set_active(active_run.version)
            logger.info("Resumed active adapter: %s", active_run.version)
        except Exception as exc:
            logger.warning("Could not load saved adapter %s: %s", active_run.version, exc)

    # ── MCP server ─────────────────────────────────────────────────────────────
    mcp = build_mcp_server(config, db, inference_manager)
    init_inference(inference_manager, config)

    # ── Cron scheduler ─────────────────────────────────────────────────────────
    async def _cron_consolidation() -> None:
        from groundcortex.consolidator import run_consolidation
        result = await run_consolidation("cron", db, config, inference_manager)
        logger.info("Cron consolidation result: %s", result)

    start_scheduler(_cron_consolidation, config)

    # ── Start both HTTP servers concurrently ───────────────────────────────────
    mcp_server = uvicorn.Server(
        uvicorn.Config(
            mcp.http_app(),
            host=config.mcp_host,
            port=config.mcp_port,
            log_level="warning",
        )
    )
    inference_server = uvicorn.Server(
        uvicorn.Config(
            inference_app,
            host=config.inference_host,
            port=config.inference_port,
            log_level="warning",
        )
    )

    logger.info(
        "MCP server:       http://%s:%d/mcp", config.mcp_host, config.mcp_port
    )
    logger.info(
        "Inference server: http://%s:%d/v1/chat/completions",
        config.inference_host,
        config.inference_port,
    )
    logger.info(
        "Exposed MCP tools: %s", ", ".join(config.mcp_exposed_tools)
    )

    await asyncio.gather(
        mcp_server.serve(),
        inference_server.serve(),
    )


def main_sync() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
