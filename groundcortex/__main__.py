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
from starlette.middleware.trustedhost import TrustedHostMiddleware


def wrap_trusted_hosts(asgi_app, allowed_hosts_cfg: str):
    """Wrap an ASGI app with DNS rebinding protection.

    When allowed_hosts_cfg is non-empty, only requests whose Host header matches
    localhost, 127.0.0.1, or one of the comma-separated values in allowed_hosts_cfg
    are accepted; others receive 400. When empty, the app is returned unchanged
    (all hosts accepted - safe when the server is bound to 127.0.0.1).

    Ports are stripped from configured entries before comparison: Starlette's
    TrustedHostMiddleware compares against the hostname only (it strips the port
    from the incoming Host header), so "192.168.1.50:4343" and "192.168.1.50"
    are equivalent as configured values.
    """
    extra = [h.strip() for h in allowed_hosts_cfg.split(",") if h.strip()]
    if not extra:
        return asgi_app
    # Strip port from each entry - TrustedHostMiddleware compares hostname-only.
    hostnames = [h.split(":")[0] for h in extra]
    return TrustedHostMiddleware(
        asgi_app,
        allowed_hosts=["localhost", "127.0.0.1"] + hostnames,
    )

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
            wrap_trusted_hosts(mcp.http_app(), config.mcp_allowed_hosts),
            host=config.mcp_host,
            port=config.mcp_port,
            forwarded_allow_ips=config.mcp_forwarded_allow_ips,
            log_level="warning",
        )
    )
    inference_server = uvicorn.Server(
        uvicorn.Config(
            wrap_trusted_hosts(inference_app, config.inference_allowed_hosts),
            host=config.inference_host,
            port=config.inference_port,
            forwarded_allow_ips=config.inference_forwarded_allow_ips,
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
