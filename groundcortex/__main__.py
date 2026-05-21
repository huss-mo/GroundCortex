"""GroundCortex service entry point.

Starts three concurrent services in a single asyncio event loop:
  1. FastMCP server  - pipeline control tools for AI agents
  2. FastAPI server  - OpenAI-compatible /v1/chat/completions inference
  3. APScheduler     - cron-triggered automatic consolidation (if enabled)

Usage:
    python -m groundcortex                    # start the server
    python -m groundcortex --switch v2        # switch active adapter
    python -m groundcortex --switch -1        # switch to latest adapter
    python -m groundcortex --switch base      # unload LoRA, use base model
    python -m groundcortex --list             # list trained adapters
    python -m groundcortex --status           # show server status
    python -m groundcortex --delete v1        # soft-delete an adapter
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

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
from groundcortex.inference.manager import create_manager
from groundcortex.inference_server import app as inference_app
from groundcortex.inference_server import init as init_inference
from groundcortex.mcp_server import build_mcp_server
from groundcortex.scheduler import start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("groundcortex")


# ──────────────────────────────────────────────────────────────────────────────
# CLI helper functions (no server required except --switch)
# ──────────────────────────────────────────────────────────────────────────────

def _complete_runs_asc(db, include_no_pass: bool = False, model_name: str | None = None):
    """Switchable runs sorted oldest-first, optionally filtered by base model."""
    return db.list_switchable_runs(include_no_pass=include_no_pass, model_name=model_name)


def _resolve_version(version_arg: str, runs):
    """Return the TrainingRun matching a version name or negative index, or None."""
    try:
        idx = int(version_arg)
        if idx < 0:
            return runs[idx] if abs(idx) <= len(runs) else None
    except ValueError:
        pass
    return next((r for r in runs if r.version == version_arg), None)


def _cli_switch(config, version: str, force: bool = False) -> None:
    import json
    import urllib.error
    import urllib.request

    # Use 127.0.0.1 when the configured host is 0.0.0.0 (bind-all address).
    connect_host = "127.0.0.1" if config.inference_host in ("0.0.0.0", "::") else config.inference_host
    url = f"http://{connect_host}:{config.inference_port}/v1/control/switch"
    headers = {"Content-Type": "application/json"}
    if config.inference_api_key:
        headers["Authorization"] = f"Bearer {config.inference_api_key}"

    data = json.dumps({"version": version, "force": force}).encode()
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        active = body.get("active_version") or "base"
        prev = body.get("previous_version") or "base"
        if body.get("noop"):
            print(f"Already active: {active}")
        else:
            print(f"Switched: {prev} → {active}")
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        try:
            detail = json.loads(raw).get("detail", exc.reason)
        except Exception:
            detail = exc.reason or f"HTTP {exc.code}"
        print(f"Error: {detail}", file=sys.stderr)
        sys.exit(1)
    except (urllib.error.URLError, OSError):
        print(
            "Error: Server is not running. Start it with: python -m groundcortex",
            file=sys.stderr,
        )
        sys.exit(1)


def _cli_delete(config, version_arg: str) -> None:
    import shutil

    db = Database(config.buffer_db)
    runs = _complete_runs_asc(db, include_no_pass=True)
    run = _resolve_version(version_arg, runs)
    if run is None:
        print(f"Error: No adapter found for '{version_arg}'.", file=sys.stderr)
        sys.exit(1)
    if run.is_active:
        print(
            f"Error: Adapter '{run.version}' is currently active. "
            "Switch to another adapter or base first:\n"
            f"  python -m groundcortex --switch base",
            file=sys.stderr,
        )
        sys.exit(1)
    db.mark_deleted(run.id)
    try:
        shutil.rmtree(run.adapter_path, ignore_errors=False)
    except FileNotFoundError:
        pass
    print(f"Deleted adapter {run.version} ({run.adapter_path}).")


def _cli_list(config) -> None:
    db = Database(config.buffer_db)
    # Show all adapters (no model filter) so history across base models is visible
    runs = _complete_runs_asc(db, include_no_pass=True)
    if not runs:
        print("No trained adapters.")
        return
    n = len(runs)
    print(f"{'INDEX':>6}  {'VERSION':<10}  {'STATUS':<10}  {'COMPAT':<6}  {'ACTIVE':<6}  {'MODEL':<35}  CREATED")
    for i, run in enumerate(runs):
        idx = i - n
        active_flag = "yes" if run.is_active else ""
        compat = "ok" if run.model_name == config.model_name else "!"
        model = run.model_name[:35]
        print(f"{idx:>6}  {run.version:<10}  {run.status:<10}  {compat:<6}  {active_flag:<6}  {model:<35}  {run.created_at}")


def _cli_status(config) -> None:
    db = Database(config.buffer_db)
    active = db.get_active_run()
    pending = db.count_pending()
    # Count only adapters compatible with the current base model
    runs = _complete_runs_asc(db, model_name=config.model_name)
    print(f"Base model     : {config.model_name}")
    print(f"Active adapter : {active.version if active else 'none (base model)'}")
    print(f"Pending count  : {pending}")
    print(f"Total adapters : {len(runs)}")
    if runs:
        last = runs[-1]
        print(f"Last trained   : {last.version} at {last.completed_at or last.created_at}")


async def main() -> None:
    config = GroundCortexConfig()
    db = Database(config.buffer_db)
    db.backfill_model_name(config.model_name)
    inference_manager = create_manager(config)

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
    init_inference(inference_manager, config, db)

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
    import argparse

    parser = argparse.ArgumentParser(
        prog="groundcortex",
        description="GroundCortex — start the server or manage adapters from the CLI.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--switch",
        metavar="VERSION",
        help="Switch active adapter. Accepts version name (v2), negative index (-1), or 'base'.",
    )
    group.add_argument(
        "--delete",
        metavar="VERSION",
        help="Soft-delete an adapter. Accepts version name or negative index.",
    )
    group.add_argument(
        "--list",
        action="store_true",
        help="List all trained (non-deleted) adapters.",
    )
    group.add_argument(
        "--status",
        action="store_true",
        help="Show active adapter and pending experience count.",
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="With --switch: allow loading a no-pass adapter that failed the quality gate.",
    )
    args = parser.parse_args()

    # CLI mode: load config only (no model, no servers)
    if args.switch or args.delete or args.list or args.status:
        from groundcortex.config import GroundCortexConfig as _Cfg
        cfg = _Cfg()
        if args.switch:
            _cli_switch(cfg, args.switch, force=getattr(args, "force", False))
        elif args.delete:
            _cli_delete(cfg, args.delete)
        elif args.list:
            _cli_list(cfg)
        elif args.status:
            _cli_status(cfg)
        return

    asyncio.run(main())


if __name__ == "__main__":
    main_sync()
