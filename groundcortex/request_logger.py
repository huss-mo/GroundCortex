"""Request/response file logger for inference and MCP servers.

Creates a logging.Logger that writes one JSON line per entry to a file,
prefixed with an ISO timestamp. Activated when GROUNDCORTEX_LOG_REQUESTS=true.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path


def make_request_logger(log_file: Path) -> logging.Logger:
    """Return a Logger that appends JSON lines to log_file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"groundcortex.requests.{log_file.stem}")
    if logger.handlers:
        return logger  # already configured (e.g. server restarted in-process)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(logger: logging.Logger, event: str, data: dict) -> None:
    logger.info("%s %s", event, json.dumps(data, ensure_ascii=False, default=str))
