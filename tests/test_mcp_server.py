"""Tests for the FastMCP server (mcp_server.py) — tool registration and handler logic."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from groundcortex.config import GroundCortexConfig
from groundcortex.mcp_server import build_mcp_server
from groundcortex.pipeline.models import TrainingRun

_ALL_TOOLS = {"trigger_consolidation", "get_cortex_status", "switch_lora_version"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _cfg(tmp_path, exposed_tools=None) -> GroundCortexConfig:
    return GroundCortexConfig(
        _env_file=None,
        output_dir=tmp_path / "adapters",
        mcp_exposed_tools=exposed_tools or [],
    )


def _db(active_run=None, pending=0):
    db = MagicMock()
    db.get_active_run.return_value = active_run
    db.count_pending.return_value = pending
    return db


def _mgr(adapters=None, active=None):
    m = MagicMock()
    m.list_loaded_adapters.return_value = list(adapters or [])
    m.get_active_version.return_value = active
    return m


def _parse(result) -> dict:
    """Extract the dict from a FastMCP call_tool result.

    FastMCP >= 2 returns a ToolResult with .structured_content (the dict directly)
    and .content (list of TextContent for protocol compatibility).
    """
    # FastMCP ToolResult
    if hasattr(result, "structured_content") and result.structured_content is not None:
        return result.structured_content
    if hasattr(result, "content") and result.content:
        item = result.content[0]
        if hasattr(item, "text"):
            return json.loads(item.text)
    # Legacy: plain list of TextContent
    if isinstance(result, list) and result:
        item = result[0]
        if hasattr(item, "text"):
            return json.loads(item.text)
        if isinstance(item, dict):
            return item
    if isinstance(result, dict):
        return result
    return {}


def _registered_names(mcp) -> set[str]:
    tools = _run(mcp.list_tools())
    return {t.name for t in tools}


# ---------------------------------------------------------------------------
# Tool registration
# ---------------------------------------------------------------------------

class TestToolRegistration:
    def test_empty_exposed_tools_registers_all_three(self, tmp_path):
        mcp = build_mcp_server(_cfg(tmp_path, []), _db(), _mgr())
        assert _registered_names(mcp) == _ALL_TOOLS

    def test_single_tool_only_that_tool_registered(self, tmp_path):
        mcp = build_mcp_server(_cfg(tmp_path, ["get_cortex_status"]), _db(), _mgr())
        assert _registered_names(mcp) == {"get_cortex_status"}

    def test_two_tools_registered(self, tmp_path):
        mcp = build_mcp_server(
            _cfg(tmp_path, ["get_cortex_status", "switch_lora_version"]),
            _db(), _mgr(),
        )
        assert _registered_names(mcp) == {"get_cortex_status", "switch_lora_version"}

    def test_all_three_explicit(self, tmp_path):
        mcp = build_mcp_server(
            _cfg(tmp_path, list(_ALL_TOOLS)), _db(), _mgr()
        )
        assert _registered_names(mcp) == _ALL_TOOLS

    def test_excluded_tools_absent(self, tmp_path):
        mcp = build_mcp_server(_cfg(tmp_path, ["get_cortex_status"]), _db(), _mgr())
        names = _registered_names(mcp)
        assert "trigger_consolidation" not in names
        assert "switch_lora_version" not in names


# ---------------------------------------------------------------------------
# get_cortex_status
# ---------------------------------------------------------------------------

class TestGetCortexStatus:
    def _build(self, tmp_path, active_run=None, pending=0, adapters=None, active=None):
        return build_mcp_server(
            _cfg(tmp_path, ["get_cortex_status"]),
            _db(active_run=active_run, pending=pending),
            _mgr(adapters=adapters, active=active),
        )

    def _call(self, mcp) -> dict:
        return _parse(_run(mcp.call_tool("get_cortex_status", {})))

    def test_returns_active_version(self, tmp_path):
        mcp = self._build(tmp_path, active="v2", adapters=["v1", "v2"])
        result = self._call(mcp)
        assert result["active_version"] == "v2"

    def test_returns_pending_count(self, tmp_path):
        mcp = self._build(tmp_path, pending=3)
        result = self._call(mcp)
        assert result["pending_count"] == 3

    def test_returns_loaded_adapters(self, tmp_path):
        mcp = self._build(tmp_path, adapters=["v1", "v2"])
        result = self._call(mcp)
        assert "v1" in result["loaded_adapters"]

    def test_no_active_run_last_run_is_none(self, tmp_path):
        mcp = self._build(tmp_path, active_run=None)
        result = self._call(mcp)
        assert result["last_run"] is None

    def test_active_run_details_present(self, tmp_path):
        active = TrainingRun(
            version="v1", trigger="mcp", adapter_path="/p",
            status="complete", is_active=True,
        )
        mcp = self._build(tmp_path, active_run=active)
        result = self._call(mcp)
        assert result["last_run"] is not None
        assert result["last_run"]["version"] == "v1"
        assert result["last_run"]["status"] == "complete"


# ---------------------------------------------------------------------------
# switch_lora_version
# ---------------------------------------------------------------------------

class TestSwitchLoraVersion:
    def _build(self, tmp_path, run=None, adapters=None):
        db = MagicMock()
        db.get_run_by_version.return_value = run
        db.set_active_run = MagicMock()
        return build_mcp_server(
            _cfg(tmp_path, ["switch_lora_version"]),
            db,
            _mgr(adapters=adapters or []),
        )

    def _call(self, mcp, version_id) -> dict:
        return _parse(_run(mcp.call_tool("switch_lora_version", {"version_id": version_id})))

    def test_version_not_found_returns_error(self, tmp_path):
        mcp = self._build(tmp_path, run=None)
        result = self._call(mcp, "v99")
        assert result["status"] == "error"
        assert "v99" in result["message"]

    def test_version_not_complete_returns_error(self, tmp_path):
        run = TrainingRun(
            version="v1", trigger="mcp", adapter_path="/p", status="training"
        )
        mcp = self._build(tmp_path, run=run)
        result = self._call(mcp, "v1")
        assert result["status"] == "error"

    def test_successful_switch_returns_ok(self, tmp_path):
        run = TrainingRun(
            version="v1", trigger="mcp", adapter_path="/adapters/v1", status="complete"
        )
        mcp = self._build(tmp_path, run=run, adapters=["v1"])
        result = self._call(mcp, "v1")
        assert result["status"] == "ok"

    def test_successful_switch_returns_active_version(self, tmp_path):
        run = TrainingRun(
            version="v1", trigger="mcp", adapter_path="/adapters/v1", status="complete"
        )
        mcp = self._build(tmp_path, run=run, adapters=["v1"])
        result = self._call(mcp, "v1")
        assert result["active_version"] == "v1"

    def test_not_loaded_adapter_triggers_load(self, tmp_path):
        run = TrainingRun(
            version="v1", trigger="mcp", adapter_path="/adapters/v1", status="complete"
        )
        mgr = _mgr(adapters=[])  # v1 not yet loaded
        db = MagicMock()
        db.get_run_by_version.return_value = run
        mcp = build_mcp_server(_cfg(tmp_path, ["switch_lora_version"]), db, mgr)
        _run(mcp.call_tool("switch_lora_version", {"version_id": "v1"}))
        mgr.load_adapter.assert_called_once_with("/adapters/v1", "v1")

    def test_already_loaded_adapter_not_loaded_again(self, tmp_path):
        run = TrainingRun(
            version="v1", trigger="mcp", adapter_path="/adapters/v1", status="complete"
        )
        mgr = _mgr(adapters=["v1"])  # already loaded
        db = MagicMock()
        db.get_run_by_version.return_value = run
        mcp = build_mcp_server(_cfg(tmp_path, ["switch_lora_version"]), db, mgr)
        _run(mcp.call_tool("switch_lora_version", {"version_id": "v1"}))
        mgr.load_adapter.assert_not_called()
