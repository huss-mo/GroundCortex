"""Tests for the FastMCP server (mcp_server.py) - tool registration and handler logic."""
from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock

import pytest

from groundcortex.config import GroundCortexConfig
from groundcortex.mcp_server import build_mcp_server
from groundcortex.pipeline.models import TrainingRun

_ALL_TOOLS = {"trigger_consolidation", "get_cortex_status", "switch_adapter", "list_adapters"}


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


def _mgr(adapters=None, active=None, training=False):
    m = MagicMock()
    m.list_loaded_adapters.return_value = list(adapters or [])
    m.get_active_version.return_value = active
    m.is_training = training
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
    def test_empty_exposed_tools_registers_all(self, tmp_path):
        mcp = build_mcp_server(_cfg(tmp_path, []), _db(), _mgr())
        assert _registered_names(mcp) == _ALL_TOOLS

    def test_single_tool_only_that_tool_registered(self, tmp_path):
        mcp = build_mcp_server(_cfg(tmp_path, ["get_cortex_status"]), _db(), _mgr())
        assert _registered_names(mcp) == {"get_cortex_status"}

    def test_two_tools_registered(self, tmp_path):
        mcp = build_mcp_server(
            _cfg(tmp_path, ["get_cortex_status", "switch_adapter"]),
            _db(), _mgr(),
        )
        assert _registered_names(mcp) == {"get_cortex_status", "switch_adapter"}

    def test_all_three_explicit(self, tmp_path):
        mcp = build_mcp_server(
            _cfg(tmp_path, list(_ALL_TOOLS)), _db(), _mgr()
        )
        assert _registered_names(mcp) == _ALL_TOOLS

    def test_excluded_tools_absent(self, tmp_path):
        mcp = build_mcp_server(_cfg(tmp_path, ["get_cortex_status"]), _db(), _mgr())
        names = _registered_names(mcp)
        assert "trigger_consolidation" not in names
        assert "switch_adapter" not in names


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
# switch_adapter
# ---------------------------------------------------------------------------

class TestSwitchLoraVersion:
    def _build(self, tmp_path, run=None, adapters=None):
        db = MagicMock()
        db.get_run_by_version.return_value = run
        db.set_active_run = MagicMock()
        return build_mcp_server(
            _cfg(tmp_path, ["switch_adapter"]),
            db,
            _mgr(adapters=adapters or []),
        )

    def _call(self, mcp, version_id) -> dict:
        return _parse(_run(mcp.call_tool("switch_adapter", {"version_id": version_id})))

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
        mcp = build_mcp_server(_cfg(tmp_path, ["switch_adapter"]), db, mgr)
        _run(mcp.call_tool("switch_adapter", {"version_id": "v1"}))
        mgr.load_adapter.assert_called_once_with("/adapters/v1", "v1")

    def test_already_loaded_adapter_not_loaded_again(self, tmp_path):
        run = TrainingRun(
            version="v1", trigger="mcp", adapter_path="/adapters/v1", status="complete"
        )
        mgr = _mgr(adapters=["v1"])  # already loaded
        db = MagicMock()
        db.get_run_by_version.return_value = run
        mcp = build_mcp_server(_cfg(tmp_path, ["switch_adapter"]), db, mgr)
        _run(mcp.call_tool("switch_adapter", {"version_id": "v1"}))
        mgr.load_adapter.assert_not_called()


def _make_runs(*versions: str) -> list[TrainingRun]:
    """Build a list of complete TrainingRun objects for the given version names."""
    return [
        TrainingRun(version=v, trigger="mcp", adapter_path=f"/adapters/{v}", status="complete")
        for v in versions
    ]


# ---------------------------------------------------------------------------
# list_adapters
# ---------------------------------------------------------------------------

class TestListLoraVersions:
    def _build(self, tmp_path, runs=None):
        db = MagicMock()
        # list_runs returns DESC; the tool reverses to get ASC
        db.list_runs.return_value = list(reversed(runs or []))
        return build_mcp_server(
            _cfg(tmp_path, ["list_adapters"]),
            db,
            _mgr(active="v2"),
        )

    def _call(self, mcp) -> dict:
        return _parse(_run(mcp.call_tool("list_adapters", {})))

    def test_empty_returns_zero_total(self, tmp_path):
        mcp = self._build(tmp_path, runs=[])
        result = self._call(mcp)
        assert result["total"] == 0
        assert result["versions"] == []

    def test_total_matches_complete_run_count(self, tmp_path):
        mcp = self._build(tmp_path, runs=_make_runs("v1", "v2", "v3"))
        result = self._call(mcp)
        assert result["total"] == 3

    def test_versions_ordered_oldest_first(self, tmp_path):
        mcp = self._build(tmp_path, runs=_make_runs("v1", "v2", "v3"))
        result = self._call(mcp)
        assert [e["version"] for e in result["versions"]] == ["v1", "v2", "v3"]

    def test_last_version_has_index_minus_one(self, tmp_path):
        mcp = self._build(tmp_path, runs=_make_runs("v1", "v2", "v3"))
        result = self._call(mcp)
        assert result["versions"][-1]["index"] == -1

    def test_first_version_has_index_minus_n(self, tmp_path):
        mcp = self._build(tmp_path, runs=_make_runs("v1", "v2", "v3"))
        result = self._call(mcp)
        assert result["versions"][0]["index"] == -3

    def test_active_version_returned(self, tmp_path):
        mcp = self._build(tmp_path, runs=_make_runs("v1", "v2"))
        result = self._call(mcp)
        assert result["active_version"] == "v2"

    def test_failed_runs_excluded(self, tmp_path):
        failed = TrainingRun(version="v1", trigger="mcp", adapter_path="/p", status="failed")
        complete = TrainingRun(version="v2", trigger="mcp", adapter_path="/p", status="complete")
        db = MagicMock()
        db.list_runs.return_value = [complete, failed]  # DESC order
        mcp = build_mcp_server(_cfg(tmp_path, ["list_adapters"]), db, _mgr())
        result = _parse(_run(mcp.call_tool("list_adapters", {})))
        assert result["total"] == 1
        assert result["versions"][0]["version"] == "v2"


# ---------------------------------------------------------------------------
# switch_adapter - negative index
# ---------------------------------------------------------------------------

class TestSwitchLoraVersionNegativeIndex:
    def _build(self, tmp_path, runs):
        db = MagicMock()
        db.list_runs.return_value = list(reversed(runs))  # DESC order from DB
        db.set_active_run = MagicMock()
        return build_mcp_server(
            _cfg(tmp_path, ["switch_adapter"]),
            db,
            _mgr(adapters=[r.version for r in runs]),
        ), db

    def _call(self, mcp, version_id) -> dict:
        return _parse(_run(mcp.call_tool("switch_adapter", {"version_id": version_id})))

    def test_minus_one_activates_latest(self, tmp_path):
        runs = _make_runs("v1", "v2", "v3")
        mcp, _ = self._build(tmp_path, runs)
        result = self._call(mcp, "-1")
        assert result["status"] == "ok"
        assert result["active_version"] == "v3"

    def test_minus_two_activates_second_to_last(self, tmp_path):
        runs = _make_runs("v1", "v2", "v3")
        mcp, _ = self._build(tmp_path, runs)
        result = self._call(mcp, "-2")
        assert result["status"] == "ok"
        assert result["active_version"] == "v2"

    def test_minus_n_activates_oldest(self, tmp_path):
        runs = _make_runs("v1", "v2", "v3")
        mcp, _ = self._build(tmp_path, runs)
        result = self._call(mcp, "-3")
        assert result["status"] == "ok"
        assert result["active_version"] == "v1"

    def test_out_of_range_returns_error(self, tmp_path):
        runs = _make_runs("v1", "v2")
        mcp, _ = self._build(tmp_path, runs)
        result = self._call(mcp, "-5")
        assert result["status"] == "error"
        assert "out of range" in result["message"]

    def test_out_of_range_message_shows_count(self, tmp_path):
        runs = _make_runs("v1", "v2")
        mcp, _ = self._build(tmp_path, runs)
        result = self._call(mcp, "-5")
        assert "2" in result["message"]

    def test_version_name_still_works_alongside_negative_index(self, tmp_path):
        runs = _make_runs("v1", "v2")
        db = MagicMock()
        db.list_runs.return_value = list(reversed(runs))
        db.get_run_by_version.return_value = runs[0]
        db.set_active_run = MagicMock()
        mcp = build_mcp_server(
            _cfg(tmp_path, ["switch_adapter"]),
            db,
            _mgr(adapters=["v1", "v2"]),
        )
        result = _parse(_run(mcp.call_tool("switch_adapter", {"version_id": "v1"})))
        assert result["status"] == "ok"
        assert result["active_version"] == "v1"
