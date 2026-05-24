"""Tests for the FastAPI inference server (inference_server.py)."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

import groundcortex.inference_server as server_mod
from groundcortex.inference_server import app


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_server_globals():
    """Restore module-level globals to None before and after each test."""
    server_mod._inference_manager = None
    server_mod._config = None
    server_mod._db = None
    yield
    server_mod._inference_manager = None
    server_mod._config = None
    server_mod._db = None


def _manager(adapters=None, active=None, ready=True, training=False, response="Test response."):
    m = MagicMock()
    m.list_loaded_adapters.return_value = list(adapters or [])
    m.get_active_version.return_value = active
    m.is_ready = ready
    m.is_training = training
    m.generate.return_value = response
    return m


def _config_with_key(api_key="", model_name="test-model"):
    cfg = MagicMock()
    cfg.inference_api_key = api_key
    cfg.model_name = model_name
    return cfg


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestListModels:
    def test_no_manager_returns_503(self):
        r = TestClient(app, raise_server_exceptions=False).get("/v1/models")
        assert r.status_code == 503

    def test_training_in_progress_returns_503(self):
        server_mod._inference_manager = _manager(training=True)
        r = TestClient(app, raise_server_exceptions=False).get("/v1/models")
        assert r.status_code == 503
        assert "training in progress" in r.json()["detail"]

    def test_returns_200_with_manager(self):
        server_mod._inference_manager = _manager()
        r = TestClient(app).get("/v1/models")
        assert r.status_code == 200

    def test_response_has_active_pseudo_model(self):
        server_mod._inference_manager = _manager()
        data = TestClient(app).get("/v1/models").json()["data"]
        ids = [m["id"] for m in data]
        assert "active" in ids

    def test_loaded_adapters_listed(self):
        server_mod._inference_manager = _manager(adapters=["v1", "v2"], active="v2")
        data = TestClient(app).get("/v1/models").json()["data"]
        ids = [m["id"] for m in data]
        assert "v1" in ids
        assert "v2" in ids

    def test_active_adapter_flagged(self):
        server_mod._inference_manager = _manager(adapters=["v1", "v2"], active="v2")
        data = TestClient(app).get("/v1/models").json()["data"]
        v2_entry = next(m for m in data if m["id"] == "v2")
        assert v2_entry.get("is_active") is True

    def test_no_adapters_loaded_still_returns_active_entry(self):
        server_mod._inference_manager = _manager(adapters=[], active=None)
        data = TestClient(app).get("/v1/models").json()["data"]
        assert any(m["id"] == "active" for m in data)


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_no_manager_returns_503(self):
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 503

    def test_training_in_progress_returns_503(self):
        server_mod._inference_manager = _manager(training=True)
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 503
        assert "training in progress" in r.json()["detail"]

    def test_model_not_ready_returns_503(self):
        server_mod._inference_manager = _manager(ready=False)
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 503

    def test_basic_completion_returns_200(self):
        server_mod._inference_manager = _manager()
        r = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert r.status_code == 200

    def test_response_has_openai_shape(self):
        server_mod._inference_manager = _manager()
        body = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        ).json()
        assert "id" in body
        assert "object" in body
        assert "choices" in body
        assert body["object"] == "chat.completion"

    def test_choice_has_assistant_message(self):
        server_mod._inference_manager = _manager(response="I know Paris.")
        body = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Tell me."}]},
        ).json()
        msg = body["choices"][0]["message"]
        assert msg["role"] == "assistant"
        assert "Paris" in msg["content"]

    def test_finish_reason_is_stop(self):
        server_mod._inference_manager = _manager()
        body = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        ).json()
        assert body["choices"][0]["finish_reason"] == "stop"

    def test_unknown_model_returns_404(self):
        server_mod._inference_manager = _manager(adapters=["v1"])
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/chat/completions",
            json={"model": "v99", "messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 404

    def test_specific_model_calls_set_active(self):
        m = _manager(adapters=["v1", "v2"])
        server_mod._inference_manager = m
        TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "v1", "messages": [{"role": "user", "content": "Hi"}]},
        )
        m.set_active.assert_called_once_with("v1")

    def test_active_model_does_not_call_set_active(self):
        m = _manager(adapters=["v1"])
        server_mod._inference_manager = m
        TestClient(app).post(
            "/v1/chat/completions",
            json={"model": "active", "messages": [{"role": "user", "content": "Hi"}]},
        )
        m.set_active.assert_not_called()

    def test_generate_called_with_messages(self):
        m = _manager()
        server_mod._inference_manager = m
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello world"}]},
        )
        call_kwargs = m.generate.call_args
        messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
        assert messages[0]["content"] == "Hello world"

    def test_max_tokens_forwarded_to_generate(self):
        m = _manager()
        server_mod._inference_manager = m
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 128},
        )
        call_kwargs = m.generate.call_args
        max_new = call_kwargs.kwargs.get("max_new_tokens")
        assert max_new == 128


# ---------------------------------------------------------------------------
# Bearer auth middleware
# ---------------------------------------------------------------------------

class TestBearerAuth:
    def test_no_api_key_configured_allows_all_requests(self):
        server_mod._inference_manager = _manager()
        server_mod._config = _config_with_key("")
        r = TestClient(app).get("/v1/models")
        assert r.status_code == 200

    def test_api_key_configured_rejects_missing_token(self):
        server_mod._inference_manager = _manager()
        server_mod._config = _config_with_key("secret")
        r = TestClient(app, raise_server_exceptions=False).get("/v1/models")
        assert r.status_code == 401

    def test_api_key_configured_accepts_correct_token(self):
        server_mod._inference_manager = _manager()
        server_mod._config = _config_with_key("secret")
        r = TestClient(app).get(
            "/v1/models", headers={"Authorization": "Bearer secret"}
        )
        assert r.status_code == 200

    def test_api_key_configured_rejects_wrong_token(self):
        server_mod._inference_manager = _manager()
        server_mod._config = _config_with_key("secret")
        r = TestClient(app, raise_server_exceptions=False).get(
            "/v1/models", headers={"Authorization": "Bearer wrong"}
        )
        assert r.status_code == 401

    def test_api_key_configured_rejects_malformed_header(self):
        server_mod._inference_manager = _manager()
        server_mod._config = _config_with_key("secret")
        r = TestClient(app, raise_server_exceptions=False).get(
            "/v1/models", headers={"Authorization": "secret"}  # missing "Bearer "
        )
        assert r.status_code == 401


# ---------------------------------------------------------------------------
# POST /v1/control/switch
# ---------------------------------------------------------------------------

from groundcortex.pipeline.models import TrainingRun


def _db_mock(runs=None, active_run=None):
    db = MagicMock()
    db.list_switchable_runs.return_value = list(runs or [])  # oldest-first
    db.get_active_run.return_value = active_run
    db.get_run_by_version.side_effect = lambda v: next(
        (r for r in (runs or []) if r.version == v), None
    )
    return db


def _switch_run(version="v1", status="complete", adapter_path="/adapters/v1", model_name="test-model"):
    return TrainingRun(
        version=version, trigger="mcp", adapter_path=adapter_path,
        status=status, model_name=model_name,
    )


class TestControlSwitch:
    def _setup(self, runs=None, adapters=None, training=False):
        server_mod._inference_manager = _manager(adapters=adapters or [], training=training)
        server_mod._db = _db_mock(runs=runs or [])
        server_mod._config = _config_with_key()

    def test_no_server_init_returns_503(self):
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/control/switch", json={"version": "v1"}
        )
        assert r.status_code == 503

    def test_training_in_progress_returns_503(self):
        self._setup(training=True)
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/control/switch", json={"version": "v1"}
        )
        assert r.status_code == 503

    def test_switch_base_calls_unload_adapter(self):
        self._setup()
        r = TestClient(app).post("/v1/control/switch", json={"version": "base"})
        assert r.status_code == 200
        server_mod._inference_manager.unload_adapter.assert_called_once()
        server_mod._db.unset_active_run.assert_called_once()

    def test_switch_base_returns_none_active(self):
        self._setup()
        body = TestClient(app).post("/v1/control/switch", json={"version": "base"}).json()
        assert body["active_version"] is None

    def test_switch_by_version_name_returns_ok(self):
        runs = [_switch_run("v1")]
        self._setup(runs=runs, adapters=["v1"])
        body = TestClient(app).post("/v1/control/switch", json={"version": "v1"}).json()
        assert body["status"] == "ok"
        assert body["active_version"] == "v1"

    def test_switch_by_negative_index_resolves_correctly(self):
        runs = [_switch_run("v1"), _switch_run("v2"), _switch_run("v3")]
        self._setup(runs=runs, adapters=["v1", "v2", "v3"])
        body = TestClient(app).post("/v1/control/switch", json={"version": "-1"}).json()
        assert body["active_version"] == "v3"

    def test_unknown_version_returns_404(self):
        self._setup()
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/control/switch", json={"version": "v99"}
        )
        assert r.status_code == 404

    def test_out_of_range_index_returns_404(self):
        runs = [_switch_run("v1")]
        self._setup(runs=runs)
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/control/switch", json={"version": "-5"}
        )
        assert r.status_code == 404

    def test_incomplete_run_returns_409(self):
        runs = [_switch_run("v1", status="training")]
        self._setup(runs=runs)
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/control/switch", json={"version": "v1"}
        )
        assert r.status_code == 409

    def test_not_loaded_adapter_triggers_load(self):
        runs = [_switch_run("v1")]
        self._setup(runs=runs, adapters=[])  # v1 not loaded
        TestClient(app).post("/v1/control/switch", json={"version": "v1"})
        server_mod._inference_manager.load_adapter.assert_called_once_with("/adapters/v1", "v1")

    def test_already_loaded_adapter_not_loaded_again(self):
        runs = [_switch_run("v1")]
        self._setup(runs=runs, adapters=["v1"])
        TestClient(app).post("/v1/control/switch", json={"version": "v1"})
        server_mod._inference_manager.load_adapter.assert_not_called()

    def test_no_pass_without_force_returns_409(self):
        runs = [_switch_run("v1", status="no-pass")]
        self._setup(runs=runs)
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/control/switch", json={"version": "v1"}
        )
        assert r.status_code == 409

    def test_no_pass_with_force_returns_ok(self):
        runs = [_switch_run("v1", status="no-pass")]
        self._setup(runs=runs, adapters=["v1"])
        body = TestClient(app).post(
            "/v1/control/switch", json={"version": "v1", "force": True}
        ).json()
        assert body["status"] == "ok"
        assert body["active_version"] == "v1"

    def test_force_field_accepted_in_request_body(self):
        runs = [_switch_run("v1")]
        self._setup(runs=runs, adapters=["v1"])
        r = TestClient(app).post(
            "/v1/control/switch", json={"version": "v1", "force": False}
        )
        assert r.status_code == 200

    def test_model_mismatch_returns_409(self):
        runs = [_switch_run("v1", model_name="other-model")]
        self._setup(runs=runs)
        r = TestClient(app, raise_server_exceptions=False).post(
            "/v1/control/switch", json={"version": "v1"}
        )
        assert r.status_code == 409
        assert "other-model" in r.json()["detail"]

    def test_model_match_returns_ok(self):
        runs = [_switch_run("v1", model_name="test-model")]
        self._setup(runs=runs, adapters=["v1"])
        body = TestClient(app).post("/v1/control/switch", json={"version": "v1"}).json()
        assert body["status"] == "ok"


# ---------------------------------------------------------------------------
# POST /v1/control/dry-run
# ---------------------------------------------------------------------------

class TestControlDryRun:
    def test_503_when_not_initialized(self):
        r = TestClient(app, raise_server_exceptions=False).post("/v1/control/dry-run")
        assert r.status_code == 503

    def test_409_when_training_in_progress(self):
        server_mod._config = _config_with_key()
        server_mod._inference_manager = _manager(training=True)
        r = TestClient(app, raise_server_exceptions=False).post("/v1/control/dry-run")
        assert r.status_code == 409

    def test_returns_ok_or_skipped(self, tmp_path):
        from unittest.mock import patch, AsyncMock
        server_mod._config = _config_with_key()
        server_mod._inference_manager = _manager()
        skipped = {"status": "skipped", "reason": "no_sources", "total_chunks": 0}
        mock_fn = AsyncMock(return_value=skipped)
        with patch("groundcortex.consolidator.run_dry_run", mock_fn):
            r = TestClient(app).post("/v1/control/dry-run")
        assert r.status_code == 200
        assert r.json()["status"] in ("ok", "skipped")


# ---------------------------------------------------------------------------
# Tool calling
# ---------------------------------------------------------------------------

_TOOL = {
    "type": "function",
    "function": {
        "name": "get_time",
        "description": "Get current time",
        "parameters": {"type": "object", "properties": {}},
    },
}

_TOOL_CALL_RESPONSE = '<tool_call>\n{"name": "get_time", "arguments": {}}\n</tool_call>'


def _setup_tool_test(response: str = _TOOL_CALL_RESPONSE):
    mgr = _manager(response=response)
    server_mod._inference_manager = mgr
    cfg = _config_with_key(api_key="", model_name="mlx-community/Qwen3.6-35B-A3B-4bit")
    server_mod._config = cfg
    return mgr


class TestToolCalling:
    def test_tool_call_returned_when_model_outputs_tool_call(self):
        _setup_tool_test()
        r = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What time is it?"}],
                "tools": [_TOOL],
            },
        )
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "tool_calls"
        assert choice["message"]["content"] is None
        tool_calls = choice["message"]["tool_calls"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["function"]["name"] == "get_time"

    def test_plain_response_when_no_tool_call(self):
        _setup_tool_test(response="It is noon.")
        r = TestClient(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "What time is it?"}],
                "tools": [_TOOL],
            },
        )
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == "It is noon."
        assert "tool_calls" not in choice["message"]

    def test_tools_not_passed_skips_parsing(self):
        # Even if the model happens to output a tool_call block, without
        # request.tools set the server must not parse it.
        _setup_tool_test(response=_TOOL_CALL_RESPONSE)
        r = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200
        choice = r.json()["choices"][0]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == _TOOL_CALL_RESPONSE

    def test_tool_message_forwarded_in_messages(self):
        mgr = _manager(response="done")
        server_mod._inference_manager = mgr
        server_mod._config = _config_with_key()
        TestClient(app).post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "user", "content": "What time?"},
                    {"role": "assistant", "content": None,
                     "tool_calls": [{"id": "call_abc", "type": "function",
                                     "function": {"name": "get_time", "arguments": "{}"}}]},
                    {"role": "tool", "tool_call_id": "call_abc", "content": "12:00"},
                ],
            },
        )
        call_args = mgr.generate.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        tool_msg = next(m for m in messages if m["role"] == "tool")
        assert tool_msg["tool_call_id"] == "call_abc"
        assert tool_msg["content"] == "12:00"


# ---------------------------------------------------------------------------
# Thinking / reasoning_effort
# ---------------------------------------------------------------------------

def _setup_thinking_test(response="Answer."):
    mgr = _manager(response=response)
    server_mod._inference_manager = mgr
    server_mod._config = _config_with_key()
    return mgr


class TestThinking:
    def test_reasoning_effort_medium_passes_enable_thinking_true(self):
        mgr = _setup_thinking_test()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}],
                  "reasoning_effort": "medium"},
        )
        call_kwargs = mgr.generate.call_args.kwargs
        assert call_kwargs["enable_thinking"] is True

    def test_reasoning_effort_high_passes_enable_thinking_true(self):
        mgr = _setup_thinking_test()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}],
                  "reasoning_effort": "high"},
        )
        assert mgr.generate.call_args.kwargs["enable_thinking"] is True

    def test_reasoning_effort_none_passes_enable_thinking_false(self):
        mgr = _setup_thinking_test()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}],
                  "reasoning_effort": "none"},
        )
        assert mgr.generate.call_args.kwargs["enable_thinking"] is False

    def test_no_reasoning_effort_passes_enable_thinking_false(self):
        mgr = _setup_thinking_test()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert mgr.generate.call_args.kwargs["enable_thinking"] is False

    def test_direct_enable_thinking_field_works(self):
        mgr = _setup_thinking_test()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}],
                  "enable_thinking": True},
        )
        assert mgr.generate.call_args.kwargs["enable_thinking"] is True

    def test_direct_enable_thinking_takes_effect_without_reasoning_effort(self):
        mgr = _setup_thinking_test()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}],
                  "enable_thinking": True, "reasoning_effort": None},
        )
        assert mgr.generate.call_args.kwargs["enable_thinking"] is True


# ---------------------------------------------------------------------------
# Request logging
# ---------------------------------------------------------------------------

import io
import json as _json
import logging as _logging


def _make_capture_logger() -> tuple[_logging.Logger, io.StringIO]:
    """Return a logger that writes to an in-memory buffer."""
    buf = io.StringIO()
    log = _logging.getLogger(f"test_capture_{id(buf)}")
    log.setLevel(_logging.INFO)
    h = _logging.StreamHandler(buf)
    h.setFormatter(_logging.Formatter("%(message)s"))
    log.addHandler(h)
    log.propagate = False
    return log, buf


def _logged_events(buf: io.StringIO) -> list[dict]:
    """Parse each logged line as 'EVENT {json}' and return list of (event, data) dicts."""
    events = []
    for line in buf.getvalue().splitlines():
        parts = line.strip().split(" ", 1)
        if len(parts) == 2:
            try:
                events.append({"event": parts[0], "data": _json.loads(parts[1])})
            except _json.JSONDecodeError:
                pass
    return events


class TestRequestLogging:
    def setup_method(self):
        server_mod._inference_manager = None
        server_mod._config = None
        server_mod._db = None
        server_mod._request_logger = None

    def _setup(self, response="Hello."):
        mgr = _manager(response=response)
        server_mod._inference_manager = mgr
        server_mod._config = _config_with_key()
        log, buf = _make_capture_logger()
        server_mod._request_logger = log
        return buf

    def test_request_event_logged(self):
        buf = self._setup()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        events = _logged_events(buf)
        assert any(e["event"] == "REQUEST" for e in events)

    def test_response_event_logged(self):
        buf = self._setup()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        events = _logged_events(buf)
        assert any(e["event"] == "RESPONSE" for e in events)

    def test_request_contains_messages(self):
        buf = self._setup()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "What time?"}]},
        )
        req = next(e for e in _logged_events(buf) if e["event"] == "REQUEST")
        assert req["data"]["messages"][0]["content"] == "What time?"

    def test_response_contains_content(self):
        buf = self._setup(response="It is noon.")
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        resp = next(e for e in _logged_events(buf) if e["event"] == "RESPONSE")
        assert resp["data"]["content"] == "It is noon."

    def test_no_logging_when_logger_is_none(self):
        server_mod._inference_manager = _manager(response="Hi")
        server_mod._config = _config_with_key()
        server_mod._request_logger = None
        # Should not raise and nothing is written anywhere
        r = TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        assert r.status_code == 200

    def test_request_and_response_share_same_id(self):
        buf = self._setup()
        TestClient(app).post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        events = _logged_events(buf)
        req_id = next(e["data"]["id"] for e in events if e["event"] == "REQUEST")
        resp_id = next(e["data"]["id"] for e in events if e["event"] == "RESPONSE")
        assert req_id == resp_id


# ---------------------------------------------------------------------------
# Sampling parameters pass-through
# ---------------------------------------------------------------------------

def _setup_sampling_test(response="Hello."):
    mgr = _manager(response=response)
    server_mod._inference_manager = mgr
    server_mod._config = _config_with_key()
    return mgr


class TestSamplingParams:
    def _post(self, payload: dict):
        base = {"messages": [{"role": "user", "content": "Hi"}]}
        return TestClient(app).post("/v1/chat/completions", json={**base, **payload})

    def test_top_p_forwarded_to_generate(self):
        mgr = _setup_sampling_test()
        self._post({"top_p": 0.9})
        assert mgr.generate.call_args.kwargs["top_p"] == 0.9

    def test_top_k_forwarded_to_generate(self):
        mgr = _setup_sampling_test()
        self._post({"top_k": 40})
        assert mgr.generate.call_args.kwargs["top_k"] == 40

    def test_min_p_forwarded_to_generate(self):
        mgr = _setup_sampling_test()
        self._post({"min_p": 0.05})
        assert mgr.generate.call_args.kwargs["min_p"] == 0.05

    def test_repetition_penalty_forwarded_to_generate(self):
        mgr = _setup_sampling_test()
        self._post({"repetition_penalty": 1.1})
        assert mgr.generate.call_args.kwargs["repetition_penalty"] == 1.1

    def test_frequency_penalty_forwarded_to_generate(self):
        mgr = _setup_sampling_test()
        self._post({"frequency_penalty": 0.2})
        assert mgr.generate.call_args.kwargs["frequency_penalty"] == 0.2

    def test_omitted_params_default_to_none(self):
        mgr = _setup_sampling_test()
        self._post({})
        kw = mgr.generate.call_args.kwargs
        for param in ("top_p", "top_k", "min_p", "repetition_penalty", "frequency_penalty"):
            assert kw[param] is None, f"expected {param}=None, got {kw[param]}"

    def test_sampling_params_forwarded_in_streaming(self):
        mgr = _setup_sampling_test()
        mgr.generate_stream.return_value = iter(["Hello"])
        TestClient(app).post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "Hi"}],
                "stream": True,
                "top_p": 0.85,
                "top_k": 20,
                "min_p": 0.01,
                "repetition_penalty": 1.2,
                "frequency_penalty": 0.3,
            },
        )
        kw = mgr.generate_stream.call_args.kwargs
        assert kw["top_p"] == 0.85
        assert kw["top_k"] == 20
        assert kw["min_p"] == 0.01
        assert kw["repetition_penalty"] == 1.2
        assert kw["frequency_penalty"] == 0.3
