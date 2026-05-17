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
    yield
    server_mod._inference_manager = None
    server_mod._config = None


def _manager(adapters=None, active=None, ready=True, training=False, response="Test response."):
    m = MagicMock()
    m.list_loaded_adapters.return_value = list(adapters or [])
    m.get_active_version.return_value = active
    m.is_ready = ready
    m.is_training = training
    m.generate.return_value = response
    return m


def _config_with_key(api_key=""):
    cfg = MagicMock()
    cfg.inference_api_key = api_key
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
