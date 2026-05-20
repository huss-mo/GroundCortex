"""Tests for wrap_trusted_hosts() (DNS rebinding protection middleware)."""
from __future__ import annotations

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from groundcortex.__main__ import wrap_trusted_hosts


# ---------------------------------------------------------------------------
# Minimal ASGI app used as the wrapped target
# ---------------------------------------------------------------------------

def _ok(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


_base_app = Starlette(routes=[Route("/", _ok)])


def _client(allowed_hosts_cfg: str, base_url: str = "http://localhost") -> TestClient:
    return TestClient(
        wrap_trusted_hosts(_base_app, allowed_hosts_cfg),
        base_url=base_url,
        raise_server_exceptions=False,
    )


# ---------------------------------------------------------------------------
# No-op when allowed_hosts_cfg is empty
# ---------------------------------------------------------------------------

class TestEmptyConfig:
    def test_returns_original_app_when_empty(self):
        assert wrap_trusted_hosts(_base_app, "") is _base_app

    def test_returns_original_app_when_whitespace_only(self):
        assert wrap_trusted_hosts(_base_app, "  , , ") is _base_app

    def test_all_hosts_allowed_when_empty(self):
        # No middleware means any Host header is accepted
        client = TestClient(_base_app, base_url="http://arbitrary.host", raise_server_exceptions=False)
        assert client.get("/").status_code == 200


# ---------------------------------------------------------------------------
# Middleware applied when allowed_hosts_cfg is non-empty
# ---------------------------------------------------------------------------

class TestAllowedHosts:
    def test_localhost_always_allowed(self):
        r = _client("192.168.1.50:4343", base_url="http://localhost").get("/")
        assert r.status_code == 200

    def test_loopback_always_allowed(self):
        r = _client("192.168.1.50:4343", base_url="http://127.0.0.1").get("/")
        assert r.status_code == 200

    def test_configured_extra_host_allowed(self):
        r = _client("192.168.1.50", base_url="http://192.168.1.50").get("/")
        assert r.status_code == 200

    def test_unlisted_host_rejected(self):
        r = _client("192.168.1.50", base_url="http://192.168.1.99").get("/")
        assert r.status_code == 400

    def test_multiple_extra_hosts_all_allowed(self):
        cfg = "192.168.1.50,myserver.local"
        assert _client(cfg, base_url="http://192.168.1.50").get("/").status_code == 200
        assert _client(cfg, base_url="http://myserver.local").get("/").status_code == 200

    def test_unlisted_host_rejected_with_multiple_configured(self):
        cfg = "192.168.1.50,myserver.local"
        r = _client(cfg, base_url="http://attacker.evil").get("/")
        assert r.status_code == 400

    def test_extra_hosts_whitespace_stripped(self):
        r = _client("  192.168.1.50  ", base_url="http://192.168.1.50").get("/")
        assert r.status_code == 200

    def test_port_in_cfg_value_is_stripped_and_still_matches(self):
        # Users may write "192.168.1.50:4343" or "192.168.1.50" - both should work
        # because TrustedHostMiddleware compares hostname-only.
        r = _client("192.168.1.50:4343", base_url="http://192.168.1.50").get("/")
        assert r.status_code == 200
