"""Tests for RemoteFileAdapter (ingestion/remote_adapter.py)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.remote_adapter import RemoteFileAdapter


def _cfg(tmp_path, urls, api_key="") -> GroundCortexConfig:
    return GroundCortexConfig(
        _env_file=None,
        output_dir=tmp_path / "adapters",
        remote_source_urls=urls,
        remote_source_api_key=api_key,
    )


def _mock_client(response_text: str | None = "Content.", error: Exception | None = None):
    """Return a context-manager-compatible mock httpx.Client."""
    mock_response = MagicMock()
    mock_response.text = response_text or ""
    mock_response.raise_for_status = MagicMock()

    mock_client = MagicMock()
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)

    if error is not None:
        mock_client.get.side_effect = error
    else:
        mock_client.get.return_value = mock_response

    return mock_client


class TestRemoteFileAdapter:
    def test_successful_get_creates_experience(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/notes.md"])
        with patch("httpx.Client", return_value=_mock_client("Remote fact content.")):
            result = RemoteFileAdapter(cfg, db).ingest()
        assert len(result) == 1
        assert result[0].source == "http://server/notes.md"
        assert result[0].status == "pending"

    def test_experience_content_matches_response(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/notes.md"])
        with patch("httpx.Client", return_value=_mock_client("Unique canary content XYZ.")):
            result = RemoteFileAdapter(cfg, db).ingest()
        assert "Unique canary content XYZ." in result[0].raw_content

    def test_api_key_sends_authorization_header(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/notes.md"], api_key="secret-token")
        captured: dict = {}

        def fake_get(url, headers=None):
            captured.update(headers or {})
            r = MagicMock()
            r.text = "Content."
            r.raise_for_status = MagicMock()
            return r

        mock_client = _mock_client()
        mock_client.get = fake_get
        with patch("httpx.Client", return_value=mock_client):
            RemoteFileAdapter(cfg, db).ingest()

        assert captured.get("Authorization") == "Bearer secret-token"

    def test_no_api_key_omits_authorization_header(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/notes.md"], api_key="")
        captured: dict = {}

        def fake_get(url, headers=None):
            captured.update(headers or {})
            r = MagicMock()
            r.text = "Content."
            r.raise_for_status = MagicMock()
            return r

        mock_client = _mock_client()
        mock_client.get = fake_get
        with patch("httpx.Client", return_value=mock_client):
            RemoteFileAdapter(cfg, db).ingest()

        assert "Authorization" not in captured

    def test_http_error_caught_returns_empty(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/missing.md"])
        with patch("httpx.Client", return_value=_mock_client(error=httpx.HTTPError("404"))):
            result = RemoteFileAdapter(cfg, db).ingest()
        assert result == []

    def test_http_error_does_not_raise(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/missing.md"])
        with patch("httpx.Client", return_value=_mock_client(error=httpx.HTTPError("500"))):
            RemoteFileAdapter(cfg, db).ingest()  # must not raise

    def test_multiple_urls_each_fetched(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/a.md", "http://server/b.md"])
        call_count = 0

        def fake_get(url, headers=None):
            nonlocal call_count
            call_count += 1
            r = MagicMock()
            r.text = f"Content from {url}"
            r.raise_for_status = MagicMock()
            return r

        mock_client = _mock_client()
        mock_client.get = fake_get
        with patch("httpx.Client", return_value=mock_client):
            result = RemoteFileAdapter(cfg, db).ingest()

        assert call_count == 2
        assert len(result) == 2

    def test_one_url_fails_others_still_processed(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/bad.md", "http://server/good.md"])
        call_count = 0

        def fake_get(url, headers=None):
            nonlocal call_count
            call_count += 1
            if "bad" in url:
                raise httpx.HTTPError("not found")
            r = MagicMock()
            r.text = "Good content."
            r.raise_for_status = MagicMock()
            return r

        mock_client = _mock_client()
        mock_client.get = fake_get
        with patch("httpx.Client", return_value=mock_client):
            result = RemoteFileAdapter(cfg, db).ingest()

        assert len(result) == 1
        assert result[0].source == "http://server/good.md"

    def test_unchanged_url_returns_empty_on_second_call(self, db, tmp_path):
        cfg = _cfg(tmp_path, ["http://server/notes.md"])
        with patch("httpx.Client", return_value=_mock_client("Stable content.")):
            RemoteFileAdapter(cfg, db).ingest()
        with patch("httpx.Client", return_value=_mock_client("Stable content.")):
            result = RemoteFileAdapter(cfg, db).ingest()
        assert result == []

    def test_no_urls_returns_empty(self, db, tmp_path):
        cfg = _cfg(tmp_path, [])
        result = RemoteFileAdapter(cfg, db).ingest()
        assert result == []
