from __future__ import annotations

import logging

import httpx

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.base import IngestionAdapter
from groundcortex.ingestion.file_adapter import parse_content
from groundcortex.pipeline.models import Experience

logger = logging.getLogger(__name__)


class RemoteFileAdapter(IngestionAdapter):
    """Fetches files from HTTP URLs and runs them through the shared parsing pipeline.

    Any HTTP server that responds to a plain GET with file content works:
    a notes API, a static file host, or a plain nginx directory - as long as
    the response body is Markdown or plain text. The URL is used as the source
    identifier in source_files.

    A single optional bearer token is applied to all configured URLs.
    """

    def __init__(self, config: GroundCortexConfig, db: Database) -> None:
        self._urls = config.remote_source_urls
        self._api_key = config.remote_source_api_key
        self._db = db

    def ingest(self) -> list[Experience]:
        results: list[Experience] = []
        headers = {}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        with httpx.Client(timeout=30.0) as client:
            for url in self._urls:
                try:
                    response = client.get(url, headers=headers)
                    response.raise_for_status()
                    content = response.text
                    new = parse_content(content, url, "remote", self._db)
                    results.extend(new)
                    logger.info("Remote ingestion: %s → %d new experiences", url, len(new))
                except httpx.HTTPError as exc:
                    logger.warning("Failed to fetch %s: %s", url, exc)

        return results
