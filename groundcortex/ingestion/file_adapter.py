from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.base import IngestionAdapter
from groundcortex.pipeline.models import Experience

# Matches GroundMemory timestamped section headers: ## 2026-05-17 14:30
_GM_HEADER = re.compile(r"^##\s+(\d{4}-\d{2}-\d{2}(?:\s+\d{2}:\d{2})?)\s*$", re.MULTILINE)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _classify_type(source_id: str) -> Literal["fact", "preference", "mindset"]:
    lower = source_id.lower()
    if "user.md" in lower:
        return "preference"
    if "/daily/" in lower or "\\daily\\" in lower:
        return "mindset"
    return "fact"


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Return list of (timestamp_or_empty, section_text) pairs.

    If the file contains GroundMemory-style ## YYYY-MM-DD [HH:MM] headers,
    each header starts a new section. Otherwise the whole file is one section.
    """
    matches = list(_GM_HEADER.finditer(content))
    if not matches:
        stripped = content.strip()
        return [("", stripped)] if stripped else []

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        ts = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if body:
            sections.append((ts, body))
    return sections


def parse_content(
    content: str,
    source_id: str,
    adapter_name: str,
    db: Database,
) -> list[Experience]:
    """Core parsing logic shared by FileAdapter and RemoteFileAdapter.

    1. Hash the full content; skip if unchanged.
    2. Supersede all old experiences for this source.
    3. Split into sections and emit new pending Experiences.
    4. Update source_files record.
    """
    file_hash = _sha256(content)
    stored_hash = db.get_file_hash(source_id)

    if stored_hash == file_hash:
        return []  # nothing changed

    # Supersede stale experiences before creating new ones
    db.supersede_source(source_id)

    exp_type = _classify_type(source_id)
    now = _now_iso()
    new_experiences: list[Experience] = []

    for _ts, body in _split_sections(content):
        exp = Experience(
            source=source_id,
            type=exp_type,
            raw_content=body,
            entities=[],          # entity extraction is a future enhancement
            content_hash=_sha256(body),
            status="pending",
            created_at=now,
        )
        db.add_experience(exp)
        new_experiences.append(exp)

    db.upsert_file(source_id, adapter_name, file_hash, now)
    return new_experiences


class FileAdapter(IngestionAdapter):
    """Reads local files listed in GROUNDCORTEX_SOURCE_PATHS."""

    def __init__(self, config: GroundCortexConfig, db: Database) -> None:
        self._paths = config.source_paths
        self._db = db

    def ingest(self) -> list[Experience]:
        results: list[Experience] = []
        for path in self._paths:
            path = Path(path).expanduser()
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            source_id = f"file:{path}"
            new = parse_content(content, source_id, "file", self._db)
            results.extend(new)
        return results
