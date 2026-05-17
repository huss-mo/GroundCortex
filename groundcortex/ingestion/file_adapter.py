from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.base import IngestionAdapter
from groundcortex.pipeline.models import Experience

# Matches any ## section header (level-2 Markdown heading with content)
_SECTION_HEADER = re.compile(r"^##\s+\S[^\n]*$", re.MULTILINE)


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Return list of (header, section_text) pairs.

    Files with ## headings are split on each heading; each heading and its
    following content become one section. Files with no ## headings are
    treated as a single section.
    """
    matches = list(_SECTION_HEADER.finditer(content))
    if not matches:
        stripped = content.strip()
        return [("", stripped)] if stripped else []

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        header = m.group(0).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if body:
            sections.append((header, body))
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

    now = _now_iso()
    new_experiences: list[Experience] = []

    for _header, body in _split_sections(content):
        exp = Experience(
            source=source_id,
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
