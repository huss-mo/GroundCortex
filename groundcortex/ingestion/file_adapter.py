from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.base import IngestionAdapter
from groundcortex.pipeline.models import Experience


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ──────────────────────────────────────────────────────────────────────────────
# Stage 1 - Heading split
# ──────────────────────────────────────────────────────────────────────────────

def _split_on_headings(content: str, depth: int) -> list[tuple[str, str]]:
    """Split content on Markdown headings up to the given depth.

    Returns (heading_chain, body) pairs where heading_chain is the full
    ancestor heading path (e.g. '# Title\n## Section') prepended for context.
    For headingless files returns a single ('', full_content) pair.
    """
    pattern = re.compile(rf"^(#{{{1},{depth}}})\s+\S[^\n]*$", re.MULTILINE)
    matches = list(pattern.finditer(content))

    if not matches:
        stripped = content.strip()
        return [("", stripped)] if stripped else []

    chain: dict[int, str] = {}
    sections: list[tuple[str, str]] = []

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading_text = m.group(0).strip()

        # Update chain: set current level, clear all deeper levels.
        chain[level] = heading_text
        for lvl in list(chain.keys()):
            if lvl > level:
                del chain[lvl]

        heading_chain = "\n".join(chain[lvl] for lvl in sorted(chain.keys()))

        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
        body = content[body_start:body_end].strip()

        if body:
            sections.append((heading_chain, body))

    return sections


# ──────────────────────────────────────────────────────────────────────────────
# Stage 2 - Paragraph split
# ──────────────────────────────────────────────────────────────────────────────

def _split_on_paragraphs(body: str, splitter: str, min_chars: int) -> list[str]:
    """Split body on splitter; discard paragraphs shorter than min_chars (0 = keep all)."""
    parts = body.split(splitter)
    result = [p.strip() for p in parts if p.strip()]
    if min_chars > 0:
        result = [p for p in result if len(p) >= min_chars]
    return result or [body.strip()]


# ──────────────────────────────────────────────────────────────────────────────
# Stage 3 - Word split
# ──────────────────────────────────────────────────────────────────────────────

def _split_on_words(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks of at most size words."""
    words = text.split()
    if len(words) <= size:
        return [text]

    step = max(1, size - overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += step

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

def _split_sections(content: str, config: GroundCortexConfig) -> list[str]:
    """Run the three-stage cascade and return a flat list of raw_content strings.

    Each string has its parent heading chain prepended (when one exists) so the
    Q&A generator has full context about where the chunk appears in the document.
    """
    chunks: list[str] = []

    for heading_chain, body in _split_on_headings(content, config.section_depth):
        if config.paragraph_split_enabled:
            paragraphs = _split_on_paragraphs(
                body, config.paragraph_splitter, config.paragraph_min_chars
            )
        else:
            paragraphs = [body] if body.strip() else []

        for para in paragraphs:
            if config.word_split_enabled:
                word_chunks = _split_on_words(
                    para, config.word_split_size, config.word_split_overlap
                )
            else:
                word_chunks = [para]

            for chunk in word_chunks:
                final = (heading_chain + "\n" + chunk).strip() if heading_chain else chunk.strip()
                if final:
                    chunks.append(final)

    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# Shared parse helper
# ──────────────────────────────────────────────────────────────────────────────

def parse_content(
    content: str,
    source_id: str,
    adapter_name: str,
    db: Database,
    config: GroundCortexConfig,
) -> list[Experience]:
    """Core parsing logic shared by FileAdapter and RemoteFileAdapter.

    1. Hash the full content; skip if unchanged.
    2. Supersede all old experiences for this source.
    3. Split into chunks via the cascading splitter and emit new pending Experiences.
    4. Update source_files record.
    """
    file_hash = _sha256(content)
    stored_hash = db.get_file_hash(source_id)

    if stored_hash == file_hash:
        return []  # nothing changed

    db.supersede_source(source_id)

    now = _now_iso()
    new_experiences: list[Experience] = []

    for raw_content in _split_sections(content, config):
        exp = Experience(
            source=source_id,
            raw_content=raw_content,
            entities=[],
            content_hash=_sha256(raw_content),
            status="pending",
            created_at=now,
        )
        db.add_experience(exp)
        new_experiences.append(exp)

    db.upsert_file(source_id, adapter_name, file_hash, now)
    return new_experiences


# ──────────────────────────────────────────────────────────────────────────────
# Adapter
# ──────────────────────────────────────────────────────────────────────────────

class FileAdapter(IngestionAdapter):
    """Reads local files listed in GROUNDCORTEX_SOURCE_PATHS."""

    def __init__(self, config: GroundCortexConfig, db: Database) -> None:
        self._config = config
        self._db = db

    def ingest(self) -> list[Experience]:
        results: list[Experience] = []
        for path in self._config.source_paths:
            path = Path(path).expanduser()
            if not path.exists():
                continue
            content = path.read_text(encoding="utf-8", errors="replace")
            source_id = f"file:{path}"
            new = parse_content(content, source_id, "file", self._db, self._config)
            results.extend(new)
        return results
