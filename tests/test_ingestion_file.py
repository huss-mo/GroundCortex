"""Tests for file ingestion - parse_content, _classify_type, _split_sections, FileAdapter."""
from __future__ import annotations

import pytest

from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.file_adapter import (
    FileAdapter,
    _classify_type,
    _split_sections,
    parse_content,
)


# ---------------------------------------------------------------------------
# _classify_type
# ---------------------------------------------------------------------------

class TestClassifyType:
    def test_user_md_is_preference(self):
        assert _classify_type("file:/home/user/.groundmemory/default/USER.md") == "preference"

    def test_user_md_case_insensitive(self):
        assert _classify_type("file:user.md") == "preference"

    def test_daily_slash_is_mindset(self):
        assert _classify_type("file:/notes/daily/2026-05-17.md") == "mindset"

    def test_daily_backslash_is_mindset(self):
        assert _classify_type("file:C:\\notes\\daily\\2026-05-17.md") == "mindset"

    def test_memory_md_is_fact(self):
        assert _classify_type("file:MEMORY.md") == "fact"

    def test_arbitrary_file_is_fact(self):
        assert _classify_type("file:project_notes.txt") == "fact"

    def test_remote_url_is_fact(self):
        assert _classify_type("http://server/facts.md") == "fact"


# ---------------------------------------------------------------------------
# _split_sections
# ---------------------------------------------------------------------------

class TestSplitSections:
    def test_plain_file_is_one_section(self):
        content = "This is a plain file with no GM headers."
        sections = _split_sections(content)
        assert len(sections) == 1
        assert sections[0][1] == content

    def test_plain_file_timestamp_is_empty_string(self):
        sections = _split_sections("Some content.")
        assert sections[0][0] == ""

    def test_gm_format_splits_on_date_headers(self):
        content = "## 2026-05-17\nFact one.\n## 2026-05-18\nFact two.\n"
        sections = _split_sections(content)
        assert len(sections) == 2

    def test_gm_format_date_time_header(self):
        content = "## 2026-05-17 09:30\nContent here.\n"
        sections = _split_sections(content)
        assert len(sections) == 1
        assert sections[0][0] == "2026-05-17 09:30"

    def test_gm_format_date_only_header(self):
        content = "## 2026-05-17\nContent here.\n"
        sections = _split_sections(content)
        assert sections[0][0] == "2026-05-17"

    def test_gm_format_section_content_correct(self):
        content = "## 2026-05-17\nFact one.\n## 2026-05-18\nFact two.\n"
        sections = _split_sections(content)
        assert "Fact one." in sections[0][1]
        assert "Fact two." in sections[1][1]

    def test_empty_sections_skipped(self):
        content = "## 2026-05-17\n\n## 2026-05-18\nHas content.\n"
        sections = _split_sections(content)
        assert len(sections) == 1
        assert "Has content." in sections[0][1]

    def test_empty_file_returns_empty(self):
        assert _split_sections("") == []

    def test_whitespace_only_returns_empty(self):
        assert _split_sections("   \n  ") == []

    def test_markdown_heading_not_treated_as_gm_header(self):
        # A regular markdown heading (# or ### etc.) is not a GM section header
        content = "# Title\nSome content here."
        sections = _split_sections(content)
        assert len(sections) == 1


# ---------------------------------------------------------------------------
# parse_content
# ---------------------------------------------------------------------------

class TestParseContent:
    def test_new_file_creates_pending_experience(self, db):
        exps = parse_content("Alice is a software engineer.", "file:notes.md", "file", db)
        assert len(exps) == 1
        assert exps[0].status == "pending"
        assert exps[0].source == "file:notes.md"
        assert exps[0].raw_content == "Alice is a software engineer."

    def test_new_file_updates_db_pending_count(self, db):
        parse_content("Some fact.", "file:notes.md", "file", db)
        assert db.count_pending() == 1

    def test_unchanged_file_returns_empty_list(self, db):
        content = "Alice is a software engineer."
        parse_content(content, "file:notes.md", "file", db)
        result = parse_content(content, "file:notes.md", "file", db)
        assert result == []

    def test_unchanged_file_does_not_duplicate_experiences(self, db):
        content = "Stable fact."
        parse_content(content, "file:notes.md", "file", db)
        parse_content(content, "file:notes.md", "file", db)
        assert db.count_pending() == 1

    def test_changed_file_supersedes_old_experiences(self, db):
        parse_content("Old content.", "file:notes.md", "file", db)
        parse_content("New content.", "file:notes.md", "file", db)
        scope = db.get_training_scope()
        assert all(e.raw_content == "New content." for e in scope)

    def test_changed_file_creates_new_pending(self, db):
        parse_content("Old content.", "file:notes.md", "file", db)
        result = parse_content("New content.", "file:notes.md", "file", db)
        assert len(result) == 1
        assert result[0].raw_content == "New content."

    def test_gm_format_creates_one_experience_per_section(self, db):
        content = (
            "## 2026-05-17\nFact one here.\n"
            "## 2026-05-18\nFact two here.\n"
        )
        exps = parse_content(content, "file:MEMORY.md", "file", db)
        assert len(exps) == 2

    def test_file_hash_stored_after_first_parse(self, db):
        parse_content("Some content.", "file:notes.md", "file", db)
        stored = db.get_file_hash("file:notes.md")
        assert stored is not None
        assert len(stored) == 64  # SHA-256 hex digest length

    def test_hash_updated_after_change(self, db):
        parse_content("Old.", "file:notes.md", "file", db)
        hash1 = db.get_file_hash("file:notes.md")
        parse_content("New.", "file:notes.md", "file", db)
        hash2 = db.get_file_hash("file:notes.md")
        assert hash1 != hash2

    def test_type_classification_user_md(self, db):
        exps = parse_content("Pref content.", "file:/home/x/USER.md", "file", db)
        assert exps[0].type == "preference"

    def test_type_classification_daily(self, db):
        exps = parse_content("Mindset content.", "file:/notes/daily/today.md", "file", db)
        assert exps[0].type == "mindset"

    def test_type_classification_fact(self, db):
        exps = parse_content("Fact content.", "file:MEMORY.md", "file", db)
        assert exps[0].type == "fact"

    def test_empty_content_creates_no_experiences(self, db):
        exps = parse_content("", "file:empty.md", "file", db)
        assert exps == []

    def test_multiple_sources_tracked_independently(self, db):
        parse_content("Content A.", "file:A.md", "file", db)
        parse_content("Content B.", "file:B.md", "file", db)
        assert db.count_pending() == 2

    def test_changing_one_source_does_not_affect_other(self, db):
        parse_content("Content A.", "file:A.md", "file", db)
        parse_content("Content B.", "file:B.md", "file", db)
        parse_content("Content A modified.", "file:A.md", "file", db)
        scope = db.get_training_scope()
        b_exps = [e for e in scope if e.source == "file:B.md"]
        assert len(b_exps) == 1
        assert b_exps[0].raw_content == "Content B."


# ---------------------------------------------------------------------------
# FileAdapter
# ---------------------------------------------------------------------------

class TestFileAdapter:
    def _make_config(self, tmp_path, paths):
        return GroundCortexConfig(
            _env_file=None,
            output_dir=tmp_path / "adapters",
            source_paths=paths,
        )

    def test_ingest_reads_existing_file(self, db, tmp_path):
        md = tmp_path / "notes.md"
        md.write_text("Alice is a software engineer.", encoding="utf-8")
        cfg = self._make_config(tmp_path, [md])
        result = FileAdapter(cfg, db).ingest()
        assert len(result) == 1

    def test_ingest_skips_missing_file(self, db, tmp_path):
        cfg = self._make_config(tmp_path, [tmp_path / "does_not_exist.md"])
        result = FileAdapter(cfg, db).ingest()
        assert result == []

    def test_ingest_multiple_files(self, db, tmp_path):
        f1 = tmp_path / "a.md"
        f2 = tmp_path / "b.md"
        f1.write_text("Content of A.", encoding="utf-8")
        f2.write_text("Content of B.", encoding="utf-8")
        cfg = self._make_config(tmp_path, [f1, f2])
        result = FileAdapter(cfg, db).ingest()
        assert len(result) == 2

    def test_ingest_second_run_unchanged_is_empty(self, db, tmp_path):
        md = tmp_path / "notes.md"
        md.write_text("Stable content.", encoding="utf-8")
        cfg = self._make_config(tmp_path, [md])
        adapter = FileAdapter(cfg, db)
        adapter.ingest()
        result = adapter.ingest()
        assert result == []

    def test_ingest_source_id_uses_full_path(self, db, tmp_path):
        md = tmp_path / "notes.md"
        md.write_text("Some fact.", encoding="utf-8")
        cfg = self._make_config(tmp_path, [md])
        result = FileAdapter(cfg, db).ingest()
        assert "notes.md" in result[0].source

    def test_ingest_empty_source_paths_returns_empty(self, db, tmp_path):
        cfg = self._make_config(tmp_path, [])
        result = FileAdapter(cfg, db).ingest()
        assert result == []
