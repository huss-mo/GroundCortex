"""Tests for file ingestion - cascading splitter and parse_content."""
from __future__ import annotations

import pytest

from groundcortex.config import GroundCortexConfig
from groundcortex.ingestion.file_adapter import (
    FileAdapter,
    _split_on_headings,
    _split_on_paragraphs,
    _split_on_words,
    _split_sections,
    parse_content,
)


def _cfg(tmp_path, **kwargs) -> GroundCortexConfig:
    kwargs.setdefault("root_dir", tmp_path)
    return GroundCortexConfig(_env_file=None, **kwargs)


SOURCE_ID = "file:notes.md"


# ---------------------------------------------------------------------------
# Stage 1 — Heading split
# ---------------------------------------------------------------------------

class TestHeadingSplit:
    def test_headingless_file_is_one_section(self):
        sections = _split_on_headings("Plain content.", depth=3)
        assert len(sections) == 1
        assert sections[0] == ("", "Plain content.")

    def test_empty_file_returns_empty(self):
        assert _split_on_headings("", depth=3) == []

    def test_whitespace_only_returns_empty(self):
        assert _split_on_headings("   \n  ", depth=3) == []

    def test_depth1_splits_only_h1(self):
        content = "# Title\nBody.\n## Sub\nMore."
        sections = _split_on_headings(content, depth=1)
        # depth=1 matches only '#'; '## Sub' is treated as body text
        assert len(sections) == 1
        assert "## Sub" in sections[0][1]

    def test_depth2_splits_h1_and_h2(self):
        content = "# Title\n## Section A\nBody A.\n## Section B\nBody B."
        sections = _split_on_headings(content, depth=2)
        assert len(sections) == 2

    def test_depth3_splits_h1_h2_h3(self):
        content = "## Section\n### Sub A\nBody A.\n### Sub B\nBody B."
        sections = _split_on_headings(content, depth=3)
        assert len(sections) == 2

    def test_h1_included_in_chain_for_h2(self):
        content = "# Main\n## Section\nContent."
        sections = _split_on_headings(content, depth=2)
        # '# Main' body is empty so only Section yields an entry
        assert len(sections) == 1
        chain = sections[0][0]
        assert "# Main" in chain
        assert "## Section" in chain

    def test_heading_chain_prepended_as_ancestors(self):
        content = "# Doc\n## Chapter\n### Topic\nFact here."
        sections = _split_on_headings(content, depth=3)
        assert len(sections) == 1
        chain, body = sections[0]
        assert chain == "# Doc\n## Chapter\n### Topic"
        assert body == "Fact here."

    def test_nested_headings_chain_resets_on_sibling(self):
        content = "## Section 1\n### Sub 1.1\nBody 1.\n## Section 2\nBody 2."
        sections = _split_on_headings(content, depth=3)
        # Section 1 has no body (immediately followed by ###), Sub 1.1 → sections[0],
        # Section 2 → sections[1]. Chain for Section 2 must not include Sub 1.1.
        assert len(sections) == 2
        chain_s2 = sections[1][0]
        assert "Sub 1.1" not in chain_s2
        assert "## Section 2" in chain_s2

    def test_empty_section_body_skipped(self):
        content = "## Topic One\n\n## Topic Two\nHas content."
        sections = _split_on_headings(content, depth=3)
        assert len(sections) == 1
        assert "Has content." in sections[0][1]

    def test_date_heading_splits_correctly(self):
        content = "## 2026-05-17\nContent here."
        sections = _split_on_headings(content, depth=3)
        assert len(sections) == 1
        assert "## 2026-05-17" in sections[0][0]

    def test_depth3_splits_on_h1_by_default(self):
        content = "# Title\nSome content here."
        sections = _split_on_headings(content, depth=3)
        assert len(sections) == 1
        assert sections[0][0] == "# Title"

    def test_section_bodies_are_correct(self):
        content = "## Topic One\nFact one.\n## Topic Two\nFact two."
        sections = _split_on_headings(content, depth=3)
        assert sections[0][1] == "Fact one."
        assert sections[1][1] == "Fact two."


# ---------------------------------------------------------------------------
# Stage 2 — Paragraph split
# ---------------------------------------------------------------------------

class TestParagraphSplit:
    def test_splits_on_double_newline(self):
        body = "First paragraph.\n\nSecond paragraph."
        result = _split_on_paragraphs(body, "\n\n", 0)
        assert result == ["First paragraph.", "Second paragraph."]

    def test_custom_splitter_single_newline(self):
        body = "Line one.\nLine two."
        result = _split_on_paragraphs(body, "\n", 0)
        assert result == ["Line one.", "Line two."]

    def test_min_chars_filters_short_paragraphs(self):
        body = "Short.\n\nThis paragraph is long enough to survive filtering."
        result = _split_on_paragraphs(body, "\n\n", 25)
        assert len(result) == 1
        assert "long enough" in result[0]

    def test_min_chars_zero_keeps_all(self):
        body = "Hi.\n\nBye."
        result = _split_on_paragraphs(body, "\n\n", 0)
        assert len(result) == 2

    def test_short_fact_survives_at_25_threshold(self):
        # "the owner hates long sentences" is 31 chars — above 25
        body = "the owner hates long sentences"
        result = _split_on_paragraphs(body, "\n\n", 25)
        assert result == [body]

    def test_all_filtered_returns_original_body(self):
        # If everything is too short, fall back to the whole body
        body = "Hi.\n\nOk."
        result = _split_on_paragraphs(body, "\n\n", 100)
        assert result == [body.strip()]

    def test_multiple_blank_lines_between_paragraphs(self):
        # split() on "\n\n" won't collapse "\n\n\n", leaving an empty part
        body = "Para one.\n\n\nPara two."
        result = _split_on_paragraphs(body, "\n\n", 0)
        # Empty string between splits is filtered out by the `if p.strip()` check
        assert "Para one." in result
        assert "Para two." in result


# ---------------------------------------------------------------------------
# Stage 3 — Word split
# ---------------------------------------------------------------------------

class TestWordSplit:
    def _words(self, n: int) -> str:
        return " ".join(f"word{i}" for i in range(n))

    def test_short_text_not_split(self):
        text = self._words(50)
        result = _split_on_words(text, size=50, overlap=5)
        assert result == [text]

    def test_long_text_splits_into_chunks(self):
        text = self._words(120)
        result = _split_on_words(text, size=50, overlap=5)
        assert len(result) == 3

    def test_overlap_shared_words(self):
        text = self._words(60)
        result = _split_on_words(text, size=50, overlap=5)
        # Last 5 words of chunk 0 == first 5 words of chunk 1
        chunk0_tail = result[0].split()[-5:]
        chunk1_head = result[1].split()[:5]
        assert chunk0_tail == chunk1_head

    def test_word_split_disabled_keeps_long_text(self):
        # _split_on_words is called with size=50; if we want to test "disabled"
        # we just don't call it. Verify the function itself returns original
        # text when len(words) == size exactly.
        text = self._words(50)
        result = _split_on_words(text, size=50, overlap=0)
        assert len(result) == 1
        assert result[0] == text

    def test_overlap_zero_no_shared_words(self):
        text = self._words(100)
        result = _split_on_words(text, size=50, overlap=0)
        chunk0_tail = result[0].split()[-1]
        chunk1_head = result[1].split()[0]
        assert chunk0_tail != chunk1_head

    def test_single_word_text(self):
        result = _split_on_words("hello", size=50, overlap=5)
        assert result == ["hello"]


# ---------------------------------------------------------------------------
# Orchestrator (_split_sections)
# ---------------------------------------------------------------------------

class TestCascade:
    def test_headingless_short_text_is_one_chunk(self, tmp_path):
        cfg = _cfg(tmp_path)
        result = _split_sections("A single fact.", cfg)
        assert result == ["A single fact."]

    def test_heading_chain_prepended_to_chunk(self, tmp_path):
        cfg = _cfg(tmp_path, word_split_enabled=False, paragraph_split_enabled=False)
        content = "## My Section\nContent here."
        result = _split_sections(content, cfg)
        assert len(result) == 1
        assert result[0].startswith("## My Section")
        assert "Content here." in result[0]

    def test_paragraph_split_disabled(self, tmp_path):
        cfg = _cfg(tmp_path, paragraph_split_enabled=False, word_split_enabled=False)
        content = "## Section\nPara one.\n\nPara two."
        result = _split_sections(content, cfg)
        assert len(result) == 1

    def test_word_split_disabled(self, tmp_path):
        cfg = _cfg(tmp_path, word_split_enabled=False)
        words = " ".join(f"w{i}" for i in range(200))
        result = _split_sections(words, cfg)
        assert len(result) == 1

    def test_full_cascade(self, tmp_path):
        long_para = " ".join(f"word{i}" for i in range(120))
        # "This is a known fact." is 21 chars — set min_chars=0 to keep it
        content = (
            "# Doc\n"
            "## Section A\n"
            "This is a known fact about the system.\n\n"
            f"{long_para}\n"
            "## Section B\n"
            "Another fact about the system."
        )
        cfg = _cfg(tmp_path, word_split_size=50, word_split_overlap=5)
        result = _split_sections(content, cfg)

        # Short-ish fact (≥25 chars) stays as one chunk, long para → 3 chunks,
        # Section B body stays as one chunk → 5 total
        assert len(result) == 5

        # All chunks from Section A carry its heading context
        section_a_chunks = [c for c in result if "## Section A" in c]
        assert len(section_a_chunks) == 4  # short fact + 3 word chunks

        # Section B chunk carries its heading
        assert any("## Section B" in c for c in result)

    def test_empty_content_returns_empty(self, tmp_path):
        cfg = _cfg(tmp_path)
        assert _split_sections("", cfg) == []


# ---------------------------------------------------------------------------
# parse_content
# ---------------------------------------------------------------------------

class TestParseContent:
    def test_new_file_creates_pending_experience(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        exps = parse_content("Alice is a software engineer.", SOURCE_ID, "file", db, cfg)
        assert len(exps) == 1
        assert exps[0].status == "pending"
        assert exps[0].source == SOURCE_ID
        assert exps[0].raw_content == "Alice is a software engineer."

    def test_new_file_updates_db_pending_count(self, db, tmp_path):
        parse_content("Some fact.", SOURCE_ID, "file", db, _cfg(tmp_path))
        assert db.count_pending() == 1

    def test_unchanged_file_returns_empty_list(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        content = "Alice is a software engineer."
        parse_content(content, SOURCE_ID, "file", db, cfg)
        result = parse_content(content, SOURCE_ID, "file", db, cfg)
        assert result == []

    def test_unchanged_file_does_not_duplicate_experiences(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        content = "Stable fact."
        parse_content(content, SOURCE_ID, "file", db, cfg)
        parse_content(content, SOURCE_ID, "file", db, cfg)
        assert db.count_pending() == 1

    def test_changed_file_supersedes_old_experiences(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        parse_content("Old content.", SOURCE_ID, "file", db, cfg)
        parse_content("New content.", SOURCE_ID, "file", db, cfg)
        scope = db.get_training_scope()
        assert all(e.raw_content == "New content." for e in scope)

    def test_changed_file_creates_new_pending(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        parse_content("Old content.", SOURCE_ID, "file", db, cfg)
        result = parse_content("New content.", SOURCE_ID, "file", db, cfg)
        assert len(result) == 1
        assert result[0].raw_content == "New content."

    def test_sectioned_file_creates_one_experience_per_section(self, db, tmp_path):
        cfg = _cfg(tmp_path, paragraph_split_enabled=False, word_split_enabled=False)
        content = (
            "## Topic One\nFact one here.\n"
            "## Topic Two\nFact two here.\n"
        )
        exps = parse_content(content, SOURCE_ID, "file", db, cfg)
        assert len(exps) == 2

    def test_file_hash_stored_after_first_parse(self, db, tmp_path):
        parse_content("Some content.", SOURCE_ID, "file", db, _cfg(tmp_path))
        stored = db.get_file_hash(SOURCE_ID)
        assert stored is not None
        assert len(stored) == 64

    def test_hash_updated_after_change(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        parse_content("Old.", SOURCE_ID, "file", db, cfg)
        hash1 = db.get_file_hash(SOURCE_ID)
        parse_content("New.", SOURCE_ID, "file", db, cfg)
        hash2 = db.get_file_hash(SOURCE_ID)
        assert hash1 != hash2

    def test_empty_content_creates_no_experiences(self, db, tmp_path):
        exps = parse_content("", "file:empty.md", "file", db, _cfg(tmp_path))
        assert exps == []

    def test_multiple_sources_tracked_independently(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        parse_content("Content A.", "file:A.md", "file", db, cfg)
        parse_content("Content B.", "file:B.md", "file", db, cfg)
        assert db.count_pending() == 2

    def test_changing_one_source_does_not_affect_other(self, db, tmp_path):
        cfg = _cfg(tmp_path)
        parse_content("Content A.", "file:A.md", "file", db, cfg)
        parse_content("Content B.", "file:B.md", "file", db, cfg)
        parse_content("Content A modified.", "file:A.md", "file", db, cfg)
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
            root_dir=tmp_path,
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
