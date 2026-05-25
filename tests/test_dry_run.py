"""Tests for run_dry_run() and _write_dry_run_report() (consolidator.py)."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from groundcortex.config import GroundCortexConfig
from groundcortex.pipeline.models import Experience, TrainingExample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(coro):
    return asyncio.run(coro)


def _cfg(tmp_path: Path, source_paths=None) -> GroundCortexConfig:
    return GroundCortexConfig(
        _env_file=None,
        root_dir=tmp_path,
        source_paths=source_paths or [],
        remote_source_urls=[],
        eval_enabled=False,
        model_name="test-model",
    )


def _make_example(question: str = "Q?", answer: str = "A.", variant: str = "generated") -> TrainingExample:
    return TrainingExample(
        run_id="dry-run",
        variant=variant,
        messages=[
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
    )


def _make_experience(source: str = "file:test.md", content: str = "Some fact.") -> Experience:
    return Experience(
        source=source,
        raw_content=content,
        entities=[],
        content_hash="",
        status="pending",
        created_at="2025-01-01T00:00:00+00:00",
    )


def _fake_gen_fn(experiences_and_examples: list[tuple[Experience, list[TrainingExample]]]):
    """Returns a generator that yields pre-built examples from a lookup dict."""
    lookup = {exp.source + exp.raw_content: exs for exp, exs in experiences_and_examples}

    def _generator(exp: Experience, run_id: str) -> list[TrainingExample]:
        return lookup.get(exp.source + exp.raw_content, [_make_example()])

    return _generator


# ---------------------------------------------------------------------------
# TestRunDryRun
# ---------------------------------------------------------------------------

class TestRunDryRun:
    def test_skipped_when_no_source_paths(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        cfg = _cfg(tmp_path, source_paths=[])
        result = _run(run_dry_run(cfg))
        assert result["status"] == "skipped"
        assert result["reason"] == "no_sources"

    def test_skipped_total_chunks_is_zero(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        cfg = _cfg(tmp_path, source_paths=[])
        result = _run(run_dry_run(cfg))
        assert result["total_chunks"] == 0

    def test_skipped_when_source_file_does_not_exist(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        cfg = _cfg(tmp_path, source_paths=[str(tmp_path / "missing.md")])
        result = _run(run_dry_run(cfg))
        assert result["status"] == "skipped"

    def test_generates_examples_for_chunks(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        src = tmp_path / "notes.md"
        src.write_text("# Title\n\nA fact about the system.", encoding="utf-8")
        cfg = _cfg(tmp_path, source_paths=[str(src)])

        mock_manager = MagicMock()
        mock_manager.generate_base.return_value = json.dumps([
            {"question": "Q?", "answer": "A."}
        ])

        result = _run(run_dry_run(cfg, mock_manager))
        assert result["status"] == "ok"
        assert result["total_chunks"] >= 1

    def test_writes_dry_run_md_to_root_dir(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        src = tmp_path / "notes.md"
        src.write_text("A standalone fact.", encoding="utf-8")
        cfg = _cfg(tmp_path, source_paths=[str(src)])

        mock_manager = MagicMock()
        mock_manager.generate_base.return_value = json.dumps([
            {"question": "Q?", "answer": "A."}
        ])

        _run(run_dry_run(cfg, mock_manager))
        assert (tmp_path / "dry-run.md").exists()

    def test_no_db_parameter(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        import inspect
        sig = inspect.signature(run_dry_run)
        assert "db" not in sig.parameters

    def test_result_has_expected_keys(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        src = tmp_path / "notes.md"
        src.write_text("A fact.", encoding="utf-8")
        cfg = _cfg(tmp_path, source_paths=[str(src)])

        mock_manager = MagicMock()
        mock_manager.generate_base.return_value = json.dumps([
            {"question": "Q?", "answer": "A."}
        ])

        result = _run(run_dry_run(cfg, mock_manager))
        assert {"status", "total_chunks", "examples_generated", "output_path"} <= result.keys()

    def test_output_path_is_in_root_dir(self, tmp_path):
        from groundcortex.consolidator import run_dry_run
        src = tmp_path / "notes.md"
        src.write_text("A fact.", encoding="utf-8")
        cfg = _cfg(tmp_path, source_paths=[str(src)])

        mock_manager = MagicMock()
        mock_manager.generate_base.return_value = json.dumps([{"question": "Q?", "answer": "A."}])

        result = _run(run_dry_run(cfg, mock_manager))
        assert result["output_path"] == str(tmp_path / "dry-run.md")

    def test_does_not_write_to_db(self, tmp_path):
        """run_dry_run takes no db arg - verify no DB file is created by the function."""
        from groundcortex.consolidator import run_dry_run
        src = tmp_path / "notes.md"
        src.write_text("A fact.", encoding="utf-8")
        cfg = _cfg(tmp_path, source_paths=[str(src)])

        mock_manager = MagicMock()
        mock_manager.generate_base.return_value = json.dumps([{"question": "Q?", "answer": "A."}])

        db_path = tmp_path / "groundcortex.db"
        assert not db_path.exists()
        _run(run_dry_run(cfg, mock_manager))
        # dry-run.md was written but the DB was not touched
        assert not db_path.exists()


# ---------------------------------------------------------------------------
# TestWriteDryRunReport
# ---------------------------------------------------------------------------

class TestWriteDryRunReport:
    def _results(self, n: int = 2) -> list[tuple[Experience, list[TrainingExample]]]:
        return [
            (_make_experience(content=f"Fact number {i}."), [_make_example(f"Q{i}?", f"A{i}.")])
            for i in range(1, n + 1)
        ]

    def test_markdown_has_chunk_heading_per_experience(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        results = self._results(3)
        out = tmp_path / "dry-run.md"
        _write_dry_run_report(results, out)
        text = out.read_text(encoding="utf-8")
        assert "## Chunk 1 of 3" in text
        assert "## Chunk 2 of 3" in text
        assert "## Chunk 3 of 3" in text

    def test_source_shown_in_chunk_section(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        exp = _make_experience(source="file:/some/path.md", content="A fact.")
        out = tmp_path / "dry-run.md"
        _write_dry_run_report([(exp, [_make_example()])], out)
        text = out.read_text(encoding="utf-8")
        assert "file:/some/path.md" in text

    def test_content_lines_are_blockquoted(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        exp = _make_experience(content="Single line fact.")
        out = tmp_path / "dry-run.md"
        _write_dry_run_report([(exp, [_make_example()])], out)
        text = out.read_text(encoding="utf-8")
        assert "> Single line fact." in text

    def test_multiline_content_each_line_blockquoted(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        exp = _make_experience(content="Line one.\nLine two.\nLine three.")
        out = tmp_path / "dry-run.md"
        _write_dry_run_report([(exp, [_make_example()])], out)
        text = out.read_text(encoding="utf-8")
        assert "> Line one." in text
        assert "> Line two." in text
        assert "> Line three." in text

    def test_qa_fenced_code_block_is_valid_json(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        results = self._results(1)
        out = tmp_path / "dry-run.md"
        _write_dry_run_report(results, out)
        text = out.read_text(encoding="utf-8")
        # Extract content between ```json and ```
        start = text.index("```json\n") + len("```json\n")
        end = text.index("\n```", start)
        parsed = json.loads(text[start:end])
        assert isinstance(parsed, list)

    def test_json_objects_have_question_answer_variant(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        results = self._results(1)
        out = tmp_path / "dry-run.md"
        _write_dry_run_report(results, out)
        text = out.read_text(encoding="utf-8")
        start = text.index("```json\n") + len("```json\n")
        end = text.index("\n```", start)
        parsed = json.loads(text[start:end])
        assert all({"question", "answer", "variant"} <= set(obj.keys()) for obj in parsed)

    def test_report_header_has_total_chunks(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        results = self._results(4)
        out = tmp_path / "dry-run.md"
        _write_dry_run_report(results, out)
        text = out.read_text(encoding="utf-8")
        assert "Total chunks: 4" in text

    def test_overwrite_existing_file(self, tmp_path):
        from groundcortex.consolidator import _write_dry_run_report
        out = tmp_path / "dry-run.md"
        out.write_text("old content", encoding="utf-8")
        results = self._results(1)
        _write_dry_run_report(results, out)
        text = out.read_text(encoding="utf-8")
        assert "old content" not in text
        assert "## Chunk 1 of 1" in text
