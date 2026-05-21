"""Shared fixtures for the GroundCortex test suite."""
from __future__ import annotations

import pytest

from groundcortex.buffer.db import Database
from groundcortex.config import GroundCortexConfig


@pytest.fixture()
def db(tmp_path):
    """Fresh isolated SQLite database for each test."""
    return Database(tmp_path / "test.db")


@pytest.fixture()
def config(tmp_path):
    """Minimal GroundCortexConfig that does not load .env and uses tmp dirs."""
    return GroundCortexConfig(
        _env_file=None,
        output_dir=tmp_path / "data" / "adapters",
        buffer_db=tmp_path / "test.db",
        source_paths=[],
        remote_source_urls=[],
        eval_enabled=False,
        model_name="test-model",
    )
