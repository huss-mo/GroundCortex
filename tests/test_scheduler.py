"""Tests for the APScheduler cron setup (scheduler.py)."""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from groundcortex.config import GroundCortexConfig
from groundcortex.scheduler import start_scheduler


def _cfg(tmp_path, enabled=True, schedule="0 2 * * *") -> GroundCortexConfig:
    return GroundCortexConfig(
        _env_file=None,
        output_dir=tmp_path / "adapters",
        cron_enabled=enabled,
        cron_schedule=schedule,
    )


# AsyncIOScheduler.start() requires a running asyncio event loop (it attaches to it).
# We patch start() so unit tests don't need to spin up an event loop.
# Job registration, trigger, and function binding all happen before start(), so they
# can be verified on the not-yet-started scheduler instance.


class TestStartScheduler:
    def test_cron_disabled_returns_none(self, tmp_path):
        result = start_scheduler(AsyncMock(), _cfg(tmp_path, enabled=False))
        assert result is None

    def test_cron_enabled_returns_async_io_scheduler(self, tmp_path):
        with patch.object(AsyncIOScheduler, "start"):
            scheduler = start_scheduler(AsyncMock(), _cfg(tmp_path, enabled=True))
        assert isinstance(scheduler, AsyncIOScheduler)

    def test_cron_enabled_calls_start(self, tmp_path):
        with patch.object(AsyncIOScheduler, "start") as mock_start:
            start_scheduler(AsyncMock(), _cfg(tmp_path, enabled=True))
        mock_start.assert_called_once()

    def test_consolidation_job_registered(self, tmp_path):
        with patch.object(AsyncIOScheduler, "start"):
            scheduler = start_scheduler(AsyncMock(), _cfg(tmp_path, enabled=True))
        job_ids = [j.id for j in scheduler.get_jobs()]
        assert "consolidation" in job_ids

    def test_exactly_one_job_registered(self, tmp_path):
        with patch.object(AsyncIOScheduler, "start"):
            scheduler = start_scheduler(AsyncMock(), _cfg(tmp_path, enabled=True))
        assert len(scheduler.get_jobs()) == 1

    def test_custom_schedule_accepted(self, tmp_path):
        with patch.object(AsyncIOScheduler, "start"):
            scheduler = start_scheduler(
                AsyncMock(), _cfg(tmp_path, enabled=True, schedule="30 6 * * 1")
            )
        job = next(j for j in scheduler.get_jobs() if j.id == "consolidation")
        assert job is not None

    def test_daily_schedule_accepted(self, tmp_path):
        with patch.object(AsyncIOScheduler, "start"):
            scheduler = start_scheduler(
                AsyncMock(), _cfg(tmp_path, enabled=True, schedule="0 0 * * *")
            )
        assert len(scheduler.get_jobs()) == 1

    def test_fn_registered_as_job_func(self, tmp_path):
        fn = AsyncMock()
        fn.__name__ = "test_consolidation_fn"
        with patch.object(AsyncIOScheduler, "start"):
            scheduler = start_scheduler(fn, _cfg(tmp_path, enabled=True))
        job = next(j for j in scheduler.get_jobs() if j.id == "consolidation")
        assert job.func is fn
