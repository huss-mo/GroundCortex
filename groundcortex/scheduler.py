from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from groundcortex.config import GroundCortexConfig

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)


def start_scheduler(
    consolidation_fn: "Callable[[], Awaitable[dict]]",
    config: GroundCortexConfig,
) -> AsyncIOScheduler | None:
    """Start the APScheduler cron job for automatic consolidation.

    Returns the running scheduler instance, or None if cron is disabled.
    The scheduler runs in the same asyncio event loop as the FastAPI/FastMCP servers.
    """
    if not config.cron_enabled:
        logger.info("Cron scheduler disabled (GROUNDCORTEX_CRON_ENABLED=false).")
        return None

    scheduler = AsyncIOScheduler()
    trigger = CronTrigger.from_crontab(config.cron_schedule)

    scheduler.add_job(
        consolidation_fn,
        trigger,
        id="consolidation",
        replace_existing=True,
    )
    scheduler.start()

    logger.info(
        "Cron scheduler started. Schedule: %s", config.cron_schedule
    )
    return scheduler
