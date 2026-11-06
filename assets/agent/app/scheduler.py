"""Job scheduling for monitoring and reporting tasks."""

import logging
import os

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger

logger = logging.getLogger(__name__)


def get_config() -> dict:
    """Load scheduler configuration from environment variables."""
    return {
        "monitoring_interval_hours": int(os.getenv("MONITORING_INTERVAL_HOURS", "4")),
        "report_timezone": os.getenv("REPORT_TIMEZONE", "UTC"),
    }


def setup_monitoring_job(scheduler: AsyncIOScheduler, job_func, interval_hours: int | None = None):
    """
    Set up the invoice monitoring job using APScheduler IntervalTrigger.

    Args:
        scheduler: AsyncIOScheduler instance
        job_func: Async function to execute for monitoring
        interval_hours: Monitoring interval in hours (defaults to config)
    """
    config = get_config()
    if interval_hours is None:
        interval_hours = config["monitoring_interval_hours"]

    try:
        trigger = IntervalTrigger(hours=interval_hours)
        scheduler.add_job(
            job_func, trigger=trigger, id="invoice_monitoring", name="Invoice Monitoring Job", replace_existing=True
        )
        logger.info(f"Monitoring job scheduled: every {interval_hours} hours")
    except Exception as e:
        logger.error(f"Failed to schedule monitoring job: {e}", exc_info=True)
        raise


def setup_reporting_job(scheduler: AsyncIOScheduler, job_func, timezone_str: str | None = None):
    """
    Set up the weekly reporting job using APScheduler CronTrigger (Monday 8 AM).

    Args:
        scheduler: AsyncIOScheduler instance
        job_func: Async function to execute for reporting
        timezone_str: Timezone name (defaults to config)
    """
    config = get_config()
    if timezone_str is None:
        timezone_str = config["report_timezone"]

    try:
        timezone = pytz.timezone(timezone_str)

        # CronTrigger: Monday at 8:00 AM
        trigger = CronTrigger(day_of_week="mon", hour=8, minute=0, timezone=timezone)

        scheduler.add_job(
            job_func, trigger=trigger, id="weekly_reporting", name="Weekly Reporting Job", replace_existing=True
        )
        logger.info(f"Reporting job scheduled: Monday 8:00 AM {timezone_str}")
    except Exception as e:
        logger.error(f"Failed to schedule reporting job: {e}", exc_info=True)
        raise


async def monitoring_job_wrapper(monitoring_func):
    """
    Wrapper for monitoring job with error handling and logging.

    Args:
        monitoring_func: Async function to execute monitoring logic
    """
    try:
        logger.info("Starting invoice monitoring job")
        await monitoring_func()
        logger.info("Invoice monitoring job completed successfully")
    except Exception as e:
        logger.error(f"Monitoring job failed: {e}", exc_info=True)


async def reporting_job_wrapper(reporting_func):
    """
    Wrapper for reporting job with error handling and logging.

    Args:
        reporting_func: Async function to execute reporting logic
    """
    try:
        logger.info("Starting weekly reporting job")
        await reporting_func()
        logger.info("Weekly reporting job completed successfully")
    except Exception as e:
        logger.error(f"Reporting job failed: {e}", exc_info=True)
