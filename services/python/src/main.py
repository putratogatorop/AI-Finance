import logging
import signal
import sys
import threading

from apscheduler.schedulers.background import BackgroundScheduler

from src.config import Settings
from src.data.binance_client import BinanceClient
from src.data.coingecko_client import CoinGeckoClient
from src.data.csv_storage import CsvStorage
from src.data.etl import EtlPipeline
from src.db.engine import create_db_engine, create_session_factory
from src.db.migrations import run_migrations
from src.scheduler.jobs import JobRunner

logger = logging.getLogger(__name__)


def get_settings() -> Settings:
    return Settings()


def main() -> None:
    settings = get_settings()
    logging.basicConfig(level=settings.LOG_LEVEL)
    logger.info("AI-Finance Python service starting...")

    engine = create_db_engine(settings)
    run_migrations(engine)
    session_factory = create_session_factory(engine)
    session = session_factory()

    scheduler = None

    if settings.SCHEDULER_ENABLED:
        binance = BinanceClient(rate_limit_ms=settings.BINANCE_RATE_LIMIT_MS)
        coingecko = CoinGeckoClient(api_key=settings.COINGECKO_API_KEY)
        csv_storage = CsvStorage(base_path=settings.DATA_LAKE_PATH)
        etl = EtlPipeline(csv_storage=csv_storage, session=session)
        job_runner = JobRunner(
            binance_client=binance, coingecko_client=coingecko,
            csv_storage=csv_storage, etl=etl,
            assets=list(settings.DEFAULT_ASSETS),
        )

        scheduler = BackgroundScheduler()
        scheduler.add_job(job_runner.run_hourly_ingestion, "interval", hours=1, id="hourly_ingestion")
        scheduler.add_job(job_runner.run_daily_aggregation, "cron", hour=0, minute=30, id="daily_aggregation")
        scheduler.add_job(job_runner.run_weekly_fundamentals, "cron", day_of_week="sun", hour=1, id="weekly_fundamentals")
        scheduler.start()
        logger.info("Scheduler started with hourly, daily, and weekly jobs")

    shutdown_event = threading.Event()

    def handle_signal(signum, frame):
        logger.info("Received shutdown signal")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    logger.info("AI-Finance Python service running...")
    shutdown_event.wait()

    if scheduler:
        scheduler.shutdown()
    session.close()
    engine.dispose()
    logger.info("AI-Finance Python service shut down.")


if __name__ == "__main__":
    main()
