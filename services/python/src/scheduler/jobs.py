import logging
from datetime import UTC, datetime

from src.data.binance_client import BinanceClient
from src.data.coingecko_client import CoinGeckoClient
from src.data.csv_storage import CsvStorage
from src.data.etl import EtlPipeline

logger = logging.getLogger(__name__)


class JobRunner:
    def __init__(self, binance_client: BinanceClient, coingecko_client: CoinGeckoClient,
                 csv_storage: CsvStorage, etl: EtlPipeline, assets: list[str]) -> None:
        self.binance = binance_client
        self.coingecko = coingecko_client
        self.csv = csv_storage
        self.etl = etl
        self.assets = assets

    def run_hourly_ingestion(self) -> None:
        now = datetime.now(UTC)
        date_str = now.strftime("%Y-%m-%d")
        logger.info(f"Starting hourly ingestion for {len(self.assets)} assets")
        for asset in self.assets:
            try:
                rows = self.binance.fetch_hourly_ohlcv(asset, limit=1)
                self.csv.save_hourly(asset, date_str, rows)
                self.etl.load_hourly(asset, date_str)
                logger.info(f"Hourly ingestion complete for {asset}")
            except Exception:
                logger.exception(f"Hourly ingestion failed for {asset}")

    def run_daily_aggregation(self) -> None:
        now = datetime.now(UTC)
        date_str = now.strftime("%Y-%m-%d")
        logger.info(f"Starting daily aggregation for {len(self.assets)} assets")
        for asset in self.assets:
            try:
                self.etl.aggregate_daily(asset, date_str)
            except Exception:
                logger.exception(f"Daily aggregation failed for {asset}")

    def run_weekly_fundamentals(self) -> None:
        now = datetime.now(UTC)
        date_str = now.strftime("%Y-%m-%d")
        logger.info("Starting weekly fundamentals refresh")
        try:
            top_assets = self.coingecko.fetch_top_n_assets(n=20)
            self.assets = top_assets
            logger.info(f"Updated asset list: {top_assets}")
            fundamentals = self.coingecko.fetch_fundamentals(self.assets)
            for fund in fundamentals:
                self.csv.save_fundamentals(fund["asset"], date_str, fund)
            logger.info(f"Saved fundamentals for {len(fundamentals)} assets")
        except Exception:
            logger.exception("Weekly fundamentals refresh failed")
