import logging
import time
from datetime import datetime, timedelta, timezone

from src.data.binance_client import BinanceClient
from src.data.csv_storage import CsvStorage

logger = logging.getLogger(__name__)


class Backfiller:
    def __init__(self, binance_client: BinanceClient, csv_storage: CsvStorage) -> None:
        self.binance = binance_client
        self.csv = csv_storage

    def generate_date_range(self, start_date: str, end_date: str) -> list[str]:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        return dates

    def backfill_asset(self, asset: str, start_date: str, end_date: str) -> int:
        dates = self.generate_date_range(start_date, end_date)
        total_saved = 0
        for date_str in dates:
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                rows = self.binance.fetch_historical_hourly(asset, since=dt, limit=24)
                if rows:
                    date_rows: dict[str, list[dict]] = {}
                    for row in rows:
                        row_date = row["timestamp"][:10]
                        date_rows.setdefault(row_date, []).append(row)
                    for rd, rd_rows in date_rows.items():
                        self.csv.save_hourly(asset, rd, rd_rows)
                        total_saved += len(rd_rows)
                logger.info(f"Backfilled {asset} for {date_str}: {len(rows)} candles")
                time.sleep(0.5)
            except Exception:
                logger.exception(f"Backfill failed for {asset} on {date_str}")
        return total_saved

    def backfill_all(self, assets: list[str], start_date: str, end_date: str) -> dict[str, int]:
        results = {}
        for asset in assets:
            logger.info(f"Starting backfill for {asset} from {start_date} to {end_date}")
            count = self.backfill_asset(asset, start_date, end_date)
            results[asset] = count
            logger.info(f"Completed backfill for {asset}: {count} total candles")
        return results
