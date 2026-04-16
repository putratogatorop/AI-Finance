"""Upsert CSV raw data → PostgreSQL.

Reads all 4h candle CSVs from data/raw/4h/ and upserts into:
  - asset_prices_hourly (4h candles)
  - asset_prices_daily (aggregated from 4h)

Uses batch inserts with ON CONFLICT for speed.
"""

import csv
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/4h")
BATCH_SIZE = 1000


def parse_symbol(symbol_usdt: str) -> str:
    """BTCUSDT → BTC"""
    return symbol_usdt.replace("USDT", "")


def upsert_hourly(engine, asset: str, rows: list[dict]):
    """Batch upsert 4h candles into asset_prices_hourly."""
    if not rows:
        return 0

    sql = text("""
        INSERT INTO asset_prices_hourly (asset, timestamp, open, high, low, close, volume)
        VALUES (:asset, :timestamp, :open, :high, :low, :close, :volume)
        ON CONFLICT (asset, timestamp) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high,
            low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def aggregate_daily(engine, asset: str):
    """Aggregate 4h candles → daily candles using SQL."""
    sql = text("""
        INSERT INTO asset_prices_daily (asset, date, open, high, low, close, volume)
        SELECT
            asset,
            date_trunc('day', timestamp) AS date,
            (array_agg(open ORDER BY timestamp))[1] AS open,
            MAX(high) AS high,
            MIN(low) AS low,
            (array_agg(close ORDER BY timestamp DESC))[1] AS close,
            SUM(volume) AS volume
        FROM asset_prices_hourly
        WHERE asset = :asset
        GROUP BY asset, date_trunc('day', timestamp)
        ON CONFLICT (asset, date) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high,
            low = EXCLUDED.low, close = EXCLUDED.close, volume = EXCLUDED.volume
    """)

    with engine.begin() as conn:
        result = conn.execute(sql, {"asset": asset})
        return result.rowcount


def process_symbol(engine, symbol_dir: Path) -> tuple[str, int, int]:
    """Process all CSV files for one symbol."""
    symbol_usdt = symbol_dir.name
    asset = parse_symbol(symbol_usdt)

    csv_files = sorted(symbol_dir.glob("*.csv"))
    if not csv_files:
        return asset, 0, 0

    all_rows = []
    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_ms = int(row["open_time"])
                    dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                    all_rows.append({
                        "asset": asset,
                        "timestamp": dt,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                    })
                except (ValueError, KeyError, OSError):
                    continue

    # Batch upsert
    hourly_count = 0
    for i in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[i:i + BATCH_SIZE]
        hourly_count += upsert_hourly(engine, asset, batch)

    # Aggregate daily
    daily_count = aggregate_daily(engine, asset)

    return asset, hourly_count, daily_count


def main():
    engine = create_engine(DB_URL)

    # Ensure tables exist
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS asset_prices_hourly (
                id SERIAL PRIMARY KEY,
                asset VARCHAR(20) NOT NULL,
                timestamp TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                CONSTRAINT uq_hourly_asset_ts UNIQUE (asset, timestamp)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS asset_prices_daily (
                id SERIAL PRIMARY KEY,
                asset VARCHAR(20) NOT NULL,
                date TIMESTAMPTZ NOT NULL,
                open DOUBLE PRECISION NOT NULL,
                high DOUBLE PRECISION NOT NULL,
                low DOUBLE PRECISION NOT NULL,
                close DOUBLE PRECISION NOT NULL,
                volume DOUBLE PRECISION NOT NULL,
                CONSTRAINT uq_daily_asset_date UNIQUE (asset, date)
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_hourly_asset ON asset_prices_hourly (asset)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_hourly_ts ON asset_prices_hourly (timestamp)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_daily_asset ON asset_prices_daily (asset)"))

    # Clear existing data for clean upsert
    logger.info("Clearing existing price data...")
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM asset_prices_hourly"))
        conn.execute(text("DELETE FROM asset_prices_daily"))

    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    total_symbols = len(symbol_dirs)
    logger.info(f"Upserting {total_symbols} symbols from {RAW_DIR}")
    logger.info("=" * 60)

    total_hourly = 0
    total_daily = 0
    start = time.time()

    for i, sym_dir in enumerate(symbol_dirs, 1):
        asset, h_count, d_count = process_symbol(engine, sym_dir)
        total_hourly += h_count
        total_daily += d_count

        if i % 10 == 0 or i == total_symbols:
            elapsed = time.time() - start
            rate = total_hourly / elapsed if elapsed > 0 else 0
            logger.info(
                f"  [{i}/{total_symbols}] {asset}: +{h_count:,} 4h, +{d_count:,} daily | "
                f"Total: {total_hourly:,} 4h, {total_daily:,} daily | "
                f"{rate:,.0f} rows/s"
            )

    # Final counts from DB
    with engine.connect() as conn:
        h_total = conn.execute(text("SELECT COUNT(*) FROM asset_prices_hourly")).scalar()
        d_total = conn.execute(text("SELECT COUNT(*) FROM asset_prices_daily")).scalar()
        h_assets = conn.execute(text("SELECT COUNT(DISTINCT asset) FROM asset_prices_hourly")).scalar()

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"  PostgreSQL: {h_total:,} hourly rows, {d_total:,} daily rows")
    logger.info(f"  Assets: {h_assets}")

    engine.dispose()


if __name__ == "__main__":
    main()
