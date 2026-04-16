"""Upsert 15m CSVs from data/raw/15m/ into asset_prices_15m table."""

import csv
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
BATCH_SIZE = 5000

V4_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "DOGEUSDT", "XRPUSDT", "AVAXUSDT", "LINKUSDT"]


def parse_symbol(symbol_usdt: str) -> str:
    return symbol_usdt.replace("USDT", "")


def upsert_15m_batch(engine, rows: list[dict]) -> int:
    if not rows:
        return 0

    sql = text("""
        INSERT INTO asset_prices_15m
            (asset, timestamp, open, high, low, close, volume,
             quote_volume, trades, taker_buy_base, taker_buy_quote)
        VALUES
            (:asset, :timestamp, :open, :high, :low, :close, :volume,
             :quote_volume, :trades, :taker_buy_base, :taker_buy_quote)
        ON CONFLICT (asset, timestamp) DO UPDATE SET
            open = EXCLUDED.open, high = EXCLUDED.high,
            low = EXCLUDED.low, close = EXCLUDED.close,
            volume = EXCLUDED.volume, quote_volume = EXCLUDED.quote_volume,
            trades = EXCLUDED.trades, taker_buy_base = EXCLUDED.taker_buy_base,
            taker_buy_quote = EXCLUDED.taker_buy_quote
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def load_symbol_csvs(symbol_dir: Path) -> list[dict]:
    symbol_usdt = symbol_dir.name
    asset = parse_symbol(symbol_usdt)
    csv_files = sorted(symbol_dir.glob("*.csv"))
    rows = []

    for csv_file in csv_files:
        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    ts_raw = int(row["open_time"])
                    # Binance Vision switched to microseconds (16 digits) in 2025
                    ts_s = ts_raw / 1_000_000 if ts_raw > 1e13 else ts_raw / 1_000
                    dt = datetime.fromtimestamp(ts_s, tz=UTC)
                    rows.append({
                        "asset": asset,
                        "timestamp": dt,
                        "open": float(row["open"]),
                        "high": float(row["high"]),
                        "low": float(row["low"]),
                        "close": float(row["close"]),
                        "volume": float(row["volume"]),
                        "quote_volume": float(row.get("quote_volume", 0)),
                        "trades": int(row.get("trades", 0)),
                        "taker_buy_base": float(row.get("taker_buy_base", 0)),
                        "taker_buy_quote": float(row.get("taker_buy_quote", 0)),
                    })
                except (ValueError, KeyError, OSError):
                    continue

    return rows


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info(f"Upserting 15m candles from {RAW_DIR}")

    total_rows = 0
    start = time.time()

    for symbol_usdt in V4_SYMBOLS:
        sym_dir = RAW_DIR / symbol_usdt
        if not sym_dir.exists():
            logger.warning(f"  {symbol_usdt}: directory not found, skipping")
            continue

        rows = load_symbol_csvs(sym_dir)
        asset = parse_symbol(symbol_usdt)

        # Batch upsert
        upserted = 0
        for i in range(0, len(rows), BATCH_SIZE):
            batch = rows[i:i + BATCH_SIZE]
            upserted += upsert_15m_batch(engine, batch)

        total_rows += upserted
        logger.info(f"  {asset}: {upserted:,} rows upserted")

    # Final DB counts
    with engine.connect() as conn:
        db_total = conn.execute(text("SELECT COUNT(*) FROM asset_prices_15m")).scalar()
        db_assets = conn.execute(text("SELECT COUNT(DISTINCT asset) FROM asset_prices_15m")).scalar()
        min_ts = conn.execute(text("SELECT MIN(timestamp) FROM asset_prices_15m")).scalar()
        max_ts = conn.execute(text("SELECT MAX(timestamp) FROM asset_prices_15m")).scalar()

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"  PostgreSQL: {db_total:,} rows, {db_assets} assets")
    logger.info(f"  Range: {min_ts} → {max_ts}")

    engine.dispose()


if __name__ == "__main__":
    main()
