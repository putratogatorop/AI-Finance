"""BTC 4-hour backfill: Gate.io → CSV + PostgreSQL.

Fetches BTC/USDT 4h candles from March 2024 to present via Gate.io
(Binance is blocked in Indonesia). Saves to both raw CSV and PostgreSQL.
Also aggregates 4h → daily candles.
"""

import csv
import logging
import os
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path

import ccxt
from sqlalchemy import create_engine, func, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, ".")
from src.db.models import AssetPriceDaily, AssetPriceHourly, Base

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/btc_4h")
START = datetime(2024, 3, 1, tzinfo=UTC)
BATCH_SIZE = 500
ASSET = "BTC"
SYMBOL = "BTC/USDT"
TIMEFRAME = "4h"


def save_csv(candles: list[dict], path: Path) -> None:
    """Append candles to a monthly CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["timestamp", "open", "high", "low", "close", "volume"]
        )
        if not exists:
            writer.writeheader()
        writer.writerows(candles)


def main():
    engine = create_engine(DB_URL)
    Session = sessionmaker(bind=engine)
    session = Session()

    ex = ccxt.gateio({"enableRateLimit": True, "timeout": 15000})
    end = datetime.now(UTC)

    # Resume from last timestamp in DB
    latest = session.query(func.max(AssetPriceHourly.timestamp)).filter(
        AssetPriceHourly.asset == ASSET
    ).scalar()
    cursor = latest + timedelta(hours=4) if latest else START

    total_candles = int((end - START).total_seconds() / (4 * 3600))
    fetched_so_far = session.query(func.count(AssetPriceHourly.id)).filter(
        AssetPriceHourly.asset == ASSET
    ).scalar() or 0

    logger.info(f"BTC 4h backfill: {START.date()} → {end.date()}")
    logger.info(f"~{total_candles} candles needed, {fetched_so_far} already in DB")
    logger.info(f"Resuming from {cursor.isoformat()}")
    logger.info("=" * 60)

    total_inserted = 0
    batch_num = 0

    while cursor < end:
        batch_num += 1
        since_ms = int(cursor.timestamp() * 1000)

        try:
            raw = ex.fetch_ohlcv(SYMBOL, TIMEFRAME, since=since_ms, limit=BATCH_SIZE)
            if not raw:
                logger.info("No more data from exchange")
                break

            candles = []
            for row in raw:
                ts_ms, o, h, l, c, v = row
                dt = datetime.fromtimestamp(ts_ms / 1000, tz=UTC)
                candles.append({
                    "timestamp": dt.isoformat(),
                    "open": o, "high": h, "low": l, "close": c, "volume": v,
                })

            # Save to CSV (grouped by month)
            by_month: dict[str, list[dict]] = {}
            for candle in candles:
                month_key = candle["timestamp"][:7]  # "2024-03"
                by_month.setdefault(month_key, []).append(candle)

            for month_key, month_candles in by_month.items():
                csv_path = RAW_DIR / f"{month_key}.csv"
                save_csv(month_candles, csv_path)

            # Save to PostgreSQL
            inserted = 0
            for candle in candles:
                dt = datetime.fromisoformat(candle["timestamp"])
                exists = session.query(AssetPriceHourly.id).filter_by(
                    asset=ASSET, timestamp=dt
                ).first()
                if exists:
                    continue
                session.add(AssetPriceHourly(
                    asset=ASSET, timestamp=dt,
                    open=candle["open"], high=candle["high"],
                    low=candle["low"], close=candle["close"],
                    volume=candle["volume"],
                ))
                inserted += 1

            session.commit()
            total_inserted += inserted

            last_dt = datetime.fromisoformat(candles[-1]["timestamp"])
            cursor = last_dt + timedelta(hours=4)

            pct = min(100, int((fetched_so_far + total_inserted) / total_candles * 100))
            logger.info(
                f"  Batch {batch_num}: +{inserted} candles "
                f"(→ {last_dt.strftime('%Y-%m-%d %H:%M')}) [{pct}%]"
            )

            time.sleep(0.5)

        except Exception as e:
            logger.error(f"  Batch {batch_num} error: {e}")
            time.sleep(3)
            cursor += timedelta(days=1)

    # Phase 2: Aggregate 4h → daily
    logger.info("=" * 60)
    logger.info("Aggregating 4h → daily candles...")

    dates = session.query(
        func.date(AssetPriceHourly.timestamp)
    ).filter(AssetPriceHourly.asset == ASSET).distinct().all()

    daily_inserted = 0
    for (date_val,) in dates:
        date_str = str(date_val)
        day_start = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC)
        day_end = day_start + timedelta(days=1)

        hourly = session.query(AssetPriceHourly).filter(
            AssetPriceHourly.asset == ASSET,
            AssetPriceHourly.timestamp >= day_start,
            AssetPriceHourly.timestamp < day_end,
        ).order_by(AssetPriceHourly.timestamp).all()

        if not hourly:
            continue

        existing = session.query(AssetPriceDaily).filter_by(
            asset=ASSET, date=day_start
        ).first()

        vals = dict(
            open=hourly[0].open,
            high=max(r.high for r in hourly),
            low=min(r.low for r in hourly),
            close=hourly[-1].close,
            volume=sum(r.volume for r in hourly),
        )

        if existing:
            for k, v in vals.items():
                setattr(existing, k, v)
        else:
            session.add(AssetPriceDaily(asset=ASSET, date=day_start, **vals))
            daily_inserted += 1

    session.commit()
    logger.info(f"  {daily_inserted} daily rows created")

    # Summary
    h_count = session.query(func.count(AssetPriceHourly.id)).filter(
        AssetPriceHourly.asset == ASSET
    ).scalar()
    d_count = session.query(func.count(AssetPriceDaily.id)).filter(
        AssetPriceDaily.asset == ASSET
    ).scalar()
    h_min = session.query(func.min(AssetPriceHourly.timestamp)).filter(
        AssetPriceHourly.asset == ASSET
    ).scalar()
    h_max = session.query(func.max(AssetPriceHourly.timestamp)).filter(
        AssetPriceHourly.asset == ASSET
    ).scalar()

    csv_files = list(RAW_DIR.glob("*.csv"))

    logger.info("=" * 60)
    logger.info(f"DONE!")
    logger.info(f"  PostgreSQL: {h_count} 4h candles, {d_count} daily candles")
    logger.info(f"  Date range: {h_min} → {h_max}")
    logger.info(f"  CSV files:  {len(csv_files)} monthly files in {RAW_DIR}")
    logger.info(f"  New rows:   {total_inserted} 4h + {daily_inserted} daily")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    main()
