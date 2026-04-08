# scripts/backfill_contract_stats.py
"""Backfill Gate.io contract stats: OI, liquidations, long/short ratio.

Endpoint: GET /api/v4/futures/usdt/contract_stats
Returns 5-minute interval data. We aggregate to 1h before storing.

Output: contract_stats_1h table in PostgreSQL
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
GATEIO_BASE = "https://api.gateio.ws/api/v4/futures/usdt/contract_stats"

V5_CONTRACTS = {
    "BTC": "BTC_USDT",
    "LINK": "LINK_USDT",
    "XRP": "XRP_USDT",
    "AVAX": "AVAX_USDT",
}

START_DATE = datetime(2024, 1, 1, tzinfo=UTC)
PAGE_LIMIT = 1000  # Max records per request


def parse_contract_stats(asset: str, data: list[dict]) -> list[dict]:
    rows = []
    for item in data:
        rows.append({
            "asset": asset,
            "timestamp": datetime.fromtimestamp(item["time"], tz=UTC),
            "open_interest_usd": float(item.get("open_interest_usd", 0)),
            "long_liq_usd": float(item.get("long_liq_usd", 0) or 0),
            "short_liq_usd": float(item.get("short_liq_usd", 0) or 0),
            "lsr_taker": float(item.get("lsr_taker", 0) or 0),
            "lsr_account": float(item.get("lsr_account", 0) or 0),
            "top_lsr_size": float(item.get("top_lsr_size", 0) or 0),
            "top_lsr_account": float(item.get("top_lsr_account", 0) or 0),
        })
    return rows


def aggregate_5min_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 5-min contract stats to 1h bars."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    agg = df.resample("1h").agg({
        "asset": "first",
        "open_interest_usd": "last",       # Snapshot: use last value
        "long_liq_usd": "sum",             # Liquidations: sum over hour
        "short_liq_usd": "sum",
        "lsr_taker": "mean",               # Ratios: average over hour
        "lsr_account": "mean",
        "top_lsr_size": "mean",
        "top_lsr_account": "mean",
    }).dropna(subset=["asset"])

    return agg.reset_index()


def fetch_stats_page(contract: str, from_ts: int, to_ts: int) -> list[dict]:
    url = f"{GATEIO_BASE}?contract={contract}&from={from_ts}&to={to_ts}&limit={PAGE_LIMIT}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    resp = urlopen(req, timeout=15)
    return json.loads(resp.read().decode("utf-8"))


def upsert_stats_batch(engine, rows: list[dict]) -> int:
    if not rows:
        return 0
    sql = text("""
        INSERT INTO contract_stats_1h
            (asset, timestamp, open_interest_usd, long_liq_usd, short_liq_usd,
             lsr_taker, lsr_account, top_lsr_size, top_lsr_account)
        VALUES
            (:asset, :timestamp, :open_interest_usd, :long_liq_usd, :short_liq_usd,
             :lsr_taker, :lsr_account, :top_lsr_size, :top_lsr_account)
        ON CONFLICT (asset, timestamp) DO UPDATE SET
            open_interest_usd = EXCLUDED.open_interest_usd,
            long_liq_usd = EXCLUDED.long_liq_usd,
            short_liq_usd = EXCLUDED.short_liq_usd,
            lsr_taker = EXCLUDED.lsr_taker,
            lsr_account = EXCLUDED.lsr_account,
            top_lsr_size = EXCLUDED.top_lsr_size,
            top_lsr_account = EXCLUDED.top_lsr_account
    """)
    with engine.begin() as conn:
        conn.execute(sql, rows)
    return len(rows)


def backfill_asset(engine, asset: str, contract: str) -> int:
    now = datetime.now(UTC)
    cursor = START_DATE
    total = 0

    while cursor < now:
        # 1000 records at 5min = ~3.5 days per request
        window_end = min(cursor + timedelta(days=3), now)
        from_ts = int(cursor.timestamp())
        to_ts = int(window_end.timestamp())

        try:
            data = fetch_stats_page(contract, from_ts, to_ts)
            if data:
                parsed = parse_contract_stats(asset, data)
                df = pd.DataFrame(parsed)
                hourly = aggregate_5min_to_1h(df)
                rows = hourly.to_dict("records")
                upsert_stats_batch(engine, rows)
                total += len(rows)
                logger.info(f"  {asset}: +{len(rows)} hours ({cursor.date()} -> {window_end.date()})")
            else:
                logger.info(f"  {asset}: no data ({cursor.date()} -> {window_end.date()})")
        except HTTPError as e:
            logger.error(f"  {asset}: HTTP {e.code} ({cursor.date()})")
        except Exception as e:
            logger.error(f"  {asset}: error: {e}")

        cursor = window_end
        time.sleep(0.3)

    return total


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info("Gate.io Contract Stats Backfill (OI + Liquidations + LSR)")
    logger.info(f"Assets: {list(V5_CONTRACTS.keys())}")
    logger.info(f"Start: {START_DATE.date()}")

    for asset, contract in V5_CONTRACTS.items():
        logger.info(f"\nBackfilling {asset} ({contract})...")
        count = backfill_asset(engine, asset, contract)
        logger.info(f"  {asset}: {count} total hourly records")

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM contract_stats_1h")).scalar()
        assets = conn.execute(text("SELECT COUNT(DISTINCT asset) FROM contract_stats_1h")).scalar()
    logger.info(f"\nDONE! {total:,} records, {assets} assets")
    engine.dispose()


if __name__ == "__main__":
    main()
