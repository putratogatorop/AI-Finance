"""Backfill funding rate history from Gate.io API.

Gate.io provides funding rates every 8 hours for perpetual contracts.
Endpoint: GET https://api.gateio.ws/api/v4/futures/usdt/funding_rate
No API key required.
"""

import json
import logging
import sys
import time
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
GATEIO_BASE = "https://api.gateio.ws/api/v4/futures/usdt/funding_rate"

# Gate.io contract names
V4_CONTRACTS = {
    "BTC": "BTC_USDT",
    "ETH": "ETH_USDT",
    "SOL": "SOL_USDT",
    "DOGE": "DOGE_USDT",
    "XRP": "XRP_USDT",
    "AVAX": "AVAX_USDT",
    "LINK": "LINK_USDT",
}

START_DATE = datetime(2023, 4, 1, tzinfo=UTC)
PAGE_LIMIT = 1000


def parse_gateio_funding(asset: str, data: list[dict]) -> list[dict]:
    rows = []
    for item in data:
        ts = int(item["t"])
        rate = float(item["r"])
        rows.append({
            "asset": asset,
            "timestamp": datetime.fromtimestamp(ts, tz=UTC),
            "funding_rate": rate,
            "source": "gateio",
        })
    return rows


def fetch_funding_page(contract: str, from_ts: int, to_ts: int) -> list[dict]:
    url = f"{GATEIO_BASE}?contract={contract}&from={from_ts}&to={to_ts}&limit={PAGE_LIMIT}"
    req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
    resp = urlopen(req, timeout=15)
    return json.loads(resp.read().decode("utf-8"))


def upsert_funding_batch(engine, rows: list[dict]) -> int:
    if not rows:
        return 0

    sql = text("""
        INSERT INTO funding_rates (asset, timestamp, funding_rate, source)
        VALUES (:asset, :timestamp, :funding_rate, :source)
        ON CONFLICT (asset, timestamp, source) DO UPDATE SET
            funding_rate = EXCLUDED.funding_rate
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    return len(rows)


def backfill_asset(engine, asset: str, contract: str) -> int:
    now = datetime.now(UTC)
    cursor = START_DATE
    total = 0

    while cursor < now:
        window_end = min(cursor + timedelta(days=90), now)
        from_ts = int(cursor.timestamp())
        to_ts = int(window_end.timestamp())

        try:
            data = fetch_funding_page(contract, from_ts, to_ts)
            rows = parse_gateio_funding(asset, data)
            if rows:
                upsert_funding_batch(engine, rows)
                total += len(rows)
                logger.info(f"  {asset}: +{len(rows)} rates ({cursor.date()} → {window_end.date()})")
            else:
                logger.info(f"  {asset}: no data ({cursor.date()} → {window_end.date()})")
        except HTTPError as e:
            logger.error(f"  {asset}: HTTP {e.code} ({cursor.date()} → {window_end.date()})")
        except Exception as e:
            logger.error(f"  {asset}: error: {e}")

        cursor = window_end
        time.sleep(0.3)  # Rate limit

    return total


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info("Gate.io Funding Rate Backfill")
    logger.info(f"Assets: {list(V4_CONTRACTS.keys())}")
    logger.info(f"Start: {START_DATE.date()}")

    grand_total = 0
    for asset, contract in V4_CONTRACTS.items():
        logger.info(f"Backfilling {asset} ({contract})...")
        count = backfill_asset(engine, asset, contract)
        grand_total += count
        logger.info(f"  {asset}: {count} total rates")

    with engine.connect() as conn:
        db_total = conn.execute(text("SELECT COUNT(*) FROM funding_rates")).scalar()
        db_assets = conn.execute(text("SELECT COUNT(DISTINCT asset) FROM funding_rates")).scalar()

    logger.info("=" * 60)
    logger.info(f"DONE! PostgreSQL: {db_total:,} funding rates, {db_assets} assets")
    engine.dispose()


if __name__ == "__main__":
    main()
