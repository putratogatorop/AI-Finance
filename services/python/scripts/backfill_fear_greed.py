# scripts/backfill_fear_greed.py
"""Backfill Fear & Greed Index from alternative.me API.

One API call gets all history. Daily values stored in fear_greed_index table.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
FNG_URL = "https://api.alternative.me/fng/?limit=0&format=json"


def main():
    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)

    logger.info("Fear & Greed Index Backfill")

    req = Request(FNG_URL, headers={"User-Agent": "Mozilla/5.0"})
    resp = urlopen(req, timeout=30)
    payload = json.loads(resp.read().decode("utf-8"))
    data = payload.get("data", [])
    logger.info(f"Fetched {len(data)} daily records")

    rows = []
    for item in data:
        ts = int(item["timestamp"])
        rows.append({
            "date": datetime.fromtimestamp(ts, tz=UTC),
            "value": int(item["value"]),
            "classification": item["value_classification"],
        })

    if rows:
        sql = text("""
            INSERT INTO fear_greed_index (date, value, classification)
            VALUES (:date, :value, :classification)
            ON CONFLICT (date) DO UPDATE SET
                value = EXCLUDED.value, classification = EXCLUDED.classification
        """)
        with engine.begin() as conn:
            conn.execute(sql, rows)

    with engine.connect() as conn:
        total = conn.execute(text("SELECT COUNT(*) FROM fear_greed_index")).scalar()
        min_d = conn.execute(text("SELECT MIN(date) FROM fear_greed_index")).scalar()
        max_d = conn.execute(text("SELECT MAX(date) FROM fear_greed_index")).scalar()

    logger.info(f"DONE! {total:,} records [{min_d} -> {max_d}]")
    engine.dispose()


if __name__ == "__main__":
    main()
