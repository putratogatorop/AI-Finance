"""Validate v4 data completeness and quality.

Checks:
1. Row counts per asset (15m candles)
2. Date range coverage
3. Gap detection (missing 15-min intervals)
4. Funding rate coverage
5. Basic data quality (no NaN, no zero prices)
"""

import logging
import sys

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
V4_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]

# 3 years of 15min candles: ~105,000 per coin. Allow 90% as passing.
MIN_ROWS_15M = 90_000
MIN_FUNDING_ROWS = 2_000  # ~3 rates/day x 1095 days = ~3285, allow margin


def main():
    engine = create_engine(DB_URL)
    issues = []

    logger.info("=" * 60)
    logger.info("V4 Data Validation")
    logger.info("=" * 60)

    # 1. 15m candle counts
    logger.info("\n--- 15m Candle Counts ---")
    with engine.connect() as conn:
        for asset in V4_ASSETS:
            result = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM asset_prices_15m WHERE asset = :asset"
            ), {"asset": asset}).fetchone()
            count, min_ts, max_ts = result
            status = "OK" if count >= MIN_ROWS_15M else "LOW"
            if status == "LOW":
                issues.append(f"{asset}: only {count:,} 15m rows (need {MIN_ROWS_15M:,})")
            logger.info(f"  {asset}: {count:,} rows [{min_ts} → {max_ts}] {status}")

    # 2. Gap detection (check for gaps > 1 hour in 15m data)
    logger.info("\n--- Gap Detection (>1h gaps) ---")
    with engine.connect() as conn:
        for asset in V4_ASSETS:
            gaps = conn.execute(text("""
                SELECT gap_start, gap_end, gap_minutes FROM (
                    SELECT
                        timestamp AS gap_start,
                        LEAD(timestamp) OVER (ORDER BY timestamp) AS gap_end,
                        EXTRACT(EPOCH FROM
                            LEAD(timestamp) OVER (ORDER BY timestamp) - timestamp
                        ) / 60 AS gap_minutes
                    FROM asset_prices_15m
                    WHERE asset = :asset
                ) sub
                WHERE gap_minutes > 60
                ORDER BY gap_minutes DESC
                LIMIT 5
            """), {"asset": asset}).fetchall()

            if gaps:
                logger.info(f"  {asset}: {len(gaps)} gaps >1h (showing top 5):")
                for g in gaps:
                    logger.info(f"    {g[0]} → {g[1]} ({g[2]:.0f} min)")
                    issues.append(f"{asset}: gap {g[2]:.0f}min at {g[0]}")
            else:
                logger.info(f"  {asset}: no gaps >1h")

    # 3. Funding rate counts
    logger.info("\n--- Funding Rate Counts ---")
    with engine.connect() as conn:
        for asset in V4_ASSETS:
            result = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM funding_rates WHERE asset = :asset"
            ), {"asset": asset}).fetchone()
            count, min_ts, max_ts = result
            status = "OK" if count >= MIN_FUNDING_ROWS else "LOW"
            if status == "LOW":
                issues.append(f"{asset}: only {count:,} funding rates (need {MIN_FUNDING_ROWS:,})")
            logger.info(f"  {asset}: {count:,} rates [{min_ts} → {max_ts}] {status}")

    # 4. Data quality
    logger.info("\n--- Data Quality ---")
    with engine.connect() as conn:
        zero_prices = conn.execute(text(
            "SELECT COUNT(*) FROM asset_prices_15m WHERE close = 0 OR open = 0"
        )).scalar()
        null_prices = conn.execute(text(
            "SELECT COUNT(*) FROM asset_prices_15m WHERE close IS NULL"
        )).scalar()
        logger.info(f"  Zero prices: {zero_prices}")
        logger.info(f"  Null prices: {null_prices}")
        if zero_prices > 0:
            issues.append(f"{zero_prices} rows with zero prices")
        if null_prices > 0:
            issues.append(f"{null_prices} rows with null prices")

    # Summary
    logger.info("\n" + "=" * 60)
    if issues:
        logger.warning(f"ISSUES FOUND ({len(issues)}):")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("ALL CHECKS PASSED")

    engine.dispose()
    return len(issues) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
