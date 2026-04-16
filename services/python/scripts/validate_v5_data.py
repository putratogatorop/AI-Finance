# scripts/validate_v5_data.py
"""Validate v5 data completeness."""

import logging
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
TRADES_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v5/agg_trades")
V5_ASSETS = ["BTC", "LINK", "XRP", "AVAX"]


def main():
    engine = create_engine(DB_URL)
    issues = []

    logger.info("=" * 60)
    logger.info("V5 Data Validation")
    logger.info("=" * 60)

    # 1. Contract stats
    logger.info("\n--- Contract Stats (1h) ---")
    with engine.connect() as conn:
        for asset in V5_ASSETS:
            r = conn.execute(text(
                "SELECT COUNT(*), MIN(timestamp), MAX(timestamp) "
                "FROM contract_stats_1h WHERE asset = :a"
            ), {"a": asset}).fetchone()
            count, min_ts, max_ts = r
            status = "OK" if count >= 10000 else "LOW"
            if status == "LOW":
                issues.append(f"{asset}: only {count:,} contract_stats rows")
            logger.info(f"  {asset}: {count:,} rows [{min_ts} -> {max_ts}] {status}")

    # 2. Fear & Greed
    logger.info("\n--- Fear & Greed Index ---")
    with engine.connect() as conn:
        r = conn.execute(text(
            "SELECT COUNT(*), MIN(date), MAX(date) FROM fear_greed_index"
        )).fetchone()
        count, min_d, max_d = r
        status = "OK" if count >= 1000 else "LOW"
        if status == "LOW":
            issues.append(f"Fear & Greed: only {count} records")
        logger.info(f"  {count:,} records [{min_d} -> {max_d}] {status}")

    # 3. Aggregated trades
    logger.info("\n--- Aggregated Trade Metrics ---")
    for symbol in ["BTCUSDT", "LINKUSDT", "XRPUSDT", "AVAXUSDT"]:
        path = TRADES_DIR / f"{symbol}_hourly_trades.parquet"
        if path.exists():
            import pandas as pd
            df = pd.read_parquet(path)
            logger.info(f"  {symbol}: {len(df):,} hourly records")
        else:
            issues.append(f"{symbol}: no aggTrades file")
            logger.info(f"  {symbol}: MISSING")

    # 4. Existing data (sanity check)
    logger.info("\n--- Existing 15m OHLCV + Funding ---")
    with engine.connect() as conn:
        for asset in V5_ASSETS:
            ohlcv = conn.execute(text(
                "SELECT COUNT(*) FROM asset_prices_15m WHERE asset = :a"
            ), {"a": asset}).scalar()
            funding = conn.execute(text(
                "SELECT COUNT(*) FROM funding_rates WHERE asset = :a"
            ), {"a": asset}).scalar()
            logger.info(f"  {asset}: {ohlcv:,} OHLCV, {funding:,} funding rates")

    logger.info("\n" + "=" * 60)
    if issues:
        logger.warning(f"ISSUES ({len(issues)}):")
        for i in issues:
            logger.warning(f"  - {i}")
    else:
        logger.info("ALL CHECKS PASSED")

    engine.dispose()
    return len(issues) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
