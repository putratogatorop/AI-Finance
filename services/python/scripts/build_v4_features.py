"""Build v4 feature matrices from PostgreSQL and save to Parquet.

Reads: asset_prices_15m, funding_rates
Outputs: data/features/v4/{ASSET}_features.parquet (one per coin)

Each Parquet contains all 29 features + target + timestamp.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base
from src.ml.features_v3 import (
    ALL_FEATURE_COLS,
    compute_ohlcv_features,
    merge_funding_features,
    compute_cross_asset_features,
    add_interaction_features,
    build_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
OUT_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v4")

V4_ALTS = ["SOL", "DOGE", "XRP", "AVAX", "LINK"]
V4_ALL = ["BTC", "ETH"] + V4_ALTS


def load_15m_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, open, high, low, close, volume,
               quote_volume, trades, taker_buy_base, taker_buy_quote
        FROM asset_prices_15m
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"asset": asset})
    return df


def load_funding_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, funding_rate
        FROM funding_rates
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"asset": asset})
    return df


def build_one_coin(
    engine,
    asset: str,
    btc_close: pd.Series,
    all_alt_ret_96: pd.DataFrame,
) -> pd.DataFrame:
    """Build complete feature matrix for one coin."""
    logger.info(f"  {asset}: loading data...")
    ohlcv = load_15m_data(engine, asset)
    funding = load_funding_data(engine, asset)

    logger.info(f"  {asset}: computing OHLCV features ({len(ohlcv):,} rows)...")
    featured = compute_ohlcv_features(ohlcv)

    logger.info(f"  {asset}: merging funding features...")
    featured = merge_funding_features(featured, funding)

    logger.info(f"  {asset}: computing cross-asset features...")
    alt_close = pd.Series(ohlcv["close"].values, dtype=float)
    cross = compute_cross_asset_features(alt_close, btc_close, all_alt_ret_96)
    for col in cross.columns:
        featured[col] = cross[col].values

    logger.info(f"  {asset}: adding interaction features...")
    featured = add_interaction_features(featured)

    logger.info(f"  {asset}: building target...")
    close = featured["close"].astype(float)
    vol = featured["vol_1d"]
    featured["target"] = build_target(close, vol, horizon=16)

    # Keep only timestamp + features + target, drop NaN warmup
    keep_cols = ["timestamp"] + ALL_FEATURE_COLS + ["target"]
    available = [c for c in keep_cols if c in featured.columns]
    result = featured[available].dropna().reset_index(drop=True)

    return result


def main():
    engine = create_engine(DB_URL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Load BTC close for cross-asset features
    logger.info("Loading BTC data for cross-asset features...")
    btc_ohlcv = load_15m_data(engine, "BTC")
    btc_close = pd.Series(btc_ohlcv["close"].values, dtype=float)

    # Load all alt 1-day returns for dispersion
    logger.info("Loading alt returns for dispersion...")
    alt_ret_cols = {}
    for alt in V4_ALTS:
        alt_ohlcv = load_15m_data(engine, alt)
        alt_c = pd.Series(alt_ohlcv["close"].values, dtype=float)
        alt_ret_cols[alt] = np.log(alt_c / alt_c.shift(96))
    all_alt_ret_96 = pd.DataFrame(alt_ret_cols)

    # Build features for each coin
    for asset in V4_ALL:
        logger.info(f"Building features for {asset}...")
        df = build_one_coin(engine, asset, btc_close, all_alt_ret_96)

        out_path = OUT_DIR / f"{asset}_features.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  {asset}: {len(df):,} rows, {len(df.columns)} cols -> {out_path}")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"Output: {OUT_DIR}")

    # Summary
    for asset in V4_ALL:
        path = OUT_DIR / f"{asset}_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            target_dist = df["target"].value_counts(normalize=True).sort_index()
            logger.info(f"  {asset}: {len(df):,} rows | "
                        f"SHORT={target_dist.get(0, 0):.1%} "
                        f"FLAT={target_dist.get(1, 0):.1%} "
                        f"LONG={target_dist.get(2, 0):.1%}")

    engine.dispose()


if __name__ == "__main__":
    main()
