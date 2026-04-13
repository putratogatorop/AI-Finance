# scripts/backfill_agg_trades.py
"""Backfill Binance aggregated trades and compute hourly large-trade metrics.

Downloads aggTrades from Binance Vision CDN, computes per-hour:
- large_trade_ratio: fraction of trades above 95th percentile size
- large_trade_imbalance: (large_buy - large_sell) / total_large
- trade_count_ratio: hourly count / 24h SMA count

Stores in asset_prices_15m is not practical for tick data.
Instead, we compute hourly metrics and store as a CSV/Parquet
that gets merged during feature building.

Output: data/features/v5/agg_trades/{ASSET}_hourly_trades.parquet
"""

import csv
import io
import logging
import os
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/spot/monthly/aggTrades"
OUTPUT_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v5/agg_trades")
START_YEAR = 2024
START_MONTH = 1

V5_SYMBOLS = ["BTCUSDT", "LINKUSDT", "XRPUSDT", "AVAXUSDT"]


def compute_hourly_trade_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute hourly large-trade metrics from tick-level aggTrades.

    Input: DataFrame with timestamp, quantity, is_buyer_maker
    Output: DataFrame with hourly large_trade_ratio, large_trade_imbalance, trade_count_ratio
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Define "large" as > 95th percentile of quantity (expanding to avoid lookahead)
    qty = df["quantity"]
    large_threshold = qty.expanding(min_periods=100).quantile(0.95)
    df["is_large"] = qty > large_threshold

    # Hourly aggregation
    hourly = df.resample("1h").agg(
        total_trades=("quantity", "count"),
        large_count=("is_large", "sum"),
        large_buy=("is_large", lambda x: ((x) & (~df.loc[x.index, "is_buyer_maker"])).sum()),
        large_sell=("is_large", lambda x: ((x) & (df.loc[x.index, "is_buyer_maker"])).sum()),
    )

    hourly["large_trade_ratio"] = hourly["large_count"] / hourly["total_trades"].replace(0, np.nan)
    total_large = (hourly["large_buy"] + hourly["large_sell"]).replace(0, np.nan)
    hourly["large_trade_imbalance"] = (hourly["large_buy"] - hourly["large_sell"]) / total_large

    # Trade count ratio: current hour / 24h SMA
    hourly["trade_count_sma"] = hourly["total_trades"].rolling(24, min_periods=1).mean()
    hourly["trade_count_ratio"] = hourly["total_trades"] / hourly["trade_count_sma"].replace(0, np.nan)

    result = hourly[["large_trade_ratio", "large_trade_imbalance", "trade_count_ratio"]].reset_index()
    return result


def download_and_process_month(symbol: str, month: str) -> pd.DataFrame | None:
    """Download one month of aggTrades and compute hourly metrics."""
    url = f"{BASE_URL}/{symbol}/{symbol}-aggTrades-{month}.zip"
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        resp = urlopen(req, timeout=60)
        zip_data = resp.read()

        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            csv_name = zf.namelist()[0]
            with zf.open(csv_name) as f:
                # aggTrades CSV: agg_trade_id, price, quantity, first_trade_id, last_trade_id, timestamp, is_buyer_maker
                df = pd.read_csv(
                    f, header=None,
                    names=["agg_id", "price", "quantity", "first_id", "last_id", "timestamp", "is_buyer_maker"],
                    usecols=["timestamp", "quantity", "is_buyer_maker"],
                )

        # Convert timestamp (milliseconds or microseconds)
        ts_raw = df["timestamp"].iloc[0]
        if ts_raw > 1e13:
            df["timestamp"] = pd.to_datetime(df["timestamp"] / 1000, unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)

        df["quantity"] = df["quantity"].astype(float)
        df["is_buyer_maker"] = df["is_buyer_maker"].astype(bool)

        return compute_hourly_trade_metrics(df)

    except HTTPError as e:
        if e.code == 404:
            return None
        logger.error(f"HTTP {e.code} for {symbol} {month}")
        return None
    except Exception as e:
        logger.error(f"Error processing {symbol} {month}: {e}")
        return None


def generate_months(start_year, start_month):
    now = datetime.now(timezone.utc)
    months = []
    y, m = start_year, start_month
    while (y, m) <= (now.year, now.month):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m = 1
            y += 1
    return months


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    months = generate_months(START_YEAR, START_MONTH)

    logger.info("Binance Aggregated Trades Backfill")
    logger.info(f"Symbols: {V5_SYMBOLS}")
    logger.info(f"Months: {len(months)} ({months[0]} to {months[-1]})")

    for symbol in V5_SYMBOLS:
        logger.info(f"\nProcessing {symbol}...")
        all_hourly = []

        for month in months:
            logger.info(f"  {symbol} {month}...")
            hourly = download_and_process_month(symbol, month)
            if hourly is not None and len(hourly) > 0:
                all_hourly.append(hourly)
                logger.info(f"    {len(hourly)} hourly records")
            else:
                logger.info(f"    no data")
            time.sleep(0.5)

        if all_hourly:
            combined = pd.concat(all_hourly, ignore_index=True)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            out_path = OUTPUT_DIR / f"{symbol}_hourly_trades.parquet"
            combined.to_parquet(out_path, index=False)
            logger.info(f"  {symbol}: {len(combined):,} total hourly records -> {out_path}")
        else:
            logger.warning(f"  {symbol}: no data at all!")

    logger.info("\nDONE!")


if __name__ == "__main__":
    main()
