# scripts/backfill_volume_breakouts.py
"""Backfill volume_breakouts table from historical 15m CSV data.

Scans 198 coins for volume spikes (>= 2x avg) on 15m, 1h, and 4h timeframes.
Records breakout context and forward returns at 6 horizons (12h to 1 month).
"""

import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

sys.path.insert(0, ".")
from scripts.backtest_momentum_scanner import compute_atr, load_coin
from src.db.models import Base, VolumeBreakout

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
VOL_RATIO_MIN = 2.0
BATCH_SIZE = 1000

# Forward return horizons in bars per timeframe
HORIZONS = {
    "15m": {"12h": 48, "24h": 96, "48h": 192, "1w": 672, "2w": 1344, "1m": 2880},
    "1h":  {"12h": 12, "24h": 24, "48h": 48,  "1w": 168, "2w": 336,  "1m": 720},
    "4h":  {"12h": 3,  "24h": 6,  "48h": 12,  "1w": 42,  "2w": 84,   "1m": 180},
}

# MFE/MAE horizons (only up to 1w -- beyond that is too noisy)
MFE_HORIZONS = {"12h", "24h", "48h", "1w"}

# Cooldown per timeframe (bars) -- prevent double-counting same move
COOLDOWN = {"15m": 8, "1h": 4, "4h": 2}


def _safe_float(v):
    """Convert to Python float, returning None for NaN/Inf."""
    if v is None:
        return None
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Compute RSI as array. Returns NaN for first `period` bars."""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 1:
        return rsi

    deltas = np.diff(close)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            rsi[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def resample_to_timeframe(df_15m: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample 15m bars to 1h or 4h. Returns DataFrame with same columns."""
    if tf == "15m":
        return df_15m.copy()

    rule = "1h" if tf == "1h" else "4h"
    df = df_15m.set_index("open_time").resample(rule).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "quote_volume": "sum",
        "trades": "sum",
    }).dropna(subset=["close"]).reset_index()
    return df


def detect_and_extract(
    df: pd.DataFrame,
    timeframe: str,
    btc_close: np.ndarray,
    btc_volume: np.ndarray,
    btc_times: np.ndarray,
) -> list[dict]:
    """Detect volume spikes and extract all features + forward returns."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_prices = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    quote_vol = df["quote_volume"].values.astype(float)
    trade_counts = df["trades"].values.astype(float)
    times = df["open_time"].values  # numpy datetime64 with UTC

    n = len(close)
    if n < 50:
        return []

    # Precompute indicators
    vol_ma = pd.Series(volume).rolling(20).mean().values
    atr = compute_atr(high, low, close, period=14)
    ema_50 = pd.Series(close).ewm(span=50).mean().values
    rsi = compute_rsi(close, period=14)
    returns = np.zeros(n)
    returns[1:] = (close[1:] - close[:-1]) / close[:-1]
    volatility = pd.Series(returns).rolling(20).std().values

    horizons = HORIZONS[timeframe]
    cooldown = COOLDOWN[timeframe]
    records = []
    last_signal_bar = -cooldown - 1

    for i in range(21, n):
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            continue

        ratio = volume[i] / vol_ma[i]
        if ratio < VOL_RATIO_MIN:
            continue

        # Cooldown
        if i - last_signal_bar < cooldown:
            continue
        last_signal_bar = i

        # Direction
        direction = 1 if close[i] >= close[i - 1] else -1

        # Bar shape
        bar_range = high[i] - low[i]
        if bar_range == 0:
            bar_range_pct = 0.0
            upper_wick_pct = 0.0
            lower_wick_pct = 0.0
            body_pct = 0.0
        else:
            bar_range_pct = bar_range / close[i]
            body = abs(close[i] - open_prices[i])
            if close[i] >= open_prices[i]:  # Green candle
                upper_wick_pct = (high[i] - close[i]) / bar_range
                lower_wick_pct = (open_prices[i] - low[i]) / bar_range
            else:  # Red candle
                upper_wick_pct = (high[i] - open_prices[i]) / bar_range
                lower_wick_pct = (close[i] - low[i]) / bar_range
            body_pct = body / bar_range

        # Price context
        price_change_1bar = float(returns[i])
        price_change_2bar = (
            float((close[i] - close[i - 2]) / close[i - 2])
            if i >= 2 and close[i - 2] > 0
            else None
        )

        # Distance from 20-bar high/low
        if i >= 20:
            high_20 = np.max(high[i - 20:i])
            low_20 = np.min(low[i - 20:i])
            dist_from_20_high = (close[i] - high_20) / high_20 if high_20 > 0 else 0.0
            dist_from_20_low = (close[i] - low_20) / low_20 if low_20 > 0 else 0.0
        else:
            dist_from_20_high = None
            dist_from_20_low = None

        price_vs_ema = (
            float(close[i] / ema_50[i] - 1)
            if not np.isnan(ema_50[i]) and ema_50[i] > 0
            else None
        )

        # BTC context
        sig_time = times[i]
        btc_idx = np.searchsorted(btc_times, sig_time)
        btc_idx = min(btc_idx, len(btc_close) - 1)
        btc_p = float(btc_close[btc_idx])

        btc_vol_ma = (
            np.mean(btc_volume[max(0, btc_idx - 20):btc_idx])
            if btc_idx >= 20
            else None
        )
        btc_vr = (
            float(btc_volume[btc_idx] / btc_vol_ma)
            if btc_vol_ma and btc_vol_ma > 0
            else None
        )

        btc_r24 = None
        if btc_idx >= 24 and btc_close[btc_idx - 24] > 0:
            btc_r24 = float(
                (btc_close[btc_idx] - btc_close[btc_idx - 24]) / btc_close[btc_idx - 24]
            )
        btc_r96 = None
        if btc_idx >= 96 and btc_close[btc_idx - 96] > 0:
            btc_r96 = float(
                (btc_close[btc_idx] - btc_close[btc_idx - 96]) / btc_close[btc_idx - 96]
            )

        # Forward returns
        fwd = {}
        for label, bars in horizons.items():
            j = i + bars
            if j < n and close[i] > 0:
                ret = (close[j] - close[i]) / close[i]
                if direction == -1:
                    ret = -ret
                fwd[f"fwd_ret_{label}"] = float(ret)
            else:
                fwd[f"fwd_ret_{label}"] = None

        # MFE / MAE
        for label, bars in horizons.items():
            if label not in MFE_HORIZONS:
                continue
            j_end = min(i + bars + 1, n)
            if j_end <= i + 1 or close[i] == 0:
                fwd[f"fwd_max_gain_{label}"] = None
                fwd[f"fwd_max_loss_{label}"] = None
                continue

            fwd_high = np.max(high[i + 1:j_end])
            fwd_low = np.min(low[i + 1:j_end])

            if direction == 1:
                fwd[f"fwd_max_gain_{label}"] = float((fwd_high - close[i]) / close[i])
                fwd[f"fwd_max_loss_{label}"] = float((close[i] - fwd_low) / close[i])
            else:
                fwd[f"fwd_max_gain_{label}"] = float((close[i] - fwd_low) / close[i])
                fwd[f"fwd_max_loss_{label}"] = float((close[i] - fwd_high) / close[i])

        # Convert numpy datetime64 to Python datetime with UTC timezone
        sig_ts = pd.Timestamp(times[i])
        if sig_ts.tzinfo is None:
            sig_dt = sig_ts.to_pydatetime().replace(tzinfo=timezone.utc)
        else:
            sig_dt = sig_ts.to_pydatetime()

        record = {
            "symbol": None,  # Filled by caller
            "timeframe": timeframe,
            "signal_time": sig_dt,
            "open": float(open_prices[i]),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "volume": float(volume[i]),
            "quote_volume": float(quote_vol[i]),
            "trades": int(trade_counts[i]) if not np.isnan(trade_counts[i]) else None,
            "vol_ratio": float(ratio),
            "vol_avg_20": float(vol_ma[i]),
            "price_change_1bar": _safe_float(price_change_1bar),
            "price_change_2bar": _safe_float(price_change_2bar),
            "atr_14": _safe_float(atr[i]),
            "bar_range_pct": _safe_float(bar_range_pct),
            "upper_wick_pct": _safe_float(upper_wick_pct),
            "lower_wick_pct": _safe_float(lower_wick_pct),
            "body_pct": _safe_float(body_pct),
            "direction": direction,
            "dist_from_20_high": _safe_float(dist_from_20_high),
            "dist_from_20_low": _safe_float(dist_from_20_low),
            "price_vs_ema_50": _safe_float(price_vs_ema),
            "rsi_14": _safe_float(rsi[i]),
            "volatility_20": _safe_float(volatility[i]),
            "btc_price": _safe_float(btc_p),
            "btc_ret_24bar": _safe_float(btc_r24),
            "btc_ret_96bar": _safe_float(btc_r96),
            "btc_vol_ratio": _safe_float(btc_vr),
        }
        # Add forward return / MFE / MAE fields (sanitize NaN)
        for k, v in fwd.items():
            record[k] = _safe_float(v)

        records.append(record)

    return records


def main():
    start_t = time.time()
    logger.info("Volume Breakout Research Backfill")
    logger.info(f"Data: {RAW_DIR}")
    logger.info(f"Vol ratio min: {VOL_RATIO_MIN}x")

    engine = create_engine(DB_URL)
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Clear existing data for clean backfill
    deleted = session.execute(text("DELETE FROM volume_breakouts")).rowcount
    session.commit()
    logger.info(f"Cleared {deleted} existing rows")

    # Load BTC
    btc_dir = RAW_DIR / "BTCUSDT"
    btc_df = load_coin(btc_dir)
    if btc_df is None:
        logger.error("Cannot load BTC data!")
        return
    btc_close = btc_df["close"].values.astype(float)
    btc_volume = btc_df["volume"].values.astype(float)
    btc_times = btc_df["open_time"].values
    logger.info(f"BTC loaded: {len(btc_df)} bars")

    # Process all coins
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    logger.info(f"Processing {len(symbol_dirs)} coins across 3 timeframes...")

    total_inserted = 0
    coins_processed = 0
    batch = []

    for sym_dir in symbol_dirs:
        symbol = sym_dir.name
        df_15m = load_coin(sym_dir)
        if df_15m is None or len(df_15m) < 50:
            continue

        # Ensure quote_volume exists
        if "quote_volume" not in df_15m.columns:
            df_15m["quote_volume"] = df_15m["volume"] * df_15m["close"]
        if "trades" not in df_15m.columns:
            df_15m["trades"] = 0

        coins_processed += 1

        for tf in ["15m", "1h", "4h"]:
            df_tf = resample_to_timeframe(df_15m, tf)
            if len(df_tf) < 50:
                continue

            records = detect_and_extract(
                df=df_tf,
                timeframe=tf,
                btc_close=btc_close,
                btc_volume=btc_volume,
                btc_times=btc_times,
            )

            for rec in records:
                rec["symbol"] = symbol
                batch.append(rec)

            if len(batch) >= BATCH_SIZE:
                session.bulk_insert_mappings(VolumeBreakout, batch)
                session.commit()
                total_inserted += len(batch)
                batch = []

        if coins_processed % 20 == 0:
            logger.info(f"  {coins_processed} coins done, {total_inserted} rows inserted...")

    # Flush remaining
    if batch:
        session.bulk_insert_mappings(VolumeBreakout, batch)
        session.commit()
        total_inserted += len(batch)

    session.close()
    elapsed = time.time() - start_t
    logger.info(f"\nBackfill complete in {elapsed:.0f}s")
    logger.info(f"Coins processed: {coins_processed}")
    logger.info(f"Total rows inserted: {total_inserted}")

    # Print summary stats
    session = Session()
    for tf in ["15m", "1h", "4h"]:
        count = session.execute(
            text("SELECT COUNT(*) FROM volume_breakouts WHERE timeframe = :tf"),
            {"tf": tf},
        ).scalar()
        avg_vr = session.execute(
            text("SELECT AVG(vol_ratio) FROM volume_breakouts WHERE timeframe = :tf"),
            {"tf": tf},
        ).scalar()
        logger.info(f"  {tf}: {count} events, avg vol_ratio={avg_vr:.1f}x")

    # Forward return distribution sanity check
    for horizon in ["12h", "24h", "48h", "1w"]:
        col = f"fwd_ret_{horizon}"
        stats = session.execute(
            text(f"""
                SELECT
                    COUNT(*) FILTER (WHERE {col} IS NOT NULL) as n,
                    ROUND(AVG({col})::numeric, 4) as mean,
                    ROUND((percentile_cont(0.25) WITHIN GROUP (ORDER BY {col}))::numeric, 4)
                        as p25,
                    ROUND((percentile_cont(0.50) WITHIN GROUP (ORDER BY {col}))::numeric, 4)
                        as median,
                    ROUND((percentile_cont(0.75) WITHIN GROUP (ORDER BY {col}))::numeric, 4)
                        as p75
                FROM volume_breakouts
                WHERE timeframe = '15m'
            """)
        ).fetchone()
        if stats and stats[0] > 0:
            logger.info(
                f"  15m fwd_ret_{horizon}: n={stats[0]}, "
                f"mean={stats[1]}, p25={stats[2]}, "
                f"median={stats[3]}, p75={stats[4]}"
            )
    session.close()


if __name__ == "__main__":
    main()
