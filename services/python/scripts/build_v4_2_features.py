# scripts/build_v4_2_features.py
"""Build v4.2 features: aggregate 15min → 1h, compute 1h-calibrated features.

Key change from v4.0/v4.1: 1-hour bars reduce noise 4x while preserving
the funding rate signal that operates on 8h timescale.

Outputs: data/features/v4.2/{ASSET}_features.parquet
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.ml.features_v3 import (
    FUNDING_FEATURE_COLS, CROSS_ASSET_FEATURE_COLS,
    INTERACTION_FEATURE_COLS, merge_funding_features,
    compute_cross_asset_features, add_interaction_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
OUT_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v4.2")

V4_2_ASSETS = ["BTC", "LINK", "XRP", "AVAX"]  # Focused coins with signal
V4_2_ALTS = ["LINK", "XRP", "AVAX"]  # For cross-asset dispersion

# 1h-calibrated feature columns (21 OHLCV + 2 time)
FEATURE_COLS_1H = [
    # Momentum (5) — recalibrated for 1h bars
    "ret_1", "ret_4", "ret_24", "ret_168", "ret_1_lag1",
    # Trend (4)
    "price_to_sma_24", "price_to_sma_168", "ema_ratio", "macd_hist_norm",
    # Oscillators (2)
    "rsi_norm", "bb_pct",
    # Volatility (4)
    "vol_4h", "vol_1d", "vol_ratio", "parkinson_vol",
    # Volume (2)
    "volume_ratio", "taker_buy_ratio",
    # Microstructure (2)
    "clv", "drawdown",
    # Time (2)
    "hour_sin", "hour_cos",
]

ALL_FEATURE_COLS_1H = (
    FEATURE_COLS_1H + FUNDING_FEATURE_COLS
    + CROSS_ASSET_FEATURE_COLS + INTERACTION_FEATURE_COLS
)


def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15min OHLCV to 1h bars."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    resampled = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "taker_buy_base": "sum",
    }).dropna()

    resampled = resampled.reset_index()
    return resampled


def compute_1h_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 21 OHLCV features calibrated for 1h bars.

    Period mapping (1h bars):
    - 1 bar = 1h, 4 bars = 4h, 24 bars = 1 day, 168 bars = 7 days
    """
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    taker_buy = out["taker_buy_base"].astype(float)

    # ── Momentum (5) ──
    out["ret_1"] = np.log(close / close.shift(1))      # 1h
    out["ret_4"] = np.log(close / close.shift(4))      # 4h
    out["ret_24"] = np.log(close / close.shift(24))    # 1d
    out["ret_168"] = np.log(close / close.shift(168))  # 7d
    out["ret_1_lag1"] = out["ret_1"].shift(1)

    # ── Trend (4) ──
    for period, name in [(24, "24"), (168, "168")]:
        sma = close.rolling(period).mean()
        out[f"price_to_sma_{name}"] = close / sma - 1

    ema_12 = close.ewm(span=12, adjust=False).mean()  # 12h
    ema_26 = close.ewm(span=26, adjust=False).mean()  # 26h
    out["ema_ratio"] = ema_12 / ema_26 - 1

    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist_norm"] = (macd - macd_signal) / close

    # ── Oscillators (2) ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()  # RSI(14) on 1h = 14h
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["rsi_norm"] = (rsi - 50) / 50

    bb_mid = close.rolling(20).mean()  # BB(20) on 1h = 20h
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower)

    # ── Volatility (4) ──
    pct = close.pct_change()
    out["vol_4h"] = pct.rolling(4).std()   # 4h
    out["vol_1d"] = pct.rolling(24).std()  # 1 day
    out["vol_ratio"] = out["vol_4h"] / out["vol_1d"].replace(0, np.nan)

    hl_log = np.log(high / low)
    out["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (hl_log ** 2).rolling(24).mean()
    )

    # ── Volume (2) ──
    vol_sma = volume.rolling(24).mean()
    out["volume_ratio"] = volume / vol_sma.replace(0, np.nan)
    out["taker_buy_ratio"] = taker_buy / volume.replace(0, np.nan)

    # ── Microstructure (2) ──
    bar_range = high - low
    out["clv"] = (2 * close - high - low) / bar_range.replace(0, np.nan)
    rolling_high = close.rolling(168).max()  # 7-day high
    out["drawdown"] = close / rolling_high - 1

    # ── Time (2) ──
    ts = pd.to_datetime(out["timestamp"])
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    return out


def build_target_1h(close: pd.Series, realized_vol: pd.Series, horizon: int = 4) -> pd.Series:
    """Build 3-class target for 1h bars. Horizon=4 bars = 4 hours."""
    forward_ret = close.shift(-horizon) / close - 1
    vol_safe = realized_vol.replace(0, np.nan)
    risk_adj = forward_ret / vol_safe

    target = pd.Series(np.nan, index=close.index)
    q20 = risk_adj.expanding(min_periods=50).quantile(0.20)
    q80 = risk_adj.expanding(min_periods=50).quantile(0.80)

    target[risk_adj <= q20] = 0  # SHORT
    target[risk_adj >= q80] = 2  # LONG
    target[(risk_adj > q20) & (risk_adj < q80)] = 1  # FLAT
    target.iloc[-horizon:] = np.nan

    return target


def load_15m_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, open, high, low, close, volume, taker_buy_base
        FROM asset_prices_15m
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"asset": asset})


def load_funding_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, funding_rate
        FROM funding_rates
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"asset": asset})


def main():
    engine = create_engine(DB_URL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    logger.info("V4.2 Feature Builder (1h bars, focused coins)")
    logger.info(f"Assets: {V4_2_ASSETS}")

    # Load and resample all assets to 1h
    hourly_data = {}
    for asset in V4_2_ASSETS:
        logger.info(f"  {asset}: loading 15m and resampling to 1h...")
        raw = load_15m_data(engine, asset)
        hourly_data[asset] = resample_to_1h(raw)
        logger.info(f"  {asset}: {len(raw):,} 15m rows → {len(hourly_data[asset]):,} 1h rows")

    # BTC close for cross-asset features
    btc_close = pd.Series(hourly_data["BTC"]["close"].values, dtype=float)

    # Alt 1-day returns for dispersion
    alt_ret_cols = {}
    for alt in V4_2_ALTS:
        alt_c = pd.Series(hourly_data[alt]["close"].values, dtype=float)
        alt_ret_cols[alt] = np.log(alt_c / alt_c.shift(24))
    all_alt_ret_24 = pd.DataFrame(alt_ret_cols)

    # Build features for each coin
    for asset in V4_2_ASSETS:
        logger.info(f"\n  {asset}: computing features...")
        ohlcv_1h = hourly_data[asset]
        funding = load_funding_data(engine, asset)

        # OHLCV features
        featured = compute_1h_features(ohlcv_1h)

        # Funding features
        featured = merge_funding_features(featured, funding)

        # Cross-asset features
        alt_close = pd.Series(ohlcv_1h["close"].values, dtype=float)
        cross = compute_cross_asset_features(alt_close, btc_close, all_alt_ret_24, window=168)
        for col in cross.columns:
            featured[col] = cross[col].values

        # Interaction
        featured = add_interaction_features(featured)

        # Target (4-bar horizon on 1h = 4 hours)
        close = featured["close"].astype(float)
        vol = featured["vol_1d"]
        featured["target"] = build_target_1h(close, vol, horizon=4)

        # Save
        keep_cols = ["timestamp"] + ALL_FEATURE_COLS_1H + ["target"]
        available = [c for c in keep_cols if c in featured.columns]
        result = featured[available].dropna().reset_index(drop=True)

        out_path = OUT_DIR / f"{asset}_features.parquet"
        result.to_parquet(out_path, index=False)

        target_dist = result["target"].value_counts(normalize=True).sort_index()
        logger.info(f"  {asset}: {len(result):,} rows, {len(result.columns)} cols")
        logger.info(f"    SHORT={target_dist.get(0, 0):.1%} "
                    f"FLAT={target_dist.get(1, 0):.1%} "
                    f"LONG={target_dist.get(2, 0):.1%}")

    elapsed = time.time() - start
    logger.info(f"\nDONE in {elapsed:.0f}s! Output: {OUT_DIR}")
    engine.dispose()


if __name__ == "__main__":
    main()
