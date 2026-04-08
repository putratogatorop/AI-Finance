"""Feature engineering v3 — 15-minute candles for LightGBM v4.

29 features total:
- 21 OHLCV-derived (this file, compute_ohlcv_features)
- 4 funding rate (merge_funding_features)
- 3 cross-asset (compute_cross_asset_features)
- 1 interaction (added in build script after funding + OHLCV merged)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 21 features from OHLCV data alone
OHLCV_FEATURE_COLS: list[str] = [
    # Momentum (5)
    "ret_4", "ret_16", "ret_96", "ret_672", "ret_4_lag1",
    # Trend (4)
    "price_to_sma_96", "price_to_sma_672", "ema_ratio", "macd_hist_norm",
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

FUNDING_FEATURE_COLS: list[str] = [
    "funding_rate", "funding_ma_3d", "funding_zscore", "cum_funding_3d",
]

CROSS_ASSET_FEATURE_COLS: list[str] = [
    "btc_ret_96", "btc_residual", "altcoin_dispersion",
]

INTERACTION_FEATURE_COLS: list[str] = [
    "funding_x_rsi",
]

ALL_FEATURE_COLS: list[str] = (
    OHLCV_FEATURE_COLS + FUNDING_FEATURE_COLS
    + CROSS_ASSET_FEATURE_COLS + INTERACTION_FEATURE_COLS
)


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 21 OHLCV-derived features from 15m candle data.

    Input must have: timestamp, open, high, low, close, volume, taker_buy_base.
    Returns original DataFrame with feature columns appended (NaN in warmup rows).
    """
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    taker_buy = out["taker_buy_base"].astype(float)

    # ── Momentum (5) ──
    for period in [4, 16, 96, 672]:
        out[f"ret_{period}"] = np.log(close / close.shift(period))
    out["ret_4_lag1"] = out["ret_4"].shift(1)

    # ── Trend / Mean-Reversion (4) ──
    for period in [96, 672]:
        sma = close.rolling(period).mean()
        out[f"price_to_sma_{period}"] = close / sma - 1

    ema_48 = close.ewm(span=48, adjust=False).mean()
    ema_104 = close.ewm(span=104, adjust=False).mean()
    out["ema_ratio"] = ema_48 / ema_104 - 1

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist_norm"] = (macd - macd_signal) / close

    # ── Oscillators (2) ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(56).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(56).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["rsi_norm"] = (rsi - 50) / 50

    bb_mid = close.rolling(80).mean()
    bb_std = close.rolling(80).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower)

    # ── Volatility (4) ──
    pct = close.pct_change()
    out["vol_4h"] = pct.rolling(16).std()
    out["vol_1d"] = pct.rolling(96).std()
    out["vol_ratio"] = out["vol_4h"] / out["vol_1d"].replace(0, np.nan)

    hl_log = np.log(high / low)
    out["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (hl_log ** 2).rolling(96).mean()
    )

    # ── Volume (2) ──
    vol_sma = volume.rolling(96).mean()
    out["volume_ratio"] = volume / vol_sma.replace(0, np.nan)
    out["taker_buy_ratio"] = taker_buy / volume.replace(0, np.nan)

    # ── Microstructure (2) ──
    bar_range = high - low
    out["clv"] = (2 * close - high - low) / bar_range.replace(0, np.nan)
    rolling_high = close.rolling(672).max()
    out["drawdown"] = close / rolling_high - 1

    # ── Time (2) ──
    ts = pd.to_datetime(out["timestamp"])
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    return out


def merge_funding_features(
    ohlcv_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge 8-hourly funding rates into 15m OHLCV and compute 4 features.

    Funding rates are forward-filled across 15m intervals, then:
    - funding_rate: raw rate
    - funding_ma_3d: rolling mean over 9 funding periods (9 x 8h = 3 days)
    - funding_zscore: (rate - mean_96) / std_96 where 96 periods = 32 days
    - cum_funding_3d: rolling sum over 9 periods

    The rolling windows operate on the 8h funding frequency before upsampling.
    """
    out = ohlcv_df.copy()
    ohlcv_ts = pd.to_datetime(out["timestamp"])

    # Prepare funding series indexed by timestamp
    fund = funding_df.copy()
    fund["timestamp"] = pd.to_datetime(fund["timestamp"])
    fund = fund.sort_values("timestamp").drop_duplicates("timestamp")
    fund = fund.set_index("timestamp")

    # Compute rolling features at 8h frequency BEFORE upsampling
    rate = fund["funding_rate"]
    fund["funding_ma_3d"] = rate.rolling(9, min_periods=1).mean()
    roll_mean = rate.rolling(96, min_periods=10).mean()
    roll_std = rate.rolling(96, min_periods=10).std()
    fund["funding_zscore"] = (rate - roll_mean) / roll_std.replace(0, np.nan)
    fund["cum_funding_3d"] = rate.rolling(9, min_periods=1).sum()

    # Reindex to 15m timestamps via merge_asof (forward-fill)
    out["_ts"] = ohlcv_ts
    out = out.sort_values("_ts")
    fund_reset = fund.reset_index()

    merged = pd.merge_asof(
        out, fund_reset[["timestamp", "funding_rate", "funding_ma_3d",
                         "funding_zscore", "cum_funding_3d"]],
        left_on="_ts", right_on="timestamp",
        direction="backward",
    )
    merged = merged.drop(columns=["_ts", "timestamp_y"], errors="ignore")
    if "timestamp_x" in merged.columns:
        merged = merged.rename(columns={"timestamp_x": "timestamp"})

    return merged
