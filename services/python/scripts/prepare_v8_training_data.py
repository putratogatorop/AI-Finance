"""Build training dataset for v8 BALANCED LightGBM classifiers.

For each historical trade in the v8 95-day backtest, computes 21 pre-entry
features with STRICT no-look-ahead. Output is a single parquet file ready
to upload to Google Drive for training in Colab.

CRITICAL — every feature in this script is computed using ONLY data with
timestamp <= entry_time. The v2 ML failure (docs/research-journal/
v2ml-audit-2026-04-24.md) was caused by sloppy feature timing. Don't
repeat it.

Output schema:
    trade_id, detector, asset, entry_time, direction, atr14_at_entry,
    feature_1, ..., feature_21,
    label (1 if pnl_pct > 0 else 0),
    pnl_pct, exit_reason, bars_held

Usage from services/python/:
    .venv/bin/python scripts/prepare_v8_training_data.py
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- CONFIG ----------------------------------------------------------------
SNAPSHOT_DATE = "2026-04-27"
TRADES_CSV = (
    Path(__file__).resolve().parents[1]
    / "results"
    / "backtest_v8_balanced_recent_v1_e4df8e27_20260430T072443Z"
    / "trades.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "training"
OUTPUT_PARQUET = OUTPUT_DIR / f"v8_trades_with_features_{SNAPSHOT_DATE}.parquet"

SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"
CANDLES_PARQUET = SNAPSHOTS_DIR / f"candles_15m_{SNAPSHOT_DATE}.parquet"
UNIVERSE_PARQUET = SNAPSHOTS_DIR / f"universe_{SNAPSHOT_DATE}.parquet"

# Detector params (must match live scanner code)
ATR_PERIOD = 14
EMA_TREND_SPAN = 50
RSI_PERIOD = 14
EARLY_TREND_FRESH_BARS = 10 * 96


# --- INDICATOR HELPERS (same code as live scanners; deterministic) -------

def _atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    return atr


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "signal": sig, "histogram": macd - sig})


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


# --- FEATURE COMPUTATION (per asset, returns aligned arrays) -------------

def compute_asset_features(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """For an asset's 15m candles, compute every per-bar feature aligned to
    the 15m grid. Each feature value at index i uses ONLY candles 0..i.

    Returns dict of feature_name → np.array of length n.
    """
    ts = df["timestamp"].to_numpy()
    h = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    o = df["open"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    v = df["volume"].to_numpy(dtype=float)
    n = len(c)

    ts_idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    s_close = pd.Series(c, index=ts_idx)
    s_high = pd.Series(h, index=ts_idx)
    s_low = pd.Series(lo, index=ts_idx)
    s_open = pd.Series(o, index=ts_idx)
    s_vol = pd.Series(v, index=ts_idx)

    atr14 = _atr14(h, lo, c)

    # ATR14 percentile rank vs trailing 90 days (8640 bars). NaN before bar 500.
    s_atr = pd.Series(atr14)
    atr_pct_rank_90d = s_atr.rolling(8640, min_periods=500).rank(pct=True).to_numpy()

    # 24h volume vs 30-day mean (z-score)
    vol_24h = s_vol.rolling(96, min_periods=96).sum()
    vol_24h_mean_30d = vol_24h.rolling(2880, min_periods=720).mean()
    vol_24h_std_30d = vol_24h.rolling(2880, min_periods=720).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z_24h = ((vol_24h - vol_24h_mean_30d) / vol_24h_std_30d).replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy(dtype=float)

    # 7-day return: c[t] / c[t - 7*96] - 1
    coin_7d_return = np.full(n, np.nan)
    if n > 7 * 96:
        coin_7d_return[7 * 96:] = c[7 * 96:] / c[: n - 7 * 96] - 1.0

    # 30-day return: c[t] / c[t - 30*96] - 1
    coin_30d_return = np.full(n, np.nan)
    if n > 30 * 96:
        coin_30d_return[30 * 96:] = c[30 * 96:] / c[: n - 30 * 96] - 1.0

    # 50-bar high / low → close-to-extreme distances normalized by ATR
    s_h50 = s_high.rolling(50, min_periods=50).max()
    s_l50 = s_low.rolling(50, min_periods=50).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        close_to_high50_atr = ((s_h50 - s_close) / s_atr.values).replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy(dtype=float)
        close_to_low50_atr = ((s_close - s_l50) / s_atr.values).replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy(dtype=float)

    # 4H bar features at each 15m index (using PRIOR-completed 4H bar — no look-ahead)
    h4_close = s_close.resample("4h").last().dropna()
    h4_open = s_open.resample("4h").first().reindex(h4_close.index)
    h4_high = s_high.resample("4h").max().reindex(h4_close.index)
    h4_low = s_low.resample("4h").min().reindex(h4_close.index)

    # Shift forward by 4h so 15m bar at 12:00 inherits the 08:00 4H bar value
    def _shift4h(s: pd.Series) -> np.ndarray:
        s2 = s.copy()
        s2.index = s2.index + pd.Timedelta("4h")
        return s2.reindex(ts_idx, method="ffill").to_numpy(dtype=float)

    h4_close_prev = _shift4h(h4_close)
    h4_open_prev = _shift4h(h4_open)
    h4_high_prev = _shift4h(h4_high)
    h4_low_prev = _shift4h(h4_low)

    with np.errstate(divide="ignore", invalid="ignore"):
        bar4h_range = h4_high_prev - h4_low_prev
        bar4h_close_pos = (h4_close_prev - h4_low_prev) / bar4h_range
        bar4h_body_pct = np.abs(h4_close_prev - h4_open_prev) / bar4h_range
        bar4h_upper_wick = (
            h4_high_prev - np.maximum(h4_open_prev, h4_close_prev)
        ) / bar4h_range
    # Sanitize divide-by-zero
    bar4h_close_pos = np.where(bar4h_range > 0, bar4h_close_pos, np.nan)
    bar4h_body_pct = np.where(bar4h_range > 0, bar4h_body_pct, np.nan)
    bar4h_upper_wick = np.where(bar4h_range > 0, bar4h_upper_wick, np.nan)

    # 4H MACD histogram (the cross-up / cross-down signal strength)
    if len(h4_close) >= 30:
        h4_macd_df = _macd(h4_close, 12, 26, 9)
        h4_macd_hist_arr = _shift4h(h4_macd_df["histogram"])
        h4_macd_macd_arr = _shift4h(h4_macd_df["macd"])
    else:
        h4_macd_hist_arr = np.full(n, np.nan)
        h4_macd_macd_arr = np.full(n, np.nan)

    # 4H RSI (for rsi_recovery features — relevant cross-section feature even for other detectors)
    if len(h4_close) >= RSI_PERIOD + EMA_TREND_SPAN:
        h4_rsi_arr = _shift4h(_rsi(h4_close, RSI_PERIOD))
        h4_ema50_arr = _shift4h(_ema(h4_close, EMA_TREND_SPAN))
        with np.errstate(divide="ignore", invalid="ignore"):
            h4_close_vs_ema50_pct = (h4_close_prev - h4_ema50_arr) / h4_ema50_arr
        h4_close_vs_ema50_pct = np.where(
            np.isfinite(h4_ema50_arr) & (h4_ema50_arr > 0), h4_close_vs_ema50_pct, np.nan
        )
    else:
        h4_rsi_arr = np.full(n, np.nan)
        h4_close_vs_ema50_pct = np.full(n, np.nan)

    # Daily MACD histogram (the regime gate signal)
    d_close = s_close.resample("1D").last().dropna()
    if len(d_close) >= 30:
        d_macd_df = _macd(d_close, 12, 26, 9)
        d_hist = d_macd_df["histogram"]
        # Shift forward by 1 day so 15m bar at day D inherits day D-1's value (no look-ahead)
        d_hist_shifted = d_hist.copy()
        d_hist_shifted.index = d_hist_shifted.index + pd.Timedelta("1D")
        daily_macd_hist = d_hist_shifted.reindex(ts_idx, method="ffill").to_numpy(dtype=float)

        # Days since daily MACD bull/bear flip
        daily_bull = ((d_macd_df["histogram"] > 0) & (d_macd_df["macd"] > d_macd_df["signal"]))
        daily_bear = ((d_macd_df["histogram"] < 0) & (d_macd_df["macd"] < d_macd_df["signal"]))
        daily_bull_arr = daily_bull.to_numpy(dtype=bool)
        daily_bear_arr = daily_bear.to_numpy(dtype=bool)
        # Days since the most recent bull or bear flip (whichever is currently true)
        days_since_bull_flip = np.full(len(daily_bull), np.nan, dtype=float)
        days_since_bear_flip = np.full(len(daily_bear), np.nan, dtype=float)
        last_bull_flip = -1
        last_bear_flip = -1
        for i in range(len(daily_bull_arr)):
            if i > 0 and daily_bull_arr[i] and not daily_bull_arr[i - 1]:
                last_bull_flip = i
            if i > 0 and daily_bear_arr[i] and not daily_bear_arr[i - 1]:
                last_bear_flip = i
            if last_bull_flip >= 0:
                days_since_bull_flip[i] = i - last_bull_flip
            if last_bear_flip >= 0:
                days_since_bear_flip[i] = i - last_bear_flip
        # Align to 15m grid via shift+ffill (same +1day shift to avoid look-ahead at flip day)
        s_dsbu = pd.Series(days_since_bull_flip, index=daily_bull.index)
        s_dsbe = pd.Series(days_since_bear_flip, index=daily_bear.index)
        s_dsbu.index = s_dsbu.index + pd.Timedelta("1D")
        s_dsbe.index = s_dsbe.index + pd.Timedelta("1D")
        days_since_bull_flip_arr = s_dsbu.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        days_since_bear_flip_arr = s_dsbe.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
    else:
        daily_macd_hist = np.full(n, np.nan)
        days_since_bull_flip_arr = np.full(n, np.nan)
        days_since_bear_flip_arr = np.full(n, np.nan)

    return {
        "atr14_pct_rank_90d": atr_pct_rank_90d,
        "vol_z_24h": vol_z_24h,
        "coin_7d_return": coin_7d_return,
        "coin_30d_return": coin_30d_return,
        "close_to_high50_atr": close_to_high50_atr,
        "close_to_low50_atr": close_to_low50_atr,
        "bar4h_close_pos_in_range": bar4h_close_pos,
        "bar4h_body_pct": bar4h_body_pct,
        "bar4h_upper_wick_pct": bar4h_upper_wick,
        "h4_macd_hist": h4_macd_hist_arr,
        "h4_macd_macd": h4_macd_macd_arr,
        "h4_rsi": h4_rsi_arr,
        "h4_close_vs_ema50_pct": h4_close_vs_ema50_pct,
        "daily_macd_hist": daily_macd_hist,
        "days_since_bull_flip": days_since_bull_flip_arr,
        "days_since_bear_flip": days_since_bear_flip_arr,
    }, ts_idx


# --- BTC-RELATIVE FEATURES (computed once globally) ----------------------

def compute_btc_features(btc_df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Returns (btc_above_4h_ema50_15m, btc_24h_return_15m, btc_realized_vol_z_15m,
    btc_score_15m), all indexed at 15m timestamps. Each is shifted so that the 15m
    bar at time t inherits values from the prior 4h period (no look-ahead).
    """
    btc = btc_df.sort_values("timestamp").reset_index(drop=True)
    ts = btc["timestamp"].to_numpy()
    c = btc["close"].to_numpy(dtype=float)
    n = len(c)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    s_close = pd.Series(c, index=ts_idx)

    h4 = s_close.resample("4h").last().dropna()
    h4_ema50 = _ema(h4, EMA_TREND_SPAN)
    btc_above_ema = (h4 > h4_ema50).astype(float)
    # Shift forward 4h so 15m@12:00 inherits 08:00 4h value
    btc_above_ema_shifted = btc_above_ema.copy()
    btc_above_ema_shifted.index = btc_above_ema_shifted.index + pd.Timedelta("4h")

    # 24h return: close[t] / close[t - 96 bars] - 1 — past only
    btc_24h_return = np.full(n, np.nan)
    if n > 96:
        btc_24h_return[96:] = c[96:] / c[:-96] - 1.0
    btc_24h_return_s = pd.Series(btc_24h_return, index=ts_idx)

    # Realized vol: std of last 96 returns, z-scored vs 30-day window
    log_ret = np.log(c[1:] / c[:-1])
    log_ret = np.concatenate([[np.nan], log_ret])
    s_logret = pd.Series(log_ret, index=ts_idx)
    rv_24h = s_logret.rolling(96, min_periods=96).std()
    rv_mean = rv_24h.rolling(2880, min_periods=720).mean()
    rv_std = rv_24h.rolling(2880, min_periods=720).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        btc_realized_vol_z = ((rv_24h - rv_mean) / rv_std).replace(
            [np.inf, -np.inf], np.nan
        )

    # btc_score (existing): tanh(4h MACD hist / 0.005×close)
    h4_macd_df = _macd(h4)
    btc_score_h4 = np.tanh(
        h4_macd_df["histogram"] / (0.005 * h4.replace(0, np.nan))
    )
    btc_score_shifted = btc_score_h4.copy()
    btc_score_shifted.index = btc_score_shifted.index + pd.Timedelta("4h")

    return btc_above_ema_shifted, btc_24h_return_s, btc_realized_vol_z, btc_score_shifted


# --- BREADTH (computed across all assets at each 4h boundary) ------------

def compute_breadth_at_h4(candles: pd.DataFrame) -> pd.DataFrame:
    """For each 4h timestamp, returns DataFrame indexed by 4h boundary with:
        breadth_up: fraction of universe with positive 4h return
        breadth_down: fraction with negative 4h return
    Both are shifted forward by 4h so a 15m bar at 12:00 inherits the 08:00 4h value.
    """
    pivot = candles.pivot_table(
        index="timestamp", columns="asset", values="close", aggfunc="last"
    )
    h4 = pivot.resample("4h").last()
    h4_ret = h4.pct_change()
    breadth_up = (h4_ret > 0).sum(axis=1) / h4_ret.notna().sum(axis=1)
    breadth_down = (h4_ret < 0).sum(axis=1) / h4_ret.notna().sum(axis=1)
    breadth_up = breadth_up.replace([np.inf, -np.inf], np.nan)
    breadth_down = breadth_down.replace([np.inf, -np.inf], np.nan)
    # Shift forward 4h so 15m@12:00 inherits 08:00 4h value
    breadth_up.index = breadth_up.index + pd.Timedelta("4h")
    breadth_down.index = breadth_down.index + pd.Timedelta("4h")
    return pd.DataFrame({"breadth_up": breadth_up, "breadth_down": breadth_down})


# --- MAIN ------------------------------------------------------------------

def main() -> None:
    started = time.monotonic()
    print(f"[{time.monotonic() - started:6.1f}s] loading snapshot {SNAPSHOT_DATE}...")
    candles = pd.read_parquet(CANDLES_PARQUET)
    universe = pd.read_parquet(UNIVERSE_PARQUET)
    trades = pd.read_csv(TRADES_CSV)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"  candles: {len(candles):,}  universe: {len(universe)}  trades: {len(trades):,}")

    # Filter to the same eligible universe used in the v8 backtest (top-100 by 24h volume)
    universe = universe.copy()
    universe["asset_key"] = universe["symbol"].str.replace("_", "", regex=False)
    eligible_uni = universe[
        (universe["is_leveraged"] == False)  # noqa: E712
        & (universe["in_delisting"] == False)  # noqa: E712
        & (universe["quote_volume_24h"] >= 500_000.0)
    ].nlargest(100, "quote_volume_24h")
    eligible_assets = set(eligible_uni["asset_key"].astype(str))
    candles = candles[candles["asset"].isin(eligible_assets)].copy()
    print(f"[{time.monotonic() - started:6.1f}s] eligible: {len(eligible_assets)} assets, {len(candles):,} candles")

    # Compute global breadth
    breadth_df = compute_breadth_at_h4(candles)
    print(f"[{time.monotonic() - started:6.1f}s] breadth pre-computed ({len(breadth_df)} 4h boundaries)")

    # Compute BTC features
    btc = candles[candles["asset"] == "BTCUSDT"].sort_values("timestamp")
    if btc.empty:
        # BTC not in eligible? grab from full snapshot
        full_candles = pd.read_parquet(CANDLES_PARQUET)
        btc = full_candles[full_candles["asset"] == "BTCUSDT"].sort_values("timestamp")
    btc_above_ema, btc_24h_ret, btc_rvz, btc_score = compute_btc_features(btc)
    print(f"[{time.monotonic() - started:6.1f}s] BTC global features computed")

    # Compute per-asset features
    candles_by_asset = {a: g.sort_values("timestamp").reset_index(drop=True)
                        for a, g in candles.groupby("asset")}
    asset_feats: dict[str, dict[str, np.ndarray]] = {}
    asset_ts_idx: dict[str, pd.DatetimeIndex] = {}
    for k, (asset, df) in enumerate(candles_by_asset.items()):
        feats, ts_idx = compute_asset_features(df)
        asset_feats[asset] = feats
        asset_ts_idx[asset] = ts_idx
        if (k + 1) % 25 == 0 or k == len(candles_by_asset) - 1:
            print(f"[{time.monotonic() - started:6.1f}s] features for {k+1}/{len(candles_by_asset)} assets")

    # For each trade, look up features at entry_time
    print(f"[{time.monotonic() - started:6.1f}s] joining features to trades...")

    # Index of trades by (entry_time, detector) for breadth-of-detector feature
    same_15m_per_detector = trades.groupby(
        ["entry_time", "detector"]
    ).size().reset_index(name="signals_same_15m_same_detector")
    trades = trades.merge(same_15m_per_detector, on=["entry_time", "detector"])

    # Hour of day & day of week (cyclical encodings)
    hour = trades["entry_time"].dt.hour
    dow = trades["entry_time"].dt.dayofweek
    trades["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    trades["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    trades["dow_sin"] = np.sin(2 * math.pi * dow / 7)
    trades["dow_cos"] = np.cos(2 * math.pi * dow / 7)

    # Per-trade per-asset features
    feature_cols = [
        "atr14_pct_rank_90d",
        "vol_z_24h",
        "coin_7d_return",
        "coin_30d_return",
        "close_to_high50_atr",
        "close_to_low50_atr",
        "bar4h_close_pos_in_range",
        "bar4h_body_pct",
        "bar4h_upper_wick_pct",
        "h4_macd_hist",
        "h4_macd_macd",
        "h4_rsi",
        "h4_close_vs_ema50_pct",
        "daily_macd_hist",
        "days_since_bull_flip",
        "days_since_bear_flip",
    ]
    for col in feature_cols:
        trades[col] = np.nan

    for col in feature_cols:
        for asset, feats in asset_feats.items():
            ts_idx = asset_ts_idx[asset]
            asset_trades_mask = trades["asset"] == asset
            if not asset_trades_mask.any():
                continue
            t_times = trades.loc[asset_trades_mask, "entry_time"]
            # For each trade entry, find the index in ts_idx
            # Use searchsorted for vectorized lookup
            arr = ts_idx.to_numpy()
            t_arr = t_times.to_numpy()
            # ts_idx is sorted; find exact match (entries should align to 15m bars)
            idx_pos = np.searchsorted(arr, t_arr)
            valid = (idx_pos < len(arr)) & (arr[np.minimum(idx_pos, len(arr) - 1)] == t_arr)
            feat_arr = feats[col]
            vals = np.where(valid, feat_arr[np.minimum(idx_pos, len(feat_arr) - 1)], np.nan)
            trades.loc[asset_trades_mask, col] = vals

    # BTC features (at entry_time, looked up from shifted-by-4h series, ffill)
    def _lookup_at(s: pd.Series, t: pd.DatetimeIndex):
        return s.reindex(t, method="ffill").to_numpy(dtype=float)

    trades["btc_above_4h_ema50"] = _lookup_at(btc_above_ema, pd.DatetimeIndex(trades["entry_time"]))
    trades["btc_24h_return"] = _lookup_at(btc_24h_ret, pd.DatetimeIndex(trades["entry_time"]))
    trades["btc_realized_vol_z"] = _lookup_at(btc_rvz, pd.DatetimeIndex(trades["entry_time"]))
    trades["btc_score"] = _lookup_at(btc_score, pd.DatetimeIndex(trades["entry_time"]))

    # Breadth features
    trades["breadth_up"] = _lookup_at(breadth_df["breadth_up"], pd.DatetimeIndex(trades["entry_time"]))
    trades["breadth_down"] = _lookup_at(breadth_df["breadth_down"], pd.DatetimeIndex(trades["entry_time"]))

    # Outcome label
    trades["label"] = (trades["pnl_pct"] > 0).astype(int)

    # Final feature list (must match what the Colab notebook expects)
    final_features = (
        feature_cols
        + [
            "btc_above_4h_ema50",
            "btc_24h_return",
            "btc_realized_vol_z",
            "btc_score",
            "breadth_up",
            "breadth_down",
            "signals_same_15m_same_detector",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
        ]
    )

    keep_cols = (
        ["detector", "asset", "entry_time", "direction", "atr14_at_entry", "label",
         "pnl_pct", "exit_reason", "bars_held"]
        + final_features
    )
    out = trades[keep_cols].copy()

    # Sanity: missing rates per feature
    print(f"\n[{time.monotonic() - started:6.1f}s] missing rates per feature:")
    for col in final_features:
        miss_pct = out[col].isna().mean() * 100
        print(f"  {col:30s}  {miss_pct:>5.1f}% missing")

    # Drop trades where any feature is missing (early-history trades)
    n_before = len(out)
    out = out.dropna(subset=final_features).reset_index(drop=True)
    n_after = len(out)
    print(f"\n  trades before NaN-drop: {n_before:,}")
    print(f"  trades after  NaN-drop: {n_after:,} (kept {n_after/n_before*100:.1f}%)")

    # Per-detector breakdown
    print(f"\n  per-detector trade counts after NaN-drop:")
    print(out.groupby("detector").size().to_string())

    # Class balance
    print(f"\n  label distribution:")
    print(out.groupby(["detector", "label"]).size().unstack().to_string())

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\n=== DONE — wall_time {time.monotonic() - started:.1f}s ===")
    print(f"output: {OUTPUT_PARQUET}")
    print(f"size:   {OUTPUT_PARQUET.stat().st_size / 1024:.1f} KB")
    print(f"shape:  {out.shape}")
    print(f"\nUpload this file to Google Drive at:")
    print(f"  /My Drive/ai-finance/training/{OUTPUT_PARQUET.name}")


if __name__ == "__main__":
    main()
