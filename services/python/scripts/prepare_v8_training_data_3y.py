"""Compute v8 BALANCED trades + features for the 3-year snapshot.

Combines backtest_v8_balanced_recent_v1.py (detector + E2 simulation) with
prepare_v8_training_data.py (27-feature pre-entry computation) into a
single pass over the 2026-04-01 snapshot (~19M candles, ~3 years).

Output: data/training/v8_trades_with_features_2026-04-01.parquet

Same methodology guards as the 95-day version:
- Strict no-look-ahead: every feature uses only candles with timestamp <= entry_time
- Universe-time-corrected: top-100 by 24h volume from snapshot, no leveraged
- E2 exits (matches deployed v8): 2× ATR SL / 6× ATR TP / 1344-bar timeout

Usage from services/python/:
    .venv/bin/python scripts/prepare_v8_training_data_3y.py
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
SNAPSHOT_DATE = "2026-04-01"
SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"
CANDLES_PARQUET = SNAPSHOTS_DIR / f"candles_15m_{SNAPSHOT_DATE}.parquet"
UNIVERSE_PARQUET = SNAPSHOTS_DIR / f"universe_{SNAPSHOT_DATE}.parquet"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "training"
OUTPUT_PARQUET = OUTPUT_DIR / f"v8_trades_with_features_{SNAPSHOT_DATE}.parquet"

TOP_N_COINS = 100
MIN_QUOTE_VOLUME_24H = 500_000.0

# Detector params (must match live)
ATR_PERIOD = 14
EMA_TREND_SPAN = 50
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
COOLDOWN_BARS = 96
EARLY_TREND_FRESH_BARS = 10 * 96

# Exit params (E2)
E2_ATR_SL_MULT = 2.0
E2_ATR_TP_MULT = 6.0
TIMEOUT_BARS = 1344
FEE_PCT = 0.0006
WORST_FILL_BUFFER = 0.005


# --- INDICATORS ------------------------------------------------------------

def _atr14(high, low, close):
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


def _apply_cooldown(idxs, cooldown_bars):
    if not idxs:
        return []
    kept = [idxs[0]]
    for idx in idxs[1:]:
        if idx - kept[-1] >= cooldown_bars:
            kept.append(idx)
    return kept


# --- SIMULATE E2 EXIT ------------------------------------------------------

def simulate_e2_exit(high, low, close, entry_idx, atr14_at_entry, direction):
    n = len(close)
    if entry_idx >= n - 1:
        return None
    ep = float(close[entry_idx])
    if not np.isfinite(ep) or ep <= 0:
        return None
    if not np.isfinite(atr14_at_entry) or atr14_at_entry <= 0:
        return None
    if direction == "long":
        sl = max(ep - E2_ATR_SL_MULT * atr14_at_entry, 0.0)
        tp = ep + E2_ATR_TP_MULT * atr14_at_entry
    else:
        sl = ep + E2_ATR_SL_MULT * atr14_at_entry
        tp = max(ep - E2_ATR_TP_MULT * atr14_at_entry, 0.0)
    exit_idx = exit_price = exit_reason = None
    end = min(entry_idx + 1 + TIMEOUT_BARS, n)
    for j in range(entry_idx + 1, end):
        hi = float(high[j]); lo = float(low[j])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        if direction == "long":
            if lo <= sl:
                exit_idx = j
                exit_price = max(sl - WORST_FILL_BUFFER * atr14_at_entry, 0.0)
                exit_reason = "stop_loss"; break
            if hi >= tp:
                exit_idx = j; exit_price = tp; exit_reason = "take_profit"; break
        else:
            if hi >= sl:
                exit_idx = j
                exit_price = sl + WORST_FILL_BUFFER * atr14_at_entry
                exit_reason = "stop_loss"; break
            if lo <= tp:
                exit_idx = j; exit_price = tp; exit_reason = "take_profit"; break
    if exit_idx is None:
        exit_idx = end - 1
        exit_price = float(close[exit_idx])
        if not np.isfinite(exit_price):
            return None
        exit_reason = "timeout"
    pnl_gross = (exit_price - ep) / ep if direction == "long" else (ep - exit_price) / ep
    return {
        "entry_idx": int(entry_idx), "exit_idx": int(exit_idx),
        "entry_price": float(ep), "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "pnl_pct": float(pnl_gross - 2 * FEE_PCT),
        "bars_held": int(exit_idx - entry_idx),
        "atr14_at_entry": float(atr14_at_entry),
    }


# --- COMPUTE PER-ASSET FEATURES (returns dict of feature_name → array) ---

def compute_asset_features(df: pd.DataFrame):
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

    # Per-bar features
    s_atr = pd.Series(atr14)
    atr_pct_rank_90d = s_atr.rolling(8640, min_periods=500).rank(pct=True).to_numpy()

    vol_24h = s_vol.rolling(96, min_periods=96).sum()
    vol_24h_mean_30d = vol_24h.rolling(2880, min_periods=720).mean()
    vol_24h_std_30d = vol_24h.rolling(2880, min_periods=720).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_z_24h = ((vol_24h - vol_24h_mean_30d) / vol_24h_std_30d).replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy(dtype=float)

    coin_7d_return = np.full(n, np.nan)
    if n > 7 * 96:
        coin_7d_return[7 * 96:] = c[7 * 96:] / c[: n - 7 * 96] - 1.0
    coin_30d_return = np.full(n, np.nan)
    if n > 30 * 96:
        coin_30d_return[30 * 96:] = c[30 * 96:] / c[: n - 30 * 96] - 1.0

    s_h50 = s_high.rolling(50, min_periods=50).max()
    s_l50 = s_low.rolling(50, min_periods=50).min()
    with np.errstate(divide="ignore", invalid="ignore"):
        close_to_high50_atr = ((s_h50 - s_close) / s_atr.values).replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy(dtype=float)
        close_to_low50_atr = ((s_close - s_l50) / s_atr.values).replace(
            [np.inf, -np.inf], np.nan
        ).to_numpy(dtype=float)

    # 4H bar-shape features (using PRIOR-completed 4H bar — no look-ahead)
    h4_close = s_close.resample("4h").last().dropna()
    h4_open = s_open.resample("4h").first().reindex(h4_close.index)
    h4_high = s_high.resample("4h").max().reindex(h4_close.index)
    h4_low = s_low.resample("4h").min().reindex(h4_close.index)

    def _shift4h(s: pd.Series) -> np.ndarray:
        s2 = s.copy()
        s2.index = s2.index + pd.Timedelta("4h")
        return s2.reindex(ts_idx, method="ffill").to_numpy(dtype=float)

    h4_close_prev = _shift4h(h4_close)
    h4_open_prev = _shift4h(h4_open)
    h4_high_prev = _shift4h(h4_high)
    h4_low_prev = _shift4h(h4_low)
    bar4h_range = h4_high_prev - h4_low_prev
    with np.errstate(divide="ignore", invalid="ignore"):
        bar4h_close_pos = np.where(
            bar4h_range > 0, (h4_close_prev - h4_low_prev) / bar4h_range, np.nan
        )
        bar4h_body_pct = np.where(
            bar4h_range > 0, np.abs(h4_close_prev - h4_open_prev) / bar4h_range, np.nan
        )
        bar4h_upper_wick = np.where(
            bar4h_range > 0,
            (h4_high_prev - np.maximum(h4_open_prev, h4_close_prev)) / bar4h_range,
            np.nan,
        )

    # 4H MACD (signal strength)
    if len(h4_close) >= 30:
        h4_macd_df = _macd(h4_close, 12, 26, 9)
        h4_macd_hist_arr = _shift4h(h4_macd_df["histogram"])
        h4_macd_macd_arr = _shift4h(h4_macd_df["macd"])
        h4_above = (h4_macd_df["macd"] > h4_macd_df["signal"]).astype(bool)
        h4_below_prev = ~h4_above.shift(1).fillna(False)
        h4_below_prev2 = ~h4_above.shift(2).fillna(False)
        h4_cross_up = h4_above & h4_below_prev & h4_below_prev2
        h4_above_prev = h4_above.shift(1).fillna(False)
        h4_above_prev2 = h4_above.shift(2).fillna(False)
        h4_cross_down = (~h4_above) & h4_above_prev & h4_above_prev2
        cu_full = h4_cross_up.reindex(ts_idx, method="ffill").fillna(False)
        h4_cross_up_15m = (cu_full & ~cu_full.shift(1).fillna(False)).to_numpy(dtype=bool)
        cd_full = h4_cross_down.reindex(ts_idx, method="ffill").fillna(False)
        h4_cross_down_15m = (cd_full & ~cd_full.shift(1).fillna(False)).to_numpy(dtype=bool)
    else:
        h4_macd_hist_arr = np.full(n, np.nan)
        h4_macd_macd_arr = np.full(n, np.nan)
        h4_cross_up_15m = np.zeros(n, dtype=bool)
        h4_cross_down_15m = np.zeros(n, dtype=bool)

    # 4H RSI + EMA50 (rsi_recovery + h4_close_vs_ema50_pct feature)
    if len(h4_close) >= RSI_PERIOD + EMA_TREND_SPAN:
        h4_rsi_series = _rsi(h4_close, RSI_PERIOD)
        h4_ema50 = _ema(h4_close, EMA_TREND_SPAN)
        h4_rsi_arr = _shift4h(h4_rsi_series)
        h4_ema50_arr = _shift4h(h4_ema50)
        with np.errstate(divide="ignore", invalid="ignore"):
            h4_close_vs_ema50_pct = np.where(
                np.isfinite(h4_ema50_arr) & (h4_ema50_arr > 0),
                (h4_close_prev - h4_ema50_arr) / h4_ema50_arr, np.nan
            )
        rsi_cross_h4 = (
            (h4_rsi_series > RSI_OVERSOLD)
            & (h4_rsi_series.shift(1) <= RSI_OVERSOLD).fillna(False)
            & (h4_close > h4_ema50)
        )
        rsi_full = rsi_cross_h4.reindex(ts_idx, method="ffill").fillna(False)
        rsi_recovery_15m = (rsi_full & ~rsi_full.shift(1).fillna(False)).to_numpy(dtype=bool)
    else:
        h4_rsi_arr = np.full(n, np.nan)
        h4_close_vs_ema50_pct = np.full(n, np.nan)
        rsi_recovery_15m = np.zeros(n, dtype=bool)

    # Daily MACD + flip features
    d_close = s_close.resample("1D").last().dropna()
    if len(d_close) >= 30:
        d_macd_df = _macd(d_close, 12, 26, 9)
        d_hist = d_macd_df["histogram"]
        d_hist_shifted = d_hist.copy()
        d_hist_shifted.index = d_hist_shifted.index + pd.Timedelta("1D")
        daily_macd_hist = d_hist_shifted.reindex(ts_idx, method="ffill").to_numpy(dtype=float)

        daily_bull = (d_macd_df["histogram"] > 0) & (d_macd_df["macd"] > d_macd_df["signal"])
        daily_bear = (d_macd_df["histogram"] < 0) & (d_macd_df["macd"] < d_macd_df["signal"])
        db_arr_d = daily_bull.to_numpy(dtype=bool)
        dbear_arr_d = daily_bear.to_numpy(dtype=bool)

        days_since_bull = np.full(len(daily_bull), np.nan, dtype=float)
        days_since_bear = np.full(len(daily_bear), np.nan, dtype=float)
        last_bull_flip = -1; last_bear_flip = -1
        for i in range(len(db_arr_d)):
            if i > 0 and db_arr_d[i] and not db_arr_d[i - 1]:
                last_bull_flip = i
            if i > 0 and dbear_arr_d[i] and not dbear_arr_d[i - 1]:
                last_bear_flip = i
            if last_bull_flip >= 0:
                days_since_bull[i] = i - last_bull_flip
            if last_bear_flip >= 0:
                days_since_bear[i] = i - last_bear_flip

        s_dsbu = pd.Series(days_since_bull, index=daily_bull.index)
        s_dsbe = pd.Series(days_since_bear, index=daily_bear.index)
        s_dsbu.index = s_dsbu.index + pd.Timedelta("1D")
        s_dsbe.index = s_dsbe.index + pd.Timedelta("1D")
        days_since_bull_flip_arr = s_dsbu.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        days_since_bear_flip_arr = s_dsbe.reindex(ts_idx, method="ffill").to_numpy(dtype=float)

        # 15m-aligned daily-bull and daily-bear regime arrays (for detectors)
        # Same +1day shift to avoid look-ahead.
        daily_bull_shifted = daily_bull.copy()
        daily_bear_shifted = daily_bear.copy()
        daily_bull_shifted.index = daily_bull_shifted.index + pd.Timedelta("1D")
        daily_bear_shifted.index = daily_bear_shifted.index + pd.Timedelta("1D")
        db_15m = daily_bull_shifted.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
        dbear_15m = daily_bear_shifted.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
    else:
        daily_macd_hist = np.full(n, np.nan)
        days_since_bull_flip_arr = np.full(n, np.nan)
        days_since_bear_flip_arr = np.full(n, np.nan)
        db_15m = np.zeros(n, dtype=bool)
        dbear_15m = np.zeros(n, dtype=bool)

    return {
        "ts_idx": ts_idx,
        "atr14": atr14,
        "h": h, "lo": lo, "c": c,
        # Detector inputs
        "db_15m": db_15m,
        "dbear_15m": dbear_15m,
        "h4_cross_up_15m": h4_cross_up_15m,
        "h4_cross_down_15m": h4_cross_down_15m,
        "rsi_recovery_15m": rsi_recovery_15m,
        # Per-bar features (aligned to 15m)
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
    }


# --- DETECTORS (just signal index lists) ---------------------------------

def detect_macd_pullback_long(n, db, h4_cu):
    return [i for i in range(n) if h4_cu[i] and db[i]]


def detect_macd_pullback_short(n, dbear, h4_cd):
    return [i for i in range(n) if h4_cd[i] and dbear[i]]


def detect_macd_early_trend_short(n, dbear, h4_cd):
    flip = np.zeros(n, dtype=bool)
    if n > 0:
        flip[0] = dbear[0]
        flip[1:] = dbear[1:] & ~dbear[:-1]
    flip_idx = np.where(flip)[0]
    fresh = np.zeros(n, dtype=bool)
    for start in flip_idx:
        end = min(n, start + EARLY_TREND_FRESH_BARS)
        fresh[start:end] = True
    return [i for i in range(n) if h4_cd[i] and dbear[i] and fresh[i]]


def detect_rsi_recovery_long(n, rsi_recovery):
    return [i for i in range(n) if rsi_recovery[i]]


# --- BTC-RELATIVE FEATURES (computed once, then aligned per asset) -------

def compute_btc_features_global(btc_df: pd.DataFrame):
    btc = btc_df.sort_values("timestamp").reset_index(drop=True)
    ts = btc["timestamp"].to_numpy()
    c = btc["close"].to_numpy(dtype=float)
    n = len(c)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
    s_close = pd.Series(c, index=ts_idx)
    h4 = s_close.resample("4h").last().dropna()
    h4_ema50 = _ema(h4, EMA_TREND_SPAN)
    btc_above_ema = (h4 > h4_ema50).astype(float)
    btc_above_ema_shifted = btc_above_ema.copy()
    btc_above_ema_shifted.index = btc_above_ema_shifted.index + pd.Timedelta("4h")
    btc_24h_return = np.full(n, np.nan)
    if n > 96:
        btc_24h_return[96:] = c[96:] / c[:-96] - 1.0
    btc_24h_return_s = pd.Series(btc_24h_return, index=ts_idx)
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
    h4_macd_df = _macd(h4)
    btc_score_h4 = np.tanh(
        h4_macd_df["histogram"] / (0.005 * h4.replace(0, np.nan))
    )
    btc_score_shifted = btc_score_h4.copy()
    btc_score_shifted.index = btc_score_shifted.index + pd.Timedelta("4h")
    return btc_above_ema_shifted, btc_24h_return_s, btc_realized_vol_z, btc_score_shifted


# --- BREADTH (cross-section, computed once) ------------------------------

def compute_breadth_at_h4(candles: pd.DataFrame) -> pd.DataFrame:
    pivot = candles.pivot_table(
        index="timestamp", columns="asset", values="close", aggfunc="last"
    )
    h4 = pivot.resample("4h").last()
    h4_ret = h4.pct_change()
    breadth_up = (h4_ret > 0).sum(axis=1) / h4_ret.notna().sum(axis=1)
    breadth_down = (h4_ret < 0).sum(axis=1) / h4_ret.notna().sum(axis=1)
    breadth_up = breadth_up.replace([np.inf, -np.inf], np.nan)
    breadth_down = breadth_down.replace([np.inf, -np.inf], np.nan)
    breadth_up.index = breadth_up.index + pd.Timedelta("4h")
    breadth_down.index = breadth_down.index + pd.Timedelta("4h")
    return pd.DataFrame({"breadth_up": breadth_up, "breadth_down": breadth_down})


# --- MAIN ------------------------------------------------------------------

FEATURE_COLS = [
    "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
    "close_to_high50_atr", "close_to_low50_atr",
    "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
    "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
    "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
]


def main() -> None:
    started = time.monotonic()

    print(f"[{time.monotonic() - started:6.1f}s] loading snapshot {SNAPSHOT_DATE}...")
    candles_all = pd.read_parquet(CANDLES_PARQUET)
    universe = pd.read_parquet(UNIVERSE_PARQUET)
    print(f"  candles: {len(candles_all):,} rows, {candles_all['asset'].nunique()} assets")

    universe = universe.copy()
    universe["asset_key"] = universe["symbol"].str.replace("_", "", regex=False)
    eligible_uni = universe[
        (universe["is_leveraged"] == False)  # noqa: E712
        & (universe["in_delisting"] == False)  # noqa: E712
        & (universe["quote_volume_24h"] >= MIN_QUOTE_VOLUME_24H)
    ].nlargest(TOP_N_COINS, "quote_volume_24h")
    eligible_assets = set(eligible_uni["asset_key"].astype(str))
    candles = candles_all[candles_all["asset"].isin(eligible_assets)].copy()
    print(f"[{time.monotonic() - started:6.1f}s] eligible: {len(eligible_assets)} assets, {len(candles):,} candles")

    # BTC features (global)
    btc = candles_all[candles_all["asset"] == "BTCUSDT"].sort_values("timestamp")
    btc_above_ema, btc_24h_ret, btc_rvz, btc_score = compute_btc_features_global(btc)
    print(f"[{time.monotonic() - started:6.1f}s] BTC global features computed")

    # Breadth
    breadth_df = compute_breadth_at_h4(candles)
    print(f"[{time.monotonic() - started:6.1f}s] breadth computed ({len(breadth_df)} 4h boundaries)")

    # Per-asset detect + features + simulate
    candles_by_asset = {a: g.sort_values("timestamp").reset_index(drop=True)
                        for a, g in candles.groupby("asset")}
    asset_list = sorted(eligible_assets)

    all_trades: list[dict] = []
    funnel = {"macd_pullback_short": 0, "macd_early_trend_short": 0,
              "macd_pullback_long": 0, "rsi_recovery_long": 0}

    for k_idx, asset in enumerate(asset_list):
        df_a = candles_by_asset.get(asset)
        if df_a is None or len(df_a) < 4 * 96 + 30 * 96:
            continue

        feats = compute_asset_features(df_a)
        ts_idx = feats["ts_idx"]; n_bars = len(ts_idx)
        atr14 = feats["atr14"]; h_arr = feats["h"]; lo_arr = feats["lo"]; c_arr = feats["c"]

        # Run detectors
        e_pb_long = _apply_cooldown(
            detect_macd_pullback_long(n_bars, feats["db_15m"], feats["h4_cross_up_15m"]),
            COOLDOWN_BARS,
        )
        e_pb_short = _apply_cooldown(
            detect_macd_pullback_short(n_bars, feats["dbear_15m"], feats["h4_cross_down_15m"]),
            COOLDOWN_BARS,
        )
        e_et_short = _apply_cooldown(
            detect_macd_early_trend_short(n_bars, feats["dbear_15m"], feats["h4_cross_down_15m"]),
            COOLDOWN_BARS,
        )
        e_rsi_long = _apply_cooldown(
            detect_rsi_recovery_long(n_bars, feats["rsi_recovery_15m"]),
            COOLDOWN_BARS,
        )

        funnel["macd_pullback_long"] += len(e_pb_long)
        funnel["macd_pullback_short"] += len(e_pb_short)
        funnel["macd_early_trend_short"] += len(e_et_short)
        funnel["rsi_recovery_long"] += len(e_rsi_long)

        # Feature lookup helper (per index)
        feat_col_arrays = {col: feats[col] for col in FEATURE_COLS}

        # BTC + breadth lookups (vectorized via reindex)
        btc_above_arr = btc_above_ema.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        btc_24h_arr = btc_24h_ret.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        btc_rvz_arr = btc_rvz.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        btc_score_arr = btc_score.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        breadth_up_arr = breadth_df["breadth_up"].reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        breadth_down_arr = breadth_df["breadth_down"].reindex(ts_idx, method="ffill").to_numpy(dtype=float)

        # Simulate + collect (with features)
        for det_name, entries, direction in (
            ("macd_pullback_long", e_pb_long, "long"),
            ("macd_pullback_short", e_pb_short, "short"),
            ("macd_early_trend_short", e_et_short, "short"),
            ("rsi_recovery_long", e_rsi_long, "long"),
        ):
            for ei in entries:
                atr_e = float(atr14[ei]) if ei < len(atr14) and np.isfinite(atr14[ei]) else float("nan")
                trade = simulate_e2_exit(h_arr, lo_arr, c_arr, ei, atr_e, direction)
                if trade is None:
                    continue
                row = {
                    "detector": det_name,
                    "asset": asset,
                    "direction": direction,
                    "entry_time": pd.Timestamp(ts_idx[ei]).isoformat(),
                    "entry_price": trade["entry_price"],
                    "exit_price": trade["exit_price"],
                    "exit_reason": trade["exit_reason"],
                    "pnl_pct": trade["pnl_pct"],
                    "bars_held": trade["bars_held"],
                    "atr14_at_entry": trade["atr14_at_entry"],
                }
                # Per-bar features
                for col in FEATURE_COLS:
                    arr = feat_col_arrays[col]
                    row[col] = float(arr[ei]) if ei < len(arr) and np.isfinite(arr[ei]) else np.nan
                # BTC + breadth
                row["btc_above_4h_ema50"] = float(btc_above_arr[ei]) if ei < len(btc_above_arr) else np.nan
                row["btc_24h_return"] = float(btc_24h_arr[ei]) if ei < len(btc_24h_arr) else np.nan
                row["btc_realized_vol_z"] = float(btc_rvz_arr[ei]) if ei < len(btc_rvz_arr) else np.nan
                row["btc_score"] = float(btc_score_arr[ei]) if ei < len(btc_score_arr) else np.nan
                row["breadth_up"] = float(breadth_up_arr[ei]) if ei < len(breadth_up_arr) else np.nan
                row["breadth_down"] = float(breadth_down_arr[ei]) if ei < len(breadth_down_arr) else np.nan
                all_trades.append(row)

        if (k_idx + 1) % 10 == 0 or k_idx == len(asset_list) - 1:
            print(f"[{time.monotonic() - started:6.1f}s] {k_idx+1}/{len(asset_list)}  "
                  f"funnel={funnel}  trades={len(all_trades):,}")

    print(f"[{time.monotonic() - started:6.1f}s] joining post-asset features (signals_same_15m, hour, dow)...")
    trades = pd.DataFrame(all_trades)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)

    # signals_same_15m_same_detector — count of trades with same entry_time + detector
    same_15m = trades.groupby(["entry_time", "detector"]).size().reset_index(name="signals_same_15m_same_detector")
    trades = trades.merge(same_15m, on=["entry_time", "detector"])

    # Time-of-week cyclical
    hour = trades["entry_time"].dt.hour
    dow = trades["entry_time"].dt.dayofweek
    trades["hour_sin"] = np.sin(2 * math.pi * hour / 24)
    trades["hour_cos"] = np.cos(2 * math.pi * hour / 24)
    trades["dow_sin"] = np.sin(2 * math.pi * dow / 7)
    trades["dow_cos"] = np.cos(2 * math.pi * dow / 7)

    # Outcome label
    trades["label"] = (trades["pnl_pct"] > 0).astype(int)

    # Final feature set
    final_features = FEATURE_COLS + [
        "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
        "breadth_up", "breadth_down", "signals_same_15m_same_detector",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    ]

    print(f"\n[{time.monotonic() - started:6.1f}s] missing-rate per feature:")
    for col in final_features:
        miss = trades[col].isna().mean() * 100
        print(f"  {col:30s}  {miss:>5.1f}% missing")

    n_before = len(trades)
    trades = trades.dropna(subset=final_features).reset_index(drop=True)
    print(f"\n  trades before NaN-drop: {n_before:,}")
    print(f"  trades after  NaN-drop: {len(trades):,}")
    print(f"\n  per-detector:")
    print(trades.groupby("detector").size().to_string())
    print(f"\n  per-quarter (across all detectors):")
    trades["quarter"] = trades["entry_time"].dt.to_period("Q").astype(str)
    print(trades.groupby(["quarter", "detector"]).size().unstack(fill_value=0).to_string())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    keep_cols = (
        ["detector", "asset", "entry_time", "direction", "atr14_at_entry", "label",
         "pnl_pct", "exit_reason", "bars_held", "entry_price", "exit_price"]
        + final_features
    )
    out = trades[keep_cols].copy()
    out.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\n=== DONE — wall_time {time.monotonic() - started:.1f}s ===")
    print(f"output: {OUTPUT_PARQUET}")
    print(f"size:   {OUTPUT_PARQUET.stat().st_size / 1024:.1f} KB")
    print(f"shape:  {out.shape}")


if __name__ == "__main__":
    main()
