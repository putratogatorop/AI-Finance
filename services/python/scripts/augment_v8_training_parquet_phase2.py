"""Phase 2 — Augment v8 training parquet with 13 new feature columns.

Reads the existing training parquet + the 15m candles snapshot, computes
new features per-asset without lookahead, and joins to each trade's entry_time.

New features added (13):
  Priority A (restore v1 drops):
    cvd_slope_1h_norm   — 4-bar CVD slope / sum-abs-vol (15m level)
    cvd_divergence_4h   — bear/bull CVD divergence flag on 16-bar window (15m)
    bb_pct_b            — Bollinger %B (20-bar, 2σ) on prior 4h close
    bb_bandwidth        — BB bandwidth / mid on prior 4h close
    fib_pos_50          — (close - low50) / (high50 - low50) on 15m

  Priority B (sister-classifier features):
    kdj_k               — KDJ K line (4h prior bar)
    kdj_j               — KDJ J line (4h prior bar)
    price_vs_ema9_pct   — (h4_close - ema9_4h) / h4_close (prior bar)
    rsi14_delta_1bar    — RSI14[t] - RSI14[t-1] on 4h
    rsi14_delta_4bar    — RSI14[t] - RSI14[t-4] on 4h
    macd_hist_momentum  — h4_macd_hist[t] - h4_macd_hist[t-2] (jerk, 4h)
    macd_signal_spread_norm — h4_macd_hist[t] / h4_close[t] (4h)
    parkinson_vol       — 96-bar Parkinson volatility (15m)

Priority C (taker_buy_ratio, funding) deferred — taker_buy_base absent from snapshot.

Usage from services/python/:
    python scripts/augment_v8_training_parquet_phase2.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── CONFIG ────────────────────────────────────────────────────────────────────

SNAPSHOT_DATE = "2026-04-01"
SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"
CANDLES_PARQUET = SNAPSHOTS_DIR / f"candles_15m_{SNAPSHOT_DATE}.parquet"
TRAINING_DIR = Path(__file__).resolve().parents[1] / "data" / "training"
INPUT_PARQUET = TRAINING_DIR / f"v8_trades_with_features_{SNAPSHOT_DATE}.parquet"
OUTPUT_PARQUET = TRAINING_DIR / f"v8_trades_with_features_{SNAPSHOT_DATE}_p2.parquet"

NEW_FEATURE_COLS = [
    "cvd_slope_1h_norm", "cvd_divergence_4h", "bb_pct_b", "bb_bandwidth",
    "fib_pos_50", "kdj_k", "kdj_j", "price_vs_ema9_pct",
    "rsi14_delta_1bar", "rsi14_delta_4bar",
    "macd_hist_momentum", "macd_signal_spread_norm", "parkinson_vol",
]


# ── INDICATOR HELPERS ─────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_hist(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9) -> pd.Series:
    m = s.ewm(span=fast, adjust=False).mean() - s.ewm(span=slow, adjust=False).mean()
    return m - m.ewm(span=sig, adjust=False).mean()


def _shift4h(s: pd.Series, ts_idx: pd.DatetimeIndex) -> np.ndarray:
    """Shift 4h series forward by 4h then ffill onto 15m index (no lookahead)."""
    s2 = s.copy()
    s2.index = s2.index + pd.Timedelta("4h")
    return s2.reindex(ts_idx, method="ffill").to_numpy(dtype=float)


# ── PER-ASSET FEATURE COMPUTATION ────────────────────────────────────────────

def compute_p2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 13 Phase-2 features for one asset's 15m candles.

    Returns a DataFrame indexed by timestamp with one column per feature.
    No lookahead: all features use only data <= current bar.
    4h features use the prior-completed 4h bar (shifted +4h so they align
    to the first 15m bar of the NEXT period — same convention as training).
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    c = df["close"].astype(float)
    h = df["high"].astype(float)
    lo = df["low"].astype(float)
    v = df["volume"].astype(float)

    s_close = pd.Series(c.values, index=ts_idx)
    s_high = pd.Series(h.values, index=ts_idx)
    s_low = pd.Series(lo.values, index=ts_idx)
    s_vol = pd.Series(v.values, index=ts_idx)

    n = len(df)
    out: dict[str, np.ndarray] = {}

    # ── 15m features ─────────────────────────────────────────────────────────

    # CVD slope over 4 15m bars (1h) — sign(close_diff) * vol proxy for taker flow
    sign_diff = np.sign(c.diff().fillna(0.0))
    cvd = (sign_diff * v).cumsum()
    s_cvd = pd.Series(cvd.values, index=ts_idx)
    cvd_4 = s_cvd.shift(4)
    vol_abs_sum4 = s_vol.rolling(4, min_periods=4).sum()
    with np.errstate(divide="ignore", invalid="ignore"):
        cvd_slope = (s_cvd - cvd_4) / vol_abs_sum4.replace(0.0, np.nan)
    out["cvd_slope_1h_norm"] = cvd_slope.to_numpy(dtype=float)

    # CVD divergence — bearish (price at 16-bar high but CVD below 15-bar CVD max)
    #                  bullish (price at 16-bar low but CVD above 15-bar CVD min)
    high16 = s_close.rolling(16, min_periods=16).max()
    low16 = s_close.rolling(16, min_periods=16).min()
    cvd_max15 = s_cvd.shift(1).rolling(15, min_periods=15).max()
    cvd_min15 = s_cvd.shift(1).rolling(15, min_periods=15).min()
    bear_div = (s_close >= high16) & (s_cvd < cvd_max15)
    bull_div = (s_close <= low16) & (s_cvd > cvd_min15)
    div = np.zeros(n, dtype=float)
    div[bear_div.values] = 1.0
    div[bull_div.values] = -1.0
    div[~np.isfinite(high16.values)] = np.nan
    out["cvd_divergence_4h"] = div

    # Fib position 50 bars
    h50 = s_close.rolling(50, min_periods=50).max()
    l50 = s_close.rolling(50, min_periods=50).min()
    rng50 = h50 - l50
    with np.errstate(divide="ignore", invalid="ignore"):
        fib = (s_close - l50) / rng50.replace(0.0, np.nan)
    out["fib_pos_50"] = fib.to_numpy(dtype=float)

    # OBV slope (24-bar linear slope / mean volume)
    obv = (np.sign(s_close.diff().fillna(0.0)) * s_vol).cumsum()
    obv_slope = obv.rolling(24).apply(
        lambda x: float(np.polyfit(range(len(x)), x, 1)[0]) if len(x) == 24 else np.nan,
        raw=True,
    )
    mean_vol_24 = s_vol.rolling(24, min_periods=24).mean().replace(0.0, np.nan)
    out["obv_slope_norm"] = (obv_slope / mean_vol_24).to_numpy(dtype=float)

    # Parkinson volatility — 96-bar (24h) rolling
    log_hl = np.log((s_high / s_low).replace(0.0, np.nan))
    parkinson = (log_hl.pow(2).rolling(96, min_periods=96).mean() / (4.0 * np.log(2.0))).apply(
        lambda x: float(np.sqrt(x)) if np.isfinite(x) and x >= 0 else np.nan
    )
    out["parkinson_vol"] = parkinson.to_numpy(dtype=float)

    # ── 4h features (prior completed bar) ────────────────────────────────────

    h4_close = s_close.resample("4h").last().dropna()
    h4_high = s_high.resample("4h").max().reindex(h4_close.index)
    h4_low = s_low.resample("4h").min().reindex(h4_close.index)

    if len(h4_close) < 30:
        for col in ["bb_pct_b", "bb_bandwidth", "kdj_k", "kdj_j",
                    "price_vs_ema9_pct", "rsi14_delta_1bar", "rsi14_delta_4bar",
                    "macd_hist_momentum", "macd_signal_spread_norm"]:
            out[col] = np.full(n, np.nan)
        early_result = pd.DataFrame(out, index=ts_idx)
        early_result.index.name = "feat_ts"
        return early_result

    # Bollinger Bands (20-bar, 2σ)
    bb_mid = h4_close.rolling(20, min_periods=20).mean()
    bb_std = h4_close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_rng = (bb_upper - bb_lower).replace(0.0, np.nan)
    bb_pct_b = (h4_close - bb_lower) / bb_rng
    bb_bandwidth = bb_rng / bb_mid.replace(0.0, np.nan)
    out["bb_pct_b"] = _shift4h(bb_pct_b, ts_idx)
    out["bb_bandwidth"] = _shift4h(bb_bandwidth, ts_idx)

    # KDJ (9-period, 3-smooth) on 4h
    roll_lo = h4_low.rolling(9, min_periods=9).min()
    roll_hi = h4_high.rolling(9, min_periods=9).max()
    span_kd = (roll_hi - roll_lo).replace(0.0, np.nan)
    rsv = 100.0 * (h4_close - roll_lo) / span_kd
    kdj_k = rsv.rolling(3, min_periods=3).mean()
    kdj_d = kdj_k.rolling(3, min_periods=3).mean()
    kdj_j = 3.0 * kdj_k - 2.0 * kdj_d
    out["kdj_k"] = _shift4h(kdj_k, ts_idx)
    out["kdj_j"] = _shift4h(kdj_j, ts_idx)

    # EMA9 on 4h — price_vs_ema9_pct = (h4_close - ema9) / h4_close
    ema9_4h = _ema(h4_close, 9)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_ema9 = (h4_close - ema9_4h) / h4_close.replace(0.0, np.nan)
    out["price_vs_ema9_pct"] = _shift4h(pct_ema9, ts_idx)

    # RSI14 delta on 4h
    rsi_4h = _rsi(h4_close, 14)
    out["rsi14_delta_1bar"] = _shift4h(rsi_4h.diff(1), ts_idx)
    out["rsi14_delta_4bar"] = _shift4h(rsi_4h.diff(4), ts_idx)

    # MACD histogram momentum and signal spread on 4h
    macd_hist_4h = _macd_hist(h4_close, 12, 26, 9)
    hist_mom = macd_hist_4h.diff(2)            # jerk: hist[t] - hist[t-2]
    spread_norm = macd_hist_4h / h4_close.replace(0.0, np.nan)
    out["macd_hist_momentum"] = _shift4h(hist_mom, ts_idx)
    out["macd_signal_spread_norm"] = _shift4h(spread_norm, ts_idx)

    result = pd.DataFrame(out, index=ts_idx)
    result.index.name = "feat_ts"
    return result


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    assert CANDLES_PARQUET.exists(), f"Candles snapshot not found: {CANDLES_PARQUET}"
    assert INPUT_PARQUET.exists(), f"Training parquet not found: {INPUT_PARQUET}"

    print(f"Loading training parquet: {INPUT_PARQUET}")
    trades = pd.read_parquet(INPUT_PARQUET)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"  Trades: {len(trades):,}  columns: {len(trades.columns)}")

    print(f"Loading candles parquet: {CANDLES_PARQUET} ({CANDLES_PARQUET.stat().st_size//1_000_000}MB)...")
    t0 = time.time()
    candles = pd.read_parquet(CANDLES_PARQUET)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    print(f"  Loaded {len(candles):,} candle rows in {time.time()-t0:.1f}s")

    assets = sorted(trades["asset"].unique())
    print(f"\nComputing Phase-2 features for {len(assets)} assets...")

    # For each asset, compute features then join to the trade rows at entry_time
    feat_rows: list[pd.DataFrame] = []
    for i, asset in enumerate(assets):
        ac = candles[candles["asset"] == asset].copy()
        if len(ac) < 200:
            continue
        t1 = time.time()
        feat_df = compute_p2_features(ac)  # indexed by 15m timestamp

        # Get the trade rows for this asset
        asset_trades = trades[trades["asset"] == asset][["entry_time"]].copy()
        if len(asset_trades) == 0:
            continue

        # Join: for each entry_time, take last available feature row (ffill)
        asset_trades = asset_trades.sort_values("entry_time")
        merged = pd.merge_asof(
            asset_trades,
            feat_df.reset_index(),
            left_on="entry_time",
            right_on="feat_ts",
            direction="backward",
        )
        merged = merged.drop(columns=["feat_ts"], errors="ignore")
        merged["asset"] = asset
        feat_rows.append(merged)

        if (i + 1) % 20 == 0 or (i + 1) == len(assets):
            print(f"  [{i+1:3d}/{len(assets)}] {asset:15s} candles={len(ac):,} "
                  f"trades={len(asset_trades):,}  {time.time()-t1:.2f}s")

    print(f"\nMerging features back to trades...")
    feat_all = pd.concat(feat_rows, ignore_index=True)
    result = trades.merge(feat_all, on=["asset", "entry_time"], how="left")

    # Verify new columns present and report null rates
    print(f"\nNew feature null rates (macd_pullback_long only):")
    sub = result[result["detector"] == "macd_pullback_long"]
    for col in NEW_FEATURE_COLS:
        if col in result.columns:
            null_pct = round(float(100.0 * sub[col].isnull().mean()), 1)
            print(f"  {col:35s}  null={null_pct}%")
        else:
            print(f"  {col:35s}  MISSING")

    print(f"\nOriginal columns: {len(trades.columns)}  →  Augmented: {len(result.columns)}")
    print(f"Total rows: {len(result):,}")

    result.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nSaved: {OUTPUT_PARQUET}")


if __name__ == "__main__":
    main()
