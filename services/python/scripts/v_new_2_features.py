"""v_new_2 — Layer 1 rule features.

Per-coin daily EMA20/50 + cycle reference levels + pullback metrics. Output is
keyed by (symbol, 4h-timestamp) for easy merge with the existing v_new_1
features_full.parquet.

Computed columns:
  d_ema20             — daily EMA20 of close, broadcast to 4h
  d_ema50             — daily EMA50 of close, broadcast to 4h
  d_ema_spread_pct    — (ema20-ema50)/ema50 × 100  (positive = uptrend strength)
  d_trend_long        — bool: ema20 > ema50 (LONG-allowed regime)
  d_trend_short       — bool: ema20 < ema50 (SHORT-allowed regime)
  days_since_long_flip   — bars since last bull-cross / 6 (in days)
  days_since_short_flip  — bars since last bear-cross / 6
  cycle_high          — highest close since last bull-cross
  cycle_low           — lowest close since last bear-cross
  pullback_pct        — 100 × (cycle_high - close) / cycle_high   (only meaningful in LONG regime)
  rise_pct            — 100 × (close - cycle_low) / cycle_low      (only meaningful in SHORT regime)
  trend_strength      — abs(ema20 - ema50) / ema50  (general trend force)

Output: data/v_new_1_v2/features_v_new_2.parquet
"""
from __future__ import annotations

import os
import pathlib
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))

EMA_FAST = 20    # daily EMA period
EMA_SLOW = 50
BARS_PER_DAY = 6 # 4h bars per day


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _process_symbol(g: pd.DataFrame) -> pd.DataFrame:
    """Compute all v_new_2 features for one symbol's 4h bar series.
    Returns the input frame with new columns appended."""
    g = g.sort_values("timestamp").reset_index(drop=True)
    if len(g) < EMA_SLOW * BARS_PER_DAY + 10:
        # Not enough history — pad with NaN
        for c in ["d_ema20","d_ema50","d_ema_spread_pct","d_trend_long","d_trend_short",
                  "days_since_long_flip","days_since_short_flip","cycle_high","cycle_low",
                  "pullback_pct","rise_pct","trend_strength"]:
            g[c] = np.nan
        return g

    # Daily resample: take last close of each UTC day
    g_idx = g.set_index("timestamp")
    daily_close = g_idx["close"].resample("1D").last().dropna()
    if len(daily_close) < EMA_SLOW + 5:
        for c in ["d_ema20","d_ema50","d_ema_spread_pct","d_trend_long","d_trend_short",
                  "days_since_long_flip","days_since_short_flip","cycle_high","cycle_low",
                  "pullback_pct","rise_pct","trend_strength"]:
            g[c] = np.nan
        return g

    # Daily EMAs
    d_ema20 = daily_close.ewm(span=EMA_FAST, adjust=False).mean()
    d_ema50 = daily_close.ewm(span=EMA_SLOW, adjust=False).mean()
    d_trend_long_daily = (d_ema20 > d_ema50).astype(int)
    d_trend_short_daily = (d_ema20 < d_ema50).astype(int)

    # Detect cross events on daily series
    long_flip = (d_trend_long_daily.diff() > 0).astype(int)   # 0→1: bull cross
    short_flip = (d_trend_short_daily.diff() > 0).astype(int) # 0→1: bear cross

    # Days since last flip — running counter
    def _days_since(flip_series: pd.Series) -> pd.Series:
        out = np.full(len(flip_series), np.nan)
        last_flip_idx = -10**6
        for i in range(len(flip_series)):
            if flip_series.iloc[i] == 1:
                last_flip_idx = i
            if last_flip_idx >= 0:
                out[i] = i - last_flip_idx
        return pd.Series(out, index=flip_series.index)

    days_since_long_flip_daily = _days_since(long_flip)
    days_since_short_flip_daily = _days_since(short_flip)

    # Cycle high since last LONG cross (using daily closes)
    def _cycle_high_since_flip(closes: pd.Series, flip: pd.Series) -> pd.Series:
        out = np.full(len(closes), np.nan)
        running_max = -np.inf
        for i in range(len(closes)):
            if flip.iloc[i] == 1:
                running_max = closes.iloc[i]
            else:
                running_max = max(running_max, closes.iloc[i]) if running_max != -np.inf else closes.iloc[i]
            out[i] = running_max
        return pd.Series(out, index=closes.index)

    def _cycle_low_since_flip(closes: pd.Series, flip: pd.Series) -> pd.Series:
        out = np.full(len(closes), np.nan)
        running_min = np.inf
        for i in range(len(closes)):
            if flip.iloc[i] == 1:
                running_min = closes.iloc[i]
            else:
                running_min = min(running_min, closes.iloc[i]) if running_min != np.inf else closes.iloc[i]
            out[i] = running_min
        return pd.Series(out, index=closes.index)

    cycle_high_daily = _cycle_high_since_flip(daily_close, long_flip)
    cycle_low_daily = _cycle_low_since_flip(daily_close, short_flip)

    # Build daily lookup frame
    daily_df = pd.DataFrame({
        "d_ema20": d_ema20,
        "d_ema50": d_ema50,
        "d_trend_long": d_trend_long_daily,
        "d_trend_short": d_trend_short_daily,
        "days_since_long_flip": days_since_long_flip_daily,
        "days_since_short_flip": days_since_short_flip_daily,
        "cycle_high_d": cycle_high_daily,
        "cycle_low_d": cycle_low_daily,
    })
    daily_df.index.name = "_day"
    daily_df = daily_df.reset_index()

    # Map 4h timestamp → date (UTC) → daily features
    g["_day"] = g["timestamp"].dt.floor("D")
    g = g.merge(daily_df, on="_day", how="left")
    g = g.drop(columns=["_day"])

    # 4h-level cycle_high / cycle_low: use the daily reference but track
    # intra-day extremes — for simplicity, use daily refs for now (good enough
    # for trend-pullback decisions). Could be refined to 4h granularity later.
    g["cycle_high"] = g["cycle_high_d"].astype(float)
    g["cycle_low"]  = g["cycle_low_d"].astype(float)
    g = g.drop(columns=["cycle_high_d", "cycle_low_d"])

    # Spreads + pullback / rise pct
    g["d_ema_spread_pct"] = (g["d_ema20"] - g["d_ema50"]) / g["d_ema50"] * 100.0
    g["pullback_pct"] = (g["cycle_high"] - g["close"]) / g["cycle_high"] * 100.0
    g["rise_pct"]     = (g["close"] - g["cycle_low"]) / g["cycle_low"] * 100.0
    g["trend_strength"] = (g["d_ema20"] - g["d_ema50"]).abs() / g["d_ema50"]

    return g


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_2 feature build — start")
    print(f"  DATA: {DATA_DIR}")
    feats = pd.read_parquet(
        DATA_DIR / "features_full.parquet",
        columns=["symbol", "timestamp", "close"],
    )
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    feats = feats.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"  loaded {len(feats):,} rows  ({feats['symbol'].nunique()} symbols)")

    out_parts = []
    syms = feats["symbol"].unique()
    for i, sym in enumerate(syms):
        g = feats[feats["symbol"] == sym].copy()
        g = _process_symbol(g)
        out_parts.append(g)
        if (i + 1) % 50 == 0:
            print(f"  {_ts()} processed {i+1}/{len(syms)} symbols  "
                  f"elapsed={time.time()-t0:.1f}s")

    out = pd.concat(out_parts, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    new_cols = ["d_ema20","d_ema50","d_ema_spread_pct","d_trend_long","d_trend_short",
                "days_since_long_flip","days_since_short_flip","cycle_high","cycle_low",
                "pullback_pct","rise_pct","trend_strength"]
    print(f"\n  output shape: {out.shape}")
    print(f"  new columns: {new_cols}")
    print(f"  NaN% per new col:")
    for c in new_cols:
        nn = out[c].isna().mean() * 100
        print(f"    {c:24s}: {nn:5.1f}%")

    # Sanity stats
    print(f"\n  d_trend_long bars:  {int(out['d_trend_long'].sum()):>10,} "
          f"({100*out['d_trend_long'].mean():.1f}%)")
    print(f"  d_trend_short bars: {int(out['d_trend_short'].sum()):>10,} "
          f"({100*out['d_trend_short'].mean():.1f}%)")
    print(f"  pullback_pct stats (LONG-trend bars only):")
    long_only = out[out["d_trend_long"] == 1]["pullback_pct"]
    print(f"    median {long_only.median():.1f}%  Q3 {long_only.quantile(0.75):.1f}%  "
          f"Q90 {long_only.quantile(0.90):.1f}%")

    out_path = DATA_DIR / "features_v_new_2.parquet"
    out.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\n{_ts()} DONE in {elapsed:.1f}s")
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
