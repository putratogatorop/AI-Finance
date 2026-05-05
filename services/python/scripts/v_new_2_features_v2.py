"""v_new_2 — Layer 1 features v2: add ATR + mcap tier on top of v1.

Reads features_v_new_2.parquet (which has daily EMAs + cycle highs) and adds:
  atr14                   — close-based ATR proxy: 14-bar rolling mean of
                              |close[i] - close[i-1]| / close[i-1]
                              (close-only since we don't have raw H/L)
  pullback_atr            — pullback distance in ATR units = pullback_pct / atr14_pct
  rise_atr                — analogous for short
  mcap_tier               — string: top25 / 26-50 / 51-100 / 101-200 / 200+
                              from universe_top100_membership avg rank
  symbol_avg_rank         — numeric average rank across the universe history
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


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _tier_label(rank: float) -> str:
    if pd.isna(rank): return "unranked"
    if rank <= 25:  return "top25"
    if rank <= 50:  return "26-50"
    if rank <= 100: return "51-100"
    if rank <= 200: return "101-200"
    return "200+"


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_2 features v2 — start")
    base = pd.read_parquet(DATA_DIR / "features_v_new_2.parquet")
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    base = base.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"  loaded {len(base):,} rows")

    # ATR-14 close-based proxy: average |Δclose|/close over 14 bars
    abs_ret = base.groupby("symbol")["close"].transform(
        lambda s: (s / s.shift(1) - 1.0).abs()
    )
    base["atr14_pct"] = abs_ret.groupby(base["symbol"]).transform(
        lambda s: s.rolling(14, min_periods=5).mean()
    ) * 100.0   # in %

    # Pullback / rise expressed in ATR units
    base["pullback_atr"] = base["pullback_pct"] / base["atr14_pct"]
    base["rise_atr"]     = base["rise_pct"] / base["atr14_pct"]

    # Mcap tier from universe membership
    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    avg_rank = mem.groupby("symbol")["rank"].mean()
    avg_rank = avg_rank.reset_index().rename(columns={"rank":"symbol_avg_rank"})
    avg_rank["mcap_tier"] = avg_rank["symbol_avg_rank"].apply(_tier_label)
    base = base.merge(avg_rank, on="symbol", how="left")
    base["mcap_tier"] = base["mcap_tier"].fillna("unranked")
    print(f"  tier distribution per symbol:")
    print(avg_rank["mcap_tier"].value_counts().to_string())

    print(f"\n  ATR_14 stats:")
    print(f"    median {base['atr14_pct'].median():.2f}%   "
          f"mean {base['atr14_pct'].mean():.2f}%   "
          f"NaN% {base['atr14_pct'].isna().mean()*100:.1f}")
    for tier in ["top25", "26-50", "51-100", "101-200"]:
        sub = base[base["mcap_tier"] == tier]["atr14_pct"]
        if len(sub) == 0: continue
        print(f"    {tier:8s}: median {sub.median():.2f}%   "
              f"Q90 {sub.quantile(0.90):.2f}%")

    out_path = DATA_DIR / "features_v_new_2_v2.parquet"
    base.to_parquet(out_path, index=False)
    print(f"\n  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")
    print(f"{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
