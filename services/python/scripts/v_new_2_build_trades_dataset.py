"""v_new_2 — Build the trade-level dataset for Layer 2 BGM training.

For each of the 8 deployable rule configs (4 tiers × 2 sides), simulate trades
on the full v_new_1_v2 dataset and record:
  - Entry timestamp, exit timestamp
  - Side, tier, pb_kind, pb_level, exit_method
  - Realized pnl_pct
  - 44 v_new_1 base features at entry bar
  - 6 rule-derived features at entry bar (pullback_pct, pullback_atr, days_in_trend, ...)
  - Binary target: profitable = (pnl_pct > 0)

Output:
  data/v_new_1_v2/v_new_2_trades_dataset.parquet  (one row per trade)

This is the input to v_new_2_train_bgm_grader.py.
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))

# Reuse the proven backtest simulator
sys.path.insert(0, str(ROOT / "services" / "python" / "scripts"))
from v_new_2_rules_backtest_v2 import _backtest_symbol  # noqa: E402

# 8 deployable configs from rules_v2_verdict
DEPLOYABLE_CONFIGS = [
    ("long",  "top25",   "atr",  2.0, "atr_trail_3"),
    ("long",  "26-50",   "pct", 15.0, "atr_trail_3"),
    ("long",  "51-100",  "atr",  3.0, "atr_trail_2"),
    ("long",  "101-200", "atr",  4.0, "atr_trail_2"),
    ("short", "top25",   "atr",  4.0, "atr_trail_2"),
    ("short", "26-50",   "pct", 10.0, "atr_trail_2"),
    ("short", "51-100",  "atr",  4.0, "atr_trail_2"),
    ("short", "101-200", "atr",  4.0, "atr_trail_2"),
]

# 44 v_new_1 base features (excluding signals_same_15m_same_detector)
V_NEW_1_FEATURES = [
    "atr14_pct_rank_90d","vol_z_24h","coin_7d_return","coin_30d_return",
    "close_to_high50_atr","close_to_low50_atr","bar4h_close_pos_in_range",
    "bar4h_body_pct","bar4h_upper_wick_pct","h4_macd_hist","h4_macd_macd",
    "h4_rsi","h4_close_vs_ema50_pct","daily_macd_hist","days_since_bull_flip",
    "days_since_bear_flip","btc_above_4h_ema50","btc_24h_return","btc_realized_vol_z",
    "btc_score","breadth_up","breadth_down","hour_sin","hour_cos","dow_sin","dow_cos",
    "cvd_slope_1h_norm","bb_pct_b","bb_bandwidth","fib_pos_50","kdj_k","kdj_j",
    "price_vs_ema9_pct","rsi14_delta_1bar","rsi14_delta_4bar","macd_hist_momentum",
    "macd_signal_spread_norm","parkinson_vol","obv_slope_norm","cfgi_value",
    "vol_rank_in_top100","days_since_last_big_move_long","days_since_last_big_move_short",
    "coin_24h_vol_zscore_30d",
]

# Rule-derived features (from features_v_new_2_v2.parquet)
RULE_FEATURES = [
    "d_ema_spread_pct","days_since_long_flip","days_since_short_flip",
    "pullback_pct","rise_pct","pullback_atr","rise_atr","atr14_pct","trend_strength",
]


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} Building v_new_2 trade-level dataset")

    # Load rule + base feature data
    print(f"  Loading features_v_new_2_v2.parquet ...")
    rules = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet")
    rules["timestamp"] = pd.to_datetime(rules["timestamp"], utc=True)
    rules = rules.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"    rules shape: {rules.shape}")

    print(f"  Loading features_full.parquet (base 44) ...")
    base = pd.read_parquet(DATA_DIR / "features_full.parquet",
                            columns=["symbol","timestamp"] + V_NEW_1_FEATURES)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    base = base.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"    base shape: {base.shape}")

    # Merge rule + base into one frame for trade enrichment
    print(f"  Merging rule + base features...")
    feats = rules.merge(base, on=["symbol","timestamp"], how="left")
    print(f"    merged shape: {feats.shape}")

    # Run the 8 deployable configs and collect trades
    print(f"\n  Running 8 deployable configs to collect trades...")
    all_trades: list[dict] = []
    for side, tier, pb_kind, pb_lvl, exit_m in DEPLOYABLE_CONFIGS:
        t_var = time.time()
        sub = feats[feats["mcap_tier"] == tier]
        syms = sub["symbol"].unique()
        n_trades_var = 0
        for sym in syms:
            g = sub[sub["symbol"]==sym].reset_index(drop=True)
            if len(g) < 50: continue
            trades = _backtest_symbol(g, side, pb_kind, pb_lvl, exit_m)
            for tr in trades:
                tr["pb_kind"] = pb_kind
                tr["pb_level"] = pb_lvl
                tr["exit_method"] = exit_m
                all_trades.append(tr)
                n_trades_var += 1
        print(f"    {side} {tier:9s}  pb={pb_kind}={pb_lvl}  exit={exit_m:13s}  "
              f"n={n_trades_var:>5}  ({time.time()-t_var:.1f}s)")

    trades_df = pd.DataFrame(all_trades)
    print(f"\n  total trades collected: {len(trades_df):,}")

    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)

    # Enrich each trade with features at entry bar
    print(f"  Enriching trades with features at entry bar...")
    feat_cols_to_pull = ["symbol","timestamp"] + V_NEW_1_FEATURES + RULE_FEATURES
    feats_for_join = feats[feat_cols_to_pull].copy()
    feats_for_join = feats_for_join.rename(columns={"timestamp": "entry_time"})

    enriched = trades_df.merge(
        feats_for_join, on=["symbol", "entry_time"], how="left",
    )
    print(f"    enriched shape: {enriched.shape}")

    # Drop rows with all-NaN base features (shouldn't happen, but safety)
    nn = enriched[V_NEW_1_FEATURES].notna().sum(axis=1)
    print(f"    rows with >=20 non-NaN base features: {(nn >= 20).sum():,}")

    # Add binary target
    enriched["is_profitable"] = (enriched["pnl_pct"] > 0).astype(int)
    print(f"    profitable rate: {enriched['is_profitable'].mean()*100:.1f}%")

    # Year/regime tags for downstream split
    enriched["entry_year"] = enriched["entry_time"].dt.year
    enriched["entry_month"] = enriched["entry_time"].dt.to_period("M").astype(str)

    # Reorder columns: trade meta first, features next, target last
    meta_cols = ["symbol","tier","side","pb_kind","pb_level","exit_method",
                 "entry_time","exit_time","bars_held","exit_reason","pnl_pct",
                 "entry_year","entry_month"]
    out_cols = (meta_cols
                + [c for c in V_NEW_1_FEATURES if c in enriched.columns]
                + [c for c in RULE_FEATURES if c in enriched.columns]
                + ["is_profitable"])
    out = enriched[out_cols]

    out_path = DATA_DIR / "v_new_2_trades_dataset.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\n  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)  rows={len(out):,}  cols={out.shape[1]}")

    # Summary by side × tier × year
    print(f"\n  Trade distribution by side × tier × year:")
    summary = out.groupby(["side","tier","entry_year"]).agg(
        n=("symbol","size"), wr=("is_profitable","mean"), avg_pnl=("pnl_pct","mean")
    ).reset_index()
    summary["wr"] = summary["wr"]*100
    summary["avg_pnl"] = summary["avg_pnl"]*100
    print(summary.to_string(index=False))

    elapsed = time.time() - t0
    print(f"\n{_ts()} DONE in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
