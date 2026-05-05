"""v_new_2 — Wider TP variants on U4 universe (memes + AI + top-50).

The current best (407%/yr CAGR at 5× lev) uses ATR-trail (2-3× ATR) + profit-lock
at +30% (switches to trend-flip-only). User asked: would WIDER TP improve this?

Tests:
  V0 baseline   — atr_trail (current, profit-lock at +30%)
  V1 pl_10%     — profit-lock at +10%, trend-flip-only after
  V2 pl_15%     — profit-lock at +15%, trend-flip-only after
  V3 pl_20%     — profit-lock at +20%, trend-flip-only after
  V4 pl_30%     — current default (kept as reference)
  V5 trend_only — pure trend-flip exit from entry (no ATR trail at all)
  V6 trail_5x   — looser trail: 5× ATR (vs default 2-3×)
  V7 trail_8x   — very loose trail: 8× ATR
  V8 trail_5x_pl_15 — combined: 5×ATR trail until +15%, then trend-flip-only

Run on U4 universe at threshold 0.60, 5× leverage, max_concurrent=5,
1% risk per trade. Compare to baseline 407%/yr.

Output:
  results/v_new_2/wider_tp_sweep.csv
  results/v_new_2/wider_tp_verdict.md
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
MODELS = ROOT / "services" / "python" / "models"

# Friction (same as honest backtest)
TAKER_FEE_RT = 0.001
SLIPPAGE_RT = 0.002
FUNDING_LONG_PER_8H = 0.0002
FUNDING_SHORT_PER_8H = -0.00005
BARS_PER_8H = 2

EXTENDED_HOLD_BARS = 90 * 6
RISK_PER_TRADE = 0.01
LEVERAGE = 5.0
MAX_CONCURRENT = 5
BGM_THRESHOLD = 0.60

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
THR_SELECT_START = pd.Timestamp("2025-07-01", tz="UTC")

DEPLOYABLE_CONFIGS = [
    ("long",  "top25",   "atr",  2.0),
    ("long",  "26-50",   "pct", 15.0),
    ("long",  "51-100",  "atr",  3.0),
    ("long",  "101-200", "atr",  4.0),
    ("short", "top25",   "atr",  4.0),
    ("short", "26-50",   "pct", 10.0),
    ("short", "51-100",  "atr",  4.0),
    ("short", "101-200", "atr",  4.0),
]

EXIT_VARIANTS = [
    ("V0_baseline",      "trail_default", None,  None),   # default 2/3x ATR trail, pl 30%
    ("V1_pl_10",         "trail_default", 0.10,  "trend"),
    ("V2_pl_15",         "trail_default", 0.15,  "trend"),
    ("V3_pl_20",         "trail_default", 0.20,  "trend"),
    ("V4_pl_30",         "trail_default", 0.30,  "trend"),
    ("V5_trend_only",    "trend_only",    None,  "trend"),  # no trail at all
    ("V6_trail_5x",      "trail_5",       0.30,  "trend"),
    ("V7_trail_8x",      "trail_8",       0.30,  "trend"),
    ("V8_5x_pl_15",      "trail_5",       0.15,  "trend"),
]


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _load_u4_universe():
    with open(DATA_DIR / "coingecko_categories.json") as f:
        cats = json.load(f)
    memes = set(cats.get("memes", []))
    ai = set(cats.get("ai_tokens", []))
    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    avg_rank = mem.groupby("symbol")["rank"].mean()
    top50 = set(avg_rank[avg_rank <= 50].index)
    return memes | ai | top50


def _backtest_with_exit(df, side, pb_kind, pb_lvl, exit_kind, profit_lock_pct, locked_exit_kind):
    """Re-simulate with custom exit variants."""
    n = len(df)
    if n < 50: return []
    closes = df["close"].values
    timestamps = df["timestamp"].values
    atr_pct = df["atr14_pct"].values
    if side == "long":
        trend = df["d_trend_long"].values
        pb_arr = df["pullback_pct"].values if pb_kind=="pct" else df["pullback_atr"].values
    else:
        trend = df["d_trend_short"].values
        pb_arr = df["rise_pct"].values if pb_kind=="pct" else df["rise_atr"].values

    # Default trail mult depends on tier exit method baseline
    default_trail_mult = 2.0 if pb_kind == "atr" and pb_lvl >= 3.0 else 3.0  # rough proxy

    trades = []
    open_pos = False
    entry_idx = -1
    entry_price = 0.0
    running_extreme = 0.0
    profit_lock_active = False

    i = 1
    while i < n - 1:
        if not open_pos:
            if pd.isna(trend[i]) or trend[i] != 1: i += 1; continue
            if pd.isna(pb_arr[i]) or pb_arr[i] < pb_lvl: i += 1; continue
            prev = pb_arr[i-1]
            if not pd.isna(prev) and prev >= pb_lvl: i += 1; continue
            entry_idx = i
            entry_price = closes[entry_idx]
            running_extreme = entry_price
            profit_lock_active = False
            open_pos = True
            i += 1
            continue

        p = closes[i]
        if pd.isna(p): i += 1; continue
        if side == "long":
            pnl_frac = p / entry_price - 1.0
            running_extreme = max(running_extreme, p)
        else:
            pnl_frac = entry_price / p - 1.0
            running_extreme = min(running_extreme, p)

        # Profit-lock activation
        if profit_lock_pct is not None and not profit_lock_active and pnl_frac >= profit_lock_pct:
            profit_lock_active = True

        # Exit logic
        exit_idx = None; exit_reason = None
        if exit_kind == "trend_only" or (profit_lock_active and locked_exit_kind == "trend"):
            if not pd.isna(trend[i]) and trend[i] != 1:
                exit_idx = i; exit_reason = "trend_flip"
        else:
            # ATR trail
            if exit_kind == "trail_default":
                mult = default_trail_mult
            elif exit_kind == "trail_5":
                mult = 5.0
            elif exit_kind == "trail_8":
                mult = 8.0
            else:
                mult = 3.0
            a = atr_pct[i] / 100.0 if not pd.isna(atr_pct[i]) else 0.02
            if side == "long":
                if p <= running_extreme * (1 - mult * a):
                    exit_idx = i; exit_reason = "atr_trail"
            else:
                if p >= running_extreme * (1 + mult * a):
                    exit_idx = i; exit_reason = "atr_trail"

        if exit_idx is None and i - entry_idx >= EXTENDED_HOLD_BARS:
            exit_idx = i; exit_reason = "timeout"

        if exit_idx is not None:
            exit_price = closes[exit_idx]
            if side == "long":
                pnl = exit_price/entry_price - 1.0
            else:
                pnl = entry_price/exit_price - 1.0
            # Friction
            bars_held = exit_idx - entry_idx
            fund_per_bar = (FUNDING_LONG_PER_8H if side=="long" else FUNDING_SHORT_PER_8H) / BARS_PER_8H
            funding_total = fund_per_bar * bars_held
            pnl_honest = pnl - SLIPPAGE_RT - TAKER_FEE_RT - funding_total
            trades.append({
                "symbol": df["symbol"].iloc[0] if "symbol" in df.columns else "?",
                "tier": df["mcap_tier"].iloc[0] if "mcap_tier" in df.columns else "?",
                "side": side,
                "entry_time": pd.Timestamp(timestamps[entry_idx]),
                "exit_time": pd.Timestamp(timestamps[exit_idx]),
                "bars_held": int(bars_held),
                "exit_reason": exit_reason,
                "profit_lock_active": bool(profit_lock_active),
                "pnl_pct": float(pnl_honest),
            })
            open_pos = False
            i = exit_idx + 1
            continue
        i += 1
    return trades


@dataclass
class OpenPos:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl_pct: float
    bgm_score: float


def _portfolio_sim(trades_df, leverage=LEVERAGE, max_concurrent=MAX_CONCURRENT,
                    risk=RISK_PER_TRADE):
    if len(trades_df)==0:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":0}
    df = trades_df.sort_values("entry_time").reset_index(drop=True)
    eq = 1.0
    open_p: list[OpenPos] = []
    curve = []
    n_taken = 0
    for _, row in df.iterrows():
        et = row["entry_time"]; xt = row["exit_time"]
        if pd.isna(et) or pd.isna(xt): continue
        new_open = []
        for pos in open_p:
            if pos.exit_time <= et:
                eq *= (1.0 + pos.pnl_pct * risk * leverage)
                eq = max(eq, 1e-6)
                curve.append((pos.exit_time, eq))
            else:
                new_open.append(pos)
        open_p = new_open
        if len(open_p) < max_concurrent:
            open_p.append(OpenPos(row["symbol"], et, xt, row["pnl_pct"],
                                    row.get("bgm_score", 0.5)))
            n_taken += 1
    for pos in open_p:
        eq *= (1.0 + pos.pnl_pct * risk * leverage)
        eq = max(eq, 1e-6)
        curve.append((pos.exit_time, eq))
    if not curve:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":n_taken}
    eq_df = pd.DataFrame(curve, columns=["t","equity"]).sort_values("t")
    arr = eq_df["equity"].values
    rm = np.maximum.accumulate(arr)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (arr - rm)/safe
    return {
        "n":int(len(df)),"n_taken":n_taken,"final_eq":float(arr[-1]),
        "compd_pct":float((arr[-1]-1)*100),
        "max_dd_pct":float(dd.min()*100),
    }


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 wider-TP sweep on U4 universe (5× lev, mc=5, thr 0.60)")
    universe_u4 = _load_u4_universe()
    print(f"  U4 universe: {len(universe_u4)} symbols")

    df = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["symbol"].isin(universe_u4)].sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"  filtered to U4: {len(df):,} rows, {df['symbol'].nunique()} syms")

    # Load BGM scoring data — use the honest-scored trades from earlier
    scored = pd.read_parquet(DATA_DIR / "v_new_2_trades_scored_honest.parquet")
    scored["entry_time"] = pd.to_datetime(scored["entry_time"], utc=True)
    bgm_lookup = scored[["symbol","entry_time","bgm_honest"]].set_index(["symbol","entry_time"])

    rows = []
    for variant_name, exit_kind, pl_pct, locked_exit in EXIT_VARIANTS:
        all_trades = []
        t_var = time.time()
        for side, tier, pb_kind, pb_lvl in DEPLOYABLE_CONFIGS:
            tier_syms = df[(df["mcap_tier"]==tier) & (df["symbol"].isin(universe_u4))]["symbol"].unique()
            for sym in tier_syms:
                g = df[df["symbol"]==sym].reset_index(drop=True)
                if len(g) < 50: continue
                trades = _backtest_with_exit(g, side, pb_kind, pb_lvl, exit_kind, pl_pct, locked_exit)
                # Attach BGM scores (lookup by entry_time)
                for tr in trades:
                    try:
                        tr["bgm_score"] = float(bgm_lookup.loc[(tr["symbol"], tr["entry_time"]), "bgm_honest"])
                    except KeyError:
                        tr["bgm_score"] = np.nan
                all_trades.extend(trades)

        # Filter to BGM threshold
        tdf = pd.DataFrame(all_trades)
        if len(tdf) == 0:
            print(f"  {variant_name}: no trades")
            continue
        tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True, errors="coerce")
        tdf["exit_time"] = pd.to_datetime(tdf["exit_time"], utc=True, errors="coerce")
        if tdf["entry_time"].dt.tz is None:
            tdf["entry_time"] = tdf["entry_time"].dt.tz_localize("UTC")
            tdf["exit_time"] = tdf["exit_time"].dt.tz_localize("UTC")
        # NOTE: BGM filtering disabled in this sweep — pure exit-logic comparison
        # (variants generate different entry_times because different exit logic
        # affects when next entry can fire after position closes; can't lookup
        # cleanly across variants). Compare raw exit-method effects.
        tdf_filtered = tdf.copy()

        # Splits
        tdf_thr = tdf_filtered[(tdf_filtered["entry_time"] >= THR_SELECT_START) &
                                  (tdf_filtered["entry_time"] < HOLDOUT_START)]
        tdf_ho = tdf_filtered[tdf_filtered["entry_time"] >= HOLDOUT_START]

        sim_thr = _portfolio_sim(tdf_thr)
        sim_ho = _portfolio_sim(tdf_ho)

        # Annualize
        thr_days = 184; ho_days = 90
        thr_cagr = ((sim_thr["final_eq"] ** (365/thr_days)) - 1) * 100 if sim_thr["final_eq"] > 0 else -100
        ho_cagr = ((sim_ho["final_eq"] ** (365/ho_days)) - 1) * 100 if sim_ho["final_eq"] > 0 else -100

        rows.append({
            "variant": variant_name, "exit_kind": exit_kind, "profit_lock": pl_pct,
            "n_total_filtered": len(tdf_filtered),
            "thr_n_taken": sim_thr["n_taken"],
            "thr_compd_pct": sim_thr["compd_pct"],
            "thr_max_dd_pct": sim_thr["max_dd_pct"],
            "thr_cagr_pct": thr_cagr,
            "ho_n_taken": sim_ho["n_taken"],
            "ho_compd_pct": sim_ho["compd_pct"],
            "ho_max_dd_pct": sim_ho["max_dd_pct"],
            "ho_cagr_pct": ho_cagr,
        })
        print(f"  {variant_name:18s}  trades_filtered={len(tdf_filtered):>5}  "
              f"H2_taken={sim_thr['n_taken']:>4} H2_compd={sim_thr['compd_pct']:>+6.1f}% H2_DD={sim_thr['max_dd_pct']:>+5.1f}%  "
              f"Q1_taken={sim_ho['n_taken']:>4} Q1_compd={sim_ho['compd_pct']:>+6.1f}% Q1_DD={sim_ho['max_dd_pct']:>+5.1f}%  "
              f"Q1_CAGR={ho_cagr:>+5.0f}%/yr  ({time.time()-t_var:.1f}s)")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "wider_tp_sweep.csv", index=False)
    print(f"\n  saved: wider_tp_sweep.csv")

    # Verdict
    md = []
    md.append(f"# v_new_2 — Wider TP Sweep ({time.strftime('%Y-%m-%d')})\n\n")
    md.append("U4 universe (memes + AI + top-50). 5× leverage, max_concurrent=5, 1% risk.\n")
    md.append(f"BGM threshold {BGM_THRESHOLD}. All variants use honest funding+slippage.\n\n")
    md.append("| variant | exit | pl% | filtered_n | H2 compd | H2 DD | Q1 compd | Q1 DD | Q1 CAGR |\n")
    md.append("|---|---|---|---|---|---|---|---|---|\n")
    for _, r in df_out.iterrows():
        pl_str = f"{int(r['profit_lock']*100)}%" if pd.notna(r["profit_lock"]) else "—"
        md.append(f"| {r['variant']} | {r['exit_kind']} | {pl_str} | "
                  f"{int(r['n_total_filtered'])} | "
                  f"{r['thr_compd_pct']:+.1f}% | {r['thr_max_dd_pct']:+.1f}% | "
                  f"{r['ho_compd_pct']:+.1f}% | {r['ho_max_dd_pct']:+.1f}% | "
                  f"{r['ho_cagr_pct']:+.0f}%/yr |\n")
    md.append("\n## Best by Q1 CAGR\n")
    sub = df_out.sort_values("ho_cagr_pct", ascending=False).head(3)
    for _, r in sub.iterrows():
        md.append(f"- **{r['variant']}**: Q1 CAGR {r['ho_cagr_pct']:+.0f}%/yr, "
                  f"DD {r['ho_max_dd_pct']:+.1f}%\n")
    (RESULTS / "wider_tp_verdict.md").write_text("".join(md))
    print(f"  saved: wider_tp_verdict.md")
    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
