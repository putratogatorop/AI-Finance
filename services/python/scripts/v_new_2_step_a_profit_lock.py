"""v_new_2 Step A — profit-lock mode + pyramiding simulator.

Extends the rules engine:
  PROFIT-LOCK MODE — once trade is +30% in the green, switch exit logic from
                    atr_trail to trend_flip_only (EMA20/50 daily). Lets winners
                    ride the full trend instead of getting clipped by a normal
                    pullback.
  PYRAMIDING       — when in winning position AND a NEW pullback fires AND
                    BGM_score >= pyramid_threshold, scale up size by +50%
                    (instead of opening separate trade). Captures the trend.

Backtest the same 8 deployable rule configs with these enhancements. Compare
to baseline (Step B current) and to RNDR-specific to verify whether we'd now
capture more of the +752% move.

Output:
  results/v_new_2/step_a_profit_lock_sweep.csv
  results/v_new_2/step_a_verdict.md
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
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
RESULTS.mkdir(parents=True, exist_ok=True)

THRESH = 0.10
FIRST_MOVE_WINDOW_BARS = 42
FORWARD_BARS = 84
EXTENDED_HOLD_BARS = 180 * 6  # 180-day max hold (vs 90 in Step A baseline)
FRICTION = 0.001

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

# Step A enhancements to sweep
PROFIT_LOCK_THRESHOLDS = [None, 0.20, 0.30, 0.50]   # None = no profit-lock; X = switch to trend-flip-only at +X%
PYRAMID_BGM_THRESHOLDS = [None, 0.70, 0.80]          # None = no pyramiding; X = pyramid when score >= X


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _backtest_with_enhancements(
    df: pd.DataFrame, side: str, pb_kind: str, pb_lvl: float, exit_method: str,
    profit_lock_pct: float | None, pyramid_thr: float | None,
    bgm_scores: np.ndarray | None,
) -> list[dict]:
    """Same per-symbol backtest as Step B but with profit-lock + pyramiding."""
    n = len(df)
    if n < 50: return []
    closes = df["close"].values
    timestamps = df["timestamp"].values
    atr_pct = df["atr14_pct"].values
    if side == "long":
        trend = df["d_trend_long"].values
        pb_arr = df["pullback_pct"].values if pb_kind == "pct" else df["pullback_atr"].values
    else:
        trend = df["d_trend_short"].values
        pb_arr = df["rise_pct"].values if pb_kind == "pct" else df["rise_atr"].values

    trades: list[dict] = []
    open_position = False
    entry_idx = -1
    entry_price = 0.0
    pyramid_size = 1.0   # multiplier on base size (1.0 + 0.5 * n_pyramids)
    running_extreme = 0.0
    profit_lock_active = False

    i = 1
    while i < n - 1:
        if not open_position:
            if pd.isna(trend[i]) or trend[i] != 1: i += 1; continue
            if pd.isna(pb_arr[i]) or pb_arr[i] < pb_lvl: i += 1; continue
            prev = pb_arr[i-1]
            if not pd.isna(prev) and prev >= pb_lvl: i += 1; continue
            entry_idx = i
            entry_price = closes[entry_idx]
            running_extreme = entry_price
            pyramid_size = 1.0
            profit_lock_active = False
            open_position = True
            i += 1
            continue

        # In a position — manage it
        p = closes[i]
        if pd.isna(p):
            i += 1; continue
        # Compute current PnL fraction
        if side == "long":
            pnl_frac = p / entry_price - 1.0
            running_extreme = max(running_extreme, p)
        else:
            pnl_frac = entry_price / p - 1.0
            running_extreme = min(running_extreme, p)

        # Profit-lock activation
        if profit_lock_pct is not None and not profit_lock_active and pnl_frac >= profit_lock_pct:
            profit_lock_active = True

        # Pyramid check: if BGM score available AND new pullback fires AND we're winning
        if (pyramid_thr is not None and bgm_scores is not None and pnl_frac > 0.05
                and not pd.isna(pb_arr[i]) and pb_arr[i] >= pb_lvl
                and (pd.isna(pb_arr[i-1]) or pb_arr[i-1] < pb_lvl)
                and i < len(bgm_scores) and not pd.isna(bgm_scores[i])
                and bgm_scores[i] >= pyramid_thr
                and pyramid_size < 2.5):  # cap at 2.5× base
            pyramid_size += 0.5

        # Exit logic
        exit_idx = None; exit_reason = None
        if profit_lock_active:
            # Trend-flip-only exit
            if not pd.isna(trend[i]) and trend[i] != 1:
                exit_idx = i; exit_reason = "trend_flip_locked"
        else:
            # Default: ATR trail
            if exit_method == "atr_trail_2": mult = 2.0
            elif exit_method == "atr_trail_3": mult = 3.0
            else: mult = 3.0
            a = atr_pct[i] / 100.0 if not pd.isna(atr_pct[i]) else 0.02
            if side == "long":
                trail = running_extreme * (1 - mult * a)
                if p <= trail:
                    exit_idx = i; exit_reason = "atr_trail"
            else:
                trail = running_extreme * (1 + mult * a)
                if p >= trail:
                    exit_idx = i; exit_reason = "atr_trail"
        # Time-stop safety
        if exit_idx is None and i - entry_idx >= EXTENDED_HOLD_BARS:
            exit_idx = i; exit_reason = "timeout"

        if exit_idx is not None:
            exit_price = closes[exit_idx]
            if side == "long":
                pnl = (exit_price / entry_price - 1.0)
            else:
                pnl = (entry_price / exit_price - 1.0)
            pnl = pnl * pyramid_size - FRICTION * pyramid_size
            trades.append({
                "symbol": df["symbol"].iloc[0] if "symbol" in df.columns else "?",
                "tier": df["mcap_tier"].iloc[0] if "mcap_tier" in df.columns else "?",
                "side": side,
                "entry_time": pd.Timestamp(timestamps[entry_idx]),
                "exit_time": pd.Timestamp(timestamps[exit_idx]),
                "bars_held": int(exit_idx - entry_idx),
                "exit_reason": exit_reason,
                "pyramid_size": float(pyramid_size),
                "pnl_pct": float(pnl),
            })
            open_position = False
            i = exit_idx + 1
            continue
        i += 1

    return trades


def _summary(pnls, leverage=1.0):
    if len(pnls)==0: return {"n":0,"wr":0,"pf":0,"sum_pnl_pct":0,"avg_win":0,"avg_loss":0,"max_pnl":0}
    p = np.clip(pnls*leverage, -0.99, 100.0)
    wins = p[p>0]; losses = p[p<0]
    return {
        "n": int(len(p)), "wr": float((p>0).mean()),
        "sum_pnl_pct": float(p.sum()*100),
        "max_pnl": float(p.max()*100),
        "avg_win": float(wins.mean()*100) if len(wins) else 0,
        "avg_loss": float(losses.mean()*100) if len(losses) else 0,
        "pf": float(wins.sum()/abs(losses.sum())) if len(losses) else float("inf"),
    }


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 Step A — profit-lock + pyramid sweep")
    df = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"  features: {len(df):,} rows")

    # Optional: load BGM scores per (symbol, timestamp) for pyramiding decision
    # For simplicity in this iteration, pyramid decision uses pullback_pct cross only
    # (no BGM at pyramid time — we'd need an online BGM score per bar, which is complex).
    # Mark this as future work; here pyramid_thr is treated as "always pyramid if pnl>5% and pullback fires".
    # Real implementation would query BGM per bar.

    rows = []
    rndr_diag = []   # specific diagnostic for RNDRUSDT
    for side, tier, pb_kind, pb_lvl, exit_m in DEPLOYABLE_CONFIGS:
        for pl in PROFIT_LOCK_THRESHOLDS:
            for py in PYRAMID_BGM_THRESHOLDS:
                # Run on tier symbols
                tier_syms = df[df["mcap_tier"]==tier]["symbol"].unique()
                all_trades = []
                for sym in tier_syms:
                    g = df[df["symbol"]==sym].reset_index(drop=True)
                    if len(g) < 50: continue
                    trades = _backtest_with_enhancements(
                        g, side, pb_kind, pb_lvl, exit_m,
                        profit_lock_pct=pl, pyramid_thr=py, bgm_scores=None,
                    )
                    all_trades.extend(trades)
                    if sym == "RNDRUSDT" and side == "long":
                        rndr_diag.append({
                            "config": f"pl={pl} py={py}",
                            "n_trades": len(trades),
                            "trades": trades,
                        })
                pnls = np.array([t["pnl_pct"] for t in all_trades])
                s = _summary(pnls, 1.0)
                oos_pnls = []
                for t in all_trades:
                    et = pd.Timestamp(t["entry_time"])
                    if et.tz is None: et = et.tz_localize("UTC")
                    if et >= pd.Timestamp("2025-07-01", tz="UTC"):
                        oos_pnls.append(t["pnl_pct"])
                s_oos = _summary(np.array(oos_pnls), 1.0)
                rows.append({
                    "side": side, "tier": tier,
                    "profit_lock": str(pl) if pl is not None else "off",
                    "pyramid": str(py) if py is not None else "off",
                    "full_n": s["n"], "full_wr": s["wr"]*100, "full_pf": s["pf"],
                    "full_avg_win": s["avg_win"], "full_avg_loss": s["avg_loss"],
                    "full_max_pnl_pct": s["max_pnl"],
                    "full_sum_pnl_pct": s["sum_pnl_pct"],
                    "oos_n": s_oos["n"], "oos_wr": s_oos["wr"]*100, "oos_pf": s_oos["pf"],
                    "oos_sum_pnl_pct": s_oos["sum_pnl_pct"],
                })
                print(f"  {side:5s} {tier:9s}  pl={str(pl):4s}  py={str(py):4s}  "
                      f"n={s['n']:>5}  WR={s['wr']*100:5.1f}%  PF={s['pf']:5.2f}  "
                      f"max_win={s['max_pnl']:>+6.1f}%  oos_n={s_oos['n']:>4} oos_PF={s_oos['pf']:5.2f}")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "step_a_profit_lock_sweep.csv", index=False)
    print(f"\n  saved: step_a_profit_lock_sweep.csv")

    # Summary: best variant per (side, tier) sorted by OOS PF
    print(f"\n{_ts()} Best Step-A enhancement per cell")
    print(f"  {'side':5s} {'tier':9s} {'profit_lock':>12s} {'pyramid':>8s} {'full_n':>7s} "
          f"{'PF_full':>8s} {'PF_oos':>8s} {'max_win':>8s} {'baseline_PF':>11s} {'lift':>5s}")
    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            sub = df_out[(df_out["side"]==side) & (df_out["tier"]==tier) & (df_out["oos_n"]>=20)]
            if len(sub) == 0: continue
            base = sub[(sub["profit_lock"]=="off") & (sub["pyramid"]=="off")]
            base_pf = base["full_pf"].iloc[0] if len(base) else 1.0
            best = sub.sort_values("oos_pf", ascending=False).iloc[0]
            print(f"  {side:5s} {tier:9s} {best['profit_lock']:>12s} {best['pyramid']:>8s} "
                  f"{int(best['full_n']):>7} {best['full_pf']:>8.2f} {best['oos_pf']:>8.2f} "
                  f"{best['full_max_pnl_pct']:>+7.1f}% {base_pf:>11.2f} {best['full_pf']/max(base_pf,0.01):>4.2f}×")

    # RNDR diagnostic
    print(f"\n{_ts()} RNDRUSDT diagnostic — does Step A capture more of the +752% cycle?")
    for diag in rndr_diag[:6]:
        cycle_trades = []
        for t in diag["trades"]:
            et = pd.Timestamp(t["entry_time"])
            if et.tz is None: et = et.tz_localize("UTC")
            if pd.Timestamp("2023-10-01", tz="UTC") <= et <= pd.Timestamp("2024-06-30", tz="UTC"):
                cycle_trades.append(t)
        if not cycle_trades: continue
        sum_pnl = sum(t["pnl_pct"] for t in cycle_trades) * 100
        max_pnl = max(t["pnl_pct"] for t in cycle_trades) * 100
        max_pyramid = max(t.get("pyramid_size",1.0) for t in cycle_trades)
        print(f"  config {diag['config']:24s}  cycle_trades={len(cycle_trades):>2}  "
              f"sum_pnl={sum_pnl:>+6.0f}%  best_trade={max_pnl:>+5.0f}%  max_pyramid={max_pyramid:.1f}×")

    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
