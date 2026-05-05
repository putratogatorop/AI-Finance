"""v_new_2 — Rules + BGM combined backtest.

Filter the rule-trigger trades by BGM score >= threshold, then measure PF/WR/PnL
per (side, tier) cell. Compare to rules-only baseline.

Output:
  services/python/results/v_new_2/rules_plus_bgm_sweep.csv
  services/python/results/v_new_2/v_new_2_verdict.md
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
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
RESULTS.mkdir(parents=True, exist_ok=True)

THRESHOLDS = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
FRICTION = 0.0   # already deducted in trades dataset


def _ts():
    return f"[{time.strftime('%H:%M:%S')}]"


def _summary(pnls, leverage=1.0):
    if len(pnls) == 0:
        return {"n":0,"wr":0,"pf":0,"sum_pnl_pct":0,"compd_pct":0,"max_dd_pct":0,
                "avg_win":0,"avg_loss":0}
    p = pnls * leverage
    p = np.clip(p, -0.99, 10.0)
    eq = np.cumprod(1.0 + p)
    rm = np.maximum.accumulate(eq)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (eq - rm) / safe
    wins = p[p > 0]; losses = p[p < 0]
    return {
        "n": int(len(p)),
        "wr": float((p > 0).mean()),
        "sum_pnl_pct": float(p.sum() * 100),
        "compd_pct": float((eq[-1] - 1.0) * 100),
        "max_dd_pct": float(dd.min() * 100),
        "avg_win": float(wins.mean()*100) if len(wins) else 0,
        "avg_loss": float(losses.mean()*100) if len(losses) else 0,
        "pf": float(wins.sum()/abs(losses.sum())) if len(losses) else float("inf"),
    }


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 rules+BGM backtest")
    df = pd.read_parquet(DATA_DIR / "v_new_2_trades_scored.parquet")
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"  loaded {len(df):,} scored trades")

    rows = []
    print(f"\n{_ts()} sweeping thresholds...")
    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            for thr in THRESHOLDS:
                # Full history
                sub_full = df[(df["side"]==side) & (df["tier"]==tier) & (df["bgm_score"]>=thr)]
                # OOS only
                oos_start = pd.Timestamp("2025-07-01", tz="UTC")
                sub_oos = sub_full[sub_full["entry_time"] >= oos_start]
                # H2 + Q1 split
                h2_end = pd.Timestamp("2026-01-01", tz="UTC")
                q1_end = pd.Timestamp("2026-04-01", tz="UTC")
                sub_h2 = sub_full[(sub_full["entry_time"]>=oos_start) & (sub_full["entry_time"]<h2_end)]
                sub_q1 = sub_full[(sub_full["entry_time"]>=h2_end) & (sub_full["entry_time"]<q1_end)]

                s_full = _summary(sub_full["pnl_pct"].values)
                s_oos = _summary(sub_oos["pnl_pct"].values)
                s_h2  = _summary(sub_h2["pnl_pct"].values)
                s_q1  = _summary(sub_q1["pnl_pct"].values)

                rows.append({
                    "side": side, "tier": tier, "threshold": thr,
                    "full_n": s_full["n"], "full_wr": s_full["wr"]*100,
                    "full_pf": s_full["pf"], "full_avg_win": s_full["avg_win"],
                    "full_avg_loss": s_full["avg_loss"],
                    "full_sum_pnl_pct": s_full["sum_pnl_pct"],
                    "oos_n": s_oos["n"], "oos_wr": s_oos["wr"]*100,
                    "oos_pf": s_oos["pf"], "oos_sum_pnl_pct": s_oos["sum_pnl_pct"],
                    "h2_n": s_h2["n"], "h2_pf": s_h2["pf"], "h2_sum_pnl_pct": s_h2["sum_pnl_pct"],
                    "q1_n": s_q1["n"], "q1_pf": s_q1["pf"], "q1_sum_pnl_pct": s_q1["sum_pnl_pct"],
                })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "rules_plus_bgm_sweep.csv", index=False)
    print(f"  saved: {RESULTS / 'rules_plus_bgm_sweep.csv'}")

    # Verdict per cell
    md = []
    md.append(f"# v_new_2 — Rules + BGM Verdict ({time.strftime('%Y-%m-%d')})\n\n")
    md.append("Layer 2 BGM grader applied as quality filter on rule-trigger trades. "
              "Sweep BGM_score threshold ∈ {0, 0.40, 0.45, 0.50, ..., 0.75}.\n\n")

    print(f"\n{_ts()} per-cell baseline → optimal-threshold lift comparison")
    print(f"  side      tier      thr=0        →  best-thr    n→n        PF→PF        OOS_PF       lift_PF")

    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            sub = df_out[(df_out["side"]==side) & (df_out["tier"]==tier)]
            base = sub[sub["threshold"]==0.0].iloc[0]
            # Pick best threshold by full_pf, requiring at least 50 trades full + 30 oos
            cand = sub[(sub["threshold"]>0.0) & (sub["full_n"]>=50) & (sub["oos_n"]>=30)]
            if len(cand) == 0:
                best = base
            else:
                best = cand.sort_values("full_pf", ascending=False).iloc[0]

            print(f"  {side:5s}  {tier:9s}  thr=0.00  n={int(base['full_n']):>5}  "
                  f"PF={base['full_pf']:.2f}  OOS_PF={base['oos_pf']:.2f}  "
                  f"→  thr={best['threshold']:.2f}  n={int(best['full_n']):>4}  "
                  f"PF={best['full_pf']:.2f}  OOS_PF={best['oos_pf']:.2f}  "
                  f"lift={best['full_pf']/max(base['full_pf'],0.01):.2f}×")

            md.append(f"## {side.upper()} {tier}\n\n")
            md.append("| threshold | n | WR | PF | avg_win | avg_loss | full_sum | OOS_n | OOS_PF | OOS_sum |\n")
            md.append("|---|---|---|---|---|---|---|---|---|---|\n")
            for _, r in sub.iterrows():
                md.append(f"| {r['threshold']:.2f} | {int(r['full_n'])} | {r['full_wr']:.1f}% | "
                          f"{r['full_pf']:.2f} | {r['full_avg_win']:+.1f}% | {r['full_avg_loss']:+.1f}% | "
                          f"{r['full_sum_pnl_pct']:+.0f}% | {int(r['oos_n'])} | {r['oos_pf']:.2f} | "
                          f"{r['oos_sum_pnl_pct']:+.0f}% |\n")
            md.append("\n")

    md.append("## Summary — best-threshold lift per cell\n\n")
    md.append("| side | tier | base PF | best thr | best PF | best n | OOS PF | lift |\n")
    md.append("|---|---|---|---|---|---|---|---|\n")
    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            sub = df_out[(df_out["side"]==side) & (df_out["tier"]==tier)]
            base = sub[sub["threshold"]==0.0].iloc[0]
            cand = sub[(sub["threshold"]>0.0) & (sub["full_n"]>=50) & (sub["oos_n"]>=30)]
            if len(cand) == 0:
                best = base
            else:
                best = cand.sort_values("full_pf", ascending=False).iloc[0]
            md.append(f"| {side} | {tier} | {base['full_pf']:.2f} | "
                      f"{best['threshold']:.2f} | {best['full_pf']:.2f} | "
                      f"{int(best['full_n'])} | {best['oos_pf']:.2f} | "
                      f"{best['full_pf']/max(base['full_pf'],0.01):.2f}× |\n")

    (RESULTS / "v_new_2_verdict.md").write_text("".join(md))
    print(f"\n  saved: {RESULTS / 'v_new_2_verdict.md'}")
    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
