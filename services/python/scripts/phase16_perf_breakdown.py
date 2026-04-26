"""Walk through one trade + month-by-month equity curve under the locked Phase 16
configuration (D1 detector + sign-locked sizing + v3 classifier + rapid-rally exit).

Two views:
  1. ANATOMY — one representative OOT trade (from signal to close).
  2. MONTH BY MONTH — realistic equity curve respecting:
       - 5% equity per trade × 5x leverage = 25% notional per trade (CLAUDE.md)
       - MAX_CONCURRENT_POSITIONS = 5 (CLAUDE.md)
       - position scale from BTC trend score (already in sized_pnl)
       - compounding: equity_t+1 = equity_t × (1 + 0.25 × sized_pnl)

Run from services/python/:
    python scripts/phase16_perf_breakdown.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES

SERVICES_PY = Path(__file__).resolve().parents[1]
TRADES_PATH = SERVICES_PY / "data" / "d1_short_trades_with_features.csv"
MODEL_PATH = SERVICES_PY / "models" / "d1_short_v3.joblib"
META_PATH = SERVICES_PY / "models" / "d1_short_v3_meta.json"

NOTIONAL_PCT = 0.25     # 5% equity × 5x leverage per CLAUDE.md
MAX_CONCURRENT = 5      # CLAUDE.md MAX_CONCURRENT_POSITIONS


def _pf(p):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _simulate_equity(df: pd.DataFrame) -> pd.DataFrame:
    """Walk trades in time, enforce MAX_CONCURRENT cap, compound NOTIONAL_PCT × sized_pnl."""
    df = df.sort_values("entry_time").reset_index(drop=True).copy()
    # Exit timestamp = entry + 15m × bars_held
    df["exit_time"] = df["entry_time"] + pd.to_timedelta(df["bars"] * 15, unit="m")

    equity = 1.0
    open_exits: list[pd.Timestamp] = []
    rows = []
    for _, t in df.iterrows():
        # close any positions that exited at or before this entry_time
        open_exits = [x for x in open_exits if x > t["entry_time"]]
        accepted = len(open_exits) < MAX_CONCURRENT
        if accepted:
            r = float(t["sized_pnl"]) * NOTIONAL_PCT
            equity *= (1.0 + r)
            open_exits.append(t["exit_time"])
            rows.append({
                "entry_time": t["entry_time"],
                "asset": t["asset"],
                "sized_pnl": float(t["sized_pnl"]),
                "equity_delta": r,
                "equity": equity,
                "accepted": True,
            })
        else:
            rows.append({
                "entry_time": t["entry_time"],
                "asset": t["asset"],
                "sized_pnl": float(t["sized_pnl"]),
                "equity_delta": 0.0,
                "equity": equity,
                "accepted": False,
            })
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])

    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]
    gated = df[df["score"] >= threshold].reset_index(drop=True)

    # === 1. Anatomy of one trade ===========================================
    print("=" * 78)
    print("ONE TRADE — anatomy of a SHORT entry under the locked config")
    print("=" * 78)
    oot = gated[gated["entry_time"] >= pd.Timestamp("2026-01-01", tz="UTC")].copy()
    winners = oot[oot["pnl_pct"] > 0].sort_values("pnl_pct").reset_index(drop=True)
    ex = winners.iloc[len(winners) // 2] if len(winners) > 0 else oot.iloc[0]
    print(f"  asset                : {ex['asset']}")
    print(f"  entry_time (UTC)     : {ex['entry_time']}")
    print(f"  classifier score     : {ex['score']:.4f}  (threshold {threshold:.4f})")
    print(f"  BTC trend score      : {ex['btc_score']:+.4f}  (must be < 0 for SHORT)")
    print(f"  pos_scale            : {ex['pos_scale']:.3f}x  (= min(1.5 × |btc_score|, 1.5))")
    print()
    print("  D1 detector trigger conditions @ entry bar:")
    print(f"    drop ≥ 2.5 × ATR   : ✓  (rolling-96-high to close, in ATR_14 units)")
    print(f"    vol_ratio          : {ex['vol_ratio']:>6.2f}   (gate ≥ 2.0)")
    print(f"    bar colour         : red (close < open) — required")
    print()
    print("  Classifier features (selected from 16):")
    show = ["price_vs_ema9_pct", "price_vs_ema50_pct",
            "rsi14_delta_1bar", "rsi14_delta_4bar",
            "macd_signal_spread_norm", "macd_hist_momentum",
            "kdj_j", "kdj_k",
            "btc_trend_score", "btc_realized_vol_24h", "concurrent_dir_breadth"]
    for k in show:
        print(f"    {k:<28} {ex[k]:>10.4f}")
    print()
    print("  Exit:")
    print(f"    exit_kind          : {ex['exit']}  (sl=−5% / tp=+15% / timeout=672 bars / "
          f"rapid-rally if BTC up >3% in 24h)")
    print(f"    bars held          : {int(ex['bars'])}  ({int(ex['bars'])*15/60:.1f}h)")
    print(f"    raw pnl_pct        : {ex['pnl_pct']*100:+.2f}%  (per-trade % of position)")
    print(f"    sized_pnl          : {ex['sized_pnl']*100:+.2f}%  (= raw × pos_scale)")
    print(f"    equity move @ "
          f"5%×5x sizing : {ex['sized_pnl']*NOTIONAL_PCT*100:+.3f}%  (= sized × 25% notional)")
    print()

    # === 2. Realistic equity simulation =====================================
    print("=" * 78)
    print(f"REALISTIC EQUITY — 5% equity × 5x lev per trade, max {MAX_CONCURRENT} concurrent")
    print(f"  per CLAUDE.md sizing rules; compounding; no extra leverage on the position cap")
    print("=" * 78)

    sim = _simulate_equity(gated)
    accepted = sim[sim["accepted"]].copy()
    rejected = sim[~sim["accepted"]]
    print(f"  signals total: {len(sim):,}  accepted: {len(accepted):,}  "
          f"dropped on cap: {len(rejected):,} ({len(rejected)/max(len(sim),1)*100:.1f}%)")

    accepted["month"] = accepted["entry_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    monthly = accepted.groupby("month").agg(
        n=("equity_delta", "size"),
        wins=("sized_pnl", lambda s: int((s > 0).sum())),
        avg_per_trade=("sized_pnl", "mean"),
        end_equity=("equity", "last"),
    )
    # monthly return = (end_equity_this_month / end_equity_prev_month) - 1
    end_eq = monthly["end_equity"].values
    start_eq = np.concatenate([[1.0], end_eq[:-1]])
    monthly["monthly_ret"] = (end_eq / start_eq) - 1.0
    monthly["wr"] = monthly["wins"] / monthly["n"]

    def _print_table(d, label):
        print(f"\n{label}")
        print(f"  {'month':<10} {'n':>5} {'wr':>6} {'avg/trade':>10} {'month %':>9} {'cum equity':>11}")
        for m, r in d.iterrows():
            print(f"  {m:<10} {int(r['n']):>5d} {r['wr']*100:>5.1f}% "
                  f"{r['avg_per_trade']*100:>+8.2f}% {r['monthly_ret']*100:>+7.2f}% "
                  f"  {r['end_equity']:>9.3f}x")

    oot_idx = monthly.index >= "2026-01"
    _print_table(monthly[~oot_idx], "TRAIN (2023-04 → 2025-12) — informational, not the ship gate")
    _print_table(monthly[oot_idx], "OOT (2026-01 → 2026-04) — the strict 8-point ship-gate hold-out")

    # Quick stats
    oot_returns = monthly.loc[oot_idx, "monthly_ret"]
    print(f"\n  OOT 3-month total: {(monthly.loc[oot_idx, 'end_equity'].iloc[-1] / monthly.loc[oot_idx, 'end_equity'].iloc[0] * (1 / (1 + oot_returns.iloc[0])) ) :.3f}x"
          if len(oot_returns) > 0 else "  no OOT trades")
    if len(oot_returns) > 0:
        oot_first_eq = monthly.loc[oot_idx, "end_equity"].iloc[0] / (1 + oot_returns.iloc[0])
        oot_last_eq = monthly.loc[oot_idx, "end_equity"].iloc[-1]
        print(f"  OOT best month  : {oot_returns.max()*100:+.2f}%   worst: {oot_returns.min()*100:+.2f}%   "
              f"median: {oot_returns.median()*100:+.2f}%")
    full_returns = monthly["monthly_ret"]
    print(f"  Full 3yr months: {len(full_returns)}  "
          f"best {full_returns.max()*100:+.1f}%   worst {full_returns.min()*100:+.1f}%   "
          f"median {full_returns.median()*100:+.2f}%   "
          f"P(loss) {(full_returns < 0).mean()*100:.1f}%")
    n_negative = int((full_returns < 0).sum())
    print(f"  {n_negative}/{len(full_returns)} months were negative.")


if __name__ == "__main__":
    main()
