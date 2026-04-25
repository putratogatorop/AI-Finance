"""Phase 6 — sensitivity + transfer analysis on the bigmover-combined backtest.

All three analyses are computed by re-aggregating the trades.csv from Phase 5
(no need to re-run the heavy backtest). Inputs already have per-trade
`proba`, `pos_scale`, `pnl_pct`, `direction`, `signal`, `period`, and
`btc_trend_score` columns.

Outputs:
  results/<run_id>/sensitivity_threshold.json
  results/<run_id>/sensitivity_sizing.json
  results/<run_id>/sensitivity_transfer.json

Run from services/python/:
    python scripts/sensitivity_bigmover_combined_v1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.sizing import SizingConfig, position_scale

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE5_RESULTS_DIR = (
    REPO_ROOT
    / "services/python/results"
    / "backtest_bigmover_combined_with_ml_18aa3e0b_20260425T165448Z"
)
OUT_DIR = PHASE5_RESULTS_DIR  # write sensitivity files alongside the source
TRADES_CSV = PHASE5_RESULTS_DIR / "trades.csv"

THRESHOLD_SWEEP = [0.45, 0.50, 0.55, 0.60]
SIZING_CAPS = [1.0, 1.5, 2.0]
SIZING_FLOORS = [0.0, 0.25]


def _pf(pnls: np.ndarray) -> float:
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return float("nan")
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses == 0:
        return float("inf")
    return float(wins / losses)


def _metrics(pnls: np.ndarray) -> dict:
    if len(pnls) == 0:
        return {"trades": 0, "wr": 0.0, "pf": float("nan"),
                "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0}
    return {
        "trades": int(len(pnls)),
        "wr": float((pnls > 0).mean()),
        "pf": _pf(pnls),
        "avg_pnl_pct": float(pnls.mean()),
        "total_pnl_pct": float(pnls.sum()),
    }


def _train_threshold(df: pd.DataFrame, q: float) -> float:
    train = df[(df["period"] == "train") & df["proba"].notna()]
    return float(train["proba"].quantile(q))


def main() -> None:
    started = time.monotonic()
    if not TRADES_CSV.exists():
        raise FileNotFoundError(f"Phase-5 trades.csv missing: {TRADES_CSV}")

    print(f"loading trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"  {len(df):,} rows", flush=True)

    # ---- 1. Threshold sweep -------------------------------------------------
    print("\n[1/3] threshold sweep ----------------------------------------")
    threshold_results = {}
    for q in [2/3, 0.5]:
        # Provided just for reference: q=2/3 was the continuation-predictor
        # default; q=0.5 was the bigmover-combined default.
        t = _train_threshold(df, q)
        threshold_results[f"q{q:.4f}"] = {
            "quantile": q,
            "threshold": t,
            "note": "diagnostic (different quantile)",
        }
    # Main sweep: lock thresholds at fixed values, evaluate OOT.
    for thr in THRESHOLD_SWEEP:
        oot_full = df[(df["period"] == "oot") & (df["proba"] >= thr) & (df["pos_scale"] > 0)]
        cell = _metrics(oot_full["sized_pnl"].to_numpy())
        # Also compute the train quantile that matches this threshold (informational).
        train_with_score = df[(df["period"] == "train") & df["proba"].notna()]
        eq_quantile = float((train_with_score["proba"] < thr).mean())
        threshold_results[f"thr_{thr:.2f}"] = {
            "threshold": thr,
            "equivalent_train_quantile": eq_quantile,
            "oot_full_metrics": cell,
        }
        print(
            f"  thr={thr:.2f}  (~q{eq_quantile:.3f} of train)  "
            f"trades={cell['trades']:>5}  PF={cell['pf']:.3f}  "
            f"WR={cell['wr']*100:.1f}%  avg={cell['avg_pnl_pct']*100:+.3f}%",
            flush=True,
        )
    (OUT_DIR / "sensitivity_threshold.json").write_text(
        json.dumps(threshold_results, indent=2, default=str)
    )

    # ---- 2. Sizing sensitivity ---------------------------------------------
    print("\n[2/3] sizing-knob sensitivity (cap x floor) ------------------")
    sizing_results = {}
    locked_threshold = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    print(f"  locked threshold from train (q=0.5): {locked_threshold:.4f}", flush=True)
    for cap in SIZING_CAPS:
        for floor in SIZING_FLOORS:
            cfg = SizingConfig(slope=1.5, intercept=0.0, floor=floor, cap=cap)
            # Re-derive scale per-trade with the new config.
            new_scale = df.apply(
                lambda r: position_scale(
                    r["btc_trend_score"], r["direction"], config=cfg
                ),
                axis=1,
            )
            new_sized_pnl = df["pnl_pct"] * new_scale
            mask = (
                (df["period"] == "oot")
                & (df["proba"] >= locked_threshold)
                & (new_scale > 0)
            )
            cell = _metrics(new_sized_pnl[mask].to_numpy())
            key = f"cap_{cap:.1f}_floor_{floor:.2f}"
            sizing_results[key] = {
                "cap": cap,
                "floor": floor,
                "oot_full_metrics": cell,
            }
            print(
                f"  {key:<22}  trades={cell['trades']:>5}  "
                f"PF={cell['pf']:.3f}  WR={cell['wr']*100:.1f}%  "
                f"avg={cell['avg_pnl_pct']*100:+.3f}%",
                flush=True,
            )
    (OUT_DIR / "sensitivity_sizing.json").write_text(
        json.dumps(sizing_results, indent=2, default=str)
    )

    # ---- 3. Transfer test (per-combo OOT) -----------------------------------
    print("\n[3/3] transfer test (per (signal, direction) combo OOT) ------")
    transfer_results = {}
    oot = df[df["period"] == "oot"].copy()
    full_mask = (oot["proba"] >= locked_threshold) & (oot["pos_scale"] > 0)
    for signal in ("baseline", "price_accel_atr", "multi_bar_confirm"):
        for direction in ("short", "long"):
            sub = oot[(oot["signal"] == signal) & (oot["direction"] == direction)]
            sub_full = sub[full_mask.loc[sub.index]]
            equal_metrics = _metrics(sub["pnl_pct"].to_numpy())          # baseline (no ML/sizing)
            full_metrics = _metrics(sub_full["sized_pnl"].to_numpy())    # ml_on + sized
            key = f"{signal}__{direction}"
            transfer_results[key] = {
                "signal": signal,
                "direction": direction,
                "oot_baseline_metrics": equal_metrics,
                "oot_full_metrics": full_metrics,
            }
            print(
                f"  {key:<32}  baseline PF={equal_metrics['pf']:.3f} "
                f"(n={equal_metrics['trades']:>5}) -> "
                f"full PF={full_metrics['pf']:.3f} (n={full_metrics['trades']:>4})",
                flush=True,
            )
    (OUT_DIR / "sensitivity_transfer.json").write_text(
        json.dumps(transfer_results, indent=2, default=str)
    )

    print(f"\nwrote sensitivity files to {OUT_DIR}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
