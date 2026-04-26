"""Phase 11b — clean momentum + RSI combos (search for best deployable config).

Phase 11 found that BTC daily RSI(14) < 20 is the cleanest single whipsaw
guard (R1: OOT PF 1.596, MC p5 1.504). %B and volume-z guards over-trigger
on the OOT period and HURT performance.

This script sweeps the cleanest 2-feature combinations of:
  - btc_90d_return <= T  (Phase 10.1's bull-rally gate)
  - btc_rsi14_daily >= R (whipsaw guard)

Locks the winner. Expected outcome: a hand-crafted rule that strictly
dominates v2 baseline on 3 of 4 ship gates.

Run from services/python/:
    python scripts/phase11b_clean_combos.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.indicators import rsi as _rsi

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
V2_RUN_DIR = (
    REPO_ROOT / "services/python/results"
    / "backtest_bigmover_combined_v2_outcome_0582b799_20260426T033736Z"
)
TRADES_CSV = V2_RUN_DIR / "trades.csv"
OUT_PATH = V2_RUN_DIR / "phase11b_clean_combos.json"

RANDOM_SEED = 42
WF_FOLD_MONTHS = 3
MC_N_RESAMPLES = 5000

# Sweep grid
MOMENTUM_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]   # btc_90d_return upper bound
RSI_FLOORS = [None, 20, 25, 30]                  # None = no RSI guard


# --- Helpers ----------------------------------------------------------------


def _pf(pnls: np.ndarray) -> float:
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _walkforward(df: pd.DataFrame, pnl_col: str = "sized_pnl") -> dict:
    if len(df) == 0:
        return {"n_folds_used": 0}
    et = pd.to_datetime(df["entry_time"], utc=True)
    s = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    fold_w = relativedelta(months=WF_FOLD_MONTHS)
    month = relativedelta(months=1)
    pfs = []
    s_ = s
    while s_ + fold_w <= end_global:
        sub = df[(et >= s_) & (et < s_ + fold_w)]
        if len(sub) >= 30:
            p = _pf(sub[pnl_col].to_numpy())
            if np.isfinite(p):
                pfs.append(p)
        s_ += month
    pfs = np.array(pfs)
    if len(pfs) == 0:
        return {"n_folds_used": 0}
    return {
        "n_folds_used": int(len(pfs)),
        "p5": float(np.percentile(pfs, 5)),
        "p25": float(np.percentile(pfs, 25)),
        "p50": float(np.percentile(pfs, 50)),
        "p75": float(np.percentile(pfs, 75)),
        "p95": float(np.percentile(pfs, 95)),
        "p_lt_1": float((pfs < 1.0).mean()),
    }


def _monte_carlo(pnls: np.ndarray, n: int = MC_N_RESAMPLES) -> dict:
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {"n": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    pfs = np.array([_pf(rng.choice(pnls, len(pnls), replace=True)) for _ in range(n)])
    finite = pfs[np.isfinite(pfs)]
    return {
        "n": int(len(pnls)),
        "point_pf": _pf(pnls),
        "p5": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
        "p50": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
        "p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
    }


def _cell(pnls: np.ndarray) -> dict:
    pnls = np.asarray(pnls, dtype=float)
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


# --- Main -------------------------------------------------------------------


def main() -> None:
    started = time.monotonic()

    print(f"loading v2 trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    threshold_predictor = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    short = df[
        (df["direction"] == "short")
        & (df["proba"] >= threshold_predictor)
        & (df["pos_scale"] > 0)
    ].copy()
    print(f"  SHORT ml_on+sized entries: {len(short):,}", flush=True)

    print(f"\nloading BTC candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
        .astype(float)
    )
    btc_daily = btc.resample("1D", label="right", closed="right").last().dropna()
    btc_90d = btc_daily.pct_change(90).shift(1).reindex(btc.index, method="ffill")
    btc_rsi14 = _rsi(btc_daily, 14).shift(1).reindex(btc.index, method="ffill")

    short["btc_90d_ret"] = btc_90d.reindex(short["entry_time"]).to_numpy()
    short["btc_rsi14"] = btc_rsi14.reindex(short["entry_time"]).to_numpy()
    short = short[short["btc_90d_ret"].notna() & short["btc_rsi14"].notna()].reset_index(drop=True)
    print(f"  trades with both BTC features: {len(short):,}", flush=True)

    print(f"\n=== Sweep: {len(MOMENTUM_THRESHOLDS)} momentum × {len(RSI_FLOORS)} RSI = "
          f"{len(MOMENTUM_THRESHOLDS) * len(RSI_FLOORS)} combos ===")
    print(f"  {'config':<35}  {'TRAIN n':>7} {'TRAIN PF':>9} {'TRAIN tot%':>11}  "
          f"{'OOT n':>5} {'OOT PF':>7}  {'WF p5':>6} {'WF p50':>7} {'P(loss)':>8}  {'MC p5':>6}",
          flush=True)
    sweep = {}
    for T in MOMENTUM_THRESHOLDS:
        for R in RSI_FLOORS:
            mask = short["btc_90d_ret"] <= T
            if R is not None:
                mask = mask & (short["btc_rsi14"] >= R)
            sub = short[mask]
            train_pnls = sub[sub["period"] == "train"]["sized_pnl"].to_numpy()
            oot_pnls = sub[sub["period"] == "oot"]["sized_pnl"].to_numpy()
            wf = _walkforward(sub)
            mc = _monte_carlo(oot_pnls)
            train_m = _cell(train_pnls)
            oot_m = _cell(oot_pnls)
            key = f"momentum_le_{T:.2f}_rsi_ge_{R}" if R is not None else f"momentum_le_{T:.2f}_no_rsi"
            sweep[key] = {
                "momentum_threshold": T,
                "rsi_floor": R,
                "train_metrics": train_m,
                "oot_metrics": oot_m,
                "walkforward": wf,
                "monte_carlo_oot": mc,
            }
            print(f"  {key:<35}  {train_m['trades']:>7} {train_m['pf']:>9.3f} "
                  f"{train_m['total_pnl_pct']*100:>+10.0f}%  "
                  f"{oot_m['trades']:>5} {oot_m['pf']:>7.3f}  "
                  f"{wf.get('p5', float('nan')):>6.3f} {wf.get('p50', float('nan')):>7.3f} "
                  f"{wf.get('p_lt_1', float('nan'))*100:>7.1f}%  "
                  f"{mc.get('p5', float('nan')):>6.3f}",
                  flush=True)

    # Lock winner on TRAIN total_pnl
    locked_name = max(
        sweep,
        key=lambda k: (
            sweep[k]["train_metrics"]["total_pnl_pct"]
            if np.isfinite(sweep[k]["train_metrics"]["pf"]) else -np.inf
        ),
    )
    locked = sweep[locked_name]
    print(f"\nLOCKED on TRAIN total_pnl: {locked_name}", flush=True)
    print(f"  momentum_threshold: {locked['momentum_threshold']:+.2f}", flush=True)
    print(f"  rsi_floor:          {locked['rsi_floor']}", flush=True)
    print(f"  TRAIN: n={locked['train_metrics']['trades']:>5}  PF={locked['train_metrics']['pf']:.3f}  "
          f"total={locked['train_metrics']['total_pnl_pct']*100:+.0f}%", flush=True)
    print(f"  OOT:   n={locked['oot_metrics']['trades']:>5}  PF={locked['oot_metrics']['pf']:.3f}",
          flush=True)
    print(f"  WF p5={locked['walkforward'].get('p5', float('nan')):.3f}  "
          f"p50={locked['walkforward'].get('p50', float('nan')):.3f}  "
          f"P(loss)={locked['walkforward'].get('p_lt_1', float('nan'))*100:.1f}%",
          flush=True)
    print(f"  MC p5={locked['monte_carlo_oot'].get('p5', float('nan')):.3f}", flush=True)

    # Ship gate at locked
    oot_pf = locked["oot_metrics"]["pf"]
    ship_gate = {
        "oot_pf_ge_130": bool(np.isfinite(oot_pf) and oot_pf >= 1.30),
        "oot_n_ge_200": bool(locked["oot_metrics"]["trades"] >= 200),
        "wf_p5_gt_1": bool(locked["walkforward"].get("p5", -1) > 1.0),
        "mc_p5_gt_1": bool(np.isfinite(locked["monte_carlo_oot"].get("p5", float("nan")))
                            and locked["monte_carlo_oot"].get("p5", -1) > 1.0),
    }
    ship_gate["overall"] = all(ship_gate.values())
    print("\n=== Ship gate at locked combo ===")
    for k, v in ship_gate.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  → {'SHIP' if ship_gate['overall'] else 'DO NOT SHIP'}")

    out = {
        "momentum_thresholds": MOMENTUM_THRESHOLDS,
        "rsi_floors": [r for r in RSI_FLOORS],
        "sweep_results": sweep,
        "locked": {
            "name": locked_name,
            "momentum_threshold": locked["momentum_threshold"],
            "rsi_floor": locked["rsi_floor"],
            "train_pf": locked["train_metrics"]["pf"],
            "oot_pf": locked["oot_metrics"]["pf"],
            "wf_p5": locked["walkforward"].get("p5"),
            "wf_p50": locked["walkforward"].get("p50"),
            "mc_p5": locked["monte_carlo_oot"].get("p5"),
            "ship_gate": ship_gate,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
