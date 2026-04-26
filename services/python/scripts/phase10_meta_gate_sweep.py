"""Phase 10.1 — hand-crafted BTC-momentum meta-regime gate sweep.

Diagnosis from Phase 8 + Phase 9: catastrophic months for SHORT trades
are concentrated in BTC bull rallies (2023-Q4, 2024-Q1, 2024-Q4, 2025-Q3).
The simplest hand-crafted gate is "block shorts when BTC's recent return
is too strong" — i.e., `btc_90d_return > T` blocks the trade.

This script is a pure re-aggregation on the existing v2 trades.csv (no
re-simulation, no re-training). Lock the threshold on TRAIN total_pnl,
evaluate on OOT.

Run from services/python/:
    python scripts/phase10_meta_gate_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
V2_RUN_DIR = (
    REPO_ROOT / "services/python/results"
    / "backtest_bigmover_combined_v2_outcome_0582b799_20260426T033736Z"
)
TRADES_CSV = V2_RUN_DIR / "trades.csv"
OUT_PATH = V2_RUN_DIR / "phase10_meta_gate_sweep.json"

# --- Knobs ------------------------------------------------------------------

THRESHOLDS = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]   # btc_90d_return upper bound
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3


# --- Helpers ----------------------------------------------------------------


def _pf(pnls: np.ndarray) -> float:
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _cell_metrics(pnls: np.ndarray) -> dict:
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


def _walkforward(df: pd.DataFrame, pnl_col: str = "sized_pnl") -> dict:
    from dateutil.relativedelta import relativedelta
    if len(df) == 0:
        return {"n_folds": 0, "folds": [], "rollup_pf": {}}
    et = pd.to_datetime(df["entry_time"], utc=True)
    start = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    month = relativedelta(months=1)
    fold_w = relativedelta(months=WF_FOLD_MONTHS)
    folds = []
    s = start
    while s + fold_w <= end_global:
        e = s + fold_w
        sub = df[(et >= s) & (et < e)]
        pnls = sub[pnl_col].to_numpy(dtype=float)
        folds.append({
            "fold_start": str(s.date()),
            "fold_end": str(e.date()),
            "trades": int(len(sub)),
            "wr": float((pnls > 0).mean()) if len(pnls) else 0.0,
            "pf": _pf(pnls),
            "total_pnl_pct": float(pnls.sum()),
        })
        s += month
    pfs = np.array([
        f["pf"] for f in folds if f["trades"] >= 30 and np.isfinite(f["pf"])
    ])
    if len(pfs):
        rollup = {
            "n_folds_used": int(len(pfs)),
            "p5": float(np.percentile(pfs, 5)),
            "p25": float(np.percentile(pfs, 25)),
            "p50": float(np.percentile(pfs, 50)),
            "p75": float(np.percentile(pfs, 75)),
            "p95": float(np.percentile(pfs, 95)),
            "p_lt_1": float((pfs < 1.0).mean()),
            "p_gte_130": float((pfs >= 1.30).mean()),
        }
    else:
        rollup = {"n_folds_used": 0}
    return {"n_folds": len(folds), "folds": folds, "rollup_pf": rollup}


def _monte_carlo(pnls: np.ndarray, n: int = MC_N_RESAMPLES) -> dict:
    if len(pnls) == 0:
        return {"n_resamples": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    pfs = np.empty(n, dtype=float)
    for k in range(n):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        pfs[k] = _pf(sample)
    finite = pfs[np.isfinite(pfs)]
    return {
        "n_resamples": int(n),
        "p5": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
        "p50": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
        "p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
        "point_pf": _pf(pnls),
        "p_pf_lt_1": float((finite < 1.0).mean()) if len(finite) else float("nan"),
    }


# --- Main ------------------------------------------------------------------


def main() -> None:
    started = time.monotonic()

    print(f"loading v2 trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"  {len(df):,} rows", flush=True)

    # ---- Restrict to SHORT-only ml_on+sized (the deployable subset) ------
    threshold_predictor = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    short_full = df[
        (df["direction"] == "short")
        & (df["proba"] >= threshold_predictor)
        & (df["pos_scale"] > 0)
    ].copy()
    print(f"  SHORT ml_on+sized entries: {len(short_full):,}", flush=True)

    # ---- Compute btc_90d_return at each entry_time -----------------------
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
    # 90-day return; shift by 1 day to ensure no-lookahead at the boundary
    # (the daily bar at date `d` becomes available only AT `d`, so a 15m
    # entry at `d` 09:00 UTC must use the previous day's value).
    btc_90d_ret = btc_daily.pct_change(90).shift(1)
    # Forward-fill onto the 15m grid for per-trade lookup.
    btc_90d_ret_15m = btc_90d_ret.reindex(btc.index, method="ffill")

    # Per-trade lookup
    short_full["btc_90d_return"] = btc_90d_ret_15m.reindex(
        short_full["entry_time"]
    ).to_numpy()
    n_with_score = int(short_full["btc_90d_return"].notna().sum())
    print(f"  short trades with btc_90d_return: {n_with_score} / {len(short_full)}",
          flush=True)
    short_full = short_full[short_full["btc_90d_return"].notna()].reset_index(drop=True)

    # ---- Sweep -------------------------------------------------------------
    print(f"\n=== Sweeping {len(THRESHOLDS)} thresholds on btc_90d_return ===")
    sweep_results = {}
    for T in THRESHOLDS:
        gated = short_full[short_full["btc_90d_return"] <= T]
        train = gated[gated["period"] == "train"]
        oot = gated[gated["period"] == "oot"]
        train_pnls = train["sized_pnl"].to_numpy()
        oot_pnls = oot["sized_pnl"].to_numpy()
        wf = _walkforward(gated, pnl_col="sized_pnl")
        mc = _monte_carlo(oot_pnls)

        key = f"T_{T:+.2f}"
        sweep_results[key] = {
            "threshold": T,
            "train_metrics": _cell_metrics(train_pnls),
            "oot_metrics": _cell_metrics(oot_pnls),
            "walkforward": wf["rollup_pf"],
            "monte_carlo_oot": mc,
        }
        tm = sweep_results[key]["train_metrics"]
        om = sweep_results[key]["oot_metrics"]
        wf_rp = wf["rollup_pf"]
        wf_p5 = wf_rp.get("p5", float("nan"))
        mc_p5 = mc.get("p5", float("nan"))
        print(
            f"  T={T:+.2f}  TRAIN n={tm['trades']:>5} PF={tm['pf']:.3f} "
            f"total_pnl={tm['total_pnl_pct']*100:+.0f}%   "
            f"OOT n={om['trades']:>4} PF={om['pf']:.3f}   "
            f"WF p5={wf_p5:.3f}   MC p5={mc_p5:.3f}",
            flush=True,
        )

    # ---- Lock winner on TRAIN total_pnl_pct ------------------------------
    locked_name = max(
        sweep_results,
        key=lambda k: (
            sweep_results[k]["train_metrics"]["total_pnl_pct"]
            if np.isfinite(sweep_results[k]["train_metrics"]["pf"]) else -np.inf
        ),
    )
    locked = sweep_results[locked_name]
    print(f"\nLOCKED on TRAIN total_pnl: {locked_name}  (T={locked['threshold']:+.2f})", flush=True)

    # Train-vs-OOT PF gap (overfit warning)
    gap = locked["oot_metrics"]["pf"] - locked["train_metrics"]["pf"]
    print(f"  train PF: {locked['train_metrics']['pf']:.3f}", flush=True)
    print(f"  oot   PF: {locked['oot_metrics']['pf']:.3f}  (delta = {gap:+.3f})", flush=True)
    print(f"  WF p5:    {locked['walkforward']['p5']:.3f}", flush=True)
    print(f"  MC p5:    {locked['monte_carlo_oot']['p5']:.3f}", flush=True)

    # ---- Per-month rescue: catastrophic months from earlier analysis -----
    catastrophic = ["2023-09", "2023-10", "2023-11", "2023-12", "2024-02",
                    "2024-05", "2024-11", "2025-08"]
    short_full["month"] = (
        short_full["entry_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
    )
    rescue = []
    print("\n=== Catastrophic-month rescue at locked threshold ===")
    print(f"  {'month':<10} {'no-gate n':>10} {'no-gate PF':>11} {'gated n':>9} {'gated PF':>10}")
    for m in catastrophic:
        in_month = short_full["month"] == m
        ng = short_full[in_month]
        ga = short_full[in_month & (short_full["btc_90d_return"] <= locked["threshold"])]
        rescue.append({
            "month": m,
            "no_gate_n": int(len(ng)), "no_gate_pf": _pf(ng["sized_pnl"].to_numpy()),
            "gated_n": int(len(ga)), "gated_pf": _pf(ga["sized_pnl"].to_numpy()),
        })
        print(f"  {m:<10} {len(ng):>10} {_pf(ng['sized_pnl'].to_numpy()):>11.3f} "
              f"{len(ga):>9} {_pf(ga['sized_pnl'].to_numpy()):>10.3f}", flush=True)

    # ---- Ship gate at locked threshold ------------------------------------
    oot = locked["oot_metrics"]
    oot_pf_ok = bool(np.isfinite(oot["pf"]) and oot["pf"] >= 1.30)
    oot_n_ok = oot["trades"] >= 200
    wf_p5_ok = bool(locked["walkforward"].get("p5", -1) > 1.0)
    mc_p5_ok = bool(np.isfinite(locked["monte_carlo_oot"].get("p5", float("nan")))
                     and locked["monte_carlo_oot"].get("p5", -1) > 1.0)
    ship_gate = {
        "oot_pf_ge_130": oot_pf_ok,
        "oot_n_ge_200": oot_n_ok,
        "walkforward_p5_gt_1": wf_p5_ok,
        "mc_p5_gt_1": mc_p5_ok,
        "overall": oot_pf_ok and oot_n_ok and wf_p5_ok and mc_p5_ok,
    }
    print("\n=== Ship gate (SHORT + ml + sizing + meta-gate at locked T) ===")
    for k, v in ship_gate.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  → overall: {'SHIP' if ship_gate['overall'] else 'DO NOT SHIP YET'}")

    # ---- Save -------------------------------------------------------------
    out = {
        "thresholds": THRESHOLDS,
        "sweep_results": sweep_results,
        "locked": {
            "name": locked_name,
            "threshold": locked["threshold"],
            "train_pf": locked["train_metrics"]["pf"],
            "oot_pf": locked["oot_metrics"]["pf"],
            "wf_p5": locked["walkforward"].get("p5"),
            "mc_p5": locked["monte_carlo_oot"].get("p5"),
            "ship_gate": ship_gate,
        },
        "catastrophic_month_rescue": rescue,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
