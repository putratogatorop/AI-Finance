"""Phase 8.1 — weekly-trend meta-gate evaluation (re-aggregation).

Hypothesis: blocking SHORT signals when BTC's WEEKLY close is above its
26-period weekly EMA will kill the catastrophic months identified in the
by-month analysis (2023-Q4 to 2024-02 rally, 2024-04 pump, 2024-11 post-
election rally to $100K, 2025-08 mid-cycle rally) and lift walk-forward p5
above 1.0.

This script is a re-aggregation only — it loads the Phase-5 trades.csv and
the snapshot's BTC closes, builds a per-trade `btc_weekly_below_ema26`
flag, and recomputes the SHORT-only ship-gate metrics with vs without the
gate. No re-simulation, no re-training.

Run from services/python/:
    python scripts/phase8_weekly_metagate_eval.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.indicators import ema as _ema

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
PHASE5_RUN_DIR = (
    REPO_ROOT / "services/python/results"
    / "backtest_bigmover_combined_with_ml_18aa3e0b_20260425T165448Z"
)
TRADES_CSV = PHASE5_RUN_DIR / "trades.csv"
OUT_PATH = PHASE5_RUN_DIR / "phase8_weekly_metagate.json"

# --- Knobs (locked) ---------------------------------------------------------

WEEKLY_EMA_PERIOD = 26
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3


# --- Helpers ---------------------------------------------------------------


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


# --- Main pipeline ----------------------------------------------------------


def main() -> None:
    started = time.monotonic()

    print(f"loading trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"  {len(df):,} rows", flush=True)

    print(f"loading BTC candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
        .astype(float)
    )
    print(f"  {len(btc):,} BTC 15m bars, "
          f"{btc.index.min()} → {btc.index.max()}", flush=True)

    # ---- Build BTC weekly close + EMA-26, shifted by 1 week (no boundary leak) ----
    btc_weekly = btc.resample("1W", label="right", closed="right").last()
    weekly_ema = _ema(btc_weekly, WEEKLY_EMA_PERIOD)
    # Below = bear-friendly for shorts. Shift by 1 weekly bar so each 15m
    # timestamp uses the LAST CLOSED weekly bar — strict no-lookahead at the
    # boundary.
    weekly_below_ema = (btc_weekly < weekly_ema).shift(1)
    # Forward-fill onto the 15m grid so per-trade lookup is O(1).
    weekly_below_ema_15m = weekly_below_ema.reindex(btc.index, method="ffill").fillna(False)

    # Per-trade lookup: at each entry_time, was BTC weekly below its EMA-26?
    df["btc_weekly_below_ema26"] = (
        weekly_below_ema_15m.reindex(df["entry_time"]).to_numpy()
    )
    # NaN entries — typically warm-up at the start of the snapshot — treat as
    # "gate not satisfied" (conservative).
    df["btc_weekly_below_ema26"] = df["btc_weekly_below_ema26"].fillna(False).astype(bool)

    short = df[df["direction"] == "short"].copy()
    print(f"\nshort trades total: {len(short):,}", flush=True)
    print(f"  passing weekly gate (btc_weekly < ema26): {int(short['btc_weekly_below_ema26'].sum()):,}",
          flush=True)
    print(f"  rejected by weekly gate: {int((~short['btc_weekly_below_ema26']).sum()):,}",
          flush=True)

    # ---- Compare full strategy vs full+meta-gate -------------------------
    threshold = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    print(f"\nlocked predictor threshold (q0.5 of train scores): {threshold:.4f}",
          flush=True)

    full_mask_no_gate = (short["proba"] >= threshold) & (short["pos_scale"] > 0)
    full_mask_gated = full_mask_no_gate & short["btc_weekly_below_ema26"]

    print("\n=== 4-cell metrics: short, ml_on + sized, gate off vs on ===")
    cells = {}
    for period in ("train", "oot"):
        for gate in ("off", "on"):
            mask = short["period"] == period
            mask = mask & (full_mask_gated if gate == "on" else full_mask_no_gate)
            pnls = short.loc[mask, "sized_pnl"].to_numpy()
            cells[f"{period}_gate_{gate}"] = _cell_metrics(pnls)
            c = cells[f"{period}_gate_{gate}"]
            print(f"  {period}_gate_{gate:<3}  trades={c['trades']:>5}  "
                  f"PF={c['pf']:.3f}  WR={c['wr']*100:.1f}%  "
                  f"avg={c['avg_pnl_pct']*100:+.3f}%", flush=True)

    # ---- Walk-forward + Monte Carlo on the GATED short pool --------------
    gated_short = short[full_mask_gated].copy()
    print("\n=== Walk-forward (short + meta-gate, 34 rolling 3-month folds) ===")
    wf = _walkforward(gated_short, pnl_col="sized_pnl")
    if wf["rollup_pf"].get("n_folds_used", 0) > 0:
        rp = wf["rollup_pf"]
        print(f"  n_folds_used={rp['n_folds_used']}  p5={rp['p5']:.3f}  "
              f"p25={rp['p25']:.3f}  p50={rp['p50']:.3f}  p75={rp['p75']:.3f}  "
              f"p95={rp['p95']:.3f}  P(loss)={rp['p_lt_1']*100:.1f}%",
              flush=True)

    print("\n=== Monte Carlo (OOT short + meta-gate, 5000 resamples) ===")
    oot_pnls = short[(short["period"] == "oot") & full_mask_gated]["sized_pnl"].to_numpy()
    mc = _monte_carlo(oot_pnls)
    if mc.get("n_resamples", 0) > 0:
        print(f"  n={len(oot_pnls)}  point PF={mc['point_pf']:.3f}  "
              f"p5={mc['p5']:.3f}  p50={mc['p50']:.3f}  p95={mc['p95']:.3f}",
              flush=True)

    # ---- Catastrophic-month rescue table ---------------------------------
    catastrophic = ["2023-09", "2023-10", "2023-11", "2023-12", "2024-02",
                    "2024-05", "2024-11", "2025-08"]
    print("\n=== Catastrophic-month rescue (short, ml_on+sized, monthly PF) ===")
    print(f"  {'month':<10} {'no-gate n':>10} {'no-gate PF':>11} "
          f"{'gated n':>9} {'gated PF':>10}")
    rescue_rows = []
    short["month"] = short["entry_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
    for m in catastrophic:
        in_month = short["month"] == m
        ng = short[in_month & full_mask_no_gate]["sized_pnl"].to_numpy()
        ga = short[in_month & full_mask_gated]["sized_pnl"].to_numpy()
        rescue_rows.append({
            "month": m,
            "no_gate_n": int(len(ng)), "no_gate_pf": _pf(ng),
            "gated_n": int(len(ga)), "gated_pf": _pf(ga),
        })
        print(f"  {m:<10} {len(ng):>10} {_pf(ng):>11.3f} "
              f"{len(ga):>9} {_pf(ga):>10.3f}", flush=True)

    # ---- Ship-gate verdict --------------------------------------------------
    oot_full = cells["oot_gate_on"]
    oot_pf_ok = bool(np.isfinite(oot_full["pf"]) and oot_full["pf"] >= 1.30)
    oot_n_ok = oot_full["trades"] >= 200
    wf_p5_ok = bool(wf["rollup_pf"].get("p5", -1) > 1.0)
    mc_p5_ok = bool(np.isfinite(mc.get("p5", float("nan"))) and mc.get("p5", -1) > 1.0)
    ship_gate = {
        "oot_pf_ge_130": oot_pf_ok,
        "oot_n_ge_200": oot_n_ok,
        "walkforward_p5_gt_1": wf_p5_ok,
        "mc_p5_gt_1": mc_p5_ok,
        "overall": oot_pf_ok and oot_n_ok and wf_p5_ok and mc_p5_ok,
    }
    print("\n=== Ship gate (short + meta-gate) ===")
    for k, v in ship_gate.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  → overall: {'SHIP' if ship_gate['overall'] else 'DO NOT SHIP YET'}")

    # ---- Save -------------------------------------------------------------
    out = {
        "weekly_ema_period": WEEKLY_EMA_PERIOD,
        "predictor_threshold": threshold,
        "by_cell": cells,
        "walkforward_gated": wf,
        "monte_carlo_oot_gated": mc,
        "catastrophic_month_rescue": rescue_rows,
        "ship_gate": ship_gate,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
