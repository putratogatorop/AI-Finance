"""Phase 16.3 — Stack D1 detector + rapid-rally exit + BTC daily RSI<20 meta-gate.

Phase 11 R1 found that skipping trades when BTC daily RSI(14) is below 20 lifted
OOT PF +14% on the Phase-9 v2 trades. The hypothesis: that signal targets the
bear-bottom whipsaw regime where SHORT entries get squeezed. This script tests
whether the same gate stacks additively on the Phase-15 D1 trades (which already
include the rapid-rally exit).

Pipeline:
  1. Load D1 SHORT trades (saved by train_d1_classifier_v3.py).
  2. Compute BTC daily RSI(14) panel and join to each trade's entry_time.
  3. Compare:
       - D1 raw (no gate)
       - D1 + RSI<20 gate (skip if BTC daily RSI(14) < 20 at entry)
  4. Optionally also stack with the v3 classifier (if backtest output exists),
     so we can compare D1 + classifier + RSI<20 vs each subset.

Decision rule:
  RSI<20 gate added iff WF p5 lifts AND OOT n stays >= 200.

Run from services/python/:
    python scripts/phase16_3_d1_stacked_gates.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES
from src.ml.indicators import rsi as _rsi

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3

CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
DATA_DIR = SERVICES_PY / "data"
MODELS_DIR = SERVICES_PY / "models"
RESULTS_DIR = SERVICES_PY / "results"
TRADES_PATH = DATA_DIR / "d1_short_trades_with_features.csv"
MODEL_PATH = MODELS_DIR / "d1_short_v3.joblib"
META_PATH = MODELS_DIR / "d1_short_v3_meta.json"
OUT_PATH = RESULTS_DIR / "phase16_3_d1_stacked_gates.json"

RSI_PERIOD = 14
RSI_FLOOR = 20.0


def _pf(pnls):
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _cell(pnls):
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


def _walkforward(df, pnl_col="sized_pnl"):
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
        "p50": float(np.percentile(pfs, 50)),
        "p95": float(np.percentile(pfs, 95)),
        "p_lt_1": float((pfs < 1.0).mean()),
        "p_gte_130": float((pfs >= 1.30).mean()),
    }


def _monte_carlo(pnls, n=MC_N_RESAMPLES):
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


def _build_btc_rsi14_daily_15m(candles: pd.DataFrame) -> pd.Series:
    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    btc_15m_close = btc["close"].astype(float)
    btc_daily = btc_15m_close.resample("1D", label="right", closed="right").last().dropna()
    rsi_d = _rsi(btc_daily, RSI_PERIOD).shift(1)  # shift 1 day to avoid lookahead
    return rsi_d.reindex(btc_15m_close.index, method="ffill")


def _summarize(df: pd.DataFrame, label: str) -> dict:
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["period"] = np.where(df["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
    train_pnls = df[df["period"] == "train"]["sized_pnl"].to_numpy()
    oot_pnls = df[df["period"] == "oot"]["sized_pnl"].to_numpy()
    train_m = _cell(train_pnls)
    oot_m = _cell(oot_pnls)
    wf = _walkforward(df, "sized_pnl")
    mc = _monte_carlo(oot_pnls)
    print(
        f"  {label:<28} TRAIN n={train_m['trades']:>5} PF={train_m['pf']:.3f}  "
        f"OOT n={oot_m['trades']:>5} PF={oot_m['pf']:.3f}  "
        f"WF p5={wf.get('p5', float('nan')):.3f}  P(loss)={wf.get('p_lt_1', float('nan'))*100:.1f}%  "
        f"MC p5={mc.get('p5', float('nan')):.3f}",
        flush=True,
    )
    return {
        "train_metrics": train_m,
        "oot_metrics": oot_m,
        "walkforward": wf,
        "monte_carlo_oot": mc,
    }


def main():
    started = time.monotonic()

    print(f"loading trades from {TRADES_PATH}", flush=True)
    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"  {len(df):,} trades loaded", flush=True)

    print(f"loading candles for BTC RSI panel...", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    print(f"computing BTC daily RSI({RSI_PERIOD}) panel...", flush=True)
    btc_rsi14_15m = _build_btc_rsi14_daily_15m(candles)
    btc_rsi_naive = btc_rsi14_15m.copy()
    btc_rsi_naive.index = pd.to_datetime(btc_rsi_naive.index, utc=True).tz_convert(None)

    entry_naive = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert(None)
    df["btc_rsi14_daily"] = btc_rsi_naive.reindex(
        pd.DatetimeIndex(entry_naive), method="ffill"
    ).to_numpy(dtype=float)
    finite_rsi = df["btc_rsi14_daily"].notna()
    print(f"  {finite_rsi.sum():,}/{len(df):,} trades have a valid BTC RSI at entry "
          f"({finite_rsi.mean()*100:.1f}%)")
    df = df[finite_rsi].reset_index(drop=True)

    # Optional classifier scores
    has_classifier = MODEL_PATH.exists() and META_PATH.exists()
    if has_classifier:
        model = joblib.load(MODEL_PATH)
        meta = json.loads(META_PATH.read_text())
        threshold = float(meta["threshold_train_q50"])
        feat_cols = FEATURE_NAMES + ["is_short"]
        X = df[feat_cols].to_numpy(dtype=float)
        df["score"] = model.predict_proba(X)[:, 1]
        print(f"  classifier loaded, threshold @ TRAIN q0.5: {threshold:.4f}", flush=True)
    else:
        threshold = float("nan")

    print("\n=== Variant comparison ===")
    raw = _summarize(df, "D1 raw")
    rsi_gated = df[df["btc_rsi14_daily"] >= RSI_FLOOR].reset_index(drop=True)
    print(f"  RSI>={RSI_FLOOR} kept {len(rsi_gated):,} of {len(df):,} "
          f"({len(rsi_gated)/max(len(df),1)*100:.1f}%)")
    rsi = _summarize(rsi_gated, f"D1 + RSI>={RSI_FLOOR:.0f}")

    if has_classifier:
        cls_gated = df[df["score"] >= threshold].reset_index(drop=True)
        cls = _summarize(cls_gated, "D1 + v3 classifier")
        full_gated = df[(df["score"] >= threshold) & (df["btc_rsi14_daily"] >= RSI_FLOOR)].reset_index(drop=True)
        full = _summarize(full_gated, f"D1 + cls + RSI>={RSI_FLOOR:.0f}")
    else:
        cls = None
        full = None

    # Decision: RSI<20 gate
    raw_oot_n = raw["oot_metrics"]["trades"]
    raw_oot_pf = raw["oot_metrics"]["pf"]
    raw_wf_p5 = raw["walkforward"].get("p5", float("nan"))
    rsi_oot_n = rsi["oot_metrics"]["trades"]
    rsi_oot_pf = rsi["oot_metrics"]["pf"]
    rsi_wf_p5 = rsi["walkforward"].get("p5", float("nan"))
    rsi_helps = (
        np.isfinite(rsi_wf_p5) and np.isfinite(raw_wf_p5)
        and rsi_wf_p5 > raw_wf_p5
        and rsi_oot_n >= 200
    )
    print(f"\nRSI<{RSI_FLOOR:.0f} meta-gate:")
    print(f"  raw       OOT n={raw_oot_n}  PF={raw_oot_pf:.3f}  WF p5={raw_wf_p5:.3f}")
    print(f"  RSI gated OOT n={rsi_oot_n}  PF={rsi_oot_pf:.3f}  WF p5={rsi_wf_p5:.3f}")
    print(f"  decision: {'STACK RSI<20 GATE' if rsi_helps else 'DROP RSI<20 GATE'}")

    out = {
        "phase": "16.3 — D1 + rapid-rally + RSI<20 stack",
        "snapshot": SNAPSHOT_DATE,
        "rsi_floor": RSI_FLOOR,
        "raw": raw,
        "rsi_gated": rsi,
        "classifier_gated": cls,
        "classifier_plus_rsi_gated": full,
        "rsi_decision": "stack" if rsi_helps else "drop",
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
