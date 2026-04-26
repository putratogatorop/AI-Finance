"""Phase 16.2 — Re-backtest D1 SHORT trades with the v3 classifier gate.

Loads d1_short_trades_with_features.csv (produced by train_d1_classifier_v3.py),
scores every trade with the saved LR pipeline, and applies a threshold gate at
TRAIN q0.5. Computes ship-gate metrics for:

  - D1 raw (no gate)         ← Phase 15 winner
  - D1 + v3 LR gate          ← did training a fresh classifier on the D1
                                distribution add edge, or repeat the D0 mistake?

Decision rule (from Phase 16 plan):
  classifier-gated OOT PF > raw OOT PF AND classifier-gated WF p5 > raw WF p5
  → keep classifier; otherwise drop and ship raw D1.

Run from services/python/:
    python scripts/backtest_d1_with_v3_classifier.py
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

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3

DATA_DIR = SERVICES_PY / "data"
MODELS_DIR = SERVICES_PY / "models"
RESULTS_DIR = SERVICES_PY / "results"

TRADES_PATH = DATA_DIR / "d1_short_trades_with_features.csv"
MODEL_PATH = MODELS_DIR / "d1_short_v3.joblib"
META_PATH = MODELS_DIR / "d1_short_v3_meta.json"
OUT_PATH = RESULTS_DIR / "phase16_2_d1_with_v3_classifier.json"


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


def _summarize(df: pd.DataFrame, label: str):
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
        f"  {label:<24} TRAIN n={train_m['trades']:>5} PF={train_m['pf']:.3f}  "
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

    print(f"loading model from {MODEL_PATH}", flush=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    print(f"  threshold @ TRAIN q0.5: {threshold:.4f}", flush=True)

    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]

    print("\n=== Variant comparison ===")
    raw = _summarize(df, "D1 raw (no gate)")
    gated = df[df["score"] >= threshold].reset_index(drop=True)
    print(f"  classifier kept {len(gated):,} of {len(df):,} trades "
          f"({len(gated)/max(len(df),1)*100:.1f}%)")
    classified = _summarize(gated, "D1 + v3 classifier")

    # Decision rule
    raw_oot_pf = raw["oot_metrics"]["pf"]
    raw_wf_p5 = raw["walkforward"].get("p5", float("nan"))
    cls_oot_pf = classified["oot_metrics"]["pf"]
    cls_wf_p5 = classified["walkforward"].get("p5", float("nan"))
    classifier_helps = (
        np.isfinite(cls_oot_pf) and np.isfinite(cls_wf_p5)
        and np.isfinite(raw_oot_pf) and np.isfinite(raw_wf_p5)
        and cls_oot_pf > raw_oot_pf and cls_wf_p5 > raw_wf_p5
    )
    print(f"\nDecision: {'KEEP CLASSIFIER' if classifier_helps else 'DROP CLASSIFIER (raw D1 wins)'}")
    print(f"  raw       OOT PF={raw_oot_pf:.3f}  WF p5={raw_wf_p5:.3f}")
    print(f"  classifier OOT PF={cls_oot_pf:.3f}  WF p5={cls_wf_p5:.3f}")

    out = {
        "phase": "16.2 — D1 + v3 classifier",
        "snapshot": SNAPSHOT_DATE,
        "model": str(MODEL_PATH),
        "meta": meta,
        "raw": raw,
        "classifier_gated": classified,
        "classifier_kept_pct": float(len(gated) / max(len(df), 1)),
        "decision": "keep_classifier" if classifier_helps else "drop_classifier",
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
