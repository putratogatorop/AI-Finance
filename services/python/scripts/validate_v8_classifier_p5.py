"""Phase 5 — Ship validation for macd_pullback_long v2p2 classifier.

Runs:
  1. Temporal walk-forward: 6-month windows across the full 3-year parquet.
     Applies the final trained model (no retraining) to each window at
     threshold 0.475. Reports per-window PF, WR, MDD.
  2. Calibration check: binned predicted probability vs realized win rate.
  3. Stability check: max single-feature SHAP % (must be < 20%).
  4. Ship gate summary: pass / fail each criterion.

USAGE (from services/python/):
    python scripts/validate_v8_classifier_p5.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgbm  # noqa: F401 (needed for joblib unpickling CalibratedLGBM)
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # noqa: F401
from sklearn.metrics import average_precision_score, brier_score_loss

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── CONFIG ────────────────────────────────────────────────────────────────────

TRAINING_PARQUET = (
    Path(__file__).resolve().parents[1]
    / "data" / "training" / "v8_trades_with_features_2026-04-01_p2.parquet"
)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "v8_classifier_lgbm_v2p2"
DETECTOR = "macd_pullback_long"
LIVE_THRESHOLD = 0.475    # locked Phase 2 optimal threshold
P1_BASELINE_OOS_PRAUC = 0.525  # Phase 1 median OOS PR-AUC
SHIP_GATE_MEDIAN_PF = 1.7
SHIP_GATE_OOS_PRAUC_DELTA = 0.04  # must beat Phase 1 by at least 0.04
SHIP_GATE_BRIER = 0.25    # must be < than prevalence*(1-prevalence) = 0.235... realistically < 0.25


# ── HELPERS ───────────────────────────────────────────────────────────────────

class CalibratedLGBM:
    """Mirror of training-script class so joblib can unpickle."""
    def __init__(self, base_model: lgbm.LGBMClassifier, calibrator: IsotonicRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])


def _pf(pnls: np.ndarray) -> float:
    w = float(pnls[pnls > 0].sum())
    l = float(-pnls[pnls <= 0].sum())
    return w / l if l > 0 else float("inf")


def _wr(labels: np.ndarray) -> float:
    return float(labels.mean()) if len(labels) > 0 else float("nan")


def _max_dd(pnls: np.ndarray, base_size: float = 0.01, start: float = 100.0) -> float:
    eq = start; peak = start; mdd = 0.0
    for p in pnls:
        eq *= 1.0 + base_size * p
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    return float(mdd * 100.0)


def _eval_window(probs: np.ndarray, y: np.ndarray, pnls: np.ndarray, thr: float, label: str) -> dict:
    mask = probs >= thr
    n_pass = int(mask.sum())
    return {
        "label": label,
        "n_total": int(len(y)),
        "n_pass": n_pass,
        "pass_pct": round(float(100.0 * mask.mean()), 1),
        "pf": round(_pf(pnls[mask]), 3) if n_pass > 0 else float("nan"),
        "wr_pct": round(float(100.0 * _wr(y[mask])), 1) if n_pass > 0 else float("nan"),
        "mdd_pct": round(_max_dd(pnls[mask]), 2) if n_pass > 0 else float("nan"),
        "pr_auc": round(float(average_precision_score(y, probs)) if len(set(y)) >= 2 else float("nan"), 4),
    }


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901
    assert TRAINING_PARQUET.exists(), f"Parquet not found: {TRAINING_PARQUET}"
    assert (MODELS_DIR / f"{DETECTOR}_lgbm_v2p2.joblib").exists(), \
        f"Model not found: {MODELS_DIR}/{DETECTOR}_lgbm_v2p2.joblib"

    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Loading Phase 5 validation data: {TRAINING_PARQUET.name}")
    df = pd.read_parquet(TRAINING_PARQUET)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df[df["detector"] == DETECTOR].sort_values("entry_time").reset_index(drop=True)
    print(f"  {len(df):,} rows  ({df.entry_time.min().date()} → {df.entry_time.max().date()})")

    meta = json.load(open(MODELS_DIR / f"{DETECTOR}_lgbm_v2p2_meta.json"))
    features = meta["features_p1"] + meta["features_p2_new"]
    X = df[features].values
    y = df["label"].values.astype(int)
    pnls = df["pnl_pct"].values

    print(f"\nLoading model: {DETECTOR}_lgbm_v2p2.joblib")
    model = joblib.load(MODELS_DIR / f"{DETECTOR}_lgbm_v2p2.joblib")
    probs = model.predict_proba(X)[:, 1]

    # ── 1. Temporal walk-forward (6-month windows, apply final model) ─────────
    print("\n── 1. TEMPORAL STABILITY (6-month windows, final model @ 0.475) ──────")
    print(f"{'Window':25s}  {'n_pass':>6s}  {'PF':>6s}  {'WR':>6s}  {'MDD':>7s}  PR-AUC")

    window_results = []
    start = df["entry_time"].min().to_period("M")
    end = df["entry_time"].max().to_period("M")
    month_range = pd.period_range(start, end, freq="M")

    i = 0
    while i < len(month_range):
        w_start = month_range[i].to_timestamp(freq="M", how="start").tz_localize("UTC")
        w_end_idx = min(i + 5, len(month_range) - 1)
        w_end = month_range[w_end_idx].to_timestamp(freq="M", how="end").tz_localize("UTC")
        mask = (df["entry_time"] >= w_start) & (df["entry_time"] <= w_end)
        if mask.sum() < 30:
            i += 6
            continue
        w_probs = probs[mask.values]
        w_y = y[mask.values]
        w_pnls = pnls[mask.values]
        res = _eval_window(w_probs, w_y, w_pnls, LIVE_THRESHOLD,
                           f"{w_start.strftime('%Y-%m')}→{w_end.strftime('%Y-%m')}")
        window_results.append(res)
        print(f"  {res['label']:25s}  {res['n_pass']:6d}  {res['pf']:6.2f}  "
              f"{res['wr_pct']:5.1f}%  {res['mdd_pct']:7.2f}%  {res['pr_auc']:.4f}")
        i += 6

    pf_values = [r["pf"] for r in window_results if np.isfinite(r["pf"])]
    wr_values = [r["wr_pct"] for r in window_results if np.isfinite(r["wr_pct"])]
    print(f"\n  Cross-window PF:  p5={np.percentile(pf_values, 5):.2f}  "
          f"p50={np.median(pf_values):.2f}  p95={np.percentile(pf_values, 95):.2f}")
    print(f"  Cross-window WR:  p5={np.percentile(wr_values, 5):.1f}%  "
          f"p50={np.median(wr_values):.1f}%  p95={np.percentile(wr_values, 95):.1f}%")

    # ── 2. Holdout evaluation (last 3 months: 2026-01-01 to 2026-03-31) ───────
    print("\n── 2. HOLDOUT EVALUATION (2026-01-01 → 2026-03-31) ──────────────────")
    holdout_mask = df["entry_time"] >= pd.Timestamp("2026-01-01", tz="UTC")
    ho_probs = probs[holdout_mask.values]
    ho_y = y[holdout_mask.values]
    ho_pnls = pnls[holdout_mask.values]

    # Monthly breakdown in holdout
    for month_label, m_start, m_end in [
        ("Jan 2026", "2026-01-01", "2026-01-31"),
        ("Feb 2026", "2026-02-01", "2026-02-28"),
        ("Mar 2026", "2026-03-01", "2026-03-31"),
    ]:
        mm = (df["entry_time"] >= pd.Timestamp(m_start, tz="UTC")) & \
             (df["entry_time"] <= pd.Timestamp(m_end, tz="UTC"))
        if mm.sum() == 0:
            continue
        r = _eval_window(probs[mm.values], y[mm.values], pnls[mm.values],
                         LIVE_THRESHOLD, month_label)
        print(f"  {r['label']:15s}  n={r['n_pass']:3d}/{r['n_total']:4d}  "
              f"PF={r['pf']:.2f}  WR={r['wr_pct']:.1f}%  MDD={r['mdd_pct']:.2f}%")

    ho_res = _eval_window(ho_probs, ho_y, ho_pnls, LIVE_THRESHOLD, "Q1 2026 total")
    brier_ho = float(brier_score_loss(ho_y, ho_probs))
    print(f"\n  Q1 2026 TOTAL:  n={ho_res['n_pass']}/{ho_res['n_total']}  "
          f"PF={ho_res['pf']:.3f}  WR={ho_res['wr_pct']:.1f}%  "
          f"MDD={ho_res['mdd_pct']:.2f}%  PR-AUC={ho_res['pr_auc']:.4f}  Brier={brier_ho:.4f}")

    # ── 3. Calibration check ──────────────────────────────────────────────────
    print("\n── 3. CALIBRATION (predicted prob vs realized WR, 0.05 bins) ─────────")
    bins = np.arange(0.0, 1.05, 0.05)
    bin_labels_center = (bins[:-1] + bins[1:]) / 2
    print(f"{'Bin':12s}  {'n':>5s}  {'pred_prob':>9s}  {'actual_wr':>9s}  {'delta':>8s}")
    cal_errors = []
    for i in range(len(bins) - 1):
        mask = (ho_probs >= bins[i]) & (ho_probs < bins[i + 1])
        if mask.sum() < 3:
            continue
        pred = float(ho_probs[mask].mean())
        actual = float(ho_y[mask].mean())
        cal_errors.append(abs(pred - actual))
        print(f"  [{bins[i]:.2f},{bins[i+1]:.2f})  {mask.sum():5d}  {pred:9.3f}  "
              f"{actual:9.3f}  {actual-pred:+8.3f}")
    ece = float(np.mean(cal_errors)) if cal_errors else float("nan")
    print(f"\n  Expected calibration error (mean |delta|): {ece:.4f}")

    # ── 4. SHAP stability check ────────────────────────────────────────────────
    print("\n── 4. SHAP STABILITY ────────────────────────────────────────────────")
    shap = meta.get("shap_importance_top20", {})
    total_shap = sum(shap.values())
    max_feat = max(shap, key=lambda k: shap[k])
    max_pct = 100.0 * shap[max_feat] / total_shap if total_shap > 0 else float("nan")
    print(f"  Max single-feature SHAP: {max_feat} = {shap[max_feat]:.4f} ({max_pct:.1f}% of top-20)")
    print(f"  Threshold: < 20% for pass (fragile if dominant)")
    for f, v in list(shap.items())[:5]:
        pct = 100.0 * v / total_shap
        print(f"    {f:35s}  {v:.4f}  ({pct:.1f}%)")

    # ── 5. Per-fold summary from meta ─────────────────────────────────────────
    print("\n── 5. TRAINING CV FOLD METRICS (from meta) ──────────────────────────")
    print(f"{'Fold':6s}  {'PR-AUC':>8s}  {'ROC-AUC':>8s}  {'Brier':>7s}  best_iter")
    fold_prauc = []
    for fm in meta.get("per_fold_metrics", []):
        print(f"  {fm['fold']:4d}  {fm['pr_auc']:8.4f}  {fm['roc_auc']:8.4f}  "
              f"{fm['brier']:7.4f}  {fm['best_iteration']}")
        fold_prauc.append(fm["pr_auc"])
    print(f"\n  Median OOS PR-AUC: {np.median(fold_prauc):.4f}")
    print(f"  Train/OOS gap:     {meta.get('train_oos_prauc_gap', float('nan')):.4f}")

    # ── 6. SHIP GATE ──────────────────────────────────────────────────────────
    print("\n── 6. SHIP GATE ─────────────────────────────────────────────────────")
    median_window_pf = float(np.median(pf_values)) if pf_values else float("nan")
    oos_prauc = float(np.median(fold_prauc))
    prauc_delta = oos_prauc - P1_BASELINE_OOS_PRAUC

    gate = {
        "median_window_pf": (median_window_pf, median_window_pf >= SHIP_GATE_MEDIAN_PF,
                             f">= {SHIP_GATE_MEDIAN_PF}"),
        "oos_prauc_delta": (prauc_delta, prauc_delta >= SHIP_GATE_OOS_PRAUC_DELTA,
                            f">= +{SHIP_GATE_OOS_PRAUC_DELTA} vs P1 baseline"),
        "brier_ho": (brier_ho, brier_ho < SHIP_GATE_BRIER,
                     f"< {SHIP_GATE_BRIER}"),
        "max_shap_pct": (max_pct, max_pct < 20.0, "< 20% (stability)"),
        "train_oos_gap": (meta.get("train_oos_prauc_gap", 1.0),
                          meta.get("train_oos_prauc_gap", 1.0) < 0.15, "< 0.15"),
    }

    all_pass = True
    for name, (val, passed, criterion) in gate.items():
        mark = "✅" if passed else "❌"
        print(f"  {mark} {name:25s}  {val:.4f}  ({criterion})")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("✅ SHIP GATE PASSED — v2p2 is ready for A/B paper deployment.")
        print(f"   Deploy at threshold {LIVE_THRESHOLD} with env var "
              f"V8_MACD_PULLBACK_LONG_CLS_FLOOR={LIVE_THRESHOLD}")
    else:
        print("❌ SHIP GATE: one or more criteria failed — review above.")

    print(f"\n  Phase 1 holdout PF baseline:     4.84 @ 0.375 (n=220/991)")
    print(f"  Phase 2 holdout PF (this model): {ho_res['pf']:.3f} @ {LIVE_THRESHOLD} "
          f"(n={ho_res['n_pass']}/{ho_res['n_total']})")


if __name__ == "__main__":
    main()
