"""v_new_1 — One-time Platt calibration setup.

Loads v_new_1_long and v_new_1_short base LightGBM models, fits LogisticRegression
Platt scalers on the cal_fit slice (2024-10-01 → 2025-12-31), and saves:
    services/python/models/v_new_1_long/platt_calibrator.joblib
    services/python/models/v_new_1_short/platt_calibrator.joblib

Run ONCE on VPS before deploying the shadow-scoring cron.

USAGE (from /opt/ai-finance/):
    services/python/.venv/bin/python3 \
        services/python/scripts/v_new_1_platt_fit.py
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time
import warnings
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# ── CalibratedLGBM stub — must match the class used when model was pickled ────
# The joblib file was serialised with this class defined in v_new_1_phase3_train.
# We must define an identical class here so pickle can reconstruct the object.
class CalibratedLGBM:
    """LGBMClassifier + IsotonicRegression calibrator (matches phase3_train.py)."""

    def __init__(self, base_model, calibrator: IsotonicRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

warnings.filterwarnings("ignore", category=UserWarning)

# ── libomp workaround for macOS (no-op on Linux) ─────────────────────────────
_SKLEARN_DYLIBS = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".venv"
    / "lib"
    / "python3.12"
    / "site-packages"
    / "sklearn"
    / ".dylibs"
)
if _SKLEARN_DYLIBS.is_dir():
    import os
    os.environ.setdefault("DYLD_LIBRARY_PATH", str(_SKLEARN_DYLIBS))

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "services" / "python"
DATA_DIR = PYTHON_ROOT / "data" / "v_new_1"
MODELS_DIR = PYTHON_ROOT / "models"

# ── Calibration fit boundaries (same as training script) ─────────────────────
CAL_FIT_START = pd.Timestamp("2024-10-01", tz="UTC")
TRAIN_FIT_END = pd.Timestamp("2024-09-30 23:59:59", tz="UTC")


def _ts(label: str = "") -> str:
    t = datetime.now(UTC).strftime("%H:%M:%S")
    return f"[{t}]{' ' + label if label else ''}"


def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() == 0:
            continue
        frac = mask.sum() / n
        ece += frac * abs(float(probs[mask].mean()) - float(y_true[mask].mean()))
    return float(ece)


def fit_platt_for_direction(direction: str) -> dict:
    """Fit and save Platt calibrator for one direction. Returns calibration metrics."""
    label_col = "long_event" if direction == "long" else "short_event"
    dir_label = f"v_new_1_{direction}"
    t0 = time.monotonic()

    print(f"\n{'='*60}")
    print(f"{_ts(dir_label)} Fitting Platt calibrator for {dir_label}")
    print(f"{'='*60}")

    # ── 1. Load base model ────────────────────────────────────────────────────
    model_path = MODELS_DIR / dir_label / f"{dir_label}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"{_ts(dir_label)} Loading model: {model_path}")
    calibrated_model = joblib.load(model_path)
    base_model = calibrated_model.base_model
    print(f"  base_model type: {type(base_model).__name__}")

    # ── 2. Load features ──────────────────────────────────────────────────────
    print(f"{_ts(dir_label)} Loading features_full.parquet ...")
    features_df = pd.read_parquet(DATA_DIR / "features_full.parquet")
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)
    print(f"  features: {len(features_df):,} rows")

    # ── 3. Load labels for cal_fit period ────────────────────────────────────
    print(f"{_ts(dir_label)} Loading labels_{direction}.parquet ...")
    labels_df = pd.read_parquet(DATA_DIR / f"labels_{direction}.parquet")
    labels_df["timestamp"] = pd.to_datetime(labels_df["timestamp"], utc=True)
    print(f"  labels: {len(labels_df):,} rows")

    # ── 4. Load feature names from meta ──────────────────────────────────────
    meta_path = MODELS_DIR / dir_label / f"{dir_label}_meta.json"
    with open(meta_path) as f:
        meta = json.load(f)
    feature_names: list[str] = meta["feature_names"]

    # Drop constant column if present
    feature_names = [f for f in feature_names if f != "signals_same_15m_same_detector"]
    print(f"  Feature count: {len(feature_names)}")

    # ── 5. Merge labels + features, filter to cal_fit ─────────────────────────
    df = labels_df.merge(
        features_df[["symbol", "timestamp"] + [f for f in feature_names if f in features_df.columns]],
        on=["symbol", "timestamp"],
        how="inner",
    )
    df = df[df["timestamp"] >= CAL_FIT_START].reset_index(drop=True)
    print(f"  cal_fit rows: {len(df):,} "
          f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()}) "
          f"pos_rate={df[label_col].mean():.4f}")

    # ── 6. Build feature matrix ───────────────────────────────────────────────
    available = [f for f in feature_names if f in df.columns]
    X = np.nan_to_num(df[available].values, nan=0.0)  # noqa: N806
    y = df[label_col].values.astype(int)
    print(f"  X shape: {X.shape}, positives: {y.sum():,} ({y.mean():.4f})")

    # ── 7. Get raw base model probabilities ───────────────────────────────────
    print(f"{_ts(dir_label)} Computing raw probabilities on cal_fit ...")
    raw_probs = base_model.predict_proba(X)[:, 1]
    print(f"  raw probs: p10={np.percentile(raw_probs, 10):.4f}  "
          f"p50={np.percentile(raw_probs, 50):.4f}  "
          f"p90={np.percentile(raw_probs, 90):.4f}  "
          f"max={raw_probs.max():.4f}")

    # ── 8. Fit Platt scaler ───────────────────────────────────────────────────
    print(f"{_ts(dir_label)} Fitting Platt LogisticRegression ...")
    platt = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    platt.fit(raw_probs.reshape(-1, 1), y)

    platt_probs = platt.predict_proba(raw_probs.reshape(-1, 1))[:, 1]

    coef = float(platt.coef_[0][0])
    intercept = float(platt.intercept_[0])
    print(f"  Platt coef={coef:.4f}  intercept={intercept:.4f}")
    print(f"  Platt probs: p10={np.percentile(platt_probs, 10):.4f}  "
          f"p50={np.percentile(platt_probs, 50):.4f}  "
          f"p90={np.percentile(platt_probs, 90):.4f}  "
          f"max={platt_probs.max():.4f}")

    # ── 9. Calibration quality metrics ───────────────────────────────────────
    brier_raw = float(brier_score_loss(y, raw_probs))
    brier_platt = float(brier_score_loss(y, platt_probs))
    ece_raw = _ece(y, raw_probs)
    ece_platt = _ece(y, platt_probs)
    print(f"  Brier raw={brier_raw:.4f}  platt={brier_platt:.4f}  "
          f"(improvement: {brier_raw - brier_platt:+.4f})")
    print(f"  ECE   raw={ece_raw:.4f}   platt={ece_platt:.4f}  "
          f"(improvement: {ece_raw - ece_platt:+.4f})")

    # ── 10. Save Platt calibrator ─────────────────────────────────────────────
    out_dir = MODELS_DIR / dir_label
    out_dir.mkdir(parents=True, exist_ok=True)
    platt_path = out_dir / "platt_calibrator.joblib"
    joblib.dump(platt, platt_path)
    print(f"{_ts(dir_label)} Saved: {platt_path}")

    # ── 11. Save metadata ─────────────────────────────────────────────────────
    platt_meta = {
        "direction": direction,
        "fitted_at": datetime.now(UTC).isoformat(),
        "cal_fit_start": str(CAL_FIT_START.date()),
        "cal_fit_n_rows": int(len(df)),
        "cal_fit_pos_rate": round(float(y.mean()), 4),
        "platt_coef": round(coef, 6),
        "platt_intercept": round(intercept, 6),
        "brier_raw": round(brier_raw, 4),
        "brier_platt": round(brier_platt, 4),
        "ece_raw": round(ece_raw, 4),
        "ece_platt": round(ece_platt, 4),
        "feature_names_used": available,
    }
    meta_out_path = out_dir / "platt_calibrator_meta.json"
    with open(meta_out_path, "w") as f:
        json.dump(platt_meta, f, indent=2)
    print(f"{_ts(dir_label)} Saved metadata: {meta_out_path}")

    elapsed = time.monotonic() - t0
    print(f"{_ts(dir_label)} Done in {elapsed:.1f}s")

    return platt_meta


def main() -> None:
    print(f"{_ts()} v_new_1 Platt Calibration Setup")
    print(f"{_ts()} Python root: {PYTHON_ROOT}")
    print(f"{_ts()} Models dir:  {MODELS_DIR}")
    print(f"{_ts()} Data dir:    {DATA_DIR}")

    results = {}
    for direction in ["long", "short"]:
        results[direction] = fit_platt_for_direction(direction)

    print(f"\n{'='*60}")
    print("PLATT SETUP COMPLETE")
    print(f"{'='*60}")
    for direction, m in results.items():
        print(f"  {direction:5s}: brier {m['brier_raw']:.4f} → {m['brier_platt']:.4f}  "
              f"ece {m['ece_raw']:.4f} → {m['ece_platt']:.4f}  "
              f"coef={m['platt_coef']:.4f} intercept={m['platt_intercept']:.4f}")

    print(f"\n{_ts()} Platt models saved:")
    for direction in ["long", "short"]:
        p = MODELS_DIR / f"v_new_1_{direction}" / "platt_calibrator.joblib"
        print(f"  {p}")

    print(f"\n{_ts()} Next step: run v_new_1_shadow_score.py to validate end-to-end")


if __name__ == "__main__":
    main()
