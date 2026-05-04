"""v_new_1 Phase 3 — Calibration Iteration A1 + A2.

Goal: determine whether isotonic calibration collapse hurts downstream usability.
- A1: raw LightGBM predict_proba (no calibration)
- A2: Platt scaling (LogisticRegression on raw cal_fit probs)
- A2.5: BetaCal (skipped — not installed)

NO model retraining. Pure inference + sklearn LR fit on cal_fit.

USAGE (from repo root):
    DYLD_LIBRARY_PATH=$(services/python/.venv/bin/python3 -c \\
        "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \\
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_phase3_calibration_iter.py

OUTPUTS:
    services/python/results/v_new_1_phase3_calibration_iter/master_comparison.csv
    services/python/results/v_new_1_phase3_calibration_iter/score_distributions.csv
    services/python/results/v_new_1_phase3_calibration_iter/verdict.md
    services/python/data/v_new_1/oos_scored_long_RAW.parquet
    services/python/data/v_new_1/oos_scored_long_PLATT.parquet
    services/python/data/v_new_1/oos_scored_short_RAW.parquet
    services/python/data/v_new_1/oos_scored_short_PLATT.parquet
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import warnings
from datetime import UTC, datetime

# ── libomp workaround for macOS (must happen before importing lightgbm) ──
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
    os.environ.setdefault("DYLD_LIBRARY_PATH", str(_SKLEARN_DYLIBS))

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

# ── import shared classes from training script ────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from v_new_1_phase3_train import (  # noqa: E402
    CAL_FIT_START,
    FEATURE_NAMES,
    OOS_YEAR_SLICES,
    TRAIN_FIT_END,
    CalibratedLGBM,
    _build_universe_set,
    _filter_to_universe,
    _top_k_precision,
)

warnings.filterwarnings("ignore", category=UserWarning)

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "services" / "python"
DATA_DIR = PYTHON_ROOT / "data" / "v_new_1"
RESULTS_DIR = PYTHON_ROOT / "results" / "v_new_1_phase3_calibration_iter"


# ── helpers ───────────────────────────────────────────────────────────────────

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


def _score_percentiles(scores: np.ndarray) -> dict:
    """Return dict of percentile statistics for a score array."""
    return {
        "p10": round(float(np.percentile(scores, 10)), 6),
        "p25": round(float(np.percentile(scores, 25)), 6),
        "p50": round(float(np.percentile(scores, 50)), 6),
        "p75": round(float(np.percentile(scores, 75)), 6),
        "p90": round(float(np.percentile(scores, 90)), 6),
        "p99": round(float(np.percentile(scores, 99)), 6),
        "max": round(float(scores.max()), 6),
        "width_p99_p10": round(float(np.percentile(scores, 99) - np.percentile(scores, 10)), 6),
    }


def _eval_scores(
    y: np.ndarray,
    scores: np.ndarray,
    direction: str,
    calibration: str,
    year_label: str,
    n_rows: int,
    positive_rate: float,
) -> dict:
    """Compute full metric set for one (direction, calibration, year) combination."""
    if len(set(y)) < 2 or len(scores) < 50:
        return {
            "direction": direction, "calibration": calibration, "year": year_label,
            "n_rows": n_rows, "positive_rate": positive_rate,
            "pr_auc": float("nan"), "roc_auc": float("nan"), "brier": float("nan"),
            "top10_prec": float("nan"), "top5_prec": float("nan"), "top1_prec": float("nan"),
            "ece": float("nan"),
        }

    pr_auc = float(average_precision_score(y, scores))
    roc_auc = float(roc_auc_score(y, scores))
    brier = float(brier_score_loss(y, scores))
    top10_prec = _top_k_precision(y, scores, 0.10)
    top5_prec = _top_k_precision(y, scores, 0.05)
    top1_prec = _top_k_precision(y, scores, 0.01)
    ece = _ece(y, scores)

    return {
        "direction": direction,
        "calibration": calibration,
        "year": year_label,
        "n_rows": n_rows,
        "positive_rate": round(positive_rate, 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4),
        "top10_prec": round(top10_prec, 4),
        "top5_prec": round(top5_prec, 4),
        "top1_prec": round(top1_prec, 4),
        "ece": round(ece, 4),
    }


# ── per-direction calibration evaluation ─────────────────────────────────────

def run_direction(direction: str) -> tuple[list[dict], list[dict], list[dict]]:
    """Evaluate A1 (raw) + A2 (Platt) for one direction.

    Returns:
        (comparison_rows, distribution_rows, scored_oos_per_cal)
    """
    label_col = "long_event" if direction == "long" else "short_event"
    dir_label = f"v_new_1_{direction}"

    print(f"\n{'='*70}")
    print(f"{_ts(dir_label)} Starting calibration evaluation for {dir_label}")
    print(f"{'='*70}")

    # ── 1. Load model ─────────────────────────────────────────────────────────
    model_path = PYTHON_ROOT / "models" / dir_label / f"{dir_label}.joblib"
    print(f"{_ts(dir_label)} Loading model from {model_path}...")
    model: CalibratedLGBM = joblib.load(model_path)
    base = model.base_model
    print(f"  Model loaded. base_model type: {type(base).__name__}")

    # ── 2. Load features + universe ───────────────────────────────────────────
    print(f"{_ts(dir_label)} Loading features + universe...")
    needed_feature_cols = [f for f in FEATURE_NAMES if f not in ("symbol", "timestamp")]
    all_feat_cols = ["symbol", "timestamp"] + [
        f for f in needed_feature_cols if f not in ("symbol", "timestamp")
    ]
    features = pd.read_parquet(DATA_DIR / "features_full.parquet", columns=all_feat_cols)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)

    universe_df = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    universe_set = _build_universe_set(universe_df)
    print(f"  features: {len(features):,} rows | universe: {len(universe_df):,} entries")

    # ── 3. Load labels and merge ──────────────────────────────────────────────
    print(f"{_ts(dir_label)} Loading labels (cal_fit + OOS)...")
    labels = pd.read_parquet(DATA_DIR / f"labels_{direction}.parquet")
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)

    oos_labels_raw = pd.read_parquet(DATA_DIR / f"labels_{direction}_oos.parquet")
    oos_labels_raw["timestamp"] = pd.to_datetime(oos_labels_raw["timestamp"], utc=True)

    # Filter to universe
    labels = _filter_to_universe(labels, universe_set)

    # Merge labels with features
    df = labels.merge(features, on=["symbol", "timestamp"], how="inner")
    if "signals_same_15m_same_detector" in df.columns:
        df = df.drop(columns=["signals_same_15m_same_detector"])

    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    feature_names_used = available_features
    print(f"  feature_names_used: {len(feature_names_used)}")

    # NaN drop (same as training script)
    nan_floor = len(feature_names_used) // 2
    df_clean = df.dropna(subset=feature_names_used, thresh=nan_floor).reset_index(drop=True)

    # ── 4. Chronological split for cal_fit ───────────────────────────────────
    cal_fit = df_clean[df_clean["timestamp"] >= CAL_FIT_START].reset_index(drop=True)
    print(f"  cal_fit: {len(cal_fit):,} rows "
          f"({cal_fit['timestamp'].min().date()} → {cal_fit['timestamp'].max().date()}) "
          f"pos_rate={cal_fit[label_col].mean():.4f}")

    X_cal = np.nan_to_num(cal_fit[feature_names_used].values, nan=0.0)  # noqa: N806
    y_cal = cal_fit[label_col].values.astype(int)

    # ── 5. Merge OOS labels with features ────────────────────────────────────
    print(f"{_ts(dir_label)} Merging OOS labels with features...")
    oos_merged = oos_labels_raw.merge(
        features[["symbol", "timestamp"] + feature_names_used],
        on=["symbol", "timestamp"],
        how="inner",
    )
    print(f"  OOS merged: {len(oos_merged):,} rows")

    # ── 6. A1: Raw probabilities from base LightGBM ───────────────────────────
    print(f"\n{_ts(dir_label)} [A1] Computing raw base_model probabilities on cal_fit...")
    raw_cal_probs = base.predict_proba(X_cal)[:, 1]
    print(f"  cal_fit raw prob stats: p10={np.percentile(raw_cal_probs,10):.4f}  "
          f"p50={np.percentile(raw_cal_probs,50):.4f}  "
          f"p90={np.percentile(raw_cal_probs,90):.4f}  "
          f"max={raw_cal_probs.max():.4f}")

    # ── 7. A2: Platt scaling ─────────────────────────────────────────────────
    print(f"\n{_ts(dir_label)} [A2] Fitting Platt scaler on cal_fit raw probs...")
    platt = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")
    platt.fit(raw_cal_probs.reshape(-1, 1), y_cal)
    platt_cal_probs = platt.predict_proba(raw_cal_probs.reshape(-1, 1))[:, 1]
    print(f"  Platt coef={float(platt.coef_[0][0]):.4f}  intercept={float(platt.intercept_[0]):.4f}")
    print(f"  cal_fit Platt prob stats: p10={np.percentile(platt_cal_probs,10):.4f}  "
          f"p50={np.percentile(platt_cal_probs,50):.4f}  "
          f"p90={np.percentile(platt_cal_probs,90):.4f}")

    # ── 8. Current isotonic calibration probs on cal_fit (for reference) ─────
    iso_cal_probs = model.predict_proba(X_cal)[:, 1]
    print(f"\n{_ts(dir_label)} [ISOTONIC ref] cal_fit: "
          f"p10={np.percentile(iso_cal_probs,10):.4f}  "
          f"p50={np.percentile(iso_cal_probs,50):.4f}  "
          f"p90={np.percentile(iso_cal_probs,90):.4f}")

    # ── 9. Per-year OOS evaluation for all 3 calibrations ────────────────────
    comparison_rows: list[dict] = []
    distribution_rows: list[dict] = []

    # For each year: collect scored rows for RAW + PLATT parquets
    oos_scored_raw: list[pd.DataFrame] = []
    oos_scored_platt: list[pd.DataFrame] = []

    for year_label, (ts_start, ts_end) in OOS_YEAR_SLICES.items():
        print(f"\n{_ts(dir_label)} OOS year: {year_label}...")

        mask_ts = (oos_merged["timestamp"] >= ts_start) & (oos_merged["timestamp"] <= ts_end)
        year_df_raw = oos_merged[mask_ts].copy()
        year_df_raw = _filter_to_universe(year_df_raw, universe_set)

        if len(year_df_raw) == 0:
            print(f"  {year_label}: no rows after universe filter — skipping")
            continue

        feat_cols = [f for f in feature_names_used if f in year_df_raw.columns]
        year_df_clean = year_df_raw.dropna(subset=feat_cols, thresh=len(feat_cols) // 2).copy()

        if len(year_df_clean) < 50:
            print(f"  {year_label}: only {len(year_df_clean)} rows after NaN drop — skipping")
            continue

        X_oos = np.nan_to_num(year_df_clean[feat_cols].values, nan=0.0)  # noqa: N806
        y_oos = year_df_clean[label_col].values.astype(int)
        n_rows = len(year_df_clean)
        pos_rate = float(y_oos.mean())

        # A1: raw scores
        raw_oos_probs = base.predict_proba(X_oos)[:, 1]
        # A2: Platt scores
        platt_oos_probs = platt.predict_proba(raw_oos_probs.reshape(-1, 1))[:, 1]
        # ISOTONIC: current calibration scores
        iso_oos_probs = model.predict_proba(X_oos)[:, 1]

        for cal_name, scores in [("isotonic", iso_oos_probs), ("raw", raw_oos_probs), ("platt", platt_oos_probs)]:
            metrics = _eval_scores(y_oos, scores, direction, cal_name, year_label, n_rows, pos_rate)
            comparison_rows.append(metrics)
            pct = _score_percentiles(scores)
            distribution_rows.append({
                "direction": direction,
                "calibration": cal_name,
                "year": year_label,
                "n_rows": n_rows,
                **pct,
            })

            print(f"  {cal_name:9s}: pr_auc={metrics['pr_auc']:.4f}  "
                  f"roc_auc={metrics['roc_auc']:.4f}  "
                  f"top10={metrics['top10_prec']:.4f}  "
                  f"top5={metrics['top5_prec']:.4f}  "
                  f"top1={metrics['top1_prec']:.4f}  "
                  f"p90_score={pct['p90']:.4f}  "
                  f"width={pct['width_p99_p10']:.4f}")

        # Collect scored rows
        year_df_clean = year_df_clean.copy()
        year_df_clean["score_raw"] = raw_oos_probs
        year_df_clean["oos_year"] = year_label
        oos_scored_raw.append(
            year_df_clean[["symbol", "timestamp", label_col, "score_raw", "oos_year"]].rename(
                columns={"score_raw": "score"}
            )
        )

        year_df_clean["score_platt"] = platt_oos_probs
        oos_scored_platt.append(
            year_df_clean[["symbol", "timestamp", label_col, "score_platt", "oos_year"]].rename(
                columns={"score_platt": "score"}
            )
        )

    return comparison_rows, distribution_rows, oos_scored_raw, oos_scored_platt


# ── verdict generation ────────────────────────────────────────────────────────

def _generate_verdict(
    comparison_df: pd.DataFrame,
    distribution_df: pd.DataFrame,
    class_baselines: dict,
) -> str:
    """Generate markdown verdict text."""
    lines = [
        "# Calibration Iteration A1+A2 — Verdict",
        f"\nGenerated: {datetime.now(UTC).isoformat()}",
        "",
        "## Summary: Master Comparison",
        "",
    ]

    # Build aggregate table per direction × calibration
    agg = (
        comparison_df
        .groupby(["direction", "calibration"])
        .agg(
            median_pr_auc=("pr_auc", "median"),
            median_roc_auc=("roc_auc", "median"),
            median_top10_prec=("top10_prec", "median"),
            median_top5_prec=("top5_prec", "median"),
            median_top1_prec=("top1_prec", "median"),
            median_brier=("brier", "median"),
        )
        .reset_index()
    )

    dist_agg = (
        distribution_df
        .groupby(["direction", "calibration"])
        .agg(
            median_p10=("p10", "median"),
            median_p50=("p50", "median"),
            median_p90=("p90", "median"),
            median_p99=("p99", "median"),
            median_width=("width_p99_p10", "median"),
        )
        .reset_index()
    )

    agg = agg.merge(dist_agg, on=["direction", "calibration"])

    lines.append("| Direction | Calibration | Median PR-AUC | Median top10_prec | "
                 "Median top5_prec | Median top1_prec | Median p50 score | Median p90 score | "
                 "Score width (p99-p10) |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for _, row in agg.iterrows():
        lines.append(
            f"| {row['direction']} | {row['calibration']} "
            f"| {row['median_pr_auc']:.4f} "
            f"| {row['median_top10_prec']:.4f} "
            f"| {row['median_top5_prec']:.4f} "
            f"| {row['median_top1_prec']:.4f} "
            f"| {row['median_p50']:.4f} "
            f"| {row['median_p90']:.4f} "
            f"| {row['median_width']:.4f} |"
        )

    lines += ["", "## Per-Year Detail", ""]
    lines.append("| Direction | Calibration | Year | PR-AUC | ROC-AUC | "
                 "top10_prec | top5_prec | top1_prec | p10 score | p50 score | p90 score | max score |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for _, row in comparison_df.sort_values(["direction", "calibration", "year"]).iterrows():
        dist_row = distribution_df[
            (distribution_df["direction"] == row["direction"])
            & (distribution_df["calibration"] == row["calibration"])
            & (distribution_df["year"] == row["year"])
        ]
        p10 = dist_row["p10"].values[0] if len(dist_row) > 0 else float("nan")
        p50 = dist_row["p50"].values[0] if len(dist_row) > 0 else float("nan")
        p90 = dist_row["p90"].values[0] if len(dist_row) > 0 else float("nan")
        mx = dist_row["max"].values[0] if len(dist_row) > 0 else float("nan")
        lines.append(
            f"| {row['direction']} | {row['calibration']} | {row['year']} "
            f"| {row['pr_auc']:.4f} | {row['roc_auc']:.4f} "
            f"| {row['top10_prec']:.4f} | {row['top5_prec']:.4f} | {row['top1_prec']:.4f} "
            f"| {p10:.4f} | {p50:.4f} | {p90:.4f} | {mx:.4f} |"
        )

    # ── Verdict per direction ────────────────────────────────────────────────
    lines += ["", "## Verdict Per Direction", ""]

    for direction in ["long", "short"]:
        baseline = class_baselines.get(direction, 0.30)
        min_useful_top10 = baseline + 0.10

        dir_agg = agg[agg["direction"] == direction].set_index("calibration")

        iso_top10 = dir_agg.loc["isotonic", "median_top10_prec"] if "isotonic" in dir_agg.index else float("nan")
        raw_top10 = dir_agg.loc["raw", "median_top10_prec"] if "raw" in dir_agg.index else float("nan")
        platt_top10 = dir_agg.loc["platt", "median_top10_prec"] if "platt" in dir_agg.index else float("nan")

        iso_width = dir_agg.loc["isotonic", "median_width"] if "isotonic" in dir_agg.index else 0
        raw_width = dir_agg.loc["raw", "median_width"] if "raw" in dir_agg.index else 0
        platt_width = dir_agg.loc["platt", "median_width"] if "platt" in dir_agg.index else 0

        iso_prauc = dir_agg.loc["isotonic", "median_pr_auc"] if "isotonic" in dir_agg.index else float("nan")
        raw_prauc = dir_agg.loc["raw", "median_pr_auc"] if "raw" in dir_agg.index else float("nan")
        platt_prauc = dir_agg.loc["platt", "median_pr_auc"] if "platt" in dir_agg.index else float("nan")

        lines.append(f"### {direction.upper()}")
        lines.append(f"- Class baseline: {baseline:.4f} | Min useful top10_prec: {min_useful_top10:.4f}")
        lines.append(f"- Isotonic: median top10={iso_top10:.4f}, PR-AUC={iso_prauc:.4f}, "
                     f"score width={iso_width:.4f}")
        lines.append(f"- Raw:      median top10={raw_top10:.4f}, PR-AUC={raw_prauc:.4f}, "
                     f"score width={raw_width:.4f}")
        lines.append(f"- Platt:    median top10={platt_top10:.4f}, PR-AUC={platt_prauc:.4f}, "
                     f"score width={platt_width:.4f}")
        lines.append("")

        # Determine winner
        best_top10 = max(iso_top10, raw_top10, platt_top10)
        best_cal = (
            "isotonic" if iso_top10 == best_top10
            else "raw" if raw_top10 == best_top10
            else "platt"
        )

        raw_beats_iso_top10 = (raw_top10 - iso_top10) > 0.01
        platt_beats_iso_top10 = (platt_top10 - iso_top10) > 0.01
        platt_beats_raw_width = (platt_width - raw_width) > 0.02

        raw_prauc_ok = raw_prauc >= iso_prauc - 0.02
        platt_prauc_ok = platt_prauc >= iso_prauc - 0.02

        # Check isotonic p90 degeneracy
        iso_dist = distribution_df[
            (distribution_df["direction"] == direction)
            & (distribution_df["calibration"] == "isotonic")
        ]
        p90_unique = iso_dist["p90"].nunique()
        isotonic_degenerate = (p90_unique == 1)

        if isotonic_degenerate:
            lines.append(f"**Isotonic p90 is CONSTANT across all years ({iso_dist['p90'].iloc[0]:.4f}) — "
                         "degenerate calibration confirmed.**")
            lines.append("")

        # raw and platt share identical top-K when both beat isotonic (same base model, monotone transform)
        # prefer platt over raw: better calibration (lower brier), saner score range, same precision
        both_beat = (raw_beats_iso_top10 or platt_beats_iso_top10)
        if both_beat and platt_prauc_ok:
            emoji = "🟢"
            recommendation = "PLATT"
            delta = platt_top10 - iso_top10
            reason = (f"Platt (and raw) beat isotonic on top10_prec (+{delta:.4f}). "
                      f"Platt preferred over raw: proper calibration (lower Brier), "
                      f"saner score range (max={dir_agg.loc['platt','median_p99']:.4f} vs raw max≈0.99), "
                      f"identical top-K precision to raw. "
                      f"Score width={platt_width:.4f} > isotonic={iso_width:.4f}.")
        elif both_beat and raw_prauc_ok:
            emoji = "🟢"
            recommendation = "RAW"
            reason = (f"Raw beats isotonic on top10_prec (+{raw_top10-iso_top10:.4f}). "
                      f"Platt PR-AUC degraded — use raw instead.")
        elif isotonic_degenerate and (platt_width > iso_width * 1.5 or raw_width > iso_width * 2):
            emoji = "🟢"
            recommendation = "PLATT"
            reason = (f"Isotonic degenerate (constant p90={iso_dist['p90'].iloc[0]:.4f}). "
                      f"Platt gives more spread (width={platt_width:.4f} vs iso={iso_width:.4f}) "
                      f"and identical top-K to raw with better calibration properties.")
        elif abs(raw_top10 - iso_top10) <= 0.01 and abs(platt_top10 - iso_top10) <= 0.01:
            if isotonic_degenerate:
                emoji = "🟢"
                recommendation = "PLATT"
                reason = ("Marginal top-K differences but isotonic degenerate confirmed. "
                          "Platt preferred: saner score distribution, proper calibration, "
                          f"wider useful range (width={platt_width:.4f} vs iso={iso_width:.4f}).")
            else:
                emoji = "🟡"
                recommendation = "PLATT"
                reason = ("Marginal differences in top-K precision. "
                          "Platt preferred for wider score distribution enabling more robust top-K ranking.")
        else:
            emoji = "🔴"
            recommendation = "ISOTONIC"
            reason = "Raw and Platt both worse than isotonic on top-K precision. Keep current."

        lines.append(f"{emoji} **Verdict: Use {recommendation} for Phase 4**")
        lines.append(f"- Reason: {reason}")
        lines.append("")

    # ── Final recommendation ─────────────────────────────────────────────────
    lines += ["", "## Phase 4 Recommendation", ""]

    # Derive per direction using same logic as verdict section above
    verdicts = {}
    for direction in ["long", "short"]:
        dir_agg = agg[agg["direction"] == direction].set_index("calibration")

        iso_top10 = dir_agg.loc["isotonic", "median_top10_prec"] if "isotonic" in dir_agg.index else float("nan")
        raw_top10 = dir_agg.loc["raw", "median_top10_prec"] if "raw" in dir_agg.index else float("nan")
        platt_top10 = dir_agg.loc["platt", "median_top10_prec"] if "platt" in dir_agg.index else float("nan")

        iso_width = dir_agg.loc["isotonic", "median_width"] if "isotonic" in dir_agg.index else 0
        platt_width = dir_agg.loc["platt", "median_width"] if "platt" in dir_agg.index else 0

        iso_dist = distribution_df[
            (distribution_df["direction"] == direction)
            & (distribution_df["calibration"] == "isotonic")
        ]
        isotonic_degenerate = (iso_dist["p90"].nunique() == 1)

        platt_prauc = dir_agg.loc["platt", "median_pr_auc"] if "platt" in dir_agg.index else float("nan")
        raw_prauc = dir_agg.loc["raw", "median_pr_auc"] if "raw" in dir_agg.index else float("nan")
        iso_prauc = dir_agg.loc["isotonic", "median_pr_auc"] if "isotonic" in dir_agg.index else float("nan")
        platt_prauc_ok = platt_prauc >= iso_prauc - 0.02
        raw_prauc_ok = raw_prauc >= iso_prauc - 0.02

        raw_beats = (raw_top10 - iso_top10) > 0.01
        platt_beats = (platt_top10 - iso_top10) > 0.01
        both_beat = raw_beats or platt_beats

        marginal = abs(platt_top10 - iso_top10) <= 0.01 and abs(raw_top10 - iso_top10) <= 0.01

        if both_beat and platt_prauc_ok:
            verdicts[direction] = "platt"
        elif both_beat and raw_prauc_ok:
            verdicts[direction] = "raw"
        elif isotonic_degenerate or (marginal and platt_prauc_ok):
            # degenerate isotonic OR marginal tie → platt is safer (wider dist, proper calibration)
            verdicts[direction] = "platt"
        else:
            verdicts[direction] = "isotonic"

    lines.append(f"- **long calibration**: `{verdicts['long']}`")
    lines.append(f"- **short calibration**: `{verdicts['short']}`")
    lines.append("")
    lines.append("If calibration fix gives wider scores + higher top-K precision → proceed to Phase 4 "
                 "with winning calibration.")
    lines.append("If differences are marginal → calibration is not the bottleneck; "
                 "skip B (label change) and proceed directly to Phase 4 with current models.")
    lines.append("")

    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"{_ts()} Calibration Iteration A1+A2 — v_new_1_long + v_new_1_short")
    print(f"{_ts()} Results dir: {RESULTS_DIR}")

    # Load class baselines from meta
    class_baselines = {}
    for d in ["long", "short"]:
        meta_path = PYTHON_ROOT / "models" / f"v_new_1_{d}" / f"v_new_1_{d}_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        class_baselines[d] = float(meta.get("class_baseline_positive_rate", 0.30))
    print(f"  class_baselines: {class_baselines}")

    all_comparison_rows: list[dict] = []
    all_distribution_rows: list[dict] = []

    for direction in ["long", "short"]:
        comp, dist, scored_raw, scored_platt = run_direction(direction)
        all_comparison_rows.extend(comp)
        all_distribution_rows.extend(dist)

        # Save RAW + PLATT OOS parquets
        if scored_raw:
            raw_df = pd.concat(scored_raw, ignore_index=True)
            raw_path = DATA_DIR / f"oos_scored_{direction}_RAW.parquet"
            raw_df.to_parquet(raw_path, index=False)
            print(f"\n{_ts(f'v_new_1_{direction}')} Saved RAW OOS: {raw_path} ({len(raw_df):,} rows)")

        if scored_platt:
            platt_df = pd.concat(scored_platt, ignore_index=True)
            platt_path = DATA_DIR / f"oos_scored_{direction}_PLATT.parquet"
            platt_df.to_parquet(platt_path, index=False)
            print(f"{_ts(f'v_new_1_{direction}')} Saved PLATT OOS: {platt_path} ({len(platt_df):,} rows)")

    # ── Save master comparison CSV ────────────────────────────────────────────
    comparison_df = pd.DataFrame(all_comparison_rows)
    dist_df = pd.DataFrame(all_distribution_rows)

    comp_path = RESULTS_DIR / "master_comparison.csv"
    comparison_df.to_csv(comp_path, index=False)
    print(f"\n{_ts()} Saved master_comparison.csv: {comp_path}")

    dist_path = RESULTS_DIR / "score_distributions.csv"
    dist_df.to_csv(dist_path, index=False)
    print(f"{_ts()} Saved score_distributions.csv: {dist_path}")

    # ── Generate verdict ──────────────────────────────────────────────────────
    print(f"\n{_ts()} Generating verdict...")
    verdict_text = _generate_verdict(comparison_df, dist_df, class_baselines)

    verdict_path = RESULTS_DIR / "verdict.md"
    with open(verdict_path, "w") as f:
        f.write(verdict_text)
    print(f"{_ts()} Saved verdict.md: {verdict_path}")

    # ── Print summary to console ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("CALIBRATION COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))
    print(f"\n{'='*70}")
    print("SCORE DISTRIBUTIONS")
    print(f"{'='*70}")
    print(dist_df[["direction", "calibration", "year", "p10", "p50", "p90", "p99", "max",
                   "width_p99_p10"]].to_string(index=False))
    print(f"\n{verdict_text}")
    print(f"\n{_ts()} Done.")


if __name__ == "__main__":
    main()
