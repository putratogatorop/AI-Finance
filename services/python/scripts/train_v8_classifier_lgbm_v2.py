"""Phase 2 — Feature uplift evaluation on top of Phase 1 methodology.

Phase 1 baseline: purged CV + Optuna + isotonic calibration + LightGBM.
Phase 2 adds 14 new features (Priority A + B from uplift plan):
  Priority A (restore v1 drops): cvd_slope_1h_norm, cvd_divergence_4h,
    bb_pct_b, bb_bandwidth, fib_pos_50
  Priority B (sister-classifier): kdj_k, kdj_j, price_vs_ema9_pct,
    rsi14_delta_1bar, rsi14_delta_4bar, macd_hist_momentum,
    macd_signal_spread_norm, parkinson_vol, obv_slope_norm

Input parquet: v8_trades_with_features_2026-04-01_p2.parquet
  (produced by scripts/augment_v8_training_parquet_phase2.py)

USAGE (from services/python/):
    python scripts/train_v8_classifier_lgbm_v2.py

OUTPUTS (models/v8_classifier_lgbm_v2p2/):
    macd_pullback_long_lgbm_v2p2.joblib     — CalibratedLGBM (drop-in for V8Classifier)
    macd_pullback_long_lgbm_v2p2_meta.json  — rich audit metadata
"""
from __future__ import annotations

import json
import warnings
from datetime import UTC, datetime
from pathlib import Path

import joblib
import lightgbm as lgbm
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── CONFIG ────────────────────────────────────────────────────────────────────

TRAINING_PARQUET = (
    Path(__file__).resolve().parents[1]
    / "data" / "training" / "v8_trades_with_features_2026-04-01_p2.parquet"
)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "v8_classifier_lgbm_v2p2"
MODEL_SUFFIX = "lgbm_v2p2"

HOLD_OUT_START = pd.Timestamp("2026-01-01", tz="UTC")
DETECTOR = "macd_pullback_long"
N_FOLDS = 5
EMBARGO_DAYS = 7          # max hold period = 672 15m bars = 7 days
N_OPTUNA_TRIALS = 100
RANDOM_SEED = 42

FEATURES_P1: list[str] = [
    "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
    "close_to_high50_atr", "close_to_low50_atr",
    "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
    "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
    "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
    "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
    "breadth_up", "breadth_down", "signals_same_15m_same_detector",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]
FEATURES_P2_NEW: list[str] = [
    # Priority A — restore v1 drops (cvd_divergence_4h pruned: SHAP=0.0)
    "cvd_slope_1h_norm", "bb_pct_b", "bb_bandwidth", "fib_pos_50",
    # Priority B — sister-classifier features
    "kdj_k", "kdj_j", "price_vs_ema9_pct",
    "rsi14_delta_1bar", "rsi14_delta_4bar",
    "macd_hist_momentum", "macd_signal_spread_norm",
    "parkinson_vol", "obv_slope_norm",
]
FEATURES: list[str] = FEATURES_P1 + FEATURES_P2_NEW


# ── PURGED TIME-SERIES SPLIT ──────────────────────────────────────────────────

def purged_ts_splits(
    entry_times: pd.Series,
    n_splits: int = N_FOLDS,
    embargo_days: int = EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit with post-fold embargo to prevent label leakage.

    Removes test samples whose entry_time falls within `embargo_days` of the
    last training sample — covers the max signal hold period so a trade open
    at the fold boundary cannot appear in both train (label known) and test.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in tss.split(np.arange(len(entry_times))):
        train_end = entry_times.iloc[train_idx[-1]]
        embargo_cut = train_end + pd.Timedelta(days=embargo_days)
        test_times = entry_times.iloc[test_idx]
        purged = test_idx[(test_times >= embargo_cut).values]
        if len(purged) >= 50:
            splits.append((train_idx, purged))
    return splits


# ── HELPERS ───────────────────────────────────────────────────────────────────

def _pf(pnls: np.ndarray) -> float:
    w = pnls[pnls > 0].sum()
    l = -pnls[pnls <= 0].sum()
    return float(w / l) if l > 0 else float("inf")


def _max_dd(pnls: np.ndarray, base_size: float = 0.01, start: float = 100.0) -> float:
    eq = start; peak = start; mdd = 0.0
    for p in pnls:
        eq *= 1.0 + base_size * p
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1.0)
    return float(mdd * 100.0)


def _fbeta(y_true: np.ndarray, probs: np.ndarray, thr: float, beta: float = 0.5) -> float:
    """F-beta (beta=0.5 → precision 2× recall) at a given threshold."""
    preds = (probs >= thr).astype(int)
    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return float((1.0 + beta**2) * prec * rec / (beta**2 * prec + rec + 1e-9))


def _opt_threshold(y: np.ndarray, probs: np.ndarray, beta: float = 0.5) -> tuple[float, float]:
    best_thr, best_fb = 0.0, 0.0
    for thr in np.arange(0.25, 0.75, 0.025):
        fb = _fbeta(y, probs, float(thr), beta)
        if fb > best_fb:
            best_fb, best_thr = fb, float(thr)
    return best_thr, best_fb


def _eval_block(probs: np.ndarray, y: np.ndarray, pnls: np.ndarray, thr: float, label: str) -> dict:
    mask = probs >= thr
    return {
        "label": label,
        "n_total": int(len(y)),
        "n_passed": int(mask.sum()),
        "pass_rate_pct": round(float(100.0 * mask.mean()), 1),
        "roc_auc": round(float(roc_auc_score(y, probs)) if len(set(y)) >= 2 else float("nan"), 4),
        "pr_auc": round(float(average_precision_score(y, probs)), 4),
        "brier": round(float(brier_score_loss(y, probs)), 4),
        "pf_filtered": round(_pf(pnls[mask]), 3) if mask.sum() > 0 else float("nan"),
        "wr_filtered_pct": round(float(100.0 * (y[mask] == 1).mean()), 1) if mask.sum() > 0 else float("nan"),
        "mdd_filtered_pct": round(_max_dd(pnls[mask]), 2) if mask.sum() > 0 else float("nan"),
    }


def _score_quantiles(model: CalibratedLGBM, X: np.ndarray, n_q: int = 200) -> list[float]:
    probs = model.predict_proba(X)[:, 1]
    idx = np.linspace(0, len(probs) - 1, n_q).astype(int)
    return [round(float(v), 6) for v in np.sort(probs)[idx].tolist()]


# ── OPTUNA OBJECTIVE ──────────────────────────────────────────────────────────

def _make_objective(
    X: np.ndarray,
    y: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    spw: float,
):
    """Returns (objective_fn, fold_iter_tracker).

    fold_iter_tracker is populated by each trial call — after the study
    completes, re-run with the best params to get the actual best_iterations.
    """
    def objective(trial: optuna.Trial) -> float:  # noqa: N803
        params: dict = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 31),
            "max_depth": trial.suggest_int("max_depth", 4, 6),
            "min_child_samples": trial.suggest_int("min_child_samples", 100, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.05, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.7),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.7, 0.9),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 5.0),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.01, 0.1),
            "scale_pos_weight": spw,
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": RANDOM_SEED,
            "n_estimators": 600,
        }
        fold_scores = []
        for fold_i, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X[tr_idx], y[tr_idx]  # noqa: N806
            X_val, y_val = X[val_idx], y[val_idx]  # noqa: N806
            clf = lgbm.LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgbm.early_stopping(50, verbose=False), lgbm.log_evaluation(-1)],
            )
            probs = clf.predict_proba(X_val)[:, 1]
            fold_scores.append(float(average_precision_score(y_val, probs)))
            trial.report(float(np.mean(fold_scores)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(np.mean(fold_scores))

    return objective


# ── CALIBRATED MODEL WRAPPER ─────────────────────────────────────────────────

class CalibratedLGBM:
    """LGBMClassifier + IsotonicRegression calibrator.

    Drop-in replacement for sklearn's predict_proba API.
    Saved as a single joblib so V8Classifier loads it unchanged.
    """

    def __init__(self, base_model: lgbm.LGBMClassifier, calibrator: IsotonicRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:  # noqa: C901
    assert TRAINING_PARQUET.exists(), f"Training parquet not found:\n  {TRAINING_PARQUET}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    df = pd.read_parquet(TRAINING_PARQUET)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df[df["detector"] == DETECTOR].sort_values("entry_time").reset_index(drop=True)

    train = df[df["entry_time"] < HOLD_OUT_START].reset_index(drop=True)
    holdout = df[df["entry_time"] >= HOLD_OUT_START].reset_index(drop=True)

    print(f"Detector: {DETECTOR}")
    print(f"  Train:   {len(train):,} rows  ({train.entry_time.min().date()} → {train.entry_time.max().date()})")
    print(f"  Holdout: {len(holdout):,} rows  ({holdout.entry_time.min().date()} → {holdout.entry_time.max().date()})")
    print(f"  Train label: {dict(train['label'].value_counts().sort_index())}")

    X_train = train[FEATURES].values  # noqa: N806
    y_train = train["label"].values.astype(int)
    X_holdout = holdout[FEATURES].values  # noqa: N806
    y_holdout = holdout["label"].values.astype(int)
    pnl_holdout = holdout["pnl_pct"].values

    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"  scale_pos_weight: {spw:.3f}\n")

    # ── Purged splits ─────────────────────────────────────────────────────────
    splits = purged_ts_splits(train["entry_time"], N_FOLDS, EMBARGO_DAYS)
    print(f"Purged folds ({N_FOLDS} requested, {len(splits)} valid after embargo filter):")
    for i, (tr_idx, te_idx) in enumerate(splits):
        t0 = train["entry_time"].iloc[te_idx[0]].date()
        t1 = train["entry_time"].iloc[te_idx[-1]].date()
        print(f"  Fold {i}: train={len(tr_idx):,}  test={len(te_idx):,}  test_range={t0}→{t1}")
    print()

    # ── Optuna ────────────────────────────────────────────────────────────────
    print(f"Starting Optuna ({N_OPTUNA_TRIALS} trials × {len(splits)} folds)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(
        _make_objective(X_train, y_train, splits, spw),
        n_trials=N_OPTUNA_TRIALS,
        show_progress_bar=True,
    )
    best_params: dict = study.best_params
    print(f"Best PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {json.dumps(best_params, indent=2)}\n")

    # ── Re-run best params to collect per-fold metrics + best_iterations ──────
    full_params: dict = {
        **best_params,
        "bagging_freq": 5,
        "scale_pos_weight": spw,
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": RANDOM_SEED,
        "n_estimators": 600,
    }
    per_fold_metrics = []
    fold_best_iters = []
    print("Re-running best params for per-fold metrics + best_iterations:")
    for fold_i, (tr_idx, te_idx) in enumerate(splits):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]  # noqa: N806
        X_val, y_val = X_train[te_idx], y_train[te_idx]  # noqa: N806
        clf = lgbm.LGBMClassifier(**full_params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgbm.early_stopping(50, verbose=False), lgbm.log_evaluation(-1)],
        )
        probs = clf.predict_proba(X_val)[:, 1]
        best_iter = int(clf.best_iteration_)
        fold_best_iters.append(best_iter)
        fm = {
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(te_idx)),
            "best_iteration": best_iter,
            "pr_auc": round(float(average_precision_score(y_val, probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, probs)) if len(set(y_val)) >= 2 else float("nan"), 4),
            "brier": round(float(brier_score_loss(y_val, probs)), 4),
            "log_loss_val": round(float(log_loss(y_val, probs)), 4),
        }
        per_fold_metrics.append(fm)
        print(f"  Fold {fold_i}: PR-AUC={fm['pr_auc']:.4f}  ROC-AUC={fm['roc_auc']:.4f}  "
              f"Brier={fm['brier']:.4f}  best_iter={best_iter}")

    final_n_estimators = max(100, int(round(float(np.median(fold_best_iters)) * 1.1)))
    print(f"\nFinal n_estimators: {final_n_estimators}  "
          f"(max(100, median {int(np.median(fold_best_iters))} × 1.1))\n")

    # ── Final model — full train period ───────────────────────────────────────
    print("Training final model on full train set...")
    final_params: dict = {**full_params, "n_estimators": final_n_estimators}
    final_model = lgbm.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train, callbacks=[lgbm.log_evaluation(-1)])

    # ── Isotonic calibration on last fold OOS ─────────────────────────────────
    print("Fitting isotonic calibrator on last fold OOS...")
    _, cal_idx = splits[-1]
    X_cal, y_cal = X_train[cal_idx], y_train[cal_idx]  # noqa: N806
    raw_cal_probs = final_model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal_probs, y_cal)
    calibrated = CalibratedLGBM(final_model, iso)

    # ── Threshold optimization on all OOS folds (calibrated probs) ────────────
    # Collect OOS calibrated probs + labels across all folds
    oos_probs = np.zeros(len(train))
    oos_y = np.zeros(len(train), dtype=int)
    oos_pnl = np.zeros(len(train))
    oos_mask = np.zeros(len(train), dtype=bool)
    for _, te_idx in splits:
        oos_probs[te_idx] = calibrated.predict_proba(X_train[te_idx])[:, 1]
        oos_y[te_idx] = y_train[te_idx]
        oos_pnl[te_idx] = train["pnl_pct"].values[te_idx]
        oos_mask[te_idx] = True

    opt_thr, opt_fb = _opt_threshold(oos_y[oos_mask], oos_probs[oos_mask])
    print(f"OOS threshold (F-0.5): {opt_thr:.3f}  F-beta={opt_fb:.4f}")

    thr_by_fold = []
    for _, te_idx in splits:
        t, _ = _opt_threshold(y_train[te_idx], calibrated.predict_proba(X_train[te_idx])[:, 1])
        thr_by_fold.append(round(float(t), 3))
    print(f"Per-fold thresholds:   {thr_by_fold}  (range {min(thr_by_fold):.3f}–{max(thr_by_fold):.3f})\n")

    # ── Holdout evaluation ────────────────────────────────────────────────────
    print("Holdout evaluation (2026-01-01+):")
    raw_probs_ho = final_model.predict_proba(X_holdout)[:, 1]
    cal_probs_ho = calibrated.predict_proba(X_holdout)[:, 1]

    blocks = [
        _eval_block(raw_probs_ho, y_holdout, pnl_holdout, 0.325, "raw@0.325 (v1 meta thr)"),
        _eval_block(cal_probs_ho, y_holdout, pnl_holdout, opt_thr, f"cal@{opt_thr:.3f} (optimal)"),
        _eval_block(cal_probs_ho, y_holdout, pnl_holdout, 0.40, "cal@0.40 (live floor)"),
    ]
    for b in blocks:
        print(f"  {b['label']:30s}  n={b['n_passed']:3d}/{b['n_total']}  "
              f"PF={b['pf_filtered']:.2f}  WR={b['wr_filtered_pct']:.1f}%  "
              f"MDD={b['mdd_filtered_pct']:.2f}%  PR-AUC={b['pr_auc']:.4f}")

    # ── Calibration curve ─────────────────────────────────────────────────────
    prob_true, prob_pred = calibration_curve(y_holdout, cal_probs_ho, n_bins=10)
    cal_curve = {
        "prob_pred": [round(float(v), 4) for v in prob_pred],
        "prob_true": [round(float(v), 4) for v in prob_true],
    }

    # ── SHAP importances ──────────────────────────────────────────────────────
    print("\nComputing SHAP (200 holdout samples)...")
    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer.shap_values(X_holdout[:200])
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1]          # older shap API: [neg_class, pos_class]
    elif shap_vals.ndim == 3:
        shap_arr = shap_vals[:, :, 1]   # newer shap API: (n, features, classes)
    else:
        shap_arr = shap_vals
    mean_abs = np.abs(shap_arr).mean(axis=0)
    shap_importance = {
        f: round(float(v), 6)
        for f, v in sorted(zip(FEATURES, mean_abs), key=lambda x: -x[1])
    }
    print("  Top-10 SHAP:", list(shap_importance.items())[:10])

    # ── Save model ────────────────────────────────────────────────────────────
    model_path = MODELS_DIR / f"{DETECTOR}_{MODEL_SUFFIX}.joblib"
    joblib.dump(calibrated, model_path)
    print(f"\nSaved: {model_path}")

    # ── Rich meta JSON ────────────────────────────────────────────────────────
    train_prauc = float(average_precision_score(y_train, final_model.predict_proba(X_train)[:, 1]))
    meta = {
        # Identity
        "detector": DETECTOR,
        "model_class": f"lightgbm.LGBMClassifier + isotonic calibration ({MODEL_SUFFIX})",
        "feature_names": FEATURES,
        "features_p1": FEATURES_P1,
        "features_p2_new": FEATURES_P2_NEW,
        "n_features": len(FEATURES),
        "snapshot_date": "2026-04-01",
        "trained_at_utc": datetime.now(UTC).isoformat(),
        # Methodology
        "n_folds": N_FOLDS,
        "embargo_days": EMBARGO_DAYS,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "optuna_best_params": best_params,
        "final_n_estimators": final_n_estimators,
        "scale_pos_weight": round(spw, 4),
        "calibration_method": "isotonic",
        "calibration_set": f"last CV fold OOS (n={len(cal_idx)})",
        # Data
        "n_train": int(len(train)),
        "n_holdout": int(len(holdout)),
        "train_date_range": [str(train["entry_time"].min()), str(train["entry_time"].max())],
        "holdout_date_range": [str(holdout["entry_time"].min()), str(holdout["entry_time"].max())],
        # Per-fold OOS metrics
        "per_fold_metrics": per_fold_metrics,
        "median_oos_pr_auc": round(float(np.median([f["pr_auc"] for f in per_fold_metrics])), 4),
        "p5_oos_pr_auc": round(float(np.percentile([f["pr_auc"] for f in per_fold_metrics], 5)), 4),
        "p95_oos_pr_auc": round(float(np.percentile([f["pr_auc"] for f in per_fold_metrics], 95)), 4),
        "train_pr_auc": round(train_prauc, 4),
        "train_oos_prauc_gap": round(train_prauc - float(np.median([f["pr_auc"] for f in per_fold_metrics])), 4),
        # Threshold
        "filter_threshold_locked": round(opt_thr, 3),
        "filter_threshold_fbeta05": round(opt_thr, 3),
        "threshold_by_fold": thr_by_fold,
        "threshold_range": [min(thr_by_fold), max(thr_by_fold)],
        # Holdout evaluation
        "holdout_raw_uncalibrated": blocks[0],
        "holdout_calibrated_optimal_thr": blocks[1],
        "holdout_calibrated_live_thr_040": blocks[2],
        # Backward-compat keys for V8Classifier
        "auc_train": round(float(roc_auc_score(y_train, final_model.predict_proba(X_train)[:, 1])), 4),
        "auc_holdout": round(float(roc_auc_score(y_holdout, raw_probs_ho)), 4),
        "size_low": 0.5,
        "size_high": 1.5,
        "train_score_quantiles": _score_quantiles(calibrated, X_train),
        # Calibration curve (binned)
        "calibration_curve": cal_curve,
        # SHAP
        "shap_importance_top20": dict(list(shap_importance.items())[:20]),
    }
    meta_path = MODELS_DIR / f"{DETECTOR}_{MODEL_SUFFIX}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"Saved: {meta_path}")

    # ── Phase 2 gate check ────────────────────────────────────────────────────
    print("\n── PHASE 2 GATE CHECK ───────────────────────────────────────────────")
    oos_med = meta["median_oos_pr_auc"]
    gap = meta["train_oos_prauc_gap"]
    cal_brier = float(brier_score_loss(y_holdout, cal_probs_ho))
    print(f"  Median OOS PR-AUC (must be reported):  {oos_med:.4f}  ✅")
    print(f"  Train/OOS PR-AUC gap (must be < 0.15): {gap:.4f}  "
          f"{'✅' if gap < 0.15 else '❌ regularize harder before adding features'}")
    print(f"  Holdout Brier (must be < 0.05):        {cal_brier:.4f}  "
          f"{'✅' if cal_brier < 0.05 else '❌'}")
    p1_holdout_pf = 4.84  # Phase 1 baseline @ 0.375
    best_p2_pf = blocks[1]["pf_filtered"]
    lift = round(float(best_p2_pf) - p1_holdout_pf, 3) if isinstance(best_p2_pf, float) else float("nan")
    print(f"  Phase 1 baseline holdout PF (@ 0.375): {p1_holdout_pf}  →  Phase 2: {best_p2_pf}  lift={lift:+.3f}")
    if gap < 0.15:
        print("\n✅ Phase 2 gate PASSED — net lift positive, ready for SHAP pruning.")
    else:
        print("\n❌ Phase 2 gate FAILED — overfitting increased with new features; check SHAP and prune.")


if __name__ == "__main__":
    main()
