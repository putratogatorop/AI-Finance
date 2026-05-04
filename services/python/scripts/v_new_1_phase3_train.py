"""v_new_1 Phase 3 — BGM Baseline Training (both directions).

Trains v_new_1_long and v_new_1_short LightGBM classifiers with:
  - Purged 5-fold CV + Optuna (50 trials) on train_fit (2023-04-01 → 2024-09-30)
  - Isotonic calibrator fit on disjoint cal_fit (2024-10-01 → 2025-12-31)
  - Threshold locked on cal_fit by max(precision × min(0.5, recall))
  - Multi-OOS evaluation on 4 regime years (2020, 2021, 2022, 2026-Q1)
  - SHAP top-20 feature importances

USAGE (from repo root):
    DYLD_LIBRARY_PATH=$(services/python/.venv/bin/python3 -c \
        "import sklearn,os; print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))") \\
    nohup services/python/.venv/bin/python3 services/python/scripts/v_new_1_phase3_train.py \\
    > services/python/results/v_new_1_phase3/training.log 2>&1 &

OUTPUTS:
    services/python/models/v_new_1_long/v_new_1_long.joblib + _meta.json
    services/python/models/v_new_1_short/v_new_1_short.joblib + _meta.json
    services/python/data/v_new_1/oos_scored_long.parquet
    services/python/data/v_new_1/oos_scored_short.parquet
    services/python/data/v_new_1/oof_scored_long.parquet         (NEW — train_fit OOF preds for v_new_1.5)
    services/python/data/v_new_1/oof_scored_short.parquet        (NEW — train_fit OOF preds for v_new_1.5)
    services/python/data/v_new_1/cal_scored_long.parquet         (NEW — cal_fit raw preds, non-leaky)
    services/python/data/v_new_1/cal_scored_short.parquet        (NEW — cal_fit raw preds, non-leaky)
    services/python/results/v_new_1_phase3/training.log (from nohup redirect)
"""

from __future__ import annotations

import gc
import json
import os
import pathlib
import sys
import time
import warnings
from datetime import UTC, datetime

# ── libomp workaround for macOS (must happen before importing lightgbm) ──
_SKLEARN_DYLIBS = pathlib.Path(__file__).resolve().parents[1] / ".venv" / "lib" / "python3.12" / "site-packages" / "sklearn" / ".dylibs"
if _SKLEARN_DYLIBS.is_dir():
    os.environ.setdefault("DYLD_LIBRARY_PATH", str(_SKLEARN_DYLIBS))

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "services" / "python"
sys.path.insert(0, str(PYTHON_ROOT))

DATA_DIR = PYTHON_ROOT / "data" / "v_new_1"
V2P3_META_PATH = PYTHON_ROOT / "models" / "v8_classifier_lgbm_v2p3" / "macd_pullback_long_lgbm_v2p3_meta.json"

# ── config ───────────────────────────────────────────────────────────────────

RANDOM_SEED = 42
N_OPTUNA_TRIALS = int(os.environ.get("N_OPTUNA_TRIALS", "50"))   # override via env: e.g. 1 for smoke, 20 for fast
N_FOLDS = 5
EMBARGO_DAYS = 10           # 10-day embargo (covers the 7-day label horizon + buffer)
LGBM_NUM_THREADS = int(os.environ.get("LGBM_NUM_THREADS", "0"))  # 0 = LightGBM picks (all cores). Set to 4 on VPS to leave 4 cores for live trading.
SHAP_SUBSAMPLE = 10_000     # rows for SHAP computation (speed/memory balance)

# v_new_1.5 mode: when set, train on features_full_v1_5.parquet (44 base + 12
# sequence features) and write artifacts under v_new_1_{long,short}_v1_5.
V_NEW_1_5_MODE = os.environ.get("V_NEW_1_5_MODE", "0") == "1"
MODE_SUFFIX = "_v1_5" if V_NEW_1_5_MODE else ""
FEATURES_PARQUET_NAME = "features_full_v1_5.parquet" if V_NEW_1_5_MODE else "features_full.parquet"

V_NEW_1_5_SEQUENCE_FEATURES = [
    "score_long_t1", "score_long_t2", "score_long_t6",
    "score_long_roll_mean_3d", "score_long_roll_std_3d",
    "score_long_change_1d",
    "score_short_t1", "score_short_t2", "score_short_t6",
    "score_short_roll_mean_3d", "score_short_roll_std_3d",
    "score_short_change_1d",
]

# Chronological split boundaries (inclusive)
TRAIN_FIT_END = pd.Timestamp("2024-09-30 23:59:59", tz="UTC")
CAL_FIT_START = pd.Timestamp("2024-10-01", tz="UTC")

# OOS year slices
OOS_YEAR_SLICES = {
    "2020": (pd.Timestamp("2020-05-01", tz="UTC"), pd.Timestamp("2020-12-31 23:59:59", tz="UTC")),
    "2021": (pd.Timestamp("2021-01-01", tz="UTC"), pd.Timestamp("2021-12-31 23:59:59", tz="UTC")),
    "2022": (pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2022-12-31 23:59:59", tz="UTC")),
    "2026_q1": (pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")),
}

# Threshold sweep grid
THR_SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
THR_MIN_PASSING = 50        # minimum n_passing in cal_fit for a threshold to be considered

# ── feature list ─────────────────────────────────────────────────────────────

def _load_features() -> list[str]:
    """Load v2p3 canonical 40 features + 5 v_new_1 additions, drop signals_same_15m_same_detector."""
    with open(V2P3_META_PATH) as f:
        v2p3_meta = json.load(f)
    base_features: list[str] = v2p3_meta["feature_names"]  # 40 features

    # Drop the constant column — always 1 in continuous-scoring context (not signal-triggered)
    DROP_CONST = "signals_same_15m_same_detector"
    filtered = [f for f in base_features if f != DROP_CONST]

    # 5 v_new_1 additions
    additions = [
        "cfgi_value",
        "vol_rank_in_top100",
        "days_since_last_big_move_long",
        "days_since_last_big_move_short",
        "coin_24h_vol_zscore_30d",
    ]
    # Deduplicate in case any addition was already present
    seen = set(filtered)
    for a in additions:
        if a not in seen:
            filtered.append(a)

    # v_new_1.5: append sequence features when toggled.
    if V_NEW_1_5_MODE:
        for s in V_NEW_1_5_SEQUENCE_FEATURES:
            if s not in seen:
                filtered.append(s)
                seen.add(s)
    return filtered


FEATURE_NAMES: list[str] = _load_features()   # 44 features at module load


# ── CalibratedLGBM wrapper (matches v8 production API exactly) ───────────────

class CalibratedLGBM:
    """LGBMClassifier + IsotonicRegression calibrator.

    Drop-in for V8Classifier.predict_proba API.
    """

    def __init__(self, base_model: lgb.LGBMClassifier, calibrator: IsotonicRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


# ── helpers ───────────────────────────────────────────────────────────────────

def _ts(label: str = "") -> str:
    t = datetime.now(UTC).strftime("%H:%M:%S")
    return f"[{t}]{' ' + label if label else ''}"


def _ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    """Expected calibration error."""
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


def _score_quantiles(model: CalibratedLGBM, X: np.ndarray, n_q: int = 200) -> list[float]:  # noqa: N803
    probs = model.predict_proba(X)[:, 1]
    idx = np.linspace(0, len(probs) - 1, n_q).astype(int)
    return [round(float(v), 6) for v in np.sort(probs)[idx].tolist()]


def _top_k_precision(y_true: np.ndarray, scores: np.ndarray, top_frac: float) -> float:
    """Precision among top-`top_frac` fraction of rows by score."""
    k = max(1, int(len(scores) * top_frac))
    top_idx = np.argsort(scores)[-k:]
    return float(y_true[top_idx].mean())


def _lock_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    thr_grid: list[float],
    min_passing: int,
) -> tuple[float, dict]:
    """Sweep threshold grid; pick max(precision × min(0.5, recall)) with n_passing >= min_passing."""
    best_thr = thr_grid[0]
    best_score = -1.0
    best_metrics: dict = {}

    for thr in thr_grid:
        mask = scores >= thr
        n_pass = int(mask.sum())
        if n_pass < min_passing:
            continue
        tp = int(((mask) & (y == 1)).sum())
        fp = int(((mask) & (y == 0)).sum())
        fn = int(((~mask) & (y == 1)).sum())
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        obj = prec * min(0.5, rec)
        if obj > best_score:
            best_score = obj
            best_thr = thr
            best_metrics = {
                "n_passing": n_pass,
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "objective": round(obj, 6),
                "pr_auc": round(float(average_precision_score(y, scores)), 4),
                "roc_auc": round(float(roc_auc_score(y, scores)) if len(set(y)) >= 2 else float("nan"), 4),
                "brier": round(float(brier_score_loss(y, scores)), 4),
            }

    if not best_metrics:
        # Fallback: take lowest threshold that meets min_passing
        for thr in thr_grid:
            if (scores >= thr).sum() >= min_passing:
                best_thr = thr
                mask = scores >= thr
                best_metrics = {"n_passing": int(mask.sum()), "pr_auc": round(float(average_precision_score(y, scores)), 4)}
                break

    return best_thr, best_metrics


# ── universe filter ──────────────────────────────────────────────────────────

def _build_universe_set(universe_df: pd.DataFrame) -> set[tuple[str, str]]:
    """Build a set of (symbol, week_start_str) for fast membership lookup."""
    universe_df = universe_df.copy()
    universe_df["week_start"] = pd.to_datetime(universe_df["week_start"], utc=True)
    universe_df["_week_key"] = universe_df["week_start"].dt.strftime("%Y-%W")
    return set(zip(universe_df["symbol"], universe_df["_week_key"]))


def _filter_to_universe(df: pd.DataFrame, universe_set: set[tuple[str, str]]) -> pd.DataFrame:
    """Filter df rows to those where (symbol, week) is in the top-100 universe."""
    week_keys = df["timestamp"].dt.strftime("%Y-%W")
    in_universe = [
        (sym, wk) in universe_set
        for sym, wk in zip(df["symbol"], week_keys)
    ]
    return df[in_universe].reset_index(drop=True)


# ── purged time-series splits ────────────────────────────────────────────────

def _purged_ts_splits(
    timestamps: pd.Series,
    n_splits: int = N_FOLDS,
    embargo_days: int = EMBARGO_DAYS,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """TimeSeriesSplit with post-fold embargo to prevent label leakage.

    Embargo removes test samples whose timestamp falls within `embargo_days`
    of the last training sample. This covers the 7-day label horizon.
    """
    tss = TimeSeriesSplit(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in tss.split(np.arange(len(timestamps))):
        train_end = timestamps.iloc[train_idx[-1]]
        embargo_cut = train_end + pd.Timedelta(days=embargo_days)
        test_times = timestamps.iloc[test_idx]
        purged = test_idx[(test_times >= embargo_cut).values]
        if len(purged) >= 50:
            splits.append((train_idx, purged))
    return splits


# ── Optuna objective ─────────────────────────────────────────────────────────

def _make_optuna_objective(
    X_train: np.ndarray,  # noqa: N803
    y_train: np.ndarray,
    splits: list[tuple[np.ndarray, np.ndarray]],
    spw: float,
    direction_label: str,
):
    """Return Optuna objective fn. Maximizes median OOS PR-AUC across purged folds."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 50, 300),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 0.9),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "bagging_freq": 5,
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "scale_pos_weight": spw,
            "objective": "binary",
            "metric": "average_precision",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "random_state": RANDOM_SEED,
            "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
            "num_threads": LGBM_NUM_THREADS,
        }
        fold_scores: list[float] = []
        for fold_i, (tr_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            clf = lgb.LGBMClassifier(**params)
            clf.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
            )
            probs = clf.predict_proba(X_val)[:, 1]
            fold_scores.append(float(average_precision_score(y_val, probs)))
            trial.report(float(np.mean(fold_scores)), step=fold_i)
            if trial.should_prune():
                raise optuna.TrialPruned()

        median_pr_auc = float(np.median(fold_scores))
        print(
            f"  {_ts(direction_label)} trial {trial.number:3d} | "
            f"PR-AUC folds={[round(s, 4) for s in fold_scores]} | "
            f"median={median_pr_auc:.4f}"
        )
        return median_pr_auc

    return objective


# ── per-year OOS evaluation ──────────────────────────────────────────────────

def _eval_oos_year(
    year_label: str,
    ts_start: pd.Timestamp,
    ts_end: pd.Timestamp,
    oos_labels: pd.DataFrame,           # labels_{direction}_oos.parquet (merged with features already)
    calibrated_model: CalibratedLGBM,
    feature_names: list[str],
    universe_set: set[tuple[str, str]],
    label_col: str,
) -> dict:
    """Score one OOS year slice and return metrics dict."""
    mask_ts = (oos_labels["timestamp"] >= ts_start) & (oos_labels["timestamp"] <= ts_end)
    year_df = oos_labels[mask_ts].copy()

    if len(year_df) == 0:
        return {"n_rows": 0, "note": "no data for this period"}

    # Filter to universe
    year_df = _filter_to_universe(year_df, universe_set)
    if len(year_df) == 0:
        return {"n_rows": 0, "note": "no rows after universe filter"}

    # Drop NaN-heavy rows (need at least half the features present)
    feat_cols_present = [f for f in feature_names if f in year_df.columns]
    year_df_clean = year_df.dropna(subset=feat_cols_present, thresh=len(feat_cols_present) // 2)

    if len(year_df_clean) < 50:
        return {"n_rows": int(len(year_df_clean)), "note": "too few rows after NaN drop"}

    X = year_df_clean[feat_cols_present].values  # noqa: N806
    # Fill any remaining NaN with 0 (LightGBM handles NaN natively but explicit fill is safer)
    X = np.nan_to_num(X, nan=0.0)
    y = year_df_clean[label_col].values.astype(int)
    scores = calibrated_model.predict_proba(X)[:, 1]

    pr_auc = float(average_precision_score(y, scores)) if len(set(y)) >= 2 else float("nan")
    roc_auc = float(roc_auc_score(y, scores)) if len(set(y)) >= 2 else float("nan")
    brier = float(brier_score_loss(y, scores))
    top10_prec = _top_k_precision(y, scores, 0.10)
    top5_prec = _top_k_precision(y, scores, 0.05)

    return {
        "year_label": year_label,
        "n_rows": int(len(year_df_clean)),
        "positive_rate": round(float(y.mean()), 4),
        "pr_auc": round(pr_auc, 4),
        "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4),
        "top_decile_precision": round(top10_prec, 4),
        "top_5pct_precision": round(top5_prec, 4),
        "median_score": round(float(np.median(scores)), 4),
        "p90_score": round(float(np.percentile(scores, 90)), 4),
    }


# ── main per-direction training function ─────────────────────────────────────

def train_direction(direction: str) -> dict:
    """Train one direction (long or short). Returns completed meta dict.

    No-contamination guarantee:
      - Optuna sees ONLY train_fit purged folds
      - Threshold is locked on cal_fit only
      - OOS years are touched exactly once for evaluation
    """
    label_col = "long_event" if direction == "long" else "short_event"
    dir_label = f"v_new_1_{direction}{MODE_SUFFIX}"
    t0_dir = time.time()

    print(f"\n{'='*70}")
    print(f"{_ts(dir_label)} Starting {dir_label} training")
    print(f"{'='*70}")

    # ── 1. Load & merge ──────────────────────────────────────────────────────
    print(f"{_ts(dir_label)} Loading labels...")
    labels = pd.read_parquet(DATA_DIR / f"labels_{direction}.parquet")
    labels["timestamp"] = pd.to_datetime(labels["timestamp"], utc=True)
    print(f"  labels: {len(labels):,} rows, positive rate={labels[label_col].mean():.4f}")

    print(f"{_ts(dir_label)} Loading features (column-selective)...")
    # Load only the columns we actually need — memory optimization for 1.17M × 45
    needed_feature_cols = [f for f in FEATURE_NAMES if f not in ("symbol", "timestamp")]
    # features_full has: symbol, timestamp, close + feature columns
    all_feat_cols = ["symbol", "timestamp"] + [
        f for f in needed_feature_cols if f not in ("symbol", "timestamp")
    ]
    features = pd.read_parquet(DATA_DIR / FEATURES_PARQUET_NAME, columns=all_feat_cols)
    features["timestamp"] = pd.to_datetime(features["timestamp"], utc=True)
    print(f"  features: {len(features):,} rows, {features.shape[1]} cols")

    print(f"{_ts(dir_label)} Loading universe membership...")
    universe_df = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    universe_set = _build_universe_set(universe_df)
    print(f"  universe: {len(universe_df):,} entries, {len(set(universe_df['symbol']))} unique symbols")

    # ── 2. Universe filter ───────────────────────────────────────────────────
    print(f"{_ts(dir_label)} Filtering labels to top-100 universe...")
    labels = _filter_to_universe(labels, universe_set)
    print(f"  after universe filter: {len(labels):,} rows")

    # ── 3. Merge labels + features ───────────────────────────────────────────
    print(f"{_ts(dir_label)} Merging labels + features...")
    df = labels.merge(features, on=["symbol", "timestamp"], how="inner")
    print(f"  merged: {len(df):,} rows")

    # ── 4. Drop signals_same_15m_same_detector (always-constant in this context) ──
    if "signals_same_15m_same_detector" in df.columns:
        df = df.drop(columns=["signals_same_15m_same_detector"])

    # Determine actual feature columns (intersection of FEATURE_NAMES with merged df)
    available_features = [f for f in FEATURE_NAMES if f in df.columns]
    if len(available_features) != len(FEATURE_NAMES):
        missing = [f for f in FEATURE_NAMES if f not in df.columns]
        print(f"  WARNING: {len(missing)} feature(s) not in merged df: {missing}")
    feature_names_used = available_features
    print(f"  features used: {len(feature_names_used)}")

    # ── 5. Drop NaN rows (allow columns with expected high-NaN rate) ─────────
    # Per plan: 4 features can be NaN-heavy. Drop rows where >50% of features are NaN.
    nan_floor = len(feature_names_used) // 2
    df_clean = df.dropna(subset=feature_names_used, thresh=nan_floor).reset_index(drop=True)
    print(f"  after NaN-row drop (thresh={nan_floor}): {len(df_clean):,} rows "
          f"(dropped {len(df)-len(df_clean):,})")

    # ── 6. Chronological split ───────────────────────────────────────────────
    train_fit = df_clean[df_clean["timestamp"] <= TRAIN_FIT_END].reset_index(drop=True)
    cal_fit = df_clean[df_clean["timestamp"] >= CAL_FIT_START].reset_index(drop=True)

    print(f"{_ts(dir_label)} Splits:")
    print(f"  train_fit: {len(train_fit):,} rows "
          f"({train_fit['timestamp'].min().date()} → {train_fit['timestamp'].max().date()}) "
          f"positive_rate={train_fit[label_col].mean():.4f}")
    print(f"  cal_fit:   {len(cal_fit):,} rows "
          f"({cal_fit['timestamp'].min().date()} → {cal_fit['timestamp'].max().date()}) "
          f"positive_rate={cal_fit[label_col].mean():.4f}")

    # ── 7. Prepare numpy arrays ──────────────────────────────────────────────
    X_train = train_fit[feature_names_used].values  # noqa: N806
    y_train = train_fit[label_col].values.astype(int)
    X_cal = cal_fit[feature_names_used].values  # noqa: N806
    y_cal = cal_fit[label_col].values.astype(int)

    # Fill NaN with 0 (LightGBM handles NaN natively; explicit fill for safety)
    X_train = np.nan_to_num(X_train, nan=0.0)
    X_cal = np.nan_to_num(X_cal, nan=0.0)

    spw = float((y_train == 0).sum()) / float((y_train == 1).sum())
    print(f"  scale_pos_weight: {spw:.4f} (positive rate: {y_train.mean():.4f})")

    # ── 8. Purged CV splits ──────────────────────────────────────────────────
    train_ts = train_fit["timestamp"].reset_index(drop=True)
    splits = _purged_ts_splits(train_ts, N_FOLDS, EMBARGO_DAYS)
    print(f"\n{_ts(dir_label)} Purged folds ({N_FOLDS} requested, {len(splits)} valid after embargo):")
    for i, (tr_idx, te_idx) in enumerate(splits):
        t_start = train_ts.iloc[te_idx[0]].date()
        t_end = train_ts.iloc[te_idx[-1]].date()
        print(f"  Fold {i}: train={len(tr_idx):,}  test={len(te_idx):,}  "
              f"test_range={t_start} → {t_end}")

    # ── 9. Optuna hyperparameter search ──────────────────────────────────────
    print(f"\n{_ts(dir_label)} Optuna search ({N_OPTUNA_TRIALS} trials, purged {len(splits)}-fold CV)...")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(
        _make_optuna_objective(X_train, y_train, splits, spw, dir_label),
        n_trials=N_OPTUNA_TRIALS,
        n_jobs=1,           # n_jobs=1 keeps stable on M2 Pro (LightGBM already multi-threaded)
        show_progress_bar=False,
    )
    best_params = study.best_params
    print(f"\n{_ts(dir_label)} Optuna done. Best median PR-AUC: {study.best_value:.4f}")
    print(f"  Best params: {json.dumps(best_params)}")

    # ── 10. Per-fold metrics with best params ────────────────────────────────
    print(f"\n{_ts(dir_label)} Re-running best params for per-fold metrics...")
    full_params = {
        **best_params,
        "bagging_freq": 5,
        "scale_pos_weight": spw,
        "objective": "binary",
        "metric": "average_precision",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "random_state": RANDOM_SEED,
        "num_threads": LGBM_NUM_THREADS,
    }

    per_fold_metrics = []
    fold_best_iters = []
    # OOF capture: each train_fit row's prediction comes from the fold where it was held out.
    # Rows in the very first TimeSeriesSplit slice and rows inside the 10-day embargo zone
    # never become test samples, so they remain NaN — that's expected and handled downstream.
    oof_probs = np.full(len(X_train), np.nan, dtype=np.float64)
    for fold_i, (tr_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]  # noqa: N806
        X_val, y_val = X_train[val_idx], y_train[val_idx]  # noqa: N806
        clf = lgb.LGBMClassifier(**full_params)
        clf.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)],
        )
        probs = clf.predict_proba(X_val)[:, 1]
        oof_probs[val_idx] = probs
        best_iter = int(clf.best_iteration_) if clf.best_iteration_ else full_params["n_estimators"]
        fold_best_iters.append(best_iter)
        fm = {
            "fold": fold_i,
            "n_train": int(len(tr_idx)),
            "n_test": int(len(val_idx)),
            "best_iteration": best_iter,
            "pr_auc": round(float(average_precision_score(y_val, probs)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, probs)) if len(set(y_val)) >= 2 else float("nan"), 4),
            "brier": round(float(brier_score_loss(y_val, probs)), 4),
        }
        per_fold_metrics.append(fm)
        print(f"  Fold {fold_i}: PR-AUC={fm['pr_auc']:.4f}  ROC-AUC={fm['roc_auc']:.4f}  "
              f"Brier={fm['brier']:.4f}  best_iter={best_iter}")

    # Final n_estimators: median across folds × 1.1, capped at 1500
    final_n_estimators = min(1500, max(200, int(round(float(np.median(fold_best_iters)) * 1.1))))
    print(f"  Final n_estimators: {final_n_estimators}")

    # ── 11. Final model: train on full train_fit ─────────────────────────────
    print(f"\n{_ts(dir_label)} Training final model on full train_fit ({len(X_train):,} rows)...")
    final_params = {**full_params, "n_estimators": final_n_estimators}
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])
    print(f"  Done.")

    # ── 12. Isotonic calibrator on cal_fit (disjoint) ────────────────────────
    print(f"\n{_ts(dir_label)} Fitting isotonic calibrator on cal_fit "
          f"({len(X_cal):,} rows, disjoint from final model)...")
    raw_cal_probs = final_model.predict_proba(X_cal)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_cal_probs, y_cal)
    calibrated = CalibratedLGBM(final_model, iso)
    print(f"  Done. Calibrator fit on {len(y_cal):,} rows.")

    # ── 13. Calibration metrics on cal_fit ───────────────────────────────────
    cal_probs = calibrated.predict_proba(X_cal)[:, 1]
    brier_cal = float(brier_score_loss(y_cal, cal_probs))
    ece_cal = _ece(y_cal, cal_probs)
    print(f"  cal_fit Brier: {brier_cal:.4f}  ECE: {ece_cal:.4f}")

    # ── 14. Threshold locking on cal_fit ─────────────────────────────────────
    print(f"\n{_ts(dir_label)} Locking entry threshold on cal_fit...")
    locked_thr, thr_metrics = _lock_threshold(y_cal, cal_probs, THR_SWEEP, THR_MIN_PASSING)
    print(f"  Locked threshold: {locked_thr:.2f}  "
          f"precision={thr_metrics.get('precision', '?'):.4f}  "
          f"recall={thr_metrics.get('recall', '?'):.4f}  "
          f"n_passing={thr_metrics.get('n_passing', '?')}")

    # ── 15. Train-set score quantiles (for live drift monitoring) ────────────
    print(f"\n{_ts(dir_label)} Computing score quantiles on train_fit...")
    train_quantiles = _score_quantiles(calibrated, X_train)
    print(f"  Done. p50={train_quantiles[100]:.4f}  p90={train_quantiles[180]:.4f}")

    # ── 16. Multi-OOS evaluation ─────────────────────────────────────────────
    print(f"\n{_ts(dir_label)} Loading OOS labels for multi-year evaluation...")
    oos_labels_raw = pd.read_parquet(DATA_DIR / f"labels_{direction}_oos.parquet")
    oos_labels_raw["timestamp"] = pd.to_datetime(oos_labels_raw["timestamp"], utc=True)

    # Merge OOS labels with features (load only needed feature columns)
    print(f"{_ts(dir_label)} Merging OOS labels with features...")
    oos_merged = oos_labels_raw.merge(
        features[["symbol", "timestamp"] + feature_names_used], on=["symbol", "timestamp"], how="inner"
    )
    print(f"  OOS merged: {len(oos_merged):,} rows")

    per_year_oos_metrics = {}
    oos_scored_rows = []

    for year_label, (ts_start, ts_end) in OOS_YEAR_SLICES.items():
        print(f"{_ts(dir_label)} Evaluating OOS year: {year_label}...")
        year_metrics = _eval_oos_year(
            year_label, ts_start, ts_end, oos_merged,
            calibrated, feature_names_used, universe_set, label_col,
        )
        per_year_oos_metrics[year_label] = year_metrics
        print(f"  {year_label}: n={year_metrics.get('n_rows',0)}  "
              f"PR-AUC={year_metrics.get('pr_auc','N/A')}  "
              f"ROC-AUC={year_metrics.get('roc_auc','N/A')}  "
              f"Brier={year_metrics.get('brier','N/A')}  "
              f"top10_prec={year_metrics.get('top_decile_precision','N/A')}")

        # Collect scored rows for oos_scored parquet
        mask_ts = (oos_merged["timestamp"] >= ts_start) & (oos_merged["timestamp"] <= ts_end)
        year_df = _filter_to_universe(oos_merged[mask_ts].copy(), universe_set)
        if len(year_df) > 0:
            feat_cols = [f for f in feature_names_used if f in year_df.columns]
            X_oos = np.nan_to_num(year_df[feat_cols].values, nan=0.0)  # noqa: N806
            year_df = year_df.copy()
            year_df["score"] = calibrated.predict_proba(X_oos)[:, 1]
            year_df["oos_year"] = year_label
            oos_scored_rows.append(year_df[["symbol", "timestamp", label_col, "score", "oos_year"]])

    # Aggregate OOS metrics
    valid_pr_aucs = [
        m["pr_auc"] for m in per_year_oos_metrics.values()
        if isinstance(m.get("pr_auc"), float) and not np.isnan(m["pr_auc"])
    ]
    aggregate_oos_metrics = {
        "median_pr_auc": round(float(np.median(valid_pr_aucs)), 4) if valid_pr_aucs else float("nan"),
        "p5_pr_auc": round(float(np.percentile(valid_pr_aucs, 5)), 4) if valid_pr_aucs else float("nan"),
        "p95_pr_auc": round(float(np.percentile(valid_pr_aucs, 95)), 4) if valid_pr_aucs else float("nan"),
    }
    print(f"\n{_ts(dir_label)} Aggregate OOS PR-AUC: "
          f"median={aggregate_oos_metrics['median_pr_auc']:.4f}  "
          f"p5={aggregate_oos_metrics['p5_pr_auc']:.4f}  "
          f"p95={aggregate_oos_metrics['p95_pr_auc']:.4f}")

    # ── 17. SHAP top-20 importances ──────────────────────────────────────────
    print(f"\n{_ts(dir_label)} Computing SHAP importances "
          f"(subsample {SHAP_SUBSAMPLE:,} rows from cal_fit)...")
    shap_n = min(SHAP_SUBSAMPLE, len(X_cal))
    rng = np.random.default_rng(RANDOM_SEED)
    shap_idx = rng.choice(len(X_cal), size=shap_n, replace=False)
    X_shap = X_cal[shap_idx]  # noqa: N806

    explainer = shap.TreeExplainer(final_model)
    shap_vals = explainer.shap_values(X_shap)
    if isinstance(shap_vals, list):
        shap_arr = shap_vals[1]           # older shap API: [neg_class, pos_class]
    elif shap_vals.ndim == 3:
        shap_arr = shap_vals[:, :, 1]    # newer shap API: (n, features, classes)
    else:
        shap_arr = shap_vals
    mean_abs = np.abs(shap_arr).mean(axis=0)
    shap_importance = {
        f: round(float(v), 6)
        for f, v in sorted(zip(feature_names_used, mean_abs), key=lambda x: -x[1])
    }
    shap_top20 = dict(list(shap_importance.items())[:20])
    print(f"  Top-5 SHAP: {list(shap_top20.items())[:5]}")

    # ── 18. Save artifacts ───────────────────────────────────────────────────
    model_dir = PYTHON_ROOT / "models" / dir_label
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{dir_label}.joblib"
    joblib.dump(calibrated, model_path)
    print(f"\n{_ts(dir_label)} Saved model: {model_path}")

    # OOS scored parquet
    oos_path = DATA_DIR / f"oos_scored_{direction}{MODE_SUFFIX}.parquet"
    if oos_scored_rows:
        oos_scored_df = pd.concat(oos_scored_rows, ignore_index=True)
        oos_scored_df.to_parquet(oos_path, index=False)
        print(f"{_ts(dir_label)} Saved OOS scored: {oos_path} ({len(oos_scored_df):,} rows)")
    else:
        print(f"{_ts(dir_label)} WARNING: no OOS scored rows to save")

    # ── 18b. OOF scored parquet (train_fit) — for v_new_1.5 lagged features ─
    # OOF covers train_fit rows where the row was a CV val sample. Rows in the
    # very first TimeSeriesSplit slice and rows inside fold-boundary embargoes
    # remain NaN. Downstream sequence-feature builder must dropna or forward-fill.
    oof_path = DATA_DIR / f"oof_scored_{direction}{MODE_SUFFIX}.parquet"
    oof_df = pd.DataFrame({
        "symbol": train_fit["symbol"].values,
        "timestamp": train_fit["timestamp"].values,
        "score_raw": oof_probs,
        label_col: y_train,
        "split": "train_fit_oof",
    })
    n_oof_valid = int((~oof_df["score_raw"].isna()).sum())
    oof_df.to_parquet(oof_path, index=False)
    print(f"{_ts(dir_label)} Saved OOF scored: {oof_path} "
          f"({n_oof_valid:,} / {len(oof_df):,} rows have OOF scores; "
          f"{len(oof_df) - n_oof_valid:,} NaN from first-slice + embargo)")

    # ── 18c. Cal scored parquet (cal_fit) — non-leaky raw probs from final model ─
    # final_model was fit on train_fit only, so its predictions on cal_fit are
    # honest. Calibrated scores would be leaky on cal_fit (calibrator was fit
    # there), so we save RAW scores for use as v_new_1.5 lagged features.
    cal_path = DATA_DIR / f"cal_scored_{direction}{MODE_SUFFIX}.parquet"
    cal_df = pd.DataFrame({
        "symbol": cal_fit["symbol"].values,
        "timestamp": cal_fit["timestamp"].values,
        "score_raw": raw_cal_probs,
        label_col: y_cal,
        "split": "cal_fit",
    })
    cal_df.to_parquet(cal_path, index=False)
    print(f"{_ts(dir_label)} Saved CAL scored: {cal_path} ({len(cal_df):,} rows)")

    # Meta JSON
    elapsed_min = round((time.time() - t0_dir) / 60.0, 1)
    meta = {
        "model_class": "CalibratedLGBM (LightGBM + isotonic)",
        "direction": direction,
        "feature_names": feature_names_used,
        "n_features": len(feature_names_used),
        "features_v2p3_base": [f for f in FEATURE_NAMES if f not in [
            "cfgi_value", "vol_rank_in_top100",
            "days_since_last_big_move_long", "days_since_last_big_move_short",
            "coin_24h_vol_zscore_30d",
        ]],
        "features_v_new_1_additions": [
            "cfgi_value", "vol_rank_in_top100",
            "days_since_last_big_move_long", "days_since_last_big_move_short",
            "coin_24h_vol_zscore_30d",
        ],
        "best_optuna_params": best_params,
        "final_n_estimators": final_n_estimators,
        "cv_scheme": "purged_kfold_5",
        "embargo_days": EMBARGO_DAYS,
        "n_optuna_trials": N_OPTUNA_TRIALS,
        "random_seed": RANDOM_SEED,
        "scale_pos_weight": round(spw, 4),
        "train_period": ["2023-04-01", "2024-09-30"],
        "cal_period": ["2024-10-01", "2025-12-31"],
        "holdout_period": "[2020 May+, 2021, 2022, 2026 Q1] — 4 OOS years",
        "per_fold_metrics": per_fold_metrics,
        "aggregate_oos_metrics": aggregate_oos_metrics,
        "calibration": {
            "method": "isotonic",
            "fit_on": "cal_fit (disjoint, last 25% of pre-OOS period 2024-10 to 2025-12)",
            "n_calibration_rows": int(len(y_cal)),
            "ece_cal_fit": round(ece_cal, 4),
            "brier_cal_fit": round(brier_cal, 4),
        },
        "train_score_quantiles": train_quantiles,
        "locked_threshold": {
            "value": locked_thr,
            "selection_method": "max(precision × min(0.5, recall)) on cal_fit",
            "min_passing_constraint": THR_MIN_PASSING,
            "cal_fit_metrics_at_threshold": thr_metrics,
        },
        "per_year_oos_metrics": per_year_oos_metrics,
        "shap_importance_top20": shap_top20,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "n_train_rows": int(len(y_train)),
        "n_cal_rows": int(len(y_cal)),
        "class_baseline_positive_rate": round(float(y_train.mean()), 4),
        "elapsed_minutes": elapsed_min,
    }

    meta_path = model_dir / f"{dir_label}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"{_ts(dir_label)} Saved meta:  {meta_path}")
    print(f"{_ts(dir_label)} {dir_label} COMPLETE in {elapsed_min:.1f} min")

    return meta


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    t0 = time.time()

    # ── validate all inputs exist before starting ────────────────────────────
    required_inputs = [
        DATA_DIR / "labels_long.parquet",
        DATA_DIR / "labels_short.parquet",
        DATA_DIR / "labels_long_oos.parquet",
        DATA_DIR / "labels_short_oos.parquet",
        DATA_DIR / "features_full.parquet",
        DATA_DIR / "universe_top100_membership.parquet",
        V2P3_META_PATH,
    ]
    print(f"{_ts()} Phase 3 training — v_new_1 (long + short)")
    print(f"{_ts()} Validating input paths...")
    for p in required_inputs:
        status = "OK" if p.exists() else "MISSING"
        print(f"  [{status}] {p}")
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}")
    print(f"{_ts()} All inputs present. n_features={len(FEATURE_NAMES)}")

    # ── create output directories ────────────────────────────────────────────
    results_dir = PYTHON_ROOT / "results" / "v_new_1_phase3"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── train long ───────────────────────────────────────────────────────────
    print(f"\n{_ts()} ════ TRAINING v_new_1_long ════")
    long_meta = train_direction("long")

    # Save immediately before starting short (crash-safety)
    print(f"\n{_ts()} v_new_1_long artifacts saved. Starting v_new_1_short...")

    # Free long-direction dataframes from memory
    gc.collect()

    # ── train short ──────────────────────────────────────────────────────────
    print(f"\n{_ts()} ════ TRAINING v_new_1_short ════")
    short_meta = train_direction("short")

    gc.collect()

    # ── final summary ────────────────────────────────────────────────────────
    total_min = round((time.time() - t0) / 60.0, 1)
    print(f"\n{'='*70}")
    print(f"{_ts()} Phase 3 training COMPLETE in {total_min:.1f} min")
    print(f"{'='*70}")
    print(f"  v_new_1_long  — median OOS PR-AUC: "
          f"{long_meta['aggregate_oos_metrics']['median_pr_auc']:.4f}  "
          f"locked_thr={long_meta['locked_threshold']['value']:.2f}")
    print(f"  v_new_1_short — median OOS PR-AUC: "
          f"{short_meta['aggregate_oos_metrics']['median_pr_auc']:.4f}  "
          f"locked_thr={short_meta['locked_threshold']['value']:.2f}")
    print()
    print("Phase 3 ding-ding gate (median OOS PR-AUC > 0.20):")
    for lbl, meta in [("long", long_meta), ("short", short_meta)]:
        pr = meta["aggregate_oos_metrics"]["median_pr_auc"]
        gate = "PASS" if pr > 0.20 else "FAIL (investigate — may need label/feature tuning)"
        print(f"  v_new_1_{lbl}: {pr:.4f} — {gate}")
    print()
    print("Artifacts written to:")
    print(f"  {PYTHON_ROOT}/models/v_new_1_long{MODE_SUFFIX}/")
    print(f"  {PYTHON_ROOT}/models/v_new_1_short{MODE_SUFFIX}/")
    print(f"  {DATA_DIR}/oos_scored_long{MODE_SUFFIX}.parquet")
    print(f"  {DATA_DIR}/oos_scored_short{MODE_SUFFIX}.parquet")
    print(f"  {DATA_DIR}/oof_scored_long{MODE_SUFFIX}.parquet")
    print(f"  {DATA_DIR}/oof_scored_short{MODE_SUFFIX}.parquet")
    print(f"  {DATA_DIR}/cal_scored_long{MODE_SUFFIX}.parquet")
    print(f"  {DATA_DIR}/cal_scored_short{MODE_SUFFIX}.parquet")
    if V_NEW_1_5_MODE:
        print(f"  (V_NEW_1_5_MODE=1: 12 sequence features added; trained on features_full_v1_5.parquet)")


if __name__ == "__main__":
    main()
