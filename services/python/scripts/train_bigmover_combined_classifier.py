"""Train the v1 bigmover-combined classifier on the 6 baseline trade CSVs.

Pipeline (matches the plan):
  1. Load all 6 (signal x direction) bigmover trade CSVs from Phase 2.
  2. Filter to labelable trades, compute direction-aware continuation labels.
  3. Build the 16 synthesized features + is_short via FeatureContext.
  4. Time-ordered split: train < 2026-01-01 UTC, OOT >= 2026-01-01.
  5. 5-fold purged time-series CV inside train (purge=192 bars = label horizon).
  6. Logistic-regression baseline → LightGBM if LR OOT AUC >= 0.55. Save best.
  7. Lock decision threshold at the 0.5 quantile of train-period scores by
     default (Phase-5 backtest will sweep + lock).

Run from services/python/:
    python scripts/train_bigmover_combined_classifier.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES, FeatureContext
from src.ml.bigmover_combined.labels import filter_labelable, make_label
from src.ml.bigmover_combined.sizing import default_sizing_config

# --- Constants ---------------------------------------------------------------

SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
CANDLES_PATH = SNAPSHOTS_DIR / f"candles_15m_{SNAPSHOT_DATE}.parquet"
UNIVERSE_PATH = SNAPSHOTS_DIR / f"universe_{SNAPSHOT_DATE}.parquet"

# Phase-2 trade CSVs (all 6 combos).
RESULTS_DIR = REPO_ROOT / "services/python/results"
TRADES_CSVS = [
    RESULTS_DIR / "backtest_bigmover_baseline_protocol_baseline_short_b198a3ee_20260425T162843Z" / "trades.csv",
    RESULTS_DIR / "backtest_bigmover_baseline_protocol_baseline_long_b198a3ee_20260425T162843Z" / "trades.csv",
    RESULTS_DIR / "backtest_bigmover_baseline_protocol_price_accel_atr_short_b198a3ee_20260425T162843Z" / "trades.csv",
    RESULTS_DIR / "backtest_bigmover_baseline_protocol_price_accel_atr_long_b198a3ee_20260425T162843Z" / "trades.csv",
    RESULTS_DIR / "backtest_bigmover_baseline_protocol_multi_bar_confirm_short_b198a3ee_20260425T162843Z" / "trades.csv",
    RESULTS_DIR / "backtest_bigmover_baseline_protocol_multi_bar_confirm_long_b198a3ee_20260425T162843Z" / "trades.csv",
]

MODELS_DIR = REPO_ROOT / "services/python/models"
MODEL_OUT = MODELS_DIR / "bigmover_combined_v1.joblib"
META_OUT = MODELS_DIR / "bigmover_combined_v1_meta.json"
SIZING_OUT = MODELS_DIR / "bigmover_combined_v1_sizing.json"

LABEL_FWD_TARGET = 0.10
LABEL_DRAWDOWN_CAP = 0.04
LABEL_MAX_BARS = 192

LR_C = 1.0
LR_AUC_FLOOR = 0.55

LGBM_PARAMS = dict(
    n_estimators=250,
    max_depth=6,
    learning_rate=0.05,
    is_unbalance=True,
    random_state=RANDOM_SEED,
    verbose=-1,
)


# --- Helpers ----------------------------------------------------------------


def _purged_kfold_indices(n: int, k: int = 5, purge: int = 192) -> list[tuple[np.ndarray, np.ndarray]]:
    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        train_end = max(0, val_start - purge)
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


def _all_columns(X: list[list[float]], finite_check: bool = True) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(X, dtype=float)
    finite = np.isfinite(arr).all(axis=1)
    return arr, finite


# --- Main pipeline -----------------------------------------------------------


def main() -> None:
    started = time.monotonic()
    np.random.seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load + concat trades ------------------------------------------
    dfs = []
    for path in TRADES_CSVS:
        if not path.exists():
            raise FileNotFoundError(f"Phase-2 trades csv missing: {path}")
        sub = pd.read_csv(path)
        dfs.append(sub)
    trades = pd.concat(dfs, ignore_index=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    print(f"loaded {len(trades):,} bigmover trades across "
          f"{trades['signal'].nunique()} signals x {trades['direction'].nunique()} dirs",
          flush=True)

    # ---- Load candles --------------------------------------------------
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    print(f"loaded {len(candles):,} candles, {candles['asset'].nunique()} assets",
          flush=True)

    universe_listed_since: dict[str, pd.Timestamp] = {}
    if UNIVERSE_PATH.exists():
        u = pd.read_parquet(UNIVERSE_PATH)
        if "asset" in u.columns and "listed_since" in u.columns:
            for _, row in u.iterrows():
                asset_with_usdt = f"{row['asset']}USDT"
                ls = row["listed_since"]
                if pd.notna(ls):
                    universe_listed_since[asset_with_usdt] = pd.Timestamp(ls, tz="UTC")
        print(f"  universe listed_since entries: {len(universe_listed_since)}", flush=True)

    # ---- Filter labelable trades + compute labels ----------------------
    snapshot_max = candles["timestamp"].max()
    keep = filter_labelable(trades, snapshot_max, max_bars=LABEL_MAX_BARS)
    trades = trades[keep].reset_index(drop=True)
    print(f"labelable trades (horizon fits): {len(trades):,}", flush=True)

    print("computing labels...", flush=True)
    labels = make_label(
        trades, candles,
        fwd_target=LABEL_FWD_TARGET,
        drawdown_cap=LABEL_DRAWDOWN_CAP,
        max_bars=LABEL_MAX_BARS,
    )
    trades["label"] = labels.values
    trades = trades[trades["label"] != -1].reset_index(drop=True)
    pos_rate = float((trades["label"] == 1).mean())
    print(f"  labeled: {len(trades):,} | positive rate: {pos_rate:.1%}", flush=True)
    pos_short = float((trades.loc[trades["direction"] == "short", "label"] == 1).mean())
    pos_long = float((trades.loc[trades["direction"] == "long", "label"] == 1).mean())
    print(f"    short positive rate: {pos_short:.1%}    long positive rate: {pos_long:.1%}",
          flush=True)

    # ---- Build features ------------------------------------------------
    print("building feature panels...", flush=True)
    ctx = FeatureContext(
        candles, btc_asset_key="BTCUSDT",
        universe_listed_since=universe_listed_since,
    )
    print("  panels built; computing per-trade features...", flush=True)
    feat_rows = []
    for i, row in enumerate(trades.itertuples(index=False)):
        feats = ctx._build_for_typed(row.symbol, row.entry_time, row.direction)
        # Concatenate FEATURE_NAMES + is_short for the model input.
        feat_rows.append([feats[n] for n in FEATURE_NAMES] + [feats["is_short"]])
        if (i + 1) % 5000 == 0:
            print(f"  features: {i+1}/{len(trades)}", flush=True)
    cols = FEATURE_NAMES + ["is_short"]
    X = pd.DataFrame(feat_rows, columns=cols)

    finite_mask = X.notna().all(axis=1)
    print(f"  feature-complete rows: {int(finite_mask.sum())}/{len(trades)}", flush=True)
    trades = trades[finite_mask].reset_index(drop=True)
    X = X[finite_mask].reset_index(drop=True)
    y = trades["label"].astype(int).to_numpy()

    is_train = (trades["entry_time"] < TRAIN_OOT_BOUNDARY).to_numpy()
    is_oot = ~is_train
    print(f"  train: {is_train.sum():,}  oot: {is_oot.sum():,}", flush=True)

    train_X = X.loc[is_train].to_numpy(dtype=float)
    train_y = y[is_train]
    oot_X = X.loc[is_oot].to_numpy(dtype=float)
    oot_y = y[is_oot]

    # ---- LR baseline + 5-fold purged time CV ---------------------------
    print("\n--- Logistic regression baseline (5-fold purged time CV, purge=192) ---")
    cv_aucs = []
    for fold_i, (tr_idx, va_idx) in enumerate(
        _purged_kfold_indices(len(train_X), k=5, purge=LABEL_MAX_BARS)
    ):
        if len(np.unique(train_y[tr_idx])) < 2 or len(np.unique(train_y[va_idx])) < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced",
            )),
        ])
        pipe.fit(train_X[tr_idx], train_y[tr_idx])
        proba_val = pipe.predict_proba(train_X[va_idx])[:, 1]
        auc = roc_auc_score(train_y[va_idx], proba_val)
        cv_aucs.append(float(auc))
        print(f"  fold {fold_i}: AUC={auc:.4f} (n_train={len(tr_idx):,}, n_val={len(va_idx):,})")

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced",
        )),
    ])
    lr_pipe.fit(train_X, train_y)
    train_auc_lr = roc_auc_score(train_y, lr_pipe.predict_proba(train_X)[:, 1])
    oot_auc_lr = (
        roc_auc_score(oot_y, lr_pipe.predict_proba(oot_X)[:, 1])
        if len(np.unique(oot_y)) >= 2 else float("nan")
    )
    print(f"  full-train AUC: {train_auc_lr:.4f}    OOT AUC: {oot_auc_lr:.4f}")

    # ---- LGBM if LR is informative -------------------------------------
    final_model = lr_pipe
    final_kind = "logistic_regression"
    train_auc_final = train_auc_lr
    oot_auc_final = oot_auc_lr
    lgbm_train_auc = None
    lgbm_oot_auc = None

    if np.isfinite(oot_auc_lr) and oot_auc_lr >= LR_AUC_FLOOR:
        print("\n--- LightGBM (LR OOT AUC clears the 0.55 floor) ---")
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as e:
            # OSError covers Mac libomp.dylib missing when no brew is installed.
            print(f"  lightgbm unavailable ({type(e).__name__}); sticking with LR.")
            LGBMClassifier = None
        if LGBMClassifier is not None:
            lgbm = LGBMClassifier(**LGBM_PARAMS)
            lgbm.fit(train_X, train_y)
            train_auc_lgbm = roc_auc_score(train_y, lgbm.predict_proba(train_X)[:, 1])
            oot_auc_lgbm = (
                roc_auc_score(oot_y, lgbm.predict_proba(oot_X)[:, 1])
                if len(np.unique(oot_y)) >= 2 else float("nan")
            )
            lgbm_train_auc = float(train_auc_lgbm)
            lgbm_oot_auc = float(oot_auc_lgbm) if np.isfinite(oot_auc_lgbm) else None
            print(f"  full-train AUC: {train_auc_lgbm:.4f}    OOT AUC: {oot_auc_lgbm:.4f}")
            # Choose whichever has higher OOT AUC.
            if np.isfinite(oot_auc_lgbm) and oot_auc_lgbm > oot_auc_lr:
                final_model = lgbm
                final_kind = "lightgbm"
                train_auc_final = train_auc_lgbm
                oot_auc_final = oot_auc_lgbm
                print("  selected: LightGBM (higher OOT AUC).")
            else:
                print("  selected: LogisticRegression (LR OOT AUC tied or higher).")
    else:
        print("\nLR OOT AUC < 0.55 — NOT escalating to LGBM (per protocol).")
        print("Final model = LR; the Phase-5 backtest will likely fall short of the 1.30 ship floor.")

    # ---- Save artifacts ------------------------------------------------
    joblib.dump(final_model, MODEL_OUT)
    meta = {
        "model": final_kind,
        "feature_names": cols,                  # FEATURE_NAMES + is_short
        "label_thresholds": {
            "fwd_target": LABEL_FWD_TARGET,
            "drawdown_cap": LABEL_DRAWDOWN_CAP,
            "max_bars": LABEL_MAX_BARS,
        },
        "train_auc": float(train_auc_final),
        "oot_auc": float(oot_auc_final) if np.isfinite(oot_auc_final) else None,
        "cv_aucs": cv_aucs,
        "lr_train_auc": float(train_auc_lr),
        "lr_oot_auc": float(oot_auc_lr) if np.isfinite(oot_auc_lr) else None,
        "lgbm_train_auc": lgbm_train_auc,
        "lgbm_oot_auc": lgbm_oot_auc,
        "trained_on_n": int(len(train_X)),
        "trained_on_pos_rate": float(np.mean(train_y)),
        "trained_on_pos_rate_short": float(np.mean(train_y[X.loc[is_train, "is_short"].to_numpy() == 1])),
        "trained_on_pos_rate_long": float(np.mean(train_y[X.loc[is_train, "is_short"].to_numpy() == 0])),
        "snapshot_date": SNAPSHOT_DATE,
        "trades_csvs": [str(p) for p in TRADES_CSVS],
    }
    META_OUT.write_text(json.dumps(meta, indent=2, default=str))
    SIZING_OUT.write_text(json.dumps(default_sizing_config().to_dict(), indent=2))

    print(f"\n{final_kind} saved to {MODEL_OUT}")
    print(f"meta written to {META_OUT}")
    print(f"sizing config written to {SIZING_OUT}")

    elapsed = time.monotonic() - started
    print(f"\ntotal wall: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
