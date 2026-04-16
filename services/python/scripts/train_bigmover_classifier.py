"""
Train a LightGBM classifier to predict which volume spikes become big movers (±10% in 24h).

Walk-forward validation:
  Train:   2023-04 to 2025-03 (24 months)
  Embargo: 2025-04 (1 month gap)
  Test:    2025-05 to 2025-10 (6 months)
  Holdout: 2025-11 to 2026-04 (final check, touch ONCE)

Run from services/python/:
  python scripts/train_bigmover_classifier.py
"""

import sys
sys.path.insert(0, ".")

import logging
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, classification_report,
)
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
MODEL_PATH = Path("models/bigmover_classifier.joblib")

FEATURE_COLS = [
    # Raw features
    "vol_ratio", "buy_ratio", "price_change_1bar", "price_change_2bar",
    "bar_range_pct", "upper_wick_pct", "lower_wick_pct", "body_pct",
    "atr_14", "rsi_14", "volatility_20",
    "dist_from_20_high", "dist_from_20_low", "price_vs_ema_50",
    "btc_ret_24bar", "btc_ret_96bar", "btc_vol_ratio",
    "direction", "tf_numeric",
    # Engineered features
    "avg_trade_size", "vol_x_buyrat", "candle_quality", "wick_ratio",
    "dollar_imbalance", "exec_vs_close", "vol_ratio_adj",
    "trend_aligned", "btc_aligned", "dist_range_pos",
    "coin_gainer_ratio",
]

LGB_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "scale_pos_weight": 7.5,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": 42,
}


def load_data() -> pd.DataFrame:
    """Load volume_breakouts from PostgreSQL."""
    logger.info("Loading data from PostgreSQL...")
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT * FROM volume_breakouts WHERE fwd_max_gain_24h IS NOT NULL",
        engine,
    )
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    logger.info(f"Loaded {len(df):,} rows, date range: {df['signal_time'].min()} to {df['signal_time'].max()}")
    logger.info(f"Big movers: {df['is_big_mover'].sum():,} ({df['is_big_mover'].mean()*100:.1f}%)")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute cross-features from existing columns."""
    df = df.copy()
    df["avg_trade_size"] = df["quote_volume"] / df["trades"].clip(lower=1)
    df["vol_x_buyrat"] = df["vol_ratio"] * df["buy_ratio"]
    df["candle_quality"] = df["body_pct"] - df["upper_wick_pct"]
    df["wick_ratio"] = df["upper_wick_pct"] / (df["lower_wick_pct"] + 0.001)
    df["dollar_imbalance"] = (
        (2 * df["taker_buy_quote"] - df["quote_volume"]) / df["quote_volume"].clip(lower=1)
    )
    df["exec_vs_close"] = (
        (df["quote_volume"] / df["volume"].clip(lower=1e-10)) / df["close"].clip(lower=1e-10)
    )
    df["vol_ratio_adj"] = df["vol_ratio"] / (df["volatility_20"] + 0.001)
    df["trend_aligned"] = (df["direction"] * df["price_vs_ema_50"]).clip(lower=0)
    df["btc_aligned"] = (df["direction"] * df["btc_ret_24bar"]).clip(lower=0)
    df["dist_range_pos"] = df["dist_from_20_high"] / (
        df["dist_from_20_high"].abs() + df["dist_from_20_low"].abs() + 0.001
    )

    tf_map = {"15m": 0, "1h": 1, "4h": 2}
    df["tf_numeric"] = df["timeframe"].map(tf_map)

    return df


def add_coin_gainer_ratio(train_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
    """Compute coin_gainer_ratio from train data and apply to target."""
    target_df = target_df.copy()
    coin_stats = train_df.groupby("symbol")["is_big_mover"].agg(["sum", "count"])
    coin_stats["coin_gainer_ratio"] = coin_stats["sum"] / coin_stats["count"]
    global_ratio = train_df["is_big_mover"].mean()
    ratio_map = coin_stats["coin_gainer_ratio"].to_dict()
    target_df["coin_gainer_ratio"] = target_df["symbol"].map(ratio_map).fillna(global_ratio)
    return target_df


def clean_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace inf/nan in feature columns."""
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def evaluate_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    fwd_max_gain: np.ndarray,
    fwd_max_loss: np.ndarray,
    label: str = "Test",
) -> pd.DataFrame:
    """Sweep thresholds, compute precision/recall/F1/PF."""
    results = []
    fwd_max_loss_abs = np.abs(fwd_max_loss)

    for thresh in np.arange(0.1, 0.91, 0.05):
        preds = (y_prob >= thresh).astype(int)
        n_pos = preds.sum()
        if n_pos == 0:
            continue

        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)

        # Profit factor: sum MFE of correct / sum MAE of incorrect
        correct_mask = (preds == 1) & (y_true == 1)
        wrong_mask = (preds == 1) & (y_true == 0)

        mfe_correct = fwd_max_gain[correct_mask].sum() if correct_mask.any() else 0
        mae_wrong = fwd_max_loss_abs[wrong_mask].sum() if wrong_mask.any() else 0
        pf = mfe_correct / mae_wrong if mae_wrong > 0 else float("inf")

        results.append({
            "threshold": round(thresh, 2),
            "n_preds": n_pos,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "profit_factor": round(pf, 3),
        })

    results_df = pd.DataFrame(results)
    logger.info(f"\n{'='*70}\n{label} — Threshold Sweep\n{'='*70}")
    logger.info(f"\n{results_df.to_string(index=False)}")
    return results_df


def evaluate_baselines(y_true, fwd_max_gain, fwd_max_loss, vol_ratio, label="Test"):
    """Compare against simple baselines."""
    fwd_max_loss_abs = np.abs(fwd_max_loss)
    base_rate = y_true.mean()

    logger.info(f"\n{'='*70}\n{label} — Baselines\n{'='*70}")

    # Random baseline
    np.random.seed(42)
    random_preds = (np.random.rand(len(y_true)) < base_rate).astype(int)
    rp = precision_score(y_true, random_preds, zero_division=0)
    rr = recall_score(y_true, random_preds, zero_division=0)
    logger.info(f"Random ({base_rate*100:.1f}% rate): precision={rp:.4f}, recall={rr:.4f}")

    # Vol ratio baselines
    for vr_thresh in [3.0, 5.0]:
        vr_preds = (vol_ratio >= vr_thresh).astype(int)
        n = vr_preds.sum()
        if n > 0:
            vp = precision_score(y_true, vr_preds, zero_division=0)
            vrec = recall_score(y_true, vr_preds, zero_division=0)
            correct = (vr_preds == 1) & (y_true == 1)
            wrong = (vr_preds == 1) & (y_true == 0)
            mfe = fwd_max_gain[correct].sum() if correct.any() else 0
            mae = fwd_max_loss_abs[wrong].sum() if wrong.any() else 0
            vpf = mfe / mae if mae > 0 else float("inf")
            logger.info(
                f"vol_ratio>={vr_thresh}: n={n}, precision={vp:.4f}, "
                f"recall={vrec:.4f}, PF={vpf:.3f}"
            )


def main():
    # ── 1. Load data ──────────────────────────────────────────────────────
    df = load_data()

    # ── 2. Engineer features ──────────────────────────────────────────────
    df = engineer_features(df)

    # ── 3. Time-based splits ──────────────────────────────────────────────
    train_mask = (df["signal_time"] >= "2023-04-01") & (df["signal_time"] < "2025-04-01")
    test_mask = (df["signal_time"] >= "2025-05-01") & (df["signal_time"] < "2025-11-01")
    holdout_mask = (df["signal_time"] >= "2025-11-01") & (df["signal_time"] < "2026-05-01")

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    holdout_df = df[holdout_mask].copy()

    logger.info(
        f"Split sizes — Train: {len(train_df):,}, Test: {len(test_df):,}, "
        f"Holdout: {len(holdout_df):,}"
    )
    logger.info(
        f"Big mover rates — Train: {train_df['is_big_mover'].mean()*100:.1f}%, "
        f"Test: {test_df['is_big_mover'].mean()*100:.1f}%, "
        f"Holdout: {holdout_df['is_big_mover'].mean()*100:.1f}%"
    )

    # ── 4. Add coin_gainer_ratio (no look-ahead) ─────────────────────────
    train_df = add_coin_gainer_ratio(train_df, train_df)
    test_df = add_coin_gainer_ratio(train_df, test_df)
    holdout_df = add_coin_gainer_ratio(train_df, holdout_df)

    # ── 5. Clean features ─────────────────────────────────────────────────
    train_df = clean_features(train_df, FEATURE_COLS)
    test_df = clean_features(test_df, FEATURE_COLS)
    holdout_df = clean_features(holdout_df, FEATURE_COLS)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["is_big_mover"].astype(int)
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["is_big_mover"].astype(int)
    X_holdout = holdout_df[FEATURE_COLS]
    y_holdout = holdout_df["is_big_mover"].astype(int)

    logger.info(f"Feature matrix: {X_train.shape[1]} features")

    # ── 6. Train LightGBM ────────────────────────────────────────────────
    logger.info("Training LightGBM...")
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(100)],
    )

    # ── 7. Feature importance ─────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    logger.info(f"\n{'='*70}\nTop 15 Feature Importances\n{'='*70}")
    logger.info(f"\n{importance.head(15).to_string(index=False)}")

    # ── 8. Evaluate on test set ───────────────────────────────────────────
    test_probs = model.predict_proba(X_test)[:, 1]
    test_results = evaluate_thresholds(
        y_test.values,
        test_probs,
        test_df["fwd_max_gain_24h"].values,
        test_df["fwd_max_loss_24h"].values,
        label="Test (2025-05 to 2025-10)",
    )

    # Best threshold by F1
    if len(test_results) > 0:
        best_row = test_results.loc[test_results["f1"].idxmax()]
        best_thresh = best_row["threshold"]
        logger.info(f"\nBest test threshold (by F1): {best_thresh}")

        test_preds = (test_probs >= best_thresh).astype(int)
        cm = confusion_matrix(y_test, test_preds)
        logger.info(f"Confusion matrix at threshold={best_thresh}:\n{cm}")
        logger.info(f"\n{classification_report(y_test, test_preds, target_names=['No Move', 'Big Mover'])}")

    # Results per timeframe
    logger.info(f"\n{'='*70}\nTest Results by Timeframe\n{'='*70}")
    for tf in ["15m", "1h", "4h"]:
        tf_mask = test_df["timeframe"] == tf
        if tf_mask.sum() == 0:
            continue
        tf_probs = test_probs[tf_mask.values]
        tf_y = y_test[tf_mask].values
        tf_preds = (tf_probs >= best_thresh).astype(int)
        n_pos = tf_preds.sum()
        if n_pos > 0:
            prec = precision_score(tf_y, tf_preds, zero_division=0)
            rec = recall_score(tf_y, tf_preds, zero_division=0)
            logger.info(f"  {tf}: n={tf_mask.sum():,}, preds={n_pos}, prec={prec:.4f}, rec={rec:.4f}")
        else:
            logger.info(f"  {tf}: n={tf_mask.sum():,}, no positive predictions at threshold={best_thresh}")

    # ── 9. Evaluate on holdout set ────────────────────────────────────────
    holdout_probs = model.predict_proba(X_holdout)[:, 1]
    holdout_results = evaluate_thresholds(
        y_holdout.values,
        holdout_probs,
        holdout_df["fwd_max_gain_24h"].values,
        holdout_df["fwd_max_loss_24h"].values,
        label="Holdout (2025-11 to 2026-04)",
    )

    if len(holdout_results) > 0:
        holdout_preds = (holdout_probs >= best_thresh).astype(int)
        cm_h = confusion_matrix(y_holdout, holdout_preds)
        logger.info(f"\nHoldout confusion matrix at threshold={best_thresh}:\n{cm_h}")
        logger.info(
            f"\n{classification_report(y_holdout, holdout_preds, target_names=['No Move', 'Big Mover'])}"
        )

    # ── 10. Baselines ────────────────────────────────────────────────────
    evaluate_baselines(
        y_test.values,
        test_df["fwd_max_gain_24h"].values,
        test_df["fwd_max_loss_24h"].values,
        test_df["vol_ratio"].values,
        label="Test Baselines",
    )
    evaluate_baselines(
        y_holdout.values,
        holdout_df["fwd_max_gain_24h"].values,
        holdout_df["fwd_max_loss_24h"].values,
        holdout_df["vol_ratio"].values,
        label="Holdout Baselines",
    )

    # ── 11. Summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    for lbl, res_df in [("Test", test_results), ("Holdout", holdout_results)]:
        if len(res_df) == 0:
            continue
        logger.info(f"\n{lbl} set — key thresholds:")
        for _, row in res_df.iterrows():
            logger.info(
                f"  t={row['threshold']:.2f}: n={int(row['n_preds']):>6}, "
                f"prec={row['precision']:.3f}, rec={row['recall']:.3f}, "
                f"F1={row['f1']:.3f}, PF={row['profit_factor']:.2f}"
            )

    # ── 12. Save model ───────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
