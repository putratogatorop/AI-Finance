"""
Train a LightGBM regressor to predict MFE (max favorable excursion) at 24h for big mover events.
This model predicts HOW BIG the move will be, enabling optimal TP/trailing stop placement.

Walk-forward validation:
  Train:   2023-04 to 2025-03 (24 months)
  Embargo: 2025-04 (1 month gap)
  Test:    2025-05 to 2025-10 (6 months)
  Holdout: 2025-11 to 2026-04 (final check, touch ONCE)

Only trains on is_big_mover = True rows (~154k).

Run from services/python/:
  python scripts/train_exit_model.py
"""

import sys
sys.path.insert(0, ".")

import logging
import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sqlalchemy import create_engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
MODEL_PATH = Path("models/bigmover_exit_model.joblib")

FEATURE_COLS = [
    # Raw features
    "vol_ratio", "buy_ratio", "price_change_1bar", "price_change_2bar",
    "bar_range_pct", "upper_wick_pct", "lower_wick_pct", "body_pct",
    "atr_14", "rsi_14", "volatility_20",
    "dist_from_20_high", "dist_from_20_low", "price_vs_ema_50",
    "btc_ret_24bar", "btc_ret_96bar", "btc_vol_ratio",
    "direction", "tf_numeric",
    # Engineered features (same as entry model)
    "avg_trade_size", "vol_x_buyrat", "candle_quality", "wick_ratio",
    "dollar_imbalance", "exec_vs_close", "vol_ratio_adj",
    "trend_aligned", "btc_aligned", "dist_range_pos",
    "coin_gainer_ratio",
    # MFE-specific features
    "price_range_to_atr", "buy_pressure_strength",
]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "min_child_samples": 100,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "verbose": -1,
    "random_state": 42,
}


def load_data() -> pd.DataFrame:
    """Load volume_breakouts from PostgreSQL, big movers only."""
    logger.info("Loading data from PostgreSQL...")
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT * FROM volume_breakouts WHERE fwd_max_gain_24h IS NOT NULL AND is_big_mover = TRUE",
        engine,
    )
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    logger.info(
        f"Loaded {len(df):,} big mover rows, "
        f"date range: {df['signal_time'].min()} to {df['signal_time'].max()}"
    )
    logger.info(
        f"MFE 24h stats — mean: {df['fwd_max_gain_24h'].mean()*100:.1f}%, "
        f"median: {df['fwd_max_gain_24h'].median()*100:.1f}%, "
        f"p75: {df['fwd_max_gain_24h'].quantile(0.75)*100:.1f}%, "
        f"p90: {df['fwd_max_gain_24h'].quantile(0.90)*100:.1f}%"
    )
    return df


def load_all_data_for_coin_ratio() -> pd.DataFrame:
    """Load ALL rows (not just big movers) for computing coin_gainer_ratio."""
    logger.info("Loading all data for coin_gainer_ratio computation...")
    engine = create_engine(DB_URL)
    df = pd.read_sql(
        "SELECT symbol, signal_time, is_big_mover FROM volume_breakouts "
        "WHERE fwd_max_gain_24h IS NOT NULL",
        engine,
    )
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    logger.info(f"Loaded {len(df):,} total rows for coin ratio")
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

    # MFE-specific features
    df["price_range_to_atr"] = df["bar_range_pct"] / (df["atr_14"] + 0.001)
    df["buy_pressure_strength"] = df["buy_ratio"] * df["vol_ratio"]

    return df


def add_coin_gainer_ratio(
    all_train_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute coin_gainer_ratio from ALL train data (not just big movers) and apply to target."""
    target_df = target_df.copy()
    coin_stats = all_train_df.groupby("symbol")["is_big_mover"].agg(["sum", "count"])
    coin_stats["coin_gainer_ratio"] = coin_stats["sum"] / coin_stats["count"]
    global_ratio = all_train_df["is_big_mover"].mean()
    ratio_map = coin_stats["coin_gainer_ratio"].to_dict()
    target_df["coin_gainer_ratio"] = target_df["symbol"].map(ratio_map).fillna(global_ratio)
    return target_df


def clean_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Replace inf/nan in feature columns."""
    df = df.copy()
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    return df


def evaluate_regression(
    y_true: np.ndarray, y_pred: np.ndarray, label: str
) -> dict:
    """Compute MAE, RMSE, R2."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    logger.info(f"\n{'='*70}\n{label} — Regression Metrics\n{'='*70}")
    logger.info(f"  MAE:  {mae*100:.2f}%")
    logger.info(f"  RMSE: {rmse*100:.2f}%")
    logger.info(f"  R2:   {r2:.4f}")
    logger.info(f"  Actual mean: {y_true.mean()*100:.1f}%, Predicted mean: {y_pred.mean()*100:.1f}%")
    return {"mae": mae, "rmse": rmse, "r2": r2}


def evaluate_mfe_buckets(
    y_true: np.ndarray, y_pred: np.ndarray, label: str
) -> None:
    """Show predicted vs actual MFE by bucket."""
    logger.info(f"\n{'='*70}\n{label} — Predicted MFE Buckets\n{'='*70}")
    buckets = [
        ("Predicted 5-10%", 0.05, 0.10),
        ("Predicted 10-15%", 0.10, 0.15),
        ("Predicted 15-25%", 0.15, 0.25),
        ("Predicted 25%+", 0.25, 999),
    ]
    for name, lo, hi in buckets:
        mask = (y_pred >= lo) & (y_pred < hi)
        n = mask.sum()
        if n > 0:
            actual_mean = y_true[mask].mean() * 100
            actual_med = np.median(y_true[mask]) * 100
            actual_std = y_true[mask].std() * 100
            pred_mean = y_pred[mask].mean() * 100
            logger.info(
                f"  {name}: n={n:,}, pred_mean={pred_mean:.1f}%, "
                f"actual_mean={actual_mean:.1f}%, actual_median={actual_med:.1f}%, "
                f"actual_std={actual_std:.1f}%"
            )
        else:
            logger.info(f"  {name}: n=0")


def simulate_exit_strategy(
    y_pred: np.ndarray,
    actual_mfe: np.ndarray,
    fwd_ret_24h: np.ndarray,
    label: str,
) -> None:
    """Simulate 3-tier exit strategy vs baselines."""
    logger.info(f"\n{'='*70}\n{label} — Trade Management Simulation\n{'='*70}")
    n = len(y_pred)

    # ML-based 3-tier exits
    tp1_levels = 0.30 * y_pred
    tp2_levels = 0.60 * y_pred

    ml_captured = np.zeros(n)
    for i in range(n):
        tp1 = tp1_levels[i]
        tp2 = tp2_levels[i]
        mfe = actual_mfe[i]

        hit_tp1 = mfe >= tp1
        hit_tp2 = mfe >= tp2

        if hit_tp2:
            # 33% at tp1, 33% at tp2, 34% trails (captures ~50% of remaining)
            ml_captured[i] = 0.33 * tp1 + 0.33 * tp2 + 0.34 * (mfe * 0.5)
        elif hit_tp1:
            # 33% at tp1, 67% trails less effectively
            ml_captured[i] = 0.33 * tp1 + 0.67 * (mfe * 0.3)
        else:
            # Barely captured
            ml_captured[i] = mfe * 0.2

    # Baseline 1: Hold to 24h close (handle NaN)
    baseline_hold = np.nan_to_num(fwd_ret_24h, nan=0.0)

    # Baseline 2: Fixed 5% TP (0.05 in fraction)
    baseline_5pct = np.minimum(actual_mfe, 0.05)

    # Baseline 3: Fixed 10% TP (0.10 in fraction)
    baseline_10pct = np.minimum(actual_mfe, 0.10)

    logger.info(f"  Events: {n:,}")
    logger.info(f"  Avg actual MFE: {actual_mfe.mean()*100:.1f}%")
    logger.info(f"  Avg predicted MFE: {y_pred.mean()*100:.1f}%")
    logger.info("")
    logger.info(f"  ML 3-Tier Exit:     avg captured = {ml_captured.mean()*100:.2f}%, "
                f"median = {np.median(ml_captured)*100:.2f}%")
    logger.info(f"  Baseline Hold 24h:  avg captured = {baseline_hold.mean()*100:.2f}%, "
                f"median = {np.median(baseline_hold)*100:.2f}%")
    logger.info(f"  Baseline Fixed 5%:  avg captured = {baseline_5pct.mean()*100:.2f}%, "
                f"median = {np.median(baseline_5pct)*100:.2f}%")
    logger.info(f"  Baseline Fixed 10%: avg captured = {baseline_10pct.mean()*100:.2f}%, "
                f"median = {np.median(baseline_10pct)*100:.2f}%")

    # Capture efficiency: total captured / total MFE available
    total_mfe = actual_mfe.sum()
    ml_efficiency = ml_captured.sum() / total_mfe * 100 if total_mfe > 0 else 0
    hold_efficiency = baseline_hold.sum() / total_mfe * 100 if total_mfe > 0 else 0
    fixed5_efficiency = baseline_5pct.sum() / total_mfe * 100 if total_mfe > 0 else 0
    fixed10_efficiency = baseline_10pct.sum() / total_mfe * 100 if total_mfe > 0 else 0

    logger.info("")
    logger.info(f"  Capture Efficiency (total captured / total MFE):")
    logger.info(f"    ML 3-Tier:     {ml_efficiency:.1f}%")
    logger.info(f"    Hold 24h:      {hold_efficiency:.1f}%")
    logger.info(f"    Fixed 5% TP:   {fixed5_efficiency:.1f}%")
    logger.info(f"    Fixed 10% TP:  {fixed10_efficiency:.1f}%")

    # Win rates (positive captured)
    logger.info("")
    logger.info(f"  Win Rates (captured > 0):")
    logger.info(f"    ML 3-Tier:     {(ml_captured > 0).mean()*100:.1f}%")
    logger.info(f"    Hold 24h:      {(baseline_hold > 0).mean()*100:.1f}%")


def main():
    # ── 1. Load data (big movers only) ───────────────────────────────────
    df = load_data()

    # Load all data for coin_gainer_ratio computation
    all_df = load_all_data_for_coin_ratio()

    # ── 2. Engineer features ─────────────────────────────────────────────
    df = engineer_features(df)

    # ── 3. Time-based splits ─────────────────────────────────────────────
    train_mask = (df["signal_time"] >= "2023-04-01") & (df["signal_time"] < "2025-04-01")
    test_mask = (df["signal_time"] >= "2025-05-01") & (df["signal_time"] < "2025-11-01")
    holdout_mask = (df["signal_time"] >= "2025-11-01") & (df["signal_time"] < "2026-05-01")

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()
    holdout_df = df[holdout_mask].copy()

    # Also split all_df for coin_gainer_ratio
    all_train_mask = (
        (all_df["signal_time"] >= "2023-04-01") & (all_df["signal_time"] < "2025-04-01")
    )
    all_train_df = all_df[all_train_mask].copy()

    logger.info(
        f"Split sizes (big movers only) — Train: {len(train_df):,}, "
        f"Test: {len(test_df):,}, Holdout: {len(holdout_df):,}"
    )

    # Target distribution
    for lbl, sdf in [("Train", train_df), ("Test", test_df), ("Holdout", holdout_df)]:
        y = sdf["fwd_max_gain_24h"]
        logger.info(
            f"  {lbl} MFE 24h — mean: {y.mean()*100:.1f}%, median: {y.median()*100:.1f}%, "
            f"std: {y.std()*100:.1f}%, min: {y.min()*100:.1f}%, max: {y.max()*100:.1f}%"
        )

    # ── 4. Add coin_gainer_ratio (no look-ahead) ────────────────────────
    train_df = add_coin_gainer_ratio(all_train_df, train_df)
    test_df = add_coin_gainer_ratio(all_train_df, test_df)
    holdout_df = add_coin_gainer_ratio(all_train_df, holdout_df)

    # ── 5. Clean features ────────────────────────────────────────────────
    train_df = clean_features(train_df, FEATURE_COLS)
    test_df = clean_features(test_df, FEATURE_COLS)
    holdout_df = clean_features(holdout_df, FEATURE_COLS)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["fwd_max_gain_24h"]
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["fwd_max_gain_24h"]
    X_holdout = holdout_df[FEATURE_COLS]
    y_holdout = holdout_df["fwd_max_gain_24h"]

    logger.info(f"Feature matrix: {X_train.shape[1]} features")

    # ── 6. Train LightGBM Regressor ─────────────────────────────────────
    logger.info("Training LightGBM Regressor...")
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.log_evaluation(100)],
    )

    # ── 7. Feature importance ────────────────────────────────────────────
    importance = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    logger.info(f"\n{'='*70}\nTop 15 Feature Importances\n{'='*70}")
    logger.info(f"\n{importance.head(15).to_string(index=False)}")

    # ── 8. Evaluate on test set ──────────────────────────────────────────
    test_pred = model.predict(X_test)
    evaluate_regression(y_test.values, test_pred, "Test (2025-05 to 2025-10)")
    evaluate_mfe_buckets(y_test.values, test_pred, "Test (2025-05 to 2025-10)")

    # ── 9. Evaluate on holdout set ───────────────────────────────────────
    holdout_pred = model.predict(X_holdout)
    evaluate_regression(y_holdout.values, holdout_pred, "Holdout (2025-11 to 2026-04)")
    evaluate_mfe_buckets(y_holdout.values, holdout_pred, "Holdout (2025-11 to 2026-04)")

    # ── 10. Trade management simulation ──────────────────────────────────
    simulate_exit_strategy(
        test_pred,
        y_test.values,
        test_df["fwd_ret_24h"].values,
        label="Test — Exit Strategy Simulation",
    )
    simulate_exit_strategy(
        holdout_pred,
        y_holdout.values,
        holdout_df["fwd_ret_24h"].values,
        label="Holdout — Exit Strategy Simulation",
    )

    # ── 11. Summary ──────────────────────────────────────────────────────
    logger.info(f"\n{'='*70}\nFINAL SUMMARY\n{'='*70}")
    test_metrics = evaluate_regression(y_test.values, test_pred, "Test Final")
    holdout_metrics = evaluate_regression(y_holdout.values, holdout_pred, "Holdout Final")

    # Prediction distribution
    logger.info(f"\n  Test predictions — "
                f"mean: {test_pred.mean()*100:.1f}%, std: {test_pred.std()*100:.1f}%, "
                f"min: {test_pred.min()*100:.1f}%, max: {test_pred.max()*100:.1f}%")
    logger.info(f"  Holdout predictions — "
                f"mean: {holdout_pred.mean()*100:.1f}%, std: {holdout_pred.std()*100:.1f}%, "
                f"min: {holdout_pred.min()*100:.1f}%, max: {holdout_pred.max()*100:.1f}%")

    # ── 12. Save model ──────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    logger.info(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
