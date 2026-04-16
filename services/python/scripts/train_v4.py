"""V4 LightGBM Walk-Forward Training.

Trains per-coin LightGBM multiclass classifiers (SHORT/FLAT/LONG)
using walk-forward validation. Outputs:
- models/v4/{ASSET}/fold_{N}.txt  (LightGBM model files)
- models/v4/{ASSET}/results.json  (per-fold metrics + aggregated)
- models/v4/summary.json          (all-coin summary)
"""

import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

sys.path.insert(0, ".")
from src.ml.walk_forward import walk_forward_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v4")
MODEL_DIR = Path("C:/Users/togat/Desktop/AI-Finance/models/v4")

V4_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]

FEATURE_COLS = [
    "ret_4", "ret_16", "ret_96", "ret_672", "ret_4_lag1",
    "price_to_sma_96", "price_to_sma_672", "ema_ratio", "macd_hist_norm",
    "rsi_norm", "bb_pct",
    "vol_4h", "vol_1d", "vol_ratio", "parkinson_vol",
    "volume_ratio", "taker_buy_ratio",
    "clv", "drawdown",
    "hour_sin", "hour_cos",
    "funding_rate", "funding_ma_3d", "funding_zscore", "cum_funding_3d",
    "btc_ret_96", "btc_residual", "altcoin_dispersion",
    "funding_x_rsi",
]

LGB_PARAMS = {
    "objective": "multiclass",
    "num_class": 3,
    "num_leaves": 47,
    "min_child_samples": 200,
    "learning_rate": 0.01,
    "feature_fraction": 0.6,
    "max_depth": 7,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

EARLY_STOPPING_ROUNDS = 50

TRAIN_SIZE = 26208
VAL_SIZE = 2880
TEST_SIZE = 2880
STEP_SIZE = 1344
EMBARGO = 16


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    params: dict,
) -> dict:
    """Train LightGBM on one fold, return metrics + predictions."""
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    proba = model.predict_proba(X_test)
    pred = model.predict(X_test)
    conviction = proba[:, 2] - proba[:, 0]

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=[0, 1, 2])

    class_acc = {}
    for c in [0, 1, 2]:
        mask = y_test == c
        if mask.sum() > 0:
            class_acc[int(c)] = float(accuracy_score(y_test[mask], pred[mask]))

    return {
        "accuracy": float(acc),
        "log_loss": float(ll),
        "class_accuracy": class_acc,
        "n_test": int(len(y_test)),
        "best_iteration": int(model.best_iteration_) if hasattr(model, "best_iteration_") else params["n_estimators"],
        "conviction": conviction.tolist(),
        "predictions": pred.tolist(),
        "actuals": y_test.tolist(),
        "model": model,
    }


def train_asset(asset: str) -> dict:
    """Run full walk-forward training for one asset."""
    parquet_path = FEATURE_DIR / f"{asset}_features.parquet"
    df = pd.read_parquet(parquet_path)
    logger.info(f"  {asset}: loaded {len(df):,} rows")

    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)
    timestamps = df["timestamp"].values

    splits = walk_forward_splits(
        n_samples=len(df),
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        embargo=EMBARGO,
    )
    logger.info(f"  {asset}: {len(splits)} walk-forward folds")

    asset_dir = MODEL_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    all_predictions = []
    all_actuals = []
    all_convictions = []
    all_timestamps = []

    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        result = train_one_fold(
            X_train=X[train_idx[0]:train_idx[1]],
            y_train=y[train_idx[0]:train_idx[1]],
            X_val=X[val_idx[0]:val_idx[1]],
            y_val=y[val_idx[0]:val_idx[1]],
            X_test=X[test_idx[0]:test_idx[1]],
            y_test=y[test_idx[0]:test_idx[1]],
            params=LGB_PARAMS,
        )

        model = result.pop("model")
        model_path = asset_dir / f"fold_{i:02d}.txt"
        model.booster_.save_model(str(model_path))

        all_predictions.extend(result["predictions"])
        all_actuals.extend(result["actuals"])
        all_convictions.extend(result["conviction"])
        all_timestamps.extend(timestamps[test_idx[0]:test_idx[1]].tolist())

        fold_results.append({
            "fold": i,
            "train_range": [int(train_idx[0]), int(train_idx[1])],
            "test_range": [int(test_idx[0]), int(test_idx[1])],
            "accuracy": result["accuracy"],
            "log_loss": result["log_loss"],
            "class_accuracy": result["class_accuracy"],
            "best_iteration": result["best_iteration"],
            "n_test": result["n_test"],
        })

        if (i + 1) % 5 == 0 or i == len(splits) - 1:
            logger.info(f"  {asset}: fold {i+1}/{len(splits)} | "
                        f"acc={result['accuracy']:.3f} ll={result['log_loss']:.3f} "
                        f"iters={result['best_iteration']}")

    all_preds = np.array(all_predictions)
    all_acts = np.array(all_actuals)
    all_conv = np.array(all_convictions)

    agg = {
        "asset": asset,
        "n_folds": len(splits),
        "total_oos_samples": len(all_preds),
        "oos_accuracy": float(accuracy_score(all_acts, all_preds)),
        "oos_log_loss": float(log_loss(all_acts,
                                        np.eye(3)[all_preds.astype(int)],
                                        labels=[0, 1, 2])),
        "mean_fold_accuracy": float(np.mean([f["accuracy"] for f in fold_results])),
        "std_fold_accuracy": float(np.std([f["accuracy"] for f in fold_results])),
        "conviction_mean": float(np.mean(all_conv)),
        "conviction_std": float(np.std(all_conv)),
        "high_conviction_pct": float(np.mean(np.abs(all_conv) > 0.35)),
    }

    for c in [0, 1, 2]:
        mask = all_acts == c
        if mask.sum() > 0:
            agg[f"oos_class_{c}_accuracy"] = float(accuracy_score(all_acts[mask], all_preds[mask]))

    results = {"aggregate": agg, "folds": fold_results}
    results_path = asset_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    oos_df = pd.DataFrame({
        "timestamp": all_timestamps,
        "prediction": all_predictions,
        "actual": all_actuals,
        "conviction": all_convictions,
    })
    oos_path = asset_dir / "oos_predictions.parquet"
    oos_df.to_parquet(oos_path, index=False)

    logger.info(f"  {asset}: OOS accuracy={agg['oos_accuracy']:.3f} "
                f"high_conviction={agg['high_conviction_pct']:.1%}")

    return agg


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    logger.info("V4 LightGBM Walk-Forward Training")
    logger.info(f"Assets: {V4_ASSETS}")
    logger.info(f"Params: leaves={LGB_PARAMS['num_leaves']} depth={LGB_PARAMS['max_depth']} "
                f"lr={LGB_PARAMS['learning_rate']} est={LGB_PARAMS['n_estimators']}")

    all_results = []
    for asset in V4_ASSETS:
        logger.info(f"\nTraining {asset}...")
        result = train_asset(asset)
        all_results.append(result)

    summary_path = MODEL_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"\nResults:")
    for r in all_results:
        logger.info(f"  {r['asset']}: OOS acc={r['oos_accuracy']:.3f} "
                    f"folds={r['n_folds']} "
                    f"high_conv={r['high_conviction_pct']:.1%}")


if __name__ == "__main__":
    main()
