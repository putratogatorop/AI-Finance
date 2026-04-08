"""V4.1 Binary LightGBM Walk-Forward Training.

Fixes v4.0's fatal flaw: the 3-class model predicted FLAT 99% of the time.
v4.1 uses two binary classifiers (LONG vs rest, SHORT vs rest) + regression
ensemble for conviction scoring.

Outputs:
- models/v4.1/{ASSET}/fold_{N}_long.txt
- models/v4.1/{ASSET}/fold_{N}_short.txt
- models/v4.1/{ASSET}/fold_{N}_reg.txt
- models/v4.1/{ASSET}/results.json
- models/v4.1/{ASSET}/oos_predictions.parquet
- models/v4.1/summary.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

sys.path.insert(0, ".")
from src.ml.walk_forward import walk_forward_splits

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v4")
MODEL_DIR = Path("C:/Users/togat/Desktop/AI-Finance/models/v4.1")

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

BINARY_PARAMS = {
    "objective": "binary",
    "num_leaves": 127,
    "min_child_samples": 50,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": -1,
    "n_estimators": 3000,
    "scale_pos_weight": 4.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

REG_PARAMS = {
    "objective": "regression",
    "num_leaves": 63,
    "min_child_samples": 100,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": -1,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

EARLY_STOPPING = 50

TRAIN_SIZE = 26208
VAL_SIZE = 2880
TEST_SIZE = 2880
STEP_SIZE = 1344
EMBARGO = 16


def train_one_fold_binary(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    params: dict,
) -> dict:
    y_train_long = (y_train == 2).astype(int)
    y_val_long = (y_val == 2).astype(int)
    y_test_long = (y_test == 2).astype(int)

    y_train_short = (y_train == 0).astype(int)
    y_val_short = (y_val == 0).astype(int)
    y_test_short = (y_test == 0).astype(int)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    long_model = lgb.LGBMClassifier(**params)
    long_model.fit(X_train, y_train_long,
                   eval_set=[(X_val, y_val_long)], callbacks=callbacks)

    short_model = lgb.LGBMClassifier(**params)
    short_model.fit(X_train, y_train_short,
                    eval_set=[(X_val, y_val_short)], callbacks=callbacks)

    y_train_reg = (y_train - 1).astype(float)
    y_val_reg = (y_val - 1).astype(float)
    reg_model = lgb.LGBMRegressor(**REG_PARAMS)
    reg_model.fit(X_train, y_train_reg,
                  eval_set=[(X_val, y_val_reg)], callbacks=callbacks)

    p_long = long_model.predict_proba(X_test)[:, 1]
    p_short = short_model.predict_proba(X_test)[:, 1]
    reg_pred = reg_model.predict(X_test)

    raw_conviction = p_long - p_short
    reg_sign = np.sign(reg_pred)
    conv_sign = np.sign(raw_conviction)
    agreement = (reg_sign == conv_sign).astype(float)
    conviction = raw_conviction * (0.5 + 0.5 * agreement)

    pred_long = (p_long > 0.5).astype(int)
    pred_short = (p_short > 0.5).astype(int)

    long_precision = precision_score(y_test_long, pred_long, zero_division=0)
    long_recall = recall_score(y_test_long, pred_long, zero_division=0)
    long_f1 = f1_score(y_test_long, pred_long, zero_division=0)

    short_precision = precision_score(y_test_short, pred_short, zero_division=0)
    short_recall = recall_score(y_test_short, pred_short, zero_division=0)
    short_f1 = f1_score(y_test_short, pred_short, zero_division=0)

    return {
        "long_precision": float(long_precision),
        "long_recall": float(long_recall),
        "long_f1": float(long_f1),
        "short_precision": float(short_precision),
        "short_recall": float(short_recall),
        "short_f1": float(short_f1),
        "long_pred_rate": float(pred_long.mean()),
        "short_pred_rate": float(pred_short.mean()),
        "conviction": conviction.tolist(),
        "actuals": y_test.tolist(),
        "n_test": int(len(y_test)),
        "long_best_iter": int(long_model.best_iteration_),
        "short_best_iter": int(short_model.best_iteration_),
        "reg_best_iter": int(reg_model.best_iteration_),
        "long_model": long_model,
        "short_model": short_model,
        "reg_model": reg_model,
    }


def train_asset(asset: str) -> dict:
    parquet_path = FEATURE_DIR / f"{asset}_features.parquet"
    df = pd.read_parquet(parquet_path)
    logger.info(f"  {asset}: loaded {len(df):,} rows")

    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)
    timestamps = df["timestamp"].values

    splits = walk_forward_splits(
        n_samples=len(df),
        train_size=TRAIN_SIZE, val_size=VAL_SIZE,
        test_size=TEST_SIZE, step_size=STEP_SIZE, embargo=EMBARGO,
    )
    logger.info(f"  {asset}: {len(splits)} walk-forward folds")

    asset_dir = MODEL_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    fold_results = []
    all_convictions = []
    all_actuals = []
    all_timestamps = []

    for i, (train_idx, val_idx, test_idx) in enumerate(splits):
        result = train_one_fold_binary(
            X_train=X[train_idx[0]:train_idx[1]],
            y_train=y[train_idx[0]:train_idx[1]],
            X_val=X[val_idx[0]:val_idx[1]],
            y_val=y[val_idx[0]:val_idx[1]],
            X_test=X[test_idx[0]:test_idx[1]],
            y_test=y[test_idx[0]:test_idx[1]],
            params=BINARY_PARAMS,
        )

        for name in ["long", "short", "reg"]:
            model = result.pop(f"{name}_model")
            path = asset_dir / f"fold_{i:02d}_{name}.txt"
            model.booster_.save_model(str(path))

        all_convictions.extend(result["conviction"])
        all_actuals.extend(result["actuals"])
        all_timestamps.extend(timestamps[test_idx[0]:test_idx[1]].tolist())

        fold_results.append({
            "fold": i,
            "long_precision": result["long_precision"],
            "long_recall": result["long_recall"],
            "long_f1": result["long_f1"],
            "short_precision": result["short_precision"],
            "short_recall": result["short_recall"],
            "short_f1": result["short_f1"],
            "long_pred_rate": result["long_pred_rate"],
            "short_pred_rate": result["short_pred_rate"],
        })

        if (i + 1) % 10 == 0 or i == len(splits) - 1:
            logger.info(
                f"  {asset}: fold {i+1}/{len(splits)} | "
                f"L_prec={result['long_precision']:.3f} L_rec={result['long_recall']:.3f} "
                f"S_prec={result['short_precision']:.3f} S_rec={result['short_recall']:.3f}"
            )

    conv = np.array(all_convictions)
    acts = np.array(all_actuals)

    buckets = {}
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
        long_mask = conv >= thresh
        short_mask = conv <= -thresh
        if long_mask.sum() > 0:
            buckets[f"long_{thresh:.2f}_acc"] = float((acts[long_mask] == 2).mean())
            buckets[f"long_{thresh:.2f}_n"] = int(long_mask.sum())
        if short_mask.sum() > 0:
            buckets[f"short_{thresh:.2f}_acc"] = float((acts[short_mask] == 0).mean())
            buckets[f"short_{thresh:.2f}_n"] = int(short_mask.sum())

    agg = {
        "asset": asset,
        "n_folds": len(splits),
        "total_oos_samples": len(conv),
        "mean_long_precision": float(np.mean([f["long_precision"] for f in fold_results])),
        "mean_long_recall": float(np.mean([f["long_recall"] for f in fold_results])),
        "mean_short_precision": float(np.mean([f["short_precision"] for f in fold_results])),
        "mean_short_recall": float(np.mean([f["short_recall"] for f in fold_results])),
        "conviction_std": float(conv.std()),
        "conviction_range": [float(conv.min()), float(conv.max())],
        "buckets": buckets,
    }

    with open(asset_dir / "results.json", "w") as f:
        json.dump({"aggregate": agg, "folds": fold_results}, f, indent=2, default=str)

    oos_df = pd.DataFrame({
        "timestamp": all_timestamps,
        "conviction": all_convictions,
        "actual": all_actuals,
    })
    oos_df.to_parquet(asset_dir / "oos_predictions.parquet", index=False)

    logger.info(f"  {asset}: L_prec={agg['mean_long_precision']:.3f} "
                f"L_rec={agg['mean_long_recall']:.3f} "
                f"S_prec={agg['mean_short_precision']:.3f} "
                f"S_rec={agg['mean_short_recall']:.3f} "
                f"conv_std={agg['conviction_std']:.3f}")

    return agg


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    logger.info("V4.1 Binary LightGBM Walk-Forward Training")
    logger.info(f"Assets: {V4_ASSETS}")
    logger.info(f"Binary params: leaves={BINARY_PARAMS['num_leaves']} "
                f"min_child={BINARY_PARAMS['min_child_samples']} "
                f"scale_pos_weight={BINARY_PARAMS['scale_pos_weight']}")

    all_results = []
    for asset in V4_ASSETS:
        logger.info(f"\nTraining {asset}...")
        result = train_asset(asset)
        all_results.append(result)

    with open(MODEL_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    for r in all_results:
        logger.info(f"  {r['asset']}: L_prec={r['mean_long_precision']:.3f} "
                    f"L_rec={r['mean_long_recall']:.3f} "
                    f"S_prec={r['mean_short_precision']:.3f} "
                    f"S_rec={r['mean_short_recall']:.3f}")
        for k, v in r.get("buckets", {}).items():
            if k.endswith("_acc"):
                n_key = k.replace("_acc", "_n")
                n = r["buckets"].get(n_key, 0)
                logger.info(f"    {k}: {v:.3f} (n={n})")


if __name__ == "__main__":
    main()
