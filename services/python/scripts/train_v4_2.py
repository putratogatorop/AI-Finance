# scripts/train_v4_2.py
"""V4.2 Binary Training on 1h bars + immediate backtest.

Same binary LONG/SHORT + regression ensemble as v4.1, but:
- 1h bars (4x less noise)
- 4 focused coins (BTC, LINK, XRP, AVAX)
- Walk-forward params adjusted for 1h (24 bars/day)
- Runs backtest immediately after training for quick feedback
"""

import json
import logging
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.ml.walk_forward import walk_forward_splits
from src.ml.backtester_v4 import backtest_coin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FEATURE_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v4.2")
MODEL_DIR = Path("C:/Users/togat/Desktop/AI-Finance/models/v4.2")
DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"

V4_2_ASSETS = ["BTC", "LINK", "XRP", "AVAX"]

FEATURE_COLS = [
    "ret_1", "ret_4", "ret_24", "ret_168", "ret_1_lag1",
    "price_to_sma_24", "price_to_sma_168", "ema_ratio", "macd_hist_norm",
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

# Walk-forward params for 1h bars (24 bars/day)
TRAIN_SIZE = 24 * 182   # 6 months = 4368 bars
VAL_SIZE = 24 * 30      # 1 month = 720 bars
TEST_SIZE = 24 * 30     # 1 month = 720 bars
STEP_SIZE = 24 * 14     # 2 weeks = 336 bars
EMBARGO = 4             # 4h target horizon


def train_one_fold(X_train, y_train, X_val, y_val, X_test, y_test):
    """Train binary LONG + SHORT + regression on one fold."""
    y_long = (y_train == 2).astype(int)
    y_val_long = (y_val == 2).astype(int)
    y_short = (y_train == 0).astype(int)
    y_val_short = (y_val == 0).astype(int)

    cb = [lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)]

    long_m = lgb.LGBMClassifier(**BINARY_PARAMS)
    long_m.fit(X_train, y_long, eval_set=[(X_val, y_val_long)], callbacks=cb)

    short_m = lgb.LGBMClassifier(**BINARY_PARAMS)
    short_m.fit(X_train, y_short, eval_set=[(X_val, y_val_short)], callbacks=cb)

    reg_m = lgb.LGBMRegressor(**REG_PARAMS)
    reg_m.fit(X_train, (y_train - 1).astype(float),
              eval_set=[(X_val, (y_val - 1).astype(float))], callbacks=cb)

    # Predict
    p_long = long_m.predict_proba(X_test)[:, 1]
    p_short = short_m.predict_proba(X_test)[:, 1]
    reg_pred = reg_m.predict(X_test)

    raw_conv = p_long - p_short
    agreement = (np.sign(reg_pred) == np.sign(raw_conv)).astype(float)
    conviction = raw_conv * (0.5 + 0.5 * agreement)

    # Metrics
    pred_l = (p_long > 0.5).astype(int)
    pred_s = (p_short > 0.5).astype(int)
    y_test_l = (y_test == 2).astype(int)
    y_test_s = (y_test == 0).astype(int)

    return {
        "conviction": conviction,
        "long_prec": precision_score(y_test_l, pred_l, zero_division=0),
        "long_rec": recall_score(y_test_l, pred_l, zero_division=0),
        "short_prec": precision_score(y_test_s, pred_s, zero_division=0),
        "short_rec": recall_score(y_test_s, pred_s, zero_division=0),
        "long_model": long_m,
        "short_model": short_m,
        "reg_model": reg_m,
    }


def train_and_backtest_asset(asset: str, engine) -> dict:
    df = pd.read_parquet(FEATURE_DIR / f"{asset}_features.parquet")
    logger.info(f"  {asset}: {len(df):,} rows")

    # Handle column name mismatch: 1h features use ret_1/ret_24 etc
    # but FEATURE_COLS references them — filter to available columns
    available_feat = [c for c in FEATURE_COLS if c in df.columns]
    # Also add 1h-specific names that might differ
    for old, new in [("ret_96", "ret_24"), ("ret_672", "ret_168"),
                     ("price_to_sma_96", "price_to_sma_24"),
                     ("price_to_sma_672", "price_to_sma_168"),
                     ("ret_4_lag1", "ret_1_lag1")]:
        if new in df.columns and old not in df.columns:
            available_feat = [new if c == old else c for c in available_feat]
    # Deduplicate and filter to what's actually in the dataframe
    feat_cols = list(dict.fromkeys(c for c in available_feat if c in df.columns))
    # Add any 1h-specific columns not yet included
    for c in df.columns:
        if c not in feat_cols and c not in ["timestamp", "target"]:
            feat_cols.append(c)
    # Final filter: only feature columns (no timestamp/target)
    feat_cols = [c for c in feat_cols if c in df.columns and c not in ["timestamp", "target"]]

    X = df[feat_cols].values
    y = df["target"].values.astype(int)
    timestamps = df["timestamp"].values

    splits = walk_forward_splits(
        n_samples=len(df), train_size=TRAIN_SIZE, val_size=VAL_SIZE,
        test_size=TEST_SIZE, step_size=STEP_SIZE, embargo=EMBARGO,
    )
    logger.info(f"  {asset}: {len(splits)} folds, {len(feat_cols)} features")

    asset_dir = MODEL_DIR / asset
    asset_dir.mkdir(parents=True, exist_ok=True)

    all_conv, all_acts, all_ts = [], [], []
    fold_metrics = []

    for i, (tr, va, te) in enumerate(splits):
        r = train_one_fold(
            X[tr[0]:tr[1]], y[tr[0]:tr[1]],
            X[va[0]:va[1]], y[va[0]:va[1]],
            X[te[0]:te[1]], y[te[0]:te[1]],
        )

        for name in ["long", "short", "reg"]:
            model = r.pop(f"{name}_model")
            model.booster_.save_model(str(asset_dir / f"fold_{i:02d}_{name}.txt"))

        all_conv.extend(r["conviction"].tolist())
        all_acts.extend(y[te[0]:te[1]].tolist())
        all_ts.extend(timestamps[te[0]:te[1]].tolist())

        fold_metrics.append({
            "fold": i, "long_prec": r["long_prec"], "long_rec": r["long_rec"],
            "short_prec": r["short_prec"], "short_rec": r["short_rec"],
        })

        if (i + 1) % 10 == 0 or i == len(splits) - 1:
            logger.info(f"  {asset}: fold {i+1}/{len(splits)} | "
                        f"L={r['long_prec']:.3f}/{r['long_rec']:.3f} "
                        f"S={r['short_prec']:.3f}/{r['short_rec']:.3f}")

    # Save OOS predictions
    conv = np.array(all_conv)
    acts = np.array(all_acts)

    oos_df = pd.DataFrame({"timestamp": all_ts, "conviction": all_conv, "actual": all_acts})
    oos_df.to_parquet(asset_dir / "oos_predictions.parquet", index=False)

    # Conviction bucket analysis
    logger.info(f"\n  {asset} conviction buckets:")
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30, 0.40]:
        l_mask = conv >= thresh
        s_mask = conv <= -thresh
        if l_mask.sum() > 10:
            logger.info(f"    LONG >={thresh:.2f}: acc={(acts[l_mask]==2).mean():.3f} n={l_mask.sum()}")
        if s_mask.sum() > 10:
            logger.info(f"    SHORT<={-thresh:.2f}: acc={(acts[s_mask]==0).mean():.3f} n={s_mask.sum()}")

    # Immediate backtest
    logger.info(f"\n  {asset} backtest:")
    prices = pd.read_sql(
        text("SELECT timestamp, high, low, close FROM asset_prices_15m WHERE asset = :a ORDER BY timestamp"),
        engine, params={"a": asset},
    )
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    oos_df["timestamp"] = pd.to_datetime(oos_df["timestamp"], utc=True)
    merged = prices.merge(oos_df[["timestamp", "conviction"]], on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    bt_results = {}
    for thresh in [0.05, 0.10, 0.15, 0.20, 0.30]:
        r = backtest_coin(merged, conviction_threshold=thresh)
        m = r["metrics"]
        weeks = len(merged) / (96 * 7)  # 15min bars in merged
        tpw = m["total_trades"] / weeks if weeks > 0 else 0
        bt_results[f"{thresh:.2f}"] = m
        logger.info(f"    @{thresh:.2f}: {m['total_trades']:4d} trades ({tpw:.1f}/wk) "
                    f"ret={m['total_return_pct']:+.1%} sharpe={m['sharpe_ratio']:.2f} "
                    f"wr={m['win_rate']:.1%} dd={m['max_drawdown_pct']:.1%}")

    # Save results
    results = {
        "asset": asset, "n_folds": len(splits),
        "mean_long_prec": float(np.mean([f["long_prec"] for f in fold_metrics])),
        "mean_long_rec": float(np.mean([f["long_rec"] for f in fold_metrics])),
        "mean_short_prec": float(np.mean([f["short_prec"] for f in fold_metrics])),
        "mean_short_rec": float(np.mean([f["short_rec"] for f in fold_metrics])),
        "conviction_std": float(conv.std()),
        "backtest": bt_results,
    }
    with open(asset_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    engine = create_engine(DB_URL)
    start = time.time()

    logger.info("V4.2 Training + Backtest (1h bars, focused coins)")
    logger.info(f"Assets: {V4_2_ASSETS}")
    logger.info(f"Walk-forward: train={TRAIN_SIZE} val={VAL_SIZE} test={TEST_SIZE} step={STEP_SIZE}")

    all_results = []
    for asset in V4_2_ASSETS:
        logger.info(f"\n{'='*60}\nTraining {asset}...")
        r = train_and_backtest_asset(asset, engine)
        all_results.append(r)

    with open(MODEL_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info(f"\n{'='*60}\nDONE in {elapsed:.0f}s!")
    engine.dispose()


if __name__ == "__main__":
    main()
