# V4 Week 5-6: LightGBM Walk-Forward Training Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train per-coin LightGBM classifiers using walk-forward validation across 3 years of 15-minute features, producing out-of-sample predictions and performance metrics for all 7 assets.

**Architecture:** Walk-forward splitter generates chronological train/val/test windows that slide every 2 weeks. For each window, a LightGBM multiclass classifier (SHORT/FLAT/LONG) trains on 6 months, validates on 1 month (early stopping), and predicts on 1 month of held-out data. Results are aggregated across ~24 OOS periods per coin.

**Tech Stack:** Python 3.12, LightGBM, pandas, numpy, scikit-learn (metrics only), joblib

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ml/walk_forward.py` | Walk-forward time-series splitter |
| `tests/test_walk_forward.py` | Unit tests for splitter logic |
| `scripts/train_v4.py` | End-to-end training: load Parquet, walk-forward train, evaluate, save |
| `tests/test_train_v4.py` | Integration test for training pipeline |

## Walk-Forward Parameters (from spec)

```
Data: 104,528 rows per coin (April 2023 → March 2026)
Train window:  6 months = 26,208 rows (96 candles/day × 273 days)
Val window:    1 month  =  2,880 rows (96 × 30)
Test window:   1 month  =  2,880 rows (96 × 30)
Slide step:    2 weeks  =  1,344 rows (96 × 14)
Embargo gap:   16 rows  (= target horizon, prevents leakage)
```

With 104,528 rows, ~24 test windows fit after the first train+val+test block.

---

### Task 1: Walk-Forward Splitter

**Files:**
- Create: `services/python/src/ml/walk_forward.py`
- Create: `services/python/tests/test_walk_forward.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_walk_forward.py
import sys
sys.path.insert(0, ".")

import pytest


def test_walk_forward_basic():
    from src.ml.walk_forward import walk_forward_splits

    splits = list(walk_forward_splits(
        n_samples=50000,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    assert len(splits) > 10

    # First split
    train, val, test = splits[0]
    assert train == (0, 26208)
    assert val == (26224, 29104)   # 26208 + 16 embargo = 26224
    assert test == (29120, 32000)  # 29104 + 16 embargo = 29120

    # No overlap
    assert train[1] + 16 <= val[0]
    assert val[1] + 16 <= test[0]


def test_walk_forward_sliding():
    from src.ml.walk_forward import walk_forward_splits

    splits = list(walk_forward_splits(
        n_samples=50000,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    # Each split slides by step_size
    _, _, test0 = splits[0]
    _, _, test1 = splits[1]
    assert test1[0] - test0[0] == 1344


def test_walk_forward_no_future_leak():
    from src.ml.walk_forward import walk_forward_splits

    splits = list(walk_forward_splits(
        n_samples=50000,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    for train, val, test in splits:
        # Train ends before val starts (with embargo)
        assert train[1] + 16 <= val[0]
        # Val ends before test starts (with embargo)
        assert val[1] + 16 <= test[0]
        # Test doesn't exceed data
        assert test[1] <= 50000


def test_walk_forward_count():
    from src.ml.walk_forward import walk_forward_splits

    # With 104,528 samples (real data size)
    splits = list(walk_forward_splits(
        n_samples=104528,
        train_size=26208,
        val_size=2880,
        test_size=2880,
        step_size=1344,
        embargo=16,
    ))

    # Should get ~24 splits
    assert 20 <= len(splits) <= 60
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_walk_forward.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement walk_forward.py**

```python
# src/ml/walk_forward.py
"""Walk-forward time-series cross-validation splitter.

Generates chronological train/val/test windows with embargo gaps
to prevent target leakage. Windows slide forward by step_size.
"""

from __future__ import annotations


def walk_forward_splits(
    n_samples: int,
    train_size: int = 26208,
    val_size: int = 2880,
    test_size: int = 2880,
    step_size: int = 1344,
    embargo: int = 16,
) -> list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]:
    """Generate walk-forward train/val/test index ranges.

    Parameters
    ----------
    n_samples : total rows in dataset
    train_size : training window (6 months of 15min = 26,208)
    val_size : validation window (1 month = 2,880)
    test_size : test window (1 month = 2,880)
    step_size : slide step (2 weeks = 1,344)
    embargo : gap between windows to prevent target leakage (= target horizon)

    Returns
    -------
    List of (train, val, test) tuples, each a (start, end) index pair.
    Indices are [start, end) — use df.iloc[start:end].
    """
    splits = []
    offset = 0

    while True:
        train_start = offset
        train_end = train_start + train_size

        val_start = train_end + embargo
        val_end = val_start + val_size

        test_start = val_end + embargo
        test_end = test_start + test_size

        if test_end > n_samples:
            break

        splits.append((
            (train_start, train_end),
            (val_start, val_end),
            (test_start, test_end),
        ))

        offset += step_size

    return splits
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_walk_forward.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/src/ml/walk_forward.py services/python/tests/test_walk_forward.py
git commit -m "feat: add walk-forward time-series splitter"
```

---

### Task 2: Training Script

**Files:**
- Create: `services/python/scripts/train_v4.py`
- Create: `services/python/tests/test_train_v4.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_train_v4.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest


def make_fake_features(n: int = 5000) -> pd.DataFrame:
    """Generate fake feature data matching v4 schema."""
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    data = {"timestamp": ts}

    feature_cols = [
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
    for col in feature_cols:
        data[col] = rng.normal(0, 1, n)

    # 3-class target: 20/60/20
    target = rng.choice([0, 1, 2], size=n, p=[0.2, 0.6, 0.2])
    data["target"] = target.astype(float)

    return pd.DataFrame(data)


def test_train_one_fold():
    from scripts.train_v4 import train_one_fold, FEATURE_COLS, LGB_PARAMS

    df = make_fake_features(5000)
    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)

    result = train_one_fold(
        X_train=X[:3000], y_train=y[:3000],
        X_val=X[3000:4000], y_val=y[3000:4000],
        X_test=X[4000:5000], y_test=y[4000:5000],
        params=LGB_PARAMS,
    )

    assert "accuracy" in result
    assert "log_loss" in result
    assert "conviction" in result
    assert len(result["conviction"]) == 1000
    # Random data: accuracy should be around 33-60%
    assert 0.2 < result["accuracy"] < 0.8


def test_conviction_range():
    from scripts.train_v4 import train_one_fold, FEATURE_COLS, LGB_PARAMS

    df = make_fake_features(5000)
    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)

    result = train_one_fold(
        X_train=X[:3000], y_train=y[:3000],
        X_val=X[3000:4000], y_val=y[3000:4000],
        X_test=X[4000:5000], y_test=y[4000:5000],
        params=LGB_PARAMS,
    )

    # Conviction = p_long - p_short, should be in [-1, 1]
    assert all(-1.01 <= c <= 1.01 for c in result["conviction"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_train_v4.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement train_v4.py**

```python
# scripts/train_v4.py
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

# Walk-forward params (15min candles)
TRAIN_SIZE = 26208   # 6 months
VAL_SIZE = 2880      # 1 month
TEST_SIZE = 2880     # 1 month
STEP_SIZE = 1344     # 2 weeks
EMBARGO = 16         # target horizon


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

    # Predict on test set
    proba = model.predict_proba(X_test)  # shape (n, 3)
    pred = model.predict(X_test)

    # Conviction: p_long - p_short
    conviction = proba[:, 2] - proba[:, 0]

    # Metrics
    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, proba, labels=[0, 1, 2])

    # Per-class accuracy
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

        # Save model
        model = result.pop("model")
        model_path = asset_dir / f"fold_{i:02d}.txt"
        model.booster_.save_model(str(model_path))

        # Collect OOS predictions
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

    # Aggregate metrics
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

    # Per-class OOS accuracy
    for c in [0, 1, 2]:
        mask = all_acts == c
        if mask.sum() > 0:
            agg[f"oos_class_{c}_accuracy"] = float(accuracy_score(all_acts[mask], all_preds[mask]))

    # Save results
    results = {"aggregate": agg, "folds": fold_results}
    results_path = asset_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Save OOS predictions for backtesting
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

    # Save summary
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd services/python && python -m pytest tests/test_train_v4.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/train_v4.py services/python/tests/test_train_v4.py
git commit -m "feat: add v4 LightGBM walk-forward training script"
```

- [ ] **Step 6: Run the training**

Run: `cd services/python && python scripts/train_v4.py`
Expected: Trains 7 assets × ~24 folds each. Each fold: ~26K train rows, 2K early-stopping rounds max. Total time ~10-30 minutes on CPU. Outputs model files + results JSON + OOS prediction Parquets.

- [ ] **Step 7: Review results**

Run:
```bash
cd services/python && python -c "
import json
from pathlib import Path
summary = json.loads(Path('C:/Users/togat/Desktop/AI-Finance/models/v4/summary.json').read_text())
for r in summary:
    print(f\"{r['asset']}: acc={r['oos_accuracy']:.3f} folds={r['n_folds']} high_conv={r['high_conviction_pct']:.1%}\")
"
```

**Success criteria from spec (Section 13):**
- OOS accuracy > 48% (random = 33%)
- High conviction trades (|conviction| > 0.35) should have higher accuracy than overall

**Red flags:**
- Accuracy > 65% likely means bug or leakage
- All folds identical accuracy = something wrong

- [ ] **Step 8: Commit results**

```bash
git add models/v4/summary.json
git commit -m "data: v4 walk-forward training results (7 assets, ~24 folds each)"
```
