"""Parallel ML Training with Crash-Resilient Checkpoints.

Runs XGBoost, LightGBM, and LSTM in parallel processes.
Saves checkpoints after every retrain — resume from last checkpoint if interrupted.
Final results saved to models/ and results/ for tournament use.
"""

import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ──
PROJECT = Path("C:/Users/togat/Desktop/AI-Finance")
DATA_PATH = PROJECT / "data/features/tournament_data.parquet"
MODEL_DIR = PROJECT / "models"
CHECKPOINT_DIR = PROJECT / "models/checkpoints"
RESULTS_DIR = PROJECT / "models/results"

# ── Training params ──
RETRAIN_EVERY = 540   # ~90 days in 4h candles
MIN_TRAIN = 2000
EMBARGO = 42          # 7-day embargo gap
CONFIDENCE = 0.55
SEQ_LEN = 30
EPOCHS = 20
BATCH_SIZE = 256
LR = 1e-3


def load_data():
    """Load and prepare feature data."""
    df = pd.read_parquet(DATA_PATH)
    feature_cols = [c for c in df.columns if not c.startswith("_")]

    valid = df["_target"].notna()
    X = df.loc[valid, feature_cols].copy()
    y = df.loc[valid, "_target"].astype(int).copy()

    if "_timestamp" in df.columns:
        ts = df.loc[valid, "_timestamp"]
        sort_idx = ts.argsort()
        X = X.iloc[sort_idx].reset_index(drop=True)
        y = y.iloc[sort_idx].reset_index(drop=True)

    return X, y, feature_cols


def load_checkpoint(model_name):
    """Load checkpoint if exists. Returns (cursor, results_so_far) or None."""
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_checkpoint.json"
    if ckpt_path.exists():
        with open(ckpt_path) as f:
            ckpt = json.load(f)
        logger.info(f"Resuming {model_name} from retrain {ckpt['retrains']}, cursor={ckpt['cursor']}")
        return ckpt
    return None


def save_checkpoint(model_name, cursor, retrains, preds, probs, actuals, model_obj=None):
    """Save checkpoint after each retrain."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "model_name": model_name,
        "cursor": cursor,
        "retrains": retrains,
        "preds": preds,
        "probs": probs,
        "actuals": actuals,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    ckpt_path = CHECKPOINT_DIR / f"{model_name}_checkpoint.json"
    with open(ckpt_path, "w") as f:
        json.dump(ckpt, f)

    # Save model object too
    if model_obj is not None:
        model_path = CHECKPOINT_DIR / f"{model_name}_latest.joblib"
        joblib.dump(model_obj, model_path)


def save_final_results(model_name, result):
    """Save final model and metrics."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save metrics
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    metrics = {
        "model": model_name,
        "retrains": result["retrains"],
        "total_predictions": len(result["actuals"]),
        "signals": int(np.sum(result["preds"])),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    actuals = np.array(result["actuals"])
    preds = np.array(result["preds"])
    probs = np.array(result["probs"])

    if len(actuals) > 0:
        try:
            metrics["auc"] = float(roc_auc_score(actuals, probs))
        except ValueError:
            metrics["auc"] = 0.0
        metrics["accuracy"] = float(accuracy_score(actuals, preds))
        if preds.sum() > 0:
            metrics["precision"] = float(actuals[preds == 1].mean())
        else:
            metrics["precision"] = 0.0

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save predictions for ensemble
    pred_path = RESULTS_DIR / f"{model_name}_predictions.npz"
    np.savez_compressed(pred_path, preds=preds, probs=probs, actuals=actuals)

    logger.info(f"{model_name} final: AUC={metrics.get('auc', 0):.3f}, "
                f"Acc={metrics.get('accuracy', 0):.3f}, "
                f"Precision={metrics.get('precision', 0):.3f}, "
                f"Signals={metrics.get('signals', 0)}")


# ── Model Training Functions ──

def train_xgboost(X, y, feature_cols):
    """Train XGBoost with walk-forward and checkpoints."""
    from xgboost import XGBClassifier

    model_name = "xgboost"
    ckpt = load_checkpoint(model_name)

    if ckpt:
        cursor = ckpt["cursor"]
        retrains = ckpt["retrains"]
        all_preds = ckpt["preds"]
        all_probs = ckpt["probs"]
        all_actuals = ckpt["actuals"]
    else:
        cursor = MIN_TRAIN
        retrains = 0
        all_preds, all_probs, all_actuals = [], [], []

    best_model = None
    t_total = time.time()

    while cursor < len(X):
        train_end = cursor - EMBARGO
        test_end = min(cursor + RETRAIN_EVERY, len(X))

        if train_end < 1000:
            cursor += RETRAIN_EVERY
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[cursor:test_end]
        y_test = y.iloc[cursor:test_end]

        if len(X_test) == 0 or y_train.nunique() < 2:
            cursor += RETRAIN_EVERY
            continue

        t0 = time.time()
        model = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
            scale_pos_weight=2.0, random_state=42, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= CONFIDENCE).astype(int)

        all_preds.extend(preds.tolist())
        all_probs.extend(proba.tolist())
        all_actuals.extend(y_test.tolist())

        retrains += 1
        cursor = test_end

        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = 0.0
        logger.info(f"XGBoost retrain {retrains}: train={len(X_train):,}, "
                     f"test={len(X_test):,}, AUC={auc:.3f}, {time.time()-t0:.1f}s")

        best_model = model
        save_checkpoint(model_name, cursor, retrains, all_preds, all_probs, all_actuals, model)

    # Save final
    if best_model:
        joblib.dump(best_model, MODEL_DIR / "xgboost_latest.joblib")

    result = {"preds": all_preds, "probs": all_probs, "actuals": all_actuals, "retrains": retrains}
    save_final_results(model_name, result)
    logger.info(f"XGBoost total: {time.time()-t_total:.0f}s")
    return result


def train_lightgbm(X, y, feature_cols):
    """Train LightGBM with walk-forward and checkpoints."""
    from lightgbm import LGBMClassifier

    model_name = "lightgbm"
    ckpt = load_checkpoint(model_name)

    if ckpt:
        cursor = ckpt["cursor"]
        retrains = ckpt["retrains"]
        all_preds = ckpt["preds"]
        all_probs = ckpt["probs"]
        all_actuals = ckpt["actuals"]
    else:
        cursor = MIN_TRAIN
        retrains = 0
        all_preds, all_probs, all_actuals = [], [], []

    best_model = None
    t_total = time.time()

    while cursor < len(X):
        train_end = cursor - EMBARGO
        test_end = min(cursor + RETRAIN_EVERY, len(X))

        if train_end < 1000:
            cursor += RETRAIN_EVERY
            continue

        X_train = X.iloc[:train_end]
        y_train = y.iloc[:train_end]
        X_test = X.iloc[cursor:test_end]
        y_test = y.iloc[cursor:test_end]

        if len(X_test) == 0 or y_train.nunique() < 2:
            cursor += RETRAIN_EVERY
            continue

        t0 = time.time()
        model = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            num_leaves=15, subsample=0.7, colsample_bytree=0.6,
            min_child_samples=50, is_unbalance=True,
            random_state=42, n_jobs=-1, verbose=-1,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= CONFIDENCE).astype(int)

        all_preds.extend(preds.tolist())
        all_probs.extend(proba.tolist())
        all_actuals.extend(y_test.tolist())

        retrains += 1
        cursor = test_end

        try:
            auc = roc_auc_score(y_test, proba)
        except ValueError:
            auc = 0.0
        logger.info(f"LightGBM retrain {retrains}: train={len(X_train):,}, "
                     f"test={len(X_test):,}, AUC={auc:.3f}, {time.time()-t0:.1f}s")

        best_model = model
        save_checkpoint(model_name, cursor, retrains, all_preds, all_probs, all_actuals, model)

    # Save final
    if best_model:
        joblib.dump(best_model, MODEL_DIR / "lightgbm_latest.joblib")

    result = {"preds": all_preds, "probs": all_probs, "actuals": all_actuals, "retrains": retrains}
    save_final_results(model_name, result)
    logger.info(f"LightGBM total: {time.time()-t_total:.0f}s")
    return result


def train_lstm(X, y, feature_cols):
    """Train LSTM with walk-forward and checkpoints."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    class LSTMClassifier(nn.Module):
        def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                batch_first=True, dropout=dropout)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32), nn.ReLU(),
                nn.Dropout(dropout), nn.Linear(32, 1),
            )

        def forward(self, x):
            out, _ = self.lstm(x)
            return self.head(out[:, -1, :]).squeeze(-1)

    def create_sequences(X_arr, y_arr, seq_len):
        seqs, targets = [], []
        for i in range(seq_len, len(X_arr)):
            seqs.append(X_arr[i - seq_len:i])
            targets.append(y_arr[i])
        return np.array(seqs), np.array(targets)

    model_name = "lstm"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"LSTM device: {device}")

    ckpt = load_checkpoint(model_name)
    if ckpt:
        cursor = ckpt["cursor"]
        retrains = ckpt["retrains"]
        all_preds = ckpt["preds"]
        all_probs = ckpt["probs"]
        all_actuals = ckpt["actuals"]
    else:
        cursor = max(MIN_TRAIN, SEQ_LEN + 100)
        retrains = 0
        all_preds, all_probs, all_actuals = [], [], []

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    y_arr = y.values

    t_total = time.time()
    latest_model = None

    while cursor < len(X_scaled):
        train_end = cursor - EMBARGO
        test_end = min(cursor + RETRAIN_EVERY, len(X_scaled))

        X_seq_train, y_seq_train = create_sequences(X_scaled[:train_end], y_arr[:train_end], SEQ_LEN)
        X_seq_test, y_seq_test = create_sequences(
            X_scaled[cursor - SEQ_LEN:test_end],
            y_arr[cursor - SEQ_LEN:test_end], SEQ_LEN,
        )

        if len(X_seq_train) < 500 or len(X_seq_test) == 0:
            cursor += RETRAIN_EVERY
            continue

        X_tr = torch.FloatTensor(X_seq_train).to(device)
        y_tr = torch.FloatTensor(y_seq_train).to(device)
        X_te = torch.FloatTensor(X_seq_test).to(device)

        train_dl = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

        model = LSTMClassifier(input_size=len(feature_cols)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        pos_weight = torch.tensor(
            [(y_seq_train == 0).sum() / max((y_seq_train == 1).sum(), 1)]
        ).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        t0 = time.time()
        model.train()
        for epoch in range(EPOCHS):
            for xb, yb in train_dl:
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        model.eval()
        with torch.no_grad():
            proba = torch.sigmoid(model(X_te)).cpu().numpy()

        preds = (proba >= CONFIDENCE).astype(int)
        all_preds.extend(preds.tolist())
        all_probs.extend(proba.tolist())
        all_actuals.extend(y_seq_test.tolist())

        retrains += 1
        cursor = test_end

        try:
            auc = roc_auc_score(y_seq_test, proba)
        except ValueError:
            auc = 0.0
        logger.info(f"LSTM retrain {retrains}: train={len(X_seq_train):,}, "
                     f"test={len(X_seq_test):,}, AUC={auc:.3f}, {time.time()-t0:.1f}s")

        latest_model = model
        # Save LSTM checkpoint (model weights + predictions)
        torch.save(model.state_dict(), CHECKPOINT_DIR / "lstm_weights.pt")
        save_checkpoint(model_name, cursor, retrains, all_preds, all_probs, all_actuals)

    # Save final
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if latest_model:
        torch.save(latest_model.state_dict(), MODEL_DIR / "lstm_latest.pt")
    joblib.dump(scaler, MODEL_DIR / "scaler_latest.joblib")

    result = {"preds": all_preds, "probs": all_probs, "actuals": all_actuals, "retrains": retrains}
    save_final_results(model_name, result)
    logger.info(f"LSTM total: {time.time()-t_total:.0f}s")
    return result


def run_ensemble():
    """Combine saved predictions into ensemble result."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    weights = {"xgboost": 0.4, "lightgbm": 0.4, "lstm": 0.2}

    all_probs = {}
    all_actuals = {}
    for name in weights:
        pred_path = RESULTS_DIR / f"{name}_predictions.npz"
        if not pred_path.exists():
            logger.warning(f"Missing predictions for {name}, skipping ensemble")
            return
        data = np.load(pred_path)
        all_probs[name] = data["probs"]
        all_actuals[name] = data["actuals"]

    min_len = min(len(p) for p in all_probs.values())
    ensemble_probs = np.zeros(min_len)
    for name, w in weights.items():
        ensemble_probs += w * all_probs[name][:min_len]

    ensemble_preds = (ensemble_probs >= CONFIDENCE).astype(int)
    actuals = all_actuals["xgboost"][:min_len]

    try:
        auc = roc_auc_score(actuals, ensemble_probs)
    except ValueError:
        auc = 0.0
    acc = accuracy_score(actuals, ensemble_preds)
    signals = int(ensemble_preds.sum())
    precision = float(actuals[ensemble_preds == 1].mean()) if signals > 0 else 0.0

    metrics = {
        "model": "ensemble",
        "weights": weights,
        "auc": auc,
        "accuracy": acc,
        "precision": precision,
        "signals": signals,
        "total_predictions": min_len,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with open(RESULTS_DIR / "ensemble_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(
        RESULTS_DIR / "ensemble_predictions.npz",
        preds=ensemble_preds, probs=ensemble_probs, actuals=actuals,
    )

    logger.info(f"Ensemble: AUC={auc:.3f}, Acc={acc:.3f}, "
                f"Precision={precision:.3f}, Signals={signals}")
    return metrics


def print_summary():
    """Print final comparison table."""
    print("\n" + "=" * 75)
    print("TRAINING RESULTS")
    print("=" * 75)
    print(f"{'Model':<15} {'AUC':>8} {'Acc':>8} {'Precision':>11} {'Signals':>9} {'Retrains':>10}")
    print("-" * 65)

    for name in ["xgboost", "lightgbm", "lstm", "ensemble"]:
        metrics_path = RESULTS_DIR / f"{name}_metrics.json"
        if not metrics_path.exists():
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        print(f"{m['model']:<15} {m.get('auc', 0):>7.3f} {m.get('accuracy', 0):>7.3f} "
              f"{m.get('precision', 0):>10.3f} {m.get('signals', 0):>9} "
              f"{m.get('retrains', 'n/a'):>10}")

    print("=" * 75)
    print(f"\nModels saved to: {MODEL_DIR}")
    print(f"Results saved to: {RESULTS_DIR}")
    print(f"Checkpoints in: {CHECKPOINT_DIR}")


def main():
    logger.info("=" * 60)
    logger.info("ML TRAINING — Parallel with Checkpoints")
    logger.info("=" * 60)

    t_start = time.time()

    # Load data once (shared across processes via fork/spawn)
    logger.info("Loading data...")
    X, y, feature_cols = load_data()
    logger.info(f"Data: {X.shape[0]:,} samples, {len(feature_cols)} features")

    # Save feature list
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODEL_DIR / "feature_cols.txt", "w") as f:
        f.write("\n".join(feature_cols))

    # Run XGBoost and LightGBM in parallel, LSTM sequentially
    # (LSTM uses torch which doesn't always play nice with multiprocessing)
    logger.info("\nStarting parallel training (XGBoost + LightGBM)...")

    with ProcessPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(train_xgboost, X, y, feature_cols): "XGBoost",
            executor.submit(train_lightgbm, X, y, feature_cols): "LightGBM",
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                future.result()
                logger.info(f"{name} completed successfully")
            except Exception as e:
                logger.error(f"{name} failed: {e}")

    # LSTM runs after tree models (can use full CPU/GPU)
    logger.info("\nStarting LSTM training...")
    try:
        train_lstm(X, y, feature_cols)
        logger.info("LSTM completed successfully")
    except Exception as e:
        logger.error(f"LSTM failed: {e}")

    # Ensemble
    logger.info("\nBuilding ensemble...")
    run_ensemble()

    # Summary
    total_time = time.time() - t_start
    logger.info(f"\nTotal training time: {total_time:.0f}s ({total_time/60:.1f}m)")
    print_summary()


if __name__ == "__main__":
    main()
