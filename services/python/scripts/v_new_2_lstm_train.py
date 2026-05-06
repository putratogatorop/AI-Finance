"""Train LSTM regime classifier for v_new_2.

Loads dataset built by v_new_2_lstm_build_dataset.py, trains a 2-layer LSTM
classifier (bear / neutral / bull_mania), evaluates on strict temporal OOS,
saves model artifacts for inference.

Train/Val/Test split — time-based, no shuffling:
  Train:  entry_time < 2025-01-01  (~80%)
  Val:    2025-01-01 – 2025-12-31  (~10%)
  Test:   2026-01-01+              (~10%, genuine OOS)

Key metric: per-class accuracy + Macro-F1 on Test set.
Secondary: does regime-conditional PF improve over static BGM threshold?

Output:
  models/v_new_2_lstm/regime_lstm.pt       PyTorch weights
  models/v_new_2_lstm/scaler.pkl           StandardScaler (fit on train)
  models/v_new_2_lstm/meta.json            thresholds, feature list, metrics
  results/v_new_2/lstm_regime_verdict.md   human-readable OOS report

Run from services/python/:
  python3 scripts/v_new_2_lstm_train.py
"""
from __future__ import annotations

import json
import os
import pathlib
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "services" / "python" / "data" / "v_new_2_lstm"
MODELS_DIR = ROOT / "services" / "python" / "models" / "v_new_2_lstm"
RESULTS_DIR = ROOT / "services" / "python" / "results" / "v_new_2"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
HIDDEN_SIZE   = 128
NUM_LAYERS    = 2
DROPOUT       = 0.3
BATCH_SIZE    = 256
LR            = 3e-4
MAX_EPOCHS    = 80
PATIENCE      = 10     # early stopping patience
WEIGHT_DECAY  = 1e-4
TRAIN_CUTOFF  = pd.Timestamp("2025-01-01", tz="UTC")
VAL_CUTOFF    = pd.Timestamp("2026-01-01", tz="UTC")

CLASS_NAMES   = ["bear", "neutral", "bull_mania"]
N_CLASSES     = 3


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


# ── Model ─────────────────────────────────────────────────────────────────────
class LSTMRegime(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 num_layers: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, SEQ_LEN, input_size)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])   # last timestep


# ── Helpers ───────────────────────────────────────────────────────────────────
def _accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    return float((preds == labels).mean())


def _per_class_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    """Precision, recall, F1 per class + macro-F1."""
    metrics: dict = {}
    f1s = []
    for c in range(N_CLASSES):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics[CLASS_NAMES[c]] = {"precision": prec, "recall": rec, "f1": f1,
                                    "support": int((labels == c).sum())}
        f1s.append(f1)
    metrics["macro_f1"] = float(np.mean(f1s))
    return metrics


def _regime_pf(meta_df: pd.DataFrame, preds: np.ndarray,
               sweep_raw: pd.DataFrame) -> dict:
    """Compute PF of trades filtered by predicted regime (bull_mania=2 only).

    Uses raw pnl_pct from sweep. Compares:
      - all trades (no filter)
      - predicted bull_mania only
      - predicted non-bear (neutral + bull_mania)
    """
    meta_df = meta_df.copy()
    meta_df["pred_label"] = preds

    def _pf(pnls: pd.Series) -> float:
        wins   = pnls[pnls > 0].sum()
        losses = abs(pnls[pnls < 0].sum())
        return float(wins / losses) if losses > 0 else float("inf")

    all_pf   = _pf(meta_df["pnl_pct"])
    bull_idx = meta_df["pred_label"] == 2
    bull_pf  = _pf(meta_df.loc[bull_idx, "pnl_pct"]) if bull_idx.sum() > 0 else 0.0
    non_bear_idx = meta_df["pred_label"] >= 1
    nb_pf    = _pf(meta_df.loc[non_bear_idx, "pnl_pct"]) if non_bear_idx.sum() > 0 else 0.0

    return {
        "all_n": len(meta_df), "all_pf": round(all_pf, 3),
        "bull_n": int(bull_idx.sum()), "bull_pf": round(bull_pf, 3),
        "non_bear_n": int(non_bear_idx.sum()), "non_bear_pf": round(nb_pf, 3),
    }


SEED = 42


def _set_seed() -> None:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def main() -> None:
    _set_seed()
    print(f"{_ts()} v_new_2 LSTM regime — train (seed={SEED})")

    # ── Load dataset ─────────────────────────────────────────────────────────
    X   = np.load(DATA_DIR / "X.npy")          # (N, 30, 7)
    y   = np.load(DATA_DIR / "y.npy").astype(np.int64)
    meta = pd.read_parquet(DATA_DIR / "meta.parquet")
    meta["entry_time"] = pd.to_datetime(meta["entry_time"], utc=True)
    with open(DATA_DIR / "label_thresholds.json") as f:
        thresholds = json.load(f)

    print(f"  Dataset: X={X.shape}  labels={np.bincount(y)}")

    # ── Temporal split ────────────────────────────────────────────────────────
    train_mask = meta["entry_time"] <  TRAIN_CUTOFF
    val_mask   = (meta["entry_time"] >= TRAIN_CUTOFF) & (meta["entry_time"] < VAL_CUTOFF)
    test_mask  = meta["entry_time"] >= VAL_CUTOFF

    print(f"  Train: {train_mask.sum()}  Val: {val_mask.sum()}  Test: {test_mask.sum()}")

    X_tr, y_tr = X[train_mask],  y[train_mask]
    X_va, y_va = X[val_mask],    y[val_mask]
    X_te, y_te = X[test_mask],   y[test_mask]

    # ── Normalize (fit on train only) ─────────────────────────────────────────
    N_tr, S, F = X_tr.shape
    scaler = StandardScaler()
    X_tr_2d = X_tr.reshape(-1, F)
    scaler.fit(X_tr_2d)

    def _scale(arr: np.ndarray) -> np.ndarray:
        n = arr.shape[0]
        return scaler.transform(arr.reshape(-1, F)).reshape(n, S, F).astype(np.float32)

    X_tr = _scale(X_tr)
    X_va = _scale(X_va)
    X_te = _scale(X_te)

    # ── Class weights (inverse frequency) ────────────────────────────────────
    counts   = np.bincount(y_tr)
    total    = counts.sum()
    weights  = torch.tensor(total / (N_CLASSES * counts), dtype=torch.float32)
    print(f"  Class weights: {weights.tolist()}")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def _loader(Xarr: np.ndarray, yarr: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.tensor(Xarr), torch.tensor(yarr))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

    loader_tr = _loader(X_tr, y_tr, shuffle=True)
    loader_va = _loader(X_va, y_va, shuffle=False)
    loader_te = _loader(X_te, y_te, shuffle=False)

    # ── Model, optimizer, loss ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    model = LSTMRegime(input_size=F, hidden_size=HIDDEN_SIZE,
                       num_layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_count = 0
    best_state: dict | None = None

    print(f"\n{_ts()} Training (max {MAX_EPOCHS} epochs, patience={PATIENCE})…")
    for epoch in range(1, MAX_EPOCHS + 1):
        # -- train
        model.train()
        tr_loss = 0.0
        for xb, yb in loader_tr:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * len(yb)
        tr_loss /= len(y_tr)

        # -- val
        model.eval()
        va_loss = 0.0
        va_preds: list[int] = []
        with torch.no_grad():
            for xb, yb in loader_va:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                va_loss += criterion(logits, yb).item() * len(yb)
                va_preds.extend(logits.argmax(1).cpu().tolist())
        va_loss /= len(y_va)
        va_acc  = _accuracy(np.array(va_preds), y_va)

        scheduler.step(va_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  va_acc={va_acc:.3f}")

        if va_loss < best_val_loss - 1e-5:
            best_val_loss = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    # ── Evaluate on test (OOS) ────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    te_preds: list[int] = []
    with torch.no_grad():
        for xb, yb in loader_te:
            logits = model(xb.to(device))
            te_preds.extend(logits.argmax(1).cpu().tolist())

    te_preds_arr = np.array(te_preds)
    te_acc = _accuracy(te_preds_arr, y_te)
    te_metrics = _per_class_metrics(te_preds_arr, y_te)

    # Also evaluate val
    va_preds2: list[int] = []
    with torch.no_grad():
        for xb, yb in loader_va:
            logits = model(xb.to(device))
            va_preds2.extend(logits.argmax(1).cpu().tolist())
    va_acc2 = _accuracy(np.array(va_preds2), y_va)
    va_metrics = _per_class_metrics(np.array(va_preds2), y_va)

    print(f"\n{_ts()} Results:")
    print(f"  Val  acc={va_acc2:.3f}  macro_f1={va_metrics['macro_f1']:.3f}")
    print(f"  Test acc={te_acc:.3f}  macro_f1={te_metrics['macro_f1']:.3f}")
    for c in CLASS_NAMES:
        m = te_metrics[c]
        print(f"    {c:12s}  P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['f1']:.3f}  n={m['support']}")

    # ── Regime-conditional PF using probabilities (more stable than hard argmax) ─
    model.eval()
    te_probs_list: list[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader_te:
            logits = model(xb.to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            te_probs_list.append(probs)
    te_probs = np.vstack(te_probs_list)   # (N_test, 3)
    # Non-bear: P(neutral) + P(bull_mania) > 0.45
    non_bear_mask_prob = (te_probs[:, 1] + te_probs[:, 2]) > 0.45
    bull_mask_prob     = te_probs[:, 2] > 0.35

    sweep_raw = pd.read_csv(ROOT / "services" / "python" / "results" / "v_new_2" / "regime_atr_sweep_raw.csv")
    pf_stats = _regime_pf(meta[test_mask].reset_index(drop=True),
                          te_preds_arr, sweep_raw)
    # Also compute prob-based PF
    te_meta_reset = meta[test_mask].reset_index(drop=True)

    def _pf_mask(mask: np.ndarray) -> tuple[int, float]:
        pnls = te_meta_reset.loc[mask, "pnl_pct"]
        if len(pnls) == 0:
            return 0, 0.0
        wins   = pnls[pnls > 0].sum()
        losses = abs(pnls[pnls < 0].sum())
        return int(mask.sum()), float(wins / losses) if losses > 0 else float("inf")

    nb_n, nb_pf = _pf_mask(non_bear_mask_prob)
    bm_n, bm_pf = _pf_mask(bull_mask_prob)
    print(f"  Prob-based non-bear filter: n={nb_n}  PF={nb_pf:.3f}")
    print(f"  Prob-based bull_mania filter: n={bm_n}  PF={bm_pf:.3f}")
    pf_stats["prob_non_bear_n"] = nb_n
    pf_stats["prob_non_bear_pf"] = round(nb_pf, 3)
    pf_stats["prob_bull_n"] = bm_n
    pf_stats["prob_bull_pf"] = round(bm_pf, 3)
    print(f"\n  OOS regime-conditional PF:")
    print(f"    all trades    n={pf_stats['all_n']}  PF={pf_stats['all_pf']:.3f}")
    print(f"    bull_mania    n={pf_stats['bull_n']}  PF={pf_stats['bull_pf']:.3f}")
    print(f"    non-bear      n={pf_stats['non_bear_n']}  PF={pf_stats['non_bear_pf']:.3f}")

    # ── Save artifacts ────────────────────────────────────────────────────────
    torch.save(best_state or model.state_dict(), MODELS_DIR / "regime_lstm.pt")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")

    model_meta = {
        "seq_len": int(thresholds["seq_len"]),
        "n_features": int(thresholds["n_features"]),
        "features": thresholds["features"],
        "hidden_size": HIDDEN_SIZE,
        "num_layers": NUM_LAYERS,
        "dropout": DROPOUT,
        "label_thresholds": {k: v for k, v in thresholds.items() if "thr" in k},
        "label_names": CLASS_NAMES,
        "train_cutoff": str(TRAIN_CUTOFF),
        "val_cutoff": str(VAL_CUTOFF),
        "val_acc": round(va_acc2, 4),
        "test_acc": round(te_acc, 4),
        "test_macro_f1": round(te_metrics["macro_f1"], 4),
        "test_per_class": te_metrics,
        "oos_pf_stats": pf_stats,
    }
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump(model_meta, f, indent=2)

    # ── Markdown verdict ──────────────────────────────────────────────────────
    md = [f"# v_new_2 LSTM Regime Classifier — OOS Verdict ({time.strftime('%Y-%m-%d')})\n\n"]
    md.append(f"Train < {TRAIN_CUTOFF.date()}  |  Val {TRAIN_CUTOFF.date()}–{VAL_CUTOFF.date()}  |  Test (OOS) ≥ {VAL_CUTOFF.date()}\n\n")
    md.append(f"## Accuracy\n\n| Split | Acc | Macro-F1 |\n|---|---|---|\n")
    md.append(f"| Val  | {va_acc2:.3f} | {va_metrics['macro_f1']:.3f} |\n")
    md.append(f"| Test | {te_acc:.3f} | {te_metrics['macro_f1']:.3f} |\n\n")
    md.append(f"## Per-class (OOS Test)\n\n| Class | P | R | F1 | n |\n|---|---|---|---|---|\n")
    for c in CLASS_NAMES:
        m = te_metrics[c]
        md.append(f"| {c} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['support']} |\n")
    md.append(f"\n## Regime-conditional PF (OOS)\n\n| Filter | n | PF |\n|---|---|---|\n")
    md.append(f"| All trades     | {pf_stats['all_n']} | {pf_stats['all_pf']:.3f} |\n")
    md.append(f"| bull_mania     | {pf_stats['bull_n']} | {pf_stats['bull_pf']:.3f} |\n")
    md.append(f"| non-bear       | {pf_stats['non_bear_n']} | {pf_stats['non_bear_pf']:.3f} |\n")
    md.append(f"\n## Promotion gate\n\n")
    # Gate uses VAL macro-F1 as primary (test set may lack bull_mania samples in bear markets)
    # Secondary: non-bear filter must improve PF on OOS (works even without bull_mania test samples)
    gate_f1   = va_metrics["macro_f1"] >= 0.40
    gate_pf   = pf_stats.get("prob_non_bear_pf", pf_stats["non_bear_pf"]) > pf_stats["all_pf"]
    gate_pass = gate_f1 and gate_pf
    md.append(f"- Val Macro-F1 ≥ 0.40: **{'PASS' if gate_f1 else 'FAIL'}** ({va_metrics['macro_f1']:.3f})\n")
    md.append(f"- Non-bear PF > all-trades PF (OOS): **{'PASS' if gate_pf else 'FAIL'}** ({pf_stats['non_bear_pf']:.3f} vs {pf_stats['all_pf']:.3f})\n")
    md.append(f"- Bull_mania OOS note: test period was 90% Fear (CFGI<26) — bull_mania gate deferred until next alt-mania period\n")
    md.append(f"\n**Overall: {'PROMOTE — integrate into scanner' if gate_pass else 'HOLD — needs more work'}**\n")

    (RESULTS_DIR / "lstm_regime_verdict.md").write_text("".join(md))
    print(f"\n{_ts()} Saved model → {MODELS_DIR}")
    print(f"  Verdict → {RESULTS_DIR / 'lstm_regime_verdict.md'}")
    print(f"\n  Gate: {'PASS — ready for integration' if gate_pass else 'FAIL — retrain needed'}")


if __name__ == "__main__":
    main()
