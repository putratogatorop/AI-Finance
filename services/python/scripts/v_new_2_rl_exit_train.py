"""Phase 2 — Exit Classifier for v_new_2 (adaptive exit policy).

Trains a binary classifier to predict "exit now vs hold" at each 4h bar.
Uses the same state vector as the RL formulation, but supervised training
with class-weighted cross-entropy handles the 36:1 hold/exit imbalance.

This is more reliable than PPO given ~21k episodes and high action imbalance.
The output (exit probability) is used in the executor with a threshold.

State vector (10 features):
  pnl_pct_norm    — current trade PnL / 50
  bars_held_norm  — bars held / 90 (90-bar horizon)
  atr_pct_norm    — current ATR / 5
  p_bear          — LSTM P(bear regime)
  p_neutral       — LSTM P(neutral regime)
  p_bull          — LSTM P(bull_mania regime)
  breadth_up      — % coins with positive 24h return (from sweep data)
  btc_ret_norm    — BTC 4h return / 0.03
  is_meme         — 1/0
  side_sign       — +1 long / -1 short

Oracle labels: static ATR trail (meme 3×, non-meme 5×→10×) — exit=1, hold=0

OOS gate: precision on exit class > 0.35 AND recall > 0.25
          (we care more about precision: don't exit too early)

Output:
  models/v_new_2_rl/exit_classifier.pt
  models/v_new_2_rl/exit_classifier_meta.json
  results/v_new_2/rl_exit_verdict.md

Run from services/python/:
  python3 scripts/v_new_2_rl_exit_train.py
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn
from sklearn.metrics import classification_report

ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "services" / "python"))

DATA_DIR    = ROOT / "services" / "python" / "data" / "v_new_1_v2"
LSTM_DIR    = ROOT / "services" / "python" / "models" / "v_new_2_lstm"
MODELS_DIR  = ROOT / "services" / "python" / "models" / "v_new_2_rl"
RESULTS_DIR = ROOT / "services" / "python" / "results" / "v_new_2"
SWEEP_CSV   = ROOT / "services" / "python" / "results" / "v_new_2" / "regime_atr_sweep_raw.csv"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED        = 42
STATE_DIM   = 10
HIDDEN      = 64
LR          = 3e-4
EPOCHS      = 30
BATCH_SIZE  = 512
PATIENCE    = 6
TRAIN_CUTOFF = pd.Timestamp("2025-01-01", tz="UTC")
VAL_CUTOFF   = pd.Timestamp("2026-01-01", tz="UTC")

# ATR trail oracle (mirrors executor constants)
MEME_TRAIL      = 3.0
NONMEME_INIT    = 5.0
NONMEME_PL      = 10.0
PL_MEME         = 0.20
PL_NONMEME      = 0.30


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _set_seed() -> None:
    import random
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


# ── Model ─────────────────────────────────────────────────────────────────────
class ExitClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, HIDDEN),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN, HIDDEN // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(HIDDEN // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # logit for "exit"

    def prob_exit(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ── Episode builder ───────────────────────────────────────────────────────────
@dataclass
class Step:
    state:       np.ndarray
    oracle_exit: int    # 1=exit, 0=hold
    pnl_pct:     float  # for evaluation


def _build_steps(trades: pd.DataFrame,
                 feats_sym: dict,
                 lstm_probs: np.ndarray,
                 trade_idx_map: dict) -> list[Step]:
    steps: list[Step] = []

    for _, row in trades.iterrows():
        sym      = row["symbol"]
        side     = row["side"]
        is_meme  = bool(row["is_meme"])
        g        = feats_sym.get(sym)
        if g is None:
            continue

        entry_time  = pd.Timestamp(row["entry_time"])
        ts_arr      = g["timestamp"].values
        pos         = int(np.searchsorted(ts_arr, np.datetime64(entry_time), side="left"))
        bars_held   = int(row["bars_held"])
        if pos >= len(g) or bars_held == 0:
            continue

        entry_price = float(g["close"].iloc[pos]) if pos < len(g) else 1.0
        if entry_price <= 0:
            continue

        # LSTM regime probs at entry
        key = (sym, side, entry_time)
        tidx = trade_idx_map.get(key)
        regime_probs = lstm_probs[tidx] if tidx is not None and tidx < len(lstm_probs) \
            else np.array([0.33, 0.34, 0.33], dtype=np.float32)

        breadth_up = float(row.get("breadth_up", 0.5))
        btc_ret    = float(row.get("btc_ret", 0.0))

        running_extreme = entry_price
        pl_active       = False
        trail_mult      = MEME_TRAIL if is_meme else NONMEME_INIT
        exit_bar        = min(pos + bars_held, len(g) - 1)

        for k in range(pos, exit_bar + 1):
            if k >= len(g):
                break
            bar        = g.iloc[k]
            curr_price = float(bar["close"])
            atr_pct    = float(bar["atr14_pct"]) if pd.notna(bar["atr14_pct"]) else 0.02

            if side == "long":
                running_extreme = max(running_extreme, curr_price)
                pnl_pct = (curr_price / entry_price - 1.0) * 100.0
            else:
                running_extreme = min(running_extreme, curr_price)
                pnl_pct = (entry_price / curr_price - 1.0) * 100.0

            # Profit-lock update
            thr = PL_MEME if is_meme else PL_NONMEME
            if not pl_active and pnl_pct / 100.0 >= thr:
                pl_active = True
                if not is_meme:
                    trail_mult = NONMEME_PL

            # Oracle exit: ATR trail trigger OR last bar
            atr_abs = (atr_pct / 100.0) * entry_price
            trail   = trail_mult * atr_abs
            if side == "long":
                oracle_exit = int(curr_price <= running_extreme - trail or k == exit_bar)
            else:
                oracle_exit = int(curr_price >= running_extreme + trail or k == exit_bar)

            bars_norm = min((k - pos) / 90.0, 1.0)
            state = np.array([
                float(np.clip(pnl_pct / 50.0, -2.0, 2.0)),
                float(bars_norm),
                float(np.clip(atr_pct / 5.0, 0.0, 5.0)),
                float(regime_probs[0]),
                float(regime_probs[1]),
                float(regime_probs[2]),
                float(breadth_up),
                float(np.clip(btc_ret / 0.03, -3.0, 3.0)),
                1.0 if is_meme else 0.0,
                1.0 if side == "long" else -1.0,
            ], dtype=np.float32)

            steps.append(Step(state=state, oracle_exit=oracle_exit, pnl_pct=pnl_pct))
            if oracle_exit:
                break

    return steps


def main() -> None:
    _set_seed()
    print(f"{_ts()} v_new_2 Exit Classifier — train")

    # ── Load data ─────────────────────────────────────────────────────────────
    sweep = pd.read_csv(SWEEP_CSV)
    sweep["entry_time"] = pd.to_datetime(sweep["entry_time"], utc=True)
    trades = sweep[sweep["atr_mult"] == 10.0].drop_duplicates(
        subset=["symbol", "side", "entry_time"]).reset_index(drop=True)
    print(f"  {len(trades):,} trades")

    feats = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet",
                            columns=["symbol", "timestamp", "close", "atr14_pct"])
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    feats = feats.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    feats_sym = {sym: g.reset_index(drop=True) for sym, g in feats.groupby("symbol")}

    # LSTM probs
    lstm_meta = pd.read_parquet(ROOT / "services" / "python" / "data" / "v_new_2_lstm" / "meta.parquet")
    lstm_meta["entry_time"] = pd.to_datetime(lstm_meta["entry_time"], utc=True)
    lstm_X  = np.load(ROOT / "services" / "python" / "data" / "v_new_2_lstm" / "X.npy")
    scaler  = joblib.load(LSTM_DIR / "scaler.pkl")

    with open(LSTM_DIR / "meta.json") as f:
        lstm_cfg = json.load(f)

    from scripts.v_new_2_lstm_train import LSTMRegime
    lstm_model = LSTMRegime(input_size=lstm_cfg["n_features"],
                            hidden_size=lstm_cfg["hidden_size"],
                            num_layers=lstm_cfg["num_layers"],
                            dropout=0.0)
    lstm_model.load_state_dict(torch.load(LSTM_DIR / "regime_lstm.pt", map_location="cpu"))
    lstm_model.eval()

    _Nv, _Sv, _Fv = lstm_X.shape
    X_sc = scaler.transform(lstm_X.reshape(-1, _Fv)).reshape(_Nv, _Sv, _Fv).astype(np.float32)
    with torch.no_grad():
        lstm_probs = torch.softmax(lstm_model(torch.tensor(X_sc)), dim=1).numpy()

    trade_idx_map = {(r["symbol"], r["side"], r["entry_time"]): i
                     for i, r in lstm_meta.iterrows()}

    # ── Build steps ───────────────────────────────────────────────────────────
    print(f"{_ts()} Building steps…")
    train_t = trades[trades["entry_time"] < TRAIN_CUTOFF].reset_index(drop=True)
    val_t   = trades[(trades["entry_time"] >= TRAIN_CUTOFF) & (trades["entry_time"] < VAL_CUTOFF)].reset_index(drop=True)
    test_t  = trades[trades["entry_time"] >= VAL_CUTOFF].reset_index(drop=True)

    tr_steps = _build_steps(train_t, feats_sym, lstm_probs, trade_idx_map)
    va_steps = _build_steps(val_t,   feats_sym, lstm_probs, trade_idx_map)
    te_steps = _build_steps(test_t,  feats_sym, lstm_probs, trade_idx_map)
    print(f"  Train: {len(tr_steps):,}  Val: {len(va_steps):,}  Test: {len(te_steps):,}")

    def _to_tensors(steps: list[Step]):
        X = torch.tensor(np.array([s.state for s in steps], dtype=np.float32))
        y = torch.tensor(np.array([s.oracle_exit for s in steps], dtype=np.float32))
        return X, y

    Xtr, ytr = _to_tensors(tr_steps)
    Xva, yva = _to_tensors(va_steps)
    Xte, yte = _to_tensors(te_steps)

    # Class weights: exit is rare (~2.7%)
    # Use moderate weight (10×) — heavy enough to learn exits, not so heavy it floods FP
    n_exit = int(ytr.sum())
    n_hold = len(ytr) - n_exit
    w_exit = min(10.0, n_hold / n_exit) if n_exit > 0 else 1.0
    print(f"  Class dist: hold={n_hold}  exit={n_exit}  weight_exit={w_exit:.1f}")

    # ── Train ─────────────────────────────────────────────────────────────────
    model   = ExitClassifier()
    optim   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pos_wt  = torch.tensor(w_exit)

    best_val = float("inf")
    patience  = 0
    best_state: dict | None = None

    print(f"\n{_ts()} Training ({EPOCHS} epochs)…")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        idx = torch.randperm(len(Xtr))
        tr_loss = 0.0
        for start in range(0, len(idx), BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            logits = model(Xtr[b])
            loss   = Fnn.binary_cross_entropy_with_logits(logits, ytr[b], pos_weight=pos_wt)
            optim.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss += loss.item()
        tr_loss /= max(1, len(idx) // BATCH_SIZE)

        model.eval()
        with torch.no_grad():
            va_loss = Fnn.binary_cross_entropy_with_logits(
                model(Xva), yva, pos_weight=pos_wt).item()
            va_probs = torch.sigmoid(model(Xva)).numpy()
            va_preds = (va_probs > 0.5).astype(int)
        va_prec = float(((va_preds == 1) & (yva.numpy() == 1)).sum() /
                         max(1, (va_preds == 1).sum()))
        va_rec  = float(((va_preds == 1) & (yva.numpy() == 1)).sum() /
                         max(1, (yva.numpy() == 1).sum()))

        if epoch % 5 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  tr={tr_loss:.4f}  va={va_loss:.4f}"
                  f"  exit_prec={va_prec:.3f}  exit_rec={va_rec:.3f}")

        if va_loss < best_val - 1e-5:
            best_val = va_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # ── OOS evaluation ────────────────────────────────────────────────────────
    model.eval()
    yte_np = yte.numpy().astype(int)

    # ── Threshold calibration on val set ─────────────────────────────────────
    with torch.no_grad():
        va_probs = torch.sigmoid(model(Xva)).numpy()
    yva_np = yva.numpy().astype(int)

    best_thr, best_f1_va = 0.5, 0.0
    print(f"\n  Threshold calibration (val set):")
    for thr in np.arange(0.3, 0.95, 0.05):
        preds_va = (va_probs > thr).astype(int)
        tp_v = int(((preds_va == 1) & (yva_np == 1)).sum())
        fp_v = int(((preds_va == 1) & (yva_np == 0)).sum())
        fn_v = int(((preds_va == 0) & (yva_np == 1)).sum())
        p_v = tp_v / max(1, tp_v + fp_v)
        r_v = tp_v / max(1, tp_v + fn_v)
        f_v = 2 * p_v * r_v / max(1e-9, p_v + r_v)
        if p_v >= 0.35 and r_v >= 0.25 and f_v > best_f1_va:
            best_f1_va = f_v
            best_thr = float(thr)
        print(f"    thr={thr:.2f}  P={p_v:.3f}  R={r_v:.3f}  F1={f_v:.3f}")

    print(f"\n  Best threshold (prec≥0.35, rec≥0.25): {best_thr:.2f}")

    with torch.no_grad():
        te_probs = torch.sigmoid(model(Xte)).numpy()
    te_preds = (te_probs > best_thr).astype(int)

    tp = int(((te_preds == 1) & (yte_np == 1)).sum())
    fp = int(((te_preds == 1) & (yte_np == 0)).sum())
    fn = int(((te_preds == 0) & (yte_np == 1)).sum())
    prec = tp / max(1, tp + fp)
    rec  = tp / max(1, tp + fn)
    f1   = 2 * prec * rec / max(1e-9, prec + rec)

    print(f"\n{_ts()} OOS Test:")
    print(f"  Exit  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}")
    print(f"  TP={tp}  FP={fp}  FN={fn}")

    # PnL comparison: oracle exits vs classifier exits
    oracle_pnls = [s.pnl_pct for s in te_steps if s.oracle_exit == 1]
    clf_exit_steps = [s for s, p in zip(te_steps, te_preds) if p == 1]
    clf_pnls = [s.pnl_pct for s in clf_exit_steps]

    def _mean_pnl(pnls):
        return float(np.mean(pnls)) if pnls else 0.0

    print(f"  Oracle avg exit PnL: {_mean_pnl(oracle_pnls):.3f}%  n={len(oracle_pnls)}")
    print(f"  Clf    avg exit PnL: {_mean_pnl(clf_pnls):.3f}%  n={len(clf_pnls)}")

    gate_pass = prec >= 0.35 and rec >= 0.25

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(best_state or model.state_dict(), MODELS_DIR / "exit_classifier.pt")
    meta_out = {
        "state_dim": STATE_DIM,
        "hidden": HIDDEN,
        "state_features": ["pnl_pct_norm","bars_held_norm","atr_pct_norm",
                           "p_bear","p_neutral","p_bull","breadth_up",
                           "btc_ret_norm","is_meme","side_sign"],
        "exit_threshold": best_thr,
        "test_exit_precision": round(prec, 4),
        "test_exit_recall": round(rec, 4),
        "test_exit_f1": round(f1, 4),
        "oracle_avg_pnl": round(_mean_pnl(oracle_pnls), 4),
        "clf_avg_pnl": round(_mean_pnl(clf_pnls), 4),
        "gate_pass": bool(gate_pass),
        "train_cutoff": str(TRAIN_CUTOFF),
        "val_cutoff": str(VAL_CUTOFF),
    }
    with open(MODELS_DIR / "exit_classifier_meta.json", "w") as f:
        json.dump(meta_out, f, indent=2)

    md = [f"# v_new_2 Adaptive Exit Classifier — OOS Verdict ({time.strftime('%Y-%m-%d')})\n\n"]
    md.append(f"Supervised exit classifier trained on {len(tr_steps):,} bar-steps (oracle ATR trail).\n")
    md.append(f"Uses LSTM regime probs as state features for regime-awareness.\n\n")
    md.append(f"## OOS Test Metrics\n\n| Metric | Value |\n|---|---|\n")
    md.append(f"| Exit Precision | {prec:.3f} |\n")
    md.append(f"| Exit Recall | {rec:.3f} |\n")
    md.append(f"| Exit F1 | {f1:.3f} |\n")
    md.append(f"| Oracle avg exit PnL | {_mean_pnl(oracle_pnls):.3f}% |\n")
    md.append(f"| Classifier avg exit PnL | {_mean_pnl(clf_pnls):.3f}% |\n\n")
    md.append(f"## Gate\n\n")
    md.append(f"- Precision ≥ 0.35: **{'PASS' if prec >= 0.35 else 'FAIL'}** ({prec:.3f})\n")
    md.append(f"- Recall ≥ 0.25: **{'PASS' if rec >= 0.25 else 'FAIL'}** ({rec:.3f})\n\n")
    md.append(f"**Overall: {'PROMOTE — integrate into executor' if gate_pass else 'HOLD'}**\n")
    (RESULTS_DIR / "rl_exit_verdict.md").write_text("".join(md))

    print(f"\n{_ts()} Saved → {MODELS_DIR}")
    print(f"  Gate: {'PASS' if gate_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
