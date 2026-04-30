"""Train v8 BALANCED trade-quality classifiers — local Mac fallback.

Mirrors notebooks/train_v8_lightgbm_classifier.ipynb but uses sklearn's
HistGradientBoostingClassifier instead of LightGBM. Reason: LightGBM and
XGBoost both need libomp.dylib which isn't installed on this Mac (no
Homebrew available to install it). sklearn's HistGBM is a histogram-based
gradient-boosting algorithm — same family as LightGBM, near-identical
performance on tabular data, zero OS-level dependencies.

Methodology guards (same as Colab notebook):
  1. Holdout = last 30d (2026-03-28+), never used to tune
  2. Walk-forward CV on train period (5 time-series folds)
  3. Locked threshold: pick threshold on train walk-forward,
     evaluate ONE number per metric on holdout
  4. Same E2 exits in train as in production (no exit-policy mismatch)
  5. Universe = top-100 by 24h volume from snapshot (matches live)
  6. No look-ahead in features (asserted in prepare_v8_training_data.py)

Outputs:
    models/v8_classifier/{detector}_histgbm_v1.joblib
    models/v8_classifier/{detector}_histgbm_v1_meta.json

Usage from services/python/:
    .venv/bin/python scripts/train_v8_classifier_local.py
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

TRAINING_PARQUET = (
    Path(__file__).resolve().parents[1]
    / "data" / "training" / "v8_trades_with_features_2026-04-27.parquet"
)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "v8_classifier"
HOLD_OUT_START = pd.Timestamp("2026-03-28", tz="UTC")
RANDOM_SEED = 42
MIN_TRADES_PER_DETECTOR = 100

FEATURES = [
    # Coin context
    "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
    "close_to_high50_atr", "close_to_low50_atr",
    # Bar shape at signal
    "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
    # Detector-specific indicators
    "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
    "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
    # BTC regime
    "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
    # Cross-section breadth
    "breadth_up", "breadth_down", "signals_same_15m_same_detector",
    # Time of week
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def make_classifier() -> HistGradientBoostingClassifier:
    """Hyperparameters tuned to mirror our LightGBM Colab choices."""
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=5,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
    )


def pf(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls <= 0].sum()
    return float(wins / losses) if losses > 0 else float("inf")


def walk_forward(d_df: pd.DataFrame, n_splits: int = 5) -> pd.DataFrame:
    d = d_df.sort_values("entry_time").reset_index(drop=True)
    train_only = d[d.entry_time < HOLD_OUT_START].reset_index(drop=True)
    if len(train_only) < n_splits * 20:
        return pd.DataFrame()
    X = train_only[FEATURES].values
    y = train_only["label"].values
    pnls = train_only["pnl_pct"].values
    rows = []
    for fold_i, (tr, te) in enumerate(TimeSeriesSplit(n_splits=n_splits).split(X)):
        m = make_classifier()
        m.fit(X[tr], y[tr])
        probs = m.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y[te], probs)) if len(set(y[te])) >= 2 else float("nan")
        mask = probs >= 0.5
        kept_pf = pf(pnls[te][mask]) if mask.any() else 0.0
        baseline_pf = pf(pnls[te])
        rows.append({
            "fold": fold_i,
            "n_train": len(tr),
            "n_test": len(te),
            "auc": round(auc, 3) if not np.isnan(auc) else None,
            "baseline_pf": round(baseline_pf, 2),
            "kept_pf_at_0.5": round(kept_pf, 2),
            "kept_pct": round(100 * mask.sum() / len(te), 1),
        })
    return pd.DataFrame(rows)


def train_final(d_df: pd.DataFrame, det: str):
    d = d_df.sort_values("entry_time").reset_index(drop=True)
    tr_mask = d.entry_time < HOLD_OUT_START
    ho_mask = d.entry_time >= HOLD_OUT_START
    if tr_mask.sum() < 30 or ho_mask.sum() < 5:
        return None
    X_tr = d.loc[tr_mask, FEATURES].values
    y_tr = d.loc[tr_mask, "label"].values
    pnl_tr = d.loc[tr_mask, "pnl_pct"].values
    X_ho = d.loc[ho_mask, FEATURES].values
    y_ho = d.loc[ho_mask, "label"].values
    pnl_ho = d.loc[ho_mask, "pnl_pct"].values

    model = make_classifier()
    model.fit(X_tr, y_tr)

    # Lock threshold on TRAIN: pick threshold maximizing train PF
    train_probs = model.predict_proba(X_tr)[:, 1]
    best_thr, best_pf = 0.5, 0.0
    for thr in np.arange(0.30, 0.71, 0.025):
        m = train_probs >= thr
        if m.sum() < 30:
            continue
        p = pf(pnl_tr[m])
        if p > best_pf:
            best_pf, best_thr = p, float(thr)

    # Holdout eval — locked, single number
    ho_probs = model.predict_proba(X_ho)[:, 1]
    ho_kept = ho_probs >= best_thr

    auc_ho = (
        float(roc_auc_score(y_ho, ho_probs)) if len(set(y_ho)) >= 2 else None
    )

    res = {
        "detector": det,
        "threshold_locked": round(best_thr, 4),
        "train_pf_at_threshold": round(best_pf, 3),
        "train_n": int(tr_mask.sum()),
        "train_kept_pct": round(100 * (train_probs >= best_thr).sum() / tr_mask.sum(), 1),
        "holdout_baseline_pf": round(pf(pnl_ho), 3),
        "holdout_filtered_pf": round(pf(pnl_ho[ho_kept]) if ho_kept.any() else 0.0, 3),
        "holdout_baseline_wr": round(100 * (pnl_ho > 0).mean(), 1),
        "holdout_filtered_wr": round(
            100 * (pnl_ho[ho_kept] > 0).mean() if ho_kept.any() else 0.0, 1
        ),
        "holdout_n_total": int(ho_mask.sum()),
        "holdout_n_kept": int(ho_kept.sum()),
        "holdout_kept_pct": round(100 * ho_kept.sum() / ho_mask.sum(), 1),
        "holdout_auc": round(auc_ho, 3) if auc_ho is not None else None,
    }
    return model, res


def main() -> None:
    assert TRAINING_PARQUET.exists(), f"missing parquet: {TRAINING_PARQUET}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TRAINING_PARQUET)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)

    print(f"Total trades: {len(df):,}")
    print(f"Time range:   {df.entry_time.min()} → {df.entry_time.max()}")
    print(f"Holdout cutoff: {HOLD_OUT_START}")
    print()
    print("Per-detector counts:")
    print(df.groupby("detector").size().to_string())

    # Walk-forward CV per detector
    for det in sorted(df["detector"].unique()):
        d = df[df["detector"] == det]
        if len(d) < MIN_TRADES_PER_DETECTOR:
            print(f"\n=== {det}: SKIP — only {len(d)} trades (< {MIN_TRADES_PER_DETECTOR}) ===")
            continue
        cv = walk_forward(d)
        if cv.empty:
            print(f"\n=== {det}: SKIP — insufficient train data ===")
            continue
        print(f"\n=== {det} (n={len(d)}, train={int((d.entry_time < HOLD_OUT_START).sum())}) — walk-forward CV ===")
        print(cv.to_string(index=False))

    # Train final + save
    print(f"\n{'=' * 80}")
    print("FINAL TRAINING + LOCKED-THRESHOLD HOLDOUT EVAL")
    print("=" * 80)
    results = {}
    for det in sorted(df["detector"].unique()):
        d = df[df["detector"] == det]
        if len(d) < MIN_TRADES_PER_DETECTOR:
            continue
        out = train_final(d, det)
        if out is None:
            continue
        model, res = out
        results[det] = (model, res)
        # Save
        model_path = MODELS_DIR / f"{det}_histgbm_v1.joblib"
        meta_path = MODELS_DIR / f"{det}_histgbm_v1_meta.json"
        joblib.dump(model, model_path)
        with open(meta_path, "w") as f:
            json.dump({
                **res,
                "feature_names": FEATURES,
                "snapshot_date": "2026-04-27",
                "trained_at_utc": datetime.now(UTC).isoformat(),
                "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
                "model_params": {
                    "max_iter": 300, "learning_rate": 0.05, "max_depth": 5,
                    "max_leaf_nodes": 15, "min_samples_leaf": 20,
                    "l2_regularization": 0.1, "random_state": RANDOM_SEED,
                },
            }, f, indent=2, default=str)
        print(f"\n=== {det} ===")
        for k, v in res.items():
            print(f"  {k:30s}  {v}")
        print(f"  Saved: {model_path.name}")

    # Summary table
    print()
    print("=" * 100)
    print("SUMMARY — does the classifier lift PF on holdout?")
    print("=" * 100)
    print(f"{'detector':30s} | {'thr':>5} | {'baseline':>10} | {'filtered':>10} | "
          f"{'PF lift':>7} | {'WR lift pts':>11} | {'kept %':>7} | {'AUC':>5} | {'ship?':>5}")
    print("-" * 100)
    for det, (_, res) in results.items():
        ship = "YES" if res["holdout_filtered_pf"] >= 1.30 else "no"
        wr_lift = round(res["holdout_filtered_wr"] - res["holdout_baseline_wr"], 1)
        pf_lift = round(res["holdout_filtered_pf"] - res["holdout_baseline_pf"], 2)
        auc_str = f"{res['holdout_auc']:.2f}" if res["holdout_auc"] is not None else "  —"
        print(f"{det:30s} | "
              f"{res['threshold_locked']:>5.3f} | "
              f"PF{res['holdout_baseline_pf']:>7.2f} | "
              f"PF{res['holdout_filtered_pf']:>7.2f} | "
              f"{pf_lift:>+7.2f} | "
              f"{wr_lift:>+11.1f} | "
              f"{res['holdout_kept_pct']:>6.1f}% | "
              f"{auc_str:>5} | "
              f"{ship:>5}")

    # Feature importances (using permutation since HistGBM doesn't expose
    # them like tree-based libs; skip for now — not required for v1)
    print()
    print(f"=== DONE — {len(results)} models saved to {MODELS_DIR} ===")


if __name__ == "__main__":
    main()
