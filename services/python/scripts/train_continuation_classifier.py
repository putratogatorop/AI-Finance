"""Train the v1 continuation classifier on Strategy-B trades.

Pipeline (per the plan):
  1. Load Strategy-B trades (68,894 rows) and snapshot candles (19M rows).
  2. Filter labelable (entry + 192 bars within snapshot).
  3. Build features. Drop rows with NaN in any feature column.
  4. Time-ordered split: train < 2026-01-01 UTC, OOT >= 2026-01-01.
  5. Gating decision A — hand-crafted rule. If filtering Strategy-B trades
     by `s = z(asset_4h_ret) + z(btc_4h_ret) + 0.5*z(atr_norm_drop)` above
     median+0.5sigma yields OOT PF >= 1.30 with n >= 200 trades, ship the
     rule and skip the model. Result saved to MODELS_DIR/continuation_v1_rule.json.
  6. Otherwise: 5-fold purged time-series CV inside train, fit
     LogisticRegression + StandardScaler. Compute train/OOT AUC.
  7. Gating decision B — if LR OOT AUC < 0.55, do NOT escalate to LGBM;
     stop and emit a "needs more features" report.
  8. If LR AUC >= 0.55, also fit LGBM as the production candidate.
  9. Save the chosen artifact to services/python/models/continuation_v1.joblib
     plus feature_meta.json.

Run from services/python/:
    python scripts/train_continuation_classifier.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

from src.ml.continuation.features import FEATURE_NAMES, FeatureContext
from src.ml.continuation.labels import filter_labelable, make_continuation_label

# --- Constants ---------------------------------------------------------------

SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOTS_DIR = REPO_ROOT / "data" / "snapshots"
CANDLES_PATH = SNAPSHOTS_DIR / f"candles_15m_{SNAPSHOT_DATE}.parquet"
UNIVERSE_PATH = SNAPSHOTS_DIR / f"universe_{SNAPSHOT_DATE}.parquet"
B_TRADES_PATH = (
    REPO_ROOT / "services/python/results"
    / "backtest_3strat_protocol_B_28975aa0_20260424T093815Z" / "trades.csv"
)

MODELS_DIR = REPO_ROOT / "services/python/models"
MODEL_OUT = MODELS_DIR / "continuation_v1.joblib"
META_OUT = MODELS_DIR / "continuation_v1_meta.json"
RULE_OUT = MODELS_DIR / "continuation_v1_rule.json"

LABEL_FWD_TARGET = 0.18
LABEL_DRAWUP_CAP = 0.04
LABEL_MAX_BARS = 192

RULE_SCORE_QUANTILE = 0.66667    # median + ~0.5 sigma of std-normal
RULE_OOT_PF_FLOOR = 1.30
RULE_OOT_N_FLOOR = 200

LR_C = 1.0


# --- Helpers ----------------------------------------------------------------


def _pf(pnls: np.ndarray) -> float:
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return float("nan")
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses == 0:
        return float("inf")
    return float(wins / losses)


def _zscore(s: pd.Series) -> pd.Series:
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or not np.isfinite(sd):
        return s * 0.0
    return (s - mu) / sd


def _purged_kfold_indices(n: int, k: int = 5, purge: int = 192) -> list[tuple[np.ndarray, np.ndarray]]:
    """Time-series K-fold with a `purge`-bar gap between train and validation.

    Splits sample indices (assumed time-ordered) into `k` contiguous segments.
    For each segment as the validation, the train fold is everything BEFORE
    that segment minus the last `purge` rows (gap), so labels can't leak.
    """
    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        train_end = max(0, val_start - purge)
        if train_end <= 0:
            continue  # first fold has no train history — skip
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


# --- Main pipeline -----------------------------------------------------------


def main() -> None:
    started = time.monotonic()
    np.random.seed(RANDOM_SEED)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load -----------------------------------------------------------
    print(f"loading Strategy-B trades from {B_TRADES_PATH}", flush=True)
    trades = pd.read_csv(B_TRADES_PATH)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    print(f"  trades: {len(trades):,}", flush=True)

    print(f"loading candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    print(f"  candles: {len(candles):,} rows, {candles['asset'].nunique()} assets", flush=True)

    universe_listed_since: dict[str, pd.Timestamp] = {}
    if UNIVERSE_PATH.exists():
        u = pd.read_parquet(UNIVERSE_PATH)
        # Universe parquet has assets as 'BTC', candles use 'BTCUSDT'.
        if "asset" in u.columns and "listed_since" in u.columns:
            for _, row in u.iterrows():
                asset_with_usdt = f"{row['asset']}USDT"
                ls = row["listed_since"]
                if pd.notna(ls):
                    universe_listed_since[asset_with_usdt] = pd.Timestamp(ls, tz="UTC")
        print(f"  universe entries with listed_since: {len(universe_listed_since)}", flush=True)

    # ---- Filter labelable trades ---------------------------------------
    snapshot_max = candles["timestamp"].max()
    keep = filter_labelable(trades, snapshot_max, max_bars=LABEL_MAX_BARS)
    trades = trades[keep].reset_index(drop=True)
    print(f"labelable trades (horizon fits): {len(trades):,}", flush=True)

    # ---- Compute labels (this scans candles per trade — minutes) -------
    print("computing labels...", flush=True)
    labels = make_continuation_label(
        trades, candles,
        fwd_target=LABEL_FWD_TARGET,
        drawup_cap=LABEL_DRAWUP_CAP,
        max_bars=LABEL_MAX_BARS,
    )
    trades["label"] = labels.values
    trades = trades[trades["label"] != -1].reset_index(drop=True)
    pos_rate = float((trades["label"] == 1).mean())
    print(f"  labeled: {len(trades):,} | positive rate: {pos_rate:.1%}", flush=True)

    # ---- Build features (heavy: panels + per-trade lookup) -------------
    print("building feature panels (this is the heavy step)...", flush=True)
    ctx = FeatureContext(candles, btc_asset_key="BTCUSDT",
                         universe_listed_since=universe_listed_since)
    print("  panels built; computing per-trade features...", flush=True)
    feat_rows = []
    for i, row in enumerate(trades.itertuples(index=False)):
        feat_rows.append(ctx._build_for_typed(row.symbol, row.entry_time))
        if (i + 1) % 5000 == 0:
            print(f"  features: {i+1}/{len(trades)}", flush=True)
    X = pd.DataFrame(feat_rows)

    # Drop rows with any NaN feature.
    finite_mask = X.notna().all(axis=1)
    print(f"  feature-complete rows: {int(finite_mask.sum())}/{len(trades)}", flush=True)
    trades = trades[finite_mask].reset_index(drop=True)
    X = X[finite_mask].reset_index(drop=True)
    y = trades["label"].astype(int).to_numpy()
    pnl = trades["pnl_pct"].to_numpy()
    et = trades["entry_time"]

    # Train/OOT split.
    is_train = et < TRAIN_OOT_BOUNDARY
    is_oot = ~is_train
    print(f"  train: {is_train.sum():,}  oot: {is_oot.sum():,}", flush=True)

    # ---- Gating decision A: hand-crafted rule --------------------------
    print("\n--- Gating A: hand-crafted score on train, evaluate on OOT ---")
    s_train = (
        _zscore(X.loc[is_train, "asset_4h_ret"])
        + _zscore(X.loc[is_train, "btc_4h_ret"])
        + 0.5 * _zscore(X.loc[is_train, "atr_norm_drop"])
    )
    score_threshold = float(s_train.quantile(RULE_SCORE_QUANTILE))
    # Apply same z-score moments to OOT to avoid in-sample leak on rule.
    moments = {
        col: (
            float(X.loc[is_train, col].mean()),
            float(X.loc[is_train, col].std(ddof=0)),
        )
        for col in ["asset_4h_ret", "btc_4h_ret", "atr_norm_drop"]
    }

    def _score(df_X: pd.DataFrame) -> pd.Series:
        a = (df_X["asset_4h_ret"] - moments["asset_4h_ret"][0]) / max(moments["asset_4h_ret"][1], 1e-12)
        b = (df_X["btc_4h_ret"] - moments["btc_4h_ret"][0]) / max(moments["btc_4h_ret"][1], 1e-12)
        c = (df_X["atr_norm_drop"] - moments["atr_norm_drop"][0]) / max(moments["atr_norm_drop"][1], 1e-12)
        return a + b + 0.5 * c

    s_full = _score(X)
    rule_keep = s_full > score_threshold
    train_kept = is_train & rule_keep
    oot_kept = is_oot & rule_keep
    rule_train_pf = _pf(pnl[train_kept])
    rule_oot_pf = _pf(pnl[oot_kept])
    rule_train_n = int(train_kept.sum())
    rule_oot_n = int(oot_kept.sum())
    print(f"  threshold (z-sum > q{RULE_SCORE_QUANTILE:.3f} on train): {score_threshold:.3f}")
    print(f"  train kept={rule_train_n}  PF={rule_train_pf:.3f}")
    print(f"  oot   kept={rule_oot_n}  PF={rule_oot_pf:.3f}")

    rule_passes = (
        np.isfinite(rule_oot_pf)
        and rule_oot_pf >= RULE_OOT_PF_FLOOR
        and rule_oot_n >= RULE_OOT_N_FLOOR
    )
    rule_summary = {
        "threshold": score_threshold,
        "moments": moments,
        "train_n": rule_train_n,
        "train_pf": rule_train_pf,
        "oot_n": rule_oot_n,
        "oot_pf": rule_oot_pf,
        "passes_gate_a": bool(rule_passes),
    }
    RULE_OUT.write_text(json.dumps(rule_summary, indent=2, default=str))
    print(f"  wrote {RULE_OUT}")

    if rule_passes:
        print("\nGate A PASSED — hand-crafted rule meets ship floor on OOT.")
        print("Skipping model training; rule artifact is sufficient.")
        elapsed = time.monotonic() - started
        print(f"\ntotal wall time: {elapsed:.1f}s")
        return

    # ---- Logistic-regression baseline ---------------------------------
    print("\n--- Gating B: logistic regression baseline (5-fold purged time CV) ---")
    train_X = X.loc[is_train].reset_index(drop=True).to_numpy(dtype=float)
    train_y = y[is_train.to_numpy()]
    oot_X = X.loc[is_oot].reset_index(drop=True).to_numpy(dtype=float)
    oot_y = y[is_oot.to_numpy()]

    cv_aucs = []
    for fold_i, (tr_idx, va_idx) in enumerate(_purged_kfold_indices(len(train_X), k=5, purge=LABEL_MAX_BARS)):
        if len(np.unique(train_y[tr_idx])) < 2 or len(np.unique(train_y[va_idx])) < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced")),
        ])
        pipe.fit(train_X[tr_idx], train_y[tr_idx])
        proba_val = pipe.predict_proba(train_X[va_idx])[:, 1]
        auc = roc_auc_score(train_y[va_idx], proba_val)
        cv_aucs.append(float(auc))
        print(f"  fold {fold_i}: AUC={auc:.4f} (n_train={len(tr_idx):,}, n_val={len(va_idx):,})")

    if not cv_aucs:
        print("  WARNING: no usable folds (label is single-class in some fold).")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced")),
    ])
    pipe.fit(train_X, train_y)
    train_auc = roc_auc_score(train_y, pipe.predict_proba(train_X)[:, 1])
    if len(np.unique(oot_y)) >= 2:
        oot_auc = roc_auc_score(oot_y, pipe.predict_proba(oot_X)[:, 1])
    else:
        oot_auc = float("nan")
    print(f"  full-train AUC: {train_auc:.4f}  OOT AUC: {oot_auc:.4f}")

    if not np.isfinite(oot_auc) or oot_auc < 0.55:
        print("\nGate B FAILED — LR OOT AUC < 0.55. Stopping per protocol; the predictor is")
        print("not informative enough to warrant model complexity. Add features or revisit labels.")
        meta = {
            "model": "none",
            "reason": "lr_oot_auc_below_0.55",
            "train_auc": float(train_auc),
            "oot_auc": float(oot_auc),
            "cv_aucs": cv_aucs,
            "feature_names": FEATURE_NAMES,
            "label_thresholds": {
                "fwd_target": LABEL_FWD_TARGET,
                "drawup_cap": LABEL_DRAWUP_CAP,
                "max_bars": LABEL_MAX_BARS,
            },
            "trained_on_n": int(len(train_X)),
            "snapshot_date": SNAPSHOT_DATE,
        }
        META_OUT.write_text(json.dumps(meta, indent=2, default=str))
        print(f"  wrote {META_OUT}")
        elapsed = time.monotonic() - started
        print(f"\ntotal wall time: {elapsed:.1f}s")
        return

    # ---- Save LR as production candidate ------------------------------
    joblib.dump(pipe, MODEL_OUT)
    meta = {
        "model": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "label_thresholds": {
            "fwd_target": LABEL_FWD_TARGET,
            "drawup_cap": LABEL_DRAWUP_CAP,
            "max_bars": LABEL_MAX_BARS,
        },
        "train_auc": float(train_auc),
        "oot_auc": float(oot_auc),
        "cv_aucs": cv_aucs,
        "trained_on_n": int(len(train_X)),
        "trained_on_pos_rate": float(np.mean(train_y)),
        "snapshot_date": SNAPSHOT_DATE,
        "rule_summary": rule_summary,
    }
    META_OUT.write_text(json.dumps(meta, indent=2, default=str))
    print(f"\nLR baseline saved to {MODEL_OUT}")
    print(f"meta written to {META_OUT}")

    elapsed = time.monotonic() - started
    print(f"\ntotal wall time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
