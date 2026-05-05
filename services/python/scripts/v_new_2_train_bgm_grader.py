"""v_new_2 — Train Layer 2 BGM grader on rule-trigger trades.

Two LightGBM binary classifiers (LONG and SHORT) trained on the trades dataset
to predict P(profitable | features at entry bar).

Splits:
  Train:  entry_year ∈ {2020, 2021, 2022, 2023, 2024}
  Cal:    entry_time ∈ [2025-01-01, 2025-06-30]
  OOS:    entry_time ∈ [2025-07-01, 2026-04-01]   (2025 H2 + 2026 Q1)

Output:
  models/v_new_2_bgm_long/{model.joblib, meta.json}
  models/v_new_2_bgm_short/{model.joblib, meta.json}
  data/v_new_1_v2/v_new_2_trades_scored.parquet  (trades + bgm_score column)
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import warnings

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                              log_loss, roc_auc_score)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
MODEL_BASE = ROOT / "services" / "python" / "models"

CAL_START = pd.Timestamp("2025-01-01", tz="UTC")
OOS_START = pd.Timestamp("2025-07-01", tz="UTC")

V_NEW_1_FEATURES = [
    "atr14_pct_rank_90d","vol_z_24h","coin_7d_return","coin_30d_return",
    "close_to_high50_atr","close_to_low50_atr","bar4h_close_pos_in_range",
    "bar4h_body_pct","bar4h_upper_wick_pct","h4_macd_hist","h4_macd_macd",
    "h4_rsi","h4_close_vs_ema50_pct","daily_macd_hist","days_since_bull_flip",
    "days_since_bear_flip","btc_above_4h_ema50","btc_24h_return","btc_realized_vol_z",
    "btc_score","breadth_up","breadth_down","hour_sin","hour_cos","dow_sin","dow_cos",
    "cvd_slope_1h_norm","bb_pct_b","bb_bandwidth","fib_pos_50","kdj_k","kdj_j",
    "price_vs_ema9_pct","rsi14_delta_1bar","rsi14_delta_4bar","macd_hist_momentum",
    "macd_signal_spread_norm","parkinson_vol","obv_slope_norm","cfgi_value",
    "vol_rank_in_top100","days_since_last_big_move_long","days_since_last_big_move_short",
    "coin_24h_vol_zscore_30d",
]
RULE_FEATURES = [
    "d_ema_spread_pct","days_since_long_flip","days_since_short_flip",
    "pullback_pct","rise_pct","pullback_atr","rise_atr","atr14_pct","trend_strength",
]
TIER_DUMMIES = ["tier_top25", "tier_26-50", "tier_51-100", "tier_101-200"]


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _split(trades: pd.DataFrame, side: str):
    sub = trades[trades["side"] == side].copy()
    sub["entry_time"] = pd.to_datetime(sub["entry_time"], utc=True)
    train = sub[sub["entry_time"] < CAL_START]
    cal = sub[(sub["entry_time"] >= CAL_START) & (sub["entry_time"] < OOS_START)]
    oos = sub[sub["entry_time"] >= OOS_START]
    return train, cal, oos


def _train_optuna(X_train, y_train, n_trials=30):
    """Quick Optuna for binary classification — small dataset, low overhead."""
    def objective(trial: optuna.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "max_depth": trial.suggest_int("max_depth", 4, 8),
            "min_child_samples": trial.suggest_int("min_child_samples", 30, 200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 0.95),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 1),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
            "objective": "binary", "metric": "average_precision",
            "verbosity": -1, "boosting_type": "gbdt", "random_state": 42,
            "num_threads": 4,
        }
        # Time-series split internal to train (no shuffling)
        n = len(X_train)
        split = int(n * 0.8)
        X_tr, X_va = X_train[:split], X_train[split:]
        y_tr, y_va = y_train[:split], y_train[split:]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        probs = clf.predict_proba(X_va)[:, 1]
        return float(average_precision_score(y_va, probs))

    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params, study.best_value


def _train_one_side(trades: pd.DataFrame, side: str):
    print(f"\n{_ts()} === Training BGM grader for {side.upper()} ===")
    train, cal, oos = _split(trades, side)
    print(f"  train rows: {len(train):,}  pos_rate: {train['is_profitable'].mean()*100:.1f}%")
    print(f"  cal rows:   {len(cal):,}    pos_rate: {cal['is_profitable'].mean()*100:.1f}%")
    print(f"  oos rows:   {len(oos):,}    pos_rate: {oos['is_profitable'].mean()*100:.1f}%")

    feat_cols = V_NEW_1_FEATURES + RULE_FEATURES + TIER_DUMMIES

    # Build tier dummies
    for df_part in [train, cal, oos]:
        for t in ["top25","26-50","51-100","101-200"]:
            df_part[f"tier_{t}"] = (df_part["tier"] == t).astype(int)

    # Drop rows with too many NaN base features
    def _clean(df_part: pd.DataFrame):
        nn = df_part[V_NEW_1_FEATURES].notna().sum(axis=1)
        return df_part[nn >= 20]

    train = _clean(train); cal = _clean(cal); oos = _clean(oos)
    print(f"  after NaN clean — train {len(train):,} cal {len(cal):,} oos {len(oos):,}")

    X_train = train[feat_cols].fillna(0).values
    y_train = train["is_profitable"].values

    print(f"\n{_ts()} Optuna search (30 trials)...")
    best_params, best_score = _train_optuna(X_train, y_train, n_trials=30)
    print(f"  best PR-AUC: {best_score:.4f}")
    print(f"  best params: {json.dumps(best_params)}")

    # Final fit on full train
    print(f"\n{_ts()} Fitting final model on all {len(train):,} train rows...")
    full_params = {**best_params,
                    "objective": "binary", "metric": "average_precision",
                    "verbosity": -1, "boosting_type": "gbdt",
                    "random_state": 42, "num_threads": 4}
    final_model = lgb.LGBMClassifier(**full_params)
    final_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])

    # Evaluate on cal + oos
    def _eval(df_part, label):
        if len(df_part) == 0:
            return {"n": 0}
        X = df_part[feat_cols].fillna(0).values
        y = df_part["is_profitable"].values
        if len(set(y)) < 2:
            return {"n": int(len(y)), "note": "single class"}
        probs = final_model.predict_proba(X)[:, 1]
        return {
            "n": int(len(y)),
            "pr_auc": round(float(average_precision_score(y, probs)), 4),
            "roc_auc": round(float(roc_auc_score(y, probs)), 4),
            "brier": round(float(brier_score_loss(y, probs)), 4),
            "logloss": round(float(log_loss(y, np.clip(probs,1e-6,1-1e-6))), 4),
            "pos_rate": round(float(y.mean()), 4),
            "top_decile_precision": round(float(y[probs >= np.percentile(probs, 90)].mean()), 4)
                                      if (probs >= np.percentile(probs, 90)).any() else 0,
        }

    cal_metrics = _eval(cal, "cal")
    oos_metrics = _eval(oos, "OOS")
    print(f"\n{_ts()} CAL  : {cal_metrics}")
    print(f"{_ts()} OOS  : {oos_metrics}")

    # Top-15 SHAP-like importance via LightGBM gain
    importance = final_model.booster_.feature_importance(importance_type="gain")
    fi = sorted(zip(feat_cols, importance), key=lambda x: -x[1])[:15]
    print(f"\n{_ts()} Top-15 features by gain:")
    for name, gain in fi:
        print(f"  {name:36s}  {gain:.0f}")

    # Save
    model_dir = MODEL_BASE / f"v_new_2_bgm_{side}"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "model.joblib")
    meta = {
        "side": side,
        "trained_at_utc": pd.Timestamp.utcnow().isoformat(),
        "n_train": int(len(train)), "n_cal": int(len(cal)), "n_oos": int(len(oos)),
        "best_params": best_params,
        "cal_metrics": cal_metrics,
        "oos_metrics": oos_metrics,
        "feature_names": feat_cols,
        "feature_importance_top15": [{"feature": n, "gain": float(g)} for n, g in fi],
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"  saved {model_dir}/model.joblib + meta.json")
    return final_model, feat_cols, train, cal, oos


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_2 BGM grader training")
    trades = pd.read_parquet(DATA_DIR / "v_new_2_trades_dataset.parquet")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"  loaded {len(trades):,} trades")

    long_model, feat_cols_l, lt, lc, lo = _train_one_side(trades, "long")
    short_model, feat_cols_s, st, sc, so = _train_one_side(trades, "short")

    # Score every trade
    print(f"\n{_ts()} Scoring all trades...")
    scored_parts = []
    for side, model, feat_cols, train, cal, oos in [
        ("long",  long_model,  feat_cols_l, lt, lc, lo),
        ("short", short_model, feat_cols_s, st, sc, so),
    ]:
        for split_name, df_part in [("train", train), ("cal", cal), ("oos", oos)]:
            if len(df_part) == 0: continue
            X = df_part[feat_cols].fillna(0).values
            df_part = df_part.copy()
            df_part["bgm_score"] = model.predict_proba(X)[:, 1]
            df_part["split"] = split_name
            scored_parts.append(df_part)
    scored = pd.concat(scored_parts, ignore_index=True)
    out_path = DATA_DIR / "v_new_2_trades_scored.parquet"
    scored.to_parquet(out_path, index=False)
    print(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB)  rows={len(scored):,}")

    elapsed = time.time() - t0
    print(f"\n{_ts()} DONE in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
