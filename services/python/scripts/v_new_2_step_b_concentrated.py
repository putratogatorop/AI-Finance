"""v_new_2 Step B — Add vol/mcap-attention features + retrain BGM concentrated.

Adds three engineered features to the BGM training set:
  vol_spec_score   = coin_24h_vol_zscore_30d × tier_multiplier
                     (tier_mult: top25=1.0, 26-50=1.4, 51-100=1.8, 101-200=2.2)
                     Captures "high vol on small cap" attention surge.
  liq_attention    = vol_z_24h × log(rank+1)
                     Speculative activity weighted by illiquidity.
  trend_persistence = days_since_long_flip × sign(d_ema_spread_pct)
                     Long-trend coins that have ridden the trend longer
                     get higher score.

Then retrains LONG and SHORT BGM with threshold sweep up to 0.90.

Output:
  models/v_new_2_bgm_long_v2/{model.joblib, meta.json}
  models/v_new_2_bgm_short_v2/{model.joblib, meta.json}
  data/v_new_1_v2/v_new_2_trades_scored_v2.parquet
  results/v_new_2/concentrated_threshold_sweep.csv
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
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
MODEL_BASE = ROOT / "services" / "python" / "models"

CAL_START = pd.Timestamp("2025-01-01", tz="UTC")
OOS_START = pd.Timestamp("2025-07-01", tz="UTC")

TIER_MULT = {"top25": 1.0, "26-50": 1.4, "51-100": 1.8, "101-200": 2.2}

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
NEW_VOL_FEATURES = ["vol_spec_score", "liq_attention", "trend_persistence"]
TIER_DUMMIES = ["tier_top25", "tier_26-50", "tier_51-100", "tier_101-200"]

THRESHOLDS = [0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93]


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["__tier_mult"] = df["tier"].map(TIER_MULT).fillna(1.5)
    df["vol_spec_score"] = df["coin_24h_vol_zscore_30d"] * df["__tier_mult"]
    df["liq_attention"]  = df["vol_z_24h"] * np.log(df["vol_rank_in_top100"].fillna(50) + 1.0)
    sign_spread = np.sign(df["d_ema_spread_pct"].fillna(0))
    df["trend_persistence"] = df["days_since_long_flip"].fillna(0) * sign_spread
    return df.drop(columns=["__tier_mult"])


def _split(trades: pd.DataFrame, side: str):
    sub = trades[trades["side"] == side].copy()
    sub["entry_time"] = pd.to_datetime(sub["entry_time"], utc=True)
    return (
        sub[sub["entry_time"] < CAL_START],
        sub[(sub["entry_time"] >= CAL_START) & (sub["entry_time"] < OOS_START)],
        sub[sub["entry_time"] >= OOS_START],
    )


def _train_optuna(X_train, y_train, n_trials=30):
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
            "verbosity": -1, "boosting_type": "gbdt", "random_state": 42, "num_threads": 4,
        }
        n = len(X_train); split = int(n * 0.8)
        X_tr, X_va = X_train[:split], X_train[split:]
        y_tr, y_va = y_train[:split], y_train[split:]
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        return float(average_precision_score(y_va, clf.predict_proba(X_va)[:,1]))
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _train_side(trades, side, feat_cols):
    print(f"\n{_ts()} === Training v2 BGM for {side.upper()} ===")
    train, cal, oos = _split(trades, side)
    for df_part in [train, cal, oos]:
        for t in ["top25","26-50","51-100","101-200"]:
            df_part[f"tier_{t}"] = (df_part["tier"] == t).astype(int)
    def _clean(df_part: pd.DataFrame):
        nn = df_part[V_NEW_1_FEATURES].notna().sum(axis=1)
        return df_part[nn >= 20]
    train = _clean(train); cal = _clean(cal); oos = _clean(oos)
    print(f"  train {len(train):,}  cal {len(cal):,}  oos {len(oos):,}")
    X_train = train[feat_cols].fillna(0).values
    y_train = train["is_profitable"].values
    print(f"  Optuna...")
    best_params = _train_optuna(X_train, y_train, n_trials=30)
    full_params = {**best_params,
                   "objective": "binary", "metric": "average_precision",
                   "verbosity": -1, "boosting_type": "gbdt",
                   "random_state": 42, "num_threads": 4}
    final_model = lgb.LGBMClassifier(**full_params)
    final_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])
    def _eval(df_part):
        if len(df_part)==0: return {"n":0}
        X = df_part[feat_cols].fillna(0).values; y = df_part["is_profitable"].values
        if len(set(y))<2: return {"n":int(len(y))}
        probs = final_model.predict_proba(X)[:,1]
        return {
            "n":int(len(y)),
            "pr_auc": round(float(average_precision_score(y, probs)), 4),
            "roc_auc": round(float(roc_auc_score(y, probs)), 4),
            "top_decile_precision": round(float(y[probs >= np.percentile(probs, 90)].mean()), 4),
            "top_5pct_precision": round(float(y[probs >= np.percentile(probs, 95)].mean()), 4),
        }
    cal_m = _eval(cal); oos_m = _eval(oos)
    print(f"  CAL: {cal_m}")
    print(f"  OOS: {oos_m}")
    importance = final_model.booster_.feature_importance(importance_type="gain")
    fi = sorted(zip(feat_cols, importance), key=lambda x:-x[1])[:15]
    print(f"  Top-15 features:")
    for name, gain in fi:
        marker = " ⭐NEW" if name in NEW_VOL_FEATURES else ""
        print(f"    {name:36s} gain={gain:.0f}{marker}")
    model_dir = MODEL_BASE / f"v_new_2_bgm_{side}_v2"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, model_dir / "model.joblib")
    with open(model_dir / "meta.json", "w") as f:
        json.dump({"side": side, "trained_at": pd.Timestamp.utcnow().isoformat(),
                    "best_params": best_params, "cal_metrics": cal_m, "oos_metrics": oos_m,
                    "feature_names": feat_cols,
                    "feature_importance_top15": [{"feature":n,"gain":float(g)} for n,g in fi]},
                   f, indent=2, default=str)
    return final_model, train, cal, oos


def _summary(pnls, leverage=1.0):
    if len(pnls)==0: return {"n":0,"wr":0,"pf":0,"sum_pnl_pct":0,"avg_win":0,"avg_loss":0}
    p = np.clip(pnls*leverage, -0.99, 10.0)
    wins = p[p>0]; losses = p[p<0]
    return {
        "n": int(len(p)), "wr": float((p>0).mean()),
        "sum_pnl_pct": float(p.sum()*100),
        "avg_win": float(wins.mean()*100) if len(wins) else 0,
        "avg_loss": float(losses.mean()*100) if len(losses) else 0,
        "pf": float(wins.sum()/abs(losses.sum())) if len(losses) else float("inf"),
    }


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 Step B — concentrated BGM")
    trades = pd.read_parquet(DATA_DIR / "v_new_2_trades_dataset.parquet")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"  loaded {len(trades):,} trades")

    print(f"\n{_ts()} Engineering vol/mcap features...")
    trades = _add_engineered(trades)
    for c in NEW_VOL_FEATURES:
        print(f"  {c:24s}  median={trades[c].median():.3f}  std={trades[c].std():.3f}  NaN%={trades[c].isna().mean()*100:.1f}")

    feat_cols = V_NEW_1_FEATURES + RULE_FEATURES + NEW_VOL_FEATURES + TIER_DUMMIES
    print(f"  total features: {len(feat_cols)} (added {len(NEW_VOL_FEATURES)} new)")

    long_model, lt, lc, lo = _train_side(trades, "long", feat_cols)
    short_model, st, sc, so = _train_side(trades, "short", feat_cols)

    # Score everything
    print(f"\n{_ts()} Scoring all trades...")
    scored_parts = []
    for side, model, train, cal, oos in [("long", long_model, lt, lc, lo),
                                            ("short", short_model, st, sc, so)]:
        for split, df_part in [("train", train), ("cal", cal), ("oos", oos)]:
            if len(df_part)==0: continue
            df_part = df_part.copy()
            df_part["bgm_score"] = model.predict_proba(df_part[feat_cols].fillna(0).values)[:,1]
            df_part["split"] = split
            scored_parts.append(df_part)
    scored = pd.concat(scored_parts, ignore_index=True)
    scored.to_parquet(DATA_DIR / "v_new_2_trades_scored_v2.parquet", index=False)
    print(f"  saved scored ({len(scored):,} rows)")

    # Threshold sweep — concentrate
    print(f"\n{_ts()} Threshold sweep — concentrated regime")
    rows = []
    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            for thr in THRESHOLDS:
                sub_full = scored[(scored["side"]==side) & (scored["tier"]==tier) & (scored["bgm_score"]>=thr)]
                sub_oos  = sub_full[sub_full["entry_time"] >= OOS_START]
                s_full = _summary(sub_full["pnl_pct"].values)
                s_oos = _summary(sub_oos["pnl_pct"].values)
                rows.append({
                    "side": side, "tier": tier, "threshold": thr,
                    "full_n": s_full["n"], "full_wr": s_full["wr"]*100, "full_pf": s_full["pf"],
                    "full_avg_win": s_full["avg_win"], "full_avg_loss": s_full["avg_loss"],
                    "full_sum_pnl_pct": s_full["sum_pnl_pct"],
                    "oos_n": s_oos["n"], "oos_wr": s_oos["wr"]*100, "oos_pf": s_oos["pf"],
                    "oos_avg_win": s_oos["avg_win"], "oos_avg_loss": s_oos["avg_loss"],
                    "oos_sum_pnl_pct": s_oos["sum_pnl_pct"],
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "concentrated_threshold_sweep.csv", index=False)
    print(f"  saved: concentrated_threshold_sweep.csv")

    # Print compressed view
    print(f"\n{_ts()} Concentrated comparison (n>=30 OOS, sorted by OOS PF)")
    print(f"  {'side':5s} {'tier':9s} {'thr':>5s} {'n_full':>6s} {'PF_full':>7s} {'WR_oos':>7s} {'n_oos':>5s} {'PF_oos':>7s} {'avg_w':>6s} {'avg_l':>6s}")
    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            sub = df_out[(df_out["side"]==side) & (df_out["tier"]==tier) & (df_out["oos_n"]>=30)]
            sub = sub.sort_values("oos_pf", ascending=False).head(3)
            for _, r in sub.iterrows():
                print(f"  {side:5s} {tier:9s} {r['threshold']:>5.2f} {int(r['full_n']):>6} "
                      f"{r['full_pf']:>7.2f} {r['oos_wr']:>6.1f}% {int(r['oos_n']):>5} {r['oos_pf']:>7.2f} "
                      f"{r['oos_avg_win']:>+5.1f}% {r['oos_avg_loss']:>+5.1f}%")

    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
