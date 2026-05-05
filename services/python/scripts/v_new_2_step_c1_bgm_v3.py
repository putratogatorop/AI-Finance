"""v_new_2 Step C1 — BGM v3 with vol/mcap feature.

Merges vol_to_mcap_24h (and a few related vol/mcap derived features) into the
trades dataset, retrains LONG + SHORT BGM, sweeps thresholds, compares to v2.

New BGM features added on top of v2 (60 features):
  vol_to_mcap_24h           — daily turnover ratio (real "trending coin" signal)
  vol_to_mcap_24h_log       — log1p of above (better-distributed)
  vol_to_mcap_above_5pct    — bool: vol/mcap >= 0.05
  vol_to_mcap_above_10pct   — bool: vol/mcap >= 0.10
  vol_to_mcap_above_20pct   — bool: vol/mcap >= 0.20  (mega-trend trigger)

Caveat: vol/mcap data covers ~50% of trades (universe coins not in CG top-500
have NaN). LightGBM handles NaN natively — those rows just ignore the
new feature.

Output:
  models/v_new_2_bgm_long_v3/{model.joblib, meta.json}
  models/v_new_2_bgm_short_v3/{model.joblib, meta.json}
  data/v_new_1_v2/v_new_2_trades_scored_v3.parquet
  results/v_new_2/concentrated_v3_threshold_sweep.csv
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
VOL_ATTN_FEATURES = ["vol_spec_score", "liq_attention", "trend_persistence"]
VMCAP_FEATURES = [
    "vol_to_mcap_24h","vol_to_mcap_24h_log",
    "vol_to_mcap_above_5pct","vol_to_mcap_above_10pct","vol_to_mcap_above_20pct",
]
TIER_DUMMIES = ["tier_top25", "tier_26-50", "tier_51-100", "tier_101-200"]

THRESHOLDS = [0.0, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.93]


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _add_engineered(df):
    df = df.copy()
    df["__tier_mult"] = df["tier"].map(TIER_MULT).fillna(1.5)
    df["vol_spec_score"] = df["coin_24h_vol_zscore_30d"] * df["__tier_mult"]
    df["liq_attention"] = df["vol_z_24h"] * np.log(df["vol_rank_in_top100"].fillna(50) + 1.0)
    sign_spread = np.sign(df["d_ema_spread_pct"].fillna(0))
    df["trend_persistence"] = df["days_since_long_flip"].fillna(0) * sign_spread
    return df.drop(columns=["__tier_mult"])


def _add_vmcap(df, vmcap):
    print(f"  merging vol/mcap...")
    vmcap = vmcap.rename(columns={"timestamp": "entry_time"})
    df = df.merge(vmcap[["symbol","entry_time","vol_to_mcap_24h"]],
                   on=["symbol","entry_time"], how="left")
    df["vol_to_mcap_24h_log"] = np.log1p(df["vol_to_mcap_24h"].fillna(0))
    df["vol_to_mcap_above_5pct"] = (df["vol_to_mcap_24h"] >= 0.05).astype(int)
    df["vol_to_mcap_above_10pct"] = (df["vol_to_mcap_24h"] >= 0.10).astype(int)
    df["vol_to_mcap_above_20pct"] = (df["vol_to_mcap_24h"] >= 0.20).astype(int)
    coverage = df["vol_to_mcap_24h"].notna().mean() * 100
    print(f"  vol/mcap coverage: {coverage:.1f}%")
    return df


def _split(trades, side):
    sub = trades[trades["side"] == side].copy()
    sub["entry_time"] = pd.to_datetime(sub["entry_time"], utc=True)
    return (
        sub[sub["entry_time"] < CAL_START],
        sub[(sub["entry_time"] >= CAL_START) & (sub["entry_time"] < OOS_START)],
        sub[sub["entry_time"] >= OOS_START],
    )


def _train_optuna(X_train, y_train, n_trials=30):
    def objective(trial):
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
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X_train[:split], y_train[:split],
                eval_set=[(X_train[split:], y_train[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        return float(average_precision_score(y_train[split:], clf.predict_proba(X_train[split:])[:,1]))
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _train_side(trades, side, feat_cols):
    print(f"\n{_ts()} === Training v3 BGM for {side.upper()} ===")
    train, cal, oos = _split(trades, side)
    for df_part in [train, cal, oos]:
        for t in ["top25","26-50","51-100","101-200"]:
            df_part[f"tier_{t}"] = (df_part["tier"] == t).astype(int)
    def _clean(df_part):
        nn = df_part[V_NEW_1_FEATURES].notna().sum(axis=1)
        return df_part[nn >= 20]
    train = _clean(train); cal = _clean(cal); oos = _clean(oos)
    print(f"  train {len(train):,}  cal {len(cal):,}  oos {len(oos):,}")
    X = train[feat_cols].fillna(0).values; y = train["is_profitable"].values
    print(f"  Optuna...")
    best = _train_optuna(X, y, n_trials=30)
    full_params = {**best, "objective":"binary","metric":"average_precision",
                    "verbosity":-1,"boosting_type":"gbdt","random_state":42,"num_threads":4}
    model = lgb.LGBMClassifier(**full_params)
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    def _eval(df_part):
        if len(df_part)==0: return {"n":0}
        Xp = df_part[feat_cols].fillna(0).values
        yp = df_part["is_profitable"].values
        if len(set(yp))<2: return {"n":int(len(yp))}
        probs = model.predict_proba(Xp)[:,1]
        return {"n":int(len(yp)),
                 "pr_auc": round(float(average_precision_score(yp, probs)), 4),
                 "roc_auc": round(float(roc_auc_score(yp, probs)), 4),
                 "top_decile_precision": round(float(yp[probs >= np.percentile(probs, 90)].mean()), 4),
                 "top_5pct_precision": round(float(yp[probs >= np.percentile(probs, 95)].mean()), 4)}

    cal_m = _eval(cal); oos_m = _eval(oos)
    print(f"  CAL: {cal_m}")
    print(f"  OOS: {oos_m}")
    importance = model.booster_.feature_importance(importance_type="gain")
    fi = sorted(zip(feat_cols, importance), key=lambda x:-x[1])[:15]
    print(f"  Top-15 features:")
    for n, g in fi:
        marker = " ⭐VMCAP" if n in VMCAP_FEATURES else (" ⭐ATTN" if n in VOL_ATTN_FEATURES else "")
        print(f"    {n:36s} gain={g:.0f}{marker}")
    md = MODEL_BASE / f"v_new_2_bgm_{side}_v3"
    md.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, md / "model.joblib")
    with open(md / "meta.json", "w") as f:
        json.dump({"side": side, "trained_at": pd.Timestamp.utcnow().isoformat(),
                    "best_params": best, "cal_metrics": cal_m, "oos_metrics": oos_m,
                    "feature_names": feat_cols,
                    "feature_importance_top15": [{"feature":n,"gain":float(g)} for n,g in fi]},
                   f, indent=2, default=str)
    return model, train, cal, oos


def _summary(pnls, leverage=1.0):
    if len(pnls)==0: return {"n":0,"wr":0,"pf":0,"sum_pnl_pct":0,"avg_win":0,"avg_loss":0}
    p = np.clip(pnls*leverage, -0.99, 100.0)
    wins = p[p>0]; losses = p[p<0]
    return {"n":int(len(p)), "wr":float((p>0).mean()),
             "sum_pnl_pct":float(p.sum()*100),
             "avg_win":float(wins.mean()*100) if len(wins) else 0,
             "avg_loss":float(losses.mean()*100) if len(losses) else 0,
             "pf":float(wins.sum()/abs(losses.sum())) if len(losses) else float("inf")}


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 Step C1 — BGM v3 with vol/mcap")
    trades = pd.read_parquet(DATA_DIR / "v_new_2_trades_dataset.parquet")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"  loaded {len(trades):,} trades")

    print(f"\n{_ts()} Engineering features...")
    trades = _add_engineered(trades)
    vmcap = pd.read_parquet(DATA_DIR / "vol_mcap_feature.parquet",
                              columns=["symbol","timestamp","vol_to_mcap_24h"])
    vmcap["timestamp"] = pd.to_datetime(vmcap["timestamp"], utc=True)
    trades = _add_vmcap(trades, vmcap)
    feat_cols = (V_NEW_1_FEATURES + RULE_FEATURES + VOL_ATTN_FEATURES
                  + VMCAP_FEATURES + TIER_DUMMIES)
    print(f"  total features: {len(feat_cols)} (added {len(VMCAP_FEATURES)} vmcap)")

    long_model, lt, lc, lo = _train_side(trades, "long", feat_cols)
    short_model, st, sc, so = _train_side(trades, "short", feat_cols)

    # Score everything
    print(f"\n{_ts()} Scoring all trades...")
    parts = []
    for side, model, train, cal, oos in [("long",long_model,lt,lc,lo),
                                            ("short",short_model,st,sc,so)]:
        for split, df_part in [("train",train),("cal",cal),("oos",oos)]:
            if len(df_part)==0: continue
            df_part = df_part.copy()
            df_part["bgm_score"] = model.predict_proba(df_part[feat_cols].fillna(0).values)[:,1]
            df_part["split"] = split
            parts.append(df_part)
    scored = pd.concat(parts, ignore_index=True)
    scored.to_parquet(DATA_DIR / "v_new_2_trades_scored_v3.parquet", index=False)
    print(f"  saved scored ({len(scored):,} rows)")

    # Sweep
    print(f"\n{_ts()} Threshold sweep")
    rows = []
    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            for thr in THRESHOLDS:
                sub = scored[(scored["side"]==side) & (scored["tier"]==tier) & (scored["bgm_score"]>=thr)]
                sub_oos = sub[sub["entry_time"] >= OOS_START]
                s_full = _summary(sub["pnl_pct"].values)
                s_oos = _summary(sub_oos["pnl_pct"].values)
                rows.append({"side":side,"tier":tier,"threshold":thr,
                              "full_n":s_full["n"],"full_wr":s_full["wr"]*100,"full_pf":s_full["pf"],
                              "full_avg_win":s_full["avg_win"],"full_avg_loss":s_full["avg_loss"],
                              "full_sum_pnl_pct":s_full["sum_pnl_pct"],
                              "oos_n":s_oos["n"],"oos_wr":s_oos["wr"]*100,"oos_pf":s_oos["pf"],
                              "oos_sum_pnl_pct":s_oos["sum_pnl_pct"]})
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "concentrated_v3_threshold_sweep.csv", index=False)
    print(f"  saved sweep")

    # Compare v2 vs v3 — best per cell at thr 0.85
    print(f"\n{_ts()} v2 vs v3 comparison @ thr=0.85 (OOS)")
    try:
        v2 = pd.read_csv(RESULTS / "concentrated_threshold_sweep.csv")
    except Exception:
        v2 = None
    print(f"  {'side':5s} {'tier':9s}  {'v2_PF':>7s} → {'v3_PF':>7s}  {'v2_WR':>6s} → {'v3_WR':>6s}  delta_PF")
    for side in ["long","short"]:
        for tier in ["top25","26-50","51-100","101-200"]:
            v3 = df_out[(df_out["side"]==side) & (df_out["tier"]==tier) & (df_out["threshold"]==0.85)]
            if len(v3)==0: continue
            v3r = v3.iloc[0]
            v2_pf = 0; v2_wr = 0
            if v2 is not None:
                v2c = v2[(v2["side"]==side) & (v2["tier"]==tier) & (v2["threshold"]==0.85)]
                if len(v2c)>0:
                    v2_pf = v2c["oos_pf"].iloc[0]; v2_wr = v2c["oos_wr"].iloc[0]
            delta = v3r["oos_pf"] - v2_pf
            print(f"  {side:5s} {tier:9s}  {v2_pf:>7.2f} → {v3r['oos_pf']:>7.2f}  "
                  f"{v2_wr:>5.1f}% → {v3r['oos_wr']:>5.1f}%  {delta:>+7.2f}")

    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
