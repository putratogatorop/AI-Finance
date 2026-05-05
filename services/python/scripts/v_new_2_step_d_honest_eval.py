"""v_new_2 Step D — HONEST evaluation per council recommendations.

Applies four fixes that the council audit flagged:
  1. FIXED-HORIZON labels: BGM target = forward 72h return >= 0 (sign-aligned),
     NOT exit-driven pnl. Base rate ~50%, not ~90%. Honest discrimination test.
  2. FUNDING + SLIPPAGE: per-trade PnL deducts realistic frictions:
        funding: +0.02%/8h on LONG (drag), -0.005%/8h on SHORT (small gain)
        slippage: 0.10% per side (entry+exit) = 0.20% per round-trip
        plus existing 0.10% taker fee round-trip = 0.30% total round-trip drag
  3. LOCKED HOLDOUT: 2026 Q1 (Jan-Mar 2026) is HOLDOUT — never seen by training,
     calibration, or threshold selection. 2025 H2 used for threshold pick.
  4. MEME FILTER: report results both for FULL universe AND for meme-only subset
     (26 CG meme coins).

Output:
  models/v_new_2_bgm_long_honest/{model.joblib, meta.json}
  models/v_new_2_bgm_short_honest/{model.joblib, meta.json}
  data/v_new_1_v2/v_new_2_trades_scored_honest.parquet
  results/v_new_2/honest_threshold_sweep.csv
  results/v_new_2/honest_verdict.md
"""
from __future__ import annotations

import json
import os
import pathlib
import time
import warnings
from typing import Optional

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
THRESHOLD_PICK_START = pd.Timestamp("2025-07-01", tz="UTC")  # 2025 H2 used for threshold pick
HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")          # 2026 Q1 = TRUE LOCKED HOLDOUT

# Friction model
TAKER_FEE_RT = 0.001        # 0.10% round-trip (Gate.io taker × 2)
SLIPPAGE_RT = 0.002         # 0.20% round-trip (0.10% per side adverse)
FUNDING_LONG_PER_8H = 0.0002    # 0.02%/8h drag (longs pay funding in bull)
FUNDING_SHORT_PER_8H = -0.00005 # -0.005%/8h gain (shorts collect in bear)
BARS_PER_8H = 2  # 2 4h bars per 8h funding period

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
THRESHOLDS = [0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _load_meme_universe():
    """Return set of meme symbols from CoinGecko categories."""
    with open(DATA_DIR / "coingecko_categories.json") as f:
        cats = json.load(f)
    return set(cats.get("memes", []))


def _add_fixed_horizon_labels(trades, features_full):
    """For each (symbol, entry_time), compute forward 24h/72h/7d returns from
    raw close prices in features_full. Uses ONLY data ≥ entry_time, no leak."""
    print(f"{_ts()} Building fixed-horizon labels...")
    feats = features_full[["symbol","timestamp","close"]].copy()
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    feats = feats.sort_values(["symbol","timestamp"]).reset_index(drop=True)

    # Forward shifts: 6 bars=24h, 18 bars=72h, 42 bars=7d
    feats["close_24h"] = feats.groupby("symbol")["close"].shift(-6)
    feats["close_72h"] = feats.groupby("symbol")["close"].shift(-18)
    feats["close_7d"]  = feats.groupby("symbol")["close"].shift(-42)
    feats["fwd_24h_ret"] = feats["close_24h"] / feats["close"] - 1.0
    feats["fwd_72h_ret"] = feats["close_72h"] / feats["close"] - 1.0
    feats["fwd_7d_ret"]  = feats["close_7d"] / feats["close"] - 1.0

    join = feats[["symbol","timestamp","close","fwd_24h_ret","fwd_72h_ret","fwd_7d_ret"]]
    join = join.rename(columns={"timestamp": "entry_time", "close": "entry_close"})
    trades = trades.merge(join, on=["symbol","entry_time"], how="left")

    # Sign-align by side
    sign = np.where(trades["side"].values == "long", 1.0, -1.0)
    trades["signed_fwd_24h"] = trades["fwd_24h_ret"] * sign
    trades["signed_fwd_72h"] = trades["fwd_72h_ret"] * sign
    trades["signed_fwd_7d"]  = trades["fwd_7d_ret"]  * sign

    # Binary labels (honest, ~50% base rate)
    trades["fwd_72h_pos"] = (trades["signed_fwd_72h"] > 0).astype(int)
    trades["fwd_72h_above_3pct"] = (trades["signed_fwd_72h"] >= 0.03).astype(int)
    trades["fwd_7d_pos"] = (trades["signed_fwd_7d"] > 0).astype(int)
    trades["fwd_7d_above_5pct"] = (trades["signed_fwd_7d"] >= 0.05).astype(int)

    print(f"  base rates:")
    for c in ["fwd_72h_pos","fwd_72h_above_3pct","fwd_7d_pos","fwd_7d_above_5pct"]:
        nn = trades[c].notna().sum()
        rate = trades[c].mean()
        print(f"    {c:24s}  base_rate={rate*100:5.1f}%  n_valid={nn:,}")
    return trades


def _add_realistic_friction(trades):
    """Recompute pnl_pct deducting funding + slippage."""
    print(f"{_ts()} Adding funding + slippage to per-trade PnL...")
    sign = np.where(trades["side"].values == "long", 1.0, -1.0)
    bars = trades["bars_held"].values.astype(float)
    funding_per_bar = np.where(trades["side"].values == "long",
                                  FUNDING_LONG_PER_8H / BARS_PER_8H,
                                  FUNDING_SHORT_PER_8H / BARS_PER_8H)
    funding_total = funding_per_bar * bars
    trades["pnl_pct_raw"] = trades["pnl_pct"]   # original (just taker fee deducted)
    trades["pnl_pct_honest"] = trades["pnl_pct_raw"] - SLIPPAGE_RT - funding_total
    print(f"  raw    pnl mean: {trades['pnl_pct_raw'].mean()*100:+.2f}%")
    print(f"  honest pnl mean: {trades['pnl_pct_honest'].mean()*100:+.2f}%")
    print(f"  per-trade friction: slippage {SLIPPAGE_RT*100:.2f}% + avg funding "
          f"{(funding_per_bar * bars).mean()*100:+.3f}%")
    return trades


def _split(trades, side, label_col):
    sub = trades[trades["side"] == side].copy()
    sub["entry_time"] = pd.to_datetime(sub["entry_time"], utc=True)
    sub = sub[sub[label_col].notna()]
    train = sub[sub["entry_time"] < CAL_START]
    cal = sub[(sub["entry_time"] >= CAL_START) & (sub["entry_time"] < THRESHOLD_PICK_START)]
    thr_select = sub[(sub["entry_time"] >= THRESHOLD_PICK_START) & (sub["entry_time"] < HOLDOUT_START)]
    holdout = sub[sub["entry_time"] >= HOLDOUT_START]
    return train, cal, thr_select, holdout


def _train_optuna(X, y, n_trials=30):
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
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "objective":"binary","metric":"average_precision",
            "verbosity":-1,"boosting_type":"gbdt","random_state":42,"num_threads":4,
        }
        n = len(X); split = int(n * 0.8)
        clf = lgb.LGBMClassifier(**params)
        clf.fit(X[:split], y[:split], eval_set=[(X[split:], y[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])
        return float(average_precision_score(y[split:], clf.predict_proba(X[split:])[:,1]))
    study = optuna.create_study(direction="maximize",
                                 sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _train_side(trades, side, feat_cols, label_col):
    print(f"\n{_ts()} === Training honest BGM for {side.upper()}  label={label_col} ===")
    train, cal, thr_select, holdout = _split(trades, side, label_col)
    for df_part in [train, cal, thr_select, holdout]:
        for t in ["top25","26-50","51-100","101-200"]:
            df_part[f"tier_{t}"] = (df_part["tier"] == t).astype(int)
    def _clean(df_part):
        nn = df_part[V_NEW_1_FEATURES].notna().sum(axis=1)
        return df_part[nn >= 20]
    train = _clean(train); cal = _clean(cal); thr_select = _clean(thr_select); holdout = _clean(holdout)
    print(f"  train {len(train):,}  cal {len(cal):,}  thr_select {len(thr_select):,}  holdout {len(holdout):,}")
    if len(train) < 500:
        print(f"  WARN: train too small")
        return None, None, None, None, None

    X = train[feat_cols].fillna(0).values
    y = train[label_col].values.astype(int)
    print(f"  base rate: {y.mean()*100:.1f}%")
    print(f"  Optuna...")
    best = _train_optuna(X, y, n_trials=30)
    full_params = {**best, "objective":"binary","metric":"average_precision",
                    "verbosity":-1,"boosting_type":"gbdt","random_state":42,"num_threads":4}
    model = lgb.LGBMClassifier(**full_params)
    model.fit(X, y, callbacks=[lgb.log_evaluation(-1)])

    def _eval(df_part, label):
        if len(df_part)==0: return None
        Xp = df_part[feat_cols].fillna(0).values
        yp = df_part[label_col].values.astype(int)
        if len(set(yp))<2: return None
        probs = model.predict_proba(Xp)[:,1]
        return {"split":label,"n":int(len(yp)),
                 "base_rate":round(float(yp.mean()),4),
                 "pr_auc": round(float(average_precision_score(yp, probs)), 4),
                 "roc_auc": round(float(roc_auc_score(yp, probs)), 4),
                 "top_decile_precision": round(float(yp[probs >= np.percentile(probs, 90)].mean()), 4),
                 "top_5pct_precision": round(float(yp[probs >= np.percentile(probs, 95)].mean()), 4),
                 "lift_top_decile": round(float(yp[probs >= np.percentile(probs, 90)].mean() / max(yp.mean(), 0.01)), 3)}
    cal_m = _eval(cal,"cal")
    thr_m = _eval(thr_select,"thr_select_2025h2")
    hold_m = _eval(holdout,"holdout_2026q1")
    print(f"  CAL:               {cal_m}")
    print(f"  THR-SELECT 2025H2: {thr_m}")
    print(f"  HOLDOUT 2026Q1:    {hold_m}  ← TRUE OOS")
    importance = model.booster_.feature_importance(importance_type="gain")
    fi = sorted(zip(feat_cols, importance), key=lambda x:-x[1])[:10]
    print(f"  Top-10 features:")
    for n, g in fi:
        print(f"    {n:36s} gain={g:.0f}")
    md = MODEL_BASE / f"v_new_2_bgm_{side}_honest"
    md.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, md / "model.joblib")
    with open(md / "meta.json", "w") as f:
        json.dump({"side":side, "label_col":label_col,
                    "trained_at":pd.Timestamp.utcnow().isoformat(),
                    "best_params":best,
                    "cal_metrics":cal_m,
                    "thr_select_metrics":thr_m,
                    "holdout_metrics":hold_m,
                    "feature_importance_top10":[{"feature":n,"gain":float(g)} for n,g in fi]},
                   f, indent=2, default=str)
    return model, train, cal, thr_select, holdout


def _summary(pnls, leverage=1.0):
    if len(pnls)==0: return {"n":0,"wr":0,"pf":0,"sum_pnl_pct":0,"avg_win":0,"avg_loss":0}
    p = np.clip(pnls*leverage, -0.99, 100.0)
    wins = p[p>0]; losses = p[p<0]
    return {"n":int(len(p)),"wr":float((p>0).mean()),
             "sum_pnl_pct":float(p.sum()*100),
             "avg_win":float(wins.mean()*100) if len(wins) else 0,
             "avg_loss":float(losses.mean()*100) if len(losses) else 0,
             "pf":float(wins.sum()/abs(losses.sum())) if len(losses) else float("inf")}


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 Step D — HONEST evaluation per council audit")

    print(f"\n{_ts()} Loading data...")
    trades = pd.read_parquet(DATA_DIR / "v_new_2_trades_dataset.parquet")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"  trades: {len(trades):,}")
    feats_full = pd.read_parquet(DATA_DIR / "features_full.parquet",
                                   columns=["symbol","timestamp","close"])

    # Step 1: fixed-horizon labels
    trades = _add_fixed_horizon_labels(trades, feats_full)
    # Step 2: realistic friction
    trades = _add_realistic_friction(trades)

    # Meme universe
    memes = _load_meme_universe()
    trades["is_meme"] = trades["symbol"].isin(memes).astype(int)
    print(f"\n  meme trades: {trades['is_meme'].sum():,} / {len(trades):,}")

    # Train BGM with honest label (fwd_72h_pos)
    label_col = "fwd_72h_pos"
    feat_cols = V_NEW_1_FEATURES + RULE_FEATURES + TIER_DUMMIES

    long_model, lt, lc, lts, lho = _train_side(trades, "long", feat_cols, label_col)
    short_model, st, sc, sts, sho = _train_side(trades, "short", feat_cols, label_col)

    # Score everything
    print(f"\n{_ts()} Scoring all trades with honest BGM...")
    scored_parts = []
    for side, model, train, cal, ts, ho in [("long",long_model,lt,lc,lts,lho),
                                               ("short",short_model,st,sc,sts,sho)]:
        if model is None: continue
        for split_name, df_part in [("train",train),("cal",cal),("thr_select",ts),("holdout",ho)]:
            if len(df_part)==0: continue
            df_part = df_part.copy()
            df_part["bgm_honest"] = model.predict_proba(df_part[feat_cols].fillna(0).values)[:,1]
            df_part["split"] = split_name
            scored_parts.append(df_part)
    if not scored_parts:
        print(f"  ERROR: no scored data")
        return
    scored = pd.concat(scored_parts, ignore_index=True)
    scored.to_parquet(DATA_DIR / "v_new_2_trades_scored_honest.parquet", index=False)
    print(f"  saved scored ({len(scored):,} rows)")

    # Threshold sweep on honest pnl
    print(f"\n{_ts()} Honest threshold sweep — pnl_pct_honest with funding+slippage")
    rows = []
    for universe, name in [(scored, "all"),
                            (scored[scored["is_meme"]==1], "meme_only")]:
        for side in ["long","short"]:
            for tier_filter, tier_label in [(None,"all_tiers")]:
                for thr in THRESHOLDS:
                    sub = universe[(universe["side"]==side) & (universe["bgm_honest"]>=thr)]
                    sub_thr = sub[sub["split"]=="thr_select"]
                    sub_ho = sub[sub["split"]=="holdout"]
                    s_thr = _summary(sub_thr["pnl_pct_honest"].values)
                    s_ho = _summary(sub_ho["pnl_pct_honest"].values)
                    rows.append({
                        "universe":name, "side":side, "tier":tier_label, "threshold":thr,
                        "thr_select_n":s_thr["n"], "thr_pf":s_thr["pf"],
                        "thr_wr":s_thr["wr"]*100,"thr_avg_win":s_thr["avg_win"],"thr_avg_loss":s_thr["avg_loss"],
                        "thr_sum_pnl_pct":s_thr["sum_pnl_pct"],
                        "ho_n":s_ho["n"],"ho_pf":s_ho["pf"],
                        "ho_wr":s_ho["wr"]*100,"ho_avg_win":s_ho["avg_win"],"ho_avg_loss":s_ho["avg_loss"],
                        "ho_sum_pnl_pct":s_ho["sum_pnl_pct"],
                    })
    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "honest_threshold_sweep.csv", index=False)
    print(f"  saved: honest_threshold_sweep.csv")

    # Print critical comparison
    print(f"\n{_ts()} HONEST READOUT — held-out 2026 Q1 only (TRUE forward OOS)")
    print(f"  {'universe':>10s} {'side':5s} {'thr':>5s}  {'thr-2025H2':>20s}  {'holdout-2026Q1':>20s}")
    for u in ["all", "meme_only"]:
        for side in ["long","short"]:
            sub = df_out[(df_out["universe"]==u) & (df_out["side"]==side)]
            for thr in [0.0, 0.50, 0.60, 0.70]:
                r = sub[sub["threshold"]==thr]
                if len(r)==0: continue
                r = r.iloc[0]
                tag = "(baseline)" if thr==0.0 else ""
                print(f"  {u:>10s} {side:5s} {thr:>5.2f}  "
                      f"n={int(r['thr_select_n']):>4} PF={r['thr_pf']:5.2f} sum={r['thr_sum_pnl_pct']:+6.0f}%  "
                      f"n={int(r['ho_n']):>4} PF={r['ho_pf']:5.2f} sum={r['ho_sum_pnl_pct']:+6.0f}% {tag}")

    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
