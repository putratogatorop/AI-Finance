"""v_new_2 — Meme pullback × exit sweep (ATR-trail + fixed TP variants).

Builds on M5 winner (pullback 20% + pl@20% + trail 5×ATR → +184%/yr CAGR).
Now sweeps:
  Meme pullback depth: 20%, 25%, 30%
  Meme exit method:
    E1  trail_5x          (M5 baseline, ATR-based)
    E2  trail_8x          (wider ATR trail)
    E3  trail_10x         (very wide ATR — ride parabolic)
    E4  pl20_trail_5x     (M5 explicit, ATR-based)
    E5  pl30_trail_8x     (later lock + wider trail, ATR-based)
    E6  tp_50pct          (fixed take-profit at +50%)
    E7  tp_100pct         (fixed take-profit at +100%)
    E8  tp_150pct         (fixed take-profit at +150%)

Total: 3 pullback × 8 exit = 24 meme variants.
Non-memes keep default (current ATR pullback per tier + atr_trail + pl 30%).
U4 universe, 5× lev, mc=5, BGM thr 0.60.
"""
from __future__ import annotations

import json
import os
import pathlib
import time
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
MODELS = ROOT / "services" / "python" / "models"

TAKER_FEE_RT = 0.001
SLIPPAGE_RT = 0.002
FUNDING_LONG_PER_8H = 0.0002
FUNDING_SHORT_PER_8H = -0.00005
BARS_PER_8H = 2
EXTENDED_HOLD_BARS = 90 * 6
RISK_PER_TRADE = 0.01
LEVERAGE = 5.0
MAX_CONCURRENT = 5
BGM_THRESHOLD = 0.60

HOLDOUT_START = pd.Timestamp("2026-01-01", tz="UTC")
THR_SELECT_START = pd.Timestamp("2025-07-01", tz="UTC")

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
FEAT_COLS = V_NEW_1_FEATURES + RULE_FEATURES + TIER_DUMMIES

ENTRY_TIER_LONG = {"top25":("atr",2.0),"26-50":("pct",15.0),"51-100":("atr",3.0),"101-200":("atr",4.0)}
ENTRY_TIER_SHORT = {"top25":("atr",4.0),"26-50":("pct",10.0),"51-100":("atr",4.0),"101-200":("atr",4.0)}

MEME_PULLBACKS = [20.0, 25.0, 30.0]
MEME_EXITS = [
    ("E1_trail_5x",      {"trail_mult":5.0,  "pl_pct":None, "tp_pct":None, "after_pl":None}),
    ("E2_trail_8x",      {"trail_mult":8.0,  "pl_pct":None, "tp_pct":None, "after_pl":None}),
    ("E3_trail_10x",     {"trail_mult":10.0, "pl_pct":None, "tp_pct":None, "after_pl":None}),
    ("E4_pl20_trail_5x", {"trail_mult":5.0,  "pl_pct":0.20, "tp_pct":None, "after_pl":"trail"}),
    ("E5_pl30_trail_8x", {"trail_mult":8.0,  "pl_pct":0.30, "tp_pct":None, "after_pl":"trail"}),
    ("E6_tp_50pct",      {"trail_mult":3.0,  "pl_pct":None, "tp_pct":0.50, "after_pl":None}),
    ("E7_tp_100pct",     {"trail_mult":3.0,  "pl_pct":None, "tp_pct":1.00, "after_pl":None}),
    ("E8_tp_150pct",     {"trail_mult":3.0,  "pl_pct":None, "tp_pct":1.50, "after_pl":None}),
]


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _load_universe():
    with open(DATA_DIR / "coingecko_categories.json") as f:
        cats = json.load(f)
    memes = set(cats.get("memes", []))
    ai = set(cats.get("ai_tokens", []))
    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    avg_rank = mem.groupby("symbol")["rank"].mean()
    top50 = set(avg_rank[avg_rank <= 50].index)
    return memes | ai | top50, memes


def _backtest_one(g, side, pb_kind, pb_lvl, is_meme, exit_cfg, default_trail_mult=2.5):
    n = len(g)
    if n < 50: return []
    closes = g["close"].values
    timestamps = g["timestamp"].values
    atr_pct = g["atr14_pct"].values
    if side == "long":
        trend = g["d_trend_long"].values
        pb_arr = g["pullback_pct"].values if pb_kind=="pct" else g["pullback_atr"].values
    else:
        trend = g["d_trend_short"].values
        pb_arr = g["rise_pct"].values if pb_kind=="pct" else g["rise_atr"].values
    if is_meme:
        cfg = exit_cfg
    else:
        cfg = {"trail_mult":default_trail_mult, "pl_pct":0.30, "tp_pct":None, "after_pl":"trend"}

    trail_mult = cfg["trail_mult"] if cfg["trail_mult"] is not None else default_trail_mult
    pl_pct = cfg["pl_pct"]
    tp_pct = cfg["tp_pct"]
    after_pl = cfg["after_pl"]

    trades = []
    open_pos = False
    i = 1
    while i < n - 1:
        if not open_pos:
            if pd.isna(trend[i]) or trend[i] != 1: i += 1; continue
            if pd.isna(pb_arr[i]) or pb_arr[i] < pb_lvl: i += 1; continue
            prev = pb_arr[i-1]
            if not pd.isna(prev) and prev >= pb_lvl: i += 1; continue
            entry_idx = i; entry_price = closes[entry_idx]
            running_extreme = entry_price
            profit_lock_active = False
            open_pos = True; i += 1; continue

        p = closes[i]
        if pd.isna(p): i += 1; continue
        if side == "long":
            pnl_frac = p / entry_price - 1.0
            running_extreme = max(running_extreme, p)
        else:
            pnl_frac = entry_price / p - 1.0
            running_extreme = min(running_extreme, p)

        if pl_pct is not None and not profit_lock_active and pnl_frac >= pl_pct:
            profit_lock_active = True

        exit_idx = None; exit_reason = None
        # Fixed TP check first
        if tp_pct is not None and pnl_frac >= tp_pct:
            exit_idx = i; exit_reason = "fixed_tp"
        # Trend-flip exit if profit-locked and after_pl=trend
        elif profit_lock_active and after_pl == "trend":
            if not pd.isna(trend[i]) and trend[i] != 1:
                exit_idx = i; exit_reason = "trend_flip_locked"
        else:
            # ATR trail
            a = atr_pct[i] / 100.0 if not pd.isna(atr_pct[i]) else 0.02
            if side == "long":
                if p <= running_extreme * (1 - trail_mult * a):
                    exit_idx = i; exit_reason = "atr_trail"
            else:
                if p >= running_extreme * (1 + trail_mult * a):
                    exit_idx = i; exit_reason = "atr_trail"

        if exit_idx is None and i - entry_idx >= EXTENDED_HOLD_BARS:
            exit_idx = i; exit_reason = "timeout"

        if exit_idx is not None:
            exit_price = closes[exit_idx]
            pnl = (exit_price/entry_price - 1.0) if side == "long" else (entry_price/exit_price - 1.0)
            bars_held = exit_idx - entry_idx
            fund = (FUNDING_LONG_PER_8H if side=="long" else FUNDING_SHORT_PER_8H) / BARS_PER_8H * bars_held
            pnl_honest = pnl - SLIPPAGE_RT - TAKER_FEE_RT - fund
            trades.append({
                "symbol": g["symbol"].iloc[0], "tier": g["mcap_tier"].iloc[0],
                "side": side, "is_meme": is_meme,
                "entry_time": pd.Timestamp(timestamps[entry_idx]),
                "exit_time": pd.Timestamp(timestamps[exit_idx]),
                "bars_held": int(bars_held), "exit_reason": exit_reason,
                "pnl_pct": float(pnl_honest),
            })
            open_pos = False; i = exit_idx + 1; continue
        i += 1
    return trades


def _score(tdf, base, ml, ms):
    if len(tdf)==0: return tdf
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    bf = base.rename(columns={"timestamp":"entry_time"})
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True)
    if tdf["entry_time"].dt.tz is None:
        tdf["entry_time"] = tdf["entry_time"].dt.tz_localize("UTC")
    merged = tdf.merge(bf, on=["symbol","entry_time"], how="left")
    for t in ["top25","26-50","51-100","101-200"]:
        merged[f"tier_{t}"] = (merged["tier"] == t).astype(int)
    bgm = np.zeros(len(merged))
    lm = (merged["side"]=="long").values; sm = ~lm
    if lm.sum()>0:
        bgm[lm] = ml.predict_proba(merged.loc[lm, FEAT_COLS].fillna(0).values)[:,1]
    if sm.sum()>0:
        bgm[sm] = ms.predict_proba(merged.loc[sm, FEAT_COLS].fillna(0).values)[:,1]
    merged["bgm_score"] = bgm
    return merged


@dataclass
class OpenP:
    symbol: str; entry_time: pd.Timestamp; exit_time: pd.Timestamp; pnl_pct: float


def _portfolio_sim(trades_df, leverage=LEVERAGE, max_concurrent=MAX_CONCURRENT, risk=RISK_PER_TRADE):
    if len(trades_df)==0:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":0}
    df = trades_df.sort_values("entry_time").reset_index(drop=True).copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    if df["entry_time"].dt.tz is None: df["entry_time"] = df["entry_time"].dt.tz_localize("UTC")
    if df["exit_time"].dt.tz is None: df["exit_time"] = df["exit_time"].dt.tz_localize("UTC")
    eq = 1.0; open_p: list[OpenP] = []; curve = []; n_taken = 0
    for _, row in df.iterrows():
        et, xt = row["entry_time"], row["exit_time"]
        if pd.isna(et) or pd.isna(xt): continue
        new_open = []
        for pos in open_p:
            if pos.exit_time <= et:
                eq *= (1.0 + pos.pnl_pct * risk * leverage); eq = max(eq, 1e-6)
                curve.append((pos.exit_time, eq))
            else: new_open.append(pos)
        open_p = new_open
        if len(open_p) < max_concurrent:
            open_p.append(OpenP(row["symbol"], et, xt, row["pnl_pct"])); n_taken += 1
    for pos in open_p:
        eq *= (1.0 + pos.pnl_pct * risk * leverage); eq = max(eq, 1e-6)
        curve.append((pos.exit_time, eq))
    if not curve:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":n_taken}
    eq_df = pd.DataFrame(curve, columns=["t","equity"]).sort_values("t")
    arr = eq_df["equity"].values
    rm = np.maximum.accumulate(arr); safe = np.where(rm>1e-10, rm, 1e-10)
    dd = (arr - rm)/safe
    return {"n":int(len(df)),"n_taken":n_taken,"final_eq":float(arr[-1]),
             "compd_pct":float((arr[-1]-1)*100),"max_dd_pct":float(dd.min()*100)}


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 meme pullback × exit sweep on U4")
    u4, memes = _load_universe()
    df = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    base = pd.read_parquet(DATA_DIR / "features_full.parquet",
                            columns=["symbol","timestamp"] + V_NEW_1_FEATURES)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    rule = df[["symbol","timestamp"] + RULE_FEATURES].copy()
    base_full = base.merge(rule, on=["symbol","timestamp"], how="left")
    ml = joblib.load(MODELS / "v_new_2_bgm_long_honest" / "model.joblib")
    ms = joblib.load(MODELS / "v_new_2_bgm_short_honest" / "model.joblib")

    rows = []
    for pb_meme in MEME_PULLBACKS:
        for ex_name, ex_cfg in MEME_EXITS:
            t_v = time.time()
            all_trades = []
            for side, scheme in [("long", ENTRY_TIER_LONG), ("short", ENTRY_TIER_SHORT)]:
                for tier, (pb_kind, pb_lvl) in scheme.items():
                    tier_syms = df[(df["mcap_tier"]==tier) & (df["symbol"].isin(u4))]["symbol"].unique()
                    for sym in tier_syms:
                        is_meme = sym in memes
                        pb_kind_use, pb_lvl_use = (("pct", pb_meme) if is_meme else (pb_kind, pb_lvl))
                        g = df[df["symbol"]==sym].reset_index(drop=True)
                        if len(g) < 50: continue
                        trades = _backtest_one(g, side, pb_kind_use, pb_lvl_use, is_meme, ex_cfg)
                        all_trades.extend(trades)

            tdf = pd.DataFrame(all_trades)
            scored = _score(tdf, base_full, ml, ms)
            filtered = scored[scored["bgm_score"] >= BGM_THRESHOLD]
            mf = filtered[filtered["is_meme"]==True]

            f_thr = filtered[(filtered["entry_time"] >= THR_SELECT_START) &
                               (filtered["entry_time"] < HOLDOUT_START)]
            f_ho = filtered[filtered["entry_time"] >= HOLDOUT_START]
            sim_thr = _portfolio_sim(f_thr); sim_ho = _portfolio_sim(f_ho)
            thr_days = 184; ho_days = 90
            thr_cagr = ((sim_thr["final_eq"]**(365/thr_days))-1)*100 if sim_thr["final_eq"]>0 else -100
            ho_cagr = ((sim_ho["final_eq"]**(365/ho_days))-1)*100 if sim_ho["final_eq"]>0 else -100

            meme_max_win = mf["pnl_pct"].max()*100 if len(mf)>0 else 0
            meme_avg = mf["pnl_pct"].mean()*100 if len(mf)>0 else 0
            meme_wr = (mf["pnl_pct"]>0).mean()*100 if len(mf)>0 else 0

            rows.append({
                "pb": pb_meme, "exit": ex_name,
                "n_filt": len(filtered), "n_meme": len(mf),
                "meme_wr": meme_wr, "meme_max_win": meme_max_win, "meme_avg_pnl": meme_avg,
                "thr_compd": sim_thr["compd_pct"], "thr_dd": sim_thr["max_dd_pct"],
                "ho_compd": sim_ho["compd_pct"], "ho_dd": sim_ho["max_dd_pct"],
                "ho_cagr": ho_cagr,
            })
            print(f"  pb={pb_meme:>4.0f}% {ex_name:18s}  meme={len(mf):>3} wr={meme_wr:>4.0f}% "
                  f"max_win={meme_max_win:>+5.0f}% avg_pnl={meme_avg:>+5.1f}%  "
                  f"H2={sim_thr['compd_pct']:>+5.1f}% Q1={sim_ho['compd_pct']:>+5.1f}%  "
                  f"Q1_DD={sim_ho['max_dd_pct']:>+5.1f}%  Q1_CAGR={ho_cagr:>+4.0f}%/yr  ({time.time()-t_v:.1f}s)")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "meme_pullback_exit_sweep.csv", index=False)

    # Verdict — top 5 by Q1 CAGR
    md = []
    md.append(f"# v_new_2 — Meme pullback × exit sweep ({time.strftime('%Y-%m-%d')})\n\n")
    md.append("U4 universe, Scheme C entries (current ATR + meme pullback override). "
              "5× lev, mc=5, BGM thr 0.60. Honest funding+slippage.\n\n")
    md.append("## All variants (sorted by Q1 CAGR)\n\n")
    md.append("| pullback | exit | n_meme | meme_wr | meme_max_win | meme_avg | H2 compd | Q1 compd | Q1 DD | Q1 CAGR |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in df_out.sort_values("ho_cagr", ascending=False).iterrows():
        md.append(f"| {r['pb']:.0f}% | {r['exit']} | {int(r['n_meme'])} | {r['meme_wr']:.0f}% | "
                  f"{r['meme_max_win']:+.0f}% | {r['meme_avg_pnl']:+.1f}% | "
                  f"{r['thr_compd']:+.1f}% | {r['ho_compd']:+.1f}% | {r['ho_dd']:+.1f}% | "
                  f"{r['ho_cagr']:+.0f}%/yr |\n")
    (RESULTS / "meme_pullback_exit_verdict.md").write_text("".join(md))
    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
