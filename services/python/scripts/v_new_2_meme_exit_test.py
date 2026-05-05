"""v_new_2 — Test meme-specific exit variants (TP/SL) on Scheme C entries.

Scheme C from prior test wins on Q1 holdout (+73%/yr CAGR, -2.1% DD):
  - Non-memes use current ATR-based pullback per tier
  - Memes use 20% absolute pullback override
  - Default exit for ALL: ATR-trail 2-3× + profit-lock at +30% → trend-flip-only

Now testing whether MEMES specifically benefit from wider/different exit logic
(since memes are "wild" with fat-tail volatility):

Variants (memes only — non-memes keep default):
  M0  default          (current: atr_trail 2-3x + pl 30%)
  M1  pl_50_trend      (later profit-lock at +50%, then trend-flip-only)
  M2  pl_20_trend      (earlier profit-lock at +20%)
  M3  trail_5x         (wider trail: 5x ATR throughout)
  M4  trail_8x         (very wide: 8x ATR)
  M5  pl_20_trail_5x   (combo: pl at +20%, after that 5x ATR until trend flip)
  M6  trend_only       (pure trend-flip-only from entry — let parabolic memes ride)
  M7  hard_sl_25       (default, but additional hard SL at -25% for catastrophic protection)

Run on U4 universe, 5× lev, mc=5, BGM thr 0.60, Scheme C entry rules.

Output:
  results/v_new_2/meme_exit_test.csv
  results/v_new_2/meme_exit_verdict.md
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

# Scheme C entries (winner from prior test)
ENTRY_TIER_LONG = {"top25":("atr",2.0),"26-50":("pct",15.0),"51-100":("atr",3.0),"101-200":("atr",4.0)}
ENTRY_TIER_SHORT = {"top25":("atr",4.0),"26-50":("pct",10.0),"51-100":("atr",4.0),"101-200":("atr",4.0)}
MEME_ENTRY_OVERRIDE = ("pct", 20.0)   # both sides

# Meme exit variants
MEME_EXITS = [
    ("M0_default",       {"trail_mult":None, "pl_pct":0.30, "after_pl":"trend", "hard_sl":None}),
    ("M1_pl_50_trend",   {"trail_mult":None, "pl_pct":0.50, "after_pl":"trend", "hard_sl":None}),
    ("M2_pl_20_trend",   {"trail_mult":None, "pl_pct":0.20, "after_pl":"trend", "hard_sl":None}),
    ("M3_trail_5x",      {"trail_mult":5.0,  "pl_pct":None, "after_pl":None,    "hard_sl":None}),
    ("M4_trail_8x",      {"trail_mult":8.0,  "pl_pct":None, "after_pl":None,    "hard_sl":None}),
    ("M5_pl_20_trail_5x",{"trail_mult":5.0,  "pl_pct":0.20, "after_pl":"trail", "hard_sl":None}),
    ("M6_trend_only",    {"trail_mult":None, "pl_pct":0.00, "after_pl":"trend", "hard_sl":None}),
    ("M7_hard_sl_25",    {"trail_mult":None, "pl_pct":0.30, "after_pl":"trend", "hard_sl":-0.25}),
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


def _backtest_one(g, side, pb_kind, pb_lvl, is_meme, meme_exit_cfg,
                    default_trail_mult=2.5):
    """Backtest one symbol's series.
       For non-memes: default exit logic (atr_trail + pl 30% → trend).
       For memes: use meme_exit_cfg if provided, else default."""
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

    # Choose exit config
    if is_meme:
        cfg = meme_exit_cfg
    else:
        cfg = {"trail_mult":None, "pl_pct":0.30, "after_pl":"trend", "hard_sl":None}

    trail_mult = cfg["trail_mult"] if cfg["trail_mult"] is not None else default_trail_mult
    pl_pct = cfg["pl_pct"]
    after_pl = cfg["after_pl"]   # "trend" | "trail" | None
    hard_sl = cfg["hard_sl"]

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
            open_pos = True
            i += 1
            continue

        p = closes[i]
        if pd.isna(p): i += 1; continue
        if side == "long":
            pnl_frac = p / entry_price - 1.0
            running_extreme = max(running_extreme, p)
        else:
            pnl_frac = entry_price / p - 1.0
            running_extreme = min(running_extreme, p)

        # Hard SL check first (catastrophic protection)
        if hard_sl is not None and pnl_frac <= hard_sl:
            exit_idx = i; exit_reason = "hard_sl"
            # Close
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

        # Profit-lock activation
        if pl_pct is not None and not profit_lock_active and pnl_frac >= pl_pct:
            profit_lock_active = True

        # Exit decision
        exit_idx = None; exit_reason = None
        if profit_lock_active and after_pl == "trend":
            if not pd.isna(trend[i]) and trend[i] != 1:
                exit_idx = i; exit_reason = "trend_flip_locked"
        else:
            # Trail-stop logic (default OR after_pl="trail")
            mult_to_use = trail_mult
            a = atr_pct[i] / 100.0 if not pd.isna(atr_pct[i]) else 0.02
            if side == "long":
                if p <= running_extreme * (1 - mult_to_use * a):
                    exit_idx = i; exit_reason = "atr_trail"
            else:
                if p >= running_extreme * (1 + mult_to_use * a):
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


def _score_trades(tdf, base_features, ml, ms, feat_cols):
    if len(tdf)==0: return tdf
    base_features["timestamp"] = pd.to_datetime(base_features["timestamp"], utc=True)
    bf = base_features.rename(columns={"timestamp":"entry_time"})
    tdf["entry_time"] = pd.to_datetime(tdf["entry_time"], utc=True)
    if tdf["entry_time"].dt.tz is None:
        tdf["entry_time"] = tdf["entry_time"].dt.tz_localize("UTC")
    merged = tdf.merge(bf, on=["symbol","entry_time"], how="left")
    for t in ["top25","26-50","51-100","101-200"]:
        merged[f"tier_{t}"] = (merged["tier"] == t).astype(int)
    bgm = np.zeros(len(merged))
    lm = (merged["side"]=="long").values
    sm = ~lm
    if lm.sum()>0:
        Xl = merged.loc[lm, feat_cols].fillna(0).values
        bgm[lm] = ml.predict_proba(Xl)[:,1]
    if sm.sum()>0:
        Xs = merged.loc[sm, feat_cols].fillna(0).values
        bgm[sm] = ms.predict_proba(Xs)[:,1]
    merged["bgm_score"] = bgm
    return merged


@dataclass
class OpenP:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl_pct: float


def _portfolio_sim(trades_df, leverage=LEVERAGE, max_concurrent=MAX_CONCURRENT,
                    risk=RISK_PER_TRADE):
    if len(trades_df)==0:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":0}
    df = trades_df.sort_values("entry_time").reset_index(drop=True).copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    if df["entry_time"].dt.tz is None: df["entry_time"] = df["entry_time"].dt.tz_localize("UTC")
    if df["exit_time"].dt.tz is None: df["exit_time"] = df["exit_time"].dt.tz_localize("UTC")
    eq = 1.0
    open_p: list[OpenP] = []
    curve = []
    n_taken = 0
    for _, row in df.iterrows():
        et = row["entry_time"]; xt = row["exit_time"]
        if pd.isna(et) or pd.isna(xt): continue
        new_open = []
        for pos in open_p:
            if pos.exit_time <= et:
                eq *= (1.0 + pos.pnl_pct * risk * leverage)
                eq = max(eq, 1e-6); curve.append((pos.exit_time, eq))
            else:
                new_open.append(pos)
        open_p = new_open
        if len(open_p) < max_concurrent:
            open_p.append(OpenP(row["symbol"], et, xt, row["pnl_pct"]))
            n_taken += 1
    for pos in open_p:
        eq *= (1.0 + pos.pnl_pct * risk * leverage)
        eq = max(eq, 1e-6); curve.append((pos.exit_time, eq))
    if not curve:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":n_taken}
    eq_df = pd.DataFrame(curve, columns=["t","equity"]).sort_values("t")
    arr = eq_df["equity"].values
    rm = np.maximum.accumulate(arr)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (arr - rm)/safe
    return {"n":int(len(df)),"n_taken":n_taken,"final_eq":float(arr[-1]),
             "compd_pct":float((arr[-1]-1)*100),"max_dd_pct":float(dd.min()*100)}


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 meme exit-variant test (Scheme C entries + meme-specific exits)")
    u4, memes = _load_universe()
    print(f"  U4 universe: {len(u4)} syms, memes within: {len(memes & u4)}")

    df = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    base = pd.read_parquet(DATA_DIR / "features_full.parquet",
                            columns=["symbol","timestamp"] + V_NEW_1_FEATURES)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    rule_feats = df[["symbol","timestamp"] + RULE_FEATURES].copy()
    base_full = base.merge(rule_feats, on=["symbol","timestamp"], how="left")

    ml = joblib.load(MODELS / "v_new_2_bgm_long_honest" / "model.joblib")
    ms = joblib.load(MODELS / "v_new_2_bgm_short_honest" / "model.joblib")

    rows = []
    for variant_name, meme_cfg in MEME_EXITS:
        t_v = time.time()
        all_trades = []
        for side, scheme in [("long", ENTRY_TIER_LONG), ("short", ENTRY_TIER_SHORT)]:
            for tier, (pb_kind, pb_lvl) in scheme.items():
                tier_syms = df[(df["mcap_tier"]==tier) & (df["symbol"].isin(u4))]["symbol"].unique()
                for sym in tier_syms:
                    is_meme = sym in memes
                    if is_meme:
                        pb_kind_use, pb_lvl_use = MEME_ENTRY_OVERRIDE
                    else:
                        pb_kind_use, pb_lvl_use = pb_kind, pb_lvl
                    g = df[df["symbol"]==sym].reset_index(drop=True)
                    if len(g) < 50: continue
                    trades = _backtest_one(g, side, pb_kind_use, pb_lvl_use, is_meme, meme_cfg)
                    all_trades.extend(trades)

        tdf = pd.DataFrame(all_trades)
        scored = _score_trades(tdf, base_full, ml, ms, FEAT_COLS)
        filtered = scored[scored["bgm_score"] >= BGM_THRESHOLD]
        meme_filtered = filtered[filtered["is_meme"]==True]

        # Splits
        f_thr = filtered[(filtered["entry_time"] >= THR_SELECT_START) &
                           (filtered["entry_time"] < HOLDOUT_START)]
        f_ho = filtered[filtered["entry_time"] >= HOLDOUT_START]
        # Meme-only splits
        m_thr = meme_filtered[(meme_filtered["entry_time"] >= THR_SELECT_START) &
                                  (meme_filtered["entry_time"] < HOLDOUT_START)]
        m_ho = meme_filtered[meme_filtered["entry_time"] >= HOLDOUT_START]

        sim_thr = _portfolio_sim(f_thr)
        sim_ho = _portfolio_sim(f_ho)

        thr_days = 184; ho_days = 90
        thr_cagr = ((sim_thr["final_eq"] ** (365/thr_days)) - 1) * 100 if sim_thr["final_eq"] > 0 else -100
        ho_cagr = ((sim_ho["final_eq"] ** (365/ho_days)) - 1) * 100 if sim_ho["final_eq"] > 0 else -100

        # Meme-specific stats
        if len(meme_filtered) > 0:
            meme_max_win = meme_filtered["pnl_pct"].max() * 100
            meme_avg_win = meme_filtered.loc[meme_filtered["pnl_pct"]>0, "pnl_pct"].mean() * 100 if (meme_filtered["pnl_pct"]>0).any() else 0
        else:
            meme_max_win = 0; meme_avg_win = 0

        rows.append({
            "variant": variant_name,
            "n_total": len(filtered), "n_meme": len(meme_filtered),
            "thr_n_taken": sim_thr["n_taken"], "thr_compd_pct": sim_thr["compd_pct"],
            "thr_max_dd_pct": sim_thr["max_dd_pct"], "thr_cagr_pct": thr_cagr,
            "ho_n_taken": sim_ho["n_taken"], "ho_compd_pct": sim_ho["compd_pct"],
            "ho_max_dd_pct": sim_ho["max_dd_pct"], "ho_cagr_pct": ho_cagr,
            "meme_max_win": meme_max_win, "meme_avg_win": meme_avg_win,
        })
        print(f"  {variant_name:22s}  total_filt={len(filtered):>4}  memes={len(meme_filtered):>3}  "
              f"H2_compd={sim_thr['compd_pct']:>+5.1f}%  H2_DD={sim_thr['max_dd_pct']:>+5.1f}%  "
              f"Q1_compd={sim_ho['compd_pct']:>+5.1f}%  Q1_DD={sim_ho['max_dd_pct']:>+5.1f}%  "
              f"Q1_CAGR={ho_cagr:>+4.0f}%/yr  meme_max_win={meme_max_win:>+5.0f}%  ({time.time()-t_v:.1f}s)")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "meme_exit_test.csv", index=False)

    md = []
    md.append(f"# v_new_2 — Meme Exit Variants Test ({time.strftime('%Y-%m-%d')})\n\n")
    md.append("U4 universe + Scheme C entries (current ATR + meme 20% override). "
              "5× lev, mc=5, BGM thr 0.60. Memes use variant exit; non-memes default.\n\n")
    md.append("| variant | n_filt | n_meme | H2 compd | H2 DD | Q1 compd | Q1 DD | Q1 CAGR | meme max_win | meme avg_win |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in df_out.iterrows():
        md.append(f"| {r['variant']} | {int(r['n_total'])} | {int(r['n_meme'])} | "
                  f"{r['thr_compd_pct']:+.1f}% | {r['thr_max_dd_pct']:+.1f}% | "
                  f"{r['ho_compd_pct']:+.1f}% | {r['ho_max_dd_pct']:+.1f}% | "
                  f"{r['ho_cagr_pct']:+.0f}%/yr | {r['meme_max_win']:+.0f}% | {r['meme_avg_win']:+.1f}% |\n")
    (RESULTS / "meme_exit_verdict.md").write_text("".join(md))
    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
