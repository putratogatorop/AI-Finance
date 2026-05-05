"""v_new_2 — Test deeper tier-specific absolute pullbacks vs current ATR scheme.

Compare:
  Scheme A (current): ATR-based per tier (top25=2×ATR, 26-50=15%, 51-100=3×ATR, 101-200=4×ATR)
  Scheme B (deeper):  Absolute % per tier (top25=5%, 26-50=10%, 51-100=15%, 101-200=20%)
  Scheme C (meme-aware): same as B but memes specifically use 20% regardless of tier

For each scheme:
  1. Re-detect rule-trigger entries on full features dataset
  2. Score each entry with existing BGM honest models (re-load features at entry bar)
  3. Run portfolio sim on U4 universe at 5× lev / mc=5 / threshold 0.60
  4. Compare CAGR + DD on held-out 2026 Q1

Output:
  results/v_new_2/deeper_pullback_test.csv
  results/v_new_2/deeper_pullback_verdict.md
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

# Three schemes
SCHEME_A_CURRENT = {  # ATR-based (current)
    "top25":   ("atr",  2.0),
    "26-50":   ("pct", 15.0),
    "51-100":  ("atr",  3.0),
    "101-200": ("atr",  4.0),
}
SCHEME_A_CURRENT_SHORT = {
    "top25":   ("atr",  4.0),
    "26-50":   ("pct", 10.0),
    "51-100":  ("atr",  4.0),
    "101-200": ("atr",  4.0),
}
SCHEME_B_DEEPER = {  # Absolute % per tier, deeper
    "top25":   ("pct",  5.0),
    "26-50":   ("pct", 10.0),
    "51-100":  ("pct", 15.0),
    "101-200": ("pct", 20.0),
}


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _load_u4_universe():
    with open(DATA_DIR / "coingecko_categories.json") as f:
        cats = json.load(f)
    memes = set(cats.get("memes", []))
    ai = set(cats.get("ai_tokens", []))
    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    avg_rank = mem.groupby("symbol")["rank"].mean()
    top50 = set(avg_rank[avg_rank <= 50].index)
    return memes | ai | top50, memes


def _backtest_one(g, side, pb_kind, pb_lvl, exit_method="atr_trail_2",
                    profit_lock_pct=0.30):
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

    trades = []
    open_pos = False
    i = 1
    while i < n - 1:
        if not open_pos:
            if pd.isna(trend[i]) or trend[i] != 1: i += 1; continue
            if pd.isna(pb_arr[i]) or pb_arr[i] < pb_lvl: i += 1; continue
            prev = pb_arr[i-1]
            if not pd.isna(prev) and prev >= pb_lvl: i += 1; continue
            entry_idx = i
            entry_price = closes[entry_idx]
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
        if profit_lock_pct is not None and not profit_lock_active and pnl_frac >= profit_lock_pct:
            profit_lock_active = True
        exit_idx = None; exit_reason = None
        if profit_lock_active:
            if not pd.isna(trend[i]) and trend[i] != 1:
                exit_idx = i; exit_reason = "trend_flip_locked"
        else:
            mult = 2.0 if exit_method == "atr_trail_2" else 3.0
            a = atr_pct[i] / 100.0 if not pd.isna(atr_pct[i]) else 0.02
            if side == "long":
                if p <= running_extreme * (1 - mult * a):
                    exit_idx = i; exit_reason = "atr_trail"
            else:
                if p >= running_extreme * (1 + mult * a):
                    exit_idx = i; exit_reason = "atr_trail"
        if exit_idx is None and i - entry_idx >= EXTENDED_HOLD_BARS:
            exit_idx = i; exit_reason = "timeout"
        if exit_idx is not None:
            exit_price = closes[exit_idx]
            if side == "long":
                pnl = exit_price/entry_price - 1.0
            else:
                pnl = entry_price/exit_price - 1.0
            bars_held = exit_idx - entry_idx
            fund_per_bar = (FUNDING_LONG_PER_8H if side=="long" else FUNDING_SHORT_PER_8H) / BARS_PER_8H
            funding_total = fund_per_bar * bars_held
            pnl_honest = pnl - SLIPPAGE_RT - TAKER_FEE_RT - funding_total
            trades.append({
                "symbol": g["symbol"].iloc[0],
                "tier": g["mcap_tier"].iloc[0],
                "side": side,
                "entry_time": pd.Timestamp(timestamps[entry_idx]),
                "exit_time": pd.Timestamp(timestamps[exit_idx]),
                "bars_held": int(bars_held),
                "exit_reason": exit_reason,
                "pnl_pct": float(pnl_honest),
                "entry_idx": int(entry_idx),
            })
            open_pos = False
            i = exit_idx + 1
            continue
        i += 1
    return trades


def _score_trades_with_bgm(trades_df, base_features, model_long, model_short, feat_cols):
    """Look up base features at each trade's entry_time, build feature vector,
    score with BGM."""
    if len(trades_df) == 0: return trades_df
    # Merge with base_features on (symbol, entry_time)
    base_features["timestamp"] = pd.to_datetime(base_features["timestamp"], utc=True)
    bf = base_features.rename(columns={"timestamp":"entry_time"})
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
    if trades_df["entry_time"].dt.tz is None:
        trades_df["entry_time"] = trades_df["entry_time"].dt.tz_localize("UTC")
    merged = trades_df.merge(bf, on=["symbol","entry_time"], how="left")
    # Add tier dummies
    for t in ["top25","26-50","51-100","101-200"]:
        merged[f"tier_{t}"] = (merged["tier"] == t).astype(int)
    # Score per side
    bgm_scores = np.zeros(len(merged))
    long_mask = (merged["side"] == "long").values
    short_mask = ~long_mask
    if long_mask.sum() > 0:
        X_long = merged.loc[long_mask, feat_cols].fillna(0).values
        bgm_scores[long_mask] = model_long.predict_proba(X_long)[:, 1]
    if short_mask.sum() > 0:
        X_short = merged.loc[short_mask, feat_cols].fillna(0).values
        bgm_scores[short_mask] = model_short.predict_proba(X_short)[:, 1]
    merged["bgm_score"] = bgm_scores
    return merged


@dataclass
class OpenPos:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl_pct: float


def _portfolio_sim(trades_df, leverage=LEVERAGE, max_concurrent=MAX_CONCURRENT,
                    risk=RISK_PER_TRADE):
    if len(trades_df)==0:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":0}
    df = trades_df.sort_values("entry_time").reset_index(drop=True).copy()
    # Ensure both entry_time and exit_time are tz-aware UTC
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["exit_time"] = pd.to_datetime(df["exit_time"], utc=True)
    if df["entry_time"].dt.tz is None:
        df["entry_time"] = df["entry_time"].dt.tz_localize("UTC")
    if df["exit_time"].dt.tz is None:
        df["exit_time"] = df["exit_time"].dt.tz_localize("UTC")
    eq = 1.0
    open_p: list[OpenPos] = []
    curve = []
    n_taken = 0
    for _, row in df.iterrows():
        et = row["entry_time"]; xt = row["exit_time"]
        if pd.isna(et) or pd.isna(xt): continue
        new_open = []
        for pos in open_p:
            if pos.exit_time <= et:
                eq *= (1.0 + pos.pnl_pct * risk * leverage)
                eq = max(eq, 1e-6)
                curve.append((pos.exit_time, eq))
            else:
                new_open.append(pos)
        open_p = new_open
        if len(open_p) < max_concurrent:
            open_p.append(OpenPos(row["symbol"], et, xt, row["pnl_pct"]))
            n_taken += 1
    for pos in open_p:
        eq *= (1.0 + pos.pnl_pct * risk * leverage)
        eq = max(eq, 1e-6)
        curve.append((pos.exit_time, eq))
    if not curve:
        return {"n":0,"final_eq":1.0,"max_dd_pct":0,"compd_pct":0,"n_taken":n_taken}
    eq_df = pd.DataFrame(curve, columns=["t","equity"]).sort_values("t")
    arr = eq_df["equity"].values
    rm = np.maximum.accumulate(arr)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (arr - rm)/safe
    return {
        "n":int(len(df)),"n_taken":n_taken,"final_eq":float(arr[-1]),
        "compd_pct":float((arr[-1]-1)*100),
        "max_dd_pct":float(dd.min()*100),
    }


def _run_scheme(scheme_long, scheme_short, df, u4, base_features, model_long, model_short,
                  meme_override=None, name="scheme"):
    """Run one scheme and return per-tier + aggregate metrics."""
    print(f"\n{_ts()} === {name} ===")
    all_trades = []
    for side, scheme in [("long", scheme_long), ("short", scheme_short)]:
        for tier, (pb_kind, pb_lvl) in scheme.items():
            tier_syms = df[(df["mcap_tier"]==tier) & (df["symbol"].isin(u4))]["symbol"].unique()
            for sym in tier_syms:
                # meme override: if symbol is meme, use deeper pullback
                if meme_override is not None and sym in meme_override["memes"]:
                    pb_kind_use, pb_lvl_use = meme_override["pullback"]
                else:
                    pb_kind_use, pb_lvl_use = pb_kind, pb_lvl
                g = df[df["symbol"]==sym].reset_index(drop=True)
                if len(g) < 50: continue
                trades = _backtest_one(g, side, pb_kind_use, pb_lvl_use)
                all_trades.extend(trades)
    tdf = pd.DataFrame(all_trades)
    print(f"  raw trades: {len(tdf):,}")

    # Score with BGM
    scored = _score_trades_with_bgm(tdf, base_features, model_long, model_short, FEAT_COLS)
    print(f"  scored trades: {len(scored):,}  median bgm: {scored['bgm_score'].median():.3f}")

    # Filter by threshold
    filtered = scored[scored["bgm_score"] >= BGM_THRESHOLD]
    print(f"  after BGM threshold {BGM_THRESHOLD}: {len(filtered):,}")

    # Splits
    f_thr = filtered[(filtered["entry_time"] >= THR_SELECT_START) &
                       (filtered["entry_time"] < HOLDOUT_START)]
    f_ho = filtered[filtered["entry_time"] >= HOLDOUT_START]

    sim_thr = _portfolio_sim(f_thr)
    sim_ho = _portfolio_sim(f_ho)

    thr_days = 184; ho_days = 90
    thr_cagr = ((sim_thr["final_eq"] ** (365/thr_days)) - 1) * 100 if sim_thr["final_eq"] > 0 else -100
    ho_cagr = ((sim_ho["final_eq"] ** (365/ho_days)) - 1) * 100 if sim_ho["final_eq"] > 0 else -100

    print(f"  H2 (2025-07→12): n_taken={sim_thr['n_taken']}  compd={sim_thr['compd_pct']:+.1f}%  "
          f"DD={sim_thr['max_dd_pct']:+.1f}%  CAGR={thr_cagr:+.0f}%/yr")
    print(f"  Q1 (2026-01→03): n_taken={sim_ho['n_taken']}  compd={sim_ho['compd_pct']:+.1f}%  "
          f"DD={sim_ho['max_dd_pct']:+.1f}%  CAGR={ho_cagr:+.0f}%/yr")
    return {
        "scheme": name,
        "n_raw": len(tdf), "n_filtered": len(filtered),
        "thr_n_taken": sim_thr["n_taken"], "thr_compd_pct": sim_thr["compd_pct"],
        "thr_max_dd_pct": sim_thr["max_dd_pct"], "thr_cagr_pct": thr_cagr,
        "ho_n_taken": sim_ho["n_taken"], "ho_compd_pct": sim_ho["compd_pct"],
        "ho_max_dd_pct": sim_ho["max_dd_pct"], "ho_cagr_pct": ho_cagr,
    }


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 deeper-pullback test on U4")
    u4, memes = _load_u4_universe()
    print(f"  U4 universe: {len(u4)} syms, of which {len(memes & u4)} memes")

    df = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"  loaded features: {len(df):,} rows")

    print(f"  loading 44 base features for BGM scoring...")
    base = pd.read_parquet(DATA_DIR / "features_full.parquet",
                            columns=["symbol","timestamp"] + V_NEW_1_FEATURES)
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    rule_feats = df[["symbol","timestamp"] + RULE_FEATURES].copy()
    base_full = base.merge(rule_feats, on=["symbol","timestamp"], how="left")
    print(f"  base features: {len(base_full):,}")

    print(f"  loading BGM models...")
    model_long = joblib.load(MODELS / "v_new_2_bgm_long_honest" / "model.joblib")
    model_short = joblib.load(MODELS / "v_new_2_bgm_short_honest" / "model.joblib")

    rows = []
    rows.append(_run_scheme(SCHEME_A_CURRENT, SCHEME_A_CURRENT_SHORT,
                              df, u4, base_full, model_long, model_short,
                              name="A_current_ATR"))
    rows.append(_run_scheme(SCHEME_B_DEEPER, SCHEME_B_DEEPER,
                              df, u4, base_full, model_long, model_short,
                              name="B_deeper_pct"))
    rows.append(_run_scheme(SCHEME_A_CURRENT, SCHEME_A_CURRENT_SHORT,
                              df, u4, base_full, model_long, model_short,
                              meme_override={"memes": memes,
                                              "pullback": ("pct", 20.0)},
                              name="C_current_meme20pct"))

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "deeper_pullback_test.csv", index=False)

    md = []
    md.append(f"# v_new_2 — Deeper Pullback Test ({time.strftime('%Y-%m-%d')})\n\n")
    md.append("U4 universe (memes + AI + top-50). 5× lev, mc=5, 1% risk, BGM thr 0.60.\n")
    md.append("Honest: funding+slippage+fees deducted.\n\n")
    md.append("| scheme | raw | filtered | H2 taken | H2 compd | H2 DD | Q1 taken | Q1 compd | Q1 DD | Q1 CAGR |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in df_out.iterrows():
        md.append(f"| {r['scheme']} | {int(r['n_raw'])} | {int(r['n_filtered'])} | "
                  f"{int(r['thr_n_taken'])} | {r['thr_compd_pct']:+.1f}% | {r['thr_max_dd_pct']:+.1f}% | "
                  f"{int(r['ho_n_taken'])} | {r['ho_compd_pct']:+.1f}% | {r['ho_max_dd_pct']:+.1f}% | "
                  f"{r['ho_cagr_pct']:+.0f}%/yr |\n")
    (RESULTS / "deeper_pullback_verdict.md").write_text("".join(md))
    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
