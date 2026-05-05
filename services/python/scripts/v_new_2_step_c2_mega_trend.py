"""v_new_2 Step C2 — Mega-trend mode.

When a trade is in a coin showing extreme trend strength, switch exit logic
from ATR-trail to "daily-EMA-flip-only" — let the winner ride the entire
parabolic phase instead of getting cut by normal volatility breathing.

Mega-trend trigger (any of):
  A) coin_30d_return > +50% (LONG) / < -50% (SHORT) at entry bar
  B) vol_to_mcap_24h > 0.10 at entry bar
  C) days_since_long_flip > 60 (LONG) / days_since_short_flip > 60 (SHORT)
     AND coin_7d_return > +20% (LONG) / < -20% (SHORT)

Exit logic per mode:
  Default mode:    ATR-trail (2-3× ATR depending on tier)
  Profit-lock:     activates at +30% unrealized — switches to trend-flip-only
  MEGA-TREND mode: triggers IMMEDIATELY at entry — exit only on EMA20/50
                   daily cross. Bypasses profit-lock, bypasses ATR trail
                   below +50%. Above +50% running peak: 8×ATR trail (very loose)
                   so we don't give back too much on a real reversal.

Test set:
  - RNDRUSDT 2023-10 → 2024-04 cycle (was +119% with default, +245% with profit-lock)
  - All "big winner" coins where the strategy historically captured <30% of buy-hold
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"

THRESH = 0.10
FIRST_MOVE_WINDOW_BARS = 42
MEGA_TREND_HOLD_BARS = 365 * 6   # up to a year for mega-trends
EXTENDED_HOLD_BARS = 180 * 6
FRICTION = 0.001

DEPLOYABLE_CONFIGS = [
    ("long",  "top25",   "atr",  2.0, "atr_trail_3"),
    ("long",  "26-50",   "pct", 15.0, "atr_trail_3"),
    ("long",  "51-100",  "atr",  3.0, "atr_trail_2"),
    ("long",  "101-200", "atr",  4.0, "atr_trail_2"),
    ("short", "top25",   "atr",  4.0, "atr_trail_2"),
    ("short", "26-50",   "pct", 10.0, "atr_trail_2"),
    ("short", "51-100",  "atr",  4.0, "atr_trail_2"),
    ("short", "101-200", "atr",  4.0, "atr_trail_2"),
]


def _ts(): return f"[{time.strftime('%H:%M:%S')}]"


def _is_mega_trend(side, row):
    """Returns True if entry bar shows mega-trend signature."""
    if side == "long":
        cond_a = row.get("coin_30d_return", 0) > 0.50
        cond_b = row.get("vol_to_mcap_24h", 0) > 0.10
        cond_c = (row.get("days_since_long_flip", 0) > 60
                   and row.get("coin_7d_return", 0) > 0.20)
    else:
        cond_a = row.get("coin_30d_return", 0) < -0.50
        cond_b = row.get("vol_to_mcap_24h", 0) > 0.10
        cond_c = (row.get("days_since_short_flip", 0) > 60
                   and row.get("coin_7d_return", 0) < -0.20)
    return cond_a or cond_b or cond_c


def _backtest_with_mega_trend(df, side, pb_kind, pb_lvl, exit_method,
                                profit_lock_pct=0.30):
    """Backtester with three modes:
      default: ATR-trail
      profit-lock: activated at +30% unrealized → trend-flip-only
      mega-trend: detected at entry → trend-flip-only with loose-ATR backstop"""
    n = len(df)
    if n < 50: return []
    closes = df["close"].values
    timestamps = df["timestamp"].values
    atr_pct = df["atr14_pct"].values
    if side == "long":
        trend = df["d_trend_long"].values
        pb_arr = df["pullback_pct"].values if pb_kind=="pct" else df["pullback_atr"].values
    else:
        trend = df["d_trend_short"].values
        pb_arr = df["rise_pct"].values if pb_kind=="pct" else df["rise_atr"].values

    coin_30d = df["coin_30d_return"].values if "coin_30d_return" in df.columns else np.zeros(n)
    coin_7d = df["coin_7d_return"].values if "coin_7d_return" in df.columns else np.zeros(n)
    days_long_flip = df["days_since_long_flip"].values if "days_since_long_flip" in df.columns else np.zeros(n)
    days_short_flip = df["days_since_short_flip"].values if "days_since_short_flip" in df.columns else np.zeros(n)
    if "vol_to_mcap_24h" in df.columns:
        vol_mcap = df["vol_to_mcap_24h"].values
    else:
        vol_mcap = np.full(n, np.nan)

    trades = []
    open_pos = False
    entry_idx = -1
    entry_price = 0.0
    running_extreme = 0.0
    profit_lock_active = False
    mega_trend_active = False

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
            # Mega-trend detection at ENTRY
            row_dict = {
                "coin_30d_return": coin_30d[i] if not pd.isna(coin_30d[i]) else 0,
                "coin_7d_return": coin_7d[i] if not pd.isna(coin_7d[i]) else 0,
                "days_since_long_flip": days_long_flip[i] if not pd.isna(days_long_flip[i]) else 0,
                "days_since_short_flip": days_short_flip[i] if not pd.isna(days_short_flip[i]) else 0,
                "vol_to_mcap_24h": vol_mcap[i] if not pd.isna(vol_mcap[i]) else 0,
            }
            mega_trend_active = _is_mega_trend(side, row_dict)
            open_pos = True
            i += 1; continue

        p = closes[i]
        if pd.isna(p): i += 1; continue
        if side == "long":
            pnl_frac = p / entry_price - 1.0
            running_extreme = max(running_extreme, p)
        else:
            pnl_frac = entry_price / p - 1.0
            running_extreme = min(running_extreme, p)

        # Profit-lock activation (only if NOT in mega-trend mode — mega-trend already locks)
        if not mega_trend_active and profit_lock_pct is not None and not profit_lock_active and pnl_frac >= profit_lock_pct:
            profit_lock_active = True

        # Exit decision
        exit_idx = None; exit_reason = None
        max_hold = MEGA_TREND_HOLD_BARS if mega_trend_active else EXTENDED_HOLD_BARS

        if mega_trend_active:
            # Trend-flip exit primary; loose 8×ATR backstop above +50%
            if not pd.isna(trend[i]) and trend[i] != 1:
                exit_idx = i; exit_reason = "trend_flip_mega"
            elif pnl_frac >= 0.50:
                a = atr_pct[i] / 100.0 if not pd.isna(atr_pct[i]) else 0.02
                if side == "long":
                    if p <= running_extreme * (1 - 8 * a):
                        exit_idx = i; exit_reason = "loose_trail_mega"
                else:
                    if p >= running_extreme * (1 + 8 * a):
                        exit_idx = i; exit_reason = "loose_trail_mega"
        elif profit_lock_active:
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

        if exit_idx is None and i - entry_idx >= max_hold:
            exit_idx = i; exit_reason = "timeout"

        if exit_idx is not None:
            exit_price = closes[exit_idx]
            if side == "long":
                pnl = exit_price/entry_price - 1.0
            else:
                pnl = entry_price/exit_price - 1.0
            pnl -= FRICTION
            trades.append({
                "symbol": df["symbol"].iloc[0] if "symbol" in df.columns else "?",
                "tier": df["mcap_tier"].iloc[0] if "mcap_tier" in df.columns else "?",
                "side": side,
                "entry_time": pd.Timestamp(timestamps[entry_idx]),
                "exit_time": pd.Timestamp(timestamps[exit_idx]),
                "bars_held": int(exit_idx - entry_idx),
                "exit_reason": exit_reason,
                "mega_trend": bool(mega_trend_active),
                "pnl_pct": float(pnl),
            })
            open_pos = False
            i = exit_idx + 1
            continue
        i += 1
    return trades


def _summary(pnls, leverage=1.0):
    if len(pnls)==0: return {"n":0,"wr":0,"pf":0,"sum_pnl":0,"max_pnl":0}
    p = np.clip(pnls*leverage, -0.99, 100.0)
    wins = p[p>0]; losses = p[p<0]
    return {"n":int(len(p)),"wr":float((p>0).mean()),
             "sum_pnl":float(p.sum()*100),"max_pnl":float(p.max()*100),
             "avg_win":float(wins.mean()*100) if len(wins) else 0,
             "avg_loss":float(losses.mean()*100) if len(losses) else 0,
             "pf":float(wins.sum()/abs(losses.sum())) if len(losses) else float("inf")}


def main():
    t0 = time.time()
    print(f"{_ts()} v_new_2 Step C2 — Mega-trend mode")
    df = pd.read_parquet(DATA_DIR / "features_v_new_2_v2.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)

    # Add coin_7d_return and coin_30d_return from base features
    print(f"  loading coin returns from base features...")
    base = pd.read_parquet(DATA_DIR / "features_full.parquet",
                           columns=["symbol","timestamp","coin_7d_return","coin_30d_return"])
    base["timestamp"] = pd.to_datetime(base["timestamp"], utc=True)
    df = df.merge(base, on=["symbol","timestamp"], how="left")

    # Add vol/mcap if available
    try:
        vmcap = pd.read_parquet(DATA_DIR / "vol_mcap_feature.parquet",
                                  columns=["symbol","timestamp","vol_to_mcap_24h"])
        vmcap["timestamp"] = pd.to_datetime(vmcap["timestamp"], utc=True)
        df = df.merge(vmcap, on=["symbol","timestamp"], how="left")
        print(f"  vol/mcap merged, coverage: {df['vol_to_mcap_24h'].notna().mean()*100:.1f}%")
    except Exception as e:
        print(f"  vol/mcap not loaded ({e}) — using None")
        df["vol_to_mcap_24h"] = np.nan

    print(f"  features ready: {df.shape}")

    # Run on all symbols, collect all trades by config
    rndr_trades = []
    all_results = []
    for side, tier, pb_kind, pb_lvl, exit_m in DEPLOYABLE_CONFIGS:
        tier_syms = df[df["mcap_tier"]==tier]["symbol"].unique()
        all_trades = []
        for sym in tier_syms:
            g = df[df["symbol"]==sym].reset_index(drop=True)
            if len(g) < 50: continue
            trades = _backtest_with_mega_trend(g, side, pb_kind, pb_lvl, exit_m)
            all_trades.extend(trades)
            if sym == "RNDRUSDT" and side == "long":
                rndr_trades.extend(trades)
        # Aggregate
        n_mega = sum(1 for t in all_trades if t["mega_trend"])
        s = _summary(np.array([t["pnl_pct"] for t in all_trades]))
        oos = [t for t in all_trades if pd.Timestamp(t["entry_time"]).tz_localize("UTC" if pd.Timestamp(t["entry_time"]).tz is None else None) >= pd.Timestamp("2025-07-01", tz="UTC")]
        oos = []
        for t in all_trades:
            et = pd.Timestamp(t["entry_time"])
            if et.tz is None: et = et.tz_localize("UTC")
            if et >= pd.Timestamp("2025-07-01", tz="UTC"):
                oos.append(t)
        s_oos = _summary(np.array([t["pnl_pct"] for t in oos]))
        all_results.append({"side":side, "tier":tier,
                             "n":s["n"], "n_mega":n_mega, "WR":s["wr"]*100, "PF":s["pf"],
                             "sum_pnl":s["sum_pnl"], "max_pnl":s["max_pnl"],
                             "oos_n":s_oos["n"], "oos_PF":s_oos["pf"]})
        print(f"  {side:5s} {tier:9s}  n={s['n']:>5}  mega={n_mega:>4}  WR={s['wr']*100:5.1f}%  "
              f"PF={s['pf']:5.2f}  max_win={s['max_pnl']:>+6.1f}%  oos_PF={s_oos['pf']:.2f}")

    pd.DataFrame(all_results).to_csv(RESULTS / "step_c2_mega_trend_summary.csv", index=False)

    # RNDR diagnostic — focus on cycle window
    print(f"\n{_ts()} === RNDRUSDT 2023-10 → 2024-06 ===")
    cycle = []
    for t in rndr_trades:
        et = pd.Timestamp(t["entry_time"])
        if et.tz is None: et = et.tz_localize("UTC")
        if pd.Timestamp("2023-10-01", tz="UTC") <= et <= pd.Timestamp("2024-06-30", tz="UTC"):
            cycle.append(t)
    print(f"  Cycle trades: {len(cycle)}")
    for t in cycle:
        et = pd.Timestamp(t["entry_time"])
        xt = pd.Timestamp(t["exit_time"])
        print(f"    {et} → {xt}  bars={t['bars_held']:>3}  reason={t['exit_reason']:>20s}  "
              f"mega={t['mega_trend']}  pnl={t['pnl_pct']*100:+6.1f}%")
    print(f"\n  Sum cycle PnL: {sum(t['pnl_pct'] for t in cycle)*100:+.0f}%  "
          f"(buy-hold was +752%)")
    print(f"  vs Step A profit-lock-only: +245%")

    # Find ALL big winners in our universe to test mega-trend mode
    print(f"\n{_ts()} === Top 10 biggest single-coin LONG sums (full history) ===")
    by_sym = {}
    for side, tier, pb_kind, pb_lvl, exit_m in [c for c in DEPLOYABLE_CONFIGS if c[0]=="long"]:
        tier_syms = df[df["mcap_tier"]==tier]["symbol"].unique()
        for sym in tier_syms:
            g = df[df["symbol"]==sym].reset_index(drop=True)
            if len(g) < 50: continue
            trades = _backtest_with_mega_trend(g, side, pb_kind, pb_lvl, exit_m)
            if not trades: continue
            sum_pnl = sum(t["pnl_pct"] for t in trades) * 100
            n_mega = sum(1 for t in trades if t["mega_trend"])
            by_sym[sym] = (sum_pnl, len(trades), n_mega)

    top10 = sorted(by_sym.items(), key=lambda x:-x[1][0])[:10]
    print(f"  {'symbol':>14s}  {'sum_pnl':>9s}  {'n_trades':>9s}  {'n_mega':>7s}")
    for sym, (sum_pnl, n_tr, n_mega) in top10:
        print(f"  {sym:>14s}  {sum_pnl:>+8.0f}%  {n_tr:>9}  {n_mega:>7}")

    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
