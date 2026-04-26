"""Compare realistic 36-month equity curves across 4 configurations on the
D1e + v4 trade ledger:

  (a) S1 × E1 — current sizing + fixed-pct exits  (matches phase16-style)
  (b) S1 × E2 — current sizing + ATR exits         (middle ground)
  (c) S4 × E1 — classifier-weighted sizing + fixed exits
  (d) S4 × E2 — classifier-weighted sizing + ATR exits  (Phase 17.E lock)

All under CLAUDE.md sizing: 5% × 5x = 25% notional, max 5 concurrent,
compounding.

Run from services/python/:
    uv run python scripts/phase17_sizing_compare.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES
from src.ml.indicators import atr as _atr, ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
TRADES_PATH = SERVICES_PY / "data" / "d1e_short_trades_with_features.csv"
MODEL_PATH = SERVICES_PY / "models" / "d1e_short_v4.joblib"
META_PATH = SERVICES_PY / "models" / "d1e_short_v4_meta.json"

NOTIONAL_PCT = 0.25
MAX_CONCURRENT = 5
ATR_PERIOD = 14
ATR_SL_MULT, ATR_TP_MULT = 2.0, 6.0
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015
RAPID_RALLY_PCT = 0.03
RAPID_RALLY_LOOKBACK_15M = 96
STOP_LOSS_PCT, TAKE_PROFIT_PCT = 0.05, 0.15


def _build_rapid_exit_panel(candles_15m):
    close = candles_15m["close"].astype(float)
    btc_daily = candles_15m.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    ema_fast = _ema(btc_daily["close"], 9).shift(1)
    ema_mid = _ema(btc_daily["close"], 21).shift(1)
    ema_slow = _ema(btc_daily["close"], 50).shift(1)
    ema_cond = (btc_daily["close"].shift(1) < ema_fast) & (ema_fast < ema_mid) & (ema_mid < ema_slow)
    rsi_cond = _rsi(btc_daily["close"], 14).shift(1) < 50.0
    md = _macd(btc_daily["close"], 12, 26, 9)
    macd_cond = md["macd"].shift(1) < md["signal"].shift(1)
    kdj_v = _kdj(btc_daily["high"], btc_daily["low"], btc_daily["close"], n=9, k_smooth=3, d_smooth=3)
    kdj_cond = kdj_v["k"].shift(1) < kdj_v["d"].shift(1)
    daily_bear = (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)
    daily_bear_15m = daily_bear.reindex(close.index, method="ffill").fillna(False)
    btc_24h_ret = close.pct_change(RAPID_RALLY_LOOKBACK_15M)
    return ((~daily_bear_15m) & (btc_24h_ret > RAPID_RALLY_PCT)).fillna(False).astype(bool)


def _sim_E1(close, high, low, open_, enter_bar, *, rally_flag):
    n = len(close)
    if enter_bar >= n - 1:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    sl_price = entry_price * (1 + STOP_LOSS_PCT)
    tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
    last_bar = min(enter_bar + 1 + TIMEOUT_BARS, n)
    for bar in range(enter_bar + 1, last_bar):
        if high[bar] >= sl_price:
            return {"pnl_pct": -STOP_LOSS_PCT - FRICTION_PCT, "bars": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            return {"pnl_pct": TAKE_PROFIT_PCT - FRICTION_PCT, "bars": int(bar - enter_bar)}
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (entry_price - ep) / entry_price - FRICTION_PCT,
                    "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[eb])) / entry_price - FRICTION_PCT,
            "bars": int(eb - enter_bar)}


def _sim_E2(close, high, low, open_, atr14, enter_bar, *, rally_flag):
    n = len(close)
    if enter_bar >= n - 1:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    a = float(atr14[enter_bar]) if enter_bar < len(atr14) else float("nan")
    if not np.isfinite(a) or a <= 0:
        return None
    sl_price = entry_price + ATR_SL_MULT * a
    tp_price = entry_price - ATR_TP_MULT * a
    if tp_price <= 0:
        return None
    last_bar = min(enter_bar + 1 + TIMEOUT_BARS, n)
    for bar in range(enter_bar + 1, last_bar):
        if high[bar] >= sl_price:
            pnl = (entry_price - sl_price) / entry_price - FRICTION_PCT
            return {"pnl_pct": pnl, "bars": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            pnl = (entry_price - tp_price) / entry_price - FRICTION_PCT
            return {"pnl_pct": pnl, "bars": int(bar - enter_bar)}
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (entry_price - ep) / entry_price - FRICTION_PCT,
                    "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[eb])) / entry_price - FRICTION_PCT,
            "bars": int(eb - enter_bar)}


def _equity_sim(df: pd.DataFrame, sized_pnl_col: str, bars_col: str):
    df = df.sort_values("entry_time").reset_index(drop=True).copy()
    df["exit_time"] = df["entry_time"] + pd.to_timedelta(df[bars_col] * 15, unit="m")
    equity = 1.0
    open_exits = []
    rows = []
    for _, t in df.iterrows():
        open_exits = [x for x in open_exits if x > t["entry_time"]]
        accepted = len(open_exits) < MAX_CONCURRENT
        if accepted:
            r = float(t[sized_pnl_col]) * NOTIONAL_PCT
            equity *= (1.0 + r)
            open_exits.append(t["exit_time"])
            rows.append({"entry_time": t["entry_time"], "asset": t["asset"],
                         "sized_pnl": float(t[sized_pnl_col]), "equity": equity, "accepted": True})
        else:
            rows.append({"entry_time": t["entry_time"], "asset": t["asset"],
                         "sized_pnl": float(t[sized_pnl_col]), "equity": equity, "accepted": False})
    return pd.DataFrame(rows)


def _summarize(sim, label):
    accepted = sim[sim["accepted"]].copy()
    accepted["month"] = accepted["entry_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    monthly = accepted.groupby("month").agg(
        n=("sized_pnl", "size"),
        wins=("sized_pnl", lambda s: int((s > 0).sum())),
        end_equity=("equity", "last"),
    )
    end_eq = monthly["end_equity"].values
    start_eq = np.concatenate([[1.0], end_eq[:-1]])
    monthly["ret"] = (end_eq / start_eq) - 1.0
    final = float(monthly["end_equity"].iloc[-1])
    rets = monthly["ret"]
    print(f"\n{label}")
    print(f"  accepted trades: {len(accepted):,} of {len(sim):,} signals "
          f"(dropped {(~sim['accepted']).sum():,}, {(~sim['accepted']).mean()*100:.1f}%)")
    print(f"  final equity   : {final:>7.2f}x")
    print(f"  best month     : {rets.max()*100:+.2f}%")
    print(f"  worst month    : {rets.min()*100:+.2f}%")
    print(f"  median month   : {rets.median()*100:+.2f}%")
    print(f"  negative months: {(rets < 0).sum()} / {len(rets)}")
    print(f"  P(loss) monthly: {(rets < 0).mean()*100:.1f}%")
    return {
        "label": label, "final_equity": final,
        "best_month": float(rets.max()), "worst_month": float(rets.min()),
        "median_month": float(rets.median()),
        "negative_months": int((rets < 0).sum()),
        "total_months": int(len(rets)),
        "p_loss": float((rets < 0).mean()),
    }


def main():
    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]
    gated = df[df["score"] >= threshold].reset_index(drop=True)
    print(f"D1e + v4 classifier kept {len(gated):,} of {len(df):,} ({len(gated)/max(len(df),1)*100:.1f}%)")

    print("re-simulating exits per trade...", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    btc = (candles[candles["asset"] == "BTCUSDT"].sort_values("timestamp").set_index("timestamp"))
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)
    rapid_exit_15m = _build_rapid_exit_panel(btc_15m)
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)

    rows = []
    for asset, gtrades in gated.groupby("asset"):
        sub = (candles[candles["asset"] == asset]
               .sort_values("timestamp").reset_index(drop=True))
        if len(sub) < 200:
            continue
        ts_naive = pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(None).to_numpy()
        close = sub["close"].to_numpy(dtype=float)
        high = sub["high"].to_numpy(dtype=float)
        low = sub["low"].to_numpy(dtype=float)
        open_ = sub["open"].to_numpy(dtype=float)
        atr14 = _atr(pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD).to_numpy()
        rally = rapid_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill").fillna(False).to_numpy().astype(bool)
        for _, r in gtrades.iterrows():
            i = int(r["enter_bar"])
            t1 = _sim_E1(close, high, low, open_, i, rally_flag=rally)
            t2 = _sim_E2(close, high, low, open_, atr14, i, rally_flag=rally)
            if t1 is None or t2 is None:
                continue
            rec = r.to_dict()
            rec["e1_pnl_pct"], rec["e1_bars"] = t1["pnl_pct"], t1["bars"]
            rec["e2_pnl_pct"], rec["e2_bars"] = t2["pnl_pct"], t2["bars"]
            rows.append(rec)
    sim = pd.DataFrame(rows)
    print(f"  exits computed for {len(sim):,} trades")

    # Sizings
    btc_score = sim["btc_score"].to_numpy(dtype=float)
    score = sim["score"].to_numpy(dtype=float)
    s1 = np.minimum(1.5 * np.abs(btc_score), 1.5)
    s4 = np.clip(3.0 * (score - 0.5), 0.0, 1.5)

    sim["entry_time"] = pd.to_datetime(sim["entry_time"], utc=True)
    sim["s1_e1_sized"] = sim["e1_pnl_pct"].to_numpy() * s1
    sim["s1_e2_sized"] = sim["e2_pnl_pct"].to_numpy() * s1
    sim["s4_e1_sized"] = sim["e1_pnl_pct"].to_numpy() * s4
    sim["s4_e2_sized"] = sim["e2_pnl_pct"].to_numpy() * s4

    print("\n=== 4-WAY EQUITY COMPARISON (D1e + v4 classifier-gated trades) ===")
    sims = {}
    for col, bars_col, label in [
        ("s1_e1_sized", "e1_bars", "(a) S1 × E1 — Phase-16 style sizing & exits"),
        ("s1_e2_sized", "e2_bars", "(b) S1 × E2 — Phase-16 sizing + ATR exits  ← MIDDLE GROUND"),
        ("s4_e1_sized", "e1_bars", "(c) S4 × E1 — classifier sizing + fixed exits"),
        ("s4_e2_sized", "e2_bars", "(d) S4 × E2 — Phase 17.E LOCK"),
    ]:
        s = _equity_sim(sim, col, bars_col)
        sims[label] = _summarize(s, label)

    print("\n=== HEADLINE COMPARISON ===")
    print(f"  {'Config':<60} {'Final':>8} {'Worst':>8} {'P(loss)':>9}")
    for label, m in sims.items():
        print(f"  {label:<60} {m['final_equity']:>7.2f}x {m['worst_month']*100:>+7.2f}% "
              f"{m['p_loss']*100:>7.1f}%")


if __name__ == "__main__":
    main()
