"""Realistic month-by-month equity curve for the Phase 17.E locked config.

Locked config: D1e detector (D1 + universe-down-breadth >= 0.6) + v4 LR
classifier (16-feature, outcome label, trained on D1e SHORT trades) + S4
classifier-weighted sizing + E2 ATR-based exits.

Sizing rules from CLAUDE.md: 5% equity per trade × 5x leverage = 25% notional;
MAX_CONCURRENT_POSITIONS = 5; compounding.

Reads: data/d1e_short_trades_with_features.csv + models/d1e_short_v4.joblib.
Re-simulates E2 exits with the v4 classifier score for S4 sizing.

Run from services/python/:
    uv run python scripts/phase17_perf_breakdown.py
"""
from __future__ import annotations

import json
import sys
import time
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

NOTIONAL_PCT = 0.25     # 5% × 5x
MAX_CONCURRENT = 5
ATR_PERIOD = 14
ATR_SL_MULT = 2.0
ATR_TP_MULT = 6.0
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015
RAPID_RALLY_PCT = 0.03
RAPID_RALLY_LOOKBACK_15M = 96


def _pf(p):
    p = np.asarray(p, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _build_rapid_exit_panel(candles_15m: pd.DataFrame) -> pd.Series:
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


def _sim_E2_atr(close, high, low, open_, atr14, enter_bar, *, rally_flag):
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
            return {"pnl_pct": pnl, "exit": "sl", "bars": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            pnl = (entry_price - tp_price) / entry_price - FRICTION_PCT
            return {"pnl_pct": pnl, "exit": "tp", "bars": int(bar - enter_bar)}
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (entry_price - ep) / entry_price - FRICTION_PCT,
                    "exit": "rapid", "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[eb])) / entry_price - FRICTION_PCT,
            "exit": "timeout", "bars": int(eb - enter_bar)}


def _simulate_equity(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("entry_time").reset_index(drop=True).copy()
    df["exit_time"] = df["entry_time"] + pd.to_timedelta(df["bars"] * 15, unit="m")
    equity = 1.0
    open_exits: list[pd.Timestamp] = []
    rows = []
    for _, t in df.iterrows():
        open_exits = [x for x in open_exits if x > t["entry_time"]]
        accepted = len(open_exits) < MAX_CONCURRENT
        if accepted:
            r = float(t["sized_pnl_realistic"]) * NOTIONAL_PCT
            equity *= (1.0 + r)
            open_exits.append(t["exit_time"])
            rows.append({
                "entry_time": t["entry_time"], "asset": t["asset"],
                "sized_pnl": float(t["sized_pnl_realistic"]),
                "equity_delta": r, "equity": equity, "accepted": True,
            })
        else:
            rows.append({
                "entry_time": t["entry_time"], "asset": t["asset"],
                "sized_pnl": float(t["sized_pnl_realistic"]),
                "equity_delta": 0.0, "equity": equity, "accepted": False,
            })
    return pd.DataFrame(rows)


def main():
    started = time.monotonic()
    print(f"loading {TRADES_PATH}", flush=True)
    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    print(f"loading model from {MODEL_PATH}", flush=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]
    gated = df[df["score"] >= threshold].reset_index(drop=True)
    print(f"  v4 kept {len(gated):,} of {len(df):,} ({len(gated)/max(len(df),1)*100:.1f}%)")

    print("loading candles for E2 re-simulation...", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    btc = (candles[candles["asset"] == "BTCUSDT"]
           .sort_values("timestamp").set_index("timestamp"))
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)
    rapid_exit_15m = _build_rapid_exit_panel(btc_15m)
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)

    print("re-simulating E2 ATR exits...", flush=True)
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
            t = _sim_E2_atr(close, high, low, open_, atr14, i, rally_flag=rally)
            if t is None:
                continue
            rec = r.to_dict()
            rec["e2_pnl_pct"] = t["pnl_pct"]
            rec["e2_exit"] = t["exit"]
            rec["e2_bars"] = int(t["bars"])
            rows.append(rec)
    sim = pd.DataFrame(rows)
    print(f"  E2 ok: {len(sim):,}/{len(gated):,}")

    # S4 sizing
    score = sim["score"].to_numpy(dtype=float)
    pos_scale_s4 = np.clip(3.0 * (score - 0.5), 0.0, 1.5)
    sim["pos_scale_s4"] = pos_scale_s4
    sim["sized_pnl_realistic"] = sim["e2_pnl_pct"].to_numpy(dtype=float) * pos_scale_s4
    sim["bars"] = sim["e2_bars"]
    sim["entry_time"] = pd.to_datetime(sim["entry_time"], utc=True)

    # Anatomy of one trade
    print("\n" + "=" * 78)
    print("ONE TRADE — D1e + v4 + S4 × E2 (locked Phase 17 config)")
    print("=" * 78)
    oot = sim[sim["entry_time"] >= TRAIN_OOT_BOUNDARY].copy()
    winners = oot[oot["e2_pnl_pct"] > 0].sort_values("e2_pnl_pct").reset_index(drop=True)
    ex = winners.iloc[len(winners) // 2] if len(winners) > 0 else oot.iloc[0]
    print(f"  asset                : {ex['asset']}")
    print(f"  entry_time           : {ex['entry_time']}")
    print(f"  v4 score             : {ex['score']:.4f}  (gate ≥ {threshold:.4f})")
    print(f"  BTC trend score      : {ex['btc_score']:+.4f}")
    print(f"  ATR_14 at entry      : {ex['atr14_at_entry']:.6f}  ({ex['atr14_at_entry']/ ex['atr14_at_entry'] * 100 if ex['atr14_at_entry']==0 else 0:.2f}…)")
    print(f"  S4 pos_scale         : {ex['pos_scale_s4']:.3f}x  (= clip(3 × (score−0.5), 0, 1.5))")
    print(f"  E2 SL = entry + 2×ATR; TP = entry − 6×ATR")
    print(f"  exit_kind            : {ex['e2_exit']}  bars held: {int(ex['e2_bars'])}")
    print(f"  E2 raw pnl_pct       : {ex['e2_pnl_pct']*100:+.2f}%")
    print(f"  S4 sized pnl         : {ex['sized_pnl_realistic']*100:+.2f}%")
    print(f"  equity move @ 5%×5x  : {ex['sized_pnl_realistic']*NOTIONAL_PCT*100:+.3f}%")

    # Realistic equity sim
    print("\n" + "=" * 78)
    print(f"REALISTIC EQUITY — 5%/5x sizing, max {MAX_CONCURRENT} concurrent, compounding")
    print("=" * 78)
    eq = _simulate_equity(sim)
    accepted = eq[eq["accepted"]].copy()
    print(f"  total signals: {len(eq):,}  accepted: {len(accepted):,}  "
          f"dropped on cap: {(~eq['accepted']).sum():,} "
          f"({(~eq['accepted']).mean()*100:.1f}%)")

    accepted["month"] = accepted["entry_time"].dt.tz_convert("UTC").dt.strftime("%Y-%m")
    monthly = accepted.groupby("month").agg(
        n=("equity_delta", "size"),
        wins=("sized_pnl", lambda s: int((s > 0).sum())),
        avg_per_trade=("sized_pnl", "mean"),
        end_equity=("equity", "last"),
    )
    end_eq = monthly["end_equity"].values
    start_eq = np.concatenate([[1.0], end_eq[:-1]])
    monthly["monthly_ret"] = (end_eq / start_eq) - 1.0
    monthly["wr"] = monthly["wins"] / monthly["n"]

    def _print_table(d, label):
        print(f"\n{label}")
        print(f"  {'month':<10} {'n':>5} {'wr':>6} {'avg/tr':>8} {'month %':>9} {'cum equity':>11}")
        for m, r in d.iterrows():
            print(f"  {m:<10} {int(r['n']):>5d} {r['wr']*100:>5.1f}% "
                  f"{r['avg_per_trade']*100:>+7.2f}% {r['monthly_ret']*100:>+7.2f}% "
                  f"  {r['end_equity']:>9.3f}x")

    oot_idx = monthly.index >= "2026-01"
    _print_table(monthly[~oot_idx], "TRAIN months (informational, not the ship gate)")
    _print_table(monthly[oot_idx], "OOT months (strict 8-point hold-out)")

    full_returns = monthly["monthly_ret"]
    print(f"\n  Full {len(full_returns)} months: best {full_returns.max()*100:+.1f}%   "
          f"worst {full_returns.min()*100:+.1f}%   median {full_returns.median()*100:+.2f}%   "
          f"P(loss) {(full_returns < 0).mean()*100:.1f}%")
    print(f"  Final equity @ end of {monthly.index[-1]}: {monthly['end_equity'].iloc[-1]:.2f}x")
    print(f"\nwall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
