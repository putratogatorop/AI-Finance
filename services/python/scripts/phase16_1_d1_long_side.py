"""Phase 16.1 — D1 detector applied to the LONG side.

Hypothesis: under the volume-only D0 detector, the long side measured PF
≈ 0.92 OOT (Phase 2). The user's working theory across Phase 15 was that
fixed-% thresholds don't translate across volatility regimes; ATR
normalization fixed that for SHORT (D1 OOT PF 2.064 vs D0 1.882). The
question this phase answers is whether the same lift carries over to
LONG, which would unlock a long+short combined deployment.

Mirror detector for LONG (D1L):
    (close - rolling_low_96) / ATR_14 >= 2.5
  & vol_ratio >= 2.0
  & close > open  (green current bar)

Sizing: sign-locked LONG — only enter when btc_score > 0 (Candidate B
4h MACD spread, K=0.005255). pos_scale = min(1.5 * abs(score), 1.5).

Rapid-bear exit (mirror of Phase 14 V2 rapid-rally):
    close LONG if BTC daily is ALL-BULLISH AND BTC 24h return < -3%.

Same SL=5% / TP=15% / timeout=672 / friction=0.0015 as SHORT side, but
with LONG semantics (TP = +15% above entry, SL = -5% below entry).

Run from services/python/:
    python scripts/phase16_1_d1_long_side.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.indicators import atr as _atr, ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
OUT_PATH = REPO_ROOT / "services/python/results" / "phase16_1_d1_long_side.json"

STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015

VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
COOLDOWN_BARS = 96

BTC_TREND_K = 0.005255

RAPID_BEAR_PCT = 0.03           # require BTC dropped >3% in 24h
RAPID_BEAR_LOOKBACK_15M = 96

WF_FOLD_MONTHS = 3
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000

TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")


def _pf(pnls):
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _cell(pnls):
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {"trades": 0, "wr": 0.0, "pf": float("nan"),
                "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0}
    return {
        "trades": int(len(pnls)),
        "wr": float((pnls > 0).mean()),
        "pf": _pf(pnls),
        "avg_pnl_pct": float(pnls.mean()),
        "total_pnl_pct": float(pnls.sum()),
    }


def _walkforward(df, pnl_col):
    if len(df) == 0:
        return {"n_folds_used": 0}
    et = pd.to_datetime(df["entry_time"], utc=True)
    s = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    fold_w = relativedelta(months=WF_FOLD_MONTHS)
    month = relativedelta(months=1)
    pfs = []
    s_ = s
    while s_ + fold_w <= end_global:
        sub = df[(et >= s_) & (et < s_ + fold_w)]
        if len(sub) >= 30:
            p = _pf(sub[pnl_col].to_numpy())
            if np.isfinite(p):
                pfs.append(p)
        s_ += month
    pfs = np.array(pfs)
    if len(pfs) == 0:
        return {"n_folds_used": 0}
    return {
        "n_folds_used": int(len(pfs)),
        "p5": float(np.percentile(pfs, 5)),
        "p50": float(np.percentile(pfs, 50)),
        "p95": float(np.percentile(pfs, 95)),
        "p_lt_1": float((pfs < 1.0).mean()),
        "p_gte_130": float((pfs >= 1.30).mean()),
    }


def _monte_carlo(pnls, n=MC_N_RESAMPLES):
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {"n": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    pfs = np.array([_pf(rng.choice(pnls, len(pnls), replace=True)) for _ in range(n)])
    finite = pfs[np.isfinite(pfs)]
    return {
        "n": int(len(pnls)),
        "point_pf": _pf(pnls),
        "p5": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
        "p50": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
        "p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
    }


# --- Panels -----------------------------------------------------------------


def _build_btc_trend_score_15m(candles_15m: pd.DataFrame) -> pd.Series:
    """Candidate B BTC trend score, range [-1, +1]. Same as Phase 9/15."""
    btc = candles_15m["close"].astype(float)
    btc_4h = btc.resample("4h", label="right", closed="right").last()
    md = _macd(btc_4h, 12, 26, 9)
    spread = ((md["macd"] - md["signal"]).shift(1) / btc_4h.shift(1))
    score_4h = spread.clip(-BTC_TREND_K, BTC_TREND_K) / BTC_TREND_K
    return score_4h.reindex(btc.index, method="ffill")


def _build_rapid_bear_panel(candles_15m: pd.DataFrame) -> pd.Series:
    """LONG-mirror of Phase 14 V2 rapid-rally exit.

    Trigger: BTC daily is ALL-BULLISH on prior close (EMA9>EMA21>EMA50,
    RSI>=50, MACD>signal, K>D) AND BTC 24h return < -3% on the 15m bar.
    Read at entry+1: long positions exit if a sharp drop hits while the
    daily regime still looks bullish (regime/price divergence)."""
    close = candles_15m["close"].astype(float)
    high = candles_15m["high"].astype(float)
    low = candles_15m["low"].astype(float)
    btc_daily = candles_15m.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    ema_fast = _ema(btc_daily["close"], 9).shift(1)
    ema_mid = _ema(btc_daily["close"], 21).shift(1)
    ema_slow = _ema(btc_daily["close"], 50).shift(1)
    ema_cond = (btc_daily["close"].shift(1) > ema_fast) & (ema_fast > ema_mid) & (ema_mid > ema_slow)
    rsi_cond = _rsi(btc_daily["close"], 14).shift(1) >= 50.0
    md = _macd(btc_daily["close"], 12, 26, 9)
    macd_cond = md["macd"].shift(1) > md["signal"].shift(1)
    kdj_v = _kdj(btc_daily["high"], btc_daily["low"], btc_daily["close"], n=9, k_smooth=3, d_smooth=3)
    kdj_cond = kdj_v["k"].shift(1) > kdj_v["d"].shift(1)
    daily_bull = (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)
    daily_bull_15m = daily_bull.reindex(close.index, method="ffill").fillna(False)
    btc_24h_ret = close.pct_change(RAPID_BEAR_LOOKBACK_15M)
    rapid_exit = (daily_bull_15m & (btc_24h_ret < -RAPID_BEAR_PCT)).fillna(False).astype(bool)
    return rapid_exit


# --- Detectors --------------------------------------------------------------


def _detect_long_d0_volume(close, high, low, open_, volume) -> np.ndarray:
    """D0L — bigmover-volume baseline mirror for LONG.

    Mirror of D0 SHORT: vol_ratio>=3, rolling_low rebound>=5%, rise>drop.
    """
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return out
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().to_numpy()
    rolling_high = pd.Series(high).rolling(PRICE_LOOKBACK, min_periods=PRICE_LOOKBACK).max().to_numpy()
    rolling_low = pd.Series(low).rolling(PRICE_LOOKBACK, min_periods=PRICE_LOOKBACK).min().to_numpy()
    for i in range(PRICE_LOOKBACK + VOL_MA_PERIOD, n):
        if vol_ma[i] <= 0:
            continue
        if volume[i] / vol_ma[i] < 3.0:
            continue
        if rolling_low[i] <= 0:
            continue
        price_rise = (close[i] - rolling_low[i]) / rolling_low[i]
        if price_rise < 0.05:
            continue
        price_drop = (rolling_high[i] - close[i]) / rolling_high[i] if rolling_high[i] > 0 else 0
        if not (price_rise > price_drop):
            continue
        out[i] = True
    return out


def _detect_long_d1_atr_rise(close, high, low, open_, volume) -> np.ndarray:
    """D1L — ATR-normalized rise + lower vol confirm + green current bar.

    Mirror of Phase-15 D1: rebound from rolling-low, measured in ATR units.
    """
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + ATR_PERIOD:
        return out
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().to_numpy()
    rolling_low = pd.Series(low).rolling(PRICE_LOOKBACK, min_periods=PRICE_LOOKBACK).min().to_numpy()
    atr14 = _atr(pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD).to_numpy()
    for i in range(PRICE_LOOKBACK + VOL_MA_PERIOD, n):
        if not (atr14[i] > 0):
            continue
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < 2.0:
            continue
        if rolling_low[i] <= 0:
            continue
        rise_in_atr = (close[i] - rolling_low[i]) / atr14[i]
        if rise_in_atr < 2.5:
            continue
        if close[i] <= open_[i]:  # require green current bar
            continue
        out[i] = True
    return out


# --- Simulators -------------------------------------------------------------


def _simulate_long(close, high, low, open_, enter_bar, *, bear_flag=None):
    n = len(close)
    if enter_bar >= n:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    sl_price = entry_price * (1 - STOP_LOSS_PCT)
    tp_price = entry_price * (1 + TAKE_PROFIT_PCT)
    last_bar = min(enter_bar + 1 + TIMEOUT_BARS, n)
    for bar in range(enter_bar + 1, last_bar):
        if low[bar] <= sl_price:
            return {"pnl_pct": -STOP_LOSS_PCT - FRICTION_PCT, "exit": "sl",
                    "bars": int(bar - enter_bar)}
        if high[bar] >= tp_price:
            return {"pnl_pct": TAKE_PROFIT_PCT - FRICTION_PCT, "exit": "tp",
                    "bars": int(bar - enter_bar)}
        if bear_flag is not None and bool(bear_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (ep - entry_price) / entry_price - FRICTION_PCT, "exit": "rapid",
                    "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (float(close[eb]) - entry_price) / entry_price - FRICTION_PCT,
            "exit": "timeout", "bars": int(eb - enter_bar)}


def _run_detector(
    candles: pd.DataFrame,
    detector_fn,
    *,
    btc_score_15m: pd.Series,
    rapid_bear_15m: pd.Series,
) -> pd.DataFrame:
    rows = []
    btc_score_naive = btc_score_15m.copy()
    btc_score_naive.index = pd.to_datetime(btc_score_naive.index, utc=True).tz_convert(None)
    bear_naive = rapid_bear_15m.copy()
    bear_naive.index = pd.to_datetime(bear_naive.index, utc=True).tz_convert(None)

    assets = sorted(candles["asset"].unique())
    for idx, asset in enumerate(assets):
        sub = (
            candles[candles["asset"] == asset]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(sub) < 200:
            continue
        ts_naive = pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(None).to_numpy()
        close = sub["close"].to_numpy(dtype=float)
        high = sub["high"].to_numpy(dtype=float)
        low = sub["low"].to_numpy(dtype=float)
        open_ = sub["open"].to_numpy(dtype=float)
        volume = sub["volume"].to_numpy(dtype=float)

        triggers = detector_fn(close, high, low, open_, volume)
        last_entry = -10**9
        entry_bars = []
        for i in range(len(triggers)):
            if not triggers[i]:
                continue
            if i - last_entry < COOLDOWN_BARS:
                continue
            entry_bars.append(i)
            last_entry = i

        btc_score = btc_score_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill").to_numpy(dtype=float)
        bear = bear_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill").fillna(False).to_numpy().astype(bool)

        for i in entry_bars:
            score = btc_score[i] if i < len(btc_score) else float("nan")
            # Sign-locked LONG: only size if BTC trend score > 0
            if not np.isfinite(score) or score <= 0:
                continue
            pos_scale = min(1.5 * abs(score), 1.5)
            if pos_scale <= 0:
                continue
            t = _simulate_long(close, high, low, open_, i, bear_flag=bear)
            if t is None:
                continue
            t["sized_pnl"] = t["pnl_pct"] * pos_scale
            t["pos_scale"] = pos_scale
            t["asset"] = asset
            t["entry_time"] = pd.Timestamp(sub["timestamp"].iloc[i + 1] if i + 1 < len(sub) else sub["timestamp"].iloc[i])
            t["btc_score"] = float(score)
            rows.append(t)
        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(assets)} assets processed, trades={len(rows):,}",
                  flush=True)
    return pd.DataFrame(rows)


def main():
    started = time.monotonic()

    print(f"loading candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)

    print("building BTC trend score + rapid-bear panels...", flush=True)
    btc_score_15m = _build_btc_trend_score_15m(btc_15m)
    rapid_bear_15m = _build_rapid_bear_panel(btc_15m)

    detectors = {
        "D0L_bigmover_volume":     _detect_long_d0_volume,
        "D1L_atr_rise_2.5_vol_2":  _detect_long_d1_atr_rise,
    }

    sweep = {}
    print("\n=== LONG-side detectors (sign-locked LONG, rapid-bear exit) ===")
    for name, fn in detectors.items():
        print(f"\n  detector: {name}", flush=True)
        sim = _run_detector(candles, fn, btc_score_15m=btc_score_15m, rapid_bear_15m=rapid_bear_15m)
        if len(sim) == 0:
            print(f"    NO TRADES", flush=True)
            sweep[name] = {"trades": 0}
            continue
        sim["entry_time"] = pd.to_datetime(sim["entry_time"], utc=True)
        sim["period"] = np.where(sim["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
        train_pnls = sim[sim["period"] == "train"]["sized_pnl"].to_numpy()
        oot_pnls = sim[sim["period"] == "oot"]["sized_pnl"].to_numpy()
        wf = _walkforward(sim, "sized_pnl")
        mc = _monte_carlo(oot_pnls)
        train_m = _cell(train_pnls)
        oot_m = _cell(oot_pnls)
        exit_counts = sim["exit"].value_counts().to_dict()
        sweep[name] = {
            "train_metrics": train_m, "oot_metrics": oot_m,
            "walkforward": wf, "monte_carlo_oot": mc,
            "exit_counts": {k: int(v) for k, v in exit_counts.items()},
        }
        print(f"    TRAIN n={train_m['trades']:>5} PF={train_m['pf']:.3f}  "
              f"OOT n={oot_m['trades']:>5} PF={oot_m['pf']:.3f}  "
              f"WF p5={wf.get('p5', float('nan')):.3f}  p50={wf.get('p50', float('nan')):.3f}  "
              f"P(loss)={wf.get('p_lt_1', float('nan'))*100:.1f}%  "
              f"MC p5={mc.get('p5', float('nan')):.3f}", flush=True)
        print(f"    exits: {exit_counts}", flush=True)

    # Decision: D1L LONG ship-gate
    locked_name = "D1L_atr_rise_2.5_vol_2"
    locked = sweep.get(locked_name)
    if locked and locked.get("oot_metrics", {}).get("trades", 0) > 0:
        oot_pf = locked["oot_metrics"]["pf"]
        ship_gate = {
            "oot_pf_ge_130": bool(np.isfinite(oot_pf) and oot_pf >= 1.30),
            "oot_n_ge_200": bool(locked["oot_metrics"]["trades"] >= 200),
            "wf_p5_gt_05": bool(locked["walkforward"].get("p5", -1) > 0.5),
            "wf_p5_gt_1": bool(locked["walkforward"].get("p5", -1) > 1.0),
            "mc_p5_gt_1": bool(np.isfinite(locked["monte_carlo_oot"].get("p5", float("nan")))
                                and locked["monte_carlo_oot"].get("p5", -1) > 1.0),
        }
        ship_gate["overall_strict"] = (ship_gate["oot_pf_ge_130"] and ship_gate["oot_n_ge_200"]
                                       and ship_gate["wf_p5_gt_1"] and ship_gate["mc_p5_gt_1"])
        ship_gate["overall_relaxed"] = (ship_gate["oot_pf_ge_130"] and ship_gate["oot_n_ge_200"]
                                        and ship_gate["wf_p5_gt_05"])
        print(f"\n=== Long ship gate at {locked_name} ===")
        for k, v in ship_gate.items():
            print(f"  {k}: {'PASS' if v else 'FAIL'}")
        print(f"  → strict ship gate (PF≥1.30, n≥200, WF p5>1.0, MC p5>1.0): "
              f"{'SHIP' if ship_gate['overall_strict'] else 'DO NOT SHIP'}")
        print(f"  → relaxed gate (PF≥1.30, n≥200, WF p5>0.5): "
              f"{'TRADE LONG' if ship_gate['overall_relaxed'] else 'STAY SHORT-ONLY'}")
    else:
        ship_gate = None

    out = {
        "phase": "16.1 — D1 LONG side",
        "snapshot": SNAPSHOT_DATE,
        "detectors": list(detectors.keys()),
        "sweep_results": sweep,
        "locked": {"name": locked_name, "ship_gate": ship_gate},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
