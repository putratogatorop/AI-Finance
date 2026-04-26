"""Phase 13 — regime-aware early exit on BTC short-TF flip.

User's insight: positions hold ~2 days median, ~7 days max, but our 4h
MACD-based btc_trend_score takes 3-5 days to react to a regime flip.
That's why catastrophic months happen — by the time the score blocks
NEW shorts, our existing shorts have already taken full SL hits over
multiple days.

Fix: add an EARLY EXIT trigger. Close any open short the moment BTC's
short-TF regime indicator flips bullish (cuts loss at small drawdown
instead of riding to SL).

This is the missing piece across all 12 prior phases — they tested
ENTRY gates but never an EXIT gate based on regime flip.

Variants tested (entries identical to v2 SHORT ml_on+sized; only exit changes):
  E0 — baseline: SL=5% / TP=15% / timeout=672 (no regime exit)
  E1 — + early exit when BTC 1h all-bearish flips False
  E2 — + early exit when BTC daily all-bearish flips False
  E3 — + early exit when BTC 15m all-bearish flips False

Run from services/python/:
    python scripts/phase13_regime_early_exit.py
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

from src.ml.indicators import ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
V2_RUN_DIR = (
    REPO_ROOT / "services/python/results"
    / "backtest_bigmover_combined_v2_outcome_0582b799_20260426T033736Z"
)
TRADES_CSV = V2_RUN_DIR / "trades.csv"
OUT_PATH = V2_RUN_DIR / "phase13_regime_early_exit.json"

STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015

EMA_FAST = 9
EMA_MID = 21
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH = 9, 3, 3

WF_FOLD_MONTHS = 3
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000


# --- Helpers ----------------------------------------------------------------


def _pf(pnls: np.ndarray) -> float:
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _cell(pnls: np.ndarray) -> dict:
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


def _walkforward(df: pd.DataFrame, pnl_col: str = "sized_pnl") -> dict:
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
        "p25": float(np.percentile(pfs, 25)),
        "p50": float(np.percentile(pfs, 50)),
        "p75": float(np.percentile(pfs, 75)),
        "p95": float(np.percentile(pfs, 95)),
        "p_lt_1": float((pfs < 1.0).mean()),
        "p_gte_130": float((pfs >= 1.30).mean()),
    }


def _monte_carlo(pnls: np.ndarray, n: int = MC_N_RESAMPLES) -> dict:
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


def _all_bearish_panel(open_, high, low, close) -> pd.Series:
    """EMA + RSI + MACD + KDJ all agree on bearish, shifted by 1 bar (no lookahead)."""
    ema_fast = _ema(close, EMA_FAST).shift(1)
    ema_mid = _ema(close, EMA_MID).shift(1)
    ema_cond = (close.shift(1) < ema_fast) & (ema_fast < ema_mid)
    rsi_cond = _rsi(close, RSI_PERIOD).shift(1) < 50.0
    md = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_cond = md["macd"].shift(1) < md["signal"].shift(1)
    kdj_v = _kdj(high, low, close, n=KDJ_N, k_smooth=KDJ_K_SMOOTH, d_smooth=KDJ_D_SMOOTH)
    kdj_cond = kdj_v["k"].shift(1) < kdj_v["d"].shift(1)
    return (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)


def _simulate_short_with_regime_exit(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, open_: np.ndarray,
    enter_bar: int,
    *,
    regime_bearish_flag: np.ndarray | None,
) -> dict:
    """Same as v2 short simulator + early exit when regime_bearish_flag flips False.

    Args:
      close, high, low, open_: per-asset arrays.
      enter_bar: index of the bar at which the trade ENTERS (= open[enter_bar]).
      regime_bearish_flag: bool array length len(close); True = "BTC regime
        still bearish". When this flips False AT bar `bar`, exit at close[bar].
        If None, no regime exit applied (matches v2 baseline).
    """
    n = len(close)
    if enter_bar >= n:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    sl_price = entry_price * (1 + STOP_LOSS_PCT)
    tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
    last_bar = min(enter_bar + 1 + TIMEOUT_BARS, n)
    for bar in range(enter_bar + 1, last_bar):
        # 1. SL
        if high[bar] >= sl_price:
            return {
                "pnl_pct": -STOP_LOSS_PCT - FRICTION_PCT,
                "exit_reason": "stop_loss",
                "bars_held": int(bar - enter_bar),
            }
        # 2. TP
        if low[bar] <= tp_price:
            return {
                "pnl_pct": TAKE_PROFIT_PCT - FRICTION_PCT,
                "exit_reason": "take_profit",
                "bars_held": int(bar - enter_bar),
            }
        # 3. Regime exit (the new gate)
        if regime_bearish_flag is not None and not bool(regime_bearish_flag[bar]):
            exit_price = float(close[bar])
            return {
                "pnl_pct": (entry_price - exit_price) / entry_price - FRICTION_PCT,
                "exit_reason": "regime_exit",
                "bars_held": int(bar - enter_bar),
            }
    # Timeout
    exit_bar = last_bar - 1
    return {
        "pnl_pct": (entry_price - float(close[exit_bar])) / entry_price - FRICTION_PCT,
        "exit_reason": "timeout",
        "bars_held": int(exit_bar - enter_bar),
    }


# --- Main -------------------------------------------------------------------


def main() -> None:
    started = time.monotonic()

    print(f"loading v2 trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    threshold_predictor = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    short = df[
        (df["direction"] == "short")
        & (df["proba"] >= threshold_predictor)
        & (df["pos_scale"] > 0)
    ][["symbol", "entry_time", "pos_scale", "period"]].copy().reset_index(drop=True)
    print(f"  SHORT ml_on+sized entries: {len(short):,}", flush=True)

    print(f"\nloading candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)

    # ---- Build BTC regime panels at 1h, 15m, daily ----------------------
    print("building BTC regime panels (1h / 15m / daily)...", flush=True)

    btc_1h = btc_15m.resample("1h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    btc_1h_bear = _all_bearish_panel(btc_1h["open"], btc_1h["high"], btc_1h["low"], btc_1h["close"])
    L1h_panel = btc_1h_bear.reindex(btc_15m.index, method="ffill").fillna(False)

    btc_daily = btc_15m.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    btc_daily_bear = _all_bearish_panel(btc_daily["open"], btc_daily["high"], btc_daily["low"], btc_daily["close"])
    Ldaily_panel = btc_daily_bear.reindex(btc_15m.index, method="ffill").fillna(False)

    btc_15m_bear = _all_bearish_panel(btc_15m["open"], btc_15m["high"], btc_15m["low"], btc_15m["close"])
    L15m_panel = btc_15m_bear  # already at 15m

    # ---- Index all per-asset candles --------------------------------------
    print("indexing per-asset candle arrays...", flush=True)
    cs = candles.sort_values(["asset", "timestamp"])
    by_symbol = {}
    for asset, sub in cs.groupby("asset", sort=False):
        ts = pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(None).to_numpy()
        by_symbol[asset] = (
            ts,
            sub["open"].to_numpy(dtype=float),
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float),
            sub["close"].to_numpy(dtype=float),
        )

    # ---- Re-simulate each variant -----------------------------------------
    variants = {
        "E0_baseline_no_regime_exit": None,
        "E1_exit_on_btc_1h_flip":     L1h_panel,
        "E2_exit_on_btc_daily_flip":  Ldaily_panel,
        "E3_exit_on_btc_15m_flip":    L15m_panel,
    }

    sweep = {}
    for name, regime_panel in variants.items():
        print(f"\n  re-simulating {name}...", flush=True)
        # Reindex regime panel to BTC's 15m index and convert to a dict keyed by ts
        # for fast per-asset lookup.
        if regime_panel is not None:
            regime_naive = regime_panel.reindex(btc_15m.index)
            regime_naive.index = pd.to_datetime(regime_naive.index, utc=True).tz_convert(None)
            regime_at_ts = regime_naive.fillna(False).astype(bool)
        else:
            regime_at_ts = None

        rows = []
        for r in short.itertuples(index=False):
            sym = r.symbol
            if sym not in by_symbol:
                continue
            ts_arr, open_a, high_a, low_a, close_a = by_symbol[sym]
            et = pd.Timestamp(r.entry_time)
            if et.tzinfo is not None:
                et = et.tz_convert(None)
            i = int(np.searchsorted(ts_arr, np.datetime64(et)))
            if i >= len(ts_arr):
                continue
            # Build regime flag aligned to THIS ASSET's bars.
            if regime_at_ts is not None:
                regime_flag = regime_at_ts.reindex(
                    pd.DatetimeIndex(ts_arr), method="ffill"
                ).fillna(False).to_numpy().astype(bool)
            else:
                regime_flag = None
            t = _simulate_short_with_regime_exit(
                close_a, high_a, low_a, open_a, i,
                regime_bearish_flag=regime_flag,
            )
            if t is None:
                continue
            t["symbol"] = sym
            t["entry_time"] = str(r.entry_time)
            t["pos_scale"] = float(r.pos_scale)
            t["period"] = r.period
            t["sized_pnl"] = t["pnl_pct"] * t["pos_scale"]
            rows.append(t)
        sim_df = pd.DataFrame(rows)
        train_pnls = sim_df[sim_df["period"] == "train"]["sized_pnl"].to_numpy()
        oot_pnls = sim_df[sim_df["period"] == "oot"]["sized_pnl"].to_numpy()
        wf = _walkforward(sim_df)
        mc = _monte_carlo(oot_pnls)
        train_m = _cell(train_pnls)
        oot_m = _cell(oot_pnls)
        # Exit-reason histogram
        exit_counts = sim_df["exit_reason"].value_counts().to_dict()
        sweep[name] = {
            "train_metrics": train_m,
            "oot_metrics": oot_m,
            "walkforward": wf,
            "monte_carlo_oot": mc,
            "exit_reason_counts": {k: int(v) for k, v in exit_counts.items()},
        }
        print(f"    TRAIN n={train_m['trades']:>5}  PF={train_m['pf']:.3f}  "
              f"OOT n={oot_m['trades']:>5}  PF={oot_m['pf']:.3f}  "
              f"WF p5={wf.get('p5', float('nan')):.3f}  p50={wf.get('p50', float('nan')):.3f}  "
              f"P(loss)={wf.get('p_lt_1', float('nan'))*100:.1f}%  "
              f"MC p5={mc.get('p5', float('nan')):.3f}", flush=True)
        print(f"    exit reasons: {exit_counts}", flush=True)

    # ---- Lock winner on TRAIN PF -----------------------------------------
    locked_name = max(
        sweep,
        key=lambda k: (
            sweep[k]["train_metrics"]["pf"]
            if np.isfinite(sweep[k]["train_metrics"]["pf"]) else -np.inf
        ),
    )
    locked = sweep[locked_name]
    print(f"\nLOCKED on TRAIN PF: {locked_name}", flush=True)
    print(f"  TRAIN: n={locked['train_metrics']['trades']:>5}  PF={locked['train_metrics']['pf']:.3f}",
          flush=True)
    print(f"  OOT:   n={locked['oot_metrics']['trades']:>5}  PF={locked['oot_metrics']['pf']:.3f}",
          flush=True)
    print(f"  WF p5={locked['walkforward'].get('p5', float('nan')):.3f}  "
          f"p50={locked['walkforward'].get('p50', float('nan')):.3f}  "
          f"P(loss)={locked['walkforward'].get('p_lt_1', float('nan'))*100:.1f}%",
          flush=True)
    print(f"  MC p5={locked['monte_carlo_oot'].get('p5', float('nan')):.3f}", flush=True)

    oot_pf = locked["oot_metrics"]["pf"]
    ship_gate = {
        "oot_pf_ge_130": bool(np.isfinite(oot_pf) and oot_pf >= 1.30),
        "oot_n_ge_200": bool(locked["oot_metrics"]["trades"] >= 200),
        "wf_p5_gt_1": bool(locked["walkforward"].get("p5", -1) > 1.0),
        "mc_p5_gt_1": bool(np.isfinite(locked["monte_carlo_oot"].get("p5", float("nan")))
                            and locked["monte_carlo_oot"].get("p5", -1) > 1.0),
    }
    ship_gate["overall"] = all(ship_gate.values())
    print("\n=== Ship gate at locked variant ===")
    for k, v in ship_gate.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  → {'SHIP' if ship_gate['overall'] else 'DO NOT SHIP'}")

    out = {
        "variants": list(variants.keys()),
        "sweep_results": sweep,
        "locked": {"name": locked_name, "ship_gate": ship_gate},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
