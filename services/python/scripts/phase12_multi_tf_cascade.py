"""Phase 12 — hierarchical multi-timeframe cascade gate.

Stacks three coarser-than-15m AND-gates on top of v2's existing per-trade
classifier + sizing:

  L1 — BTC daily indicators all agree on direction
  L2 — BTC 1h    indicators all agree on direction
  L3 — coin 4h   indicators all agree on direction (per-asset)
  L4 — coin 15m  v2 ML threshold + sizing (already in v2 trades.csv)

A trade fires iff all enabled layers agree. We sweep through 4 variants
(C0/C1/C2/C3) to see how many trades each layer kills and how the
ship-gate metrics move.

This is a re-aggregation script — no retraining, no candle re-loop. The
v2 trades.csv supplies the entries; we just compute the cascade booleans
at each entry_time and AND them with v2's pass.

Run from services/python/:
    python scripts/phase12_multi_tf_cascade.py
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
OUT_PATH = V2_RUN_DIR / "phase12_multi_tf_cascade.json"

TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
WF_FOLD_MONTHS = 3
MC_N_RESAMPLES = 5000

# --- Indicator parameters (textbook defaults, NOT tuned on this data) -------

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50    # only used at L1 (BTC daily)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH = 9, 3, 3


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


def _all_bearish_panel(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series,
    *, include_ema_slow: bool = False,
) -> pd.Series:
    """Return a boolean Series indexed by the input timestamps where True
    means EMA + RSI + MACD + KDJ all agree on bearish.

    All indicators shifted by 1 of their own bar before being evaluated to
    enforce no-lookahead at the original timestamp.
    """
    ema_fast = _ema(close, EMA_FAST).shift(1)
    ema_mid = _ema(close, EMA_MID).shift(1)
    ema_cond = (close.shift(1) < ema_fast) & (ema_fast < ema_mid)
    if include_ema_slow:
        ema_slow = _ema(close, EMA_SLOW).shift(1)
        ema_cond = ema_cond & (ema_mid < ema_slow)

    rsi_v = _rsi(close, RSI_PERIOD).shift(1)
    rsi_cond = rsi_v < 50.0

    md = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_cond = (md["macd"].shift(1) < md["signal"].shift(1))

    kdj_v = _kdj(high, low, close, n=KDJ_N, k_smooth=KDJ_K_SMOOTH, d_smooth=KDJ_D_SMOOTH)
    kdj_cond = (kdj_v["k"].shift(1) < kdj_v["d"].shift(1))

    return (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)


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
    ].copy()
    print(f"  SHORT ml_on+sized entries: {len(short):,}", flush=True)

    # ---- Load BTC candles ------------------------------------------------
    print(f"\nloading BTC + per-asset candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)

    # ---- L1: BTC daily ---------------------------------------------------
    print("[L1] computing BTC daily all-bearish panel...", flush=True)
    btc_daily = btc_15m.resample("1D", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    btc_daily_bear = _all_bearish_panel(
        btc_daily["open"], btc_daily["high"], btc_daily["low"], btc_daily["close"],
        include_ema_slow=True,
    )
    L1_panel_15m = btc_daily_bear.reindex(btc_15m.index, method="ffill").fillna(False)

    # ---- L2: BTC 1h ------------------------------------------------------
    print("[L2] computing BTC 1h all-bearish panel...", flush=True)
    btc_1h = btc_15m.resample("1h", label="right", closed="right").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
    }).dropna()
    btc_1h_bear = _all_bearish_panel(
        btc_1h["open"], btc_1h["high"], btc_1h["low"], btc_1h["close"],
    )
    L2_panel_15m = btc_1h_bear.reindex(btc_15m.index, method="ffill").fillna(False)

    # ---- L3: coin 4h (per asset) ----------------------------------------
    print("[L3] computing per-asset coin 4h all-bearish panels...", flush=True)
    L3_panels: dict[str, pd.Series] = {}
    assets_seen = sorted(short["symbol"].unique())
    for i, sym in enumerate(assets_seen, 1):
        sub = candles[candles["asset"] == sym].sort_values("timestamp").set_index("timestamp")
        if len(sub) < 200:
            continue
        coin_4h = sub[["open", "high", "low", "close"]].astype(float).resample(
            "4h", label="right", closed="right"
        ).agg({"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        coin_4h_bear = _all_bearish_panel(
            coin_4h["open"], coin_4h["high"], coin_4h["low"], coin_4h["close"],
        )
        L3_panels[sym] = coin_4h_bear
        if i % 50 == 0:
            print(f"  L3 panels built: {i}/{len(assets_seen)}", flush=True)

    # ---- Per-trade lookup -----------------------------------------------
    print("\nlooking up cascade booleans per trade...", flush=True)
    short["L1_pass"] = L1_panel_15m.reindex(short["entry_time"]).fillna(False).to_numpy()
    short["L2_pass"] = L2_panel_15m.reindex(short["entry_time"]).fillna(False).to_numpy()
    L3_vals = []
    for r in short.itertuples(index=False):
        panel = L3_panels.get(r.symbol)
        if panel is None or len(panel) == 0:
            L3_vals.append(False)
            continue
        try:
            v = panel.asof(r.entry_time)
            L3_vals.append(bool(v) if pd.notna(v) else False)
        except KeyError:
            L3_vals.append(False)
    short["L3_pass"] = L3_vals

    # ---- Layer pass-rate diagnostics -----------------------------------
    print("\n=== Per-layer pass rates (SHORT v2 ml_on+sized entries) ===")
    for period_name, mask in [
        ("ALL", pd.Series(True, index=short.index)),
        ("TRAIN", short["period"] == "train"),
        ("OOT", short["period"] == "oot"),
    ]:
        sub = short[mask]
        l1 = sub["L1_pass"].mean()
        l2 = sub["L2_pass"].mean()
        l3 = sub["L3_pass"].mean()
        l1_l3 = (sub["L1_pass"] & sub["L3_pass"]).mean()
        l1_l2_l3 = (sub["L1_pass"] & sub["L2_pass"] & sub["L3_pass"]).mean()
        print(
            f"  {period_name:<6} n={len(sub):>5}  "
            f"L1={l1*100:.1f}%  L2={l2*100:.1f}%  L3={l3*100:.1f}%  "
            f"L1∧L3={l1_l3*100:.1f}%  L1∧L2∧L3={l1_l2_l3*100:.1f}%",
            flush=True,
        )

    # ---- Sweep variants -------------------------------------------------
    print("\n=== Cascade variants ===")
    print(f"  {'name':<24}  {'TRAIN n':>7} {'TRAIN PF':>8}  "
          f"{'OOT n':>5} {'OOT PF':>7}  {'WF p5':>6} {'WF p50':>7} {'P(loss)':>8}  "
          f"{'MC p5':>6}", flush=True)

    variants = {
        "C0_v2_baseline":             pd.Series(True, index=short.index),
        "C1_L1_only":                 short["L1_pass"],
        "C2_L1_AND_L3":               short["L1_pass"] & short["L3_pass"],
        "C3_L1_AND_L2_AND_L3":        short["L1_pass"] & short["L2_pass"] & short["L3_pass"],
    }
    sweep = {}
    for name, mask in variants.items():
        sub = short[mask]
        train_pnls = sub[sub["period"] == "train"]["sized_pnl"].to_numpy()
        oot_pnls = sub[sub["period"] == "oot"]["sized_pnl"].to_numpy()
        wf = _walkforward(sub)
        mc = _monte_carlo(oot_pnls)
        train_m = _cell(train_pnls)
        oot_m = _cell(oot_pnls)
        sweep[name] = {
            "train_metrics": train_m,
            "oot_metrics": oot_m,
            "walkforward": wf,
            "monte_carlo_oot": mc,
        }
        print(f"  {name:<24}  {train_m['trades']:>7} {train_m['pf']:>8.3f}  "
              f"{oot_m['trades']:>5} {oot_m['pf']:>7.3f}  "
              f"{wf.get('p5', float('nan')):>6.3f} {wf.get('p50', float('nan')):>7.3f} "
              f"{wf.get('p_lt_1', float('nan'))*100:>7.1f}%  "
              f"{mc.get('p5', float('nan')):>6.3f}",
              flush=True)

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
    print(f"  TRAIN: n={locked['train_metrics']['trades']:>5}  "
          f"PF={locked['train_metrics']['pf']:.3f}", flush=True)
    print(f"  OOT:   n={locked['oot_metrics']['trades']:>5}  "
          f"PF={locked['oot_metrics']['pf']:.3f}", flush=True)
    print(f"  WF p5={locked['walkforward'].get('p5', float('nan')):.3f}  "
          f"p50={locked['walkforward'].get('p50', float('nan')):.3f}  "
          f"P(loss)={locked['walkforward'].get('p_lt_1', float('nan'))*100:.1f}%", flush=True)
    print(f"  MC p5={locked['monte_carlo_oot'].get('p5', float('nan')):.3f}", flush=True)

    # Ship gate at locked variant
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
        "indicator_params": {
            "ema_fast": EMA_FAST, "ema_mid": EMA_MID, "ema_slow": EMA_SLOW,
            "rsi_period": RSI_PERIOD,
            "macd_fast": MACD_FAST, "macd_slow": MACD_SLOW, "macd_signal": MACD_SIGNAL,
            "kdj_n": KDJ_N, "kdj_k_smooth": KDJ_K_SMOOTH, "kdj_d_smooth": KDJ_D_SMOOTH,
        },
        "layer_pass_rates": {
            period: {
                "n": int((short["period"] == p_filter).sum()) if p_filter else int(len(short)),
                "L1": float((short.loc[(short["period"] == p_filter) if p_filter else short.index, "L1_pass"]).mean()),
                "L2": float((short.loc[(short["period"] == p_filter) if p_filter else short.index, "L2_pass"]).mean()),
                "L3": float((short.loc[(short["period"] == p_filter) if p_filter else short.index, "L3_pass"]).mean()),
            }
            for period, p_filter in [("ALL", None), ("TRAIN", "train"), ("OOT", "oot")]
        },
        "sweep_results": sweep,
        "locked": {
            "name": locked_name,
            "ship_gate": ship_gate,
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
