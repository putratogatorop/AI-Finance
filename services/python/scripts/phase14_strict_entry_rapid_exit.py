"""Phase 14 — strict entry (BTC big-TF AND small-TF both bear) + rapid-rally exit.

Combines two ideas from the prior phases:
  - Phase 12 entry filter: only enter when BTC daily AND BTC 1h are both
    all-bearish (EMA + RSI + MACD + KDJ agreement on each TF).
  - Phase 13 exit filter: don't exit on every 1h bounce (too noisy);
    require CONFIRMATION = regime flip AND BTC moved fast (>+3% in 24h).

Variant matrix (entries from v2 SHORT ml_on+sized; gates layered on top):

  V0 — baseline (= Phase 5 v2 baseline; no extras)
  V1 — strict entry only (Phase 12 C3; no exit change)
  V2 — rapid-rally exit only (no entry change)
  V3 — strict entry + rapid-rally exit (full operationalization of user's intuition)

Run from services/python/:
    python scripts/phase14_strict_entry_rapid_exit.py
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
OUT_PATH = V2_RUN_DIR / "phase14_strict_entry_rapid_exit.json"

STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015

EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50    # only used at L1 (BTC daily)
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH = 9, 3, 3

# Rapid-rally exit thresholds
RAPID_RALLY_PCT = 0.03     # BTC moved > +3% in last 24h
RAPID_RALLY_LOOKBACK_15M = 96   # 24h on 15m grid

WF_FOLD_MONTHS = 3
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000


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


def _walkforward(df, pnl_col="sized_pnl"):
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


def _all_bearish_panel(open_, high, low, close, *, include_ema_slow=False):
    ema_fast = _ema(close, EMA_FAST).shift(1)
    ema_mid = _ema(close, EMA_MID).shift(1)
    ema_cond = (close.shift(1) < ema_fast) & (ema_fast < ema_mid)
    if include_ema_slow:
        ema_slow = _ema(close, EMA_SLOW).shift(1)
        ema_cond = ema_cond & (ema_mid < ema_slow)
    rsi_cond = _rsi(close, RSI_PERIOD).shift(1) < 50.0
    md = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_cond = md["macd"].shift(1) < md["signal"].shift(1)
    kdj_v = _kdj(high, low, close, n=KDJ_N, k_smooth=KDJ_K_SMOOTH, d_smooth=KDJ_D_SMOOTH)
    kdj_cond = kdj_v["k"].shift(1) < kdj_v["d"].shift(1)
    return (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)


def _simulate_short(close, high, low, open_, enter_bar, *,
                    rally_exit_flag=None):
    """Standard short simulator + optional rapid-rally exit.

    rally_exit_flag: bool ndarray length len(close); True at bar i means
      "BTC daily flipped bullish AT bar i AND BTC has moved >+3% over the
      prior 24h". When True at any bar between entry+1 and last_bar, exit
      at close[bar].
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
        if high[bar] >= sl_price:
            return {"pnl_pct": -STOP_LOSS_PCT - FRICTION_PCT,
                    "exit_reason": "stop_loss",
                    "bars_held": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            return {"pnl_pct": TAKE_PROFIT_PCT - FRICTION_PCT,
                    "exit_reason": "take_profit",
                    "bars_held": int(bar - enter_bar)}
        if rally_exit_flag is not None and bool(rally_exit_flag[bar]):
            exit_price = float(close[bar])
            return {"pnl_pct": (entry_price - exit_price) / entry_price - FRICTION_PCT,
                    "exit_reason": "rapid_rally_exit",
                    "bars_held": int(bar - enter_bar)}
    exit_bar = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[exit_bar])) / entry_price - FRICTION_PCT,
            "exit_reason": "timeout",
            "bars_held": int(exit_bar - enter_bar)}


def main():
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

    print("building BTC daily/1h all-bearish panels...", flush=True)
    btc_daily = btc_15m.resample("1D", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    btc_daily_bear = _all_bearish_panel(
        btc_daily["open"], btc_daily["high"], btc_daily["low"], btc_daily["close"],
        include_ema_slow=True,
    )
    Ldaily_panel_15m = btc_daily_bear.reindex(btc_15m.index, method="ffill").fillna(False)

    btc_1h = btc_15m.resample("1h", label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    btc_1h_bear = _all_bearish_panel(btc_1h["open"], btc_1h["high"], btc_1h["low"], btc_1h["close"])
    L1h_panel_15m = btc_1h_bear.reindex(btc_15m.index, method="ffill").fillna(False)

    # Strict entry mask: both daily AND 1h all-bearish
    strict_entry_15m = Ldaily_panel_15m & L1h_panel_15m

    # Per-trade strict-entry pass
    short["strict_entry_pass"] = strict_entry_15m.reindex(
        short["entry_time"]
    ).fillna(False).to_numpy()

    print(f"\n=== Strict-entry pass rates ===")
    for label, mask in [("ALL", pd.Series(True, index=short.index)),
                         ("TRAIN", short["period"] == "train"),
                         ("OOT", short["period"] == "oot")]:
        sub = short[mask]
        rate = sub["strict_entry_pass"].mean()
        print(f"  {label:<6}  n={len(sub):>5}  pass={rate*100:.1f}%  n_kept={int(sub['strict_entry_pass'].sum())}",
              flush=True)

    # ---- Rapid-rally exit panel: BTC daily flips bullish AT bar i AND BTC 24h ret > +3% ----
    print("\nbuilding rapid-rally exit panel (BTC daily flip + 24h move > +3%)...",
          flush=True)
    daily_now_bear = Ldaily_panel_15m.astype(bool)
    # "Flipped bullish AT bar i" means: at bar i, daily-bear is False AND at bar i-1 it was True.
    # Reindexing to per-asset bars later, so encode flip transition on the BTC 15m grid first:
    daily_was_bear = daily_now_bear.shift(1).fillna(False)
    daily_flipped_to_bull = (~daily_now_bear) & daily_was_bear
    # 24h BTC return
    btc_close_15m = btc_15m["close"]
    btc_24h_ret = btc_close_15m.pct_change(RAPID_RALLY_LOOKBACK_15M)
    rapid_rally_active = (btc_24h_ret > RAPID_RALLY_PCT).fillna(False).astype(bool)
    # Combined exit signal: at bar i, exit IFF daily is now bullish AND 24h move > +3%
    # We want to fire on persistent rally state, not just flip moment, so use:
    # rally_exit = (NOT daily_now_bear) AND rapid_rally_active
    rapid_exit_signal_15m = ((~daily_now_bear) & rapid_rally_active).astype(bool)
    print(f"  rapid_exit_signal active rate (over BTC 15m timeline): "
          f"{rapid_exit_signal_15m.mean()*100:.1f}%", flush=True)

    # Index rapid_exit signal by tz-naive datetime for per-asset reindex.
    rapid_exit_naive = rapid_exit_signal_15m.copy()
    rapid_exit_naive.index = pd.to_datetime(rapid_exit_naive.index, utc=True).tz_convert(None)

    # ---- Index per-asset candles --------------------------------------
    print("\nindexing per-asset candle arrays...", flush=True)
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

    def _simulate_variant(name, *, strict_entry, rapid_rally_exit):
        """Run a variant; return DataFrame of trades and metrics."""
        pool = short[short["strict_entry_pass"]] if strict_entry else short
        rows = []
        for r in pool.itertuples(index=False):
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
            if rapid_rally_exit:
                rally_flag = (
                    rapid_exit_naive
                    .reindex(pd.DatetimeIndex(ts_arr), method="ffill")
                    .fillna(False)
                    .to_numpy()
                    .astype(bool)
                )
            else:
                rally_flag = None
            t = _simulate_short(close_a, high_a, low_a, open_a, i, rally_exit_flag=rally_flag)
            if t is None:
                continue
            t["symbol"] = sym
            t["entry_time"] = str(r.entry_time)
            t["pos_scale"] = float(r.pos_scale)
            t["period"] = r.period
            t["sized_pnl"] = t["pnl_pct"] * t["pos_scale"]
            rows.append(t)
        return pd.DataFrame(rows)

    print("\n=== Running 4 variants ===")
    print(f"  {'name':<48}  {'TRAIN n':>7} {'TRAIN PF':>9}  "
          f"{'OOT n':>5} {'OOT PF':>7}  {'WF p5':>6} {'WF p50':>7} {'P(loss)':>8}  {'MC p5':>6}",
          flush=True)
    sweep = {}
    variants = [
        ("V0_baseline_no_extras",                      dict(strict_entry=False, rapid_rally_exit=False)),
        ("V1_strict_entry_only",                       dict(strict_entry=True,  rapid_rally_exit=False)),
        ("V2_rapid_rally_exit_only",                   dict(strict_entry=False, rapid_rally_exit=True)),
        ("V3_strict_entry_plus_rapid_rally_exit",      dict(strict_entry=True,  rapid_rally_exit=True)),
    ]
    for name, kwargs in variants:
        sim = _simulate_variant(name, **kwargs)
        train_pnls = sim[sim["period"] == "train"]["sized_pnl"].to_numpy()
        oot_pnls = sim[sim["period"] == "oot"]["sized_pnl"].to_numpy()
        wf = _walkforward(sim)
        mc = _monte_carlo(oot_pnls)
        train_m = _cell(train_pnls)
        oot_m = _cell(oot_pnls)
        exit_counts = sim["exit_reason"].value_counts().to_dict() if len(sim) else {}
        sweep[name] = {
            "train_metrics": train_m, "oot_metrics": oot_m,
            "walkforward": wf, "monte_carlo_oot": mc,
            "exit_reason_counts": {k: int(v) for k, v in exit_counts.items()},
        }
        print(f"  {name:<48}  {train_m['trades']:>7} {train_m['pf']:>9.3f}  "
              f"{oot_m['trades']:>5} {oot_m['pf']:>7.3f}  "
              f"{wf.get('p5', float('nan')):>6.3f} {wf.get('p50', float('nan')):>7.3f} "
              f"{wf.get('p_lt_1', float('nan'))*100:>7.1f}%  "
              f"{mc.get('p5', float('nan')):>6.3f}",
              flush=True)
        print(f"    exits: {exit_counts}", flush=True)

    # Lock on TRAIN PF
    locked_name = max(
        sweep, key=lambda k: (
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
        "variants": [n for n, _ in variants],
        "rapid_rally_pct": RAPID_RALLY_PCT,
        "rapid_rally_lookback_15m": RAPID_RALLY_LOOKBACK_15M,
        "sweep_results": sweep,
        "locked": {"name": locked_name, "ship_gate": ship_gate},
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
