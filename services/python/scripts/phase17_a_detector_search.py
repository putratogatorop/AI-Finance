"""Phase 17.A — Controlled head-to-head SHORT detector search.

Goal: find a SHORT detector composition that lifts ship-gate metrics above
the Phase-16 locked baseline (D1 + sign-locked sizing + v3 LR + rapid-rally
exit) which currently sits at OOT PF=2.607, WF p5=0.719 — highest ever, but
still below the WF p5 > 1.0 ship floor.

This phase 17.A test does NOT use a classifier (no LightGBM, no LR). Pure
detector + sign-locked Candidate-B sizing + Phase-14 V2 rapid-rally exit.
We compare the RAW detector quality across 5 variants using the same
SL=5% / TP=15% / timeout=672 / friction=0.0015 simulator.

5 SHORT detectors:

  D1a — ATR-only:
        drop_in_atr_units >= 2.5  (no vol gate, no red-bar requirement)

  D1  — current baseline (= Phase 15 D1):
        drop_in_atr_units >= 2.5
      & vol_ratio >= 2.0
      & close < open  (red current bar)

  D1c — ATR + vol + lower-low confirmation:
        D1
      & low[i] < min(low[i-3], low[i-2], low[i-1])

  D1d — ATR + vol + bars-since-32-high gate (fresh breakdown):
        D1
      & argmax(high[i-31:i+1]) implies high happened within last 4 bars
        (i.e. offset_to_max <= 4)

  D1e — ATR + vol + breadth filter:
        D1
      & concurrent universe down-breadth (4h returns) >= 0.6
        (i.e. >=60% of currently-listed universe is down on 4h returns
         at the entry timestamp; same panel as
         src/ml/bigmover_combined/features.py::_ret_4h_panel)

PRE-COMMITTED DECISION RULE
---------------------------
A variant "wins" iff ALL of the following hold:
  (1) TRAIN PF is the highest among all 5 variants
  (2) OOT n >= 200
  (3) OOT PF >= 2.064            (Phase-15 D1 raw OOT PF baseline)
  (4) WF p5  >= 0.605            (Phase-15 D1 raw WF p5  baseline)

Lock the winner using TRAIN PF only (NEVER on OOT). Print PASS/FAIL per
gate condition and final decision: "ADOPT <variant>" or "STAY ON D1".

HARD CONSTRAINTS
----------------
- No LightGBM. No model training in this strand.
- Pre-committed thresholds; no post-hoc tuning.
- No live changes. No git commits.
- Reproducibility: re-run metrics on the same trades dataframe and assert
  byte-identical metrics for the top variant.

Run from services/python/:
    uv run python scripts/phase17_a_detector_search.py
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
UNIVERSE_PATH = REPO_ROOT / "data/snapshots" / f"universe_{SNAPSHOT_DATE}.parquet"
OUT_PATH = REPO_ROOT / "services/python/results" / "phase17_a_detector_search.json"

# --- Simulator constants (identical to Phase 15/16) -------------------------

STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015

VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
COOLDOWN_BARS = 96

# Sizing knobs (Candidate B BTC trend score)
BTC_TREND_K = 0.005255

# Rapid-rally exit (from Phase 14 V2)
RAPID_RALLY_PCT = 0.03
RAPID_RALLY_LOOKBACK_15M = 96

# Cross-sectional breadth panel — match features.py BREADTH_4H_LOOKBACK
BREADTH_4H_LOOKBACK = 16  # 16 * 15m = 4h

WF_FOLD_MONTHS = 3
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000

TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")

# --- Phase-15 D1 raw baseline (used in pre-committed gate thresholds) -------
# Source: services/python/results/phase15_atr_detector_vs_volume.json
PHASE15_D1_OOT_PF_BASELINE = 2.064
PHASE15_D1_WF_P5_BASELINE = 0.605

# --- Bars-since-high gate constant (matches features.py bars_since_high32) --
BARS_SINCE_HIGH_LOOKBACK = 32
BARS_SINCE_HIGH_MAX = 4  # fresh breakdown: high within last 4 bars

# --- Lower-low confirmation constants --------------------------------------
LOWER_LOW_LOOKBACK = 3  # low[i] < min(low[i-3], low[i-2], low[i-1])

# --- Breadth gate constants -------------------------------------------------
BREADTH_THRESHOLD = 0.6  # >=60% of universe down on 4h returns


# --- Metric helpers (verbatim from Phase 15) --------------------------------


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


def _compute_all_metrics(sim: pd.DataFrame) -> dict:
    """Compute the full metric block from a trades DataFrame.
    Used twice: once during the search, once for the reproducibility check."""
    sim = sim.copy()
    sim["entry_time"] = pd.to_datetime(sim["entry_time"], utc=True)
    sim["period"] = np.where(sim["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
    train_pnls = sim[sim["period"] == "train"]["sized_pnl"].to_numpy()
    oot_pnls = sim[sim["period"] == "oot"]["sized_pnl"].to_numpy()
    wf = _walkforward(sim, "sized_pnl")
    mc = _monte_carlo(oot_pnls)
    train_m = _cell(train_pnls)
    oot_m = _cell(oot_pnls)
    exit_counts = sim["exit"].value_counts().to_dict()
    return {
        "train_metrics": train_m,
        "oot_metrics": oot_m,
        "walkforward": wf,
        "monte_carlo_oot": mc,
        "exit_counts": {k: int(v) for k, v in exit_counts.items()},
    }


# --- BTC trend / rapid-exit panels (verbatim from Phase 15) -----------------


def _build_btc_trend_score_15m(candles_15m: pd.DataFrame) -> pd.Series:
    """Candidate B BTC trend score, range [-1, +1]. Reuse from Phase 9."""
    btc = candles_15m["close"].astype(float)
    btc_4h = btc.resample("4h", label="right", closed="right").last()
    md = _macd(btc_4h, 12, 26, 9)
    spread = ((md["macd"] - md["signal"]).shift(1) / btc_4h.shift(1))
    score_4h = spread.clip(-BTC_TREND_K, BTC_TREND_K) / BTC_TREND_K
    return score_4h.reindex(btc.index, method="ffill")


def _build_rapid_exit_panel(candles_15m: pd.DataFrame) -> pd.Series:
    """Phase 14 V2 rapid-rally exit panel."""
    close = candles_15m["close"].astype(float)
    high = candles_15m["high"].astype(float)
    low = candles_15m["low"].astype(float)
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
    btc_24h_ret = close.pct_change(96)
    rapid_exit = ((~daily_bear_15m) & (btc_24h_ret > RAPID_RALLY_PCT)).fillna(False).astype(bool)
    return rapid_exit


# --- Cross-sectional breadth panel (mirrors features._ret_4h_panel) ---------


def _build_breadth_short_panel(
    candles: pd.DataFrame,
    universe_listed_since: dict[str, pd.Timestamp] | None = None,
) -> pd.Series:
    """Build a 15m-indexed Series of `frac of currently-listed universe whose
    4h return < 0`. This mirrors how
    src/ml/bigmover_combined/features.py::_ret_4h_panel computes
    `concurrent_dir_breadth` for direction='short'.
    """
    ret_4h_panel = (
        candles.sort_values(["asset", "timestamp"])
        .assign(
            ret4h=lambda df: df.groupby("asset")["close"].transform(
                lambda s: s.pct_change(BREADTH_4H_LOOKBACK)
            )
        )
        .pivot(index="timestamp", columns="asset", values="ret4h")
    )
    if universe_listed_since:
        mask = pd.DataFrame(False, index=ret_4h_panel.index, columns=ret_4h_panel.columns)
        for sym in ret_4h_panel.columns:
            listed = universe_listed_since.get(sym)
            mask[sym] = (ret_4h_panel.index >= listed) if listed is not None else True
        ret_4h_panel = ret_4h_panel.where(mask)

    # For each timestamp: fraction of non-NaN columns with ret < 0.
    notna = ret_4h_panel.notna().sum(axis=1)
    down = (ret_4h_panel < 0).sum(axis=1)
    breadth = (down / notna).where(notna > 0)
    return breadth


# --- Detector implementations ----------------------------------------------


def _common_d1_arrays(close, high, low, open_, volume):
    """Pre-compute shared rolling arrays used by D1, D1a, D1c, D1d, D1e."""
    n = len(close)
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().to_numpy()
    rolling_high = pd.Series(high).rolling(PRICE_LOOKBACK, min_periods=PRICE_LOOKBACK).max().to_numpy()
    atr14 = _atr(pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD).to_numpy()
    return vol_ma, rolling_high, atr14, n


def _detect_short_d1a_atr_only(close, high, low, open_, volume) -> np.ndarray:
    """D1a — ATR-only: drop_in_atr_units >= 2.5. No vol gate, no red bar."""
    vol_ma, rolling_high, atr14, n = _common_d1_arrays(close, high, low, open_, volume)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + ATR_PERIOD:
        return out
    for i in range(PRICE_LOOKBACK + VOL_MA_PERIOD, n):
        if not (atr14[i] > 0):
            continue
        if rolling_high[i] <= 0:
            continue
        drop_in_atr = (rolling_high[i] - close[i]) / atr14[i]
        if drop_in_atr < 2.5:
            continue
        out[i] = True
    return out


def _detect_short_d1_baseline(close, high, low, open_, volume) -> np.ndarray:
    """D1 — current baseline (= Phase 15 D1).
       drop_in_atr_units >= 2.5 AND vol_ratio >= 2.0 AND close < open."""
    vol_ma, rolling_high, atr14, n = _common_d1_arrays(close, high, low, open_, volume)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + ATR_PERIOD:
        return out
    for i in range(PRICE_LOOKBACK + VOL_MA_PERIOD, n):
        if not (atr14[i] > 0):
            continue
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < 2.0:
            continue
        if rolling_high[i] <= 0:
            continue
        drop_in_atr = (rolling_high[i] - close[i]) / atr14[i]
        if drop_in_atr < 2.5:
            continue
        if close[i] >= open_[i]:  # require red current bar (directional)
            continue
        out[i] = True
    return out


def _detect_short_d1c_lower_low(close, high, low, open_, volume) -> np.ndarray:
    """D1c — D1 AND low[i] < min(low[i-3], low[i-2], low[i-1])."""
    vol_ma, rolling_high, atr14, n = _common_d1_arrays(close, high, low, open_, volume)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + ATR_PERIOD:
        return out
    for i in range(PRICE_LOOKBACK + VOL_MA_PERIOD, n):
        if i < LOWER_LOW_LOOKBACK:
            continue
        if not (atr14[i] > 0):
            continue
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < 2.0:
            continue
        if rolling_high[i] <= 0:
            continue
        drop_in_atr = (rolling_high[i] - close[i]) / atr14[i]
        if drop_in_atr < 2.5:
            continue
        if close[i] >= open_[i]:
            continue
        prior_min_low = min(low[i - 3], low[i - 2], low[i - 1])
        if not (low[i] < prior_min_low):
            continue
        out[i] = True
    return out


def _detect_short_d1d_fresh_breakdown(close, high, low, open_, volume) -> np.ndarray:
    """D1d — D1 AND offset of max-high within last 32 bars <= 4.

    `bars_since_high32` is computed (matching features.py) as
    32 - 1 - argmax(high[i-31:i+1]) (i.e. how many bars ago the peak was).
    We require this to be <= BARS_SINCE_HIGH_MAX (=4) — fresh breakdown.
    """
    vol_ma, rolling_high, atr14, n = _common_d1_arrays(close, high, low, open_, volume)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + ATR_PERIOD:
        return out
    high_arr = np.asarray(high, dtype=float)
    for i in range(PRICE_LOOKBACK + VOL_MA_PERIOD, n):
        if not (atr14[i] > 0):
            continue
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < 2.0:
            continue
        if rolling_high[i] <= 0:
            continue
        drop_in_atr = (rolling_high[i] - close[i]) / atr14[i]
        if drop_in_atr < 2.5:
            continue
        if close[i] >= open_[i]:
            continue
        lo = max(0, i - (BARS_SINCE_HIGH_LOOKBACK - 1))
        window = high_arr[lo:i + 1]
        # offset_to_max: how many bars ago the max-high was (0 = current bar).
        offset_to_max = (len(window) - 1) - int(np.argmax(window))
        if offset_to_max > BARS_SINCE_HIGH_MAX:
            continue
        out[i] = True
    return out


def _detect_short_d1e_breadth(close, high, low, open_, volume,
                               *, breadth_at_bar: np.ndarray | None = None) -> np.ndarray:
    """D1e — D1 AND universe-concurrent down-breadth >= BREADTH_THRESHOLD.

    `breadth_at_bar` is the per-bar breadth value already aligned to this
    asset's bars (passed in by the runner because it depends on the global
    panel and the asset's timestamp index)."""
    vol_ma, rolling_high, atr14, n = _common_d1_arrays(close, high, low, open_, volume)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + ATR_PERIOD:
        return out
    if breadth_at_bar is None:
        return out
    for i in range(PRICE_LOOKBACK + VOL_MA_PERIOD, n):
        if not (atr14[i] > 0):
            continue
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < 2.0:
            continue
        if rolling_high[i] <= 0:
            continue
        drop_in_atr = (rolling_high[i] - close[i]) / atr14[i]
        if drop_in_atr < 2.5:
            continue
        if close[i] >= open_[i]:
            continue
        b = breadth_at_bar[i] if i < len(breadth_at_bar) else float("nan")
        if not np.isfinite(b) or b < BREADTH_THRESHOLD:
            continue
        out[i] = True
    return out


# --- Simulator (verbatim from Phase 15) ------------------------------------


def _simulate_short(close, high, low, open_, enter_bar, *, rally_flag=None):
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
            return {"pnl_pct": -STOP_LOSS_PCT - FRICTION_PCT, "exit": "sl",
                    "bars": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            return {"pnl_pct": TAKE_PROFIT_PCT - FRICTION_PCT, "exit": "tp",
                    "bars": int(bar - enter_bar)}
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (entry_price - ep) / entry_price - FRICTION_PCT, "exit": "rapid",
                    "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[eb])) / entry_price - FRICTION_PCT,
            "exit": "timeout", "bars": int(eb - enter_bar)}


# --- Detector-runner with sign-locked Candidate-B sizing -------------------


def _run_detector(
    candles: pd.DataFrame,
    detector_fn,
    *,
    btc_score_15m: pd.Series,
    rapid_exit_15m: pd.Series,
    breadth_panel_naive: pd.Series | None = None,
) -> pd.DataFrame:
    """Run detector across all assets, sized by BTC trend score (sign-locked
    SHORT, 1.5x cap), with rapid-rally exit applied. Returns trade DataFrame."""
    rows = []
    btc_score_naive = btc_score_15m.copy()
    btc_score_naive.index = pd.to_datetime(btc_score_naive.index, utc=True).tz_convert(None)
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)

    needs_breadth = breadth_panel_naive is not None

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

        if needs_breadth:
            breadth_at_bar = (
                breadth_panel_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill")
                .to_numpy(dtype=float)
            )
            triggers = detector_fn(close, high, low, open_, volume,
                                    breadth_at_bar=breadth_at_bar)
        else:
            triggers = detector_fn(close, high, low, open_, volume)

        # Apply cooldown
        last_entry = -10**9
        entry_bars = []
        for i in range(len(triggers)):
            if not triggers[i]:
                continue
            if i - last_entry < COOLDOWN_BARS:
                continue
            entry_bars.append(i)
            last_entry = i

        # BTC score / rally flags per timestamp
        btc_score = btc_score_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill").to_numpy(dtype=float)
        rally = rapid_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill").fillna(False).to_numpy().astype(bool)

        for i in entry_bars:
            score = btc_score[i] if i < len(btc_score) else float("nan")
            # Sign-locked sizing for SHORT: only size if score < 0
            if not np.isfinite(score) or score >= 0:
                continue
            pos_scale = min(1.5 * abs(score), 1.5)
            if pos_scale <= 0:
                continue
            t = _simulate_short(close, high, low, open_, i, rally_flag=rally)
            if t is None:
                continue
            t["sized_pnl"] = t["pnl_pct"] * pos_scale
            t["pos_scale"] = pos_scale
            t["asset"] = asset
            t["entry_time"] = pd.Timestamp(
                sub["timestamp"].iloc[i + 1] if i + 1 < len(sub) else sub["timestamp"].iloc[i]
            )
            t["btc_score"] = float(score)
            rows.append(t)
        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(assets)} assets processed, trades={len(rows):,}",
                  flush=True)
    return pd.DataFrame(rows)


# --- Main -------------------------------------------------------------------


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

    print("building BTC trend score + rapid-exit panels...", flush=True)
    btc_score_15m = _build_btc_trend_score_15m(btc_15m)
    rapid_exit_15m = _build_rapid_exit_panel(btc_15m)

    # Universe listing dates (mirror features.py FeatureContext usage).
    universe_listed_since: dict[str, pd.Timestamp] = {}
    if UNIVERSE_PATH.exists():
        try:
            uni = pd.read_parquet(UNIVERSE_PATH)
            sym_col = "symbol" if "symbol" in uni.columns else (
                "asset" if "asset" in uni.columns else None
            )
            list_col = "listed_since" if "listed_since" in uni.columns else (
                "listed_at" if "listed_at" in uni.columns else None
            )
            if sym_col and list_col:
                for _, r in uni.iterrows():
                    ts = r[list_col]
                    if pd.notna(ts):
                        universe_listed_since[str(r[sym_col])] = pd.Timestamp(ts, tz="UTC") \
                            if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert("UTC")
            print(f"  loaded universe listed_since for {len(universe_listed_since)} symbols",
                  flush=True)
        except Exception as e:
            print(f"  could not load universe parquet: {e}; using ts >= 0 mask", flush=True)

    print("building cross-sectional breadth panel (4h returns < 0 fraction)...",
          flush=True)
    breadth_short_15m = _build_breadth_short_panel(
        candles, universe_listed_since=universe_listed_since or None
    )
    breadth_panel_naive = breadth_short_15m.copy()
    breadth_panel_naive.index = pd.to_datetime(
        breadth_panel_naive.index, utc=True
    ).tz_convert(None)

    detectors = [
        ("D1a_atr_only",        _detect_short_d1a_atr_only,       False),
        ("D1_baseline",         _detect_short_d1_baseline,        False),
        ("D1c_lower_low",       _detect_short_d1c_lower_low,      False),
        ("D1d_fresh_breakdown", _detect_short_d1d_fresh_breakdown,False),
        ("D1e_breadth",         _detect_short_d1e_breadth,        True),
    ]

    sweep: dict[str, dict] = {}
    trades_cache: dict[str, pd.DataFrame] = {}

    print("\n=== Running 5 SHORT detectors (same exits + sizing + rapid_rally) ===")
    for name, fn, needs_breadth in detectors:
        print(f"\n  detector: {name}", flush=True)
        sim = _run_detector(
            candles,
            fn,
            btc_score_15m=btc_score_15m,
            rapid_exit_15m=rapid_exit_15m,
            breadth_panel_naive=breadth_panel_naive if needs_breadth else None,
        )
        if len(sim) == 0:
            print("    NO TRADES", flush=True)
            sweep[name] = {"trades": 0}
            trades_cache[name] = sim
            continue
        trades_cache[name] = sim.copy()
        m = _compute_all_metrics(sim)
        sweep[name] = m
        print(
            f"    TRAIN n={m['train_metrics']['trades']:>5} PF={m['train_metrics']['pf']:.3f}  "
            f"OOT n={m['oot_metrics']['trades']:>5} PF={m['oot_metrics']['pf']:.3f}  "
            f"WF p5={m['walkforward'].get('p5', float('nan')):.3f}  "
            f"p50={m['walkforward'].get('p50', float('nan')):.3f}  "
            f"P(loss)={m['walkforward'].get('p_lt_1', float('nan'))*100:.1f}%  "
            f"MC p5={m['monte_carlo_oot'].get('p5', float('nan')):.3f}",
            flush=True,
        )
        print(f"    exits: {m['exit_counts']}", flush=True)

    # --- Pre-committed decision rule: lock on TRAIN PF only -----------------
    valid = {k: v for k, v in sweep.items() if v.get("train_metrics", {}).get("trades", 0) > 0}
    if valid:
        locked_name = max(valid, key=lambda k: (
            valid[k]["train_metrics"]["pf"]
            if np.isfinite(valid[k]["train_metrics"]["pf"]) else -np.inf
        ))
        locked = valid[locked_name]
        print(f"\nLOCKED on TRAIN PF: {locked_name}", flush=True)
        print(f"  TRAIN: n={locked['train_metrics']['trades']:>5}  PF={locked['train_metrics']['pf']:.3f}",
              flush=True)
        print(f"  OOT:   n={locked['oot_metrics']['trades']:>5}  PF={locked['oot_metrics']['pf']:.3f}",
              flush=True)

        oot_pf = locked["oot_metrics"]["pf"]
        oot_n = locked["oot_metrics"]["trades"]
        wf_p5 = locked["walkforward"].get("p5", float("nan"))
        # The 4 pre-committed gate conditions
        train_pf_is_max = True  # by construction, locked is argmax(train PF)
        gate = {
            "train_pf_is_max":          bool(train_pf_is_max),
            "oot_n_ge_200":             bool(oot_n >= 200),
            "oot_pf_ge_phase15_baseline": bool(np.isfinite(oot_pf)
                                                and oot_pf >= PHASE15_D1_OOT_PF_BASELINE),
            "wf_p5_ge_phase15_baseline":  bool(np.isfinite(wf_p5)
                                                and wf_p5 >= PHASE15_D1_WF_P5_BASELINE),
        }
        gate["overall_win"] = all(gate.values())

        print("\n=== Pre-committed decision-rule gates (lock + adopt) ===")
        for k, v in gate.items():
            print(f"  {k}: {'PASS' if v else 'FAIL'}")

        # Final decision string
        if locked_name == "D1_baseline":
            # Argmax already landed on baseline; can never beat itself.
            decision_str = "STAY ON D1"
        elif gate["overall_win"]:
            decision_str = f"ADOPT {locked_name}"
        else:
            decision_str = "STAY ON D1"
        print(f"\nFINAL DECISION: {decision_str}")
    else:
        gate = None
        locked_name = None
        decision_str = "STAY ON D1"
        print("\nFINAL DECISION: STAY ON D1 (no valid detector produced trades)")

    # --- Reproducibility check on the locked detector's trade frame ---------
    repro_ok = True
    repro_detail: dict = {}
    if locked_name and locked_name in trades_cache and len(trades_cache[locked_name]) > 0:
        first = sweep[locked_name]
        second = _compute_all_metrics(trades_cache[locked_name])
        # Compare the entire metric dict via JSON serialization (byte-identical).
        a = json.dumps(first, sort_keys=True, default=str)
        b = json.dumps(second, sort_keys=True, default=str)
        repro_ok = (a == b)
        repro_detail = {
            "variant": locked_name,
            "first_hash": hash(a),
            "second_hash": hash(b),
            "byte_identical": bool(repro_ok),
        }
        if repro_ok:
            print(f"\nREPRODUCIBILITY: PASS (variant={locked_name})")
        else:
            print(f"\nREPRODUCIBILITY: FAIL (variant={locked_name})")
            # Surface a short diff
            for k in first:
                if json.dumps(first[k], sort_keys=True, default=str) \
                        != json.dumps(second.get(k), sort_keys=True, default=str):
                    print(f"  diff in {k}:")
                    print(f"    first:  {first[k]}")
                    print(f"    second: {second.get(k)}")
    else:
        print("\nREPRODUCIBILITY: SKIPPED (no locked variant with trades)")

    out = {
        "phase": "17.A",
        "snapshot_date": SNAPSHOT_DATE,
        "detectors": [n for n, _, _ in detectors],
        "phase15_d1_baseline_used_in_gates": {
            "oot_pf": PHASE15_D1_OOT_PF_BASELINE,
            "wf_p5":  PHASE15_D1_WF_P5_BASELINE,
        },
        "sweep_results": sweep,
        "locked": {
            "name": locked_name,
            "decision": decision_str if locked_name else "STAY ON D1",
            "gate": gate,
        },
        "reproducibility": repro_detail,
        "constants": {
            "STOP_LOSS_PCT": STOP_LOSS_PCT,
            "TAKE_PROFIT_PCT": TAKE_PROFIT_PCT,
            "TIMEOUT_BARS": TIMEOUT_BARS,
            "FRICTION_PCT": FRICTION_PCT,
            "VOL_MA_PERIOD": VOL_MA_PERIOD,
            "PRICE_LOOKBACK": PRICE_LOOKBACK,
            "ATR_PERIOD": ATR_PERIOD,
            "COOLDOWN_BARS": COOLDOWN_BARS,
            "BTC_TREND_K": BTC_TREND_K,
            "RAPID_RALLY_PCT": RAPID_RALLY_PCT,
            "BREADTH_4H_LOOKBACK": BREADTH_4H_LOOKBACK,
            "BREADTH_THRESHOLD": BREADTH_THRESHOLD,
            "BARS_SINCE_HIGH_LOOKBACK": BARS_SINCE_HIGH_LOOKBACK,
            "BARS_SINCE_HIGH_MAX": BARS_SINCE_HIGH_MAX,
            "LOWER_LOW_LOOKBACK": LOWER_LOW_LOOKBACK,
            "WF_FOLD_MONTHS": WF_FOLD_MONTHS,
            "RANDOM_SEED": RANDOM_SEED,
            "MC_N_RESAMPLES": MC_N_RESAMPLES,
            "TRAIN_OOT_BOUNDARY": str(TRAIN_OOT_BOUNDARY),
        },
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
