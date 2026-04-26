"""Phase 17.D — 4×4 sweep over SIZING curves and EXIT policies on D1 SHORT trades.

Phase 16 locked: D1 detector + sign-locked Candidate-B BTC sizing + v3 LR
classifier + Phase-14 V2 rapid-rally exit. Locked OOT PF=2.607, WF p5=0.719,
n=2336 on the classifier-gated subset (score >= 0.4991 = TRAIN q0.5).

This script searches for a (SIZING, EXIT) cell that beats the locked OOT PF
while keeping WF p5 >= 0.65.

PRE-COMMITTED DECISION RULE:
  - A cell "wins" iff its OOT PF is the highest in the grid AND WF p5 >= 0.65
    AND OOT n >= 1500 (smaller-trade-set risk floor).
  - If the cell with highest OOT PF has WF p5 < 0.65, walk down OOT-PF rank
    until a cell satisfies the WF p5 floor and the OOT n >= 1500 floor.
  - If no cell satisfies all three: print "DROP — current S1×E1 holds".
  - Otherwise print "ADOPT S{i}×E{j}" with cell metrics.

GRID (16 cells):
  Sizing: S1 current | S2 sqrt | S3 step | S4 classifier-weighted
  Exits:  E1 fixed-pct | E2 ATR-based | E3 trailing | E4 regime-flip-only

HARD CONSTRAINTS (all enforced):
  TRAIN_OOT_BOUNDARY=2026-01-01 UTC
  WF_FOLD_MONTHS=3 (1-month step)
  RANDOM_SEED=42, MC_N_RESAMPLES=5000
  Friction=0.0015 in every exit policy
  Timeout=672 bars in E1, E2, E3, E4
  Classifier gate: v3 score >= 0.4991 (TRAIN q0.5) applied first.
  No retraining. No live changes.

Run from services/python/:
    uv run python scripts/phase17_d_sizing_exit_sweep.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES
from src.ml.indicators import atr as _atr, ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi


# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
DATA_DIR = SERVICES_PY / "data"
MODELS_DIR = SERVICES_PY / "models"
RESULTS_DIR = SERVICES_PY / "results"
TRADES_PATH = DATA_DIR / "d1_short_trades_with_features.csv"
MODEL_PATH = MODELS_DIR / "d1_short_v3.joblib"
META_PATH = MODELS_DIR / "d1_short_v3_meta.json"
OUT_PATH = RESULTS_DIR / "phase17_d_sizing_exit_sweep.json"

# --- Constants --------------------------------------------------------------

TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
WF_FOLD_MONTHS = 3
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000

STOP_LOSS_PCT = 0.05      # E1 only
TAKE_PROFIT_PCT = 0.15    # E1, E2, E3
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015
ATR_PERIOD = 14

# Phase 14 V2 rapid-rally exit knobs
RAPID_RALLY_PCT = 0.03
RAPID_RALLY_LOOKBACK_15M = 96

# Sizing knobs (Candidate B BTC trend score)
BTC_TREND_K = 0.005255

# Pre-committed thresholds for adopt rule
WF_P5_FLOOR = 0.65
OOT_N_FLOOR = 1500


# --- Metrics helpers (copy-pasted plumbing) ---------------------------------


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


# --- BTC panels (rapid rally for E1/E2/E3, regime flip = same panel for E4) -


def _build_rapid_exit_panel(candles_15m: pd.DataFrame) -> pd.Series:
    """Phase 14 V2 rapid-rally exit panel — boolean per 15m bar on BTC.

    Replicated EXACTLY from phase15_atr_detector_vs_volume.py.
    """
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
    rapid_exit = ((~daily_bear_15m) & (btc_24h_ret > RAPID_RALLY_PCT)).fillna(False).astype(bool)
    return rapid_exit


# --- Per-bar SHORT simulators for E1..E4 ------------------------------------


def _sim_E1_fixed_pct(close, high, low, open_, enter_bar, rally_flag):
    """E1 — current: SL=−5%, TP=+15%, timeout=672, rapid-rally."""
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
            return {"pnl_pct": -STOP_LOSS_PCT - FRICTION_PCT, "exit": "sl",
                    "bars": int(bar - enter_bar)}
        if low[bar] <= tp_price:
            return {"pnl_pct": TAKE_PROFIT_PCT - FRICTION_PCT, "exit": "tp",
                    "bars": int(bar - enter_bar)}
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (entry_price - ep) / entry_price - FRICTION_PCT,
                    "exit": "rapid", "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[eb])) / entry_price - FRICTION_PCT,
            "exit": "timeout", "bars": int(eb - enter_bar)}


def _sim_E2_atr(close, high, low, open_, atr14, enter_bar, rally_flag):
    """E2 — ATR-based: SL = entry*(1+2*ATR/entry), TP = entry*(1-6*ATR/entry).
    Same timeout=672, same rapid-rally panel. Friction 0.0015.
    """
    n = len(close)
    if enter_bar >= n - 1:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    a = float(atr14[enter_bar]) if enter_bar < len(atr14) else float("nan")
    if not np.isfinite(a) or a <= 0:
        return None
    # Per spec: SL_short = entry_price * (1 + 2*ATR_14_at_entry/entry_price)
    sl_price = entry_price * (1.0 + 2.0 * a / entry_price)  # = entry_price + 2*ATR
    tp_price = entry_price * (1.0 - 6.0 * a / entry_price)  # = entry_price - 6*ATR
    if tp_price <= 0:
        # ATR too large; skip
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


def _sim_E3_trailing(close, high, low, open_, enter_bar, rally_flag):
    """E3 — trailing: initial SL=−5%. Once price has dropped >=5% from entry,
    shift SL to lowest_low_so_far * 1.02 and trail. TP=+15%, timeout=672, rapid.
    """
    n = len(close)
    if enter_bar >= n - 1:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    sl_price = entry_price * (1 + STOP_LOSS_PCT)
    tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
    profit_trigger = entry_price * (1 - STOP_LOSS_PCT)  # price needs to drop 5%
    last_bar = min(enter_bar + 1 + TIMEOUT_BARS, n)
    lowest_low = float("inf")
    trailing = False
    for bar in range(enter_bar + 1, last_bar):
        # Update lowest_low with current bar
        if low[bar] < lowest_low:
            lowest_low = float(low[bar])
        # Activate trailing once we've reached +5% unrealized profit
        # i.e. price has dropped to <= entry_price * (1 - 0.05)
        if (not trailing) and (lowest_low <= profit_trigger):
            trailing = True
        if trailing:
            new_sl = lowest_low * 1.02
            # Trailing SL only ratchets DOWN for shorts
            if new_sl < sl_price:
                sl_price = new_sl
        # Check TP first (would require low <= tp_price). For shorts, TP hit if low <= tp_price.
        # Order of checks: SL > TP > rapid (consistent with E1, but here SL might already be tighter
        # than entry after trailing). Use same order as E1.
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


def _sim_E4_regime_only(close, high, low, open_, enter_bar, rally_flag):
    """E4 — no SL, no TP. Exit only on rapid-rally trigger or timeout."""
    n = len(close)
    if enter_bar >= n - 1:
        return None
    entry_price = float(open_[enter_bar])
    if entry_price <= 0 or not np.isfinite(entry_price):
        return None
    last_bar = min(enter_bar + 1 + TIMEOUT_BARS, n)
    for bar in range(enter_bar + 1, last_bar):
        if rally_flag is not None and bool(rally_flag[bar]):
            ep = float(close[bar])
            return {"pnl_pct": (entry_price - ep) / entry_price - FRICTION_PCT,
                    "exit": "rapid", "bars": int(bar - enter_bar)}
    eb = last_bar - 1
    return {"pnl_pct": (entry_price - float(close[eb])) / entry_price - FRICTION_PCT,
            "exit": "timeout", "bars": int(eb - enter_bar)}


# --- Per-asset cache --------------------------------------------------------


def _build_asset_cache(candles: pd.DataFrame, rapid_naive: pd.Series) -> dict:
    """Build a per-asset cache of (close/high/low/open arrays, atr14, rally_flag,
    timestamp→bar_idx map). Done once. Returns dict[asset] -> dict.
    """
    cache = {}
    assets = sorted(candles["asset"].unique())
    for asset in assets:
        sub = (
            candles[candles["asset"] == asset]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(sub) < 200:
            continue
        ts_naive = pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(None)
        close = sub["close"].to_numpy(dtype=float)
        high = sub["high"].to_numpy(dtype=float)
        low = sub["low"].to_numpy(dtype=float)
        open_ = sub["open"].to_numpy(dtype=float)
        atr14 = _atr(pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD).to_numpy()
        rally = (
            rapid_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill")
            .fillna(False).to_numpy().astype(bool)
        )
        # ts -> bar idx (use int timestamps in ns as keys for fast lookup)
        ts_ns = ts_naive.astype("int64").to_numpy()
        ts_to_idx = {int(t): i for i, t in enumerate(ts_ns)}
        cache[asset] = {
            "close": close, "high": high, "low": low, "open": open_,
            "atr14": atr14, "rally": rally, "ts_to_idx": ts_to_idx,
        }
    return cache


# --- Sizing curves ----------------------------------------------------------


def _sizing_S1(btc_score: float, cls_score: float) -> float:
    """S1 — current: pos_scale = min(1.5*|btc_score|, 1.5). Sign-locked already
    enforced by data: btc_score < 0 for all SHORT trades."""
    s = abs(float(btc_score)) if np.isfinite(btc_score) else 0.0
    return float(min(1.5 * s, 1.5))


def _sizing_S2(btc_score: float, cls_score: float) -> float:
    """S2 — sqrt: pos_scale = min(1.5*sqrt(|btc_score|), 1.5)."""
    s = abs(float(btc_score)) if np.isfinite(btc_score) else 0.0
    return float(min(1.5 * np.sqrt(s), 1.5))


def _sizing_S3(btc_score: float, cls_score: float) -> float:
    """S3 — threshold step: 1.5 if |btc_score| >= 0.5 else 0."""
    s = abs(float(btc_score)) if np.isfinite(btc_score) else 0.0
    return 1.5 if s >= 0.5 else 0.0


def _sizing_S4(btc_score: float, cls_score: float) -> float:
    """S4 — classifier-score weighted: clip(1.5*(score-0.5)*2, 0, 1.5).
    Only sizes when classifier confidence > 0.5."""
    if not np.isfinite(cls_score):
        return 0.0
    raw = 1.5 * (float(cls_score) - 0.5) * 2.0
    return float(np.clip(raw, 0.0, 1.5))


SIZING_FNS = {
    "S1_current": _sizing_S1,
    "S2_sqrt": _sizing_S2,
    "S3_step": _sizing_S3,
    "S4_cls_weighted": _sizing_S4,
}

EXIT_NAMES = ["E1_fixed", "E2_atr", "E3_trail", "E4_regime"]


# --- Re-simulate exit policies for all (gated) trades -----------------------


def _resim_exits(
    df_gated: pd.DataFrame,
    cache: dict,
) -> dict:
    """For each gated trade, re-simulate E1, E2, E3, E4 once. Return dict of
    np.ndarray of pnl_pct keyed by exit name, aligned with df_gated row order.

    Uses entry_bar = (idx of bar with timestamp == entry_time) - 1 — empirically
    confirmed to reproduce the saved pnl_pct values.
    """
    n = len(df_gated)
    pnl_E1 = np.full(n, np.nan)
    pnl_E2 = np.full(n, np.nan)
    pnl_E3 = np.full(n, np.nan)
    pnl_E4 = np.full(n, np.nan)
    exit_E1 = np.full(n, "skip", dtype=object)
    exit_E2 = np.full(n, "skip", dtype=object)
    exit_E3 = np.full(n, "skip", dtype=object)
    exit_E4 = np.full(n, "skip", dtype=object)

    et_naive_ns = (
        pd.to_datetime(df_gated["entry_time"], utc=True)
        .dt.tz_convert(None)
        .astype("int64")
        .to_numpy()
    )
    assets = df_gated["asset"].to_numpy()

    miss_count = 0
    for i in range(n):
        asset = assets[i]
        c = cache.get(asset)
        if c is None:
            miss_count += 1
            continue
        idx_at_ts = c["ts_to_idx"].get(int(et_naive_ns[i]))
        if idx_at_ts is None:
            miss_count += 1
            continue
        # entry_time is timestamp[trigger_bar + 1] in detector code,
        # but execution uses open[trigger_bar]. So enter_bar = idx_at_ts - 1.
        enter_bar = idx_at_ts - 1
        if enter_bar < 0:
            miss_count += 1
            continue

        close = c["close"]; high = c["high"]; low = c["low"]; open_ = c["open"]
        atr14 = c["atr14"]; rally = c["rally"]

        t1 = _sim_E1_fixed_pct(close, high, low, open_, enter_bar, rally)
        if t1 is not None:
            pnl_E1[i] = t1["pnl_pct"]; exit_E1[i] = t1["exit"]
        t2 = _sim_E2_atr(close, high, low, open_, atr14, enter_bar, rally)
        if t2 is not None:
            pnl_E2[i] = t2["pnl_pct"]; exit_E2[i] = t2["exit"]
        t3 = _sim_E3_trailing(close, high, low, open_, enter_bar, rally)
        if t3 is not None:
            pnl_E3[i] = t3["pnl_pct"]; exit_E3[i] = t3["exit"]
        t4 = _sim_E4_regime_only(close, high, low, open_, enter_bar, rally)
        if t4 is not None:
            pnl_E4[i] = t4["pnl_pct"]; exit_E4[i] = t4["exit"]
    return {
        "pnl_E1_fixed": pnl_E1, "exit_E1_fixed": exit_E1,
        "pnl_E2_atr": pnl_E2, "exit_E2_atr": exit_E2,
        "pnl_E3_trail": pnl_E3, "exit_E3_trail": exit_E3,
        "pnl_E4_regime": pnl_E4, "exit_E4_regime": exit_E4,
        "miss": miss_count,
    }


def _verify_E1_reproduces_saved(df_gated, pnl_E1):
    """Quick sanity: E1 re-sim should match the saved pnl_pct in the CSV
    on a 20-trade sample (after the classifier gate)."""
    sample = df_gated.head(20).reset_index(drop=True)
    sample_pnl = pnl_E1[:20]
    saved = sample["pnl_pct"].to_numpy()
    diffs = np.abs(saved - sample_pnl)
    max_diff = float(np.nanmax(diffs)) if len(diffs) else 0.0
    matches = int(np.sum(diffs < 1e-6))
    return matches, len(saved), max_diff


# --- Main grid eval ---------------------------------------------------------


def _eval_cell(df: pd.DataFrame, sizing_name: str, exit_name: str) -> dict:
    """df has: entry_time, btc_score, score (cls), pnl_E*_*, exit_E*_*.
    Build sized_pnl from the chosen sizing curve × the chosen exit pnl, then
    drop trades with zero size. Compute TRAIN/OOT/WF/MC."""
    sfn = SIZING_FNS[sizing_name]
    pnl_col = {
        "E1_fixed": "pnl_E1_fixed",
        "E2_atr": "pnl_E2_atr",
        "E3_trail": "pnl_E3_trail",
        "E4_regime": "pnl_E4_regime",
    }[exit_name]
    pnl_pct = df[pnl_col].to_numpy(dtype=float)
    btc_score = df["btc_score"].to_numpy(dtype=float)
    cls_score = df["score"].to_numpy(dtype=float)

    pos_scale = np.array([sfn(btc_score[i], cls_score[i]) for i in range(len(df))])
    sized_pnl = pnl_pct * pos_scale

    keep = np.isfinite(sized_pnl) & (pos_scale > 0)
    sub = df.loc[keep, ["entry_time"]].copy()
    sub["sized_pnl"] = sized_pnl[keep]
    sub["entry_time"] = pd.to_datetime(sub["entry_time"], utc=True)
    sub["period"] = np.where(sub["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
    train_pnls = sub[sub["period"] == "train"]["sized_pnl"].to_numpy()
    oot_pnls = sub[sub["period"] == "oot"]["sized_pnl"].to_numpy()

    train_m = _cell(train_pnls)
    oot_m = _cell(oot_pnls)
    wf = _walkforward(sub, "sized_pnl")
    mc = _monte_carlo(oot_pnls)

    return {
        "sizing": sizing_name,
        "exit": exit_name,
        "n_total_after_size": int(keep.sum()),
        "train_metrics": train_m,
        "oot_metrics": oot_m,
        "walkforward": wf,
        "monte_carlo_oot": mc,
    }


def main():
    started = time.monotonic()

    print(f"[1/5] loading trades from {TRADES_PATH}", flush=True)
    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"      {len(df):,} trades loaded", flush=True)

    print(f"[2/5] loading v3 classifier…", flush=True)
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]
    print(f"      threshold @ TRAIN q0.5 = {threshold:.4f}", flush=True)
    print(f"      score range = [{df['score'].min():.4f}, {df['score'].max():.4f}]",
          flush=True)

    df_gated = df[df["score"] >= threshold].reset_index(drop=True).copy()
    print(f"      classifier-gated trades: {len(df_gated):,} of {len(df):,}",
          flush=True)

    print(f"[3/5] loading candles + building rapid-rally panel…", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)
    rapid_exit_15m = _build_rapid_exit_panel(btc_15m)
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)

    print(f"[4/5] building per-asset cache (close/high/low/open/atr14/rally)…",
          flush=True)
    t0 = time.monotonic()
    cache = _build_asset_cache(candles, rapid_naive)
    print(f"      cache built: {len(cache)} assets in {time.monotonic()-t0:.1f}s",
          flush=True)

    # Free the big candles DataFrame; cache has what we need
    del candles

    print(f"[5/5] re-simulating E1..E4 for {len(df_gated):,} gated trades…",
          flush=True)
    t0 = time.monotonic()
    sim = _resim_exits(df_gated, cache)
    print(f"      re-sim wall = {time.monotonic()-t0:.1f}s, miss={sim['miss']}",
          flush=True)

    df_gated["pnl_E1_fixed"]   = sim["pnl_E1_fixed"]
    df_gated["exit_E1_fixed"]  = sim["exit_E1_fixed"]
    df_gated["pnl_E2_atr"]     = sim["pnl_E2_atr"]
    df_gated["exit_E2_atr"]    = sim["exit_E2_atr"]
    df_gated["pnl_E3_trail"]   = sim["pnl_E3_trail"]
    df_gated["exit_E3_trail"]  = sim["exit_E3_trail"]
    df_gated["pnl_E4_regime"]  = sim["pnl_E4_regime"]
    df_gated["exit_E4_regime"] = sim["exit_E4_regime"]

    # Sanity: re-sim E1 must reproduce saved pnl_pct on a 20-trade sample
    matches, total, max_diff = _verify_E1_reproduces_saved(
        df_gated, sim["pnl_E1_fixed"]
    )
    print(f"\nSANITY: E1 re-sim vs saved pnl_pct on first 20 gated trades: "
          f"{matches}/{total} match (max abs diff = {max_diff:.2e})", flush=True)

    # === 4×4 GRID =====
    print(f"\n=== 4×4 GRID (rows=sizing, cols=exit) ===")
    grid = {}
    for sname in SIZING_FNS.keys():
        grid[sname] = {}
        for ename in EXIT_NAMES:
            cell = _eval_cell(df_gated, sname, ename)
            grid[sname][ename] = cell
            tm = cell["train_metrics"]; om = cell["oot_metrics"]
            wf = cell["walkforward"]; mc = cell["monte_carlo_oot"]
            print(f"  {sname:>17} × {ename:<10}  "
                  f"OOT n={om['trades']:>5}  PF={om['pf']:.3f}  "
                  f"WF p5={wf.get('p5', float('nan')):.3f}  "
                  f"MC p5={mc.get('p5', float('nan')):.3f}",
                  flush=True)

    # === DECISION RULE ============================================
    print(f"\n=== DECISION RULE ===")
    print(f"  win iff OOT PF is highest in grid AND WF p5 >= {WF_P5_FLOOR} "
          f"AND OOT n >= {OOT_N_FLOOR}", flush=True)

    rank = []
    for sname in SIZING_FNS.keys():
        for ename in EXIT_NAMES:
            c = grid[sname][ename]
            om = c["oot_metrics"]; wf = c["walkforward"]
            rank.append({
                "sname": sname, "ename": ename,
                "oot_pf": om["pf"], "oot_n": om["trades"],
                "wf_p5": wf.get("p5", float("nan")),
                "mc_p5": c["monte_carlo_oot"].get("p5", float("nan")),
            })
    rank.sort(key=lambda r: (r["oot_pf"] if np.isfinite(r["oot_pf"]) else -np.inf),
              reverse=True)

    # Walk down rank until floors are met
    winner = None
    for r in rank:
        if (np.isfinite(r["oot_pf"])
                and np.isfinite(r["wf_p5"]) and r["wf_p5"] >= WF_P5_FLOOR
                and r["oot_n"] >= OOT_N_FLOOR):
            winner = r
            break

    if winner is None:
        verdict = "DROP — current S1×E1 holds"
        print(f"  {verdict}", flush=True)
    else:
        verdict = f"ADOPT {winner['sname']}×{winner['ename']}"
        print(f"  {verdict}", flush=True)
        print(f"     OOT n={winner['oot_n']}  PF={winner['oot_pf']:.4f}  "
              f"WF p5={winner['wf_p5']:.4f}  MC p5={winner['mc_p5']:.4f}",
              flush=True)

    # === REPRODUCIBILITY: re-run winning cell metrics twice ===========
    if winner is not None:
        c1 = _eval_cell(df_gated, winner["sname"], winner["ename"])
        c2 = _eval_cell(df_gated, winner["sname"], winner["ename"])
        keys = [
            ("train_pf", c1["train_metrics"]["pf"], c2["train_metrics"]["pf"]),
            ("oot_pf",   c1["oot_metrics"]["pf"],   c2["oot_metrics"]["pf"]),
            ("wf_p5",    c1["walkforward"]["p5"],   c2["walkforward"]["p5"]),
            ("mc_p5",    c1["monte_carlo_oot"]["p5"], c2["monte_carlo_oot"]["p5"]),
        ]
        all_match = all(round(a, 4) == round(b, 4) for _, a, b in keys)
        print(f"\nREPRODUCIBILITY: {'PASS' if all_match else 'FAIL'}", flush=True)
        for k, a, b in keys:
            print(f"  {k}: {a:.6f} vs {b:.6f}", flush=True)
    else:
        all_match = True  # Trivially reproducible if no winner
        # Still verify S1×E1 reproduces twice
        c1 = _eval_cell(df_gated, "S1_current", "E1_fixed")
        c2 = _eval_cell(df_gated, "S1_current", "E1_fixed")
        keys = [
            ("oot_pf", c1["oot_metrics"]["pf"], c2["oot_metrics"]["pf"]),
            ("wf_p5",  c1["walkforward"]["p5"], c2["walkforward"]["p5"]),
        ]
        all_match = all(round(a, 4) == round(b, 4) for _, a, b in keys)
        print(f"\nREPRODUCIBILITY (S1×E1 fallback): "
              f"{'PASS' if all_match else 'FAIL'}", flush=True)

    # === Sanity print: S1×E1 vs locked Phase-16 numbers =================
    s1e1 = grid["S1_current"]["E1_fixed"]
    print(f"\n=== Sanity: S1_current × E1_fixed should reproduce Phase-16 lock ===")
    print(f"  expected: OOT PF=2.607  WF p5=0.719  OOT n=2336", flush=True)
    print(f"  got:      OOT PF={s1e1['oot_metrics']['pf']:.3f}  "
          f"WF p5={s1e1['walkforward'].get('p5', float('nan')):.3f}  "
          f"OOT n={s1e1['oot_metrics']['trades']}", flush=True)

    # === Write JSON =====================================================
    out = {
        "phase": "17.D — sizing × exit 4x4 sweep",
        "snapshot": SNAPSHOT_DATE,
        "classifier_threshold": threshold,
        "wf_p5_floor": WF_P5_FLOOR,
        "oot_n_floor": OOT_N_FLOOR,
        "sanity_E1_resim": {
            "matches": matches, "total": total, "max_abs_diff": max_diff,
        },
        "sanity_S1xE1_vs_phase16_lock": {
            "expected": {"oot_pf": 2.607, "wf_p5": 0.719, "oot_n": 2336},
            "got": {
                "oot_pf": s1e1["oot_metrics"]["pf"],
                "wf_p5": s1e1["walkforward"].get("p5", float("nan")),
                "oot_n": s1e1["oot_metrics"]["trades"],
            },
        },
        "grid": grid,
        "ranked": rank,
        "winner": winner,
        "verdict": verdict,
        "reproducibility_pass": bool(all_match),
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
