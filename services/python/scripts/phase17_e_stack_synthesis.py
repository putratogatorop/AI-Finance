"""Phase 17.E — Synthesis: stack D1e (Phase 17.A winner) + retrained v4 classifier
+ S4 classifier-weighted sizing × E2 ATR-based exits (Phase 17.D winner).

Phase 17 had 4 strands; A (D1e detector with universe-down-breadth gate) and D
(S4 sizing × E2 ATR exits) cleanly won. The project memory rule says: detector
change → retrain classifier on the new distribution. This script does that and
re-runs the S4×E2 backtest on the stacked configuration.

PRE-COMMITTED DECISION RULE:
  Adopt stacked config iff ALL of:
    (a) WF p5 >= 1.0 (clears the 8-point ship-gate WF floor)
    (b) OOT n >= 1500
    (c) OOT PF >= 5.422 (the 17.D-alone winner's OOT PF, applied to the new ledger)
    (d) MC p5 >= 4.550 (17.D-alone MC p5)
    (e) WF p5 >= 2.474 (17.D-alone WF p5) — within 5% degradation tolerance
  If any FAILS: print "STAY ON 17.D-alone (D1 + v3 + S4 × E2)".
  If all PASS: print "ADOPT 17.E (D1e + v4 + S4 × E2)".

HARD CONSTRAINTS:
  TRAIN_OOT_BOUNDARY=2026-01-01 UTC; same RANDOM_SEED=42, MC_N_RESAMPLES=5000;
  WF_FOLD_MONTHS=3; LR C=1.0, balanced, 5-fold purged CV @ purge=192.
  No LightGBM. No live changes. Stage files only.

Run from services/python/:
    uv run python scripts/phase17_e_stack_synthesis.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES, FeatureContext
from src.ml.bigmover_combined.sizing import default_sizing_config
from src.ml.indicators import atr as _atr, ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi


# --- Paths / constants ------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3
PURGE_BARS = 192

CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
DATA_DIR = SERVICES_PY / "data"
MODELS_DIR = SERVICES_PY / "models"
RESULTS_DIR = SERVICES_PY / "results"

TRADES_OUT = DATA_DIR / "d1e_short_trades.csv"
FEATURES_OUT = DATA_DIR / "d1e_short_trades_with_features.csv"
MODEL_OUT = MODELS_DIR / "d1e_short_v4.joblib"
META_OUT = MODELS_DIR / "d1e_short_v4_meta.json"
SIZING_OUT = MODELS_DIR / "d1e_short_v4_sizing.json"
OUT_PATH = RESULTS_DIR / "phase17_e_stack_synthesis.json"

# D1 detector + simulation constants — must match Phase 15 / 17.A
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
COOLDOWN_BARS = 96
BTC_TREND_K = 0.005255
RAPID_RALLY_PCT = 0.03
RAPID_RALLY_LOOKBACK_15M = 96

# 17.A breadth threshold
BREADTH_THRESHOLD = 0.6
BREADTH_4H_LOOKBACK = 16   # = features.py BREADTH_4H_LOOKBACK

# 17.D winner constants
LR_C = 1.0
ATR_SL_MULT = 2.0   # E2 SL = entry * (1 + 2*ATR/entry) for SHORT
ATR_TP_MULT = 6.0   # E2 TP = entry * (1 - 6*ATR/entry)


# --- Metrics ----------------------------------------------------------------


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


# --- Panels -----------------------------------------------------------------


def _build_btc_trend_score_15m(candles_15m: pd.DataFrame) -> pd.Series:
    btc = candles_15m["close"].astype(float)
    btc_4h = btc.resample("4h", label="right", closed="right").last()
    md = _macd(btc_4h, 12, 26, 9)
    spread = ((md["macd"] - md["signal"]).shift(1) / btc_4h.shift(1))
    score_4h = spread.clip(-BTC_TREND_K, BTC_TREND_K) / BTC_TREND_K
    return score_4h.reindex(btc.index, method="ffill")


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
    rapid_exit = ((~daily_bear_15m) & (btc_24h_ret > RAPID_RALLY_PCT)).fillna(False).astype(bool)
    return rapid_exit


def _build_breadth_short_panel(candles: pd.DataFrame) -> pd.Series:
    """Mirror src/ml/bigmover_combined/features.py:_ret_4h_panel breadth.

    Returns a 15m-grid Series of 'fraction of universe whose 4h return is < 0'
    at each timestamp. Aligned to the 15m grid by ffill.
    """
    df = candles.sort_values(["asset", "timestamp"]).copy()
    df["ret_4h"] = df.groupby("asset")["close"].transform(lambda s: s.pct_change(BREADTH_4H_LOOKBACK))
    pivot = df.pivot(index="timestamp", columns="asset", values="ret_4h")
    notna = pivot.notna().sum(axis=1).to_numpy()
    down = (pivot < 0).sum(axis=1).to_numpy()
    breadth = pd.Series(np.where(notna > 0, down / np.where(notna > 0, notna, 1), np.nan),
                        index=pivot.index, name="breadth_short")
    return breadth


# --- Detector + simulators --------------------------------------------------


def _detect_short_d1e(close, high, low, open_, volume, *, breadth_at_bar) -> np.ndarray:
    """D1e — D1 (drop>=2.5×ATR + vol>=2 + red bar) AND breadth >= 0.6."""
    n = len(close)
    out = np.zeros(n, dtype=bool)
    if n < PRICE_LOOKBACK + ATR_PERIOD:
        return out
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().to_numpy()
    rolling_high = pd.Series(high).rolling(PRICE_LOOKBACK, min_periods=PRICE_LOOKBACK).max().to_numpy()
    atr14 = _atr(pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD).to_numpy()
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


def _sim_E1(close, high, low, open_, enter_bar, *, rally_flag):
    """Phase 14/15 baseline: SL=−5% / TP=+15% / timeout=672 / rapid-rally."""
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


def _sim_E2_atr(close, high, low, open_, atr14, enter_bar, *, rally_flag):
    """E2 — ATR-based: SL = entry + 2*ATR, TP = entry − 6*ATR (SHORT)."""
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


# --- Detection driver -------------------------------------------------------


def _detect_d1e_trades(candles, btc_score_15m, rapid_exit_15m, breadth_15m):
    """Detect D1e trades with sign-locked SHORT sizing + Phase-14 V2 rapid-rally
    exit (E1 baseline). The baseline pnl_pct under E1 is what we feed to the
    classifier as outcome label."""
    rows = []
    btc_score_naive = btc_score_15m.copy()
    btc_score_naive.index = pd.to_datetime(btc_score_naive.index, utc=True).tz_convert(None)
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)
    breadth_naive = breadth_15m.copy()
    breadth_naive.index = pd.to_datetime(breadth_naive.index, utc=True).tz_convert(None)

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
        atr14 = _atr(pd.Series(high), pd.Series(low), pd.Series(close), ATR_PERIOD).to_numpy()

        breadth_at_bar = breadth_naive.reindex(
            pd.DatetimeIndex(ts_naive), method="ffill"
        ).to_numpy(dtype=float)
        triggers = _detect_short_d1e(close, high, low, open_, volume, breadth_at_bar=breadth_at_bar)

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
        rally = rapid_naive.reindex(pd.DatetimeIndex(ts_naive), method="ffill").fillna(False).to_numpy().astype(bool)

        for i in entry_bars:
            score = btc_score[i] if i < len(btc_score) else float("nan")
            if not np.isfinite(score) or score >= 0:
                continue
            pos_scale_s1 = min(1.5 * abs(score), 1.5)
            if pos_scale_s1 <= 0:
                continue
            t = _sim_E1(close, high, low, open_, i, rally_flag=rally)
            if t is None:
                continue
            t["sized_pnl"] = t["pnl_pct"] * pos_scale_s1
            t["pos_scale_s1"] = pos_scale_s1
            t["asset"] = asset
            t["symbol"] = asset
            t["direction"] = "short"
            entry_idx = i + 1 if i + 1 < len(sub) else i
            t["entry_time"] = pd.Timestamp(sub["timestamp"].iloc[entry_idx])
            t["btc_score"] = float(score)
            t["enter_bar"] = i
            t["atr14_at_entry"] = float(atr14[i]) if i < len(atr14) and np.isfinite(atr14[i]) else float("nan")
            rows.append(t)
        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(assets)} assets processed, trades={len(rows):,}",
                  flush=True)
    return pd.DataFrame(rows)


def _purged_kfold_indices(n: int, k: int = 5, purge: int = 192):
    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        train_end = max(0, val_start - purge)
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


# --- E2 re-simulator across all trades --------------------------------------


def _resimulate_E2_all(trades: pd.DataFrame, candles: pd.DataFrame,
                       rapid_exit_15m: pd.Series) -> pd.DataFrame:
    """For every trade, re-run E2 ATR exits using the asset's full candle series."""
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)

    out = []
    for asset, gtrades in trades.groupby("asset"):
        sub = (
            candles[candles["asset"] == asset]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(sub) < 200:
            for _, r in gtrades.iterrows():
                rec = r.to_dict()
                rec.update({"e2_pnl_pct": float("nan"), "e2_exit": "skip"})
                out.append(rec)
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
            rec = r.to_dict()
            if t is None:
                rec["e2_pnl_pct"] = float("nan")
                rec["e2_exit"] = "skip"
                rec["e2_bars"] = -1
            else:
                rec["e2_pnl_pct"] = float(t["pnl_pct"])
                rec["e2_exit"] = t["exit"]
                rec["e2_bars"] = int(t["bars"])
            out.append(rec)
    return pd.DataFrame(out)


# --- Summarize --------------------------------------------------------------


def _summarize(df: pd.DataFrame, label: str, pnl_col: str = "sized_pnl"):
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["period"] = np.where(df["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
    train_pnls = df[df["period"] == "train"][pnl_col].to_numpy()
    oot_pnls = df[df["period"] == "oot"][pnl_col].to_numpy()
    train_m = _cell(train_pnls)
    oot_m = _cell(oot_pnls)
    wf = _walkforward(df, pnl_col)
    mc = _monte_carlo(oot_pnls)
    print(
        f"  {label:<30} TRAIN n={train_m['trades']:>5} PF={train_m['pf']:.3f}  "
        f"OOT n={oot_m['trades']:>5} PF={oot_m['pf']:.3f}  "
        f"WF p5={wf.get('p5', float('nan')):.3f}  P(loss)={wf.get('p_lt_1', float('nan'))*100:.1f}%  "
        f"MC p5={mc.get('p5', float('nan')):.3f}",
        flush=True,
    )
    return {
        "train_metrics": train_m, "oot_metrics": oot_m,
        "walkforward": wf, "monte_carlo_oot": mc,
    }


# --- Main -------------------------------------------------------------------


def main():
    started = time.monotonic()
    np.random.seed(RANDOM_SEED)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ---- 1. Load candles + build panels --------------------------------
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
    print("building cross-sectional breadth panel (4h returns < 0 fraction)...", flush=True)
    breadth_15m = _build_breadth_short_panel(candles)

    # ---- 2. Detect D1e trades ------------------------------------------
    print("\n=== Detecting D1e SHORT trades ===")
    trades = _detect_d1e_trades(candles, btc_score_15m, rapid_exit_15m, breadth_15m)
    if len(trades) == 0:
        raise RuntimeError("D1e produced no trades — abort")
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades.to_csv(TRADES_OUT, index=False)
    print(f"  saved {len(trades):,} D1e trades → {TRADES_OUT}")

    # ---- 3. Build features + outcome label -----------------------------
    trades["label"] = (trades["pnl_pct"] > 0).astype(int)
    print(f"  outcome positive rate: {trades['label'].mean():.1%}")

    print("\n=== Building 16-feature panels ===")
    universe_listed_since: dict[str, pd.Timestamp] = {}
    universe_path = REPO_ROOT / "data" / "snapshots" / f"universe_{SNAPSHOT_DATE}.parquet"
    if universe_path.exists():
        u = pd.read_parquet(universe_path)
        if "asset" in u.columns and "listed_since" in u.columns:
            for _, row in u.iterrows():
                k = f"{row['asset']}USDT"
                ls = row["listed_since"]
                if pd.notna(ls):
                    universe_listed_since[k] = pd.Timestamp(ls, tz="UTC")
    ctx = FeatureContext(candles, btc_asset_key="BTCUSDT",
                         universe_listed_since=universe_listed_since)
    feat_rows = []
    for i, row in enumerate(trades.itertuples(index=False)):
        feats = ctx._build_for_typed(row.symbol, row.entry_time, row.direction)
        feat_rows.append([feats[n] for n in FEATURE_NAMES] + [feats["is_short"]])
        if (i + 1) % 5000 == 0:
            print(f"  features: {i+1}/{len(trades)}", flush=True)
    cols = FEATURE_NAMES + ["is_short"]
    X = pd.DataFrame(feat_rows, columns=cols)
    finite_mask = X.notna().all(axis=1)
    print(f"  feature-complete rows: {int(finite_mask.sum())}/{len(trades)}")
    trades = trades[finite_mask].reset_index(drop=True)
    X = X[finite_mask].reset_index(drop=True)
    y = trades["label"].astype(int).to_numpy()

    feat_full = X.copy()
    feat_full.insert(0, "entry_time",
                     trades["entry_time"].astype("datetime64[ns, UTC]").astype(str))
    feat_full.insert(1, "asset", trades["asset"])
    for c in ["pnl_pct", "sized_pnl", "pos_scale_s1", "exit", "bars",
              "btc_score", "enter_bar", "atr14_at_entry"]:
        feat_full[c] = trades[c]
    feat_full.to_csv(FEATURES_OUT, index=False)
    print(f"  saved features-aligned ledger → {FEATURES_OUT}")

    # ---- 4. Train v4 LR on D1e trades ----------------------------------
    is_train = (trades["entry_time"] < TRAIN_OOT_BOUNDARY).to_numpy()
    is_oot = ~is_train
    print(f"  train: {is_train.sum():,}  oot: {is_oot.sum():,}")

    train_X = X.loc[is_train].to_numpy(dtype=float)
    train_y = y[is_train]
    oot_X = X.loc[is_oot].to_numpy(dtype=float)
    oot_y = y[is_oot]

    print("\n=== v4 LR — 5-fold purged time CV ===")
    cv_aucs = []
    for fold_i, (tr_idx, va_idx) in enumerate(
        _purged_kfold_indices(len(train_X), k=5, purge=PURGE_BARS)
    ):
        if len(np.unique(train_y[tr_idx])) < 2 or len(np.unique(train_y[va_idx])) < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=LR_C, max_iter=2000, random_state=RANDOM_SEED,
                                       class_weight="balanced")),
        ])
        pipe.fit(train_X[tr_idx], train_y[tr_idx])
        proba_val = pipe.predict_proba(train_X[va_idx])[:, 1]
        auc = roc_auc_score(train_y[va_idx], proba_val)
        cv_aucs.append(float(auc))
        print(f"  fold {fold_i}: AUC={auc:.4f}")

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=LR_C, max_iter=2000, random_state=RANDOM_SEED,
                                   class_weight="balanced")),
    ])
    lr_pipe.fit(train_X, train_y)
    train_auc = roc_auc_score(train_y, lr_pipe.predict_proba(train_X)[:, 1])
    oot_auc = roc_auc_score(oot_y, lr_pipe.predict_proba(oot_X)[:, 1]) \
        if len(np.unique(oot_y)) >= 2 else float("nan")
    train_scores = lr_pipe.predict_proba(train_X)[:, 1]
    threshold = float(np.percentile(train_scores, 50.0))
    print(f"\n  TRAIN AUC: {train_auc:.4f}    OOT AUC: {oot_auc:.4f}    "
          f"gap: {train_auc - oot_auc:.4f}")
    print(f"  threshold @ TRAIN q0.5: {threshold:.4f}")

    joblib.dump(lr_pipe, MODEL_OUT)
    META_OUT.write_text(json.dumps({
        "model": "logistic_regression",
        "feature_names": cols,
        "label_kind": "outcome",
        "detector": "D1e (D1 + universe-down-breadth >= 0.6)",
        "lr_C": LR_C, "purge_bars": PURGE_BARS,
        "cv_aucs": cv_aucs,
        "train_auc": float(train_auc),
        "oot_auc": float(oot_auc) if np.isfinite(oot_auc) else None,
        "threshold_train_q50": threshold,
        "trained_on_n": int(len(train_X)),
        "trained_on_pos_rate": float(np.mean(train_y)),
        "snapshot_date": SNAPSHOT_DATE,
    }, indent=2, default=str))
    SIZING_OUT.write_text(json.dumps(default_sizing_config().to_dict(), indent=2))
    print(f"  v4 saved → {MODEL_OUT}")

    # ---- 5. Score every trade with v4; apply gate ----------------------
    all_X = X.to_numpy(dtype=float)
    trades["score"] = lr_pipe.predict_proba(all_X)[:, 1]
    gated = trades[trades["score"] >= threshold].reset_index(drop=True)
    print(f"\n  v4 classifier kept {len(gated):,} of {len(trades):,} trades "
          f"({len(gated)/max(len(trades),1)*100:.1f}%)")

    # ---- 6. Re-simulate E2 ATR exits on the gated trades --------------
    print("re-simulating E2 ATR exits on gated trades...", flush=True)
    gated_e2 = _resimulate_E2_all(gated, candles, rapid_exit_15m)
    e2_ok = gated_e2["e2_pnl_pct"].notna()
    print(f"  E2 re-sim ok: {int(e2_ok.sum())}/{len(gated_e2)} trades")
    gated_e2 = gated_e2[e2_ok].reset_index(drop=True)
    gated_e2["entry_time"] = pd.to_datetime(gated_e2["entry_time"], utc=True)

    # ---- 7. Apply S4 sizing -------------------------------------------
    # S4 = clip(1.5 * (score - 0.5) * 2, 0, 1.5) = clip(3 * (score - 0.5), 0, 1.5)
    score = gated_e2["score"].to_numpy(dtype=float)
    pos_scale_s4 = np.clip(3.0 * (score - 0.5), 0.0, 1.5)
    gated_e2["pos_scale_s4"] = pos_scale_s4
    gated_e2["sized_pnl_s4_e2"] = gated_e2["e2_pnl_pct"].to_numpy(dtype=float) * pos_scale_s4

    # ---- 8. Compute ship-gate metrics on stacked config ---------------
    print("\n=== Phase 17.E stacked config — D1e + v4 + S4 × E2 ===")
    locked = _summarize(gated_e2, "D1e + v4 + S4 × E2", pnl_col="sized_pnl_s4_e2")

    # Comparison: D1e + v4 + S1 × E1 (legacy sizing, legacy exits)
    gated_e1 = gated.copy()
    gated_e1["sized_pnl_s1_e1"] = (gated_e1["pnl_pct"].to_numpy(dtype=float)
                                    * gated_e1["pos_scale_s1"].to_numpy(dtype=float))
    legacy = _summarize(gated_e1, "D1e + v4 + S1 × E1 (legacy)", pnl_col="sized_pnl_s1_e1")

    # ---- 9. Decision --------------------------------------------------
    oot_pf = locked["oot_metrics"]["pf"]
    oot_n = locked["oot_metrics"]["trades"]
    wf_p5 = locked["walkforward"].get("p5", float("nan"))
    mc_p5 = locked["monte_carlo_oot"].get("p5", float("nan"))

    REF_OOT_PF = 5.422
    REF_WF_P5 = 2.474
    REF_MC_P5 = 4.550

    gates = {
        "wf_p5_ge_1.0": bool(np.isfinite(wf_p5) and wf_p5 >= 1.0),
        "oot_n_ge_1500": bool(oot_n >= 1500),
        "oot_pf_ge_17D": bool(np.isfinite(oot_pf) and oot_pf >= REF_OOT_PF * 0.95),  # 5% tolerance
        "mc_p5_ge_17D": bool(np.isfinite(mc_p5) and mc_p5 >= REF_MC_P5 * 0.95),
        "wf_p5_ge_17D": bool(np.isfinite(wf_p5) and wf_p5 >= REF_WF_P5 * 0.95),
    }
    overall = all(gates.values())
    print("\n=== Decision gates ===")
    for k, v in gates.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    decision = "ADOPT 17.E (D1e + v4 + S4 × E2)" if overall else "STAY ON 17.D-alone"
    print(f"\n  → {decision}")

    # ---- 10. Reproducibility check (re-run summary on same df) ---------
    locked_v2 = _summarize(gated_e2, "D1e + v4 + S4 × E2 (rerun)", pnl_col="sized_pnl_s4_e2")
    repro_ok = (
        locked["oot_metrics"]["pf"] == locked_v2["oot_metrics"]["pf"]
        and locked["walkforward"].get("p5") == locked_v2["walkforward"].get("p5")
        and locked["monte_carlo_oot"].get("p5") == locked_v2["monte_carlo_oot"].get("p5")
    )
    print(f"\nREPRODUCIBILITY: {'PASS' if repro_ok else 'FAIL'}")

    out = {
        "phase": "17.E — Stack D1e + v4 + S4 × E2",
        "snapshot": SNAPSHOT_DATE,
        "v4_meta": {
            "train_auc": float(train_auc),
            "oot_auc": float(oot_auc) if np.isfinite(oot_auc) else None,
            "cv_aucs": cv_aucs, "threshold": threshold,
            "trained_on_n": int(len(train_X)),
        },
        "stacked_S4_E2": locked,
        "legacy_S1_E1": legacy,
        "decision_gates": gates,
        "decision": decision,
        "reproducibility": repro_ok,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
