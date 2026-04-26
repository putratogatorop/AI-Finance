"""Phase 16.2 — Train a fresh classifier on D1 SHORT trades.

The Phase-15 finding was that the v2 classifier (trained on D0 volume-detector
trades) was destructive when applied to D1 ATR-detector trades. This script
trains a v3 classifier on the D1 distribution itself.

Protocol (mirrors train_bigmover_combined_v2_outcome_label.py):
  1. Re-run D1 SHORT detector on the 2026-04-01 snapshot, capturing every
     trade with entry metadata + final pnl_pct.
  2. Build the 16-feature set (services/python/src/ml/bigmover_combined/features.py).
  3. Outcome-aligned label: label = 1 iff pnl_pct > 0 (Phase 9).
  4. Train LR with C=1.0, class_weight='balanced', 5-fold purged time-series
     CV (purge=192 bars).
  5. NO LightGBM escalation regardless of AUC (per Phase 16 plan: lesson from
     Phases 9/10 is more model complexity overfits tiny windows).
  6. Save model + meta + sizing to services/python/models/d1_short_v3.{joblib,
     _meta.json, _sizing.json}, and the trade ledger to services/python/data/
     d1_short_trades.csv for re-use by the backtest script.

Run from services/python/:
    python scripts/train_d1_classifier_v3.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
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

CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
UNIVERSE_PATH = REPO_ROOT / "data" / "snapshots" / f"universe_{SNAPSHOT_DATE}.parquet"

DATA_DIR = SERVICES_PY / "data"
MODELS_DIR = SERVICES_PY / "models"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

TRADES_OUT = DATA_DIR / "d1_short_trades.csv"
MODEL_OUT = MODELS_DIR / "d1_short_v3.joblib"
META_OUT = MODELS_DIR / "d1_short_v3_meta.json"
SIZING_OUT = MODELS_DIR / "d1_short_v3_sizing.json"

# D1 SHORT detector + simulation constants — must match Phase 15.
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

LR_C = 1.0
PURGE_BARS = 192


# --- Reused panels ----------------------------------------------------------


def _build_btc_trend_score_15m(candles_15m: pd.DataFrame) -> pd.Series:
    btc = candles_15m["close"].astype(float)
    btc_4h = btc.resample("4h", label="right", closed="right").last()
    md = _macd(btc_4h, 12, 26, 9)
    spread = ((md["macd"] - md["signal"]).shift(1) / btc_4h.shift(1))
    score_4h = spread.clip(-BTC_TREND_K, BTC_TREND_K) / BTC_TREND_K
    return score_4h.reindex(btc.index, method="ffill")


def _build_rapid_exit_panel(candles_15m: pd.DataFrame) -> pd.Series:
    """Phase 14 V2 rapid-rally exit panel (BTC daily NOT all-bearish AND BTC up >3% in 24h)."""
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


def _detect_short_d1_atr_drop(close, high, low, open_, volume) -> np.ndarray:
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
        out[i] = True
    return out


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


def _detect_d1_short_trades(candles, btc_score_15m, rapid_exit_15m) -> pd.DataFrame:
    rows = []
    btc_score_naive = btc_score_15m.copy()
    btc_score_naive.index = pd.to_datetime(btc_score_naive.index, utc=True).tz_convert(None)
    rapid_naive = rapid_exit_15m.copy()
    rapid_naive.index = pd.to_datetime(rapid_naive.index, utc=True).tz_convert(None)

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

        triggers = _detect_short_d1_atr_drop(close, high, low, open_, volume)
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
            pos_scale = min(1.5 * abs(score), 1.5)
            if pos_scale <= 0:
                continue
            t = _simulate_short(close, high, low, open_, i, rally_flag=rally)
            if t is None:
                continue
            t["sized_pnl"] = t["pnl_pct"] * pos_scale
            t["pos_scale"] = pos_scale
            t["asset"] = asset
            t["symbol"] = asset
            t["direction"] = "short"
            entry_idx = i + 1 if i + 1 < len(sub) else i
            t["entry_time"] = pd.Timestamp(sub["timestamp"].iloc[entry_idx])
            t["btc_score"] = float(score)
            rows.append(t)
        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(assets)} assets processed, trades={len(rows):,}", flush=True)
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


def main():
    started = time.monotonic()
    np.random.seed(RANDOM_SEED)

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

    print("\n=== Detecting D1 SHORT trades ===")
    trades = _detect_d1_short_trades(candles, btc_score_15m, rapid_exit_15m)
    if len(trades) == 0:
        raise RuntimeError("no D1 SHORT trades detected — check snapshot/path")
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades.to_csv(TRADES_OUT, index=False)
    print(f"  saved {len(trades):,} trades → {TRADES_OUT}")

    # universe listed_since for the breadth feature
    universe_listed_since: dict[str, pd.Timestamp] = {}
    if UNIVERSE_PATH.exists():
        u = pd.read_parquet(UNIVERSE_PATH)
        if "asset" in u.columns and "listed_since" in u.columns:
            for _, row in u.iterrows():
                asset_with_usdt = f"{row['asset']}USDT"
                ls = row["listed_since"]
                if pd.notna(ls):
                    universe_listed_since[asset_with_usdt] = pd.Timestamp(ls, tz="UTC")
        print(f"  universe listed_since entries: {len(universe_listed_since)}", flush=True)

    # outcome-aligned label
    trades["label"] = (trades["pnl_pct"] > 0).astype(int)
    pos_rate = float(trades["label"].mean())
    print(f"  outcome-label positive rate: {pos_rate:.1%}", flush=True)

    print("\n=== Building features ===")
    ctx = FeatureContext(
        candles, btc_asset_key="BTCUSDT",
        universe_listed_since=universe_listed_since,
    )
    feat_rows = []
    for i, row in enumerate(trades.itertuples(index=False)):
        feats = ctx._build_for_typed(row.symbol, row.entry_time, row.direction)
        feat_rows.append([feats[n] for n in FEATURE_NAMES] + [feats["is_short"]])
        if (i + 1) % 5000 == 0:
            print(f"  features: {i+1}/{len(trades)}", flush=True)
    cols = FEATURE_NAMES + ["is_short"]
    X = pd.DataFrame(feat_rows, columns=cols)

    finite_mask = X.notna().all(axis=1)
    print(f"  feature-complete rows: {int(finite_mask.sum())}/{len(trades)}", flush=True)
    trades = trades[finite_mask].reset_index(drop=True)
    X = X[finite_mask].reset_index(drop=True)
    y = trades["label"].astype(int).to_numpy()

    # write the *feature-complete* trades back so the backtest script can rebuild
    # the same X without re-running detection.
    feat_full = X.copy()
    feat_full.insert(0, "entry_time", trades["entry_time"].astype("datetime64[ns, UTC]").astype(str))
    feat_full.insert(1, "asset", trades["asset"])
    feat_full["pnl_pct"] = trades["pnl_pct"]
    feat_full["sized_pnl"] = trades["sized_pnl"]
    feat_full["pos_scale"] = trades["pos_scale"]
    feat_full["exit"] = trades["exit"]
    feat_full["bars"] = trades["bars"]
    feat_full["btc_score"] = trades["btc_score"]
    feat_full.to_csv(DATA_DIR / "d1_short_trades_with_features.csv", index=False)
    print(f"  saved features-aligned ledger → {DATA_DIR / 'd1_short_trades_with_features.csv'}")

    is_train = (trades["entry_time"] < TRAIN_OOT_BOUNDARY).to_numpy()
    is_oot = ~is_train
    print(f"  train: {is_train.sum():,}  oot: {is_oot.sum():,}", flush=True)

    train_X = X.loc[is_train].to_numpy(dtype=float)
    train_y = y[is_train]
    oot_X = X.loc[is_oot].to_numpy(dtype=float)
    oot_y = y[is_oot]

    print("\n=== LR (C=1.0, class_weight=balanced) — 5-fold purged time CV ===")
    cv_aucs = []
    for fold_i, (tr_idx, va_idx) in enumerate(
        _purged_kfold_indices(len(train_X), k=5, purge=PURGE_BARS)
    ):
        if len(np.unique(train_y[tr_idx])) < 2 or len(np.unique(train_y[va_idx])) < 2:
            continue
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced",
            )),
        ])
        pipe.fit(train_X[tr_idx], train_y[tr_idx])
        proba_val = pipe.predict_proba(train_X[va_idx])[:, 1]
        auc = roc_auc_score(train_y[va_idx], proba_val)
        cv_aucs.append(float(auc))
        print(f"  fold {fold_i}: AUC={auc:.4f} (n_train={len(tr_idx):,}, n_val={len(va_idx):,})")

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced",
        )),
    ])
    lr_pipe.fit(train_X, train_y)
    train_auc = roc_auc_score(train_y, lr_pipe.predict_proba(train_X)[:, 1])
    oot_auc = (
        roc_auc_score(oot_y, lr_pipe.predict_proba(oot_X)[:, 1])
        if len(np.unique(oot_y)) >= 2 else float("nan")
    )
    print(f"\n  full-train AUC: {train_auc:.4f}    OOT AUC: {oot_auc:.4f}    "
          f"gap: {train_auc - oot_auc:.4f}")

    # Lock threshold at TRAIN q0.5
    train_scores = lr_pipe.predict_proba(train_X)[:, 1]
    threshold = float(np.percentile(train_scores, 50.0))
    print(f"  threshold @ TRAIN q0.5: {threshold:.4f}", flush=True)

    joblib.dump(lr_pipe, MODEL_OUT)
    meta = {
        "model": "logistic_regression",
        "feature_names": cols,
        "label_kind": "outcome",
        "label_thresholds": {
            "kind": "outcome_pnl_positive",
            "exit_sl": STOP_LOSS_PCT,
            "exit_tp": TAKE_PROFIT_PCT,
            "exit_timeout": TIMEOUT_BARS,
            "exit_friction": FRICTION_PCT,
        },
        "detector": "D1_atr_drop_2.5_vol_2 (Phase 15 winner)",
        "lr_C": LR_C,
        "purge_bars": PURGE_BARS,
        "cv_aucs": cv_aucs,
        "train_auc": float(train_auc),
        "oot_auc": float(oot_auc) if np.isfinite(oot_auc) else None,
        "threshold_train_q50": threshold,
        "trained_on_n": int(len(train_X)),
        "trained_on_pos_rate": float(np.mean(train_y)),
        "snapshot_date": SNAPSHOT_DATE,
        "trades_csv": str(TRADES_OUT),
    }
    META_OUT.write_text(json.dumps(meta, indent=2, default=str))
    SIZING_OUT.write_text(json.dumps(default_sizing_config().to_dict(), indent=2))

    print(f"\nLR saved to {MODEL_OUT}")
    print(f"meta written to {META_OUT}")
    print(f"sizing written to {SIZING_OUT}")
    print(f"\ntotal wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
