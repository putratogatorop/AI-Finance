"""Feature-builder tests for src/ml/bigmover_combined/features.py.

Three guarantees we hold (mirrors continuation predictor's tests):
  1. **No lookahead.** Permuting future bars does NOT change feature values
     for any trade. Strongest practical guarantee.
  2. **Warmup respected.** Features that need N prior bars are NaN before
     bar N (and finite after).
  3. **Direction-aware features behave correctly.** is_short toggles by
     direction; concurrent_dir_breadth flips meaning.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.ml.bigmover_combined.features import FEATURE_NAMES, FeatureContext


def _synthetic_candles(n_assets: int = 6, n_bars: int = 800) -> pd.DataFrame:
    """Synthetic candle frame with `n_assets` symbols (one is BTCUSDT)."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    rows = []
    syms = ["BTCUSDT"] + [f"COIN{i}USDT" for i in range(n_assets - 1)]
    for sym in syms:
        log_ret = rng.normal(0.0, 0.005, n_bars)
        close = 100.0 * np.exp(np.cumsum(log_ret))
        opens = np.concatenate([[100.0], close[:-1]])
        highs = np.maximum(opens, close) * (1 + np.abs(rng.normal(0, 0.001, n_bars)))
        lows = np.minimum(opens, close) * (1 - np.abs(rng.normal(0, 0.001, n_bars)))
        vol = np.abs(rng.normal(1000.0, 200.0, n_bars))
        ts = [start + timedelta(minutes=15 * i) for i in range(n_bars)]
        rows.append(
            pd.DataFrame(
                {
                    "asset": sym,
                    "timestamp": pd.to_datetime(ts, utc=True),
                    "open": opens,
                    "high": highs,
                    "low": lows,
                    "close": close,
                    "volume": vol,
                }
            )
        )
    return pd.concat(rows, ignore_index=True)


def test_features_invariant_under_future_permutation():
    """Mutate every bar AFTER each trade's entry_time. Features must not change."""
    candles = _synthetic_candles()
    # Pick entry_time well past warmup (BTC trend SMA-50 daily = 50 days = 4800 bars
    # — too many for our 800-bar synthetic. Use bar 600 — still satisfies the
    # 14/26/96 lookbacks, BTC trend score will be NaN in this fixture; that's OK,
    # we only test invariance, not value.).
    entry_ts = candles["timestamp"].iloc[600]
    trade = {"symbol": "COIN0USDT", "entry_time": entry_ts, "direction": "short"}

    ctx_orig = FeatureContext(candles)
    feats_orig = ctx_orig.build_for(trade)

    rng = np.random.default_rng(123)
    corrupted = candles.copy()
    mask_future = corrupted["timestamp"] > entry_ts
    perm = rng.permutation(int(mask_future.sum()))
    for col in ["open", "high", "low", "close", "volume"]:
        future_vals = corrupted.loc[mask_future, col].to_numpy()
        corrupted.loc[mask_future, col] = future_vals[perm]
    assert not np.array_equal(
        corrupted.loc[mask_future, "close"].to_numpy(),
        candles.loc[mask_future, "close"].to_numpy(),
    )

    ctx_corrupt = FeatureContext(corrupted)
    feats_corrupt = ctx_corrupt.build_for(trade)
    for name in FEATURE_NAMES:
        a, b = feats_orig[name], feats_corrupt[name]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, f"{name!r} changed under future permutation: orig={a}, corrupted={b}"


def test_warm_features_are_finite_for_indicators():
    """Past warmup, indicator features should be finite (BTC trend may still be NaN
    given the short fixture; that's OK — we test only the per-asset indicators)."""
    candles = _synthetic_candles(n_bars=400)
    entry_ts = candles["timestamp"].iloc[300]
    trade = {"symbol": "COIN0USDT", "entry_time": entry_ts, "direction": "short"}
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    for name in [
        "vol_ratio", "price_drop_pct", "price_rise_pct", "accel_atr_norm",
        "bars_since_high32",
        "price_vs_ema9_pct", "price_vs_ema50_pct",
        "rsi14_delta_1bar", "rsi14_delta_4bar",
        "macd_signal_spread_norm", "macd_hist_momentum",
        "kdj_j", "kdj_k",
        "concurrent_dir_breadth",
    ]:
        assert np.isfinite(feats[name]), f"{name} should be finite past warmup but got {feats[name]}"


def test_warmup_yields_nan_for_indicators():
    """At a very early bar, indicator features should be NaN."""
    candles = _synthetic_candles(n_bars=300)
    entry_ts = candles["timestamp"].iloc[5]  # before any rolling lookback warms up
    trade = {"symbol": "COIN0USDT", "entry_time": entry_ts, "direction": "short"}
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    # macd_signal_spread_norm uses EMA (ewm has no min_periods), so it's
    # finite immediately — exclude from warmup-NaN list.
    for name in ["price_drop_pct", "rsi14_delta_4bar", "kdj_j", "bars_since_high32"]:
        assert np.isnan(feats[name]), f"{name} should be NaN at warmup, got {feats[name]}"


def test_is_short_toggles_with_direction():
    candles = _synthetic_candles(n_bars=400)
    entry_ts = candles["timestamp"].iloc[300]
    ctx = FeatureContext(candles)
    short_feats = ctx.build_for(
        {"symbol": "COIN0USDT", "entry_time": entry_ts, "direction": "short"}
    )
    long_feats = ctx.build_for(
        {"symbol": "COIN0USDT", "entry_time": entry_ts, "direction": "long"}
    )
    assert short_feats["is_short"] == 1.0
    assert long_feats["is_short"] == 0.0


def test_concurrent_dir_breadth_flips_with_direction():
    """concurrent_dir_breadth_short + concurrent_dir_breadth_long should sum to ~1
    minus the fraction of universe with exactly 0 4h-return (vanishingly small in
    the synthetic random walk)."""
    candles = _synthetic_candles(n_bars=400)
    entry_ts = candles["timestamp"].iloc[300]
    ctx = FeatureContext(candles)
    s = ctx.build_for(
        {"symbol": "COIN0USDT", "entry_time": entry_ts, "direction": "short"}
    )["concurrent_dir_breadth"]
    l = ctx.build_for(
        {"symbol": "COIN0USDT", "entry_time": entry_ts, "direction": "long"}
    )["concurrent_dir_breadth"]
    if np.isfinite(s) and np.isfinite(l):
        assert 0.99 <= s + l <= 1.01


def test_unknown_symbol_yields_all_nan_for_features():
    candles = _synthetic_candles(n_bars=400)
    trade = {
        "symbol": "GHOSTUSDT",
        "entry_time": candles["timestamp"].iloc[300],
        "direction": "short",
    }
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    for name in FEATURE_NAMES:
        assert np.isnan(feats[name]), f"{name} should be NaN for unknown symbol"
    # is_short is direction-only metadata — still set
    assert feats["is_short"] == 1.0


def test_feature_dict_has_all_columns():
    candles = _synthetic_candles(n_bars=400)
    trade = {
        "symbol": "COIN0USDT",
        "entry_time": candles["timestamp"].iloc[300],
        "direction": "short",
    }
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    expected = set(FEATURE_NAMES) | {"is_short"}
    assert set(feats.keys()) == expected
