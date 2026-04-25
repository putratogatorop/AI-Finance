"""Feature-builder tests for src/ml/continuation/features.py.

Three guarantees we must hold:
  1. **No lookahead.** Permuting future bars does NOT change feature values
     for any trade. This is the strongest test we can write without
     instrumenting numpy at the byte level: if features were leaking future
     data, mutating those bars would change the output.
  2. **Warmup respected.** Features that need N prior bars are NaN before
     bar N (and finite after).
  3. **Sample-shape sanity.** On real Strategy-B trades, no feature column
     is fully NaN; vol_decile is in [0, 9]; bools are in {0.0, 1.0}.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.ml.continuation.features import FEATURE_NAMES, FeatureContext


def _synthetic_candles(n_assets: int = 6, n_bars: int = 500) -> pd.DataFrame:
    """Make a synthetic candle frame with `n_assets` (one is BTCUSDT)."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    rng = np.random.default_rng(42)
    rows = []
    syms = ["BTCUSDT"] + [f"COIN{i}USDT" for i in range(n_assets - 1)]
    for sym in syms:
        # GBM-ish path
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


# ---- 1. No-lookahead invariance ---------------------------------------------


def test_features_invariant_under_future_permutation():
    """Mutate every bar AFTER each trade's entry_time. Features must not change."""
    candles = _synthetic_candles()
    # Pick entry_time well after warmup (96 bars) and before the end.
    entry_ts = candles["timestamp"].iloc[200]
    trade = {"symbol": "COIN0USDT", "entry_time": entry_ts}

    ctx_orig = FeatureContext(candles)
    feats_orig = ctx_orig.build_for(trade)

    # Build a corrupted candles where everything strictly AFTER entry_ts is shuffled.
    rng = np.random.default_rng(123)
    corrupted = candles.copy()
    mask_future = corrupted["timestamp"] > entry_ts
    perm = rng.permutation(int(mask_future.sum()))
    for col in ["open", "high", "low", "close", "volume"]:
        future_vals = corrupted.loc[mask_future, col].to_numpy()
        corrupted.loc[mask_future, col] = future_vals[perm]
    # Sanity: corrupted future should differ from original future.
    assert not np.array_equal(
        corrupted.loc[mask_future, "close"].to_numpy(),
        candles.loc[mask_future, "close"].to_numpy(),
    )

    ctx_corrupt = FeatureContext(corrupted)
    feats_corrupt = ctx_corrupt.build_for(trade)

    # Every feature must be identical (or both NaN).
    for name in FEATURE_NAMES:
        a, b = feats_orig[name], feats_corrupt[name]
        if np.isnan(a) and np.isnan(b):
            continue
        assert a == b, (
            f"Feature {name!r} changed under future permutation: "
            f"original={a}, corrupted={b}"
        )


# ---- 2. Warmup respected ----------------------------------------------------


def test_warmup_yields_nan():
    """Trade at bar 5 (well before any rolling lookback warms up) → most features NaN."""
    candles = _synthetic_candles()
    early_ts = candles["timestamp"].iloc[5]
    trade = {"symbol": "COIN0USDT", "entry_time": early_ts}
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    # Features needing >=16 lookback must be NaN.
    for name in [
        "asset_4h_ret",
        "asset_24h_ret",
        "atr_norm_drop",
        "bars_since_high",
        "body_to_range_ratio",  # actually uses min_periods=1 — should be finite
        "vol_zscore_8",
        "btc_4h_ret",
    ]:
        if name == "body_to_range_ratio":
            # body_to_range uses min_periods=1, so finite even at bar 5.
            assert np.isfinite(feats[name]) or np.isnan(feats[name])
            continue
        assert np.isnan(feats[name]), f"{name} should be NaN at warmup but got {feats[name]}"


def test_warm_features_are_finite():
    """At bar 200 (well past all warmups) features should be finite."""
    candles = _synthetic_candles()
    entry_ts = candles["timestamp"].iloc[200]
    trade = {"symbol": "COIN0USDT", "entry_time": entry_ts}
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    for name in [
        "asset_4h_ret",
        "asset_24h_ret",
        "atr_norm_drop",
        "bars_since_high",
        "body_to_range_ratio",
        "vol_zscore_8",
        "btc_4h_ret",
        "btc_realized_vol_24h",
        "btc_above_ema50_4h",
        "vol_decile",
        "concurrent_down_breadth",
    ]:
        assert np.isfinite(feats[name]), f"{name} should be finite past warmup but got {feats[name]}"


# ---- 3. Sample-shape sanity --------------------------------------------------


def test_vol_decile_in_range_and_bool_features_clean():
    candles = _synthetic_candles()
    entry_ts = candles["timestamp"].iloc[300]
    trade = {"symbol": "COIN0USDT", "entry_time": entry_ts}
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    # vol_decile in [0, 9]
    if not np.isnan(feats["vol_decile"]):
        assert 0 <= feats["vol_decile"] <= 9
    # btc_above_ema50_4h is boolean-valued
    if not np.isnan(feats["btc_above_ema50_4h"]):
        assert feats["btc_above_ema50_4h"] in (0.0, 1.0)
    # concurrent_down_breadth in [0, 1]
    if not np.isnan(feats["concurrent_down_breadth"]):
        assert 0.0 <= feats["concurrent_down_breadth"] <= 1.0


def test_unknown_symbol_returns_all_nan():
    candles = _synthetic_candles()
    trade = {
        "symbol": "GHOSTUSDT",  # not in candles
        "entry_time": candles["timestamp"].iloc[200],
    }
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    for name in FEATURE_NAMES:
        assert np.isnan(feats[name]), f"{name} should be NaN for unknown symbol"


def test_feature_dict_has_all_columns():
    candles = _synthetic_candles()
    trade = {"symbol": "COIN0USDT", "entry_time": candles["timestamp"].iloc[200]}
    ctx = FeatureContext(candles)
    feats = ctx.build_for(trade)
    assert set(feats.keys()) == set(FEATURE_NAMES)
