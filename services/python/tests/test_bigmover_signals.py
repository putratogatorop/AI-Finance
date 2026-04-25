"""Unit tests for src/ml/bigmover_signals.py.

Protects against the look-ahead bias class that was fixed in commit
backtest_bigmover_longshort_3y_leakfix.py: multi_bar_confirm used
close.shift(-1) while the simulator entered at open_[i], a 2-bar leak. Any
regression of that pattern (positive-offset shift, centered rolling, intrabar
fill) will make `test_multi_bar_confirm_no_lookahead` fail.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.bigmover_signals import (
    PRICE_LOOKBACK,
    VOL_MA_PERIOD,
    _self_check_no_lookahead,
    baseline_entry_mask_long,
    baseline_entry_mask_short,
    compute_atr,
    detect_signal,
)


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """200-bar synthetic OHLCV with occasional vol spikes + price drops."""
    rng = np.random.default_rng(42)
    n = 200
    close = 100 + rng.normal(0, 1, n).cumsum()
    # Force a vol-spike + drop at bar 150 to make baseline fire there for shorts.
    drops = np.zeros(n)
    drops[150] = -8.0  # -8% drop on bar 150
    close = close + drops
    high = close + np.abs(rng.normal(0, 0.3, n))
    low = close - np.abs(rng.normal(0, 0.3, n))
    volume = rng.uniform(1.0, 2.0, n)
    volume[150] = 10.0  # vol spike matching the drop
    return pd.DataFrame({
        "timestamp": np.arange(n, dtype=np.int64) * 900,
        "open": close,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_baseline_short_returns_bool_series_of_len_n(synthetic_df):
    mask = baseline_entry_mask_short(synthetic_df)
    assert len(mask) == len(synthetic_df)
    assert mask.dtype == bool


def test_baseline_long_returns_bool_series_of_len_n(synthetic_df):
    mask = baseline_entry_mask_long(synthetic_df)
    assert len(mask) == len(synthetic_df)
    assert mask.dtype == bool


def test_baseline_short_below_minimum_bars_is_all_false():
    too_small = pd.DataFrame({
        "timestamp": np.arange(PRICE_LOOKBACK + VOL_MA_PERIOD - 1, dtype=np.int64),
        "open": np.ones(PRICE_LOOKBACK + VOL_MA_PERIOD - 1),
        "high": np.ones(PRICE_LOOKBACK + VOL_MA_PERIOD - 1),
        "low": np.ones(PRICE_LOOKBACK + VOL_MA_PERIOD - 1),
        "close": np.ones(PRICE_LOOKBACK + VOL_MA_PERIOD - 1),
        "volume": np.ones(PRICE_LOOKBACK + VOL_MA_PERIOD - 1),
    })
    assert not baseline_entry_mask_short(too_small).any()
    assert not baseline_entry_mask_long(too_small).any()


def test_compute_atr_never_negative_tracks_range(synthetic_df):
    atr = compute_atr(synthetic_df)
    # ATR is NaN for the first (period-1) bars; the rest must be non-negative.
    assert (atr.dropna() >= 0).all()
    # Not wildly implausible vs median bar range
    bar_range = (synthetic_df["high"] - synthetic_df["low"]).median()
    assert atr.dropna().median() > 0
    assert atr.dropna().median() < 10 * bar_range


def test_detect_signal_baseline_short_equals_baseline_mask(synthetic_df):
    atr = compute_atr(synthetic_df)
    direct = baseline_entry_mask_short(synthetic_df)
    dispatched = detect_signal(synthetic_df, "baseline", "short", atr)
    assert (direct.values == dispatched.values).all()


def test_detect_signal_price_accel_atr_is_subset_of_baseline(synthetic_df):
    atr = compute_atr(synthetic_df)
    base = baseline_entry_mask_short(synthetic_df)
    gated = detect_signal(synthetic_df, "price_accel_atr", "short", atr)
    # Every price_accel_atr hit must also be a baseline hit.
    assert (gated & ~base).sum() == 0


def test_detect_signal_multi_bar_confirm_uses_prev_baseline_not_current(synthetic_df):
    """mask[j] fires iff baseline[j-1] AND close[j] moves in trade direction."""
    atr = compute_atr(synthetic_df)
    base = baseline_entry_mask_short(synthetic_df)
    mask = detect_signal(synthetic_df, "multi_bar_confirm", "short", atr)
    prev_close = synthetic_df["close"].shift(1)
    # For every True index in mask, verify baseline fired at the previous bar
    # and close[j] < close[j-1].
    true_idx = np.where(mask.values)[0]
    for j in true_idx:
        assert bool(base.iloc[j - 1]), f"mask[{j}] true but baseline[{j - 1}] false"
        assert synthetic_df["close"].iloc[j] < prev_close.iloc[j], (
            f"mask[{j}] true but close did not move for short"
        )


def test_detect_signal_multi_bar_confirm_no_lookahead():
    """Mutating future bars must not change past mask values."""
    _self_check_no_lookahead()  # raises AssertionError on regression


def test_detect_signal_unknown_signal_raises(synthetic_df):
    atr = compute_atr(synthetic_df)
    with pytest.raises(ValueError, match="unknown signal"):
        detect_signal(synthetic_df, "nonexistent", "short", atr)


def test_detect_signal_unknown_direction_raises(synthetic_df):
    atr = compute_atr(synthetic_df)
    with pytest.raises(ValueError, match="unknown direction"):
        detect_signal(synthetic_df, "baseline", "sideways", atr)
