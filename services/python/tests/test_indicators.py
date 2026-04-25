"""Behavior + reference tests for src/ml/indicators.py.

Three guarantees:
  1. EMA / RSI / MACD match the existing inline implementations in
     src/ml/features.py (so existing artifacts stay reproducible).
  2. KDJ values match a hand-computed reference (within 1e-9) on a small,
     deterministic fixture.
  3. No-lookahead: every indicator at bar i depends only on bars [..., i].
     We verify this by permuting future bars and asserting indicator values
     up to bar i are unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.ml.indicators import atr, ema, kdj, macd, rsi


# Deterministic fixture used across tests
def _close_path() -> pd.Series:
    rng = np.random.default_rng(42)
    return pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 200))))


def _ohlc_path() -> tuple[pd.Series, pd.Series, pd.Series]:
    """Synthetic OHLC where high>=close>=low for every bar."""
    rng = np.random.default_rng(43)
    close = pd.Series(100.0 * np.exp(np.cumsum(rng.normal(0, 0.005, 200))))
    high = close * (1.0 + np.abs(rng.normal(0, 0.002, 200)))
    low = close * (1.0 - np.abs(rng.normal(0, 0.002, 200)))
    return high, low, close


# --- 1. EMA / RSI / MACD parity with existing inline implementations --------


def test_ema_matches_pandas_inline():
    """ema(close, 20) == close.ewm(span=20, adjust=False).mean()  (the call
    site formula in regime.py and features.py)."""
    c = _close_path()
    expected = c.ewm(span=20, adjust=False).mean()
    pd.testing.assert_series_equal(ema(c, 20), expected, check_names=False)


def test_rsi_matches_inline_features_formula():
    """RSI matches the implementation in src/ml/features.py:_compute_rsi."""
    c = _close_path()
    delta = c.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    expected = 100 - (100 / (1 + rs))
    pd.testing.assert_series_equal(rsi(c, 14), expected, check_names=False)


def test_macd_matches_inline_formula():
    """MACD matches the inline definition in src/ml/features.py."""
    c = _close_path()
    ema_12 = c.ewm(span=12, adjust=False).mean()
    ema_26 = c.ewm(span=26, adjust=False).mean()
    expected_macd = ema_12 - ema_26
    expected_signal = expected_macd.ewm(span=9, adjust=False).mean()
    expected_hist = expected_macd - expected_signal
    out = macd(c, 12, 26, 9)
    pd.testing.assert_series_equal(out["macd"], expected_macd, check_names=False)
    pd.testing.assert_series_equal(out["signal"], expected_signal, check_names=False)
    pd.testing.assert_series_equal(out["histogram"], expected_hist, check_names=False)


# --- 2. KDJ reference values ------------------------------------------------


def test_kdj_hand_computed_reference():
    """KDJ on a tiny step-shaped fixture matches a hand-computation.

    Fixture: 9 bars where close climbs in steps. RSV at bar 8 (the first
    fully-warm bar) is 100 since close = max-of-window.
    """
    n = 9
    # Closes: 1, 2, ..., 9. high = close + 0.5, low = close - 0.5.
    closes = pd.Series([float(i + 1) for i in range(n)])
    highs = closes + 0.5
    lows = closes - 0.5

    out = kdj(highs, lows, closes, n=n, k_smooth=3, d_smooth=3)
    # At bar n-1 (i=8), RSV = 100 * (9 - 0.5) / (9.5 - 0.5) = 100 * 8.5 / 9.0
    expected_rsv_last = 100.0 * 8.5 / 9.0
    # K, D require k_smooth and d_smooth more bars — should still be NaN here.
    assert np.isnan(out["k"].iloc[-1])
    assert np.isnan(out["d"].iloc[-1])
    assert np.isnan(out["j"].iloc[-1])

    # Add 4 more bars to reach K and D warmup.
    closes2 = pd.concat([closes, pd.Series([float(n + 1 + i) for i in range(4)])], ignore_index=True)
    highs2 = closes2 + 0.5
    lows2 = closes2 - 0.5
    out2 = kdj(highs2, lows2, closes2, n=n, k_smooth=3, d_smooth=3)

    # RSV at the last bar (i=12, close=13): window is closes[4..12]=[5..13].
    # min_low = 4.5, max_high = 13.5, close = 13 → RSV = 100*(13-4.5)/(13.5-4.5).
    expected_rsv_12 = 100.0 * (13.0 - 4.5) / (13.5 - 4.5)
    # The full RSV series — we recompute and compare K to the SMA(RSV, 3).
    rolling_low = lows2.rolling(window=n, min_periods=n).min()
    rolling_high = highs2.rolling(window=n, min_periods=n).max()
    expected_rsv = 100.0 * (closes2 - rolling_low) / (rolling_high - rolling_low)
    expected_k = expected_rsv.rolling(window=3, min_periods=3).mean()
    expected_d = expected_k.rolling(window=3, min_periods=3).mean()
    expected_j = 3.0 * expected_k - 2.0 * expected_d

    pd.testing.assert_series_equal(out2["k"], expected_k, check_names=False)
    pd.testing.assert_series_equal(out2["d"], expected_d, check_names=False)
    pd.testing.assert_series_equal(out2["j"], expected_j, check_names=False)
    # And the RSV at i=12 hand-check:
    assert abs(expected_rsv.iloc[12] - expected_rsv_12) < 1e-9


def test_kdj_flat_range_yields_nan():
    """When the n-bar high == low (zero span), RSV is NaN, which propagates."""
    n = 9
    flat = pd.Series([100.0] * (n + 5))
    out = kdj(flat, flat, flat, n=n, k_smooth=3, d_smooth=3)
    # All values should be NaN (flat range = no information).
    assert out.isna().all().all()


def test_kdj_input_length_mismatch_raises():
    with pytest.raises(ValueError):
        kdj(pd.Series([1, 2, 3]), pd.Series([1, 2]), pd.Series([1, 2, 3]))


# --- 3. No-lookahead invariance ---------------------------------------------


def test_no_lookahead_ema():
    c = _close_path()
    out_full = ema(c, 20)
    # Permute bars 100..end. ema[..99] must be unchanged.
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(c) - 100)
    c_corrupt = c.copy()
    c_corrupt.iloc[100:] = c.iloc[100:].to_numpy()[perm]
    out_corrupt = ema(c_corrupt, 20)
    pd.testing.assert_series_equal(out_full.iloc[:100], out_corrupt.iloc[:100], check_names=False)


def test_no_lookahead_rsi():
    c = _close_path()
    out_full = rsi(c, 14)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(c) - 100)
    c_corrupt = c.copy()
    c_corrupt.iloc[100:] = c.iloc[100:].to_numpy()[perm]
    out_corrupt = rsi(c_corrupt, 14)
    pd.testing.assert_series_equal(out_full.iloc[:100], out_corrupt.iloc[:100], check_names=False)


def test_no_lookahead_macd():
    c = _close_path()
    out_full = macd(c)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(c) - 100)
    c_corrupt = c.copy()
    c_corrupt.iloc[100:] = c.iloc[100:].to_numpy()[perm]
    out_corrupt = macd(c_corrupt)
    for col in ("macd", "signal", "histogram"):
        pd.testing.assert_series_equal(out_full[col].iloc[:100], out_corrupt[col].iloc[:100], check_names=False)


def test_no_lookahead_kdj():
    h, l, c = _ohlc_path()
    out_full = kdj(h, l, c)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(c) - 100)
    h_c = h.copy(); l_c = l.copy(); c_c = c.copy()
    h_c.iloc[100:] = h.iloc[100:].to_numpy()[perm]
    l_c.iloc[100:] = l.iloc[100:].to_numpy()[perm]
    c_c.iloc[100:] = c.iloc[100:].to_numpy()[perm]
    out_corrupt = kdj(h_c, l_c, c_c)
    for col in ("k", "d", "j"):
        pd.testing.assert_series_equal(out_full[col].iloc[:100], out_corrupt[col].iloc[:100], check_names=False)


def test_no_lookahead_atr():
    h, l, c = _ohlc_path()
    out_full = atr(h, l, c, 14)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(c) - 100)
    h_c = h.copy(); l_c = l.copy(); c_c = c.copy()
    h_c.iloc[100:] = h.iloc[100:].to_numpy()[perm]
    l_c.iloc[100:] = l.iloc[100:].to_numpy()[perm]
    c_c.iloc[100:] = c.iloc[100:].to_numpy()[perm]
    out_corrupt = atr(h_c, l_c, c_c, 14)
    pd.testing.assert_series_equal(out_full.iloc[:100], out_corrupt.iloc[:100], check_names=False)
