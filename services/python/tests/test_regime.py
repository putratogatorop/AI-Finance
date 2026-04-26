"""Behavior-preserving regression tests for src/ml/regime.py.

These tests pin the new shared regime utility against the inline
implementations that previously lived in:
  - live_scanner_v2.py:190-235 (short)
  - live_scanner_long_v2.py:153-178 (long)
  - backtest_short_regime.py:63-111 (history)

Each test reproduces the inline math against a synthetic candle cache
and asserts that the new module returns the same boolean / dict.
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.ml.regime import (
    compute_long_regime,
    compute_short_regime,
    regime_history,
    regime_state_now,
)


def _make_candles_15m(close_path: list[float], start: datetime) -> pd.DataFrame:
    """Build a minimal 15m DataFrame from a list of closes."""
    n = len(close_path)
    ts = [start + timedelta(minutes=15 * i) for i in range(n)]
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(ts, utc=True),
            "open": close_path,
            "high": close_path,
            "low": close_path,
            "close": close_path,
            "volume": [1.0] * n,
        }
    )
    return df


def _bars_for_n_days(days: int) -> int:
    return days * 96


def _legacy_short_compute_regime(candle_cache, btc_df):
    """Verbatim copy of live_scanner_v2.compute_regime as of pre-refactor."""
    if len(btc_df) < 20 * 96:
        return False
    btc_daily = (
        btc_df.set_index("timestamp")
        .resample("1D")
        .agg({"close": "last"})
        .dropna()
    )
    if len(btc_daily) < 20:
        return False
    btc_ema20 = btc_daily["close"].ewm(span=20, adjust=False).mean()
    btc_bearish = float(btc_daily["close"].iloc[-1]) < float(btc_ema20.iloc[-1])
    if not btc_bearish:
        return False
    above_count = 0
    total_count = 0
    for pair, df in candle_cache.items():
        if pair == "BTC_USDT" or len(df) < 20 * 96:
            continue
        coin_daily = (
            df.set_index("timestamp")
            .resample("1D")
            .agg({"close": "last"})
            .dropna()
        )
        if len(coin_daily) < 20:
            continue
        ema20 = coin_daily["close"].ewm(span=20, adjust=False).mean()
        total_count += 1
        if float(coin_daily["close"].iloc[-1]) > float(ema20.iloc[-1]):
            above_count += 1
    breadth = above_count / total_count if total_count > 0 else 0.5
    return breadth < 0.60


def _legacy_long_compute_regime(candle_cache, btc_df):
    """Verbatim copy of live_scanner_long_v2.compute_regime as of pre-refactor."""
    if len(btc_df) < 20 * 96:
        return False
    btc_daily = btc_df.set_index("timestamp").resample("1D").agg({"close": "last"}).dropna()
    if len(btc_daily) < 20:
        return False
    btc_ema20 = btc_daily["close"].ewm(span=20, adjust=False).mean()
    btc_bullish = float(btc_daily["close"].iloc[-1]) > float(btc_ema20.iloc[-1])
    if not btc_bullish:
        return False
    above_count = 0
    total_count = 0
    for pair, df in candle_cache.items():
        if pair == "BTC_USDT" or len(df) < 20 * 96:
            continue
        coin_daily = df.set_index("timestamp").resample("1D").agg({"close": "last"}).dropna()
        if len(coin_daily) < 20:
            continue
        ema20 = coin_daily["close"].ewm(span=20, adjust=False).mean()
        total_count += 1
        if float(coin_daily["close"].iloc[-1]) > float(ema20.iloc[-1]):
            above_count += 1
    breadth = above_count / total_count if total_count > 0 else 0.5
    return breadth > 0.40


# Fixtures -------------------------------------------------------------------

START = datetime(2026, 1, 1, tzinfo=UTC)
N_BARS_25D = _bars_for_n_days(25)


@pytest.fixture
def warm_uptrend_cache():
    """BTC slowly rising → bullish gate; coins in lockstep → high breadth."""
    rng = np.random.default_rng(42)
    btc_close = 100.0 * np.exp(np.linspace(0, 0.10, N_BARS_25D))  # +10% over 25d
    btc = _make_candles_15m(btc_close.tolist(), START)
    coins = {}
    for i in range(50):
        # Each coin is a noisy version of BTC also trending up.
        path = btc_close * np.exp(rng.normal(0, 0.01, N_BARS_25D)) * (1 + i * 0.001)
        coins[f"COIN{i}_USDT"] = _make_candles_15m(path.tolist(), START)
    coins["BTC_USDT"] = btc
    return coins, btc


@pytest.fixture
def warm_downtrend_cache():
    """BTC falling → bearish gate; coins falling → low breadth."""
    rng = np.random.default_rng(43)
    btc_close = 100.0 * np.exp(np.linspace(0, -0.10, N_BARS_25D))  # -10% over 25d
    btc = _make_candles_15m(btc_close.tolist(), START)
    coins = {}
    for i in range(50):
        path = btc_close * np.exp(rng.normal(0, 0.01, N_BARS_25D)) * (1 - i * 0.001)
        coins[f"COIN{i}_USDT"] = _make_candles_15m(path.tolist(), START)
    coins["BTC_USDT"] = btc
    return coins, btc


@pytest.fixture
def cold_cache():
    """Insufficient warmup — should always return regime not allowed."""
    btc_close = list(np.linspace(100.0, 102.0, 100))
    btc = _make_candles_15m(btc_close, START)
    coins = {"BTC_USDT": btc, "COINX_USDT": _make_candles_15m(btc_close, START)}
    return coins, btc


# Tests ----------------------------------------------------------------------


def test_short_regime_matches_legacy_uptrend(warm_uptrend_cache):
    cache, btc = warm_uptrend_cache
    legacy = _legacy_short_compute_regime(cache, btc)
    new = compute_short_regime(cache, btc)
    assert new == legacy
    # Sanity: in an uptrend, shorts should NOT be allowed.
    assert legacy is False


def test_short_regime_matches_legacy_downtrend(warm_downtrend_cache):
    cache, btc = warm_downtrend_cache
    legacy = _legacy_short_compute_regime(cache, btc)
    new = compute_short_regime(cache, btc)
    assert new == legacy
    # Sanity: in a downtrend, shorts SHOULD be allowed.
    assert legacy is True


def test_long_regime_matches_legacy_uptrend(warm_uptrend_cache):
    cache, btc = warm_uptrend_cache
    legacy = _legacy_long_compute_regime(cache, btc)
    new = compute_long_regime(cache, btc)
    assert new == legacy
    # Sanity: longs should be allowed.
    assert legacy is True


def test_long_regime_matches_legacy_downtrend(warm_downtrend_cache):
    cache, btc = warm_downtrend_cache
    legacy = _legacy_long_compute_regime(cache, btc)
    new = compute_long_regime(cache, btc)
    assert new == legacy
    assert legacy is False


def test_cold_cache_returns_not_ready(cold_cache):
    cache, btc = cold_cache
    state = regime_state_now(cache, btc)
    assert state["ready"] is False
    assert state["allows_short"] is False
    assert state["allows_long"] is False
    # Legacy paths return False for short/long when warmup insufficient.
    assert _legacy_short_compute_regime(cache, btc) is False
    assert _legacy_long_compute_regime(cache, btc) is False


def test_state_includes_breadth(warm_downtrend_cache):
    cache, btc = warm_downtrend_cache
    state = regime_state_now(cache, btc)
    assert state["ready"] is True
    assert state["btc_bearish"] is True
    assert state["btc_bullish"] is False
    assert isinstance(state["breadth_pct"], float)
    assert 0.0 <= state["breadth_pct"] <= 1.0


def test_regime_history_keys_and_values(warm_downtrend_cache):
    cache, btc = warm_downtrend_cache
    # The history function uses a different ts_col convention (`open_time`).
    # Provide that view by aliasing the column.
    btc_alias = btc.rename(columns={"timestamp": "open_time"})
    cache_alias = {
        sym: df.rename(columns={"timestamp": "open_time"})
        for sym, df in cache.items()
    }
    # Use BTCUSDT key (backtest convention) instead of BTC_USDT.
    cache_alias["BTCUSDT"] = cache_alias.pop("BTC_USDT")
    hist = regime_history(cache_alias, btc_alias, ts_col="open_time", btc_key="BTCUSDT")
    assert len(hist) > 0
    sample = next(iter(hist.values()))
    for key in ("btc_bearish", "btc_bullish", "breadth", "breadth_low",
                "breadth_high", "allowed", "allowed_long"):
        assert key in sample
    # Spot check: in our synthetic downtrend the latest day should be allowed-short.
    latest = hist[max(hist.keys())]
    assert latest["btc_bearish"] is True
    assert latest["allowed"] is True
