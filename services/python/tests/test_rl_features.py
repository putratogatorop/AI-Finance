"""Tests for cross-token and BTC/ETH indicator features."""

import numpy as np
import pandas as pd
import pytest

from src.rl.features import build_cross_token_features, build_indicator_features


def _make_price_series(n: int, base: float, seed: int) -> pd.DataFrame:
    """Generate fake OHLCV data with a random-walk close price."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(0, 0.01, size=n)
    close = base * np.cumprod(1 + returns)
    high = close * (1 + rng.uniform(0, 0.005, size=n))
    low = close * (1 - rng.uniform(0, 0.005, size=n))
    open_ = close * (1 + rng.normal(0, 0.002, size=n))
    volume = rng.uniform(100, 10000, size=n)
    return pd.DataFrame({
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


class TestBuildCrossTokenFeatures:
    """Tests for build_cross_token_features."""

    def test_returns_three_columns(self):
        n = 500
        btc = _make_price_series(n, 40000, seed=1)
        alt = _make_price_series(n, 100, seed=2)
        result = build_cross_token_features(alt["close"], btc["close"])
        assert list(result.columns) == [
            "btc_alt_beta",
            "btc_alt_corr",
            "btc_alt_residual",
        ]

    def test_values_finite_after_warmup(self):
        n = 500
        window = 180
        btc = _make_price_series(n, 40000, seed=3)
        alt = _make_price_series(n, 100, seed=4)
        result = build_cross_token_features(
            alt["close"], btc["close"], window=window
        )
        after_warmup = result.iloc[window + 1:]
        for col in ["btc_alt_beta", "btc_alt_corr"]:
            vals = after_warmup[col].dropna()
            assert len(vals) > 0, f"No valid values for {col}"
            assert np.all(np.isfinite(vals)), f"Non-finite values in {col}"

    def test_residual_has_values(self):
        n = 500
        btc = _make_price_series(n, 40000, seed=5)
        alt = _make_price_series(n, 100, seed=6)
        result = build_cross_token_features(alt["close"], btc["close"])
        after_warmup = result.iloc[181:]
        valid = after_warmup["btc_alt_residual"].dropna()
        assert len(valid) > 0

    def test_custom_window(self):
        n = 300
        btc = _make_price_series(n, 40000, seed=7)
        alt = _make_price_series(n, 100, seed=8)
        result = build_cross_token_features(
            alt["close"], btc["close"], window=50
        )
        after_warmup = result.iloc[51:]
        valid_pct = after_warmup["btc_alt_beta"].notna().mean()
        assert valid_pct > 0.8


class TestBuildIndicatorFeatures:
    """Tests for build_indicator_features."""

    def test_returns_ten_columns(self):
        n = 500
        btc = _make_price_series(n, 40000, seed=10)
        eth = _make_price_series(n, 2500, seed=11)
        result = build_indicator_features(btc, eth)
        expected = [
            "btc_ret_1", "btc_ret_2", "btc_ret_3",
            "btc_vol_24", "btc_rsi",
            "eth_ret_1", "eth_ret_2", "eth_ret_3",
            "eth_btc_ratio_chg", "btc_above_sma50",
        ]
        assert list(result.columns) == expected

    def test_at_least_80_pct_valid_after_warmup(self):
        n = 500
        warmup = 50  # SMA50 is the longest warmup
        btc = _make_price_series(n, 40000, seed=12)
        eth = _make_price_series(n, 2500, seed=13)
        result = build_indicator_features(btc, eth)
        after_warmup = result.iloc[warmup:]
        for col in result.columns:
            valid_pct = after_warmup[col].notna().mean()
            assert valid_pct >= 0.80, (
                f"Column {col} has only {valid_pct:.1%} valid rows"
            )

    def test_btc_above_sma50_is_binary(self):
        n = 500
        btc = _make_price_series(n, 40000, seed=14)
        eth = _make_price_series(n, 2500, seed=15)
        result = build_indicator_features(btc, eth)
        vals = result["btc_above_sma50"].dropna().unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_rsi_in_range(self):
        n = 500
        btc = _make_price_series(n, 40000, seed=16)
        eth = _make_price_series(n, 2500, seed=17)
        result = build_indicator_features(btc, eth)
        rsi = result["btc_rsi"].dropna()
        assert rsi.min() >= -1.0 - 1e-9
        assert rsi.max() <= 1.0 + 1e-9

    def test_lagged_returns_are_shifted(self):
        n = 500
        btc = _make_price_series(n, 40000, seed=18)
        eth = _make_price_series(n, 2500, seed=19)
        result = build_indicator_features(btc, eth)
        btc_ret = btc["close"].pct_change()
        # btc_ret_1 at index i should equal btc_ret at index i-1
        pd.testing.assert_series_equal(
            result["btc_ret_1"].iloc[2:].reset_index(drop=True),
            btc_ret.iloc[1:-1].reset_index(drop=True),
            check_names=False,
        )
