# tests/test_build_v4_2.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest


def test_resample_15m_to_1h():
    from scripts.build_v4_2_features import resample_to_1h

    n = 96 * 3  # 3 days of 15min data
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    rng = np.random.RandomState(42)
    close = 100.0 + np.cumsum(rng.normal(0, 0.1, n))
    df = pd.DataFrame({
        "timestamp": ts,
        "open": close + rng.normal(0, 0.05, n),
        "high": close + abs(rng.normal(0, 0.5, n)),
        "low": close - abs(rng.normal(0, 0.5, n)),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
        "taker_buy_base": rng.uniform(50, 500, n),
    })

    result = resample_to_1h(df)

    # 3 days * 24 hours = 72 rows
    assert len(result) == 72
    assert "timestamp" in result.columns
    assert "open" in result.columns
    assert "close" in result.columns
    assert "volume" in result.columns
    assert "taker_buy_base" in result.columns

    # Volume should be summed (4x per hour)
    first_hour_vol = df["volume"].iloc[:4].sum()
    assert abs(result["volume"].iloc[0] - first_hour_vol) < 0.01

    # High should be max of 4 bars
    first_hour_high = df["high"].iloc[:4].max()
    assert abs(result["high"].iloc[0] - first_hour_high) < 0.01


def test_compute_1h_features():
    from scripts.build_v4_2_features import compute_1h_features, FEATURE_COLS_1H

    n = 200  # 200 hourly bars (~8 days)
    ts = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
    rng = np.random.RandomState(42)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))

    df = pd.DataFrame({
        "timestamp": ts,
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + rng.uniform(0, 0.01, n)),
        "low": close * (1 - rng.uniform(0, 0.01, n)),
        "close": close,
        "volume": rng.uniform(100, 1000, n),
        "taker_buy_base": rng.uniform(50, 500, n),
    })

    result = compute_1h_features(df)

    # Should have all feature columns
    for col in FEATURE_COLS_1H:
        assert col in result.columns, f"Missing {col}"

    # ret_1 should be 1-bar log return (= 1h)
    expected = np.log(df["close"].iloc[100] / df["close"].iloc[99])
    assert abs(result["ret_1"].iloc[100] - expected) < 1e-10
