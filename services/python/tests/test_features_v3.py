import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 15m OHLCV data."""
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.uniform(100, 1000, n)
    taker_buy_base = volume * rng.uniform(0.3, 0.7, n)

    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "taker_buy_base": taker_buy_base,
    })


def test_momentum_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    for col in ["ret_4", "ret_16", "ret_96", "ret_672", "ret_4_lag1"]:
        assert col in result.columns, f"Missing {col}"
        assert result[col].notna().sum() > 0

    # ret_4 should be log return over 4 bars
    expected = np.log(df["close"].iloc[700] / df["close"].iloc[696])
    assert abs(result["ret_4"].iloc[700] - expected) < 1e-10


def test_trend_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    for col in ["price_to_sma_96", "price_to_sma_672", "ema_ratio", "macd_hist_norm"]:
        assert col in result.columns, f"Missing {col}"

    # price_to_sma_96 should be close/SMA(96) - 1
    sma = df["close"].rolling(96).mean()
    expected = df["close"].iloc[700] / sma.iloc[700] - 1
    assert abs(result["price_to_sma_96"].iloc[700] - expected) < 1e-10


def test_oscillator_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "rsi_norm" in result.columns
    assert "bb_pct" in result.columns
    # RSI should be in [-1, 1]
    valid = result["rsi_norm"].dropna()
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01


def test_volatility_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    for col in ["vol_4h", "vol_1d", "vol_ratio", "parkinson_vol"]:
        assert col in result.columns, f"Missing {col}"
    # vol_ratio = vol_4h / vol_1d
    idx = 750
    expected = result["vol_4h"].iloc[idx] / result["vol_1d"].iloc[idx]
    assert abs(result["vol_ratio"].iloc[idx] - expected) < 1e-10


def test_volume_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "volume_ratio" in result.columns
    assert "taker_buy_ratio" in result.columns
    # taker_buy_ratio should be between 0 and 1
    valid = result["taker_buy_ratio"].dropna()
    assert valid.min() >= 0
    assert valid.max() <= 1


def test_microstructure_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "clv" in result.columns
    assert "drawdown" in result.columns
    # CLV should be in [-1, 1]
    valid = result["clv"].dropna()
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01
    # drawdown should be <= 0
    assert result["drawdown"].dropna().max() <= 0.001


def test_time_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    # sin^2 + cos^2 should be ~1
    check = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
    assert abs(check.iloc[100] - 1.0) < 1e-10


def test_feature_count():
    from src.ml.features_v3 import compute_ohlcv_features, OHLCV_FEATURE_COLS

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    # 19 OHLCV features + 2 time features = 21
    assert len(OHLCV_FEATURE_COLS) == 21
    for col in OHLCV_FEATURE_COLS:
        assert col in result.columns, f"Missing {col}"
