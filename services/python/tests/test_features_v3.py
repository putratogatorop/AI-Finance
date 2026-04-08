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


def make_funding(n: int = 100) -> pd.DataFrame:
    """Generate synthetic 8-hourly funding rate data."""
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="8h")
    rates = rng.normal(0.0001, 0.0005, n)
    return pd.DataFrame({"timestamp": ts, "funding_rate": rates})


def test_merge_funding_features():
    from src.ml.features_v3 import merge_funding_features

    ohlcv = make_ohlcv(1000)
    funding = make_funding(200)
    result = merge_funding_features(ohlcv, funding)

    for col in ["funding_rate", "funding_ma_3d", "funding_zscore", "cum_funding_3d"]:
        assert col in result.columns, f"Missing {col}"

    # funding_rate should be forward-filled (not all NaN)
    assert result["funding_rate"].notna().sum() > 500


def test_funding_zscore_range():
    from src.ml.features_v3 import merge_funding_features

    ohlcv = make_ohlcv(1000)
    funding = make_funding(200)
    result = merge_funding_features(ohlcv, funding)

    valid = result["funding_zscore"].dropna()
    # Z-score should be roughly centered around 0
    assert abs(valid.mean()) < 2.0


def test_cross_asset_features():
    from src.ml.features_v3 import compute_cross_asset_features

    n = 1000
    rng = np.random.RandomState(42)

    # BTC close
    btc_close = pd.Series(50000 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))
    # Alt close (correlated with BTC)
    alt_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))))
    # All alt returns for dispersion (5 coins)
    all_alt_ret_96 = pd.DataFrame({
        f"alt_{i}": pd.Series(np.exp(np.cumsum(rng.normal(0, 0.002, n)))).pct_change(96)
        for i in range(5)
    })

    result = compute_cross_asset_features(alt_close, btc_close, all_alt_ret_96)

    assert "btc_ret_96" in result.columns
    assert "btc_residual" in result.columns
    assert "altcoin_dispersion" in result.columns
    assert len(result) == n


def test_interaction_feature():
    from src.ml.features_v3 import add_interaction_features

    df = pd.DataFrame({
        "funding_zscore": [0.5, -1.0, 2.0, 0.0],
        "rsi_norm": [0.3, -0.5, 0.8, 0.0],
    })
    result = add_interaction_features(df)
    assert "funding_x_rsi" in result.columns
    assert result["funding_x_rsi"].iloc[0] == pytest.approx(0.15)
    assert result["funding_x_rsi"].iloc[2] == pytest.approx(1.6)
