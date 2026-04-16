import numpy as np
import pandas as pd
import pytest

from src.ml.features import FeatureEngineer


@pytest.fixture
def sample_daily_prices() -> pd.DataFrame:
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    base_price = 50000.0
    returns = np.random.normal(0.001, 0.02, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_prices = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.uniform(1000, 10000, n)
    return pd.DataFrame({
        "date": dates, "asset": "BTC",
        "open": open_prices, "high": high, "low": low,
        "close": close, "volume": volume,
    })


@pytest.fixture
def sample_fundamentals() -> pd.DataFrame:
    return pd.DataFrame({
        "asset": ["BTC", "ETH", "BNB"],
        "market_cap_rank": [1, 2, 3],
        "total_volume_24h": [35e9, 15e9, 5e9],
        "market_cap": [1.4e12, 4e11, 9e10],
    })


@pytest.fixture
def sample_market_context() -> pd.DataFrame:
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "btc_dominance": np.random.uniform(40, 60, n),
        "fear_greed_index": np.random.randint(10, 90, n),
        "total_market_cap": np.random.uniform(2e12, 3e12, n),
    })


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer()


class TestTechnicalFeatures:
    def test_sma_computed(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        expected_sma5 = sample_daily_prices["close"].iloc[:5].mean()
        assert abs(result["sma_5"].iloc[4] - expected_sma5) < 1e-6

    def test_ema_computed(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "ema_5" in result.columns
        assert "ema_50" in result.columns
        assert not np.isnan(result["ema_5"].iloc[10])

    def test_rsi_computed(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "rsi_14" in result.columns
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_computed(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_bollinger_bands(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "bb_upper" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        valid_idx = result[["bb_upper", "bb_lower"]].dropna().index
        assert (
            result.loc[valid_idx, "bb_upper"]
            >= result.loc[valid_idx, "bb_lower"]
        ).all()

    def test_atr_computed(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "atr_14" in result.columns
        valid = result["atr_14"].dropna()
        assert (valid > 0).all()

    def test_momentum_features(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "momentum_7d" in result.columns
        assert "momentum_14d" in result.columns
        assert "momentum_28d" in result.columns

    def test_volume_features(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "volume_sma_20" in result.columns
        assert "volume_ratio" in result.columns

    def test_volatility_std(self, engineer, sample_daily_prices):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "volatility_20d" in result.columns


class TestFundamentalFeatures:
    def test_fundamental_features(self, engineer, sample_fundamentals):
        result = engineer.compute_fundamental_features(
            sample_fundamentals, "BTC"
        )
        assert "market_cap_rank" in result
        assert "volume_rank" in result
        assert result["market_cap_rank"] == 1

    def test_fundamental_missing_asset(self, engineer, sample_fundamentals):
        result = engineer.compute_fundamental_features(
            sample_fundamentals, "UNKNOWN"
        )
        assert result["market_cap_rank"] == 0
        assert result["volume_rank"] == 0


class TestMarketContextFeatures:
    def test_market_context(self, engineer, sample_market_context):
        result = engineer.compute_market_context_features(
            sample_market_context
        )
        assert "btc_dominance" in result.columns
        assert "fear_greed_index" in result.columns
        assert "total_market_cap_change_7d" in result.columns


class TestFullFeaturePipeline:
    def test_build_feature_matrix(
        self, engineer, sample_daily_prices,
        sample_fundamentals, sample_market_context,
    ):
        result = engineer.build_feature_matrix(
            daily_prices=sample_daily_prices,
            fundamentals=sample_fundamentals,
            market_context=sample_market_context,
            asset="BTC",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        assert not result.isnull().any().any()
        assert "sma_5" in result.columns
        assert "rsi_14" in result.columns
        assert "market_cap_rank" in result.columns
        assert "btc_dominance" in result.columns

    def test_build_targets(self, engineer, sample_daily_prices):
        targets = engineer.build_targets(sample_daily_prices)
        assert "target_7d" in targets.columns
        assert "target_14d" in targets.columns
        assert "target_28d" in targets.columns
        assert "target_42d" in targets.columns

    def test_feature_names_list(self, engineer):
        names = engineer.feature_names()
        assert isinstance(names, list)
        assert len(names) > 20
