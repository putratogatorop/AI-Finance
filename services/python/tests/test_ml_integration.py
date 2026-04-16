"""Integration test: full ML pipeline from features through signal generation.

This test exercises the complete flow:
1. Feature engineering on synthetic data
2. Target computation
3. Train all three models (XGBoost, LightGBM, LSTM)
4. Ensemble consensus
5. Signal generation
6. Risk management check
7. Backtest validation
"""

import tempfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest import BacktestConfig, BacktestEngine
from src.ml.ensemble import EnsembleConsensus
from src.ml.features import FeatureEngineer
from src.ml.models.lightgbm_model import LightGBMModel
from src.ml.models.lstm_model import LSTMModel
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.risk import PortfolioState, RiskManager
from src.ml.signals import SignalGenerator
from src.ml.versioning import ModelVersionManager


@pytest.fixture
def synthetic_data() -> dict:
    """Generate synthetic market data for integration testing."""
    np.random.seed(42)
    n_days = 400  # ~1.1 years

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    base_price = 50000.0
    returns = np.random.normal(0.001, 0.02, n_days)
    close = base_price * np.cumprod(1 + returns)

    daily_prices = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "open": close * (1 + np.random.normal(0, 0.005, n_days)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_days),
    })

    fundamentals = pd.DataFrame({
        "asset": ["BTC", "ETH", "SOL"],
        "market_cap_rank": [1, 2, 5],
        "total_volume_24h": [35e9, 15e9, 3e9],
        "market_cap": [1.4e12, 4e11, 3e10],
    })

    market_context = pd.DataFrame({
        "date": dates,
        "btc_dominance": np.random.uniform(45, 55, n_days),
        "fear_greed_index": np.random.randint(20, 80, n_days),
        "total_market_cap": np.random.uniform(2e12, 3e12, n_days),
    })

    return {
        "daily_prices": daily_prices,
        "fundamentals": fundamentals,
        "market_context": market_context,
    }


class TestFullMLPipeline:
    def test_end_to_end_pipeline(self, synthetic_data: dict):
        """Run the complete ML pipeline end-to-end."""
        daily_prices = synthetic_data["daily_prices"]
        fundamentals = synthetic_data["fundamentals"]
        market_context = synthetic_data["market_context"]

        # 1. Feature Engineering
        engineer = FeatureEngineer()
        features = engineer.build_feature_matrix(
            daily_prices=daily_prices,
            fundamentals=fundamentals,
            market_context=market_context,
            asset="BTC",
        )
        assert len(features) > 0
        assert not features.isnull().any().any()

        # 2. Build Targets
        targets = engineer.build_targets(daily_prices)
        # Align targets with features (drop warmup rows and future NaN rows)
        n_warmup = len(daily_prices) - len(features)
        targets_aligned = targets.iloc[n_warmup : n_warmup + len(features)].reset_index(drop=True)

        # Drop rows where targets are NaN (last 42 rows)
        target_cols = ["target_7d", "target_14d", "target_28d", "target_42d"]
        valid_mask = ~targets_aligned[target_cols].isnull().any(axis=1)
        X = features[valid_mask].reset_index(drop=True)
        y = targets_aligned[valid_mask][target_cols].reset_index(drop=True)

        assert len(X) > 100, f"Not enough valid samples: {len(X)}"

        # 3. Train/Val Split (80/20, no shuffle for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 4. Train Models
        xgb_model = XGBoostModel(n_estimators=20, max_depth=3)
        xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val)
        assert "train_rmse_7d" in xgb_metrics

        lgb_model = LightGBMModel(n_estimators=20, max_depth=3)
        lgb_metrics = lgb_model.train(X_train, y_train, X_val, y_val)
        assert "train_rmse_7d" in lgb_metrics

        lstm_model = LSTMModel(
            sequence_length=20, hidden_size=16, num_layers=1,
            dropout=0.0, epochs=3, batch_size=32,
        )
        lstm_metrics = lstm_model.train(X_train, y_train, X_val, y_val)
        assert "train_loss" in lstm_metrics

        # 5. Generate Predictions
        xgb_pred = xgb_model.predict(X_val)
        lgb_pred = lgb_model.predict(X_val)
        lstm_pred = lstm_model.predict(X_val)

        assert xgb_pred.model_name == "xgboost"
        assert lgb_pred.model_name == "lightgbm"
        assert lstm_pred.model_name == "lstm"

        # 6. Ensemble Consensus
        ensemble = EnsembleConsensus(
            model_weights={"xgboost": 0.4, "lightgbm": 0.4, "lstm": 0.2},
            confidence_threshold=0.3,  # Low threshold for synthetic data
        )
        result = ensemble.combine([xgb_pred, lgb_pred, lstm_pred])
        # Result might be None if confidence is too low, that's OK

        # 7. Signal Generation
        generator = SignalGenerator(stop_loss_pct=-8.0)
        signal = generator.generate(
            asset="BTC",
            ensemble_result=result,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )

        # 8. Risk Management
        risk_mgr = RiskManager(
            max_positions=8,
            max_single_asset_pct=25.0,
            portfolio_drawdown_pause_pct=20.0,
        )
        if signal is not None:
            state = PortfolioState(
                positions=[],
                total_capital_idr=10_000_000,
                peak_value_idr=10_000_000,
                current_value_idr=10_000_000,
            )
            risk_result = risk_mgr.check(signal, state)
            assert isinstance(risk_result.allowed, bool)

    def test_model_versioning_roundtrip(self, synthetic_data: dict):
        """Test save and reload of all models."""
        daily_prices = synthetic_data["daily_prices"]
        fundamentals = synthetic_data["fundamentals"]
        market_context = synthetic_data["market_context"]

        engineer = FeatureEngineer()
        features = engineer.build_feature_matrix(
            daily_prices=daily_prices,
            fundamentals=fundamentals,
            market_context=market_context,
            asset="BTC",
        )
        targets = engineer.build_targets(daily_prices)
        n_warmup = len(daily_prices) - len(features)
        targets_aligned = targets.iloc[n_warmup : n_warmup + len(features)].reset_index(drop=True)

        target_cols = ["target_7d", "target_14d", "target_28d", "target_42d"]
        valid_mask = ~targets_aligned[target_cols].isnull().any(axis=1)
        X = features[valid_mask].reset_index(drop=True)
        y = targets_aligned[valid_mask][target_cols].reset_index(drop=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(base_dir=tmpdir)

            # Train and save XGBoost
            xgb = XGBoostModel(n_estimators=10, max_depth=2)
            xgb.train(X, y)
            pred_before = xgb.predict(X.iloc[[-1]])

            manager.save_version(
                model=xgb, metrics={"rmse": 0.03},
                timestamp=datetime(2026, 3, 30, tzinfo=UTC),
            )

            # Load into new instance
            xgb2 = XGBoostModel()
            manager.load_latest(xgb2, "xgboost")
            pred_after = xgb2.predict(X.iloc[[-1]])

            assert abs(
                pred_before.horizons[0].predicted_return
                - pred_after.horizons[0].predicted_return
            ) < 1e-6

    def test_backtest_with_generated_signals(self, synthetic_data: dict):
        """Run a backtest using the pipeline's generated signals."""
        daily_prices = synthetic_data["daily_prices"]

        # Create simple signals for backtesting
        np.random.seed(99)
        signal_dates = pd.date_range("2024-04-01", "2025-02-01", freq="14D")
        signals = pd.DataFrame({
            "date": signal_dates,
            "asset": "BTC",
            "action": "BUY",
            "confidence": np.random.uniform(0.7, 0.9, len(signal_dates)),
            "suggested_hold_days": 14,
            "stop_loss_pct": -8.0,
            "expected_return_pct": np.random.uniform(2, 10, len(signal_dates)),
            "model_agreement": "3/3",
        })

        config = BacktestConfig(
            position_size_idr=1_000_000,
            stop_loss_pct=-8.0,
            max_positions=8,
            confidence_threshold=0.7,
        )
        engine = BacktestEngine(config)
        result = engine.run(daily_prices, signals)

        assert result.total_trades > 0
        assert 0.0 <= result.win_rate <= 1.0
        assert len(result.monthly_breakdown) > 0
        assert len(result.per_asset_breakdown) > 0

        # Verify no lookahead
        for trade in result.trades:
            assert trade.entry_date >= trade.signal_date
