import numpy as np
import pandas as pd
import pytest

from src.ml.backtest import BacktestConfig, BacktestEngine, BacktestResult


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(
        position_size_idr=1_000_000,
        stop_loss_pct=-8.0,
        max_positions=8,
        max_single_asset_pct=25.0,
        confidence_threshold=0.7,
    )


@pytest.fixture
def price_data() -> pd.DataFrame:
    """Generate 2 years of synthetic daily price data for multiple assets."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    assets = ["BTC", "ETH", "SOL"]
    rows = []
    for asset in assets:
        base = {"BTC": 30000, "ETH": 2000, "SOL": 25}[asset]
        prices = base * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "asset": asset,
                "open": prices[i] * 0.999,
                "high": prices[i] * 1.01,
                "low": prices[i] * 0.99,
                "close": prices[i],
                "volume": np.random.uniform(1000, 10000),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def signals_data() -> pd.DataFrame:
    """Generate synthetic signals over the price data period."""
    np.random.seed(123)
    rows = []
    dates = pd.date_range("2023-03-01", "2024-12-01", freq="14D")
    assets = ["BTC", "ETH", "SOL"]
    for date in dates:
        asset = np.random.choice(assets)
        action = np.random.choice(["BUY", "SELL"], p=[0.7, 0.3])
        rows.append({
            "date": date,
            "asset": asset,
            "action": action,
            "confidence": np.random.uniform(0.65, 0.95),
            "suggested_hold_days": np.random.choice([7, 14, 28, 42]),
            "stop_loss_pct": -8.0,
            "expected_return_pct": np.random.uniform(-5, 15),
            "model_agreement": np.random.choice(["2/3", "3/3"]),
        })
    return pd.DataFrame(rows)


class TestBacktestConfig:
    def test_config_defaults(self):
        config = BacktestConfig()
        assert config.position_size_idr == 1_000_000
        assert config.stop_loss_pct == -8.0
        assert config.max_positions == 8

    def test_config_custom(self, config: BacktestConfig):
        assert config.max_single_asset_pct == 25.0


class TestBacktestEngine:
    def test_run_returns_result(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        assert isinstance(result, BacktestResult)

    def test_result_has_all_metrics(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)

        assert hasattr(result, "total_return_idr")
        assert hasattr(result, "total_return_pct")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "reward_risk_ratio")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown_pct")
        assert hasattr(result, "total_trades")
        assert hasattr(result, "trades")
        assert hasattr(result, "monthly_breakdown")
        assert hasattr(result, "per_asset_breakdown")

    def test_trades_have_no_lookahead(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        for trade in result.trades:
            assert trade.entry_date >= trade.signal_date

    def test_stop_loss_respected(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        slippage_tolerance = 2.0
        for trade in result.trades:
            if trade.pnl_pct < 0:
                assert trade.pnl_pct >= config.stop_loss_pct - slippage_tolerance

    def test_max_positions_respected(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        assert result.max_concurrent_positions <= config.max_positions

    def test_position_size_fixed(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        for trade in result.trades:
            assert trade.entry_amount_idr == config.position_size_idr

    def test_win_rate_bounded(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_monthly_breakdown_format(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        for entry in result.monthly_breakdown:
            assert "month" in entry
            assert "pnl_idr" in entry
            assert "trades" in entry

    def test_per_asset_breakdown_format(
        self, config: BacktestConfig, price_data: pd.DataFrame, signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        for entry in result.per_asset_breakdown:
            assert "asset" in entry
            assert "total_pnl_idr" in entry
            assert "win_rate" in entry
            assert "trades" in entry

    def test_empty_signals(self, config: BacktestConfig, price_data: pd.DataFrame):
        engine = BacktestEngine(config)
        empty_signals = pd.DataFrame(columns=[
            "date", "asset", "action", "confidence", "suggested_hold_days",
            "stop_loss_pct", "expected_return_pct", "model_agreement",
        ])
        result = engine.run(price_data, empty_signals)
        assert result.total_trades == 0
        assert result.total_return_idr == 0
