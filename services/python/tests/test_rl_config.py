"""Tests for RL config constants."""

import pytest

from src.rl.config import RLConfig


@pytest.fixture
def cfg() -> RLConfig:
    return RLConfig()


class TestRLConfigDefaults:
    """Verify all default values are correct."""

    def test_position_constraints(self, cfg: RLConfig) -> None:
        assert cfg.MAX_POSITIONS == 10
        assert cfg.MAX_SINGLE_POSITION_PCT == 0.10
        assert cfg.MAX_EXPOSURE_PCT == 0.80

    def test_sl_tp_ranges(self, cfg: RLConfig) -> None:
        assert cfg.SL_MIN == 0.02
        assert cfg.SL_MAX == 0.08
        assert cfg.TP_MIN == 0.03
        assert cfg.TP_MAX == 0.15

    def test_costs(self, cfg: RLConfig) -> None:
        assert cfg.TAKER_FEE == 0.001
        assert cfg.SLIPPAGE == 0.0005
        assert cfg.FUNDING_RATE_8H == 0.0001

    def test_capital(self, cfg: RLConfig) -> None:
        assert cfg.STARTING_CAPITAL == 10_000_000

    def test_episode(self, cfg: RLConfig) -> None:
        assert cfg.EPISODE_WINDOWS == 365
        assert cfg.MAX_DRAWDOWN_KILL == 0.30
        assert cfg.WINDOW_HOURS == 12
        assert cfg.CANDLES_PER_WINDOW == 3

    def test_reward_weights(self, cfg: RLConfig) -> None:
        assert cfg.REWARD_SHARPE_W == 0.6
        assert cfg.REWARD_RETURN_W == 0.3
        assert cfg.REWARD_DD_PENALTY_W == 0.1
        assert cfg.REWARD_DD_THRESHOLD == 0.20

    def test_features(self, cfg: RLConfig) -> None:
        assert cfg.N_ALT_FEATURES == 28
        assert cfg.N_INDICATOR_FEATURES == 9
        assert cfg.N_PORTFOLIO_FEATURES == 5
        assert cfg.LSTM_LOOKBACK == 10

    def test_indicator_assets(self, cfg: RLConfig) -> None:
        assert cfg.INDICATOR_ASSETS == ("BTC", "ETH")
        assert isinstance(cfg.INDICATOR_ASSETS, tuple)


class TestRLConfigConstraints:
    """Verify derived constraints hold."""

    def test_reward_weights_sum_to_one(self, cfg: RLConfig) -> None:
        total = cfg.REWARD_SHARPE_W + cfg.REWARD_RETURN_W + cfg.REWARD_DD_PENALTY_W
        assert abs(total - 1.0) < 1e-9

    def test_feature_counts_match(self, cfg: RLConfig) -> None:
        total = cfg.N_ALT_FEATURES + cfg.N_INDICATOR_FEATURES + cfg.N_PORTFOLIO_FEATURES
        assert total == 42

    def test_sl_range_valid(self, cfg: RLConfig) -> None:
        assert cfg.SL_MIN < cfg.SL_MAX

    def test_tp_range_valid(self, cfg: RLConfig) -> None:
        assert cfg.TP_MIN < cfg.TP_MAX


class TestRLConfigFrozen:
    """Verify the dataclass is frozen."""

    def test_cannot_mutate(self, cfg: RLConfig) -> None:
        with pytest.raises(AttributeError):
            cfg.MAX_POSITIONS = 20  # type: ignore[misc]
