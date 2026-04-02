"""Tests for Portfolio state tracker."""

import pytest

from src.rl.config import RLConfig
from src.rl.portfolio import Portfolio


@pytest.fixture
def cfg() -> RLConfig:
    return RLConfig()


@pytest.fixture
def portfolio(cfg: RLConfig) -> Portfolio:
    return Portfolio(cfg)


class TestInitialState:
    def test_starting_cash(self, portfolio: Portfolio):
        assert portfolio.cash == 10_000_000

    def test_no_positions(self, portfolio: Portfolio):
        assert portfolio.n_positions == 0

    def test_equity_equals_cash(self, portfolio: Portfolio):
        assert portfolio.equity == 10_000_000


class TestOpenLong:
    def test_open_long_success(self, portfolio: Portfolio):
        ok = portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        assert ok is True
        assert portfolio.n_positions == 1

    def test_long_side(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        pos = portfolio.positions["BTC"]
        assert pos.side == 1

    def test_long_entry_price(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        pos = portfolio.positions["BTC"]
        assert pos.entry_price == 500_000_000

    def test_long_sl_price(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        pos = portfolio.positions["BTC"]
        # sl = entry * (1 - 0.05)
        assert pos.sl_price == pytest.approx(475_000_000)

    def test_long_tp_price(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        pos = portfolio.positions["BTC"]
        # tp = entry * (1 + 0.10)
        assert pos.tp_price == pytest.approx(550_000_000)

    def test_cash_deducted_with_fees(self, portfolio: Portfolio, cfg: RLConfig):
        initial_cash = portfolio.cash
        size_pct = 0.10
        portfolio.open_position("BTC", 1, size_pct, 500_000_000, 0.05, 0.10)
        size_idr = initial_cash * size_pct  # equity == cash at start
        fee_rate = cfg.TAKER_FEE + cfg.SLIPPAGE
        expected_cash = initial_cash - size_idr - size_idr * fee_rate
        assert portfolio.cash == pytest.approx(expected_cash)


class TestOpenShort:
    def test_short_sl_price(self, portfolio: Portfolio):
        portfolio.open_position("ETH", -1, 0.10, 50_000_000, 0.05, 0.10)
        pos = portfolio.positions["ETH"]
        # short sl = entry * (1 + 0.05)
        assert pos.sl_price == pytest.approx(52_500_000)

    def test_short_tp_price(self, portfolio: Portfolio):
        portfolio.open_position("ETH", -1, 0.10, 50_000_000, 0.05, 0.10)
        pos = portfolio.positions["ETH"]
        # short tp = entry * (1 - 0.10)
        assert pos.tp_price == pytest.approx(45_000_000)


class TestConstraints:
    def test_reject_max_positions(self, portfolio: Portfolio, cfg: RLConfig):
        # Fill up to MAX_POSITIONS
        for i in range(cfg.MAX_POSITIONS):
            ok = portfolio.open_position(
                f"COIN{i}", 1, 0.05, 100_000, 0.05, 0.10
            )
            assert ok is True
        # Next should fail
        ok = portfolio.open_position("EXTRA", 1, 0.05, 100_000, 0.05, 0.10)
        assert ok is False

    def test_reject_duplicate_asset(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        ok = portfolio.open_position("BTC", -1, 0.05, 500_000_000, 0.05, 0.10)
        assert ok is False

    def test_reject_exposure_over_80pct(self, portfolio: Portfolio):
        # Open positions until we approach 80%
        # Each at 10% => 8 positions = 80%, but exposure check is >
        # After 7 positions at 10% each, equity changes due to fees
        # so let's open big chunks
        for i in range(8):
            portfolio.open_position(
                f"COIN{i}", 1, 0.10, 100_000, 0.05, 0.10
            )
        # With 8 positions at ~10% each, exposure is ~80%
        # The 9th should be rejected due to > 80%
        ok = portfolio.open_position("COINX", 1, 0.10, 100_000, 0.05, 0.10)
        assert ok is False


class TestClosePosition:
    def test_close_long_profit(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        pnl = portfolio.close_position("BTC", 520_000_000)
        # Positive PnL for price going up on a long
        assert pnl > 0
        assert portfolio.n_positions == 0

    def test_close_short_profit(self, portfolio: Portfolio):
        portfolio.open_position("ETH", -1, 0.10, 50_000_000, 0.05, 0.10)
        pnl = portfolio.close_position("ETH", 48_000_000)
        # Positive PnL for price going down on a short
        assert pnl > 0
        assert portfolio.n_positions == 0


class TestSLTP:
    def test_sl_hit_long(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        sl_price = portfolio.positions["BTC"].sl_price
        # low <= sl_price triggers SL
        fills = portfolio.check_sl_tp(
            {"BTC": {"high": 510_000_000, "low": sl_price}}
        )
        assert "BTC" in fills
        assert fills["BTC"]["reason"] == "sl"
        assert fills["BTC"]["pnl"] < 0

    def test_tp_hit_long(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        tp_price = portfolio.positions["BTC"].tp_price
        fills = portfolio.check_sl_tp(
            {"BTC": {"high": tp_price, "low": 499_000_000}}
        )
        assert "BTC" in fills
        assert fills["BTC"]["reason"] == "tp"
        assert fills["BTC"]["pnl"] > 0

    def test_sl_hit_short(self, portfolio: Portfolio):
        portfolio.open_position("ETH", -1, 0.10, 50_000_000, 0.05, 0.10)
        sl_price = portfolio.positions["ETH"].sl_price
        # short SL: high >= sl_price
        fills = portfolio.check_sl_tp(
            {"ETH": {"high": sl_price, "low": 49_000_000}}
        )
        assert "ETH" in fills
        assert fills["ETH"]["reason"] == "sl"
        assert fills["ETH"]["pnl"] < 0

    def test_tp_hit_short(self, portfolio: Portfolio):
        portfolio.open_position("ETH", -1, 0.10, 50_000_000, 0.05, 0.10)
        tp_price = portfolio.positions["ETH"].tp_price
        # short TP: low <= tp_price
        fills = portfolio.check_sl_tp(
            {"ETH": {"high": 50_000_000, "low": tp_price}}
        )
        assert "ETH" in fills
        assert fills["ETH"]["reason"] == "tp"
        assert fills["ETH"]["pnl"] > 0

    def test_no_trigger(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        fills = portfolio.check_sl_tp(
            {"BTC": {"high": 510_000_000, "low": 490_000_000}}
        )
        assert len(fills) == 0


class TestFunding:
    def test_funding_deduction(self, portfolio: Portfolio, cfg: RLConfig):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        equity_before = portfolio.equity
        portfolio.deduct_funding(n_8h_periods=1)
        equity_after = portfolio.equity
        assert equity_after < equity_before

    def test_funding_amount(self, portfolio: Portfolio, cfg: RLConfig):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        cash_before = portfolio.cash
        pos_size = portfolio.positions["BTC"].size_idr
        portfolio.deduct_funding(n_8h_periods=2)
        expected = cash_before - pos_size * cfg.FUNDING_RATE_8H * 2
        assert portfolio.cash == pytest.approx(expected)


class TestStateVector:
    def test_length(self, portfolio: Portfolio):
        vec = portfolio.get_state_vector({})
        assert len(vec) == 5

    def test_cash_ratio_initial(self, portfolio: Portfolio):
        vec = portfolio.get_state_vector({})
        assert 0 <= vec[0] <= 1
        assert vec[0] == pytest.approx(1.0)

    def test_cash_ratio_with_position(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        vec = portfolio.get_state_vector({"BTC": 500_000_000})
        assert 0 <= vec[0] <= 1
        assert vec[0] < 1.0


class TestIncrementHoldTimes:
    def test_increment(self, portfolio: Portfolio):
        portfolio.open_position("BTC", 1, 0.10, 500_000_000, 0.05, 0.10)
        assert portfolio.positions["BTC"].windows_held == 0
        portfolio.increment_hold_times()
        assert portfolio.positions["BTC"].windows_held == 1
        portfolio.increment_hold_times()
        assert portfolio.positions["BTC"].windows_held == 2
