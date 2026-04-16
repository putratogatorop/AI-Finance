from datetime import UTC, datetime

import pytest

from src.ml.risk import OpenPosition, PortfolioState, RiskManager
from src.ml.signals import TradeSignal


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(
        max_positions=8,
        max_single_asset_pct=25.0,
        portfolio_drawdown_pause_pct=20.0,
        stop_loss_pct=-8.0,
        position_size_idr=1_000_000,
    )


def _make_signal(
    asset: str = "ETH",
    action: str = "BUY",
    confidence: float = 0.84,
) -> TradeSignal:
    return TradeSignal(
        asset=asset,
        action=action,
        confidence=confidence,
        suggested_hold_days=14,
        stop_loss_pct=-8.0,
        expected_return_pct=8.5,
        model_agreement="3/3",
        timestamp=datetime(2026, 3, 30, tzinfo=UTC),
    )


def _make_portfolio_state(
    positions: list[OpenPosition] | None = None,
    total_capital_idr: int = 10_000_000,
    peak_value_idr: int = 10_000_000,
    current_value_idr: int = 10_000_000,
) -> PortfolioState:
    return PortfolioState(
        positions=positions or [],
        total_capital_idr=total_capital_idr,
        peak_value_idr=peak_value_idr,
        current_value_idr=current_value_idr,
    )


class TestRiskManager:
    def test_allow_valid_signal(self, risk_manager: RiskManager):
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state()
        result = risk_manager.check(signal, state)
        assert result.allowed is True
        assert len(result.violations) == 0

    def test_reject_max_positions_exceeded(self, risk_manager: RiskManager):
        positions = [
            OpenPosition(
                asset=f"ASSET_{i}",
                entry_amount_idr=1_000_000,
                current_value_idr=1_000_000,
            )
            for i in range(8)
        ]
        signal = _make_signal(asset="NEW_ASSET", action="BUY")
        state = _make_portfolio_state(positions=positions)
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert any("max positions" in v.lower() for v in result.violations)

    def test_reject_single_asset_exposure(self, risk_manager: RiskManager):
        """25% of 10M = 2.5M. Already 2M in ETH, adding 1M would be 3M = 30%."""
        positions = [
            OpenPosition(
                asset="ETH",
                entry_amount_idr=2_000_000,
                current_value_idr=2_000_000,
            ),
        ]
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state(
            positions=positions,
            total_capital_idr=10_000_000,
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert any("single asset" in v.lower() for v in result.violations)

    def test_allow_single_asset_within_limit(self, risk_manager: RiskManager):
        """25% of 10M = 2.5M. 1M in ETH + 1M new = 2M = 20%, OK."""
        positions = [
            OpenPosition(
                asset="ETH",
                entry_amount_idr=1_000_000,
                current_value_idr=1_000_000,
            ),
        ]
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state(
            positions=positions,
            total_capital_idr=10_000_000,
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is True

    def test_reject_portfolio_drawdown_pause(self, risk_manager: RiskManager):
        """20% drawdown from peak => pause new signals."""
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state(
            peak_value_idr=10_000_000,
            current_value_idr=7_900_000,  # -21% drawdown
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert any("drawdown" in v.lower() for v in result.violations)

    def test_allow_sell_during_drawdown(self, risk_manager: RiskManager):
        """SELL/EXIT signals should still be allowed during drawdown pause."""
        signal = _make_signal(asset="ETH", action="SELL")
        state = _make_portfolio_state(
            peak_value_idr=10_000_000,
            current_value_idr=7_900_000,  # -21% drawdown
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is True

    def test_multiple_violations(self, risk_manager: RiskManager):
        positions = [
            OpenPosition(
                asset=f"ASSET_{i}",
                entry_amount_idr=1_000_000,
                current_value_idr=1_000_000,
            )
            for i in range(8)
        ]
        signal = _make_signal(asset="ASSET_0", action="BUY")
        state = _make_portfolio_state(
            positions=positions,
            total_capital_idr=10_000_000,
            peak_value_idr=10_000_000,
            current_value_idr=7_500_000,  # -25% drawdown
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert len(result.violations) >= 2  # max positions + drawdown

    def test_allow_exit_signals_always(self, risk_manager: RiskManager):
        """EXIT signals bypass all risk checks."""
        positions = [
            OpenPosition(
                asset=f"ASSET_{i}",
                entry_amount_idr=1_000_000,
                current_value_idr=1_000_000,
            )
            for i in range(8)
        ]
        signal = _make_signal(asset="ASSET_0", action="EXIT")
        state = _make_portfolio_state(
            positions=positions,
            peak_value_idr=10_000_000,
            current_value_idr=7_000_000,
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is True

    def test_compute_stop_loss_price(self, risk_manager: RiskManager):
        price = risk_manager.compute_stop_loss_price(entry_price=100.0)
        assert price == 92.0  # 100 * (1 + (-8/100)) = 92

    def test_alert_level_green(self, risk_manager: RiskManager):
        level = risk_manager.get_alert_level(
            entry_price=100.0, current_price=105.0, confidence=0.80
        )
        assert level == "GREEN"

    def test_alert_level_yellow(self, risk_manager: RiskManager):
        level = risk_manager.get_alert_level(
            entry_price=100.0, current_price=97.0, confidence=0.55
        )
        assert level == "YELLOW"

    def test_alert_level_red(self, risk_manager: RiskManager):
        level = risk_manager.get_alert_level(
            entry_price=100.0, current_price=91.0, confidence=0.80
        )
        assert level == "RED"
