"""Risk management checks for the trading system.

Validates signals against portfolio-level rules:
- Max concurrent positions (8)
- Max single asset exposure (25%)
- Portfolio drawdown pause (20%)
- Stop-loss calculations (-8%)
- Alert level assessment (GREEN/YELLOW/RED)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ml.signals import TradeSignal


@dataclass
class OpenPosition:
    """Represents a currently open position in the portfolio."""

    asset: str
    entry_amount_idr: int
    current_value_idr: int


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk checking."""

    positions: list[OpenPosition] = field(default_factory=list)
    total_capital_idr: int = 10_000_000
    peak_value_idr: int = 10_000_000
    current_value_idr: int = 10_000_000


@dataclass
class RiskCheckResult:
    """Result of risk management validation."""

    allowed: bool
    violations: list[str] = field(default_factory=list)


class RiskManager:
    """Validates trade signals against portfolio risk rules."""

    def __init__(
        self,
        max_positions: int = 8,
        max_single_asset_pct: float = 25.0,
        portfolio_drawdown_pause_pct: float = 20.0,
        stop_loss_pct: float = -8.0,
        position_size_idr: int = 1_000_000,
    ) -> None:
        self._max_positions = max_positions
        self._max_single_asset_pct = max_single_asset_pct
        self._portfolio_drawdown_pause_pct = portfolio_drawdown_pause_pct
        self._stop_loss_pct = stop_loss_pct
        self._position_size_idr = position_size_idr

    def check(self, signal: TradeSignal, state: PortfolioState) -> RiskCheckResult:
        """Validate a trade signal against all risk rules."""
        # EXIT and SELL signals always pass
        if signal.action in ("EXIT", "SELL", "HOLD"):
            return RiskCheckResult(allowed=True, violations=[])

        # Only BUY signals need risk checking
        violations: list[str] = []

        # Check 1: Max concurrent positions
        if len(state.positions) >= self._max_positions:
            violations.append(
                f"Max positions ({self._max_positions}) reached. "
                f"Currently have {len(state.positions)} open positions."
            )

        # Check 2: Max single asset exposure
        current_asset_exposure = sum(
            p.current_value_idr
            for p in state.positions
            if p.asset == signal.asset
        )
        new_exposure = current_asset_exposure + self._position_size_idr
        max_allowed = state.total_capital_idr * (self._max_single_asset_pct / 100)
        if new_exposure > max_allowed:
            violations.append(
                f"Single asset exposure for {signal.asset} would be "
                f"{new_exposure:,} IDR "
                f"({new_exposure / state.total_capital_idr * 100:.1f}%), "
                f"exceeding max {self._max_single_asset_pct}% "
                f"= {max_allowed:,.0f} IDR."
            )

        # Check 3: Portfolio drawdown pause
        if state.peak_value_idr > 0:
            drawdown_pct = (
                (state.peak_value_idr - state.current_value_idr)
                / state.peak_value_idr
                * 100
            )
            if drawdown_pct >= self._portfolio_drawdown_pause_pct:
                violations.append(
                    f"Portfolio drawdown is {drawdown_pct:.1f}%, "
                    f"exceeding pause threshold of "
                    f"{self._portfolio_drawdown_pause_pct}%. "
                    f"New entries paused until recovery."
                )

        allowed = len(violations) == 0
        return RiskCheckResult(allowed=allowed, violations=violations)

    def compute_stop_loss_price(self, entry_price: float) -> float:
        """Calculate the stop-loss price for an entry."""
        return entry_price * (1 + self._stop_loss_pct / 100)

    def get_alert_level(
        self,
        entry_price: float,
        current_price: float,
        confidence: float,
    ) -> str:
        """Determine the alert level for an open position."""
        price_change_pct = ((current_price - entry_price) / entry_price) * 100

        # RED: stop-loss hit or within 1% of it
        stop_loss_threshold = self._stop_loss_pct + 1.0
        if price_change_pct <= stop_loss_threshold:
            return "RED"

        # YELLOW: confidence dropping or moderate loss
        if confidence < 0.6 or price_change_pct < -5.0:
            return "YELLOW"

        # GREEN: on track
        return "GREEN"
