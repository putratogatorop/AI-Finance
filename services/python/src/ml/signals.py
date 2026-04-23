"""Signal generation from ensemble predictions.

Converts ensemble results into actionable trade signals with
all required fields for the trading system.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from src.ml.ensemble import EnsembleResult


@dataclass
class TradeSignal:
    """A complete trade signal ready for the portfolio manager."""

    asset: str
    action: str  # BUY, SELL, HOLD, EXIT
    confidence: float
    suggested_hold_days: int
    stop_loss_pct: float
    expected_return_pct: float
    model_agreement: str
    timestamp: datetime

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class SignalGenerator:
    """Generates trade signals from ensemble predictions."""

    def __init__(self, stop_loss_pct: float = -8.0) -> None:
        self._stop_loss_pct = stop_loss_pct

    def generate(
        self,
        asset: str,
        ensemble_result: EnsembleResult | None,
        timestamp: datetime,
    ) -> TradeSignal | None:
        """Generate a trade signal from an ensemble result."""
        if ensemble_result is None:
            return None

        return TradeSignal(
            asset=asset,
            action=ensemble_result.action,
            confidence=ensemble_result.confidence,
            suggested_hold_days=ensemble_result.suggested_hold_days,
            stop_loss_pct=self._stop_loss_pct,
            expected_return_pct=ensemble_result.expected_return_pct,
            model_agreement=ensemble_result.model_agreement,
            timestamp=timestamp,
        )

    def generate_batch(
        self,
        ensemble_results: dict[str, EnsembleResult | None],
        timestamp: datetime,
    ) -> list[TradeSignal]:
        """Generate signals for multiple assets."""
        signals: list[TradeSignal] = []
        for asset, result in ensemble_results.items():
            signal = self.generate(asset, result, timestamp)
            if signal is not None:
                signals.append(signal)
        return signals

    def evaluate_open_position(
        self,
        asset: str,
        entry_price: float,
        current_price: float,
        ensemble_result: EnsembleResult | None,
        timestamp: datetime,
    ) -> TradeSignal:
        """Re-evaluate an open position and decide HOLD or EXIT."""
        # Check stop-loss first (hard rule)
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        if price_change_pct <= self._stop_loss_pct:
            return TradeSignal(
                asset=asset,
                action="EXIT",
                confidence=1.0,
                suggested_hold_days=0,
                stop_loss_pct=self._stop_loss_pct,
                expected_return_pct=price_change_pct,
                model_agreement="N/A",
                timestamp=timestamp,
            )

        # If no ensemble result, default to HOLD
        if ensemble_result is None:
            return TradeSignal(
                asset=asset,
                action="HOLD",
                confidence=0.5,
                suggested_hold_days=7,
                stop_loss_pct=self._stop_loss_pct,
                expected_return_pct=0.0,
                model_agreement="N/A",
                timestamp=timestamp,
            )

        # If model now says SELL (direction reversal), signal EXIT
        if ensemble_result.action == "SELL":
            return TradeSignal(
                asset=asset,
                action="EXIT",
                confidence=ensemble_result.confidence,
                suggested_hold_days=0,
                stop_loss_pct=self._stop_loss_pct,
                expected_return_pct=price_change_pct,
                model_agreement=ensemble_result.model_agreement,
                timestamp=timestamp,
            )

        # Model still says BUY => HOLD
        return TradeSignal(
            asset=asset,
            action="HOLD",
            confidence=ensemble_result.confidence,
            suggested_hold_days=ensemble_result.suggested_hold_days,
            stop_loss_pct=self._stop_loss_pct,
            expected_return_pct=ensemble_result.expected_return_pct,
            model_agreement=ensemble_result.model_agreement,
            timestamp=timestamp,
        )
