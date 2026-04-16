from datetime import UTC, datetime

import pytest

from src.ml.ensemble import EnsembleResult
from src.ml.models.base import HorizonPrediction
from src.ml.signals import SignalGenerator, TradeSignal


@pytest.fixture
def generator() -> SignalGenerator:
    return SignalGenerator(stop_loss_pct=-8.0)


def _make_ensemble_result(
    action: str = "BUY",
    confidence: float = 0.84,
    hold_days: int = 14,
    expected_return_pct: float = 8.5,
    agreement: str = "3/3",
) -> EnsembleResult:
    horizons = [
        HorizonPrediction(horizon_days=7, predicted_return=0.05, confidence=0.75),
        HorizonPrediction(horizon_days=14, predicted_return=0.085, confidence=confidence),
        HorizonPrediction(horizon_days=28, predicted_return=0.10, confidence=0.70),
        HorizonPrediction(horizon_days=42, predicted_return=0.12, confidence=0.60),
    ]
    return EnsembleResult(
        horizons=horizons,
        model_agreement=agreement,
        action=action,
        suggested_hold_days=hold_days,
        expected_return_pct=expected_return_pct,
        confidence=confidence,
    )


class TestTradeSignal:
    def test_signal_dataclass(self):
        signal = TradeSignal(
            asset="ETH",
            action="BUY",
            confidence=0.84,
            suggested_hold_days=18,
            stop_loss_pct=-8.0,
            expected_return_pct=12.0,
            model_agreement="3/3",
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert signal.asset == "ETH"
        assert signal.action == "BUY"
        assert signal.confidence == 0.84

    def test_signal_to_dict(self):
        signal = TradeSignal(
            asset="BTC",
            action="SELL",
            confidence=0.75,
            suggested_hold_days=7,
            stop_loss_pct=-8.0,
            expected_return_pct=-5.5,
            model_agreement="2/3",
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        d = signal.to_dict()
        assert d["asset"] == "BTC"
        assert d["action"] == "SELL"
        assert "timestamp" in d


class TestSignalGenerator:
    def test_generate_buy_signal(self, generator: SignalGenerator):
        ensemble_result = _make_ensemble_result(action="BUY", confidence=0.84)
        signal = generator.generate(
            asset="ETH",
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert signal is not None
        assert signal.asset == "ETH"
        assert signal.action == "BUY"
        assert signal.confidence == 0.84
        assert signal.stop_loss_pct == -8.0
        assert signal.model_agreement == "3/3"

    def test_generate_sell_signal(self, generator: SignalGenerator):
        ensemble_result = _make_ensemble_result(
            action="SELL", confidence=0.78, expected_return_pct=-6.0
        )
        signal = generator.generate(
            asset="BTC",
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert signal is not None
        assert signal.action == "SELL"

    def test_none_ensemble_returns_none(self, generator: SignalGenerator):
        signal = generator.generate(
            asset="BTC",
            ensemble_result=None,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert signal is None

    def test_generate_for_multiple_assets(self, generator: SignalGenerator):
        results = {
            "BTC": _make_ensemble_result(action="BUY", confidence=0.85),
            "ETH": None,
            "SOL": _make_ensemble_result(action="SELL", confidence=0.72),
        }
        signals = generator.generate_batch(
            ensemble_results=results,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert len(signals) == 2  # ETH filtered out (None result)
        assert signals[0].asset == "BTC"
        assert signals[1].asset == "SOL"

    def test_evaluate_open_position_exit(self, generator: SignalGenerator):
        """When model confidence drops, signal EXIT."""
        ensemble_result = _make_ensemble_result(
            action="SELL", confidence=0.78, expected_return_pct=-5.0
        )
        signal = generator.evaluate_open_position(
            asset="ETH",
            entry_price=3500.0,
            current_price=3400.0,
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert signal is not None
        assert signal.action == "EXIT"

    def test_evaluate_open_position_hold(self, generator: SignalGenerator):
        """When model still confident, signal HOLD."""
        ensemble_result = _make_ensemble_result(
            action="BUY", confidence=0.80, expected_return_pct=8.0
        )
        signal = generator.evaluate_open_position(
            asset="ETH",
            entry_price=3500.0,
            current_price=3600.0,
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert signal is not None
        assert signal.action == "HOLD"

    def test_evaluate_stop_loss_triggered(self, generator: SignalGenerator):
        """When price drops below stop-loss, signal EXIT regardless of model."""
        ensemble_result = _make_ensemble_result(
            action="BUY", confidence=0.85, expected_return_pct=10.0
        )
        signal = generator.evaluate_open_position(
            asset="ETH",
            entry_price=3500.0,
            current_price=3200.0,  # -8.57% drop > -8% stop-loss
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )
        assert signal is not None
        assert signal.action == "EXIT"
