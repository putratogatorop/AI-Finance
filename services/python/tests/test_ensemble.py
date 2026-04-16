import pytest

from src.ml.ensemble import EnsembleConsensus
from src.ml.models.base import HorizonPrediction, ModelPrediction


@pytest.fixture
def ensemble() -> EnsembleConsensus:
    return EnsembleConsensus(
        model_weights={"xgboost": 0.4, "lightgbm": 0.4, "lstm": 0.2},
        confidence_threshold=0.7,
    )


def _make_prediction(
    model_name: str,
    returns: list[float],
    confidences: list[float],
) -> ModelPrediction:
    """Helper to build a ModelPrediction."""
    horizons = [
        HorizonPrediction(horizon_days=d, predicted_return=r, confidence=c)
        for d, r, c in zip([7, 14, 28, 42], returns, confidences, strict=True)
    ]
    return ModelPrediction(model_name=model_name, horizons=horizons)


class TestEnsembleConsensus:
    def test_weighted_average_returns(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)

        assert result is not None
        # Weighted average: 0.4*0.05 + 0.4*0.04 + 0.2*0.06 = 0.048
        expected_7d = 0.4 * 0.05 + 0.4 * 0.04 + 0.2 * 0.06
        assert abs(result.horizons[0].predicted_return - expected_7d) < 1e-6

    def test_confidence_weighted_average(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # Confidence is also weighted averaged
        expected_conf_7d = 0.4 * 0.8 + 0.4 * 0.75 + 0.2 * 0.85
        assert abs(result.horizons[0].confidence - expected_conf_7d) < 1e-6

    def test_below_confidence_threshold_returns_none(self):
        high_threshold = EnsembleConsensus(
            model_weights={"xgboost": 0.5, "lightgbm": 0.5},
            confidence_threshold=0.9,
        )
        predictions = [
            _make_prediction("xgboost", [0.01, 0.01, 0.01, 0.01], [0.5, 0.5, 0.5, 0.5]),
            _make_prediction("lightgbm", [0.01, 0.01, 0.01, 0.01], [0.5, 0.5, 0.5, 0.5]),
        ]
        result = high_threshold.combine(predictions)
        # All confidences below 0.9 => no horizon passes => None
        assert result is None

    def test_model_agreement_all_agree(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # All models predict positive => agreement = "3/3"
        assert result.model_agreement == "3/3"

    def test_model_agreement_partial(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [-0.02, -0.01, 0.01, 0.02], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # For 7d horizon: xgboost positive, lightgbm negative, lstm positive => 2/3
        assert result.model_agreement == "2/3"

    def test_best_horizon_selection(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.02, 0.08, 0.10, 0.05], [0.6, 0.85, 0.75, 0.5]),
            _make_prediction("lightgbm", [0.01, 0.07, 0.09, 0.04], [0.55, 0.82, 0.70, 0.45]),
            _make_prediction("lstm", [0.03, 0.09, 0.11, 0.06], [0.65, 0.88, 0.80, 0.55]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # 14d horizon should have highest ensemble confidence
        assert result.best_horizon.horizon_days == 14

    def test_suggested_hold_days(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.02, 0.08, 0.10, 0.05], [0.6, 0.85, 0.75, 0.5]),
            _make_prediction("lightgbm", [0.01, 0.07, 0.09, 0.04], [0.55, 0.82, 0.70, 0.45]),
            _make_prediction("lstm", [0.03, 0.09, 0.11, 0.06], [0.65, 0.88, 0.80, 0.55]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        assert result.suggested_hold_days == 14

    def test_action_buy_for_positive_return(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.75, 0.7, 0.6]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.78, 0.73, 0.68, 0.58]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.82, 0.77, 0.72, 0.62]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        assert result.action == "BUY"

    def test_action_sell_for_negative_return(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [-0.05, -0.08, -0.10, -0.12], [0.8, 0.75, 0.7, 0.6]),
            _make_prediction("lightgbm", [-0.04, -0.07, -0.09, -0.11], [0.78, 0.73, 0.68, 0.58]),
            _make_prediction("lstm", [-0.06, -0.09, -0.11, -0.13], [0.82, 0.77, 0.72, 0.62]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        assert result.action == "SELL"
