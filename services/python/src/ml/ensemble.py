"""Ensemble consensus logic for combining predictions from multiple models.

Combines XGBoost, LightGBM, and LSTM predictions via weighted average.
Applies confidence threshold and model agreement checks.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.ml.models.base import HorizonPrediction, ModelPrediction


@dataclass
class EnsembleResult:
    """Result of ensemble consensus across all models."""

    horizons: list[HorizonPrediction]
    model_agreement: str  # e.g., "3/3", "2/3"
    action: str  # "BUY", "SELL", or "HOLD"
    suggested_hold_days: int
    expected_return_pct: float
    confidence: float

    @property
    def best_horizon(self) -> HorizonPrediction:
        """Return the horizon with highest confidence."""
        return max(self.horizons, key=lambda h: h.confidence)


class EnsembleConsensus:
    """Combines predictions from multiple models using weighted averaging."""

    def __init__(
        self,
        model_weights: dict[str, float] | None = None,
        confidence_threshold: float = 0.7,
    ) -> None:
        self._model_weights = model_weights or {
            "xgboost": 0.4,
            "lightgbm": 0.4,
            "lstm": 0.2,
        }
        self._confidence_threshold = confidence_threshold

        total = sum(self._model_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Model weights must sum to 1.0, got {total}.")

    def combine(
        self, predictions: list[ModelPrediction]
    ) -> EnsembleResult | None:
        """Combine predictions from all models into an ensemble result."""
        if not predictions:
            return None

        weight_lookup: dict[str, float] = {}
        for pred in predictions:
            if pred.model_name in self._model_weights:
                weight_lookup[pred.model_name] = self._model_weights[pred.model_name]

        total_weight = sum(weight_lookup.values())
        if total_weight == 0:
            return None
        normalized_weights = {
            k: v / total_weight for k, v in weight_lookup.items()
        }

        horizon_days_list = [7, 14, 28, 42]
        ensemble_horizons: list[HorizonPrediction] = []

        for h_idx, horizon_days in enumerate(horizon_days_list):
            weighted_return = 0.0
            weighted_confidence = 0.0

            for pred in predictions:
                w = normalized_weights.get(pred.model_name, 0.0)
                if h_idx < len(pred.horizons):
                    h = pred.horizons[h_idx]
                    weighted_return += w * h.predicted_return
                    weighted_confidence += w * h.confidence

            ensemble_horizons.append(
                HorizonPrediction(
                    horizon_days=horizon_days,
                    predicted_return=weighted_return,
                    confidence=weighted_confidence,
                )
            )

        passing_horizons = [
            h for h in ensemble_horizons if h.confidence >= self._confidence_threshold
        ]

        if not passing_horizons:
            return None

        best = max(passing_horizons, key=lambda h: h.confidence)

        best_idx = horizon_days_list.index(best.horizon_days)
        agree_count = 0
        total_models = len(predictions)

        best_direction = 1 if best.predicted_return >= 0 else -1
        for pred in predictions:
            if best_idx < len(pred.horizons):
                pred_direction = (
                    1 if pred.horizons[best_idx].predicted_return >= 0 else -1
                )
                if pred_direction == best_direction:
                    agree_count += 1

        model_agreement = f"{agree_count}/{total_models}"

        if best.predicted_return >= 0:
            action = "BUY"
        else:
            action = "SELL"

        return EnsembleResult(
            horizons=ensemble_horizons,
            model_agreement=model_agreement,
            action=action,
            suggested_hold_days=best.horizon_days,
            expected_return_pct=best.predicted_return * 100,
            confidence=best.confidence,
        )
