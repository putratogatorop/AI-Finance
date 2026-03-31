"""Base model interface for all ML models in the ensemble.

Every model must implement train, predict, save, and load.
Predictions are multi-horizon: each model predicts returns for 1w, 2w, 4w, 6w.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class HorizonPrediction:
    """Prediction for a single time horizon."""

    horizon_days: int  # 7, 14, 28, or 42
    predicted_return: float  # Expected % return (e.g., 0.05 = 5%)
    confidence: float  # 0.0 to 1.0


@dataclass
class ModelPrediction:
    """Complete prediction from one model, covering all horizons."""

    model_name: str
    horizons: list[HorizonPrediction] = field(default_factory=list)

    @property
    def best_horizon(self) -> HorizonPrediction:
        """Return the horizon with the highest confidence."""
        return max(self.horizons, key=lambda h: h.confidence)


class BaseModel(ABC):
    """Abstract base class for all ML models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the model name (e.g., 'xgboost', 'lightgbm', 'lstm')."""
        ...

    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train the model on feature matrix X and target matrix y.

        Args:
            X_train: Training features.
            y_train: Training targets with columns
                     [target_7d, target_14d, target_28d, target_42d].
            X_val: Optional validation features.
            y_val: Optional validation targets.

        Returns:
            Dict of training metrics (e.g., {'loss': 0.01, 'val_loss': 0.02}).
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Generate predictions for all horizons.

        Args:
            X: Feature matrix (single row or batch -- last row used for signal).

        Returns:
            ModelPrediction with predictions for each horizon.
        """
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model weights/state to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights/state from disk."""
        ...

    # --- Shared Utilities ---

    TARGET_COLUMNS: list[str] = ["target_7d", "target_14d", "target_28d", "target_42d"]
    HORIZON_DAYS: list[int] = [7, 14, 28, 42]

    def _return_to_confidence(self, predicted_return: float) -> float:
        """Convert a predicted return to a confidence score between 0 and 1.

        Uses a sigmoid-like mapping: higher absolute predicted returns map
        to higher confidence.
        """
        scaled = abs(predicted_return) * 10
        confidence = float(1.0 / (1.0 + np.exp(-scaled + 1.0)))
        return confidence
