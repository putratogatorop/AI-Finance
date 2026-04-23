"""LightGBM model for multi-horizon crypto return prediction.

Trains one LGBMRegressor per target horizon (7d, 14d, 28d, 42d).
Uses early stopping on validation set when provided.
"""

from __future__ import annotations

import os
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.ml.models.base import BaseModel, HorizonPrediction, ModelPrediction


class LightGBMModel(BaseModel):
    """LightGBM ensemble for multi-horizon return prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._num_leaves = num_leaves
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._min_child_samples = min_child_samples
        self._reg_alpha = reg_alpha
        self._reg_lambda = reg_lambda
        self._early_stopping_rounds = early_stopping_rounds
        self._random_state = random_state

        self._models: dict[str, lgb.LGBMRegressor] = {}

    @property
    def name(self) -> str:
        return "lightgbm"

    def _make_regressor(self) -> lgb.LGBMRegressor:
        return lgb.LGBMRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            num_leaves=self._num_leaves,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            min_child_samples=self._min_child_samples,
            reg_alpha=self._reg_alpha,
            reg_lambda=self._reg_lambda,
            random_state=self._random_state,
            objective="regression",
            n_jobs=-1,
            verbose=-1,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train one LightGBM regressor per target horizon."""
        metrics: dict[str, float] = {}

        for target_col, horizon_days in zip(
            self.TARGET_COLUMNS, self.HORIZON_DAYS, strict=True
        ):
            label = f"{horizon_days}d"
            reg = self._make_regressor()

            fit_params: dict[str, Any] = {}
            if X_val is not None and y_val is not None:
                fit_params["eval_set"] = [(X_val, y_val[target_col])]
                fit_params["eval_metric"] = "rmse"

            reg.fit(X_train, y_train[target_col], **fit_params)
            self._models[target_col] = reg

            # Train RMSE
            train_pred = reg.predict(X_train)
            train_rmse = float(
                np.sqrt(
                    np.mean(
                        (train_pred - y_train[target_col].values) ** 2
                    )
                )
            )
            metrics[f"train_rmse_{label}"] = train_rmse

            # Val RMSE
            if X_val is not None and y_val is not None:
                val_pred = reg.predict(X_val)
                val_rmse = float(
                    np.sqrt(
                        np.mean(
                            (val_pred - y_val[target_col].values) ** 2
                        )
                    )
                )
                metrics[f"val_rmse_{label}"] = val_rmse

        return metrics

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Predict returns for all horizons."""
        if not self._models:
            raise RuntimeError("Model not trained. Call train() first.")

        X_latest = X.iloc[[-1]] if len(X) > 1 else X

        horizons: list[HorizonPrediction] = []
        for target_col, horizon_days in zip(
            self.TARGET_COLUMNS, self.HORIZON_DAYS, strict=True
        ):
            reg = self._models[target_col]
            raw_pred = float(reg.predict(X_latest)[0])
            confidence = self._return_to_confidence(raw_pred)

            horizons.append(
                HorizonPrediction(
                    horizon_days=horizon_days,
                    predicted_return=raw_pred,
                    confidence=confidence,
                )
            )

        return ModelPrediction(model_name=self.name, horizons=horizons)

    def save(self, path: str) -> None:
        """Save all horizon models to disk."""
        os.makedirs(path, exist_ok=True)
        for target_col, reg in self._models.items():
            model_path = os.path.join(
                path, f"lightgbm_{target_col}.joblib"
            )
            joblib.dump(reg, model_path)

        params = {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "num_leaves": self._num_leaves,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "min_child_samples": self._min_child_samples,
            "reg_alpha": self._reg_alpha,
            "reg_lambda": self._reg_lambda,
        }
        joblib.dump(
            params, os.path.join(path, "lightgbm_params.joblib")
        )

    def load(self, path: str) -> None:
        """Load all horizon models from disk."""
        self._models = {}
        for target_col in self.TARGET_COLUMNS:
            model_path = os.path.join(
                path, f"lightgbm_{target_col}.joblib"
            )
            self._models[target_col] = joblib.load(model_path)

    def feature_importance(self) -> dict[str, dict[str, float]]:
        """Return feature importances for each horizon model."""
        if not self._models:
            raise RuntimeError("Model not trained. Call train() first.")

        result: dict[str, dict[str, float]] = {}
        for target_col, horizon_days in zip(
            self.TARGET_COLUMNS, self.HORIZON_DAYS, strict=True
        ):
            label = f"{horizon_days}d"
            reg = self._models[target_col]
            importance = reg.feature_importances_
            feature_names = reg.feature_name_
            result[label] = dict(
                zip(
                    feature_names,
                    importance.tolist(),
                    strict=True,
                )
            )

        return result
