"""XGBoost model for multi-horizon crypto return prediction.

Trains one XGBRegressor per target horizon (7d, 14d, 28d, 42d).
Uses early stopping on validation set when provided.
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.ml.models.base import BaseModel, HorizonPrediction, ModelPrediction


class XGBoostModel(BaseModel):
    """XGBoost ensemble for multi-horizon return prediction."""

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 3,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate
        self._subsample = subsample
        self._colsample_bytree = colsample_bytree
        self._min_child_weight = min_child_weight
        self._reg_alpha = reg_alpha
        self._reg_lambda = reg_lambda
        self._early_stopping_rounds = early_stopping_rounds
        self._random_state = random_state

        # One model per horizon
        self._models: dict[str, xgb.XGBRegressor] = {}

    @property
    def name(self) -> str:
        return "xgboost"

    def _make_regressor(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            subsample=self._subsample,
            colsample_bytree=self._colsample_bytree,
            min_child_weight=self._min_child_weight,
            reg_alpha=self._reg_alpha,
            reg_lambda=self._reg_lambda,
            random_state=self._random_state,
            objective="reg:squarederror",
            n_jobs=-1,
            verbosity=0,
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train one XGBoost regressor per target horizon."""
        metrics: dict[str, float] = {}

        for target_col, horizon_days in zip(
            self.TARGET_COLUMNS, self.HORIZON_DAYS, strict=True
        ):
            label = f"{horizon_days}d"
            reg = self._make_regressor()

            fit_params: dict = {}
            if X_val is not None and y_val is not None:
                fit_params["eval_set"] = [(X_val, y_val[target_col])]
                fit_params["verbose"] = False

            reg.fit(X_train, y_train[target_col], **fit_params)
            self._models[target_col] = reg

            # Compute train RMSE
            train_pred = reg.predict(X_train)
            train_rmse = float(
                np.sqrt(
                    np.mean((train_pred - y_train[target_col].values) ** 2)
                )
            )
            metrics[f"train_rmse_{label}"] = train_rmse

            # Compute val RMSE if validation data provided
            if X_val is not None and y_val is not None:
                val_pred = reg.predict(X_val)
                val_rmse = float(
                    np.sqrt(
                        np.mean((val_pred - y_val[target_col].values) ** 2)
                    )
                )
                metrics[f"val_rmse_{label}"] = val_rmse

        return metrics

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Predict returns for all horizons.

        Uses the last row for signal generation.
        """
        if not self._models:
            raise RuntimeError("Model not trained. Call train() first.")

        # Use last row for signal (most recent data point)
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
            model_path = os.path.join(path, f"xgboost_{target_col}.joblib")
            joblib.dump(reg, model_path)

        # Save hyperparams for reproducibility
        params = {
            "n_estimators": self._n_estimators,
            "max_depth": self._max_depth,
            "learning_rate": self._learning_rate,
            "subsample": self._subsample,
            "colsample_bytree": self._colsample_bytree,
            "min_child_weight": self._min_child_weight,
            "reg_alpha": self._reg_alpha,
            "reg_lambda": self._reg_lambda,
        }
        joblib.dump(params, os.path.join(path, "xgboost_params.joblib"))

    def load(self, path: str) -> None:
        """Load all horizon models from disk."""
        self._models = {}
        for target_col in self.TARGET_COLUMNS:
            model_path = os.path.join(
                path, f"xgboost_{target_col}.joblib"
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
            feature_names = reg.get_booster().feature_names or [
                f"f{i}" for i in range(len(importance))
            ]
            result[label] = dict(
                zip(feature_names, importance.tolist(), strict=True)
            )

        return result
