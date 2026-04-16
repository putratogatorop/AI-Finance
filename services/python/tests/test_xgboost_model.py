import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.ml.models.base import ModelPrediction
from src.ml.models.xgboost_model import XGBoostModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic training data."""
    np.random.seed(42)
    n = 500
    n_features = 30
    X = pd.DataFrame(
        np.random.randn(n, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )
    y = pd.DataFrame({
        "target_7d": np.random.randn(n) * 0.05,
        "target_14d": np.random.randn(n) * 0.08,
        "target_28d": np.random.randn(n) * 0.10,
        "target_42d": np.random.randn(n) * 0.12,
    })
    return X, y


@pytest.fixture
def model() -> XGBoostModel:
    return XGBoostModel(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
    )


class TestXGBoostModel:
    def test_name(self, model: XGBoostModel):
        assert model.name == "xgboost"

    def test_train_returns_metrics(
        self,
        model: XGBoostModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        split = int(len(X) * 0.8)
        metrics = model.train(
            X_train=X.iloc[:split],
            y_train=y.iloc[:split],
            X_val=X.iloc[split:],
            y_val=y.iloc[split:],
        )
        assert "train_rmse_7d" in metrics
        assert "val_rmse_7d" in metrics
        assert "train_rmse_14d" in metrics
        assert metrics["train_rmse_7d"] >= 0

    def test_predict_returns_model_prediction(
        self,
        model: XGBoostModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)

        pred = model.predict(X.iloc[-5:])
        assert isinstance(pred, ModelPrediction)
        assert pred.model_name == "xgboost"
        assert len(pred.horizons) == 4
        for h in pred.horizons:
            assert h.horizon_days in [7, 14, 28, 42]
            assert 0.0 <= h.confidence <= 1.0

    def test_predict_single_row(
        self,
        model: XGBoostModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)

        pred = model.predict(X.iloc[[-1]])
        assert isinstance(pred, ModelPrediction)
        assert len(pred.horizons) == 4

    def test_save_and_load(
        self,
        model: XGBoostModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)

        pred_before = model.predict(X.iloc[[-1]])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            # Verify files were created
            files = os.listdir(tmpdir)
            assert any("xgboost" in f for f in files)

            # Load into new model instance
            model2 = XGBoostModel()
            model2.load(tmpdir)
            pred_after = model2.predict(X.iloc[[-1]])

        # Predictions should be identical
        for h_before, h_after in zip(
            pred_before.horizons, pred_after.horizons, strict=True
        ):
            assert abs(h_before.predicted_return - h_after.predicted_return) < 1e-6

    def test_feature_importance(
        self,
        model: XGBoostModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)

        importance = model.feature_importance()
        assert isinstance(importance, dict)
        # Should have importance for each target horizon
        assert "7d" in importance
        assert len(importance["7d"]) == X.shape[1]
