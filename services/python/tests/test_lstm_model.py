import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.ml.models.base import ModelPrediction
from src.ml.models.lstm_model import LSTMModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic sequential training data.

    LSTM needs enough rows for sequence windowing.
    """
    np.random.seed(42)
    n = 300
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
def model() -> LSTMModel:
    return LSTMModel(
        sequence_length=20,
        hidden_size=32,
        num_layers=1,
        dropout=0.0,
        learning_rate=0.001,
        epochs=5,
        batch_size=32,
    )


class TestLSTMModel:
    def test_name(self, model: LSTMModel):
        assert model.name == "lstm"

    def test_train_returns_metrics(
        self,
        model: LSTMModel,
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
        assert "train_loss" in metrics
        assert "val_loss" in metrics
        assert metrics["train_loss"] >= 0

    def test_predict_returns_model_prediction(
        self,
        model: LSTMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)

        # Need at least sequence_length rows for prediction
        pred = model.predict(X.iloc[-30:])
        assert isinstance(pred, ModelPrediction)
        assert pred.model_name == "lstm"
        assert len(pred.horizons) == 4
        for h in pred.horizons:
            assert h.horizon_days in [7, 14, 28, 42]
            assert 0.0 <= h.confidence <= 1.0

    def test_save_and_load(
        self,
        model: LSTMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)
        pred_before = model.predict(X.iloc[-30:])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            files = os.listdir(tmpdir)
            assert any("lstm" in f for f in files)

            model2 = LSTMModel(
                sequence_length=20,
                hidden_size=32,
                num_layers=1,
                dropout=0.0,
            )
            model2.load(tmpdir)
            pred_after = model2.predict(X.iloc[-30:])

        for h_before, h_after in zip(
            pred_before.horizons, pred_after.horizons, strict=True
        ):
            assert abs(h_before.predicted_return - h_after.predicted_return) < 1e-4

    def test_predict_with_exact_sequence_length(
        self,
        model: LSTMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)
        # Exactly sequence_length rows
        pred = model.predict(X.iloc[-20:])
        assert len(pred.horizons) == 4

    def test_predict_too_few_rows_raises(
        self,
        model: LSTMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)
        with pytest.raises(ValueError, match="at least"):
            model.predict(X.iloc[-5:])  # Less than sequence_length=20
