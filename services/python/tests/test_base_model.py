import pandas as pd
import pytest

from src.ml.models.base import BaseModel, HorizonPrediction, ModelPrediction


class TestModelPrediction:
    def test_prediction_dataclass(self):
        hp = HorizonPrediction(
            horizon_days=7, predicted_return=0.05, confidence=0.82,
        )
        assert hp.horizon_days == 7
        assert hp.predicted_return == 0.05
        assert hp.confidence == 0.82

    def test_model_prediction(self):
        horizons = [
            HorizonPrediction(
                horizon_days=7,
                predicted_return=0.05,
                confidence=0.8,
            ),
            HorizonPrediction(
                horizon_days=14,
                predicted_return=0.08,
                confidence=0.75,
            ),
        ]
        pred = ModelPrediction(model_name="xgboost", horizons=horizons)
        assert pred.model_name == "xgboost"
        assert len(pred.horizons) == 2
        assert pred.best_horizon.horizon_days == 7

    def test_model_prediction_best_horizon_by_confidence(self):
        horizons = [
            HorizonPrediction(
                horizon_days=7, predicted_return=0.02, confidence=0.6,
            ),
            HorizonPrediction(
                horizon_days=14, predicted_return=0.10, confidence=0.9,
            ),
            HorizonPrediction(
                horizon_days=28, predicted_return=0.15, confidence=0.7,
            ),
        ]
        pred = ModelPrediction(model_name="test", horizons=horizons)
        assert pred.best_horizon.horizon_days == 14


class TestBaseModelInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseModel()

    def test_concrete_implementation(self):
        class DummyModel(BaseModel):
            @property
            def name(self) -> str:
                return "dummy"

            def train(self, X_train, y_train, X_val=None, y_val=None):
                return {"loss": 0.1}

            def predict(self, X):
                horizons = [
                    HorizonPrediction(
                        horizon_days=7,
                        predicted_return=0.05,
                        confidence=0.8,
                    )
                ]
                return ModelPrediction(
                    model_name=self.name, horizons=horizons
                )

            def save(self, path):
                pass

            def load(self, path):
                pass

        model = DummyModel()
        assert model.name == "dummy"
        X = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        y = pd.DataFrame({"target_7d": [0.01, 0.02, 0.03]})
        metrics = model.train(X, y)
        assert "loss" in metrics
        pred = model.predict(X)
        assert pred.model_name == "dummy"
        assert len(pred.horizons) == 1
