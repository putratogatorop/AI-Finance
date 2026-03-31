import os
import tempfile
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from src.ml.versioning import ModelVersion, ModelVersionManager


@pytest.fixture
def version_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def manager(version_dir: str) -> ModelVersionManager:
    return ModelVersionManager(base_dir=version_dir)


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    np.random.seed(42)
    n = 200
    X = pd.DataFrame(
        np.random.randn(n, 10), columns=[f"f_{i}" for i in range(10)]
    )
    y = pd.DataFrame({
        "target_7d": np.random.randn(n) * 0.05,
        "target_14d": np.random.randn(n) * 0.08,
        "target_28d": np.random.randn(n) * 0.10,
        "target_42d": np.random.randn(n) * 0.12,
    })
    return X, y


class TestModelVersion:
    def test_version_dataclass(self):
        v = ModelVersion(
            version_id="v_20260330_120000",
            model_name="xgboost",
            created_at=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
            metrics={"train_rmse_7d": 0.03, "val_rmse_7d": 0.04},
            path="/models/xgboost/v_20260330_120000",
        )
        assert v.version_id == "v_20260330_120000"
        assert v.model_name == "xgboost"
        assert v.metrics["val_rmse_7d"] == 0.04


class TestModelVersionManager:
    def test_save_version(self, manager: ModelVersionManager, version_dir: str):
        from src.ml.models.xgboost_model import XGBoostModel

        model = XGBoostModel(n_estimators=10, max_depth=2)

        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({
            "target_7d": np.random.randn(100) * 0.05,
            "target_14d": np.random.randn(100) * 0.08,
            "target_28d": np.random.randn(100) * 0.10,
            "target_42d": np.random.randn(100) * 0.12,
        })
        model.train(X, y)

        metrics = {"train_rmse_7d": 0.03}
        version = manager.save_version(
            model=model,
            metrics=metrics,
            timestamp=datetime(2026, 3, 30, 12, 0, tzinfo=UTC),
        )

        assert version.version_id.startswith("v_")
        assert version.model_name == "xgboost"
        assert os.path.exists(version.path)

    def test_list_versions(self, manager: ModelVersionManager):
        from src.ml.models.xgboost_model import XGBoostModel

        model = XGBoostModel(n_estimators=10, max_depth=2)
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({
            "target_7d": np.random.randn(100) * 0.05,
            "target_14d": np.random.randn(100) * 0.08,
            "target_28d": np.random.randn(100) * 0.10,
            "target_42d": np.random.randn(100) * 0.12,
        })
        model.train(X, y)

        manager.save_version(
            model=model, metrics={"rmse": 0.03},
            timestamp=datetime(2026, 3, 23, tzinfo=UTC),
        )
        manager.save_version(
            model=model, metrics={"rmse": 0.025},
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )

        versions = manager.list_versions("xgboost")
        assert len(versions) == 2
        assert versions[0].created_at > versions[1].created_at

    def test_load_latest(self, manager: ModelVersionManager):
        from src.ml.models.xgboost_model import XGBoostModel

        model = XGBoostModel(n_estimators=10, max_depth=2)
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({
            "target_7d": np.random.randn(100) * 0.05,
            "target_14d": np.random.randn(100) * 0.08,
            "target_28d": np.random.randn(100) * 0.10,
            "target_42d": np.random.randn(100) * 0.12,
        })
        model.train(X, y)
        pred_original = model.predict(X.iloc[[-1]])

        manager.save_version(
            model=model, metrics={"rmse": 0.03},
            timestamp=datetime(2026, 3, 30, tzinfo=UTC),
        )

        loaded_model = XGBoostModel()
        manager.load_latest(loaded_model, "xgboost")
        pred_loaded = loaded_model.predict(X.iloc[[-1]])

        assert abs(
            pred_original.horizons[0].predicted_return
            - pred_loaded.horizons[0].predicted_return
        ) < 1e-6

    def test_should_replace_better_metrics(self, manager: ModelVersionManager):
        old_metrics = {"val_rmse_7d": 0.05, "val_rmse_14d": 0.06}
        new_metrics = {"val_rmse_7d": 0.04, "val_rmse_14d": 0.05}
        assert manager.should_replace(old_metrics, new_metrics) is True

    def test_should_not_replace_worse_metrics(self, manager: ModelVersionManager):
        old_metrics = {"val_rmse_7d": 0.03, "val_rmse_14d": 0.04}
        new_metrics = {"val_rmse_7d": 0.05, "val_rmse_14d": 0.06}
        assert manager.should_replace(old_metrics, new_metrics) is False

    def test_should_replace_equal_metrics(self, manager: ModelVersionManager):
        old_metrics = {"val_rmse_7d": 0.04, "val_rmse_14d": 0.05}
        new_metrics = {"val_rmse_7d": 0.04, "val_rmse_14d": 0.05}
        assert manager.should_replace(old_metrics, new_metrics) is True

    def test_generate_version_id(self, manager: ModelVersionManager):
        ts = datetime(2026, 3, 30, 14, 30, 0, tzinfo=UTC)
        vid = manager._generate_version_id(ts)
        assert vid == "v_20260330_143000"

    def test_retrain_decision_weekly(self, manager: ModelVersionManager):
        from src.ml.models.xgboost_model import XGBoostModel

        model = XGBoostModel(n_estimators=10, max_depth=2)
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({
            "target_7d": np.random.randn(100) * 0.05,
            "target_14d": np.random.randn(100) * 0.08,
            "target_28d": np.random.randn(100) * 0.10,
            "target_42d": np.random.randn(100) * 0.12,
        })
        model.train(X, y)

        manager.save_version(
            model=model, metrics={"rmse": 0.03},
            timestamp=datetime(2026, 3, 20, tzinfo=UTC),
        )

        needs_retrain = manager.needs_retraining(
            model_name="xgboost",
            current_time=datetime(2026, 3, 30, tzinfo=UTC),
            retrain_interval_days=7,
        )
        assert needs_retrain is True

    def test_no_retrain_if_recent(self, manager: ModelVersionManager):
        from src.ml.models.xgboost_model import XGBoostModel

        model = XGBoostModel(n_estimators=10, max_depth=2)
        np.random.seed(42)
        X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f{i}" for i in range(5)])
        y = pd.DataFrame({
            "target_7d": np.random.randn(100) * 0.05,
            "target_14d": np.random.randn(100) * 0.08,
            "target_28d": np.random.randn(100) * 0.10,
            "target_42d": np.random.randn(100) * 0.12,
        })
        model.train(X, y)

        manager.save_version(
            model=model, metrics={"rmse": 0.03},
            timestamp=datetime(2026, 3, 28, tzinfo=UTC),
        )

        needs_retrain = manager.needs_retraining(
            model_name="xgboost",
            current_time=datetime(2026, 3, 30, tzinfo=UTC),
            retrain_interval_days=7,
        )
        assert needs_retrain is False
