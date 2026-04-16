"""Model versioning and weekly retraining logic.

Manages model lifecycle:
- Save trained models with version IDs and timestamps.
- Load the latest or a specific version.
- Compare new models against old ones (only replace if equal or better).
- Track retraining schedule (weekly on Sundays).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime

from src.ml.models.base import BaseModel


@dataclass
class ModelVersion:
    """Metadata about a saved model version."""

    version_id: str
    model_name: str
    created_at: datetime
    metrics: dict[str, float]
    path: str


class ModelVersionManager:
    """Manages model saving, loading, versioning, and retraining decisions."""

    def __init__(self, base_dir: str = "models") -> None:
        self._base_dir = base_dir

    def _generate_version_id(self, timestamp: datetime) -> str:
        """Generate a version ID from a timestamp."""
        return f"v_{timestamp.strftime('%Y%m%d_%H%M%S')}"

    def save_version(
        self,
        model: BaseModel,
        metrics: dict[str, float],
        timestamp: datetime | None = None,
    ) -> ModelVersion:
        """Save a trained model as a new version."""
        if timestamp is None:
            timestamp = datetime.now(UTC)

        version_id = self._generate_version_id(timestamp)
        model_dir = os.path.join(self._base_dir, model.name, version_id)
        os.makedirs(model_dir, exist_ok=True)

        model.save(model_dir)

        version = ModelVersion(
            version_id=version_id,
            model_name=model.name,
            created_at=timestamp,
            metrics=metrics,
            path=model_dir,
        )

        metadata = {
            "version_id": version.version_id,
            "model_name": version.model_name,
            "created_at": version.created_at.isoformat(),
            "metrics": version.metrics,
        }
        metadata_path = os.path.join(model_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return version

    def list_versions(self, model_name: str) -> list[ModelVersion]:
        """List all saved versions for a model, most recent first."""
        model_dir = os.path.join(self._base_dir, model_name)
        if not os.path.exists(model_dir):
            return []

        versions: list[ModelVersion] = []
        for entry in os.listdir(model_dir):
            metadata_path = os.path.join(model_dir, entry, "metadata.json")
            if os.path.exists(metadata_path):
                with open(metadata_path) as f:
                    meta = json.load(f)
                versions.append(
                    ModelVersion(
                        version_id=meta["version_id"],
                        model_name=meta["model_name"],
                        created_at=datetime.fromisoformat(meta["created_at"]),
                        metrics=meta["metrics"],
                        path=os.path.join(model_dir, entry),
                    )
                )

        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def load_latest(self, model: BaseModel, model_name: str) -> ModelVersion:
        """Load the latest saved version into a model instance."""
        versions = self.list_versions(model_name)
        if not versions:
            raise FileNotFoundError(
                f"No saved versions found for model '{model_name}' "
                f"in {self._base_dir}"
            )

        latest = versions[0]
        model.load(latest.path)
        return latest

    def load_version(
        self, model: BaseModel, model_name: str, version_id: str
    ) -> ModelVersion:
        """Load a specific version into a model instance."""
        versions = self.list_versions(model_name)
        for v in versions:
            if v.version_id == version_id:
                model.load(v.path)
                return v

        raise FileNotFoundError(
            f"Version '{version_id}' not found for model '{model_name}'"
        )

    def should_replace(
        self,
        old_metrics: dict[str, float],
        new_metrics: dict[str, float],
    ) -> bool:
        """Decide whether new model should replace old model."""
        val_keys = [k for k in old_metrics if "val_rmse" in k]

        if val_keys:
            old_avg = sum(old_metrics[k] for k in val_keys) / len(val_keys)
            new_val_keys = [k for k in new_metrics if "val_rmse" in k]
            if new_val_keys:
                new_avg = (
                    sum(new_metrics[k] for k in new_val_keys) / len(new_val_keys)
                )
                return new_avg <= old_avg

        common_keys = set(old_metrics.keys()) & set(new_metrics.keys())
        if not common_keys:
            return True

        old_avg = sum(old_metrics[k] for k in common_keys) / len(common_keys)
        new_avg = sum(new_metrics[k] for k in common_keys) / len(common_keys)
        return new_avg <= old_avg

    def needs_retraining(
        self,
        model_name: str,
        current_time: datetime | None = None,
        retrain_interval_days: int = 7,
    ) -> bool:
        """Check if a model needs retraining based on time since last version."""
        if current_time is None:
            current_time = datetime.now(UTC)

        versions = self.list_versions(model_name)
        if not versions:
            return True

        latest = versions[0]
        latest_time = latest.created_at
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=UTC)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=UTC)

        days_since_last = (current_time - latest_time).days
        return days_since_last >= retrain_interval_days
