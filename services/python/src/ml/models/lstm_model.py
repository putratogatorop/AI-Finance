"""LSTM model (PyTorch) for multi-horizon crypto return prediction.

Uses a sequence of N days of features to predict forward returns at 4 horizons.
Includes feature scaling (StandardScaler) as part of the pipeline.
"""

from __future__ import annotations

import os
from typing import Any, cast

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.ml.models.base import BaseModel, HorizonPrediction, ModelPrediction


class _LSTMNetwork(nn.Module):
    """PyTorch LSTM network for multi-output regression."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: sequence in, multi-horizon prediction out."""
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use output of last timestep
        last_hidden = lstm_out[:, -1, :]
        return cast(torch.Tensor, self.fc(last_hidden))


class LSTMModel(BaseModel):
    """LSTM model for multi-horizon return prediction."""

    def __init__(
        self,
        sequence_length: int = 30,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        patience: int = 10,
    ) -> None:
        self._sequence_length = sequence_length
        self._hidden_size = hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._batch_size = batch_size
        self._patience = patience

        self._network: _LSTMNetwork | None = None
        self._scaler_mean: np.ndarray[Any, np.dtype[Any]] | None = None
        self._scaler_std: np.ndarray[Any, np.dtype[Any]] | None = None
        self._n_features: int = 0
        self._device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "lstm"

    def _create_sequences(
        self,
        X: np.ndarray[Any, np.dtype[Any]],
        y: np.ndarray[Any, np.dtype[Any]] | None = None,
    ) -> tuple[np.ndarray[Any, np.dtype[Any]], np.ndarray[Any, np.dtype[Any]] | None]:
        """Create sliding window sequences from time-series data."""
        n = len(X)
        seq_len = self._sequence_length
        if n < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows for sequences, got {n}."
            )

        X_seqs = []
        y_seqs: list[np.ndarray[Any, np.dtype[Any]]] | None = [] if y is not None else None

        for i in range(n - seq_len + 1):
            X_seqs.append(X[i : i + seq_len])
            if y is not None and y_seqs is not None:
                y_seqs.append(y[i + seq_len - 1])

        X_out = np.array(X_seqs, dtype=np.float32)
        y_out = (
            np.array(y_seqs, dtype=np.float32) if y_seqs is not None else None
        )
        return (
            cast(np.ndarray[Any, np.dtype[Any]], X_out),
            cast(np.ndarray[Any, np.dtype[Any]], y_out) if y_out is not None else None,
        )

    def _fit_scaler(self, X: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """Fit and apply standard scaling."""
        values = X.values.astype(np.float32)
        self._scaler_mean = values.mean(axis=0)
        self._scaler_std = values.std(axis=0)
        self._scaler_std[self._scaler_std == 0] = 1.0
        return cast(np.ndarray[Any, np.dtype[Any]], (values - self._scaler_mean) / self._scaler_std)

    def _apply_scaler(self, X: pd.DataFrame) -> np.ndarray[Any, np.dtype[Any]]:
        """Apply previously fit scaling."""
        if self._scaler_mean is None or self._scaler_std is None:
            raise RuntimeError("Scaler not fitted. Call train() first.")
        values = X.values.astype(np.float32)
        return cast(np.ndarray[Any, np.dtype[Any]], (values - self._scaler_mean) / self._scaler_std)

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: pd.DataFrame | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> dict[str, float]:
        """Train the LSTM on sequential data."""
        self._n_features = X_train.shape[1]
        n_targets = len(self.TARGET_COLUMNS)

        # Scale features
        X_scaled = self._fit_scaler(X_train)
        y_arr = y_train[self.TARGET_COLUMNS].values.astype(np.float32)

        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_arr)
        if y_seq is None:
            raise RuntimeError("y_seq should not be None during training.")

        # Build network
        self._network = _LSTMNetwork(
            input_size=self._n_features,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            output_size=n_targets,
            dropout=self._dropout,
        ).to(self._device)

        # DataLoader
        train_dataset = TensorDataset(
            torch.from_numpy(X_seq), torch.from_numpy(y_seq)
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self._batch_size, shuffle=True
        )

        # Validation data
        val_loader: DataLoader[Any] | None = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self._apply_scaler(X_val)
            y_val_arr = y_val[self.TARGET_COLUMNS].values.astype(np.float32)
            X_val_seq, y_val_seq = self._create_sequences(
                X_val_scaled, y_val_arr
            )
            if y_val_seq is not None:
                val_dataset = TensorDataset(
                    torch.from_numpy(X_val_seq),
                    torch.from_numpy(y_val_seq),
                )
                val_loader = DataLoader(
                    val_dataset, batch_size=self._batch_size, shuffle=False
                )

        # Optimizer and loss
        optimizer = torch.optim.Adam(
            self._network.parameters(), lr=self._learning_rate
        )
        criterion = nn.MSELoss()

        # Training loop with early stopping
        best_val_loss = float("inf")
        patience_counter = 0
        best_state: dict[str, Any] = {}

        self._network.train()
        final_train_loss = 0.0
        final_val_loss = 0.0

        for _epoch in range(self._epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                predictions = self._network(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._network.parameters(), 1.0
                )
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            final_train_loss = epoch_loss / max(n_batches, 1)

            # Validation
            if val_loader is not None:
                self._network.eval()
                val_loss = 0.0
                val_batches = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self._device)
                        y_batch = y_batch.to(self._device)
                        predictions = self._network(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()
                        val_batches += 1

                final_val_loss = val_loss / max(val_batches, 1)
                self._network.train()

                # Early stopping
                if final_val_loss < best_val_loss:
                    best_val_loss = final_val_loss
                    best_state = {
                        k: v.clone()
                        for k, v in self._network.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self._patience:
                        break

        # Restore best weights if early stopping was used
        if best_state:
            self._network.load_state_dict(best_state)

        self._network.eval()

        metrics: dict[str, float] = {"train_loss": final_train_loss}
        if val_loader is not None:
            metrics["val_loss"] = (
                best_val_loss if best_state else final_val_loss
            )

        return metrics

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Predict returns for all horizons using the last sequence_length rows."""
        if self._network is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if len(X) < self._sequence_length:
            raise ValueError(
                f"Need at least {self._sequence_length} rows "
                f"for prediction, got {len(X)}."
            )

        # Scale and create one sequence from the last sequence_length rows
        X_scaled = self._apply_scaler(X.iloc[-self._sequence_length :])
        X_tensor = torch.from_numpy(
            X_scaled.reshape(1, self._sequence_length, self._n_features)
        ).to(self._device)

        self._network.eval()
        with torch.no_grad():
            raw_preds = self._network(X_tensor).cpu().numpy()[0]

        horizons: list[HorizonPrediction] = []
        for i, (_, horizon_days) in enumerate(
            zip(self.TARGET_COLUMNS, self.HORIZON_DAYS, strict=True)
        ):
            predicted_return = float(raw_preds[i])
            confidence = self._return_to_confidence(predicted_return)
            horizons.append(
                HorizonPrediction(
                    horizon_days=horizon_days,
                    predicted_return=predicted_return,
                    confidence=confidence,
                )
            )

        return ModelPrediction(model_name=self.name, horizons=horizons)

    def save(self, path: str) -> None:
        """Save LSTM network weights and scaler parameters."""
        if self._network is None:
            raise RuntimeError("Model not trained. Call train() first.")

        os.makedirs(path, exist_ok=True)

        # Save network weights
        torch.save(
            self._network.state_dict(),
            os.path.join(path, "lstm_weights.pt"),
        )

        # Save scaler and config
        config = {
            "sequence_length": self._sequence_length,
            "hidden_size": self._hidden_size,
            "num_layers": self._num_layers,
            "dropout": self._dropout,
            "n_features": self._n_features,
            "scaler_mean": self._scaler_mean,
            "scaler_std": self._scaler_std,
        }
        joblib.dump(config, os.path.join(path, "lstm_config.joblib"))

    def load(self, path: str) -> None:
        """Load LSTM network weights and scaler parameters."""
        config = joblib.load(os.path.join(path, "lstm_config.joblib"))
        self._sequence_length = config["sequence_length"]
        self._hidden_size = config["hidden_size"]
        self._num_layers = config["num_layers"]
        self._dropout = config["dropout"]
        self._n_features = config["n_features"]
        self._scaler_mean = config["scaler_mean"]
        self._scaler_std = config["scaler_std"]

        # Rebuild network architecture
        self._network = _LSTMNetwork(
            input_size=self._n_features,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            output_size=len(self.TARGET_COLUMNS),
            dropout=self._dropout,
        ).to(self._device)

        # Load weights
        state_dict = torch.load(
            os.path.join(path, "lstm_weights.pt"),
            map_location=self._device,
            weights_only=True,
        )
        self._network.load_state_dict(state_dict)
        self._network.eval()
