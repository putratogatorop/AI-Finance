# Plan 2: ML Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full ML pipeline -- feature engineering, three model types (XGBoost, LightGBM, LSTM), ensemble consensus, signal generation, backtesting engine, risk management, and model versioning with weekly retraining.

**Architecture:** Python runs as a background scheduler/ML service only (no FastAPI, no API server). Feature engineering computes technical indicators, fundamental ranks, and market context from daily price data in PostgreSQL. Three independent models (XGBoost, LightGBM, LSTM) each predict multi-horizon forward returns (1w, 2w, 4w, 6w). An ensemble layer combines predictions via weighted average with a confidence threshold of 0.7 and model agreement checks. The signal generator emits actionable buy/sell signals, validated by risk management rules (stop-loss, position limits, drawdown pause). All ML results (signals, features, model scores) are written directly to PostgreSQL via SQLAlchemy. Next.js handles all API routes (see Plan 3). A backtesting engine validates everything against 2-3 years of historical data with no lookahead bias.

**Tech Stack:** Python 3.12+, pandas, numpy, scikit-learn, xgboost, lightgbm, PyTorch (LSTM), SQLAlchemy, joblib, pytest

---

## File Structure

```
services/python/src/ml/
├── __init__.py
├── features.py          # Feature engineering (technical + fundamental + market)
├── models/
│   ├── __init__.py
│   ├── base.py          # Base model interface
│   ├── xgboost_model.py
│   ├── lightgbm_model.py
│   └── lstm_model.py
├── ensemble.py          # Ensemble consensus logic
├── signals.py           # Signal generation from ensemble output
├── backtest.py          # Backtesting engine
├── risk.py              # Risk management checks
└── versioning.py        # Model save/load/versioning

services/python/tests/
├── test_features.py
├── test_base_model.py
├── test_xgboost_model.py
├── test_lightgbm_model.py
├── test_lstm_model.py
├── test_ensemble.py
├── test_signals.py
├── test_backtest.py
├── test_risk.py
└── test_versioning.py
```

## Dependencies to add to pyproject.toml

```toml
# Add to [project] dependencies:
"scikit-learn>=1.5.0",
"xgboost>=2.1.0",
"lightgbm>=4.5.0",
"torch>=2.4.0",
"joblib>=1.4.0",
```

---

### Task 1: Feature Engineering

**Files:**
- Create: `services/python/src/ml/__init__.py`
- Create: `services/python/src/ml/features.py`
- Create: `services/python/tests/test_features.py`
- Modify: `services/python/pyproject.toml`

- [ ] **Step 1: Add ML dependencies to pyproject.toml**

Modify `services/python/pyproject.toml` -- add these to the `dependencies` list:

```toml
[project]
name = "ai-finance-python"
version = "0.1.0"
description = "Crypto trading ML engine and data pipeline"
requires-python = ">=3.12"
dependencies = [
    "sqlalchemy>=2.0.0",
    "psycopg2-binary>=2.9.0",
    "pandas>=2.2.0",
    "numpy>=1.26.0",
    "ccxt>=4.0.0",
    "pycoingecko>=3.1.0",
    "apscheduler>=3.10.0",
    "pydantic>=2.0.0",
    "pydantic-settings>=2.0.0",
    "httpx>=0.27.0",
    "scikit-learn>=1.5.0",
    "xgboost>=2.1.0",
    "lightgbm>=4.5.0",
    "torch>=2.4.0",
    "joblib>=1.4.0",
]
```

- [ ] **Step 2: Create ML package init**

Create `services/python/src/ml/__init__.py`:

```python
```

- [ ] **Step 3: Write failing test for feature engineering**

Create `services/python/tests/test_features.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.ml.features import FeatureEngineer


@pytest.fixture
def sample_daily_prices() -> pd.DataFrame:
    """Generate 120 days of synthetic daily OHLCV data for testing."""
    np.random.seed(42)
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    base_price = 50000.0
    returns = np.random.normal(0.001, 0.02, n)
    close = base_price * np.cumprod(1 + returns)
    high = close * (1 + np.abs(np.random.normal(0, 0.01, n)))
    low = close * (1 - np.abs(np.random.normal(0, 0.01, n)))
    open_prices = close * (1 + np.random.normal(0, 0.005, n))
    volume = np.random.uniform(1000, 10000, n)

    return pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "open": open_prices,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def sample_fundamentals() -> pd.DataFrame:
    """Fundamental data snapshot."""
    return pd.DataFrame({
        "asset": ["BTC", "ETH", "BNB"],
        "market_cap_rank": [1, 2, 3],
        "total_volume_24h": [35e9, 15e9, 5e9],
        "market_cap": [1.4e12, 4e11, 9e10],
    })


@pytest.fixture
def sample_market_context() -> pd.DataFrame:
    """Market-level context data (BTC dominance, fear & greed)."""
    n = 120
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "btc_dominance": np.random.uniform(40, 60, n),
        "fear_greed_index": np.random.randint(10, 90, n),
        "total_market_cap": np.random.uniform(2e12, 3e12, n),
    })


@pytest.fixture
def engineer() -> FeatureEngineer:
    return FeatureEngineer()


class TestTechnicalFeatures:
    def test_sma_computed(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "sma_5" in result.columns
        assert "sma_10" in result.columns
        assert "sma_20" in result.columns
        assert "sma_50" in result.columns
        # SMA_5 at row 4 (0-indexed) should be mean of first 5 closes
        expected_sma5 = sample_daily_prices["close"].iloc[:5].mean()
        assert abs(result["sma_5"].iloc[4] - expected_sma5) < 1e-6

    def test_ema_computed(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "ema_5" in result.columns
        assert "ema_10" in result.columns
        assert "ema_20" in result.columns
        assert "ema_50" in result.columns
        # EMA should not be NaN after warmup
        assert not np.isnan(result["ema_5"].iloc[10])

    def test_rsi_computed(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "rsi_14" in result.columns
        # RSI is between 0 and 100
        valid = result["rsi_14"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_macd_computed(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "macd" in result.columns
        assert "macd_signal" in result.columns
        assert "macd_histogram" in result.columns

    def test_bollinger_bands(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "bb_upper" in result.columns
        assert "bb_middle" in result.columns
        assert "bb_lower" in result.columns
        assert "bb_width" in result.columns
        # Upper > middle > lower where defined
        valid_idx = result[["bb_upper", "bb_middle", "bb_lower"]].dropna().index
        assert (result.loc[valid_idx, "bb_upper"] >= result.loc[valid_idx, "bb_middle"]).all()
        assert (result.loc[valid_idx, "bb_middle"] >= result.loc[valid_idx, "bb_lower"]).all()

    def test_atr_computed(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "atr_14" in result.columns
        valid = result["atr_14"].dropna()
        assert (valid > 0).all()

    def test_momentum_features(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "momentum_7d" in result.columns
        assert "momentum_14d" in result.columns
        assert "momentum_28d" in result.columns

    def test_volume_features(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "volume_sma_20" in result.columns
        assert "volume_ratio" in result.columns

    def test_volatility_std(self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame):
        result = engineer.compute_technical_features(sample_daily_prices)
        assert "volatility_20d" in result.columns


class TestFundamentalFeatures:
    def test_fundamental_features(
        self, engineer: FeatureEngineer, sample_fundamentals: pd.DataFrame
    ):
        result = engineer.compute_fundamental_features(sample_fundamentals, "BTC")
        assert "market_cap_rank" in result
        assert "volume_rank" in result
        assert result["market_cap_rank"] == 1

    def test_fundamental_missing_asset(
        self, engineer: FeatureEngineer, sample_fundamentals: pd.DataFrame
    ):
        result = engineer.compute_fundamental_features(sample_fundamentals, "UNKNOWN")
        assert result["market_cap_rank"] == 0
        assert result["volume_rank"] == 0


class TestMarketContextFeatures:
    def test_market_context(
        self, engineer: FeatureEngineer, sample_market_context: pd.DataFrame
    ):
        result = engineer.compute_market_context_features(sample_market_context)
        assert "btc_dominance" in result.columns
        assert "fear_greed_index" in result.columns
        assert "total_market_cap_change_7d" in result.columns


class TestFullFeaturePipeline:
    def test_build_feature_matrix(
        self,
        engineer: FeatureEngineer,
        sample_daily_prices: pd.DataFrame,
        sample_fundamentals: pd.DataFrame,
        sample_market_context: pd.DataFrame,
    ):
        result = engineer.build_feature_matrix(
            daily_prices=sample_daily_prices,
            fundamentals=sample_fundamentals,
            market_context=sample_market_context,
            asset="BTC",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
        # Should have dropped rows with NaN from warmup
        assert not result.isnull().any().any()
        # Should have technical + fundamental + market features
        assert "sma_5" in result.columns
        assert "rsi_14" in result.columns
        assert "market_cap_rank" in result.columns
        assert "btc_dominance" in result.columns

    def test_build_targets(
        self, engineer: FeatureEngineer, sample_daily_prices: pd.DataFrame
    ):
        targets = engineer.build_targets(sample_daily_prices)
        assert "target_7d" in targets.columns
        assert "target_14d" in targets.columns
        assert "target_28d" in targets.columns
        assert "target_42d" in targets.columns
        # Targets are forward returns (pct change)
        # Last 42 rows should be NaN for target_42d
        assert targets["target_42d"].iloc[-1] != targets["target_42d"].iloc[-1]  # NaN check

    def test_feature_names_list(self, engineer: FeatureEngineer):
        names = engineer.feature_names()
        assert isinstance(names, list)
        assert len(names) > 20  # We expect ~30+ features
```

- [ ] **Step 4: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_features.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml'`

- [ ] **Step 5: Implement features.py**

Create `services/python/src/ml/features.py`:

```python
"""Feature engineering for crypto ML models.

Computes technical indicators, fundamental ranks, and market context features
from daily OHLCV data. All computations use only past data (no lookahead).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Computes all features for the ML pipeline."""

    # --- Technical Indicators ---

    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators from daily OHLCV data.

        Args:
            df: DataFrame with columns [date, asset, open, high, low, close, volume].
                Must be sorted by date ascending.

        Returns:
            Copy of df with technical feature columns appended.
        """
        out = df.copy()
        close = out["close"]
        high = out["high"]
        low = out["low"]
        volume = out["volume"]

        # Simple Moving Averages
        for period in (5, 10, 20, 50):
            out[f"sma_{period}"] = close.rolling(window=period).mean()

        # Exponential Moving Averages
        for period in (5, 10, 20, 50):
            out[f"ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # RSI (14-period)
        out["rsi_14"] = self._compute_rsi(close, period=14)

        # MACD (12, 26, 9)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        out["macd"] = ema_12 - ema_26
        out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()
        out["macd_histogram"] = out["macd"] - out["macd_signal"]

        # Bollinger Bands (20-period, 2 std)
        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        out["bb_upper"] = bb_middle + 2 * bb_std
        out["bb_middle"] = bb_middle
        out["bb_lower"] = bb_middle - 2 * bb_std
        out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]

        # ATR (14-period)
        out["atr_14"] = self._compute_atr(high, low, close, period=14)

        # Price Momentum (percentage returns over periods)
        for days, label in [(7, "7d"), (14, "14d"), (28, "28d")]:
            out[f"momentum_{label}"] = close.pct_change(periods=days)

        # Volume features
        out["volume_sma_20"] = volume.rolling(window=20).mean()
        out["volume_ratio"] = volume / out["volume_sma_20"]

        # Volatility (20-day rolling std of daily returns)
        daily_returns = close.pct_change()
        out["volatility_20d"] = daily_returns.rolling(window=20).std()

        # Price relative to moving averages (normalized)
        out["price_to_sma_20"] = close / out["sma_20"] - 1
        out["price_to_sma_50"] = close / out["sma_50"] - 1

        # Daily return
        out["daily_return"] = daily_returns

        return out

    def _compute_rsi(self, close: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index."""
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_atr(
        self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
    ) -> pd.Series:
        """Compute Average True Range."""
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    # --- Fundamental Features ---

    def compute_fundamental_features(
        self, fundamentals: pd.DataFrame, asset: str
    ) -> dict[str, float]:
        """Extract fundamental features for a specific asset.

        Args:
            fundamentals: DataFrame with columns [asset, market_cap_rank, total_volume_24h, market_cap].
            asset: The asset ticker to look up.

        Returns:
            Dict with fundamental feature values. Returns zeros if asset not found.
        """
        row = fundamentals[fundamentals["asset"] == asset]
        if row.empty:
            return {
                "market_cap_rank": 0.0,
                "volume_rank": 0.0,
                "market_cap_log": 0.0,
                "volume_24h_log": 0.0,
            }

        row = row.iloc[0]

        # Volume rank: rank by total_volume_24h descending (1 = highest)
        fundamentals_sorted = fundamentals.sort_values(
            "total_volume_24h", ascending=False
        ).reset_index(drop=True)
        volume_rank_idx = fundamentals_sorted[
            fundamentals_sorted["asset"] == asset
        ].index
        volume_rank = int(volume_rank_idx[0]) + 1 if len(volume_rank_idx) > 0 else 0

        market_cap_val = row.get("market_cap", 0) or 0
        volume_val = row.get("total_volume_24h", 0) or 0

        return {
            "market_cap_rank": float(row.get("market_cap_rank", 0) or 0),
            "volume_rank": float(volume_rank),
            "market_cap_log": float(np.log1p(market_cap_val)),
            "volume_24h_log": float(np.log1p(volume_val)),
        }

    # --- Market Context Features ---

    def compute_market_context_features(
        self, market_context: pd.DataFrame
    ) -> pd.DataFrame:
        """Compute market-level context features.

        Args:
            market_context: DataFrame with columns [date, btc_dominance, fear_greed_index, total_market_cap].

        Returns:
            DataFrame with market context feature columns.
        """
        out = market_context.copy()

        # 7-day change in total market cap
        out["total_market_cap_change_7d"] = out["total_market_cap"].pct_change(
            periods=7
        )

        # 7-day change in BTC dominance
        out["btc_dominance_change_7d"] = out["btc_dominance"].diff(periods=7)

        # Fear & greed moving average (smoothed)
        out["fear_greed_sma_7"] = out["fear_greed_index"].rolling(window=7).mean()

        return out

    # --- Target Variable ---

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute multi-horizon forward return targets.

        Args:
            df: DataFrame with a 'close' column, sorted by date.

        Returns:
            DataFrame with target columns (forward % returns). Future rows will be NaN.
        """
        out = df[["date", "close"]].copy()
        close = out["close"]

        # Forward returns: % change from current close to future close
        for days, label in [(7, "7d"), (14, "14d"), (28, "28d"), (42, "42d")]:
            out[f"target_{label}"] = close.shift(-days) / close - 1

        return out

    # --- Full Pipeline ---

    def build_feature_matrix(
        self,
        daily_prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        market_context: pd.DataFrame,
        asset: str,
    ) -> pd.DataFrame:
        """Build the complete feature matrix for one asset.

        Merges technical, fundamental, and market context features.
        Drops rows with NaN values from indicator warmup periods.

        Args:
            daily_prices: Daily OHLCV for this asset.
            fundamentals: Fundamental data for all assets.
            market_context: Market-level context data.
            asset: The asset ticker.

        Returns:
            Clean DataFrame with all features, no NaN values.
        """
        # 1. Technical features
        tech = self.compute_technical_features(daily_prices)

        # 2. Market context features
        ctx = self.compute_market_context_features(market_context)

        # 3. Merge on date
        merged = tech.merge(
            ctx[["date", "btc_dominance", "fear_greed_index",
                 "total_market_cap_change_7d", "btc_dominance_change_7d",
                 "fear_greed_sma_7"]],
            on="date",
            how="left",
        )

        # 4. Add fundamental features (static values broadcast to all rows)
        fund_feats = self.compute_fundamental_features(fundamentals, asset)
        for k, v in fund_feats.items():
            merged[k] = v

        # 5. Drop non-feature columns and rows with NaN
        drop_cols = ["date", "asset", "open", "high", "low", "close", "volume"]
        feature_cols = [c for c in merged.columns if c not in drop_cols]
        result = merged[feature_cols].copy()

        # Drop rows with NaN (warmup period for indicators)
        result = result.dropna().reset_index(drop=True)

        return result

    def feature_names(self) -> list[str]:
        """Return the list of all feature column names produced by build_feature_matrix.

        This is the canonical ordering used by all models.
        """
        return [
            # Moving averages
            "sma_5", "sma_10", "sma_20", "sma_50",
            "ema_5", "ema_10", "ema_20", "ema_50",
            # RSI
            "rsi_14",
            # MACD
            "macd", "macd_signal", "macd_histogram",
            # Bollinger Bands
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            # ATR
            "atr_14",
            # Momentum
            "momentum_7d", "momentum_14d", "momentum_28d",
            # Volume
            "volume_sma_20", "volume_ratio",
            # Volatility
            "volatility_20d",
            # Price relative
            "price_to_sma_20", "price_to_sma_50",
            # Daily return
            "daily_return",
            # Market context
            "btc_dominance", "fear_greed_index",
            "total_market_cap_change_7d", "btc_dominance_change_7d",
            "fear_greed_sma_7",
            # Fundamental
            "market_cap_rank", "volume_rank",
            "market_cap_log", "volume_24h_log",
        ]
```

- [ ] **Step 6: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_features.py -v
```

Expected: All tests PASSED

- [ ] **Step 7: Commit**

```bash
git checkout -b feat/task-1-feature-engineering
git add services/python/src/ml/__init__.py services/python/src/ml/features.py \
  services/python/tests/test_features.py services/python/pyproject.toml
git commit -m "feat: add feature engineering module with technical, fundamental, and market context indicators"
git push -u origin feat/task-1-feature-engineering
```

---

### Task 2: Base Model Interface

**Files:**
- Create: `services/python/src/ml/models/__init__.py`
- Create: `services/python/src/ml/models/base.py`
- Create: `services/python/tests/test_base_model.py`

- [ ] **Step 1: Write failing test for base model**

Create `services/python/tests/test_base_model.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.ml.models.base import BaseModel, ModelPrediction, HorizonPrediction


class TestModelPrediction:
    def test_prediction_dataclass(self):
        hp = HorizonPrediction(
            horizon_days=7,
            predicted_return=0.05,
            confidence=0.82,
        )
        assert hp.horizon_days == 7
        assert hp.predicted_return == 0.05
        assert hp.confidence == 0.82

    def test_model_prediction(self):
        horizons = [
            HorizonPrediction(horizon_days=7, predicted_return=0.05, confidence=0.8),
            HorizonPrediction(horizon_days=14, predicted_return=0.08, confidence=0.75),
        ]
        pred = ModelPrediction(
            model_name="xgboost",
            horizons=horizons,
        )
        assert pred.model_name == "xgboost"
        assert len(pred.horizons) == 2
        assert pred.best_horizon.horizon_days == 7  # highest confidence

    def test_model_prediction_best_horizon_by_confidence(self):
        horizons = [
            HorizonPrediction(horizon_days=7, predicted_return=0.02, confidence=0.6),
            HorizonPrediction(horizon_days=14, predicted_return=0.10, confidence=0.9),
            HorizonPrediction(horizon_days=28, predicted_return=0.15, confidence=0.7),
        ]
        pred = ModelPrediction(model_name="test", horizons=horizons)
        assert pred.best_horizon.horizon_days == 14


class TestBaseModelInterface:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseModel()  # type: ignore[abstract]

    def test_concrete_implementation(self):
        class DummyModel(BaseModel):
            @property
            def name(self) -> str:
                return "dummy"

            def train(
                self,
                X_train: pd.DataFrame,
                y_train: pd.DataFrame,
                X_val: pd.DataFrame | None = None,
                y_val: pd.DataFrame | None = None,
            ) -> dict[str, float]:
                return {"loss": 0.1}

            def predict(self, X: pd.DataFrame) -> ModelPrediction:
                horizons = [
                    HorizonPrediction(
                        horizon_days=7, predicted_return=0.05, confidence=0.8
                    )
                ]
                return ModelPrediction(model_name=self.name, horizons=horizons)

            def save(self, path: str) -> None:
                pass

            def load(self, path: str) -> None:
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_base_model.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.models'`

- [ ] **Step 3: Implement base model**

Create `services/python/src/ml/models/__init__.py`:

```python
```

Create `services/python/src/ml/models/base.py`:

```python
"""Base model interface for all ML models in the ensemble.

Every model must implement train, predict, save, and load.
Predictions are multi-horizon: each model predicts returns for 1w, 2w, 4w, 6w.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

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
            y_train: Training targets with columns [target_7d, target_14d, target_28d, target_42d].
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
        """Save model weights/state to disk.

        Args:
            path: Directory path to save model files into.
        """
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model weights/state from disk.

        Args:
            path: Directory path containing saved model files.
        """
        ...

    # --- Shared Utilities ---

    TARGET_COLUMNS: list[str] = ["target_7d", "target_14d", "target_28d", "target_42d"]
    HORIZON_DAYS: list[int] = [7, 14, 28, 42]

    def _return_to_confidence(self, predicted_return: float) -> float:
        """Convert a predicted return to a confidence score between 0 and 1.

        Uses a sigmoid-like mapping: higher absolute predicted returns map
        to higher confidence. The sign indicates direction (buy vs sell).

        Args:
            predicted_return: Raw predicted return (e.g., 0.05 = 5%).

        Returns:
            Confidence score between 0.0 and 1.0.
        """
        import numpy as np

        # Scale: 10% return maps to ~0.73 confidence, 20% to ~0.88
        scaled = abs(predicted_return) * 10
        confidence = float(1.0 / (1.0 + np.exp(-scaled + 1.0)))
        return confidence
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_base_model.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-2-base-model
git add services/python/src/ml/models/__init__.py services/python/src/ml/models/base.py \
  services/python/tests/test_base_model.py
git commit -m "feat: add base model interface with multi-horizon prediction dataclasses"
git push -u origin feat/task-2-base-model
```

---

### Task 3: XGBoost Model

**Files:**
- Create: `services/python/src/ml/models/xgboost_model.py`
- Create: `services/python/tests/test_xgboost_model.py`

- [ ] **Step 1: Write failing test for XGBoost model**

Create `services/python/tests/test_xgboost_model.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_xgboost_model.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.models.xgboost_model'`

- [ ] **Step 3: Implement XGBoost model**

Create `services/python/src/ml/models/xgboost_model.py`:

```python
"""XGBoost model for multi-horizon crypto return prediction.

Trains one XGBRegressor per target horizon (7d, 14d, 28d, 42d).
Uses early stopping on validation set when provided.
"""

from __future__ import annotations

import os
from pathlib import Path

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
        """Train one XGBoost regressor per target horizon.

        Returns dict with train/val RMSE for each horizon.
        """
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
                np.sqrt(np.mean((train_pred - y_train[target_col].values) ** 2))
            )
            metrics[f"train_rmse_{label}"] = train_rmse

            # Compute val RMSE if validation data provided
            if X_val is not None and y_val is not None:
                val_pred = reg.predict(X_val)
                val_rmse = float(
                    np.sqrt(np.mean((val_pred - y_val[target_col].values) ** 2))
                )
                metrics[f"val_rmse_{label}"] = val_rmse

        return metrics

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Predict returns for all horizons. Uses the last row for signal generation."""
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
            model_path = os.path.join(path, f"xgboost_{target_col}.joblib")
            self._models[target_col] = joblib.load(model_path)

    def feature_importance(self) -> dict[str, dict[str, float]]:
        """Return feature importances for each horizon model.

        Returns:
            Dict mapping horizon label to dict of {feature_name: importance}.
        """
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
            result[label] = dict(zip(feature_names, importance.tolist(), strict=True))

        return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_xgboost_model.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-3-xgboost-model
git add services/python/src/ml/models/xgboost_model.py \
  services/python/tests/test_xgboost_model.py
git commit -m "feat: add XGBoost multi-horizon regression model"
git push -u origin feat/task-3-xgboost-model
```

---

### Task 4: LightGBM Model

**Files:**
- Create: `services/python/src/ml/models/lightgbm_model.py`
- Create: `services/python/tests/test_lightgbm_model.py`

- [ ] **Step 1: Write failing test for LightGBM model**

Create `services/python/tests/test_lightgbm_model.py`:

```python
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.ml.models.base import ModelPrediction
from src.ml.models.lightgbm_model import LightGBMModel


@pytest.fixture
def training_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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
def model() -> LightGBMModel:
    return LightGBMModel(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
    )


class TestLightGBMModel:
    def test_name(self, model: LightGBMModel):
        assert model.name == "lightgbm"

    def test_train_returns_metrics(
        self,
        model: LightGBMModel,
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
        assert metrics["train_rmse_7d"] >= 0

    def test_predict_returns_model_prediction(
        self,
        model: LightGBMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)

        pred = model.predict(X.iloc[-5:])
        assert isinstance(pred, ModelPrediction)
        assert pred.model_name == "lightgbm"
        assert len(pred.horizons) == 4
        for h in pred.horizons:
            assert h.horizon_days in [7, 14, 28, 42]
            assert 0.0 <= h.confidence <= 1.0

    def test_predict_single_row(
        self,
        model: LightGBMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)

        pred = model.predict(X.iloc[[-1]])
        assert len(pred.horizons) == 4

    def test_save_and_load(
        self,
        model: LightGBMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)
        pred_before = model.predict(X.iloc[[-1]])

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)
            files = os.listdir(tmpdir)
            assert any("lightgbm" in f for f in files)

            model2 = LightGBMModel()
            model2.load(tmpdir)
            pred_after = model2.predict(X.iloc[[-1]])

        for h_before, h_after in zip(
            pred_before.horizons, pred_after.horizons, strict=True
        ):
            assert abs(h_before.predicted_return - h_after.predicted_return) < 1e-6

    def test_feature_importance(
        self,
        model: LightGBMModel,
        training_data: tuple[pd.DataFrame, pd.DataFrame],
    ):
        X, y = training_data
        model.train(X_train=X, y_train=y)
        importance = model.feature_importance()
        assert isinstance(importance, dict)
        assert "7d" in importance
        assert len(importance["7d"]) == X.shape[1]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_lightgbm_model.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.models.lightgbm_model'`

- [ ] **Step 3: Implement LightGBM model**

Create `services/python/src/ml/models/lightgbm_model.py`:

```python
"""LightGBM model for multi-horizon crypto return prediction.

Trains one LGBMRegressor per target horizon (7d, 14d, 28d, 42d).
Uses early stopping on validation set when provided.
"""

from __future__ import annotations

import os

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

            fit_params: dict = {}
            if X_val is not None and y_val is not None:
                fit_params["eval_set"] = [(X_val, y_val[target_col])]
                fit_params["eval_metric"] = "rmse"

            reg.fit(X_train, y_train[target_col], **fit_params)
            self._models[target_col] = reg

            # Train RMSE
            train_pred = reg.predict(X_train)
            train_rmse = float(
                np.sqrt(np.mean((train_pred - y_train[target_col].values) ** 2))
            )
            metrics[f"train_rmse_{label}"] = train_rmse

            # Val RMSE
            if X_val is not None and y_val is not None:
                val_pred = reg.predict(X_val)
                val_rmse = float(
                    np.sqrt(np.mean((val_pred - y_val[target_col].values) ** 2))
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
            model_path = os.path.join(path, f"lightgbm_{target_col}.joblib")
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
        joblib.dump(params, os.path.join(path, "lightgbm_params.joblib"))

    def load(self, path: str) -> None:
        """Load all horizon models from disk."""
        self._models = {}
        for target_col in self.TARGET_COLUMNS:
            model_path = os.path.join(path, f"lightgbm_{target_col}.joblib")
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
                zip(feature_names, importance.tolist(), strict=True)
            )

        return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_lightgbm_model.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-4-lightgbm-model
git add services/python/src/ml/models/lightgbm_model.py \
  services/python/tests/test_lightgbm_model.py
git commit -m "feat: add LightGBM multi-horizon regression model"
git push -u origin feat/task-4-lightgbm-model
```

---

### Task 5: LSTM Model (PyTorch)

**Files:**
- Create: `services/python/src/ml/models/lstm_model.py`
- Create: `services/python/tests/test_lstm_model.py`

- [ ] **Step 1: Write failing test for LSTM model**

Create `services/python/tests/test_lstm_model.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_lstm_model.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.models.lstm_model'`

- [ ] **Step 3: Implement LSTM model**

Create `services/python/src/ml/models/lstm_model.py`:

```python
"""LSTM model (PyTorch) for multi-horizon crypto return prediction.

Uses a sequence of N days of features to predict forward returns at 4 horizons.
Includes feature scaling (StandardScaler) as part of the pipeline.
"""

from __future__ import annotations

import os
from typing import Any

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
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use output of last timestep
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)


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
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
        self._n_features: int = 0
        self._device = torch.device("cpu")

    @property
    def name(self) -> str:
        return "lstm"

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Create sliding window sequences from time-series data.

        Args:
            X: Feature array of shape (n_samples, n_features).
            y: Target array of shape (n_samples, n_targets) or None.

        Returns:
            Tuple of (X_seq, y_seq) where X_seq has shape (n_seqs, seq_len, n_features)
            and y_seq has shape (n_seqs, n_targets).
        """
        n = len(X)
        seq_len = self._sequence_length
        if n < seq_len:
            raise ValueError(
                f"Need at least {seq_len} rows for sequences, got {n}."
            )

        X_seqs = []
        y_seqs = [] if y is not None else None

        for i in range(n - seq_len + 1):
            X_seqs.append(X[i : i + seq_len])
            if y is not None and y_seqs is not None:
                # Target is the value at the end of the sequence
                y_seqs.append(y[i + seq_len - 1])

        X_out = np.array(X_seqs, dtype=np.float32)
        y_out = np.array(y_seqs, dtype=np.float32) if y_seqs is not None else None
        return X_out, y_out

    def _fit_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and apply standard scaling."""
        values = X.values.astype(np.float32)
        self._scaler_mean = values.mean(axis=0)
        self._scaler_std = values.std(axis=0)
        # Prevent division by zero
        self._scaler_std[self._scaler_std == 0] = 1.0
        return (values - self._scaler_mean) / self._scaler_std

    def _apply_scaler(self, X: pd.DataFrame) -> np.ndarray:
        """Apply previously fit scaling."""
        if self._scaler_mean is None or self._scaler_std is None:
            raise RuntimeError("Scaler not fitted. Call train() first.")
        values = X.values.astype(np.float32)
        return (values - self._scaler_mean) / self._scaler_std

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
        val_loader: DataLoader | None = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self._apply_scaler(X_val)
            y_val_arr = y_val[self.TARGET_COLUMNS].values.astype(np.float32)
            X_val_seq, y_val_seq = self._create_sequences(X_val_scaled, y_val_arr)
            if y_val_seq is not None:
                val_dataset = TensorDataset(
                    torch.from_numpy(X_val_seq), torch.from_numpy(y_val_seq)
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

        for epoch in range(self._epochs):
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                predictions = self._network(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
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
                        k: v.clone() for k, v in self._network.state_dict().items()
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
            metrics["val_loss"] = best_val_loss if best_state else final_val_loss

        return metrics

    def predict(self, X: pd.DataFrame) -> ModelPrediction:
        """Predict returns for all horizons using the last sequence_length rows."""
        if self._network is None:
            raise RuntimeError("Model not trained. Call train() first.")

        if len(X) < self._sequence_length:
            raise ValueError(
                f"Need at least {self._sequence_length} rows for prediction, got {len(X)}."
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
        for i, (target_col, horizon_days) in enumerate(
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
            self._network.state_dict(), os.path.join(path, "lstm_weights.pt")
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_lstm_model.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-5-lstm-model
git add services/python/src/ml/models/lstm_model.py \
  services/python/tests/test_lstm_model.py
git commit -m "feat: add LSTM (PyTorch) multi-horizon regression model"
git push -u origin feat/task-5-lstm-model
```

---

### Task 6: Ensemble Consensus Logic

**Files:**
- Create: `services/python/src/ml/ensemble.py`
- Create: `services/python/tests/test_ensemble.py`

- [ ] **Step 1: Write failing test for ensemble**

Create `services/python/tests/test_ensemble.py`:

```python
import numpy as np
import pandas as pd
import pytest

from src.ml.ensemble import EnsembleConsensus
from src.ml.models.base import HorizonPrediction, ModelPrediction


@pytest.fixture
def ensemble() -> EnsembleConsensus:
    return EnsembleConsensus(
        model_weights={"xgboost": 0.4, "lightgbm": 0.4, "lstm": 0.2},
        confidence_threshold=0.7,
    )


def _make_prediction(
    model_name: str,
    returns: list[float],
    confidences: list[float],
) -> ModelPrediction:
    """Helper to build a ModelPrediction."""
    horizons = [
        HorizonPrediction(horizon_days=d, predicted_return=r, confidence=c)
        for d, r, c in zip([7, 14, 28, 42], returns, confidences, strict=True)
    ]
    return ModelPrediction(model_name=model_name, horizons=horizons)


class TestEnsembleConsensus:
    def test_weighted_average_returns(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)

        assert result is not None
        # Weighted average: 0.4*0.05 + 0.4*0.04 + 0.2*0.06 = 0.048
        expected_7d = 0.4 * 0.05 + 0.4 * 0.04 + 0.2 * 0.06
        assert abs(result.horizons[0].predicted_return - expected_7d) < 1e-6

    def test_confidence_weighted_average(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # Confidence is also weighted averaged
        expected_conf_7d = 0.4 * 0.8 + 0.4 * 0.75 + 0.2 * 0.85
        assert abs(result.horizons[0].confidence - expected_conf_7d) < 1e-6

    def test_below_confidence_threshold_returns_none(self):
        high_threshold = EnsembleConsensus(
            model_weights={"xgboost": 0.5, "lightgbm": 0.5},
            confidence_threshold=0.9,
        )
        predictions = [
            _make_prediction("xgboost", [0.01, 0.01, 0.01, 0.01], [0.5, 0.5, 0.5, 0.5]),
            _make_prediction("lightgbm", [0.01, 0.01, 0.01, 0.01], [0.5, 0.5, 0.5, 0.5]),
        ]
        result = high_threshold.combine(predictions)
        # All confidences below 0.9 => no horizon passes => None
        assert result is None

    def test_model_agreement_all_agree(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # All models predict positive => agreement = "3/3"
        assert result.model_agreement == "3/3"

    def test_model_agreement_partial(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.7, 0.6, 0.5]),
            _make_prediction("lightgbm", [-0.02, -0.01, 0.01, 0.02], [0.75, 0.72, 0.65, 0.55]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.85, 0.78, 0.7, 0.6]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # For 7d horizon: xgboost positive, lightgbm negative, lstm positive => 2/3
        assert result.model_agreement == "2/3"

    def test_best_horizon_selection(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.02, 0.08, 0.10, 0.05], [0.6, 0.85, 0.75, 0.5]),
            _make_prediction("lightgbm", [0.01, 0.07, 0.09, 0.04], [0.55, 0.82, 0.70, 0.45]),
            _make_prediction("lstm", [0.03, 0.09, 0.11, 0.06], [0.65, 0.88, 0.80, 0.55]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        # 14d horizon should have highest ensemble confidence
        assert result.best_horizon.horizon_days == 14

    def test_suggested_hold_days(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.02, 0.08, 0.10, 0.05], [0.6, 0.85, 0.75, 0.5]),
            _make_prediction("lightgbm", [0.01, 0.07, 0.09, 0.04], [0.55, 0.82, 0.70, 0.45]),
            _make_prediction("lstm", [0.03, 0.09, 0.11, 0.06], [0.65, 0.88, 0.80, 0.55]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        assert result.suggested_hold_days == 14

    def test_action_buy_for_positive_return(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [0.05, 0.08, 0.10, 0.12], [0.8, 0.75, 0.7, 0.6]),
            _make_prediction("lightgbm", [0.04, 0.07, 0.09, 0.11], [0.78, 0.73, 0.68, 0.58]),
            _make_prediction("lstm", [0.06, 0.09, 0.11, 0.13], [0.82, 0.77, 0.72, 0.62]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        assert result.action == "BUY"

    def test_action_sell_for_negative_return(self, ensemble: EnsembleConsensus):
        predictions = [
            _make_prediction("xgboost", [-0.05, -0.08, -0.10, -0.12], [0.8, 0.75, 0.7, 0.6]),
            _make_prediction("lightgbm", [-0.04, -0.07, -0.09, -0.11], [0.78, 0.73, 0.68, 0.58]),
            _make_prediction("lstm", [-0.06, -0.09, -0.11, -0.13], [0.82, 0.77, 0.72, 0.62]),
        ]
        result = ensemble.combine(predictions)
        assert result is not None
        assert result.action == "SELL"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_ensemble.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.ensemble'`

- [ ] **Step 3: Implement ensemble consensus logic**

Create `services/python/src/ml/ensemble.py`:

```python
"""Ensemble consensus logic for combining predictions from multiple models.

Combines XGBoost, LightGBM, and LSTM predictions via weighted average.
Applies confidence threshold and model agreement checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    """Combines predictions from multiple models using weighted averaging.

    Attributes:
        model_weights: Dict mapping model name to weight (must sum to 1.0).
        confidence_threshold: Minimum ensemble confidence to emit a signal (default 0.7).
    """

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

        # Validate weights sum to ~1.0
        total = sum(self._model_weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"Model weights must sum to 1.0, got {total}."
            )

    def combine(
        self, predictions: list[ModelPrediction]
    ) -> EnsembleResult | None:
        """Combine predictions from all models into an ensemble result.

        Args:
            predictions: List of ModelPrediction objects, one per model.

        Returns:
            EnsembleResult if at least one horizon passes the confidence threshold,
            otherwise None (no signal).
        """
        if not predictions:
            return None

        # Build weight lookup from available predictions
        weight_lookup: dict[str, float] = {}
        for pred in predictions:
            if pred.model_name in self._model_weights:
                weight_lookup[pred.model_name] = self._model_weights[pred.model_name]

        # Normalize weights to sum to 1.0 in case some models are missing
        total_weight = sum(weight_lookup.values())
        if total_weight == 0:
            return None
        normalized_weights = {
            k: v / total_weight for k, v in weight_lookup.items()
        }

        # Compute weighted average for each horizon
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

        # Find the best horizon (highest confidence that passes threshold)
        passing_horizons = [
            h for h in ensemble_horizons if h.confidence >= self._confidence_threshold
        ]

        if not passing_horizons:
            return None

        best = max(passing_horizons, key=lambda h: h.confidence)

        # Model agreement: how many models agree on direction for the best horizon
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

        # Action based on direction of best horizon
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
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_ensemble.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-6-ensemble-consensus
git add services/python/src/ml/ensemble.py services/python/tests/test_ensemble.py
git commit -m "feat: add ensemble consensus logic with weighted averaging and model agreement"
git push -u origin feat/task-6-ensemble-consensus
```

---

### Task 7: Signal Generation

**Files:**
- Create: `services/python/src/ml/signals.py`
- Create: `services/python/tests/test_signals.py`

- [ ] **Step 1: Write failing test for signal generation**

Create `services/python/tests/test_signals.py`:

```python
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.ml.ensemble import EnsembleConsensus, EnsembleResult
from src.ml.models.base import HorizonPrediction, ModelPrediction
from src.ml.signals import SignalGenerator, TradeSignal


@pytest.fixture
def generator() -> SignalGenerator:
    return SignalGenerator(stop_loss_pct=-8.0)


def _make_ensemble_result(
    action: str = "BUY",
    confidence: float = 0.84,
    hold_days: int = 14,
    expected_return_pct: float = 8.5,
    agreement: str = "3/3",
) -> EnsembleResult:
    horizons = [
        HorizonPrediction(horizon_days=7, predicted_return=0.05, confidence=0.75),
        HorizonPrediction(horizon_days=14, predicted_return=0.085, confidence=confidence),
        HorizonPrediction(horizon_days=28, predicted_return=0.10, confidence=0.70),
        HorizonPrediction(horizon_days=42, predicted_return=0.12, confidence=0.60),
    ]
    return EnsembleResult(
        horizons=horizons,
        model_agreement=agreement,
        action=action,
        suggested_hold_days=hold_days,
        expected_return_pct=expected_return_pct,
        confidence=confidence,
    )


class TestTradeSignal:
    def test_signal_dataclass(self):
        signal = TradeSignal(
            asset="ETH",
            action="BUY",
            confidence=0.84,
            suggested_hold_days=18,
            stop_loss_pct=-8.0,
            expected_return_pct=12.0,
            model_agreement="3/3",
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert signal.asset == "ETH"
        assert signal.action == "BUY"
        assert signal.confidence == 0.84

    def test_signal_to_dict(self):
        signal = TradeSignal(
            asset="BTC",
            action="SELL",
            confidence=0.75,
            suggested_hold_days=7,
            stop_loss_pct=-8.0,
            expected_return_pct=-5.5,
            model_agreement="2/3",
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        d = signal.to_dict()
        assert d["asset"] == "BTC"
        assert d["action"] == "SELL"
        assert "timestamp" in d


class TestSignalGenerator:
    def test_generate_buy_signal(self, generator: SignalGenerator):
        ensemble_result = _make_ensemble_result(action="BUY", confidence=0.84)
        signal = generator.generate(
            asset="ETH",
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert signal is not None
        assert signal.asset == "ETH"
        assert signal.action == "BUY"
        assert signal.confidence == 0.84
        assert signal.stop_loss_pct == -8.0
        assert signal.model_agreement == "3/3"

    def test_generate_sell_signal(self, generator: SignalGenerator):
        ensemble_result = _make_ensemble_result(
            action="SELL", confidence=0.78, expected_return_pct=-6.0
        )
        signal = generator.generate(
            asset="BTC",
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert signal is not None
        assert signal.action == "SELL"

    def test_none_ensemble_returns_none(self, generator: SignalGenerator):
        signal = generator.generate(
            asset="BTC",
            ensemble_result=None,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert signal is None

    def test_generate_for_multiple_assets(self, generator: SignalGenerator):
        assets = ["BTC", "ETH", "SOL"]
        results = {
            "BTC": _make_ensemble_result(action="BUY", confidence=0.85),
            "ETH": None,
            "SOL": _make_ensemble_result(action="SELL", confidence=0.72),
        }
        signals = generator.generate_batch(
            ensemble_results=results,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert len(signals) == 2  # ETH filtered out (None result)
        assert signals[0].asset == "BTC"
        assert signals[1].asset == "SOL"

    def test_evaluate_open_position_exit(self, generator: SignalGenerator):
        """When model confidence drops, signal EXIT."""
        ensemble_result = _make_ensemble_result(
            action="SELL", confidence=0.78, expected_return_pct=-5.0
        )
        signal = generator.evaluate_open_position(
            asset="ETH",
            entry_price=3500.0,
            current_price=3400.0,
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert signal is not None
        assert signal.action == "EXIT"

    def test_evaluate_open_position_hold(self, generator: SignalGenerator):
        """When model still confident, signal HOLD."""
        ensemble_result = _make_ensemble_result(
            action="BUY", confidence=0.80, expected_return_pct=8.0
        )
        signal = generator.evaluate_open_position(
            asset="ETH",
            entry_price=3500.0,
            current_price=3600.0,
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert signal is not None
        assert signal.action == "HOLD"

    def test_evaluate_stop_loss_triggered(self, generator: SignalGenerator):
        """When price drops below stop-loss, signal EXIT regardless of model."""
        ensemble_result = _make_ensemble_result(
            action="BUY", confidence=0.85, expected_return_pct=10.0
        )
        signal = generator.evaluate_open_position(
            asset="ETH",
            entry_price=3500.0,
            current_price=3200.0,  # -8.57% drop > -8% stop-loss
            ensemble_result=ensemble_result,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        assert signal is not None
        assert signal.action == "EXIT"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_signals.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.signals'`

- [ ] **Step 3: Implement signal generation**

Create `services/python/src/ml/signals.py`:

```python
"""Signal generation from ensemble predictions.

Converts ensemble results into actionable trade signals with
all required fields for the trading system.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime

from src.ml.ensemble import EnsembleResult


@dataclass
class TradeSignal:
    """A complete trade signal ready for the portfolio manager."""

    asset: str
    action: str  # BUY, SELL, HOLD, EXIT
    confidence: float
    suggested_hold_days: int
    stop_loss_pct: float
    expected_return_pct: float
    model_agreement: str
    timestamp: datetime

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


class SignalGenerator:
    """Generates trade signals from ensemble predictions.

    Handles new entry signals (BUY/SELL) and re-evaluation of
    open positions (HOLD/EXIT).
    """

    def __init__(self, stop_loss_pct: float = -8.0) -> None:
        """
        Args:
            stop_loss_pct: Hard stop-loss percentage (negative, e.g., -8.0 for -8%).
        """
        self._stop_loss_pct = stop_loss_pct

    def generate(
        self,
        asset: str,
        ensemble_result: EnsembleResult | None,
        timestamp: datetime,
    ) -> TradeSignal | None:
        """Generate a trade signal from an ensemble result.

        Args:
            asset: Asset ticker (e.g., "ETH").
            ensemble_result: Output from EnsembleConsensus.combine(), or None if no signal.
            timestamp: Signal generation timestamp.

        Returns:
            TradeSignal or None if no actionable signal.
        """
        if ensemble_result is None:
            return None

        return TradeSignal(
            asset=asset,
            action=ensemble_result.action,
            confidence=ensemble_result.confidence,
            suggested_hold_days=ensemble_result.suggested_hold_days,
            stop_loss_pct=self._stop_loss_pct,
            expected_return_pct=ensemble_result.expected_return_pct,
            model_agreement=ensemble_result.model_agreement,
            timestamp=timestamp,
        )

    def generate_batch(
        self,
        ensemble_results: dict[str, EnsembleResult | None],
        timestamp: datetime,
    ) -> list[TradeSignal]:
        """Generate signals for multiple assets.

        Args:
            ensemble_results: Dict mapping asset ticker to ensemble result (or None).
            timestamp: Signal generation timestamp.

        Returns:
            List of TradeSignal objects, excluding assets with no signal.
        """
        signals: list[TradeSignal] = []
        for asset, result in ensemble_results.items():
            signal = self.generate(asset, result, timestamp)
            if signal is not None:
                signals.append(signal)
        return signals

    def evaluate_open_position(
        self,
        asset: str,
        entry_price: float,
        current_price: float,
        ensemble_result: EnsembleResult | None,
        timestamp: datetime,
    ) -> TradeSignal:
        """Re-evaluate an open position and decide HOLD or EXIT.

        Exit triggers:
        1. Price dropped below stop-loss level.
        2. Model now predicts negative returns (direction reversal).

        Args:
            asset: Asset ticker.
            entry_price: Original entry price.
            current_price: Current market price.
            ensemble_result: Latest ensemble prediction, or None.
            timestamp: Evaluation timestamp.

        Returns:
            TradeSignal with action HOLD or EXIT.
        """
        # Check stop-loss first (hard rule)
        price_change_pct = ((current_price - entry_price) / entry_price) * 100
        if price_change_pct <= self._stop_loss_pct:
            return TradeSignal(
                asset=asset,
                action="EXIT",
                confidence=1.0,  # Certain exit
                suggested_hold_days=0,
                stop_loss_pct=self._stop_loss_pct,
                expected_return_pct=price_change_pct,
                model_agreement="N/A",
                timestamp=timestamp,
            )

        # If no ensemble result, default to HOLD
        if ensemble_result is None:
            return TradeSignal(
                asset=asset,
                action="HOLD",
                confidence=0.5,
                suggested_hold_days=7,
                stop_loss_pct=self._stop_loss_pct,
                expected_return_pct=0.0,
                model_agreement="N/A",
                timestamp=timestamp,
            )

        # If model now says SELL (direction reversal), signal EXIT
        if ensemble_result.action == "SELL":
            return TradeSignal(
                asset=asset,
                action="EXIT",
                confidence=ensemble_result.confidence,
                suggested_hold_days=0,
                stop_loss_pct=self._stop_loss_pct,
                expected_return_pct=price_change_pct,
                model_agreement=ensemble_result.model_agreement,
                timestamp=timestamp,
            )

        # Model still says BUY => HOLD
        return TradeSignal(
            asset=asset,
            action="HOLD",
            confidence=ensemble_result.confidence,
            suggested_hold_days=ensemble_result.suggested_hold_days,
            stop_loss_pct=self._stop_loss_pct,
            expected_return_pct=ensemble_result.expected_return_pct,
            model_agreement=ensemble_result.model_agreement,
            timestamp=timestamp,
        )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_signals.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-7-signal-generation
git add services/python/src/ml/signals.py services/python/tests/test_signals.py
git commit -m "feat: add signal generation with entry signals and open position re-evaluation"
git push -u origin feat/task-7-signal-generation
```

---

### Task 8: Risk Management

**Files:**
- Create: `services/python/src/ml/risk.py`
- Create: `services/python/tests/test_risk.py`

- [ ] **Step 1: Write failing test for risk management**

Create `services/python/tests/test_risk.py`:

```python
from datetime import datetime, timezone

import pytest

from src.ml.risk import RiskManager, RiskCheckResult, PortfolioState, OpenPosition
from src.ml.signals import TradeSignal


@pytest.fixture
def risk_manager() -> RiskManager:
    return RiskManager(
        max_positions=8,
        max_single_asset_pct=25.0,
        portfolio_drawdown_pause_pct=20.0,
        stop_loss_pct=-8.0,
        position_size_idr=1_000_000,
    )


def _make_signal(
    asset: str = "ETH",
    action: str = "BUY",
    confidence: float = 0.84,
) -> TradeSignal:
    return TradeSignal(
        asset=asset,
        action=action,
        confidence=confidence,
        suggested_hold_days=14,
        stop_loss_pct=-8.0,
        expected_return_pct=8.5,
        model_agreement="3/3",
        timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
    )


def _make_portfolio_state(
    positions: list[OpenPosition] | None = None,
    total_capital_idr: int = 10_000_000,
    peak_value_idr: int = 10_000_000,
    current_value_idr: int = 10_000_000,
) -> PortfolioState:
    return PortfolioState(
        positions=positions or [],
        total_capital_idr=total_capital_idr,
        peak_value_idr=peak_value_idr,
        current_value_idr=current_value_idr,
    )


class TestRiskManager:
    def test_allow_valid_signal(self, risk_manager: RiskManager):
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state()
        result = risk_manager.check(signal, state)
        assert result.allowed is True
        assert len(result.violations) == 0

    def test_reject_max_positions_exceeded(self, risk_manager: RiskManager):
        positions = [
            OpenPosition(asset=f"ASSET_{i}", entry_amount_idr=1_000_000, current_value_idr=1_000_000)
            for i in range(8)
        ]
        signal = _make_signal(asset="NEW_ASSET", action="BUY")
        state = _make_portfolio_state(positions=positions)
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert any("max positions" in v.lower() for v in result.violations)

    def test_reject_single_asset_exposure(self, risk_manager: RiskManager):
        """25% of 10M = 2.5M. Already 2M in ETH, adding 1M would be 3M = 30%."""
        positions = [
            OpenPosition(asset="ETH", entry_amount_idr=2_000_000, current_value_idr=2_000_000),
        ]
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state(
            positions=positions,
            total_capital_idr=10_000_000,
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert any("single asset" in v.lower() for v in result.violations)

    def test_allow_single_asset_within_limit(self, risk_manager: RiskManager):
        """25% of 10M = 2.5M. 1M in ETH + 1M new = 2M = 20%, OK."""
        positions = [
            OpenPosition(asset="ETH", entry_amount_idr=1_000_000, current_value_idr=1_000_000),
        ]
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state(
            positions=positions,
            total_capital_idr=10_000_000,
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is True

    def test_reject_portfolio_drawdown_pause(self, risk_manager: RiskManager):
        """20% drawdown from peak => pause new signals."""
        signal = _make_signal(asset="ETH", action="BUY")
        state = _make_portfolio_state(
            peak_value_idr=10_000_000,
            current_value_idr=7_900_000,  # -21% drawdown
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert any("drawdown" in v.lower() for v in result.violations)

    def test_allow_sell_during_drawdown(self, risk_manager: RiskManager):
        """SELL/EXIT signals should still be allowed during drawdown pause."""
        signal = _make_signal(asset="ETH", action="SELL")
        state = _make_portfolio_state(
            peak_value_idr=10_000_000,
            current_value_idr=7_900_000,  # -21% drawdown
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is True

    def test_multiple_violations(self, risk_manager: RiskManager):
        positions = [
            OpenPosition(asset=f"ASSET_{i}", entry_amount_idr=1_000_000, current_value_idr=1_000_000)
            for i in range(8)
        ]
        signal = _make_signal(asset="ASSET_0", action="BUY")
        state = _make_portfolio_state(
            positions=positions,
            total_capital_idr=10_000_000,
            peak_value_idr=10_000_000,
            current_value_idr=7_500_000,  # -25% drawdown
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is False
        assert len(result.violations) >= 2  # max positions + drawdown

    def test_allow_exit_signals_always(self, risk_manager: RiskManager):
        """EXIT signals bypass all risk checks."""
        positions = [
            OpenPosition(asset=f"ASSET_{i}", entry_amount_idr=1_000_000, current_value_idr=1_000_000)
            for i in range(8)
        ]
        signal = _make_signal(asset="ASSET_0", action="EXIT")
        state = _make_portfolio_state(
            positions=positions,
            peak_value_idr=10_000_000,
            current_value_idr=7_000_000,
        )
        result = risk_manager.check(signal, state)
        assert result.allowed is True

    def test_compute_stop_loss_price(self, risk_manager: RiskManager):
        price = risk_manager.compute_stop_loss_price(entry_price=100.0)
        assert price == 92.0  # 100 * (1 + (-8/100)) = 92

    def test_alert_level_green(self, risk_manager: RiskManager):
        level = risk_manager.get_alert_level(
            entry_price=100.0, current_price=105.0, confidence=0.80
        )
        assert level == "GREEN"

    def test_alert_level_yellow(self, risk_manager: RiskManager):
        level = risk_manager.get_alert_level(
            entry_price=100.0, current_price=97.0, confidence=0.55
        )
        assert level == "YELLOW"

    def test_alert_level_red(self, risk_manager: RiskManager):
        level = risk_manager.get_alert_level(
            entry_price=100.0, current_price=91.0, confidence=0.80
        )
        assert level == "RED"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_risk.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.risk'`

- [ ] **Step 3: Implement risk management**

Create `services/python/src/ml/risk.py`:

```python
"""Risk management checks for the trading system.

Validates signals against portfolio-level rules:
- Max concurrent positions (8)
- Max single asset exposure (25%)
- Portfolio drawdown pause (20%)
- Stop-loss calculations (-8%)
- Alert level assessment (GREEN/YELLOW/RED)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.ml.signals import TradeSignal


@dataclass
class OpenPosition:
    """Represents a currently open position in the portfolio."""

    asset: str
    entry_amount_idr: int
    current_value_idr: int


@dataclass
class PortfolioState:
    """Current state of the portfolio for risk checking."""

    positions: list[OpenPosition] = field(default_factory=list)
    total_capital_idr: int = 10_000_000
    peak_value_idr: int = 10_000_000
    current_value_idr: int = 10_000_000


@dataclass
class RiskCheckResult:
    """Result of risk management validation."""

    allowed: bool
    violations: list[str] = field(default_factory=list)


class RiskManager:
    """Validates trade signals against portfolio risk rules.

    Attributes:
        max_positions: Maximum number of concurrent open positions.
        max_single_asset_pct: Maximum % of total capital in a single asset.
        portfolio_drawdown_pause_pct: Pause new entries if portfolio drops this % from peak.
        stop_loss_pct: Hard stop-loss percentage (negative).
        position_size_idr: Fixed position size per trade in IDR.
    """

    def __init__(
        self,
        max_positions: int = 8,
        max_single_asset_pct: float = 25.0,
        portfolio_drawdown_pause_pct: float = 20.0,
        stop_loss_pct: float = -8.0,
        position_size_idr: int = 1_000_000,
    ) -> None:
        self._max_positions = max_positions
        self._max_single_asset_pct = max_single_asset_pct
        self._portfolio_drawdown_pause_pct = portfolio_drawdown_pause_pct
        self._stop_loss_pct = stop_loss_pct
        self._position_size_idr = position_size_idr

    def check(self, signal: TradeSignal, state: PortfolioState) -> RiskCheckResult:
        """Validate a trade signal against all risk rules.

        EXIT and SELL signals always pass -- we never block exits.
        BUY signals are checked against all portfolio-level rules.

        Args:
            signal: The trade signal to validate.
            state: Current portfolio state.

        Returns:
            RiskCheckResult indicating whether the signal is allowed.
        """
        # EXIT and SELL signals always pass
        if signal.action in ("EXIT", "SELL", "HOLD"):
            return RiskCheckResult(allowed=True, violations=[])

        # Only BUY signals need risk checking
        violations: list[str] = []

        # Check 1: Max concurrent positions
        if len(state.positions) >= self._max_positions:
            violations.append(
                f"Max positions ({self._max_positions}) reached. "
                f"Currently have {len(state.positions)} open positions."
            )

        # Check 2: Max single asset exposure
        current_asset_exposure = sum(
            p.current_value_idr
            for p in state.positions
            if p.asset == signal.asset
        )
        new_exposure = current_asset_exposure + self._position_size_idr
        max_allowed = state.total_capital_idr * (self._max_single_asset_pct / 100)
        if new_exposure > max_allowed:
            violations.append(
                f"Single asset exposure for {signal.asset} would be "
                f"{new_exposure:,} IDR ({new_exposure / state.total_capital_idr * 100:.1f}%), "
                f"exceeding max {self._max_single_asset_pct}% = {max_allowed:,.0f} IDR."
            )

        # Check 3: Portfolio drawdown pause
        if state.peak_value_idr > 0:
            drawdown_pct = (
                (state.peak_value_idr - state.current_value_idr)
                / state.peak_value_idr
                * 100
            )
            if drawdown_pct >= self._portfolio_drawdown_pause_pct:
                violations.append(
                    f"Portfolio drawdown is {drawdown_pct:.1f}%, "
                    f"exceeding pause threshold of {self._portfolio_drawdown_pause_pct}%. "
                    f"New entries paused until recovery."
                )

        allowed = len(violations) == 0
        return RiskCheckResult(allowed=allowed, violations=violations)

    def compute_stop_loss_price(self, entry_price: float) -> float:
        """Calculate the stop-loss price for an entry.

        Args:
            entry_price: The entry price of the position.

        Returns:
            The price at which to trigger stop-loss exit.
        """
        return entry_price * (1 + self._stop_loss_pct / 100)

    def get_alert_level(
        self,
        entry_price: float,
        current_price: float,
        confidence: float,
    ) -> str:
        """Determine the alert level for an open position.

        Alert levels:
        - RED: Stop-loss hit or very close to it (within 1%).
        - YELLOW: Confidence dropping (<0.6) or moderate loss (>5%).
        - GREEN: On track.

        Args:
            entry_price: Original entry price.
            current_price: Current market price.
            confidence: Latest model confidence for this position.

        Returns:
            "RED", "YELLOW", or "GREEN".
        """
        price_change_pct = ((current_price - entry_price) / entry_price) * 100

        # RED: stop-loss hit or within 1% of it
        stop_loss_threshold = self._stop_loss_pct + 1.0  # e.g., -7% for -8% stop-loss
        if price_change_pct <= stop_loss_threshold:
            return "RED"

        # YELLOW: confidence dropping or moderate loss
        if confidence < 0.6 or price_change_pct < -5.0:
            return "YELLOW"

        # GREEN: on track
        return "GREEN"
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_risk.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-8-risk-management
git add services/python/src/ml/risk.py services/python/tests/test_risk.py
git commit -m "feat: add risk management with position limits, drawdown pause, and alert levels"
git push -u origin feat/task-8-risk-management
```

---

### Task 9: Backtesting Engine

**Files:**
- Create: `services/python/src/ml/backtest.py`
- Create: `services/python/tests/test_backtest.py`

- [ ] **Step 1: Write failing test for backtesting engine**

Create `services/python/tests/test_backtest.py`:

```python
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest import BacktestEngine, BacktestConfig, BacktestResult, BacktestTrade


@pytest.fixture
def config() -> BacktestConfig:
    return BacktestConfig(
        position_size_idr=1_000_000,
        stop_loss_pct=-8.0,
        max_positions=8,
        max_single_asset_pct=25.0,
        confidence_threshold=0.7,
    )


@pytest.fixture
def price_data() -> pd.DataFrame:
    """Generate 2 years of synthetic daily price data for multiple assets."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", "2024-12-31", freq="D")
    assets = ["BTC", "ETH", "SOL"]
    rows = []
    for asset in assets:
        base = {"BTC": 30000, "ETH": 2000, "SOL": 25}[asset]
        prices = base * np.cumprod(1 + np.random.normal(0.0005, 0.02, len(dates)))
        for i, date in enumerate(dates):
            rows.append({
                "date": date,
                "asset": asset,
                "open": prices[i] * 0.999,
                "high": prices[i] * 1.01,
                "low": prices[i] * 0.99,
                "close": prices[i],
                "volume": np.random.uniform(1000, 10000),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def signals_data() -> pd.DataFrame:
    """Generate synthetic signals over the price data period."""
    np.random.seed(123)
    rows = []
    dates = pd.date_range("2023-03-01", "2024-12-01", freq="14D")
    assets = ["BTC", "ETH", "SOL"]
    for date in dates:
        asset = np.random.choice(assets)
        action = np.random.choice(["BUY", "SELL"], p=[0.7, 0.3])
        rows.append({
            "date": date,
            "asset": asset,
            "action": action,
            "confidence": np.random.uniform(0.65, 0.95),
            "suggested_hold_days": np.random.choice([7, 14, 28, 42]),
            "stop_loss_pct": -8.0,
            "expected_return_pct": np.random.uniform(-5, 15),
            "model_agreement": np.random.choice(["2/3", "3/3"]),
        })
    return pd.DataFrame(rows)


class TestBacktestConfig:
    def test_config_defaults(self):
        config = BacktestConfig()
        assert config.position_size_idr == 1_000_000
        assert config.stop_loss_pct == -8.0
        assert config.max_positions == 8

    def test_config_custom(self, config: BacktestConfig):
        assert config.max_single_asset_pct == 25.0


class TestBacktestEngine:
    def test_run_returns_result(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        assert isinstance(result, BacktestResult)

    def test_result_has_all_metrics(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)

        assert hasattr(result, "total_return_idr")
        assert hasattr(result, "total_return_pct")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "reward_risk_ratio")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "max_drawdown_pct")
        assert hasattr(result, "total_trades")
        assert hasattr(result, "trades")
        assert hasattr(result, "monthly_breakdown")
        assert hasattr(result, "per_asset_breakdown")

    def test_trades_have_no_lookahead(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        """Verify that entry dates are always on or after signal dates."""
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)

        for trade in result.trades:
            # Entry should not happen before signal date
            assert trade.entry_date >= trade.signal_date

    def test_stop_loss_respected(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        """No trade should have a loss worse than stop-loss + slippage tolerance."""
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)

        slippage_tolerance = 2.0  # Allow 2% slippage
        for trade in result.trades:
            if trade.pnl_pct < 0:
                assert trade.pnl_pct >= config.stop_loss_pct - slippage_tolerance, (
                    f"Trade {trade.asset} on {trade.entry_date} lost {trade.pnl_pct:.1f}%, "
                    f"exceeding stop-loss of {config.stop_loss_pct}%"
                )

    def test_max_positions_respected(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        # Check max concurrent positions across all dates
        assert result.max_concurrent_positions <= config.max_positions

    def test_position_size_fixed(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        """Every trade should use the fixed position size."""
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)

        for trade in result.trades:
            assert trade.entry_amount_idr == config.position_size_idr

    def test_win_rate_bounded(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)
        if result.total_trades > 0:
            assert 0.0 <= result.win_rate <= 1.0

    def test_monthly_breakdown_format(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)

        for entry in result.monthly_breakdown:
            assert "month" in entry
            assert "pnl_idr" in entry
            assert "trades" in entry

    def test_per_asset_breakdown_format(
        self,
        config: BacktestConfig,
        price_data: pd.DataFrame,
        signals_data: pd.DataFrame,
    ):
        engine = BacktestEngine(config)
        result = engine.run(price_data, signals_data)

        for entry in result.per_asset_breakdown:
            assert "asset" in entry
            assert "total_pnl_idr" in entry
            assert "win_rate" in entry
            assert "trades" in entry

    def test_empty_signals(self, config: BacktestConfig, price_data: pd.DataFrame):
        engine = BacktestEngine(config)
        empty_signals = pd.DataFrame(columns=[
            "date", "asset", "action", "confidence", "suggested_hold_days",
            "stop_loss_pct", "expected_return_pct", "model_agreement",
        ])
        result = engine.run(price_data, empty_signals)
        assert result.total_trades == 0
        assert result.total_return_idr == 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_backtest.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.backtest'`

- [ ] **Step 3: Implement backtesting engine**

Create `services/python/src/ml/backtest.py`:

```python
"""Backtesting engine for validating the trading system against historical data.

Simulates trades exactly as the live system would: signals, entries, exits,
stop-losses, and position limits. Uses no lookahead bias -- the engine only
sees data available at the time of each decision.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    position_size_idr: int = 1_000_000
    stop_loss_pct: float = -8.0
    max_positions: int = 8
    max_single_asset_pct: float = 25.0
    confidence_threshold: float = 0.7
    total_capital_idr: int = 10_000_000


@dataclass
class BacktestTrade:
    """Record of a single completed trade in the backtest."""

    asset: str
    signal_date: datetime
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    entry_amount_idr: int
    exit_amount_idr: int
    pnl_idr: int
    pnl_pct: float
    hold_days: int
    exit_reason: str  # "target_hold", "stop_loss", "signal_exit"


@dataclass
class BacktestResult:
    """Complete results of a backtest run."""

    total_return_idr: int
    total_return_pct: float
    win_rate: float
    reward_risk_ratio: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    max_concurrent_positions: int
    trades: list[BacktestTrade]
    monthly_breakdown: list[dict[str, Any]]
    per_asset_breakdown: list[dict[str, Any]]


@dataclass
class _OpenPosition:
    """Internal tracking of an open position during backtesting."""

    asset: str
    entry_date: datetime
    entry_price: float
    entry_amount_idr: int
    quantity: float
    stop_loss_price: float
    target_exit_date: datetime
    signal_date: datetime


class BacktestEngine:
    """Simulates the trading system against historical data.

    The engine processes signals chronologically, enters positions on the
    day after the signal (to avoid lookahead), checks stop-losses daily,
    and exits positions at their target hold period or when stop-loss triggers.
    """

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """Run the backtest simulation.

        Args:
            price_data: DataFrame with columns [date, asset, open, high, low, close, volume].
            signals: DataFrame with columns [date, asset, action, confidence,
                     suggested_hold_days, stop_loss_pct, expected_return_pct, model_agreement].

        Returns:
            BacktestResult with all metrics and trade records.
        """
        if signals.empty:
            return BacktestResult(
                total_return_idr=0,
                total_return_pct=0.0,
                win_rate=0.0,
                reward_risk_ratio=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                max_concurrent_positions=0,
                trades=[],
                monthly_breakdown=[],
                per_asset_breakdown=[],
            )

        # Build price lookup: {(date, asset): row}
        price_data = price_data.copy()
        price_data["date"] = pd.to_datetime(price_data["date"])
        signals = signals.copy()
        signals["date"] = pd.to_datetime(signals["date"])

        # Build per-asset sorted price DataFrames for efficient lookup
        asset_prices: dict[str, pd.DataFrame] = {}
        for asset in price_data["asset"].unique():
            asset_df = (
                price_data[price_data["asset"] == asset]
                .sort_values("date")
                .set_index("date")
            )
            asset_prices[asset] = asset_df

        # Get all unique trading dates
        all_dates = sorted(price_data["date"].unique())

        # State
        open_positions: list[_OpenPosition] = []
        completed_trades: list[BacktestTrade] = []
        max_concurrent = 0
        portfolio_value_history: list[float] = []

        # Process each date
        signals_by_date: dict[Any, pd.DataFrame] = {}
        for date, group in signals.groupby("date"):
            signals_by_date[date] = group

        for date in all_dates:
            # 1. Check stop-losses and target exits on open positions
            positions_to_close: list[tuple[int, str]] = []  # (index, reason)
            for i, pos in enumerate(open_positions):
                if pos.asset not in asset_prices:
                    continue
                if date not in asset_prices[pos.asset].index:
                    continue

                current_low = asset_prices[pos.asset].loc[date, "low"]
                current_close = asset_prices[pos.asset].loc[date, "close"]

                # Handle potential duplicate dates (take first)
                if isinstance(current_low, pd.Series):
                    current_low = current_low.iloc[0]
                if isinstance(current_close, pd.Series):
                    current_close = current_close.iloc[0]

                # Stop-loss check (use low price for intraday trigger)
                if current_low <= pos.stop_loss_price:
                    positions_to_close.append((i, "stop_loss"))
                    continue

                # Target hold period exit
                if date >= pos.target_exit_date:
                    positions_to_close.append((i, "target_hold"))
                    continue

            # Close positions (reverse order to maintain indices)
            for idx, reason in sorted(positions_to_close, reverse=True):
                pos = open_positions.pop(idx)
                if pos.asset in asset_prices and date in asset_prices[pos.asset].index:
                    exit_price_raw = asset_prices[pos.asset].loc[date, "close"]
                    if isinstance(exit_price_raw, pd.Series):
                        exit_price_raw = exit_price_raw.iloc[0]

                    if reason == "stop_loss":
                        exit_price = float(pos.stop_loss_price)
                    else:
                        exit_price = float(exit_price_raw)

                    exit_amount = int(pos.quantity * exit_price)
                    pnl_idr = exit_amount - pos.entry_amount_idr
                    pnl_pct = ((exit_price - pos.entry_price) / pos.entry_price) * 100

                    completed_trades.append(
                        BacktestTrade(
                            asset=pos.asset,
                            signal_date=pos.signal_date,
                            entry_date=pos.entry_date,
                            exit_date=date,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            entry_amount_idr=pos.entry_amount_idr,
                            exit_amount_idr=exit_amount,
                            pnl_idr=pnl_idr,
                            pnl_pct=pnl_pct,
                            hold_days=(date - pos.entry_date).days,
                            exit_reason=reason,
                        )
                    )

            # 2. Process new signals for this date (entry on NEXT available date)
            if date in signals_by_date:
                for _, sig in signals_by_date[date].iterrows():
                    if sig["action"] != "BUY":
                        continue
                    if sig["confidence"] < self._config.confidence_threshold:
                        continue

                    # Position limit check
                    if len(open_positions) >= self._config.max_positions:
                        continue

                    # Single asset exposure check
                    asset_exposure = sum(
                        p.entry_amount_idr
                        for p in open_positions
                        if p.asset == sig["asset"]
                    )
                    max_allowed = self._config.total_capital_idr * (
                        self._config.max_single_asset_pct / 100
                    )
                    if (
                        asset_exposure + self._config.position_size_idr
                        > max_allowed
                    ):
                        continue

                    # Find next trading day for entry (avoid lookahead)
                    asset = sig["asset"]
                    if asset not in asset_prices:
                        continue

                    future_dates = asset_prices[asset].index[
                        asset_prices[asset].index > date
                    ]
                    if len(future_dates) == 0:
                        continue

                    entry_date = future_dates[0]
                    entry_price_raw = asset_prices[asset].loc[entry_date, "open"]
                    if isinstance(entry_price_raw, pd.Series):
                        entry_price_raw = entry_price_raw.iloc[0]
                    entry_price = float(entry_price_raw)

                    quantity = self._config.position_size_idr / entry_price
                    stop_loss_price = entry_price * (
                        1 + self._config.stop_loss_pct / 100
                    )
                    hold_days = int(sig.get("suggested_hold_days", 14))
                    target_exit = entry_date + pd.Timedelta(days=hold_days)

                    open_positions.append(
                        _OpenPosition(
                            asset=asset,
                            entry_date=entry_date,
                            entry_price=entry_price,
                            entry_amount_idr=self._config.position_size_idr,
                            quantity=quantity,
                            stop_loss_price=stop_loss_price,
                            target_exit_date=target_exit,
                            signal_date=date,
                        )
                    )

            # Track max concurrent positions
            max_concurrent = max(max_concurrent, len(open_positions))

            # Track portfolio value for drawdown calculation
            total_value = float(self._config.total_capital_idr)
            for pos in open_positions:
                if pos.asset in asset_prices and date in asset_prices[pos.asset].index:
                    cp = asset_prices[pos.asset].loc[date, "close"]
                    if isinstance(cp, pd.Series):
                        cp = cp.iloc[0]
                    total_value += pos.quantity * float(cp) - pos.entry_amount_idr
            portfolio_value_history.append(total_value)

        # Force-close any remaining open positions at last available price
        last_date = all_dates[-1]
        for pos in open_positions:
            if pos.asset in asset_prices:
                dates_available = asset_prices[pos.asset].index
                if len(dates_available) > 0:
                    last_avail = dates_available[-1]
                    exit_price_raw = asset_prices[pos.asset].loc[last_avail, "close"]
                    if isinstance(exit_price_raw, pd.Series):
                        exit_price_raw = exit_price_raw.iloc[0]
                    exit_price = float(exit_price_raw)
                    exit_amount = int(pos.quantity * exit_price)
                    pnl_idr = exit_amount - pos.entry_amount_idr
                    pnl_pct = (
                        (exit_price - pos.entry_price) / pos.entry_price
                    ) * 100
                    completed_trades.append(
                        BacktestTrade(
                            asset=pos.asset,
                            signal_date=pos.signal_date,
                            entry_date=pos.entry_date,
                            exit_date=last_avail,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            entry_amount_idr=pos.entry_amount_idr,
                            exit_amount_idr=exit_amount,
                            pnl_idr=pnl_idr,
                            pnl_pct=pnl_pct,
                            hold_days=(last_avail - pos.entry_date).days,
                            exit_reason="backtest_end",
                        )
                    )

        # Compute metrics
        return self._compute_metrics(
            completed_trades, max_concurrent, portfolio_value_history
        )

    def _compute_metrics(
        self,
        trades: list[BacktestTrade],
        max_concurrent: int,
        portfolio_value_history: list[float],
    ) -> BacktestResult:
        """Compute all backtest performance metrics."""
        total_trades = len(trades)

        if total_trades == 0:
            return BacktestResult(
                total_return_idr=0,
                total_return_pct=0.0,
                win_rate=0.0,
                reward_risk_ratio=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                max_concurrent_positions=max_concurrent,
                trades=[],
                monthly_breakdown=[],
                per_asset_breakdown=[],
            )

        # Total return
        total_pnl = sum(t.pnl_idr for t in trades)
        total_invested = sum(t.entry_amount_idr for t in trades)
        total_return_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0.0

        # Win rate
        winning_trades = [t for t in trades if t.pnl_idr > 0]
        win_rate = len(winning_trades) / total_trades

        # Reward/Risk ratio
        avg_win = (
            np.mean([t.pnl_idr for t in winning_trades]) if winning_trades else 0.0
        )
        losing_trades = [t for t in trades if t.pnl_idr < 0]
        avg_loss = (
            abs(np.mean([t.pnl_idr for t in losing_trades])) if losing_trades else 1.0
        )
        reward_risk_ratio = float(avg_win / avg_loss) if avg_loss > 0 else 0.0

        # Sharpe ratio (annualized, assuming ~252 trading days)
        daily_returns = self._compute_daily_returns(portfolio_value_history)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = float(
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0.0

        # Max drawdown
        max_drawdown_pct = self._compute_max_drawdown(portfolio_value_history)

        # Monthly breakdown
        monthly_breakdown = self._compute_monthly_breakdown(trades)

        # Per-asset breakdown
        per_asset_breakdown = self._compute_per_asset_breakdown(trades)

        return BacktestResult(
            total_return_idr=total_pnl,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            reward_risk_ratio=reward_risk_ratio,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
            max_concurrent_positions=max_concurrent,
            trades=trades,
            monthly_breakdown=monthly_breakdown,
            per_asset_breakdown=per_asset_breakdown,
        )

    def _compute_daily_returns(
        self, portfolio_values: list[float]
    ) -> list[float]:
        """Compute daily returns from portfolio value history."""
        if len(portfolio_values) < 2:
            return []
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        return returns.tolist()

    def _compute_max_drawdown(self, portfolio_values: list[float]) -> float:
        """Compute maximum drawdown percentage."""
        if not portfolio_values:
            return 0.0

        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak * 100
        return float(np.min(drawdowns))

    def _compute_monthly_breakdown(
        self, trades: list[BacktestTrade]
    ) -> list[dict[str, Any]]:
        """Group trades by exit month and compute P&L."""
        monthly: dict[str, dict[str, Any]] = {}

        for trade in trades:
            month_key = trade.exit_date.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = {"month": month_key, "pnl_idr": 0, "trades": 0}
            monthly[month_key]["pnl_idr"] += trade.pnl_idr
            monthly[month_key]["trades"] += 1

        return sorted(monthly.values(), key=lambda x: x["month"])

    def _compute_per_asset_breakdown(
        self, trades: list[BacktestTrade]
    ) -> list[dict[str, Any]]:
        """Group trades by asset and compute metrics."""
        by_asset: dict[str, list[BacktestTrade]] = {}
        for trade in trades:
            by_asset.setdefault(trade.asset, []).append(trade)

        result: list[dict[str, Any]] = []
        for asset, asset_trades in sorted(by_asset.items()):
            total_pnl = sum(t.pnl_idr for t in asset_trades)
            wins = sum(1 for t in asset_trades if t.pnl_idr > 0)
            win_rate = wins / len(asset_trades) if asset_trades else 0.0

            result.append({
                "asset": asset,
                "total_pnl_idr": total_pnl,
                "win_rate": win_rate,
                "trades": len(asset_trades),
                "avg_hold_days": np.mean([t.hold_days for t in asset_trades]),
            })

        return result
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_backtest.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-9-backtesting-engine
git add services/python/src/ml/backtest.py services/python/tests/test_backtest.py
git commit -m "feat: add backtesting engine with full trade simulation and performance metrics"
git push -u origin feat/task-9-backtesting-engine
```

---

### Task 10: Model Versioning and Weekly Retraining

**Files:**
- Create: `services/python/src/ml/versioning.py`
- Create: `services/python/tests/test_versioning.py`

- [ ] **Step 1: Write failing test for model versioning**

Create `services/python/tests/test_versioning.py`:

```python
import os
import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.ml.versioning import ModelVersionManager, ModelVersion


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
            created_at=datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
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
            timestamp=datetime(2026, 3, 30, 12, 0, tzinfo=timezone.utc),
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
            timestamp=datetime(2026, 3, 23, tzinfo=timezone.utc),
        )
        manager.save_version(
            model=model, metrics={"rmse": 0.025},
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )

        versions = manager.list_versions("xgboost")
        assert len(versions) == 2
        # Most recent first
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
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
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
        ts = datetime(2026, 3, 30, 14, 30, 0, tzinfo=timezone.utc)
        vid = manager._generate_version_id(ts)
        assert vid == "v_20260330_143000"

    def test_retrain_decision_weekly(self, manager: ModelVersionManager):
        """Should retrain if last version is older than 7 days."""
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

        # Saved 10 days ago
        manager.save_version(
            model=model, metrics={"rmse": 0.03},
            timestamp=datetime(2026, 3, 20, tzinfo=timezone.utc),
        )

        # Check if retrain needed (now = March 30)
        needs_retrain = manager.needs_retraining(
            model_name="xgboost",
            current_time=datetime(2026, 3, 30, tzinfo=timezone.utc),
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

        # Saved 2 days ago
        manager.save_version(
            model=model, metrics={"rmse": 0.03},
            timestamp=datetime(2026, 3, 28, tzinfo=timezone.utc),
        )

        needs_retrain = manager.needs_retraining(
            model_name="xgboost",
            current_time=datetime(2026, 3, 30, tzinfo=timezone.utc),
            retrain_interval_days=7,
        )
        assert needs_retrain is False
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd services/python && python -m pytest tests/test_versioning.py -v
```

Expected: FAIL -- `ModuleNotFoundError: No module named 'src.ml.versioning'`

- [ ] **Step 3: Implement model versioning**

Create `services/python/src/ml/versioning.py`:

```python
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
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any

import joblib

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
    """Manages model saving, loading, versioning, and retraining decisions.

    Directory structure:
        base_dir/
        ├── xgboost/
        │   ├── v_20260323_000000/
        │   │   ├── model files...
        │   │   └── metadata.json
        │   └── v_20260330_000000/
        │       ├── model files...
        │       └── metadata.json
        ├── lightgbm/
        │   └── ...
        └── lstm/
            └── ...
    """

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
        """Save a trained model as a new version.

        Args:
            model: Trained model instance (must implement save()).
            metrics: Training/validation metrics for comparison.
            timestamp: Version timestamp (defaults to now UTC).

        Returns:
            ModelVersion metadata for the saved version.
        """
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)

        version_id = self._generate_version_id(timestamp)
        model_dir = os.path.join(self._base_dir, model.name, version_id)
        os.makedirs(model_dir, exist_ok=True)

        # Save model weights
        model.save(model_dir)

        # Save metadata
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
        """List all saved versions for a model, most recent first.

        Args:
            model_name: Model name (e.g., "xgboost").

        Returns:
            List of ModelVersion sorted by created_at descending.
        """
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

        # Sort by created_at descending (most recent first)
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions

    def load_latest(self, model: BaseModel, model_name: str) -> ModelVersion:
        """Load the latest saved version into a model instance.

        Args:
            model: Model instance to load weights into.
            model_name: Model name to look up versions for.

        Returns:
            The ModelVersion that was loaded.

        Raises:
            FileNotFoundError: If no versions exist.
        """
        versions = self.list_versions(model_name)
        if not versions:
            raise FileNotFoundError(
                f"No saved versions found for model '{model_name}' in {self._base_dir}"
            )

        latest = versions[0]
        model.load(latest.path)
        return latest

    def load_version(
        self, model: BaseModel, model_name: str, version_id: str
    ) -> ModelVersion:
        """Load a specific version into a model instance.

        Args:
            model: Model instance to load weights into.
            model_name: Model name.
            version_id: Specific version ID to load.

        Returns:
            The ModelVersion that was loaded.

        Raises:
            FileNotFoundError: If the version does not exist.
        """
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
        """Decide whether new model should replace old model.

        New model replaces old if its average validation RMSE is equal or better.
        Only compares metrics that contain 'val_rmse' in the key. If no such
        metrics exist, falls back to comparing all common numeric keys (lower is better).

        Args:
            old_metrics: Metrics from the currently deployed model.
            new_metrics: Metrics from the newly trained model.

        Returns:
            True if the new model should replace the old one.
        """
        # Find validation RMSE metrics
        val_keys = [k for k in old_metrics if "val_rmse" in k]

        if val_keys:
            old_avg = sum(old_metrics[k] for k in val_keys) / len(val_keys)
            new_val_keys = [k for k in new_metrics if "val_rmse" in k]
            if new_val_keys:
                new_avg = sum(new_metrics[k] for k in new_val_keys) / len(new_val_keys)
                return new_avg <= old_avg

        # Fallback: compare all common numeric keys (lower is better)
        common_keys = set(old_metrics.keys()) & set(new_metrics.keys())
        if not common_keys:
            return True  # No basis for comparison, accept new model

        old_avg = sum(old_metrics[k] for k in common_keys) / len(common_keys)
        new_avg = sum(new_metrics[k] for k in common_keys) / len(common_keys)
        return new_avg <= old_avg

    def needs_retraining(
        self,
        model_name: str,
        current_time: datetime | None = None,
        retrain_interval_days: int = 7,
    ) -> bool:
        """Check if a model needs retraining based on time since last version.

        Args:
            model_name: Model name to check.
            current_time: Current timestamp (defaults to now UTC).
            retrain_interval_days: Number of days between retrains (default 7 = weekly).

        Returns:
            True if the model should be retrained.
        """
        if current_time is None:
            current_time = datetime.now(timezone.utc)

        versions = self.list_versions(model_name)
        if not versions:
            return True  # Never trained

        latest = versions[0]
        # Ensure both are timezone-aware for comparison
        latest_time = latest.created_at
        if latest_time.tzinfo is None:
            latest_time = latest_time.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        days_since_last = (current_time - latest_time).days
        return days_since_last >= retrain_interval_days
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd services/python && python -m pytest tests/test_versioning.py -v
```

Expected: All tests PASSED

- [ ] **Step 5: Commit**

```bash
git checkout -b feat/task-10-model-versioning
git add services/python/src/ml/versioning.py services/python/tests/test_versioning.py
git commit -m "feat: add model versioning with save/load, comparison, and weekly retraining logic"
git push -u origin feat/task-10-model-versioning
```

---

### Task 11: Integration -- Full ML Pipeline Test

**Files:**
- Create: `services/python/tests/test_ml_integration.py`

- [ ] **Step 1: Write integration test that exercises the full pipeline**

Create `services/python/tests/test_ml_integration.py`:

```python
"""Integration test: full ML pipeline from features through signal generation.

This test exercises the complete flow:
1. Feature engineering on synthetic data
2. Target computation
3. Train all three models (XGBoost, LightGBM, LSTM)
4. Ensemble consensus
5. Signal generation
6. Risk management check
7. Backtest validation
"""

import tempfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.ml.backtest import BacktestConfig, BacktestEngine
from src.ml.ensemble import EnsembleConsensus
from src.ml.features import FeatureEngineer
from src.ml.models.lightgbm_model import LightGBMModel
from src.ml.models.lstm_model import LSTMModel
from src.ml.models.xgboost_model import XGBoostModel
from src.ml.risk import OpenPosition, PortfolioState, RiskManager
from src.ml.signals import SignalGenerator
from src.ml.versioning import ModelVersionManager


@pytest.fixture
def synthetic_data() -> dict:
    """Generate synthetic market data for integration testing."""
    np.random.seed(42)
    n_days = 400  # ~1.1 years

    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    base_price = 50000.0
    returns = np.random.normal(0.001, 0.02, n_days)
    close = base_price * np.cumprod(1 + returns)

    daily_prices = pd.DataFrame({
        "date": dates,
        "asset": "BTC",
        "open": close * (1 + np.random.normal(0, 0.005, n_days)),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, n_days))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, n_days))),
        "close": close,
        "volume": np.random.uniform(1000, 10000, n_days),
    })

    fundamentals = pd.DataFrame({
        "asset": ["BTC", "ETH", "SOL"],
        "market_cap_rank": [1, 2, 5],
        "total_volume_24h": [35e9, 15e9, 3e9],
        "market_cap": [1.4e12, 4e11, 3e10],
    })

    market_context = pd.DataFrame({
        "date": dates,
        "btc_dominance": np.random.uniform(45, 55, n_days),
        "fear_greed_index": np.random.randint(20, 80, n_days),
        "total_market_cap": np.random.uniform(2e12, 3e12, n_days),
    })

    return {
        "daily_prices": daily_prices,
        "fundamentals": fundamentals,
        "market_context": market_context,
    }


class TestFullMLPipeline:
    def test_end_to_end_pipeline(self, synthetic_data: dict):
        """Run the complete ML pipeline end-to-end."""
        daily_prices = synthetic_data["daily_prices"]
        fundamentals = synthetic_data["fundamentals"]
        market_context = synthetic_data["market_context"]

        # 1. Feature Engineering
        engineer = FeatureEngineer()
        features = engineer.build_feature_matrix(
            daily_prices=daily_prices,
            fundamentals=fundamentals,
            market_context=market_context,
            asset="BTC",
        )
        assert len(features) > 0
        assert not features.isnull().any().any()

        # 2. Build Targets
        targets = engineer.build_targets(daily_prices)
        # Align targets with features (drop warmup rows and future NaN rows)
        n_warmup = len(daily_prices) - len(features)
        targets_aligned = targets.iloc[n_warmup : n_warmup + len(features)].reset_index(drop=True)

        # Drop rows where targets are NaN (last 42 rows)
        valid_mask = ~targets_aligned[["target_7d", "target_14d", "target_28d", "target_42d"]].isnull().any(axis=1)
        X = features[valid_mask].reset_index(drop=True)
        y = targets_aligned[valid_mask][["target_7d", "target_14d", "target_28d", "target_42d"]].reset_index(drop=True)

        assert len(X) > 100, f"Not enough valid samples: {len(X)}"

        # 3. Train/Val Split (80/20, no shuffle for time series)
        split_idx = int(len(X) * 0.8)
        X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

        # 4. Train Models
        xgb_model = XGBoostModel(n_estimators=20, max_depth=3)
        xgb_metrics = xgb_model.train(X_train, y_train, X_val, y_val)
        assert "train_rmse_7d" in xgb_metrics

        lgb_model = LightGBMModel(n_estimators=20, max_depth=3)
        lgb_metrics = lgb_model.train(X_train, y_train, X_val, y_val)
        assert "train_rmse_7d" in lgb_metrics

        lstm_model = LSTMModel(
            sequence_length=20, hidden_size=16, num_layers=1,
            dropout=0.0, epochs=3, batch_size=32,
        )
        lstm_metrics = lstm_model.train(X_train, y_train, X_val, y_val)
        assert "train_loss" in lstm_metrics

        # 5. Generate Predictions
        xgb_pred = xgb_model.predict(X_val)
        lgb_pred = lgb_model.predict(X_val)
        lstm_pred = lstm_model.predict(X_val)

        assert xgb_pred.model_name == "xgboost"
        assert lgb_pred.model_name == "lightgbm"
        assert lstm_pred.model_name == "lstm"

        # 6. Ensemble Consensus
        ensemble = EnsembleConsensus(
            model_weights={"xgboost": 0.4, "lightgbm": 0.4, "lstm": 0.2},
            confidence_threshold=0.3,  # Low threshold for synthetic data
        )
        result = ensemble.combine([xgb_pred, lgb_pred, lstm_pred])
        # Result might be None if confidence is too low, that's OK
        # Just verify the pipeline doesn't crash

        # 7. Signal Generation
        generator = SignalGenerator(stop_loss_pct=-8.0)
        signal = generator.generate(
            asset="BTC",
            ensemble_result=result,
            timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
        )
        # Signal may or may not be generated depending on confidence

        # 8. Risk Management
        risk_mgr = RiskManager(
            max_positions=8,
            max_single_asset_pct=25.0,
            portfolio_drawdown_pause_pct=20.0,
        )
        if signal is not None:
            state = PortfolioState(
                positions=[],
                total_capital_idr=10_000_000,
                peak_value_idr=10_000_000,
                current_value_idr=10_000_000,
            )
            risk_result = risk_mgr.check(signal, state)
            assert isinstance(risk_result.allowed, bool)

    def test_model_versioning_roundtrip(self, synthetic_data: dict):
        """Test save and reload of all models."""
        daily_prices = synthetic_data["daily_prices"]
        fundamentals = synthetic_data["fundamentals"]
        market_context = synthetic_data["market_context"]

        engineer = FeatureEngineer()
        features = engineer.build_feature_matrix(
            daily_prices=daily_prices,
            fundamentals=fundamentals,
            market_context=market_context,
            asset="BTC",
        )
        targets = engineer.build_targets(daily_prices)
        n_warmup = len(daily_prices) - len(features)
        targets_aligned = targets.iloc[n_warmup : n_warmup + len(features)].reset_index(drop=True)

        valid_mask = ~targets_aligned[["target_7d", "target_14d", "target_28d", "target_42d"]].isnull().any(axis=1)
        X = features[valid_mask].reset_index(drop=True)
        y = targets_aligned[valid_mask][["target_7d", "target_14d", "target_28d", "target_42d"]].reset_index(drop=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            manager = ModelVersionManager(base_dir=tmpdir)

            # Train and save XGBoost
            xgb = XGBoostModel(n_estimators=10, max_depth=2)
            xgb.train(X, y)
            pred_before = xgb.predict(X.iloc[[-1]])

            manager.save_version(
                model=xgb, metrics={"rmse": 0.03},
                timestamp=datetime(2026, 3, 30, tzinfo=timezone.utc),
            )

            # Load into new instance
            xgb2 = XGBoostModel()
            manager.load_latest(xgb2, "xgboost")
            pred_after = xgb2.predict(X.iloc[[-1]])

            assert abs(
                pred_before.horizons[0].predicted_return
                - pred_after.horizons[0].predicted_return
            ) < 1e-6

    def test_backtest_with_generated_signals(self, synthetic_data: dict):
        """Run a backtest using the pipeline's generated signals."""
        daily_prices = synthetic_data["daily_prices"]

        # Create simple signals for backtesting
        np.random.seed(99)
        signal_dates = pd.date_range("2024-04-01", "2025-02-01", freq="14D")
        signals = pd.DataFrame({
            "date": signal_dates,
            "asset": "BTC",
            "action": "BUY",
            "confidence": np.random.uniform(0.7, 0.9, len(signal_dates)),
            "suggested_hold_days": 14,
            "stop_loss_pct": -8.0,
            "expected_return_pct": np.random.uniform(2, 10, len(signal_dates)),
            "model_agreement": "3/3",
        })

        config = BacktestConfig(
            position_size_idr=1_000_000,
            stop_loss_pct=-8.0,
            max_positions=8,
            confidence_threshold=0.7,
        )
        engine = BacktestEngine(config)
        result = engine.run(daily_prices, signals)

        assert result.total_trades > 0
        assert 0.0 <= result.win_rate <= 1.0
        assert len(result.monthly_breakdown) > 0
        assert len(result.per_asset_breakdown) > 0

        # Verify no lookahead
        for trade in result.trades:
            assert trade.entry_date >= trade.signal_date
```

- [ ] **Step 2: Run integration test**

```bash
cd services/python && python -m pytest tests/test_ml_integration.py -v --timeout=120
```

Expected: All tests PASSED

- [ ] **Step 3: Run full test suite**

```bash
cd services/python && python -m pytest tests/ -v --timeout=120
```

Expected: All tests PASSED

- [ ] **Step 4: Commit**

```bash
git checkout -b feat/task-11-ml-integration-test
git add services/python/tests/test_ml_integration.py
git commit -m "test: add full ML pipeline integration test covering features through backtesting"
git push -u origin feat/task-11-ml-integration-test
```
