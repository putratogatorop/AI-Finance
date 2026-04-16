# V4 Week 3-4: Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a 29-feature pipeline (15min-calibrated) that reads from PostgreSQL and outputs per-coin Parquet files ready for LightGBM walk-forward training.

**Architecture:** Single `FeatureEngineerV3` class computes OHLCV-derived features. A separate `build_v4_features.py` script loads 15m candles + funding rates from PostgreSQL, computes features per coin (including cross-asset features using BTC data), constructs the 3-class target variable, and saves to Parquet. Feature #24 (oi_change_pct) is deferred — we don't have OI data yet.

**Tech Stack:** Python 3.12, pandas, numpy, SQLAlchemy, pyarrow (Parquet)

---

## File Structure

| File | Responsibility |
|---|---|
| `src/ml/features_v3.py` | `FeatureEngineerV3` class — computes all 29 features from OHLCV + funding data |
| `tests/test_features_v3.py` | Unit tests for each feature group |
| `scripts/build_v4_features.py` | Load from PG, compute features + target, save per-coin Parquet |
| `tests/test_build_v4_features.py` | Integration test for the build script |

## Feature List (from spec Section 4)

All periods calibrated for 15-minute candles (96 candles = 1 day).

| # | Name | Group | Formula |
|---|---|---|---|
| 1 | `ret_4` | Momentum | log(close / close.shift(4)) — 1h return |
| 2 | `ret_16` | Momentum | log(close / close.shift(16)) — 4h return |
| 3 | `ret_96` | Momentum | log(close / close.shift(96)) — 1d return |
| 4 | `ret_672` | Momentum | log(close / close.shift(672)) — 7d return |
| 5 | `ret_4_lag1` | Momentum | ret_4.shift(1) — lagged 1h return |
| 6 | `price_to_sma_96` | Trend | close / SMA(96) - 1 |
| 7 | `price_to_sma_672` | Trend | close / SMA(672) - 1 |
| 8 | `ema_ratio` | Trend | EMA(48) / EMA(104) - 1 |
| 9 | `macd_hist_norm` | Trend | (MACD - signal) / close |
| 10 | `rsi_norm` | Oscillator | (RSI(56) - 50) / 50 |
| 11 | `bb_pct` | Oscillator | (close - BB_lower) / (BB_upper - BB_lower), BB(80) |
| 12 | `vol_4h` | Volatility | pct_change().rolling(16).std() |
| 13 | `vol_1d` | Volatility | pct_change().rolling(96).std() |
| 14 | `vol_ratio` | Volatility | vol_4h / vol_1d |
| 15 | `parkinson_vol` | Volatility | sqrt((1/4ln2) * rolling_mean(ln(H/L)^2, 96)) |
| 16 | `volume_ratio` | Volume | volume / SMA(96, volume) |
| 17 | `taker_buy_ratio` | Volume | taker_buy_base / volume |
| 18 | `clv` | Microstructure | (2*close - high - low) / (high - low) |
| 19 | `drawdown` | Microstructure | close / rolling_max(672) - 1 |
| 20 | `funding_rate` | Funding | raw rate (forward-filled to 15m) |
| 21 | `funding_ma_3d` | Funding | funding_rate.rolling(9).mean() (9 × 8h = 3d) |
| 22 | `funding_zscore` | Funding | (rate - rolling_mean(96)) / rolling_std(96) (96 × 8h = 32d) |
| 23 | `cum_funding_3d` | Funding | funding_rate.rolling(9).sum() |
| 24 | `btc_ret_96` | Cross-Asset | BTC log return 1d (broadcast) |
| 25 | `btc_residual` | Cross-Asset | coin_ret_96 - beta * btc_ret_96 |
| 26 | `altcoin_dispersion` | Cross-Asset | std of ret_96 across 5 alt coins |
| 27 | `hour_sin` | Time | sin(2π * hour / 24) |
| 28 | `hour_cos` | Time | cos(2π * hour / 24) |
| 29 | `funding_x_rsi` | Interaction | funding_zscore * rsi_norm |

**Note:** Feature #24 from spec (oi_change_pct) deferred — no OI data backfilled yet. Feature numbering above is sequential 1-29.

---

### Task 1: Core OHLCV Features (1-19, 27-28)

**Files:**
- Create: `services/python/src/ml/features_v3.py`
- Create: `services/python/tests/test_features_v3.py`

- [ ] **Step 1: Write failing tests for momentum features**

```python
# tests/test_features_v3.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(n: int = 1000, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 15m OHLCV data."""
    rng = np.random.RandomState(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    open_ = close * (1 + rng.normal(0, 0.001, n))
    volume = rng.uniform(100, 1000, n)
    taker_buy_base = volume * rng.uniform(0.3, 0.7, n)

    ts = pd.date_range("2024-01-01", periods=n, freq="15min")
    return pd.DataFrame({
        "timestamp": ts,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume, "taker_buy_base": taker_buy_base,
    })


def test_momentum_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    for col in ["ret_4", "ret_16", "ret_96", "ret_672", "ret_4_lag1"]:
        assert col in result.columns, f"Missing {col}"
        assert result[col].notna().sum() > 0

    # ret_4 should be log return over 4 bars
    expected = np.log(df["close"].iloc[700] / df["close"].iloc[696])
    assert abs(result["ret_4"].iloc[700] - expected) < 1e-10


def test_trend_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    for col in ["price_to_sma_96", "price_to_sma_672", "ema_ratio", "macd_hist_norm"]:
        assert col in result.columns, f"Missing {col}"

    # price_to_sma_96 should be close/SMA(96) - 1
    sma = df["close"].rolling(96).mean()
    expected = df["close"].iloc[700] / sma.iloc[700] - 1
    assert abs(result["price_to_sma_96"].iloc[700] - expected) < 1e-10


def test_oscillator_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "rsi_norm" in result.columns
    assert "bb_pct" in result.columns
    # RSI should be in [-1, 1]
    valid = result["rsi_norm"].dropna()
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01


def test_volatility_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    for col in ["vol_4h", "vol_1d", "vol_ratio", "parkinson_vol"]:
        assert col in result.columns, f"Missing {col}"
    # vol_ratio = vol_4h / vol_1d
    idx = 750
    expected = result["vol_4h"].iloc[idx] / result["vol_1d"].iloc[idx]
    assert abs(result["vol_ratio"].iloc[idx] - expected) < 1e-10


def test_volume_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "volume_ratio" in result.columns
    assert "taker_buy_ratio" in result.columns
    # taker_buy_ratio should be between 0 and 1
    valid = result["taker_buy_ratio"].dropna()
    assert valid.min() >= 0
    assert valid.max() <= 1


def test_microstructure_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "clv" in result.columns
    assert "drawdown" in result.columns
    # CLV should be in [-1, 1]
    valid = result["clv"].dropna()
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01
    # drawdown should be <= 0
    assert result["drawdown"].dropna().max() <= 0.001


def test_time_features():
    from src.ml.features_v3 import compute_ohlcv_features

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    assert "hour_sin" in result.columns
    assert "hour_cos" in result.columns
    # sin^2 + cos^2 should be ~1
    check = result["hour_sin"] ** 2 + result["hour_cos"] ** 2
    assert abs(check.iloc[100] - 1.0) < 1e-10


def test_feature_count():
    from src.ml.features_v3 import compute_ohlcv_features, OHLCV_FEATURE_COLS

    df = make_ohlcv(800)
    result = compute_ohlcv_features(df)

    # 19 OHLCV features + 2 time features = 21
    assert len(OHLCV_FEATURE_COLS) == 21
    for col in OHLCV_FEATURE_COLS:
        assert col in result.columns, f"Missing {col}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_features_v3.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ml.features_v3'`

- [ ] **Step 3: Implement compute_ohlcv_features**

```python
# src/ml/features_v3.py
"""Feature engineering v3 — 15-minute candles for LightGBM v4.

29 features total:
- 21 OHLCV-derived (this file, compute_ohlcv_features)
- 4 funding rate (merge_funding_features)
- 3 cross-asset (compute_cross_asset_features)
- 1 interaction (added in build script after funding + OHLCV merged)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# 21 features from OHLCV data alone
OHLCV_FEATURE_COLS: list[str] = [
    # Momentum (5)
    "ret_4", "ret_16", "ret_96", "ret_672", "ret_4_lag1",
    # Trend (4)
    "price_to_sma_96", "price_to_sma_672", "ema_ratio", "macd_hist_norm",
    # Oscillators (2)
    "rsi_norm", "bb_pct",
    # Volatility (4)
    "vol_4h", "vol_1d", "vol_ratio", "parkinson_vol",
    # Volume (2)
    "volume_ratio", "taker_buy_ratio",
    # Microstructure (2)
    "clv", "drawdown",
    # Time (2)
    "hour_sin", "hour_cos",
]

FUNDING_FEATURE_COLS: list[str] = [
    "funding_rate", "funding_ma_3d", "funding_zscore", "cum_funding_3d",
]

CROSS_ASSET_FEATURE_COLS: list[str] = [
    "btc_ret_96", "btc_residual", "altcoin_dispersion",
]

INTERACTION_FEATURE_COLS: list[str] = [
    "funding_x_rsi",
]

ALL_FEATURE_COLS: list[str] = (
    OHLCV_FEATURE_COLS + FUNDING_FEATURE_COLS
    + CROSS_ASSET_FEATURE_COLS + INTERACTION_FEATURE_COLS
)


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 21 OHLCV-derived features from 15m candle data.

    Input must have: timestamp, open, high, low, close, volume, taker_buy_base.
    Returns original DataFrame with feature columns appended (NaN in warmup rows).
    """
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    taker_buy = out["taker_buy_base"].astype(float)

    # ── Momentum (5) ──
    for period in [4, 16, 96, 672]:
        out[f"ret_{period}"] = np.log(close / close.shift(period))
    out["ret_4_lag1"] = out["ret_4"].shift(1)

    # ── Trend / Mean-Reversion (4) ──
    for period in [96, 672]:
        sma = close.rolling(period).mean()
        out[f"price_to_sma_{period}"] = close / sma - 1

    ema_48 = close.ewm(span=48, adjust=False).mean()
    ema_104 = close.ewm(span=104, adjust=False).mean()
    out["ema_ratio"] = ema_48 / ema_104 - 1

    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    out["macd_hist_norm"] = (macd - macd_signal) / close

    # ── Oscillators (2) ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(56).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(56).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["rsi_norm"] = (rsi - 50) / 50

    bb_mid = close.rolling(80).mean()
    bb_std = close.rolling(80).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    out["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower)

    # ── Volatility (4) ──
    pct = close.pct_change()
    out["vol_4h"] = pct.rolling(16).std()
    out["vol_1d"] = pct.rolling(96).std()
    out["vol_ratio"] = out["vol_4h"] / out["vol_1d"].replace(0, np.nan)

    hl_log = np.log(high / low)
    out["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (hl_log ** 2).rolling(96).mean()
    )

    # ── Volume (2) ──
    vol_sma = volume.rolling(96).mean()
    out["volume_ratio"] = volume / vol_sma.replace(0, np.nan)
    out["taker_buy_ratio"] = taker_buy / volume.replace(0, np.nan)

    # ── Microstructure (2) ──
    bar_range = high - low
    out["clv"] = (2 * close - high - low) / bar_range.replace(0, np.nan)
    rolling_high = close.rolling(672).max()
    out["drawdown"] = close / rolling_high - 1

    # ── Time (2) ──
    ts = pd.to_datetime(out["timestamp"])
    out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)

    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/python && python -m pytest tests/test_features_v3.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/src/ml/features_v3.py services/python/tests/test_features_v3.py
git commit -m "feat: add FeatureEngineerV3 — 21 OHLCV features for 15min candles"
```

---

### Task 2: Funding Rate Features (20-23)

**Files:**
- Modify: `services/python/src/ml/features_v3.py`
- Modify: `services/python/tests/test_features_v3.py`

- [ ] **Step 1: Write failing test for funding features**

Append to `tests/test_features_v3.py`:

```python
def make_funding(n: int = 100) -> pd.DataFrame:
    """Generate synthetic 8-hourly funding rate data."""
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="8h")
    rates = rng.normal(0.0001, 0.0005, n)
    return pd.DataFrame({"timestamp": ts, "funding_rate": rates})


def test_merge_funding_features():
    from src.ml.features_v3 import merge_funding_features

    ohlcv = make_ohlcv(1000)
    funding = make_funding(200)
    result = merge_funding_features(ohlcv, funding)

    for col in ["funding_rate", "funding_ma_3d", "funding_zscore", "cum_funding_3d"]:
        assert col in result.columns, f"Missing {col}"

    # funding_rate should be forward-filled (not all NaN)
    assert result["funding_rate"].notna().sum() > 500


def test_funding_zscore_range():
    from src.ml.features_v3 import merge_funding_features

    ohlcv = make_ohlcv(1000)
    funding = make_funding(200)
    result = merge_funding_features(ohlcv, funding)

    valid = result["funding_zscore"].dropna()
    # Z-score should be roughly centered around 0
    assert abs(valid.mean()) < 2.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_features_v3.py::test_merge_funding_features -v`
Expected: FAIL with `ImportError: cannot import name 'merge_funding_features'`

- [ ] **Step 3: Implement merge_funding_features**

Add to `src/ml/features_v3.py`:

```python
def merge_funding_features(
    ohlcv_df: pd.DataFrame,
    funding_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge 8-hourly funding rates into 15m OHLCV and compute 4 features.

    Funding rates are forward-filled across 15m intervals, then:
    - funding_rate: raw rate
    - funding_ma_3d: rolling mean over 9 funding periods (9 × 8h = 3 days)
    - funding_zscore: (rate - mean_96) / std_96 where 96 periods = 32 days
    - cum_funding_3d: rolling sum over 9 periods

    The rolling windows operate on the 8h funding frequency before upsampling.
    """
    out = ohlcv_df.copy()
    ohlcv_ts = pd.to_datetime(out["timestamp"])

    # Prepare funding series indexed by timestamp
    fund = funding_df.copy()
    fund["timestamp"] = pd.to_datetime(fund["timestamp"])
    fund = fund.sort_values("timestamp").drop_duplicates("timestamp")
    fund = fund.set_index("timestamp")

    # Compute rolling features at 8h frequency BEFORE upsampling
    rate = fund["funding_rate"]
    fund["funding_ma_3d"] = rate.rolling(9, min_periods=1).mean()
    roll_mean = rate.rolling(96, min_periods=10).mean()
    roll_std = rate.rolling(96, min_periods=10).std()
    fund["funding_zscore"] = (rate - roll_mean) / roll_std.replace(0, np.nan)
    fund["cum_funding_3d"] = rate.rolling(9, min_periods=1).sum()

    # Reindex to 15m timestamps via merge_asof (forward-fill)
    out["_ts"] = ohlcv_ts
    out = out.sort_values("_ts")
    fund_reset = fund.reset_index()

    merged = pd.merge_asof(
        out, fund_reset[["timestamp", "funding_rate", "funding_ma_3d",
                         "funding_zscore", "cum_funding_3d"]],
        left_on="_ts", right_on="timestamp",
        direction="backward",
    )
    merged = merged.drop(columns=["_ts", "timestamp_y"], errors="ignore")
    if "timestamp_x" in merged.columns:
        merged = merged.rename(columns={"timestamp_x": "timestamp"})

    return merged
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/python && python -m pytest tests/test_features_v3.py -v`
Expected: PASS (10 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/src/ml/features_v3.py services/python/tests/test_features_v3.py
git commit -m "feat: add funding rate features (4) with 8h→15m forward-fill"
```

---

### Task 3: Cross-Asset Features (24-26) and Interaction (29)

**Files:**
- Modify: `services/python/src/ml/features_v3.py`
- Modify: `services/python/tests/test_features_v3.py`

- [ ] **Step 1: Write failing tests**

Append to `tests/test_features_v3.py`:

```python
def test_cross_asset_features():
    from src.ml.features_v3 import compute_cross_asset_features

    n = 1000
    rng = np.random.RandomState(42)

    # BTC close
    btc_close = pd.Series(50000 * np.exp(np.cumsum(rng.normal(0, 0.001, n))))
    # Alt close (correlated with BTC)
    alt_close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, n))))
    # All alt returns for dispersion (5 coins)
    all_alt_ret_96 = pd.DataFrame({
        f"alt_{i}": pd.Series(np.exp(np.cumsum(rng.normal(0, 0.002, n)))).pct_change(96)
        for i in range(5)
    })

    result = compute_cross_asset_features(alt_close, btc_close, all_alt_ret_96)

    assert "btc_ret_96" in result.columns
    assert "btc_residual" in result.columns
    assert "altcoin_dispersion" in result.columns
    assert len(result) == n


def test_interaction_feature():
    from src.ml.features_v3 import add_interaction_features

    df = pd.DataFrame({
        "funding_zscore": [0.5, -1.0, 2.0, 0.0],
        "rsi_norm": [0.3, -0.5, 0.8, 0.0],
    })
    result = add_interaction_features(df)
    assert "funding_x_rsi" in result.columns
    assert result["funding_x_rsi"].iloc[0] == pytest.approx(0.15)
    assert result["funding_x_rsi"].iloc[2] == pytest.approx(1.6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_features_v3.py::test_cross_asset_features -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement cross-asset and interaction features**

Add to `src/ml/features_v3.py`:

```python
def compute_cross_asset_features(
    alt_close: pd.Series,
    btc_close: pd.Series,
    all_alt_ret_96: pd.DataFrame,
    window: int = 672,
) -> pd.DataFrame:
    """Compute 3 cross-asset features.

    Parameters
    ----------
    alt_close : pd.Series — this coin's close prices
    btc_close : pd.Series — BTC close prices (same length, aligned)
    all_alt_ret_96 : pd.DataFrame — 1-day returns for all 5 alt coins (columns)
    window : int — rolling window for beta calculation (672 = 7 days of 15m)

    Returns
    -------
    DataFrame with: btc_ret_96, btc_residual, altcoin_dispersion
    """
    n = len(alt_close)
    result = pd.DataFrame(index=range(n))

    btc_ret_96 = np.log(btc_close / btc_close.shift(96))
    alt_ret_96 = np.log(alt_close / alt_close.shift(96))

    result["btc_ret_96"] = btc_ret_96.values

    # Rolling beta: cov(alt, btc) / var(btc)
    alt_ret = alt_close.pct_change()
    btc_ret = btc_close.pct_change()
    rolling_cov = alt_ret.rolling(window).cov(btc_ret)
    rolling_var = btc_ret.rolling(window).var()
    beta = rolling_cov / rolling_var.replace(0, np.nan)

    result["btc_residual"] = (alt_ret_96 - beta * btc_ret_96).values

    # Altcoin dispersion: std of 1-day returns across all alt coins
    result["altcoin_dispersion"] = all_alt_ret_96.std(axis=1).values

    return result


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add interaction feature: funding_zscore * rsi_norm."""
    out = df.copy()
    out["funding_x_rsi"] = out["funding_zscore"] * out["rsi_norm"]
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/python && python -m pytest tests/test_features_v3.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/src/ml/features_v3.py services/python/tests/test_features_v3.py
git commit -m "feat: add cross-asset features (3) and interaction feature"
```

---

### Task 4: Target Variable Construction

**Files:**
- Modify: `services/python/src/ml/features_v3.py`
- Modify: `services/python/tests/test_features_v3.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_features_v3.py`:

```python
def test_build_target():
    from src.ml.features_v3 import build_target

    n = 500
    rng = np.random.RandomState(42)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))))
    vol = pd.Series(rng.normal(0, 0.003, n)).rolling(96).std()

    target = build_target(close, vol, horizon=16)

    # Should have 3 classes: 0 (SHORT), 1 (FLAT), 2 (LONG)
    valid = target.dropna()
    assert set(valid.unique()).issubset({0, 1, 2})

    # Last `horizon` rows should be NaN
    assert target.iloc[-1] != target.iloc[-1]  # NaN check

    # FLAT should be the majority (~60%)
    flat_pct = (valid == 1).sum() / len(valid)
    assert 0.45 < flat_pct < 0.75


def test_build_target_distribution():
    from src.ml.features_v3 import build_target

    n = 5000
    rng = np.random.RandomState(123)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.003, n))))
    vol = pd.Series(rng.normal(0, 0.003, n)).rolling(96).std()

    target = build_target(close, vol, horizon=16)
    valid = target.dropna()

    # SHORT and LONG should each be ~20%
    short_pct = (valid == 0).sum() / len(valid)
    long_pct = (valid == 2).sum() / len(valid)
    assert 0.10 < short_pct < 0.30
    assert 0.10 < long_pct < 0.30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_features_v3.py::test_build_target -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement build_target**

Add to `src/ml/features_v3.py`:

```python
def build_target(
    close: pd.Series,
    realized_vol: pd.Series,
    horizon: int = 16,
) -> pd.Series:
    """Build 3-class target: SHORT (0), FLAT (1), LONG (2).

    target = classify(forward_return_16bars / realized_vol_96bars)
      SHORT: bottom 20% of risk-adjusted returns
      FLAT:  middle 60%
      LONG:  top 20%

    Parameters
    ----------
    close : pd.Series — close prices
    realized_vol : pd.Series — rolling volatility (e.g. vol_1d)
    horizon : int — forward look period in bars (16 bars = 4 hours at 15min)

    Returns
    -------
    pd.Series of 0, 1, 2 (NaN for last `horizon` rows)
    """
    forward_ret = close.shift(-horizon) / close - 1
    vol_safe = realized_vol.replace(0, np.nan)
    risk_adj = forward_ret / vol_safe

    # Use expanding quantiles to avoid look-ahead bias
    target = pd.Series(np.nan, index=close.index)

    # Compute quantiles on available data up to each point
    q20 = risk_adj.expanding(min_periods=100).quantile(0.20)
    q80 = risk_adj.expanding(min_periods=100).quantile(0.80)

    target[risk_adj <= q20] = 0  # SHORT
    target[risk_adj >= q80] = 2  # LONG
    target[(risk_adj > q20) & (risk_adj < q80)] = 1  # FLAT

    # NaN out the last `horizon` rows (no future data)
    target.iloc[-horizon:] = np.nan

    return target
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/python && python -m pytest tests/test_features_v3.py -v`
Expected: PASS (14 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/src/ml/features_v3.py services/python/tests/test_features_v3.py
git commit -m "feat: add 3-class target variable (SHORT/FLAT/LONG)"
```

---

### Task 5: Build Script — Load from PG, Compute, Save Parquet

**Files:**
- Create: `services/python/scripts/build_v4_features.py`
- Create: `services/python/tests/test_build_v4_features.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_build_v4_features.py
import sys
sys.path.insert(0, ".")

import pandas as pd
from sqlalchemy import create_engine, text
from src.db.models import Base


def test_load_15m_from_pg():
    from scripts.build_v4_features import load_15m_data

    engine = create_engine("postgresql://postgres:MySQL100%25@localhost:5432/market")
    Base.metadata.create_all(engine)

    df = load_15m_data(engine, "BTC")
    assert len(df) > 100_000
    assert "close" in df.columns
    assert "taker_buy_base" in df.columns
    assert "timestamp" in df.columns


def test_load_funding_from_pg():
    from scripts.build_v4_features import load_funding_data

    engine = create_engine("postgresql://postgres:MySQL100%25@localhost:5432/market")
    df = load_funding_data(engine, "BTC")
    assert len(df) > 3000
    assert "funding_rate" in df.columns
    assert "timestamp" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd services/python && python -m pytest tests/test_build_v4_features.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement build_v4_features.py**

```python
# scripts/build_v4_features.py
"""Build v4 feature matrices from PostgreSQL and save to Parquet.

Reads: asset_prices_15m, funding_rates
Outputs: data/features/v4/{ASSET}_features.parquet (one per coin)

Each Parquet contains all 29 features + target + timestamp.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.db.models import Base
from src.ml.features_v3 import (
    ALL_FEATURE_COLS,
    compute_ohlcv_features,
    merge_funding_features,
    compute_cross_asset_features,
    add_interaction_features,
    build_target,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
OUT_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/features/v4")

V4_ALTS = ["SOL", "DOGE", "XRP", "AVAX", "LINK"]
V4_ALL = ["BTC", "ETH"] + V4_ALTS


def load_15m_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, open, high, low, close, volume,
               quote_volume, trades, taker_buy_base, taker_buy_quote
        FROM asset_prices_15m
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"asset": asset})
    return df


def load_funding_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, funding_rate
        FROM funding_rates
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"asset": asset})
    return df


def build_one_coin(
    engine,
    asset: str,
    btc_close: pd.Series,
    all_alt_ret_96: pd.DataFrame,
) -> pd.DataFrame:
    """Build complete feature matrix for one coin."""
    logger.info(f"  {asset}: loading data...")
    ohlcv = load_15m_data(engine, asset)
    funding = load_funding_data(engine, asset)

    logger.info(f"  {asset}: computing OHLCV features ({len(ohlcv):,} rows)...")
    featured = compute_ohlcv_features(ohlcv)

    logger.info(f"  {asset}: merging funding features...")
    featured = merge_funding_features(featured, funding)

    logger.info(f"  {asset}: computing cross-asset features...")
    alt_close = pd.Series(ohlcv["close"].values, dtype=float)
    cross = compute_cross_asset_features(alt_close, btc_close, all_alt_ret_96)
    for col in cross.columns:
        featured[col] = cross[col].values

    logger.info(f"  {asset}: adding interaction features...")
    featured = add_interaction_features(featured)

    logger.info(f"  {asset}: building target...")
    close = featured["close"].astype(float)
    vol = featured["vol_1d"]
    featured["target"] = build_target(close, vol, horizon=16)

    # Keep only timestamp + features + target, drop NaN warmup
    keep_cols = ["timestamp"] + ALL_FEATURE_COLS + ["target"]
    available = [c for c in keep_cols if c in featured.columns]
    result = featured[available].dropna().reset_index(drop=True)

    return result


def main():
    engine = create_engine(DB_URL)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    # Load BTC close for cross-asset features
    logger.info("Loading BTC data for cross-asset features...")
    btc_ohlcv = load_15m_data(engine, "BTC")
    btc_close = pd.Series(btc_ohlcv["close"].values, dtype=float)

    # Load all alt 1-day returns for dispersion
    logger.info("Loading alt returns for dispersion...")
    alt_ret_cols = {}
    for alt in V4_ALTS:
        alt_ohlcv = load_15m_data(engine, alt)
        alt_c = pd.Series(alt_ohlcv["close"].values, dtype=float)
        alt_ret_cols[alt] = np.log(alt_c / alt_c.shift(96))
    all_alt_ret_96 = pd.DataFrame(alt_ret_cols)

    # Build features for each coin
    for asset in V4_ALL:
        logger.info(f"Building features for {asset}...")
        df = build_one_coin(engine, asset, btc_close, all_alt_ret_96)

        out_path = OUT_DIR / f"{asset}_features.parquet"
        df.to_parquet(out_path, index=False)
        logger.info(f"  {asset}: {len(df):,} rows, {len(df.columns)} cols → {out_path}")

    elapsed = time.time() - start
    logger.info("=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"Output: {OUT_DIR}")

    # Summary
    for asset in V4_ALL:
        path = OUT_DIR / f"{asset}_features.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            target_dist = df["target"].value_counts(normalize=True).sort_index()
            logger.info(f"  {asset}: {len(df):,} rows | "
                        f"SHORT={target_dist.get(0, 0):.1%} "
                        f"FLAT={target_dist.get(1, 0):.1%} "
                        f"LONG={target_dist.get(2, 0):.1%}")

    engine.dispose()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/python && python -m pytest tests/test_build_v4_features.py -v`
Expected: PASS (2 tests). Requires local PostgreSQL with data from Week 1.

- [ ] **Step 5: Commit**

```bash
git add services/python/scripts/build_v4_features.py services/python/tests/test_build_v4_features.py
git commit -m "feat: add build_v4_features.py — full 29-feature pipeline to Parquet"
```

- [ ] **Step 6: Run the build script**

Run: `cd services/python && python scripts/build_v4_features.py`
Expected: Generates 7 Parquet files in `data/features/v4/`. Each should have ~100K rows, 31 columns (timestamp + 29 features + target). Target distribution ~20/60/20.

- [ ] **Step 7: Verify output**

Run:
```bash
python -c "
import pandas as pd
from pathlib import Path
for p in sorted(Path('C:/Users/togat/Desktop/AI-Finance/data/features/v4').glob('*.parquet')):
    df = pd.read_parquet(p)
    print(f'{p.stem}: {len(df):,} rows, {len(df.columns)} cols, NaN={df.isna().sum().sum()}')
"
```
Expected: 7 files, ~100K rows each, 31 columns, 0 NaN.

- [ ] **Step 8: Commit**

```bash
git add scripts/build_v4_features.py
git commit -m "data: build v4 feature matrices (29 features + target, 7 assets)"
```
