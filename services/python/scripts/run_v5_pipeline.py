# scripts/run_v5_pipeline.py
"""V5 Two-Stage Extreme Move Predictor Pipeline.

End-to-end pipeline:
1. Load data from PostgreSQL (asset_prices_15m, funding_rates, contract_stats_1h,
   fear_greed_index)
2. Resample 15min → 1h, merge all sources
3. Compute ~30 features (OHLCV, contract stats, funding, fear/greed, interactions)
4. Build targets: Stage 1 = extreme move (|ret_4h| > 2*ATR), Stage 2 = direction
5. Walk-forward train binary LightGBM for both stages
6. Backtest the two-stage signal
7. Print results

Requires:
- PostgreSQL at DB_URL with tables: asset_prices_15m, funding_rates,
  contract_stats_1h, fear_greed_index
- lightgbm, pandas, numpy, sqlalchemy
"""

from __future__ import annotations

import logging
import sys
import time
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.ml.walk_forward import walk_forward_splits
from src.ml.backtester_v4 import backtest_coin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ─── Config ───────────────────────────────────────────────────────────────────

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"

ASSETS = ["BTC", "LINK", "XRP", "AVAX"]

# Data cutoff: contract_stats_1h starts Dec 2023
CONTRACT_STATS_START = "2023-12-01"

# Walk-forward (1h bars)
TRAIN_SIZE = 24 * 182   # 6 months = 4368 bars
VAL_SIZE = 24 * 30      # 1 month  = 720 bars
TEST_SIZE = 24 * 30     # 1 month  = 720 bars
STEP_SIZE = 24 * 14     # 2 weeks  = 336 bars
EMBARGO = 4             # 4h target horizon

STAGE1_PARAMS = {
    "objective": "binary",
    "num_leaves": 63,
    "min_child_samples": 100,
    "learning_rate": 0.01,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": -1,
    "n_estimators": 3000,
    "scale_pos_weight": 4.0,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

STAGE2_PARAMS = {
    "objective": "binary",
    "num_leaves": 31,
    "min_child_samples": 50,
    "learning_rate": 0.01,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "max_depth": 5,
    "n_estimators": 2000,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42,
}

EARLY_STOPPING = 50

STAGE1_THRESHOLD = 0.6
STAGE2_LONG_THRESHOLD = 0.6
STAGE2_SHORT_THRESHOLD = 0.4

# Feature column lists (used to select from merged DataFrame)
STAGE1_FEATURE_COLS = [
    # OHLCV
    "ret_1", "ret_4", "ret_24",
    "vol_4h", "vol_1d", "vol_ratio", "parkinson_vol",
    "volume_ratio", "taker_buy_ratio",
    "clv", "rsi_norm", "bb_pct",
    # Funding
    "funding_rate", "funding_ma_3d", "funding_zscore", "cum_funding_3d",
    # Contract stats
    "oi_change_1h", "oi_change_4h", "oi_zscore",
    "liq_long_1h", "liq_short_1h", "liq_imbalance",
    "ls_ratio", "ls_ratio_change_4h",
    # Fear/greed
    "fear_greed", "fear_greed_change_3d",
    # Interaction
    "funding_x_liq",
]

STAGE2_FEATURE_COLS = STAGE1_FEATURE_COLS + [
    # Additional momentum for direction
    "ret_1", "ret_4", "ret_24",
    "rsi_norm", "bb_pct",
    "taker_buy_ratio", "clv",
    "ls_ratio", "ls_ratio_change_4h",
]
# Deduplicate while preserving order
_seen: set = set()
_deduped = []
for _c in STAGE2_FEATURE_COLS:
    if _c not in _seen:
        _seen.add(_c)
        _deduped.append(_c)
STAGE2_FEATURE_COLS = _deduped
del _seen, _deduped, _c


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_15m_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, open, high, low, close, volume, taker_buy_base
        FROM asset_prices_15m
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"asset": asset})


def load_funding_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, funding_rate
        FROM funding_rates
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"asset": asset})


def load_contract_stats(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, open_interest_usd, long_liq_usd, short_liq_usd,
               lsr_taker, lsr_account
        FROM contract_stats_1h
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"asset": asset})


def load_fear_greed(engine) -> pd.DataFrame:
    query = text("""
        SELECT date AS timestamp, value
        FROM fear_greed_index
        ORDER BY date
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn)


# ─── Feature Engineering ──────────────────────────────────────────────────────

def resample_to_1h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 15min OHLCV to 1h bars."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    resampled = df.resample("1h").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
        "taker_buy_base": "sum",
    }).dropna()
    return resampled.reset_index()


def compute_ohlcv_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute OHLCV features on 1h bars."""
    out = df.copy()
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    volume = out["volume"].astype(float)
    taker_buy = out["taker_buy_base"].astype(float)

    # Log returns
    out["ret_1"] = np.log(close / close.shift(1))
    out["ret_4"] = np.log(close / close.shift(4))
    out["ret_24"] = np.log(close / close.shift(24))

    # Volatility
    pct = close.pct_change()
    out["vol_4h"] = pct.rolling(4).std()
    out["vol_1d"] = pct.rolling(24).std()
    out["vol_ratio"] = out["vol_4h"] / out["vol_1d"].replace(0, np.nan)

    hl_log = np.log(high / low)
    out["parkinson_vol"] = np.sqrt(
        (1 / (4 * np.log(2))) * (hl_log ** 2).rolling(24).mean()
    )

    # Volume
    vol_sma = volume.rolling(24).mean()
    out["volume_ratio"] = volume / vol_sma.replace(0, np.nan)
    out["taker_buy_ratio"] = taker_buy / volume.replace(0, np.nan)

    # Microstructure
    bar_range = (high - low).replace(0, np.nan)
    out["clv"] = (2 * close - high - low) / bar_range

    # RSI(14)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    out["rsi_norm"] = (rsi - 50) / 50

    # Bollinger Bands (20)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_range = (bb_upper - bb_lower).replace(0, np.nan)
    out["bb_pct"] = (close - bb_lower) / bb_range

    return out


def compute_contract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 8 features from contract_stats_1h data.

    Parameters
    ----------
    df : DataFrame with columns: timestamp, open_interest_usd, long_liq_usd,
         short_liq_usd, lsr_taker, lsr_account

    Returns
    -------
    DataFrame with original columns plus 8 new feature columns.
    """
    out = df.copy()
    oi = out["open_interest_usd"].astype(float)
    long_liq = out["long_liq_usd"].astype(float)
    short_liq = out["short_liq_usd"].astype(float)
    lsr = out["lsr_taker"].astype(float)

    # OI features
    out["oi_change_1h"] = oi.pct_change(1)
    out["oi_change_4h"] = oi.pct_change(4)

    oi_roll_mean = out["oi_change_1h"].rolling(168, min_periods=24).mean()
    oi_roll_std = out["oi_change_1h"].rolling(168, min_periods=24).std()
    out["oi_zscore"] = (out["oi_change_1h"] - oi_roll_mean) / oi_roll_std.replace(0, np.nan)

    # Liquidation features
    out["liq_long_1h"] = np.log1p(long_liq)
    out["liq_short_1h"] = np.log1p(short_liq)
    denom = long_liq + short_liq + 1
    out["liq_imbalance"] = (long_liq - short_liq) / denom

    # LSR features
    out["ls_ratio"] = lsr
    out["ls_ratio_change_4h"] = lsr.pct_change(4)

    return out


def merge_funding_features(ohlcv_1h: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
    """Merge 8h funding rates into 1h bars via merge_asof (forward-fill).

    Computes: funding_rate, funding_ma_3d (9 periods), funding_zscore (96 periods),
    cum_funding_3d (9 periods sum).
    """
    out = ohlcv_1h.copy()
    ohlcv_ts = pd.to_datetime(out["timestamp"], utc=True)

    fund = funding_df.copy()
    fund["timestamp"] = pd.to_datetime(fund["timestamp"], utc=True)
    fund = fund.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    rate = fund["funding_rate"]
    fund["funding_ma_3d"] = rate.rolling(9, min_periods=1).mean()
    roll_mean = rate.rolling(96, min_periods=10).mean()
    roll_std = rate.rolling(96, min_periods=10).std()
    fund["funding_zscore"] = (rate - roll_mean) / roll_std.replace(0, np.nan)
    fund["cum_funding_3d"] = rate.rolling(9, min_periods=1).sum()

    out["_ts"] = ohlcv_ts
    out = out.sort_values("_ts")
    fund_reset = fund.reset_index()

    merged = pd.merge_asof(
        out,
        fund_reset[["timestamp", "funding_rate", "funding_ma_3d",
                    "funding_zscore", "cum_funding_3d"]],
        left_on="_ts",
        right_on="timestamp",
        direction="backward",
    )
    merged = merged.drop(columns=["_ts", "timestamp_y"], errors="ignore")
    if "timestamp_x" in merged.columns:
        merged = merged.rename(columns={"timestamp_x": "timestamp"})
    return merged


def merge_fear_greed_features(ohlcv_1h: pd.DataFrame, fear_greed_df: pd.DataFrame) -> pd.DataFrame:
    """Merge daily fear/greed index into 1h bars via merge_asof (forward-fill).

    Computes: fear_greed (normalized 0-1), fear_greed_change_3d.
    """
    out = ohlcv_1h.copy()
    ohlcv_ts = pd.to_datetime(out["timestamp"], utc=True)

    fg = fear_greed_df.copy()
    fg["timestamp"] = pd.to_datetime(fg["timestamp"], utc=True)
    fg = fg.sort_values("timestamp").drop_duplicates("timestamp").set_index("timestamp")

    fg["fear_greed"] = fg["value"] / 100.0
    fg["fear_greed_change_3d"] = fg["value"].diff(3) / 100.0

    out["_ts"] = ohlcv_ts
    out = out.sort_values("_ts")
    fg_reset = fg.reset_index()

    merged = pd.merge_asof(
        out,
        fg_reset[["timestamp", "fear_greed", "fear_greed_change_3d"]],
        left_on="_ts",
        right_on="timestamp",
        direction="backward",
    )
    merged = merged.drop(columns=["_ts", "timestamp_y"], errors="ignore")
    if "timestamp_x" in merged.columns:
        merged = merged.rename(columns={"timestamp_x": "timestamp"})
    return merged


def merge_contract_stats(ohlcv_1h: pd.DataFrame, contract_df: pd.DataFrame) -> pd.DataFrame:
    """Merge contract_stats_1h features into 1h OHLCV by exact timestamp join."""
    ohlcv_ts = pd.to_datetime(ohlcv_1h["timestamp"], utc=True)
    contract_ts = pd.to_datetime(contract_df["timestamp"], utc=True)

    ohlcv_copy = ohlcv_1h.copy()
    ohlcv_copy["timestamp"] = ohlcv_ts

    contract_copy = contract_df.copy()
    contract_copy["timestamp"] = contract_ts

    contract_feat_cols = [
        "timestamp", "oi_change_1h", "oi_change_4h", "oi_zscore",
        "liq_long_1h", "liq_short_1h", "liq_imbalance",
        "ls_ratio", "ls_ratio_change_4h",
    ]
    available = [c for c in contract_feat_cols if c in contract_copy.columns]

    merged = pd.merge(ohlcv_copy, contract_copy[available], on="timestamp", how="inner")
    return merged


def _compute_atr_series(high: pd.Series, low: pd.Series, close: pd.Series,
                         period: int = 4) -> pd.Series:
    """Compute ATR as a fraction of close price (unit-less, comparable to log returns).

    True range is expressed as a percentage of the previous close, so the resulting
    ATR is in the same units as log returns and can be compared directly with
    ``abs(forward_log_return) > multiplier * ATR``.
    """
    h = high.astype(float).values
    l = low.astype(float).values
    c = close.astype(float).values
    n = len(c)

    # TR as fraction of close (unit-less)
    tr = np.empty(n)
    tr[0] = (h[0] - l[0]) / c[0] if c[0] != 0 else 0.0
    for i in range(1, n):
        hl = (h[i] - l[i]) / c[i] if c[i] != 0 else 0.0
        hpc = abs(h[i] - c[i - 1]) / c[i - 1] if c[i - 1] != 0 else 0.0
        lpc = abs(l[i] - c[i - 1]) / c[i - 1] if c[i - 1] != 0 else 0.0
        tr[i] = max(hl, hpc, lpc)

    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period

    return pd.Series(atr, index=close.index)


def build_extreme_move_target(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon: int = 4,
    atr_multiplier: float = 2.0,
    atr_period: int = 14,
) -> tuple[pd.Series, pd.Series]:
    """Build two-stage targets.

    Stage 1 (y1): 1 if |forward_4h_return| > atr_multiplier * ATR, else 0.
    Stage 2 (y2): 1 if forward return is positive (up), 0 if negative (down).
                  Only defined where y1 == 1; NaN elsewhere.

    Parameters
    ----------
    close, high, low : price series (aligned index)
    horizon : number of bars ahead to compute forward return
    atr_multiplier : multiplier for ATR threshold (default 2.0)
    atr_period : ATR lookback period

    Returns
    -------
    y1 : pd.Series, dtype int (0/1), NaN at tail
    y2 : pd.Series, dtype float, NaN where y1==0 or at tail
    """
    close = close.astype(float)
    high = high.astype(float)
    low = low.astype(float)

    # ATR on current bars
    atr = _compute_atr_series(high, low, close, period=atr_period)

    # Forward return over horizon bars
    forward_ret = np.log(close.shift(-horizon) / close)

    # Stage 1: extreme move flag
    threshold = atr_multiplier * atr
    y1 = (forward_ret.abs() > threshold).astype(float)
    y1.iloc[-horizon:] = np.nan

    # Stage 2: direction (only where extreme move happened)
    y2 = pd.Series(np.nan, index=close.index)
    extreme_mask = y1 == 1
    y2[extreme_mask] = (forward_ret[extreme_mask] > 0).astype(float)
    y2.iloc[-horizon:] = np.nan

    return y1.astype("Int64").astype(float), y2


def build_all_features(
    ohlcv_1h: pd.DataFrame,
    contract_df: pd.DataFrame,
    funding_df: pd.DataFrame,
    fear_greed_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge all data sources and compute all features.

    Returns DataFrame with all feature columns and OHLCV columns,
    filtered to the overlap period (Dec 2023+).
    """
    # Compute OHLCV features
    featured = compute_ohlcv_features(ohlcv_1h)

    # Merge contract stats (exact 1h join)
    contract_feats = compute_contract_features(contract_df)
    featured = merge_contract_stats(featured, contract_feats)

    # Merge funding (merge_asof forward-fill)
    featured = merge_funding_features(featured, funding_df)

    # Merge fear/greed (merge_asof forward-fill)
    featured = merge_fear_greed_features(featured, fear_greed_df)

    # Interaction feature
    if "funding_zscore" in featured.columns and "liq_imbalance" in featured.columns:
        featured["funding_x_liq"] = (
            featured["funding_zscore"].fillna(0) * featured["liq_imbalance"].fillna(0)
        )

    # Filter to Dec 2023+ (when contract_stats starts)
    featured["timestamp"] = pd.to_datetime(featured["timestamp"], utc=True)
    cutoff = pd.Timestamp(CONTRACT_STATS_START, tz="UTC")
    featured = featured[featured["timestamp"] >= cutoff].reset_index(drop=True)

    return featured


# ─── Walk-Forward Training ────────────────────────────────────────────────────

def train_stage1_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> lgb.LGBMClassifier:
    """Train Stage 1 model (extreme move detector) on one fold."""
    cb = [lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)]
    model = lgb.LGBMClassifier(**STAGE1_PARAMS)
    model.fit(X_train, y_train.astype(int),
              eval_set=[(X_val, y_val.astype(int))],
              callbacks=cb)
    return model


def train_stage2_fold(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> Optional[lgb.LGBMClassifier]:
    """Train Stage 2 model (direction) on extreme-move samples only."""
    # Filter to extreme move samples
    mask_train = ~np.isnan(y_train) & (y_train >= 0)
    mask_val = ~np.isnan(y_val) & (y_val >= 0)

    if mask_train.sum() < 50 or mask_val.sum() < 10:
        return None

    Xt = X_train[mask_train]
    yt = y_train[mask_train].astype(int)
    Xv = X_val[mask_val]
    yv = y_val[mask_val].astype(int)

    cb = [lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.log_evaluation(0)]
    model = lgb.LGBMClassifier(**STAGE2_PARAMS)
    model.fit(Xt, yt, eval_set=[(Xv, yv)], callbacks=cb)
    return model


def predict_fold(
    stage1_model: lgb.LGBMClassifier,
    stage2_model: Optional[lgb.LGBMClassifier],
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate two-stage predictions for a test fold.

    Returns
    -------
    stage1_prob : P(extreme move)
    stage2_prob : P(direction=up), 0.5 if no stage2 model
    conviction  : stage1_prob * |stage2_prob - 0.5| * 2, signed by direction
    """
    s1_prob = stage1_model.predict_proba(X_test)[:, 1]

    if stage2_model is not None:
        s2_prob = stage2_model.predict_proba(X_test)[:, 1]
    else:
        s2_prob = np.full(len(X_test), 0.5)

    # Signed conviction: positive = long signal, negative = short signal
    direction = np.where(s2_prob >= 0.5, 1.0, -1.0)
    conviction = s1_prob * np.abs(s2_prob - 0.5) * 2.0 * direction

    return s1_prob, s2_prob, conviction


def get_feature_importance(model: lgb.LGBMClassifier, feature_names: list[str],
                            top_n: int = 10) -> list[tuple[str, float]]:
    """Extract top N feature importances from a LightGBM model."""
    importances = model.feature_importances_
    pairs = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)
    return pairs[:top_n]


def walk_forward_asset(
    df: pd.DataFrame,
    asset: str,
    feat_cols_s1: list[str],
    feat_cols_s2: list[str],
) -> dict:
    """Run full walk-forward training + OOS predictions for one asset.

    Returns dict with oos_df (timestamps + predictions) and fold_metrics.
    """
    # Filter to available features
    s1_cols = [c for c in feat_cols_s1 if c in df.columns]
    s2_cols = [c for c in feat_cols_s2 if c in df.columns]

    if not s1_cols:
        logger.error(f"  {asset}: No stage1 feature columns available!")
        return {}

    # Build arrays — drop rows where y1 or all features are NaN
    valid_mask = df["y1"].notna() & df[s1_cols].notna().all(axis=1)
    df_valid = df[valid_mask].reset_index(drop=True)

    X_s1 = df_valid[s1_cols].values.astype(np.float32)
    X_s2 = df_valid[s2_cols].values.astype(np.float32) if s2_cols else X_s1
    y1 = df_valid["y1"].values.astype(float)
    y2 = df_valid["y2"].values.astype(float)
    timestamps = df_valid["timestamp"].values

    n = len(df_valid)
    splits = walk_forward_splits(
        n_samples=n,
        train_size=TRAIN_SIZE,
        val_size=VAL_SIZE,
        test_size=TEST_SIZE,
        step_size=STEP_SIZE,
        embargo=EMBARGO,
    )

    if not splits:
        logger.warning(f"  {asset}: Not enough data for walk-forward (n={n})")
        return {}

    logger.info(f"  {asset}: {n:,} valid rows, {len(splits)} folds, "
                f"{len(s1_cols)} S1 features, {len(s2_cols)} S2 features")

    all_s1_prob, all_s2_prob, all_conviction = [], [], []
    all_y1, all_y2, all_ts = [], [], []
    fold_metrics = []

    # Accumulate feature importances
    s1_importance_acc: dict[str, float] = {c: 0.0 for c in s1_cols}
    s2_importance_acc: dict[str, float] = {c: 0.0 for c in s2_cols}
    s1_folds_counted = 0
    s2_folds_counted = 0

    for i, (tr, va, te) in enumerate(splits):
        # Stage 1
        s1_model = train_stage1_fold(
            X_s1[tr[0]:tr[1]], y1[tr[0]:tr[1]],
            X_s1[va[0]:va[1]], y1[va[0]:va[1]],
        )

        # Stage 2 (trained on extreme-move samples from train+val windows)
        y2_train = y2[tr[0]:tr[1]]
        y2_val = y2[va[0]:va[1]]
        s2_model = train_stage2_fold(
            X_s2[tr[0]:tr[1]], y2_train,
            X_s2[va[0]:va[1]], y2_val,
        )

        # Predict on test window
        s1p, s2p, conv = predict_fold(s1_model, s2_model, X_s1[te[0]:te[1]])

        all_s1_prob.extend(s1p.tolist())
        all_s2_prob.extend(s2p.tolist())
        all_conviction.extend(conv.tolist())
        all_y1.extend(y1[te[0]:te[1]].tolist())
        all_y2.extend(y2[te[0]:te[1]].tolist())
        all_ts.extend(timestamps[te[0]:te[1]].tolist())

        # Metrics for this fold
        y1_test = y1[te[0]:te[1]]
        y1_pred = (s1p >= STAGE1_THRESHOLD).astype(int)
        valid_y1 = ~np.isnan(y1_test)

        s1_prec = precision_score(y1_test[valid_y1].astype(int), y1_pred[valid_y1],
                                   zero_division=0)
        s1_rec = recall_score(y1_test[valid_y1].astype(int), y1_pred[valid_y1],
                               zero_division=0)
        try:
            s1_auc = roc_auc_score(y1_test[valid_y1].astype(int), s1p[valid_y1])
        except ValueError:
            s1_auc = 0.5

        fold_metrics.append({
            "fold": i, "s1_prec": s1_prec, "s1_rec": s1_rec, "s1_auc": s1_auc,
            "s2_available": s2_model is not None,
        })

        # Accumulate feature importances
        for col, imp in zip(s1_cols, s1_model.feature_importances_):
            s1_importance_acc[col] += imp
        s1_folds_counted += 1

        if s2_model is not None:
            for col, imp in zip(s2_cols, s2_model.feature_importances_):
                s2_importance_acc[col] = s2_importance_acc.get(col, 0.0) + imp
            s2_folds_counted += 1

        if (i + 1) % 5 == 0 or i == len(splits) - 1:
            logger.info(f"  {asset}: fold {i+1}/{len(splits)} | "
                        f"S1 prec={s1_prec:.3f} rec={s1_rec:.3f} auc={s1_auc:.3f}")

    # Aggregate feature importances
    s1_top10 = sorted(s1_importance_acc.items(), key=lambda x: x[1], reverse=True)[:10]
    s2_top10 = sorted(s2_importance_acc.items(), key=lambda x: x[1], reverse=True)[:10]

    logger.info(f"\n  {asset} Stage1 top10 features:")
    for col, imp in s1_top10:
        logger.info(f"    {col}: {imp / max(s1_folds_counted, 1):.1f}")

    logger.info(f"\n  {asset} Stage2 top10 features:")
    for col, imp in s2_top10:
        logger.info(f"    {col}: {imp / max(s2_folds_counted, 1):.1f}")

    oos_df = pd.DataFrame({
        "timestamp": all_ts,
        "stage1_prob": all_s1_prob,
        "stage2_prob": all_s2_prob,
        "conviction": all_conviction,
        "y1_actual": all_y1,
        "y2_actual": all_y2,
    })

    return {
        "oos_df": oos_df,
        "fold_metrics": fold_metrics,
        "s1_top_features": s1_top10,
        "s2_top_features": s2_top10,
    }


# ─── Backtest ─────────────────────────────────────────────────────────────────

def run_backtest(oos_df: pd.DataFrame, ohlcv_15m: pd.DataFrame,
                 asset: str) -> dict:
    """Backtest two-stage signal using backtester_v4.

    The backtester expects a DataFrame with columns: timestamp, high, low, close,
    conviction. We use the signed conviction from our two-stage signal.
    """
    # Prepare 15m prices
    prices = ohlcv_15m.copy()
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)

    oos = oos_df.copy()
    oos["timestamp"] = pd.to_datetime(oos["timestamp"], utc=True)

    # Merge: OOS predictions are on 1h bars — align to 15m bars
    # Use merge_asof to forward-fill 1h signals to 15m bars
    prices = prices.sort_values("timestamp")
    oos_sorted = oos.sort_values("timestamp")

    merged = pd.merge_asof(
        prices[["timestamp", "high", "low", "close"]],
        oos_sorted[["timestamp", "conviction"]],
        on="timestamp",
        direction="backward",
    )
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    merged["conviction"] = merged["conviction"].fillna(0.0)

    # Filter to signal period (where we have OOS predictions)
    start_ts = oos["timestamp"].min()
    merged = merged[merged["timestamp"] >= start_ts].reset_index(drop=True)

    if len(merged) < 100:
        logger.warning(f"  {asset}: Not enough bars for backtest ({len(merged)})")
        return {}

    bt_results = {}
    for thresh in [0.10, 0.15, 0.20, 0.25, 0.30]:
        result = backtest_coin(merged, conviction_threshold=thresh)
        m = result["metrics"]
        weeks = len(merged) / (96 * 7)
        tpw = m["total_trades"] / weeks if weeks > 0 else 0
        bt_results[f"{thresh:.2f}"] = m
        logger.info(f"    @{thresh:.2f}: {m['total_trades']:4d} trades ({tpw:.1f}/wk) "
                    f"ret={m['total_return_pct']:+.1%} sharpe={m['sharpe_ratio']:.2f} "
                    f"wr={m['win_rate']:.1%} dd={m['max_drawdown_pct']:.1%}")

    return bt_results


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    engine = create_engine(DB_URL)
    start = time.time()

    logger.info("V5 Two-Stage Extreme Move Predictor Pipeline")
    logger.info(f"Assets: {ASSETS}")
    logger.info(f"Data cutoff: {CONTRACT_STATS_START}+")
    logger.info(
        f"Walk-forward: train={TRAIN_SIZE} val={VAL_SIZE} "
        f"test={TEST_SIZE} step={STEP_SIZE} embargo={EMBARGO}"
    )

    # Load fear/greed once (shared across assets)
    logger.info("\nLoading fear/greed index...")
    fear_greed_df = load_fear_greed(engine)
    logger.info(f"  fear_greed: {len(fear_greed_df):,} rows")

    all_results = []

    for asset in ASSETS:
        logger.info(f"\n{'='*60}\nProcessing {asset}...")

        # Load raw data
        logger.info(f"  {asset}: loading 15m OHLCV...")
        raw_15m = load_15m_data(engine, asset)
        logger.info(f"  {asset}: {len(raw_15m):,} 15m rows")

        logger.info(f"  {asset}: resampling to 1h...")
        ohlcv_1h = resample_to_1h(raw_15m)
        logger.info(f"  {asset}: {len(ohlcv_1h):,} 1h rows")

        logger.info(f"  {asset}: loading funding rates...")
        funding_df = load_funding_data(engine, asset)
        logger.info(f"  {asset}: {len(funding_df):,} funding rows")

        logger.info(f"  {asset}: loading contract stats...")
        contract_df = load_contract_stats(engine, asset)
        logger.info(f"  {asset}: {len(contract_df):,} contract_stats rows")

        # Build features
        logger.info(f"  {asset}: computing features...")
        df = build_all_features(ohlcv_1h, contract_df, funding_df, fear_greed_df)
        logger.info(f"  {asset}: {len(df):,} rows after cutoff filter")

        # Build targets
        logger.info(f"  {asset}: building targets...")
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)

        y1, y2 = build_extreme_move_target(close, high, low, horizon=4)
        df["y1"] = y1.values
        df["y2"] = y2.values

        # Log target stats
        valid_y1 = y1.dropna()
        pos_rate = valid_y1.mean()
        logger.info(f"  {asset}: y1 positive rate = {pos_rate:.1%} "
                    f"(n={len(valid_y1):,}, extreme={int(valid_y1.sum()):,})")

        # Walk-forward training
        logger.info(f"  {asset}: walk-forward training...")
        wf_result = walk_forward_asset(
            df, asset,
            feat_cols_s1=STAGE1_FEATURE_COLS,
            feat_cols_s2=STAGE2_FEATURE_COLS,
        )

        if not wf_result:
            logger.warning(f"  {asset}: Walk-forward returned no results, skipping backtest")
            continue

        oos_df = wf_result["oos_df"]
        fold_metrics = wf_result["fold_metrics"]

        # Summary metrics
        mean_s1_auc = np.mean([f["s1_auc"] for f in fold_metrics])
        mean_s1_prec = np.mean([f["s1_prec"] for f in fold_metrics])
        mean_s1_rec = np.mean([f["s1_rec"] for f in fold_metrics])
        logger.info(f"\n  {asset} OOS summary: {len(fold_metrics)} folds | "
                    f"S1 AUC={mean_s1_auc:.3f} prec={mean_s1_prec:.3f} rec={mean_s1_rec:.3f}")

        # Backtest
        logger.info(f"\n  {asset} backtest:")
        bt_results = run_backtest(oos_df, raw_15m, asset)

        all_results.append({
            "asset": asset,
            "n_rows": len(df),
            "n_folds": len(fold_metrics),
            "y1_positive_rate": float(pos_rate),
            "mean_s1_auc": float(mean_s1_auc),
            "mean_s1_prec": float(mean_s1_prec),
            "mean_s1_rec": float(mean_s1_rec),
            "s1_top_features": wf_result["s1_top_features"],
            "s2_top_features": wf_result["s2_top_features"],
            "backtest": bt_results,
        })

    # Final summary
    elapsed = time.time() - start
    logger.info(f"\n{'='*60}")
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"\nV5 Pipeline Summary:")
    for r in all_results:
        logger.info(f"  {r['asset']}: {r['n_folds']} folds | "
                    f"y1_rate={r['y1_positive_rate']:.1%} | "
                    f"S1 AUC={r['mean_s1_auc']:.3f}")
        if r["backtest"]:
            best_thresh = max(
                r["backtest"].items(),
                key=lambda kv: kv[1].get("sharpe_ratio", -999),
            )
            m = best_thresh[1]
            logger.info(f"    best@{best_thresh[0]}: ret={m['total_return_pct']:+.1%} "
                        f"sharpe={m['sharpe_ratio']:.2f} wr={m['win_rate']:.1%}")

    engine.dispose()
    return all_results


if __name__ == "__main__":
    main()
