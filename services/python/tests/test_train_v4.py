import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def make_fake_features(n: int = 5000) -> pd.DataFrame:
    """Generate fake feature data matching v4 schema."""
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    data = {"timestamp": ts}

    feature_cols = [
        "ret_4", "ret_16", "ret_96", "ret_672", "ret_4_lag1",
        "price_to_sma_96", "price_to_sma_672", "ema_ratio", "macd_hist_norm",
        "rsi_norm", "bb_pct",
        "vol_4h", "vol_1d", "vol_ratio", "parkinson_vol",
        "volume_ratio", "taker_buy_ratio",
        "clv", "drawdown",
        "hour_sin", "hour_cos",
        "funding_rate", "funding_ma_3d", "funding_zscore", "cum_funding_3d",
        "btc_ret_96", "btc_residual", "altcoin_dispersion",
        "funding_x_rsi",
    ]
    for col in feature_cols:
        data[col] = rng.normal(0, 1, n)

    target = rng.choice([0, 1, 2], size=n, p=[0.2, 0.6, 0.2])
    data["target"] = target.astype(float)

    return pd.DataFrame(data)


def test_train_one_fold():
    from scripts.train_v4 import FEATURE_COLS, LGB_PARAMS, train_one_fold

    df = make_fake_features(5000)
    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)

    result = train_one_fold(
        X_train=X[:3000], y_train=y[:3000],
        X_val=X[3000:4000], y_val=y[3000:4000],
        X_test=X[4000:5000], y_test=y[4000:5000],
        params=LGB_PARAMS,
    )

    assert "accuracy" in result
    assert "log_loss" in result
    assert "conviction" in result
    assert len(result["conviction"]) == 1000
    assert 0.2 < result["accuracy"] < 0.8


def test_conviction_range():
    from scripts.train_v4 import FEATURE_COLS, LGB_PARAMS, train_one_fold

    df = make_fake_features(5000)
    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)

    result = train_one_fold(
        X_train=X[:3000], y_train=y[:3000],
        X_val=X[3000:4000], y_val=y[3000:4000],
        X_test=X[4000:5000], y_test=y[4000:5000],
        params=LGB_PARAMS,
    )

    assert all(-1.01 <= c <= 1.01 for c in result["conviction"])
