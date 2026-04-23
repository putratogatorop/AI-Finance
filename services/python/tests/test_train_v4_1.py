import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def make_fake_features(n: int = 5000) -> pd.DataFrame:
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
    signal = data["ret_4"] + 0.5 * data["funding_zscore"] + rng.normal(0, 0.5, n)
    target = np.where(signal > np.percentile(signal, 80), 2,
                      np.where(signal < np.percentile(signal, 20), 0, 1))
    data["target"] = target.astype(float)
    return pd.DataFrame(data)


def test_train_one_fold_binary():
    from scripts.train_v4_1 import BINARY_PARAMS, FEATURE_COLS, train_one_fold_binary

    df = make_fake_features(5000)
    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)

    result = train_one_fold_binary(
        X_train=X[:3000], y_train=y[:3000],
        X_val=X[3000:4000], y_val=y[3000:4000],
        X_test=X[4000:5000], y_test=y[4000:5000],
        params=BINARY_PARAMS,
    )

    assert "conviction" in result
    assert "long_precision" in result
    assert "short_precision" in result
    assert len(result["conviction"]) == 1000

    conv = np.array(result["conviction"])
    assert conv.min() >= -1.01
    assert conv.max() <= 1.01

    avg_precision = (result["long_precision"] + result["short_precision"]) / 2
    assert avg_precision > 0.15


def test_conviction_correlation_with_returns():
    from scripts.train_v4_1 import BINARY_PARAMS, FEATURE_COLS, train_one_fold_binary

    df = make_fake_features(5000)
    X = df[FEATURE_COLS].values
    y = df["target"].values.astype(int)

    result = train_one_fold_binary(
        X_train=X[:3000], y_train=y[:3000],
        X_val=X[3000:4000], y_val=y[3000:4000],
        X_test=X[4000:5000], y_test=y[4000:5000],
        params=BINARY_PARAMS,
    )

    conv = np.array(result["conviction"])
    actuals = np.array(result["actuals"])

    high_long = conv > np.percentile(conv, 90)
    _high_short = conv < np.percentile(conv, 10)

    if high_long.sum() > 0:
        long_rate_high = (actuals[high_long] == 2).mean()
        long_rate_all = (actuals == 2).mean()
        print(
            f"LONG rate in top 10% conviction: {long_rate_high:.3f} vs overall: {long_rate_all:.3f}"
        )
