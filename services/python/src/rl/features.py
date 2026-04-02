"""Cross-token and BTC/ETH indicator features for RL trading gym."""

import numpy as np
import pandas as pd


def build_cross_token_features(
    alt_close: pd.Series,
    btc_close: pd.Series,
    window: int = 180,
) -> pd.DataFrame:
    """Compute 3 features capturing BTC-alt dependency.

    Parameters
    ----------
    alt_close : pd.Series
        Alt-coin close prices.
    btc_close : pd.Series
        BTC close prices.
    window : int
        Rolling window size in candles (default 180).

    Returns
    -------
    pd.DataFrame with columns:
        btc_alt_beta, btc_alt_corr, btc_alt_residual
    """
    alt_ret = alt_close.pct_change()
    btc_ret = btc_close.pct_change()

    rolling_cov = alt_ret.rolling(window).cov(btc_ret)
    rolling_var = btc_ret.rolling(window).var()

    # Avoid division by zero
    rolling_var_safe = rolling_var.replace(0, np.nan)

    beta = rolling_cov / rolling_var_safe
    corr = alt_ret.rolling(window).corr(btc_ret)
    residual = alt_ret - (beta * btc_ret)

    return pd.DataFrame({
        "btc_alt_beta": beta,
        "btc_alt_corr": corr,
        "btc_alt_residual": residual,
    })


def build_indicator_features(
    btc_df: pd.DataFrame,
    eth_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute 10 BTC/ETH indicator features (leading signals).

    Parameters
    ----------
    btc_df : pd.DataFrame
        BTC OHLCV with columns: open, high, low, close, volume.
    eth_df : pd.DataFrame
        ETH OHLCV with columns: open, high, low, close, volume.

    Returns
    -------
    pd.DataFrame with 10 columns.
    """
    btc_ret = btc_df["close"].pct_change()
    eth_ret = eth_df["close"].pct_change()

    # BTC lagged returns
    btc_ret_1 = btc_ret.shift(1)
    btc_ret_2 = btc_ret.shift(2)
    btc_ret_3 = btc_ret.shift(3)

    # BTC volatility
    btc_vol_24 = btc_ret.rolling(24).std()

    # BTC RSI(14) normalized to [-1, 1]
    delta = btc_df["close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    loss_safe = loss.replace(0, np.nan)
    rs = gain / loss_safe
    btc_rsi = (100 - 100 / (1 + rs) - 50) / 50

    # ETH lagged returns
    eth_ret_1 = eth_ret.shift(1)
    eth_ret_2 = eth_ret.shift(2)
    eth_ret_3 = eth_ret.shift(3)

    # ETH/BTC ratio change
    btc_close_safe = btc_df["close"].replace(0, np.nan)
    eth_btc_ratio_chg = (eth_df["close"] / btc_close_safe).pct_change(6)

    # BTC above SMA50
    btc_above_sma50 = (
        btc_df["close"] > btc_df["close"].rolling(50).mean()
    ).astype(float)

    return pd.DataFrame({
        "btc_ret_1": btc_ret_1,
        "btc_ret_2": btc_ret_2,
        "btc_ret_3": btc_ret_3,
        "btc_vol_24": btc_vol_24,
        "btc_rsi": btc_rsi,
        "eth_ret_1": eth_ret_1,
        "eth_ret_2": eth_ret_2,
        "eth_ret_3": eth_ret_3,
        "eth_btc_ratio_chg": eth_btc_ratio_chg,
        "btc_above_sma50": btc_above_sma50,
    })
