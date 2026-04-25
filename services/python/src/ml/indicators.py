"""Classical technical indicators — single source of truth.

This module hosts pure-function indicator computations used across
backtests, live scanners, and the continuation/bigmover predictors.

Conventions
-----------
- All functions take pandas Series of float close (and high/low/volume
  where needed) and return Series of identical length aligned to the
  input index. NaN before warmup.
- All computations are causal: a value at index `i` reads only inputs at
  indices `<= i`. (Strict no-lookahead; cross-checked by tests/test_indicators.py.)
- Defaults match the values quoted in the canonical references:
  * EMA span (no canonical default — caller picks)
  * RSI(14)
  * MACD(12, 26, 9)
  * KDJ(9, 3, 3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---- EMA --------------------------------------------------------------------


def ema(close: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. `span` matches pandas convention.

    Equivalent to `close.ewm(span=span, adjust=False).mean()`.
    """
    return close.ewm(span=span, adjust=False).mean()


# ---- RSI --------------------------------------------------------------------


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI (using simple moving average of gains/losses).

    Matches the existing implementation in src/ml/features.py:_compute_rsi
    so existing artifacts remain reproducible.
    """
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    # When avg_loss is 0 and avg_gain > 0, rs is +inf and rsi is 100.
    # When both are 0, the value is undefined → NaN (consistent with pandas).
    return out


# ---- MACD -------------------------------------------------------------------


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD, signal line, and histogram in a 3-column DataFrame.

    Columns: ``macd``, ``signal``, ``histogram``.

    `macd = EMA(close, fast) - EMA(close, slow)`
    `signal = EMA(macd, signal)`
    `histogram = macd - signal`
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    sig = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig
    return pd.DataFrame(
        {"macd": macd_line, "signal": sig, "histogram": hist},
        index=close.index,
    )


# ---- KDJ --------------------------------------------------------------------


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    *,
    n: int = 9,
    k_smooth: int = 3,
    d_smooth: int = 3,
) -> pd.DataFrame:
    """KDJ stochastic oscillator (China-popular variant).

    Steps:
      RSV[i] = 100 * (close[i] - min(low[i-n+1..i])) / (max(high[i-n+1..i]) - min(...))
      K[i]   = SMA(RSV, k_smooth) at i
      D[i]   = SMA(K, d_smooth) at i
      J[i]   = 3*K[i] - 2*D[i]

    Returns a DataFrame with columns ``k``, ``d``, ``j``. NaN before warmup
    (i < n + k_smooth + d_smooth - 2).

    Note: classical Chinese definition uses Wilder smoothing; we use SMA
    (matches what most modern crypto tooling reports). The ``n=9, k_smooth=3,
    d_smooth=3`` defaults mirror most TA libraries.
    """
    if not (len(high) == len(low) == len(close)):
        raise ValueError("high, low, close must all be the same length")
    rolling_low = low.rolling(window=n, min_periods=n).min()
    rolling_high = high.rolling(window=n, min_periods=n).max()
    span = rolling_high - rolling_low
    # Where span==0 (flat range) RSV is undefined; pandas yields NaN, which
    # we propagate (caller treats as "indicator not available this bar").
    rsv = 100.0 * (close - rolling_low) / span.where(span > 0)
    k = rsv.rolling(window=k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(window=d_smooth, min_periods=d_smooth).mean()
    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"k": k, "d": d, "j": j}, index=close.index)


# ---- Convenience -----------------------------------------------------------


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range with simple moving average over `period` true ranges.

    Same definition the bigmover detector uses internally
    (services/python/src/ml/bigmover_signals.py).
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()
