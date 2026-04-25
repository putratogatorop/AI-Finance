"""Shared BTC + altcoin-breadth regime computation.

Both live scanners (`live_scanner_v2.py`, `live_scanner_long_v2.py`) and
multiple backtests previously inlined near-identical regime logic. This
module is the single source of truth.

Behavior is intentionally unchanged from the previous inline versions:

  - SHORT allowed iff BTC daily close < 20d EMA AND breadth < 60%.
  - LONG  allowed iff BTC daily close > 20d EMA AND breadth > 40%.

Where breadth = fraction of non-BTC coins whose latest daily close is above
their own 20d daily EMA. Coins with < 20 days of data are excluded from
breadth (they were excluded by the inline versions too).

The module exposes two API shapes:

  - `regime_state_now()`           — single-snapshot, used by live scanners.
  - `regime_history()`             — date-keyed time series, used by backtests.

Plus the original boolean wrappers (`compute_short_regime`, `compute_long_regime`)
so callers can drop-in replace the inline `compute_regime()` they had.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd

# Tuned to match existing in-tree behavior. Do not change in this refactor.
_DAILY_EMA_SPAN = 20
_BARS_PER_DAY_15M = 96
_MIN_15M_BARS = _DAILY_EMA_SPAN * _BARS_PER_DAY_15M  # 1920
_MIN_DAILY_BARS = _DAILY_EMA_SPAN
_BREADTH_SHORT_MAX = 0.60
_BREADTH_LONG_MIN = 0.40


def _resample_close_daily(df: pd.DataFrame, ts_col: str) -> pd.Series:
    """Daily-last close series from 15m candles; empty Series if no rows."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    daily = (
        df.set_index(ts_col)
        .resample("1D")
        .agg({"close": "last"})
        .dropna()
    )
    return daily["close"]


def _btc_above_ema20_at(close_series: pd.Series, ts) -> bool | None:
    """Is BTC daily close > 20d EMA at the given timestamp?

    `ts` may be any datetime-like; we pick the daily row at-or-before that ts.
    Returns None if the warmup is insufficient.
    """
    if len(close_series) < _MIN_DAILY_BARS:
        return None
    ema = close_series.ewm(span=_DAILY_EMA_SPAN, adjust=False).mean()
    # Pick the latest daily bar <= ts.
    upto = close_series.loc[: ts]
    if len(upto) < _MIN_DAILY_BARS:
        return None
    last_close = float(upto.iloc[-1])
    last_ema = float(ema.loc[upto.index[-1]])
    return last_close > last_ema


def regime_state_now(
    candle_cache: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    btc_key: str = "BTC_USDT",
) -> dict[str, Any]:
    """Compute the latest regime state from a cache of 15m candles.

    Args:
      candle_cache: mapping symbol -> 15m DataFrame. Used for breadth.
      btc_df: BTC 15m DataFrame. Used for the BTC EMA gate.
      ts_col: timestamp column name (live: "timestamp", backtest: "open_time").
      btc_key: key in `candle_cache` to skip when computing breadth.

    Returns:
      dict with keys:
        ready: bool — true when warmup is sufficient
        btc_above_ema20_d: bool | None
        btc_bullish: bool | None  (alias for above)
        btc_bearish: bool | None  (alias for not-above)
        breadth_pct: float | None — None when no coins reached warmup
        allows_short: bool
        allows_long: bool

    Single-snapshot version. For live scanners, equivalent to the prior
    `compute_regime()` pair (short/long).
    """
    btc_close = _resample_close_daily(btc_df, ts_col)
    if len(btc_close) < _MIN_DAILY_BARS or len(btc_df) < _MIN_15M_BARS:
        return _empty_state()
    btc_ema = btc_close.ewm(span=_DAILY_EMA_SPAN, adjust=False).mean()
    btc_above = float(btc_close.iloc[-1]) > float(btc_ema.iloc[-1])

    above_count = 0
    total_count = 0
    for sym, sub in candle_cache.items():
        if sym == btc_key:
            continue
        if sub is None or len(sub) < _MIN_15M_BARS:
            continue
        coin_close = _resample_close_daily(sub, ts_col)
        if len(coin_close) < _MIN_DAILY_BARS:
            continue
        coin_ema = coin_close.ewm(span=_DAILY_EMA_SPAN, adjust=False).mean()
        total_count += 1
        if float(coin_close.iloc[-1]) > float(coin_ema.iloc[-1]):
            above_count += 1

    breadth: float | None = None
    if total_count > 0:
        breadth = above_count / total_count
    # Match prior behavior: when no coin has enough data, fall back to neutral
    # 0.5 for the gate decision (short would fail breadth<0.60? actually 0.5<0.60
    # passes — but BTC gate would still need to be bearish). Keep the explicit
    # fallback for the gate while exposing breadth=None for analytics.
    breadth_for_gate = breadth if breadth is not None else 0.5

    allows_short = (not btc_above) and (breadth_for_gate < _BREADTH_SHORT_MAX)
    allows_long = btc_above and (breadth_for_gate > _BREADTH_LONG_MIN)

    return {
        "ready": True,
        "btc_above_ema20_d": btc_above,
        "btc_bullish": btc_above,
        "btc_bearish": not btc_above,
        "breadth_pct": breadth,
        "allows_short": bool(allows_short),
        "allows_long": bool(allows_long),
    }


def _empty_state() -> dict[str, Any]:
    return {
        "ready": False,
        "btc_above_ema20_d": None,
        "btc_bullish": None,
        "btc_bearish": None,
        "breadth_pct": None,
        "allows_short": False,
        "allows_long": False,
    }


def compute_short_regime(
    candle_cache: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    btc_key: str = "BTC_USDT",
) -> bool:
    """Drop-in replacement for the inline `compute_regime()` in
    `live_scanner_v2.py`. Returns True iff a SHORT signal is allowed."""
    return regime_state_now(
        candle_cache, btc_df, ts_col=ts_col, btc_key=btc_key
    )["allows_short"]


def compute_long_regime(
    candle_cache: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    *,
    ts_col: str = "timestamp",
    btc_key: str = "BTC_USDT",
) -> bool:
    """Drop-in replacement for the inline `compute_regime()` in
    `live_scanner_long_v2.py`. Returns True iff a LONG signal is allowed."""
    return regime_state_now(
        candle_cache, btc_df, ts_col=ts_col, btc_key=btc_key
    )["allows_long"]


def regime_history(
    all_coin_data: dict[str, pd.DataFrame],
    btc_df: pd.DataFrame,
    *,
    ts_col: str = "open_time",
    btc_key: str = "BTCUSDT",
) -> dict[date, dict[str, Any]]:
    """Daily regime time-series for backtests.

    Drop-in replacement for `compute_daily_regime()` in
    `backtest_short_regime.py` and `backtest_long_v2.py`. The returned dict
    is keyed by `date` and carries the same fields the existing backtests
    consume: `btc_bearish`, `breadth`, `breadth_low`, `allowed` (short side).

    The long-side equivalent (`btc_bullish`, `breadth_high`, `allowed_long`)
    is added in the same dict for code that needs both directions.
    """
    btc_daily = (
        btc_df.set_index(ts_col)
        .resample("1D")
        .agg({"close": "last"})
        .dropna()
    )
    btc_daily["ema20"] = btc_daily["close"].ewm(span=_DAILY_EMA_SPAN, adjust=False).mean()
    btc_daily["bearish"] = btc_daily["close"] < btc_daily["ema20"]
    btc_daily["bullish"] = btc_daily["close"] > btc_daily["ema20"]

    coin_above_total: dict[date, list[int]] = {}
    for sym, df in all_coin_data.items():
        if sym == btc_key:
            continue
        if df is None or len(df) == 0:
            continue
        coin_daily = df.set_index(ts_col).resample("1D").agg({"close": "last"}).dropna()
        if len(coin_daily) < _MIN_DAILY_BARS:
            continue
        coin_daily["ema20"] = coin_daily["close"].ewm(span=_DAILY_EMA_SPAN, adjust=False).mean()
        coin_daily["above"] = coin_daily["close"] > coin_daily["ema20"]
        for ts, row in coin_daily.iterrows():
            d = ts.date()
            slot = coin_above_total.setdefault(d, [0, 0])
            slot[1] += 1
            if bool(row["above"]):
                slot[0] += 1

    out: dict[date, dict[str, Any]] = {}
    for d in sorted(coin_above_total.keys()):
        above, total = coin_above_total[d]
        breadth = above / total if total > 0 else 0.5
        btc_row = btc_daily[btc_daily.index.date == d]
        if len(btc_row) > 0:
            btc_bear = bool(btc_row["bearish"].iloc[0])
            btc_bull = bool(btc_row["bullish"].iloc[0])
        else:
            btc_bear = False
            btc_bull = False
        out[d] = {
            "btc_bearish": btc_bear,
            "btc_bullish": btc_bull,
            "breadth": breadth,
            "breadth_low": breadth < _BREADTH_SHORT_MAX,
            "breadth_high": breadth > _BREADTH_LONG_MIN,
            "allowed": btc_bear and breadth < _BREADTH_SHORT_MAX,
            "allowed_long": btc_bull and breadth > _BREADTH_LONG_MIN,
        }
    return out
