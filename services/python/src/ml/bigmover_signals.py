"""Leak-free detectors for the bigmover signal family.

Three signals: `baseline` (vol-spike + directional dominance), `price_accel_atr`
(baseline gated by |close acceleration| >= k * ATR), and `multi_bar_confirm`
(baseline on bar j-1 confirmed by bar j moving in the trade direction).

Every mask returned by `detect_signal` here is computable from bars with
`timestamp <= bar_j`, where `j` is the mask's own index. Callers that enter at
`open_[j+1]` (the next bar's open) are guaranteed leak-free. The original copies
of these detectors in five `scripts/backtest_bigmover_*.py` files leaked two bars
— `multi_bar_confirm` via `close.shift(-1)`, and all three signals by entering at
`open_[i]` on the same bar whose `close[i]` built the mask.

Constants are module-level so callers can import them unchanged. Where a script
needs an override (e.g. a different ATR multiplier), `detect_signal` accepts
`accel_mult_of_atr` as a keyword.

Imports: numpy + pandas only. Safe to import from any script or test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Signal-parameter defaults — match the original 3y / grid / sweep scripts.
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
ACCEL_MULT_OF_ATR = 1.0


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def baseline_entry_mask_short(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(df)
    out = pd.Series([False] * n, index=df.index)
    if n < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return out
    vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
    rolling_high = high.rolling(PRICE_LOOKBACK).max()
    rolling_low = low.rolling(PRICE_LOOKBACK).min()
    price_drop = (rolling_high - close) / rolling_high
    price_rise = (close - rolling_low) / rolling_low
    vol_ratio = volume / vol_ma
    return (
        (vol_ratio >= VOL_SPIKE)
        & (price_drop >= PRICE_MOVE_THRESH)
        & (price_drop > price_rise)
    ).fillna(False)


def baseline_entry_mask_long(df: pd.DataFrame) -> pd.Series:
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(df)
    out = pd.Series([False] * n, index=df.index)
    if n < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return out
    vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
    rolling_high = high.rolling(PRICE_LOOKBACK).max()
    rolling_low = low.rolling(PRICE_LOOKBACK).min()
    price_drop = (rolling_high - close) / rolling_high
    price_rise = (close - rolling_low) / rolling_low
    vol_ratio = volume / vol_ma
    return (
        (vol_ratio >= VOL_SPIKE)
        & (price_rise >= PRICE_MOVE_THRESH)
        & (price_rise > price_drop)
    ).fillna(False)


def detect_signal(
    df: pd.DataFrame,
    signal: str,
    direction: str,
    atr: pd.Series,
    *,
    accel_mult_of_atr: float = ACCEL_MULT_OF_ATR,
) -> pd.Series:
    """Dispatch to one of the three signals. Mask[j] references only bars <= j."""
    if direction == "short":
        base = baseline_entry_mask_short(df)
    elif direction == "long":
        base = baseline_entry_mask_long(df)
    else:
        raise ValueError(f"unknown direction: {direction}")

    if signal == "baseline":
        return base

    if signal == "price_accel_atr":
        prev_close = df["close"].shift(1)
        prev_prev = df["close"].shift(2)
        accel = (df["close"] - prev_close) - (prev_close - prev_prev)
        return (base & (accel.abs() >= atr * accel_mult_of_atr)).fillna(False)

    if signal == "multi_bar_confirm":
        # Baseline fires at bar j-1; confirmation is bar j closing in the trade
        # direction. Mask[j] is thus computable at close of bar j using bars <= j.
        prev_close = df["close"].shift(1)
        prev_base = base.shift(1, fill_value=False)
        if direction == "short":
            confirm = df["close"] < prev_close
        else:
            confirm = df["close"] > prev_close
        return (prev_base & confirm).fillna(False)

    raise ValueError(f"unknown signal: {signal}")


def _self_check_no_lookahead() -> None:
    """Mutate future bars in a synthetic df; assert past mask values unchanged.

    Runs all three signals in both directions. Any look-ahead leak (positive-offset
    shift, centered rolling, intrabar fill assumption) would make mask values at
    indices < `cut` depend on bars >= `cut`.
    """
    rng = np.random.default_rng(12345)
    n_bars = PRICE_LOOKBACK + VOL_MA_PERIOD + 100
    base_close = 100 + rng.normal(0, 1, n_bars).cumsum()
    df = pd.DataFrame({
        "timestamp": np.arange(n_bars, dtype=np.int64) * 900,
        "open": base_close,
        "high": base_close + np.abs(rng.normal(0, 0.5, n_bars)),
        "low": base_close - np.abs(rng.normal(0, 0.5, n_bars)),
        "close": base_close,
        "volume": rng.uniform(1, 10, n_bars),
    })

    cut = n_bars - 30
    df_mut = df.copy()
    df_mut.loc[cut:, "close"] = df_mut.loc[cut:, "close"] * 3.0
    df_mut.loc[cut:, "open"] = df_mut.loc[cut:, "open"] * 3.0
    df_mut.loc[cut:, "high"] = df_mut.loc[cut:, "high"] * 3.0
    df_mut.loc[cut:, "low"] = df_mut.loc[cut:, "low"] * 3.0
    df_mut.loc[cut:, "volume"] = df_mut.loc[cut:, "volume"] * 5.0

    atr_a = compute_atr(df)
    atr_b = compute_atr(df_mut)

    for signal in ("baseline", "price_accel_atr", "multi_bar_confirm"):
        for direction in ("short", "long"):
            mask_a = detect_signal(df, signal, direction, atr_a)
            mask_b = detect_signal(df_mut, signal, direction, atr_b)
            if not (mask_a.iloc[:cut].values == mask_b.iloc[:cut].values).all():
                first_diff = int(np.where(
                    mask_a.iloc[:cut].values != mask_b.iloc[:cut].values
                )[0][0])
                raise AssertionError(
                    f"LEAK REGRESSION ({signal}/{direction}): mask at past bars "
                    f"changed after mutating future bars. First diff idx={first_diff}"
                )
