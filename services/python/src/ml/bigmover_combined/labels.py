"""Direction-aware continuation labels for bigmover trades.

For a SHORT entry:
  Positive (continuation=1) iff price falls >= fwd_target from entry within
  max_bars BEFORE any rally of > drawdown_cap from entry.
For a LONG entry:
  Positive iff price rises >= fwd_target from entry within max_bars BEFORE
  any drop of > drawdown_cap from entry.

Defaults (per plan): fwd_target=0.10, drawdown_cap=0.04, max_bars=192 (=48h
on 15m). The bigmover detector's median holding period is shorter than
Strategy B's, so the target threshold is relaxed from 0.18 to 0.10.

The label uses FUTURE bars by design — only used for training, never as a
feature.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    fwd_target: float = 0.10
    drawdown_cap: float = 0.04
    max_bars: int = 192


def make_label(
    trades_df: pd.DataFrame,
    candles: pd.DataFrame,
    *,
    fwd_target: float = 0.10,
    drawdown_cap: float = 0.04,
    max_bars: int = 192,
    symbol_col: str = "symbol",
    entry_time_col: str = "entry_time",
    entry_price_col: str = "entry_price",
    direction_col: str = "direction",
    candle_symbol_col: str = "asset",
    candle_ts_col: str = "timestamp",
) -> pd.Series:
    """Direction-aware continuation label for bigmover trades.

    Returns:
      pd.Series of int8 (0/1), aligned to `trades_df.index`. Trades whose
      label cannot be determined (insufficient post-entry bars in the snapshot
      OR direction not in {'short','long'}) yield -1; the caller drops those
      rows before training.
    """
    if len(trades_df) == 0:
        return pd.Series([], dtype=np.int8)

    by_symbol: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    cs = candles.sort_values([candle_symbol_col, candle_ts_col])
    for sym, sub in cs.groupby(candle_symbol_col, sort=False):
        ts = pd.to_datetime(sub[candle_ts_col], utc=True).to_numpy()
        by_symbol[sym] = (
            ts,
            sub["high"].to_numpy(dtype=float),
            sub["low"].to_numpy(dtype=float),
        )

    entry_ts = pd.to_datetime(trades_df[entry_time_col], utc=True).to_numpy()
    entry_px = trades_df[entry_price_col].to_numpy(dtype=float)
    symbols = trades_df[symbol_col].to_numpy()
    directions = trades_df[direction_col].astype(str).to_numpy()

    out = np.full(len(trades_df), -1, dtype=np.int8)

    for i in range(len(trades_df)):
        sym = symbols[i]
        d = directions[i]
        if sym not in by_symbol or d not in ("short", "long"):
            continue
        if entry_px[i] <= 0:
            continue

        ts_arr, high_arr, low_arr = by_symbol[sym]
        start = int(np.searchsorted(ts_arr, entry_ts[i], side="right"))
        end = min(start + max_bars, len(ts_arr))
        if start >= end:
            continue

        ep = entry_px[i]
        if d == "short":
            # SHORT: favorable = price falls; adverse = price rises.
            adverse_trigger = ep * (1.0 + drawdown_cap)   # rally hits SL-side
            favorable_trigger = ep * (1.0 - fwd_target)   # drop hits TP-side
            for j in range(start, end):
                if high_arr[j] >= adverse_trigger:
                    out[i] = 0
                    break
                if low_arr[j] <= favorable_trigger:
                    out[i] = 1
                    break
            else:
                out[i] = 0
        else:  # long
            # LONG: favorable = price rises; adverse = price falls.
            adverse_trigger = ep * (1.0 - drawdown_cap)
            favorable_trigger = ep * (1.0 + fwd_target)
            for j in range(start, end):
                if low_arr[j] <= adverse_trigger:
                    out[i] = 0
                    break
                if high_arr[j] >= favorable_trigger:
                    out[i] = 1
                    break
            else:
                out[i] = 0

    return pd.Series(out, index=trades_df.index, name="continuation")


def filter_labelable(
    trades_df: pd.DataFrame,
    candles_max_ts: pd.Timestamp,
    *,
    max_bars: int = 192,
    bars_per_step_minutes: int = 15,
    entry_time_col: str = "entry_time",
) -> pd.Series:
    """Boolean mask: True for trades whose label horizon fits in the snapshot."""
    if len(trades_df) == 0:
        return pd.Series([], dtype=bool)
    entry_ts = pd.to_datetime(trades_df[entry_time_col], utc=True)
    horizon = pd.Timedelta(minutes=max_bars * bars_per_step_minutes)
    return entry_ts + horizon <= candles_max_ts
