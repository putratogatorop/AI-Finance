"""Continuation labeling for short-side trades.

For each Strategy-B short trade row, we want a binary label:
  1 = "the move continued" — price fell >= fwd_target from entry within
      max_bars *before* any rally of > drawup_cap from entry.
  0 = "stalled / reverted" — otherwise (drawup hit first, or neither
      threshold reached within the window).

Justification (from signal-timing analyst's diagnostic on 2026-04-01):
winners' moves extend to ~23% from peak, losers stall at ~13% — implying
~18% as the discriminator on the post-entry leg. drawup_cap=4% sits below
the 5% baseline stop-loss so the label respects what would-be live
execution survives. max_bars=192 (=48h) brackets B's 133-bar median hold.

The labeler uses FUTURE bars by design — only used for training. Features
in features.py must NEVER read post-entry data. See tests/test_labels.py
for the four labeling branches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelConfig:
    """Knobs for continuation labeling. Defaults reflect the v1 plan."""
    fwd_target: float = 0.18      # required favorable move from entry (drop, since short)
    drawup_cap: float = 0.04      # adverse move from entry that, if hit first, kills the label
    max_bars: int = 192           # 48h on 15m


def make_continuation_label(
    trades_df: pd.DataFrame,
    candles: pd.DataFrame,
    *,
    fwd_target: float = 0.18,
    drawup_cap: float = 0.04,
    max_bars: int = 192,
    symbol_col: str = "symbol",
    entry_time_col: str = "entry_time",
    entry_price_col: str = "entry_price",
    candle_symbol_col: str = "asset",
    candle_ts_col: str = "timestamp",
) -> pd.Series:
    """Compute the continuation label for every row in `trades_df`.

    Args:
      trades_df: one row per trade with columns `symbol`, `entry_time`, `entry_price`.
      candles: 15m OHLC parquet — must have `asset`, `timestamp`, `high`, `low`.
      fwd_target: required favorable price drop from entry (e.g. 0.18 = 18%).
      drawup_cap: adverse rally from entry that disqualifies the label.
      max_bars: scan horizon in 15m bars.

    Returns:
      pd.Series of int (0/1), same index/length as `trades_df`. Trades whose
      label cannot be determined (insufficient post-entry bars in the snapshot)
      yield -1 — caller is expected to drop those rows before training.
    """
    if len(trades_df) == 0:
        return pd.Series([], dtype=np.int8)

    # Index candles for fast per-symbol lookups:
    #   dict[symbol] -> (ts ndarray, high ndarray, low ndarray)
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

    out = np.full(len(trades_df), -1, dtype=np.int8)

    for i in range(len(trades_df)):
        sym = symbols[i]
        if sym not in by_symbol:
            continue
        ts_arr, high_arr, low_arr = by_symbol[sym]
        # First post-entry bar index. searchsorted returns insertion point so
        # `right` gives the first ts strictly after entry_ts.
        start = int(np.searchsorted(ts_arr, entry_ts[i], side="right"))
        end = min(start + max_bars, len(ts_arr))
        if start >= end:
            continue
        if entry_px[i] <= 0:
            continue

        ep = entry_px[i]
        sl_trigger_price = ep * (1.0 + drawup_cap)  # short adverse: price up
        tp_trigger_price = ep * (1.0 - fwd_target)  # short favorable: price down
        # First-hit semantics: SL beats TP at the same bar (matches simulate_trade).
        for j in range(start, end):
            if high_arr[j] >= sl_trigger_price:
                out[i] = 0
                break
            if low_arr[j] <= tp_trigger_price:
                out[i] = 1
                break
        else:
            # Window exhausted without hitting either: stalled = label 0.
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
    """Boolean mask: True for trades whose label horizon fits in the snapshot.

    Trades with `entry_time + max_bars*15m > snapshot_end` cannot have a
    well-defined label — they may be in a continuation that the snapshot just
    doesn't cover. Drop them before computing labels.
    """
    if len(trades_df) == 0:
        return pd.Series([], dtype=bool)
    entry_ts = pd.to_datetime(trades_df[entry_time_col], utc=True)
    horizon = pd.Timedelta(minutes=max_bars * bars_per_step_minutes)
    return entry_ts + horizon <= candles_max_ts
