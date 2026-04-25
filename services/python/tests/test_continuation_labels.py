"""Unit tests for src/ml/continuation/labels.py.

Covers all four labeling branches:
  1. Continuation: TP-side breached before SL-side.
  2. Drawup-first: SL-side (rally) breached before TP-side (drop).
  3. Stalled: neither threshold reached within `max_bars`.
  4. Ran-out: not enough post-entry candles in the snapshot to evaluate.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.ml.continuation.labels import (
    LabelConfig,  # noqa: F401  — exposed for downstream callers
    filter_labelable,
    make_continuation_label,
)


def _candles(symbol: str, start: datetime, highs: list[float], lows: list[float]) -> pd.DataFrame:
    """Build a synthetic candles frame for one symbol."""
    n = len(highs)
    assert len(lows) == n
    ts = [start + timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "asset": [symbol] * n,
            "timestamp": pd.to_datetime(ts, utc=True),
            "open": [(h + l) / 2 for h, l in zip(highs, lows)],
            "high": highs,
            "low": lows,
            "close": [(h + l) / 2 for h, l in zip(highs, lows)],
            "volume": [1.0] * n,
        }
    )


def _trade_row(
    symbol: str = "TESTUSDT",
    entry_time: datetime | None = None,
    entry_price: float = 100.0,
) -> pd.DataFrame:
    if entry_time is None:
        entry_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return pd.DataFrame(
        {
            "symbol": [symbol],
            "entry_time": [entry_time],
            "entry_price": [entry_price],
        }
    )


# ---------- continuation: drop hits TP within window --------------------------


def test_label_continuation_drop_to_target():
    """Short entered at 100; price grinds down to 80 (>=18% drop) without
    a >4% rally first → label = 1."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    # 10 pre-entry bars + post-entry bars
    pre_h = [101.0] * 10
    pre_l = [99.0] * 10
    # post-entry: stays under 104 (no >4% rally) and dips to 81 by bar 30 (>18% drop).
    post_h = [102.0] * 30
    post_l = [99.0 - i * 0.7 for i in range(30)]  # eventually goes to ~78
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    # entry at the 10th (last pre) bar: timestamp = start + 9*15m
    entry_ts = start + timedelta(minutes=15 * 9)
    trades = _trade_row(entry_time=entry_ts, entry_price=100.0)
    labels = make_continuation_label(trades, candles)
    assert labels.iloc[0] == 1


def test_label_drawup_first():
    """Short at 100; price rallies to 105 (>4% drawup) BEFORE dropping → label = 0."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h = [100.5] * 10
    pre_l = [99.5] * 10
    # post-entry: bar 0 spikes high to 105, then drops sharply.
    post_h = [105.0] + [101.0] * 19
    post_l = [99.0] + [80.0] * 19
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = start + timedelta(minutes=15 * 9)
    trades = _trade_row(entry_time=entry_ts, entry_price=100.0)
    labels = make_continuation_label(trades, candles)
    assert labels.iloc[0] == 0


def test_label_stalled_within_window():
    """Neither 18% drop nor 4% rally hit within max_bars → label = 0 (stalled)."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h = [100.5] * 10
    pre_l = [99.5] * 10
    # post-entry: chop between 99 and 102 forever.
    post_h = [102.0] * 200
    post_l = [99.0] * 200
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = start + timedelta(minutes=15 * 9)
    trades = _trade_row(entry_time=entry_ts, entry_price=100.0)
    labels = make_continuation_label(trades, candles, max_bars=192)
    assert labels.iloc[0] == 0


def test_label_ran_out_of_bars():
    """Entry near the end of the candle series: not enough post-entry bars to
    evaluate the label horizon. Returns -1 (un-labelable)."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    n = 20
    h = [101.0] * n
    l = [99.0] * n
    candles = _candles("TESTUSDT", start, h, l)
    # Entry at the very last bar — only 0 future bars available.
    entry_ts = candles["timestamp"].iloc[-1]
    trades = _trade_row(entry_time=entry_ts, entry_price=100.0)
    labels = make_continuation_label(trades, candles, max_bars=192)
    assert labels.iloc[0] == -1


def test_filter_labelable_drops_too_late_entries():
    """Trades whose entry_time + max_bars*15m exceed snapshot end are filtered out."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot_end = start + timedelta(hours=24)
    trades = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "entry_time": [
                start + timedelta(hours=1),                  # plenty of room
                start + timedelta(hours=23),                 # 1h to end -- too late
                start + timedelta(hours=15),                 # 9h to end, max_bars=192=48h -- too late
            ],
            "entry_price": [100, 100, 100],
        }
    )
    mask = filter_labelable(trades, snapshot_end, max_bars=192)
    assert list(mask) == [False, False, False]
    # With smaller horizon (4h), the first one fits.
    mask4 = filter_labelable(trades, snapshot_end, max_bars=16)
    assert list(mask4) == [True, False, True]


def test_label_skips_unknown_symbol():
    """Trade for a symbol absent from candles → label = -1, not crash."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = _candles("OTHERUSDT", start, [101] * 200, [99] * 200)
    entry_ts = start + timedelta(hours=1)
    trades = _trade_row(symbol="MISSINGUSDT", entry_time=entry_ts, entry_price=100.0)
    labels = make_continuation_label(trades, candles)
    assert labels.iloc[0] == -1


def test_label_sl_beats_tp_in_same_bar():
    """If a single bar hits both the drawup_cap and the fwd_target, SL is
    checked first (matches simulate_trade conservative ordering) → label = 0."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h = [100.5] * 10
    pre_l = [99.5] * 10
    # post-entry bar 0: hi=110 (rally trigger), lo=80 (target trigger). SL wins.
    post_h = [110.0] + [101.0] * 19
    post_l = [80.0] + [99.0] * 19
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = start + timedelta(minutes=15 * 9)
    trades = _trade_row(entry_time=entry_ts, entry_price=100.0)
    labels = make_continuation_label(trades, candles)
    assert labels.iloc[0] == 0
