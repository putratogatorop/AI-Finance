"""Direction-aware continuation label tests."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from src.ml.bigmover_combined.labels import filter_labelable, make_label


def _candles(symbol: str, start: datetime, highs: list[float], lows: list[float]) -> pd.DataFrame:
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


def _trade(direction: str, entry_time: datetime, entry_price: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "symbol": ["TESTUSDT"],
            "entry_time": [entry_time],
            "entry_price": [entry_price],
            "direction": [direction],
        }
    )


# ---- SHORT direction ---------------------------------------------------------


def test_short_continuation_drop_to_target():
    """Short at 100; price grinds down to 88 (>=10% drop) without a >4% rally → label=1."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h = [101.0] * 5
    pre_l = [99.0] * 5
    # 30 post bars: stay below 104 (no >4% rally), reach 88 by bar 25.
    post_h = [102.0] * 30
    post_l = [99.0 - i * 0.5 for i in range(30)]  # eventually ~84
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = candles["timestamp"].iloc[4]  # last pre-entry bar
    trades = _trade("short", entry_ts, 100.0)
    assert make_label(trades, candles).iloc[0] == 1


def test_short_drawup_first_kills_label():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h, pre_l = [100.5] * 5, [99.5] * 5
    # post bar 0 spikes high to 105 (>4% rally) before any drop.
    post_h = [105.0] + [101.0] * 19
    post_l = [99.0] + [80.0] * 19
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = candles["timestamp"].iloc[4]
    trades = _trade("short", entry_ts, 100.0)
    assert make_label(trades, candles).iloc[0] == 0


# ---- LONG direction ----------------------------------------------------------


def test_long_continuation_rally_to_target():
    """Long at 100; price grinds up to 112 without a >4% drop → label=1."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h, pre_l = [101.0] * 5, [99.0] * 5
    post_h = [101.0 + i * 0.5 for i in range(30)]
    post_l = [99.0] * 30  # never drops >4% from 100
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = candles["timestamp"].iloc[4]
    trades = _trade("long", entry_ts, 100.0)
    assert make_label(trades, candles).iloc[0] == 1


def test_long_drawdown_first_kills_label():
    """Long at 100; price drops to 95 (>4%) before reaching +10% → label=0."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h, pre_l = [100.5] * 5, [99.5] * 5
    post_h = [101.0] + [115.0] * 19
    post_l = [95.0] + [99.0] * 19  # bar 0 dips to 95 (>4% drop) before any rally
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = candles["timestamp"].iloc[4]
    trades = _trade("long", entry_ts, 100.0)
    assert make_label(trades, candles).iloc[0] == 0


# ---- common ------------------------------------------------------------------


def test_label_stalled_in_window():
    """Neither target nor cap reached within max_bars → label=0 (stalled)."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pre_h, pre_l = [100.5] * 5, [99.5] * 5
    post_h = [101.0] * 200
    post_l = [99.0] * 200
    candles = _candles("TESTUSDT", start, pre_h + post_h, pre_l + post_l)
    entry_ts = candles["timestamp"].iloc[4]
    trades = _trade("short", entry_ts, 100.0)
    assert make_label(trades, candles, max_bars=192).iloc[0] == 0


def test_unknown_direction_yields_minus_one():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = _candles("TESTUSDT", start, [101] * 200, [99] * 200)
    trades = _trade("flat", candles["timestamp"].iloc[5], 100.0)
    assert make_label(trades, candles).iloc[0] == -1


def test_unknown_symbol_yields_minus_one():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = _candles("OTHERUSDT", start, [101] * 200, [99] * 200)
    trades = pd.DataFrame(
        {
            "symbol": ["MISSINGUSDT"],
            "entry_time": [candles["timestamp"].iloc[5]],
            "entry_price": [100.0],
            "direction": ["short"],
        }
    )
    assert make_label(trades, candles).iloc[0] == -1


def test_label_ran_out_of_bars():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    candles = _candles("TESTUSDT", start, [101] * 20, [99] * 20)
    entry_ts = candles["timestamp"].iloc[-1]  # last bar = no future bars
    trades = _trade("short", entry_ts, 100.0)
    assert make_label(trades, candles, max_bars=192).iloc[0] == -1


def test_filter_labelable_drops_too_late_entries():
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    snapshot_end = start + timedelta(hours=24)
    trades = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "entry_time": [
                start + timedelta(hours=1),
                start + timedelta(hours=23),
                start + timedelta(hours=15),
            ],
            "entry_price": [100, 100, 100],
            "direction": ["short", "long", "short"],
        }
    )
    mask48 = filter_labelable(trades, snapshot_end, max_bars=192)
    assert list(mask48) == [False, False, False]
    mask4 = filter_labelable(trades, snapshot_end, max_bars=16)
    assert list(mask4) == [True, False, True]
