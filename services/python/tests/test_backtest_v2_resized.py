"""Tests for backtest_v2_resized.py — deterministic units."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


def _make_trades(rows):
    """Build a trades DataFrame from rows of (signal_time, ml_prob, pnl_pct, bars_held).

    signal_time is parsed as ISO string. Index is reset.
    """
    df = pd.DataFrame(
        rows,
        columns=["signal_time", "ml_prob", "pnl_pct", "bars_held"],
    )
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    return df.reset_index(drop=True)


def test_fixture_helper_builds_expected_shape():
    df = _make_trades([
        ("2025-06-01T00:00:00Z", 0.80, 0.03, 20),
        ("2025-06-01T00:15:00Z", 0.92, 0.05, 15),
    ])
    assert len(df) == 2
    assert list(df.columns) == ["signal_time", "ml_prob", "pnl_pct", "bars_held"]
    assert df["signal_time"].dtype == "datetime64[ns, UTC]"
