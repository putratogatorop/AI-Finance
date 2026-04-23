"""Tests for backtest_v2_resized.py — deterministic units."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_v2_resized import build_kelly_table  # noqa: E402


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


def test_kelly_table_single_bin_all_winners_at_max_cap():
    """All wins in one bin + no losses: p=1, b=inf, kelly=1.0, clipped to 0.15."""
    rows = [
        ("2025-01-01T00:00:00Z", 0.92, 0.05, 20),
        ("2025-01-01T00:15:00Z", 0.93, 0.04, 20),
        ("2025-01-01T00:30:00Z", 0.94, 0.06, 20),
        ("2025-01-01T00:45:00Z", 0.95, 0.05, 20),
    ]
    df = _make_trades(rows)
    tbl = build_kelly_table(df)
    row_0_90 = tbl[tbl["bin_low"] == 0.90].iloc[0]
    assert row_0_90["p"] == 1.0
    assert row_0_90["f_capped"] == 0.15


def test_kelly_table_empty_bin_uses_floor():
    """Bin with zero samples (or insufficient) falls back to floor 0.02."""
    rows = [
        ("2025-01-01T00:00:00Z", 0.92, 0.05, 20),
        ("2025-01-01T00:15:00Z", 0.93, 0.04, 20),
    ]
    df = _make_trades(rows)
    tbl = build_kelly_table(df)
    # [0.70, 0.75) bin has no data
    row_low = tbl[tbl["bin_low"] == 0.70].iloc[0]
    assert row_low["f_capped"] == 0.02


def test_kelly_table_known_p_b_example():
    """Bin with p=0.8, win=+3%, loss=-1% => b=3, Kelly=0.8 - 0.2/3 = 0.733.
    Half-Kelly = 0.366, clipped to 0.15."""
    rows = []
    # 8 wins at 3%
    for i in range(8):
        rows.append((f"2025-01-01T00:{i:02d}:00Z", 0.82, 0.03, 20))
    # 2 losses at -1%
    for i in range(2):
        rows.append((f"2025-01-01T01:{i:02d}:00Z", 0.82, -0.01, 20))
    df = _make_trades(rows)
    tbl = build_kelly_table(df)
    row = tbl[tbl["bin_low"] == 0.80].iloc[0]
    assert row["p"] == pytest.approx(0.8)
    assert row["b"] == pytest.approx(3.0)
    # f_raw = 0.8 - 0.2/3 = 0.7333
    # f_half = 0.3667 -> clipped to 0.15
    assert row["f_raw"] == pytest.approx(0.7333, abs=1e-3)
    assert row["f_capped"] == 0.15


def test_kelly_table_returns_all_bins():
    """Output always has one row per bin edge pair, in order."""
    rows = [("2025-01-01T00:00:00Z", 0.75, 0.03, 20)]
    df = _make_trades(rows)
    tbl = build_kelly_table(df)
    # 5 bins: [0.70,0.75) [0.75,0.80) [0.80,0.85) [0.85,0.90) [0.90,1.01)
    assert len(tbl) == 5
    assert list(tbl["bin_low"]) == [0.70, 0.75, 0.80, 0.85, 0.90]
