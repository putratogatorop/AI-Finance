"""Tests for backtest_v2_resized.py — deterministic units."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from backtest_v2_resized import (  # noqa: E402
    _kelly_fraction_for,
    build_kelly_table,
    monte_carlo_pf,
    simulate,
)


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
    # pandas 2.x uses us precision by default; older used ns. Accept either.
    dtype_str = str(df["signal_time"].dtype)
    assert pd.api.types.is_datetime64_any_dtype(df["signal_time"]) and "UTC" in dtype_str


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


def test_kelly_fraction_for_maps_to_correct_bin():
    tbl = pd.DataFrame([
        {"bin_low": 0.70, "bin_high": 0.75, "f_capped": 0.03},
        {"bin_low": 0.75, "bin_high": 0.80, "f_capped": 0.05},
        {"bin_low": 0.80, "bin_high": 0.85, "f_capped": 0.10},
        {"bin_low": 0.85, "bin_high": 0.90, "f_capped": 0.15},
        {"bin_low": 0.90, "bin_high": 1.01, "f_capped": 0.15},
    ])
    assert _kelly_fraction_for(0.72, tbl) == 0.03
    assert _kelly_fraction_for(0.80, tbl) == 0.10  # inclusive-left
    assert _kelly_fraction_for(0.95, tbl) == 0.15


def test_kelly_fraction_below_range_uses_floor():
    """ml_prob below min bin (shouldn't happen given threshold, defensive)."""
    tbl = pd.DataFrame([
        {"bin_low": 0.70, "bin_high": 0.75, "f_capped": 0.03},
    ])
    assert _kelly_fraction_for(0.50, tbl) == 0.02  # KELLY_FRACTION_FLOOR


def test_simulate_flat_sizing_when_kelly_table_all_equal():
    """When kelly_table has same f_capped in every bin, result = flat sizing."""
    rows = [
        ("2025-06-01T00:00:00Z", 0.80, 0.03, 20),   # +3%
        ("2025-06-01T06:00:00Z", 0.85, -0.01, 20),  # -1%
    ]
    df = _make_trades(rows)
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": 0.05}
        for lo, hi in zip([0.70, 0.75, 0.80, 0.85, 0.90], [0.75, 0.80, 0.85, 0.90, 1.01])
    ])
    ledger, metrics = simulate(df, flat_tbl, max_concurrent=5, label="test")
    # Both trades taken: trade 1 at 0.05 * 600 * 5 = 150 notional, pnl net = 0.03-0.0012 = 0.0288
    # trade 1 pnl_usd = 150 * 0.0288 = 4.32
    # trade 2 pnl net = -0.01 - 0.0012 = -0.0112, pnl_usd = 150 * -0.0112 = -1.68
    assert len(ledger) == 2
    assert ledger.iloc[0]["pnl_usd"] == pytest.approx(4.32, abs=0.01)
    assert ledger.iloc[1]["pnl_usd"] == pytest.approx(-1.68, abs=0.01)
    assert metrics["trades"] == 2
    assert metrics["wr"] == 0.5


def test_simulate_concurrency_drops_overlapping_signals():
    """When MAX_CONCURRENT=1, second simultaneous signal is dropped."""
    rows = [
        ("2025-06-01T00:00:00Z", 0.80, 0.03, 20),   # exits at 00:00 + 20*15 = 05:00
        ("2025-06-01T01:00:00Z", 0.80, 0.05, 20),   # opens before first exits
        ("2025-06-01T10:00:00Z", 0.80, 0.02, 20),   # opens after first exits
    ]
    df = _make_trades(rows)
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": 0.05}
        for lo, hi in zip([0.70, 0.75, 0.80, 0.85, 0.90], [0.75, 0.80, 0.85, 0.90, 1.01])
    ])
    ledger, metrics = simulate(df, flat_tbl, max_concurrent=1, label="test")
    # Trade 1 taken. Trade 2 dropped (concurrent). Trade 3 taken.
    assert metrics["trades"] == 2
    assert metrics["skipped_concurrency"] == 1


def test_simulate_kelly_fractions_scale_notional():
    """Two trades with different ml_probs get proportionally different notionals."""
    rows = [
        ("2025-06-01T00:00:00Z", 0.72, 0.03, 20),   # f=0.02 -> 60 notional
        ("2025-06-01T10:00:00Z", 0.92, 0.03, 20),   # f=0.10 -> 300 notional
    ]
    df = _make_trades(rows)
    stepped_tbl = pd.DataFrame([
        {"bin_low": 0.70, "bin_high": 0.75, "f_capped": 0.02},
        {"bin_low": 0.75, "bin_high": 0.80, "f_capped": 0.05},
        {"bin_low": 0.80, "bin_high": 0.85, "f_capped": 0.05},
        {"bin_low": 0.85, "bin_high": 0.90, "f_capped": 0.05},
        {"bin_low": 0.90, "bin_high": 1.01, "f_capped": 0.10},
    ])
    ledger, metrics = simulate(df, stepped_tbl, max_concurrent=5, label="test")
    # Trade 1: notional = 600 * 0.02 * 5 = 60. pnl_usd = 60 * 0.0288 = 1.728
    # Trade 2: notional = 600 * 0.10 * 5 = 300. pnl_usd = 300 * 0.0288 = 8.64
    assert ledger.iloc[0]["notional_usd"] == pytest.approx(60.0)
    assert ledger.iloc[1]["notional_usd"] == pytest.approx(300.0)
    assert ledger.iloc[1]["pnl_usd"] == pytest.approx(8.64, abs=0.01)


def test_monte_carlo_pf_reproducible_with_seed():
    pnls = np.array([0.03, -0.01, 0.04, -0.01, 0.02, -0.01])
    p5_a, p50_a, p95_a = monte_carlo_pf(pnls, iters=100, seed=42)
    p5_b, p50_b, p95_b = monte_carlo_pf(pnls, iters=100, seed=42)
    assert p5_a == p5_b
    assert p50_a == p50_b
    assert p95_a == p95_b


def test_monte_carlo_pf_all_winners_returns_inf():
    pnls = np.array([0.01, 0.02, 0.03, 0.04])
    p5, p50, p95 = monte_carlo_pf(pnls, iters=50, seed=42)
    # No losers in any bootstrap sample -> inf
    assert p5 == float("inf")
    assert p50 == float("inf")


def test_monte_carlo_pf_quantile_order():
    """p5 <= p50 <= p95 always."""
    pnls = np.array([0.03, -0.01, 0.04, -0.01, 0.02, -0.01, 0.05, -0.02])
    p5, p50, p95 = monte_carlo_pf(pnls, iters=500, seed=42)
    # Filter out inf cases for ordering assertion
    if p95 != float("inf"):
        assert p5 <= p50 <= p95


from backtest_v2_resized import threshold_kelly_crosscheck  # noqa: E402


def test_threshold_kelly_crosscheck_returns_row_per_threshold():
    rows = [
        ("2025-06-01T00:00:00Z", 0.72, 0.03, 20),
        ("2025-06-01T10:00:00Z", 0.82, 0.04, 20),
        ("2025-06-02T00:00:00Z", 0.92, 0.05, 20),
    ]
    df = _make_trades(rows)
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": 0.05}
        for lo, hi in zip([0.70, 0.75, 0.80, 0.85, 0.90], [0.75, 0.80, 0.85, 0.90, 1.01])
    ])
    cc = threshold_kelly_crosscheck(df, flat_tbl, thresholds=[0.70, 0.80, 0.90])
    assert list(cc["threshold"]) == [0.70, 0.80, 0.90]
    assert set(cc.columns) >= {"threshold", "A_trades", "A_ret_pct", "B_ret_pct", "C_ret_pct"}
    # At threshold 0.70 all 3 trades pass; at 0.90 only one.
    assert cc.iloc[0]["A_trades"] == 3
    assert cc.iloc[2]["A_trades"] == 1
