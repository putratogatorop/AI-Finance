# v2+ML Resizing Analysis — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single research script that replays the v2+ML OOT trade stream under 3 sizing configurations (flat baseline, half-Kelly by ml_prob bin, half-Kelly + concurrency 10) and outputs a clean attribution table answering "how much PnL can we unlock with same trades, better capital allocation?"

**Architecture:** One script (`backtest_v2_resized.py`) that (a) loads pre-OOT + OOT trades from the existing `scanner_short_v2_ml_filtered` DB table, (b) calibrates a Kelly table from pre-OOT only, (c) simulates 3 scenarios deterministically, (d) runs Monte Carlo on each, (e) writes results to `results/<run_id>/` per the backtest protocol. Pure analysis — no new model, no new features, no prod impact.

**Tech Stack:** Python 3.12, pandas, numpy, sqlalchemy, pytest. All already in the project.

**Spec:** `docs/superpowers/specs/2026-04-23-v2-resizing-analysis-design.md`

## File Structure

- Create: `services/python/scripts/backtest_v2_resized.py` — the analysis script (~300 lines)
- Create: `services/python/tests/test_backtest_v2_resized.py` — pytest tests for deterministic units (Kelly, simulate, MC)

Pure testable units (no DB mocking needed):
- `build_kelly_table(pre_oot_df)` → DataFrame — given a deterministic input df, outputs deterministic bin table
- `simulate(trades_df, kelly_table, max_concurrent)` → (ledger, metrics) — deterministic given fixed inputs
- `monte_carlo_pf(pnl_series, iters, seed)` → (p5, p50, p95) — deterministic with seed
- `threshold_kelly_crosscheck(trades_df, kelly_table, thresholds)` → DataFrame

DB-dependent thin wrappers (smoke test only, not unit tested):
- `load_trades(engine)` → pd.DataFrame
- `main()` — orchestrator

---

### Task 1: Script Scaffold with Constants

**Files:**
- Create: `services/python/scripts/backtest_v2_resized.py`

Creates the empty script shell with module docstring, imports, constants, and a placeholder main(). No logic yet. Just the frame.

- [ ] **Step 1: Create the script file**

```python
"""v2+ML Resizing Analysis — 3-run layered attribution.

Re-simulates the v2+ML OOT trade stream under 3 sizing configurations:
  A (baseline)     flat 25% notional, MAX_CONCURRENT=5
  B (+Lever 1)     half-Kelly by ml_prob bin, MAX_CONCURRENT=5
  C (+Lever 1+4)   half-Kelly by ml_prob bin, MAX_CONCURRENT=10

Outputs PnL / PF / drawdown per scenario + monthly comparison +
threshold-Kelly crosscheck. Pure research, no prod impact.

Spec: docs/superpowers/specs/2026-04-23-v2-resizing-analysis-design.md

Run from services/python/:
    python scripts/backtest_v2_resized.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time as _time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, ".")

# --- CONSTANTS ---
RANDOM_SEED = 42

# Source
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)
SOURCE_TABLE = "scanner_short_v2_ml_filtered"

# Sizing (matches CLAUDE.md trading rules)
START_EQUITY_USD = 600.0
LEVERAGE = 5
BASELINE_EQUITY_FRACTION = 0.05  # 5% of equity per trade = 25% notional at 5x
KELLY_MULTIPLIER = 0.5           # half-Kelly, industry standard, hardcoded
KELLY_FRACTION_FLOOR = 0.02      # min 2% equity per trade
KELLY_FRACTION_CAP = 0.15        # max 15% equity per trade
ML_THRESHOLD = 0.70              # dashboard threshold (see spec for rationale)
ROUND_TRIP_FEE = 0.0012          # 0.06% taker x 2 legs (Gate.io futures)

# OOT split
HOLD_OUT_MONTHS = 3

# Scenarios
MAX_CONCURRENT_A = 5
MAX_CONCURRENT_B = 5
MAX_CONCURRENT_C = 10

# Kelly bins (edges inclusive-left, exclusive-right; last bin captures 0.95+)
KELLY_BIN_EDGES = [0.70, 0.75, 0.80, 0.85, 0.90, 1.01]

# Monte Carlo
MC_ITER = 1000
MC_SEED = 42

# Threshold crosscheck
CROSSCHECK_THRESHOLDS = [0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.88, 0.90]

# Bar size for exit_time derivation
BAR_MINUTES = 15


def main():
    raise NotImplementedError("Filled in by later tasks")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the script parses**

Run:
```
cd services/python && python -c "import scripts.backtest_v2_resized"
```
Expected: no output, no error. Just a clean import.

- [ ] **Step 3: Commit**

```
git add services/python/scripts/backtest_v2_resized.py
git commit -m "feat(backtest): scaffold v2 resizing analysis script"
```

---

### Task 2: Test File Scaffold + Synthetic Fixture Helper

**Files:**
- Create: `services/python/tests/test_backtest_v2_resized.py`

Creates the test module with a reusable fixture helper that fabricates a small DataFrame matching the shape of what the script will process. Keeps all tests deterministic.

- [ ] **Step 1: Create the test file**

```python
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
```

- [ ] **Step 2: Run test**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 1 passed.

- [ ] **Step 3: Commit**

```
git add services/python/tests/test_backtest_v2_resized.py
git commit -m "test(backtest): fixture helper for resizing analysis"
```

---

### Task 3: `build_kelly_table()` — Core Logic

**Files:**
- Modify: `services/python/scripts/backtest_v2_resized.py` — add function above `main()`
- Modify: `services/python/tests/test_backtest_v2_resized.py` — add tests

Given a DataFrame of pre-OOT trades with columns `ml_prob, pnl_pct`, bin by ml_prob and compute an empirical half-Kelly fraction per bin with floor/cap.

- [ ] **Step 1: Write failing tests**

Add to `test_backtest_v2_resized.py`:

```python
from backtest_v2_resized import build_kelly_table  # noqa: E402


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
```

- [ ] **Step 2: Run tests — expect failure (function not defined)**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 4 failures with `ImportError: cannot import name 'build_kelly_table'`.

- [ ] **Step 3: Implement `build_kelly_table()`**

Add to `scripts/backtest_v2_resized.py` above `main()`:

```python
def build_kelly_table(pre_oot_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bin half-Kelly fraction from pre-OOT trades.

    For each [bin_low, bin_high) bin in KELLY_BIN_EDGES:
      p = win_rate_in_bin (empirical)
      b = avg_winner_pnl / |avg_loser_pnl|  (empirical)
      f_raw = p - (1-p)/b                    (full Kelly)
      f_half = f_raw * KELLY_MULTIPLIER
      f_capped = clip(f_half, FLOOR, CAP)

    Empty bins (or bins with no losers / no winners) fall back to FLOOR.
    Returns DataFrame with columns:
      bin_low, bin_high, trades, p, b, f_raw, f_half, f_capped
    """
    rows = []
    for lo, hi in zip(KELLY_BIN_EDGES[:-1], KELLY_BIN_EDGES[1:], strict=False):
        mask = (pre_oot_df["ml_prob"] >= lo) & (pre_oot_df["ml_prob"] < hi)
        sub = pre_oot_df.loc[mask, "pnl_pct"]
        trades = len(sub)
        if trades == 0:
            rows.append({
                "bin_low": lo, "bin_high": hi, "trades": 0,
                "p": 0.0, "b": 0.0,
                "f_raw": 0.0, "f_half": 0.0, "f_capped": KELLY_FRACTION_FLOOR,
            })
            continue
        wins = sub[sub > 0]
        losses = sub[sub <= 0]
        p = len(wins) / trades
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        if avg_loss == 0 or len(losses) == 0:
            # No losers in this bin. Full Kelly is +inf; half still clips to CAP.
            f_raw, f_half = 1.0, KELLY_MULTIPLIER
            b = float("inf")
        elif avg_win == 0:
            f_raw, f_half, b = 0.0, 0.0, 0.0
        else:
            b = avg_win / avg_loss
            f_raw = p - (1.0 - p) / b
            f_half = f_raw * KELLY_MULTIPLIER
        f_capped = float(np.clip(f_half, KELLY_FRACTION_FLOOR, KELLY_FRACTION_CAP))
        rows.append({
            "bin_low": lo, "bin_high": hi, "trades": trades,
            "p": p, "b": b, "f_raw": f_raw, "f_half": f_half, "f_capped": f_capped,
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 5 passed (4 new + 1 fixture).

- [ ] **Step 5: Commit**

```
git add services/python/scripts/backtest_v2_resized.py services/python/tests/test_backtest_v2_resized.py
git commit -m "feat(backtest): build_kelly_table + tests"
```

---

### Task 4: `_kelly_fraction_for()` Helper

**Files:**
- Modify: `services/python/scripts/backtest_v2_resized.py`
- Modify: `services/python/tests/test_backtest_v2_resized.py`

Small utility to map a single ml_prob value to its bin's capped fraction.

- [ ] **Step 1: Write failing tests**

```python
from backtest_v2_resized import _kelly_fraction_for  # noqa: E402


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
```

- [ ] **Step 2: Run tests — expect import failure**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 2 new failures with `ImportError`.

- [ ] **Step 3: Implement `_kelly_fraction_for()`**

Add to `backtest_v2_resized.py` after `build_kelly_table()`:

```python
def _kelly_fraction_for(ml_prob: float, kelly_table: pd.DataFrame) -> float:
    """Map a single ml_prob to its bin's f_capped. Defensive floor for out-of-range."""
    for _, row in kelly_table.iterrows():
        if row["bin_low"] <= ml_prob < row["bin_high"]:
            return float(row["f_capped"])
    return KELLY_FRACTION_FLOOR
```

- [ ] **Step 4: Run tests**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 7 passed.

- [ ] **Step 5: Commit**

```
git add services/python/scripts/backtest_v2_resized.py services/python/tests/test_backtest_v2_resized.py
git commit -m "feat(backtest): _kelly_fraction_for lookup helper"
```

---

### Task 5: `simulate()` — Core Scenario Runner

**Files:**
- Modify: `services/python/scripts/backtest_v2_resized.py`
- Modify: `services/python/tests/test_backtest_v2_resized.py`

Deterministic function that walks a trade stream in time order, applies sizing via the kelly_table, enforces concurrency, and returns a ledger + metrics.

- [ ] **Step 1: Write failing tests**

```python
from backtest_v2_resized import simulate  # noqa: E402


def test_simulate_flat_sizing_when_kelly_table_all_equal():
    """When kelly_table has same f_capped in every bin, result = flat sizing."""
    rows = [
        ("2025-06-01T00:00:00Z", 0.80, 0.03, 20),   # +3%
        ("2025-06-01T06:00:00Z", 0.85, -0.01, 20),  # -1%
    ]
    df = _make_trades(rows)
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": 0.05}
        for lo, hi in zip([0.70,0.75,0.80,0.85,0.90], [0.75,0.80,0.85,0.90,1.01])
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
        for lo, hi in zip([0.70,0.75,0.80,0.85,0.90], [0.75,0.80,0.85,0.90,1.01])
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
```

- [ ] **Step 2: Run tests — expect import failure**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 3 new failures with `ImportError`.

- [ ] **Step 3: Implement `simulate()`**

Add to `backtest_v2_resized.py`:

```python
def simulate(
    trades_df: pd.DataFrame,
    kelly_table: pd.DataFrame,
    max_concurrent: int,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """Replay trades under a given sizing/concurrency rule.

    Input `trades_df` must have columns: signal_time, ml_prob, pnl_pct, bars_held.
    Sort order inside the function is by signal_time.

    Sizing per trade: notional = START_EQUITY_USD * kelly_fraction * LEVERAGE.
    Fees applied as pnl_net_pct = pnl_pct - ROUND_TRIP_FEE before converting to USD.

    Concurrency: prune finished positions, skip new if len(open) >= max_concurrent.

    Returns (ledger_df, metrics_dict).
    ledger_df columns: signal_time, ml_prob, pnl_pct_gross, pnl_pct_net,
                       kelly_fraction, notional_usd, pnl_usd, exit_time, taken
    metrics_dict: trades, skipped_concurrency, wr, pf_net, avg_pnl_pct_net,
                  total_pnl_usd, total_return_pct, max_dd_pct, label
    """
    df = trades_df.sort_values("signal_time").reset_index(drop=True)
    bar_td = pd.Timedelta(minutes=BAR_MINUTES)

    ledger_rows = []
    open_positions: list[pd.Timestamp] = []
    skipped = 0

    for _, t in df.iterrows():
        sig_time = t["signal_time"]
        # Prune finished positions
        open_positions = [exit_t for exit_t in open_positions if exit_t > sig_time]

        f = _kelly_fraction_for(t["ml_prob"], kelly_table)
        notional = START_EQUITY_USD * f * LEVERAGE
        pnl_net = t["pnl_pct"] - ROUND_TRIP_FEE
        pnl_usd = notional * pnl_net
        exit_time = sig_time + bar_td * int(t["bars_held"])

        if len(open_positions) >= max_concurrent:
            skipped += 1
            ledger_rows.append({
                "signal_time": sig_time, "ml_prob": t["ml_prob"],
                "pnl_pct_gross": t["pnl_pct"], "pnl_pct_net": pnl_net,
                "kelly_fraction": f, "notional_usd": notional,
                "pnl_usd": 0.0, "exit_time": exit_time, "taken": False,
            })
            continue

        open_positions.append(exit_time)
        ledger_rows.append({
            "signal_time": sig_time, "ml_prob": t["ml_prob"],
            "pnl_pct_gross": t["pnl_pct"], "pnl_pct_net": pnl_net,
            "kelly_fraction": f, "notional_usd": notional,
            "pnl_usd": pnl_usd, "exit_time": exit_time, "taken": True,
        })

    ledger = pd.DataFrame(ledger_rows)
    taken = ledger[ledger["taken"]]

    # Metrics on taken trades only
    if len(taken) == 0:
        metrics = {
            "label": label, "trades": 0, "skipped_concurrency": skipped,
            "wr": 0.0, "pf_net": 0.0, "avg_pnl_pct_net": 0.0,
            "total_pnl_usd": 0.0, "total_return_pct": 0.0, "max_dd_pct": 0.0,
        }
        return ledger, metrics

    wins = taken[taken["pnl_usd"] > 0]["pnl_usd"]
    losses = taken[taken["pnl_usd"] <= 0]["pnl_usd"]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    total_pnl_usd = taken["pnl_usd"].sum()

    # Max drawdown on cumulative USD equity curve
    equity = START_EQUITY_USD + taken["pnl_usd"].cumsum()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min()) * 100.0  # negative number

    metrics = {
        "label": label,
        "trades": int(len(taken)),
        "skipped_concurrency": int(skipped),
        "wr": float((taken["pnl_usd"] > 0).mean()),
        "pf_net": float(pf) if pf != float("inf") else float("inf"),
        "avg_pnl_pct_net": float(taken["pnl_pct_net"].mean()),
        "total_pnl_usd": float(total_pnl_usd),
        "total_return_pct": float(total_pnl_usd / START_EQUITY_USD * 100.0),
        "max_dd_pct": max_dd,
    }
    return ledger, metrics
```

- [ ] **Step 4: Run tests**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 10 passed.

- [ ] **Step 5: Commit**

```
git add services/python/scripts/backtest_v2_resized.py services/python/tests/test_backtest_v2_resized.py
git commit -m "feat(backtest): simulate() scenario runner with Kelly + concurrency"
```

---

### Task 6: `monte_carlo_pf()` — Bootstrap PF Robustness

**Files:**
- Modify: `services/python/scripts/backtest_v2_resized.py`
- Modify: `services/python/tests/test_backtest_v2_resized.py`

Bootstrap the PF distribution by resampling trade order with replacement (1000 iters). Satisfies Gate 7 of the Tighter Backtest Standard.

- [ ] **Step 1: Write failing tests**

```python
from backtest_v2_resized import monte_carlo_pf  # noqa: E402


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
```

- [ ] **Step 2: Run tests — expect import failure**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 3 new failures with `ImportError`.

- [ ] **Step 3: Implement `monte_carlo_pf()`**

Add to `backtest_v2_resized.py`:

```python
def monte_carlo_pf(pnls: np.ndarray, iters: int = MC_ITER, seed: int = MC_SEED) -> tuple[float, float, float]:
    """Bootstrap PF distribution by resampling trade order with replacement.

    Returns (p5, p50, p95) of PF across `iters` bootstrap samples.
    PF is inf when a sample has no losers.
    """
    rng = np.random.default_rng(seed)
    n = len(pnls)
    if n == 0:
        return 0.0, 0.0, 0.0
    pfs = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = pnls[rng.integers(0, n, size=n)]
        wins = sample[sample > 0]
        losses = sample[sample <= 0]
        if len(losses) == 0 or losses.sum() == 0:
            pfs[i] = np.inf
        else:
            pfs[i] = wins.sum() / abs(losses.sum())
    # Quantiles ignore inf by masking
    finite = pfs[np.isfinite(pfs)]
    if len(finite) == 0:
        return float("inf"), float("inf"), float("inf")
    # Use all pfs for quantile (inf participates via sorting), but represent inf cleanly
    pfs_sorted = np.sort(pfs)
    p5 = pfs_sorted[int(0.05 * iters)]
    p50 = pfs_sorted[int(0.50 * iters)]
    p95 = pfs_sorted[int(0.95 * iters)]
    return float(p5), float(p50), float(p95)
```

- [ ] **Step 4: Run tests**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 13 passed.

- [ ] **Step 5: Commit**

```
git add services/python/scripts/backtest_v2_resized.py services/python/tests/test_backtest_v2_resized.py
git commit -m "feat(backtest): monte_carlo_pf bootstrap + tests"
```

---

### Task 7: `threshold_kelly_crosscheck()` — Byproduct Output

**Files:**
- Modify: `services/python/scripts/backtest_v2_resized.py`
- Modify: `services/python/tests/test_backtest_v2_resized.py`

For each threshold in CROSSCHECK_THRESHOLDS, re-run the 3 scenarios and collect total_return_pct into a single DataFrame. Answers "is 0.70 still best under Kelly + concurrency?"

- [ ] **Step 1: Write failing test**

```python
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
        for lo, hi in zip([0.70,0.75,0.80,0.85,0.90], [0.75,0.80,0.85,0.90,1.01])
    ])
    cc = threshold_kelly_crosscheck(df, flat_tbl, thresholds=[0.70, 0.80, 0.90])
    assert list(cc["threshold"]) == [0.70, 0.80, 0.90]
    assert set(cc.columns) >= {"threshold", "A_trades", "A_ret_pct", "B_ret_pct", "C_ret_pct"}
    # At threshold 0.70 all 3 trades pass; at 0.90 only one.
    assert cc.iloc[0]["A_trades"] == 3
    assert cc.iloc[2]["A_trades"] == 1
```

- [ ] **Step 2: Run tests — expect import failure**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 1 new failure with `ImportError`.

- [ ] **Step 3: Implement `threshold_kelly_crosscheck()`**

Add to `backtest_v2_resized.py`:

```python
def threshold_kelly_crosscheck(
    trades_df: pd.DataFrame,
    kelly_table: pd.DataFrame,
    thresholds: list[float] = CROSSCHECK_THRESHOLDS,
) -> pd.DataFrame:
    """For each threshold, re-run scenarios A, B, C and collect total_return_pct.

    Scenario A uses flat sizing (f=BASELINE_EQUITY_FRACTION for every bin);
    B and C use the provided kelly_table.

    Returns DataFrame with columns: threshold, A_trades, A_ret_pct, B_ret_pct, C_ret_pct.
    """
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": BASELINE_EQUITY_FRACTION}
        for lo, hi in zip(KELLY_BIN_EDGES[:-1], KELLY_BIN_EDGES[1:], strict=False)
    ])
    rows = []
    for th in thresholds:
        sub = trades_df[trades_df["ml_prob"] >= th].copy()
        _, m_a = simulate(sub, flat_tbl, max_concurrent=MAX_CONCURRENT_A, label=f"A_th{th}")
        _, m_b = simulate(sub, kelly_table, max_concurrent=MAX_CONCURRENT_B, label=f"B_th{th}")
        _, m_c = simulate(sub, kelly_table, max_concurrent=MAX_CONCURRENT_C, label=f"C_th{th}")
        rows.append({
            "threshold": th,
            "A_trades": m_a["trades"],
            "A_ret_pct": m_a["total_return_pct"],
            "B_ret_pct": m_b["total_return_pct"],
            "C_ret_pct": m_c["total_return_pct"],
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run tests**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 14 passed.

- [ ] **Step 5: Commit**

```
git add services/python/scripts/backtest_v2_resized.py services/python/tests/test_backtest_v2_resized.py
git commit -m "feat(backtest): threshold_kelly_crosscheck byproduct table"
```

---

### Task 8: `load_trades()` — DB Reader with Schema Guard

**Files:**
- Modify: `services/python/scripts/backtest_v2_resized.py`

Thin DB wrapper. Not unit-tested (needs DB) but validates schema and returns a clean DataFrame with the 4 required columns.

- [ ] **Step 1: Implement `load_trades()`**

Add to `backtest_v2_resized.py`:

```python
def load_trades(engine) -> pd.DataFrame:
    """Load ML-filtered trades from DB and validate schema.

    Expected columns (at minimum): signal_time, ml_prob, pnl_pct, bars_held.
    Drops rows with NaN in any required column. Sorts by signal_time.
    """
    required = ["signal_time", "ml_prob", "pnl_pct", "bars_held"]
    sql = f"SELECT {', '.join(required)} FROM {SOURCE_TABLE} ORDER BY signal_time"
    df = pd.read_sql(sql, engine)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{SOURCE_TABLE} missing required columns: {missing}")
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < before:
        print(f"  dropped {before - len(df)} rows with NaN in required columns")
    return df


def _split_oot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (pre_oot, oot) using last HOLD_OUT_MONTHS calendar months."""
    max_date = df["signal_time"].max()
    oot_start = (max_date - pd.DateOffset(months=HOLD_OUT_MONTHS)).floor("D")
    pre = df[df["signal_time"] < oot_start].reset_index(drop=True)
    oot = df[df["signal_time"] >= oot_start].reset_index(drop=True)
    return pre, oot
```

- [ ] **Step 2: Smoke test — can it connect and read?**

Run:
```
cd services/python && python -c "
import sys; sys.path.insert(0, '.')
import os
from sqlalchemy import create_engine
from scripts.backtest_v2_resized import load_trades, _split_oot

eng = create_engine(os.environ.get('DATABASE_URL', 'postgresql://postgres:MySQL100%25@localhost:5432/market'))
df = load_trades(eng)
print(f'Loaded {len(df)} trades, range {df[\"signal_time\"].min().date()} -> {df[\"signal_time\"].max().date()}')
pre, oot = _split_oot(df)
print(f'Pre-OOT: {len(pre)}  OOT: {len(oot)}')
"
```
Expected output: `Loaded 1132 trades ...` and `Pre-OOT: ~828  OOT: ~304` (approximate; depends on current DB state).

- [ ] **Step 3: Commit**

```
git add services/python/scripts/backtest_v2_resized.py
git commit -m "feat(backtest): load_trades DB reader + OOT split helper"
```

---

### Task 9: `main()` — Orchestrator + Result Writers

**Files:**
- Modify: `services/python/scripts/backtest_v2_resized.py`

Replace the placeholder `main()` with the full orchestrator: load, split, calibrate, run 3 scenarios, MC, crosscheck, write outputs.

- [ ] **Step 1: Implement helpers for run metadata and output writing**

Add to `backtest_v2_resized.py` above `main()`:

```python
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _git_sha_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def _run_id() -> str:
    sha = _git_sha_short()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"backtest_v2_resized_{sha}_{ts}"


def _write_outputs(
    run_dir: Path,
    metrics_a: dict,
    metrics_b: dict,
    metrics_c: dict,
    mc_a: tuple,
    mc_b: tuple,
    mc_c: tuple,
    ledger_a: pd.DataFrame,
    ledger_b: pd.DataFrame,
    ledger_c: pd.DataFrame,
    kelly_table: pd.DataFrame,
    crosscheck: pd.DataFrame,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    def _mc_to_dict(mc):
        return {"pf_p5": mc[0], "pf_p50": mc[1], "pf_p95": mc[2]}

    def _merge(metrics, mc):
        out = dict(metrics)
        out.update(_mc_to_dict(mc))
        # json-serialize inf as null
        for k, v in list(out.items()):
            if isinstance(v, float) and (v == float("inf") or v != v):
                out[k] = None
        return out

    lift_b_minus_a = metrics_b["total_return_pct"] - metrics_a["total_return_pct"]
    lift_c_minus_b = metrics_c["total_return_pct"] - metrics_b["total_return_pct"]
    lift_c_minus_a = metrics_c["total_return_pct"] - metrics_a["total_return_pct"]

    metrics_out = {
        "scenario_A": _merge(metrics_a, mc_a),
        "scenario_B": _merge(metrics_b, mc_b),
        "scenario_C": _merge(metrics_c, mc_c),
        "attribution": {
            "lever_1_lift_pct": lift_b_minus_a,
            "lever_4_lift_pct": lift_c_minus_b,
            "combined_lift_pct": lift_c_minus_a,
        },
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # params.json
    params = {
        "SNAPSHOT_SOURCE": SOURCE_TABLE,
        "ML_THRESHOLD": ML_THRESHOLD,
        "START_EQUITY_USD": START_EQUITY_USD,
        "LEVERAGE": LEVERAGE,
        "BASELINE_EQUITY_FRACTION": BASELINE_EQUITY_FRACTION,
        "KELLY_MULTIPLIER": KELLY_MULTIPLIER,
        "KELLY_FRACTION_FLOOR": KELLY_FRACTION_FLOOR,
        "KELLY_FRACTION_CAP": KELLY_FRACTION_CAP,
        "KELLY_BIN_EDGES": KELLY_BIN_EDGES,
        "ROUND_TRIP_FEE": ROUND_TRIP_FEE,
        "HOLD_OUT_MONTHS": HOLD_OUT_MONTHS,
        "MAX_CONCURRENT_A": MAX_CONCURRENT_A,
        "MAX_CONCURRENT_B": MAX_CONCURRENT_B,
        "MAX_CONCURRENT_C": MAX_CONCURRENT_C,
        "MC_ITER": MC_ITER,
        "MC_SEED": MC_SEED,
        "BAR_MINUTES": BAR_MINUTES,
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Ledgers
    ledger_a.to_csv(run_dir / "trades_A.csv", index=False)
    ledger_b.to_csv(run_dir / "trades_B.csv", index=False)
    ledger_c.to_csv(run_dir / "trades_C.csv", index=False)

    # Kelly table
    kelly_table.to_csv(run_dir / "kelly_table.csv", index=False)

    # Skipped by concurrency (A vs C; B same as A)
    skipped_a = ledger_a[~ledger_a["taken"]].copy()
    skipped_a["scenario"] = "A"
    skipped_c = ledger_c[~ledger_c["taken"]].copy()
    skipped_c["scenario"] = "C"
    pd.concat([skipped_a, skipped_c], ignore_index=True).to_csv(
        run_dir / "skipped_by_concurrency.csv", index=False
    )

    # Monthly comparison
    def _monthly(ledger, col):
        taken = ledger[ledger["taken"]].copy()
        taken["month"] = taken["signal_time"].dt.to_period("M").astype(str)
        return taken.groupby("month")["pnl_usd"].sum().rename(col)

    monthly = pd.concat(
        [_monthly(ledger_a, "A_pnl_usd"), _monthly(ledger_b, "B_pnl_usd"), _monthly(ledger_c, "C_pnl_usd")],
        axis=1,
    ).fillna(0.0).reset_index()
    for sc in ["A", "B", "C"]:
        monthly[f"{sc}_ret_pct"] = monthly[f"{sc}_pnl_usd"] / START_EQUITY_USD * 100.0
    monthly.to_csv(run_dir / "monthly_comparison.csv", index=False)

    # Crosscheck
    crosscheck.to_csv(run_dir / "threshold_kelly_crosscheck.csv", index=False)
```

- [ ] **Step 2: Replace the placeholder `main()`**

Replace the `def main(): raise NotImplementedError(...)` with:

```python
def main():
    engine = create_engine(DB_URL)
    print("Loading trades ...")
    df = load_trades(engine)
    print(f"  {len(df)} trades  {df['signal_time'].min().date()} -> {df['signal_time'].max().date()}")

    pre_oot, oot = _split_oot(df)
    print(f"  pre-OOT: {len(pre_oot)}  OOT: {len(oot)}")

    # Apply threshold gate to OOT (pre-OOT uses same threshold for Kelly calibration)
    pre_oot_th = pre_oot[pre_oot["ml_prob"] >= ML_THRESHOLD].reset_index(drop=True)
    oot_th = oot[oot["ml_prob"] >= ML_THRESHOLD].reset_index(drop=True)
    print(f"  pre-OOT (ml>={ML_THRESHOLD}): {len(pre_oot_th)}  OOT: {len(oot_th)}")

    # --- Calibrate Kelly table on pre-OOT ONLY ---
    print("\nCalibrating Kelly table on pre-OOT ...")
    kelly_table = build_kelly_table(pre_oot_th)
    print(kelly_table.to_string(index=False))

    # --- Flat sizing table (baseline) ---
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": BASELINE_EQUITY_FRACTION,
         "trades": 0, "p": 0.0, "b": 0.0, "f_raw": 0.0, "f_half": 0.0}
        for lo, hi in zip(KELLY_BIN_EDGES[:-1], KELLY_BIN_EDGES[1:], strict=False)
    ])

    # --- Three scenarios on OOT ---
    print("\nRunning scenario A (flat, conc=5) ...")
    ledger_a, metrics_a = simulate(oot_th, flat_tbl, MAX_CONCURRENT_A, "A")
    print(f"  trades={metrics_a['trades']} ret={metrics_a['total_return_pct']:.1f}% pf={metrics_a['pf_net']:.2f}")

    print("\nRunning scenario B (half-Kelly, conc=5) ...")
    ledger_b, metrics_b = simulate(oot_th, kelly_table, MAX_CONCURRENT_B, "B")
    print(f"  trades={metrics_b['trades']} ret={metrics_b['total_return_pct']:.1f}% pf={metrics_b['pf_net']:.2f}")

    print("\nRunning scenario C (half-Kelly, conc=10) ...")
    ledger_c, metrics_c = simulate(oot_th, kelly_table, MAX_CONCURRENT_C, "C")
    print(f"  trades={metrics_c['trades']} ret={metrics_c['total_return_pct']:.1f}% pf={metrics_c['pf_net']:.2f}")

    # --- Monte Carlo on each scenario's taken trades ---
    print("\nMonte Carlo PF bootstrap ...")
    mc_a = monte_carlo_pf(ledger_a[ledger_a["taken"]]["pnl_pct_net"].to_numpy())
    mc_b = monte_carlo_pf(ledger_b[ledger_b["taken"]]["pnl_pct_net"].to_numpy())
    mc_c = monte_carlo_pf(ledger_c[ledger_c["taken"]]["pnl_pct_net"].to_numpy())
    print(f"  A PF: p5={mc_a[0]:.2f} p50={mc_a[1]:.2f} p95={mc_a[2]:.2f}")
    print(f"  B PF: p5={mc_b[0]:.2f} p50={mc_b[1]:.2f} p95={mc_b[2]:.2f}")
    print(f"  C PF: p5={mc_c[0]:.2f} p50={mc_c[1]:.2f} p95={mc_c[2]:.2f}")

    # --- Threshold crosscheck ---
    print("\nThreshold x Kelly crosscheck ...")
    crosscheck = threshold_kelly_crosscheck(oot, kelly_table)
    print(crosscheck.to_string(index=False))

    # --- Write outputs ---
    run_dir = RESULTS_ROOT / _run_id()
    _write_outputs(run_dir, metrics_a, metrics_b, metrics_c, mc_a, mc_b, mc_c,
                   ledger_a, ledger_b, ledger_c, kelly_table, crosscheck)
    print(f"\nWrote results to: {run_dir}")

    # --- Attribution summary ---
    print("\n=== ATTRIBUTION ===")
    print(f"A baseline return:     {metrics_a['total_return_pct']:>7.1f}%")
    print(f"B (+Kelly) return:     {metrics_b['total_return_pct']:>7.1f}%  (lift: {metrics_b['total_return_pct']-metrics_a['total_return_pct']:+.1f}%)")
    print(f"C (+Kelly+conc) return: {metrics_c['total_return_pct']:>7.1f}%  (lift: {metrics_c['total_return_pct']-metrics_a['total_return_pct']:+.1f}%)")
```

- [ ] **Step 3: Commit**

```
git add services/python/scripts/backtest_v2_resized.py
git commit -m "feat(backtest): main orchestrator + output writers"
```

---

### Task 10: End-to-End Smoke Run

**Files:**
- None modified. Executes the script and inspects outputs.

Final integration check: script runs to completion against the real DB, outputs appear, numbers match the rough expectations from the in-chat exploration (A ≈ +210%).

- [ ] **Step 1: Run the script**

```
cd services/python && python scripts/backtest_v2_resized.py
```
Expected: completes in under 60 seconds. Final stdout shows 3 scenarios with A ≈ 210% ± 20% (matches chat baseline at threshold 0.70).

- [ ] **Step 2: Verify outputs exist**

```
ls services/python/results/backtest_v2_resized_*/
```
Expected files:
```
metrics.json
params.json
trades_A.csv
trades_B.csv
trades_C.csv
kelly_table.csv
skipped_by_concurrency.csv
monthly_comparison.csv
threshold_kelly_crosscheck.csv
```

- [ ] **Step 3: Sanity-check the attribution**

```
cd services/python && python -c "
import json, glob
run = sorted(glob.glob('results/backtest_v2_resized_*'))[-1]
m = json.load(open(f'{run}/metrics.json'))
print('A:', m['scenario_A']['total_return_pct'])
print('B:', m['scenario_B']['total_return_pct'])
print('C:', m['scenario_C']['total_return_pct'])
print('Attribution:', m['attribution'])
"
```
Expected: A in 180-240% range. B and C should be >= A (sizing and concurrency should not hurt PnL on this data). If B < A or C < B, inspect kelly_table.csv — a poorly-calibrated bin might be pushing size into a loss-heavy bin.

- [ ] **Step 4: Commit the result folder**

```
git add services/python/results/backtest_v2_resized_*
git commit -m "test(backtest): v2 resizing analysis first run results"
```

---

### Task 11: Ruff + Final Smoke

**Files:**
- None. Lint check.

- [ ] **Step 1: Run ruff on new files**

```
cd services/python && ruff check scripts/backtest_v2_resized.py tests/test_backtest_v2_resized.py
```
Expected: no issues. If any, fix them inline — prefer simple changes (rename unused vars, remove unused imports). Do not touch logic.

- [ ] **Step 2: Run full test file one more time**

```
cd services/python && pytest tests/test_backtest_v2_resized.py -v
```
Expected: 14 passed.

- [ ] **Step 3: Commit lint fixes (if any)**

```
git commit -am "chore(backtest): ruff fixes on resizing analysis"
```

---

## Success Criteria

After Task 11:

1. `services/python/scripts/backtest_v2_resized.py` exists, passes ruff, and runs end-to-end against the local DB in under 60s.
2. `services/python/tests/test_backtest_v2_resized.py` has 14 passing tests covering `build_kelly_table`, `_kelly_fraction_for`, `simulate`, `monte_carlo_pf`, `threshold_kelly_crosscheck`.
3. One `results/backtest_v2_resized_<sha>_<ts>/` folder exists with 9 expected output files, committed to git.
4. `metrics.json.scenario_A.total_return_pct` is within ~20% of the 210% baseline we computed in chat.
5. Attribution summary printed at end of run shows clean lift numbers for Kelly (B − A) and concurrency (C − B).

## Decision Framework After Run

See spec section "Decision Framework" for interpretation of the attribution numbers and what to do next.
