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


def _kelly_fraction_for(ml_prob: float, kelly_table: pd.DataFrame) -> float:
    """Map a single ml_prob to its bin's f_capped. Defensive floor for out-of-range."""
    for _, row in kelly_table.iterrows():
        if row["bin_low"] <= ml_prob < row["bin_high"]:
            return float(row["f_capped"])
    return KELLY_FRACTION_FLOOR


def main():
    raise NotImplementedError("Filled in by later tasks")


if __name__ == "__main__":
    main()
