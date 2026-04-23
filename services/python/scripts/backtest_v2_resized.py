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


def main():
    raise NotImplementedError("Filled in by later tasks")


if __name__ == "__main__":
    main()
