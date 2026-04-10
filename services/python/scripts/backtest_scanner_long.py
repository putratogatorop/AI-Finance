# scripts/backtest_scanner_long.py
"""Scanner Long Backtester.

Loads big_movers from DB, simulates long entries with stop-loss,
take-profit, and timeout exits on 15m bar data.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from scripts.backtest_momentum_scanner import load_coin  # noqa: E402

# ── Database ─────────────────────────────────────────────────────────
DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"

# ── Paths ────────────────────────────────────────────────────────────
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

# ── Strategy params ──────────────────────────────────────────────────
VOL_SPIKE_A = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK_A = 96
PRICE_LOOKBACK_B = 32
VOL_SPIKE_B = 2.0
VOL_SPIKE_C = 2.0
PRICE_LOOKBACK_C = 96

# ── Exit params ──────────────────────────────────────────────────────
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
COOLDOWN_BARS = 96


def load_big_movers() -> pd.DataFrame:
    """Query big_movers table for qualifying long moves."""
    engine = create_engine(DB_URL)
    query = text("""
        SELECT symbol, move_start, move_peak, start_price, peak_price,
               move_pct, duration_bars, max_vol_ratio_during, avg_buy_ratio_during
        FROM big_movers
        WHERE move_pct > 0.2
          AND peak_price > start_price
          AND direction = 1
        ORDER BY symbol, move_start
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df


def simulate_trade(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_bar: int,
    stop_pct: float = STOP_LOSS_PCT,
    tp_pct: float = TAKE_PROFIT_PCT,
    max_bars: int = TIMEOUT_BARS,
) -> dict | None:
    """Simulate a long trade from entry_bar.

    Entry at close[entry_bar] (approximating next-bar open).
    Returns dict with pnl_pct, exit_reason, bars_held, entry_price, exit_price,
    or None if not enough data.
    """
    if entry_bar + 1 >= len(close):
        return None

    entry_price = close[entry_bar]
    if entry_price <= 0:
        return None

    sl_price = entry_price * (1 - stop_pct)
    tp_price = entry_price * (1 + tp_pct)

    last_bar = min(entry_bar + 1 + max_bars, len(close))

    for bar in range(entry_bar + 1, last_bar):
        # Check stop loss first (conservative)
        if low[bar] <= sl_price:
            pnl_pct = (sl_price - entry_price) / entry_price
            return {
                "pnl_pct": pnl_pct,
                "exit_reason": "stop_loss",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }

        # Check take profit
        if high[bar] >= tp_price:
            pnl_pct = (tp_price - entry_price) / entry_price
            return {
                "pnl_pct": pnl_pct,
                "exit_reason": "take_profit",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": tp_price,
            }

    # Timeout — exit at close of last bar
    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (exit_price - entry_price) / entry_price
    return {
        "pnl_pct": pnl_pct,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


if __name__ == "__main__":
    movers = load_big_movers()
    print(f"Loaded {len(movers)} big movers")
