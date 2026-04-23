# scripts/backtest_short_v2_exit_sweep.py
"""Exit strategy sweep for Scanner Short v2 entries.

Re-uses the exact same entry logic from backtest_short_v2.py (delayed entry +
regime filter), but simulates MULTIPLE exit strategies on each entry to find
the optimal exit.

Exit types:
  - Fixed: standard SL / TP / timeout
  - Trailing: SL + trailing stop activated after X% profit
  - Layered: 50% at fixed TP, 50% trailing stop

Run from services/python/:
    python scripts/backtest_short_v2_exit_sweep.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from scripts.backtest_momentum_scanner import load_coin  # noqa: E402

# ── Database ─────────────────────────────────────────────────────────
import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")

# ── Paths ────────────────────────────────────────────────────────────
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

# ── Entry params (same as v2) ────────────────────────────────────────
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
COOLDOWN_BARS = 96
BOUNCE_WINDOW = 96
BOUNCE_MIN_PCT = 0.02
REJECTION_PCT = 0.03

# ── Exit strategies to sweep ────────────────────────────────────────
# (label, stop_pct, tp_pct, timeout_bars, trailing_after, trailing_pct)
# trailing_after = activate trail after X% profit, trailing_pct = trail distance
# For "LAYERED" strategies: 50% exits at tp_pct, 50% uses trailing

# Round-trip friction per trade (Gate.io fees + spread/slippage on altcoins)
FRICTION_PCT = 0.0015

EXIT_STRATEGIES = [
    # Fixed exits
    ("5%/5% 48h (current)", 0.05, 0.05, 192, None, None),
    ("5%/8% 48h", 0.05, 0.08, 192, None, None),
    ("5%/10% 96h", 0.05, 0.10, 384, None, None),
    ("5%/12% 96h", 0.05, 0.12, 384, None, None),
    ("5%/15% 1w", 0.05, 0.15, 672, None, None),
    # Trailing stop exits
    ("5%SL + trail@3% (2%trail) 96h", 0.05, None, 384, 0.03, 0.02),
    ("5%SL + trail@5% (2%trail) 96h", 0.05, None, 384, 0.05, 0.02),
    ("5%SL + trail@3% (3%trail) 96h", 0.05, None, 384, 0.03, 0.03),
    # Layered: 50% at fixed TP, 50% trailing
    ("LAYERED 50%@5% + 50%trail@5%(2%) 96h", 0.05, 0.05, 384, 0.05, 0.02),
    ("LAYERED 50%@5% + 50%trail@3%(2%) 96h", 0.05, 0.05, 384, 0.03, 0.02),
]

engine = create_engine(DB_URL)


# ── Regime (identical to v2) ─────────────────────────────────────────

def compute_daily_regime(all_coin_data: dict, btc_df: pd.DataFrame) -> dict:
    """Precompute daily regime signals.

    Returns dict: {date -> {"btc_bearish": bool, "breadth": float,
                             "breadth_low": bool, "allowed": bool}}
    """
    btc_daily = btc_df.set_index("open_time").resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    btc_daily["ema20"] = btc_daily["close"].ewm(span=20, adjust=False).mean()
    btc_daily["bearish"] = btc_daily["close"] < btc_daily["ema20"]

    coin_above_ema: dict = {}
    for symbol, df in all_coin_data.items():
        if symbol == "BTCUSDT":
            continue
        coin_daily = df.set_index("open_time").resample("1D").agg({"close": "last"}).dropna()
        coin_daily["ema20"] = coin_daily["close"].ewm(span=20, adjust=False).mean()
        coin_daily["above"] = coin_daily["close"] > coin_daily["ema20"]
        for date, row in coin_daily.iterrows():
            d = date.date()
            if d not in coin_above_ema:
                coin_above_ema[d] = [0, 0]
            coin_above_ema[d][1] += 1
            if row["above"]:
                coin_above_ema[d][0] += 1

    regime = {}
    for date in sorted(coin_above_ema.keys()):
        above, total = coin_above_ema[date]
        breadth = above / total if total > 0 else 0.5
        btc_row = btc_daily[btc_daily.index.date == date]
        btc_bear = bool(btc_row["bearish"].iloc[0]) if len(btc_row) > 0 else False
        allowed = btc_bear and breadth < 0.60
        regime[date] = {
            "btc_bearish": btc_bear,
            "breadth": breadth,
            "breadth_low": breadth < 0.60,
            "allowed": allowed,
        }
    return regime


# ── Simulate functions ───────────────────────────────────────────────

def simulate_short_fixed(close, high, low, entry_bar, stop_pct, tp_pct, max_bars):
    """Standard fixed SL/TP short."""
    if entry_bar + 1 >= len(close):
        return None
    entry_price = close[entry_bar]
    if entry_price <= 0:
        return None

    sl_price = entry_price * (1 + stop_pct)
    tp_price = entry_price * (1 - tp_pct)
    last_bar = min(entry_bar + 1 + max_bars, len(close))

    for bar in range(entry_bar + 1, last_bar):
        if high[bar] >= sl_price:
            return {
                "pnl_pct": -stop_pct - FRICTION_PCT,
                "exit_reason": "stop_loss",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }
        if low[bar] <= tp_price:
            return {
                "pnl_pct": tp_pct - FRICTION_PCT,
                "exit_reason": "take_profit",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": tp_price,
            }

    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (entry_price - exit_price) / entry_price - FRICTION_PCT
    return {
        "pnl_pct": pnl_pct,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


def simulate_short_trailing(
    close, high, low, entry_bar, stop_pct, max_bars, trail_activate, trail_pct
):
    """Short with trailing stop. No fixed TP.

    - Initial stop at entry*(1+stop_pct)
    - Once price drops trail_activate% below entry, activate trailing stop
    - Trail stop follows at trail_pct above the lowest close since entry
    - Trail stop can only move DOWN (tighter), never up
    """
    if entry_bar + 1 >= len(close):
        return None
    entry_price = close[entry_bar]
    if entry_price <= 0:
        return None

    sl_price = entry_price * (1 + stop_pct)
    lowest_close = entry_price
    trailing_active = False
    trail_stop = None
    last_bar = min(entry_bar + 1 + max_bars, len(close))

    for bar in range(entry_bar + 1, last_bar):
        # Check fixed SL first
        if high[bar] >= sl_price:
            return {
                "pnl_pct": -stop_pct - FRICTION_PCT,
                "exit_reason": "stop_loss",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }

        # Track lowest close
        if close[bar] < lowest_close:
            lowest_close = close[bar]

        # Activate trailing after sufficient profit
        profit = (entry_price - lowest_close) / entry_price
        if profit >= trail_activate and not trailing_active:
            trailing_active = True

        if trailing_active:
            new_trail = lowest_close * (1 + trail_pct)
            if trail_stop is None or new_trail < trail_stop:
                trail_stop = new_trail

            # Check trail stop hit
            if high[bar] >= trail_stop:
                pnl = (entry_price - trail_stop) / entry_price - FRICTION_PCT
                return {
                    "pnl_pct": pnl,
                    "exit_reason": "trail_stop",
                    "bars_held": bar - entry_bar,
                    "entry_price": entry_price,
                    "exit_price": trail_stop,
                }

    # Timeout
    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (entry_price - exit_price) / entry_price - FRICTION_PCT
    return {
        "pnl_pct": pnl_pct,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


def simulate_short_layered(
    close, high, low, entry_bar, stop_pct, tp_pct, max_bars, trail_activate, trail_pct
):
    """Layered exit: 50% at fixed TP, 50% trailing.

    - Lot 1 (50%): exits at fixed TP
    - Lot 2 (50%): trailing stop after trail_activate%
    - If SL hit before any TP: both lots exit at SL
    - Combined PnL = average of both lots
    """
    if entry_bar + 1 >= len(close):
        return None
    entry_price = close[entry_bar]
    if entry_price <= 0:
        return None

    sl_price = entry_price * (1 + stop_pct)
    tp_price = entry_price * (1 - tp_pct)
    lot1_closed = False
    lot1_pnl = 0.0
    lowest_close = entry_price
    trailing_active = False
    trail_stop = None
    last_bar = min(entry_bar + 1 + max_bars, len(close))

    for bar in range(entry_bar + 1, last_bar):
        # SL check (applies to whichever lots are still open)
        if high[bar] >= sl_price:
            if lot1_closed:
                lot2_pnl = -stop_pct if sl_price > entry_price else (
                    (entry_price - sl_price) / entry_price
                )
                # sl_price may have moved to breakeven
                lot2_pnl = (entry_price - sl_price) / entry_price
                combined = (lot1_pnl + lot2_pnl) / 2 - FRICTION_PCT
                reason = "partial_then_sl" if lot2_pnl < 0 else "partial_then_be"
                return {
                    "pnl_pct": combined,
                    "exit_reason": reason,
                    "bars_held": bar - entry_bar,
                    "entry_price": entry_price,
                    "exit_price": sl_price,
                }
            else:
                return {
                    "pnl_pct": -stop_pct - FRICTION_PCT,
                    "exit_reason": "stop_loss",
                    "bars_held": bar - entry_bar,
                    "entry_price": entry_price,
                    "exit_price": sl_price,
                }

        # Track lowest close
        if close[bar] < lowest_close:
            lowest_close = close[bar]

        # Lot 1: fixed TP
        if not lot1_closed and low[bar] <= tp_price:
            lot1_pnl = tp_pct
            lot1_closed = True
            # Move SL to breakeven for lot 2
            sl_price = entry_price

        # Lot 2: trailing
        profit = (entry_price - lowest_close) / entry_price
        if profit >= trail_activate:
            trailing_active = True

        if trailing_active and lot1_closed:
            new_trail = lowest_close * (1 + trail_pct)
            if trail_stop is None or new_trail < trail_stop:
                trail_stop = new_trail
            if high[bar] >= trail_stop:
                lot2_pnl = (entry_price - trail_stop) / entry_price
                combined = (lot1_pnl + lot2_pnl) / 2 - FRICTION_PCT
                return {
                    "pnl_pct": combined,
                    "exit_reason": "layered_complete",
                    "bars_held": bar - entry_bar,
                    "entry_price": entry_price,
                    "exit_price": trail_stop,
                }

    # Timeout
    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    if lot1_closed:
        lot2_pnl = (entry_price - exit_price) / entry_price
        combined = (lot1_pnl + lot2_pnl) / 2 - FRICTION_PCT
    else:
        combined = (entry_price - exit_price) / entry_price - FRICTION_PCT
    return {
        "pnl_pct": combined,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


def run_strategy(close, high, low, entry_bar, strategy):
    """Dispatch to the right simulate function based on strategy config."""
    label, stop_pct, tp_pct, max_bars, trail_after, trail_pct = strategy

    if label.startswith("LAYERED"):
        return simulate_short_layered(
            close, high, low, entry_bar,
            stop_pct, tp_pct, max_bars, trail_after, trail_pct,
        )
    elif trail_after is not None and tp_pct is None:
        return simulate_short_trailing(
            close, high, low, entry_bar,
            stop_pct, max_bars, trail_after, trail_pct,
        )
    else:
        return simulate_short_fixed(
            close, high, low, entry_bar,
            stop_pct, tp_pct, max_bars,
        )


# ── Entry detection (identical gates to v2) ──────────────────────────

def find_entries(symbol, df, regime):
    """Find all delayed-entry signals for one coin. Returns list of entry_bar indices."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["open_time"].values

    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    entries = []
    cooldown_until = 0

    for i in range(VOL_MA_PERIOD + PRICE_LOOKBACK, len(close) - BOUNCE_WINDOW):
        if i < cooldown_until:
            continue

        # Phase 1: Volume spike
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
            continue

        # Phase 1: Price drop from rolling high
        rolling_high = np.max(high[i - PRICE_LOOKBACK: i])
        if rolling_high <= 0:
            continue
        price_drop = (rolling_high - close[i]) / rolling_high
        if price_drop < PRICE_MOVE_THRESH:
            continue

        # Phase 1: Regime gate
        bar_date = pd.Timestamp(times[i]).date()
        if bar_date not in regime or not regime[bar_date]["allowed"]:
            continue

        dump_close = close[i]
        dump_bar = i

        # Phase 2: Bounce detection
        bounce_high = dump_close
        bounce_bar = None
        for j in range(dump_bar + 1, min(dump_bar + BOUNCE_WINDOW + 1, len(close))):
            if close[j] > bounce_high:
                bounce_high = close[j]
                bounce_bar = j

        if bounce_bar is None or (bounce_high - dump_close) / dump_close < BOUNCE_MIN_PCT:
            continue

        # Phase 3: Re-rejection
        rejection_level = bounce_high * (1 - REJECTION_PCT)
        entry_bar = None
        for j in range(bounce_bar + 1, min(dump_bar + BOUNCE_WINDOW + 1, len(close))):
            if close[j] < rejection_level:
                entry_bar = j
                break

        if entry_bar is None:
            continue

        entry_time = pd.Timestamp(times[entry_bar])
        entries.append({
            "entry_bar": entry_bar,
            "signal_time": entry_time,
            "entry_price": close[entry_bar],
        })
        cooldown_until = entry_bar + COOLDOWN_BARS

    return entries


# ── Main sweep ───────────────────────────────────────────────────────

def run_sweep():
    """Load data, find entries, sweep all exit strategies."""
    t0 = time.time()
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(symbol_dirs)} coin directories")

    # Step 1: Load all coin data
    print("Loading all coin data...")
    all_coin_data: dict[str, pd.DataFrame] = {}
    for sym_dir in symbol_dirs:
        df = load_coin(sym_dir)
        if df is not None and len(df) >= 500:
            all_coin_data[sym_dir.name] = df
    print(f"  Loaded {len(all_coin_data)} coins with >= 500 bars")

    if "BTCUSDT" not in all_coin_data:
        raise RuntimeError("BTCUSDT data not found")

    btc_df = all_coin_data["BTCUSDT"]

    # Step 2: Compute regime
    print("Computing daily regime...")
    regime = compute_daily_regime(all_coin_data, btc_df)
    allowed_days = sum(1 for v in regime.values() if v["allowed"])
    print(f"  Regime: {allowed_days}/{len(regime)} days allowed "
          f"({allowed_days / len(regime) * 100:.1f}%)")

    # Step 3: Find all entries and simulate all strategies
    print("Scanning coins and simulating all exit strategies...")
    n_strategies = len(EXIT_STRATEGIES)
    # results[strategy_idx] = list of trade dicts
    results: list[list[dict]] = [[] for _ in range(n_strategies)]
    # Keep best-strategy trades with full info for DB save
    best_trades_raw: list[list[dict]] = [[] for _ in range(n_strategies)]

    total_entries = 0
    for idx, (symbol, df) in enumerate(all_coin_data.items()):
        if symbol == "BTCUSDT":
            continue

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        times = df["open_time"].values

        entries = find_entries(symbol, df, regime)
        total_entries += len(entries)

        for entry_info in entries:
            entry_bar = entry_info["entry_bar"]

            for si, strategy in enumerate(EXIT_STRATEGIES):
                trade = run_strategy(close, high, low, entry_bar, strategy)
                if trade is not None:
                    trade["symbol"] = symbol
                    trade["signal_time"] = entry_info["signal_time"]
                    results[si].append(trade)

        if (idx + 1) % 50 == 0:
            print(f"  {idx + 1}/{len(all_coin_data)} coins, "
                  f"{total_entries} entries so far")

    elapsed = time.time() - t0
    print(f"\nScan complete: {total_entries} entries across "
          f"{len(all_coin_data) - 1} coins in {elapsed:.1f}s")

    return results


# ── Reporting ────────────────────────────────────────────────────────

def compute_stats(trades: list[dict]) -> dict:
    """Compute summary stats for a list of trades."""
    if not trades:
        return None
    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["pnl_pct"] > 0).sum()
    wr = wins / n * 100
    avg_pnl = df["pnl_pct"].mean() * 100

    gross_profit = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    gross_loss = abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Exit reason breakdown
    reasons = df["exit_reason"].value_counts()
    tp_n = reasons.get("take_profit", 0)
    sl_n = reasons.get("stop_loss", 0)
    trail_n = reasons.get("trail_stop", 0)
    layered_n = reasons.get("layered_complete", 0)
    partial_sl_n = reasons.get("partial_then_sl", 0)
    partial_be_n = reasons.get("partial_then_be", 0)
    to_n = reasons.get("timeout", 0)

    return {
        "trades": n,
        "wr": wr,
        "pf": pf,
        "avg_pnl": avg_pnl,
        "total_pnl": df["pnl_pct"].sum() * 100,
        "tp_pct": tp_n / n * 100,
        "sl_pct": sl_n / n * 100,
        "trail_pct": (trail_n + layered_n) / n * 100,
        "partial_pct": (partial_sl_n + partial_be_n) / n * 100,
        "to_pct": to_n / n * 100,
        "avg_bars": df["bars_held"].mean(),
    }


def print_sweep_results(results: list[list[dict]]):
    """Print the comparison grid sorted by PF."""
    print("\n" + "=" * 110)
    print("EXIT STRATEGY SWEEP -- v2 Entries (Delayed + Regime)")
    print("=" * 110)

    rows = []
    for si, strategy in enumerate(EXIT_STRATEGIES):
        label = strategy[0]
        stats = compute_stats(results[si])
        if stats is None:
            continue
        rows.append((label, stats))

    # Sort by PF descending
    rows.sort(key=lambda x: x[1]["pf"], reverse=True)

    header = (
        f"{'Strategy':<42} {'Trades':>6} {'WR%':>6} {'PF':>6} "
        f"{'AvgPnL':>8} {'TotPnL':>9} {'TP%':>5} {'SL%':>5} "
        f"{'Trail%':>6} {'Part%':>5} {'TO%':>5} {'AvgBars':>7}"
    )
    print(header)
    print("-" * 110)

    for label, s in rows:
        line = (
            f"{label:<42} {s['trades']:>6} {s['wr']:>5.1f}% {s['pf']:>6.2f} "
            f"{s['avg_pnl']:>+7.2f}% {s['total_pnl']:>+8.1f}% "
            f"{s['tp_pct']:>4.1f}% {s['sl_pct']:>4.1f}% "
            f"{s['trail_pct']:>5.1f}% {s['partial_pct']:>4.1f}% "
            f"{s['to_pct']:>4.1f}% {s['avg_bars']:>6.0f}"
        )
        print(line)

    print("=" * 110)
    print(f"Best strategy by PF: {rows[0][0]} (PF={rows[0][1]['pf']:.2f})")

    return rows


def save_best_to_db(results: list[list[dict]], rows):
    """Save the best strategy's trades to DB."""
    best_label = rows[0][0]
    # Find the index of the best strategy
    best_idx = None
    for si, strategy in enumerate(EXIT_STRATEGIES):
        if strategy[0] == best_label:
            best_idx = si
            break

    if best_idx is None:
        print("Could not find best strategy index.")
        return

    trades = results[best_idx]
    if not trades:
        print("No trades to save.")
        return

    df = pd.DataFrame(trades)
    df["strategy"] = best_label

    table_name = "scanner_short_v2_best_exit"
    with engine.connect() as conn:
        conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
        conn.commit()
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} trades ({best_label}) to {table_name} table.")


if __name__ == "__main__":
    results = run_sweep()
    rows = print_sweep_results(results)
    save_best_to_db(results, rows)
