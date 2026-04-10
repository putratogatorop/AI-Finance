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


engine = create_engine(DB_URL)


def load_big_movers() -> pd.DataFrame:
    """Query big_movers table for qualifying long moves."""
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


def detect_strategy_a(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> list[int]:
    """Strategy A: Volume spike >= 3x 20-bar avg + price up 5% from 96-bar low."""
    n = len(close)
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    entries: list[int] = []
    cooldown_until = 0
    begin = max(start_idx, VOL_MA_PERIOD + PRICE_LOOKBACK_A)
    finish = min(end_idx, n)
    for i in range(begin, finish):
        if i < cooldown_until:
            continue
        if vol_ma[i] <= 0:
            continue
        # Gate 1: volume spike
        if volume[i] / vol_ma[i] < VOL_SPIKE_A:
            continue
        # Gate 2: price already up from rolling low
        rolling_low = np.min(low[i - PRICE_LOOKBACK_A : i])
        if rolling_low <= 0:
            continue
        if (close[i] - rolling_low) / rolling_low < PRICE_MOVE_THRESH:
            continue
        entries.append(i)
        cooldown_until = i + COOLDOWN_BARS
    return entries


def detect_strategy_b(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> list[int]:
    """Strategy B: Price up 5% in 32 bars + at least one 2x volume bar in window."""
    n = len(close)
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    entries: list[int] = []
    cooldown_until = 0
    begin = max(start_idx, VOL_MA_PERIOD + PRICE_LOOKBACK_B)
    finish = min(end_idx, n)
    for i in range(begin, finish):
        if i < cooldown_until:
            continue
        # Gate 1: price move
        rolling_low = np.min(low[i - PRICE_LOOKBACK_B : i])
        if rolling_low <= 0:
            continue
        if (close[i] - rolling_low) / rolling_low < PRICE_MOVE_THRESH:
            continue
        # Gate 2: any bar in the window with 2x volume spike
        has_vol_spike = False
        for j in range(i - PRICE_LOOKBACK_B, i + 1):
            if vol_ma[j] > 0 and volume[j] / vol_ma[j] >= VOL_SPIKE_B:
                has_vol_spike = True
                break
        if not has_vol_spike:
            continue
        entries.append(i)
        cooldown_until = i + COOLDOWN_BARS
    return entries


def load_volume_breakouts() -> pd.DataFrame:
    """Load volume breakout signals from DB (long direction only)."""
    query = text("""
        SELECT symbol, signal_time, vol_ratio, close as signal_close
        FROM volume_breakouts
        WHERE direction = 1
          AND vol_ratio >= 2.0
          AND timeframe = '15m'
        ORDER BY symbol, signal_time
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    return df


def detect_strategy_c(
    close: np.ndarray,
    low: np.ndarray,
    times: np.ndarray,
    vb_times: list,
    start_idx: int,
    end_idx: int,
) -> list[int]:
    """Strategy C: Volume breakout signal exists + price up 5% in 96 bars."""
    n = len(close)
    entries: list[int] = []
    cooldown_until = 0
    vb_set = set(vb_times)
    begin = max(start_idx, PRICE_LOOKBACK_C)
    finish = min(end_idx, n)
    for i in range(begin, finish):
        if i < cooldown_until:
            continue
        ts = pd.Timestamp(times[i])
        if ts not in vb_set:
            continue
        # Gate: price already up from rolling low
        rolling_low = np.min(low[i - PRICE_LOOKBACK_C : i])
        if rolling_low <= 0:
            continue
        if (close[i] - rolling_low) / rolling_low < PRICE_MOVE_THRESH:
            continue
        entries.append(i)
        cooldown_until = i + COOLDOWN_BARS
    return entries


def run_backtest() -> dict:
    """Orchestrate backtest across all big mover windows and 3 strategies."""
    big_movers = load_big_movers()
    vb_df = load_volume_breakouts()
    print(f"Loaded {len(big_movers)} big movers, {len(vb_df)} volume breakouts")

    vb_by_symbol = vb_df.groupby("symbol")["signal_time"].apply(list).to_dict()

    results: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    symbols = big_movers["symbol"].unique()

    for idx, symbol in enumerate(symbols):
        if (idx + 1) % 20 == 0:
            print(f"  Progress: {idx + 1}/{len(symbols)} coins")

        coin_dir = RAW_DIR / symbol
        df = load_coin(coin_dir)
        if df is None or len(df) < 200:
            continue

        close = df["close"].values
        high = df["high"].values
        low = df["low"].values
        volume = df["volume"].values
        times = df["open_time"].values

        # Make times naive for comparison
        times_naive = pd.DatetimeIndex(times).tz_localize(None)

        symbol_movers = big_movers[big_movers["symbol"] == symbol]
        vb_times_for_symbol = vb_by_symbol.get(symbol, [])

        for _, row in symbol_movers.iterrows():
            move_start = pd.Timestamp(row["move_start"]).tz_localize(None)
            move_peak = pd.Timestamp(row["move_peak"]).tz_localize(None)

            # Window: 48 bars before move_start to 96 bars after move_peak
            win_start = np.searchsorted(times_naive, move_start) - 48
            win_end = np.searchsorted(times_naive, move_peak) + 96
            win_start = max(win_start, 0)
            win_end = min(win_end, len(close))

            if win_end - win_start < 10:
                continue

            # Strategy A
            entries_a = detect_strategy_a(close, high, low, volume, win_start, win_end)
            for eb in entries_a:
                trade = simulate_trade(close, high, low, eb)
                if trade:
                    trade["symbol"] = symbol
                    trade["strategy"] = "A"
                    trade["signal_time"] = str(times_naive[eb])
                    results["A"].append(trade)

            # Strategy B
            entries_b = detect_strategy_b(close, high, low, volume, win_start, win_end)
            for eb in entries_b:
                trade = simulate_trade(close, high, low, eb)
                if trade:
                    trade["symbol"] = symbol
                    trade["strategy"] = "B"
                    trade["signal_time"] = str(times_naive[eb])
                    results["B"].append(trade)

            # Strategy C — need tz-aware times for VB matching
            # Normalize VB times to naive for comparison
            vb_naive = [pd.Timestamp(t).tz_localize(None) for t in vb_times_for_symbol]
            entries_c = detect_strategy_c(close, low, times_naive, vb_naive, win_start, win_end)
            for eb in entries_c:
                trade = simulate_trade(close, high, low, eb)
                if trade:
                    trade["symbol"] = symbol
                    trade["strategy"] = "C"
                    trade["signal_time"] = str(times_naive[eb])
                    results["C"].append(trade)

    return results


def print_summary(results: dict) -> None:
    """Print comparison table for all strategies."""
    print("\n" + "=" * 80)
    print(f"SCANNER LONG BACKTEST RESULTS")
    print(f"Exit params: SL={STOP_LOSS_PCT:.0%}  TP={TAKE_PROFIT_PCT:.0%}  "
          f"Timeout={TIMEOUT_BARS} bars ({TIMEOUT_BARS * 15 / 60:.0f}h)")
    print("=" * 80)

    labels = {"A": "Vol+Momentum", "B": "PriceScan", "C": "VB Piggyback"}
    best_pf = -1.0
    best_strat = ""

    print(f"\n{'Strategy':<18} {'Trades':>7} {'WR%':>7} {'AvgPnL':>8} {'PF':>7} "
          f"{'AvgBars':>8} {'TP%':>6} {'SL%':>6} {'TO%':>6}")
    print("-" * 80)

    for key in ["A", "B", "C"]:
        trades = results[key]
        if not trades:
            print(f"{labels[key]:<18} {'0':>7}")
            continue

        df = pd.DataFrame(trades)
        n = len(df)
        wins = (df["pnl_pct"] > 0).sum()
        wr = wins / n * 100
        avg_pnl = df["pnl_pct"].mean() * 100
        avg_bars = df["bars_held"].mean()

        gross_profit = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
        gross_loss = abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        tp_pct = (df["exit_reason"] == "take_profit").sum() / n * 100
        sl_pct = (df["exit_reason"] == "stop_loss").sum() / n * 100
        to_pct = (df["exit_reason"] == "timeout").sum() / n * 100

        print(f"{labels[key]:<18} {n:>7} {wr:>6.1f}% {avg_pnl:>7.2f}% {pf:>7.2f} "
              f"{avg_bars:>7.1f} {tp_pct:>5.1f}% {sl_pct:>5.1f}% {to_pct:>5.1f}%")

        if pf > best_pf:
            best_pf = pf
            best_strat = labels[key]

    if best_strat:
        print(f"\n>>> Winner by Profit Factor: {best_strat} (PF={best_pf:.2f})")

    # Per-strategy breakdown
    for key in ["A", "B", "C"]:
        trades = results[key]
        if not trades:
            continue
        df = pd.DataFrame(trades)
        print(f"\n--- {labels[key]} ({key}) breakdown ---")
        print(f"  Unique coins: {df['symbol'].nunique()}")
        grouped = df.groupby("exit_reason").agg(
            count=("pnl_pct", "size"),
            avg_pnl=("pnl_pct", "mean"),
            avg_bars=("bars_held", "mean"),
        )
        for reason, row in grouped.iterrows():
            print(f"  {reason:<12} count={int(row['count']):>4}  "
                  f"avg_pnl={row['avg_pnl'] * 100:>+6.2f}%  avg_bars={row['avg_bars']:.0f}")


def save_to_db(results: dict) -> None:
    """Save all trades to scanner_long_backtest table."""
    all_trades = []
    for key in ["A", "B", "C"]:
        all_trades.extend(results[key])

    if not all_trades:
        print("No trades to save.")
        return

    df = pd.DataFrame(all_trades)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_long_backtest"))
        conn.commit()
    df.to_sql("scanner_long_backtest", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} trades to scanner_long_backtest table.")


if __name__ == "__main__":
    results = run_backtest()
    print_summary(results)
    save_to_db(results)
