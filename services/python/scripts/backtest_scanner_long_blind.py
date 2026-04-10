# scripts/backtest_scanner_long_blind.py
"""Blind Scanner Long Backtester.

Scans ALL coins across ALL bars chronologically for Strategy A (Vol+Momentum)
entries — NO big_movers lookahead. This is a "live simulation" pretending we're
scanning from the past to now.

Includes BTC trend filter: only go long when BTC close > BTC 30-bar EMA.
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
VOL_SPIKE = 3.0           # 3x volume spike
PRICE_MOVE_THRESH = 0.05  # 5% from rolling low
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96       # 24h lookback for rolling low
BTC_EMA_PERIOD = 30       # BTC trend filter

# ── Exit params ──────────────────────────────────────────────────────
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
COOLDOWN_BARS = 96

engine = create_engine(DB_URL)


def load_btc():
    """Load BTC 15m data and compute 30-bar EMA."""
    df = load_coin(RAW_DIR / "BTCUSDT")
    if df is None or len(df) < BTC_EMA_PERIOD:
        raise RuntimeError("Cannot load BTC data — needed for trend filter")
    close = df["close"].values.astype(float)
    times = df["open_time"].values
    ema = pd.Series(close).ewm(span=BTC_EMA_PERIOD, adjust=False).mean().values
    return times, close, ema


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

    Entry at close[entry_bar]. Returns dict with pnl_pct, exit_reason,
    bars_held, entry_price, exit_price, or None if not enough data.
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


def scan_coin(symbol, df, btc_times, btc_close, btc_ema):
    """Scan all bars of one coin for Strategy A entries with BTC filter."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["open_time"].values

    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    trades = []
    cooldown_until = 0

    for i in range(VOL_MA_PERIOD + PRICE_LOOKBACK, len(close)):
        if i < cooldown_until:
            continue

        # Gate 1: Volume spike
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
            continue

        # Gate 2: Price up from rolling low
        rolling_low = np.min(low[i - PRICE_LOOKBACK : i])
        if rolling_low <= 0:
            continue
        if (close[i] - rolling_low) / rolling_low < PRICE_MOVE_THRESH:
            continue

        # Gate 3: BTC trend filter
        btc_idx = np.searchsorted(btc_times, times[i])
        btc_idx = min(btc_idx, len(btc_close) - 1)
        if btc_close[btc_idx] < btc_ema[btc_idx]:
            continue  # Skip if BTC is below EMA (bearish)

        # Entry signal — simulate trade
        trade = simulate_trade(close, high, low, i)
        if trade:
            trade["symbol"] = symbol
            trade["signal_time"] = str(times[i])
            trades.append(trade)
            cooldown_until = i + COOLDOWN_BARS

    return trades


def run_backtest():
    """Scan every coin, every bar — no big_movers lookahead."""
    print("Loading BTC data for trend filter...")
    btc_times, btc_close, btc_ema = load_btc()
    print(f"  BTC bars: {len(btc_close)}")

    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(symbol_dirs)} coin directories")

    all_trades = []

    for idx, sym_dir in enumerate(symbol_dirs):
        symbol = sym_dir.name
        if symbol == "BTCUSDT":
            continue  # Skip BTC itself

        df = load_coin(sym_dir)
        if df is None or len(df) < 500:
            continue

        trades = scan_coin(symbol, df, btc_times, btc_close, btc_ema)
        all_trades.extend(trades)

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(symbol_dirs)} coins, {len(all_trades)} trades so far")

    print(f"\nScan complete: {len(all_trades)} total trades across all coins")
    return all_trades


def print_summary(trades):
    """Print detailed summary of blind backtest results."""
    print("\n" + "=" * 80)
    print("BLIND SCANNER LONG BACKTEST — Strategy A (Vol+Momentum)")
    print("** NO big_movers lookahead — this is a live simulation **")
    print(f"Exit params: SL={STOP_LOSS_PCT:.0%}  TP={TAKE_PROFIT_PCT:.0%}  "
          f"Timeout={TIMEOUT_BARS} bars ({TIMEOUT_BARS * 15 / 60:.0f}h)")
    print(f"Filters: vol_spike>={VOL_SPIKE}x  price_move>={PRICE_MOVE_THRESH:.0%}  "
          f"BTC>{BTC_EMA_PERIOD}-EMA  cooldown={COOLDOWN_BARS} bars")
    print("=" * 80)

    if not trades:
        print("No trades found.")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["pnl_pct"] > 0).sum()
    losses = (df["pnl_pct"] <= 0).sum()
    wr = wins / n * 100
    avg_pnl = df["pnl_pct"].mean() * 100
    total_pnl = df["pnl_pct"].sum() * 100
    avg_bars = df["bars_held"].mean()

    gross_profit = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    gross_loss = abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    tp_n = (df["exit_reason"] == "take_profit").sum()
    sl_n = (df["exit_reason"] == "stop_loss").sum()
    to_n = (df["exit_reason"] == "timeout").sum()

    # ── 1. Overall stats ────────────────────────────────────────────
    print(f"\n{'OVERALL STATS':}")
    print(f"  Trades:       {n}")
    print(f"  Wins/Losses:  {wins}/{losses}")
    print(f"  Win Rate:     {wr:.1f}%")
    print(f"  Profit Factor:{pf:>7.2f}")
    print(f"  Avg PnL:      {avg_pnl:+.2f}%")
    print(f"  Total PnL:    {total_pnl:+.1f}%")
    print(f"  Avg Bars:     {avg_bars:.0f} ({avg_bars * 15 / 60:.1f}h)")
    print(f"  TP/SL/TO:     {tp_n} ({tp_n/n*100:.1f}%) / "
          f"{sl_n} ({sl_n/n*100:.1f}%) / "
          f"{to_n} ({to_n/n*100:.1f}%)")
    print(f"  Unique coins: {df['symbol'].nunique()}")

    # ── 2. Monthly breakdown ────────────────────────────────────────
    print(f"\n{'MONTHLY BREAKDOWN':}")
    df["month"] = pd.to_datetime(df["signal_time"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades=("pnl_pct", "size"),
        wins=("pnl_pct", lambda x: (x > 0).sum()),
        losses=("pnl_pct", lambda x: (x <= 0).sum()),
        total_pnl=("pnl_pct", "sum"),
        avg_pnl=("pnl_pct", "mean"),
    )
    monthly["wr"] = monthly["wins"] / monthly["trades"] * 100

    print(f"  {'Month':<10} {'Trades':>7} {'Wins':>5} {'Loss':>5} "
          f"{'WR%':>6} {'TotPnL':>8} {'AvgPnL':>8}")
    print("  " + "-" * 60)
    for period, row in monthly.iterrows():
        print(f"  {str(period):<10} {int(row['trades']):>7} {int(row['wins']):>5} "
              f"{int(row['losses']):>5} {row['wr']:>5.1f}% "
              f"{row['total_pnl']*100:>+7.1f}% {row['avg_pnl']*100:>+7.2f}%")

    # ── 3. Top 10 coins by total PnL ───────────────────────────────
    print(f"\n{'TOP 10 COINS BY TOTAL PNL':}")
    coin_pnl = df.groupby("symbol").agg(
        trades=("pnl_pct", "size"),
        total_pnl=("pnl_pct", "sum"),
        avg_pnl=("pnl_pct", "mean"),
        wr=("pnl_pct", lambda x: (x > 0).sum() / len(x) * 100),
    ).sort_values("total_pnl", ascending=False).head(10)

    print(f"  {'Symbol':<14} {'Trades':>7} {'TotPnL':>8} {'AvgPnL':>8} {'WR%':>6}")
    print("  " + "-" * 50)
    for sym, row in coin_pnl.iterrows():
        print(f"  {sym:<14} {int(row['trades']):>7} "
              f"{row['total_pnl']*100:>+7.1f}% {row['avg_pnl']*100:>+7.2f}% "
              f"{row['wr']:>5.1f}%")

    # ── 4. Bottom 10 coins ──────────────────────────────────────────
    print(f"\n{'BOTTOM 10 COINS BY TOTAL PNL':}")
    coin_bottom = df.groupby("symbol").agg(
        trades=("pnl_pct", "size"),
        total_pnl=("pnl_pct", "sum"),
        avg_pnl=("pnl_pct", "mean"),
        wr=("pnl_pct", lambda x: (x > 0).sum() / len(x) * 100),
    ).sort_values("total_pnl", ascending=True).head(10)

    print(f"  {'Symbol':<14} {'Trades':>7} {'TotPnL':>8} {'AvgPnL':>8} {'WR%':>6}")
    print("  " + "-" * 50)
    for sym, row in coin_bottom.iterrows():
        print(f"  {sym:<14} {int(row['trades']):>7} "
              f"{row['total_pnl']*100:>+7.1f}% {row['avg_pnl']*100:>+7.2f}% "
              f"{row['wr']:>5.1f}%")

    print("\n" + "=" * 80)
    print("This is a BLIND backtest -- no big_movers lookahead.")
    print("These numbers reflect what the scanner would have produced in real-time.")
    print("=" * 80)


def save_to_db(trades):
    """Save all trades to scanner_long_blind table."""
    if not trades:
        print("No trades to save.")
        return

    df = pd.DataFrame(trades)
    if "month" in df.columns:
        df = df.drop(columns=["month"])

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_long_blind"))
        conn.commit()
    df.to_sql("scanner_long_blind", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} trades to scanner_long_blind table.")


if __name__ == "__main__":
    trades = run_backtest()
    print_summary(trades)
    save_to_db(trades)
