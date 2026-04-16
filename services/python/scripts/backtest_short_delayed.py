# scripts/backtest_short_delayed.py
"""Delayed Entry (Bounce Rejection) Short Backtester.

Instead of shorting at the dump, this uses the vol spike + 5% drop as a
WATCHLIST trigger, then waits for the relief bounce to fail before entry.

Phase 1: Detect dump (vol spike + price drop + BTC bearish)
Phase 2: Wait for bounce (+2% from dump close)
Phase 3: Enter on re-rejection (close < bounce_high * 0.97)

Tests two R:R variants:
  A: 5% stop / 5% target / 192-bar timeout (1:1, 48h)
  B: 5% stop / 10% target / 384-bar timeout (2:1, 96h)
"""

import sys
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

# ── Strategy params ──────────────────────────────────────────────────
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
BTC_EMA_PERIOD = 30
COOLDOWN_BARS = 96

# ── Bounce detection params ──────────────────────────────────────────
BOUNCE_WINDOW = 96          # max bars after dump to find bounce + rejection
BOUNCE_MIN_PCT = 0.02       # bounce must be at least +2% from dump close
REJECTION_PCT = 0.03        # price gives back 3% from bounce high

# ── Variant params ───────────────────────────────────────────────────
VARIANT_A = {"stop_pct": 0.05, "tp_pct": 0.05, "max_bars": 192, "label": "A (5%/5% 48h)"}
VARIANT_B = {"stop_pct": 0.05, "tp_pct": 0.10, "max_bars": 384, "label": "B (5%/10% 96h)"}

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


def simulate_short(close, high, low, entry_bar, stop_pct, tp_pct, max_bars):
    """Simulate a short trade from entry_bar.

    Entry at close[entry_bar].
    SL: price goes UP by stop_pct.
    TP: price goes DOWN by tp_pct.
    """
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
                "pnl_pct": -stop_pct,
                "exit_reason": "stop_loss",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }
        if low[bar] <= tp_price:
            return {
                "pnl_pct": tp_pct,
                "exit_reason": "take_profit",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": tp_price,
            }

    # Timeout
    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (entry_price - exit_price) / entry_price
    return {
        "pnl_pct": pnl_pct,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


def scan_coin(symbol, df, btc_times, btc_close, btc_ema):
    """Scan all bars for delayed-entry short signals."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["open_time"].values

    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    trades_a = []
    trades_b = []
    cooldown_until = 0

    for i in range(VOL_MA_PERIOD + PRICE_LOOKBACK, len(close) - BOUNCE_WINDOW):
        if i < cooldown_until:
            continue

        # Phase 1: Detect dump (watchlist trigger)
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
            continue

        rolling_high = np.max(high[i - PRICE_LOOKBACK : i])
        if rolling_high <= 0:
            continue
        price_drop = (rolling_high - close[i]) / rolling_high
        if price_drop < PRICE_MOVE_THRESH:
            continue

        btc_idx = np.searchsorted(btc_times, times[i])
        btc_idx = min(btc_idx, len(btc_close) - 1)
        if btc_close[btc_idx] >= btc_ema[btc_idx]:
            continue

        dump_close = close[i]
        dump_bar = i

        # Phase 2: Wait for bounce (+2% from dump close)
        bounce_high = dump_close
        bounce_bar = None
        for j in range(dump_bar + 1, min(dump_bar + BOUNCE_WINDOW + 1, len(close))):
            if close[j] > bounce_high:
                bounce_high = close[j]
                bounce_bar = j

        if bounce_bar is None or (bounce_high - dump_close) / dump_close < BOUNCE_MIN_PCT:
            continue  # No meaningful bounce

        # Phase 3: Enter on re-rejection (close < bounce_high * (1 - REJECTION_PCT))
        rejection_level = bounce_high * (1 - REJECTION_PCT)
        entry_bar = None
        for j in range(bounce_bar + 1, min(dump_bar + BOUNCE_WINDOW + 1, len(close))):
            if close[j] < rejection_level:
                entry_bar = j
                break

        if entry_bar is None:
            continue  # No rejection within window

        entry_time = pd.Timestamp(times[entry_bar])
        dump_time = pd.Timestamp(times[dump_bar])
        bounce_time = pd.Timestamp(times[bounce_bar])
        bars_to_entry = entry_bar - dump_bar
        bounce_pct = (bounce_high - dump_close) / dump_close

        # Simulate Variant A
        trade_a = simulate_short(
            close, high, low, entry_bar,
            VARIANT_A["stop_pct"], VARIANT_A["tp_pct"], VARIANT_A["max_bars"],
        )
        if trade_a:
            trade_a["symbol"] = symbol
            trade_a["signal_time"] = entry_time
            trade_a["dump_time"] = dump_time
            trade_a["bounce_time"] = bounce_time
            trade_a["variant"] = "A"
            trade_a["bounce_pct"] = bounce_pct
            trade_a["bars_to_entry"] = bars_to_entry
            trades_a.append(trade_a)

        # Simulate Variant B
        trade_b = simulate_short(
            close, high, low, entry_bar,
            VARIANT_B["stop_pct"], VARIANT_B["tp_pct"], VARIANT_B["max_bars"],
        )
        if trade_b:
            trade_b["symbol"] = symbol
            trade_b["signal_time"] = entry_time
            trade_b["dump_time"] = dump_time
            trade_b["bounce_time"] = bounce_time
            trade_b["variant"] = "B"
            trade_b["bounce_pct"] = bounce_pct
            trade_b["bars_to_entry"] = bars_to_entry
            trades_b.append(trade_b)

        cooldown_until = entry_bar + COOLDOWN_BARS

    return trades_a, trades_b


def run_backtest():
    """Scan every coin for delayed-entry short signals."""
    print("Loading BTC data for trend filter...")
    btc_times, btc_close, btc_ema = load_btc()
    print(f"  BTC bars: {len(btc_close)}")

    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(symbol_dirs)} coin directories")

    all_a = []
    all_b = []

    for idx, sym_dir in enumerate(symbol_dirs):
        symbol = sym_dir.name
        if symbol == "BTCUSDT":
            continue

        df = load_coin(sym_dir)
        if df is None or len(df) < 500:
            continue

        trades_a, trades_b = scan_coin(symbol, df, btc_times, btc_close, btc_ema)
        all_a.extend(trades_a)
        all_b.extend(trades_b)

        if (idx + 1) % 20 == 0:
            print(
                f"  {idx + 1}/{len(symbol_dirs)} coins, "
                f"A={len(all_a)} B={len(all_b)} trades"
            )

    print(f"\nScan complete: A={len(all_a)}, B={len(all_b)} trades")
    return all_a, all_b


def print_variant_summary(trades, label):
    """Print summary for one variant."""
    if not trades:
        print(f"\n  {label}: No trades found.")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["pnl_pct"] > 0).sum()
    losses = (df["pnl_pct"] <= 0).sum()
    wr = wins / n * 100
    avg_pnl = df["pnl_pct"].mean() * 100
    total_pnl = df["pnl_pct"].sum() * 100

    gross_profit = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    gross_loss = abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    tp_n = (df["exit_reason"] == "take_profit").sum()
    sl_n = (df["exit_reason"] == "stop_loss").sum()
    to_n = (df["exit_reason"] == "timeout").sum()
    avg_bars = df["bars_held"].mean()
    avg_bounce = df["bounce_pct"].mean() * 100
    avg_bars_to_entry = df["bars_to_entry"].mean()

    print(f"\n  {label}:")
    print(f"    Trades:       {n}")
    print(f"    Wins/Losses:  {wins}/{losses}")
    print(f"    Win Rate:     {wr:.1f}%")
    print(f"    Profit Factor:{pf:>7.2f}")
    print(f"    Avg PnL:      {avg_pnl:+.2f}%")
    print(f"    Total PnL:    {total_pnl:+.1f}%")
    print(f"    Avg Bars Held:{avg_bars:.0f} ({avg_bars * 15 / 60:.1f}h)")
    print(f"    TP/SL/TO:     {tp_n} ({tp_n/n*100:.1f}%) / "
          f"{sl_n} ({sl_n/n*100:.1f}%) / "
          f"{to_n} ({to_n/n*100:.1f}%)")
    print(f"    Avg Bounce:   {avg_bounce:.1f}% before rejection")
    print(f"    Avg Bars to Entry: {avg_bars_to_entry:.0f} "
          f"({avg_bars_to_entry * 15 / 60:.1f}h after dump)")
    print(f"    Unique coins: {df['symbol'].nunique()}")


def print_summary(trades_a, trades_b):
    """Print comparison of both variants."""
    print("\n" + "=" * 80)
    print("DELAYED ENTRY (BOUNCE REJECTION) SHORT BACKTEST")
    print("Phase 1: vol spike >= 3x + price drop >= 5% + BTC < 30-EMA")
    print(f"Phase 2: wait for bounce >= {BOUNCE_MIN_PCT:.0%} from dump close")
    print(f"Phase 3: enter when close < bounce_high * {1 - REJECTION_PCT:.2f}")
    print(f"Cooldown: {COOLDOWN_BARS} bars after entry")
    print("=" * 80)

    print_variant_summary(trades_a, f"Variant {VARIANT_A['label']}")
    print_variant_summary(trades_b, f"Variant {VARIANT_B['label']}")

    # Monthly breakdown for each variant
    for trades, label in [(trades_a, "A"), (trades_b, "B")]:
        if not trades:
            continue
        df = pd.DataFrame(trades)
        print(f"\n  MONTHLY BREAKDOWN — Variant {label}:")
        df["month"] = pd.to_datetime(df["signal_time"]).dt.to_period("M")
        monthly = df.groupby("month").agg(
            trades=("pnl_pct", "size"),
            wins=("pnl_pct", lambda x: (x > 0).sum()),
            total_pnl=("pnl_pct", "sum"),
            avg_pnl=("pnl_pct", "mean"),
        )
        monthly["wr"] = monthly["wins"] / monthly["trades"] * 100

        print(f"  {'Month':<10} {'Trades':>7} {'Wins':>5} "
              f"{'WR%':>6} {'TotPnL':>8} {'AvgPnL':>8}")
        print("  " + "-" * 55)
        for period, row in monthly.iterrows():
            print(f"  {str(period):<10} {int(row['trades']):>7} {int(row['wins']):>5} "
                  f"{row['wr']:>5.1f}% "
                  f"{row['total_pnl']*100:>+7.1f}% {row['avg_pnl']*100:>+7.2f}%")

    print("\n  vs Original (5%/15% 1w): 59545 trades, 32% WR, PF 1.09")

    print("\n" + "=" * 80)
    print("This is a BLIND backtest — no big_movers lookahead.")
    print("Delayed entry waits for the bounce to fail before shorting.")
    print("=" * 80)


def save_to_db(trades_a, trades_b):
    """Save all trades to scanner_short_delayed table."""
    all_trades = trades_a + trades_b
    if not all_trades:
        print("No trades to save.")
        return

    df = pd.DataFrame(all_trades)
    if "month" in df.columns:
        df = df.drop(columns=["month"])

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_short_delayed"))
        conn.commit()
    df.to_sql("scanner_short_delayed", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} trades to scanner_short_delayed table.")


if __name__ == "__main__":
    trades_a, trades_b = run_backtest()
    print_summary(trades_a, trades_b)
    save_to_db(trades_a, trades_b)
