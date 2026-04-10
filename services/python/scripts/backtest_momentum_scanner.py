# scripts/backtest_momentum_scanner.py
"""V6 Momentum Scanner Backtester.

Scans 200+ coins on 4h bars for volume+price breakouts.
Simulates pullback entries, ATR stops, and trailing profits.

Uses existing 4h CSV data from data/raw/15m/.
"""

import csv
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
MIN_VOLUME_USD = 1_000_000  # Skip illiquid coins
MIN_BARS = 200  # Need enough history for volume average

# Scanner parameters — TIGHTENED from v1
VOL_MULT = 4.0        # Volume must be 4x the 20-bar average
PRICE_THRESH = 0.035  # 3.5% price change in 2 bars
LOOKBACK = 20         # 20-bar rolling window for volume average

# Trade parameters
PULLBACK_PCT = 0.02     # Wait for 2% pullback to enter (was 3%)
ATR_STOP_MULT = 2.0     # Stop at 2x ATR from entry
ATR_TP_MULT = 5.0       # TP at 5x ATR (2.5:1 R:R)
MAX_HOLD_BARS = 192     # 48h max hold (192 x 15min bars) — big moves play out over 1-2 days
RISK_PER_TRADE = 0.02   # 2% equity risk
MAX_LEVERAGE = 3.0
MAX_CONCURRENT = 2
MAX_DAILY_MOVE = 0.30   # Skip if already up >30% (give room for big moves)

# Regime filter
BTC_TREND_PERIOD = 30   # 30 bars = 5 days on 4h for trend filter


def load_coin(symbol_dir: Path) -> pd.DataFrame | None:
    """Load all 4h CSVs for one coin, return DataFrame or None."""
    csv_files = sorted(symbol_dir.glob("*.csv"))
    if not csv_files:
        return None

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            if "close" in df.columns and len(df) > 0:
                frames.append(df)
        except Exception:
            continue

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["open_time"] = pd.to_numeric(combined["open_time"], errors="coerce")
    combined = combined.dropna(subset=["open_time", "close"])

    # Handle mixed timestamps: some files use ms (13 digits), others us (16 digits)
    ts = combined["open_time"].values.astype(float)
    ts = np.where(ts > 1e13, ts / 1e3, ts)  # Convert microseconds to milliseconds
    combined["open_time"] = pd.to_datetime(ts, unit="ms", utc=True)

    combined = combined.sort_values("open_time").reset_index(drop=True)

    for col in ["open", "high", "low", "close", "volume"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined = combined.dropna(subset=["close", "volume"])
    return combined


def compute_atr(high, low, close, period=14):
    """Compute ATR as fraction of price."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = (high[0] - low[0]) / close[0] if close[0] > 0 else 0
    for i in range(1, n):
        if close[i] == 0:
            tr[i] = 0
            continue
        tr[i] = max(
            (high[i] - low[i]) / close[i],
            abs(high[i] - close[i-1]) / close[i],
            abs(low[i] - close[i-1]) / close[i],
        )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period-1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def detect_breakouts(df: pd.DataFrame, vol_mult: float = VOL_MULT,
                     price_thresh: float = PRICE_THRESH,
                     lookback: int = LOOKBACK) -> list[dict]:
    """Detect volume+price breakout bars in a single coin's data."""
    close = df["close"].values.astype(float)
    volume = df["volume"].values.astype(float)
    n = len(close)

    signals = []
    vol_ma = pd.Series(volume).rolling(lookback).mean().values

    for i in range(lookback + 1, n):
        if np.isnan(vol_ma[i]) or vol_ma[i] == 0:
            continue

        # Volume spike
        if volume[i] < vol_mult * vol_ma[i]:
            continue

        # Price change
        if close[i-1] == 0:
            continue
        ret = (close[i] - close[i-1]) / close[i-1]
        if abs(ret) < price_thresh:
            continue

        # Check not already up >10% in last 6 bars (1.5h on 15min)
        # Use 24h window (96 bars) for the "too late" filter
        if i >= 96:
            day_ret = (close[i] - close[i-96]) / close[i-96]
            if abs(day_ret) > MAX_DAILY_MOVE:
                continue

        # Cooldown: skip if we already signaled this coin in last 8 bars (2h)
        if signals and signals[-1]["bar"] > i - 8:
            continue

        direction = 1 if ret > 0 else -1
        signals.append({
            "bar": i,
            "direction": direction,
            "price": close[i],
            "volume_ratio": volume[i] / vol_ma[i],
            "price_change": ret,
        })

    return signals


def simulate_pullback_trade(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray | None,
    signal_bar: int, direction: int,
    pullback_pct: float, atr_stop_mult: float,
    atr_value: float, max_hold_bars: int,
    atr_tp_mult: float = ATR_TP_MULT,
) -> dict | None:
    """Simulate a momentum trade designed to catch 8-16% of a 10-20% move.

    Key changes from v1:
    - ENTER FAST: next bar after signal (don't wait for pullback)
    - STOP WIDE: 3% fixed minimum (don't get noise-stopped on big moves)
    - NO FIXED TP: let winners run, exit on momentum death
    - TRAIL AGGRESSIVELY: once in profit, trail to lock in gains
    - EXIT ON VOLUME DEATH: if volume drops below average, momentum is over
    """
    n = len(close)
    if signal_bar + 1 >= n:
        return None

    # ENTER FAST: next bar open (no pullback waiting)
    entry_bar = signal_bar + 1
    entry_price = close[entry_bar]
    if entry_price == 0:
        return None

    # STOP: wider of 3% or 2x ATR (don't get stopped on noise)
    stop_pct = max(0.03, atr_value * atr_stop_mult)

    if direction == 1:
        stop_price = entry_price * (1 - stop_pct)
    else:
        stop_price = entry_price * (1 + stop_pct)

    # Track high water mark for trailing
    hwm = entry_price  # High water mark
    trail_active = False

    exit_bar = None
    exit_price = None
    exit_reason = None

    # Volume average for momentum death detection
    if volume is not None and signal_bar >= 20:
        vol_avg = np.mean(volume[signal_bar - 20:signal_bar])
    else:
        vol_avg = None

    for i in range(entry_bar + 1, min(entry_bar + max_hold_bars + 1, n)):
        # Update high water mark
        if direction == 1:
            if close[i] > hwm:
                hwm = close[i]
        else:
            if close[i] < hwm:
                hwm = close[i]

        # CHECK STOP
        if direction == 1:
            if low[i] <= stop_price:
                exit_bar = i
                exit_price = stop_price
                exit_reason = "stop_loss"
                break
        else:
            if high[i] >= stop_price:
                exit_bar = i
                exit_price = stop_price
                exit_reason = "stop_loss"
                break

        # TRAILING STOP: activate after +5% profit, trail loosely (keep 30% of max gain)
        # This lets winners run: a +12% move can pull back to +8.4% before we exit
        if direction == 1:
            unrealized = (close[i] - entry_price) / entry_price
            if unrealized > 0.05:
                trail_active = True
                # Trail: lock in only 30% of max gain (give back 70% — let it breathe)
                new_stop = entry_price + (hwm - entry_price) * 0.30
                # But never below breakeven once trail is active
                new_stop = max(new_stop, entry_price * 1.005)
                if new_stop > stop_price:
                    stop_price = new_stop
        else:
            unrealized = (entry_price - close[i]) / entry_price
            if unrealized > 0.05:
                trail_active = True
                new_stop = entry_price - (entry_price - hwm) * 0.30
                new_stop = min(new_stop, entry_price * 0.995)
                if new_stop < stop_price:
                    stop_price = new_stop

        # MOMENTUM DEATH: volume drops below average for 2 bars AND we're in profit
        if vol_avg is not None and volume is not None and i >= 2 and trail_active:
            vol_dying = volume[i] < vol_avg and volume[i-1] < vol_avg
            if vol_dying:
                exit_bar = i
                exit_price = close[i]
                exit_reason = "momentum_death"
                break

    # Time exit
    if exit_bar is None:
        exit_bar = min(entry_bar + max_hold_bars, n - 1)
        exit_price = close[exit_bar]
        exit_reason = "time_exit"

    # Calculate PnL
    if direction == 1:
        pnl_pct = (exit_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - exit_price) / entry_price

    return {
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "hold_bars": exit_bar - entry_bar,
        "exit_reason": exit_reason,
        "signal_volume_ratio": 0,
    }


def main():
    start = time.time()
    logger.info("V6 Momentum Scanner Backtester")
    logger.info(f"Data: {RAW_DIR}")
    logger.info(f"Params: vol_mult={VOL_MULT} price_thresh={PRICE_THRESH} "
                f"pullback={PULLBACK_PCT} atr_stop={ATR_STOP_MULT} atr_tp={ATR_TP_MULT}")

    # Load BTC for trend filter
    btc_dir = RAW_DIR / "BTCUSDT"
    btc_df = load_coin(btc_dir)
    if btc_df is None:
        logger.error("Cannot load BTC data!")
        return
    btc_close = btc_df["close"].values.astype(float)
    btc_ema = pd.Series(btc_close).ewm(span=BTC_TREND_PERIOD).mean().values
    btc_times = btc_df["open_time"].values
    logger.info(f"BTC loaded: {len(btc_df)} bars")

    # Load all coins
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    logger.info(f"Loading {len(symbol_dirs)} coins...")

    all_trades = []
    coins_with_signals = 0
    total_signals = 0

    for sym_dir in symbol_dirs:
        symbol = sym_dir.name
        df = load_coin(sym_dir)
        if df is None or len(df) < MIN_BARS:
            continue

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        times = df["open_time"].values

        # Check minimum volume (rough filter)
        avg_vol = np.nanmean(volume[-100:]) if len(volume) >= 100 else np.nanmean(volume)
        avg_price = np.nanmean(close[-100:]) if len(close) >= 100 else np.nanmean(close)
        if avg_vol * avg_price < MIN_VOLUME_USD / 96:  # 15min bar, 96 bars per day
            continue

        # Detect breakouts
        signals = detect_breakouts(df)
        if not signals:
            continue

        coins_with_signals += 1
        total_signals += len(signals)

        # Compute ATR
        atr = compute_atr(high, low, close)

        for sig in signals:
            bar = sig["bar"]
            if np.isnan(atr[bar]) or atr[bar] == 0:
                continue

            # BTC trend filter: find closest BTC bar
            # Only long if BTC > EMA, only short if BTC < EMA
            sig_time = times[bar]
            btc_idx = np.searchsorted(btc_times, sig_time)
            btc_idx = min(btc_idx, len(btc_close) - 1)
            if sig["direction"] == 1 and btc_close[btc_idx] < btc_ema[btc_idx]:
                continue  # No longs in BTC downtrend
            if sig["direction"] == -1 and btc_close[btc_idx] > btc_ema[btc_idx]:
                continue  # No shorts in BTC uptrend

            trade = simulate_pullback_trade(
                close=close, high=high, low=low,
                volume=volume,
                signal_bar=bar, direction=sig["direction"],
                pullback_pct=PULLBACK_PCT, atr_stop_mult=ATR_STOP_MULT,
                atr_value=atr[bar], max_hold_bars=MAX_HOLD_BARS,
                atr_tp_mult=ATR_TP_MULT,
            )

            if trade is not None:
                trade["symbol"] = symbol
                trade["signal_time"] = str(times[bar])
                trade["signal_volume_ratio"] = sig["volume_ratio"]
                trade["signal_price_change"] = sig["price_change"]
                all_trades.append(trade)

    elapsed = time.time() - start
    logger.info(f"\nScanning complete in {elapsed:.0f}s")
    logger.info(f"Coins with signals: {coins_with_signals}")
    logger.info(f"Total signals: {total_signals}")
    logger.info(f"Total trades: {len(all_trades)}")

    if not all_trades:
        logger.warning("No trades generated!")
        return

    # Compute metrics
    trades_df = pd.DataFrame(all_trades)
    pnls = trades_df["pnl_pct"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls)
    avg_win = np.mean(wins) if len(wins) > 0 else 0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
    profit_factor = sum(wins) / abs(sum(losses)) if len(losses) > 0 and sum(losses) != 0 else float("inf")
    expectancy = np.mean(pnls)

    # Estimate monthly return
    total_bars = len(btc_df)
    total_months = total_bars / (6 * 30)  # 6 bars/day * 30 days
    trades_per_month = len(all_trades) / total_months
    monthly_return = expectancy * trades_per_month * MAX_LEVERAGE

    # Exit reason breakdown
    exit_reasons = trades_df["exit_reason"].value_counts()

    logger.info("\n" + "=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total trades: {len(all_trades)}")
    logger.info(f"Trades/month: {trades_per_month:.1f}")
    logger.info(f"Win rate: {win_rate:.1%}")
    logger.info(f"Avg winner: {avg_win:.2%}")
    logger.info(f"Avg loser: {avg_loss:.2%}")
    logger.info(f"R:R ratio: {avg_win/avg_loss:.2f}:1" if avg_loss > 0 else "R:R: inf")
    logger.info(f"Profit factor: {profit_factor:.2f}")
    logger.info(f"Expectancy/trade: {expectancy:.2%}")
    logger.info(f"Est monthly return (3x lev): {monthly_return:.1%}")
    logger.info(f"\nExit reasons:")
    for reason, count in exit_reasons.items():
        logger.info(f"  {reason}: {count} ({count/len(all_trades):.1%})")

    # Top performers
    logger.info(f"\nTop 10 trades:")
    top = trades_df.nlargest(10, "pnl_pct")
    for _, t in top.iterrows():
        logger.info(f"  {t['symbol']}: {t['pnl_pct']:+.1%} ({t['exit_reason']}) "
                    f"vol_ratio={t['signal_volume_ratio']:.1f}x")

    # Worst trades
    logger.info(f"\nBottom 5 trades:")
    bottom = trades_df.nsmallest(5, "pnl_pct")
    for _, t in bottom.iterrows():
        logger.info(f"  {t['symbol']}: {t['pnl_pct']:+.1%} ({t['exit_reason']})")

    # Monthly breakdown (approximate)
    trades_df["month"] = pd.to_datetime(trades_df["signal_time"]).dt.to_period("M")
    monthly = trades_df.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])
    monthly["leveraged"] = monthly["sum"] * MAX_LEVERAGE
    logger.info(f"\nMonthly returns (3x leverage):")
    for month, row in monthly.iterrows():
        logger.info(f"  {month}: {row['leveraged']:+.1%} ({int(row['count'])} trades)")

    profitable_months = (monthly["leveraged"] > 0).sum()
    total_months_count = len(monthly)
    logger.info(f"\nProfitable months: {profitable_months}/{total_months_count} "
                f"({profitable_months/total_months_count:.0%})")


if __name__ == "__main__":
    main()
