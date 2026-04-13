# scripts/backtest_experiment_tiered.py
"""Tiered SL + Widening Trail Backtest Experiment.

Tests two improvements over the baseline V6 scanner:
1. Tiered Stop Loss:
   - At -3%: sell 50% of position
   - At -4%: sell 75% cumulative (25% more)
   - At -5%: sell 100% (fully out)
   Average loss becomes ~3.5% instead of full -5%

2. Widening Trailing Take Profit:
   - +5% to +10%:  trail at 2.5x ATR (normal)
   - +10% to +20%: trail at 3.5x ATR (wider)
   - +20%+:        trail at 5.0x ATR (monster move, let it ride)

3. Staged Partial TP: sell 33% at +8% (same as baseline)
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from scripts.backtest_momentum_scanner import (
    compute_atr,
    detect_breakouts,
    load_coin,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
MIN_VOLUME_USD = 1_000_000
MIN_BARS = 200

# Scanner params — same as baseline
VOL_MULT = 4.0
PRICE_THRESH = 0.035
LOOKBACK = 20

# Trade params
ATR_STOP_MULT = 2.0
MAX_HOLD_BARS = 192
MAX_LEVERAGE = 3.0
MAX_DAILY_MOVE = 0.30

# BTC trend filter
BTC_TREND_PERIOD = 30

# Tiered stop loss levels (as fraction of entry)
TIER1_PCT = 0.03   # -3%: sell 50% cumulative
TIER2_PCT = 0.04   # -4%: sell 75% cumulative
TIER3_PCT = 0.05   # -5%: sell 100% (fully out)

# Widening trail ATR multipliers
TRAIL_MULT_NORMAL = 2.5   # +5% to +10%
TRAIL_MULT_WIDE   = 3.5   # +10% to +20%
TRAIL_MULT_MONSTER = 5.0  # +20%+

# Staged partial TP (same as baseline)
PARTIAL_TP_PCT   = 0.08   # +8% trigger
PARTIAL_TP_FRAC  = 0.33   # sell 33%


def simulate_tiered_trade(
    close: np.ndarray, high: np.ndarray, low: np.ndarray,
    volume: np.ndarray | None,
    signal_bar: int, direction: int,
    atr_value: float, max_hold_bars: int,
) -> dict | None:
    """Simulate trade with tiered stop loss and widening trailing stop.

    Exit reasons:
    - tiered_stop:     hit one of the 3 tiered stop levels
    - trailing_stop:   ATR trail (widening) triggered
    - momentum_death:  volume drying up while in trail
    - time_exit:       max_hold_bars safety net
    """
    n = len(close)
    if signal_bar + 1 >= n:
        return None

    entry_bar = signal_bar + 1
    entry_price = close[entry_bar]
    if entry_price == 0:
        return None

    # Position sizing state
    # We track what fraction of original position is still open
    # and the PnL locked in from closed portions.
    pos_frac = 1.0   # fraction of original position still open
    locked_pnl = 0.0  # PnL locked in from partial closes

    # Tiered stop state
    tier1_triggered = False  # -3% → sold 50%
    tier2_triggered = False  # -4% → sold 75% cumulative (25% more)
    # tier3 = fully out — handled by break

    # Partial TP state
    partial_taken = False

    # Trail state
    trail_active = False
    hwm = entry_price  # high-water-mark (low-water for shorts)
    trail_dist = atr_value * TRAIL_MULT_NORMAL  # current trail distance (fraction)
    stop_price = None  # will be set once trail activates

    # Volume average for momentum death
    if volume is not None and signal_bar >= 20:
        vol_avg = np.mean(volume[signal_bar - 20:signal_bar])
    else:
        vol_avg = None

    exit_bar = None
    exit_price = None
    exit_reason = None

    for i in range(entry_bar + 1, min(entry_bar + max_hold_bars + 1, n)):
        c = close[i]
        h = high[i]
        lo = low[i]

        # Current unrealized PnL on remaining position
        if direction == 1:
            unrealized = (c - entry_price) / entry_price
            unrealized_low  = (lo - entry_price) / entry_price  # worst case this bar
        else:
            unrealized = (entry_price - c) / entry_price
            unrealized_low  = (entry_price - h) / entry_price   # worst case this bar (short)

        # --- TIERED STOP LOSS ---
        # Check worst-case intra-bar move for stop triggers
        if pos_frac > 0:
            if not tier1_triggered and unrealized_low <= -TIER1_PCT:
                # -3%: close 50% at -3%
                close_pct = 0.50
                pnl_this = -TIER1_PCT * close_pct
                locked_pnl += pnl_this
                pos_frac -= close_pct
                tier1_triggered = True

            if not tier2_triggered and unrealized_low <= -TIER2_PCT:
                # -4%: close another 25% (total 75% closed) at -4%
                close_pct = 0.25
                pnl_this = -TIER2_PCT * close_pct
                locked_pnl += pnl_this
                pos_frac -= close_pct
                tier2_triggered = True

            if unrealized_low <= -TIER3_PCT:
                # -5%: close remaining at -5%
                pnl_this = -TIER3_PCT * pos_frac
                locked_pnl += pnl_this
                pos_frac = 0.0
                exit_bar = i
                exit_price = entry_price * (1 - TIER3_PCT) if direction == 1 else entry_price * (1 + TIER3_PCT)
                exit_reason = "tiered_stop"
                break

        # --- PARTIAL TP at +8% ---
        if not partial_taken and unrealized > PARTIAL_TP_PCT and pos_frac > 0:
            close_frac = PARTIAL_TP_FRAC * pos_frac  # 33% of current remaining
            pnl_this = PARTIAL_TP_PCT * close_frac
            locked_pnl += pnl_this
            pos_frac -= close_frac
            partial_taken = True

        # --- UPDATE HIGH WATER MARK ---
        if direction == 1:
            if h > hwm:
                hwm = h
        else:
            if lo < hwm:
                hwm = lo

        # --- WIDENING TRAIL ---
        if unrealized > 0.05 and pos_frac > 0:
            trail_active = True

            # Determine current trail multiplier based on unrealized profit
            if unrealized >= 0.20:
                trail_mult = TRAIL_MULT_MONSTER
            elif unrealized >= 0.10:
                trail_mult = TRAIL_MULT_WIDE
            else:
                trail_mult = TRAIL_MULT_NORMAL

            trail_dist = atr_value * trail_mult

            if direction == 1:
                new_stop = hwm * (1 - trail_dist)
                new_stop = max(new_stop, entry_price * 1.002)  # never below breakeven
                if stop_price is None or new_stop > stop_price:
                    stop_price = new_stop
                # Check if trail stop triggered this bar
                if lo <= stop_price:
                    exit_bar = i
                    exit_price = stop_price
                    exit_reason = "trailing_stop"
                    locked_pnl += (exit_price - entry_price) / entry_price * pos_frac
                    pos_frac = 0.0
                    break
            else:
                new_stop = hwm * (1 + trail_dist)
                new_stop = min(new_stop, entry_price * 0.998)
                if stop_price is None or new_stop < stop_price:
                    stop_price = new_stop
                if h >= stop_price:
                    exit_bar = i
                    exit_price = stop_price
                    exit_reason = "trailing_stop"
                    locked_pnl += (entry_price - exit_price) / entry_price * pos_frac
                    pos_frac = 0.0
                    break

        # --- MOMENTUM DEATH ---
        if vol_avg is not None and volume is not None and i >= 3 and trail_active and pos_frac > 0:
            vol_dying = (
                volume[i]   < vol_avg * 0.8
                and volume[i-1] < vol_avg * 0.8
                and volume[i-2] < vol_avg * 0.8
            )
            if vol_dying:
                exit_bar = i
                exit_price = c
                exit_reason = "momentum_death"
                if direction == 1:
                    locked_pnl += (exit_price - entry_price) / entry_price * pos_frac
                else:
                    locked_pnl += (entry_price - exit_price) / entry_price * pos_frac
                pos_frac = 0.0
                break

    # Safety net time exit
    if exit_bar is None:
        exit_bar = min(entry_bar + max_hold_bars, n - 1)
        exit_price = close[exit_bar]
        exit_reason = "time_exit"
        if direction == 1:
            locked_pnl += (exit_price - entry_price) / entry_price * pos_frac
        else:
            locked_pnl += (entry_price - exit_price) / entry_price * pos_frac
        pos_frac = 0.0

    pnl_pct = locked_pnl

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
    logger.info("Tiered SL + Widening Trail Backtest Experiment")
    logger.info(f"Data: {RAW_DIR}")
    logger.info(f"Scanner params: VOL_MULT={VOL_MULT} PRICE_THRESH={PRICE_THRESH}")
    logger.info(f"Tiered SL: -3%(50%) / -4%(75%) / -5%(100%)")
    logger.info(f"Widening trail: 2.5x(<10%) / 3.5x(<20%) / 5.0x(20%+) ATR")
    logger.info(f"Staged partial TP: 33% at +8%")

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

        # Minimum liquidity filter
        avg_vol = np.nanmean(volume[-100:]) if len(volume) >= 100 else np.nanmean(volume)
        avg_price = np.nanmean(close[-100:]) if len(close) >= 100 else np.nanmean(close)
        if avg_vol * avg_price < MIN_VOLUME_USD / 96:
            continue

        signals = detect_breakouts(df, vol_mult=VOL_MULT, price_thresh=PRICE_THRESH)
        if not signals:
            continue

        coins_with_signals += 1
        total_signals += len(signals)

        atr = compute_atr(high, low, close)

        for sig in signals:
            bar = sig["bar"]
            if np.isnan(atr[bar]) or atr[bar] == 0:
                continue

            # BTC trend filter
            sig_time = times[bar]
            btc_idx = np.searchsorted(btc_times, sig_time)
            btc_idx = min(btc_idx, len(btc_close) - 1)
            if sig["direction"] == 1 and btc_close[btc_idx] < btc_ema[btc_idx]:
                continue
            if sig["direction"] == -1 and btc_close[btc_idx] > btc_ema[btc_idx]:
                continue

            trade = simulate_tiered_trade(
                close=close, high=high, low=low,
                volume=volume,
                signal_bar=bar, direction=sig["direction"],
                atr_value=atr[bar], max_hold_bars=MAX_HOLD_BARS,
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

    trades_df = pd.DataFrame(all_trades)
    pnls = trades_df["pnl_pct"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
    profit_factor = (
        float(sum(wins)) / float(abs(sum(losses)))
        if len(losses) > 0 and sum(losses) != 0
        else float("inf")
    )
    expectancy = float(np.mean(pnls))

    total_bars = len(btc_df)
    total_months = total_bars / (96 * 30)  # 96 bars/day on 15min * 30 days
    trades_per_month = len(all_trades) / total_months if total_months > 0 else 0
    monthly_return = expectancy * trades_per_month * MAX_LEVERAGE

    exit_reasons = trades_df["exit_reason"].value_counts()

    logger.info("\n" + "=" * 60)
    logger.info("TIERED EXPERIMENT RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total trades:        {len(all_trades)}")
    logger.info(f"Trades/month:        {trades_per_month:.1f}")
    logger.info(f"Win rate:            {win_rate:.1%}")
    logger.info(f"Avg winner:          {avg_win:.2%}")
    logger.info(f"Avg loser:           {avg_loss:.2%}")
    if avg_loss > 0:
        logger.info(f"R:R ratio:           {avg_win/avg_loss:.2f}:1")
    else:
        logger.info("R:R ratio:           inf")
    logger.info(f"Profit factor:       {profit_factor:.2f}")
    logger.info(f"Expectancy/trade:    {expectancy:.2%}")
    logger.info(f"Est monthly return (3x lev): {monthly_return:.1%}")

    logger.info(f"\nExit reasons:")
    for reason, count in exit_reasons.items():
        logger.info(f"  {reason}: {count} ({count/len(all_trades):.1%})")

    logger.info(f"\nTop 10 winners:")
    top = trades_df.nlargest(10, "pnl_pct")
    for _, t in top.iterrows():
        logger.info(f"  {t['symbol']}: {t['pnl_pct']:+.1%} ({t['exit_reason']}) "
                    f"vol_ratio={t['signal_volume_ratio']:.1f}x")

    logger.info(f"\nBottom 5 losers:")
    bottom = trades_df.nsmallest(5, "pnl_pct")
    for _, t in bottom.iterrows():
        logger.info(f"  {t['symbol']}: {t['pnl_pct']:+.1%} ({t['exit_reason']})")

    # Monthly breakdown
    trades_df["month"] = pd.to_datetime(trades_df["signal_time"]).dt.to_period("M")
    monthly = trades_df.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])
    monthly["leveraged"] = monthly["sum"] * MAX_LEVERAGE
    logger.info(f"\nMonthly returns (3x leverage):")
    for month, row in monthly.iterrows():
        logger.info(f"  {month}: {row['leveraged']:+.1%} ({int(row['count'])} trades)")

    profitable_months = (monthly["leveraged"] > 0).sum()
    total_months_count = len(monthly)
    logger.info(
        f"\nProfitable months: {profitable_months}/{total_months_count} "
        f"({profitable_months/total_months_count:.0%})"
    )


if __name__ == "__main__":
    main()
