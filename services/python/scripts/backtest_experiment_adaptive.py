# scripts/backtest_experiment_adaptive.py
"""Pure ATR-Based Adaptive Stops Backtest Experiment.

The Problem: Fixed percentage stops treat all coins equally.
  - BTC ATR = 0.32% per 15min candle → 3% stop = 9.4x ATR (too wide, wastes capital)
  - ALPACA ATR = 9.10% per candle   → 3% stop = 0.3x ATR (instant stop)

The Fix: Scale ALL stops and trails to each coin's ATR at signal time.

Tiered Stop Loss (ATR multiples):
  - At -2x ATR: sell 50% of position
  - At -3x ATR: sell 100% (fully out)

Widening Trailing TP (ATR multiples):
  - At +3x ATR profit: activate trail at 2x ATR from high
  - At +6x ATR profit: widen trail to 3x ATR
  - At +10x ATR profit: widen trail to 4x ATR (monster move)

Staged Partial: sell 33% at +5x ATR profit

Momentum death: volume below 0.8x average for 3 bars

Per-coin-category stats by ATR bucket:
  - Low vol:  ATR < 0.5%
  - Medium:   ATR 0.5% – 2%
  - High vol: ATR > 2%
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

# Scanner params — identical to baseline
VOL_MULT = 4.0
PRICE_THRESH = 0.035
LOOKBACK = 20

# Trade params
MAX_HOLD_BARS = 192   # 48h max hold (192 × 15min bars)
MAX_LEVERAGE = 3.0
MAX_DAILY_MOVE = 0.30

# BTC trend filter
BTC_TREND_PERIOD = 30

# ATR volatility buckets (as fraction of price, e.g. 0.005 = 0.5%)
ATR_LOW_THRESH = 0.005    # < 0.5%  → low vol
ATR_HIGH_THRESH = 0.02    # > 2.0%  → high vol


def simulate_adaptive_trade(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray | None,
    signal_bar: int,
    direction: int,
    atr_value: float,
    max_hold: int = MAX_HOLD_BARS,
) -> dict | None:
    """Simulate a trade where all stops/targets are in ATR multiples.

    atr_value: ATR as fraction of price (e.g. 0.0032 for BTC, 0.091 for ALPACA)

    Exit reasons:
      sl_tier1      : -2x ATR, sold 50%
      sl_tier2      : -3x ATR, sold remaining 50% (fully out)
      trail_stop    : trailing stop hit (activated at +3x ATR)
      partial_tp    : 33% partial taken at +5x ATR, rest closed by trail/time
      momentum_death: volume < 0.8x avg for 3 consecutive bars while in profit
      time_exit     : max_hold_bars safety net
    """
    n = len(close)
    if signal_bar + 1 >= n:
        return None

    entry_bar = signal_bar + 1
    entry_price = close[entry_bar]
    if entry_price == 0:
        return None

    # ATR in actual price units
    atr_price = atr_value * entry_price

    # ── Tiered stop levels ────────────────────────────────────────────────────
    sl_tier1_dist = 2.0 * atr_price   # -2x ATR → sell 50%
    sl_tier2_dist = 3.0 * atr_price   # -3x ATR → sell 100%

    if direction == 1:
        sl_tier1_price = entry_price - sl_tier1_dist
        sl_tier2_price = entry_price - sl_tier2_dist
    else:
        sl_tier1_price = entry_price + sl_tier1_dist
        sl_tier2_price = entry_price + sl_tier2_dist

    # ── Trailing TP levels ────────────────────────────────────────────────────
    trail_activate_dist  = 3.0 * atr_price   # profit to activate trail
    trail_medium_dist    = 6.0 * atr_price   # profit to widen to 3x ATR
    trail_wide_dist      = 10.0 * atr_price  # profit to widen to 4x ATR
    partial_tp_dist      = 5.0 * atr_price   # profit to take 33% partial

    # ── State ─────────────────────────────────────────────────────────────────
    hwm = entry_price          # high water mark in price
    trail_active = False
    trail_stop = None          # trailing stop price (None until activated)

    tier1_hit = False          # whether tier1 stop (50%) has been triggered
    partial_taken = False      # whether 33% partial TP has been taken

    remaining_frac = 1.0       # fraction of position still open
    locked_pnl = 0.0           # realised PnL from partial exits

    exit_bar = None
    exit_price = None
    exit_reason = None

    # Volume average for momentum death
    if volume is not None and signal_bar >= 20:
        vol_avg = np.mean(volume[signal_bar - 20:signal_bar])
    else:
        vol_avg = None

    for i in range(entry_bar + 1, min(entry_bar + max_hold + 1, n)):
        bar_high = high[i]
        bar_low = low[i]
        bar_close = close[i]

        # ── Update high water mark ────────────────────────────────────────────
        if direction == 1:
            if bar_high > hwm:
                hwm = bar_high
        else:
            if bar_low < hwm:
                hwm = bar_low

        # ── Current unrealised move in price units ────────────────────────────
        if direction == 1:
            move = bar_close - entry_price   # positive = profit
        else:
            move = entry_price - bar_close

        # ── Tiered stop loss check ────────────────────────────────────────────
        if not tier1_hit:
            if direction == 1:
                hit_tier1 = bar_low <= sl_tier1_price
            else:
                hit_tier1 = bar_high >= sl_tier1_price

            if hit_tier1:
                # Sell 50% at tier1 price
                exec_price = sl_tier1_price
                if direction == 1:
                    partial_ret = (exec_price - entry_price) / entry_price
                else:
                    partial_ret = (entry_price - exec_price) / entry_price
                locked_pnl += partial_ret * 0.50
                remaining_frac -= 0.50
                tier1_hit = True
                exit_reason = "sl_tier1_partial"  # intermediate, may change

        # After tier1, check tier2 (full exit)
        if tier1_hit:
            if direction == 1:
                hit_tier2 = bar_low <= sl_tier2_price
            else:
                hit_tier2 = bar_high >= sl_tier2_price

            if hit_tier2:
                exit_bar = i
                exit_price = sl_tier2_price
                exit_reason = "sl_tier2"
                break

        # ── Partial TP at +5x ATR ─────────────────────────────────────────────
        if not partial_taken and move >= partial_tp_dist:
            if direction == 1:
                partial_ret = (bar_close - entry_price) / entry_price
            else:
                partial_ret = (entry_price - bar_close) / entry_price
            locked_pnl += partial_ret * 0.33
            remaining_frac -= 0.33
            partial_taken = True

        # ── Trailing TP activation / widening ────────────────────────────────
        if move >= trail_activate_dist:
            trail_active = True

            # Determine current trail distance
            if move >= trail_wide_dist:
                trail_atr_mult = 4.0
            elif move >= trail_medium_dist:
                trail_atr_mult = 3.0
            else:
                trail_atr_mult = 2.0

            trail_dist = trail_atr_mult * atr_price

            if direction == 1:
                new_trail = hwm - trail_dist
                # Only ratchet up, never down; also never below entry
                if trail_stop is None:
                    trail_stop = max(new_trail, entry_price)
                else:
                    trail_stop = max(trail_stop, new_trail, entry_price)
            else:
                new_trail = hwm + trail_dist
                if trail_stop is None:
                    trail_stop = min(new_trail, entry_price)
                else:
                    trail_stop = min(trail_stop, new_trail, entry_price)

        # ── Check trail stop ─────────────────────────────────────────────────
        if trail_active and trail_stop is not None:
            if direction == 1:
                if bar_low <= trail_stop:
                    exit_bar = i
                    exit_price = trail_stop
                    exit_reason = "trail_stop"
                    break
            else:
                if bar_high >= trail_stop:
                    exit_bar = i
                    exit_price = trail_stop
                    exit_reason = "trail_stop"
                    break

        # ── Momentum death (only once trailing, i.e. profitable) ─────────────
        if vol_avg is not None and volume is not None and trail_active and i >= 3:
            vol_dying = (
                volume[i]   < vol_avg * 0.8
                and volume[i-1] < vol_avg * 0.8
                and volume[i-2] < vol_avg * 0.8
            )
            if vol_dying:
                exit_bar = i
                exit_price = bar_close
                exit_reason = "momentum_death"
                break

    # ── Safety net time exit ──────────────────────────────────────────────────
    if exit_bar is None:
        exit_bar = min(entry_bar + max_hold, n - 1)
        exit_price = close[exit_bar]
        exit_reason = "time_exit"

    # ── Final PnL ─────────────────────────────────────────────────────────────
    if direction == 1:
        remaining_ret = (exit_price - entry_price) / entry_price
    else:
        remaining_ret = (entry_price - exit_price) / entry_price
    pnl_pct = locked_pnl + remaining_ret * remaining_frac

    return {
        "entry_bar": entry_bar,
        "exit_bar": exit_bar,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "pnl_pct": pnl_pct,
        "hold_bars": exit_bar - entry_bar,
        "exit_reason": exit_reason,
        "atr_value": atr_value,
        "partial_taken": partial_taken,
        "tier1_hit": tier1_hit,
    }


def atr_bucket(atr: float) -> str:
    """Classify a coin's ATR into a volatility bucket."""
    if atr < ATR_LOW_THRESH:
        return "low_vol (<0.5%)"
    elif atr < ATR_HIGH_THRESH:
        return "medium_vol (0.5-2%)"
    else:
        return "high_vol (>2%)"


def main():
    start = time.time()
    logger.info("Adaptive ATR-Based Stops Backtest")
    logger.info(f"Data: {RAW_DIR}")
    logger.info(f"Scanner: VOL_MULT={VOL_MULT} PRICE_THRESH={PRICE_THRESH}")
    logger.info("Stops: Tier1=-2xATR(50%), Tier2=-3xATR(100%)")
    logger.info("Trail: +3xATR activate @2xATR, +6xATR @3xATR, +10xATR @4xATR")
    logger.info("Partial: 33% at +5xATR")

    # ── BTC for trend filter ───────────────────────────────────────────────────
    btc_dir = RAW_DIR / "BTCUSDT"
    btc_df = load_coin(btc_dir)
    if btc_df is None:
        logger.error("Cannot load BTC data!")
        return
    btc_close = btc_df["close"].values.astype(float)
    btc_ema = pd.Series(btc_close).ewm(span=BTC_TREND_PERIOD).mean().values
    btc_times = btc_df["open_time"].values
    logger.info(f"BTC loaded: {len(btc_df)} bars")

    # ── Scan all coins ─────────────────────────────────────────────────────────
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

        # Liquidity filter
        avg_vol = np.nanmean(volume[-100:]) if len(volume) >= 100 else np.nanmean(volume)
        avg_price = np.nanmean(close[-100:]) if len(close) >= 100 else np.nanmean(close)
        if avg_vol * avg_price < MIN_VOLUME_USD / 96:
            continue

        signals = detect_breakouts(df)
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

            trade = simulate_adaptive_trade(
                close=close, high=high, low=low, volume=volume,
                signal_bar=bar, direction=sig["direction"],
                atr_value=atr[bar], max_hold=MAX_HOLD_BARS,
            )

            if trade is not None:
                trade["symbol"] = symbol
                trade["signal_time"] = str(times[bar])
                trade["signal_volume_ratio"] = sig["volume_ratio"]
                trade["signal_price_change"] = sig["price_change"]
                all_trades.append(trade)

    elapsed = time.time() - start
    logger.info(f"Scan complete in {elapsed:.0f}s")
    logger.info(f"Coins with signals: {coins_with_signals}")
    logger.info(f"Total raw signals:  {total_signals}")
    logger.info(f"Total trades:       {len(all_trades)}")

    if not all_trades:
        logger.warning("No trades generated!")
        return

    trades_df = pd.DataFrame(all_trades)

    # ── Overall metrics ────────────────────────────────────────────────────────
    pnls = trades_df["pnl_pct"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls)
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0.0
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = np.mean(pnls)

    # Rough trades/month based on BTC bar count
    total_bars = len(btc_df)
    total_months = total_bars / (96 * 30)   # 96 15min bars/day × 30 days
    trades_per_month = len(all_trades) / total_months if total_months > 0 else 0
    monthly_return = expectancy * trades_per_month * MAX_LEVERAGE

    exit_reasons = trades_df["exit_reason"].value_counts()

    sep = "=" * 64
    logger.info(f"\n{sep}")
    logger.info("ADAPTIVE ATR BACKTEST — RESULTS")
    logger.info(sep)
    logger.info(f"Total trades:         {len(all_trades)}")
    logger.info(f"Trades / month:       {trades_per_month:.1f}")
    logger.info(f"Win rate:             {win_rate:.1%}")
    logger.info(f"Avg winner:           {avg_win:.2%}")
    logger.info(f"Avg loser:            {avg_loss:.2%}")
    if avg_loss > 0:
        logger.info(f"R:R ratio:            {avg_win / avg_loss:.2f}:1")
    else:
        logger.info("R:R ratio:            inf")
    logger.info(f"Profit factor:        {profit_factor:.2f}")
    logger.info(f"Expectancy / trade:   {expectancy:.2%}")
    logger.info(f"Est monthly (3x lev): {monthly_return:.1%}")

    logger.info("\nExit reasons:")
    for reason, count in exit_reasons.items():
        logger.info(f"  {reason:<20s}: {count:4d}  ({count/len(all_trades):.1%})")

    # ── Top 10 winners ─────────────────────────────────────────────────────────
    logger.info("\nTop 10 winners:")
    top = trades_df.nlargest(10, "pnl_pct")
    for _, t in top.iterrows():
        logger.info(
            f"  {t['symbol']:<14s}  {t['pnl_pct']:+.1%}  "
            f"atr={t['atr_value']:.3%}  exit={t['exit_reason']}"
        )

    # ── Bottom 5 losers ────────────────────────────────────────────────────────
    logger.info("\nBottom 5 losers:")
    bottom = trades_df.nsmallest(5, "pnl_pct")
    for _, t in bottom.iterrows():
        logger.info(
            f"  {t['symbol']:<14s}  {t['pnl_pct']:+.1%}  "
            f"atr={t['atr_value']:.3%}  exit={t['exit_reason']}"
        )

    # ── Per ATR-bucket stats ───────────────────────────────────────────────────
    trades_df["atr_bucket"] = trades_df["atr_value"].apply(atr_bucket)

    logger.info(f"\n{'─'*64}")
    logger.info("PER VOLATILITY BUCKET (proves adaptive scaling)")
    logger.info(f"{'─'*64}")
    logger.info(f"{'Bucket':<26s}  {'Trades':>6}  {'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  {'PF':>6}")
    logger.info(f"{'─'*64}")

    bucket_order = ["low_vol (<0.5%)", "medium_vol (0.5-2%)", "high_vol (>2%)"]
    for bucket in bucket_order:
        sub = trades_df[trades_df["atr_bucket"] == bucket]
        if len(sub) == 0:
            logger.info(f"  {bucket:<26s}  {'—':>6}")
            continue
        sub_pnls = sub["pnl_pct"].values
        sub_wins = sub_pnls[sub_pnls > 0]
        sub_losses = sub_pnls[sub_pnls <= 0]
        sub_wr = len(sub_wins) / len(sub_pnls)
        sub_aw = np.mean(sub_wins) if len(sub_wins) > 0 else 0.0
        sub_al = abs(np.mean(sub_losses)) if len(sub_losses) > 0 else 0.0
        sub_gp = sub_wins.sum()
        sub_gl = abs(sub_losses.sum())
        sub_pf = sub_gp / sub_gl if sub_gl > 0 else float("inf")
        logger.info(
            f"  {bucket:<26s}  {len(sub):>6d}  {sub_wr:>6.1%}  "
            f"{sub_aw:>7.2%}  {sub_al:>7.2%}  {sub_pf:>6.2f}"
        )

    logger.info(f"{'─'*64}")

    # Also show ATR distribution of signals
    logger.info("\nATR distribution of all signals:")
    atr_vals = trades_df["atr_value"].values
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        logger.info(f"  p{p:<3d}: {np.percentile(atr_vals, p):.3%}")

    # ── Monthly breakdown ──────────────────────────────────────────────────────
    trades_df["month"] = pd.to_datetime(trades_df["signal_time"]).dt.to_period("M")
    monthly = trades_df.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])
    monthly["leveraged"] = monthly["sum"] * MAX_LEVERAGE

    logger.info(f"\nMonthly returns (3x leverage):")
    for month, row in monthly.iterrows():
        logger.info(f"  {month}:  {row['leveraged']:+.1%}  ({int(row['count'])} trades)")

    profitable_months = (monthly["leveraged"] > 0).sum()
    total_months_count = len(monthly)
    logger.info(
        f"\nProfitable months: {profitable_months}/{total_months_count} "
        f"({profitable_months / total_months_count:.0%})"
    )

    logger.info(f"\nTotal elapsed: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
