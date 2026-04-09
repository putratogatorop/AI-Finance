# scripts/backtest_experiment_sizing.py
"""ATR-Scaled Position Sizing Backtest Experiment.

The Idea: Instead of skipping high-volatility coins (ATR > 2%), trade ALL coins
but scale position size inversely to volatility. Small-cap volatile coins get
tiny positions (lottery tickets), stable large-caps get bigger positions.

Position Sizing Formula:
    risk_dollars = equity * 0.02            # 2% of equity
    stop_dist    = 3 * atr_fraction         # 3x ATR as fraction
    notional     = risk / stop_dist          # auto-scaled
    notional     = min(notional, equity * 3) # 3x leverage cap

Key insight: every trade risks the same 2% of equity regardless of coin volatility.
  BTC (ATR 0.3%):   notional = risk / 0.009 = 222x risk → big position
  ALPACA (ATR 9%):  notional = risk / 0.27  = 7.4x risk → tiny position

Trade Simulation: Adaptive ATR tiered stops + widening trail from backtest_experiment_adaptive.py

Reporting:
  1. Overall equity-weighted metrics
  2. Per-volatility-bucket contribution
  3. Cumulative equity curve ($600 start)
  4. Max drawdown on equity
  5. Monthly equity returns
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
from scripts.backtest_experiment_adaptive import simulate_adaptive_trade, atr_bucket
from scripts.backtest_scanner_ml_filter import (
    extract_features,
    build_walk_forward_splits,
    FEATURE_COLS,
    ML_PARAMS,
    MIN_TRAIN_SIGNALS,
)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

# Scanner settings (matching ml_filter baseline)
MIN_VOLUME_USD = 1_000_000
MIN_BARS = 200
VOL_MULT = 4.0
PRICE_THRESH = 0.035
LOOKBACK = 20

# BTC trend filter
BTC_TREND_PERIOD = 30

# ML filter thresholds to sweep
ML_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
ML_DEFAULT_THRESH = 0.60  # Default for equity simulation

# Position sizing
STARTING_EQUITY = 600.0
RISK_PCT = 0.02            # 2% of equity per trade
STOP_ATR_MULT = 3.0        # stop distance = 3x ATR
MAX_LEVERAGE = 3.0         # cap notional at 3x equity

# Trade params (same as adaptive experiment)
MAX_HOLD_BARS = 192

# ATR volatility buckets
ATR_LOW_THRESH = 0.005     # < 0.5%  → low vol
ATR_HIGH_THRESH = 0.02     # > 2.0%  → high vol


def compute_position_sizing(equity: float, atr_value: float) -> dict:
    """Compute ATR-scaled position size.

    Args:
        equity: Current portfolio equity in dollars
        atr_value: ATR as a fraction of price (e.g. 0.003 for BTC, 0.09 for ALPACA)

    Returns:
        dict with risk_dollars, stop_dist_frac, notional, leverage
    """
    risk_dollars = equity * RISK_PCT
    stop_dist_frac = STOP_ATR_MULT * atr_value  # stop distance as fraction

    if stop_dist_frac <= 0:
        return {
            "risk_dollars": risk_dollars,
            "stop_dist_frac": 0.0,
            "notional": 0.0,
            "leverage": 0.0,
        }

    notional = risk_dollars / stop_dist_frac
    max_notional = equity * MAX_LEVERAGE
    notional = min(notional, max_notional)
    leverage = notional / equity

    return {
        "risk_dollars": risk_dollars,
        "stop_dist_frac": stop_dist_frac,
        "notional": notional,
        "leverage": leverage,
    }


def simulate_equity_curve(
    oos_trades: pd.DataFrame,
    starting_equity: float = STARTING_EQUITY,
) -> tuple[list[float], pd.DataFrame]:
    """Simulate portfolio equity curve from OOS trades.

    Trades are processed chronologically. For each trade:
        position_notional = min(risk/stop_dist, equity * 3)
        dollar_pnl = pnl_pct * notional
        equity += dollar_pnl

    Returns:
        equity_curve: list of equity values after each trade
        trades_with_sizing: DataFrame with position sizing columns added
    """
    trades = oos_trades.sort_values("signal_time").reset_index(drop=True)

    equity = starting_equity
    equity_curve = [equity]
    records = []

    for _, trade in trades.iterrows():
        atr = float(trade["atr_value"])
        pnl_pct = float(trade["pnl_pct"])

        sizing = compute_position_sizing(equity, atr)
        notional = sizing["notional"]

        if notional <= 0:
            records.append({**trade.to_dict(), "notional": 0, "dollar_pnl": 0,
                             "equity_before": equity, "equity_after": equity,
                             "equity_pnl_pct": 0})
            continue

        dollar_pnl = pnl_pct * notional
        equity_after = equity + dollar_pnl
        equity_pnl_pct = dollar_pnl / equity  # impact relative to equity before trade

        records.append({
            **trade.to_dict(),
            "notional": notional,
            "leverage": sizing["leverage"],
            "stop_dist_frac": sizing["stop_dist_frac"],
            "dollar_pnl": dollar_pnl,
            "equity_before": equity,
            "equity_after": equity_after,
            "equity_pnl_pct": equity_pnl_pct,
        })

        equity = equity_after
        equity_curve.append(equity)

    result_df = pd.DataFrame(records)
    return equity_curve, result_df


def compute_max_drawdown(equity_curve: list[float]) -> tuple[float, float]:
    """Compute maximum drawdown from equity curve.

    Returns:
        max_dd_pct: max drawdown as a fraction (e.g. 0.25 = 25%)
        max_dd_dollar: max drawdown in dollars
    """
    if len(equity_curve) < 2:
        return 0.0, 0.0

    arr = np.array(equity_curve, dtype=float)
    peak = arr[0]
    max_dd_pct = 0.0
    max_dd_dollar = 0.0

    for val in arr:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        dd_dollar = peak - val
        if dd > max_dd_pct:
            max_dd_pct = dd
            max_dd_dollar = dd_dollar

    return max_dd_pct, max_dd_dollar


def print_bucket_stats(trades_df: pd.DataFrame, label: str = ""):
    """Print per-volatility-bucket equity-impact stats."""
    bucket_order = ["low_vol (<0.5%)", "medium_vol (0.5-2%)", "high_vol (>2%)"]
    sep = "-" * 90
    logger.info(f"\n{sep}")
    logger.info(f"PER VOLATILITY BUCKET — {label}")
    logger.info(f"{sep}")
    logger.info(
        f"{'Bucket':<26s}  {'Trades':>6}  {'WR':>6}  {'AvgPos%':>8}  "
        f"{'AvgEqImpact':>12}  {'TotalDollar':>12}  {'PF':>6}"
    )
    logger.info(sep)

    for bucket in bucket_order:
        sub = trades_df[trades_df["atr_bucket"] == bucket]
        if len(sub) == 0:
            logger.info(f"  {bucket:<26s}  {'—':>6}")
            continue

        pnls = sub["pnl_pct"].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / len(pnls)

        eq_impacts = sub["equity_pnl_pct"].values
        avg_eq = float(np.mean(eq_impacts))

        dollar_pnls = sub["dollar_pnl"].values
        total_dollar = float(np.sum(dollar_pnls))

        gross_profit = float(np.sum(dollar_pnls[dollar_pnls > 0]))
        gross_loss = float(abs(np.sum(dollar_pnls[dollar_pnls <= 0])))
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        avg_pos_pct = float(np.mean(sub["leverage"].values)) if "leverage" in sub.columns else 0.0

        logger.info(
            f"  {bucket:<26s}  {len(sub):>6d}  {wr:>6.1%}  {avg_pos_pct:>8.2f}x  "
            f"{avg_eq:>+12.3%}  {total_dollar:>+12.2f}  {pf:>6.2f}"
        )

    logger.info(sep)


def main():
    t_start = time.time()
    logger.info("ATR-Scaled Position Sizing Backtest")
    logger.info(f"Data: {RAW_DIR}")
    logger.info(
        f"Scanner: VOL_MULT={VOL_MULT} PRICE_THRESH={PRICE_THRESH} | "
        f"NO vol filter (all coins)"
    )
    logger.info(
        f"Sizing: risk={RISK_PCT:.0%} equity | stop={STOP_ATR_MULT}x ATR | "
        f"max_lev={MAX_LEVERAGE}x"
    )
    logger.info(f"Starting equity: ${STARTING_EQUITY:.0f}")

    # ── Load BTC ──────────────────────────────────────────────────────────────
    btc_dir = RAW_DIR / "BTCUSDT"
    btc_df = load_coin(btc_dir)
    if btc_df is None:
        logger.error("Cannot load BTC data!")
        return
    btc_close = btc_df["close"].values.astype(float)
    btc_volume = btc_df["volume"].values.astype(float)
    btc_ema = pd.Series(btc_close).ewm(span=BTC_TREND_PERIOD).mean().values
    btc_times = btc_df["open_time"].values
    logger.info(f"BTC loaded: {len(btc_df)} bars")

    # ── Scan ALL coins (no vol filter) ────────────────────────────────────────
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    logger.info(f"Scanning {len(symbol_dirs)} coins (no ATR filter)...")

    all_signals = []
    coins_processed = 0
    coins_with_signals = 0

    for sym_dir in symbol_dirs:
        symbol = sym_dir.name
        df = load_coin(sym_dir)
        if df is None or len(df) < MIN_BARS:
            continue

        coins_processed += 1
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        times = df["open_time"].values

        # Liquidity filter (keep; we still need some minimum volume)
        avg_vol = np.nanmean(volume[-100:]) if len(volume) >= 100 else np.nanmean(volume)
        avg_price = np.nanmean(close[-100:]) if len(close) >= 100 else np.nanmean(close)
        if avg_vol * avg_price < MIN_VOLUME_USD / 96:
            continue

        # NOTE: No ATR/vol filter here — all coins pass through
        signals = detect_breakouts(df, vol_mult=VOL_MULT, price_thresh=PRICE_THRESH, lookback=LOOKBACK)
        if not signals:
            continue

        coins_with_signals += 1

        atr = compute_atr(high, low, close)
        atr_14 = atr
        atr_96 = compute_atr(high, low, close, period=96)

        for sig in signals:
            bar = sig["bar"]
            if np.isnan(atr[bar]) or atr[bar] == 0:
                continue

            # BTC trend filter
            sig_time = times[bar]
            btc_idx = int(np.searchsorted(btc_times, sig_time))
            btc_idx = min(btc_idx, len(btc_close) - 1)
            if sig["direction"] == 1 and btc_close[btc_idx] < btc_ema[btc_idx]:
                continue
            if sig["direction"] == -1 and btc_close[btc_idx] > btc_ema[btc_idx]:
                continue

            # Simulate adaptive trade (same as adaptive experiment)
            trade = simulate_adaptive_trade(
                close=close,
                high=high,
                low=low,
                volume=volume,
                signal_bar=bar,
                direction=sig["direction"],
                atr_value=atr[bar],
                max_hold=MAX_HOLD_BARS,
            )
            if trade is None:
                continue

            # Extract ML features
            feats = extract_features(
                df=df,
                bar=bar,
                btc_close=btc_close,
                btc_volume=btc_volume,
                btc_times=btc_times,
                sig_vol_ratio=sig["volume_ratio"],
                sig_price_change=sig["price_change"],
                atr_14=atr_14,
                atr_96=atr_96,
            )
            if feats is None:
                continue

            label = 1 if trade["pnl_pct"] > 0 else 0

            record = {
                **feats,
                **trade,
                "label": label,
                "symbol": symbol,
                "signal_time": str(times[bar]),
                "direction": sig["direction"],
                "atr_value": atr[bar],
            }
            all_signals.append(record)

    logger.info(f"Coins processed: {coins_processed} | with signals: {coins_with_signals}")
    logger.info(f"Total signals collected: {len(all_signals)}")

    if not all_signals:
        logger.error("No signals collected!")
        return

    # ── Build DataFrame ───────────────────────────────────────────────────────
    signals_df = pd.DataFrame(all_signals)
    signals_df["signal_time"] = pd.to_datetime(signals_df["signal_time"])
    signals_df = signals_df.sort_values("signal_time").reset_index(drop=True)

    # Fill num_concurrent
    signals_df["time_bucket"] = signals_df["signal_time"].dt.floor("15min")
    concurrent_counts = signals_df.groupby("time_bucket").size()
    signals_df["num_concurrent"] = signals_df["time_bucket"].map(concurrent_counts)
    signals_df = signals_df.drop(columns=["time_bucket"])

    # Sanitise features
    signals_df[FEATURE_COLS] = (
        signals_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    )

    label_dist = signals_df["label"].value_counts()
    logger.info(
        f"Label distribution: winners={label_dist.get(1, 0)} "
        f"losers={label_dist.get(0, 0)}"
    )

    # Add ATR bucket
    signals_df["atr_bucket"] = signals_df["atr_value"].apply(atr_bucket)

    # ── Walk-forward ML filter ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("WALK-FORWARD ML FILTER")
    logger.info("=" * 70)

    splits = build_walk_forward_splits(signals_df)
    logger.info(f"Walk-forward windows: {len(splits)}")

    if not splits:
        logger.error("Not enough data for walk-forward splits!")
        return

    all_oos_list = []
    window_results = []

    for i, (train_data, test_data) in enumerate(splits):
        train_start = pd.to_datetime(train_data["signal_time"]).min().date()
        train_end = pd.to_datetime(train_data["signal_time"]).max().date()
        test_start = pd.to_datetime(test_data["signal_time"]).min().date()
        test_end = pd.to_datetime(test_data["signal_time"]).max().date()

        logger.info(
            f"\nWindow {i+1}: train [{train_start} -> {train_end}] "
            f"({len(train_data)} signals), "
            f"test [{test_start} -> {test_end}] ({len(test_data)} signals)"
        )

        X_train = train_data[FEATURE_COLS].values
        y_train = train_data["label"].values
        X_test = test_data[FEATURE_COLS].values

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        logger.info(f"  Train labels: {n_pos} winners / {n_neg} losers")

        if n_pos < 10 or n_neg < 10:
            logger.warning("  Insufficient class balance, skipping window")
            test_data = test_data.copy()
            test_data["ml_proba"] = 0.5
        else:
            model = LGBMClassifier(**ML_PARAMS)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            test_data = test_data.copy()
            test_data["ml_proba"] = proba

            # Feature importance
            fi = dict(zip(FEATURE_COLS, model.feature_importances_))
            top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(
                "  Top features: "
                + ", ".join(f"{k}={v:.0f}" for k, v in top_features)
            )

        # Per-threshold pass rates
        for thresh in ML_THRESHOLDS:
            filt = test_data[test_data["ml_proba"] > thresh]
            if len(filt) > 0:
                pnls = filt["pnl_pct"].values
                wr = (pnls > 0).mean()
                logger.info(f"    @{thresh:.2f}: {len(filt)} trades, WR={wr:.1%}")

        filtered = test_data[test_data["ml_proba"] > ML_DEFAULT_THRESH]
        logger.info(
            f"  OOS signals: {len(test_data)} total, "
            f"{len(filtered)} pass ML>{ML_DEFAULT_THRESH:.2f} "
            f"({len(filtered)/len(test_data):.1%})"
        )

        window_results.append({
            "window": i + 1,
            "train_n": len(train_data),
            "test_n": len(test_data),
            "filtered_n": len(filtered),
        })

        all_oos_list.append(test_data)

    all_oos = pd.concat(all_oos_list, ignore_index=True)

    # ── Threshold sweep on OOS ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("THRESHOLD SWEEP — OOS AGGREGATE (pct-based)")
    logger.info("=" * 70)
    logger.info(
        f"{'Thresh':>7}  {'Trades':>7}  {'WR':>6}  {'AvgW':>7}  {'AvgL':>7}  "
        f"{'PF':>6}  {'Exp':>7}"
    )

    for thresh in ML_THRESHOLDS:
        filt = all_oos[all_oos["ml_proba"] > thresh]
        if len(filt) == 0:
            continue
        pnls = filt["pnl_pct"].values
        wins = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        wr = len(wins) / len(pnls)
        aw = float(np.mean(wins)) if len(wins) > 0 else 0.0
        al = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
        gp = wins.sum()
        gl = abs(losses.sum())
        pf = gp / gl if gl > 0 else float("inf")
        exp = float(np.mean(pnls))
        logger.info(
            f"  >{thresh:.2f}  {len(filt):>7d}  {wr:>6.1%}  {aw:>7.2%}  "
            f"{al:>7.2%}  {pf:>6.2f}  {exp:>+7.3%}"
        )

    # ── Equity simulation at default threshold ────────────────────────────────
    filtered_oos = all_oos[all_oos["ml_proba"] > ML_DEFAULT_THRESH].copy()
    logger.info(f"\nEquity simulation: {len(filtered_oos)} trades @ ML>{ML_DEFAULT_THRESH:.2f}")

    if len(filtered_oos) == 0:
        logger.warning("No filtered trades — cannot simulate equity curve")
        return

    equity_curve, trades_sized = simulate_equity_curve(filtered_oos, STARTING_EQUITY)

    final_equity = equity_curve[-1]
    total_return = (final_equity - STARTING_EQUITY) / STARTING_EQUITY
    max_dd_pct, max_dd_dollar = compute_max_drawdown(equity_curve)

    sep = "=" * 70
    logger.info(f"\n{sep}")
    logger.info("EQUITY SIMULATION RESULTS")
    logger.info(sep)
    logger.info(f"Starting equity:   ${STARTING_EQUITY:.2f}")
    logger.info(f"Final equity:      ${final_equity:.2f}")
    logger.info(f"Total return:      {total_return:+.1%}")
    logger.info(f"Max drawdown:      {max_dd_pct:.1%}  (${max_dd_dollar:.2f})")

    # Overall equity-weighted metrics
    eq_impacts = trades_sized["equity_pnl_pct"].values
    dollar_pnls = trades_sized["dollar_pnl"].values

    wins_eq = eq_impacts[eq_impacts > 0]
    losses_eq = eq_impacts[eq_impacts <= 0]
    wr_eq = len(wins_eq) / len(eq_impacts) if len(eq_impacts) > 0 else 0.0
    avg_win_eq = float(np.mean(wins_eq)) if len(wins_eq) > 0 else 0.0
    avg_loss_eq = float(abs(np.mean(losses_eq))) if len(losses_eq) > 0 else 0.0
    gross_profit_eq = float(np.sum(dollar_pnls[dollar_pnls > 0]))
    gross_loss_eq = float(abs(np.sum(dollar_pnls[dollar_pnls <= 0])))
    pf_eq = gross_profit_eq / gross_loss_eq if gross_loss_eq > 0 else float("inf")
    exp_eq = float(np.mean(eq_impacts))

    avg_notional = float(np.mean(trades_sized["notional"].values))
    avg_leverage = float(np.mean(trades_sized["leverage"].values))

    logger.info(f"\nEquity-Weighted Metrics (all {len(trades_sized)} sized trades):")
    logger.info(f"  Win rate (equity):   {wr_eq:.1%}")
    logger.info(f"  Avg win (equity):    {avg_win_eq:+.3%}")
    logger.info(f"  Avg loss (equity):   {avg_loss_eq:+.3%}")
    logger.info(f"  Profit factor:       {pf_eq:.2f}")
    logger.info(f"  Expectancy (equity): {exp_eq:+.4%}")
    logger.info(f"  Avg position size:   ${avg_notional:.2f}")
    logger.info(f"  Avg leverage:        {avg_leverage:.2f}x")

    # Exit reason breakdown
    logger.info("\nExit reasons:")
    for reason, count in trades_sized["exit_reason"].value_counts().items():
        logger.info(f"  {reason:<22s}: {count:4d}  ({count/len(trades_sized):.1%})")

    # ── Per-volatility-bucket stats ───────────────────────────────────────────
    print_bucket_stats(trades_sized, label="Equity-Adjusted")

    # ── Monthly equity returns ────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("MONTHLY EQUITY RETURNS")
    logger.info("=" * 70)

    trades_sized["month"] = pd.to_datetime(trades_sized["signal_time"]).dt.to_period("M")
    monthly = trades_sized.groupby("month").agg(
        dollar_pnl=("dollar_pnl", "sum"),
        n_trades=("dollar_pnl", "count"),
        avg_eq_impact=("equity_pnl_pct", "mean"),
    )

    # Build equity at start of each month for return calculation
    # Use cumulative equity: equity at start of month = starting + sum(dollar_pnl up to month)
    monthly["cumulative_equity"] = STARTING_EQUITY + monthly["dollar_pnl"].cumsum() - monthly["dollar_pnl"]
    monthly["monthly_return"] = monthly["dollar_pnl"] / monthly["cumulative_equity"]

    logger.info(
        f"{'Month':<10}  {'Trades':>7}  {'DollarPnL':>10}  {'MonthReturn':>12}  {'CumEquity':>10}"
    )
    for month, row in monthly.iterrows():
        logger.info(
            f"  {str(month):<10}  {int(row['n_trades']):>7d}  "
            f"{row['dollar_pnl']:>+10.2f}  {row['monthly_return']:>+12.1%}  "
            f"{row['cumulative_equity'] + row['dollar_pnl']:>10.2f}"
        )

    profitable_months = (monthly["monthly_return"] > 0).sum()
    total_months_count = len(monthly)
    logger.info(
        f"\nProfitable months: {profitable_months}/{total_months_count} "
        f"({profitable_months/total_months_count:.0%})"
    )

    # ── Equity curve highlights ───────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("EQUITY CURVE HIGHLIGHTS")
    logger.info("=" * 70)

    eq_arr = np.array(equity_curve)
    peak_equity = float(eq_arr.max())
    trough_equity = float(eq_arr.min())
    peak_idx = int(eq_arr.argmax())
    trough_idx = int(eq_arr.argmin())

    logger.info(f"  Starting equity:   ${STARTING_EQUITY:.2f}")
    logger.info(f"  Peak equity:       ${peak_equity:.2f}  (after trade #{peak_idx})")
    logger.info(f"  Trough equity:     ${trough_equity:.2f}  (after trade #{trough_idx})")
    logger.info(f"  Final equity:      ${final_equity:.2f}")
    logger.info(f"  Max drawdown:      {max_dd_pct:.1%}  (${max_dd_dollar:.2f})")
    logger.info(f"  Total return:      {total_return:+.1%}")

    # Best and worst individual trades (by dollar impact)
    best_trade = trades_sized.loc[trades_sized["dollar_pnl"].idxmax()]
    worst_trade = trades_sized.loc[trades_sized["dollar_pnl"].idxmin()]

    logger.info(f"\n  Best trade:  {best_trade['symbol']}  "
                f"pnl={best_trade['pnl_pct']:+.2%}  "
                f"notional=${best_trade['notional']:.2f}  "
                f"dollar_pnl=${best_trade['dollar_pnl']:+.2f}  "
                f"atr={best_trade['atr_value']:.3%}")
    logger.info(f"  Worst trade: {worst_trade['symbol']}  "
                f"pnl={worst_trade['pnl_pct']:+.2%}  "
                f"notional=${worst_trade['notional']:.2f}  "
                f"dollar_pnl=${worst_trade['dollar_pnl']:+.2f}  "
                f"atr={worst_trade['atr_value']:.3%}")

    # ── ATR distribution of sized trades ─────────────────────────────────────
    logger.info("\nATR distribution of sized trades:")
    atr_vals = trades_sized["atr_value"].values
    for p in [10, 25, 50, 75, 90]:
        logger.info(f"  p{p:<3d}: {np.percentile(atr_vals, p):.3%}")

    # ── Top 10 winners and losers by dollar PnL ───────────────────────────────
    logger.info("\nTop 10 trades by dollar PnL:")
    top_dollar = trades_sized.nlargest(10, "dollar_pnl")
    for _, t in top_dollar.iterrows():
        logger.info(
            f"  {str(t['signal_time'])[:19]}  {t['symbol']:<14s}  "
            f"pnl={t['pnl_pct']:+7.2%}  notional=${t['notional']:6.2f}  "
            f"$pnl={t['dollar_pnl']:+7.2f}  atr={t['atr_value']:.3%}  "
            f"lev={t['leverage']:.2f}x"
        )

    logger.info("\nBottom 5 trades by dollar PnL:")
    bot_dollar = trades_sized.nsmallest(5, "dollar_pnl")
    for _, t in bot_dollar.iterrows():
        logger.info(
            f"  {str(t['signal_time'])[:19]}  {t['symbol']:<14s}  "
            f"pnl={t['pnl_pct']:+7.2%}  notional=${t['notional']:6.2f}  "
            f"$pnl={t['dollar_pnl']:+7.2f}  atr={t['atr_value']:.3%}  "
            f"lev={t['leverage']:.2f}x"
        )

    logger.info(f"\nTotal elapsed: {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()
