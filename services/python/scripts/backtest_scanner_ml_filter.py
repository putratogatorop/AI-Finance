# scripts/backtest_scanner_ml_filter.py
"""ML-Filtered Momentum Scanner Backtester.

Runs the momentum scanner on all coins, extracts features at each signal,
then uses walk-forward LightGBM to filter out fake breakouts.

Imports scanner logic from backtest_momentum_scanner.py.
"""

import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# Import scanner functions
from scripts.backtest_momentum_scanner import (
    compute_atr,
    detect_breakouts,
    load_coin,
    simulate_pullback_trade,
)

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
MIN_VOLUME_USD = 1_000_000
MIN_BARS = 200

# Scanner settings — LOOSE to get ~2749 signals
VOL_MULT = 4.0
PRICE_THRESH = 0.035
LOOKBACK = 20

# Trade parameters (same as base scanner)
PULLBACK_PCT = 0.02
ATR_STOP_MULT = 2.0
ATR_TP_MULT = 5.0
MAX_HOLD_BARS = 48
MAX_LEVERAGE = 3.0
MAX_DAILY_MOVE = 0.10
BTC_TREND_PERIOD = 30

# ML filter threshold
ML_PROB_THRESHOLD = 0.55

# Walk-forward config (time-based)
TRAIN_MONTHS = 18
TEST_MONTHS = 6
MIN_TRAIN_SIGNALS = 200

# LightGBM params — simple to avoid overfitting on small dataset
ML_PARAMS = {
    "objective": "binary",
    "num_leaves": 15,
    "min_child_samples": 30,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "feature_fraction": 0.8,
    "verbose": -1,
    "random_state": 42,
}

FEATURE_COLS = [
    "vol_ratio",
    "price_change",
    "price_change_2bar",
    "bar_range_norm",
    "upper_wick_pct",
    "vol_trend",
    "price_trend",
    "volatility",
    "rel_volume_rank",
    "btc_ret_4bar",
    "btc_ret_24bar",
    "btc_vol_ratio",
    "hour_of_day",
    "day_of_week",
]


def extract_features(
    df: pd.DataFrame,
    bar: int,
    btc_close: np.ndarray,
    btc_volume: np.ndarray,
    btc_times: np.ndarray,
    sig_vol_ratio: float,
    sig_price_change: float,
) -> dict | None:
    """Extract ML features at signal bar."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["open_time"].values

    n = len(close)
    if bar < 100 or bar >= n:
        return None

    # --- Breakout bar features ---
    vol_ratio = sig_vol_ratio
    price_change = sig_price_change

    # 2-bar return
    if bar >= 2 and close[bar - 2] > 0:
        price_change_2bar = (close[bar] - close[bar - 2]) / close[bar - 2]
    else:
        price_change_2bar = 0.0

    # Candle size
    bar_range = high[bar] - low[bar]
    bar_range_norm = bar_range / close[bar] if close[bar] > 0 else 0.0

    # Upper wick fraction (rejection signal for longs, entry signal for shorts)
    body_top = max(open_[bar], close[bar])
    upper_wick = high[bar] - body_top
    upper_wick_pct = (upper_wick / bar_range) if bar_range > 0 else 0.0

    # --- Recent history features (last 20 bars before signal) ---
    start20 = max(0, bar - 20)
    vol_window = volume[start20:bar]
    close_window = close[start20:bar]

    # Volume trend: linear slope over last 20 bars (positive = building)
    if len(vol_window) >= 2:
        x = np.arange(len(vol_window), dtype=float)
        vol_mean = np.mean(vol_window)
        if vol_mean > 0:
            vol_norm = vol_window / vol_mean
            vol_trend = float(np.polyfit(x, vol_norm, 1)[0])
        else:
            vol_trend = 0.0
    else:
        vol_trend = 0.0

    # Price trend: return over last 20 bars
    if len(close_window) >= 2 and close_window[0] > 0:
        price_trend = (close_window[-1] - close_window[0]) / close_window[0]
    else:
        price_trend = 0.0

    # Volatility: std of returns over last 20 bars
    if len(close_window) >= 2:
        rets = np.diff(close_window) / close_window[:-1]
        rets = rets[np.isfinite(rets)]
        volatility = float(np.std(rets)) if len(rets) > 0 else 0.0
    else:
        volatility = 0.0

    # Relative volume rank: where does this bar's volume rank in last 100 bars
    start100 = max(0, bar - 100)
    vol_window100 = volume[start100:bar + 1]
    if len(vol_window100) > 1:
        rank = float(np.sum(vol_window100[:-1] < volume[bar])) / (len(vol_window100) - 1)
        rel_volume_rank = rank
    else:
        rel_volume_rank = 0.5

    # --- BTC context features ---
    sig_time = times[bar]
    btc_idx = int(np.searchsorted(btc_times, sig_time))
    btc_idx = min(btc_idx, len(btc_close) - 1)

    # BTC 4-bar return
    if btc_idx >= 4 and btc_close[btc_idx - 4] > 0:
        btc_ret_4bar = (btc_close[btc_idx] - btc_close[btc_idx - 4]) / btc_close[btc_idx - 4]
    else:
        btc_ret_4bar = 0.0

    # BTC 24-bar return (6h on 15min bars)
    if btc_idx >= 24 and btc_close[btc_idx - 24] > 0:
        btc_ret_24bar = (btc_close[btc_idx] - btc_close[btc_idx - 24]) / btc_close[btc_idx - 24]
    else:
        btc_ret_24bar = 0.0

    # BTC volume ratio
    btc_vol_window = btc_volume[max(0, btc_idx - 20):btc_idx]
    btc_vol_avg = np.mean(btc_vol_window) if len(btc_vol_window) > 0 else 0
    if btc_vol_avg > 0:
        btc_vol_ratio = btc_volume[btc_idx] / btc_vol_avg
    else:
        btc_vol_ratio = 1.0

    # --- Market structure features ---
    sig_ts = pd.Timestamp(sig_time)
    if sig_ts.tzinfo is None:
        sig_ts = sig_ts.tz_localize("UTC")
    hour_of_day = sig_ts.hour
    day_of_week = sig_ts.dayofweek

    return {
        "vol_ratio": vol_ratio,
        "price_change": price_change,
        "price_change_2bar": price_change_2bar,
        "bar_range_norm": bar_range_norm,
        "upper_wick_pct": upper_wick_pct,
        "vol_trend": vol_trend,
        "price_trend": price_trend,
        "volatility": volatility,
        "rel_volume_rank": rel_volume_rank,
        "btc_ret_4bar": btc_ret_4bar,
        "btc_ret_24bar": btc_ret_24bar,
        "btc_vol_ratio": btc_vol_ratio,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
    }


def build_walk_forward_splits(
    signals_df: pd.DataFrame,
    train_months: int = TRAIN_MONTHS,
    test_months: int = TEST_MONTHS,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Build time-based walk-forward splits."""
    times = pd.to_datetime(signals_df["signal_time"])
    min_time = times.min()
    max_time = times.max()

    splits = []
    train_start = min_time
    train_end = train_start + pd.DateOffset(months=train_months)

    while True:
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_start >= max_time:
            break

        # Cap test_end at max_time
        test_end = min(test_end, max_time)

        train_mask = (times >= train_start) & (times < train_end)
        test_mask = (times >= test_start) & (times < test_end)

        train_data = signals_df[train_mask]
        test_data = signals_df[test_mask]

        if len(train_data) < MIN_TRAIN_SIGNALS:
            logger.warning(
                f"Window {train_start.date()} -> {train_end.date()}: "
                f"only {len(train_data)} train signals, skipping"
            )
        elif len(test_data) == 0:
            logger.warning(f"Window test {test_start.date()} -> {test_end.date()}: empty test set")
        else:
            splits.append((train_data, test_data))

        # Slide: keep all history up to new train_end (expanding) or slide
        train_end = train_end + pd.DateOffset(months=test_months)

    return splits


def compute_metrics(trades: pd.DataFrame, label: str = "") -> dict:
    """Compute backtest metrics on a set of trades."""
    if len(trades) == 0:
        return {"label": label, "n_trades": 0}

    pnls = trades["pnl_pct"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]

    win_rate = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
    gross_profit = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_loss = float(abs(np.sum(losses))) if len(losses) > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = float(np.mean(pnls))

    return {
        "label": label,
        "n_trades": len(trades),
        "win_rate": win_rate,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
    }


def print_metrics(m: dict):
    """Print metrics dict."""
    if m["n_trades"] == 0:
        logger.info(f"  {m['label']}: No trades")
        return
    logger.info(
        f"  {m['label']}: {m['n_trades']} trades | "
        f"WR={m['win_rate']:.1%} | "
        f"PF={m['profit_factor']:.2f} | "
        f"E={m['expectancy']:+.2%} | "
        f"AvgW={m['avg_win']:.2%} AvgL={m['avg_loss']:.2%}"
    )


def main():
    start_t = time.time()
    logger.info("ML-Filtered Momentum Scanner Backtester")
    logger.info(f"Data: {RAW_DIR}")
    logger.info(
        f"Scanner: vol_mult={VOL_MULT} price_thresh={PRICE_THRESH} | "
        f"ML threshold={ML_PROB_THRESHOLD}"
    )

    # --- Load BTC ---
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

    # --- Load all coins and collect signals + features ---
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    logger.info(f"Loading {len(symbol_dirs)} coins...")

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

        # Liquidity filter
        avg_vol = np.nanmean(volume[-100:]) if len(volume) >= 100 else np.nanmean(volume)
        avg_price = np.nanmean(close[-100:]) if len(close) >= 100 else np.nanmean(close)
        if avg_vol * avg_price < MIN_VOLUME_USD / 96:
            continue

        # Detect breakouts
        signals = detect_breakouts(
            df, vol_mult=VOL_MULT, price_thresh=PRICE_THRESH, lookback=LOOKBACK
        )
        if not signals:
            continue

        coins_with_signals += 1

        # Compute ATR
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

            # Simulate trade to get label
            trade = simulate_pullback_trade(
                close=close,
                high=high,
                low=low,
                signal_bar=bar,
                direction=sig["direction"],
                pullback_pct=PULLBACK_PCT,
                atr_stop_mult=ATR_STOP_MULT,
                atr_value=atr[bar],
                max_hold_bars=MAX_HOLD_BARS,
                atr_tp_mult=ATR_TP_MULT,
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
            )
            if feats is None:
                continue

            label = 1 if trade["pnl_pct"] > 0 else 0

            record = {
                **feats,
                "label": label,
                "pnl_pct": trade["pnl_pct"],
                "exit_reason": trade["exit_reason"],
                "symbol": symbol,
                "signal_time": str(times[bar]),
                "direction": sig["direction"],
            }
            all_signals.append(record)

    logger.info(f"Coins processed: {coins_processed}, with signals: {coins_with_signals}")
    logger.info(f"Total signals collected: {len(all_signals)}")

    if not all_signals:
        logger.error("No signals collected!")
        return

    # --- Build DataFrame ---
    signals_df = pd.DataFrame(all_signals)
    signals_df["signal_time"] = pd.to_datetime(signals_df["signal_time"])
    signals_df = signals_df.sort_values("signal_time").reset_index(drop=True)

    # Sanity check features
    for col in FEATURE_COLS:
        if col not in signals_df.columns:
            logger.error(f"Missing feature column: {col}")
            return

    # Replace any inf/nan in features
    signals_df[FEATURE_COLS] = signals_df[FEATURE_COLS].replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)

    label_dist = signals_df["label"].value_counts()
    logger.info(f"Label distribution: winners={label_dist.get(1, 0)} losers={label_dist.get(0, 0)}")

    # --- Walk-forward ML filter ---
    logger.info("\n" + "=" * 60)
    logger.info("WALK-FORWARD ML FILTER")
    logger.info("=" * 60)

    splits = build_walk_forward_splits(signals_df, TRAIN_MONTHS, TEST_MONTHS)
    logger.info(f"Walk-forward windows: {len(splits)}")

    if not splits:
        logger.error("Not enough data for walk-forward splits!")
        return

    all_filtered_trades = []
    all_unfiltered_oos_trades = []
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

        # Check class balance
        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)
        logger.info(f"  Train labels: {n_pos} winners / {n_neg} losers")

        if n_pos < 10 or n_neg < 10:
            logger.warning("  Insufficient class balance, skipping window")
            all_unfiltered_oos_trades.append(test_data)
            all_filtered_trades.append(test_data.iloc[0:0])  # empty
            continue

        # Train LightGBM
        model = LGBMClassifier(**ML_PARAMS)
        model.fit(X_train, y_train)

        # Predict on test
        proba = model.predict_proba(X_test)[:, 1]
        test_data = test_data.copy()
        test_data["ml_proba"] = proba

        # Filter
        filtered = test_data[proba > ML_PROB_THRESHOLD]
        unfiltered = test_data

        logger.info(
            f"  OOS signals: {len(test_data)} total, "
            f"{len(filtered)} pass ML filter ({len(filtered)/len(test_data):.1%})"
        )

        # Per-window metrics
        m_all = compute_metrics(unfiltered, "Unfiltered")
        m_filtered = compute_metrics(filtered, f"ML>{ML_PROB_THRESHOLD}")
        print_metrics(m_all)
        print_metrics(m_filtered)

        all_unfiltered_oos_trades.append(unfiltered)
        all_filtered_trades.append(filtered)

        # Feature importance
        fi = dict(zip(FEATURE_COLS, model.feature_importances_))
        top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info(
            f"  Top features: "
            + ", ".join(f"{k}={v:.0f}" for k, v in top_features)
        )

        window_results.append({
            "window": i + 1,
            "train_n": len(train_data),
            "test_n": len(test_data),
            "filtered_n": len(filtered),
            "filter_rate": len(filtered) / len(test_data) if len(test_data) > 0 else 0,
            "unfiltered_wr": m_all.get("win_rate", 0),
            "filtered_wr": m_filtered.get("win_rate", 0),
            "unfiltered_pf": m_all.get("profit_factor", 0),
            "filtered_pf": m_filtered.get("profit_factor", 0),
        })

    # --- Aggregate OOS results ---
    logger.info("\n" + "=" * 60)
    logger.info("AGGREGATE OOS RESULTS")
    logger.info("=" * 60)

    all_oos = pd.concat(all_unfiltered_oos_trades, ignore_index=True)
    all_filtered = pd.concat(all_filtered_trades, ignore_index=True)

    # Baseline: all signals (including in-sample)
    m_baseline = compute_metrics(signals_df, "ALL signals (in+OOS)")
    m_oos = compute_metrics(all_oos, "OOS unfiltered")
    m_filtered_total = compute_metrics(all_filtered, f"OOS ML>{ML_PROB_THRESHOLD}")

    logger.info("\nBaseline (all signals including in-sample):")
    print_metrics(m_baseline)

    logger.info("\nOOS comparison:")
    print_metrics(m_oos)
    print_metrics(m_filtered_total)

    # Win rate improvement
    if m_oos.get("win_rate", 0) > 0 and m_filtered_total.get("win_rate", 0) > 0:
        wr_improvement = m_filtered_total["win_rate"] - m_oos["win_rate"]
        pf_improvement = m_filtered_total["profit_factor"] - m_oos["profit_factor"]
        trade_reduction = 1 - (m_filtered_total["n_trades"] / m_oos["n_trades"])
        logger.info(f"\nML Filter Impact:")
        logger.info(f"  Win rate: {m_oos['win_rate']:.1%} -> {m_filtered_total['win_rate']:.1%} "
                    f"({wr_improvement:+.1%})")
        logger.info(f"  Profit factor: {m_oos['profit_factor']:.2f} -> "
                    f"{m_filtered_total['profit_factor']:.2f} ({pf_improvement:+.2f})")
        logger.info(f"  Trade count: {m_oos['n_trades']} -> {m_filtered_total['n_trades']} "
                    f"({trade_reduction:.1%} filtered out)")

    # Monthly breakdown for filtered OOS trades
    if len(all_filtered) > 0:
        all_filtered = all_filtered.copy()
        all_filtered["month"] = pd.to_datetime(
            all_filtered["signal_time"]
        ).dt.to_period("M")
        monthly_f = all_filtered.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])
        monthly_f["leveraged"] = monthly_f["sum"] * MAX_LEVERAGE
        logger.info(f"\nOOS Filtered Monthly returns (3x leverage):")
        for month, row in monthly_f.iterrows():
            logger.info(f"  {month}: {row['leveraged']:+.1%} ({int(row['count'])} trades)")

        profitable_months = (monthly_f["leveraged"] > 0).sum()
        total_months = len(monthly_f)
        logger.info(f"\nProfitable months: {profitable_months}/{total_months} "
                    f"({profitable_months/total_months:.0%})")

    # Exit reason breakdown
    if len(all_filtered) > 0:
        logger.info(f"\nFiltered OOS exit reasons:")
        for reason, count in all_filtered["exit_reason"].value_counts().items():
            logger.info(f"  {reason}: {count} ({count/len(all_filtered):.1%})")

    # Window summary table
    if window_results:
        logger.info("\n" + "=" * 60)
        logger.info("WINDOW SUMMARY")
        logger.info("=" * 60)
        logger.info(
            f"{'Win':>4} {'TrainN':>7} {'TestN':>6} {'FiltN':>6} "
            f"{'FiltRate':>9} {'UfWR':>6} {'MlWR':>6} {'UfPF':>6} {'MlPF':>6}"
        )
        for w in window_results:
            logger.info(
                f"{w['window']:>4} {w['train_n']:>7} {w['test_n']:>6} {w['filtered_n']:>6} "
                f"{w['filter_rate']:>9.1%} {w['unfiltered_wr']:>6.1%} "
                f"{w['filtered_wr']:>6.1%} {w['unfiltered_pf']:>6.2f} {w['filtered_pf']:>6.2f}"
            )

    elapsed = time.time() - start_t
    logger.info(f"\nTotal elapsed: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
