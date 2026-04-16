# scripts/backfill_scanner_signals.py
"""Backfill scanner_signals table with historical ML-filtered signals.

This populates the Next.js dashboard with historical data for validation.
Runs the full scanner + ML filter and writes OOS filtered signals to PostgreSQL.
"""

import logging
import sys
import time

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
ML_THRESHOLD = 0.65


def create_table(engine):
    """Create scanner_signals table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                direction INT NOT NULL,
                ml_prob FLOAT NOT NULL,
                vol_ratio FLOAT NOT NULL,
                price_change FLOAT NOT NULL,
                signal_time TIMESTAMPTZ NOT NULL,
                status VARCHAR(20) DEFAULT 'active',
                pnl_pct FLOAT,
                exit_reason VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_scanner_symbol ON scanner_signals(symbol)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_scanner_time ON scanner_signals(signal_time)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_scanner_status ON scanner_signals(status)"))


def main():
    engine = create_engine(DB_URL)
    start = time.time()

    logger.info("Backfilling scanner_signals table for dashboard")

    # Create table
    create_table(engine)

    # Clear existing data
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM scanner_signals"))
    logger.info("Cleared existing scanner_signals")

    # Import and run the ML filter pipeline
    from pathlib import Path
    from scripts.backtest_momentum_scanner import (
        load_coin, detect_breakouts, compute_atr, simulate_pullback_trade,
    )

    RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
    VOL_MULT = 4.0
    PRICE_THRESH = 0.035
    LOOKBACK = 20
    PULLBACK_PCT = 0.02
    ATR_STOP_MULT = 2.0
    ATR_TP_MULT = 5.0
    MAX_HOLD_BARS = 192
    MAX_DAILY_MOVE = 0.30
    BTC_TREND_PERIOD = 30
    MIN_VOLUME_USD = 1_000_000

    # Load BTC
    btc_df = load_coin(RAW_DIR / "BTCUSDT")
    btc_close = btc_df["close"].values.astype(float)
    btc_volume = btc_df["volume"].values.astype(float)
    btc_ema = pd.Series(btc_close).ewm(span=BTC_TREND_PERIOD).mean().values
    btc_times = btc_df["open_time"].values
    logger.info(f"BTC loaded: {len(btc_df)} bars")

    # Collect all signals with features (same as ML filter script)
    from scripts.backtest_scanner_ml_filter import extract_features, FEATURE_COLS, build_walk_forward_splits

    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    all_signals = []

    logger.info(f"Scanning {len(symbol_dirs)} coins...")
    for sym_dir in symbol_dirs:
        symbol = sym_dir.name
        df = load_coin(sym_dir)
        if df is None or len(df) < 200:
            continue

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        times = df["open_time"].values

        avg_vol = np.nanmean(volume[-100:]) if len(volume) >= 100 else np.nanmean(volume)
        avg_price = np.nanmean(close[-100:]) if len(close) >= 100 else np.nanmean(close)
        if avg_vol * avg_price < MIN_VOLUME_USD / 96:
            continue

        signals = detect_breakouts(df, vol_mult=VOL_MULT, price_thresh=PRICE_THRESH, lookback=LOOKBACK)
        if not signals:
            continue

        atr = compute_atr(high, low, close)

        for sig in signals:
            bar = sig["bar"]
            if np.isnan(atr[bar]) or atr[bar] == 0:
                continue

            sig_time = times[bar]
            btc_idx = np.searchsorted(btc_times, sig_time)
            btc_idx = min(btc_idx, len(btc_close) - 1)
            if sig["direction"] == 1 and btc_close[btc_idx] < btc_ema[btc_idx]:
                continue
            if sig["direction"] == -1 and btc_close[btc_idx] > btc_ema[btc_idx]:
                continue

            trade = simulate_pullback_trade(
                close=close, high=high, low=low, volume=volume,
                signal_bar=bar, direction=sig["direction"],
                pullback_pct=PULLBACK_PCT, atr_stop_mult=ATR_STOP_MULT,
                atr_value=atr[bar], max_hold_bars=MAX_HOLD_BARS,
                atr_tp_mult=ATR_TP_MULT,
            )
            if trade is None:
                continue

            feats = extract_features(
                df=df, bar=bar,
                btc_close=btc_close, btc_volume=btc_volume,
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
                "vol_ratio": sig["volume_ratio"],
                "price_change": sig["price_change"],
            }
            all_signals.append(record)

    logger.info(f"Collected {len(all_signals)} signals")

    if not all_signals:
        logger.error("No signals!")
        return

    # Build DataFrame and run walk-forward ML
    signals_df = pd.DataFrame(all_signals)
    signals_df["signal_time"] = pd.to_datetime(signals_df["signal_time"])
    signals_df = signals_df.sort_values("signal_time").reset_index(drop=True)
    signals_df[FEATURE_COLS] = signals_df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        logger.error("LightGBM not installed!")
        return

    ML_PARAMS = {
        "objective": "binary", "num_leaves": 15, "min_child_samples": 30,
        "learning_rate": 0.05, "n_estimators": 200, "feature_fraction": 0.8,
        "verbose": -1, "random_state": 42,
    }

    # Walk-forward: train 18mo, test 6mo
    splits = build_walk_forward_splits(signals_df, 18, 6)
    logger.info(f"Walk-forward: {len(splits)} windows")

    all_filtered = []
    for i, (train_data, test_data) in enumerate(splits):
        X_train = train_data[FEATURE_COLS].values
        y_train = train_data["label"].values
        X_test = test_data[FEATURE_COLS].values

        model = LGBMClassifier(**ML_PARAMS)
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        test_data = test_data.copy()
        test_data["ml_proba"] = proba

        filtered = test_data[proba > ML_THRESHOLD]
        all_filtered.append(filtered)
        logger.info(f"  Window {i+1}: {len(test_data)} signals, {len(filtered)} pass ML>{ML_THRESHOLD}")

    if not all_filtered:
        logger.error("No filtered signals!")
        return

    combined = pd.concat(all_filtered, ignore_index=True)
    logger.info(f"Total filtered signals: {len(combined)}")

    # Write to DB
    rows = []
    for _, s in combined.iterrows():
        rows.append({
            "symbol": s["symbol"],
            "direction": int(s["direction"]),
            "ml_prob": float(s["ml_proba"]),
            "vol_ratio": float(s["vol_ratio"]),
            "price_change": float(s["price_change"]),
            "signal_time": pd.to_datetime(s["signal_time"]),
            "status": "traded",
            "pnl_pct": float(s["pnl_pct"]),
            "exit_reason": s["exit_reason"],
        })

    sql = text("""
        INSERT INTO scanner_signals
            (symbol, direction, ml_prob, vol_ratio, price_change, signal_time, status, pnl_pct, exit_reason)
        VALUES
            (:symbol, :direction, :ml_prob, :vol_ratio, :price_change, :signal_time, :status, :pnl_pct, :exit_reason)
    """)

    with engine.begin() as conn:
        conn.execute(sql, rows)

    # Summary
    wins = len([r for r in rows if r["pnl_pct"] > 0])
    total = len(rows)
    with engine.connect() as conn:
        db_count = conn.execute(text("SELECT COUNT(*) FROM scanner_signals")).scalar()

    elapsed = time.time() - start
    logger.info(f"\nDONE in {elapsed:.0f}s!")
    logger.info(f"  Written: {db_count} signals to scanner_signals")
    logger.info(f"  Win rate: {wins}/{total} = {wins/total:.1%}")
    logger.info(f"  Dashboard ready at: http://localhost:3000/scanner")

    engine.dispose()


if __name__ == "__main__":
    main()
