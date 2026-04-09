"""Live Momentum Scanner Daemon.

Polls Gate.io every 60 seconds, detects breakouts on all USDT pairs,
runs ML filter, writes signals to scanner_signals PostgreSQL table.

Usage: python scripts/live_scanner.py
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_scanner.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
GATEIO_BASE = "https://api.gateio.ws/api/v4"

# Scanner params (same as backtest)
VOL_MULT = 4.0
PRICE_THRESH = 0.035
LOOKBACK = 20
MIN_VOLUME_USD = 1_000_000
MAX_DAILY_MOVE = 0.30
ML_THRESHOLD = 0.65

SCAN_INTERVAL = 60  # seconds
CANDLE_INTERVAL = 900  # 15 min in seconds
HISTORY_BARS = 200

# Feature columns (same as ML filter)
FEATURE_COLS = [
    "vol_ratio", "price_change", "price_change_2bar", "bar_range_norm",
    "upper_wick_pct", "vol_trend", "price_trend", "volatility",
    "rel_volume_rank", "btc_ret_4bar", "btc_ret_24bar", "btc_vol_ratio",
    "hour_of_day", "day_of_week",
    "consolidation_bars", "vol_acceleration", "num_concurrent",
    "atr_compression", "daily_trend_alignment",
]


def fetch_json(url: str, retries: int = 2) -> dict | list | None:
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            resp = urlopen(req, timeout=10)
            return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"API error (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_all_tickers() -> dict[str, dict]:
    """Fetch all Gate.io spot tickers. Returns {symbol: {last, vol, ...}}."""
    data = fetch_json(f"{GATEIO_BASE}/spot/tickers")
    if not data:
        return {}

    tickers = {}
    for t in data:
        pair = t.get("currency_pair", "")
        if not pair.endswith("_USDT"):
            continue
        try:
            tickers[pair] = {
                "last": float(t.get("last", 0)),
                "base_volume": float(t.get("base_volume", 0)),
                "quote_volume": float(t.get("quote_volume", 0)),
                "high_24h": float(t.get("high_24h", 0)),
                "low_24h": float(t.get("low_24h", 0)),
                "change_pct": float(t.get("change_percentage", 0)),
            }
        except (ValueError, TypeError):
            continue
    return tickers


def fetch_candles(
    pair: str, interval: int = CANDLE_INTERVAL, limit: int = HISTORY_BARS
) -> pd.DataFrame | None:
    """Fetch 15min candles for one pair from Gate.io."""
    url = (
        f"{GATEIO_BASE}/spot/candlesticks"
        f"?currency_pair={pair}&interval={interval}&limit={limit}"
    )
    data = fetch_json(url)
    if not data or len(data) < 10:
        return None

    rows = []
    for c in data:
        # Gate.io candle format: [timestamp, volume, close, high, low, open, is_window_closed]
        try:
            rows.append({
                "timestamp": datetime.fromtimestamp(int(c[0]), tz=timezone.utc),
                "volume": float(c[1]),
                "close": float(c[2]),
                "high": float(c[3]),
                "low": float(c[4]),
                "open": float(c[5]),
            })
        except (ValueError, IndexError):
            continue

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


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
            abs(high[i] - close[i - 1]) / close[i],
            abs(low[i] - close[i - 1]) / close[i],
        )
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def detect_breakout(df: pd.DataFrame) -> dict | None:
    """Check if the latest candle is a breakout."""
    close = df["close"].values
    volume = df["volume"].values
    high = df["high"].values
    low = df["low"].values
    open_ = df["open"].values
    n = len(close)

    if n < LOOKBACK + 2:
        return None

    i = n - 1  # Latest bar

    # Volume average
    vol_ma = np.mean(volume[max(0, i - LOOKBACK):i])
    if vol_ma == 0:
        return None

    vol_ratio = volume[i] / vol_ma
    if vol_ratio < VOL_MULT:
        return None

    # Price change
    if close[i - 1] == 0:
        return None
    ret = (close[i] - close[i - 1]) / close[i - 1]
    if abs(ret) < PRICE_THRESH:
        return None

    # Too-late filter: check 96-bar (24h) move
    if i >= 96:
        day_ret = (close[i] - close[i - 96]) / close[i - 96]
        if abs(day_ret) > MAX_DAILY_MOVE:
            return None

    # Cooldown: don't signal same coin twice within 8 bars
    # (caller handles this)

    direction = 1 if ret > 0 else -1
    return {
        "direction": direction,
        "vol_ratio": vol_ratio,
        "price_change": ret,
        "bar_index": i,
    }


def extract_features(
    df: pd.DataFrame,
    bar: int,
    btc_df: pd.DataFrame | None,
    signal: dict,
) -> dict | None:
    """Extract ML features at signal bar. Simplified from backtest version."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_ = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)

    n = len(close)
    if bar < 100 or bar >= n:
        return None

    vol_ratio = signal["vol_ratio"]
    price_change = signal["price_change"]

    # 2-bar return
    price_change_2bar = (
        (close[bar] - close[bar - 2]) / close[bar - 2]
        if bar >= 2 and close[bar - 2] > 0
        else 0
    )

    # Candle size
    bar_range = high[bar] - low[bar]
    bar_range_norm = bar_range / close[bar] if close[bar] > 0 else 0

    # Upper wick
    body_top = max(open_[bar], close[bar])
    upper_wick = high[bar] - body_top
    upper_wick_pct = upper_wick / bar_range if bar_range > 0 else 0

    # Volume trend
    vol_window = volume[max(0, bar - 20):bar]
    if len(vol_window) >= 2:
        vol_mean = np.mean(vol_window)
        if vol_mean > 0:
            x = np.arange(len(vol_window), dtype=float)
            vol_trend = float(np.polyfit(x, vol_window / vol_mean, 1)[0])
        else:
            vol_trend = 0.0
    else:
        vol_trend = 0.0

    # Price trend
    close_window = close[max(0, bar - 20):bar]
    price_trend = (
        (close_window[-1] - close_window[0]) / close_window[0]
        if len(close_window) >= 2 and close_window[0] > 0
        else 0
    )

    # Volatility
    if len(close_window) >= 2:
        rets = np.diff(close_window) / close_window[:-1]
        rets = rets[np.isfinite(rets)]
        volatility = float(np.std(rets)) if len(rets) > 0 else 0
    else:
        volatility = 0

    # Volume rank
    vol_window100 = volume[max(0, bar - 100):bar + 1]
    rel_volume_rank = float(np.sum(vol_window100[:-1] < volume[bar])) / max(
        len(vol_window100) - 1, 1
    )

    # BTC context
    btc_ret_4bar = 0.0
    btc_ret_24bar = 0.0
    btc_vol_ratio = 1.0
    if btc_df is not None and len(btc_df) > 24:
        btc_c = btc_df["close"].values.astype(float)
        btc_v = btc_df["volume"].values.astype(float)
        btc_n = len(btc_c)
        if btc_n >= 5 and btc_c[btc_n - 5] > 0:
            btc_ret_4bar = (btc_c[-1] - btc_c[btc_n - 5]) / btc_c[btc_n - 5]
        if btc_n >= 25 and btc_c[btc_n - 25] > 0:
            btc_ret_24bar = (btc_c[-1] - btc_c[btc_n - 25]) / btc_c[btc_n - 25]
        btc_vol_avg = (
            np.mean(btc_v[max(0, btc_n - 20):btc_n - 1])
            if btc_n > 20
            else np.mean(btc_v)
        )
        if btc_vol_avg > 0:
            btc_vol_ratio = btc_v[-1] / btc_vol_avg

    # Time
    ts = df["timestamp"].iloc[bar]
    if hasattr(ts, "hour"):
        hour_of_day = ts.hour
        day_of_week = ts.weekday()
    else:
        hour_of_day = 0
        day_of_week = 0

    # ATR features
    atr_14 = compute_atr(high, low, close, 14)
    atr_96 = compute_atr(high, low, close, 96)

    # Consolidation bars
    consolidation_bars = 0
    if bar >= 20:
        recent_atr = atr_14[max(0, bar - 50):bar]
        recent_atr = recent_atr[~np.isnan(recent_atr)]
        if len(recent_atr) > 0:
            atr_median = np.median(recent_atr)
            for j in range(bar - 1, max(bar - 100, 0), -1):
                if j < len(atr_14) and not np.isnan(atr_14[j]) and atr_14[j] < atr_median:
                    consolidation_bars += 1
                else:
                    break

    # Volume acceleration
    vol_acceleration = (
        (volume[bar] - volume[bar - 2]) / volume[bar - 2]
        if bar >= 3 and volume[bar - 2] > 0
        else 0
    )

    # Concurrent (caller fills)
    num_concurrent = 0

    # ATR compression
    atr_compression = (
        atr_14[bar] / atr_96[bar]
        if not np.isnan(atr_14[bar]) and not np.isnan(atr_96[bar]) and atr_96[bar] > 0
        else 1.0
    )

    # Daily trend alignment
    if bar >= 96:
        ema = pd.Series(close[:bar + 1]).ewm(span=96).mean().values
        ema_slope = (ema[bar] - ema[bar - 24]) / ema[bar - 24] if ema[bar - 24] > 0 else 0
        daily_trend_alignment = ema_slope * (1 if price_change > 0 else -1)
    else:
        daily_trend_alignment = 0

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
        "consolidation_bars": consolidation_bars,
        "vol_acceleration": vol_acceleration,
        "num_concurrent": num_concurrent,
        "atr_compression": atr_compression,
        "daily_trend_alignment": daily_trend_alignment,
    }


def write_signal(engine, symbol: str, signal: dict, ml_prob: float):
    """Write a signal to the scanner_signals table."""
    sql = text("""
        INSERT INTO scanner_signals
            (symbol, direction, ml_prob, vol_ratio, price_change, signal_time, status)
        VALUES
            (:symbol, :direction, :ml_prob, :vol_ratio, :price_change, :signal_time, 'active')
    """)
    with engine.begin() as conn:
        conn.execute(sql, {
            "symbol": symbol,
            "direction": signal["direction"],
            "ml_prob": ml_prob,
            "vol_ratio": signal["vol_ratio"],
            "price_change": signal["price_change"],
            "signal_time": datetime.now(timezone.utc),
        })


def main():
    # Ensure log dir exists
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 60)
    logger.info("LIVE MOMENTUM SCANNER DAEMON")
    logger.info("=" * 60)
    logger.info(
        f"Params: vol_mult={VOL_MULT} price_thresh={PRICE_THRESH} ml_threshold={ML_THRESHOLD}"
    )
    logger.info(f"Scan interval: {SCAN_INTERVAL}s")

    # Connect to DB
    engine = create_engine(DB_URL)

    # Ensure table exists
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

    # Try to load ML model
    model = None
    model_path = Path("models/live_scanner_model.joblib")
    if model_path.exists():
        import joblib
        model = joblib.load(model_path)
        logger.info(f"ML model loaded from {model_path}")
    else:
        logger.warning(
            "No ML model found — running without ML filter (all breakouts will be signals)"
        )
        logger.warning(f"To enable ML filter, save a trained model to {model_path}")

    # Load initial candle history
    logger.info("Fetching initial candle history...")
    candle_cache: dict[str, pd.DataFrame] = {}
    btc_df = None

    # Get list of all USDT pairs
    tickers = fetch_all_tickers()
    logger.info(f"Found {len(tickers)} USDT pairs on Gate.io")

    # Fetch BTC candles first
    btc_df = fetch_candles("BTC_USDT")
    if btc_df is not None:
        candle_cache["BTC_USDT"] = btc_df
        logger.info(f"BTC history loaded: {len(btc_df)} candles")

    # BTC EMA for trend filter
    btc_ema = None
    if btc_df is not None:
        btc_ema = pd.Series(btc_df["close"].values).ewm(span=30).mean().values

    # Track recent signals to avoid duplicates
    recent_signals: dict[str, float] = {}  # {symbol: last_signal_time}

    # Track last candle update time
    last_candle_update = 0

    cycle = 0
    logger.info("Scanner started. Press Ctrl+C to stop.")
    logger.info("=" * 60)

    while True:
        try:
            cycle += 1
            cycle_start = time.time()

            # Fetch all tickers
            tickers = fetch_all_tickers()
            if not tickers:
                logger.warning(f"Cycle {cycle}: Failed to fetch tickers, skipping")
                time.sleep(SCAN_INTERVAL)
                continue

            # Update candle history every 15 minutes
            now = time.time()
            if now - last_candle_update > CANDLE_INTERVAL:
                logger.info(f"Cycle {cycle}: Updating 15min candle history...")

                # Update BTC
                btc_new = fetch_candles("BTC_USDT")
                if btc_new is not None:
                    btc_df = btc_new
                    candle_cache["BTC_USDT"] = btc_df
                    btc_ema = pd.Series(btc_df["close"].values).ewm(span=30).mean().values

                # Update top pairs by volume (limit API calls)
                sorted_pairs = sorted(
                    tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True
                )
                updated = 0
                for pair, _ in sorted_pairs[:100]:  # Top 100 by volume
                    if pair == "BTC_USDT":
                        continue
                    df = fetch_candles(pair)
                    if df is not None:
                        candle_cache[pair] = df
                        updated += 1
                    time.sleep(0.1)  # Rate limit

                last_candle_update = now
                logger.info(f"  Updated {updated} pairs' candle history")

            # Scan for breakouts
            breakouts = []
            for pair, ticker in tickers.items():
                if pair not in candle_cache:
                    continue

                # Volume filter
                if ticker["quote_volume"] < MIN_VOLUME_USD / 96:
                    continue

                df = candle_cache[pair]
                signal = detect_breakout(df)
                if signal is None:
                    continue

                # BTC trend filter
                if btc_df is not None and btc_ema is not None:
                    btc_close_now = btc_df["close"].iloc[-1]
                    btc_ema_now = btc_ema[-1]
                    if signal["direction"] == 1 and btc_close_now < btc_ema_now:
                        continue
                    if signal["direction"] == -1 and btc_close_now > btc_ema_now:
                        continue

                # Cooldown: skip if we signaled this coin in last 2 hours
                if pair in recent_signals:
                    if now - recent_signals[pair] < 7200:
                        continue

                # Extract features
                feats = extract_features(df, signal["bar_index"], btc_df, signal)
                if feats is None:
                    continue

                breakouts.append({"pair": pair, "signal": signal, "features": feats})

            # Fill num_concurrent
            for b in breakouts:
                b["features"]["num_concurrent"] = len(breakouts)

            # Run ML filter
            signals_written = 0
            for b in breakouts:
                pair = b["pair"]
                signal = b["signal"]
                feats = b["features"]

                if model is not None:
                    # Run ML prediction
                    X = np.array([[feats[col] for col in FEATURE_COLS]])
                    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

                    try:
                        if hasattr(model, "predict_proba"):
                            ml_prob = float(model.predict_proba(X)[0][1])
                        else:
                            ml_prob = float(model.predict(X)[0])
                    except Exception as e:
                        logger.error(f"ML prediction error for {pair}: {e}")
                        continue

                    if ml_prob < ML_THRESHOLD:
                        continue
                else:
                    ml_prob = 0.0  # No ML model — pass all breakouts

                # Write signal
                direction_str = "LONG" if signal["direction"] == 1 else "SHORT"
                logger.info(
                    f"  SIGNAL: {pair} {direction_str} | "
                    f"ml={ml_prob:.2f} vol={signal['vol_ratio']:.1f}x "
                    f"price={signal['price_change']:+.1%}"
                )

                # Convert BTC_USDT → BTCUSDT (remove underscore) to match existing data format
                write_signal(engine, pair.replace("_", ""), signal, ml_prob)
                recent_signals[pair] = now
                signals_written += 1

            # Cycle summary
            elapsed = time.time() - cycle_start
            if cycle % 5 == 0 or signals_written > 0:
                logger.info(
                    f"Cycle {cycle}: {len(tickers)} pairs, "
                    f"{len(breakouts)} breakouts, "
                    f"{signals_written} signals written "
                    f"({elapsed:.1f}s)"
                )

            # Sleep until next cycle
            sleep_time = max(0, SCAN_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Scanner stopped by user.")
            break
        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
            time.sleep(SCAN_INTERVAL)

    engine.dispose()
    logger.info("Scanner shutdown complete.")


if __name__ == "__main__":
    main()
