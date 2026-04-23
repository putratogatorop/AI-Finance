"""Live Scanner — Volume-Surge v1 (SHORT-ONLY).

Simple signal: volume spike + price direction + cooldown.
No 3-phase state machine — fires immediately on detection.

Catches 76.6% of top movers vs 41% for the 3-phase pattern.
PF 1.23 on shorts with trailing stop (learn/22042026/analysis_3).

Usage: python scripts/live_scanner_surge_v1.py
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

from src import db_adapters  # noqa: F401  # registers numpy→psycopg2 adapters

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
GATEIO_BASE = "https://api.gateio.ws/api/v4"
SCANNER_TOP_N = int(os.environ.get("SCANNER_TOP_N", "100"))

# Scanner params (match learn/22042026/analysis_3 exactly)
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
COOLDOWN_SECONDS = 96 * 900  # 96 bars * 15min = 24h

SCAN_INTERVAL = 60
CANDLE_INTERVAL = 900         # 15 min in seconds
CANDLE_INTERVAL_STR = "15m"
HISTORY_BARS = 200            # enough for 96-bar lookback + 20-bar MA
MIN_VOLUME_USD = 500_000

# ── Logging setup ─────────────────────────────────────────────────
logger = logging.getLogger("scanner_surge")


def setup_logging():
    """Configure logging after ensuring logs/ directory exists."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/scanner_surge.log", mode="a"),
        ],
    )


# ── Gate.io API helpers ───────────────────────────────────────────


def fetch_json(url: str, retries: int = 2) -> dict | list | None:
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            req = Request(
                url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
            )
            resp = urlopen(req, timeout=10)
            return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_all_tickers() -> dict[str, dict]:
    """Fetch all Gate.io USDT-perpetual FUTURES tickers.

    Returns {contract: {last, vol, ...}}.
    Short signals require futures — spot has no short mechanic.
    """
    data = fetch_json(f"{GATEIO_BASE}/futures/usdt/tickers")
    if not data:
        return {}

    tickers = {}
    for t in data:
        pair = t.get("contract", "")
        if not pair.endswith("_USDT"):
            continue
        try:
            tickers[pair] = {
                "last": float(t.get("last", 0)),
                "base_volume": float(t.get("volume_24h_base", 0)),
                "quote_volume": float(t.get("volume_24h_quote", 0)),
                "high_24h": float(t.get("high_24h", 0)),
                "low_24h": float(t.get("low_24h", 0)),
                "change_pct": float(t.get("change_percentage", 0)),
            }
        except (ValueError, TypeError):
            continue
    return tickers


_db_engine = None


def get_db_engine():
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(DB_URL)
    return _db_engine


def fetch_candles(
    pair: str, interval: str = CANDLE_INTERVAL_STR, limit: int = HISTORY_BARS
) -> pd.DataFrame | None:
    """Load 15min candles from DB (source of truth)."""
    asset = pair.replace("_", "")
    eng = get_db_engine()
    with eng.connect() as conn:
        result = conn.execute(text("""
            SELECT timestamp, open, high, low, close, volume
            FROM asset_prices_15m WHERE asset = :a
            ORDER BY timestamp DESC LIMIT :lim
        """), {"a": asset, "lim": limit}).fetchall()
    if not result or len(result) < 10:
        return None
    df = pd.DataFrame(result, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.sort_values("timestamp").reset_index(drop=True)


# ── DB helpers ────────────────────────────────────────────────────


def ensure_table(engine):
    """Create scanner_signals_surge_v1 table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals_surge_v1 (
                id SERIAL PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT NOT NULL DEFAULT 'short',
                vol_ratio FLOAT,
                price_move FLOAT,
                signal_time TIMESTAMPTZ NOT NULL,
                status TEXT NOT NULL DEFAULT 'active',
                entry_price FLOAT,
                pnl_pct FLOAT,
                exit_reason TEXT,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                regime_allowed BOOLEAN DEFAULT TRUE,
                UNIQUE(symbol, signal_time)
            )
        """))


def write_signal(engine, symbol: str, vol_ratio: float, price_move: float, entry_price: float):
    """Write signal to scanner_signals_surge_v1 table."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO scanner_signals_surge_v1
                (symbol, direction, vol_ratio, price_move, signal_time, status,
                 entry_price, regime_allowed)
                VALUES (:symbol, 'short', :vol_ratio, :price_move,
                        :signal_time, 'active', :entry_price, TRUE)
                ON CONFLICT (symbol, signal_time) DO NOTHING
            """),
            {
                "symbol": symbol,
                "vol_ratio": vol_ratio,
                "price_move": price_move,
                "signal_time": datetime.now(timezone.utc),
                "entry_price": entry_price,
            },
        )


# ── Detection logic ───────────────────────────────────────────────


def check_surge_short(df: pd.DataFrame) -> dict | None:
    """Check if latest bar has volume surge + price drop.

    Conditions:
    1. vol_ratio >= 3x (volume vs 20-bar MA)
    2. Price dropped >= 5% from 96-bar high
    3. Drop magnitude > rise magnitude (bearish bias)

    Returns dict with vol_ratio, price_move, entry_price, or None.
    """
    i = len(df) - 1
    if i < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return None

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # Volume MA
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    if vol_ma[i] <= 0:
        return None

    vol_ratio = volume[i] / vol_ma[i]
    if vol_ratio < VOL_SPIKE:
        return None

    # Price drop from 96-bar high
    lookback_slice = high[max(0, i - PRICE_LOOKBACK):i]
    rolling_high = np.max(lookback_slice)
    if rolling_high <= 0:
        return None

    price_drop = (rolling_high - close[i]) / rolling_high
    if price_drop < PRICE_MOVE_THRESH:
        return None

    # Drop > rise check: compare drop from high vs rise from low
    rolling_low = np.min(low[max(0, i - PRICE_LOOKBACK):i])
    if rolling_low <= 0:
        return None

    price_rise = (close[i] - rolling_low) / rolling_low
    if price_drop <= price_rise:
        return None

    return {
        "vol_ratio": vol_ratio,
        "price_move": price_drop,
        "entry_price": float(close[i]),
    }


# ── Main loop ─────────────────────────────────────────────────────


def main():
    setup_logging()

    logger.info("=" * 60)
    logger.info("LIVE SCANNER — VOLUME-SURGE v1 (SHORT-ONLY)")
    logger.info("=" * 60)
    logger.info(
        f"Params: vol_spike={VOL_SPIKE} price_thresh={PRICE_MOVE_THRESH} "
        f"lookback={PRICE_LOOKBACK} cooldown={COOLDOWN_SECONDS}s"
    )
    logger.info(f"Scan interval: {SCAN_INTERVAL}s | Top {SCANNER_TOP_N} pairs")

    engine = create_engine(DB_URL)
    ensure_table(engine)

    candle_cache: dict[str, pd.DataFrame] = {}
    recent_signals: dict[str, float] = {}  # {symbol: epoch_time}

    # Candle refresh timing
    last_candle_update = 0.0

    cycle = 0
    logger.info("Fetching initial candle history...")

    # Get initial ticker list
    tickers = fetch_all_tickers()
    logger.info(f"Found {len(tickers)} USDT futures perps on Gate.io")

    # Pre-load top pairs by volume
    sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True)
    loaded = 0
    for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
        df = fetch_candles(pair)
        if df is not None:
            candle_cache[pair] = df
            loaded += 1
        time.sleep(0.1)  # Rate limit
    last_candle_update = time.time()
    logger.info(f"Pre-loaded {loaded} pairs' candle history")

    logger.info("Scanner started. Press Ctrl+C to stop.")
    logger.info("=" * 60)

    while True:
        try:
            cycle += 1
            cycle_start = time.time()
            now = cycle_start

            # Fetch tickers
            tickers = fetch_all_tickers()
            if not tickers:
                logger.warning(f"Cycle {cycle}: Failed to fetch tickers, skipping")
                time.sleep(SCAN_INTERVAL)
                continue

            # Update candle history every 15 minutes
            if now - last_candle_update > CANDLE_INTERVAL:
                logger.info(f"Cycle {cycle}: Updating 15min candle history...")
                sorted_pairs = sorted(
                    tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True
                )
                updated = 0
                for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
                    df = fetch_candles(pair)
                    if df is not None:
                        candle_cache[pair] = df
                        updated += 1
                    time.sleep(0.1)
                last_candle_update = now
                logger.info(f"  Updated {updated} pairs' candle history")

            # Scan each pair for surge signal
            signals_written = 0
            for pair, ticker in tickers.items():
                if pair not in candle_cache:
                    continue
                if ticker["quote_volume"] < MIN_VOLUME_USD:
                    continue

                symbol = pair.replace("_", "")

                # Cooldown check
                if symbol in recent_signals:
                    if now - recent_signals[symbol] < COOLDOWN_SECONDS:
                        continue

                df = candle_cache[pair]
                result = check_surge_short(df)
                if result is None:
                    continue

                write_signal(
                    engine, symbol,
                    result["vol_ratio"], result["price_move"], result["entry_price"],
                )
                recent_signals[symbol] = now
                signals_written += 1

                logger.info(
                    f"  SIGNAL: {pair} SHORT | vol={result['vol_ratio']:.1f}x "
                    f"drop={result['price_move']:.1%} entry=${result['entry_price']:.4f}"
                )

            # Cycle summary
            elapsed = time.time() - cycle_start
            if cycle % 5 == 0 or signals_written > 0:
                logger.info(
                    f"Cycle {cycle}: {len(candle_cache)} pairs scanned, "
                    f"{signals_written} signals ({elapsed:.1f}s)"
                )

            sleep_time = max(0, SCAN_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Scanner surge v1 stopped by user.")
            break
        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
            time.sleep(SCAN_INTERVAL)

    engine.dispose()
    logger.info("Scanner surge v1 shutdown complete.")


if __name__ == "__main__":
    main()
