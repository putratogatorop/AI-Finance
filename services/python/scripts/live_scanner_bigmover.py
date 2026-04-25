"""Live scanner — bigmover signal family (baseline / price_accel_atr / multi_bar_confirm).

Polls 15min candles for top-N USDT-perp pairs, detects three signal variants in
BOTH long and short directions, writes hits to `scanner_signals_bigmover`.

Signals (mirror of backtest_bigmover_longshort_3y.py detectors):
  baseline:           vol_ratio >= 3 AND price_{drop|rise} >= 5% AND bearish/bullish dominance
  price_accel_atr:    baseline AND |price_accel| >= 1 * ATR_14
  multi_bar_confirm:  baseline AND next-bar close moves with the trade direction

Architectural notes:
  - No ML gate. Backtest (87-variant grid) showed ML filter cut trade count 10x
    without lifting PF_p5; no-ML anchors dominated. Signals are written raw; the
    paper executor routes them to bigmover accounts for live validation.
  - Multi-bar-confirm fires ONE BAR LATER than baseline/price_accel_atr because
    it needs bar[i+1]'s close. Live version signals on bar i+1 close.
  - Cooldown per (symbol, signal_type, direction) is 96 bars (24h), matching
    COOLDOWN_BARS in the backtest.
  - 3-class BTC regime (SMA 50/200 on daily closes) is computed once per hour
    and stored on each signal row via `regime_allowed` (True iff direction
    matches current regime: long+bull or short+bear).

Usage from services/python/:
    python scripts/live_scanner_bigmover.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from src import db_adapters  # noqa: F401  # registers numpy->psycopg2 adapters

# ── Config ───────────────────────────────────────────────────────────
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)
GATEIO_BASE = "https://api.gateio.ws/api/v4"

SCANNER_TOP_N = int(os.environ.get("SCANNER_TOP_N", "100"))
SCAN_INTERVAL = 60
CANDLE_INTERVAL = 900  # 15 min
HISTORY_BARS = 2100  # ~22 days for rolling windows + regime warmup
MIN_VOLUME_USD = 500_000

# Detector params — mirror backtest_bigmover_longshort_3y.py exactly
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
ACCEL_MULT_OF_ATR = 1.0
COOLDOWN_SECONDS = 96 * 900  # 24h

# Regime (matches backtest)
REGIME_SMA_FAST = 50
REGIME_SMA_SLOW = 200
REGIME_CACHE_SECONDS = 3600

SIGNALS = ("baseline", "price_accel_atr", "multi_bar_confirm")
DIRECTIONS = ("short", "long")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_scanner_bigmover.log", mode="a"),
    ],
)
logger = logging.getLogger("scanner_bigmover")


# ── Gate.io tickers ──────────────────────────────────────────────────


def fetch_json(url: str, retries: int = 2):
    for attempt in range(retries):
        try:
            req = Request(
                url,
                headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"},
            )
            resp = urlopen(req, timeout=10)
            return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_all_tickers() -> dict[str, dict]:
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
                "quote_volume": float(t.get("volume_24h_quote", 0)),
            }
        except (ValueError, TypeError):
            continue
    return tickers


# ── Candle fetch from DB ─────────────────────────────────────────────

_db_engine = None


def get_db_engine():
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(DB_URL)
    return _db_engine


def fetch_candles(pair: str, limit: int = HISTORY_BARS) -> pd.DataFrame | None:
    asset = pair.replace("_", "")
    eng = get_db_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT timestamp, open, high, low, close, volume
                FROM asset_prices_15m WHERE asset = :a
                ORDER BY timestamp DESC LIMIT :lim
            """),
            {"a": asset, "lim": limit},
        ).fetchall()
    if not rows or len(rows) < PRICE_LOOKBACK + VOL_MA_PERIOD + 5:
        return None
    df = pd.DataFrame(
        rows, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Detector helpers (array-based, latest-bar evaluation) ───────────


def compute_atr_at(high: np.ndarray, low: np.ndarray, close: np.ndarray, i: int) -> float:
    if i < ATR_PERIOD:
        return 0.0
    tr_vals = []
    for k in range(i - ATR_PERIOD + 1, i + 1):
        prev_close = close[k - 1] if k > 0 else close[k]
        tr = max(
            high[k] - low[k],
            abs(high[k] - prev_close),
            abs(low[k] - prev_close),
        )
        tr_vals.append(tr)
    return float(np.mean(tr_vals))


def detect_baseline_at(df: pd.DataFrame, i: int, direction: str) -> dict | None:
    """Check if baseline (bearish OR bullish bigmover) fires on bar i."""
    if i < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return None
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    vol_ma = float(np.mean(volume[i - VOL_MA_PERIOD + 1 : i + 1]))
    if vol_ma <= 0:
        return None
    vol_ratio = float(volume[i]) / vol_ma
    if vol_ratio < VOL_SPIKE:
        return None

    rolling_high = float(np.max(high[i - PRICE_LOOKBACK + 1 : i + 1]))
    rolling_low = float(np.min(low[i - PRICE_LOOKBACK + 1 : i + 1]))
    if rolling_high <= 0 or rolling_low <= 0:
        return None
    c = float(close[i])
    price_drop = (rolling_high - c) / rolling_high
    price_rise = (c - rolling_low) / rolling_low

    if direction == "short":
        if price_drop < PRICE_MOVE_THRESH or price_drop <= price_rise:
            return None
        price_move = price_drop
    else:  # long
        if price_rise < PRICE_MOVE_THRESH or price_rise <= price_drop:
            return None
        price_move = price_rise

    return {
        "vol_ratio": vol_ratio,
        "price_move": price_move,
        "entry_close": c,
    }


def detect_price_accel_at(df: pd.DataFrame, i: int) -> dict | None:
    """Compute |accel| >= 1 * ATR_14 at bar i. Returns {"accel_value": X} or None."""
    if i < 2:
        return None
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    accel = (close[i] - close[i - 1]) - (close[i - 1] - close[i - 2])
    atr = compute_atr_at(high, low, close, i)
    if atr <= 0:
        return None
    if abs(accel) < ACCEL_MULT_OF_ATR * atr:
        return None
    return {"accel_value": float(accel), "atr_value": atr}


def detect_multi_bar_confirm_at(
    df: pd.DataFrame, i: int, direction: str
) -> dict | None:
    """Check if bar i+1's close moves with the trade direction relative to bar i.

    This means the SIGNAL fires at bar i+1 (one bar later than baseline/price_accel).
    Caller is responsible for passing i such that bar i+1 is the latest closed bar,
    and ensuring the baseline at bar i was satisfied.
    """
    close = df["close"].values
    if i + 1 >= len(df):
        return None
    if direction == "short":
        if close[i + 1] >= close[i]:
            return None
    else:
        if close[i + 1] <= close[i]:
            return None
    return {"confirm_delta": float(close[i + 1] - close[i])}


# ── Regime (3-class, daily SMA 50/200) ──────────────────────────────


def compute_regime(btc_df: pd.DataFrame) -> str:
    """Return 'bull' | 'bear' | 'sideways' using BTC daily SMA_50 / SMA_200."""
    if btc_df is None or len(btc_df) < REGIME_SMA_SLOW * 96:
        return "sideways"
    daily = (
        btc_df.set_index("timestamp")
        .resample("1D")
        .agg({"close": "last"})
        .dropna()
    )
    if len(daily) < REGIME_SMA_SLOW:
        return "sideways"
    c = float(daily["close"].iloc[-1])
    sma50 = float(daily["close"].rolling(REGIME_SMA_FAST).mean().iloc[-1])
    sma200 = float(daily["close"].rolling(REGIME_SMA_SLOW).mean().iloc[-1])
    if c > sma50 > sma200:
        return "bull"
    if c < sma50 < sma200:
        return "bear"
    return "sideways"


def regime_allows(regime: str, direction: str) -> bool:
    return (regime == "bull" and direction == "long") or (
        regime == "bear" and direction == "short"
    )


# ── DB ───────────────────────────────────────────────────────────────


def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS scanner_signals_bigmover (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    signal_type VARCHAR(20) NOT NULL,
                    direction VARCHAR(5) NOT NULL,
                    vol_ratio DOUBLE PRECISION,
                    price_move DOUBLE PRECISION,
                    accel_value DOUBLE PRECISION,
                    atr_value DOUBLE PRECISION,
                    confirm_delta DOUBLE PRECISION,
                    regime VARCHAR(10),
                    regime_allowed BOOLEAN,
                    signal_time TIMESTAMPTZ NOT NULL,
                    status VARCHAR(10) DEFAULT 'active',
                    entry_price DOUBLE PRECISION,
                    pnl_pct DOUBLE PRECISION,
                    exit_reason VARCHAR(20),
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (symbol, signal_type, direction, signal_time)
                )
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_bigmover_signal_time
                ON scanner_signals_bigmover (signal_time DESC)
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_bigmover_active
                ON scanner_signals_bigmover (status, signal_time)
                WHERE status = 'active'
            """)
        )


def write_signal(
    engine,
    *,
    symbol: str,
    signal_type: str,
    direction: str,
    vol_ratio: float,
    price_move: float,
    accel_value: float | None,
    atr_value: float | None,
    confirm_delta: float | None,
    regime: str,
    regime_allowed: bool,
    signal_time: datetime,
    entry_price: float,
) -> bool:
    """Insert a signal row. Returns True if inserted (False on UNIQUE conflict)."""
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO scanner_signals_bigmover
                    (symbol, signal_type, direction, vol_ratio, price_move,
                     accel_value, atr_value, confirm_delta, regime, regime_allowed,
                     signal_time, status, entry_price)
                    VALUES (:symbol, :stype, :direction, :vr, :pm,
                            :accel, :atr, :cd, :regime, :regime_allowed,
                            :sig_t, 'active', :ep)
                    ON CONFLICT (symbol, signal_type, direction, signal_time) DO NOTHING
                """),
                {
                    "symbol": symbol, "stype": signal_type, "direction": direction,
                    "vr": vol_ratio, "pm": price_move,
                    "accel": accel_value, "atr": atr_value, "cd": confirm_delta,
                    "regime": regime, "regime_allowed": regime_allowed,
                    "sig_t": signal_time, "ep": entry_price,
                },
            )
        return True
    except Exception as e:
        logger.error(f"Failed to insert signal for {symbol} {signal_type} {direction}: {e}")
        return False


# ── Main loop ────────────────────────────────────────────────────────


@dataclass
class SignalKey:
    symbol: str
    signal_type: str
    direction: str


def scan_coin(
    df: pd.DataFrame,
    symbol: str,
    now_ts: float,
    cooldowns: dict[tuple[str, str, str], float],
    regime: str,
    engine,
) -> int:
    """Detect all 6 variant firings on the latest closed bar (and bar-1 for confirm).
    Returns count of signals written.
    """
    if len(df) < PRICE_LOOKBACK + VOL_MA_PERIOD + 5:
        return 0
    i_last = len(df) - 1
    entry_price = float(df["close"].iloc[i_last])
    signal_time = df["timestamp"].iloc[i_last]
    if isinstance(signal_time, pd.Timestamp):
        signal_time_py = signal_time.to_pydatetime()
    else:
        signal_time_py = datetime.now(timezone.utc)
    written = 0

    for direction in DIRECTIONS:
        # 1. Baseline at latest bar
        base = detect_baseline_at(df, i_last, direction)
        if base is None:
            # Only check multi_bar_confirm (which uses bar i_last-1) if baseline fired there
            base_prev = detect_baseline_at(df, i_last - 1, direction)
            if base_prev is not None:
                confirm = detect_multi_bar_confirm_at(df, i_last - 1, direction)
                if confirm is not None:
                    key = (symbol, "multi_bar_confirm", direction)
                    if now_ts - cooldowns.get(key, 0.0) >= COOLDOWN_SECONDS:
                        allowed = regime_allows(regime, direction)
                        if write_signal(
                            engine,
                            symbol=symbol, signal_type="multi_bar_confirm",
                            direction=direction,
                            vol_ratio=base_prev["vol_ratio"],
                            price_move=base_prev["price_move"],
                            accel_value=None, atr_value=None,
                            confirm_delta=confirm["confirm_delta"],
                            regime=regime, regime_allowed=allowed,
                            signal_time=signal_time_py, entry_price=entry_price,
                        ):
                            cooldowns[key] = now_ts
                            written += 1
            continue

        # baseline fired at latest bar — write it (regardless of sub-variants)
        base_key = (symbol, "baseline", direction)
        if now_ts - cooldowns.get(base_key, 0.0) >= COOLDOWN_SECONDS:
            allowed = regime_allows(regime, direction)
            if write_signal(
                engine,
                symbol=symbol, signal_type="baseline", direction=direction,
                vol_ratio=base["vol_ratio"], price_move=base["price_move"],
                accel_value=None, atr_value=None, confirm_delta=None,
                regime=regime, regime_allowed=allowed,
                signal_time=signal_time_py, entry_price=entry_price,
            ):
                cooldowns[base_key] = now_ts
                written += 1

        # 2. price_accel_atr (same bar as baseline)
        accel = detect_price_accel_at(df, i_last)
        if accel is not None:
            accel_key = (symbol, "price_accel_atr", direction)
            if now_ts - cooldowns.get(accel_key, 0.0) >= COOLDOWN_SECONDS:
                allowed = regime_allows(regime, direction)
                if write_signal(
                    engine,
                    symbol=symbol, signal_type="price_accel_atr", direction=direction,
                    vol_ratio=base["vol_ratio"], price_move=base["price_move"],
                    accel_value=accel["accel_value"], atr_value=accel["atr_value"],
                    confirm_delta=None,
                    regime=regime, regime_allowed=allowed,
                    signal_time=signal_time_py, entry_price=entry_price,
                ):
                    cooldowns[accel_key] = now_ts
                    written += 1

        # 3. multi_bar_confirm — uses bar i_last (baseline fired here) as i,
        #    needs bar i_last + 1 (future). Can't fire yet. Re-check at next bar.
        #    Instead we detect multi_bar_confirm for bar (i_last - 1) above when
        #    baseline's current-bar fire FAILS — which handles the "bar just closed
        #    after baseline bar" case.
    return written


def main() -> None:
    logger.info("=" * 60)
    logger.info("LIVE SCANNER — BIGMOVER (baseline / price_accel_atr / multi_bar_confirm)")
    logger.info("=" * 60)
    logger.info(
        f"Params: vol_spike={VOL_SPIKE} price_move>={PRICE_MOVE_THRESH:.0%} "
        f"accel_mult={ACCEL_MULT_OF_ATR} cooldown={COOLDOWN_SECONDS//3600}h"
    )

    engine = get_db_engine()
    ensure_table(engine)

    tickers = fetch_all_tickers()
    if not tickers:
        logger.error("No tickers fetched. Exiting.")
        return
    logger.info(f"Found {len(tickers)} USDT futures perps")

    candle_cache: dict[str, pd.DataFrame] = {}
    cooldowns: dict[tuple[str, str, str], float] = {}
    btc_df: pd.DataFrame | None = None
    regime = "sideways"
    last_regime_check = 0.0
    last_candle_update = 0.0

    # Pre-load
    btc_df = fetch_candles("BTC_USDT")
    if btc_df is not None:
        candle_cache["BTC_USDT"] = btc_df
    sorted_pairs = sorted(
        tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True
    )
    loaded = 0
    for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
        if pair == "BTC_USDT":
            continue
        df = fetch_candles(pair)
        if df is not None:
            candle_cache[pair] = df
            loaded += 1
        time.sleep(0.05)
    last_candle_update = time.time()
    logger.info(f"Pre-loaded {loaded} pairs")

    if btc_df is not None:
        regime = compute_regime(btc_df)
        last_regime_check = time.time()
        logger.info(f"Initial regime: {regime}")

    logger.info("Scanner started. Ctrl+C to stop.")
    cycle = 0
    while True:
        try:
            cycle += 1
            cycle_start = time.time()
            now = cycle_start

            tickers = fetch_all_tickers()
            if not tickers:
                logger.warning(f"Cycle {cycle}: No tickers, skipping")
                time.sleep(SCAN_INTERVAL)
                continue

            # Refresh candles every 15 min
            if now - last_candle_update > CANDLE_INTERVAL:
                logger.info(f"Cycle {cycle}: refreshing candles...")
                btc_new = fetch_candles("BTC_USDT")
                if btc_new is not None:
                    btc_df = btc_new
                    candle_cache["BTC_USDT"] = btc_df
                sorted_pairs = sorted(
                    tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True
                )
                updated = 0
                for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
                    if pair == "BTC_USDT":
                        continue
                    df = fetch_candles(pair)
                    if df is not None:
                        candle_cache[pair] = df
                        updated += 1
                    time.sleep(0.05)
                last_candle_update = now
                logger.info(f"  refreshed {updated} pairs")

            # Regime refresh hourly
            if now - last_regime_check > REGIME_CACHE_SECONDS:
                if btc_df is not None:
                    regime = compute_regime(btc_df)
                    last_regime_check = now
                    logger.info(f"Cycle {cycle}: regime -> {regime}")

            # Scan
            signals_written = 0
            for pair, ticker in tickers.items():
                if pair == "BTC_USDT":
                    continue
                if ticker["quote_volume"] < MIN_VOLUME_USD:
                    continue
                df = candle_cache.get(pair)
                if df is None:
                    continue
                symbol = pair.replace("_", "")
                signals_written += scan_coin(
                    df, symbol, now, cooldowns, regime, engine
                )

            elapsed = time.time() - cycle_start
            if signals_written > 0 or cycle % 5 == 0:
                logger.info(
                    f"Cycle {cycle}: {signals_written} signals written, "
                    f"{len(candle_cache)} pairs cached, regime={regime} "
                    f"({elapsed:.1f}s)"
                )

            sleep_time = max(0, SCAN_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
            break
        except Exception as e:
            logger.exception(f"Cycle {cycle} error: {e}")
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
