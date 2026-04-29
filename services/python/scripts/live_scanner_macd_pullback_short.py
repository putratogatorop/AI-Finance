"""Live scanner — macd_pullback_short detector (Week 8 v8 BALANCED, locked).

Polls 15min candles for top-N USDT-perp pairs, detects macd_pullback_short
SHORT signals, and writes them to `scanner_signals_macd_pullback_short`. The
paper executor reads this table and routes signals to v8 paper accounts.

Detector spec (pre-registered, locked 2026-04-29 — symmetric mirror of long):
    BIG TF:  daily MACD(12,26,9) bear when hist < 0 AND macd < signal
    SMALL TF: 4h MACD(12,26,9) cross-DOWN
              = (macd < signal at j) AND (macd >= signal at j-1)
              AND macd was ABOVE signal for ≥ 2 prior 4h bars
    Entry: first 15m bar after each 4h cross-down where daily_bear is True
    Cooldown: 24h per symbol (96 × 15m bars)

Validated 3y backtest (run id 92373355_20260429T041931Z):
    holdout PF 4.289, DSR 1.000, MC p5 3.320, wf_p5 2.102 — 5/6 ship gates.
    Strongest single config in v5 effort.
    portfolio_metrics.json wf_p5 1.486 (with macd_early_trend_short_e2 +
    macd_pullback_long_e2 + rsi_recovery_long_e2 = the v8 BALANCED stack).

Architectural notes:
  - Computes daily MACD + 4h MACD per-asset on each candle refresh (15 min).
    No FeatureContext usage — MACD math is self-contained for portability.
  - Same DB read pattern as live_scanner_d1e.py (asset_prices_15m).
  - No classifier — pure rule-based per Week 8 v8 paper-stack decision.
  - 15m close + entry_price + atr14_at_entry written to row so paper executor
    can size with v6 ATR-based sizing module (src/ml/v8/sizing.py).

Usage from services/python/:
    python scripts/live_scanner_macd_pullback_short.py
"""
from __future__ import annotations

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

from src import db_adapters  # noqa: F401  registers numpy->psycopg2 adapters

# ── Config ───────────────────────────────────────────────────────────
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)
GATEIO_BASE = "https://api.gateio.ws/api/v4"

SCANNER_TOP_N = int(os.environ.get("SCANNER_TOP_N", "100"))
SCAN_INTERVAL = 60
CANDLE_INTERVAL = 900  # 15 min — bar boundary
HISTORY_BARS = 4032  # ~42 days for daily/4h MACD warmup
MIN_VOLUME_USD = 500_000

# Detector params (locked Week 8)
ATR_PERIOD = 14
COOLDOWN_SECONDS = 96 * 900  # 24h per symbol
H4_BARS_PRIOR_ABOVE = 2  # ≥ 2 prior 4h bars above signal before cross-down

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_scanner_macd_pullback_short.log", mode="a"),
    ],
)
logger = logging.getLogger("scanner_macd_pb_short")


# ── Gate.io tickers ──────────────────────────────────────────────────


def fetch_json(url: str, retries: int = 2):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
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
    out: dict[str, dict] = {}
    for t in data:
        pair = t.get("contract", "")
        if not pair.endswith("_USDT"):
            continue
        try:
            out[pair] = {
                "last": float(t.get("last", 0)),
                "quote_volume": float(t.get("volume_24h_quote", 0)),
            }
        except (ValueError, TypeError):
            continue
    return out


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
    if not rows or len(rows) < 4 * 96:  # ≥ 4 days of 15m bars
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["asset"] = asset
    return df.sort_values("timestamp").reset_index(drop=True)


# ── Indicator math ───────────────────────────────────────────────────


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    """MACD(12,26,9). Returns df with columns: macd, signal, histogram."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "signal": sig, "histogram": hist})


def _atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Wilder ATR(14). Standard implementation matching backtest code."""
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.full(n, np.nan)
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    return atr


# ── BTC score (regime, used as audit metadata only) ─────────────────


def _compute_btc_score(btc_df: pd.DataFrame | None) -> float:
    """BTC trend score in [-1, +1] from 4h MACD spread.

    Uses a simple normalization: tanh(macd_hist / 0.005×close).
    Matches the canonical FeatureContext metric direction (positive = bull).
    """
    if btc_df is None or len(btc_df) < 30 * 96:
        return float("nan")
    s = btc_df.set_index("timestamp")["close"].astype(float)
    h4 = s.resample("4h").last().dropna()
    if len(h4) < 30:
        return float("nan")
    m = _macd(h4)
    hist = m["histogram"].iloc[-1]
    px = h4.iloc[-1]
    if not np.isfinite(hist) or not np.isfinite(px) or px <= 0:
        return float("nan")
    return float(np.tanh(hist / (0.005 * px)))


# ── Detector ─────────────────────────────────────────────────────────


def detect_macd_pullback_short(df: pd.DataFrame, btc_score: float) -> dict | None:
    """Evaluate macd_pullback_short detector at the latest 15m bar.

    Returns a dict with detector internals + entry context, or None if any
    gate fails. Math matches the batch detector in
    backtest_5strat_production_v1.py:detect_macd_pullback_short bit-for-bit.
    """
    if len(df) < 4 * 96:
        return None

    ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    s_close = pd.Series(df["close"].astype(float).to_numpy(), index=ts_idx)
    high_arr = df["high"].astype(float).to_numpy()
    low_arr = df["low"].astype(float).to_numpy()
    close_arr = df["close"].astype(float).to_numpy()

    # Daily MACD bear
    d_close = s_close.resample("1D").last().dropna()
    if len(d_close) < 30:
        return None
    d_macd = _macd(d_close)
    daily_bear_d = (d_macd["histogram"] < 0) & (d_macd["macd"] < d_macd["signal"])

    # 4h MACD cross-down
    h4_close = s_close.resample("4h").last().dropna()
    if len(h4_close) < 30:
        return None
    h4_macd = _macd(h4_close)
    h4_above = (h4_macd["macd"] > h4_macd["signal"]).astype(bool)
    h4_above_prev = h4_above.shift(1).fillna(False)
    h4_above_prev2 = h4_above.shift(2).fillna(False)
    h4_cross_down = (~h4_above) & h4_above_prev & h4_above_prev2

    # The current 4h bucket: floor latest 15m to 4h
    last_15m_ts = ts_idx[-1]
    current_4h_ts = last_15m_ts.floor("4h")

    # Get the 4h MACD values AT the current 4h bucket
    if current_4h_ts not in h4_macd.index:
        return None
    h4_idx = h4_macd.index.get_loc(current_4h_ts)
    if h4_idx < 2:
        return None
    fired_this_4h = bool(h4_cross_down.iloc[h4_idx])
    if not fired_this_4h:
        return None

    # Daily must be bear at this date
    current_day_ts = last_15m_ts.floor("D")
    if current_day_ts not in daily_bear_d.index:
        return None
    if not bool(daily_bear_d.loc[current_day_ts]):
        return None

    # Entry condition: this is the FIRST 15m bar of the new 4h period where the
    # cross fires. Check: previous 15m bar was in the previous 4h bucket OR
    # this is the first bar of the day.
    if len(ts_idx) < 2:
        return None
    prev_15m_4h = ts_idx[-2].floor("4h")
    if prev_15m_4h == current_4h_ts:
        # Already fired earlier in this 4h period — skip (cooldown will catch it too).
        return None

    # Compute ATR14 on 15m for sizing
    atr_arr = _atr14(high_arr, low_arr, close_arr)
    atr14 = float(atr_arr[-1]) if len(atr_arr) > 0 and np.isfinite(atr_arr[-1]) else float("nan")
    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    entry_price = float(close_arr[-1])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    return {
        "symbol": df["asset"].iloc[0] if "asset" in df.columns else None,
        "direction": "short",
        "signal_time": last_15m_ts.to_pydatetime().replace(tzinfo=timezone.utc) if last_15m_ts.tzinfo is None else last_15m_ts.to_pydatetime(),
        "entry_price": entry_price,
        "atr14_at_entry": atr14,
        "daily_macd": float(d_macd["macd"].loc[current_day_ts]),
        "daily_signal": float(d_macd["signal"].loc[current_day_ts]),
        "daily_hist": float(d_macd["histogram"].loc[current_day_ts]),
        "h4_macd": float(h4_macd["macd"].iloc[h4_idx]),
        "h4_signal": float(h4_macd["signal"].iloc[h4_idx]),
        "h4_hist": float(h4_macd["histogram"].iloc[h4_idx]),
        "btc_score": btc_score,
    }


# ── DB schema ────────────────────────────────────────────────────────


def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS scanner_signals_macd_pullback_short (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    direction VARCHAR(5) NOT NULL DEFAULT 'short',
                    signal_time TIMESTAMPTZ NOT NULL,
                    status VARCHAR(10) DEFAULT 'active',
                    entry_price DOUBLE PRECISION,
                    atr14_at_entry DOUBLE PRECISION,
                    daily_macd DOUBLE PRECISION,
                    daily_signal DOUBLE PRECISION,
                    daily_hist DOUBLE PRECISION,
                    h4_macd DOUBLE PRECISION,
                    h4_signal DOUBLE PRECISION,
                    h4_hist DOUBLE PRECISION,
                    btc_score DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (symbol, direction, signal_time)
                )
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_macdpbs_signal_time
                ON scanner_signals_macd_pullback_short (signal_time DESC)
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_macdpbs_active
                ON scanner_signals_macd_pullback_short (status, signal_time)
                WHERE status = 'active'
            """)
        )


def write_signal(engine, *, hit: dict) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO scanner_signals_macd_pullback_short
                    (symbol, direction, signal_time, status,
                     entry_price, atr14_at_entry,
                     daily_macd, daily_signal, daily_hist,
                     h4_macd, h4_signal, h4_hist, btc_score)
                    VALUES (:symbol, :direction, :sig_t, 'active',
                            :ep, :atr, :dm, :ds, :dh, :hm, :hs, :hh, :bs)
                    ON CONFLICT (symbol, direction, signal_time) DO NOTHING
                """),
                {
                    "symbol": hit["symbol"], "direction": hit["direction"],
                    "sig_t": hit["signal_time"],
                    "ep": hit["entry_price"], "atr": hit["atr14_at_entry"],
                    "dm": hit["daily_macd"], "ds": hit["daily_signal"], "dh": hit["daily_hist"],
                    "hm": hit["h4_macd"], "hs": hit["h4_signal"], "hh": hit["h4_hist"],
                    "bs": hit["btc_score"] if np.isfinite(hit["btc_score"]) else None,
                },
            )
        return True
    except Exception as e:
        logger.error(f"Failed to insert macd_pullback_short signal for {hit.get('symbol')}: {e}")
        return False


# ── Main loop ────────────────────────────────────────────────────────


def main() -> None:
    logger.info("=" * 60)
    logger.info("LIVE SCANNER — macd_pullback_short (Week 8 v8 BALANCED, locked)")
    logger.info("=" * 60)
    logger.info(
        f"Detector: daily MACD bear AND 4h MACD cross-down after ≥ "
        f"{H4_BARS_PRIOR_ABOVE} prior 4h bars above signal, "
        f"cooldown {COOLDOWN_SECONDS//3600}h"
    )

    engine = get_db_engine()
    ensure_table(engine)
    logger.info("scanner_signals_macd_pullback_short table ready")

    tickers = fetch_all_tickers()
    if not tickers:
        logger.error("No tickers fetched. Exiting.")
        return
    logger.info(f"Found {len(tickers)} USDT futures perps")

    candle_cache: dict[str, pd.DataFrame] = {}
    cooldowns: dict[str, float] = {}
    last_candle_update = 0.0
    btc_score_cached = float("nan")

    # Pre-load BTC + top-N
    btc_df = fetch_candles("BTC_USDT")
    if btc_df is not None:
        candle_cache["BTC_USDT"] = btc_df
        btc_score_cached = _compute_btc_score(btc_df)
    sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True)
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
    logger.info(f"Pre-loaded {loaded} pairs (+ BTC). btc_score={btc_score_cached:.3f}")
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
                    candle_cache["BTC_USDT"] = btc_new
                    btc_score_cached = _compute_btc_score(btc_new)
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
                logger.info(f"  refreshed {updated} pairs, btc_score={btc_score_cached:.3f}")

            # Scan
            signals_written = 0
            scanned = 0
            for pair, ticker in tickers.items():
                if pair == "BTC_USDT":
                    continue
                if ticker["quote_volume"] < MIN_VOLUME_USD:
                    continue
                df = candle_cache.get(pair)
                if df is None:
                    continue
                symbol = pair.replace("_", "")
                scanned += 1

                # Cooldown
                if now - cooldowns.get(symbol, 0.0) < COOLDOWN_SECONDS:
                    continue

                hit = detect_macd_pullback_short(df, btc_score_cached)
                if hit is None:
                    continue
                hit["symbol"] = symbol  # ensure set

                ok = write_signal(engine, hit=hit)
                if ok:
                    cooldowns[symbol] = now
                    signals_written += 1
                    logger.info(f"  FIRED: {symbol} ep={hit['entry_price']:.4f} "
                                f"atr={hit['atr14_at_entry']:.4f} "
                                f"d_hist={hit['daily_hist']:+.4f} "
                                f"h4_hist={hit['h4_hist']:+.4f}")

            elapsed = time.time() - cycle_start
            if signals_written > 0 or cycle % 5 == 0:
                logger.info(
                    f"Cycle {cycle}: scanned={scanned} fired={signals_written} ({elapsed:.1f}s)"
                )

            sleep_time = max(0.0, SCAN_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
            break
        except Exception as e:
            logger.exception(f"Cycle {cycle} error: {e}")
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
