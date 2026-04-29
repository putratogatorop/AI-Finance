"""Live scanner — macd_pullback_long detector (Week 8 v8 BALANCED, locked).

Symmetric mirror of macd_pullback_short. Fires a LONG entry when:
  1. Daily MACD(12,26,9) bull: hist > 0 AND macd > signal.
  2. 4h MACD(12,26,9) cross-UP after ≥ 2 prior 4h bars below signal.
  3. Entry on the first 15m bar of the new 4h period where both conditions met.
  4. 24h per-symbol cooldown.

Validated in run d71e63c6_20260429T035916Z:
  macd_pullback_long_e2: holdout PF 1.566, wf_p5 1.099, DSR 1.000, MC p5 1.388.
  Portfolio-level wf_p5 = 1.486 in BALANCED 4-det stack alongside
  macd_pullback_short_e2, macd_early_trend_short_e2, rsi_recovery_long_e2.

Usage from services/python/:
    python scripts/live_scanner_macd_pullback_long.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import UTC
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from src import db_adapters  # noqa: F401

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)
GATEIO_BASE = "https://api.gateio.ws/api/v4"

SCANNER_TOP_N = int(os.environ.get("SCANNER_TOP_N", "100"))
SCAN_INTERVAL = 60
CANDLE_INTERVAL = 900
HISTORY_BARS = 4032
MIN_VOLUME_USD = 500_000

ATR_PERIOD = 14
COOLDOWN_SECONDS = 96 * 900  # 24h
H4_BARS_PRIOR_BELOW = 2  # ≥ 2 prior 4h bars below signal before cross-up

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_scanner_macd_pullback_long.log", mode="a"),
    ],
)
logger = logging.getLogger("scanner_macd_pb_long")


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
    if not rows or len(rows) < 4 * 96:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["asset"] = asset
    return df.sort_values("timestamp").reset_index(drop=True)


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    hist = macd - sig
    return pd.DataFrame({"macd": macd, "signal": sig, "histogram": hist})


def _atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
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


def _compute_btc_score(btc_df: pd.DataFrame | None) -> float:
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


def detect_macd_pullback_long(df: pd.DataFrame, btc_score: float) -> dict | None:
    """Evaluate macd_pullback_long at the latest 15m bar.

    Mirror of macd_pullback_short with direction inverted:
    - Daily MACD bull (hist > 0 AND macd > signal)
    - 4h MACD cross-UP after ≥ 2 prior 4h bars BELOW signal
    - First 15m bar of the new 4h period where both are True
    """
    if len(df) < 4 * 96:
        return None

    ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    s_close = pd.Series(df["close"].astype(float).to_numpy(), index=ts_idx)
    high_arr = df["high"].astype(float).to_numpy()
    low_arr = df["low"].astype(float).to_numpy()
    close_arr = df["close"].astype(float).to_numpy()

    # Daily MACD bull
    d_close = s_close.resample("1D").last().dropna()
    if len(d_close) < 30:
        return None
    d_macd = _macd(d_close)
    daily_bull_d = (d_macd["histogram"] > 0) & (d_macd["macd"] > d_macd["signal"])

    # 4h MACD cross-UP (≥ 2 prior bars BELOW signal)
    h4_close = s_close.resample("4h").last().dropna()
    if len(h4_close) < 30:
        return None
    h4_macd = _macd(h4_close)
    h4_above = (h4_macd["macd"] > h4_macd["signal"]).astype(bool)
    h4_below_prev = ~h4_above.shift(1).fillna(True)   # prev bar was below
    h4_below_prev2 = ~h4_above.shift(2).fillna(True)  # bar before prev was also below
    h4_cross_up = h4_above & h4_below_prev & h4_below_prev2

    last_15m_ts = ts_idx[-1]
    current_4h_ts = last_15m_ts.floor("4h")

    if current_4h_ts not in h4_macd.index:
        return None
    h4_idx = h4_macd.index.get_loc(current_4h_ts)
    if h4_idx < 2:
        return None
    if not bool(h4_cross_up.iloc[h4_idx]):
        return None

    # Daily must be bull at this date
    current_day_ts = last_15m_ts.floor("D")
    if current_day_ts not in daily_bull_d.index:
        return None
    if not bool(daily_bull_d.loc[current_day_ts]):
        return None

    # First 15m bar of the new 4h period
    if len(ts_idx) < 2:
        return None
    if ts_idx[-2].floor("4h") == current_4h_ts:
        return None

    atr_arr = _atr14(high_arr, low_arr, close_arr)
    atr14 = float(atr_arr[-1]) if len(atr_arr) > 0 and np.isfinite(atr_arr[-1]) else float("nan")
    if not np.isfinite(atr14) or atr14 <= 0:
        return None

    entry_price = float(close_arr[-1])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    sig_time = (
        last_15m_ts.to_pydatetime().replace(tzinfo=UTC)
        if last_15m_ts.tzinfo is None
        else last_15m_ts.to_pydatetime()
    )

    return {
        "symbol": df["asset"].iloc[0] if "asset" in df.columns else None,
        "direction": "long",
        "signal_time": sig_time,
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


def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals_macd_pullback_long (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                direction VARCHAR(5) NOT NULL DEFAULT 'long',
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
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_macdpbl_signal_time
            ON scanner_signals_macd_pullback_long (signal_time DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_macdpbl_active
            ON scanner_signals_macd_pullback_long (status, signal_time)
            WHERE status = 'active'
        """))


def write_signal(engine, *, hit: dict) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO scanner_signals_macd_pullback_long
                (symbol, direction, signal_time, status,
                 entry_price, atr14_at_entry,
                 daily_macd, daily_signal, daily_hist,
                 h4_macd, h4_signal, h4_hist, btc_score)
                VALUES (:symbol, :direction, :sig_t, 'active',
                        :ep, :atr, :dm, :ds, :dh, :hm, :hs, :hh, :bs)
                ON CONFLICT (symbol, direction, signal_time) DO NOTHING
            """), {
                "symbol": hit["symbol"], "direction": hit["direction"],
                "sig_t": hit["signal_time"],
                "ep": hit["entry_price"], "atr": hit["atr14_at_entry"],
                "dm": hit["daily_macd"], "ds": hit["daily_signal"], "dh": hit["daily_hist"],
                "hm": hit["h4_macd"], "hs": hit["h4_signal"], "hh": hit["h4_hist"],
                "bs": hit["btc_score"] if np.isfinite(hit["btc_score"]) else None,
            })
        return True
    except Exception as e:
        logger.error(f"Failed to insert signal for {hit.get('symbol')}: {e}")
        return False


def main() -> None:
    logger.info("=" * 60)
    logger.info("LIVE SCANNER — macd_pullback_long (Week 8 v8 BALANCED, locked)")
    logger.info("=" * 60)
    logger.info(
        f"Detector: daily MACD bull AND 4h MACD cross-UP after ≥"
        f"{H4_BARS_PRIOR_BELOW} prior bars below signal, "
        f"cooldown {COOLDOWN_SECONDS // 3600}h"
    )

    engine = get_db_engine()
    ensure_table(engine)
    logger.info("scanner_signals_macd_pullback_long table ready")

    tickers = fetch_all_tickers()
    if not tickers:
        logger.error("No tickers fetched. Exiting.")
        return
    logger.info(f"Found {len(tickers)} USDT futures perps")

    candle_cache: dict[str, pd.DataFrame] = {}
    cooldowns: dict[str, float] = {}
    last_candle_update = 0.0
    btc_score_cached = float("nan")

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

                if now - cooldowns.get(symbol, 0.0) < COOLDOWN_SECONDS:
                    continue

                hit = detect_macd_pullback_long(df, btc_score_cached)
                if hit is None:
                    continue
                hit["symbol"] = symbol

                ok = write_signal(engine, hit=hit)
                if ok:
                    cooldowns[symbol] = now
                    signals_written += 1
                    logger.info(
                        f"  FIRED: {symbol} ep={hit['entry_price']:.4f} "
                        f"atr={hit['atr14_at_entry']:.4f} "
                        f"d_hist={hit['daily_hist']:+.4f} "
                        f"h4_hist={hit['h4_hist']:+.4f}"
                    )

            elapsed = time.time() - cycle_start
            if signals_written > 0 or cycle % 5 == 0:
                logger.info(
                    f"Cycle {cycle}: scanned={scanned} fired={signals_written} ({elapsed:.1f}s)"
                )

            time.sleep(max(0.0, SCAN_INTERVAL - elapsed))

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
            break
        except Exception as e:
            logger.exception(f"Cycle {cycle} error: {e}")
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
