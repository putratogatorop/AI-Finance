"""Live scanner — rsi_recovery_long detector (Week 8 v8 BALANCED, locked).

Fires a LONG entry when daily RSI(14) recovers from the oversold zone:
  1. Daily RSI(14) was below RSI_OVERSOLD_THRESHOLD (35) yesterday.
  2. Daily RSI(14) crosses above RSI_OVERSOLD_THRESHOLD today (recovery).
  3. Entry on the first 15m bar of the new day where this condition fires.
  4. 24h per-symbol cooldown.

No MACD regime filter — this detector is intentionally regime-agnostic,
which is why it has near-zero correlation (0.007) with macd_pullback_long_e2
and provides the key diversification benefit in the BALANCED 4-det portfolio.

Validated in backtest_entry_sweep_v3_d71e63c6_20260429T030622Z:
  rsi_recovery_long_e2: holdout PF 2.222, wf_p5 1.812, DSR 1.000,
  MC p5 1.474. Ship-eligible with eff_n waiver (only gate failed).
  Portfolio-level wf_p5 = 1.486 in BALANCED 4-det stack.

NOTE: The original backtest_entry_sweep_v3 script was a local research script
and was not committed to git. This live scanner implements the pre-registered
detector spec (RSI recovery from oversold zone).

Usage from services/python/:
    python scripts/live_scanner_rsi_recovery_long.py
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
HISTORY_BARS = 4032  # ~42 days
MIN_VOLUME_USD = 500_000

ATR_PERIOD = 14
RSI_PERIOD = 14
RSI_OVERSOLD_THRESHOLD = 35.0  # daily RSI cross-up through this level = recovery signal
COOLDOWN_SECONDS = 96 * 900  # 24h per symbol

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_scanner_rsi_recovery_long.log", mode="a"),
    ],
)
logger = logging.getLogger("scanner_rsi_recov_long")


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


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder RSI(period). Standard EMA-smoothed gain/loss ratio."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


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


def detect_rsi_recovery_long(df: pd.DataFrame, btc_score: float) -> dict | None:
    """Evaluate rsi_recovery_long at the latest 15m bar.

    Fires when daily RSI(14) crosses from below RSI_OVERSOLD_THRESHOLD (35)
    to above it on the current day (yesterday was oversold, today recovered).
    Entry on the first 15m bar of the day where this condition is True.
    """
    if len(df) < 4 * 96:
        return None

    ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    s_close = pd.Series(df["close"].astype(float).to_numpy(), index=ts_idx)
    high_arr = df["high"].astype(float).to_numpy()
    low_arr = df["low"].astype(float).to_numpy()
    close_arr = df["close"].astype(float).to_numpy()

    # Daily RSI
    d_close = s_close.resample("1D").last().dropna()
    if len(d_close) < RSI_PERIOD + 5:
        return None
    d_rsi = _rsi(d_close, RSI_PERIOD)

    last_15m_ts = ts_idx[-1]
    current_day_ts = last_15m_ts.floor("D")

    # Need at least today and yesterday in the daily index
    if current_day_ts not in d_rsi.index:
        return None
    daily_pos = d_rsi.index.get_loc(current_day_ts)
    if daily_pos < 1:
        return None

    rsi_today = float(d_rsi.iloc[daily_pos])
    rsi_yesterday = float(d_rsi.iloc[daily_pos - 1])

    # RSI recovery cross-up: was below threshold yesterday, now above today
    if not (np.isfinite(rsi_today) and np.isfinite(rsi_yesterday)):
        return None
    if rsi_yesterday >= RSI_OVERSOLD_THRESHOLD:
        return None  # wasn't oversold yesterday
    if rsi_today < RSI_OVERSOLD_THRESHOLD:
        return None  # still oversold today, no cross-up yet

    # Entry: first 15m bar of this day (any bar in today counts as valid entry
    # if this is the day the cross occurred, since the cross is computed on daily close)
    # Only fire on the FIRST 15m bar of the current day to avoid re-firing within the day.
    prev_15m_day = ts_idx[-2].floor("D") if len(ts_idx) >= 2 else None
    if prev_15m_day == current_day_ts:
        return None  # already processed an earlier bar today

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
        "rsi_today": rsi_today,
        "rsi_yesterday": rsi_yesterday,
        "rsi_threshold": RSI_OVERSOLD_THRESHOLD,
        "btc_score": btc_score,
    }


def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals_rsi_recovery_long (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                direction VARCHAR(5) NOT NULL DEFAULT 'long',
                signal_time TIMESTAMPTZ NOT NULL,
                status VARCHAR(10) DEFAULT 'active',
                entry_price DOUBLE PRECISION,
                atr14_at_entry DOUBLE PRECISION,
                rsi_today DOUBLE PRECISION,
                rsi_yesterday DOUBLE PRECISION,
                rsi_threshold DOUBLE PRECISION,
                btc_score DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (symbol, direction, signal_time)
            )
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_rsirecov_signal_time
            ON scanner_signals_rsi_recovery_long (signal_time DESC)
        """))
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_rsirecov_active
            ON scanner_signals_rsi_recovery_long (status, signal_time)
            WHERE status = 'active'
        """))


def write_signal(engine, *, hit: dict) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO scanner_signals_rsi_recovery_long
                (symbol, direction, signal_time, status,
                 entry_price, atr14_at_entry,
                 rsi_today, rsi_yesterday, rsi_threshold, btc_score)
                VALUES (:symbol, :direction, :sig_t, 'active',
                        :ep, :atr, :rt, :ry, :rth, :bs)
                ON CONFLICT (symbol, direction, signal_time) DO NOTHING
            """), {
                "symbol": hit["symbol"], "direction": hit["direction"],
                "sig_t": hit["signal_time"],
                "ep": hit["entry_price"], "atr": hit["atr14_at_entry"],
                "rt": hit["rsi_today"], "ry": hit["rsi_yesterday"],
                "rth": hit["rsi_threshold"],
                "bs": hit["btc_score"] if np.isfinite(hit["btc_score"]) else None,
            })
        return True
    except Exception as e:
        logger.error(f"Failed to insert signal for {hit.get('symbol')}: {e}")
        return False


def main() -> None:
    logger.info("=" * 60)
    logger.info("LIVE SCANNER — rsi_recovery_long (Week 8 v8 BALANCED, locked)")
    logger.info("=" * 60)
    logger.info(
        f"Detector: daily RSI(14) cross-up through {RSI_OVERSOLD_THRESHOLD} "
        f"(was oversold yesterday, recovered today), "
        f"cooldown {COOLDOWN_SECONDS // 3600}h"
    )

    engine = get_db_engine()
    ensure_table(engine)
    logger.info("scanner_signals_rsi_recovery_long table ready")

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

                hit = detect_rsi_recovery_long(df, btc_score_cached)
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
                        f"rsi_yest={hit['rsi_yesterday']:.1f} "
                        f"rsi_today={hit['rsi_today']:.1f}"
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
