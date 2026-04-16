"""Paper Executor — dry-run order executor for Gate.io futures.

Simulates placing orders on Gate.io WITHOUT sending real requests.
- Polls scanner_signals_v2 (short) and scanner_signals_long_v2 (long) for new signals.
- For each new signal, logs the order it WOULD place (entry, size, TP, SL).
- Tracks each paper trade against live Gate.io prices; marks TP/SL/timeout exits.
- Records everything to `paper_trades` table for comparison vs backtest.

Safety:
- DRY_RUN hardcoded TRUE — no real orders under any condition.
- When ready for real, a separate `live_executor.py` will extend this.

Usage: python scripts/paper_executor.py
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

# ── HARD SAFETY ──────────────────────────────────────────────────────
DRY_RUN = True  # NEVER change to False in this file — real trading is a separate script
assert DRY_RUN, "paper_executor.py must always be DRY_RUN"

# ── Config ───────────────────────────────────────────────────────────
import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
GATEIO_BASE = "https://api.gateio.ws/api/v4"

# Portfolio sim
STARTING_EQUITY_USD = 100.0   # paper $100; adjust to match what you'd start live
POSITION_PCT = 0.10           # 10% of equity per trade (matches backtest)
LEVERAGE = 1                  # no leverage initially
FEE_RATE = 0.0006             # 0.06% taker fee per side (Gate.io futures)

# Exit params (match backtest Variant A — TP 5% / SL 5% / 192 bars = 48h)
TP_PCT = 0.05
SL_PCT = 0.05
MAX_BARS_HOLD = 192           # 48h at 15min — hard timeout

# Poll / daemon
POLL_INTERVAL_SEC = 60

# Safety guardrails (even in paper mode — trains discipline)
MAX_DAILY_LOSS_PCT = 0.05     # halt new entries if day -5%
MAX_OPEN_POSITIONS = 3

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/paper_executor.log", mode="a"),
    ],
)
logger = logging.getLogger("paper_executor")


# ── Gate.io price fetch (READ-ONLY — safe) ───────────────────────────

def fetch_json(url: str, retries: int = 2):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            resp = urlopen(req, timeout=10)
            return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"API error: {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_last_price(symbol_usdt: str) -> float | None:
    """Fetch last spot price. symbol_usdt e.g. 'BTCUSDT' -> pair 'BTC_USDT'."""
    pair = symbol_usdt.replace("USDT", "_USDT") if "_" not in symbol_usdt else symbol_usdt
    data = fetch_json(f"{GATEIO_BASE}/spot/tickers?currency_pair={pair}")
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    try:
        return float(data[0].get("last", 0))
    except (ValueError, TypeError):
        return None


# ── DB schema ────────────────────────────────────────────────────────

def ensure_signal_tables(engine):
    """Create signal tables if the scanners haven't run yet, so JOINs don't fail."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals_v2 (
                id SERIAL PRIMARY KEY, symbol VARCHAR(20) NOT NULL,
                direction SMALLINT DEFAULT -1, ml_prob DOUBLE PRECISION,
                vol_ratio DOUBLE PRECISION, price_drop DOUBLE PRECISION,
                bounce_pct DOUBLE PRECISION, signal_time TIMESTAMPTZ NOT NULL,
                status VARCHAR(10) DEFAULT 'active', entry_price DOUBLE PRECISION,
                pnl_pct DOUBLE PRECISION, exit_reason VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals_long_v2 (
                id SERIAL PRIMARY KEY, symbol VARCHAR(20) NOT NULL,
                direction SMALLINT DEFAULT 1, ml_prob DOUBLE PRECISION,
                vol_ratio DOUBLE PRECISION, price_rise DOUBLE PRECISION,
                pullback_pct DOUBLE PRECISION, signal_time TIMESTAMPTZ NOT NULL,
                status VARCHAR(10) DEFAULT 'active', entry_price DOUBLE PRECISION,
                pnl_pct DOUBLE PRECISION, exit_reason VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))


def ensure_paper_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id SERIAL PRIMARY KEY,
                source_table VARCHAR(40) NOT NULL,
                source_id INTEGER NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                direction VARCHAR(5) NOT NULL,  -- 'long' or 'short'
                ml_prob DOUBLE PRECISION,
                entry_time TIMESTAMPTZ NOT NULL,
                entry_price DOUBLE PRECISION NOT NULL,
                position_usd DOUBLE PRECISION NOT NULL,
                tp_price DOUBLE PRECISION NOT NULL,
                sl_price DOUBLE PRECISION NOT NULL,
                status VARCHAR(15) DEFAULT 'open',  -- open, won, lost, timeout, cancelled
                exit_time TIMESTAMPTZ,
                exit_price DOUBLE PRECISION,
                exit_reason VARCHAR(20),
                pnl_pct DOUBLE PRECISION,
                pnl_usd DOUBLE PRECISION,
                fees_usd DOUBLE PRECISION,
                equity_after DOUBLE PRECISION,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE (source_table, source_id)
            )
        """))


# ── Paper trade logic ────────────────────────────────────────────────

@dataclass
class Signal:
    source_table: str
    source_id: int
    symbol: str
    direction: str  # 'long' or 'short'
    ml_prob: float
    entry_price: float
    signal_time: datetime


def fetch_new_signals(engine) -> list[Signal]:
    """Fetch signals from both tables not yet in paper_trades."""
    signals: list[Signal] = []
    with engine.begin() as conn:
        # Long signals
        rows = conn.execute(text("""
            SELECT s.id, s.symbol, s.ml_prob, s.entry_price, s.signal_time
            FROM scanner_signals_long_v2 s
            LEFT JOIN paper_trades p
              ON p.source_table = 'scanner_signals_long_v2' AND p.source_id = s.id
            WHERE p.id IS NULL AND s.status = 'active'
            ORDER BY s.signal_time
        """)).fetchall()
        for r in rows:
            signals.append(Signal(
                source_table="scanner_signals_long_v2",
                source_id=r[0], symbol=r[1], direction="long",
                ml_prob=float(r[2] or 0), entry_price=float(r[3]),
                signal_time=r[4],
            ))
        # Short signals
        rows = conn.execute(text("""
            SELECT s.id, s.symbol, s.ml_prob, s.entry_price, s.signal_time
            FROM scanner_signals_v2 s
            LEFT JOIN paper_trades p
              ON p.source_table = 'scanner_signals_v2' AND p.source_id = s.id
            WHERE p.id IS NULL AND s.status = 'active'
            ORDER BY s.signal_time
        """)).fetchall()
        for r in rows:
            signals.append(Signal(
                source_table="scanner_signals_v2",
                source_id=r[0], symbol=r[1], direction="short",
                ml_prob=float(r[2] or 0), entry_price=float(r[3]),
                signal_time=r[4],
            ))
    return signals


def current_equity(engine) -> float:
    """Compute current equity: starting + sum of closed pnl."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COALESCE(SUM(pnl_usd), 0) FROM paper_trades
            WHERE status IN ('won', 'lost', 'timeout')
        """)).fetchone()
    realized = float(row[0]) if row else 0.0
    return STARTING_EQUITY_USD + realized


def open_positions(engine) -> int:
    with engine.begin() as conn:
        row = conn.execute(text("SELECT COUNT(*) FROM paper_trades WHERE status = 'open'")).fetchone()
    return int(row[0]) if row else 0


def today_pnl_pct(engine) -> float:
    """Realized PnL today as fraction of starting equity (for kill-switch)."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COALESCE(SUM(pnl_usd), 0) FROM paper_trades
            WHERE exit_time >= CURRENT_DATE AND status IN ('won', 'lost', 'timeout')
        """)).fetchone()
    realized = float(row[0]) if row else 0.0
    return realized / STARTING_EQUITY_USD


def open_paper_trade(engine, sig: Signal):
    """Simulate placing a market entry + TP + SL order set on Gate.io."""
    equity = current_equity(engine)
    position_usd = equity * POSITION_PCT * LEVERAGE

    if sig.direction == "long":
        tp_price = sig.entry_price * (1 + TP_PCT)
        sl_price = sig.entry_price * (1 - SL_PCT)
    else:
        tp_price = sig.entry_price * (1 - TP_PCT)
        sl_price = sig.entry_price * (1 + SL_PCT)

    # This is what WOULD be sent to Gate.io. Log it clearly.
    logger.info("=" * 60)
    logger.info(f"[DRY-RUN ORDER] {sig.direction.upper()} {sig.symbol}")
    logger.info(f"  Entry:    ${sig.entry_price:.6f} (market)")
    logger.info(f"  Size:     ${position_usd:.2f} @ {LEVERAGE}x leverage")
    logger.info(f"  TP:       ${tp_price:.6f}  (+{TP_PCT*100:.1f}%)")
    logger.info(f"  SL:       ${sl_price:.6f}  (-{SL_PCT*100:.1f}%)")
    logger.info(f"  ML:       {sig.ml_prob:.2f}")
    logger.info(f"  Equity:   ${equity:.2f}  (open pos: {open_positions(engine)+1}/{MAX_OPEN_POSITIONS})")
    logger.info("=" * 60)

    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_trades
            (source_table, source_id, symbol, direction, ml_prob, entry_time,
             entry_price, position_usd, tp_price, sl_price, status)
            VALUES (:st, :sid, :sym, :dir, :ml, :et, :ep, :pos, :tp, :sl, 'open')
            ON CONFLICT (source_table, source_id) DO NOTHING
        """), {
            "st": sig.source_table, "sid": sig.source_id, "sym": sig.symbol,
            "dir": sig.direction, "ml": sig.ml_prob,
            "et": sig.signal_time, "ep": sig.entry_price,
            "pos": position_usd, "tp": tp_price, "sl": sl_price,
        })


def check_and_close_trades(engine):
    """For each open trade, fetch current price; mark exit if TP/SL/timeout hit."""
    with engine.begin() as conn:
        trades = conn.execute(text("""
            SELECT id, symbol, direction, entry_price, entry_time,
                   position_usd, tp_price, sl_price
            FROM paper_trades WHERE status = 'open'
        """)).fetchall()

    for t in trades:
        trade_id, symbol, direction, entry, entry_time, pos_usd, tp, sl = t
        price = fetch_last_price(symbol)
        if price is None:
            continue

        # Timeout check — 192 bars * 15 min = 48h
        age_sec = (datetime.now(timezone.utc) - entry_time).total_seconds()
        age_bars = age_sec / (15 * 60)

        exit_price = None
        exit_reason = None

        if direction == "long":
            if price >= tp:
                exit_price, exit_reason = tp, "take_profit"
            elif price <= sl:
                exit_price, exit_reason = sl, "stop_loss"
        else:  # short
            if price <= tp:
                exit_price, exit_reason = tp, "take_profit"
            elif price >= sl:
                exit_price, exit_reason = sl, "stop_loss"

        if exit_reason is None and age_bars >= MAX_BARS_HOLD:
            exit_price, exit_reason = price, "timeout"

        if exit_reason is None:
            continue

        # Compute PnL
        if direction == "long":
            pnl_pct = (exit_price - entry) / entry
        else:
            pnl_pct = (entry - exit_price) / entry

        fees = pos_usd * FEE_RATE * 2  # entry + exit
        pnl_usd = pos_usd * pnl_pct - fees
        equity_after = current_equity(engine) + pnl_usd

        status = "won" if pnl_usd > 0 else ("lost" if exit_reason == "stop_loss" else "timeout")

        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE paper_trades
                SET status = :st, exit_time = :et, exit_price = :ep,
                    exit_reason = :er, pnl_pct = :pp, pnl_usd = :pu,
                    fees_usd = :f, equity_after = :eq
                WHERE id = :id
            """), {
                "st": status, "et": datetime.now(timezone.utc),
                "ep": exit_price, "er": exit_reason,
                "pp": pnl_pct, "pu": pnl_usd, "f": fees, "eq": equity_after,
                "id": trade_id,
            })

        emoji = "W" if status == "won" else "L"
        logger.info(
            f"[DRY-RUN EXIT {emoji}] {direction.upper()} {symbol} "
            f"{exit_reason} | entry=${entry:.6f} exit=${exit_price:.6f} "
            f"pnl={pnl_pct*100:+.2f}% (${pnl_usd:+.2f}) | equity=${equity_after:.2f}"
        )


# ── Main loop ────────────────────────────────────────────────────────

def main():
    engine = create_engine(DB_URL)
    ensure_signal_tables(engine)
    ensure_paper_table(engine)

    logger.info("=" * 60)
    logger.info("PAPER EXECUTOR (DRY-RUN) — no real orders will be placed")
    logger.info(f"Starting equity: ${STARTING_EQUITY_USD:.2f}")
    logger.info(f"Position size:   {POSITION_PCT*100:.0f}% @ {LEVERAGE}x")
    logger.info(f"TP/SL:           +{TP_PCT*100:.0f}% / -{SL_PCT*100:.0f}% (max {MAX_BARS_HOLD} bars = 48h)")
    logger.info(f"Kill-switch:     halt new entries if day PnL < -{MAX_DAILY_LOSS_PCT*100:.0f}%")
    logger.info(f"Max positions:   {MAX_OPEN_POSITIONS}")
    logger.info(f"Poll interval:   {POLL_INTERVAL_SEC}s")
    logger.info("=" * 60)

    cycle = 0
    while True:
        try:
            cycle += 1
            # 1. Check and close existing open trades
            check_and_close_trades(engine)

            # 2. Fetch new signals
            new_sigs = fetch_new_signals(engine)

            # Apply safety gates
            if new_sigs:
                halted = today_pnl_pct(engine) <= -MAX_DAILY_LOSS_PCT
                if halted:
                    logger.warning(f"KILL-SWITCH: daily PnL <= -{MAX_DAILY_LOSS_PCT*100:.0f}%, "
                                   f"dropping {len(new_sigs)} new signals")
                    new_sigs = []

            for sig in new_sigs:
                if open_positions(engine) >= MAX_OPEN_POSITIONS:
                    logger.warning(f"MAX POSITIONS ({MAX_OPEN_POSITIONS}) reached — "
                                   f"skipping {sig.direction.upper()} {sig.symbol}")
                    continue
                open_paper_trade(engine, sig)

            # Periodic status
            if cycle % 10 == 0:
                eq = current_equity(engine)
                op = open_positions(engine)
                logger.info(f"Cycle {cycle}: equity=${eq:.2f} open={op} "
                            f"ret={(eq/STARTING_EQUITY_USD-1)*100:+.1f}%")

            time.sleep(POLL_INTERVAL_SEC)

        except KeyboardInterrupt:
            logger.info("Paper executor stopped by user.")
            break
        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
            time.sleep(POLL_INTERVAL_SEC)

    engine.dispose()


if __name__ == "__main__":
    main()
