"""Paper Executor — dry-run order executor for Gate.io futures.

Simulates placing orders on Gate.io WITHOUT sending real requests.
- Polls scanner_signals_v2 (short) and scanner_signals_long_v2 (long) for new signals.
- For each new signal, logs the order it WOULD place (entry, size, SL).
- Tracks each paper trade against live Gate.io prices; marks trailing stop/SL/timeout exits.
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

from src import db_adapters  # noqa: F401  # registers numpy→psycopg2 adapters

# ── HARD SAFETY ──────────────────────────────────────────────────────
DRY_RUN = True  # NEVER change to False in this file — real trading is a separate script
assert DRY_RUN, "paper_executor.py must always be DRY_RUN"

# ── Config ───────────────────────────────────────────────────────────
import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
GATEIO_BASE = "https://api.gateio.ws/api/v4"

# Multi-threshold paper accounts — each runs independently
THRESHOLDS = [0.80, 0.75, 0.70, 0.65, 0.60]

# Strategy key for the existing v2 accounts (failed-bounce short scanner).
V2_STRATEGY = "v2_failed_bounce"

# Bigmover paper accounts (added 2026-04-23). 6 separate portfolios, one per
# (signal_type, direction). Each reads from scanner_signals_bigmover. Threshold
# is stored as sentinel 1.0 — bigmover has no ML gate.
#   strategy_key matches scanner_signals_bigmover.signal_type + direction.
BIGMOVER_ACCOUNTS: list[dict] = [
    {"strategy": "bigmover_baseline_short",
     "signal_type": "baseline", "direction": "short"},
    {"strategy": "bigmover_baseline_long",
     "signal_type": "baseline", "direction": "long"},
    {"strategy": "bigmover_price_accel_atr_short",
     "signal_type": "price_accel_atr", "direction": "short"},
    {"strategy": "bigmover_price_accel_atr_long",
     "signal_type": "price_accel_atr", "direction": "long"},
    {"strategy": "bigmover_multi_bar_confirm_short",
     "signal_type": "multi_bar_confirm", "direction": "short"},
    {"strategy": "bigmover_multi_bar_confirm_long",
     "signal_type": "multi_bar_confirm", "direction": "long"},
]
BIGMOVER_SENTINEL_THRESHOLD = 1.0  # no ML gate; distinguishes rows from v2 ones

# Portfolio sim
STARTING_EQUITY_USD = 100.0   # paper $100; adjust to match what you'd start live
POSITION_PCT = 0.10           # 10% of equity per trade (matches backtest)
LEVERAGE = 1                  # no leverage initially
FEE_RATE = 0.0006             # 0.06% taker fee per side (Gate.io futures)

# Exit params — trailing stop (backed by learn/22042026 analysis)
SL_PCT = 0.05                 # hard stop-loss at -5%
TRAIL_ACTIVATION = 0.02       # start trailing after +2% unrealized
TRAIL_PCT = 0.03              # exit when price gives back 3% from peak
MAX_BARS_HOLD = 192           # 48h timeout unchanged

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


def fetch_last_price(symbol_usdt: str, venue: str = "spot") -> float | None:
    """Fetch last price for a symbol. venue='spot' (long) or 'futures' (short).

    Long trades mark against spot — that's where a 1x long would actually clear.
    Short trades mark against the USDT perp — short is only tradeable on futures,
    and the perp's last price is what determines PnL / liquidation on a real short.
    Marking a short against spot would introduce spot-perp basis drift (0.05-0.3%
    on altcoins) between paper and any eventual live execution.
    """
    pair = symbol_usdt.replace("USDT", "_USDT") if "_" not in symbol_usdt else symbol_usdt
    if venue == "futures":
        data = fetch_json(f"{GATEIO_BASE}/futures/usdt/tickers?contract={pair}")
    else:
        data = fetch_json(f"{GATEIO_BASE}/spot/tickers?currency_pair={pair}")
    if not data or not isinstance(data, list) or len(data) == 0:
        return None
    try:
        return float(data[0].get("last", 0))
    except (ValueError, TypeError):
        return None


def venue_for_direction(direction: str) -> str:
    """Long can execute on spot or futures; we use spot (simpler, no funding).
    Short requires futures — spot has no short mechanic.
    """
    return "futures" if direction == "short" else "spot"


# ── DB schema ────────────────────────────────────────────────────────

def ensure_bigmover_signal_table(engine):
    """Create scanner_signals_bigmover table (mirror of live_scanner_bigmover.py)
    so the executor's join doesn't fail if the scanner hasn't run yet.
    """
    with engine.begin() as conn:
        conn.execute(text("""
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
        """))


def ensure_signal_tables(engine):
    """Create signal tables if the scanners haven't run yet, so JOINs don't fail."""
    ensure_bigmover_signal_table(engine)
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


def ensure_paper_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS paper_trades (
                id SERIAL PRIMARY KEY,
                source_table VARCHAR(40) NOT NULL,
                source_id INTEGER NOT NULL,
                threshold DOUBLE PRECISION NOT NULL DEFAULT 0.80,
                symbol VARCHAR(20) NOT NULL,
                direction VARCHAR(5) NOT NULL,
                ml_prob DOUBLE PRECISION,
                entry_time TIMESTAMPTZ NOT NULL,
                entry_price DOUBLE PRECISION NOT NULL,
                position_usd DOUBLE PRECISION NOT NULL,
                tp_price DOUBLE PRECISION NOT NULL,
                sl_price DOUBLE PRECISION NOT NULL,
                status VARCHAR(15) DEFAULT 'open',
                exit_time TIMESTAMPTZ,
                exit_price DOUBLE PRECISION,
                exit_reason VARCHAR(20),
                pnl_pct DOUBLE PRECISION,
                pnl_usd DOUBLE PRECISION,
                fees_usd DOUBLE PRECISION,
                equity_after DOUBLE PRECISION,
                regime_allowed BOOLEAN,
                venue VARCHAR(10),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))
        conn.execute(text("""
            ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS threshold
            DOUBLE PRECISION NOT NULL DEFAULT 0.80
        """))
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS regime_allowed BOOLEAN"
        ))
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS venue VARCHAR(10)"
        ))
        conn.execute(text("""
            ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS peak_pnl_pct FLOAT DEFAULT 0.0
        """))
        conn.execute(text(
            "ALTER TABLE paper_trades DROP CONSTRAINT IF EXISTS paper_trades_source_table_source_id_key"
        ))
        # Bigmover: add strategy discriminator. Default existing rows to legacy v2.
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS strategy "
            f"VARCHAR(40) NOT NULL DEFAULT '{V2_STRATEGY}'"
        ))
        # Replace the (source_table, source_id, threshold) uniqueness so bigmover
        # rows (sharing threshold=1.0) can coexist when routed to different strategies.
        conn.execute(text(
            "DROP INDEX IF EXISTS uq_paper_src_threshold"
        ))
        conn.execute(text("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_paper_src_threshold_strategy
            ON paper_trades (source_table, source_id, threshold, strategy)
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
    regime_allowed: bool | None  # None for pre-migration rows
    # Bigmover-only: None for v2 signals, else one of the SIGNALS constants
    signal_type: str | None = None


def fetch_recent_signals(engine) -> list[Signal]:
    """Fetch all active signals from last 48h. Threshold filtering happens in open loop."""
    signals: list[Signal] = []
    with engine.begin() as conn:
        for tbl, direction in [
            ("scanner_signals_long_v2", "long"),
            ("scanner_signals_v2", "short"),
        ]:
            rows = conn.execute(text(f"""
                SELECT id, symbol, ml_prob, entry_price, signal_time, regime_allowed
                FROM {tbl}
                WHERE status = 'active'
                  AND signal_time >= NOW() - INTERVAL '48 hours'
                ORDER BY signal_time
            """)).fetchall()
            for r in rows:
                signals.append(Signal(
                    source_table=tbl, source_id=r[0], symbol=r[1],
                    direction=direction, ml_prob=float(r[2] or 0),
                    entry_price=float(r[3]), signal_time=r[4],
                    regime_allowed=r[5],
                ))
        # Volume-surge signals (no ML model — use vol_ratio/10 as proxy)
        surge_rows = conn.execute(text("""
            SELECT id, symbol, direction, COALESCE(vol_ratio, 0) as ml_prob,
                   signal_time, entry_price, COALESCE(regime_allowed, true)
            FROM scanner_signals_surge_v1
            WHERE signal_time > NOW() - INTERVAL '48 hours'
              AND status = 'active'
            ORDER BY signal_time DESC
        """)).fetchall()

        for r in surge_rows:
            signals.append(Signal(
                source_table="scanner_signals_surge_v1",
                source_id=r[0], symbol=r[1], direction=r[2],
                ml_prob=min(r[3] / 10, 1.0),
                signal_time=r[4], entry_price=float(r[5]),
                regime_allowed=r[6],
            ))

        # Bigmover signals (all 3 signal types x 2 directions, no ML gate)
        bigmover_rows = conn.execute(text("""
            SELECT id, symbol, signal_type, direction, signal_time,
                   entry_price, regime_allowed
            FROM scanner_signals_bigmover
            WHERE status = 'active'
              AND signal_time >= NOW() - INTERVAL '48 hours'
            ORDER BY signal_time
        """)).fetchall()
        for r in bigmover_rows:
            signals.append(Signal(
                source_table="scanner_signals_bigmover",
                source_id=r[0], symbol=r[1], direction=r[3],
                ml_prob=1.0,  # sentinel — no ML gate for bigmover
                signal_time=r[4], entry_price=float(r[5]),
                regime_allowed=r[6],
                signal_type=r[2],
            ))

    return signals


def current_equity(engine, threshold: float, strategy: str = V2_STRATEGY) -> float:
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COALESCE(SUM(pnl_usd), 0) FROM paper_trades
            WHERE threshold = :th AND strategy = :strat
              AND status IN ('won', 'lost', 'timeout')
        """), {"th": threshold, "strat": strategy}).fetchone()
    return STARTING_EQUITY_USD + (float(row[0]) if row else 0.0)


def open_positions(engine, threshold: float, strategy: str = V2_STRATEGY) -> int:
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COUNT(*) FROM paper_trades
            WHERE threshold = :th AND strategy = :strat AND status = 'open'
        """), {"th": threshold, "strat": strategy}).fetchone()
    return int(row[0]) if row else 0


def today_pnl_pct(engine, threshold: float, strategy: str = V2_STRATEGY) -> float:
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COALESCE(SUM(pnl_usd), 0) FROM paper_trades
            WHERE threshold = :th AND strategy = :strat
              AND exit_time >= CURRENT_DATE
              AND status IN ('won', 'lost', 'timeout')
        """), {"th": threshold, "strat": strategy}).fetchone()
    return (float(row[0]) if row else 0.0) / STARTING_EQUITY_USD


def open_paper_trade(
    engine, sig: Signal, threshold: float, strategy: str = V2_STRATEGY
):
    # Short-only mode for v2 accounts — skip longs there until long edge proven.
    # Bigmover accounts explicitly enable longs (direction is part of the strategy key).
    if strategy == V2_STRATEGY and sig.direction == "long":
        return

    equity = current_equity(engine, threshold, strategy)
    position_usd = equity * POSITION_PCT * LEVERAGE

    if sig.direction == "short":
        tp_price = 0
        sl_price = sig.entry_price * (1 + SL_PCT)
    else:
        tp_price = 0
        sl_price = sig.entry_price * (1 - SL_PCT)

    logger.info(
        f"[DRY-RUN {strategy}@{threshold}] {sig.direction.upper()} {sig.symbol} "
        f"ml={sig.ml_prob:.2f} entry=${sig.entry_price:.6f} eq=${equity:.2f}"
    )

    venue = venue_for_direction(sig.direction)
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_trades
            (source_table, source_id, threshold, strategy, symbol, direction, ml_prob,
             entry_time, entry_price, position_usd, tp_price, sl_price, status,
             regime_allowed, venue)
            VALUES (:st, :sid, :th, :strat, :sym, :dir, :ml, :et, :ep, :pos, :tp, :sl,
                    'open', :regime, :venue)
            ON CONFLICT (source_table, source_id, threshold, strategy) DO NOTHING
        """), {
            "st": sig.source_table, "sid": sig.source_id, "th": threshold,
            "strat": strategy, "sym": sig.symbol, "dir": sig.direction,
            "ml": sig.ml_prob, "et": sig.signal_time, "ep": sig.entry_price,
            "pos": position_usd, "tp": tp_price, "sl": sl_price,
            "regime": sig.regime_allowed, "venue": venue,
        })


def check_and_close_trades(engine):
    """For each open trade across all thresholds/strategies, check trail/SL/timeout."""
    with engine.begin() as conn:
        trades = conn.execute(text("""
            SELECT id, symbol, direction, entry_price, entry_time,
                   position_usd, tp_price, sl_price, threshold, peak_pnl_pct,
                   strategy
            FROM paper_trades WHERE status = 'open'
        """)).fetchall()

    for t in trades:
        (trade_id, symbol, direction, entry, entry_time,
         pos_usd, tp, sl, threshold, peak_pnl, strategy) = t
        peak_pnl = float(peak_pnl or 0.0)
        strategy = strategy or V2_STRATEGY

        price = fetch_last_price(symbol, venue=venue_for_direction(direction))
        if price is None:
            continue

        # Current unrealized PnL
        if direction == "long":
            cur_pnl = (price - entry) / entry
        else:
            cur_pnl = (entry - price) / entry

        # Update peak PnL if new high
        if cur_pnl > peak_pnl:
            peak_pnl = cur_pnl
            with engine.begin() as conn:
                conn.execute(text(
                    "UPDATE paper_trades SET peak_pnl_pct = :peak WHERE id = :id"
                ), {"peak": peak_pnl, "id": trade_id})

        # Timeout check — 192 bars * 15 min = 48h
        age_sec = (datetime.now(timezone.utc) - entry_time).total_seconds()
        age_bars = age_sec / (15 * 60)

        exit_price = None
        exit_reason = None

        # 1. Hard stop-loss
        if direction == "long":
            if price <= sl:
                exit_price, exit_reason = sl, "stop_loss"
        else:  # short
            if price >= sl:
                exit_price, exit_reason = sl, "stop_loss"

        # 2. Trailing stop: activate after +TRAIL_ACTIVATION%, exit on TRAIL_PCT% drawdown
        if exit_reason is None and peak_pnl >= TRAIL_ACTIVATION:
            drawdown_from_peak = peak_pnl - cur_pnl
            if drawdown_from_peak >= TRAIL_PCT:
                exit_price, exit_reason = price, "trail_stop"

        # 3. Timeout
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
        equity_after = current_equity(engine, threshold, strategy) + pnl_usd

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
            f"pnl={pnl_pct*100:+.2f}% peak={peak_pnl*100:+.2f}% "
            f"(${pnl_usd:+.2f}) | equity=${equity_after:.2f}"
        )


# ── Main loop ────────────────────────────────────────────────────────

def main():
    engine = create_engine(DB_URL)
    ensure_signal_tables(engine)
    ensure_paper_table(engine)

    logger.info("=" * 60)
    logger.info("PAPER EXECUTOR (DRY-RUN) — v2 multi-threshold + 6 bigmover accounts")
    logger.info(f"v2 thresholds:   {THRESHOLDS} (shorts only)")
    logger.info(f"Bigmover:        {len(BIGMOVER_ACCOUNTS)} accounts ({', '.join(a['strategy'] for a in BIGMOVER_ACCOUNTS)})")
    logger.info(f"Starting equity: ${STARTING_EQUITY_USD:.2f} per account")
    logger.info(f"Position size:   {POSITION_PCT*100:.0f}% @ {LEVERAGE}x")
    logger.info(f"Exit:            trail {TRAIL_PCT*100:.0f}% (activate >{TRAIL_ACTIVATION*100:.0f}%) | SL -{SL_PCT*100:.0f}% | timeout {MAX_BARS_HOLD} bars")
    logger.info(f"Kill-switch:     halt new entries if day PnL < -{MAX_DAILY_LOSS_PCT*100:.0f}%")
    logger.info(f"Max positions:   {MAX_OPEN_POSITIONS} per account")
    logger.info(f"Poll interval:   {POLL_INTERVAL_SEC}s")
    logger.info("=" * 60)

    cycle = 0
    while True:
        try:
            cycle += 1
            check_and_close_trades(engine)

            new_sigs = fetch_recent_signals(engine)
            v2_sigs = [s for s in new_sigs if s.signal_type is None]
            bm_sigs = [s for s in new_sigs if s.signal_type is not None]

            # v2 accounts (5 ML thresholds)
            for th in THRESHOLDS:
                halted = today_pnl_pct(engine, th, V2_STRATEGY) <= -MAX_DAILY_LOSS_PCT
                if halted:
                    continue
                for sig in v2_sigs:
                    if sig.ml_prob < th:
                        continue
                    if open_positions(engine, th, V2_STRATEGY) >= MAX_OPEN_POSITIONS:
                        break
                    open_paper_trade(engine, sig, th, V2_STRATEGY)

            # Bigmover accounts (6 separate portfolios, one per signal x direction)
            for acct in BIGMOVER_ACCOUNTS:
                halted = today_pnl_pct(
                    engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                ) <= -MAX_DAILY_LOSS_PCT
                if halted:
                    continue
                for sig in bm_sigs:
                    if sig.signal_type != acct["signal_type"]:
                        continue
                    if sig.direction != acct["direction"]:
                        continue
                    if open_positions(
                        engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    ) >= MAX_OPEN_POSITIONS:
                        break
                    open_paper_trade(
                        engine, sig, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    )

            if cycle % 10 == 0:
                parts = []
                for th in THRESHOLDS:
                    eq = current_equity(engine, th, V2_STRATEGY)
                    op = open_positions(engine, th, V2_STRATEGY)
                    parts.append(f"v2@{th}=${eq:.0f}({op})")
                for acct in BIGMOVER_ACCOUNTS:
                    eq = current_equity(
                        engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    )
                    op = open_positions(
                        engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    )
                    short_name = acct["strategy"].replace("bigmover_", "bm_")
                    parts.append(f"{short_name}=${eq:.0f}({op})")
                logger.info(f"Cycle {cycle}: {' | '.join(parts)}")

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
