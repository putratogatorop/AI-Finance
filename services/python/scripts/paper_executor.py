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
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from src import db_adapters  # noqa: F401  # registers numpy→psycopg2 adapters
from src.ml.v8 import sizing as v8_sizing  # v8 phase-switch ATR sizing module

# ── HARD SAFETY ──────────────────────────────────────────────────────
DRY_RUN = True  # NEVER change to False in this file — real trading is a separate script
assert DRY_RUN, "paper_executor.py must always be DRY_RUN"

# ── Config ───────────────────────────────────────────────────────────

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
GATEIO_BASE = "https://api.gateio.ws/api/v4"

# Multi-threshold paper accounts — each runs independently
THRESHOLDS = [0.80, 0.75, 0.70, 0.65, 0.60]

# All 5 v2 threshold accounts disabled 2026-04-24. Four-agent audit found:
#   1. SEVERE survivorship bias — live scanner fetches ALL Gate.io futures
#      (including 5x leveraged tokens like FARTCOIN5L) while the backtest
#      universe was ~100 coins in local CSV storage. Model trades coins it
#      was never trained on.
#   2. P-hack on threshold selection — PF 5.05 is best-of-6 thresholds
#      swept on the same walk-forward OOS used for evaluation. No Bonferroni.
#   3. Exit logic mismatch — backtest used fixed TP (+5% or +10%); paper
#      executor runs trailing-stop with tp_price=0. The 5/5 winning streak
#      is measuring a strategy no backtest ever characterized.
# See docs/research-journal/v2ml-audit-2026-04-24.md for the full report.
# Flip V2_ENABLED back to True only after fixing (1) universe filter,
# (2) proper train/test/holdout split with Bonferroni correction, and
# (3) backtesting the actual trail-stop exit the live executor uses.
V2_ENABLED = False

# Strategy key for the existing v2 accounts (failed-bounce short scanner).
V2_STRATEGY = "v2_failed_bounce"

# Bigmover-specific sizing overrides (derived from 3y sizing-sweep backtest
# commit 195388b9: 15%/5x/SL7% dominates 10%/5x/SL5% on PF / PF_p5 / Calmar).
BIGMOVER_POSITION_PCT = 0.15
BIGMOVER_LEVERAGE = 5
BIGMOVER_SL_PCT = 0.07

# Notional cap for bigmover compounding — prevents live equity from compounding
# into astronomical notionals. Gate.io per-position liquidity for top-100 alts
# tops out around $500k-$5M depending on tier; $100k is safe and realistic for
# an initial deploy. Raise later if account grows and liquidity allows.
BIGMOVER_NOTIONAL_CAP_USD = float(
    os.environ.get("BIGMOVER_NOTIONAL_CAP_USD", "100000")
)

# Bigmover paper accounts (added 2026-04-23). 7 separate portfolios:
#   6 single-direction (one per signal_type x direction)
#   1 COMBINED (multi_bar_confirm long+short sharing a single 5-slot pool)
# Each reads from scanner_signals_bigmover. Threshold is stored as sentinel 1.0
# because bigmover has no ML gate.
#   strategy_key matches scanner_signals_bigmover.signal_type + direction.
#   direction=None on the combined account means "accept both sides".
# All 7 accounts disabled 2026-04-23: their justifying backtest had a look-ahead
# bias (see src/ml/bigmover_signals.py and the SUPERSEDED header in
# results/_BIGMOVER_RESEARCH_README.md). Leak-fixed PF is 0.84-0.93, below the
# 1.30 ship floor. Flip "enabled" to True after a clean backtest passes.
BIGMOVER_ACCOUNTS: list[dict] = [
    {"strategy": "bigmover_baseline_short",
     "signal_type": "baseline", "direction": "short", "enabled": False},
    {"strategy": "bigmover_baseline_long",
     "signal_type": "baseline", "direction": "long", "enabled": False},
    {"strategy": "bigmover_price_accel_atr_short",
     "signal_type": "price_accel_atr", "direction": "short", "enabled": False},
    {"strategy": "bigmover_price_accel_atr_long",
     "signal_type": "price_accel_atr", "direction": "long", "enabled": False},
    {"strategy": "bigmover_multi_bar_confirm_short",
     "signal_type": "multi_bar_confirm", "direction": "short", "enabled": False},
    {"strategy": "bigmover_multi_bar_confirm_long",
     "signal_type": "multi_bar_confirm", "direction": "long", "enabled": False},
    # Combined long+short, shared 5-slot pool.
    {"strategy": "bigmover_multi_bar_confirm_combined",
     "signal_type": "multi_bar_confirm", "direction": None, "enabled": False},
]
BIGMOVER_SENTINEL_THRESHOLD = 1.0  # no ML gate; distinguishes rows from v2 ones


def _is_bigmover_strategy(strategy: str) -> bool:
    return strategy.startswith("bigmover_")


def _is_d1e_strategy(strategy: str) -> bool:
    return strategy.startswith("d1e_")


# ── D1e (Phase 17 locked) — paper accounts ───────────────────────────
#
# Source: scanner_signals_d1e (written by live_scanner_d1e.py).
# Each account routes the SAME signal stream through a different
# (sizing, exit) tuple and tracks its own equity in paper_trades.
# Threshold sentinel for D1e rows — distinguishes from v2/bigmover.
D1E_SENTINEL_THRESHOLD = 2.0
D1E_CLS_THRESHOLD = 0.5010418460370951  # v4 TRAIN q50, locked at training time
D1E_POSITION_PCT = 0.05      # 5% of equity per trade (CLAUDE.md)
D1E_LEVERAGE = 5             # 5x leverage (CLAUDE.md)
D1E_MAX_POS_SCALE = 1.5      # S1/S4 caps. NOTE: gross notional with 5 concurrent
                             # × 1.5 × 5 × 0.05 = 187.5%, exceeding CLAUDE.md
                             # 125%; we honor the research config and rely on
                             # MAX_OPEN_POSITIONS to bound concurrency.
D1E_MAX_OPEN = 5             # CLAUDE.md MAX_CONCURRENT_POSITIONS
D1E_NOTIONAL_CAP_USD = float(os.environ.get("D1E_NOTIONAL_CAP_USD", "100000"))
# E1 fixed-pct exits.
D1E_E1_SL_PCT = 0.05
D1E_E1_TP_PCT = 0.15
# E2 ATR-scaled exits.
D1E_E2_ATR_SL_MULT = 2.0
D1E_E2_ATR_TP_MULT = 6.0
# Both exits share these.
D1E_TIMEOUT_BARS = 672          # 7 days at 15m
D1E_RAPID_RALLY_PCT = 0.03
D1E_RAPID_LOOKBACK_15M = 96     # 24h
# AUC kill-switch (rolling 60-trade window per account; pause new entries below).
D1E_AUC_KILL_THRESHOLD = 0.52
D1E_AUC_KILL_WINDOW = 60

# ── v8 BALANCED accounts (Week 8, 2026-04-29) ───────────────────────
# Pre-registered detector portfolio with portfolio wf_p5 = 1.486 on 3y backtest
# (run id 92373355_20260429T041931Z portfolio_v8day2). Sizing uses
# src.ml.v8.sizing — phase-switch ATR-based concurrent-aware Kelly:
#   PHASE 1 (eq < $10K): Kelly 0.50, ceil 3.0%, gross 100%, per-pos 25%
#   PHASE 2 (eq ≥ $10K): Kelly 0.25, ceil 1.5%, gross 60%, per-pos 12%
#
# First v8 detector deployed: macd_pullback_short_e2 (strongest single config
# in the v5/v8 effort: holdout PF 4.289, wf_p5 2.102, DSR 1.000).
# Other v8 detectors (macd_early_trend_short, macd_pullback_long,
# rsi_recovery_long) ship in follow-up PRs after 7-14d of paper validation.
V8_SENTINEL_THRESHOLD = 3.0  # distinguishes v8 from v2 (0.6-0.8), bigmover (1.0), d1e (2.0)
V8_E1_SL_PCT = 0.05
V8_E1_TP_PCT = 0.15
V8_E2_ATR_SL_MULT = 2.0
V8_E2_ATR_TP_MULT = 6.0
V8_TIMEOUT_BARS = 672  # 7 days at 15m
V8_STARTING_EQUITY_USD = float(os.environ.get("V8_STARTING_EQUITY_USD", "200"))

V8_ACCOUNTS: list[dict] = [
    # macd_pullback_short_e2 — Week 8 PRIMARY SHORT (wf_p5 2.10, holdout PF 4.29)
    {"strategy": "v8_macd_pullback_short_e2",
     "signal_table": "scanner_signals_macd_pullback_short",
     "direction": "short", "exit": "e2", "enabled": True},
    # macd_early_trend_short_e2 — FRESH bear only, ≤10d since bear-flip (wf_p5 2.05)
    {"strategy": "v8_macd_early_trend_short_e2",
     "signal_table": "scanner_signals_macd_early_trend_short",
     "direction": "short", "exit": "e2", "enabled": True},
    # macd_pullback_long_e2 — symmetric long mirror, daily MACD bull + 4h cross-up (wf_p5 1.10)
    {"strategy": "v8_macd_pullback_long_e2",
     "signal_table": "scanner_signals_macd_pullback_long",
     "direction": "long", "exit": "e2", "enabled": True},
    # rsi_recovery_long_e2 — daily RSI(14) oversold recovery, de-correlated (wf_p5 1.81)
    {"strategy": "v8_rsi_recovery_long_e2",
     "signal_table": "scanner_signals_rsi_recovery_long",
     "direction": "long", "exit": "e2", "enabled": True},
]

# Map each v8 signal table to its strategy_group for fetch_recent_signals.
V8_SCANNER_TABLES = [
    {"table": "scanner_signals_macd_pullback_short", "strategy_group": "v8_macd_pullback_short"},
    {"table": "scanner_signals_macd_early_trend_short", "strategy_group": "v8_macd_early_trend_short"},  # noqa: E501
    {"table": "scanner_signals_macd_pullback_long", "strategy_group": "v8_macd_pullback_long"},
    {"table": "scanner_signals_rsi_recovery_long", "strategy_group": "v8_rsi_recovery_long"},
]


D1E_ACCOUNTS: list[dict] = [
    # All 4 enabled at deploy. Disable individually after observation if any
    # account violates kill-switch or PF < 1.0 over a rolling 60-trade window.
    {"strategy": "d1e_s1_e1", "sizing": "s1", "exit": "e1", "enabled": True},
    {"strategy": "d1e_s1_e2", "sizing": "s1", "exit": "e2", "enabled": True},
    {"strategy": "d1e_s4_e1", "sizing": "s4", "exit": "e1", "enabled": True},
    {"strategy": "d1e_s4_e2", "sizing": "s4", "exit": "e2", "enabled": True},
]


def _sizing_s1(btc_score: float, cls_score: float) -> float:
    """Phase 17 S1: linear sign-locked Candidate-B BTC trend score sizing.
    Returns 0 when btc_score >= 0 (no SHORT entries against an up-trending BTC).
    """
    if not (btc_score is not None and btc_score == btc_score):  # nan-safe
        return 0.0
    if btc_score >= 0:
        return 0.0
    return min(D1E_MAX_POS_SCALE * abs(float(btc_score)), D1E_MAX_POS_SCALE)


def _sizing_s4(btc_score: float, cls_score: float) -> float:
    """Phase 17 S4: classifier-confidence-weighted sizing.
    Zeroes out low-confidence trades; max scale at cls_score == 1.0.
    """
    if not (cls_score is not None and cls_score == cls_score):
        return 0.0
    return max(0.0, min(3.0 * (float(cls_score) - 0.5), D1E_MAX_POS_SCALE))


_SIZING_FNS = {"s1": _sizing_s1, "s4": _sizing_s4}

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
            "ALTER TABLE paper_trades DROP CONSTRAINT IF EXISTS "
            "paper_trades_source_table_source_id_key"
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
        # D1e per-trade context (added 2026-04-26 with Phase 17 deploy).
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS atr14_at_entry DOUBLE PRECISION"
        ))
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS btc_score DOUBLE PRECISION"
        ))
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS cls_score DOUBLE PRECISION"
        ))
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS pos_scale DOUBLE PRECISION"
        ))
        conn.execute(text(
            "ALTER TABLE paper_trades ADD COLUMN IF NOT EXISTS exit_kind VARCHAR(8)"
        ))


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
    # D1e-only (None for v2/bigmover signals)
    atr14_at_entry: float | None = None
    btc_score: float | None = None
    cls_score: float | None = None
    cls_kept: bool | None = None
    # v8-only — strategy_group discriminator. None for v2/bigmover/d1e.
    strategy_group: str | None = None


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

        # D1e signals (Phase 17, SHORT-only) — table may not exist if scanner
        # hasn't deployed yet; skip silently in that case.
        try:
            d1e_rows = conn.execute(text("""
                SELECT id, symbol, direction, signal_time, entry_price,
                       atr14_at_entry, btc_score, cls_score, cls_kept
                FROM scanner_signals_d1e
                WHERE status = 'active'
                  AND signal_time >= NOW() - INTERVAL '48 hours'
                ORDER BY signal_time
            """)).fetchall()
        except Exception:
            d1e_rows = []
        for r in d1e_rows:
            signals.append(Signal(
                source_table="scanner_signals_d1e",
                source_id=r[0], symbol=r[1], direction=r[2],
                ml_prob=float(r[7] or 0.0),  # cls_score doubles as ml_prob for logging
                signal_time=r[3], entry_price=float(r[4]),
                regime_allowed=None,
                signal_type="d1e",
                atr14_at_entry=float(r[5]) if r[5] is not None else None,
                btc_score=float(r[6]) if r[6] is not None else None,
                cls_score=float(r[7]) if r[7] is not None else None,
                cls_kept=bool(r[8]) if r[8] is not None else None,
            ))

        # v8 signals — one query per scanner table; try/except so a missing table
        # doesn't break the other scanners or d1e accounts.
        for v8_tbl in V8_SCANNER_TABLES:
            tbl_name = v8_tbl["table"]
            strat_grp = v8_tbl["strategy_group"]
            try:
                v8_rows = conn.execute(text(f"""
                    SELECT id, symbol, direction, signal_time, entry_price,
                           atr14_at_entry, btc_score
                    FROM {tbl_name}
                    WHERE status = 'active'
                      AND signal_time >= NOW() - INTERVAL '48 hours'
                    ORDER BY signal_time
                """)).fetchall()  # noqa: S608 — tbl_name is internal constant
            except Exception:
                v8_rows = []
            for r in v8_rows:
                signals.append(Signal(
                    source_table=tbl_name,
                    source_id=r[0], symbol=r[1], direction=r[2],
                    ml_prob=1.0,  # sentinel — v8 has no classifier
                    signal_time=r[3], entry_price=float(r[4]),
                    regime_allowed=None,
                    signal_type="v8",
                    strategy_group=strat_grp,
                    atr14_at_entry=float(r[5]) if r[5] is not None else None,
                    btc_score=float(r[6]) if r[6] is not None else None,
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

    if _is_bigmover_strategy(strategy):
        # Bigmover: use 15%/5x sizing (sweep winner), apply notional cap, use SL=7%.
        desired_notional = equity * BIGMOVER_POSITION_PCT * BIGMOVER_LEVERAGE
        position_usd = min(desired_notional, BIGMOVER_NOTIONAL_CAP_USD)
        sl_pct_used = BIGMOVER_SL_PCT
    else:
        # Legacy v2 sizing.
        position_usd = equity * POSITION_PCT * LEVERAGE
        sl_pct_used = SL_PCT

    if sig.direction == "short":
        tp_price = 0
        sl_price = sig.entry_price * (1 + sl_pct_used)
    else:
        tp_price = 0
        sl_price = sig.entry_price * (1 - sl_pct_used)

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


def open_d1e_paper_trade(engine, sig: Signal, account: dict):
    """Open a paper trade on a D1e signal under one S×E account spec.

    `account` carries `strategy`, `sizing` (s1|s4), `exit` (e1|e2). Sizing is
    computed from `sig.btc_score` / `sig.cls_score`; SL/TP from `account['exit']`
    using the locked Phase 17 params.
    """
    if sig.direction != "short":
        return  # D1e is SHORT-only by research design
    if sig.cls_kept is False:
        return  # paper executor honors the v4 gate
    sizing_fn = _SIZING_FNS.get(account["sizing"])
    if sizing_fn is None:
        return
    pos_scale = sizing_fn(
        sig.btc_score if sig.btc_score is not None else float("nan"),
        sig.cls_score if sig.cls_score is not None else float("nan"),
    )
    if pos_scale <= 0:
        return  # sign-locked or low-confidence — skip without recording

    equity = current_equity(engine, D1E_SENTINEL_THRESHOLD, account["strategy"])
    desired_notional = equity * D1E_POSITION_PCT * D1E_LEVERAGE * pos_scale
    position_usd = min(desired_notional, D1E_NOTIONAL_CAP_USD)
    if position_usd <= 0:
        return

    # Exit setup
    if account["exit"] == "e1":
        sl_price = sig.entry_price * (1 + D1E_E1_SL_PCT)
        tp_price = sig.entry_price * (1 - D1E_E1_TP_PCT)
    elif account["exit"] == "e2":
        atr = sig.atr14_at_entry
        if atr is None or not (atr > 0):
            return  # cannot run E2 without ATR
        sl_price = sig.entry_price + D1E_E2_ATR_SL_MULT * atr
        tp_price = max(sig.entry_price - D1E_E2_ATR_TP_MULT * atr, 0.0)
    else:
        return

    venue = venue_for_direction(sig.direction)
    logger.info(
        f"[DRY-RUN {account['strategy']}] SHORT {sig.symbol} "
        f"size={pos_scale:.2f} btc={sig.btc_score:+.3f} cls={sig.cls_score:.3f} "
        f"entry=${sig.entry_price:.6f} sl=${sl_price:.6f} tp=${tp_price:.6f} "
        f"eq=${equity:.2f}"
    )
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_trades
            (source_table, source_id, threshold, strategy, symbol, direction, ml_prob,
             entry_time, entry_price, position_usd, tp_price, sl_price, status,
             regime_allowed, venue,
             atr14_at_entry, btc_score, cls_score, pos_scale, exit_kind)
            VALUES (:st, :sid, :th, :strat, :sym, 'short', :ml, :et, :ep, :pos,
                    :tp, :sl, 'open', NULL, :venue,
                    :atr, :bs, :cs, :psc, :ek)
            ON CONFLICT (source_table, source_id, threshold, strategy) DO NOTHING
        """), {
            "st": sig.source_table, "sid": sig.source_id,
            "th": D1E_SENTINEL_THRESHOLD, "strat": account["strategy"],
            "sym": sig.symbol, "ml": sig.ml_prob, "et": sig.signal_time,
            "ep": sig.entry_price, "pos": position_usd, "tp": tp_price, "sl": sl_price,
            "venue": venue,
            "atr": sig.atr14_at_entry, "bs": sig.btc_score, "cs": sig.cls_score,
            "psc": pos_scale, "ek": account["exit"],
        })


def _v8_open_gross_notional_usd(engine, strategy: str) -> float:
    """Sum of currently-open paper notional for a v8 strategy."""
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COALESCE(SUM(position_usd), 0)
            FROM paper_trades
            WHERE strategy = :strat AND status = 'open'
        """), {"strat": strategy}).fetchone()
    return float(row[0]) if row else 0.0


def open_v8_paper_trade(engine, sig: Signal, account: dict):
    """Open a paper trade on a v8 signal using v6 phase-switch ATR sizing."""
    if sig.direction != account["direction"]:
        return
    if sig.atr14_at_entry is None or sig.atr14_at_entry <= 0:
        return
    if sig.entry_price is None or sig.entry_price <= 0:
        return

    strategy = account["strategy"]
    # Equity tracked separately per strategy; uses V8_STARTING_EQUITY_USD as the seed.
    with engine.begin() as conn:
        row = conn.execute(text("""
            SELECT COALESCE(SUM(pnl_usd), 0) FROM paper_trades
            WHERE threshold = :th AND strategy = :strat
              AND status IN ('won', 'lost', 'timeout')
        """), {"th": V8_SENTINEL_THRESHOLD, "strat": strategy}).fetchone()
    realized_pnl_usd = float(row[0]) if row else 0.0
    equity = V8_STARTING_EQUITY_USD + realized_pnl_usd

    # Load phase + recent_pnl Kelly window
    state = v8_sizing.load_state(engine, strategy)

    open_gross_usd = _v8_open_gross_notional_usd(engine, strategy)

    decision = v8_sizing.compute_size(
        current_equity=equity,
        exit_kind=account["exit"],
        atr14_at_entry=sig.atr14_at_entry,
        entry_price=sig.entry_price,
        open_gross_notional_usd=open_gross_usd,
        recent_pnl=state.recent_pnl,
        current_phase=state.phase,
    )

    if not decision.take:
        logger.info(
            f"[DRY-RUN {strategy}] SKIP {sig.symbol} reason={decision.reason} "
            f"phase={decision.phase} eq=${equity:.2f}"
        )
        return

    # Persist the new phase if it switched
    if decision.phase != state.phase:
        state.phase = decision.phase
        state.phase_switch_ts = datetime.now(UTC)
        state.phase_switch_eq = equity
        v8_sizing.save_state(engine, state)

    # Exit setup — direction-aware (long: SL below entry, TP above; short: reversed).
    if account["exit"] == "e1":
        if sig.direction == "short":
            sl_price = sig.entry_price * (1 + V8_E1_SL_PCT)
            tp_price = sig.entry_price * (1 - V8_E1_TP_PCT)
        else:
            sl_price = sig.entry_price * (1 - V8_E1_SL_PCT)
            tp_price = sig.entry_price * (1 + V8_E1_TP_PCT)
    elif account["exit"] == "e2":
        atr = sig.atr14_at_entry
        if sig.direction == "short":
            sl_price = sig.entry_price + V8_E2_ATR_SL_MULT * atr
            tp_price = max(sig.entry_price - V8_E2_ATR_TP_MULT * atr, 0.0)
        else:
            sl_price = max(sig.entry_price - V8_E2_ATR_SL_MULT * atr, 0.0)
            tp_price = sig.entry_price + V8_E2_ATR_TP_MULT * atr
    else:
        return

    venue = venue_for_direction(sig.direction)
    logger.info(
        f"[DRY-RUN {strategy}] {sig.direction.upper()} {sig.symbol} phase={decision.phase} "
        f"risk={decision.risk_pct*100:.2f}% sl_dist={decision.sl_distance_pct*100:.2f}% "
        f"notional=${decision.notional_usd:.2f} ({decision.notional_pct*100:.1f}% eq) "
        f"entry=${sig.entry_price:.6f} sl=${sl_price:.6f} tp=${tp_price:.6f} "
        f"eq=${equity:.2f}"
    )
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO paper_trades
            (source_table, source_id, threshold, strategy, symbol, direction, ml_prob,
             entry_time, entry_price, position_usd, tp_price, sl_price, status,
             regime_allowed, venue,
             atr14_at_entry, btc_score, pos_scale, exit_kind)
            VALUES (:st, :sid, :th, :strat, :sym, :dir, :ml, :et, :ep, :pos,
                    :tp, :sl, 'open', NULL, :venue,
                    :atr, :bs, :psc, :ek)
            ON CONFLICT (source_table, source_id, threshold, strategy) DO NOTHING
        """), {
            "st": sig.source_table, "sid": sig.source_id,
            "th": V8_SENTINEL_THRESHOLD, "strat": strategy,
            "sym": sig.symbol, "dir": sig.direction, "ml": sig.ml_prob,
            "et": sig.signal_time, "ep": sig.entry_price,
            "pos": decision.notional_usd, "tp": tp_price, "sl": sl_price,
            "venue": venue,
            "atr": sig.atr14_at_entry, "bs": sig.btc_score,
            "psc": decision.notional_pct,  # store notional_pct as pos_scale for v8
            "ek": account["exit"],
        })


# ── Rapid-rally cache (E1 + E2 share this exit signal) ──────────────

_rapid_rally_state: dict = {"active": False, "checked_at": 0.0}


def _btc_daily_bearish() -> bool:
    """Compute BTC daily ALL-BEARISH using EMA 9/21/50 + RSI<50 + MACD<sig + KDJ K<D.
    Mirrors phase17_sizing_compare._build_rapid_exit_panel logic. shift(1) so
    we use yesterday's daily indicator values (no lookahead at boundary).
    """
    import pandas as pd

    from src.ml.indicators import atr as _atr  # noqa: F401
    from src.ml.indicators import ema as _ema
    from src.ml.indicators import kdj as _kdj
    from src.ml.indicators import macd as _macd
    from src.ml.indicators import rsi as _rsi
    eng = create_engine(DB_URL)
    try:
        with eng.connect() as conn:
            rows = conn.execute(text("""
                SELECT timestamp, open, high, low, close
                FROM asset_prices_15m WHERE asset = 'BTCUSDT'
                ORDER BY timestamp DESC LIMIT 20000
            """)).fetchall()
    finally:
        eng.dispose()
    if not rows or len(rows) < 200 * 96:
        return False
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    daily = (df.set_index("timestamp")
               .resample("1D", label="right", closed="right")
               .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
               .dropna())
    if len(daily) < 60:
        return False
    ema_fast = _ema(daily["close"], 9).shift(1)
    ema_mid = _ema(daily["close"], 21).shift(1)
    ema_slow = _ema(daily["close"], 50).shift(1)
    ema_cond = (daily["close"].shift(1) < ema_fast) & (ema_fast < ema_mid) & (ema_mid < ema_slow)
    rsi_cond = _rsi(daily["close"], 14).shift(1) < 50.0
    md = _macd(daily["close"], 12, 26, 9)
    macd_cond = md["macd"].shift(1) < md["signal"].shift(1)
    kdj_v = _kdj(daily["high"], daily["low"], daily["close"], n=9, k_smooth=3, d_smooth=3)
    kdj_cond = kdj_v["k"].shift(1) < kdj_v["d"].shift(1)
    daily_bear = (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)
    return bool(daily_bear.iloc[-1])


def _btc_24h_return() -> float:
    """BTC 24h return from latest 15m bars."""
    import pandas as pd
    eng = create_engine(DB_URL)
    try:
        with eng.connect() as conn:
            rows = conn.execute(text("""
                SELECT timestamp, close FROM asset_prices_15m
                WHERE asset = 'BTCUSDT'
                ORDER BY timestamp DESC LIMIT 100
            """)).fetchall()
    finally:
        eng.dispose()
    if not rows or len(rows) < D1E_RAPID_LOOKBACK_15M + 1:
        return 0.0
    df = pd.DataFrame(rows, columns=["timestamp", "close"]).iloc[::-1]
    return float(df["close"].iloc[-1] / df["close"].iloc[-D1E_RAPID_LOOKBACK_15M - 1] - 1.0)


def rapid_rally_active() -> bool:
    """True iff (NOT BTC daily ALL-BEARISH) AND (BTC 24h return > +3%).
    Cached for 5 minutes; recomputed lazily on next check.
    """
    now = time.time()
    if now - _rapid_rally_state["checked_at"] > 300:
        try:
            bear = _btc_daily_bearish()
            ret24 = _btc_24h_return()
            _rapid_rally_state["active"] = (not bear) and ret24 > D1E_RAPID_RALLY_PCT
        except Exception as e:
            logger.warning(f"rapid_rally check failed: {e}")
            _rapid_rally_state["active"] = False
        _rapid_rally_state["checked_at"] = now
    return bool(_rapid_rally_state["active"])


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

        # Timeout check — D1e uses 672-bar (7d) timeout, others use 192-bar (48h).
        age_sec = (datetime.now(UTC) - entry_time).total_seconds()
        age_bars = age_sec / (15 * 60)

        exit_price = None
        exit_reason = None

        is_d1e = _is_d1e_strategy(strategy)
        is_v8 = strategy.startswith("v8_")
        timeout_bars = (
            V8_TIMEOUT_BARS if is_v8 else
            D1E_TIMEOUT_BARS if is_d1e else
            MAX_BARS_HOLD
        )

        # 1. Hard stop-loss (price-based, both v2/bigmover and D1e)
        if direction == "long":
            if price <= sl:
                exit_price, exit_reason = sl, "stop_loss"
        else:  # short
            if price >= sl:
                exit_price, exit_reason = sl, "stop_loss"

        if is_v8:
            # v8 exit: SL (price-based, above), then TP (price-based), then timeout.
            # No trail-stop, no rapid-rally — pure ATR-based exits like d1e.
            if exit_reason is None and tp is not None and tp > 0:
                if direction == "short" and price <= tp:
                    exit_price, exit_reason = tp, "take_profit"
                elif direction == "long" and price >= tp:
                    exit_price, exit_reason = tp, "take_profit"
        elif is_d1e:
            # D1e exit stack: SL (above), then TP (price-based), then rapid-rally.
            if exit_reason is None and tp is not None and tp > 0:
                if direction == "short" and price <= tp:
                    exit_price, exit_reason = tp, "take_profit"
                elif direction == "long" and price >= tp:
                    exit_price, exit_reason = tp, "take_profit"
            if exit_reason is None and rapid_rally_active():
                exit_price, exit_reason = price, "rapid_rally"
        else:
            # Legacy v2/bigmover: trail-stop after +TRAIL_ACTIVATION peak.
            if exit_reason is None and peak_pnl >= TRAIL_ACTIVATION:
                drawdown_from_peak = peak_pnl - cur_pnl
                if drawdown_from_peak >= TRAIL_PCT:
                    exit_price, exit_reason = price, "trail_stop"

        # Final: timeout (per-strategy bar count)
        if exit_reason is None and age_bars >= timeout_bars:
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
                "st": status, "et": datetime.now(UTC),
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
    v8_sizing.ensure_state_table(engine)

    logger.info("=" * 60)
    logger.info("PAPER EXECUTOR (DRY-RUN) — v2 + bigmover + D1e (Phase 17)")
    logger.info(f"v2 thresholds:   {THRESHOLDS} (V2_ENABLED={V2_ENABLED})")
    logger.info(
        f"Bigmover:        {sum(1 for a in BIGMOVER_ACCOUNTS if a.get('enabled'))}"
        f" of {len(BIGMOVER_ACCOUNTS)} accounts enabled"
    )
    enabled_d1e = [a['strategy'] for a in D1E_ACCOUNTS if a.get('enabled')]
    logger.info(f"D1e accounts:    {len(enabled_d1e)} enabled — {', '.join(enabled_d1e)}")
    enabled_v8 = [a['strategy'] for a in V8_ACCOUNTS if a.get('enabled')]
    logger.info(f"v8 accounts:     {len(enabled_v8)} enabled — {', '.join(enabled_v8)}")
    logger.info(
        f"v8 sizing:       phase-switch ATR-based "
        f"(P1: Kelly 0.50, ceil 3%, gross 100%; P2: Kelly 0.25, ceil 1.5%, gross 60%; "
        f"trigger ${v8_sizing.PHASE2_TRIGGER_USD:,.0f})"
    )
    logger.info(f"v8 starting eq: ${V8_STARTING_EQUITY_USD:.2f} per account")
    logger.info(
        f"D1e sizing:      {D1E_POSITION_PCT*100:.0f}%/{D1E_LEVERAGE}x × pos_scale "
        f"(S1: |btc|, S4: cls-0.5; cap {D1E_MAX_POS_SCALE})"
    )
    logger.info(
        f"D1e exits:       E1=fixed SL{D1E_E1_SL_PCT*100:.0f}%/TP{D1E_E1_TP_PCT*100:.0f}%, "
        f"E2=ATR ×{D1E_E2_ATR_SL_MULT}/×{D1E_E2_ATR_TP_MULT}, "
        f"timeout {D1E_TIMEOUT_BARS} bars (7d), "
        f"rapid-rally on +{D1E_RAPID_RALLY_PCT*100:.0f}% BTC 24h"
    )
    logger.info(f"D1e notional cap: ${D1E_NOTIONAL_CAP_USD:,.0f} per trade")
    logger.info(f"Starting equity: ${STARTING_EQUITY_USD:.2f} per account")
    logger.info(f"Kill-switch:     halt new entries if day PnL < -{MAX_DAILY_LOSS_PCT*100:.0f}%")
    logger.info(
        f"Max positions:   {MAX_OPEN_POSITIONS} (legacy) / {D1E_MAX_OPEN} (D1e) per account"
    )
    logger.info(f"Poll interval:   {POLL_INTERVAL_SEC}s")
    logger.info("=" * 60)

    cycle = 0
    while True:
        try:
            cycle += 1
            check_and_close_trades(engine)

            new_sigs = fetch_recent_signals(engine)
            v2_sigs = [s for s in new_sigs if s.signal_type is None]
            bm_sigs = [s for s in new_sigs
                       if s.signal_type is not None and s.signal_type not in ("d1e", "v8")]
            d1e_sigs = [s for s in new_sigs if s.signal_type == "d1e"]
            v8_sigs = [s for s in new_sigs if s.signal_type == "v8"]

            # v2 accounts (5 ML thresholds) — gated by V2_ENABLED flag.
            if V2_ENABLED:
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

            # Bigmover accounts (7 portfolios: 6 single-direction + 1 combined)
            for acct in BIGMOVER_ACCOUNTS:
                if not acct.get("enabled", True):
                    continue
                halted = today_pnl_pct(
                    engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                ) <= -MAX_DAILY_LOSS_PCT
                if halted:
                    continue
                for sig in bm_sigs:
                    if sig.signal_type != acct["signal_type"]:
                        continue
                    # direction=None on an account means "accept both sides"
                    # (used by bigmover_multi_bar_confirm_combined).
                    if acct["direction"] is not None and sig.direction != acct["direction"]:
                        continue
                    if open_positions(
                        engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    ) >= MAX_OPEN_POSITIONS:
                        break
                    open_paper_trade(
                        engine, sig, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    )

            # D1e accounts (Phase 17 locked, SHORT-only).
            for acct in D1E_ACCOUNTS:
                if not acct.get("enabled", True):
                    continue
                halted = today_pnl_pct(
                    engine, D1E_SENTINEL_THRESHOLD, acct["strategy"]
                ) <= -MAX_DAILY_LOSS_PCT
                if halted:
                    continue
                for sig in d1e_sigs:
                    if open_positions(
                        engine, D1E_SENTINEL_THRESHOLD, acct["strategy"]
                    ) >= D1E_MAX_OPEN:
                        break
                    open_d1e_paper_trade(engine, sig, acct)

            # v8 accounts (Week 8 BALANCED, phase-switch ATR sizing).
            for acct in V8_ACCOUNTS:
                if not acct.get("enabled", True):
                    continue
                halted = today_pnl_pct(
                    engine, V8_SENTINEL_THRESHOLD, acct["strategy"]
                ) <= -MAX_DAILY_LOSS_PCT
                if halted:
                    continue
                # No fixed max-open count — gross notional cap inside v6 sizing
                # already constrains concurrency. We still skip if a stale
                # position somehow stacks above 20 (sanity bound).
                if open_positions(
                    engine, V8_SENTINEL_THRESHOLD, acct["strategy"]
                ) >= 20:
                    continue
                for sig in v8_sigs:
                    # account.signal_table filter — only process signals from
                    # the account's own scanner table (one v8 account per scanner).
                    if sig.source_table != acct.get("signal_table"):
                        continue
                    try:
                        open_v8_paper_trade(engine, sig, acct)
                    except Exception as e:
                        logger.exception(
                            f"v8 open_paper_trade error on {sig.symbol}: {e}"
                        )

            if cycle % 10 == 0:
                parts = []
                if V2_ENABLED:
                    for th in THRESHOLDS:
                        eq = current_equity(engine, th, V2_STRATEGY)
                        op = open_positions(engine, th, V2_STRATEGY)
                        parts.append(f"v2@{th}=${eq:.0f}({op})")
                for acct in BIGMOVER_ACCOUNTS:
                    if not acct.get("enabled", True):
                        continue
                    eq = current_equity(
                        engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    )
                    op = open_positions(
                        engine, BIGMOVER_SENTINEL_THRESHOLD, acct["strategy"]
                    )
                    short_name = acct["strategy"].replace("bigmover_", "bm_")
                    parts.append(f"{short_name}=${eq:.0f}({op})")
                for acct in D1E_ACCOUNTS:
                    if not acct.get("enabled", True):
                        continue
                    eq = current_equity(
                        engine, D1E_SENTINEL_THRESHOLD, acct["strategy"]
                    )
                    op = open_positions(
                        engine, D1E_SENTINEL_THRESHOLD, acct["strategy"]
                    )
                    parts.append(f"{acct['strategy']}=${eq:.0f}({op})")
                for acct in V8_ACCOUNTS:
                    if not acct.get("enabled", True):
                        continue
                    with engine.begin() as conn:
                        rrow = conn.execute(text("""
                            SELECT COALESCE(SUM(pnl_usd), 0) FROM paper_trades
                            WHERE threshold = :th AND strategy = :strat
                              AND status IN ('won', 'lost', 'timeout')
                        """), {"th": V8_SENTINEL_THRESHOLD, "strat": acct["strategy"]}).fetchone()
                    eq = V8_STARTING_EQUITY_USD + (float(rrow[0]) if rrow else 0.0)
                    op = open_positions(
                        engine, V8_SENTINEL_THRESHOLD, acct["strategy"]
                    )
                    state = v8_sizing.load_state(engine, acct["strategy"])
                    parts.append(f"{acct['strategy']}=${eq:.0f}({op}/{state.phase})")
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
