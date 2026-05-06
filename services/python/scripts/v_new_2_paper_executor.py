"""v_new_2 — Paper executor (paper-deploy, $20 phase).

Single-shot tick runnable via cron. Three independent passes per tick:

  1. open_new_positions  — pick up untaken signals, open paper positions
                           (caps at MAX_CONCURRENT, $10 notional default)
  2. manage_open_positions — for each open position, pull latest 4h close,
                             update trail, evaluate exits (meme/non-meme rules,
                             90d timeout), close on exit
  3. snapshot_equity     — write a row to v_new_2_paper_account

Recommended cron (every 5 min):
    */5 * * * * /opt/ai-finance/services/python/.venv/bin/python3 \\
        /opt/ai-finance/services/python/scripts/v_new_2_paper_executor.py

Env config:
    DATABASE_URL                   — Postgres
    V_NEW_2_PAPER_BALANCE_INIT     — initial paper balance (default 20.0)
    V_NEW_2_PAPER_RISK_PCT         — % risk per trade (default 5.0)
    V_NEW_2_PAPER_LEVERAGE         — leverage (default 10)
    V_NEW_2_PAPER_MAX_CONCURRENT   — max open (default 2)
    V_NEW_2_PAPER_DRY_RUN          — if "1", logs only, no writes
    V_NEW_2_PAPER_HARD_TIMEOUT_DAYS — timeout (default 90)
"""
from __future__ import annotations

import logging
import math
import os
import sys
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("v_new_2_executor")

DB_URL = os.environ.get("DATABASE_URL",
                          "postgresql://aifinance:6bnn9pcfRpf9xW-CO4wsMmVtyA094nbK1XUQ72Tyi1U@localhost:5432/aifinance")
DRY_RUN = os.environ.get("V_NEW_2_PAPER_DRY_RUN", "0") == "1"

PAPER_BALANCE_INIT = float(os.environ.get("V_NEW_2_PAPER_BALANCE_INIT", "20.0"))
RISK_PCT = float(os.environ.get("V_NEW_2_PAPER_RISK_PCT", "5.0"))
LEVERAGE = float(os.environ.get("V_NEW_2_PAPER_LEVERAGE", "10"))
MAX_CONCURRENT = int(os.environ.get("V_NEW_2_PAPER_MAX_CONCURRENT", "5"))
HARD_TIMEOUT_DAYS = int(os.environ.get("V_NEW_2_PAPER_HARD_TIMEOUT_DAYS", "90"))

# Exit logic constants
MEME_PROFIT_LOCK_PCT = 20.0        # arm profit-lock at +20%
MEME_TRAIL_ATR_MULT = 3.0          # 3×ATR trail for memes (OOS sweep: 3x beats 5x, PF 5.60 vs 3.61)

DEFAULT_PROFIT_LOCK_PCT = 30.0     # arm profit-lock at +30% for non-memes
DEFAULT_TRAIL_ATR_MULT = 10.0      # 10×ATR trail after profit-lock (OOS Q1-2026: beats 8x on sum_pnl 96% vs 77% AND PF 4.56 vs 4.44)
DEFAULT_INITIAL_STOP_ATR = 5.0     # 5×ATR initial stop (no profit-lock yet)


# ─── DB helpers ─────────────────────────────────────────────────────────────
_engine = None


def _engine_lazy():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=300)
    return _engine


def _current_equity(engine) -> float:
    """Current paper account equity = initial + sum(realized pnl_usd) + sum(open mark-to-market).

    For paper, we keep it simple: equity = INIT + realized_pnl. Open MTM tracked separately.
    """
    with engine.connect() as c:
        r = c.execute(text("SELECT COALESCE(SUM(pnl_usd), 0) FROM v_new_2_paper_trades")).scalar()
    return PAPER_BALANCE_INIT + float(r or 0.0)


def _open_position_count(engine) -> int:
    with engine.connect() as c:
        r = c.execute(text("SELECT COUNT(*) FROM v_new_2_paper_positions WHERE status='open'")).scalar()
    return int(r or 0)


def _fetch_latest_4h_close(engine, symbol: str) -> tuple[pd.Timestamp | None, float | None, float | None]:
    """Return (latest_4h_close_ts, latest_4h_close_price, atr14_pct) for symbol.

    Uses 15m candles resampled to 4h. ATR computed from last 14 4h bars
    using mean of |ret| × 100 (matches scanner convention).
    """
    sql = text("""
        SELECT timestamp, open, high, low, close
        FROM asset_prices_15m
        WHERE symbol = :sym
          AND timestamp >= NOW() - INTERVAL '20 days'
        ORDER BY timestamp ASC
    """)
    with engine.connect() as c:
        rows = c.execute(sql, {"sym": symbol}).fetchall()
    if not rows or len(rows) < 50:
        return None, None, None
    df = pd.DataFrame(rows, columns=["timestamp","open","high","low","close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df_4h = df.resample("4h").agg({"open":"first","high":"max","low":"min","close":"last"}).dropna()
    if len(df_4h) < 16:
        return None, None, None
    last = df_4h.iloc[-1]
    abs_ret = (df_4h["close"] / df_4h["close"].shift(1) - 1).abs()
    atr14_pct = float(abs_ret.rolling(14, min_periods=5).mean().iloc[-1] * 100.0)
    return df_4h.index[-1], float(last["close"]), atr14_pct


def _compute_pnl_pct(entry: float, current: float, side: str) -> float:
    if side == "long":
        return (current / entry - 1.0) * 100.0
    return (entry / current - 1.0) * 100.0


# ─── Step 1: open new positions ─────────────────────────────────────────────
def open_new_positions(engine):
    open_count = _open_position_count(engine)
    if open_count >= MAX_CONCURRENT:
        log.info("[open] at capacity (%d/%d open) — skipping", open_count, MAX_CONCURRENT)
        return

    slots = MAX_CONCURRENT - open_count
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT id, signal_time, symbol, side, tier, is_meme, bgm_score,
                   d_ema_spread, close_at_signal, pullback_pct, pullback_atr
            FROM v_new_2_signals
            WHERE taken = FALSE
              AND signal_time >= NOW() - INTERVAL '8 hours'
            ORDER BY bgm_score DESC, signal_time DESC
            LIMIT :n
        """), {"n": slots}).fetchall()

    if not rows:
        log.info("[open] no untaken signals (last 8h)")
        return

    equity = _current_equity(engine)
    notional = equity * (RISK_PCT / 100.0) * LEVERAGE   # 5% × 10x × $20 = $10

    for r in rows:
        sig_id = r[0]
        sig_time, sym, side, tier, is_meme, bgm = r[1], r[2], r[3], r[4], r[5], r[6]
        signal_close = float(r[8])

        # Use signal_close as entry_price (paper, no slippage modeled here yet)
        entry_price = signal_close
        entry_time = pd.Timestamp(sig_time).to_pydatetime()
        log.info("[open] %s %s tier=%s meme=%s bgm=%.3f notional=$%.2f entry=%.6f",
                  sym, side, tier, is_meme, float(bgm), notional, entry_price)

        if DRY_RUN:
            continue

        with engine.connect() as c:
            c.execute(text("""
                INSERT INTO v_new_2_paper_positions
                  (signal_id, symbol, side, is_meme, tier, entry_time, entry_price,
                   bgm_score, position_size_pct, leverage, notional_usd,
                   account_at_entry, running_extreme, profit_lock_active, status)
                VALUES (:sid, :sym, :side, :ismeme, :tier, :ts, :entry,
                        :bgm, :pct, :lev, :notional,
                        :acct, :extreme, FALSE, 'open')
            """), {
                "sid": sig_id, "sym": sym, "side": side, "ismeme": bool(is_meme),
                "tier": tier, "ts": entry_time, "entry": entry_price,
                "bgm": float(bgm), "pct": RISK_PCT, "lev": LEVERAGE,
                "notional": notional, "acct": equity,
                "extreme": entry_price,
            })
            c.execute(text("UPDATE v_new_2_signals SET taken=TRUE WHERE id=:id"), {"id": sig_id})
            c.commit()


# ─── Step 2: manage open positions ──────────────────────────────────────────
def manage_open_positions(engine):
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT id, symbol, side, is_meme, tier, entry_time, entry_price,
                   running_extreme, profit_lock_active, notional_usd, bgm_score
            FROM v_new_2_paper_positions
            WHERE status = 'open'
            ORDER BY entry_time ASC
        """)).fetchall()

    if not rows:
        log.info("[manage] no open positions")
        return

    log.info("[manage] %d open positions", len(rows))
    for r in rows:
        pos_id, sym, side, is_meme, tier, entry_time, entry_price = r[0], r[1], r[2], r[3], r[4], r[5], r[6]
        running_extreme, pl_active, notional, bgm = r[7], r[8], r[9], r[10]

        latest_ts, latest_close, atr14_pct = _fetch_latest_4h_close(engine, sym)
        if latest_close is None:
            log.warning("[manage]   %s: no fresh data, skipping tick", sym)
            continue

        # Update running extreme
        if side == "long":
            new_extreme = max(float(running_extreme), latest_close)
        else:
            new_extreme = min(float(running_extreme), latest_close)

        pnl_pct = _compute_pnl_pct(float(entry_price), latest_close, side)

        # Profit-lock arming
        new_pl_active = bool(pl_active)
        threshold = MEME_PROFIT_LOCK_PCT if is_meme else DEFAULT_PROFIT_LOCK_PCT
        if not new_pl_active and pnl_pct >= threshold:
            new_pl_active = True
            log.info("[manage]   %s: profit-lock ARMED at pnl=%.2f%%", sym, pnl_pct)

        # Determine exit
        exit_reason = None
        exit_price = None

        # Hard timeout
        entry_dt = pd.Timestamp(entry_time)
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.tz_localize("UTC")
        held_days = (datetime.now(UTC) - entry_dt.to_pydatetime()).total_seconds() / 86400.0
        if held_days >= HARD_TIMEOUT_DAYS:
            exit_reason = "timeout_90d"
            exit_price = latest_close

        # Trailing stop
        if exit_reason is None and atr14_pct is not None and atr14_pct > 0:
            atr_dist_abs = (atr14_pct / 100.0) * float(entry_price)
            if is_meme:
                trail_dist = MEME_TRAIL_ATR_MULT * atr_dist_abs
                if side == "long":
                    stop = new_extreme - trail_dist
                    if latest_close <= stop:
                        exit_reason = "meme_trail_3atr"
                        exit_price = latest_close
                else:
                    stop = new_extreme + trail_dist
                    if latest_close >= stop:
                        exit_reason = "meme_trail_3atr"
                        exit_price = latest_close
            else:
                # Non-meme: 5×ATR initial stop, widens to 10×ATR trail after profit-lock arms
                trail_mult = DEFAULT_TRAIL_ATR_MULT if new_pl_active else DEFAULT_INITIAL_STOP_ATR
                trail_dist = trail_mult * atr_dist_abs
                if side == "long":
                    stop = new_extreme - trail_dist
                    if latest_close <= stop:
                        exit_reason = "atr_trail"
                        exit_price = latest_close
                else:
                    stop = new_extreme + trail_dist
                    if latest_close >= stop:
                        exit_reason = "atr_trail"
                        exit_price = latest_close

        if exit_reason:
            final_pnl_pct = _compute_pnl_pct(float(entry_price), exit_price, side)
            pnl_usd = (final_pnl_pct / 100.0) * float(notional)
            bars_held = int(held_days * 6)
            log.info("[manage]   %s: EXIT %s pnl=%.2f%% ($%.2f) held=%dbars",
                      sym, exit_reason, final_pnl_pct, pnl_usd, bars_held)
            if not DRY_RUN:
                with engine.connect() as c:
                    c.execute(text("""
                        INSERT INTO v_new_2_paper_trades
                          (position_id, symbol, side, is_meme, tier,
                           entry_time, exit_time, entry_price, exit_price,
                           bars_held, exit_reason, bgm_score,
                           position_size_pct, leverage, pnl_pct, pnl_usd)
                        VALUES (:pid, :sym, :side, :ismeme, :tier,
                                :et, :xt, :ep, :xp,
                                :bars, :reason, :bgm,
                                :pct, :lev, :pnlpct, :pnlusd)
                    """), {
                        "pid": pos_id, "sym": sym, "side": side, "ismeme": bool(is_meme),
                        "tier": tier, "et": entry_dt.to_pydatetime(),
                        "xt": datetime.now(UTC), "ep": float(entry_price), "xp": float(exit_price),
                        "bars": bars_held, "reason": exit_reason, "bgm": float(bgm) if bgm else None,
                        "pct": RISK_PCT, "lev": LEVERAGE,
                        "pnlpct": float(final_pnl_pct), "pnlusd": float(pnl_usd),
                    })
                    c.execute(text("UPDATE v_new_2_paper_positions SET status='closed' WHERE id=:id"),
                                {"id": pos_id})
                    c.commit()
        else:
            log.info("[manage]   %s: HOLD pnl=%.2f%% extreme=%.6f close=%.6f",
                      sym, pnl_pct, new_extreme, latest_close)
            if not DRY_RUN:
                with engine.connect() as c:
                    c.execute(text("""
                        UPDATE v_new_2_paper_positions
                           SET running_extreme = :ext, profit_lock_active = :pla
                         WHERE id = :id
                    """), {"ext": new_extreme, "pla": new_pl_active, "id": pos_id})
                    c.commit()


# ─── Step 3: equity snapshot ────────────────────────────────────────────────
def snapshot_equity(engine):
    equity = _current_equity(engine)
    open_count = _open_position_count(engine)
    with engine.connect() as c:
        n_trades = c.execute(text("SELECT COUNT(*) FROM v_new_2_paper_trades")).scalar() or 0
    log.info("[snapshot] equity=$%.2f open=%d trades=%d", equity, open_count, n_trades)
    if DRY_RUN:
        return
    with engine.connect() as c:
        c.execute(text("""
            INSERT INTO v_new_2_paper_account (ts, equity_usd, open_positions, realized_trades)
            VALUES (:ts, :eq, :op, :n)
        """), {"ts": datetime.now(UTC), "eq": equity, "op": open_count, "n": int(n_trades)})
        c.commit()


def main():
    log.info("v_new_2 paper executor — start (DRY_RUN=%s init_balance=$%.2f risk=%.1f%% lev=%.1fx max=%d)",
              DRY_RUN, PAPER_BALANCE_INIT, RISK_PCT, LEVERAGE, MAX_CONCURRENT)
    engine = _engine_lazy()
    open_new_positions(engine)
    manage_open_positions(engine)
    snapshot_equity(engine)
    log.info("v_new_2 paper executor — done")


if __name__ == "__main__":
    main()
