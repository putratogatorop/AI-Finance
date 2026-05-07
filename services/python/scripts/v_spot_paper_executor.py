"""v_spot — Paper executor for the 1-2 position spot portfolio.

Strategy locked 2026-05-07 from cross-OOS backtest (sum +6077% across thr_select+cal+holdout
vs BTC HODL -37%):

  - Long-only (skip side='short' signals)
  - cap=2 concurrent positions
  - 50% equity per slot at entry (no leverage)
  - BGM honest score gate >= 0.65
  - No tiered profit-taking (FLAT — empirically beat all moon-bag ladders by 100-500pp)
  - Exits inherited from v_new_2 paper executor: ATR trail (3x meme / 5-10x non-meme post-lock)
    + 90-day hard timeout

Reads v_new_2_signals (read-only — does NOT mutate the `taken` flag, so v_new_2's executor
is undisturbed). Consumption is tracked through v_spot_paper_positions.signal_id.

Recommended cron (every 5 min, alongside v_new_2_paper_executor):
    */5 * * * * /opt/ai-finance/services/python/.venv/bin/python3 \\
        /opt/ai-finance/services/python/scripts/v_spot_paper_executor.py

Env config:
    DATABASE_URL                   — Postgres
    V_SPOT_PAPER_BALANCE_INIT      — initial paper balance (default 20.0)
    V_SPOT_PAPER_POS_PCT           — % equity per slot (default 50.0)
    V_SPOT_PAPER_MAX_CONCURRENT    — max open (default 2)
    V_SPOT_PAPER_BGM_THRESHOLD     — BGM gate (default 0.65)
    V_SPOT_PAPER_HARD_TIMEOUT_DAYS — timeout (default 90)
    V_SPOT_PAPER_DRY_RUN           — if "1", logs only, no writes
"""
from __future__ import annotations

import logging
import os
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("v_spot_executor")

DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://aifinance:6bnn9pcfRpf9xW-CO4wsMmVtyA094nbK1XUQ72Tyi1U@localhost:5432/aifinance",
)
DRY_RUN = os.environ.get("V_SPOT_PAPER_DRY_RUN", "0") == "1"

PAPER_BALANCE_INIT = float(os.environ.get("V_SPOT_PAPER_BALANCE_INIT", "20.0"))
POS_PCT = float(os.environ.get("V_SPOT_PAPER_POS_PCT", "50.0"))
MAX_CONCURRENT = int(os.environ.get("V_SPOT_PAPER_MAX_CONCURRENT", "2"))
BGM_THRESHOLD = float(os.environ.get("V_SPOT_PAPER_BGM_THRESHOLD", "0.65"))
HARD_TIMEOUT_DAYS = int(os.environ.get("V_SPOT_PAPER_HARD_TIMEOUT_DAYS", "90"))

# Exit logic constants (inherited from v_new_2 paper executor)
MEME_PROFIT_LOCK_PCT = 20.0
MEME_TRAIL_ATR_MULT = 3.0

DEFAULT_PROFIT_LOCK_PCT = 30.0
DEFAULT_TRAIL_ATR_MULT = 10.0
DEFAULT_INITIAL_STOP_ATR = 5.0


_engine = None


def _engine_lazy():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=300)
    return _engine


def _current_equity(engine) -> float:
    """Paper equity = INIT + sum(realized pnl_usd from v_spot_paper_trades)."""
    with engine.connect() as c:
        r = c.execute(text("SELECT COALESCE(SUM(pnl_usd), 0) FROM v_spot_paper_trades")).scalar()
    return PAPER_BALANCE_INIT + float(r or 0.0)


def _open_position_count(engine) -> int:
    with engine.connect() as c:
        r = c.execute(text("SELECT COUNT(*) FROM v_spot_paper_positions WHERE status='open'")).scalar()
    return int(r or 0)


def _fetch_latest_4h_close(engine, symbol: str) -> tuple[pd.Timestamp | None, float | None, float | None]:
    """Return (latest_4h_close_ts, latest_4h_close_price, atr14_pct) for symbol.

    Same logic as v_new_2 paper executor: 15m candles resampled to 4h, ATR=mean(|ret|)*100
    over last 14 4h bars.
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
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp")
    df_4h = df.resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
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
    # Filter: long-only, BGM>=threshold, recent, NOT already consumed by v_spot
    # (independent of v_new_2's `taken` flag — both executors run independently).
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT s.id, s.signal_time, s.symbol, s.side, s.tier, s.is_meme, s.bgm_score,
                   s.d_ema_spread, s.close_at_signal, s.pullback_pct, s.pullback_atr,
                   s.lstm_regime, s.regime_meme_trail, s.regime_nonmeme_trail
            FROM v_new_2_signals s
            WHERE s.side = 'long'
              AND s.bgm_score >= :bgm
              AND s.signal_time >= NOW() - INTERVAL '8 hours'
              AND NOT EXISTS (
                  SELECT 1 FROM v_spot_paper_positions p
                   WHERE p.signal_id = s.id
              )
              AND NOT EXISTS (
                  SELECT 1 FROM v_spot_paper_positions p
                   WHERE p.symbol = s.symbol AND p.status = 'open'
              )
            ORDER BY s.bgm_score DESC, s.signal_time DESC
            LIMIT :n
        """), {"bgm": BGM_THRESHOLD, "n": slots}).fetchall()

    if not rows:
        log.info("[open] no eligible long signals (bgm>=%.2f, last 8h, not already taken)",
                  BGM_THRESHOLD)
        return

    equity = _current_equity(engine)
    # Spot sizing: notional = equity × POS_PCT/100, no leverage.
    # With cap=2 and POS_PCT=50, each slot = 50% of equity → 100% deployed when both filled.
    notional = equity * (POS_PCT / 100.0)

    for r in rows:
        sig_id = r[0]
        sig_time, sym, side, tier, is_meme, bgm = r[1], r[2], r[3], r[4], r[5], r[6]
        signal_close = float(r[8])
        lstm_regime = r[11] or "neutral"
        regime_meme_trail = float(r[12]) if r[12] is not None else MEME_TRAIL_ATR_MULT
        regime_nonmeme_trail = float(r[13]) if r[13] is not None else DEFAULT_TRAIL_ATR_MULT

        entry_price = signal_close
        entry_time = pd.Timestamp(sig_time).to_pydatetime()
        log.info("[open] %s %s tier=%s meme=%s bgm=%.3f regime=%s trail=%.0f/%.0f notional=$%.2f entry=%.6f",
                  sym, side, tier, is_meme, float(bgm), lstm_regime,
                  regime_meme_trail, regime_nonmeme_trail, notional, entry_price)

        if DRY_RUN:
            continue

        with engine.connect() as c:
            c.execute(text("""
                INSERT INTO v_spot_paper_positions
                  (signal_id, symbol, side, is_meme, tier, entry_time, entry_price,
                   bgm_score, position_size_pct, notional_usd,
                   account_at_entry, running_extreme, profit_lock_active, status,
                   lstm_regime, regime_meme_trail, regime_nonmeme_trail)
                VALUES (:sid, :sym, :side, :ismeme, :tier, :ts, :entry,
                        :bgm, :pct, :notional,
                        :acct, :extreme, FALSE, 'open',
                        :regime, :meme_trail, :nonmeme_trail)
            """), {
                "sid": sig_id, "sym": sym, "side": side, "ismeme": bool(is_meme),
                "tier": tier, "ts": entry_time, "entry": entry_price,
                "bgm": float(bgm), "pct": POS_PCT,
                "notional": notional, "acct": equity,
                "extreme": entry_price,
                "regime": lstm_regime,
                "meme_trail": regime_meme_trail,
                "nonmeme_trail": regime_nonmeme_trail,
            })
            c.commit()


# ─── Step 2: manage open positions ──────────────────────────────────────────
def manage_open_positions(engine):
    with engine.connect() as c:
        rows = c.execute(text("""
            SELECT id, symbol, side, is_meme, tier, entry_time, entry_price,
                   running_extreme, profit_lock_active, notional_usd, bgm_score,
                   lstm_regime, regime_meme_trail, regime_nonmeme_trail
            FROM v_spot_paper_positions
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
        pos_meme_trail = float(r[12]) if r[12] is not None else MEME_TRAIL_ATR_MULT
        pos_nonmeme_trail = float(r[13]) if r[13] is not None else DEFAULT_TRAIL_ATR_MULT

        latest_ts, latest_close, atr14_pct = _fetch_latest_4h_close(engine, sym)
        if latest_close is None:
            log.warning("[manage]   %s: no fresh data, skipping tick", sym)
            continue

        # spot is long-only, but keep the side-aware extreme update for safety
        if side == "long":
            new_extreme = max(float(running_extreme), latest_close)
        else:
            new_extreme = min(float(running_extreme), latest_close)

        pnl_pct = _compute_pnl_pct(float(entry_price), latest_close, side)

        new_pl_active = bool(pl_active)
        threshold = MEME_PROFIT_LOCK_PCT if is_meme else DEFAULT_PROFIT_LOCK_PCT
        if not new_pl_active and pnl_pct >= threshold:
            new_pl_active = True
            log.info("[manage]   %s: profit-lock ARMED at pnl=%.2f%%", sym, pnl_pct)

        exit_reason = None
        exit_price = None

        entry_dt = pd.Timestamp(entry_time)
        if entry_dt.tzinfo is None:
            entry_dt = entry_dt.tz_localize("UTC")
        held_days = (datetime.now(UTC) - entry_dt.to_pydatetime()).total_seconds() / 86400.0
        if held_days >= HARD_TIMEOUT_DAYS:
            exit_reason = f"timeout_{HARD_TIMEOUT_DAYS}d"
            exit_price = latest_close

        if exit_reason is None and atr14_pct is not None and atr14_pct > 0:
            atr_dist_abs = (atr14_pct / 100.0) * float(entry_price)
            if is_meme:
                trail_dist = pos_meme_trail * atr_dist_abs
                if side == "long":
                    stop = new_extreme - trail_dist
                    if latest_close <= stop:
                        exit_reason = f"meme_trail_{pos_meme_trail:.0f}atr"
                        exit_price = latest_close
                else:
                    stop = new_extreme + trail_dist
                    if latest_close >= stop:
                        exit_reason = f"meme_trail_{pos_meme_trail:.0f}atr"
                        exit_price = latest_close
            else:
                trail_mult = pos_nonmeme_trail if new_pl_active else DEFAULT_INITIAL_STOP_ATR
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
            # Spot: no leverage, pnl_usd = pnl_pct/100 × notional (the dollar amount we put in)
            pnl_usd = (final_pnl_pct / 100.0) * float(notional)
            bars_held = int(held_days * 6)
            log.info("[manage]   %s: EXIT %s pnl=%.2f%% ($%.2f) held=%dbars",
                      sym, exit_reason, final_pnl_pct, pnl_usd, bars_held)
            if not DRY_RUN:
                with engine.connect() as c:
                    c.execute(text("""
                        INSERT INTO v_spot_paper_trades
                          (position_id, symbol, side, is_meme, tier,
                           entry_time, exit_time, entry_price, exit_price,
                           bars_held, exit_reason, bgm_score,
                           position_size_pct, pnl_pct, pnl_usd)
                        VALUES (:pid, :sym, :side, :ismeme, :tier,
                                :et, :xt, :ep, :xp,
                                :bars, :reason, :bgm,
                                :pct, :pnlpct, :pnlusd)
                    """), {
                        "pid": pos_id, "sym": sym, "side": side, "ismeme": bool(is_meme),
                        "tier": tier, "et": entry_dt.to_pydatetime(),
                        "xt": datetime.now(UTC), "ep": float(entry_price), "xp": float(exit_price),
                        "bars": bars_held, "reason": exit_reason,
                        "bgm": float(bgm) if bgm else None,
                        "pct": POS_PCT,
                        "pnlpct": float(final_pnl_pct), "pnlusd": float(pnl_usd),
                    })
                    c.execute(
                        text("UPDATE v_spot_paper_positions SET status='closed' WHERE id=:id"),
                        {"id": pos_id},
                    )
                    c.commit()
        else:
            log.info("[manage]   %s: HOLD pnl=%.2f%% extreme=%.6f close=%.6f",
                      sym, pnl_pct, new_extreme, latest_close)
            if not DRY_RUN:
                with engine.connect() as c:
                    c.execute(text("""
                        UPDATE v_spot_paper_positions
                           SET running_extreme = :ext, profit_lock_active = :pla
                         WHERE id = :id
                    """), {"ext": new_extreme, "pla": new_pl_active, "id": pos_id})
                    c.commit()


# ─── Step 3: equity snapshot ────────────────────────────────────────────────
def snapshot_equity(engine):
    equity = _current_equity(engine)
    open_count = _open_position_count(engine)
    with engine.connect() as c:
        n_trades = c.execute(text("SELECT COUNT(*) FROM v_spot_paper_trades")).scalar() or 0
    log.info("[snapshot] equity=$%.2f open=%d trades=%d", equity, open_count, n_trades)
    if DRY_RUN:
        return
    with engine.connect() as c:
        c.execute(text("""
            INSERT INTO v_spot_paper_account (ts, equity_usd, open_positions, realized_trades)
            VALUES (:ts, :eq, :op, :n)
        """), {"ts": datetime.now(UTC), "eq": equity, "op": open_count, "n": int(n_trades)})
        c.commit()


def main():
    log.info("v_spot paper executor — start (DRY_RUN=%s init=$%.2f pos_pct=%.1f%% max=%d bgm>=%.2f)",
              DRY_RUN, PAPER_BALANCE_INIT, POS_PCT, MAX_CONCURRENT, BGM_THRESHOLD)
    engine = _engine_lazy()
    open_new_positions(engine)
    manage_open_positions(engine)
    snapshot_equity(engine)
    log.info("v_spot paper executor — done")


if __name__ == "__main__":
    main()
