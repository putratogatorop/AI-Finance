"""v8 production sizing module — phase-switch v6 ATR-aware risk-based sizing.

Promoted from research scripts (research_sizing_v6_atr.py +
phase_switch_sim_v8.py) into a callable production module:

    decision = compute_size(
        current_equity=200.0,
        exit_kind="e2",
        atr14_at_entry=0.012,
        entry_price=1.42,
        open_gross_notional_usd=0.0,
        recent_pnl=[...],   # closed pnl_pct window
        current_phase="P1",
    )
    if decision.take:
        # open position with decision.notional_usd

Phase-switch (one-way P1 → P2 at $10K equity):

    PHASE 1 (aggressive — small bankroll growth, $200 → $10K)
        Kelly mult     0.50
        Risk ceil      3.0% per trade
        Per-pos cap    25% of equity (notional)
        Gross cap      100% of equity
        DD expected    ~12-18% (real-world ~20-25%)

    PHASE 2 (conservative — auto-active when equity ≥ $10K)
        Kelly mult     0.25
        Risk ceil      1.5% per trade
        Per-pos cap    12% of equity
        Gross cap      60% of equity
        DD expected    ~6-12%

Validated against 3y backtest portfolio (run 92373355_20260429T041931Z):
    portfolio_metrics.json wf_p5 = 1.486 on the v8 BALANCED 4-detector stack
    phase_switch_monthly.csv: $200 → $991K over 36 months (raw backtest;
    realistic ~30-50% haircut applies).

Hard safety: RUIN_FLOOR_USD = $50 — refuses to size below that.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from sqlalchemy import text


# ── Phase configs (LOCKED — change requires backtest re-citation) ────
@dataclass(frozen=True)
class PhaseConfig:
    kelly_mult: float
    risk_ceil_pct: float
    gross_cap: float
    per_pos_cap: float


PHASE1 = PhaseConfig(kelly_mult=0.50, risk_ceil_pct=0.030,
                     gross_cap=1.00, per_pos_cap=0.25)
PHASE2 = PhaseConfig(kelly_mult=0.25, risk_ceil_pct=0.015,
                     gross_cap=0.60, per_pos_cap=0.12)
PHASE2_TRIGGER_USD = 10_000.0

# Sizing constants
LEVERAGE = 5.0
ROLLING_KELLY_WINDOW = 60
KELLY_MIN_TRADES = 30
RISK_FLOOR_PCT = 0.001          # 0.10% min risk per trade
RISK_BOOTSTRAP_PCT = 0.0025     # 0.25% during bootstrap (< KELLY_MIN_TRADES closed)
MIN_NOTIONAL_PCT = 0.005        # 0.5% — below this fees dominate
RUIN_FLOOR_USD = 50.0           # halt new entries if equity drops below

# Exit-aware SL distance
E1_SL_PCT = 0.05
E2_ATR_SL_MULT = 2.0
DEFAULT_SL_PCT = 0.05  # fallback if atr/entry_price malformed


@dataclass
class SizingDecision:
    take: bool
    notional_usd: float
    notional_pct: float           # of current equity
    margin_usd: float             # notional / leverage
    risk_pct: float               # of current equity (target risk per trade)
    sl_distance_pct: float        # of entry_price
    phase: str                    # "P1" or "P2"
    reason: str = ""              # if not take, the reason string


@dataclass
class SizingState:
    """In-memory state, mirrored to DB via load_state/save_state."""
    strategy_group: str
    phase: str = "P1"
    phase_switch_ts: datetime | None = None
    phase_switch_eq: float | None = None
    recent_pnl: list[float] = field(default_factory=list)


# ── Pure functions (no I/O) ──────────────────────────────────────────


def determine_phase(equity_usd: float, current_phase: str) -> str:
    """One-way P1 → P2 at $10K. P2 never reverts to P1."""
    if current_phase == "P1" and equity_usd >= PHASE2_TRIGGER_USD:
        return "P2"
    return current_phase


def compute_sl_distance_pct(exit_kind: str, atr14_at_entry: float,
                             entry_price: float) -> float:
    """Returns SL distance as fraction of entry price.

    E1 = fixed 5%; E2/E3/E4 = 2 × ATR14 / entry_price (clipped to [0.5%, 30%]).
    """
    if exit_kind == "e1":
        return E1_SL_PCT
    if (not np.isfinite(atr14_at_entry) or atr14_at_entry <= 0
            or not np.isfinite(entry_price) or entry_price <= 0):
        return DEFAULT_SL_PCT
    sl = E2_ATR_SL_MULT * atr14_at_entry / entry_price
    return float(np.clip(sl, 0.005, 0.30))


def compute_kelly_risk_pct(recent_pnl: list[float], kelly_mult: float,
                            risk_ceil: float) -> float:
    """Per-trade target risk as fraction of equity.

    Uses last ROLLING_KELLY_WINDOW closed pnl_pcts. Bootstraps at
    RISK_BOOTSTRAP_PCT until KELLY_MIN_TRADES closed trades available.
    """
    if len(recent_pnl) < KELLY_MIN_TRADES:
        return RISK_BOOTSTRAP_PCT
    recent = np.array(recent_pnl[-ROLLING_KELLY_WINDOW:], dtype=float)
    wins = recent[recent > 0]
    losses = recent[recent < 0]
    if len(wins) == 0 or len(losses) == 0:
        return RISK_BOOTSTRAP_PCT
    p = len(wins) / len(recent)
    avg_win = wins.mean()
    avg_loss = -losses.mean()
    if avg_loss <= 0:
        return RISK_BOOTSTRAP_PCT
    R = avg_win / avg_loss
    f_star = (p * R - (1 - p)) / R
    kelly = max(f_star, 0.0)
    return float(np.clip(kelly_mult * kelly, RISK_FLOOR_PCT, risk_ceil))


def compute_size(*, current_equity: float, exit_kind: str,
                  atr14_at_entry: float, entry_price: float,
                  open_gross_notional_usd: float, recent_pnl: list[float],
                  current_phase: str) -> SizingDecision:
    """The single production-grade entry: returns a SizingDecision.

    Caller must:
      1. Load `current_phase` and `recent_pnl` from `paper_sizing_state` table.
      2. Compute `current_equity` from running paper equity.
      3. Compute `open_gross_notional_usd` = sum of currently-open notional USD
         across the strategy_group.
      4. Pass `atr14_at_entry` and `entry_price` from the signal row.

    Caller does NOT need to track phase transitions — this function reports
    `decision.phase` which the caller persists back via save_state().
    """
    # Ruin floor — halt new entries if equity dies
    if current_equity < RUIN_FLOOR_USD:
        return SizingDecision(False, 0.0, 0.0, 0.0, 0.0, 0.0, current_phase,
                              f"ruin_floor_hit:eq={current_equity:.2f}<${RUIN_FLOOR_USD}")

    phase = determine_phase(current_equity, current_phase)
    cfg = PHASE1 if phase == "P1" else PHASE2

    risk_pct = compute_kelly_risk_pct(recent_pnl, cfg.kelly_mult, cfg.risk_ceil_pct)
    sl_dist = compute_sl_distance_pct(exit_kind, atr14_at_entry, entry_price)
    if sl_dist <= 0:
        return SizingDecision(False, 0.0, 0.0, 0.0, risk_pct, sl_dist, phase,
                              "bad_sl_distance")

    desired_notional_pct = risk_pct / sl_dist
    notional_pct = min(desired_notional_pct, cfg.per_pos_cap)

    # Concurrent gross-notional cap
    current_gross_pct = (open_gross_notional_usd / current_equity
                         if current_equity > 0 else 1.0)
    headroom = max(cfg.gross_cap - current_gross_pct, 0.0)
    notional_pct = min(notional_pct, headroom)

    if notional_pct < MIN_NOTIONAL_PCT:
        return SizingDecision(False, 0.0, 0.0, 0.0, risk_pct, sl_dist, phase,
                              f"below_min_notional:notional_pct={notional_pct:.4%}")

    notional_usd = notional_pct * current_equity
    margin_usd = notional_usd / LEVERAGE
    return SizingDecision(
        take=True, notional_usd=notional_usd, notional_pct=notional_pct,
        margin_usd=margin_usd, risk_pct=risk_pct, sl_distance_pct=sl_dist,
        phase=phase, reason="",
    )


# ── DB state persistence ─────────────────────────────────────────────


def ensure_state_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS paper_sizing_state (
                    strategy_group VARCHAR(50) PRIMARY KEY,
                    phase VARCHAR(5) NOT NULL DEFAULT 'P1',
                    phase_switch_ts TIMESTAMPTZ,
                    phase_switch_eq DOUBLE PRECISION,
                    recent_pnl_window JSONB NOT NULL DEFAULT '[]'::jsonb,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                )
            """)
        )


def load_state(engine, strategy_group: str) -> SizingState:
    """Read phase + rolling PnL window from DB. Initializes P1 row if missing."""
    with engine.connect() as conn:
        row = conn.execute(
            text("""
                SELECT phase, phase_switch_ts, phase_switch_eq, recent_pnl_window
                FROM paper_sizing_state WHERE strategy_group = :sg
            """),
            {"sg": strategy_group},
        ).fetchone()
    if row is None:
        # First call — create the row with defaults
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO paper_sizing_state (strategy_group, phase)
                    VALUES (:sg, 'P1') ON CONFLICT DO NOTHING
                """),
                {"sg": strategy_group},
            )
        return SizingState(strategy_group=strategy_group, phase="P1", recent_pnl=[])

    phase, switch_ts, switch_eq, pnl_raw = row
    if isinstance(pnl_raw, list):
        pnl_list = [float(x) for x in pnl_raw]
    elif isinstance(pnl_raw, str):
        pnl_list = [float(x) for x in json.loads(pnl_raw)]
    else:
        pnl_list = []
    return SizingState(
        strategy_group=strategy_group,
        phase=phase or "P1",
        phase_switch_ts=switch_ts,
        phase_switch_eq=switch_eq,
        recent_pnl=pnl_list,
    )


def save_state(engine, state: SizingState) -> None:
    pnl_window = state.recent_pnl[-ROLLING_KELLY_WINDOW:]
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO paper_sizing_state
                    (strategy_group, phase, phase_switch_ts, phase_switch_eq,
                     recent_pnl_window, updated_at)
                VALUES (:sg, :ph, :pts, :pe, CAST(:pnl AS JSONB), NOW())
                ON CONFLICT (strategy_group) DO UPDATE
                SET phase = EXCLUDED.phase,
                    phase_switch_ts = COALESCE(EXCLUDED.phase_switch_ts,
                                               paper_sizing_state.phase_switch_ts),
                    phase_switch_eq = COALESCE(EXCLUDED.phase_switch_eq,
                                               paper_sizing_state.phase_switch_eq),
                    recent_pnl_window = EXCLUDED.recent_pnl_window,
                    updated_at = NOW()
            """),
            {
                "sg": state.strategy_group, "ph": state.phase,
                "pts": state.phase_switch_ts, "pe": state.phase_switch_eq,
                "pnl": json.dumps(pnl_window),
            },
        )


def record_close(engine, strategy_group: str, closed_pnl_pct: float,
                  new_equity: float) -> SizingState:
    """Call after a position closes. Updates Kelly window + phase."""
    state = load_state(engine, strategy_group)
    state.recent_pnl.append(closed_pnl_pct)
    state.recent_pnl = state.recent_pnl[-ROLLING_KELLY_WINDOW:]
    new_phase = determine_phase(new_equity, state.phase)
    if new_phase != state.phase:
        state.phase = new_phase
        state.phase_switch_ts = datetime.now(UTC)
        state.phase_switch_eq = new_equity
    save_state(engine, state)
    return state
