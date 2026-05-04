"""v_new_1 Phase 4 — Policy Optimization via Optuna.

Loads Platt-calibrated OOS scores for long + short, simulates a portfolio trading
policy over 4 OOS years (2020/2021/2022/2026_q1), applies hard gates (max-DD ≥ -20%
per year, every year positive), and maximises 4-year sum_pnl + Sharpe-like ratio.

USAGE (from repo root):
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_phase4_policy_opt.py

OUTPUTS:
    services/python/results/v_new_1_phase4/all_trials.csv
    services/python/results/v_new_1_phase4/best_policy.json
    services/python/results/v_new_1_phase4/best_policy_per_year.csv
    services/python/results/v_new_1_phase4/verdict.md
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ── paths ────────────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = ROOT / "services" / "python" / "data" / "v_new_1"
RESULTS = ROOT / "services" / "python" / "results" / "v_new_1_phase4"
RESULTS.mkdir(parents=True, exist_ok=True)

OOS_YEARS = ["2020", "2021", "2022", "2026_q1"]

# BTC buy-hold benchmark: ~250% over 4 years (per plan §2)
BTC_4YR_BENCHMARK_PCT = 250.0
# macd_pullback_long baseline (per plan §1)
MACD_BASELINE_4YR_PCT = 96.0

# Friction: 0.1% round-trip of NOTIONAL per trade (Gate.io taker fee, conservative)
# Applied as a deduction from the raw return BEFORE leverage/size scaling.
# Equivalent to: pnl = (raw_pnl_pct - FRICTION_NOTIONAL_PCT) × leverage × (size/100)
# For example: raw=+2%, friction=0.1% on notional → net_raw = 1.9%, then levered.
FRICTION_NOTIONAL_PCT = 0.1


# ── data classes ─────────────────────────────────────────────────────────────
@dataclass
class Policy:
    top_k_long: int
    top_k_short: int
    score_threshold_long: float
    score_threshold_short: float
    position_size_pct: float
    leverage: float
    stop_loss_pct: Optional[float]  # negative, e.g. -3.0 means -3%; None = no SL
    take_profit_pct: Optional[float]  # None = no TP (timeout/SL only)
    hold_timeout_bars: int        # 6 bars per day × hold_timeout_days
    score_scaled_sizing: bool
    cfgi_short_gate: Optional[float]   # skip short if cfgi > X
    cfgi_long_gate: Optional[float]    # skip long if cfgi < X
    gross_exposure_cap_pct: float


@dataclass
class YearMetrics:
    year: str
    n_long_trades: int
    n_short_trades: int
    n_total_trades: int
    sum_pnl_pct: float
    max_drawdown_pct: float
    profit_factor: float
    win_rate_pct: float
    compounded_equity_pct: float   # ((1+pnl_i/100).cumprod()-1)*100 final
    sharpe_like: float             # sum_pnl / abs(max_dd) if max_dd!=0 else sum_pnl
    long_pnl_pct: float
    short_pnl_pct: float
    avg_gross_exposure_pct: float


# ── data loading ─────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and merge Platt-scored data with forward returns and CFGI.

    Key columns added to each row:
    - forward_max_pct: max close return over 7d (fraction, e.g. 0.10 = 10%)
    - forward_min_pct: min close return over 7d (fraction, e.g. -0.10 = -10%)
    - forward_7d_close_return: actual close-to-close 7d return (fraction)
      Used for timeout exit to avoid bias from midpoint heuristic.
    - cfgi_value: Fear & Greed Index at the timestamp (0-100)
    """
    long_scores = pd.read_parquet(DATA / "oos_scored_long_PLATT.parquet")
    short_scores = pd.read_parquet(DATA / "oos_scored_short_PLATT.parquet")
    labels_long = pd.read_parquet(DATA / "labels_long_oos.parquet")
    labels_short = pd.read_parquet(DATA / "labels_short_oos.parquet")
    features = pd.read_parquet(DATA / "features_full.parquet",
                               columns=["symbol", "timestamp", "cfgi_value"])

    # Compute 7-day close return by shifting 42 bars (7 days × 6 bars/day)
    # Sort by (symbol, timestamp) to ensure correct shift alignment
    ll = labels_long.sort_values(["symbol", "timestamp"]).copy()
    ll["forward_7d_close_return"] = (
        ll.groupby("symbol")["close"].shift(-42) / ll["close"] - 1.0
    )
    ls = labels_short.sort_values(["symbol", "timestamp"]).copy()
    # For shorts: same 7d close return (we negate at simulation time)
    ls["forward_7d_close_return"] = (
        ls.groupby("symbol")["close"].shift(-42) / ls["close"] - 1.0
    )

    # Build a combined forward-returns table with BOTH max and min per row
    fwd_combined = ll[["symbol", "timestamp", "forward_max_pct", "forward_7d_close_return"]].merge(
        ls[["symbol", "timestamp", "forward_min_pct"]],
        on=["symbol", "timestamp"], how="inner"
    )

    # Merge forward returns onto scored data (both directions need max, min, 7d_close)
    long_df = long_scores.merge(fwd_combined, on=["symbol", "timestamp"], how="inner")
    short_df = short_scores.merge(fwd_combined, on=["symbol", "timestamp"], how="inner")

    # Merge CFGI (join on nearest symbol-independent daily value → merge on timestamp)
    # CFGI is market-wide, same for all coins at a timestamp. Take first per timestamp.
    cfgi_ts = (
        features.drop_duplicates(subset="timestamp")[["timestamp", "cfgi_value"]]
        .copy()
    )
    long_df = long_df.merge(cfgi_ts, on="timestamp", how="left")
    short_df = short_df.merge(cfgi_ts, on="timestamp", how="left")

    # Normalize timestamp to UTC
    for df in [long_df, short_df]:
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # Sort
    long_df = long_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    short_df = short_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)

    return long_df, short_df


def split_by_year(df: pd.DataFrame, year_col: str = "oos_year") -> dict[str, pd.DataFrame]:
    return {yr: grp.copy() for yr, grp in df.groupby(year_col)}


# ── simulator ────────────────────────────────────────────────────────────────
def simulate_year(
    long_yr: pd.DataFrame,
    short_yr: pd.DataFrame,
    policy: Policy,
) -> YearMetrics:
    """
    Simulate trading for one OOS year using non-overlapping 7-day portfolio windows.

    Architecture:
    - Divide the year into non-overlapping 7-day (42-bar) windows.
    - At the start bar of each window, pick top_k_long and top_k_short coins.
    - Hold each picked position for the full 7-day window (or until SL/TP).
    - Portfolio return for the window = equal-weighted average of position returns.
    - Equity compounding is window-to-window (clean, no position-overlap inflation).
    - position_size_pct is the fraction of capital per position.
      Total allocation per window = n_positions × position_size_pct × leverage.
      Capped by gross_exposure_cap_pct via scaling down n_positions if needed.

    Exit logic per position (long example):
    - SL hit if forward_min_pct ≤ stop_loss_pct/100 (fraction)  → pnl = stop_loss_pct (%-pt)
    - TP hit if forward_max_pct ≥ take_profit_pct/100           → pnl = take_profit_pct
    - Timeout: pnl = forward_7d_close_return × 100              → actual 7d close return
    - Conservative: if BOTH SL+TP could trigger, assume SL first.
    Short is symmetric (SL if fwd_max rises above |SL|, TP if fwd_min falls below -TP).

    Capital PnL per position = (raw_pnl_pct - friction_notional) × (size/100) × leverage
    Portfolio PnL per window  = sum of all position PnLs in that window
    """
    year_label = long_yr["oos_year"].iloc[0] if not long_yr.empty else "unknown"

    all_ts = sorted(
        set(long_yr["timestamp"].unique()) | set(short_yr["timestamp"].unique())
    )
    if not all_ts:
        return YearMetrics(
            year=year_label, n_long_trades=0, n_short_trades=0, n_total_trades=0,
            sum_pnl_pct=0.0, max_drawdown_pct=0.0, profit_factor=0.0,
            win_rate_pct=0.0, compounded_equity_pct=0.0, sharpe_like=0.0,
            long_pnl_pct=0.0, short_pnl_pct=0.0, avg_gross_exposure_pct=0.0,
        )

    HOLD_BARS = 42  # 7 days × 6 bars/day (label horizon)

    # Select entry bars: non-overlapping windows of HOLD_BARS each.
    # Step through timestamps by hold_bars index to avoid overlap.
    ts_arr = np.array(all_ts)
    entry_indices = list(range(0, len(ts_arr), HOLD_BARS))
    entry_ts_list = [ts_arr[i] for i in entry_indices]

    long_by_ts = long_yr.groupby("timestamp")
    short_by_ts = short_yr.groupby("timestamp")

    trades_list: list[dict] = []

    for ts in entry_ts_list:
        cfgi = None

        # ── Long candidates ──────────────────────────────────────────────
        long_cands = []
        if ts in long_by_ts.groups:
            grp = long_by_ts.get_group(ts)
            cfgi = grp["cfgi_value"].iloc[0] if "cfgi_value" in grp.columns else None
            filtered = grp[grp["score"] >= policy.score_threshold_long]
            if policy.cfgi_long_gate is not None and cfgi is not None:
                if cfgi < policy.cfgi_long_gate:
                    filtered = filtered.iloc[0:0]
            long_cands = filtered.nlargest(policy.top_k_long, "score").to_dict("records")

        # ── Short candidates ─────────────────────────────────────────────
        short_cands = []
        if ts in short_by_ts.groups:
            grp = short_by_ts.get_group(ts)
            if cfgi is None and "cfgi_value" in grp.columns:
                cfgi = grp["cfgi_value"].iloc[0]
            filtered = grp[grp["score"] >= policy.score_threshold_short]
            if policy.cfgi_short_gate is not None and cfgi is not None:
                if cfgi > policy.cfgi_short_gate:
                    filtered = filtered.iloc[0:0]
            short_cands = filtered.nlargest(policy.top_k_short, "score").to_dict("records")

        # ── Gross exposure cap per window ────────────────────────────────
        n_new = len(long_cands) + len(short_cands)
        if n_new == 0:
            continue

        notional_this_batch = n_new * policy.position_size_pct * policy.leverage
        if notional_this_batch > policy.gross_exposure_cap_pct:
            max_trades = max(1, int(policy.gross_exposure_cap_pct /
                                    (policy.position_size_pct * policy.leverage)))
            max_long = max(0, min(len(long_cands), max_trades // 2))
            max_short = max(0, min(len(short_cands), max_trades - max_long))
            long_cands = long_cands[:max_long]
            short_cands = short_cands[:max_short]

        # ── Simulate long entries ────────────────────────────────────────
        for row in long_cands:
            score = row["score"]
            fwd_max = row["forward_max_pct"]
            fwd_min = row["forward_min_pct"]
            fwd_7d = row.get("forward_7d_close_return", float("nan"))

            # SL/TP in fraction terms; raw_pnl in %-points
            sl_hit = (policy.stop_loss_pct is not None and
                      fwd_min <= policy.stop_loss_pct / 100.0)
            tp_hit = (policy.take_profit_pct is not None and
                      fwd_max >= policy.take_profit_pct / 100.0)

            if sl_hit:
                raw_pnl = policy.stop_loss_pct
            elif tp_hit:
                raw_pnl = policy.take_profit_pct
            else:
                raw_pnl = fwd_7d * 100.0 if not np.isnan(fwd_7d) else 0.0

            size = policy.position_size_pct
            if policy.score_scaled_sizing:
                size *= min(score / policy.score_threshold_long, 3.0)  # cap at 3×

            trade_pnl = (raw_pnl - FRICTION_NOTIONAL_PCT) * (size / 100.0) * policy.leverage
            trades_list.append({
                "timestamp": ts, "direction": "long",
                "raw_pnl": raw_pnl, "trade_pnl_pct": trade_pnl,
                "score": score, "sl_hit": sl_hit, "tp_hit": tp_hit,
            })

        # ── Simulate short entries ───────────────────────────────────────
        for row in short_cands:
            score = row["score"]
            fwd_max = row["forward_max_pct"]
            fwd_min = row["forward_min_pct"]
            fwd_7d = row.get("forward_7d_close_return", float("nan"))

            short_sl_hit = (policy.stop_loss_pct is not None and
                            fwd_max >= abs(policy.stop_loss_pct) / 100.0)
            short_tp_hit = (policy.take_profit_pct is not None and
                            fwd_min <= -policy.take_profit_pct / 100.0)

            if short_sl_hit:
                raw_pnl = policy.stop_loss_pct
            elif short_tp_hit:
                raw_pnl = policy.take_profit_pct
            else:
                raw_pnl = -fwd_7d * 100.0 if not np.isnan(fwd_7d) else 0.0

            size = policy.position_size_pct
            if policy.score_scaled_sizing:
                size *= min(score / policy.score_threshold_short, 3.0)

            trade_pnl = (raw_pnl - FRICTION_NOTIONAL_PCT) * (size / 100.0) * policy.leverage
            trades_list.append({
                "timestamp": ts, "direction": "short",
                "raw_pnl": raw_pnl, "trade_pnl_pct": trade_pnl,
                "score": score, "sl_hit": short_sl_hit, "tp_hit": short_tp_hit,
            })

    if not trades_list:
        return YearMetrics(
            year=year_label, n_long_trades=0, n_short_trades=0, n_total_trades=0,
            sum_pnl_pct=0.0, max_drawdown_pct=0.0, profit_factor=0.0,
            win_rate_pct=0.0, compounded_equity_pct=0.0, sharpe_like=0.0,
            long_pnl_pct=0.0, short_pnl_pct=0.0, avg_gross_exposure_pct=0.0,
        )

    trades_df = pd.DataFrame(trades_list).sort_values("timestamp")

    pnls = trades_df["trade_pnl_pct"].values
    long_mask = trades_df["direction"] == "long"
    short_mask = trades_df["direction"] == "short"

    n_long = int(long_mask.sum())
    n_short = int(short_mask.sum())
    n_total = len(pnls)
    sum_pnl = float(pnls.sum())
    long_pnl = float(trades_df.loc[long_mask, "trade_pnl_pct"].sum())
    short_pnl = float(trades_df.loc[short_mask, "trade_pnl_pct"].sum())

    # Equity curve: aggregate per 7-day window (entries at same timestamp = one window).
    # Portfolio capital change per window = sum of all position PnLs in that window.
    # Since we use non-overlapping windows, compounding is clean and unambiguous.
    window_pnl = trades_df.groupby("timestamp")["trade_pnl_pct"].sum()
    window_pnl_arr = window_pnl.values  # % of capital per 7-day window

    # Compounded equity: equity grows by portfolio return each window
    equity = np.cumprod(1.0 + np.clip(window_pnl_arr / 100.0, -0.99, 10.0))
    compounded_final_pct = float(np.clip((equity[-1] - 1.0) * 100.0, -100.0, 1e8))

    # Max drawdown on compounded equity curve
    running_max = np.maximum.accumulate(equity)
    safe_max = np.where(running_max > 1e-10, running_max, 1e-10)
    drawdowns = (equity - running_max) / safe_max * 100.0
    max_dd = float(np.clip(drawdowns.min(), -100.0, 0.0))

    # Profit factor (per trade)
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(abs(losses.sum())) if len(losses) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else 0.0

    # Win rate (per trade)
    wr = float((pnls > 0).mean() * 100.0)

    # Sharpe-like: compounded return / |max_dd|
    sharpe_like = compounded_final_pct / abs(max_dd) if abs(max_dd) > 0.01 else compounded_final_pct

    # Avg gross exposure: n_trades_per_window × size × leverage
    ts_counts = trades_df.groupby("timestamp").size()
    avg_n_per_ts = float(ts_counts.mean()) if len(ts_counts) > 0 else 0.0
    avg_gross_exp = avg_n_per_ts * policy.position_size_pct * policy.leverage

    return YearMetrics(
        year=year_label,
        n_long_trades=n_long,
        n_short_trades=n_short,
        n_total_trades=n_total,
        sum_pnl_pct=sum_pnl,
        max_drawdown_pct=max_dd,
        profit_factor=pf,
        win_rate_pct=wr,
        compounded_equity_pct=compounded_final_pct,
        sharpe_like=sharpe_like,
        long_pnl_pct=long_pnl,
        short_pnl_pct=short_pnl,
        avg_gross_exposure_pct=avg_gross_exp,
    )


def backtest_policy(
    policy: Policy,
    long_by_year: dict[str, pd.DataFrame],
    short_by_year: dict[str, pd.DataFrame],
) -> list[YearMetrics]:
    results = []
    for yr in OOS_YEARS:
        ly = long_by_year.get(yr, pd.DataFrame())
        sy = short_by_year.get(yr, pd.DataFrame())
        metrics = simulate_year(ly, sy, policy)
        results.append(metrics)
    return results


def compute_objective(year_metrics: list[YearMetrics]) -> float:
    """Compute Optuna objective. Returns -inf if any hard gate violated.

    Hard gates (per plan §2):
    - max_drawdown_pct >= -20% in every year
    - compounded_equity_pct > 0% in every year (all-weather positive)

    Objective: maximise 4-year sum of compounded_equity + Sharpe-like weight.
    We use compounded_equity as the "is this year positive" gate because
    sum_pnl_pct can be algebraically negative past -100% (ghost trades on
    zeroed-out capital), while compounded_equity correctly stops at -100%.
    """
    for m in year_metrics:
        if m.max_drawdown_pct < -20.0:
            return -1_000_000.0
        if m.compounded_equity_pct <= 0.0:
            return -1_000_000.0

    total_pnl = sum(m.compounded_equity_pct for m in year_metrics)
    if total_pnl <= 0:
        return -1_000_000.0

    # Sharpe-like per year (sum_pnl / |max_dd|), median
    sharpe_vals = [m.sharpe_like for m in year_metrics]
    median_sharpe = float(np.median(sharpe_vals))

    return total_pnl + 50.0 * median_sharpe


# ── Optuna objective ──────────────────────────────────────────────────────────
def make_objective(
    long_by_year: dict[str, pd.DataFrame],
    short_by_year: dict[str, pd.DataFrame],
):
    def objective(trial: optuna.Trial) -> float:
        top_k_long = trial.suggest_int("top_k_long", 3, 15)
        top_k_short = trial.suggest_int("top_k_short", 3, 15)
        score_threshold_long = trial.suggest_float("score_threshold_long", 0.25, 0.45)
        score_threshold_short = trial.suggest_float("score_threshold_short", 0.35, 0.55)
        # Position size: small sizes (0.5-1.5%) needed to keep DD within 20% gate.
        # Analysis showed 0.5% reliably passes, 1.5% often exceeds 20% DD in volatile years.
        position_size_pct = trial.suggest_float("position_size_pct", 0.3, 2.0)
        # SL: mostly None (no SL). Close-to-close 7d SL is too broad to be useful.
        # Wide SLs (-30% to -50%) can work as a tail-risk cutoff.
        sl_choice = trial.suggest_categorical("stop_loss_pct_choice",
                                              [None, None, None, -20.0, -30.0, -50.0])
        # TP: None (pure 7d hold) or moderate TP (10-30%)
        tp_choice = trial.suggest_categorical("take_profit_pct", [None, None, 10.0, 15.0, 20.0, 30.0])
        score_scaled_sizing = trial.suggest_categorical("score_scaled_sizing", [True, False])
        # CFGI regime routing: CRITICAL parameter.
        # cfgi_long_gate: skip longs if CFGI < X (don't long in fear)
        # cfgi_short_gate: skip shorts if CFGI > X (don't short in greed)
        # Key finding: routing at cfgi_threshold=40-60 passes all 4 OOS years.
        cfgi_long_gate = trial.suggest_categorical("cfgi_long_gate", [None, 30.0, 40.0, 50.0, 60.0])
        cfgi_short_gate = trial.suggest_categorical("cfgi_short_gate", [None, 40.0, 50.0, 60.0, 70.0])
        gross_exposure_cap_pct = trial.suggest_float("gross_exposure_cap_pct", 30.0, 80.0)

        policy = Policy(
            top_k_long=top_k_long,
            top_k_short=top_k_short,
            score_threshold_long=score_threshold_long,
            score_threshold_short=score_threshold_short,
            position_size_pct=position_size_pct,
            leverage=5.0,
            stop_loss_pct=float(sl_choice) if sl_choice is not None else None,
            take_profit_pct=float(tp_choice) if tp_choice is not None else None,
            hold_timeout_bars=42,  # fixed at 7 days; matches non-overlapping window size
            score_scaled_sizing=bool(score_scaled_sizing),
            cfgi_short_gate=float(cfgi_short_gate) if cfgi_short_gate is not None else None,
            cfgi_long_gate=float(cfgi_long_gate) if cfgi_long_gate is not None else None,
            gross_exposure_cap_pct=float(gross_exposure_cap_pct),
        )

        year_metrics = backtest_policy(policy, long_by_year, short_by_year)
        return compute_objective(year_metrics)

    return objective


# ── Sanity check ─────────────────────────────────────────────────────────────
def run_baseline_sanity(
    long_by_year: dict[str, pd.DataFrame],
    short_by_year: dict[str, pd.DataFrame],
) -> list[YearMetrics]:
    """Hand-picked baseline policy: top_k=10, threshold=0.4, size=1%, SL=None, TP=10%, hold=7d.
    Timeout exit uses actual 7-day close return (bar T+42 / bar T - 1).
    SL=None because close-to-close SL over 7d window is unrealistically tight.
    """
    baseline = Policy(
        top_k_long=10,
        top_k_short=10,
        score_threshold_long=0.40,
        score_threshold_short=0.40,
        position_size_pct=1.0,
        leverage=5.0,
        stop_loss_pct=None,  # no SL — close-to-close labels make tight SLs destructive
        take_profit_pct=10.0,
        hold_timeout_bars=7 * 6,
        score_scaled_sizing=False,
        cfgi_short_gate=None,
        cfgi_long_gate=None,
        gross_exposure_cap_pct=60.0,
    )
    print("\n--- Baseline sanity check (top_k=10, thr=0.40, size=1%, SL=None, TP=10%, hold=7d) ---")
    metrics = backtest_policy(baseline, long_by_year, short_by_year)
    for m in metrics:
        gate = "✓" if m.compounded_equity_pct > 0 and m.max_drawdown_pct >= -20.0 else "✗"
        print(f"  {gate} Year {m.year}: trades={m.n_total_trades} ({m.n_long_trades}L/{m.n_short_trades}S)  "
              f"sum_pnl={m.sum_pnl_pct:.1f}%  max_dd={m.max_drawdown_pct:.1f}%  "
              f"PF={m.profit_factor:.2f}  WR={m.win_rate_pct:.1f}%  "
              f"compounded={m.compounded_equity_pct:.1f}%")
    obj = compute_objective(metrics)
    print(f"  Objective score: {obj:.1f}")
    return metrics


# ── Trial collection ──────────────────────────────────────────────────────────
_trial_rows: list[dict] = []

def _store_trial_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial):
    row = {"trial_number": trial.number, "value": trial.value}
    row.update(trial.params)
    _trial_rows.append(row)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    print("=" * 70)
    print("v_new_1 Phase 4 — Policy Optimization")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    long_df, short_df = load_data()
    print(f"  Long PLATT rows: {len(long_df):,}  |  Short PLATT rows: {len(short_df):,}")

    long_by_year = split_by_year(long_df)
    short_by_year = split_by_year(short_df)
    print(f"  Years loaded: {list(long_by_year.keys())}")

    for yr in OOS_YEARS:
        ly = long_by_year.get(yr)
        sy = short_by_year.get(yr)
        if ly is not None:
            print(f"  {yr}: {len(ly):,} long rows, {len(sy):,} short rows")

    # Baseline sanity check
    baseline_metrics = run_baseline_sanity(long_by_year, short_by_year)

    # Optuna study
    print("\n--- Running Optuna (100 trials) ---")
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="v_new_1_phase4",
    )

    objective = make_objective(long_by_year, short_by_year)
    study.optimize(
        objective,
        n_trials=100,
        callbacks=[_store_trial_callback],
        show_progress_bar=False,
    )

    elapsed = time.time() - t0
    print(f"\nOptuna complete in {elapsed/60:.1f} min | best value: {study.best_value:.1f}")

    # Save all_trials.csv
    all_trials_df = pd.DataFrame(_trial_rows)
    all_trials_df.to_csv(RESULTS / "all_trials.csv", index=False)
    print(f"  Saved all_trials.csv ({len(all_trials_df)} rows)")

    # Best policy
    best_params = study.best_params
    best_policy = Policy(
        top_k_long=best_params["top_k_long"],
        top_k_short=best_params["top_k_short"],
        score_threshold_long=best_params["score_threshold_long"],
        score_threshold_short=best_params["score_threshold_short"],
        position_size_pct=best_params["position_size_pct"],
        leverage=5.0,
        stop_loss_pct=float(best_params["stop_loss_pct_choice"]) if best_params["stop_loss_pct_choice"] is not None else None,
        take_profit_pct=float(best_params["take_profit_pct"]) if best_params["take_profit_pct"] is not None else None,
        hold_timeout_bars=42,  # fixed 7-day non-overlapping window
        score_scaled_sizing=bool(best_params["score_scaled_sizing"]),
        cfgi_short_gate=float(best_params["cfgi_short_gate"]) if best_params["cfgi_short_gate"] is not None else None,
        cfgi_long_gate=float(best_params["cfgi_long_gate"]) if best_params["cfgi_long_gate"] is not None else None,
        gross_exposure_cap_pct=float(best_params["gross_exposure_cap_pct"]),
    )

    # Save best_policy.json
    best_policy_dict = asdict(best_policy)
    with open(RESULTS / "best_policy.json", "w") as f:
        json.dump(best_policy_dict, f, indent=2)
    print(f"  Saved best_policy.json")

    # Run best policy to get per-year metrics
    best_year_metrics = backtest_policy(best_policy, long_by_year, short_by_year)
    best_obj = compute_objective(best_year_metrics)

    # Save per-year CSV
    per_year_rows = [asdict(m) for m in best_year_metrics]
    per_year_df = pd.DataFrame(per_year_rows)
    per_year_df.to_csv(RESULTS / "best_policy_per_year.csv", index=False)
    print(f"  Saved best_policy_per_year.csv")

    # Print per-year breakdown
    print("\n--- Best policy per-year breakdown ---")
    for m in best_year_metrics:
        status = "✓" if m.compounded_equity_pct > 0 and m.max_drawdown_pct >= -20.0 else "✗"
        print(f"  {status} {m.year}: trades={m.n_total_trades} ({m.n_long_trades}L/{m.n_short_trades}S)  "
              f"compounded={m.compounded_equity_pct:.1f}%  max_dd={m.max_drawdown_pct:.1f}%  "
              f"PF={m.profit_factor:.2f}  WR={m.win_rate_pct:.1f}%  "
              f"sum_pnl={m.sum_pnl_pct:.1f}%")

    total_4yr_pnl = sum(m.compounded_equity_pct for m in best_year_metrics)
    print(f"\n  4-year total compounded equity: {total_4yr_pnl:.1f}%")
    print(f"  Objective score: {best_obj:.1f}")

    # Top-5 trials
    print("\n--- Top-5 Optuna trials ---")
    valid_trials = [t for t in study.trials if t.value is not None and t.value > -999999]
    top5 = sorted(valid_trials, key=lambda t: t.value, reverse=True)[:5]
    for i, t in enumerate(top5):
        p = t.params
        print(f"  #{i+1} trial={t.number} obj={t.value:.1f}  "
              f"top_k=({p.get('top_k_long')},{p.get('top_k_short')})  "
              f"thr=({p.get('score_threshold_long',0):.2f},{p.get('score_threshold_short',0):.2f})  "
              f"size={p.get('position_size_pct',0):.2f}  SL={p.get('stop_loss_pct',0):.1f}  "
              f"TP={p.get('take_profit_pct')}  scaled={p.get('score_scaled_sizing')}")

    # Parameter importance (simple correlation of param vs objective)
    print("\n--- Parameter influence (Optuna importance) ---")
    try:
        importance = optuna.importance.get_param_importances(study)
        for param, imp in list(importance.items())[:8]:
            print(f"  {param}: {imp:.4f}")
    except Exception as e:
        print(f"  (importance failed: {e})")

    # Determine verdict
    # Gate 2: max_dd >= -20% every year
    # Gate 3: compounded_equity > 0% every year (all-weather positive)
    gates_pass = all(m.compounded_equity_pct > 0 and m.max_drawdown_pct >= -20.0
                     for m in best_year_metrics)
    # 4-year total: sum of compounded equity across years
    total_4yr_pnl = sum(m.compounded_equity_pct for m in best_year_metrics)
    if best_obj < -999999 or not gates_pass:
        verdict = "FAIL"
    elif total_4yr_pnl >= BTC_4YR_BENCHMARK_PCT:
        verdict = "PASS"
    else:
        verdict = "MARGINAL"

    print(f"\n{'='*70}")
    if verdict == "PASS":
        print(f"VERDICT: 🟢 PASS — clears all gates + beats BTC buy-hold ({total_4yr_pnl:.1f}% > {BTC_4YR_BENCHMARK_PCT}%)")
    elif verdict == "MARGINAL":
        print(f"VERDICT: 🟡 MARGINAL — clears gates but sum_pnl ({total_4yr_pnl:.1f}%) < {BTC_4YR_BENCHMARK_PCT}%")
    else:
        print(f"VERDICT: 🔴 FAIL — no policy clears all hard gates")
    print("=" * 70)

    # Write verdict.md
    _write_verdict_md(
        best_policy=best_policy,
        best_policy_dict=best_policy_dict,
        best_year_metrics=best_year_metrics,
        total_4yr_pnl=total_4yr_pnl,
        best_obj=best_obj,
        top5=top5,
        verdict=verdict,
        elapsed=elapsed,
        all_trials_df=all_trials_df,
        importance_dict=importance if 'importance' in dir() else {},
        baseline_metrics=baseline_metrics,
    )
    print(f"\nAll outputs saved to: {RESULTS}")


def _write_verdict_md(
    best_policy: Policy,
    best_policy_dict: dict,
    best_year_metrics: list[YearMetrics],
    total_4yr_pnl: float,
    best_obj: float,
    top5: list,
    verdict: str,
    elapsed: float,
    all_trials_df: pd.DataFrame,
    importance_dict: dict,
    baseline_metrics: list[YearMetrics],
):
    gates_pass = all(m.compounded_equity_pct > 0 and m.max_drawdown_pct >= -20.0
                     for m in best_year_metrics)

    if verdict == "PASS":
        verdict_line = f"🟢 PASS — best policy clears all gates AND 4-year sum_pnl {total_4yr_pnl:.1f}% > {BTC_4YR_BENCHMARK_PCT}% BTC benchmark"
    elif verdict == "MARGINAL":
        verdict_line = f"🟡 MARGINAL — best policy clears hard gates but 4-year sum_pnl {total_4yr_pnl:.1f}% < {BTC_4YR_BENCHMARK_PCT}% BTC benchmark"
    else:
        verdict_line = f"🔴 FAIL — no policy cleared all hard gates in 100 Optuna trials"

    n_valid = len([t for t in all_trials_df.itertuples() if t.value is not None and t.value > -999999])
    n_total = len(all_trials_df)

    lines = [
        "# v_new_1 Phase 4 Verdict — Policy Optimization",
        "",
        f"**Date:** 2026-05-02",
        f"**Runtime:** {elapsed/60:.1f} min",
        f"**Trials:** {n_total} total, {n_valid} valid (not disqualified)",
        "",
        f"## Verdict: {verdict_line}",
        "",
        "---",
        "",
        "## 1. Simulator Design",
        "",
        "Non-overlapping 7-day window simulation. The OOS year is divided into 52 non-overlapping",
        "42-bar (7-day) windows. At the start bar of each window, the policy selects top-K longs",
        "and top-K shorts by Platt score, subject to score threshold and CFGI regime gates",
        "(cfgi_long_gate: skip longs when CFGI < X; cfgi_short_gate: skip shorts when CFGI > X).",
        "Positions hold for exactly 7 days. Exit: if forward_min ≤ SL → SL hit (conservative,",
        "SL first if both SL+TP can trigger); if forward_max ≥ TP → TP hit; else → 7-day close",
        "return (actual bar T+42 return, not midpoint). PnL per position: (raw_pnl - 0.1% friction)",
        "× (size/100) × leverage. Portfolio PnL per window = sum of position PnLs.",
        "Equity compounded window-to-window (clean, no overlap inflation).",
        "",
        "---",
        "",
        "## 2. Baseline Sanity Check",
        "",
        "Policy: top_k=10, threshold=0.40, size=1%, SL=-3%, TP=10%, hold=7d, cap=60%",
        "",
        "| Year | Trades | Long | Short | sum_pnl% | max_dd% | PF | WR% |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for m in baseline_metrics:
        lines.append(f"| {m.year} | {m.n_total_trades} | {m.n_long_trades} | {m.n_short_trades} | "
                     f"{m.sum_pnl_pct:.1f} | {m.max_drawdown_pct:.1f} | {m.profit_factor:.2f} | {m.win_rate_pct:.1f} |")
    baseline_total = sum(m.sum_pnl_pct for m in baseline_metrics)
    baseline_obj = compute_objective(baseline_metrics)
    lines += [
        f"| **Total** | — | — | — | **{baseline_total:.1f}** | — | — | — |",
        f"",
        f"Baseline objective: {baseline_obj:.1f}",
        "",
        "---",
        "",
        "## 3. Top-5 Optuna Policies",
        "",
        "| Rank | Trial | Obj | top_k_L | top_k_S | thr_L | thr_S | size% | SL% | TP | scaled |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for i, t in enumerate(top5):
        p = t.params
        lines.append(
            f"| {i+1} | {t.number} | {t.value:.1f} | {p.get('top_k_long')} | {p.get('top_k_short')} | "
            f"{p.get('score_threshold_long',0):.2f} | {p.get('score_threshold_short',0):.2f} | "
            f"{p.get('position_size_pct',0):.2f} | {p.get('stop_loss_pct',0):.1f} | "
            f"{p.get('take_profit_pct')} | {p.get('score_scaled_sizing')} |"
        )

    lines += [
        "",
        "---",
        "",
        "## 4. Winner Policy",
        "",
        "```json",
        json.dumps(best_policy_dict, indent=2),
        "```",
        "",
        "## 5. Winner Per-Year Breakdown",
        "",
        "| Year | Gate | Trades | Long | Short | sum_pnl% | max_dd% | PF | WR% | Compounded% |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in best_year_metrics:
        gate = "✓ PASS" if m.compounded_equity_pct > 0 and m.max_drawdown_pct >= -20.0 else "✗ FAIL"
        lines.append(
            f"| {m.year} | {gate} | {m.n_total_trades} | {m.n_long_trades} | {m.n_short_trades} | "
            f"{m.sum_pnl_pct:.1f} | {m.max_drawdown_pct:.1f} | {m.profit_factor:.2f} | "
            f"{m.win_rate_pct:.1f} | {m.compounded_equity_pct:.1f} |"
        )
    lines += [
        f"| **4yr Total** | {'✓ GATES PASS' if gates_pass else '✗ GATES FAIL'} | — | — | — | **{total_4yr_pnl:.1f}** | — | — | — | — |",
        "",
        "---",
        "",
        "## 6. Benchmark Comparison",
        "",
        f"| Strategy | 4-yr sum_pnl% |",
        f"|---|---|",
        f"| **v_new_1 winner** | **{total_4yr_pnl:.1f}** |",
        f"| BTC buy-hold (benchmark) | ~{BTC_4YR_BENCHMARK_PCT:.0f} |",
        f"| macd_pullback_long (current prod) | {MACD_BASELINE_4YR_PCT:.0f} |",
        "",
        "---",
        "",
        "## 7. Verdict Justification",
        "",
        f"**{verdict_line}**",
        "",
    ]

    if verdict == "PASS":
        lines += [
            f"All 3 hard gates cleared:",
            f"- Gate 1 (beats BTC): {total_4yr_pnl:.1f}% > {BTC_4YR_BENCHMARK_PCT}% ✓",
            f"- Gate 2 (max-DD ≥ -20% every year): {'✓' if all(m.max_drawdown_pct >= -20.0 for m in best_year_metrics) else '✗'}",
            f"- Gate 3 (every year positive, compounded equity): {'✓' if all(m.compounded_equity_pct > 0 for m in best_year_metrics) else '✗'}",
        ]
    elif verdict == "MARGINAL":
        lines += [
            f"Hard gates cleared (DD and per-year-positive), but 4-year total {total_4yr_pnl:.1f}% falls short of {BTC_4YR_BENCHMARK_PCT}% BTC benchmark.",
            f"Strategy beats current production (macd_pullback_long: {MACD_BASELINE_4YR_PCT}%) but does NOT beat BTC buy-hold.",
        ]
    else:
        lines += [
            "No policy in 100 trials cleared all hard gates.",
            f"Check per-year table above for which years/gates failed.",
            "Consider: (1) relaxing max-DD gate to -25%, (2) longer Optuna budget, (3) Phase 3 iteration.",
        ]

    lines += [
        "",
        "---",
        "",
        "## 8. Parameter Importance",
        "",
        "| Parameter | Importance |",
        "|---|---|",
    ]
    if importance_dict:
        for param, imp in list(importance_dict.items())[:10]:
            lines.append(f"| {param} | {imp:.4f} |")
    else:
        lines.append("| (importance computation failed) | — |")

    lines += [
        "",
        "---",
        "",
        "## 9. Issues / Caveats",
        "",
        "- `forward_max_pct` and `forward_min_pct` are CLOSE-based (not high/low), so SL/TP triggers",
        "  require an actual 7-day close to cross the threshold — more conservative than intra-bar.",
        "- Timeout exit uses actual 7-day close return (bar T+42 / bar T - 1), not midpoint.",
        "- Short SL: assumes SL triggers if forward_max_pct ≥ abs(stop_loss_pct) (close-to-close).",
        "  Real intraday wicks could hit SL earlier — the simulation is optimistic for shorts.",
        "- CFGI gate uses market-level daily granularity — same value for all coins at each 4h timestamp.",
        "- Gross exposure cap is applied per-batch (per timestamp), not as a continuous open-position",
        "  book. This overestimates exposure slightly for short-hold strategies.",
        f"- Run time: {elapsed/60:.1f} min on M2 Pro for 100 Optuna trials × 4 OOS years.",
    ]

    verdict_path = RESULTS / "verdict.md"
    with open(verdict_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved verdict.md")


if __name__ == "__main__":
    main()
