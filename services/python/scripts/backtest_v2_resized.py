"""v2+ML Resizing Analysis — 3-run layered attribution.

Re-simulates the v2+ML OOT trade stream under 3 sizing configurations:
  A (baseline)     flat 25% notional, MAX_CONCURRENT=5
  B (+Lever 1)     half-Kelly by ml_prob bin, MAX_CONCURRENT=5
  C (+Lever 1+4)   half-Kelly by ml_prob bin, MAX_CONCURRENT=10

Outputs PnL / PF / drawdown per scenario + monthly comparison +
threshold-Kelly crosscheck. Pure research, no prod impact.

Spec: docs/superpowers/specs/2026-04-23-v2-resizing-analysis-design.md

Run from services/python/:
    python scripts/backtest_v2_resized.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

sys.path.insert(0, ".")

# --- CONSTANTS ---
RANDOM_SEED = 42

# Source
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)
SOURCE_TABLE = "scanner_short_v2_ml_filtered"

# Sizing (matches CLAUDE.md trading rules)
START_EQUITY_USD = 600.0
LEVERAGE = 5
BASELINE_EQUITY_FRACTION = 0.05  # 5% of equity per trade = 25% notional at 5x
KELLY_MULTIPLIER = 0.5           # half-Kelly, industry standard, hardcoded
KELLY_FRACTION_FLOOR = 0.02      # min 2% equity per trade
KELLY_FRACTION_CAP = 0.15        # max 15% equity per trade
ML_THRESHOLD = 0.70              # dashboard threshold (see spec for rationale)
ROUND_TRIP_FEE = 0.0012          # 0.06% taker x 2 legs (Gate.io futures)

# OOT split
HOLD_OUT_MONTHS = 3

# Scenarios
MAX_CONCURRENT_A = 5
MAX_CONCURRENT_B = 5
MAX_CONCURRENT_C = 10

# Kelly bins (edges inclusive-left, exclusive-right; last bin captures 0.95+)
KELLY_BIN_EDGES = [0.70, 0.75, 0.80, 0.85, 0.90, 1.01]

# Monte Carlo
MC_ITER = 1000
MC_SEED = 42

# Threshold crosscheck
CROSSCHECK_THRESHOLDS = [0.70, 0.72, 0.75, 0.78, 0.80, 0.85, 0.88, 0.90]

# Bar size for exit_time derivation
BAR_MINUTES = 15


def build_kelly_table(pre_oot_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-bin half-Kelly fraction from pre-OOT trades.

    For each [bin_low, bin_high) bin in KELLY_BIN_EDGES:
      p = win_rate_in_bin (empirical)
      b = avg_winner_pnl / |avg_loser_pnl|  (empirical)
      f_raw = p - (1-p)/b                    (full Kelly)
      f_half = f_raw * KELLY_MULTIPLIER
      f_capped = clip(f_half, FLOOR, CAP)

    Empty bins (or bins with no losers / no winners) fall back to FLOOR.
    Returns DataFrame with columns:
      bin_low, bin_high, trades, p, b, f_raw, f_half, f_capped
    """
    rows = []
    for lo, hi in zip(KELLY_BIN_EDGES[:-1], KELLY_BIN_EDGES[1:], strict=False):
        mask = (pre_oot_df["ml_prob"] >= lo) & (pre_oot_df["ml_prob"] < hi)
        sub = pre_oot_df.loc[mask, "pnl_pct"]
        trades = len(sub)
        if trades == 0:
            rows.append({
                "bin_low": lo, "bin_high": hi, "trades": 0,
                "p": 0.0, "b": 0.0,
                "f_raw": 0.0, "f_half": 0.0, "f_capped": KELLY_FRACTION_FLOOR,
            })
            continue
        wins = sub[sub > 0]
        losses = sub[sub <= 0]
        p = len(wins) / trades
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        if avg_loss == 0 or len(losses) == 0:
            # No losers in this bin. Full Kelly is +inf; half still clips to CAP.
            f_raw, f_half = 1.0, KELLY_MULTIPLIER
            b = float("inf")
        elif avg_win == 0:
            f_raw, f_half, b = 0.0, 0.0, 0.0
        else:
            b = avg_win / avg_loss
            f_raw = p - (1.0 - p) / b
            f_half = f_raw * KELLY_MULTIPLIER
        f_capped = float(np.clip(f_half, KELLY_FRACTION_FLOOR, KELLY_FRACTION_CAP))
        rows.append({
            "bin_low": lo, "bin_high": hi, "trades": trades,
            "p": p, "b": b, "f_raw": f_raw, "f_half": f_half, "f_capped": f_capped,
        })
    return pd.DataFrame(rows)


def _kelly_fraction_for(ml_prob: float, kelly_table: pd.DataFrame) -> float:
    """Map a single ml_prob to its bin's f_capped. Defensive floor for out-of-range."""
    for _, row in kelly_table.iterrows():
        if row["bin_low"] <= ml_prob < row["bin_high"]:
            return float(row["f_capped"])
    return KELLY_FRACTION_FLOOR


def simulate(
    trades_df: pd.DataFrame,
    kelly_table: pd.DataFrame,
    max_concurrent: int,
    label: str,
) -> tuple[pd.DataFrame, dict]:
    """Replay trades under a given sizing/concurrency rule.

    Input `trades_df` must have columns: signal_time, ml_prob, pnl_pct, bars_held.
    Sort order inside the function is by signal_time.

    Sizing per trade: notional = START_EQUITY_USD * kelly_fraction * LEVERAGE.
    Fees applied as pnl_net_pct = pnl_pct - ROUND_TRIP_FEE before converting to USD.

    Concurrency: prune finished positions, skip new if len(open) >= max_concurrent.

    Returns (ledger_df, metrics_dict).
    ledger_df columns: signal_time, ml_prob, pnl_pct_gross, pnl_pct_net,
                       kelly_fraction, notional_usd, pnl_usd, exit_time, taken
    metrics_dict: trades, skipped_concurrency, wr, pf_net, avg_pnl_pct_net,
                  total_pnl_usd, total_return_pct, max_dd_pct, label
    """
    df = trades_df.sort_values("signal_time").reset_index(drop=True)
    bar_td = pd.Timedelta(minutes=BAR_MINUTES)

    ledger_rows = []
    open_positions: list[pd.Timestamp] = []
    skipped = 0

    for _, t in df.iterrows():
        sig_time = t["signal_time"]
        # Prune finished positions
        open_positions = [exit_t for exit_t in open_positions if exit_t > sig_time]

        f = _kelly_fraction_for(t["ml_prob"], kelly_table)
        notional = START_EQUITY_USD * f * LEVERAGE
        pnl_net = t["pnl_pct"] - ROUND_TRIP_FEE
        pnl_usd = notional * pnl_net
        exit_time = sig_time + bar_td * int(t["bars_held"])

        if len(open_positions) >= max_concurrent:
            skipped += 1
            ledger_rows.append({
                "signal_time": sig_time, "ml_prob": t["ml_prob"],
                "pnl_pct_gross": t["pnl_pct"], "pnl_pct_net": pnl_net,
                "kelly_fraction": f, "notional_usd": notional,
                "pnl_usd": 0.0, "exit_time": exit_time, "taken": False,
            })
            continue

        open_positions.append(exit_time)
        ledger_rows.append({
            "signal_time": sig_time, "ml_prob": t["ml_prob"],
            "pnl_pct_gross": t["pnl_pct"], "pnl_pct_net": pnl_net,
            "kelly_fraction": f, "notional_usd": notional,
            "pnl_usd": pnl_usd, "exit_time": exit_time, "taken": True,
        })

    ledger = pd.DataFrame(ledger_rows)
    taken = ledger[ledger["taken"]]

    # Metrics on taken trades only
    if len(taken) == 0:
        metrics = {
            "label": label, "trades": 0, "skipped_concurrency": skipped,
            "wr": 0.0, "pf_net": 0.0, "avg_pnl_pct_net": 0.0,
            "total_pnl_usd": 0.0, "total_return_pct": 0.0, "max_dd_pct": 0.0,
        }
        return ledger, metrics

    wins = taken[taken["pnl_usd"] > 0]["pnl_usd"]
    losses = taken[taken["pnl_usd"] <= 0]["pnl_usd"]
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    total_pnl_usd = taken["pnl_usd"].sum()

    # Max drawdown on cumulative USD equity curve
    equity = START_EQUITY_USD + taken["pnl_usd"].cumsum()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max
    max_dd = float(dd.min()) * 100.0  # negative number

    metrics = {
        "label": label,
        "trades": int(len(taken)),
        "skipped_concurrency": int(skipped),
        "wr": float((taken["pnl_usd"] > 0).mean()),
        "pf_net": float(pf) if pf != float("inf") else float("inf"),
        "avg_pnl_pct_net": float(taken["pnl_pct_net"].mean()),
        "total_pnl_usd": float(total_pnl_usd),
        "total_return_pct": float(total_pnl_usd / START_EQUITY_USD * 100.0),
        "max_dd_pct": max_dd,
    }
    return ledger, metrics


def monte_carlo_pf(
    pnls: np.ndarray, iters: int = MC_ITER, seed: int = MC_SEED
) -> tuple[float, float, float]:
    """Bootstrap PF distribution by resampling trade order with replacement.

    Returns (p5, p50, p95) of PF across `iters` bootstrap samples. Samples
    with no losers contribute PF=inf, which sorts to the tail and therefore
    participates in p50/p95 (a strategy whose bootstraps frequently land
    no-loss IS robust — that signal should not be masked). p5 stays finite
    when <5% of samples are no-loss, which is what we actually gate on.
    """
    rng = np.random.default_rng(seed)
    n = len(pnls)
    if n == 0:
        return 0.0, 0.0, 0.0
    pfs = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        sample = pnls[rng.integers(0, n, size=n)]
        wins = sample[sample > 0]
        losses = sample[sample <= 0]
        if len(losses) == 0 or losses.sum() == 0:
            pfs[i] = np.inf
        else:
            pfs[i] = wins.sum() / abs(losses.sum())
    pfs_sorted = np.sort(pfs)
    p5 = pfs_sorted[int(0.05 * iters)]
    p50 = pfs_sorted[int(0.50 * iters)]
    p95 = pfs_sorted[int(0.95 * iters)]
    return float(p5), float(p50), float(p95)


def threshold_kelly_crosscheck(
    trades_df: pd.DataFrame,
    kelly_table: pd.DataFrame,
    thresholds: list[float] = CROSSCHECK_THRESHOLDS,
) -> pd.DataFrame:
    """For each threshold, re-run scenarios A, B, C and collect total_return_pct.

    Scenario A uses flat sizing (f=BASELINE_EQUITY_FRACTION for every bin);
    B and C use the provided kelly_table.

    Returns DataFrame with columns: threshold, A_trades, A_ret_pct, B_ret_pct, C_ret_pct.
    """
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": BASELINE_EQUITY_FRACTION}
        for lo, hi in zip(KELLY_BIN_EDGES[:-1], KELLY_BIN_EDGES[1:], strict=False)
    ])
    rows = []
    for th in thresholds:
        sub = trades_df[trades_df["ml_prob"] >= th].copy()
        _, m_a = simulate(sub, flat_tbl, max_concurrent=MAX_CONCURRENT_A, label=f"A_th{th}")
        _, m_b = simulate(sub, kelly_table, max_concurrent=MAX_CONCURRENT_B, label=f"B_th{th}")
        _, m_c = simulate(sub, kelly_table, max_concurrent=MAX_CONCURRENT_C, label=f"C_th{th}")
        rows.append({
            "threshold": th,
            "A_trades": m_a["trades"],
            "A_ret_pct": m_a["total_return_pct"],
            "B_ret_pct": m_b["total_return_pct"],
            "C_ret_pct": m_c["total_return_pct"],
        })
    return pd.DataFrame(rows)


def load_trades(engine) -> pd.DataFrame:
    """Load ML-filtered trades from DB and validate schema.

    Expected columns (at minimum): signal_time, ml_prob, pnl_pct, bars_held.
    Drops rows with NaN in any required column. Sorts by signal_time.
    """
    required = ["signal_time", "ml_prob", "pnl_pct", "bars_held"]
    sql = f"SELECT {', '.join(required)} FROM {SOURCE_TABLE} ORDER BY signal_time"
    df = pd.read_sql(sql, engine)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{SOURCE_TABLE} missing required columns: {missing}")
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    before = len(df)
    df = df.dropna(subset=required).reset_index(drop=True)
    if len(df) < before:
        print(f"  dropped {before - len(df)} rows with NaN in required columns")
    return df


def _split_oot(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (pre_oot, oot) using last HOLD_OUT_MONTHS calendar months."""
    max_date = df["signal_time"].max()
    oot_start = (max_date - pd.DateOffset(months=HOLD_OUT_MONTHS)).floor("D")
    pre = df[df["signal_time"] < oot_start].reset_index(drop=True)
    oot = df[df["signal_time"] >= oot_start].reset_index(drop=True)
    return pre, oot


RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def _git_sha_short() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short=8", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def _run_id() -> str:
    sha = _git_sha_short()
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"backtest_v2_resized_{sha}_{ts}"


def _write_outputs(
    run_dir: Path,
    metrics_a: dict,
    metrics_b: dict,
    metrics_c: dict,
    mc_a: tuple,
    mc_b: tuple,
    mc_c: tuple,
    ledger_a: pd.DataFrame,
    ledger_b: pd.DataFrame,
    ledger_c: pd.DataFrame,
    kelly_table: pd.DataFrame,
    crosscheck: pd.DataFrame,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    def _mc_to_dict(mc):
        return {"pf_p5": mc[0], "pf_p50": mc[1], "pf_p95": mc[2]}

    def _merge(metrics, mc):
        out = dict(metrics)
        out.update(_mc_to_dict(mc))
        # json-serialize inf as null
        for k, v in list(out.items()):
            if isinstance(v, float) and (v == float("inf") or v != v):
                out[k] = None
        return out

    lift_b_minus_a = metrics_b["total_return_pct"] - metrics_a["total_return_pct"]
    lift_c_minus_b = metrics_c["total_return_pct"] - metrics_b["total_return_pct"]
    lift_c_minus_a = metrics_c["total_return_pct"] - metrics_a["total_return_pct"]

    metrics_out = {
        "scenario_A": _merge(metrics_a, mc_a),
        "scenario_B": _merge(metrics_b, mc_b),
        "scenario_C": _merge(metrics_c, mc_c),
        "attribution": {
            "lever_1_lift_pct": lift_b_minus_a,
            "lever_4_lift_pct": lift_c_minus_b,
            "combined_lift_pct": lift_c_minus_a,
        },
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    # params.json
    params = {
        "SNAPSHOT_SOURCE": SOURCE_TABLE,
        "ML_THRESHOLD": ML_THRESHOLD,
        "START_EQUITY_USD": START_EQUITY_USD,
        "LEVERAGE": LEVERAGE,
        "BASELINE_EQUITY_FRACTION": BASELINE_EQUITY_FRACTION,
        "KELLY_MULTIPLIER": KELLY_MULTIPLIER,
        "KELLY_FRACTION_FLOOR": KELLY_FRACTION_FLOOR,
        "KELLY_FRACTION_CAP": KELLY_FRACTION_CAP,
        "KELLY_BIN_EDGES": KELLY_BIN_EDGES,
        "ROUND_TRIP_FEE": ROUND_TRIP_FEE,
        "HOLD_OUT_MONTHS": HOLD_OUT_MONTHS,
        "MAX_CONCURRENT_A": MAX_CONCURRENT_A,
        "MAX_CONCURRENT_B": MAX_CONCURRENT_B,
        "MAX_CONCURRENT_C": MAX_CONCURRENT_C,
        "MC_ITER": MC_ITER,
        "MC_SEED": MC_SEED,
        "BAR_MINUTES": BAR_MINUTES,
    }
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    # Ledgers
    ledger_a.to_csv(run_dir / "trades_A.csv", index=False)
    ledger_b.to_csv(run_dir / "trades_B.csv", index=False)
    ledger_c.to_csv(run_dir / "trades_C.csv", index=False)

    # Kelly table
    kelly_table.to_csv(run_dir / "kelly_table.csv", index=False)

    # Skipped by concurrency (A vs C; B same as A)
    skipped_a = ledger_a[~ledger_a["taken"]].copy()
    skipped_a["scenario"] = "A"
    skipped_c = ledger_c[~ledger_c["taken"]].copy()
    skipped_c["scenario"] = "C"
    pd.concat([skipped_a, skipped_c], ignore_index=True).to_csv(
        run_dir / "skipped_by_concurrency.csv", index=False
    )

    # Monthly comparison
    def _monthly(ledger, col):
        taken = ledger[ledger["taken"]].copy()
        taken["month"] = taken["signal_time"].dt.to_period("M").astype(str)
        return taken.groupby("month")["pnl_usd"].sum().rename(col)

    monthly = pd.concat(
        [
            _monthly(ledger_a, "A_pnl_usd"),
            _monthly(ledger_b, "B_pnl_usd"),
            _monthly(ledger_c, "C_pnl_usd"),
        ],
        axis=1,
    ).fillna(0.0).reset_index()
    for sc in ["A", "B", "C"]:
        monthly[f"{sc}_ret_pct"] = monthly[f"{sc}_pnl_usd"] / START_EQUITY_USD * 100.0
    monthly.to_csv(run_dir / "monthly_comparison.csv", index=False)

    # Crosscheck
    crosscheck.to_csv(run_dir / "threshold_kelly_crosscheck.csv", index=False)


def main():
    engine = create_engine(DB_URL)
    print("Loading trades ...")
    df = load_trades(engine)
    date_min = df["signal_time"].min().date()
    date_max = df["signal_time"].max().date()
    print(f"  {len(df)} trades  {date_min} -> {date_max}")

    pre_oot, oot = _split_oot(df)
    print(f"  pre-OOT: {len(pre_oot)}  OOT: {len(oot)}")

    # Apply threshold gate to OOT (pre-OOT uses same threshold for Kelly calibration)
    pre_oot_th = pre_oot[pre_oot["ml_prob"] >= ML_THRESHOLD].reset_index(drop=True)
    oot_th = oot[oot["ml_prob"] >= ML_THRESHOLD].reset_index(drop=True)
    print(f"  pre-OOT (ml>={ML_THRESHOLD}): {len(pre_oot_th)}  OOT: {len(oot_th)}")

    # --- Calibrate Kelly table on pre-OOT ONLY ---
    print("\nCalibrating Kelly table on pre-OOT ...")
    kelly_table = build_kelly_table(pre_oot_th)
    print(kelly_table.to_string(index=False))

    # --- Flat sizing table (baseline) ---
    flat_tbl = pd.DataFrame([
        {"bin_low": lo, "bin_high": hi, "f_capped": BASELINE_EQUITY_FRACTION,
         "trades": 0, "p": 0.0, "b": 0.0, "f_raw": 0.0, "f_half": 0.0}
        for lo, hi in zip(KELLY_BIN_EDGES[:-1], KELLY_BIN_EDGES[1:], strict=False)
    ])

    # --- Three scenarios on OOT ---
    print("\nRunning scenario A (flat, conc=5) ...")
    ledger_a, metrics_a = simulate(oot_th, flat_tbl, MAX_CONCURRENT_A, "A")
    print(
        f"  trades={metrics_a['trades']} ret={metrics_a['total_return_pct']:.1f}%"
        f" pf={metrics_a['pf_net']:.2f}"
    )

    print("\nRunning scenario B (half-Kelly, conc=5) ...")
    ledger_b, metrics_b = simulate(oot_th, kelly_table, MAX_CONCURRENT_B, "B")
    print(
        f"  trades={metrics_b['trades']} ret={metrics_b['total_return_pct']:.1f}%"
        f" pf={metrics_b['pf_net']:.2f}"
    )

    print("\nRunning scenario C (half-Kelly, conc=10) ...")
    ledger_c, metrics_c = simulate(oot_th, kelly_table, MAX_CONCURRENT_C, "C")
    print(
        f"  trades={metrics_c['trades']} ret={metrics_c['total_return_pct']:.1f}%"
        f" pf={metrics_c['pf_net']:.2f}"
    )

    # --- Monte Carlo on each scenario's taken trades ---
    print("\nMonte Carlo PF bootstrap ...")
    mc_a = monte_carlo_pf(ledger_a[ledger_a["taken"]]["pnl_pct_net"].to_numpy())
    mc_b = monte_carlo_pf(ledger_b[ledger_b["taken"]]["pnl_pct_net"].to_numpy())
    mc_c = monte_carlo_pf(ledger_c[ledger_c["taken"]]["pnl_pct_net"].to_numpy())
    print(f"  A PF: p5={mc_a[0]:.2f} p50={mc_a[1]:.2f} p95={mc_a[2]:.2f}")
    print(f"  B PF: p5={mc_b[0]:.2f} p50={mc_b[1]:.2f} p95={mc_b[2]:.2f}")
    print(f"  C PF: p5={mc_c[0]:.2f} p50={mc_c[1]:.2f} p95={mc_c[2]:.2f}")

    # --- Threshold crosscheck ---
    print("\nThreshold x Kelly crosscheck ...")
    crosscheck = threshold_kelly_crosscheck(oot, kelly_table)
    print(crosscheck.to_string(index=False))

    # --- Write outputs ---
    run_dir = RESULTS_ROOT / _run_id()
    _write_outputs(run_dir, metrics_a, metrics_b, metrics_c, mc_a, mc_b, mc_c,
                   ledger_a, ledger_b, ledger_c, kelly_table, crosscheck)
    print(f"\nWrote results to: {run_dir}")

    # --- Attribution summary ---
    print("\n=== ATTRIBUTION ===")
    print(f"A baseline return:     {metrics_a['total_return_pct']:>7.1f}%")
    lift_b = metrics_b["total_return_pct"] - metrics_a["total_return_pct"]
    lift_c = metrics_c["total_return_pct"] - metrics_a["total_return_pct"]
    print(f"B (+Kelly) return:     {metrics_b['total_return_pct']:>7.1f}%  (lift: {lift_b:+.1f}%)")
    print(f"C (+Kelly+conc) return: {metrics_c['total_return_pct']:>7.1f}%  (lift: {lift_c:+.1f}%)")


if __name__ == "__main__":
    main()
