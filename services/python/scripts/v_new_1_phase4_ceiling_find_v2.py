"""v_new_1 Phase 4 — Ceiling-find sweep (v2 data).

Determines the achievable backtest CAGR for the BGM architecture across a grid
of (leverage × DD cap × universe × sizing aggressiveness × exit) configs, so we
can decide whether the BGM has enough edge to defend with LSTM/RL.

OUTPUT:
    services/python/results/v_new_1_phase4_ceiling_find_v2/
        all_trials.csv         — every Optuna trial w/ params + metrics
        pareto_table.csv       — best trial per (leverage, dd_cap) cell
        verdict.md             — headline ceiling at -30% DD / 10× / top-200

Reuses Policy + simulate_year + backtest_policy from v_new_1_phase4_policy_opt.

USAGE:
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_phase4_ceiling_find_v2.py
Env:
    V_NEW_1_DATA_DIR    default services/python/data/v_new_1_v2
    MODEL_VARIANT       "" (baseline) or "_v1_5" (sequence-feature model)
    N_TRIALS            default 200
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time
import warnings
from dataclasses import asdict
from typing import Optional

import numpy as np
import optuna
import pandas as pd

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reuse simulator and dataclasses from the legacy phase4 script
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))
from v_new_1_phase4_policy_opt import (  # noqa: E402
    Policy, YearMetrics, simulate_year, backtest_policy, FRICTION_NOTIONAL_PCT,
)

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                    str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
RESULTS = ROOT / "services" / "python" / "results" / "v_new_1_phase4_ceiling_find_v2"
RESULTS.mkdir(parents=True, exist_ok=True)

MODEL_VARIANT = os.environ.get("MODEL_VARIANT", "")  # "" baseline; "_v1_5" sequence feats
N_TRIALS = int(os.environ.get("N_TRIALS", "200"))

# Single-cell ceiling target the user wants to hit
TARGET_CELL = {"leverage": 10.0, "dd_cap_pct": -30.0}

# Slice keys present in the v2 chain output
OOS_SLICES = ["2025_h2", "2026_q1"]

# Sweep grids (used to bin trials post-hoc; Optuna actually samples continuously)
LEVERAGE_GRID = [5.0, 8.0, 10.0, 15.0]
DD_CAP_GRID = [None, -20.0, -30.0, -40.0]


def _load_v2_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Loads OOS-scored long/short for the v2 dataset, attaches forward
    returns (max, min, 7d close) and CFGI, and tags each row with its OOS slice
    based on timestamp."""
    suffix = MODEL_VARIANT
    long_path = DATA / f"oos_scored_long{suffix}.parquet"
    short_path = DATA / f"oos_scored_short{suffix}.parquet"
    print(f"  long scores:  {long_path}")
    print(f"  short scores: {short_path}")
    long_scores = pd.read_parquet(long_path)
    short_scores = pd.read_parquet(short_path)

    labels_long = pd.read_parquet(DATA / "labels_long_oos.parquet")
    labels_short = pd.read_parquet(DATA / "labels_short_oos.parquet")
    features = pd.read_parquet(
        DATA / "features_full.parquet", columns=["symbol", "timestamp", "cfgi_value"],
    )

    # 7-day close return (42 × 4h bars)
    ll = labels_long.sort_values(["symbol", "timestamp"]).copy()
    ll["forward_7d_close_return"] = (
        ll.groupby("symbol")["close"].shift(-42) / ll["close"] - 1.0
    )
    ls = labels_short.sort_values(["symbol", "timestamp"]).copy()
    ls["forward_7d_close_return"] = (
        ls.groupby("symbol")["close"].shift(-42) / ls["close"] - 1.0
    )

    fwd_combined = ll[["symbol", "timestamp", "forward_max_pct", "forward_7d_close_return"]].merge(
        ls[["symbol", "timestamp", "forward_min_pct"]],
        on=["symbol", "timestamp"], how="inner",
    )

    long_df = long_scores.merge(fwd_combined, on=["symbol", "timestamp"], how="inner")
    short_df = short_scores.merge(fwd_combined, on=["symbol", "timestamp"], how="inner")

    cfgi_ts = features.drop_duplicates(subset="timestamp")[["timestamp", "cfgi_value"]].copy()
    long_df = long_df.merge(cfgi_ts, on="timestamp", how="left")
    short_df = short_df.merge(cfgi_ts, on="timestamp", how="left")

    for df in (long_df, short_df):
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        # tag oos_year if not already present in the scored parquet
        if "oos_year" not in df.columns:
            ts = df["timestamp"]
            df["oos_year"] = "unknown"
            df.loc[(ts >= "2025-07-01") & (ts <= "2025-12-31 23:59:59"), "oos_year"] = "2025_h2"
            df.loc[(ts >= "2026-01-01") & (ts <= "2026-03-31 23:59:59"), "oos_year"] = "2026_q1"

    long_df = long_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    short_df = short_df.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    return long_df, short_df


def _split_by_slice(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {sl: grp.copy() for sl, grp in df.groupby("oos_year") if sl in OOS_SLICES}


def _backtest_for_slices(
    policy: Policy,
    long_by_slice: dict[str, pd.DataFrame],
    short_by_slice: dict[str, pd.DataFrame],
) -> list[YearMetrics]:
    """Same as backtest_policy but uses our slice list (2025_h2, 2026_q1) and
    calls simulate_year directly (which is slice-agnostic — it just receives
    rows and a year label)."""
    results = []
    for sl in OOS_SLICES:
        ly = long_by_slice.get(sl, pd.DataFrame())
        sy = short_by_slice.get(sl, pd.DataFrame())
        results.append(simulate_year(ly, sy, policy))
        # simulate_year tags year from `year_label`; we override here so the
        # YearMetrics record has the slice key we want.
        results[-1].year = sl
    return results


def _annualised_cagr(metrics: list[YearMetrics]) -> float:
    """Compute annualised CAGR across the 9-month OOS window, compounding
    2025_h2 (6 mo) → 2026_q1 (3 mo). Returns CAGR as a percent (e.g. 500.0 means
    500%/yr)."""
    eq = 1.0
    months = 0.0
    for m in metrics:
        if m.year == "2025_h2":
            months += 6.0
        elif m.year == "2026_q1":
            months += 3.0
        eq *= (1.0 + np.clip(m.compounded_equity_pct / 100.0, -0.99, 1e6))
    if months == 0 or eq <= 0:
        return -100.0
    years = months / 12.0
    cagr = (eq ** (1.0 / years) - 1.0) * 100.0
    return float(np.clip(cagr, -100.0, 1e6))


def _max_dd_per_slice(metrics: list[YearMetrics]) -> float:
    """Worst per-slice DD across the 2 slices. Negative number."""
    return float(min(m.max_drawdown_pct for m in metrics)) if metrics else 0.0


def _passes_dd_cap(metrics: list[YearMetrics], dd_cap_pct: Optional[float]) -> bool:
    if dd_cap_pct is None:
        return True  # No cap → all configs valid
    worst = _max_dd_per_slice(metrics)
    return worst >= dd_cap_pct  # e.g. -25% >= -30% is True (within cap)


def _make_objective(
    long_by_slice: dict[str, pd.DataFrame],
    short_by_slice: dict[str, pd.DataFrame],
    trial_log: list[dict],
):
    def objective(trial: optuna.Trial) -> float:
        # Wider search space than the original phase4 — we want to find the
        # ceiling, not a conservative production policy.
        leverage = trial.suggest_categorical("leverage", LEVERAGE_GRID)
        dd_cap_pct = trial.suggest_categorical("dd_cap_pct",
                                               ["none", "-20", "-30", "-40"])
        dd_cap_val = None if dd_cap_pct == "none" else float(dd_cap_pct)

        top_k_long = trial.suggest_int("top_k_long", 3, 30)
        top_k_short = trial.suggest_int("top_k_short", 3, 30)
        score_threshold_long = trial.suggest_float("score_threshold_long", 0.20, 0.55)
        score_threshold_short = trial.suggest_float("score_threshold_short", 0.30, 0.65)
        position_size_pct = trial.suggest_float("position_size_pct", 0.2, 5.0)
        sl_choice = trial.suggest_categorical(
            "stop_loss_pct_choice", [None, None, -20.0, -30.0, -50.0],
        )
        tp_choice = trial.suggest_categorical(
            "take_profit_pct", [None, None, 10.0, 15.0, 20.0, 30.0, 50.0],
        )
        score_scaled_sizing = trial.suggest_categorical("score_scaled_sizing", [True, False])
        cfgi_long_gate = trial.suggest_categorical(
            "cfgi_long_gate", [None, 30.0, 40.0, 50.0, 60.0],
        )
        cfgi_short_gate = trial.suggest_categorical(
            "cfgi_short_gate", [None, 40.0, 50.0, 60.0, 70.0],
        )
        # Wider gross exposure range — at 10×+ leverage we expect 100-300% gross
        gross_exposure_cap_pct = trial.suggest_float("gross_exposure_cap_pct", 30.0, 300.0)

        policy = Policy(
            top_k_long=top_k_long,
            top_k_short=top_k_short,
            score_threshold_long=score_threshold_long,
            score_threshold_short=score_threshold_short,
            position_size_pct=position_size_pct,
            leverage=float(leverage),
            stop_loss_pct=float(sl_choice) if sl_choice is not None else None,
            take_profit_pct=float(tp_choice) if tp_choice is not None else None,
            hold_timeout_bars=42,
            score_scaled_sizing=bool(score_scaled_sizing),
            cfgi_short_gate=float(cfgi_short_gate) if cfgi_short_gate is not None else None,
            cfgi_long_gate=float(cfgi_long_gate) if cfgi_long_gate is not None else None,
            gross_exposure_cap_pct=float(gross_exposure_cap_pct),
        )

        try:
            metrics = _backtest_for_slices(policy, long_by_slice, short_by_slice)
        except Exception as e:
            print(f"    trial {trial.number} ERROR: {e}")
            return -1e6

        cagr = _annualised_cagr(metrics)
        worst_dd = _max_dd_per_slice(metrics)
        passes_dd = _passes_dd_cap(metrics, dd_cap_val)

        row = {
            "trial": trial.number,
            "leverage": float(leverage),
            "dd_cap_pct": dd_cap_pct,
            "dd_cap_val": dd_cap_val,
            "top_k_long": top_k_long,
            "top_k_short": top_k_short,
            "score_threshold_long": score_threshold_long,
            "score_threshold_short": score_threshold_short,
            "position_size_pct": position_size_pct,
            "stop_loss_pct": float(sl_choice) if sl_choice is not None else None,
            "take_profit_pct": float(tp_choice) if tp_choice is not None else None,
            "score_scaled_sizing": bool(score_scaled_sizing),
            "cfgi_long_gate": float(cfgi_long_gate) if cfgi_long_gate is not None else None,
            "cfgi_short_gate": float(cfgi_short_gate) if cfgi_short_gate is not None else None,
            "gross_exposure_cap_pct": float(gross_exposure_cap_pct),
            "cagr_pct": cagr,
            "worst_dd_pct": worst_dd,
            "passes_dd_cap": passes_dd,
        }
        for m in metrics:
            row[f"{m.year}_compounded_pct"] = m.compounded_equity_pct
            row[f"{m.year}_max_dd_pct"] = m.max_drawdown_pct
            row[f"{m.year}_n_trades"] = m.n_total_trades
        trial_log.append(row)

        # Optuna maximises this. We return CAGR if the DD cap is honoured,
        # otherwise heavily penalise. The penalty preserves direction — Optuna
        # still gradients toward configs that respect the cap.
        if not passes_dd:
            return -100.0 + 0.001 * cagr
        return cagr
    return objective


def _verdict(trials: pd.DataFrame) -> str:
    lines = []
    lines.append("# v_new_1 Phase 4 — Ceiling-find Verdict (v2)\n")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d')}\n")
    lines.append(f"**Model variant:** {MODEL_VARIANT or 'baseline (v_new_1)'}\n")
    lines.append(f"**OOS slices:** {OOS_SLICES}\n")
    lines.append(f"**Trials evaluated:** {len(trials)}\n\n")

    lines.append("## Pareto frontier (best CAGR per leverage × dd_cap cell)\n\n")
    lines.append("| leverage | dd_cap | CAGR (%/yr) | worst DD (%) | n_trades | trial |\n")
    lines.append("|---|---|---|---|---|---|\n")

    for lev in LEVERAGE_GRID:
        for dd_cap_str in ["none", "-20", "-30", "-40"]:
            mask = (
                (trials["leverage"] == lev)
                & (trials["dd_cap_pct"] == dd_cap_str)
                & (trials["passes_dd_cap"])
            )
            cell = trials.loc[mask]
            if cell.empty:
                lines.append(f"| {lev}× | {dd_cap_str} | — | — | — | — |\n")
                continue
            best = cell.loc[cell["cagr_pct"].idxmax()]
            n_trades = int(
                best.get("2025_h2_n_trades", 0) + best.get("2026_q1_n_trades", 0)
            )
            lines.append(
                f"| {lev}× | {dd_cap_str} | {best['cagr_pct']:.1f} | "
                f"{best['worst_dd_pct']:.1f} | {n_trades} | "
                f"#{int(best['trial'])} |\n"
            )

    lines.append("\n## Target operating point\n\n")
    target_mask = (
        (trials["leverage"] == TARGET_CELL["leverage"])
        & (trials["dd_cap_pct"] == "-30")
        & (trials["passes_dd_cap"])
    )
    target = trials.loc[target_mask]
    if target.empty:
        lines.append(
            "**🔴 No config at -30% DD / 10× leverage cleared the DD cap.** "
            "The architecture cannot deliver the requested risk/return profile.\n"
        )
    else:
        best = target.loc[target["cagr_pct"].idxmax()]
        cagr = best["cagr_pct"]
        if cagr >= 500.0:
            verdict = "🟢 BGM ceiling clears 500%/yr at the target — proceed to v_new_2 LSTM."
        elif cagr >= 300.0:
            verdict = (
                "🟡 BGM ceiling between 300-500%/yr — partial edge. "
                "LSTM may close the gap; worth running v_new_2 with caution."
            )
        else:
            verdict = (
                "🔴 BGM ceiling <300%/yr at the target. Architecture is the "
                "bottleneck. LSTM unlikely to fix; pivot to a different "
                "strategy class (parabolic capture, options, leveraged ETFs)."
            )
        lines.append(f"**Best at -30% DD / 10× leverage / top-200:**\n\n")
        lines.append(f"- CAGR: **{cagr:.1f}%/yr**\n")
        lines.append(f"- Worst slice DD: {best['worst_dd_pct']:.1f}%\n")
        lines.append(f"- top_k: {int(best['top_k_long'])}L / {int(best['top_k_short'])}S\n")
        lines.append(f"- Position size: {best['position_size_pct']:.2f}%\n")
        lines.append(f"- Score thresholds: {best['score_threshold_long']:.3f}L / "
                     f"{best['score_threshold_short']:.3f}S\n")
        lines.append(f"- Score-scaled sizing: {best['score_scaled_sizing']}\n")
        lines.append(f"- CFGI gates: long≥{best['cfgi_long_gate']}, "
                     f"short≤{best['cfgi_short_gate']}\n")
        lines.append(f"- Gross-exposure cap: {best['gross_exposure_cap_pct']:.1f}%\n")
        lines.append(f"\n**Verdict:** {verdict}\n")

    return "".join(lines)


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print(f"v_new_1 Phase 4 — Ceiling-find Sweep (v2, model={MODEL_VARIANT or 'baseline'})")
    print("=" * 70)
    print(f"DATA: {DATA}")
    print(f"RESULTS: {RESULTS}")
    print(f"N_TRIALS: {N_TRIALS}\n")

    long_df, short_df = _load_v2_data()
    print(f"  Loaded long: {len(long_df):,}  short: {len(short_df):,}")
    print(f"  Slices in long: {long_df['oos_year'].value_counts().to_dict()}")

    long_by_slice = _split_by_slice(long_df)
    short_by_slice = _split_by_slice(short_df)
    for sl in OOS_SLICES:
        ly = long_by_slice.get(sl, pd.DataFrame())
        sy = short_by_slice.get(sl, pd.DataFrame())
        print(f"  {sl}: {len(ly):,} long rows, {len(sy):,} short rows")

    print(f"\nRunning Optuna ceiling-find ({N_TRIALS} trials)...")
    trial_log: list[dict] = []
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(
        _make_objective(long_by_slice, short_by_slice, trial_log),
        n_trials=N_TRIALS,
        n_jobs=1,
        show_progress_bar=False,
    )

    trials = pd.DataFrame(trial_log)
    trials.sort_values("cagr_pct", ascending=False, inplace=True)
    trials_path = RESULTS / f"all_trials{MODEL_VARIANT}.csv"
    trials.to_csv(trials_path, index=False)
    print(f"\n  Saved: {trials_path} ({len(trials)} trials)")

    verdict_text = _verdict(trials)
    verdict_path = RESULTS / f"verdict{MODEL_VARIANT}.md"
    verdict_path.write_text(verdict_text)
    print(f"  Saved: {verdict_path}")

    elapsed = time.time() - t0
    print(f"\nDONE in {elapsed:.1f}s\n")
    print(verdict_text)


if __name__ == "__main__":
    main()
