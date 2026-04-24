"""Sizing sweep for the 2 winning bigmover signals across 3 years.

36 variants = 2 signals (price_accel_atr, multi_bar_confirm)
             x 2 directions (long, short)
             x 9 sizing configs (varying position_pct, leverage, sl_pct)

Sizing configs:
  1. 05pct_05x_sl05  (position 5%, 5x lev, SL 5%)   — risk 1.25% per trade
  2. 10pct_05x_sl05  (position 10%, 5x lev, SL 5%)  — risk 2.5%   (current baseline)
  3. 15pct_05x_sl05  (position 15%, 5x lev, SL 5%)  — risk 3.75%
  4. 20pct_05x_sl05  (position 20%, 5x lev, SL 5%)  — risk 5%
  5. 10pct_03x_sl05  (position 10%, 3x lev, SL 5%)  — risk 1.5%
  6. 10pct_07x_sl05  (position 10%, 7x lev, SL 5%)  — risk 3.5%
  7. 15pct_03x_sl05  (position 15%, 3x lev, SL 5%)  — risk 2.25%
  8. 10pct_05x_sl07  (position 10%, 5x lev, SL 7%)  — risk 3.5%   (wider SL)
  9. 15pct_05x_sl07  (position 15%, 5x lev, SL 7%)  — risk 5.25%  (wider SL)

Reuses the bidirectional simulator and detectors from
backtest_bigmover_longshort_3y.py. Flat sizing (no compounding), 3-year window.

Run from services/python/:
    python scripts/backtest_bigmover_sizing_sweep_v1.py

Output:
  - Per-variant folders in results/<run_id>__<label>/ with trades.csv etc.
  - results/_sizing_sweep_summary_<sha8>_<ts>.csv (36 rows)
  - Leaderboard printed to stdout (filter: trades>=200 AND PF_p5>=1.30, sort by return)
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time as _time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from src.ml.bigmover_signals import (  # noqa: E402
    ACCEL_MULT_OF_ATR,
    ATR_PERIOD,
    PRICE_LOOKBACK,
    PRICE_MOVE_THRESH,
    VOL_MA_PERIOD,
    VOL_SPIKE,
    baseline_entry_mask_long,
    baseline_entry_mask_short,
    compute_atr,
    detect_signal,
)

# =============================================================================
# CONSTANTS
# =============================================================================
SNAPSHOT_DATE = "2026-04-24"
RANDOM_SEED = 42

TOP_N = 100
MIN_QUOTE_VOL_24H = 500_000
LEVERAGED_TOKEN_RE = r"[35][LS]_USDT$"

# Portfolio-level cooldown (detector params come from src.ml.bigmover_signals).
COOLDOWN_BARS = 96

MAX_CONCURRENT_POSITIONS = 5
WORST_FILL_BUFFER = 0.005
MAX_HOLD_BARS = 192
TRAIL_PCT = 0.03
TRAIL_ACTIVATION = 0.02

START_EQUITY_USD = 100.0
ROUND_TRIP_FEE = 0.0012

LISTING_WARMUP_DAYS = 3
MC_ITER = 1000
MC_SEED = 42

SIGNALS = ["price_accel_atr", "multi_bar_confirm"]
DIRECTIONS = ["short", "long"]

# Sizing configurations: (label, position_pct, leverage, sl_pct)
SIZING_CONFIGS = [
    ("05pct_05x_sl05", 0.05, 5, 0.05),
    ("10pct_05x_sl05", 0.10, 5, 0.05),   # current baseline
    ("15pct_05x_sl05", 0.15, 5, 0.05),
    ("20pct_05x_sl05", 0.20, 5, 0.05),
    ("10pct_03x_sl05", 0.10, 3, 0.05),
    ("10pct_07x_sl05", 0.10, 7, 0.05),
    ("15pct_03x_sl05", 0.15, 3, 0.05),
    ("10pct_05x_sl07", 0.10, 5, 0.07),
    ("15pct_05x_sl07", 0.15, 5, 0.07),
]


# =============================================================================
# CANONICAL HELPERS (copied from backtest_bigmover_longshort_3y.py)
# =============================================================================

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest_row(manifest_text, date):
    for line in manifest_text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5 or not _DATE_RE.fullmatch(parts[0]) or parts[0] != date:
            continue
        return parts[2], parts[4]
    raise ValueError(f"date {date} not in MANIFEST.md")


def load_snapshot(date):
    snapshots_dir = _SNAPSHOTS_DIR
    manifest = snapshots_dir / "MANIFEST.md"
    candles_path = snapshots_dir / f"candles_15m_{date}.parquet"
    universe_path = snapshots_dir / f"universe_{date}.parquet"
    expected_c, expected_u = _parse_manifest_row(manifest.read_text(), date)
    if _sha256_file(candles_path) != expected_c:
        raise ValueError(f"sha256 mismatch for {candles_path}")
    if _sha256_file(universe_path) != expected_u:
        raise ValueError(f"sha256 mismatch for {universe_path}")
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _git_dirty():
    return bool(
        subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    )


def _pip_freeze():
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    return sorted(out.decode().strip().splitlines())


def write_results(
    *, run_id, trades_df, metrics, params, snapshot_date,
    snapshot_candles_sha256, snapshot_universe_sha256,
    git_dirty, wall_time_seconds, results_root=Path("results"),
):
    run_dir = Path(results_root) / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    trades_df.to_csv(run_dir / "trades.csv", index=False)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)
    metadata = {
        "run_id": run_id, "git_sha": _git_sha(), "git_dirty": git_dirty,
        "snapshot_date": snapshot_date,
        "snapshot_candles_sha256": snapshot_candles_sha256,
        "snapshot_universe_sha256": snapshot_universe_sha256,
        "python_version": sys.version, "pip_freeze": _pip_freeze(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_time_seconds": float(wall_time_seconds),
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)
    return run_dir


def monte_carlo_pf(trades_df, n_iter=MC_ITER, seed=MC_SEED):
    if len(trades_df) < 10 or "pnl_pct" not in trades_df.columns:
        return {"p5": 0.0, "p50": 0.0, "p95": 0.0}
    pnls = trades_df["pnl_pct"].values
    rng = np.random.default_rng(seed)
    pfs = []
    n = len(pnls)
    for _ in range(n_iter):
        sample = rng.choice(pnls, size=n, replace=True)
        wins = sample[sample > 0].sum()
        losses = -sample[sample < 0].sum()
        if losses == 0:
            pfs.append(float("inf") if wins > 0 else 0.0)
        else:
            pfs.append(wins / losses)
    finite = [p for p in pfs if p != float("inf")]
    if not finite:
        return {"p5": float("inf"), "p50": float("inf"), "p95": float("inf")}
    return {
        "p5": float(np.percentile(finite, 5)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
    }


def apply_universe_filter(universe_df):
    leveraged_re = re.compile(LEVERAGED_TOKEN_RE)
    df = universe_df[
        (~universe_df["symbol"].str.contains(leveraged_re, na=False))
        & (~universe_df["in_delisting"].fillna(False))
        & (universe_df["quote_volume_24h"].fillna(0) >= MIN_QUOTE_VOL_24H)
    ].copy()
    df = df.sort_values("quote_volume_24h", ascending=False).head(TOP_N)
    return set(df["symbol"].str.replace("_", "", regex=False).tolist())


# compute_atr, baseline_entry_mask_short, baseline_entry_mask_long, and
# detect_signal are imported from src.ml.bigmover_signals above.


def compute_listing_gates(candles_by_asset):
    gates = {}
    warmup_s = LISTING_WARMUP_DAYS * 86400
    for asset, df in candles_by_asset.items():
        if len(df) == 0:
            continue
        gates[asset] = int(df["timestamp"].iloc[0]) + warmup_s
    return gates


# =============================================================================
# BIDIRECTIONAL SIMULATOR (parameterised on sl_pct)
# =============================================================================


def simulate_portfolio_bidirectional(
    candles_by_asset, entry_masks_by_dir, sl_pct, cooldown_bars=COOLDOWN_BARS
):
    """Short+long exit engine with specified SL. Trail params kept at 2%/3% activate/exit."""
    candidate_trades = []
    for direction, masks in entry_masks_by_dir.items():
        for asset, mask in masks.items():
            if asset not in candles_by_asset or not mask.any():
                continue
            df = candles_by_asset[asset]
            ts = df["timestamp"].values
            open_ = df["open"].values.astype(float)
            high = df["high"].values.astype(float)
            low = df["low"].values.astype(float)
            n = len(df)
            last_entry_bar = -cooldown_bars
            for i in np.where(mask.values)[0]:
                if i - last_entry_bar < cooldown_bars or i >= n:
                    continue
                entry_price = float(open_[i])
                if entry_price <= 0:
                    continue
                entry_ts = int(ts[i])
                if direction == "short":
                    sl_trigger = entry_price * (1 + sl_pct + WORST_FILL_BUFFER)
                    peak_low = entry_price
                    exit_price = entry_price
                    exit_bar = min(i + MAX_HOLD_BARS, n - 1)
                    exit_reason = "timeout"
                    for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
                        bar_high, bar_low = high[j], low[j]
                        if bar_low < peak_low:
                            peak_low = bar_low
                        if bar_high >= sl_trigger:
                            exit_price, exit_bar, exit_reason = sl_trigger, j, "stop_loss"
                            break
                        act_price = entry_price * (1 - TRAIL_ACTIVATION)
                        if peak_low <= act_price:
                            trail_price = peak_low * (1 + TRAIL_PCT)
                            if bar_high >= trail_price:
                                exit_price, exit_bar, exit_reason = trail_price, j, "trail_stop"
                                break
                    pnl_pct = (entry_price - exit_price) / entry_price
                else:
                    sl_trigger = entry_price * (1 - sl_pct - WORST_FILL_BUFFER)
                    peak_high = entry_price
                    exit_price = entry_price
                    exit_bar = min(i + MAX_HOLD_BARS, n - 1)
                    exit_reason = "timeout"
                    for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
                        bar_high, bar_low = high[j], low[j]
                        if bar_high > peak_high:
                            peak_high = bar_high
                        if bar_low <= sl_trigger:
                            exit_price, exit_bar, exit_reason = sl_trigger, j, "stop_loss"
                            break
                        act_price = entry_price * (1 + TRAIL_ACTIVATION)
                        if peak_high >= act_price:
                            trail_price = peak_high * (1 - TRAIL_PCT)
                            if bar_low <= trail_price:
                                exit_price, exit_bar, exit_reason = trail_price, j, "trail_stop"
                                break
                    pnl_pct = (exit_price - entry_price) / entry_price
                candidate_trades.append({
                    "direction": direction,
                    "pnl_pct": pnl_pct,
                    "bars_held": exit_bar - i,
                    "symbol": asset,
                    "entry_time": entry_ts,
                    "exit_time": int(ts[exit_bar]),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "exit_reason": exit_reason,
                })
                last_entry_bar = i

    if not candidate_trades:
        return pd.DataFrame(columns=[
            "direction", "pnl_pct", "bars_held", "symbol", "entry_time",
            "exit_time", "entry_price", "exit_price", "exit_reason",
        ])

    candidate_trades.sort(key=lambda t: t["entry_time"])
    accepted = []
    for t in candidate_trades:
        open_count = sum(
            1 for a in accepted
            if a["entry_time"] <= t["entry_time"] < a["exit_time"]
        )
        if open_count >= MAX_CONCURRENT_POSITIONS:
            continue
        accepted.append(t)
    return pd.DataFrame(accepted)


# =============================================================================
# SIZING APPLICATION + METRICS
# =============================================================================


def apply_sizing_and_metrics(
    trades_df, position_pct, leverage, sl_pct
):
    """Apply flat sizing: notional = $100 * position_pct * leverage per trade.
    Subtracts round-trip fee. Computes equity curve + max DD.
    """
    df = trades_df.copy()
    n = len(df)
    if n == 0:
        return df, {
            "trades": 0, "wr": 0.0, "pf": 0.0,
            "pf_p5": 0.0, "pf_p50": 0.0, "pf_p95": 0.0,
            "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0,
            "final_equity_usd": START_EQUITY_USD,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "risk_per_trade_pct": position_pct * leverage * sl_pct * 100,
            "avg_winner_pct": 0.0, "avg_loser_pct": 0.0,
            "median_hold_bars": 0.0,
        }
    notional = START_EQUITY_USD * position_pct * leverage
    df["notional_usd"] = notional
    df["pnl_pct_net"] = df["pnl_pct"] - ROUND_TRIP_FEE
    df["pnl_usd"] = notional * df["pnl_pct_net"]

    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    equity = START_EQUITY_USD + df_sorted["pnl_usd"].cumsum()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    max_dd = float(dd.min()) if dd.notna().any() else 0.0

    wins = df.loc[df["pnl_pct"] > 0, "pnl_pct"]
    losses = df.loc[df["pnl_pct"] < 0, "pnl_pct"]
    wins_sum = float(wins.sum())
    losses_sum = -float(losses.sum())
    pf = float("inf") if losses_sum == 0 else wins_sum / losses_sum
    mc = monte_carlo_pf(df)

    ret = df["pnl_pct_net"]
    sharpe = 0.0
    if ret.std(ddof=0) > 0:
        sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(len(ret)))

    total_pnl_usd = float(df["pnl_usd"].sum())
    metrics = {
        "trades": int(n),
        "wr": float((df["pnl_pct"] > 0).mean()),
        "pf": pf,
        "pf_p5": mc["p5"], "pf_p50": mc["p50"], "pf_p95": mc["p95"],
        "avg_pnl_pct": float(df["pnl_pct"].mean()),
        "total_pnl_pct": float(df["pnl_pct"].sum()),
        "final_equity_usd": float(START_EQUITY_USD + total_pnl_usd),
        "total_return_pct": float(total_pnl_usd / START_EQUITY_USD * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "sharpe": sharpe,
        "risk_per_trade_pct": float(position_pct * leverage * sl_pct * 100),
        "avg_winner_pct": float(wins.mean()) if len(wins) > 0 else 0.0,
        "avg_loser_pct": float(losses.mean()) if len(losses) > 0 else 0.0,
        "median_hold_bars": float(df["bars_held"].median()),
    }
    return df, metrics


# =============================================================================
# MAIN
# =============================================================================


def main():
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    overall_start = _time.monotonic()
    git_dirty = _git_dirty()
    if git_dirty:
        print("WARNING: git dirty — not exactly reproducible.", file=sys.stderr)

    print(f"[sizing] loading snapshot {SNAPSHOT_DATE} ...")
    candles, universe = load_snapshot(SNAPSHOT_DATE)
    if pd.api.types.is_datetime64_any_dtype(candles["timestamp"]):
        candles = candles.copy()
        candles["timestamp"] = candles["timestamp"].astype("int64") // 10**9
    print(f"[sizing] candles: {len(candles):,} rows")

    universe_assets = apply_universe_filter(universe)
    candles_tradable = candles[candles["asset"].isin(universe_assets)].copy()
    candles_by_asset = {
        asset: g.sort_values("timestamp").reset_index(drop=True)
        for asset, g in candles_tradable.groupby("asset")
    }
    print(f"[sizing] tradable assets: {len(candles_by_asset)}")

    listing_gates = compute_listing_gates(candles_by_asset)

    manifest_path = _SNAPSHOTS_DIR / "MANIFEST.md"
    candles_sha, universe_sha = _parse_manifest_row(
        manifest_path.read_text(), SNAPSHOT_DATE
    )

    base_run_id = (
        f"backtest_bigmover_sizing_sweep_v1_{_git_sha()[:8]}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )

    # Pre-compute entry masks per (signal, direction) — invariant to SL
    print("[sizing] detecting signals per (signal, direction)...")
    entry_masks_cache = {}
    for signal in SIGNALS:
        for direction in DIRECTIONS:
            t0 = _time.monotonic()
            masks = {}
            for asset, df in candles_by_asset.items():
                atr = compute_atr(df)
                mask = detect_signal(df, signal, direction, atr)
                gate_ts = listing_gates.get(asset)
                if gate_ts is not None:
                    mask = mask & (df["timestamp"] >= gate_ts)
                if mask.any():
                    masks[asset] = mask
            entry_masks_cache[(signal, direction)] = masks
            n_signals = sum(m.sum() for m in masks.values())
            print(f"  {signal}/{direction}: {int(n_signals)} raw signals [{_time.monotonic()-t0:.1f}s]")

    # Run simulator per (signal, direction, sl_pct) — sl is the one exit param that matters
    unique_sl = sorted({c[3] for c in SIZING_CONFIGS})
    print(f"[sizing] running simulator for {len(SIGNALS)} x {len(DIRECTIONS)} x {len(unique_sl)} = "
          f"{len(SIGNALS)*len(DIRECTIONS)*len(unique_sl)} simulations...")
    trade_cache = {}
    for signal in SIGNALS:
        for direction in DIRECTIONS:
            masks = entry_masks_cache[(signal, direction)]
            for sl in unique_sl:
                t0 = _time.monotonic()
                trades = simulate_portfolio_bidirectional(
                    candles_by_asset, {direction: masks}, sl_pct=sl
                )
                trade_cache[(signal, direction, sl)] = trades
                print(f"  {signal}/{direction}/sl={sl:.2f}: {len(trades)} trades "
                      f"[{_time.monotonic()-t0:.1f}s]")

    # Apply sizing, write per-variant outputs, accumulate summary
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)
    summary_rows = []
    print(f"\n[sizing] writing {len(SIGNALS)*len(DIRECTIONS)*len(SIZING_CONFIGS)} variants...")

    for signal in SIGNALS:
        for direction in DIRECTIONS:
            for (label, pos_pct, lev, sl) in SIZING_CONFIGS:
                v_start = _time.monotonic()
                trades = trade_cache[(signal, direction, sl)]
                sized, metrics = apply_sizing_and_metrics(trades, pos_pct, lev, sl)
                variant_label = f"{signal}__{direction}__{label}"
                run_id = f"{base_run_id}__{variant_label}"
                params = {
                    "signal": signal, "direction": direction,
                    "position_pct": pos_pct, "leverage": lev, "sl_pct": sl,
                    "trail_pct": TRAIL_PCT, "trail_activation": TRAIL_ACTIVATION,
                    "max_hold_bars": MAX_HOLD_BARS,
                    "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
                    "round_trip_fee": ROUND_TRIP_FEE,
                    "start_equity_usd": START_EQUITY_USD,
                    "snapshot_date": SNAPSHOT_DATE,
                    "listing_warmup_days": LISTING_WARMUP_DAYS,
                }
                write_results(
                    run_id=run_id, trades_df=sized, metrics=metrics, params=params,
                    snapshot_date=SNAPSHOT_DATE,
                    snapshot_candles_sha256=candles_sha,
                    snapshot_universe_sha256=universe_sha,
                    git_dirty=git_dirty,
                    wall_time_seconds=_time.monotonic() - v_start,
                )
                summary_rows.append({
                    "variant": variant_label,
                    "signal": signal, "direction": direction,
                    "sizing": label, "position_pct": pos_pct,
                    "leverage": lev, "sl_pct": sl,
                    **metrics,
                    "run_id": run_id,
                })

    summary_df = pd.DataFrame(summary_rows).sort_values(
        "total_return_pct", ascending=False
    )
    sha8 = _git_sha()[:8]
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = results_root / f"_sizing_sweep_summary_{sha8}_{ts_str}.csv"

    # Column ordering
    cols = [
        "variant", "signal", "direction", "sizing",
        "position_pct", "leverage", "sl_pct", "risk_per_trade_pct",
        "trades", "wr", "pf", "pf_p5", "pf_p50", "pf_p95",
        "total_return_pct", "max_drawdown_pct", "final_equity_usd",
        "avg_winner_pct", "avg_loser_pct", "sharpe",
        "avg_pnl_pct", "total_pnl_pct", "median_hold_bars", "run_id",
    ]
    summary_df[cols].to_csv(summary_path, index=False)

    elapsed = _time.monotonic() - overall_start
    print(f"\n[sizing] done in {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"[sizing] summary: {summary_path}")
    print("\n=== TOP 10 by total_return_pct (filter trades>=200 AND pf_p5>=1.30) ===\n")
    passing = summary_df[(summary_df["trades"] >= 200) & (summary_df["pf_p5"] >= 1.30)]
    top = passing.head(10)
    short_cols = [
        "variant", "risk_per_trade_pct", "trades", "wr", "pf", "pf_p5",
        "total_return_pct", "max_drawdown_pct", "sharpe",
    ]
    print(top[short_cols].to_string(index=False))
    print(f"\nTotal passing filter: {len(passing)}/{len(summary_df)}")


if __name__ == "__main__":
    main()
