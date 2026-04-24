"""Combined long+short portfolio — multi_bar_confirm @ 15%/5x/SL7%.

ONE unified $100 account takes BOTH long and short multi_bar_confirm signals,
competing for a single shared 5-slot pool. No regime gate.

Produces two scenarios:
  1. Flat sizing (constant notional) — apples-to-apples vs separate LONG/SHORT 3y.
  2. Compounding @ $100k notional cap — realistic live-deploy projection.

Each scenario writes: trades.csv, monthly.csv, metrics.json, params.json, run_metadata.json

Run from services/python/:
    python scripts/backtest_bigmover_combined_v1.py
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
    compute_atr,
    detect_signal,
)

# =============================================================================
# CONSTANTS (same as longshort_3y)
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
POSITION_PCT = 0.15
LEVERAGE = 5
SL_PCT = 0.07
ROUND_TRIP_FEE = 0.0012

NOTIONAL_CAP_USD = 100_000  # compounding cap for realistic deploy
LISTING_WARMUP_DAYS = 3

# =============================================================================
# CANONICAL HELPERS
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
    cp = snapshots_dir / f"candles_15m_{date}.parquet"
    up = snapshots_dir / f"universe_{date}.parquet"
    ec, eu = _parse_manifest_row(manifest.read_text(), date)
    if _sha256_file(cp) != ec:
        raise ValueError(f"sha mismatch {cp}")
    if _sha256_file(up) != eu:
        raise ValueError(f"sha mismatch {up}")
    return pd.read_parquet(cp), pd.read_parquet(up)


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _git_dirty():
    return bool(subprocess.check_output(["git", "status", "--porcelain"]).decode().strip())


def _pip_freeze():
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    return sorted(out.decode().strip().splitlines())


def apply_universe_filter(universe_df):
    leveraged_re = re.compile(LEVERAGED_TOKEN_RE)
    df = universe_df[
        (~universe_df["symbol"].str.contains(leveraged_re, na=False))
        & (~universe_df["in_delisting"].fillna(False))
        & (universe_df["quote_volume_24h"].fillna(0) >= MIN_QUOTE_VOL_24H)
    ].copy()
    df = df.sort_values("quote_volume_24h", ascending=False).head(TOP_N)
    return set(df["symbol"].str.replace("_", "", regex=False).tolist())


def compute_listing_gates(candles_by_asset):
    gates = {}
    warmup_s = LISTING_WARMUP_DAYS * 86400
    for asset, df in candles_by_asset.items():
        if len(df) == 0:
            continue
        gates[asset] = int(df["timestamp"].iloc[0]) + warmup_s
    return gates


# Signal detection comes from src.ml.bigmover_signals.detect_signal. This
# script uses only the multi_bar_confirm dispatch.


def detect_multi_bar_confirm(df, direction):
    return detect_signal(df, "multi_bar_confirm", direction, compute_atr(df))


# =============================================================================
# BIDIRECTIONAL SIMULATOR — shared 5-slot pool (copy from longshort_3y)
# =============================================================================


def simulate_portfolio_bidirectional(
    candles_by_asset, entry_masks_by_dir, sl_pct=SL_PCT, cooldown_bars=COOLDOWN_BARS
):
    """Short+long exit engine with MAX_CONCURRENT_POSITIONS=5 shared across directions."""
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
                        act = entry_price * (1 - TRAIL_ACTIVATION)
                        if peak_low <= act:
                            tp_ = peak_low * (1 + TRAIL_PCT)
                            if bar_high >= tp_:
                                exit_price, exit_bar, exit_reason = tp_, j, "trail_stop"
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
                        act = entry_price * (1 + TRAIL_ACTIVATION)
                        if peak_high >= act:
                            tp_ = peak_high * (1 - TRAIL_PCT)
                            if bar_low <= tp_:
                                exit_price, exit_bar, exit_reason = tp_, j, "trail_stop"
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
    rejected = []
    for t in candidate_trades:
        open_count = sum(
            1 for a in accepted
            if a["entry_time"] <= t["entry_time"] < a["exit_time"]
        )
        if open_count >= MAX_CONCURRENT_POSITIONS:
            rejected.append(t)
            continue
        accepted.append(t)
    return pd.DataFrame(accepted), pd.DataFrame(rejected)


# =============================================================================
# FLAT SIZING + COMPOUNDING (copy/adapt from existing scripts)
# =============================================================================


def apply_flat_sizing(trades_df, position_pct, leverage):
    df = trades_df.copy()
    if len(df) == 0:
        for c in ("notional_usd", "pnl_pct_net", "pnl_usd"):
            df[c] = []
        return df
    df["notional_usd"] = START_EQUITY_USD * position_pct * leverage
    df["pnl_pct_net"] = df["pnl_pct"] - ROUND_TRIP_FEE
    df["pnl_usd"] = df["notional_usd"] * df["pnl_pct_net"]
    return df


def apply_compounding(
    trades_df, position_pct, leverage, notional_cap=NOTIONAL_CAP_USD,
):
    df = trades_df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        for c in ("equity_at_entry", "notional_usd", "pnl_pct_net", "pnl_usd",
                  "equity_after_exit", "was_capped"):
            df[c] = []
        return df

    events = []
    for i, row in df.iterrows():
        events.append((int(row["entry_time"]), 1, i, "open"))
        events.append((int(row["exit_time"]), 0, i, "close"))
    events.sort(key=lambda e: (e[0], e[1]))

    equity = float(START_EQUITY_USD)
    equity_at_entry = np.zeros(n)
    notional_arr = np.zeros(n)
    pnl_net_arr = np.zeros(n)
    pnl_usd_arr = np.zeros(n)
    equity_after = np.zeros(n)
    was_capped = np.zeros(n, dtype=bool)

    for ts, _, idx, kind in events:
        if kind == "open":
            eq = max(equity, 1.0)
            desired = eq * position_pct * leverage
            if notional_cap is not None and desired > notional_cap:
                notional = float(notional_cap)
                was_capped[idx] = True
            else:
                notional = desired
            equity_at_entry[idx] = eq
            notional_arr[idx] = notional
            pnl_pct = float(df.at[idx, "pnl_pct"])
            pnl_net = pnl_pct - ROUND_TRIP_FEE
            pnl_net_arr[idx] = pnl_net
            pnl_usd_arr[idx] = notional * pnl_net
        else:
            equity += pnl_usd_arr[idx]
            equity_after[idx] = equity

    df["equity_at_entry"] = equity_at_entry
    df["notional_usd"] = notional_arr
    df["pnl_pct_net"] = pnl_net_arr
    df["pnl_usd"] = pnl_usd_arr
    df["equity_after_exit"] = equity_after
    df["was_capped"] = was_capped
    return df


# =============================================================================
# METRICS + MONTHLY
# =============================================================================


def _safe_pf(wins, losses):
    return float("inf") if losses == 0 else float(wins / losses)


def flat_metrics(df):
    if len(df) == 0:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "total_return_pct": 0.0,
                "max_drawdown_pct": 0.0, "final_equity_usd": START_EQUITY_USD,
                "long_trades": 0, "short_trades": 0}
    wins = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = -df.loc[df["pnl_pct"] < 0, "pnl_pct"].sum()
    df_sorted = df.sort_values("entry_time").reset_index(drop=True)
    equity = START_EQUITY_USD + df_sorted["pnl_usd"].cumsum()
    rm = equity.cummax()
    dd = (equity - rm) / rm.replace(0, np.nan)
    max_dd = float(dd.min()) if dd.notna().any() else 0.0
    total_pnl_usd = float(df["pnl_usd"].sum())
    return {
        "trades": int(len(df)),
        "wr": float((df["pnl_pct"] > 0).mean()),
        "pf": _safe_pf(wins, losses),
        "avg_pnl_pct": float(df["pnl_pct"].mean()),
        "total_return_pct": float(total_pnl_usd / START_EQUITY_USD * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "final_equity_usd": float(START_EQUITY_USD + total_pnl_usd),
        "long_trades": int((df["direction"] == "long").sum()),
        "short_trades": int((df["direction"] == "short").sum()),
        "avg_winner_pct": float(df.loc[df["pnl_pct"] > 0, "pnl_pct"].mean()) if (df["pnl_pct"] > 0).any() else 0.0,
        "avg_loser_pct": float(df.loc[df["pnl_pct"] < 0, "pnl_pct"].mean()) if (df["pnl_pct"] < 0).any() else 0.0,
    }


def compound_metrics(df):
    if len(df) == 0:
        return {"trades": 0, "trades_capped": 0, "wr": 0.0, "pf": 0.0,
                "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "final_equity_usd": START_EQUITY_USD, "avg_notional_usd": 0.0}
    closed = df.sort_values("exit_time").reset_index(drop=True)
    eq = closed["equity_after_exit"].values
    rm = np.maximum.accumulate(np.concatenate([[START_EQUITY_USD], eq]))[1:]
    dd = (eq - rm) / rm
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    wins = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = -df.loc[df["pnl_pct"] < 0, "pnl_pct"].sum()
    final = float(closed["equity_after_exit"].iloc[-1])
    return {
        "trades": int(len(df)),
        "trades_capped": int(df["was_capped"].sum()),
        "wr": float((df["pnl_pct"] > 0).mean()),
        "pf": _safe_pf(wins, losses),
        "total_return_pct": float((final - START_EQUITY_USD) / START_EQUITY_USD * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "final_equity_usd": final,
        "avg_notional_usd": float(df["notional_usd"].mean()),
        "final_notional_usd": float(df["notional_usd"].iloc[-1]),
        "long_trades": int((df["direction"] == "long").sum()),
        "short_trades": int((df["direction"] == "short").sum()),
    }


def monthly_aggregate(df, compounding=True):
    if len(df) == 0:
        return pd.DataFrame()
    work = df.copy()
    work["entry_dt"] = pd.to_datetime(work["entry_time"], unit="s", utc=True)
    work["month"] = work["entry_dt"].dt.strftime("%Y-%m")
    rows = []
    if compounding:
        # Initialize running-from-$100 equity progression
        prev_eq = START_EQUITY_USD
        for month, grp in work.groupby("month", sort=True):
            long_n = int((grp["direction"] == "long").sum())
            short_n = int((grp["direction"] == "short").sum())
            wins = grp.loc[grp["pnl_pct"] > 0, "pnl_pct"].sum()
            losses = -grp.loc[grp["pnl_pct"] < 0, "pnl_pct"].sum()
            closed = grp.sort_values("exit_time").reset_index(drop=True)
            eq_end = float(closed["equity_after_exit"].iloc[-1]) if len(closed) else prev_eq
            if len(closed):
                eq_series = closed["equity_after_exit"].values
                rm = np.maximum.accumulate(np.concatenate([[prev_eq], eq_series]))[1:]
                dd_series = (eq_series - rm) / rm
                dd_in_month = float(dd_series.min()) if len(dd_series) else 0.0
            else:
                dd_in_month = 0.0
            rows.append({
                "month": month,
                "trades": int(len(grp)),
                "long_trades": long_n,
                "short_trades": short_n,
                "wr": float((grp["pnl_pct"] > 0).mean()),
                "pf": _safe_pf(wins, losses),
                "avg_pnl_pct": float(grp["pnl_pct"].mean()),
                "total_pnl_usd": float(grp["pnl_usd"].sum()),
                "equity_end": eq_end,
                "month_return_pct": float((eq_end - prev_eq) / prev_eq * 100),
                "drawdown_in_month_pct": float(dd_in_month * 100),
                "avg_notional_usd": float(grp["notional_usd"].mean()),
            })
            prev_eq = eq_end
    else:
        # Flat: cumulative PnL over $100 start
        for month, grp in work.groupby("month", sort=True):
            long_n = int((grp["direction"] == "long").sum())
            short_n = int((grp["direction"] == "short").sum())
            wins = grp.loc[grp["pnl_pct"] > 0, "pnl_pct"].sum()
            losses = -grp.loc[grp["pnl_pct"] < 0, "pnl_pct"].sum()
            rows.append({
                "month": month,
                "trades": int(len(grp)),
                "long_trades": long_n,
                "short_trades": short_n,
                "wr": float((grp["pnl_pct"] > 0).mean()),
                "pf": _safe_pf(wins, losses),
                "avg_pnl_pct": float(grp["pnl_pct"].mean()),
                "total_pnl_usd": float(grp["pnl_usd"].sum()),
            })
    return pd.DataFrame(rows)


# =============================================================================
# WRITE HELPERS
# =============================================================================


def write_scenario(run_dir, trades_df, monthly_df, metrics, params,
                   candles_sha, universe_sha, git_dirty, wall_time):
    run_dir.mkdir(parents=True, exist_ok=False)
    trades_df.to_csv(run_dir / "trades.csv", index=False)
    monthly_df.to_csv(run_dir / "monthly.csv", index=False)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)
    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump({
            "run_id": run_dir.name, "git_sha": _git_sha(),
            "git_dirty": git_dirty,
            "snapshot_date": SNAPSHOT_DATE,
            "snapshot_candles_sha256": candles_sha,
            "snapshot_universe_sha256": universe_sha,
            "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_time_seconds": float(wall_time),
            "python_version": sys.version,
            "pip_freeze": _pip_freeze(),
        }, f, indent=2, sort_keys=True, default=str)


# =============================================================================
# MAIN
# =============================================================================


def main():
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    git_dirty = _git_dirty()
    if git_dirty:
        print("WARNING: git dirty — not reproducible.", file=sys.stderr)

    print(f"[combined] loading snapshot {SNAPSHOT_DATE} ...")
    candles, universe = load_snapshot(SNAPSHOT_DATE)
    if pd.api.types.is_datetime64_any_dtype(candles["timestamp"]):
        candles = candles.copy()
        candles["timestamp"] = candles["timestamp"].astype("int64") // 10**9
    print(f"[combined] candles: {len(candles):,} rows")

    universe_assets = apply_universe_filter(universe)
    candles_tradable = candles[candles["asset"].isin(universe_assets)].copy()
    candles_by_asset = {
        asset: g.sort_values("timestamp").reset_index(drop=True)
        for asset, g in candles_tradable.groupby("asset")
    }
    print(f"[combined] tradable assets: {len(candles_by_asset)}")
    listing_gates = compute_listing_gates(candles_by_asset)

    # Detect both-direction masks
    print("[combined] detecting multi_bar_confirm long + short ...")
    masks_by_dir = {}
    for direction in ("long", "short"):
        t0 = _time.monotonic()
        d_masks = {}
        for asset, df in candles_by_asset.items():
            mask = detect_multi_bar_confirm(df, direction)
            gate = listing_gates.get(asset)
            if gate is not None:
                mask = mask & (df["timestamp"] >= gate)
            if mask.any():
                d_masks[asset] = mask
        masks_by_dir[direction] = d_masks
        raw_count = sum(m.sum() for m in d_masks.values())
        print(f"  {direction}: {int(raw_count)} raw signals [{_time.monotonic()-t0:.1f}s]")

    # Simulate with shared 5-slot pool
    print("[combined] running bidirectional simulator (shared 5-slot pool)...")
    t0 = _time.monotonic()
    accepted_trades, rejected_trades = simulate_portfolio_bidirectional(
        candles_by_asset, masks_by_dir, sl_pct=SL_PCT
    )
    sim_dt = _time.monotonic() - t0
    print(f"[combined] accepted={len(accepted_trades)}  rejected_slot_contention="
          f"{len(rejected_trades)}  [{sim_dt:.1f}s]")
    if len(accepted_trades) > 0:
        long_share = (accepted_trades["direction"] == "long").mean()
        print(f"  accepted split: long {long_share*100:.1f}% / short {(1-long_share)*100:.1f}%")

    manifest_path = _SNAPSHOTS_DIR / "MANIFEST.md"
    candles_sha, universe_sha = _parse_manifest_row(
        manifest_path.read_text(), SNAPSHOT_DATE
    )

    sha8 = _git_sha()[:8]
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    base_id = f"backtest_bigmover_combined_v1_{sha8}_{ts_str}"
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)

    # Scenario 1: FLAT
    print("\n[combined] scenario 1: FLAT sizing ...")
    t0 = _time.monotonic()
    flat_trades = apply_flat_sizing(accepted_trades, POSITION_PCT, LEVERAGE)
    flat_m = flat_metrics(flat_trades)
    flat_monthly = monthly_aggregate(flat_trades, compounding=False)
    flat_run_dir = results_root / f"{base_id}__combined__flat"
    write_scenario(
        flat_run_dir, flat_trades, flat_monthly, flat_m,
        {
            "mode": "flat", "signal": "multi_bar_confirm",
            "directions": ["long", "short"], "regime_gate": None,
            "position_pct": POSITION_PCT, "leverage": LEVERAGE, "sl_pct": SL_PCT,
            "trail_pct": TRAIL_PCT, "trail_activation": TRAIL_ACTIVATION,
            "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
            "max_hold_bars": MAX_HOLD_BARS, "start_equity_usd": START_EQUITY_USD,
            "round_trip_fee": ROUND_TRIP_FEE, "snapshot_date": SNAPSHOT_DATE,
        },
        candles_sha, universe_sha, git_dirty, _time.monotonic() - t0,
    )
    print(f"  trades={flat_m['trades']} (L{flat_m['long_trades']}/S{flat_m['short_trades']}) "
          f"wr={flat_m['wr']:.2f} pf={flat_m['pf']:.2f} "
          f"ret=+{flat_m['total_return_pct']:.0f}% dd={flat_m['max_drawdown_pct']:.1f}%")

    # Scenario 2: COMPOUNDING @ $100k cap
    print("\n[combined] scenario 2: COMPOUNDING @ $100k cap ...")
    t0 = _time.monotonic()
    comp_trades = apply_compounding(
        accepted_trades, POSITION_PCT, LEVERAGE, notional_cap=NOTIONAL_CAP_USD,
    )
    comp_m = compound_metrics(comp_trades)
    comp_monthly = monthly_aggregate(comp_trades, compounding=True)
    comp_run_dir = results_root / f"{base_id}__combined__cap100k"
    write_scenario(
        comp_run_dir, comp_trades, comp_monthly, comp_m,
        {
            "mode": "compounding", "signal": "multi_bar_confirm",
            "directions": ["long", "short"], "regime_gate": None,
            "position_pct": POSITION_PCT, "leverage": LEVERAGE, "sl_pct": SL_PCT,
            "notional_cap_usd": NOTIONAL_CAP_USD,
            "trail_pct": TRAIL_PCT, "trail_activation": TRAIL_ACTIVATION,
            "max_concurrent_positions": MAX_CONCURRENT_POSITIONS,
            "max_hold_bars": MAX_HOLD_BARS, "start_equity_usd": START_EQUITY_USD,
            "round_trip_fee": ROUND_TRIP_FEE, "snapshot_date": SNAPSHOT_DATE,
        },
        candles_sha, universe_sha, git_dirty, _time.monotonic() - t0,
    )
    print(f"  trades={comp_m['trades']} (L{comp_m['long_trades']}/S{comp_m['short_trades']}) "
          f"capped={comp_m['trades_capped']} final=${comp_m['final_equity_usd']:,.0f} "
          f"ret=+{comp_m['total_return_pct']:,.0f}% dd={comp_m['max_drawdown_pct']:.1f}%")

    # Short comparison vs prior separate-accounts numbers
    print("\n=== COMPARISON vs separate LONG/SHORT accounts (prior runs) ===")
    print("  Separate LONG only @15/5/7 (flat):    PF 3.09  ret +4,700%  DD -8.1%")
    print("  Separate SHORT only @15/5/7 (flat):   PF 3.14  ret +3,755%  DD -8.1%")
    print(f"  Combined (this run, flat):            PF {flat_m['pf']:.2f}  "
          f"ret +{flat_m['total_return_pct']:,.0f}%  DD {flat_m['max_drawdown_pct']:.1f}%")
    print("  Separate LONG compounding@100k:       final $5,370,240   DD -24.8%")
    print("  Separate SHORT compounding@100k:      final $4,031,310   DD -30.6%")
    print(f"  Combined compounding@100k:            final ${comp_m['final_equity_usd']:,.0f}   "
          f"DD {comp_m['max_drawdown_pct']:.1f}%")

    # Summary CSV
    summary_rows = [
        {"scenario": "combined_flat_15_5_7", **flat_m,
         "run_dir": flat_run_dir.name},
        {"scenario": "combined_compound_15_5_7_cap100k", **comp_m,
         "run_dir": comp_run_dir.name},
    ]
    summary_path = results_root / f"_combined_summary_{sha8}_{ts_str}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n[combined] summary: {summary_path}")
    print(f"[combined] flat monthly: {flat_run_dir / 'monthly.csv'}")
    print(f"[combined] compound monthly: {comp_run_dir / 'monthly.csv'}")


if __name__ == "__main__":
    main()
