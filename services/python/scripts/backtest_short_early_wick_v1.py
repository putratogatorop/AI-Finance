"""Short-entry alternative: upper-wick rejection + red follow-through.

Thesis (one sentence): an upper-wick rejection on high volume is the footprint of
aggressive sellers absorbing a rally at a local top; if the next bar closes red
and below the rejection bar's low, distribution is *confirmed in progress* — this
fires hours before price has completed a 5% drop from the 96-bar high.

Protocol-compliant short backtest on SNAPSHOT_DATE = 2026-04-01. Same exit rules
and short-trade simulator as baseline backtest_3strat_protocol.py (SL 5%, TP 15%,
timeout 672, friction 0.15%). The short-trade simulator and canonical helpers are
ported verbatim.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Module-level names to exclude from params.json even though they're UPPERCASE.
_NON_PARAM_NAMES = frozenset({"UTC"})

# --- CONSTANTS ---
SNAPSHOT_DATE = "2026-04-01"
RANDOM_SEED = 42

# --- PARAMS (entry: wick-rejection + red follow-through) ---
VOL_MA_PERIOD = 20
# Prior-bar rejection candle parameters
REJECTION_UPPER_WICK_MIN_FRAC = 0.60   # upper wick ≥ 60% of candle range
REJECTION_BODY_MAX_FRAC = 0.30         # body ≤ 30% of candle range
REJECTION_VOL_MULT = 2.0               # vol_spike on the rejection bar
# "Local high" context for the rejection candle
LOCAL_HIGH_LOOKBACK = 32               # rejection bar must print a 32-bar high
# Current-bar confirmation
CONFIRM_VOL_MULT = 1.0                 # current bar volume ≥ 1× MA20
# Diagnostic only (not an entry gate): how far off the 96-bar high is price when we enter?
DROP_DIAG_LOOKBACK = 96

# Exit rules (same as baseline)
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
COOLDOWN_BARS = 96
FRICTION_PCT = 0.0015

MIN_BARS_PER_ASSET = VOL_MA_PERIOD + max(LOCAL_HIGH_LOOKBACK, DROP_DIAG_LOOKBACK) + 10

# --- CANONICAL HELPERS (do not edit) ---

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest_row(manifest_text: str, date: str) -> tuple[str, str]:
    for line in manifest_text.splitlines():
        if not line.startswith("|"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) < 5:
            continue
        if not _DATE_RE.fullmatch(parts[0]):
            continue
        if parts[0] != date:
            continue
        return parts[2], parts[4]
    raise ValueError(f"date {date} not in MANIFEST.md")


def load_snapshot(
    date: str,
    snapshots_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots_dir = Path(snapshots_dir) if snapshots_dir is not None else _SNAPSHOTS_DIR
    manifest = snapshots_dir / "MANIFEST.md"
    if not manifest.exists():
        raise FileNotFoundError(f"missing MANIFEST.md at {manifest}")
    candles_path = snapshots_dir / f"candles_15m_{date}.parquet"
    universe_path = snapshots_dir / f"universe_{date}.parquet"
    if not candles_path.exists():
        raise FileNotFoundError(
            f"snapshot candles file missing for date {date}. "
            f"Run export_snapshot.py --date {date} or pick a date from MANIFEST.md."
        )
    if not universe_path.exists():
        raise FileNotFoundError(
            f"snapshot universe file missing for date {date}. "
            f"Run export_snapshot.py --date {date} or pick a date from MANIFEST.md."
        )
    expected_candles_sha, expected_universe_sha = _parse_manifest_row(manifest.read_text(), date)
    actual_candles_sha = _sha256_file(candles_path)
    actual_universe_sha = _sha256_file(universe_path)
    if actual_candles_sha != expected_candles_sha:
        raise ValueError(
            f"sha256 mismatch for {candles_path}: expected {expected_candles_sha}, "
            f"got {actual_candles_sha}"
        )
    if actual_universe_sha != expected_universe_sha:
        raise ValueError(
            f"sha256 mismatch for {universe_path}: expected {expected_universe_sha}, "
            f"got {actual_universe_sha}"
        )
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)


def _git_sha() -> str:
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode().strip()


def _git_dirty() -> bool:
    out = subprocess.check_output(["git", "status", "--porcelain"])
    return bool(out.strip())


def _pip_freeze() -> list[str]:
    out = subprocess.check_output([sys.executable, "-m", "pip", "freeze"])
    return sorted(out.decode().strip().splitlines())


def write_results(
    *,
    run_id: str,
    trades_df: pd.DataFrame,
    metrics: dict,
    params: dict,
    snapshot_date: str,
    snapshot_candles_sha256: str,
    snapshot_universe_sha256: str,
    git_dirty: bool,
    wall_time_seconds: float,
    results_root: Path | str = Path("results"),
) -> Path:
    results_root = Path(results_root)
    run_dir = results_root / run_id
    if run_dir.exists():
        raise FileExistsError(f"run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    trades_df.to_csv(run_dir / "trades.csv", index=False)

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, sort_keys=True, default=str)

    with open(run_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2, sort_keys=True, default=str)

    metadata = {
        "run_id": run_id,
        "git_sha": _git_sha(),
        "git_dirty": git_dirty,
        "snapshot_date": snapshot_date,
        "snapshot_candles_sha256": snapshot_candles_sha256,
        "snapshot_universe_sha256": snapshot_universe_sha256,
        "python_version": sys.version,
        "pip_freeze": _pip_freeze(),
        "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wall_time_seconds": float(wall_time_seconds),
    }
    with open(run_dir / "run_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True, default=str)

    return run_dir


def compute_metrics(trades_df: pd.DataFrame) -> dict:
    def _stats(df: pd.DataFrame) -> dict:
        n = len(df)
        if n == 0:
            return {"trades": 0, "wr": 0.0, "pf": 0.0, "avg_pnl_pct": 0.0,
                    "total_pnl_pct": 0.0, "median_hold_bars": 0.0}
        wins = df[df["pnl_pct"] > 0]["pnl_pct"].sum()
        losses = -df[df["pnl_pct"] < 0]["pnl_pct"].sum()
        pf = float("inf") if losses == 0 else float(wins / losses)
        return {
            "trades": int(n),
            "wr": float((df["pnl_pct"] > 0).sum() / n),
            "pf": pf,
            "avg_pnl_pct": float(df["pnl_pct"].mean()),
            "total_pnl_pct": float(df["pnl_pct"].sum()),
            "median_hold_bars": float(df["bars_held"].median()),
        }

    overall = _stats(trades_df)
    overall["by_direction"] = {
        "long": _stats(trades_df[trades_df["direction"] == "long"]),
        "short": _stats(trades_df[trades_df["direction"] == "short"]),
    }
    return overall


# --- RUN ID ---


def build_run_id(script_path: str | Path) -> str:
    script_stem = Path(script_path).stem
    sha8 = _git_sha()[:8]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{script_stem}_{sha8}_{ts}"


# --- SHORT-TRADE SIMULATOR (ported verbatim from backtest_3strat_protocol.py) ---


def _simulate_short_trade(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_bar: int,
) -> dict | None:
    n = len(close)
    if entry_bar + 1 >= n:
        return None
    entry_price = float(close[entry_bar])
    if entry_price <= 0:
        return None
    sl_price = entry_price * (1 + STOP_LOSS_PCT)
    tp_price = entry_price * (1 - TAKE_PROFIT_PCT)
    last_bar = min(entry_bar + 1 + TIMEOUT_BARS, n)
    for bar in range(entry_bar + 1, last_bar):
        if high[bar] >= sl_price:
            return {
                "pnl_pct": -STOP_LOSS_PCT - FRICTION_PCT,
                "exit_reason": "stop_loss",
                "bars_held": int(bar - entry_bar),
                "entry_price": entry_price,
                "exit_price": float(sl_price),
            }
        if low[bar] <= tp_price:
            return {
                "pnl_pct": TAKE_PROFIT_PCT - FRICTION_PCT,
                "exit_reason": "take_profit",
                "bars_held": int(bar - entry_bar),
                "entry_price": entry_price,
                "exit_price": float(tp_price),
            }
    exit_bar = last_bar - 1
    exit_price = float(close[exit_bar])
    return {
        "pnl_pct": (entry_price - exit_price) / entry_price - FRICTION_PCT,
        "exit_reason": "timeout",
        "bars_held": int(exit_bar - entry_bar),
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


# --- NEW ENTRY DETECTOR ---


def _detect_wick_early(
    open_: np.ndarray,
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
) -> list[tuple[int, float]]:
    """Upper-wick rejection at a local high on high volume, confirmed by a red
    break-of-low on the next bar.

    Returns list of (entry_bar_index, drop_pct_from_96_bar_high) tuples so we can
    report how early entries fire compared to the baseline's ~5% drop gate.
    """
    n = len(close)
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().to_numpy()
    # rolling high of the PRIOR bar over LOCAL_HIGH_LOOKBACK bars — use shift(1)
    # so the rejection candle's own high is not included when checking "bar[i-1]
    # made a local high".
    local_high_excl = pd.Series(high).rolling(LOCAL_HIGH_LOOKBACK).max().shift(1).to_numpy()
    # 96-bar rolling high (excluding current bar) — diagnostic only.
    diag_high = pd.Series(high).rolling(DROP_DIAG_LOOKBACK).max().shift(1).to_numpy()

    entries: list[tuple[int, float]] = []
    cooldown_until = 0
    begin = VOL_MA_PERIOD + max(LOCAL_HIGH_LOOKBACK, DROP_DIAG_LOOKBACK) + 1
    for i in range(begin, n):
        if i < cooldown_until:
            continue
        # --- Prior bar must be a valid rejection candle at a local high ---
        j = i - 1
        rng = high[j] - low[j]
        if rng <= 0:
            continue
        body = abs(close[j] - open_[j])
        upper_wick = high[j] - max(close[j], open_[j])
        if upper_wick / rng < REJECTION_UPPER_WICK_MIN_FRAC:
            continue
        if body / rng > REJECTION_BODY_MAX_FRAC:
            continue
        vm_j = vol_ma[j]
        if not (vm_j > 0):
            continue
        if volume[j] / vm_j < REJECTION_VOL_MULT:
            continue
        # Rejection bar must print a local high over the prior LOCAL_HIGH_LOOKBACK
        # bars (the comparison window stops at j-1 thanks to shift(1)).
        lh = local_high_excl[j]
        if not (lh > 0) or high[j] < lh:
            continue
        # --- Current bar must confirm: red, closes below prior LOW, vol not dead ---
        if close[i] >= open_[i]:
            continue
        if close[i] >= low[j]:
            continue
        vm_i = vol_ma[i]
        if not (vm_i > 0):
            continue
        if volume[i] / vm_i < CONFIRM_VOL_MULT:
            continue
        # Diagnostic: drop from 96-bar high AT entry (same definition as baseline).
        rh = diag_high[i]
        drop_pct = float((rh - close[i]) / rh) if (rh > 0) else float("nan")
        entries.append((i, drop_pct))
        cooldown_until = i + COOLDOWN_BARS
    return entries


def _run_strategy(candles: pd.DataFrame) -> pd.DataFrame:
    trades: list[dict] = []
    assets = sorted(candles["asset"].unique())
    for idx, asset in enumerate(assets):
        sub = (
            candles[candles["asset"] == asset]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(sub) < MIN_BARS_PER_ASSET:
            continue
        open_ = sub["open"].to_numpy(dtype=float)
        close = sub["close"].to_numpy(dtype=float)
        high = sub["high"].to_numpy(dtype=float)
        low = sub["low"].to_numpy(dtype=float)
        volume = sub["volume"].to_numpy(dtype=float)
        timestamps = sub["timestamp"].to_numpy()

        for eb, drop_pct in _detect_wick_early(open_, close, high, low, volume):
            t = _simulate_short_trade(close, high, low, eb)
            if t is None:
                continue
            t["direction"] = "short"
            t["symbol"] = asset
            t["strategy"] = "wick_early"
            t["entry_time"] = str(timestamps[eb])
            t["drop_at_entry"] = drop_pct
            trades.append(t)

        if (idx + 1) % 25 == 0:
            print(
                f"  [{idx + 1}/{len(assets)}] trades={len(trades):,}",
                flush=True,
            )

    if not trades:
        return pd.DataFrame(
            columns=["direction", "pnl_pct", "bars_held", "drop_at_entry"]
        )
    return pd.DataFrame(trades)


# --- MAIN ---


def main() -> None:
    import random
    import time

    started = time.monotonic()
    git_dirty_at_start = _git_dirty()

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if git_dirty_at_start:
        print(
            "WARNING: git working tree is dirty. "
            "This run cannot be exactly reproduced from git SHA alone.",
            file=sys.stderr,
        )

    candles, _universe = load_snapshot(SNAPSHOT_DATE)
    print(
        f"loaded {len(candles):,} candles across "
        f"{candles['asset'].nunique()} assets",
        flush=True,
    )

    trades_df = _run_strategy(candles)

    # Extra diagnostic metrics added to metrics.json for comparison with baseline.
    metrics = compute_metrics(trades_df)
    if "drop_at_entry" in trades_df.columns and len(trades_df) > 0:
        drops = trades_df["drop_at_entry"].dropna()
        metrics["drop_at_entry"] = {
            "median": float(drops.median()) if len(drops) else 0.0,
            "mean": float(drops.mean()) if len(drops) else 0.0,
            "pct_under_5pct_drop": float((drops < 0.05).mean()) if len(drops) else 0.0,
            "pct_under_2pct_drop": float((drops < 0.02).mean()) if len(drops) else 0.0,
            "n": int(len(drops)),
        }

    c_sha, u_sha = _parse_manifest_row(
        (_SNAPSHOTS_DIR / "MANIFEST.md").read_text(), SNAPSHOT_DATE
    )

    params = {
        k: v for k, v in globals().items()
        if k.isupper()
        and not k.startswith("_")
        and k not in _NON_PARAM_NAMES
    }

    run_id = build_run_id(__file__)
    write_results(
        run_id=run_id,
        trades_df=trades_df,
        metrics=metrics,
        params=params,
        snapshot_date=SNAPSHOT_DATE,
        snapshot_candles_sha256=c_sha,
        snapshot_universe_sha256=u_sha,
        git_dirty=git_dirty_at_start,
        wall_time_seconds=time.monotonic() - started,
    )
    print(f"run_id: {run_id}")
    print(f"metrics: {metrics}")


if __name__ == "__main__":
    main()
