"""Backtest template — copy this file to start a new backtest.

Every new backtest MUST:
  - Start from this file: cp backtest_template.py backtest_<name>.py
  - Edit SNAPSHOT_DATE to a date listed in data/snapshots/MANIFEST.md
  - Fill in the strategy logic block (marked with TODO)
  - Leave load_snapshot, compute_metrics, write_results UNTOUCHED

See docs/backtest-protocol.md for the full rules.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

# Module-level names to exclude from params.json even though they're UPPERCASE.
# Extend this set if you `from X import UPPER_NAME` something that isn't a param.
_NON_PARAM_NAMES = frozenset({"UTC"})

# --- CONSTANTS (edit SNAPSHOT_DATE when copying) ---
SNAPSHOT_DATE = "2026-04-01"  # must match a row in data/snapshots/MANIFEST.md
RANDOM_SEED = 42

# --- PARAMS (edit for your strategy) ---
ASSET = "BTCUSDT"
SMA_FAST = 20
SMA_SLOW = 50
HOLD_BARS = 96  # 24h on 15m

# --- CANONICAL HELPERS (do not edit) ---

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Resolve snapshots dir relative to this file so the script can be run from any CWD.
# scripts/ → services/python/ → services/ → repo-root → data/snapshots
_SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file (streaming, safe for large Parquet)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest_row(manifest_text: str, date: str) -> tuple[str, str]:
    """Return (candles_sha, universe_sha) for `date` from MANIFEST.md.

    Only lines whose column 0 is a YYYY-MM-DD date are considered data rows
    (filters out the separator line `|---|---|` and any other `|`-starting content).
    Raises ValueError if the date is not found.
    """
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
        # Columns: Date | Candles rows | Candles sha256 | Universe rows | Universe sha256 | Notes
        return parts[2], parts[4]
    raise ValueError(f"date {date} not in MANIFEST.md")


def load_snapshot(
    date: str,
    snapshots_dir: Path | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load (candles, universe) parquet files for a snapshot date.

    Verifies both files' sha256 against MANIFEST.md. Raises:
      - FileNotFoundError if either Parquet file is missing.
      - ValueError if the date is not listed in MANIFEST.md.
      - ValueError if either file's sha256 does not match the manifest.
    """
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
    """Return current HEAD SHA (40 chars)."""
    out = subprocess.check_output(["git", "rev-parse", "HEAD"])
    return out.decode().strip()


def _git_dirty() -> bool:
    """True if working tree has uncommitted changes."""
    out = subprocess.check_output(["git", "status", "--porcelain"])
    return bool(out.strip())


def _pip_freeze() -> list[str]:
    """Return installed packages, sorted, one per line."""
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
    """Write the four canonical output files to results/<run_id>/.

    Files:
      - trades.csv (one row per closed trade)
      - metrics.json (compute_metrics output)
      - params.json (the strategy params)
      - run_metadata.json (git SHA, dirty flag, snapshot shas, versions, wall time)

    Raises FileExistsError if results/<run_id>/ already exists — never overwrite.
    """
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
    """Canonical PF/WR/avg-PnL calculation.

    trades_df columns:
      - direction: 'long' or 'short'
      - pnl_pct: fractional PnL per trade (0.05 = +5%)
      - bars_held: int

    Returns dict with the keys the protocol requires. The same function is
    used by every backtest so numbers are directly comparable.

    PF conventions: an empty DataFrame yields ``pf = 0.0`` ("no trades, no
    edge"); a non-empty DataFrame with no losing trades yields
    ``pf = float('inf')`` ("all winners, ratio undefined").
    """
    def _stats(df: pd.DataFrame) -> dict:
        n = len(df)
        if n == 0:
            return {"trades": 0, "wr": 0.0, "pf": 0.0, "avg_pnl_pct": 0.0,
                    "total_pnl_pct": 0.0, "median_hold_bars": 0.0}
        wins = df[df["pnl_pct"] > 0]["pnl_pct"].sum()
        losses = -df[df["pnl_pct"] < 0]["pnl_pct"].sum()  # positive number
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
    """Compose the canonical run id for a backtest script invocation.

    Format: {script_stem}_{git_sha[:8]}_{YYYYMMDDTHHMMSSZ}
    """
    script_stem = Path(script_path).stem
    sha8 = _git_sha()[:8]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{script_stem}_{sha8}_{ts}"


# --- MAIN (edit this per backtest) ---


def main() -> None:
    import random
    import time

    import numpy as np

    # Capture before any writes — git_dirty must reflect pre-run state, not the
    # untracked files we are about to create under results/<run_id>/.
    started = time.monotonic()
    git_dirty_at_start = _git_dirty()

    # Seed EVERY source of randomness you touch. Add torch here if you import it.
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    if git_dirty_at_start:
        print(
            "WARNING: git working tree is dirty. "
            "This run cannot be exactly reproduced from git SHA alone.",
            file=sys.stderr,
        )

    candles, universe = load_snapshot(SNAPSHOT_DATE)

    # --- 20/50 SMA crossover on BTC ---
    btc = candles[candles["asset"] == ASSET].sort_values("timestamp").reset_index(drop=True)
    if len(btc) < SMA_SLOW + 2:
        trades = pd.DataFrame(columns=["direction", "pnl_pct", "bars_held"])
    else:
        close = btc["close"].astype(float).values
        fast = pd.Series(close).rolling(SMA_FAST).mean().values
        slow = pd.Series(close).rolling(SMA_SLOW).mean().values
        rows = []
        i = SMA_SLOW
        while i < len(close) - HOLD_BARS - 1:
            if fast[i - 1] <= slow[i - 1] and fast[i] > slow[i]:
                entry = float(close[i + 1])
                exit_ = float(close[i + 1 + HOLD_BARS])
                rows.append({
                    "direction": "long",
                    "pnl_pct": (exit_ - entry) / entry,
                    "bars_held": HOLD_BARS,
                    "entry_idx": int(i + 1),
                })
                i += HOLD_BARS + 1
            else:
                i += 1
        trades = pd.DataFrame(rows) if rows else pd.DataFrame(
            columns=["direction", "pnl_pct", "bars_held"]
        )

    metrics = compute_metrics(trades)

    # Snapshot shas are read from the manifest by a convenience wrapper.
    c_sha, u_sha = _parse_manifest_row(
        (_SNAPSHOTS_DIR / "MANIFEST.md").read_text(), SNAPSHOT_DATE
    )

    run_id = build_run_id(__file__)
    params = {
        k: v for k, v in globals().items()
        if k.isupper()
        and not k.startswith("_")
        and k not in _NON_PARAM_NAMES
    }
    write_results(
        run_id=run_id,
        trades_df=trades,
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
