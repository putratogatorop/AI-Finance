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

# --- PARAMS — Strategy B (wait_bounce) entry, ported from backtest_3strat_protocol.py ---
VOL_MA_PERIOD = 20
VOL_SPIKE_B = 2.0
PRICE_MOVE_THRESH = 0.05
PRICE_LOOKBACK_B = 32
COOLDOWN_BARS = 96

# --- PARAMS — exit (unchanged from baseline) ---
STOP_LOSS_PCT = 0.05
TAKE_PROFIT_PCT = 0.15
TIMEOUT_BARS = 672
FRICTION_PCT = 0.0015

# --- PARAMS — regime filter (the new gates) ---
BTC_ASSET = "BTCUSDT"
BTC_TREND_LOOKBACK = 96       # bars over which we measure BTC pct change
BTC_FLAT_BAND = 0.02          # +/- 2% counts as "flat"
ASSET_VOL_LOOKBACK = 96       # bars for realized vol per asset
VOL_DECILE_MAX = 2            # keep entries only when asset is in deciles 0..2 cross-sectionally

# --- PARAMS — OOT split per the 8-point standard ---
OOT_SPLIT_DATE = "2026-01-01"  # train ends here; OOT = (OOT_SPLIT_DATE, SNAPSHOT_DATE)

MIN_BARS_PER_ASSET = VOL_MA_PERIOD + max(PRICE_LOOKBACK_B, ASSET_VOL_LOOKBACK) + 10

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


import numpy as np


def _simulate_short_trade(
    close: np.ndarray, high: np.ndarray, low: np.ndarray, entry_bar: int,
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


def _detect_b_entries(close, high, volume) -> list[int]:
    """Strategy B raw entries — same as backtest_3strat_protocol.py.

    Filtered downstream by BTC regime + cross-sectional vol decile.
    """
    n = len(close)
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().to_numpy()
    rolling_high = pd.Series(high).rolling(PRICE_LOOKBACK_B).max().shift(1).to_numpy()
    vol_ratio = np.where(vol_ma > 0, volume / np.where(vol_ma > 0, vol_ma, 1.0), 0.0)
    rolling_vol_max = pd.Series(vol_ratio).rolling(PRICE_LOOKBACK_B + 1).max().to_numpy()
    entries: list[int] = []
    cooldown_until = 0
    begin = VOL_MA_PERIOD + PRICE_LOOKBACK_B
    for i in range(begin, n):
        if i < cooldown_until:
            continue
        rh = rolling_high[i]
        if not (rh > 0):
            continue
        if (rh - close[i]) / rh < PRICE_MOVE_THRESH:
            continue
        rvm = rolling_vol_max[i]
        if not (rvm >= VOL_SPIKE_B):
            continue
        entries.append(i)
        cooldown_until = i + COOLDOWN_BARS
    return entries


def _build_btc_regime_series(candles: pd.DataFrame) -> pd.Series:
    """Map each timestamp → 'btc_up' / 'btc_flat' / 'btc_down'.

    Regime = sign of (close[t] - close[t-96]) / close[t-96] vs +/- BTC_FLAT_BAND.
    """
    btc = (
        candles[candles["asset"] == BTC_ASSET]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
        .astype(float)
    )
    pct = btc.pct_change(BTC_TREND_LOOKBACK)
    regime = pd.Series("btc_flat", index=pct.index)
    regime[pct > BTC_FLAT_BAND] = "btc_up"
    regime[pct < -BTC_FLAT_BAND] = "btc_down"
    return regime


def _build_vol_decile_panel(candles: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional vol-decile (0..9) per (timestamp, asset).

    Step 1: per-asset 96-bar realized vol = stddev of 15m log returns.
    Step 2: pivot wide (timestamp x asset).
    Step 3: rank assets cross-sectionally per timestamp via .rank(pct=True), then
            multiply by 10 floored to 0..9; deciles are recomputed only over
            assets with non-NaN vol at that timestamp (universe-time-correct).
    """
    df = candles[["timestamp", "asset", "close"]].copy()
    df = df.sort_values(["asset", "timestamp"])
    df["log_ret"] = (
        df.groupby("asset")["close"]
          .transform(lambda s: np.log(s).diff())
    )
    df["vol96"] = (
        df.groupby("asset")["log_ret"]
          .transform(lambda s: s.rolling(ASSET_VOL_LOOKBACK).std())
    )
    wide = df.pivot(index="timestamp", columns="asset", values="vol96")
    pct_rank = wide.rank(axis=1, pct=True)  # NaN -> NaN, others ranked among themselves
    decile = (pct_rank * 10.0 - 1e-9).clip(lower=0).apply(np.floor).astype("Int64")
    return decile  # cells: Int64 0..9 or pd.NA


def _run_strategy_b_filtered(candles: pd.DataFrame) -> pd.DataFrame:
    """Detect Strategy B entries, then keep only those that pass the regime filter."""
    print("[1/3] computing BTC regime series...", flush=True)
    btc_regime = _build_btc_regime_series(candles)

    print("[2/3] computing cross-sectional vol-decile panel "
          "(this is the heavy step)...", flush=True)
    decile_panel = _build_vol_decile_panel(candles)

    print("[3/3] running Strategy B per asset, filtering, simulating shorts...",
          flush=True)
    rows = []
    assets = sorted(candles["asset"].unique())
    for idx, asset in enumerate(assets):
        sub = (
            candles[candles["asset"] == asset]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(sub) < MIN_BARS_PER_ASSET:
            continue
        close = sub["close"].to_numpy(dtype=float)
        high = sub["high"].to_numpy(dtype=float)
        low = sub["low"].to_numpy(dtype=float)
        volume = sub["volume"].to_numpy(dtype=float)
        timestamps = sub["timestamp"].to_numpy()

        for eb in _detect_b_entries(close, high, volume):
            entry_ts = timestamps[eb]

            # Filter 1: BTC regime must be flat at entry
            try:
                rg = btc_regime.loc[entry_ts]
            except KeyError:
                continue
            if rg != "btc_flat":
                continue

            # Filter 2: asset's cross-sectional vol decile <= VOL_DECILE_MAX at entry
            try:
                d = decile_panel.at[entry_ts, asset]
            except KeyError:
                continue
            if d is pd.NA or pd.isna(d):
                continue
            if int(d) > VOL_DECILE_MAX:
                continue

            t = _simulate_short_trade(close, high, low, eb)
            if t is None:
                continue
            t["direction"] = "short"
            t["symbol"] = asset
            t["entry_time"] = str(entry_ts)
            t["btc_regime"] = rg
            t["vol_decile"] = int(d)
            rows.append(t)

        if (idx + 1) % 25 == 0:
            print(f"  [{idx + 1}/{len(assets)}] kept-trades-so-far={len(rows):,}",
                  flush=True)

    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["direction", "pnl_pct", "bars_held"])


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
    print(f"loaded {len(candles):,} candles across "
          f"{candles['asset'].nunique()} assets", flush=True)

    trades_df = _run_strategy_b_filtered(candles)

    # Tag train vs OOT — boundary is OOT_SPLIT_DATE 00:00 UTC.
    if len(trades_df):
        ts = pd.to_datetime(trades_df["entry_time"], utc=True)
        boundary = pd.Timestamp(OOT_SPLIT_DATE, tz="UTC")
        trades_df["period"] = np.where(ts < boundary, "train", "oot")
    else:
        trades_df["period"] = pd.Series([], dtype="object")

    overall_metrics = compute_metrics(trades_df)
    train_metrics = compute_metrics(
        trades_df[trades_df["period"] == "train"]
        if len(trades_df) else trades_df
    )
    oot_metrics = compute_metrics(
        trades_df[trades_df["period"] == "oot"]
        if len(trades_df) else trades_df
    )
    metrics = {
        "overall": overall_metrics,
        "train": train_metrics,
        "oot": oot_metrics,
        "ship_floor_pf": 1.30,
        "oot_passes_ship_floor": bool(oot_metrics["pf"] >= 1.30 and oot_metrics["trades"] >= 100),
    }

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
    print(f"overall: {overall_metrics}")
    print(f"train:   {train_metrics}")
    print(f"oot:     {oot_metrics}")
    print(f"ship_floor (PF>=1.30, n>=100 OOT trades): "
          f"{'PASS' if metrics['oot_passes_ship_floor'] else 'FAIL'}")


if __name__ == "__main__":
    main()
