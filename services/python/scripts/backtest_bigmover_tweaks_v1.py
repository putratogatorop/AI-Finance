"""Big-mover coverage research grid — 9 variants (baseline + 8 tweaks).

All evaluation is OOT-only (last 3 calendar months of the snapshot window).
Coverage metric: of all (pair, day) pairs where intraday open->low drop >= 10%,
what fraction did we enter?

Tweaks:
  0. baseline         — vol_spike + 3-phase, trail 3%, 96-bar cooldown
  1. cooldown_24      — shorter cooldown (24 bars instead of 96)
  2. ml_thresh_055    — deterministic ML proxy: score=vol_ratio/5.0 >= 0.55
                        (illustrative; real ML rerun is a separate project)
  3. vol_rank_pctl90  — only fire signals in top-10% vol_ratio across universe
  4. price_accel_atr  — require |price acceleration| >= 1x ATR
  5. vol_4bar_momentum — union: baseline OR (vol_spike + 4-bar downtrend -2%)
  6. multi_bar_confirm — baseline AND next bar closes lower
  7. btc_correlation  — only fire when |btc_ret_4bar| >= 2%
  8. rank_top3_per_bar — post-filter: keep top-3 by vol_ratio*price_drop per bar

Run from services/python/:
    python scripts/backtest_bigmover_tweaks_v1.py
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

# Module-level names to exclude from params.json even though they're UPPERCASE.
_NON_PARAM_NAMES = frozenset({"UTC"})

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
SNAPSHOT_DATE = "2026-04-23"
RANDOM_SEED = 42

# Universe
TOP_N = 100
MIN_QUOTE_VOL_24H = 500_000
LEVERAGED_TOKEN_RE = r"[35][LS]_USDT$"

# Detector params (baseline = current production)
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
COOLDOWN_BARS = 96  # baseline; tweak 1 changes to 24

# Exit (matches production)
SL_PCT = 0.05
TRAIL_ACTIVATION = 0.02
TRAIL_PCT = 0.03
MAX_HOLD_BARS = 192
WORST_FILL_BUFFER = 0.005

# Sizing
POSITION_PCT = 0.05
LEVERAGE = 5
MAX_CONCURRENT_POSITIONS = 5
FEE_RATE = 0.0006

# OOT split
HOLD_OUT_MONTHS = 3

# Monte Carlo
MC_ITER = 1000
MC_SEED = 42

# Coverage metric
BIG_MOVER_THRESHOLD = 0.10  # 10% intraday max-favorable move

# Tweak-specific params
TWEAK1_COOLDOWN_BARS = 24
TWEAK3_VOL_RANK_PCTL = 90
TWEAK4_ACCEL_MULT_OF_ATR = 1.0
TWEAK7_BTC_RET_THRESHOLD = 0.02
TWEAK8_RANK_TOP_K = 3

# Variant names (order matters — used for iteration)
VARIANTS = [
    "baseline",
    "cooldown_24",
    "ml_thresh_055",
    "vol_rank_pctl90",
    "price_accel_atr",
    "vol_4bar_momentum",
    "multi_bar_confirm",
    "btc_correlation",
    "rank_top3_per_bar",
]

# ---------------------------------------------------------------------------
# CANONICAL HELPERS (copied from backtest_template.py — do not edit)
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"


def _sha256_file(path: Path) -> str:
    """Compute SHA-256 of a file (streaming, safe for large Parquet)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_manifest_row(manifest_text: str, date: str) -> tuple[str, str]:
    """Return (candles_sha, universe_sha) for `date` from MANIFEST.md."""
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
    """Load (candles, universe) parquet files for a snapshot date."""
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
    """Write the four canonical output files to results/<run_id>/."""
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
    """Canonical PF/WR/avg-PnL calculation."""
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


def build_run_id(script_path: str | Path) -> str:
    """Compose the canonical run id for a backtest script invocation."""
    script_stem = Path(script_path).stem
    sha8 = _git_sha()[:8]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{script_stem}_{sha8}_{ts}"


# ---------------------------------------------------------------------------
# STRATEGY HELPERS (copied + adapted from backtest_short_grid_v1.py)
# ---------------------------------------------------------------------------


def apply_universe_filter(universe_df: pd.DataFrame) -> set[str]:
    """Top N by quote_volume_24h, exclude leveraged + delisting + low-volume."""
    leveraged_re = re.compile(LEVERAGED_TOKEN_RE)
    df = universe_df[
        (~universe_df["symbol"].str.contains(leveraged_re, na=False))
        & (~universe_df["in_delisting"].fillna(False))
        & (universe_df["quote_volume_24h"].fillna(0) >= MIN_QUOTE_VOL_24H)
    ].copy()
    df = df.sort_values("quote_volume_24h", ascending=False).head(TOP_N)
    return set(df["symbol"].str.replace("_", "", regex=False).tolist())


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """SMA-based ATR over `period` bars from OHLC."""
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def simulate_portfolio(
    candles_by_asset: dict[str, pd.DataFrame],
    entry_masks: dict[str, pd.Series],
    exit_params: dict,
    cooldown_bars: int = COOLDOWN_BARS,
) -> pd.DataFrame:
    """Single time-ordered event loop across all assets, enforcing MAX_CONCURRENT_POSITIONS.

    The cooldown_bars kwarg overrides the module-level COOLDOWN_BARS constant so
    per-variant cooldown values (e.g., tweak 1 uses 24 bars) can be passed without
    touching module globals.

    Returns trades DataFrame with columns:
      direction, pnl_pct, bars_held, symbol, entry_time, exit_time,
      exit_reason, peak_pnl_pct, entry_price, exit_price.
    """
    sl_pct = exit_params["sl_pct"]
    tp_pct = exit_params.get("tp_pct")
    trail_pct = exit_params.get("trail_pct")
    trail_activation = exit_params.get("trail_activation")

    candidate_trades = []
    for asset, df in candles_by_asset.items():
        if asset not in entry_masks:
            continue
        mask = entry_masks[asset]
        if not mask.any():
            continue
        ts = df["timestamp"].values
        open_ = df["open"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        n = len(df)
        last_entry_bar = -cooldown_bars
        for i in np.where(mask.values)[0]:
            if i - last_entry_bar < cooldown_bars:
                continue
            if i >= n:
                continue
            entry_price = float(open_[i])
            if entry_price <= 0:
                continue
            sl_trigger = entry_price * (1 + sl_pct + WORST_FILL_BUFFER)
            tp_trigger = entry_price * (1 - tp_pct) if tp_pct is not None else None
            peak_low = entry_price
            exit_price = entry_price
            exit_bar = min(i + MAX_HOLD_BARS, n - 1)
            exit_reason = "timeout"
            for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
                bar_high = high[j]
                bar_low = low[j]
                if bar_low < peak_low:
                    peak_low = bar_low
                if bar_high >= sl_trigger:
                    exit_price = sl_trigger
                    exit_bar = j
                    exit_reason = "stop_loss"
                    break
                if tp_trigger is not None and bar_low <= tp_trigger:
                    exit_price = tp_trigger
                    exit_bar = j
                    exit_reason = "take_profit"
                    break
                if trail_pct is not None and trail_activation is not None:
                    activation_price = entry_price * (1 - trail_activation)
                    if peak_low <= activation_price:
                        trail_price = peak_low * (1 + trail_pct)
                        if bar_high >= trail_price:
                            exit_price = trail_price
                            exit_bar = j
                            exit_reason = "trail_stop"
                            break
            pnl_pct = (entry_price - exit_price) / entry_price
            peak_pnl_pct = (entry_price - peak_low) / entry_price
            candidate_trades.append({
                "direction": "short",
                "pnl_pct": pnl_pct,
                "bars_held": exit_bar - i,
                "symbol": asset,
                "entry_time": int(ts[i]),
                "exit_time": int(ts[exit_bar]),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "peak_pnl_pct": peak_pnl_pct,
                "exit_reason": exit_reason,
            })
            last_entry_bar = i

    if not candidate_trades:
        return pd.DataFrame(columns=[
            "direction", "pnl_pct", "bars_held", "symbol", "entry_time",
            "exit_time", "entry_price", "exit_price", "peak_pnl_pct",
            "exit_reason",
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


# ---------------------------------------------------------------------------
# NEW HELPERS
# ---------------------------------------------------------------------------


def compute_oot_split(candles: pd.DataFrame, hold_out_months: int = 3) -> tuple[int, int]:
    """Return (oot_start_ts, max_ts) as epoch seconds. OOT = last hold_out_months of data."""
    max_ts = int(candles["timestamp"].max())
    oot_start_ts = max_ts - (hold_out_months * 30 * 24 * 3600)
    return oot_start_ts, max_ts


def compute_big_mover_coverage(
    trades_df: pd.DataFrame,
    candles_by_asset: dict,
    threshold: float = BIG_MOVER_THRESHOLD,
    oot_start_ts: int = 0,
) -> float:
    """Coverage metric: fraction of big-mover (pair, day) events we entered on.

    Big mover = a day where open-to-low drop >= threshold (short-favorable).
    Returns float in [0, 1].
    """
    big_movers: set[tuple[str, int]] = set()
    for asset, df in candles_by_asset.items():
        oot_df = df[df["timestamp"] >= oot_start_ts]
        if len(oot_df) == 0:
            continue
        oot_df = oot_df.copy()
        oot_df["day"] = (oot_df["timestamp"] // 86400).astype(int)
        for day_id, day_df in oot_df.groupby("day"):
            day_open = float(day_df["open"].iloc[0])
            day_low = float(day_df["low"].min())
            if day_open <= 0:
                continue
            max_drop = (day_open - day_low) / day_open
            if max_drop >= threshold:
                big_movers.add((asset, int(day_id)))
    if not big_movers:
        return 0.0
    entered: set[tuple[str, int]] = set()
    if "symbol" in trades_df.columns and "entry_time" in trades_df.columns:
        for _, row in trades_df.iterrows():
            entered.add((str(row["symbol"]), int(row["entry_time"]) // 86400))
    caught = big_movers & entered
    return len(caught) / len(big_movers)


def monte_carlo_pf(trades_df: pd.DataFrame, n_iter: int = MC_ITER, seed: int = MC_SEED) -> dict:
    """Bootstrap trade-order, return {p5, p50, p95} of PF distribution."""
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
    finite_pfs = [p for p in pfs if p != float("inf")]
    if not finite_pfs:
        return {"p5": float("inf"), "p50": float("inf"), "p95": float("inf")}
    return {
        "p5": float(np.percentile(finite_pfs, 5)),
        "p50": float(np.percentile(finite_pfs, 50)),
        "p95": float(np.percentile(finite_pfs, 95)),
    }


# ---------------------------------------------------------------------------
# DETECTOR LOGIC
# ---------------------------------------------------------------------------


def baseline_entry_mask(df: pd.DataFrame) -> pd.Series:
    """Vol-spike + price-drop bearish dominance — mirror of detect_vol_spike from
    backtest_short_grid_v1.py (DO NOT change; this is the production baseline).
    """
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    volume = df["volume"].astype(float)
    n = len(df)
    out = pd.Series([False] * n, index=df.index)
    if n < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return out
    vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
    rolling_high = high.rolling(PRICE_LOOKBACK).max()
    rolling_low = low.rolling(PRICE_LOOKBACK).min()
    price_drop = (rolling_high - close) / rolling_high
    price_rise = (close - rolling_low) / rolling_low
    vol_ratio = volume / vol_ma
    return (
        (vol_ratio >= VOL_SPIKE)
        & (price_drop >= PRICE_MOVE_THRESH)
        & (price_drop > price_rise)
    ).fillna(False)


def detect_with_tweak(df: pd.DataFrame, tweak: str, atr: pd.Series) -> pd.Series:
    """Return entry mask for a given tweak, layered on top of the baseline.

    Cross-asset tweaks (vol_rank_pctl90, btc_correlation, rank_top3_per_bar)
    are marked with a sentinel False series here; they are handled in the run loop.
    """
    base = baseline_entry_mask(df)

    if tweak == "baseline":
        return base

    if tweak == "cooldown_24":
        # Cooldown reduction is enforced in simulate_portfolio via cooldown_bars kwarg.
        # Signal mask itself is identical to baseline.
        return base

    if tweak == "ml_thresh_055":
        # Deterministic ML proxy: score = min(1.0, vol_ratio / 5.0).
        # Require score >= 0.55, i.e. vol_ratio >= 2.75.
        # This is illustrative — a real ML rerun (with joblib model) is a separate project.
        volume = df["volume"].astype(float)
        vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
        vol_ratio = (volume / vol_ma).clip(upper=5.0)
        score = vol_ratio / 5.0
        return base & (score >= 0.55)

    if tweak == "price_accel_atr":
        prev_close = df["close"].shift(1)
        prev_prev = df["close"].shift(2)
        accel = (df["close"] - prev_close) - (prev_close - prev_prev)
        return base & (accel.abs() >= atr * TWEAK4_ACCEL_MULT_OF_ATR)

    if tweak == "vol_4bar_momentum":
        # Union: baseline OR (vol_spike AND 4-bar downtrend >= 2%)
        volume = df["volume"].astype(float)
        vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
        ratio = volume / vol_ma
        prev4 = df["close"].shift(4)
        ret4 = (df["close"] - prev4) / prev4.replace(0, float("nan"))
        secondary = (ratio >= VOL_SPIKE) & (ret4 < -0.02)
        return (base | secondary).fillna(False)

    if tweak == "multi_bar_confirm":
        # Require base signal AND next bar closes lower (shift(-1) looks ahead — entry is
        # the bar after signal, so confirmation is known before we act).
        next_close = df["close"].shift(-1)
        confirm = next_close < df["close"]
        return (base & confirm).fillna(False)

    # Cross-asset tweaks handled outside this function — return baseline as placeholder.
    # vol_rank_pctl90, btc_correlation, rank_top3_per_bar are all post-processed
    # in the main run loop after per-pair masks are assembled.
    if tweak in ("vol_rank_pctl90", "btc_correlation", "rank_top3_per_bar"):
        return base

    raise ValueError(f"unknown tweak: {tweak}")


# ---------------------------------------------------------------------------
# CROSS-ASSET POST-FILTERS
# ---------------------------------------------------------------------------


def apply_vol_rank_filter(
    entry_masks: dict[str, pd.Series],
    candles_by_asset: dict[str, pd.DataFrame],
    pctl: float = TWEAK3_VOL_RANK_PCTL,
) -> dict[str, pd.Series]:
    """Keep only signals where the pair's vol_ratio is in the top pctl% across
    the universe at that timestamp.

    Approach: build a vol_ratio matrix (timestamp x asset), compute row-wise
    percentile cutoff, mask out entries below the cutoff.

    NOTE: This is an approximate cross-asset filter. The vol_ratio values are
    computed per-pair independently (no survivorship adjustment). Bars with fewer
    than 10 active pairs fall back to the per-pair mask unchanged.
    """
    # Build aligned vol_ratio DataFrame
    vol_ratio_dict: dict[str, pd.Series] = {}
    ts_index_dict: dict[str, pd.Series] = {}
    for asset, df in candles_by_asset.items():
        if asset not in entry_masks:
            continue
        volume = df["volume"].astype(float)
        vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
        vr = (volume / vol_ma).fillna(0.0)
        vol_ratio_dict[asset] = vr.values
        ts_index_dict[asset] = df["timestamp"].values

    if not vol_ratio_dict:
        return entry_masks

    # Align on a common timestamp index (union of all)
    all_ts = sorted(set(
        int(ts)
        for arr in ts_index_dict.values()
        for ts in arr
    ))
    ts_to_idx: dict[int, int] = {ts: i for i, ts in enumerate(all_ts)}
    n_ts = len(all_ts)
    assets = list(vol_ratio_dict.keys())
    mat = np.zeros((n_ts, len(assets)), dtype=float)
    for col_i, asset in enumerate(assets):
        ts_arr = ts_index_dict[asset]
        vr_arr = vol_ratio_dict[asset]
        for row_i, (ts, vr) in enumerate(zip(ts_arr, vr_arr)):
            mat[ts_to_idx[int(ts)], col_i] = vr

    # Row-wise percentile cutoff
    cutoff = np.percentile(mat, pctl, axis=1)  # shape (n_ts,)

    # Rebuild masks: only keep True where vol_ratio >= row cutoff
    new_masks: dict[str, pd.Series] = {}
    for col_i, asset in enumerate(assets):
        old_mask = entry_masks[asset]
        df = candles_by_asset[asset]
        ts_arr = df["timestamp"].values
        new_vals = old_mask.values.copy()
        for row_i, ts in enumerate(ts_arr):
            if not new_vals[row_i]:
                continue
            global_row = ts_to_idx.get(int(ts))
            if global_row is None:
                new_vals[row_i] = False
                continue
            if mat[global_row, col_i] < cutoff[global_row]:
                new_vals[row_i] = False
        new_masks[asset] = pd.Series(new_vals, index=old_mask.index)
    return new_masks


def apply_btc_filter(
    entry_masks: dict[str, pd.Series],
    candles_by_asset: dict[str, pd.DataFrame],
    btc_ret_4bar: pd.Series | None,
    btc_df: pd.DataFrame | None,
    threshold: float = TWEAK7_BTC_RET_THRESHOLD,
) -> dict[str, pd.Series]:
    """Keep only signals at bars where |btc_ret_4bar| >= threshold.

    Maps BTC timestamps to a bool lookup, then masks per-pair entries.
    NOTE: BTC must be in the universe (key "BTCUSDT" or similar). If BTC data
    is unavailable, entries are unfiltered (conservative fallback).
    """
    if btc_ret_4bar is None or btc_df is None:
        return entry_masks

    btc_ts = btc_df["timestamp"].values
    btc_active = (btc_ret_4bar.abs() >= threshold).fillna(False).values
    ts_to_active: dict[int, bool] = {
        int(ts): bool(active)
        for ts, active in zip(btc_ts, btc_active)
    }

    new_masks: dict[str, pd.Series] = {}
    for asset, mask in entry_masks.items():
        df = candles_by_asset[asset]
        ts_arr = df["timestamp"].values
        new_vals = mask.values.copy()
        for i, ts in enumerate(ts_arr):
            if new_vals[i] and not ts_to_active.get(int(ts), False):
                new_vals[i] = False
        new_masks[asset] = pd.Series(new_vals, index=mask.index)
    return new_masks


def apply_rank_top3_filter(
    entry_masks: dict[str, pd.Series],
    candles_by_asset: dict[str, pd.DataFrame],
    top_k: int = TWEAK8_RANK_TOP_K,
) -> dict[str, pd.Series]:
    """Post-filter: at each timestamp, keep only top_k signals by vol_ratio * price_drop.

    Walk timestamps in ascending order. At each timestamp, gather all (asset, score)
    pairs with an active signal; rank descending; zero out all but top_k.

    NOTE: approximate — uses the bar's own vol_ratio and close/rolling_high price_drop.
    Pairs without 20+ bars of history will have NaN scores and are dropped.
    """
    # Precompute score series per asset
    score_series: dict[str, np.ndarray] = {}
    ts_series: dict[str, np.ndarray] = {}
    for asset, df in candles_by_asset.items():
        if asset not in entry_masks:
            continue
        volume = df["volume"].astype(float)
        vol_ma = volume.rolling(VOL_MA_PERIOD).mean()
        vol_ratio = volume / vol_ma
        high = df["high"].astype(float)
        close = df["close"].astype(float)
        rolling_high = high.rolling(PRICE_LOOKBACK).max()
        price_drop = (rolling_high - close) / rolling_high.replace(0, float("nan"))
        scores = (vol_ratio * price_drop).fillna(0.0).values
        score_series[asset] = scores
        ts_series[asset] = df["timestamp"].values

    # Build unified sorted timestamp list where at least one signal fires
    all_signal_ts: set[int] = set()
    for asset, mask in entry_masks.items():
        df = candles_by_asset[asset]
        fired_ts = df.loc[mask, "timestamp"].values
        for ts in fired_ts:
            all_signal_ts.add(int(ts))

    # For each bar with signals, rank and keep top_k
    # Build position index maps per asset
    asset_ts_to_row: dict[str, dict[int, int]] = {}
    for asset, ts_arr in ts_series.items():
        asset_ts_to_row[asset] = {int(ts): i for i, ts in enumerate(ts_arr)}

    # Start with copies of existing mask arrays
    new_mask_vals: dict[str, np.ndarray] = {
        asset: mask.values.copy() for asset, mask in entry_masks.items()
    }

    for ts in sorted(all_signal_ts):
        candidates = []
        for asset in entry_masks:
            row_map = asset_ts_to_row.get(asset, {})
            row = row_map.get(ts)
            if row is None:
                continue
            if not new_mask_vals[asset][row]:
                continue
            score = float(score_series[asset][row]) if asset in score_series else 0.0
            candidates.append((score, asset, row))
        if len(candidates) <= top_k:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        for _, asset, row in candidates[top_k:]:
            new_mask_vals[asset][row] = False

    return {
        asset: pd.Series(new_mask_vals[asset], index=entry_masks[asset].index)
        for asset in entry_masks
    }


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main() -> None:
    import random

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    overall_start = _time.monotonic()
    git_dirty_at_start = _git_dirty()
    if git_dirty_at_start:
        print(
            "WARNING: git working tree is dirty. "
            "This run cannot be exactly reproduced from git SHA alone.",
            file=sys.stderr,
        )

    print(f"[bigmover] loading snapshot {SNAPSHOT_DATE} ...")
    candles, universe = load_snapshot(SNAPSHOT_DATE)
    print(f"[bigmover] candles: {len(candles):,} rows, universe: {len(universe)} pairs")

    universe_assets = apply_universe_filter(universe)
    print(f"[bigmover] tradable universe: {len(universe_assets)} pairs")

    candles = candles[candles["asset"].isin(universe_assets)].copy()
    print(f"[bigmover] candles for tradable assets: {len(candles):,} rows")

    candles_by_asset = {
        asset: g.sort_values("timestamp").reset_index(drop=True)
        for asset, g in candles.groupby("asset")
    }
    print(f"[bigmover] candles grouped: {len(candles_by_asset)} assets")

    # OOT split
    oot_start_ts, max_ts = compute_oot_split(candles, HOLD_OUT_MONTHS)
    oot_start_dt = datetime.fromtimestamp(oot_start_ts, tz=UTC)
    max_ts_dt = datetime.fromtimestamp(max_ts, tz=UTC)
    print(f"[bigmover] OOT window: {oot_start_dt.date()} -> {max_ts_dt.date()}")

    # BTC 4-bar return series (for tweak 7)
    btc_key = next(
        (k for k in candles_by_asset if "BTC" in k and "USDT" in k and len(k) <= 7),
        None,
    )
    # Try common key forms: BTCUSDT, BTC_USDT stripped
    for candidate in ["BTCUSDT", "BTCUSDT_PERP", "BTCPERP"]:
        if candidate in candles_by_asset:
            btc_key = candidate
            break
    btc_df = candles_by_asset.get(btc_key) if btc_key else None
    if btc_df is not None:
        btc_ret_4bar = btc_df["close"] / btc_df["close"].shift(4) - 1
        print(f"[bigmover] BTC key: {btc_key}")
    else:
        btc_ret_4bar = None
        print("[bigmover] WARNING: BTC not found in universe — tweak 7 will be unfiltered")

    # Snapshot shas for write_results
    manifest_path = Path(__file__).resolve().parents[3] / "data" / "snapshots" / "MANIFEST.md"
    manifest_text = manifest_path.read_text()
    candles_sha, universe_sha = _parse_manifest_row(manifest_text, SNAPSHOT_DATE)

    exit_params = {
        "sl_pct": SL_PCT,
        "tp_pct": None,
        "trail_pct": TRAIL_PCT,
        "trail_activation": TRAIL_ACTIVATION,
    }

    base_run_id = build_run_id(__file__)

    summary_rows: list[dict] = []

    for variant in VARIANTS:
        v_start = _time.monotonic()
        print(f"\n[bigmover] variant: {variant} ...")

        cooldown_bars = TWEAK1_COOLDOWN_BARS if variant == "cooldown_24" else COOLDOWN_BARS

        # Per-pair detection (baseline + per-pair tweaks)
        entry_masks: dict[str, pd.Series] = {}
        for asset, df in candles_by_asset.items():
            atr = compute_atr(df)
            mask = detect_with_tweak(df, variant, atr)
            # Restrict to OOT bars only
            oot_mask = df["timestamp"] >= oot_start_ts
            mask = mask & oot_mask
            if mask.any():
                entry_masks[asset] = mask

        # Cross-asset post-filters
        if variant == "vol_rank_pctl90":
            # NOTE: vol_rank_pctl90 post-filter is approximate — it ranks vol_ratio
            # across the universe at each bar using independently-computed vol_ma per pair.
            # Pairs with insufficient history (< VOL_MA_PERIOD bars) have vol_ratio=NaN
            # and are treated as 0, which may undercount their rank. The OOT mask is applied
            # before ranking, so only OOT bars participate.
            entry_masks = apply_vol_rank_filter(entry_masks, candles_by_asset)

        elif variant == "btc_correlation":
            entry_masks = apply_btc_filter(
                entry_masks, candles_by_asset, btc_ret_4bar, btc_df
            )

        elif variant == "rank_top3_per_bar":
            # NOTE: rank_top3_per_bar scores are vol_ratio * price_drop computed from
            # full-history rolling windows (not re-computed OOT-only), which is correct
            # for ranking at each bar but means pre-OOT rolling state influences scores.
            entry_masks = apply_rank_top3_filter(entry_masks, candles_by_asset)

        # Simulate
        trades_df = simulate_portfolio(
            candles_by_asset, entry_masks, exit_params, cooldown_bars=cooldown_bars
        )

        # Compute metrics and coverage
        metrics = compute_metrics(trades_df)
        big_mover_cov = compute_big_mover_coverage(
            trades_df, candles_by_asset, BIG_MOVER_THRESHOLD, oot_start_ts
        )
        mc = monte_carlo_pf(trades_df)

        # Winners subset
        winners = trades_df[trades_df["pnl_pct"] > 0]["pnl_pct"]
        avg_winner_pct = float(winners.mean()) if len(winners) > 0 else 0.0
        median_winner_pct = float(winners.median()) if len(winners) > 0 else 0.0

        # Write results
        run_id = f"{base_run_id}__{variant}"
        params = {
            "variant": variant,
            "SNAPSHOT_DATE": SNAPSHOT_DATE,
            "HOLD_OUT_MONTHS": HOLD_OUT_MONTHS,
            "oot_start_ts": oot_start_ts,
            "cooldown_bars": cooldown_bars,
            "VOL_SPIKE": VOL_SPIKE,
            "PRICE_MOVE_THRESH": PRICE_MOVE_THRESH,
            "SL_PCT": SL_PCT,
            "TRAIL_PCT": TRAIL_PCT,
            "TRAIL_ACTIVATION": TRAIL_ACTIVATION,
            "MAX_HOLD_BARS": MAX_HOLD_BARS,
            "POSITION_PCT": POSITION_PCT,
            "LEVERAGE": LEVERAGE,
            "MAX_CONCURRENT_POSITIONS": MAX_CONCURRENT_POSITIONS,
            "FEE_RATE": FEE_RATE,
            "BIG_MOVER_THRESHOLD": BIG_MOVER_THRESHOLD,
        }
        write_results(
            run_id=run_id,
            trades_df=trades_df,
            metrics=metrics,
            params=params,
            snapshot_date=SNAPSHOT_DATE,
            snapshot_candles_sha256=candles_sha,
            snapshot_universe_sha256=universe_sha,
            git_dirty=git_dirty_at_start,
            wall_time_seconds=_time.monotonic() - v_start,
        )

        v_elapsed = _time.monotonic() - v_start
        print(
            f"  trades={metrics['trades']:5d}  wr={metrics['wr']:.3f}  "
            f"pf={metrics['pf']:.3f}  coverage={big_mover_cov:.3f}  "
            f"mc_p50={mc['p50']:.3f}  ({v_elapsed:.1f}s)"
        )

        summary_rows.append({
            "variant": variant,
            "trades": metrics["trades"],
            "wr": metrics["wr"],
            "pf": metrics["pf"],
            "pf_p5": mc["p5"],
            "pf_p50": mc["p50"],
            "pf_p95": mc["p95"],
            "avg_winner_pct": avg_winner_pct,
            "median_winner_pct": median_winner_pct,
            "big_mover_coverage_pct": big_mover_cov,
            "run_id": run_id,
        })

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows).sort_values("big_mover_coverage_pct", ascending=False)
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)
    git_sha = _git_sha()
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = results_root / f"_bigmover_tweaks_summary_{git_sha[:8]}_{ts_str}.csv"
    summary_df.to_csv(summary_path, index=False)

    total_elapsed = _time.monotonic() - overall_start
    print(f"\n[bigmover] done in {total_elapsed:.0f}s ({total_elapsed / 60:.1f}m)")
    print(f"[bigmover] summary: {summary_path}")
    print("\nSummary (sorted by big_mover_coverage_pct):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
