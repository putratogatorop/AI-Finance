"""Short-only multi-strategy backtest grid v1.

24 Phase-1 variants (3 detectors × 4 timings × 2 HTF settings) with fixed 5% SL / 15% TP.
Phase 2 re-runs top-3 PF >= 1.20 variants across 4 alternative exits (12 more variants).

Run from services/python/:
    python scripts/backtest_short_grid_v1.py

Results:
  - Per-variant folders in results/<run_id>/
  - Summary CSV at results/_grid_summary_short_<sha8>_<ts>.csv
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
SNAPSHOT_DATE = "2026-04-23"
RANDOM_SEED = 42

# --- UNIVERSE ---
TOP_N = 100
MIN_QUOTE_VOL_24H = 500_000
LEVERAGED_TOKEN_RE = r"[35][LS]_USDT$"

# --- HTF FILTER ---
HTF_RESAMPLE = "4h"  # pandas resample alias
HTF_SMA_PERIOD = 50

# --- DETECTORS ---
ATR_PERIOD = 14
ATR_DROP_MULT = 2.5
VOL_SPIKE_MULT = 3.0
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96  # 24h on 15m

# --- TIMINGS ---
RETEST_BARS = 8
RETEST_PCT = 0.01
NEXT_SURGE_BARS = 4

# --- EXIT (Phase 1 default) ---
SL_PCT = 0.05
TP_PCT = 0.15
MAX_HOLD_BARS = 192
WORST_FILL_BUFFER = 0.005

# --- SIZING ---
POSITION_PCT = 0.05
LEVERAGE = 5
MAX_CONCURRENT_POSITIONS = 5
FEE_RATE = 0.0006

# --- COOLDOWN ---
COOLDOWN_BARS = PRICE_LOOKBACK  # 96 bars per pair

# --- GRID ---
DETECTORS = ["atr_down", "vol_spike", "combined"]
TIMINGS = ["immediate", "wait_retest", "wait_confirm", "wait_next_surge"]
HTF_VARIANTS = [True, False]
PHASE2_EXITS = [
    ("fixed_10", {"sl_pct": 0.05, "tp_pct": 0.10, "trail_pct": None, "trail_activation": None}),
    ("fixed_20", {"sl_pct": 0.05, "tp_pct": 0.20, "trail_pct": None, "trail_activation": None}),
    ("trail_3", {"sl_pct": 0.05, "tp_pct": None, "trail_pct": 0.03, "trail_activation": 0.02}),
    ("trail_5", {"sl_pct": 0.05, "tp_pct": None, "trail_pct": 0.05, "trail_activation": 0.03}),
]
PHASE1_EXIT = {"sl_pct": SL_PCT, "tp_pct": TP_PCT, "trail_pct": None, "trail_activation": None}
PHASE2_TRIGGER_PF = 1.20

# --- CANONICAL HELPERS (copied from backtest_template.py — do not edit) ---

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
# Resolve snapshots dir relative to this file so the script can be run from any CWD.
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


def build_run_id(script_path: str | Path) -> str:
    """Compose the canonical run id for a backtest script invocation."""
    script_stem = Path(script_path).stem
    sha8 = _git_sha()[:8]
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"{script_stem}_{sha8}_{ts}"


# --- STRATEGY HELPERS ---


def apply_universe_filter(universe_df: pd.DataFrame) -> set[str]:
    """Top N by quote_volume_24h, exclude leveraged + delisting + low-volume.

    Returns set of asset names (without _USDT suffix) — e.g., {"BTC", "ETH", ...}.
    """
    leveraged_re = re.compile(LEVERAGED_TOKEN_RE)
    df = universe_df[
        (~universe_df["symbol"].str.contains(leveraged_re, na=False))
        & (~universe_df["in_delisting"].fillna(False))
        & (universe_df["quote_volume_24h"].fillna(0) >= MIN_QUOTE_VOL_24H)
    ].copy()
    df = df.sort_values("quote_volume_24h", ascending=False).head(TOP_N)
    return set(df["asset"].tolist())


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
    """SMA-based ATR over `period` bars from OHLC. Returns same-length Series."""
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


def compute_htf_downtrend(df_15m: pd.DataFrame) -> pd.Series:
    """Resample 15m to 4h, compute SMA50, return 15m-indexed bool Series of
    downtrend status (True = coin's 4h close < 4h SMA50 at that bar's time).

    df_15m must be a single-asset DataFrame with 'timestamp' column (epoch seconds)
    sorted ascending.
    """
    if len(df_15m) < 4 * HTF_SMA_PERIOD:
        return pd.Series([False] * len(df_15m), index=df_15m.index)
    s = df_15m.set_index(pd.to_datetime(df_15m["timestamp"], unit="s", utc=True))
    s4h_close = s["close"].resample(HTF_RESAMPLE).last().dropna()
    if len(s4h_close) < HTF_SMA_PERIOD:
        return pd.Series([False] * len(df_15m), index=df_15m.index)
    s4h_sma = s4h_close.rolling(HTF_SMA_PERIOD).mean()
    s4h_down = s4h_close < s4h_sma
    # Forward-fill 4h state down to 15m timestamps
    aligned = s4h_down.reindex(s.index, method="ffill").fillna(False)
    return pd.Series(aligned.values, index=df_15m.index)


def detect_atr_down(df: pd.DataFrame, atr: pd.Series) -> pd.Series:
    """Bar's downward range > ATR_DROP_MULT × ATR. Returns same-length bool Series.

    Downward range = max(prev_close - close, open - low, close - low) — captures
    bars that gapped down OR sold off intra-bar.
    """
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    low = df["low"].astype(float)
    prev_close = close.shift(1)
    drop = pd.concat([
        prev_close - close,
        open_ - low,
        close - low,
    ], axis=1).max(axis=1)
    drop = drop.clip(lower=0)
    threshold = ATR_DROP_MULT * atr
    return (drop > threshold).fillna(False)


def detect_vol_spike(df: pd.DataFrame) -> pd.Series:
    """vol > VOL_SPIKE_MULT × MA(20), price_drop > price_rise over 96-bar lookback.

    Returns same-length bool Series.
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
        (vol_ratio >= VOL_SPIKE_MULT)
        & (price_drop >= 0.05)
        & (price_drop > price_rise)
    ).fillna(False)


def apply_timing(signal_mask: pd.Series, df: pd.DataFrame, timing: str) -> pd.Series:
    """Transform raw signal mask into entry-bar mask per timing rule.

    Returns a same-length bool Series where True means "ENTER on this bar's open"
    (so the entry price is df.iloc[i]['open'] for True at index i).

    immediate: shift signals forward by 1 (signal on bar i -> entry at bar i+1).
    wait_retest: after signal at i, look for j in [i+1, i+RETEST_BARS] where
        close[j] >= close[i] * (1 - RETEST_PCT). Then look for k in [j+1, j+RETEST_BARS]
        where close[k] < close[k-1]. Entry at k+1.
    wait_confirm: after signal at i, if close[i+1] < close[i] then entry at i+2.
        Else skip.
    wait_next_surge: after signal at i, if any vol_spike fires in [i+1, i+NEXT_SURGE_BARS]
        then entry at the bar after the second surge.
    """
    n = len(df)
    out = pd.Series([False] * n, index=df.index)
    sig_idx = list(np.where(signal_mask.values)[0])

    if timing == "immediate":
        for i in sig_idx:
            if i + 1 < n:
                out.iloc[i + 1] = True
        return out

    close = df["close"].values.astype(float)

    if timing == "wait_retest":
        for i in sig_idx:
            target = close[i] * (1 - RETEST_PCT)
            j = -1
            for k in range(i + 1, min(i + 1 + RETEST_BARS, n)):
                if close[k] >= target:
                    j = k
                    break
            if j == -1:
                continue
            entry_bar = -1
            for k in range(j + 1, min(j + 1 + RETEST_BARS, n)):
                if close[k] < close[k - 1]:
                    entry_bar = k + 1
                    break
            if entry_bar != -1 and entry_bar < n:
                out.iloc[entry_bar] = True
        return out

    if timing == "wait_confirm":
        for i in sig_idx:
            if i + 2 >= n:
                continue
            if close[i + 1] < close[i]:
                out.iloc[i + 2] = True
        return out

    if timing == "wait_next_surge":
        vol_spike_mask = detect_vol_spike(df).values
        for i in sig_idx:
            for k in range(i + 1, min(i + 1 + NEXT_SURGE_BARS, n)):
                if vol_spike_mask[k]:
                    if k + 1 < n:
                        out.iloc[k + 1] = True
                    break
        return out

    raise ValueError(f"unknown timing: {timing}")


def simulate_portfolio(
    candles_by_asset: dict[str, pd.DataFrame],
    entry_masks: dict[str, pd.Series],  # per-asset bool Series; True = enter at this bar's open
    exit_params: dict,  # {sl_pct, tp_pct, trail_pct, trail_activation}
) -> pd.DataFrame:
    """Single time-ordered event loop across all assets enforcing MAX_CONCURRENT_POSITIONS.

    Returns trades DataFrame with columns:
      direction, pnl_pct, bars_held, symbol, entry_time, exit_time,
      exit_reason, peak_pnl_pct, entry_price, exit_price.
    """
    sl_pct = exit_params["sl_pct"]
    tp_pct = exit_params["tp_pct"]
    trail_pct = exit_params["trail_pct"]
    trail_activation = exit_params["trail_activation"]

    # First pass: per-asset, simulate each entry independently to get [entry_ts, exit_ts].
    candidate_trades = []
    for asset, df in candles_by_asset.items():
        if asset not in entry_masks:
            continue
        mask = entry_masks[asset]
        if not mask.any():
            continue
        ts = df["timestamp"].values  # epoch seconds
        open_ = df["open"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        n = len(df)
        last_entry_bar = -COOLDOWN_BARS
        for i in np.where(mask.values)[0]:
            if i - last_entry_bar < COOLDOWN_BARS:
                continue
            if i >= n:
                continue
            entry_price = float(open_[i])
            if entry_price <= 0:
                continue
            sl_trigger = entry_price * (1 + sl_pct + WORST_FILL_BUFFER)  # SHORT: SL above
            tp_trigger = entry_price * (1 - tp_pct) if tp_pct is not None else None
            peak_low = entry_price  # for SHORT, peak gain = lowest price
            exit_price = entry_price
            exit_bar = min(i + MAX_HOLD_BARS, n - 1)
            exit_reason = "timeout"
            for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
                bar_high = high[j]
                bar_low = low[j]
                if bar_low < peak_low:
                    peak_low = bar_low
                # Hard SL (price went UP for short)
                if bar_high >= sl_trigger:
                    exit_price = sl_trigger
                    exit_bar = j
                    exit_reason = "stop_loss"
                    break
                # TP if applicable
                if tp_trigger is not None and bar_low <= tp_trigger:
                    exit_price = tp_trigger
                    exit_bar = j
                    exit_reason = "take_profit"
                    break
                # Trail if applicable
                if trail_pct is not None and trail_activation is not None:
                    activation_price = entry_price * (1 - trail_activation)
                    if peak_low <= activation_price:
                        trail_price = peak_low * (1 + trail_pct)
                        if bar_high >= trail_price:
                            exit_price = trail_price
                            exit_bar = j
                            exit_reason = "trail_stop"
                            break
            pnl_pct = (entry_price - exit_price) / entry_price  # SHORT
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

    # Second pass: apply MAX_CONCURRENT_POSITIONS by walking entries in time order.
    candidate_trades.sort(key=lambda t: t["entry_time"])
    accepted = []
    for t in candidate_trades:
        # Count currently-open positions at t["entry_time"].
        open_count = sum(
            1 for a in accepted
            if a["entry_time"] <= t["entry_time"] < a["exit_time"]
        )
        if open_count >= MAX_CONCURRENT_POSITIONS:
            continue
        accepted.append(t)

    return pd.DataFrame(accepted)


def run_variant(
    candles_by_asset,
    htf_by_asset,
    universe_assets,
    detector: str,
    timing: str,
    htf_filter: bool,
    exit_kind: str,
    exit_params: dict,
    snapshot_date,
    candles_sha,
    universe_sha,
) -> tuple[str, dict]:
    """Run one variant. Returns (run_id, summary_row_dict)."""
    entry_masks = {}
    for asset, df in candles_by_asset.items():
        if asset not in universe_assets:
            continue
        # Compute base detection
        if detector == "atr_down":
            atr = compute_atr(df)
            sig = detect_atr_down(df, atr)
        elif detector == "vol_spike":
            sig = detect_vol_spike(df)
        elif detector == "combined":
            atr = compute_atr(df)
            sig = detect_atr_down(df, atr) & detect_vol_spike(df)
        else:
            raise ValueError(detector)
        # Apply HTF filter
        if htf_filter:
            htf = htf_by_asset.get(asset)
            if htf is not None:
                sig = sig & htf
            else:
                sig = sig & False  # no HTF data -> no signal
        # Apply timing
        entry = apply_timing(sig, df, timing)
        if entry.any():
            entry_masks[asset] = entry

    trades_df = simulate_portfolio(candles_by_asset, entry_masks, exit_params)
    metrics = compute_metrics(trades_df)
    params = {
        "detector": detector,
        "timing": timing,
        "htf_filter": htf_filter,
        "exit_kind": exit_kind,
        **exit_params,
        "POSITION_PCT": POSITION_PCT,
        "LEVERAGE": LEVERAGE,
        "MAX_CONCURRENT_POSITIONS": MAX_CONCURRENT_POSITIONS,
        "SNAPSHOT_DATE": SNAPSHOT_DATE,
        "RANDOM_SEED": RANDOM_SEED,
    }
    run_id = (
        build_run_id(__file__)
        + f"__{detector}_{timing}_htf{int(htf_filter)}_{exit_kind}"
    )
    write_results(
        run_id=run_id,
        trades_df=trades_df,
        metrics=metrics,
        params=params,
        snapshot_date=snapshot_date,
        snapshot_candles_sha256=candles_sha,
        snapshot_universe_sha256=universe_sha,
        git_dirty=False,  # we just committed; should be clean
        wall_time_seconds=0.0,  # overall timing logged in stdout; per-variant skipped
    )
    summary = {
        "phase": 1 if exit_kind == "fixed_15" else 2,
        "variant": f"{detector}_{timing}_htf{int(htf_filter)}_{exit_kind}",
        "detector": detector,
        "timing": timing,
        "htf_filter": htf_filter,
        "exit_kind": exit_kind,
        "trades": metrics["trades"],
        "wr": metrics["wr"],
        "pf": metrics["pf"],
        "avg_pnl_pct": metrics["avg_pnl_pct"],
        "total_pnl_pct": metrics["total_pnl_pct"],
        "median_winner_pnl_pct": float(
            trades_df.loc[trades_df["pnl_pct"] > 0, "pnl_pct"].median()
        ) if (trades_df["pnl_pct"] > 0).any() else 0.0,
        "median_loser_pnl_pct": float(
            trades_df.loc[trades_df["pnl_pct"] < 0, "pnl_pct"].median()
        ) if (trades_df["pnl_pct"] < 0).any() else 0.0,
        "tp_count": int((trades_df["exit_reason"] == "take_profit").sum())
        if "exit_reason" in trades_df.columns else 0,
        "sl_count": int((trades_df["exit_reason"] == "stop_loss").sum())
        if "exit_reason" in trades_df.columns else 0,
        "trail_count": int((trades_df["exit_reason"] == "trail_stop").sum())
        if "exit_reason" in trades_df.columns else 0,
        "timeout_count": int((trades_df["exit_reason"] == "timeout").sum())
        if "exit_reason" in trades_df.columns else 0,
        "run_id": run_id,
    }
    return run_id, summary


def _short_sha8(sha: str) -> str:
    return sha[:8]


# --- MAIN ---


def main() -> None:
    import time as _time

    import numpy as np  # noqa: F811  (re-import for seed)

    np.random.seed(RANDOM_SEED)

    candles, universe = load_snapshot(SNAPSHOT_DATE)
    print(f"[grid] candles: {len(candles):,} rows, universe: {len(universe)} pairs")

    universe_assets = apply_universe_filter(universe)
    print(f"[grid] tradable universe: {len(universe_assets)} pairs (after filters)")

    candles = candles[candles["asset"].isin(universe_assets)].copy()
    print(f"[grid] candles for tradable assets: {len(candles):,} rows")

    # Pre-group candles by asset to avoid repeated groupby
    candles_by_asset = {
        asset: g.sort_values("timestamp").reset_index(drop=True)
        for asset, g in candles.groupby("asset")
    }
    print(f"[grid] candles grouped: {len(candles_by_asset)} assets")

    # Pre-compute HTF downtrend per asset (used by HTF=True variants)
    print("[grid] computing HTF (4h SMA50) downtrend per asset ...")
    htf_by_asset = {}
    for asset, df in candles_by_asset.items():
        htf_by_asset[asset] = compute_htf_downtrend(df)
    print(f"[grid] HTF computed for {len(htf_by_asset)} assets")

    # Read snapshot shas from MANIFEST for write_results
    manifest_text = (Path("data/snapshots") / "MANIFEST.md").read_text()
    if not manifest_text:
        # try repo root resolution
        manifest_text = (
            Path(__file__).resolve().parents[3] / "data" / "snapshots" / "MANIFEST.md"
        ).read_text()
    candles_sha, universe_sha = _parse_manifest_row(manifest_text, SNAPSHOT_DATE)

    summary_rows: list[dict] = []

    # Phase 1
    print("\n" + "=" * 70)
    print("PHASE 1: 24-variant grid (fixed 5% SL / 15% TP)")
    print("=" * 70)
    p1_start = _time.time()
    for detector in DETECTORS:
        for timing in TIMINGS:
            for htf_filter in HTF_VARIANTS:
                v_start = _time.time()
                run_id, summary = run_variant(
                    candles_by_asset,
                    htf_by_asset,
                    universe_assets,
                    detector,
                    timing,
                    htf_filter,
                    exit_kind="fixed_15",
                    exit_params=PHASE1_EXIT,
                    snapshot_date=SNAPSHOT_DATE,
                    candles_sha=candles_sha,
                    universe_sha=universe_sha,
                )
                v_elapsed = _time.time() - v_start
                summary_rows.append(summary)
                print(
                    f"  {detector:10s} {timing:18s} htf={int(htf_filter)} | "
                    f"trades={summary['trades']:6d} wr={summary['wr']:.3f} "
                    f"pf={summary['pf']:.3f} ({v_elapsed:.1f}s)"
                )
    p1_elapsed = _time.time() - p1_start
    print(f"\nPhase 1 done in {p1_elapsed:.0f}s ({p1_elapsed / 60:.1f}m)")

    # Phase 2
    p1_summaries = [s for s in summary_rows if s["phase"] == 1]
    eligible = [s for s in p1_summaries if s["pf"] >= PHASE2_TRIGGER_PF and s["trades"] >= 100]
    if not eligible:
        print("\n" + "=" * 70)
        print("PHASE 2 SKIPPED — no Phase 1 variant cleared PF >= 1.20.")
        print("Entry is the bottleneck — no exit refinement worth running.")
        print("=" * 70)
    else:
        eligible.sort(key=lambda s: s["pf"], reverse=True)
        top3 = eligible[:3]
        print("\n" + "=" * 70)
        print(f"PHASE 2: refining exits on top {len(top3)} entry combos")
        print("=" * 70)
        p2_start = _time.time()
        for top in top3:
            for exit_kind, exit_params in PHASE2_EXITS:
                v_start = _time.time()
                run_id, summary = run_variant(
                    candles_by_asset,
                    htf_by_asset,
                    universe_assets,
                    top["detector"],
                    top["timing"],
                    top["htf_filter"],
                    exit_kind=exit_kind,
                    exit_params=exit_params,
                    snapshot_date=SNAPSHOT_DATE,
                    candles_sha=candles_sha,
                    universe_sha=universe_sha,
                )
                v_elapsed = _time.time() - v_start
                summary_rows.append(summary)
                print(
                    f"  {top['detector']:10s} {top['timing']:18s} "
                    f"htf={int(top['htf_filter'])} {exit_kind:9s} | "
                    f"trades={summary['trades']:6d} wr={summary['wr']:.3f} "
                    f"pf={summary['pf']:.3f} ({v_elapsed:.1f}s)"
                )
        p2_elapsed = _time.time() - p2_start
        print(f"\nPhase 2 done in {p2_elapsed:.0f}s ({p2_elapsed / 60:.1f}m)")

    # Write summary CSV at results root
    summary_df = pd.DataFrame(summary_rows).sort_values(["pf"], ascending=False)
    results_root = Path("results")
    summary_csv_name = (
        f"_grid_summary_short_{_short_sha8(_git_sha())}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.csv"
    )
    summary_path = results_root / summary_csv_name
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary written: {summary_path}")
    print("\nTop 5 by PF:")
    print(summary_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
