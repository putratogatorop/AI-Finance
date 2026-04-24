"""Leak-fixed fork of backtest_bigmover_longshort_3y.py.

Two look-ahead biases present in the original are removed here so the resulting
PF numbers reflect what the strategy could actually have executed live.

Fixes vs the original:

  1) `multi_bar_confirm` used `close[i+1]` (via `.shift(-1)`) to decide whether
     a signal at bar `i` should fire, while the simulator then entered at
     `open[i]`. That is a 2-bar look-ahead. Fix: shift the final mask forward
     by 1 bar so `mask[j] = baseline[j-1] AND (close[j] < close[j-1])` for
     shorts (symmetric for longs). Under this definition the mask at index
     `j` only reads bars `<= j`, so entry at `open[j+1]` is clean.

  2) All signals entered at `open[i]` despite computing masks from `close[i]`.
     That is a 1-bar look-ahead affecting baseline, price_accel_atr, and
     multi_bar_confirm. Fix: entry moves to `open[i+1]` with a bounds check,
     and the exit scan window shifts accordingly.

No other logic changes. Same snapshot, same universe, same sizing, same
fees, same exit params. Output directory and summary filename carry a
`_leakfix` tag so results do not collide with the buggy run.

Run from services/python/:
    python scripts/backtest_bigmover_longshort_3y_leakfix.py
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

_NON_PARAM_NAMES = frozenset({"UTC"})

# =============================================================================
# CONSTANTS (identical to the original 3y script)
# =============================================================================
SNAPSHOT_DATE = "2026-04-24"
RANDOM_SEED = 42

TOP_N = 100
MIN_QUOTE_VOL_24H = 500_000
LEVERAGED_TOKEN_RE = r"[35][LS]_USDT$"

VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
COOLDOWN_BARS = 96
ACCEL_MULT_OF_ATR = 1.0

MAX_CONCURRENT_POSITIONS = 5
WORST_FILL_BUFFER = 0.005
MAX_HOLD_BARS = 192

START_EQUITY_USD = 100.0
LEVERAGE = 5
POSITION_PCT = 0.10
ROUND_TRIP_FEE = 0.0012

EXIT_PARAMS = {
    "sl_pct": 0.05, "tp_pct": None,
    "trail_pct": 0.03, "trail_activation": 0.02,
}

BTC_ASSET = "BTCUSDT"
REGIME_SMA_FAST = 50
REGIME_SMA_SLOW = 200

MC_ITER = 1000
MC_SEED = 42

LISTING_WARMUP_DAYS = 3

# OOT hold-out window per CLAUDE.md gate 8: last 3 months of the snapshot.
# 2026-01-23 00:00:00 UTC == 1769126400
OOT_START_EPOCH_S = 1769126400


# =============================================================================
# CANONICAL HELPERS (unchanged)
# =============================================================================

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
    date: str, snapshots_dir: Path | str | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    snapshots_dir = Path(snapshots_dir) if snapshots_dir is not None else _SNAPSHOTS_DIR
    manifest = snapshots_dir / "MANIFEST.md"
    if not manifest.exists():
        raise FileNotFoundError(f"missing MANIFEST.md at {manifest}")
    candles_path = snapshots_dir / f"candles_15m_{date}.parquet"
    universe_path = snapshots_dir / f"universe_{date}.parquet"
    if not candles_path.exists():
        raise FileNotFoundError(f"snapshot candles file missing for date {date}")
    if not universe_path.exists():
        raise FileNotFoundError(f"snapshot universe file missing for date {date}")
    expected_candles_sha, expected_universe_sha = _parse_manifest_row(
        manifest.read_text(), date
    )
    actual_candles_sha = _sha256_file(candles_path)
    actual_universe_sha = _sha256_file(universe_path)
    if actual_candles_sha != expected_candles_sha:
        raise ValueError(f"sha256 mismatch for {candles_path}")
    if actual_universe_sha != expected_universe_sha:
        raise ValueError(f"sha256 mismatch for {universe_path}")
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    )


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
    extra_files: dict[str, pd.DataFrame] | None = None,
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
    if extra_files:
        for name, df in extra_files.items():
            df.to_csv(run_dir / name, index=False)
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
    def _stats(df):
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


def monte_carlo_pf(trades_df: pd.DataFrame, n_iter: int = MC_ITER, seed: int = MC_SEED) -> dict:
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


def apply_universe_filter(universe_df: pd.DataFrame) -> set[str]:
    leveraged_re = re.compile(LEVERAGED_TOKEN_RE)
    df = universe_df[
        (~universe_df["symbol"].str.contains(leveraged_re, na=False))
        & (~universe_df["in_delisting"].fillna(False))
        & (universe_df["quote_volume_24h"].fillna(0) >= MIN_QUOTE_VOL_24H)
    ].copy()
    df = df.sort_values("quote_volume_24h", ascending=False).head(TOP_N)
    return set(df["symbol"].str.replace("_", "", regex=False).tolist())


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD) -> pd.Series:
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


# =============================================================================
# SIGNAL DETECTORS (baseline unchanged; multi_bar_confirm leak-fixed)
# =============================================================================


def baseline_entry_mask_short(df: pd.DataFrame) -> pd.Series:
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


def baseline_entry_mask_long(df: pd.DataFrame) -> pd.Series:
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
        & (price_rise >= PRICE_MOVE_THRESH)
        & (price_rise > price_drop)
    ).fillna(False)


def detect_signal(df: pd.DataFrame, signal: str, direction: str, atr: pd.Series) -> pd.Series:
    """Dispatch to baseline + tweak. Leak-fixed for multi_bar_confirm."""
    if direction == "short":
        base = baseline_entry_mask_short(df)
    elif direction == "long":
        base = baseline_entry_mask_long(df)
    else:
        raise ValueError(f"unknown direction: {direction}")

    if signal == "price_accel_atr":
        prev_close = df["close"].shift(1)
        prev_prev = df["close"].shift(2)
        accel = (df["close"] - prev_close) - (prev_close - prev_prev)
        return (base & (accel.abs() >= atr * ACCEL_MULT_OF_ATR)).fillna(False)

    if signal == "multi_bar_confirm":
        # LEAK FIX: original used `close.shift(-1)` to look one bar into the
        # future and decide whether a baseline at bar `i` should fire. The
        # simulator then entered at `open[i]`, so that was a 2-bar leak.
        # Clean formulation: baseline fired on bar `j-1`, confirmed by bar `j`
        # moving in the trade direction. `mask[j] = True` means entry at
        # `open[j+1]` (after the simulator shift below) reads only bars <= j.
        prev_close = df["close"].shift(1)
        prev_base = base.shift(1).fillna(False)
        if direction == "short":
            confirm = df["close"] < prev_close
        else:
            confirm = df["close"] > prev_close
        return (prev_base & confirm).fillna(False)

    raise ValueError(f"unknown signal: {signal}")


# =============================================================================
# BTC REGIME CLASSIFIER (unchanged)
# =============================================================================


def classify_btc_regime(btc_df_15m: pd.DataFrame) -> pd.DataFrame:
    df = btc_df_15m.copy()
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        ts = df["timestamp"]
    else:
        ts = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.assign(ts=ts).set_index("ts")
    daily = df.resample("1D").agg({"close": "last"}).dropna().reset_index()
    daily = daily.rename(columns={"ts": "date"})
    daily["sma_50"] = daily["close"].rolling(REGIME_SMA_FAST).mean()
    daily["sma_200"] = daily["close"].rolling(REGIME_SMA_SLOW).mean()
    regime = pd.Series("sideways", index=daily.index, dtype=object)
    bull = (daily["close"] > daily["sma_50"]) & (daily["sma_50"] > daily["sma_200"])
    bear = (daily["close"] < daily["sma_50"]) & (daily["sma_50"] < daily["sma_200"])
    regime[bull.fillna(False).astype(bool)] = "bull"
    regime[bear.fillna(False).astype(bool)] = "bear"
    daily["regime"] = regime
    return daily[["date", "close", "sma_50", "sma_200", "regime"]]


def build_regime_map(regime_df: pd.DataFrame) -> dict[int, str]:
    out: dict[int, str] = {}
    for _, row in regime_df.iterrows():
        day_ts = int(row["date"].value // 10**9)
        day_id = day_ts // 86400
        out[day_id] = str(row["regime"])
    return out


# =============================================================================
# BIDIRECTIONAL PORTFOLIO SIMULATOR — entry shifted to open[i+1]
# =============================================================================


def simulate_portfolio_bidirectional(
    candles_by_asset: dict[str, pd.DataFrame],
    entry_masks_by_dir: dict[str, dict[str, pd.Series]],
    exit_params: dict,
    cooldown_bars: int = COOLDOWN_BARS,
    regime_map: dict[int, str] | None = None,
    regime_gate: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Leak-fixed simulator. Signal at bar `i` produces an entry at `open[i+1]`.

    Exit scan starts one bar after the entry bar, so the entry bar itself
    cannot satisfy SL / TP / trail (preserving the original no-same-bar-exit
    convention).
    """
    sl_pct = exit_params["sl_pct"]
    tp_pct = exit_params.get("tp_pct")
    trail_pct = exit_params.get("trail_pct")
    trail_activation = exit_params.get("trail_activation")

    if regime_gate and regime_map is None:
        raise ValueError("regime_gate set but regime_map not provided")

    candidate_trades = []
    for direction, masks in entry_masks_by_dir.items():
        for asset, mask in masks.items():
            if asset not in candles_by_asset:
                continue
            df = candles_by_asset[asset]
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
                # LEAK FIX: enter on the bar AFTER the signal bar.
                entry_bar_idx = i + 1
                if entry_bar_idx >= n:
                    continue
                entry_price = float(open_[entry_bar_idx])
                if entry_price <= 0:
                    continue

                entry_ts = int(ts[entry_bar_idx])
                entry_regime = "sideways"
                if regime_map is not None:
                    entry_regime = regime_map.get(entry_ts // 86400, "sideways")
                if regime_gate:
                    wanted = regime_gate.get(direction)
                    if wanted is not None and entry_regime != wanted:
                        continue

                if direction == "short":
                    sl_trigger = entry_price * (1 + sl_pct + WORST_FILL_BUFFER)
                    tp_trigger = entry_price * (1 - tp_pct) if tp_pct is not None else None
                    peak_low = entry_price
                    exit_price = entry_price
                    exit_bar = min(entry_bar_idx + MAX_HOLD_BARS, n - 1)
                    exit_reason = "timeout"
                    for j in range(entry_bar_idx + 1, min(entry_bar_idx + 1 + MAX_HOLD_BARS, n)):
                        bar_high = high[j]
                        bar_low = low[j]
                        if bar_low < peak_low:
                            peak_low = bar_low
                        if bar_high >= sl_trigger:
                            exit_price, exit_bar, exit_reason = (
                                sl_trigger, j, "stop_loss"
                            )
                            break
                        if tp_trigger is not None and bar_low <= tp_trigger:
                            exit_price, exit_bar, exit_reason = (
                                tp_trigger, j, "take_profit"
                            )
                            break
                        if trail_pct is not None and trail_activation is not None:
                            act_price = entry_price * (1 - trail_activation)
                            if peak_low <= act_price:
                                trail_price = peak_low * (1 + trail_pct)
                                if bar_high >= trail_price:
                                    exit_price, exit_bar, exit_reason = (
                                        trail_price, j, "trail_stop"
                                    )
                                    break
                    pnl_pct = (entry_price - exit_price) / entry_price
                    peak_pnl_pct = (entry_price - peak_low) / entry_price
                else:
                    sl_trigger = entry_price * (1 - sl_pct - WORST_FILL_BUFFER)
                    tp_trigger = entry_price * (1 + tp_pct) if tp_pct is not None else None
                    peak_high = entry_price
                    exit_price = entry_price
                    exit_bar = min(entry_bar_idx + MAX_HOLD_BARS, n - 1)
                    exit_reason = "timeout"
                    for j in range(entry_bar_idx + 1, min(entry_bar_idx + 1 + MAX_HOLD_BARS, n)):
                        bar_high = high[j]
                        bar_low = low[j]
                        if bar_high > peak_high:
                            peak_high = bar_high
                        if bar_low <= sl_trigger:
                            exit_price, exit_bar, exit_reason = (
                                sl_trigger, j, "stop_loss"
                            )
                            break
                        if tp_trigger is not None and bar_high >= tp_trigger:
                            exit_price, exit_bar, exit_reason = (
                                tp_trigger, j, "take_profit"
                            )
                            break
                        if trail_pct is not None and trail_activation is not None:
                            act_price = entry_price * (1 + trail_activation)
                            if peak_high >= act_price:
                                trail_price = peak_high * (1 - trail_pct)
                                if bar_low <= trail_price:
                                    exit_price, exit_bar, exit_reason = (
                                        trail_price, j, "trail_stop"
                                    )
                                    break
                    pnl_pct = (exit_price - entry_price) / entry_price
                    peak_pnl_pct = (peak_high - entry_price) / entry_price

                candidate_trades.append({
                    "direction": direction,
                    "pnl_pct": pnl_pct,
                    "bars_held": exit_bar - entry_bar_idx,
                    "symbol": asset,
                    "entry_time": entry_ts,
                    "exit_time": int(ts[exit_bar]),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "peak_pnl_pct": peak_pnl_pct,
                    "exit_reason": exit_reason,
                    "entry_regime": entry_regime,
                })
                last_entry_bar = i

    if not candidate_trades:
        return pd.DataFrame(columns=[
            "direction", "pnl_pct", "bars_held", "symbol", "entry_time",
            "exit_time", "entry_price", "exit_price", "peak_pnl_pct",
            "exit_reason", "entry_regime",
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
# SIZING + METRICS (unchanged)
# =============================================================================


def apply_fixed10_sizing(trades_df: pd.DataFrame) -> pd.DataFrame:
    df = trades_df.copy()
    if len(df) == 0:
        df["notional_usd"] = pd.Series(dtype=float)
        df["pnl_pct_net"] = pd.Series(dtype=float)
        df["pnl_usd"] = pd.Series(dtype=float)
        return df
    df["notional_usd"] = START_EQUITY_USD * POSITION_PCT * LEVERAGE
    df["pnl_pct_net"] = df["pnl_pct"] - ROUND_TRIP_FEE
    df["pnl_usd"] = df["notional_usd"] * df["pnl_pct_net"]
    return df


def compute_usd_metrics(trades_df: pd.DataFrame) -> dict:
    if len(trades_df) == 0 or "pnl_usd" not in trades_df.columns:
        return {
            "final_equity_usd": START_EQUITY_USD,
            "total_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
        }
    df = trades_df.sort_values("entry_time").reset_index(drop=True)
    equity = START_EQUITY_USD + df["pnl_usd"].cumsum()
    running_max = equity.cummax()
    dd = (equity - running_max) / running_max.replace(0, np.nan)
    max_dd = float(dd.min()) if dd.notna().any() else 0.0
    total_pnl_usd = float(df["pnl_usd"].sum())
    ret = df["pnl_pct_net"]
    sharpe = 0.0
    if ret.std(ddof=0) > 0:
        sharpe = float(ret.mean() / ret.std(ddof=0) * np.sqrt(len(ret)))
    return {
        "final_equity_usd": float(START_EQUITY_USD + total_pnl_usd),
        "total_return_pct": float(total_pnl_usd / START_EQUITY_USD * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "sharpe": sharpe,
    }


# =============================================================================
# OOT HOLD-OUT METRICS (new — CLAUDE.md gate 8)
# =============================================================================


def compute_oot_metrics(trades_df: pd.DataFrame) -> dict:
    """PF on the strict OOT hold-out window (last 3 months: 2026-01-23 onward)."""
    if len(trades_df) == 0:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "total_pnl_pct": 0.0}
    mask = trades_df["entry_time"] >= OOT_START_EPOCH_S
    sub = trades_df.loc[mask]
    n = len(sub)
    if n == 0:
        return {"trades": 0, "wr": 0.0, "pf": 0.0, "total_pnl_pct": 0.0}
    wins = sub.loc[sub["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = -sub.loc[sub["pnl_pct"] < 0, "pnl_pct"].sum()
    pf = float("inf") if losses == 0 else float(wins / losses)
    return {
        "trades": int(n),
        "wr": float((sub["pnl_pct"] > 0).mean()),
        "pf": pf,
        "total_pnl_pct": float(sub["pnl_pct"].sum()),
    }


# =============================================================================
# VARIANTS (same 6 as the original)
# =============================================================================


VARIANTS = [
    {"label": "price_accel_atr__short__all", "signal": "price_accel_atr",
     "directions": ["short"], "regime_gate": None},
    {"label": "price_accel_atr__long__all", "signal": "price_accel_atr",
     "directions": ["long"], "regime_gate": None},
    {"label": "multi_bar_confirm__short__all", "signal": "multi_bar_confirm",
     "directions": ["short"], "regime_gate": None},
    {"label": "multi_bar_confirm__long__all", "signal": "multi_bar_confirm",
     "directions": ["long"], "regime_gate": None},
    {"label": "price_accel_atr__combo__regime", "signal": "price_accel_atr",
     "directions": ["short", "long"],
     "regime_gate": {"long": "bull", "short": "bear"}},
    {"label": "multi_bar_confirm__combo__regime", "signal": "multi_bar_confirm",
     "directions": ["short", "long"],
     "regime_gate": {"long": "bull", "short": "bear"}},
]


def compute_listing_gates(candles_by_asset: dict) -> dict[str, int]:
    gates: dict[str, int] = {}
    warmup_s = LISTING_WARMUP_DAYS * 86400
    for asset, df in candles_by_asset.items():
        if len(df) == 0:
            continue
        first_ts = int(df["timestamp"].iloc[0])
        gates[asset] = first_ts + warmup_s
    return gates


def run_variant(
    variant: dict,
    candles_by_asset: dict,
    regime_map: dict[int, str],
    signal_mask_cache: dict,
    listing_gates: dict[str, int],
) -> tuple[pd.DataFrame, dict]:
    signal = variant["signal"]
    entry_masks_by_dir: dict[str, dict[str, pd.Series]] = {}
    for direction in variant["directions"]:
        cache_key = (signal, direction)
        if cache_key not in signal_mask_cache:
            masks: dict[str, pd.Series] = {}
            for asset, df in candles_by_asset.items():
                atr = compute_atr(df)
                mask = detect_signal(df, signal, direction, atr)
                gate_ts = listing_gates.get(asset)
                if gate_ts is not None:
                    after_gate = df["timestamp"] >= gate_ts
                    mask = mask & after_gate
                if mask.any():
                    masks[asset] = mask
            signal_mask_cache[cache_key] = masks
        entry_masks_by_dir[direction] = signal_mask_cache[cache_key]

    trades = simulate_portfolio_bidirectional(
        candles_by_asset,
        entry_masks_by_dir,
        EXIT_PARAMS,
        cooldown_bars=COOLDOWN_BARS,
        regime_map=regime_map,
        regime_gate=variant["regime_gate"],
    )
    trades = apply_fixed10_sizing(trades)
    return trades, {}


# =============================================================================
# SELF-CHECK — verify multi_bar_confirm mask is lookahead-free
# =============================================================================


def _self_check_multi_bar_confirm_no_lookahead() -> None:
    """Mutating future close values must not change past mask values."""
    rng = np.random.default_rng(12345)
    n_bars = PRICE_LOOKBACK + VOL_MA_PERIOD + 100
    base_close = 100 + rng.normal(0, 1, n_bars).cumsum()
    df = pd.DataFrame({
        "timestamp": np.arange(n_bars, dtype=np.int64) * 900,
        "open": base_close,
        "high": base_close + np.abs(rng.normal(0, 0.5, n_bars)),
        "low": base_close - np.abs(rng.normal(0, 0.5, n_bars)),
        "close": base_close,
        "volume": rng.uniform(1, 10, n_bars),
    })
    atr_a = compute_atr(df)
    mask_a_short = detect_signal(df, "multi_bar_confirm", "short", atr_a)
    mask_a_long = detect_signal(df, "multi_bar_confirm", "long", atr_a)

    cut = n_bars - 30
    df_mut = df.copy()
    # Replace the tail with something wildly different
    df_mut.loc[cut:, "close"] = df_mut.loc[cut:, "close"] * 3.0
    df_mut.loc[cut:, "open"] = df_mut.loc[cut:, "open"] * 3.0
    df_mut.loc[cut:, "high"] = df_mut.loc[cut:, "high"] * 3.0
    df_mut.loc[cut:, "low"] = df_mut.loc[cut:, "low"] * 3.0
    df_mut.loc[cut:, "volume"] = df_mut.loc[cut:, "volume"] * 5.0
    atr_b = compute_atr(df_mut)
    mask_b_short = detect_signal(df_mut, "multi_bar_confirm", "short", atr_b)
    mask_b_long = detect_signal(df_mut, "multi_bar_confirm", "long", atr_b)

    # mask values at indices strictly before `cut` must not depend on mutated bars.
    # baseline uses rolling windows (period=20 for volume, 96 for price), so it
    # can only "see" back `max(period)` bars. As long as our check index is
    # `cut - 1` (one bar before the mutation), no past rolling aggregate
    # references a mutated bar. For safety, check all indices < cut.
    idx = slice(0, cut)
    if not (mask_a_short.iloc[idx].values == mask_b_short.iloc[idx].values).all():
        first_diff = np.where(
            mask_a_short.iloc[idx].values != mask_b_short.iloc[idx].values
        )[0]
        raise AssertionError(
            f"LEAK REGRESSION (short): multi_bar_confirm mask at past bars changed "
            f"after mutating future bars. First diff at idx={int(first_diff[0])}"
        )
    if not (mask_a_long.iloc[idx].values == mask_b_long.iloc[idx].values).all():
        first_diff = np.where(
            mask_a_long.iloc[idx].values != mask_b_long.iloc[idx].values
        )[0]
        raise AssertionError(
            f"LEAK REGRESSION (long): multi_bar_confirm mask at past bars changed "
            f"after mutating future bars. First diff at idx={int(first_diff[0])}"
        )
    print("[leakfix] self-check PASS: multi_bar_confirm mask uses only past bars")


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    _self_check_multi_bar_confirm_no_lookahead()

    overall_start = _time.monotonic()
    git_dirty_at_start = _git_dirty()
    if git_dirty_at_start:
        print(
            "WARNING: git working tree is dirty — not exactly reproducible.",
            file=sys.stderr,
        )

    print(f"[leakfix] loading snapshot {SNAPSHOT_DATE} ...")
    candles, universe = load_snapshot(SNAPSHOT_DATE)
    if pd.api.types.is_datetime64_any_dtype(candles["timestamp"]):
        candles = candles.copy()
        candles["timestamp"] = candles["timestamp"].astype("int64") // 10**9
    print(f"[leakfix] candles: {len(candles):,} rows, universe: {len(universe)} pairs")

    universe_assets = apply_universe_filter(universe)
    print(f"[leakfix] tradable universe: {len(universe_assets)} pairs")
    candles_tradable = candles[candles["asset"].isin(universe_assets)].copy()
    candles_by_asset = {
        asset: g.sort_values("timestamp").reset_index(drop=True)
        for asset, g in candles_tradable.groupby("asset")
    }

    if BTC_ASSET not in candles_by_asset:
        btc_full = candles[candles["asset"] == BTC_ASSET].copy()
        if len(btc_full) == 0:
            raise RuntimeError(f"{BTC_ASSET} not found in snapshot")
        btc_full = btc_full.sort_values("timestamp").reset_index(drop=True)
    else:
        btc_full = candles_by_asset[BTC_ASSET]

    btc_for_regime = btc_full.copy()
    btc_for_regime["timestamp"] = pd.to_datetime(
        btc_for_regime["timestamp"], unit="s", utc=True
    )
    print("[leakfix] classifying BTC daily regime ...")
    regime_df = classify_btc_regime(btc_for_regime)
    regime_map = build_regime_map(regime_df)

    manifest_path = _SNAPSHOTS_DIR / "MANIFEST.md"
    candles_sha, universe_sha = _parse_manifest_row(
        manifest_path.read_text(), SNAPSHOT_DATE
    )

    base_run_id_prefix = (
        f"backtest_bigmover_longshort_3y_leakfix_{_git_sha()[:8]}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )

    results_root = Path("results")
    results_root.mkdir(exist_ok=True)
    signal_mask_cache: dict = {}
    summary_rows: list[dict] = []

    listing_gates = compute_listing_gates(candles_by_asset)
    late_listed = sum(1 for ts in listing_gates.values() if ts > 1682035200)
    print(f"[leakfix] listing gates: {len(listing_gates)} assets; "
          f"{late_listed} listed after 2023-04-21")

    for i, variant in enumerate(VARIANTS, 1):
        v_start = _time.monotonic()
        label = variant["label"]
        print(f"\n[leakfix] variant {i}/{len(VARIANTS)}: {label} ...")

        trades, _ = run_variant(
            variant, candles_by_asset, regime_map, signal_mask_cache, listing_gates
        )

        metrics_pct = compute_metrics(trades)
        mc = monte_carlo_pf(trades)
        metrics_usd = compute_usd_metrics(trades)
        oot = compute_oot_metrics(trades)

        headline = {
            "variant": label,
            "signal": variant["signal"],
            "directions": variant["directions"],
            "regime_gate": variant["regime_gate"],
            "trades": metrics_pct["trades"],
            "wr": metrics_pct["wr"],
            "pf": metrics_pct["pf"],
            "pf_p5": mc["p5"], "pf_p50": mc["p50"], "pf_p95": mc["p95"],
            "avg_pnl_pct": metrics_pct["avg_pnl_pct"],
            "total_pnl_pct": metrics_pct["total_pnl_pct"],
            "oot_trades": oot["trades"],
            "oot_wr": oot["wr"],
            "oot_pf": oot["pf"],
            "oot_total_pnl_pct": oot["total_pnl_pct"],
            **metrics_usd,
        }

        metrics = {**metrics_pct, **metrics_usd, **mc, "oot": oot}
        params = {
            "variant_label": label,
            "signal": variant["signal"],
            "directions": variant["directions"],
            "regime_gate": variant["regime_gate"],
            "SNAPSHOT_DATE": SNAPSHOT_DATE,
            "VOL_SPIKE": VOL_SPIKE, "PRICE_MOVE_THRESH": PRICE_MOVE_THRESH,
            "PRICE_LOOKBACK": PRICE_LOOKBACK, "ACCEL_MULT_OF_ATR": ACCEL_MULT_OF_ATR,
            "COOLDOWN_BARS": COOLDOWN_BARS, "MAX_HOLD_BARS": MAX_HOLD_BARS,
            "MAX_CONCURRENT_POSITIONS": MAX_CONCURRENT_POSITIONS,
            "POSITION_PCT": POSITION_PCT, "LEVERAGE": LEVERAGE,
            "START_EQUITY_USD": START_EQUITY_USD, "ROUND_TRIP_FEE": ROUND_TRIP_FEE,
            "EXIT_PARAMS": EXIT_PARAMS,
            "REGIME_SMA_FAST": REGIME_SMA_FAST, "REGIME_SMA_SLOW": REGIME_SMA_SLOW,
            "LEAK_FIXED": True,
            "OOT_START_EPOCH_S": OOT_START_EPOCH_S,
        }

        run_id = f"{base_run_id_prefix}__{label}"
        write_results(
            run_id=run_id, trades_df=trades, metrics=metrics, params=params,
            snapshot_date=SNAPSHOT_DATE,
            snapshot_candles_sha256=candles_sha,
            snapshot_universe_sha256=universe_sha,
            git_dirty=git_dirty_at_start,
            wall_time_seconds=_time.monotonic() - v_start,
        )
        headline["run_id"] = run_id
        summary_rows.append(headline)

        v_elapsed = _time.monotonic() - v_start
        print(
            f"  trades={metrics_pct['trades']:5d}  wr={metrics_pct['wr']:.2f}  "
            f"pf={metrics_pct['pf']:.2f}  p5={mc['p5']:.2f}  "
            f"ret={metrics_usd['total_return_pct']:+.0f}%  "
            f"dd={metrics_usd['max_drawdown_pct']:.1f}%  "
            f"|| OOT: n={oot['trades']} pf={oot['pf']:.2f}  [{v_elapsed:.1f}s]"
        )

    summary_df = pd.DataFrame(summary_rows)
    sha8 = _git_sha()[:8]
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = results_root / f"_longshort_3y_leakfix_summary_{sha8}_{ts_str}.csv"
    cols_front = [
        "variant", "signal", "trades", "wr", "pf", "pf_p5",
        "total_return_pct", "max_drawdown_pct", "sharpe",
        "oot_trades", "oot_wr", "oot_pf", "oot_total_pnl_pct",
        "run_id",
    ]
    cols = [c for c in cols_front if c in summary_df.columns] + [
        c for c in summary_df.columns if c not in cols_front
    ]
    summary_df[cols].to_csv(summary_path, index=False)

    total_elapsed = _time.monotonic() - overall_start
    print(f"\n[leakfix] done in {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"[leakfix] summary: {summary_path}")
    print("\nCross-variant headline (trades / pf / oot_pf):")
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
