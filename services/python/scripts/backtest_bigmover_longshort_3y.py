"""3-year long+short regime backtest for price_accel_atr & multi_bar_confirm.

6 variants across the full snapshot (2023-04-01 -> 2026-04-24):
  1. price_accel_atr__short__all     (shorts, all regimes)
  2. price_accel_atr__long__all      (longs, all regimes)
  3. multi_bar_confirm__short__all
  4. multi_bar_confirm__long__all
  5. price_accel_atr__combo__regime  (long in bull + short in bear, skip sideways)
  6. multi_bar_confirm__combo__regime

BTC regime (3-class, daily):
  bull:     close > SMA_50 AND SMA_50 > SMA_200
  bear:     close < SMA_50 AND SMA_50 < SMA_200
  sideways: otherwise (incl. first 200 days that lack SMA_200)

Sizing + exit held constant across variants: fixed_10 (10% equity x 5x lev) +
trail_3 (5% SL + trail 3%, activate +2%, 48h timeout).

Run from services/python/:
    python scripts/backtest_bigmover_longshort_3y.py

Outputs:
  - results/<run_id>__<variant_label>/ (trades.csv, metrics.json, params.json,
    run_metadata.json, monthly_regime_summary.csv, regime_summary.csv)
  - results/_longshort_3y_summary_<sha8>_<ts>.csv (cross-variant headline)
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

_NON_PARAM_NAMES = frozenset({"UTC"})

# =============================================================================
# CONSTANTS
# =============================================================================
SNAPSHOT_DATE = "2026-04-24"
RANDOM_SEED = 42

# Universe
TOP_N = 100
MIN_QUOTE_VOL_24H = 500_000
LEVERAGED_TOKEN_RE = r"[35][LS]_USDT$"

# Portfolio-level cooldown (signal-detector constants come from src.ml.bigmover_signals)
COOLDOWN_BARS = 96

# Portfolio
MAX_CONCURRENT_POSITIONS = 5
WORST_FILL_BUFFER = 0.005
MAX_HOLD_BARS = 192

# Sizing + exit (fixed across variants)
START_EQUITY_USD = 100.0
LEVERAGE = 5
POSITION_PCT = 0.10
ROUND_TRIP_FEE = 0.0012  # 0.06% x 2

EXIT_PARAMS = {
    "sl_pct": 0.05, "tp_pct": None,
    "trail_pct": 0.03, "trail_activation": 0.02,
}

# Regime (daily)
BTC_ASSET = "BTCUSDT"
REGIME_SMA_FAST = 50
REGIME_SMA_SLOW = 200

# Monte Carlo
MC_ITER = 1000
MC_SEED = 42

# Listing-aware universe (partial survivorship-bias mitigation)
# Each coin's effective listing date = its first snapshot candle + this warmup buffer.
# Rolling windows (PRICE_LOOKBACK=96 bars = 24h, plus VOL_MA_PERIOD=20) need ~3 days
# of data to produce stable signals; we require at least this before any entry.
LISTING_WARMUP_DAYS = 3


# =============================================================================
# CANONICAL HELPERS (copied from backtest_bigmover_tweaks_v1.py)
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


# Signal detectors (compute_atr, baseline_entry_mask_short/long, detect_signal)
# are imported from src.ml.bigmover_signals — single source of truth, leak-free
# multi_bar_confirm, covered by tests/test_bigmover_signals.py. See commit
# 730564bb for the extraction.


# =============================================================================
# BTC REGIME CLASSIFIER (3-class, daily)
# =============================================================================


def classify_btc_regime(btc_df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample BTC 15m candles to daily, compute SMA_50 & SMA_200, label each day.

    Returns DataFrame with columns: date (datetime64[ns, UTC] at 00:00), close,
    sma_50, sma_200, regime ("bull", "bear", "sideways").
    """
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


def regime_lookup_for_ts(regime_df: pd.DataFrame, ts_epoch_s: int) -> str:
    """Map an epoch-seconds timestamp to its daily regime."""
    dt = datetime.fromtimestamp(ts_epoch_s, tz=UTC)
    day = pd.Timestamp(dt.year, dt.month, dt.day, tz="UTC")
    row = regime_df.loc[regime_df["date"] == day]
    if len(row) == 0:
        return "sideways"
    return str(row["regime"].iloc[0])


def build_regime_map(regime_df: pd.DataFrame) -> dict[int, str]:
    """Lookup table: epoch-day (ts // 86400) -> regime string."""
    out: dict[int, str] = {}
    for _, row in regime_df.iterrows():
        day_ts = int(row["date"].value // 10**9)
        day_id = day_ts // 86400
        out[day_id] = str(row["regime"])
    return out


# =============================================================================
# BIDIRECTIONAL PORTFOLIO SIMULATOR
# =============================================================================


def simulate_portfolio_bidirectional(
    candles_by_asset: dict[str, pd.DataFrame],
    entry_masks_by_dir: dict[str, dict[str, pd.Series]],
    exit_params: dict,
    cooldown_bars: int = COOLDOWN_BARS,
    regime_map: dict[int, str] | None = None,
    regime_gate: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Short+long exit engine with MAX_CONCURRENT_POSITIONS=5.

    entry_masks_by_dir: {"short": {asset: mask}, "long": {asset: mask}}
    regime_gate: optional dict like {"long": "bull", "short": "bear"} — only accept
      trades where entry-day regime matches. If regime_gate is truthy but regime_map
      is None, raises. If direction is in gate keys, it must match; otherwise skip.
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
                if i >= n:
                    continue
                entry_price = float(open_[i])
                if entry_price <= 0:
                    continue

                entry_ts = int(ts[i])
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
                    exit_bar = min(i + MAX_HOLD_BARS, n - 1)
                    exit_reason = "timeout"
                    for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
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
                    # long
                    sl_trigger = entry_price * (1 - sl_pct - WORST_FILL_BUFFER)
                    tp_trigger = entry_price * (1 + tp_pct) if tp_pct is not None else None
                    peak_high = entry_price
                    exit_price = entry_price
                    exit_bar = min(i + MAX_HOLD_BARS, n - 1)
                    exit_reason = "timeout"
                    for j in range(i + 1, min(i + 1 + MAX_HOLD_BARS, n)):
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
                    "bars_held": exit_bar - i,
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
# SIZING + METRICS (per-trade USD)
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
# MONTHLY x REGIME SUMMARY
# =============================================================================


def _safe_pf(wins: float, losses: float) -> float:
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return wins / losses


def build_monthly_regime_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if len(trades_df) == 0:
        return pd.DataFrame(columns=[
            "month", "regime", "direction", "trades", "wr", "pf",
            "avg_pnl_pct", "total_pnl_pct", "total_pnl_usd",
            "max_drawdown_pct",
        ])
    df = trades_df.copy()
    dt = pd.to_datetime(df["entry_time"], unit="s", utc=True)
    df["month"] = dt.dt.strftime("%Y-%m")
    rows = []
    for (month, regime, direction), grp in df.groupby(
        ["month", "entry_regime", "direction"]
    ):
        wins = grp.loc[grp["pnl_pct"] > 0, "pnl_pct"].sum()
        losses = -grp.loc[grp["pnl_pct"] < 0, "pnl_pct"].sum()
        # Per-group equity curve for localized DD
        g = grp.sort_values("entry_time").reset_index(drop=True)
        eq = START_EQUITY_USD + g["pnl_usd"].cumsum()
        rm = eq.cummax()
        dd = (eq - rm) / rm.replace(0, np.nan)
        max_dd = float(dd.min()) if dd.notna().any() else 0.0
        rows.append({
            "month": month, "regime": regime, "direction": direction,
            "trades": int(len(grp)),
            "wr": float((grp["pnl_pct"] > 0).mean()),
            "pf": _safe_pf(wins, losses),
            "avg_pnl_pct": float(grp["pnl_pct"].mean()),
            "total_pnl_pct": float(grp["pnl_pct"].sum()),
            "total_pnl_usd": float(grp["pnl_usd"].sum()),
            "max_drawdown_pct": float(max_dd * 100.0),
        })
    out = pd.DataFrame(rows).sort_values(["month", "regime", "direction"])
    return out.reset_index(drop=True)


def build_regime_summary(trades_df: pd.DataFrame) -> pd.DataFrame:
    if len(trades_df) == 0:
        return pd.DataFrame(columns=[
            "regime", "trades", "wr", "pf", "pf_p5",
            "avg_pnl_pct", "total_pnl_pct", "total_pnl_usd", "max_drawdown_pct",
        ])
    rows = []
    for regime in ["bull", "bear", "sideways"]:
        grp = trades_df[trades_df["entry_regime"] == regime]
        if len(grp) == 0:
            rows.append({
                "regime": regime, "trades": 0, "wr": 0.0, "pf": 0.0,
                "pf_p5": 0.0, "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0,
                "total_pnl_usd": 0.0, "max_drawdown_pct": 0.0,
            })
            continue
        wins = grp.loc[grp["pnl_pct"] > 0, "pnl_pct"].sum()
        losses = -grp.loc[grp["pnl_pct"] < 0, "pnl_pct"].sum()
        g = grp.sort_values("entry_time").reset_index(drop=True)
        eq = START_EQUITY_USD + g["pnl_usd"].cumsum()
        rm = eq.cummax()
        dd = (eq - rm) / rm.replace(0, np.nan)
        max_dd = float(dd.min()) if dd.notna().any() else 0.0
        mc = monte_carlo_pf(grp)
        rows.append({
            "regime": regime, "trades": int(len(grp)),
            "wr": float((grp["pnl_pct"] > 0).mean()),
            "pf": _safe_pf(wins, losses),
            "pf_p5": mc["p5"],
            "avg_pnl_pct": float(grp["pnl_pct"].mean()),
            "total_pnl_pct": float(grp["pnl_pct"].sum()),
            "total_pnl_usd": float(grp["pnl_usd"].sum()),
            "max_drawdown_pct": float(max_dd * 100.0),
        })
    return pd.DataFrame(rows)


# =============================================================================
# VARIANT RUNNER
# =============================================================================


VARIANTS = [
    {
        "label": "price_accel_atr__short__all",
        "signal": "price_accel_atr",
        "directions": ["short"], "regime_gate": None,
    },
    {
        "label": "price_accel_atr__long__all",
        "signal": "price_accel_atr",
        "directions": ["long"], "regime_gate": None,
    },
    {
        "label": "multi_bar_confirm__short__all",
        "signal": "multi_bar_confirm",
        "directions": ["short"], "regime_gate": None,
    },
    {
        "label": "multi_bar_confirm__long__all",
        "signal": "multi_bar_confirm",
        "directions": ["long"], "regime_gate": None,
    },
    {
        "label": "price_accel_atr__combo__regime",
        "signal": "price_accel_atr",
        "directions": ["short", "long"],
        "regime_gate": {"long": "bull", "short": "bear"},
    },
    {
        "label": "multi_bar_confirm__combo__regime",
        "signal": "multi_bar_confirm",
        "directions": ["short", "long"],
        "regime_gate": {"long": "bull", "short": "bear"},
    },
]


def compute_listing_gates(candles_by_asset: dict) -> dict[str, int]:
    """Return {asset: earliest_tradable_ts_seconds} based on first snapshot candle
    + LISTING_WARMUP_DAYS. Prevents entry signals from firing before a coin has
    enough data for rolling windows to stabilize.
    """
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
    """Build entry masks (respecting per-coin listing gates), simulate, size."""
    signal = variant["signal"]
    entry_masks_by_dir: dict[str, dict[str, pd.Series]] = {}
    for direction in variant["directions"]:
        cache_key = (signal, direction)
        if cache_key not in signal_mask_cache:
            masks: dict[str, pd.Series] = {}
            for asset, df in candles_by_asset.items():
                atr = compute_atr(df)
                mask = detect_signal(df, signal, direction, atr)
                # Gate out any signal before the coin's effective listing date
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
# MAIN
# =============================================================================


def main() -> None:
    import random
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    overall_start = _time.monotonic()
    git_dirty_at_start = _git_dirty()
    if git_dirty_at_start:
        print(
            "WARNING: git working tree is dirty — not exactly reproducible.",
            file=sys.stderr,
        )

    print(f"[3y] loading snapshot {SNAPSHOT_DATE} ...")
    candles, universe = load_snapshot(SNAPSHOT_DATE)
    if pd.api.types.is_datetime64_any_dtype(candles["timestamp"]):
        candles = candles.copy()
        candles["timestamp"] = candles["timestamp"].astype("int64") // 10**9
    print(f"[3y] candles: {len(candles):,} rows, universe: {len(universe)} pairs")

    universe_assets = apply_universe_filter(universe)
    print(f"[3y] tradable universe: {len(universe_assets)} pairs")
    candles_tradable = candles[candles["asset"].isin(universe_assets)].copy()
    candles_by_asset = {
        asset: g.sort_values("timestamp").reset_index(drop=True)
        for asset, g in candles_tradable.groupby("asset")
    }
    print(f"[3y] candles grouped: {len(candles_by_asset)} assets")

    # BTC regime — use full-universe BTCUSDT if tradable universe drops it
    if BTC_ASSET not in candles_by_asset:
        btc_full = candles[candles["asset"] == BTC_ASSET].copy()
        if len(btc_full) == 0:
            raise RuntimeError(f"{BTC_ASSET} not found in snapshot — cannot classify regime")
        btc_full = btc_full.sort_values("timestamp").reset_index(drop=True)
    else:
        btc_full = candles_by_asset[BTC_ASSET]

    # Need datetime64 for resample — convert epoch_s back
    btc_for_regime = btc_full.copy()
    btc_for_regime["timestamp"] = pd.to_datetime(
        btc_for_regime["timestamp"], unit="s", utc=True
    )
    print("[3y] classifying BTC daily regime ...")
    regime_df = classify_btc_regime(btc_for_regime)
    regime_counts = regime_df["regime"].value_counts().to_dict()
    print(
        f"[3y] regime distribution (days): bull={regime_counts.get('bull', 0)} "
        f"bear={regime_counts.get('bear', 0)} sideways={regime_counts.get('sideways', 0)}"
    )
    regime_map = build_regime_map(regime_df)

    # Snapshot shas
    manifest_path = _SNAPSHOTS_DIR / "MANIFEST.md"
    candles_sha, universe_sha = _parse_manifest_row(
        manifest_path.read_text(), SNAPSHOT_DATE
    )

    base_run_id_prefix = (
        f"backtest_bigmover_longshort_3y_{_git_sha()[:8]}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )

    results_root = Path("results")
    results_root.mkdir(exist_ok=True)
    signal_mask_cache: dict = {}
    summary_rows: list[dict] = []

    # Per-coin listing gate (first candle ts + 3-day warmup)
    listing_gates = compute_listing_gates(candles_by_asset)
    gated_pct = sum(
        1 for ts in listing_gates.values()
        if ts > (candles.attrs.get("min_ts", 0) if hasattr(candles, "attrs") else 0)
    )
    late_listed = sum(
        1 for asset, ts in listing_gates.items()
        if ts > 1682035200  # 2023-04-21 epoch — roughly 20 days post-backtest-start
    )
    print(f"[3y] listing gates computed for {len(listing_gates)} assets; "
          f"{late_listed} listed after 2023-04-21 (warmup={LISTING_WARMUP_DAYS}d)")

    for i, variant in enumerate(VARIANTS, 1):
        v_start = _time.monotonic()
        label = variant["label"]
        print(f"\n[3y] variant {i}/{len(VARIANTS)}: {label} ...")

        trades, _ = run_variant(
            variant, candles_by_asset, regime_map, signal_mask_cache, listing_gates
        )

        metrics_pct = compute_metrics(trades)
        mc = monte_carlo_pf(trades)
        metrics_usd = compute_usd_metrics(trades)

        monthly_regime = build_monthly_regime_summary(trades)
        regime_summary = build_regime_summary(trades)

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
            **metrics_usd,
        }
        # Regime bucket quick pf/trades
        for r in ["bull", "bear", "sideways"]:
            sub = regime_summary[regime_summary["regime"] == r]
            if len(sub) == 0:
                headline[f"{r}_trades"] = 0
                headline[f"{r}_pf"] = 0.0
                headline[f"{r}_total_pnl_usd"] = 0.0
            else:
                row = sub.iloc[0]
                headline[f"{r}_trades"] = int(row["trades"])
                headline[f"{r}_pf"] = float(row["pf"])
                headline[f"{r}_total_pnl_usd"] = float(row["total_pnl_usd"])

        metrics = {**metrics_pct, **metrics_usd, **mc,
                   "regime_summary": regime_summary.to_dict(orient="records")}

        params = {
            "variant_label": label,
            "signal": variant["signal"],
            "directions": variant["directions"],
            "regime_gate": variant["regime_gate"],
            "SNAPSHOT_DATE": SNAPSHOT_DATE,
            "VOL_SPIKE": VOL_SPIKE, "PRICE_MOVE_THRESH": PRICE_MOVE_THRESH,
            "PRICE_LOOKBACK": PRICE_LOOKBACK, "ACCEL_MULT_OF_ATR": ACCEL_MULT_OF_ATR,
            "COOLDOWN_BARS": COOLDOWN_BARS,
            "MAX_HOLD_BARS": MAX_HOLD_BARS,
            "MAX_CONCURRENT_POSITIONS": MAX_CONCURRENT_POSITIONS,
            "POSITION_PCT": POSITION_PCT, "LEVERAGE": LEVERAGE,
            "START_EQUITY_USD": START_EQUITY_USD,
            "ROUND_TRIP_FEE": ROUND_TRIP_FEE,
            "EXIT_PARAMS": EXIT_PARAMS,
            "REGIME_SMA_FAST": REGIME_SMA_FAST,
            "REGIME_SMA_SLOW": REGIME_SMA_SLOW,
        }

        run_id = f"{base_run_id_prefix}__{label}"
        write_results(
            run_id=run_id, trades_df=trades, metrics=metrics, params=params,
            snapshot_date=SNAPSHOT_DATE,
            snapshot_candles_sha256=candles_sha,
            snapshot_universe_sha256=universe_sha,
            git_dirty=git_dirty_at_start,
            wall_time_seconds=_time.monotonic() - v_start,
            extra_files={
                "monthly_regime_summary.csv": monthly_regime,
                "regime_summary.csv": regime_summary,
            },
        )
        headline["run_id"] = run_id
        summary_rows.append(headline)

        v_elapsed = _time.monotonic() - v_start
        print(
            f"  trades={metrics_pct['trades']:5d}  wr={metrics_pct['wr']:.2f}  "
            f"pf={metrics_pct['pf']:.2f}  p5={mc['p5']:.2f}  "
            f"ret={metrics_usd['total_return_pct']:+.0f}%  "
            f"dd={metrics_usd['max_drawdown_pct']:.1f}%  "
            f"(bull PF {headline['bull_pf']:.2f} / bear PF {headline['bear_pf']:.2f} / "
            f"sideways PF {headline['sideways_pf']:.2f})  [{v_elapsed:.1f}s]"
        )

    # Cross-variant summary
    summary_df = pd.DataFrame(summary_rows)
    sha8 = _git_sha()[:8]
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = results_root / f"_longshort_3y_summary_{sha8}_{ts_str}.csv"
    # Tidy column ordering
    cols_front = [
        "variant", "signal", "trades", "wr", "pf", "pf_p5",
        "total_return_pct", "max_drawdown_pct", "sharpe",
        "bull_trades", "bull_pf", "bull_total_pnl_usd",
        "bear_trades", "bear_pf", "bear_total_pnl_usd",
        "sideways_trades", "sideways_pf", "sideways_total_pnl_usd",
        "run_id",
    ]
    cols = [c for c in cols_front if c in summary_df.columns] + [
        c for c in summary_df.columns if c not in cols_front
    ]
    summary_df[cols].to_csv(summary_path, index=False)

    total_elapsed = _time.monotonic() - overall_start
    print(f"\n[3y] done in {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"[3y] summary: {summary_path}")
    print("\nCross-variant headline (trades / pf / bull / bear / sideways):")
    print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
