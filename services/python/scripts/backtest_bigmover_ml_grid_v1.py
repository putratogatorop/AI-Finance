"""Bigmover ML grid backtest v1 — 3 signals x 7 ML thresholds x 2 sizing x 2 exits.

Grid dimensions (87 variants total):
  - Signals (3):    baseline, price_accel_atr, multi_bar_confirm
  - ML thresholds:  main grid {0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90} (7)
                    plus 3 no-ML anchors (threshold=0.00, fixed_10 / trail_3 only)
  - Sizing (2):     fixed_10 (10% equity per trade, 5x leverage)
                    kelly_cap15 (half-Kelly by ML bin, [2%, 15%])
  - Exits (2):      trail_3 (5% SL + trail 3%, activate +2%, 48h TO)
                    fixed_5_15 (5% SL + 15% TP, 48h TO)

Hybrid data flow:
  - Signal DETECTION from snapshot candles (reproducible, matches bigmover_tweaks_v1).
  - Signal FEATURES from postgres `volume_breakouts` (matches classifier training data).
  - Signals without DB feature match are SKIPPED from ML variants (conservative).
  - No-ML anchor variants take all signals, ignoring DB presence.

Run from services/python/:
  # sanity check
  python scripts/backtest_bigmover_ml_grid_v1.py --dry-run

  # one variant smoke test
  python scripts/backtest_bigmover_ml_grid_v1.py --single baseline__thr070__fixed_10__trail_3

  # full grid
  python scripts/backtest_bigmover_ml_grid_v1.py

Outputs:
  - Per-variant folders in results/<run_id>__<variant_label>/
  - Summary CSV at results/_grid_summary_bigmover_ml_v1_<sha8>_<ts>.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time as _time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

# Module-level names to exclude from params.json even though they're UPPERCASE.
_NON_PARAM_NAMES = frozenset({"UTC"})

# =============================================================================
# CONSTANTS
# =============================================================================
SNAPSHOT_DATE = "2026-04-24"
RANDOM_SEED = 42

# --- Universe ---
TOP_N = 100
MIN_QUOTE_VOL_24H = 500_000
LEVERAGED_TOKEN_RE = r"[35][LS]_USDT$"

# --- Detector params ---
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
ATR_PERIOD = 14
COOLDOWN_BARS = 96
TWEAK4_ACCEL_MULT_OF_ATR = 1.0  # price_accel_atr filter

# --- Portfolio ---
MAX_CONCURRENT_POSITIONS = 5
WORST_FILL_BUFFER = 0.005
MAX_HOLD_BARS = 192

# --- Sizing ---
START_EQUITY_USD = 100.0
LEVERAGE = 5
FIXED_10_FRACTION = 0.10
ROUND_TRIP_FEE = 0.0012  # 0.06% x 2 legs (Gate.io)

# Kelly config (half-Kelly by ML-probability bin; mirror of backtest_v2_resized.py)
KELLY_MULTIPLIER = 0.5
KELLY_FRACTION_FLOOR = 0.02
KELLY_FRACTION_CAP = 0.15
KELLY_BIN_EDGES = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 1.01]

# --- OOT ---
HOLD_OUT_MONTHS = 3

# --- Monte Carlo ---
MC_ITER = 1000
MC_SEED = 42

# --- ML ---
ML_MODEL_PATH = Path("models/bigmover_classifier.joblib")

# --- Grid ---
SIGNALS = ["baseline", "price_accel_atr", "multi_bar_confirm"]
ML_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
SIZING_METHODS = ["fixed_10", "kelly_cap15"]
EXIT_METHODS = ["trail_3", "fixed_5_15"]

EXIT_CONFIGS = {
    "trail_3": {
        "sl_pct": 0.05, "tp_pct": None,
        "trail_pct": 0.03, "trail_activation": 0.02,
    },
    "fixed_5_15": {
        "sl_pct": 0.05, "tp_pct": 0.15,
        "trail_pct": None, "trail_activation": None,
    },
}

# DB
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)

# Feature order must match classifier's feature_name_
ML_FEATURE_COLS = [
    "vol_ratio", "buy_ratio", "price_change_1bar", "price_change_2bar",
    "bar_range_pct", "upper_wick_pct", "lower_wick_pct", "body_pct",
    "atr_14", "rsi_14", "volatility_20",
    "dist_from_20_high", "dist_from_20_low", "price_vs_ema_50",
    "btc_ret_24bar", "btc_ret_96bar", "btc_vol_ratio",
    "direction", "tf_numeric",
    "avg_trade_size", "vol_x_buyrat", "candle_quality", "wick_ratio",
    "dollar_imbalance", "exec_vs_close", "vol_ratio_adj",
    "trend_aligned", "btc_aligned", "dist_range_pos",
    "coin_gainer_ratio",
]


# =============================================================================
# CANONICAL HELPERS (copied verbatim from backtest_bigmover_tweaks_v1.py)
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
    """Canonical PF/WR/avg-PnL using pnl_pct column."""
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


def compute_oot_split(candles: pd.DataFrame, hold_out_months: int = 3) -> tuple[int, int]:
    max_ts = int(candles["timestamp"].max())
    oot_start_ts = max_ts - (hold_out_months * 30 * 24 * 3600)
    return oot_start_ts, max_ts


# =============================================================================
# STRATEGY HELPERS (copied from bigmover_tweaks_v1)
# =============================================================================


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


def baseline_entry_mask(df: pd.DataFrame) -> pd.Series:
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


def detect_signal(df: pd.DataFrame, signal: str, atr: pd.Series) -> pd.Series:
    """Return entry mask for one of: baseline, price_accel_atr, multi_bar_confirm."""
    base = baseline_entry_mask(df)
    if signal == "baseline":
        return base
    if signal == "price_accel_atr":
        prev_close = df["close"].shift(1)
        prev_prev = df["close"].shift(2)
        accel = (df["close"] - prev_close) - (prev_close - prev_prev)
        return base & (accel.abs() >= atr * TWEAK4_ACCEL_MULT_OF_ATR)
    if signal == "multi_bar_confirm":
        next_close = df["close"].shift(-1)
        return (base & (next_close < df["close"])).fillna(False)
    raise ValueError(f"unknown signal: {signal}")


def simulate_portfolio(
    candles_by_asset: dict[str, pd.DataFrame],
    entry_masks: dict[str, pd.Series],
    exit_params: dict,
    cooldown_bars: int = COOLDOWN_BARS,
) -> pd.DataFrame:
    """Short-only exit engine with MAX_CONCURRENT_POSITIONS=5 + cooldown."""
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


# =============================================================================
# ML FEATURE LOADING AND SCORING
# =============================================================================


def engineer_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived/engineered columns required by bigmover_classifier.

    Mirrors engineer_features() in train_bigmover_classifier.py.
    Input df must have raw columns from volume_breakouts.
    """
    df = df.copy()
    df["avg_trade_size"] = df["quote_volume"] / df["trades"].clip(lower=1)
    df["vol_x_buyrat"] = df["vol_ratio"] * df["buy_ratio"]
    df["candle_quality"] = df["body_pct"] - df["upper_wick_pct"]
    df["wick_ratio"] = df["upper_wick_pct"] / (df["lower_wick_pct"] + 0.001)
    df["dollar_imbalance"] = (
        (2 * df["taker_buy_quote"] - df["quote_volume"]) / df["quote_volume"].clip(lower=1)
    )
    df["exec_vs_close"] = (
        (df["quote_volume"] / df["volume"].clip(lower=1e-10)) / df["close"].clip(lower=1e-10)
    )
    df["vol_ratio_adj"] = df["vol_ratio"] / (df["volatility_20"] + 0.001)
    df["trend_aligned"] = (df["direction"] * df["price_vs_ema_50"]).clip(lower=0)
    df["btc_aligned"] = (df["direction"] * df["btc_ret_24bar"]).clip(lower=0)
    df["dist_range_pos"] = df["dist_from_20_high"] / (
        df["dist_from_20_high"].abs() + df["dist_from_20_low"].abs() + 0.001
    )
    tf_map = {"15m": 0, "1h": 1, "4h": 2}
    df["tf_numeric"] = df["timeframe"].map(tf_map)
    return df


def add_coin_gainer_ratio(
    train_df: pd.DataFrame, target_df: pd.DataFrame
) -> pd.DataFrame:
    """Compute coin_gainer_ratio from train_df is_big_mover rates per symbol."""
    target_df = target_df.copy()
    coin_stats = train_df.groupby("symbol")["is_big_mover"].agg(["sum", "count"])
    coin_stats["coin_gainer_ratio"] = coin_stats["sum"] / coin_stats["count"]
    global_ratio = train_df["is_big_mover"].mean()
    ratio_map = coin_stats["coin_gainer_ratio"].to_dict()
    target_df["coin_gainer_ratio"] = (
        target_df["symbol"].map(ratio_map).fillna(global_ratio)
    )
    return target_df


def load_db_features(
    engine, oot_start_ts: int, oot_end_ts: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pull volume_breakouts rows for (timeframe='15m', direction=-1, vol_ratio>=3.0).

    Returns (pre_oot_df, oot_df) split on oot_start_ts. Both are post-engineered
    (engineer_ml_features + coin_gainer_ratio calibrated on pre_oot).

    pre_oot is used for Kelly calibration AND coin_gainer_ratio baseline.
    oot is the scoring target.
    """
    # NOTE: we intentionally DO NOT filter by `direction = -1`. The DB's direction
    # is the CURRENT bar's move direction (up/down). Our baseline detector fires on
    # macro-bearish context (close below 96-bar midpoint) which may coincide with
    # either up or down bars (e.g., counter-trend rally into supply = short setup).
    # direction is kept as a FEATURE in the classifier; restricting here would
    # cut ~90% of valid matches.
    sql = text("""
        SELECT symbol, signal_time, timeframe,
               vol_ratio, buy_ratio, price_change_1bar, price_change_2bar,
               bar_range_pct, upper_wick_pct, lower_wick_pct, body_pct,
               atr_14, rsi_14, volatility_20,
               dist_from_20_high, dist_from_20_low, price_vs_ema_50,
               btc_ret_24bar, btc_ret_96bar, btc_vol_ratio,
               direction, volume, quote_volume, trades, taker_buy_quote, close,
               fwd_max_gain_24h, fwd_max_loss_24h, is_big_mover
        FROM volume_breakouts
        WHERE timeframe = '15m'
          AND vol_ratio >= :vs
        ORDER BY signal_time
    """)
    df = pd.read_sql(sql, engine, params={"vs": VOL_SPIKE})
    # timestamps to epoch seconds (int64) to match snapshot convention
    df["signal_ts"] = (
        pd.to_datetime(df["signal_time"], utc=True).astype("int64") // 10**9
    )
    # asset key matches snapshot (symbol has no underscore already e.g. BTCUSDT)
    df["asset"] = df["symbol"].str.replace("_", "", regex=False)

    df = engineer_ml_features(df)

    # Split pre-OOT vs OOT
    pre_oot = df[df["signal_ts"] < oot_start_ts].reset_index(drop=True)
    oot = df[(df["signal_ts"] >= oot_start_ts) & (df["signal_ts"] <= oot_end_ts)].reset_index(
        drop=True
    )

    # Coin-gainer-ratio from pre-OOT
    pre_oot = add_coin_gainer_ratio(pre_oot, pre_oot)
    oot = add_coin_gainer_ratio(pre_oot, oot)

    # Clean inf/nan in feature columns
    for col in ML_FEATURE_COLS:
        if col in pre_oot.columns:
            pre_oot[col] = pre_oot[col].replace([np.inf, -np.inf], np.nan).fillna(0)
        if col in oot.columns:
            oot[col] = oot[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    return pre_oot, oot


def score_with_classifier(
    classifier, feature_df: pd.DataFrame
) -> np.ndarray:
    """Run classifier.predict_proba on feature_df, return P(class=1) column."""
    X = feature_df[ML_FEATURE_COLS]
    probs = classifier.predict_proba(X)
    # Binary classifier: class 1 is positive ("big mover")
    return probs[:, 1]


# =============================================================================
# KELLY SIZING (copied from backtest_v2_resized.py)
# =============================================================================


def build_kelly_table(pre_oot_trades: pd.DataFrame) -> pd.DataFrame:
    """Half-Kelly fraction per [bin_low, bin_high) ml_prob bin.

    pre_oot_trades must have columns: ml_prob, pnl_pct (decimal).
    """
    rows = []
    for lo, hi in zip(KELLY_BIN_EDGES[:-1], KELLY_BIN_EDGES[1:], strict=False):
        mask = (pre_oot_trades["ml_prob"] >= lo) & (pre_oot_trades["ml_prob"] < hi)
        sub = pre_oot_trades.loc[mask, "pnl_pct"]
        trades = len(sub)
        if trades == 0:
            rows.append({
                "bin_low": lo, "bin_high": hi, "trades": 0,
                "p": 0.0, "b": 0.0,
                "f_raw": 0.0, "f_half": 0.0,
                "f_capped": KELLY_FRACTION_FLOOR,
            })
            continue
        wins = sub[sub > 0]
        losses = sub[sub <= 0]
        p = len(wins) / trades
        avg_win = wins.mean() if len(wins) > 0 else 0.0
        avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
        if avg_loss == 0 or len(losses) == 0:
            f_raw, f_half = 1.0, KELLY_MULTIPLIER
            b = float("inf")
        elif avg_win == 0:
            f_raw, f_half, b = 0.0, 0.0, 0.0
        else:
            b = avg_win / avg_loss
            f_raw = p - (1.0 - p) / b
            f_half = f_raw * KELLY_MULTIPLIER
        f_capped = float(np.clip(f_half, KELLY_FRACTION_FLOOR, KELLY_FRACTION_CAP))
        rows.append({
            "bin_low": lo, "bin_high": hi, "trades": trades,
            "p": p, "b": b, "f_raw": f_raw, "f_half": f_half,
            "f_capped": f_capped,
        })
    return pd.DataFrame(rows)


def kelly_fraction_for(ml_prob: float, kelly_table: pd.DataFrame) -> float:
    for _, row in kelly_table.iterrows():
        if row["bin_low"] <= ml_prob < row["bin_high"]:
            return float(row["f_capped"])
    return KELLY_FRACTION_FLOOR


# =============================================================================
# SIZING APPLICATION (post-simulation)
# =============================================================================


def apply_sizing(
    trades_df: pd.DataFrame,
    sizing_method: str,
    kelly_table: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute pnl_usd per trade after fees, given sizing method.

    Adds columns: kelly_fraction, notional_usd, pnl_pct_net, pnl_usd.
    Uses flat START_EQUITY_USD (no compounding) — matches backtest_v2_resized.
    """
    df = trades_df.copy()
    if len(df) == 0:
        df["kelly_fraction"] = pd.Series(dtype=float)
        df["notional_usd"] = pd.Series(dtype=float)
        df["pnl_pct_net"] = pd.Series(dtype=float)
        df["pnl_usd"] = pd.Series(dtype=float)
        return df

    if sizing_method == "fixed_10":
        df["kelly_fraction"] = FIXED_10_FRACTION
    elif sizing_method == "kelly_cap15":
        if kelly_table is None or kelly_table.empty:
            df["kelly_fraction"] = KELLY_FRACTION_FLOOR
        else:
            df["kelly_fraction"] = df["ml_prob"].apply(
                lambda p: kelly_fraction_for(float(p), kelly_table)
            )
    else:
        raise ValueError(f"unknown sizing method: {sizing_method}")

    df["notional_usd"] = START_EQUITY_USD * df["kelly_fraction"] * LEVERAGE
    df["pnl_pct_net"] = df["pnl_pct"] - ROUND_TRIP_FEE
    df["pnl_usd"] = df["notional_usd"] * df["pnl_pct_net"]
    return df


def compute_usd_metrics(trades_df: pd.DataFrame) -> dict:
    """Final-equity / max-drawdown / sharpe on the pnl_usd column."""
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
    # Sharpe on per-trade returns (unannualized — trades are irregular)
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
# SIGNAL DETECTION + ML SCORE MERGE
# =============================================================================


def detect_and_score_signals(
    signal_name: str,
    candles_by_asset: dict,
    oot_df_scored: pd.DataFrame,
    oot_start_ts: int,
    oot_end_ts: int,
) -> tuple[dict[str, pd.Series], pd.DataFrame, int, int]:
    """For a given signal variant, detect entry bars in snapshot AND merge ml_prob.

    Returns:
      entry_masks:   {asset -> bool Series}  (all signals in OOT window)
      scored_signals_df: (asset, timestamp, ml_prob) rows for signals with DB match
      n_detected:    int (total signals detected in snapshot OOT)
      n_matched:     int (signals with DB feature match, hence ml_prob)
    """
    entry_masks: dict[str, pd.Series] = {}
    detected_tuples: list[tuple[str, int]] = []

    for asset, df in candles_by_asset.items():
        atr = compute_atr(df)
        mask = detect_signal(df, signal_name, atr)
        in_oot = (df["timestamp"] >= oot_start_ts) & (df["timestamp"] <= oot_end_ts)
        mask = mask & in_oot
        if not mask.any():
            continue
        entry_masks[asset] = mask
        fired_ts = df.loc[mask, "timestamp"].astype("int64").values
        for ts in fired_ts:
            detected_tuples.append((asset, int(ts)))

    n_detected = len(detected_tuples)
    detected_df = pd.DataFrame(detected_tuples, columns=["asset", "signal_ts"])
    if n_detected == 0:
        return entry_masks, pd.DataFrame(columns=["asset", "signal_ts", "ml_prob"]), 0, 0

    # Merge with oot_df_scored on (asset, signal_ts)
    merged = detected_df.merge(
        oot_df_scored[["asset", "signal_ts", "ml_prob"]],
        on=["asset", "signal_ts"],
        how="left",
    )
    matched = merged.dropna(subset=["ml_prob"])
    n_matched = len(matched)
    return entry_masks, matched.reset_index(drop=True), n_detected, n_matched


def filter_masks_by_threshold(
    entry_masks: dict[str, pd.Series],
    candles_by_asset: dict,
    scored_signals: pd.DataFrame,
    ml_threshold: float,
) -> dict[str, pd.Series]:
    """Restrict each asset's entry_mask to bars where ml_prob >= threshold.

    Signals without DB feature match get DROPPED (unless ml_threshold == 0.0, in
    which case the no-ML anchor keeps all detected signals regardless).
    """
    if ml_threshold <= 0.0:
        return entry_masks
    # Keep only signals with score >= threshold
    passing = scored_signals[scored_signals["ml_prob"] >= ml_threshold]
    # Build set of (asset, signal_ts) that pass
    pass_set: dict[str, set[int]] = {}
    for asset, group in passing.groupby("asset"):
        pass_set[asset] = set(int(t) for t in group["signal_ts"].values)
    new_masks: dict[str, pd.Series] = {}
    for asset, mask in entry_masks.items():
        df = candles_by_asset[asset]
        ts_arr = df["timestamp"].astype("int64").values
        pass_ts = pass_set.get(asset, set())
        new_vals = mask.values.copy()
        for i, ts in enumerate(ts_arr):
            if new_vals[i] and int(ts) not in pass_ts:
                new_vals[i] = False
        new_masks[asset] = pd.Series(new_vals, index=mask.index)
    return new_masks


# =============================================================================
# GRID ORCHESTRATION
# =============================================================================


def variant_label(signal: str, ml_threshold: float, sizing: str, exit_name: str) -> str:
    """Canonical label string for a variant row."""
    thr_str = f"thr{int(ml_threshold * 100):03d}" if ml_threshold > 0 else "thrNOML"
    return f"{signal}__{thr_str}__{sizing}__{exit_name}"


def build_grid() -> list[dict]:
    """Build the full 87-variant plan as a list of dicts.

    84 main-grid cells (3 signals x 7 thresholds x 2 sizing x 2 exits)
    + 3 no-ML anchors (threshold=0.0, fixed_10, trail_3).
    """
    cells = []
    for sig in SIGNALS:
        for thr in ML_THRESHOLDS:
            for sizing in SIZING_METHODS:
                for exit_name in EXIT_METHODS:
                    cells.append({
                        "signal": sig, "ml_threshold": thr,
                        "sizing": sizing, "exit": exit_name,
                    })
    # No-ML anchors (one per signal, default sizing + exit)
    for sig in SIGNALS:
        cells.append({
            "signal": sig, "ml_threshold": 0.0,
            "sizing": "fixed_10", "exit": "trail_3",
        })
    return cells


def build_trades_with_scores(
    entry_masks: dict[str, pd.Series],
    candles_by_asset: dict,
    exit_params: dict,
    scored_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Run simulate_portfolio then attach ml_prob to each trade by (symbol, entry_time).

    Returns a trades DataFrame extended with ml_prob column (NaN if no match).
    """
    trades = simulate_portfolio(candles_by_asset, entry_masks, exit_params)
    if len(trades) == 0:
        trades["ml_prob"] = []
        return trades
    if len(scored_signals) == 0:
        trades["ml_prob"] = np.nan
        return trades
    lookup = scored_signals.set_index(["asset", "signal_ts"])["ml_prob"].to_dict()
    trades["ml_prob"] = [
        float(lookup.get((row["symbol"], int(row["entry_time"])), np.nan))
        for _, row in trades.iterrows()
    ]
    return trades


def simulate_pre_oot_for_kelly(
    signal_name: str,
    candles_by_asset: dict,
    pre_oot_df_scored: pd.DataFrame,
    oot_start_ts: int,
    ml_threshold: float,
    exit_params: dict,
) -> pd.DataFrame:
    """Generate a pre-OOT trade stream at the same ml_threshold so Kelly can calibrate.

    Only used when sizing == 'kelly_cap15'. Returns trades DF with ml_prob + pnl_pct.
    """
    entry_masks: dict[str, pd.Series] = {}
    detected_tuples: list[tuple[str, int]] = []
    for asset, df in candles_by_asset.items():
        atr = compute_atr(df)
        mask = detect_signal(df, signal_name, atr)
        in_pre = df["timestamp"] < oot_start_ts
        mask = mask & in_pre
        if not mask.any():
            continue
        entry_masks[asset] = mask
        fired_ts = df.loc[mask, "timestamp"].astype("int64").values
        for ts in fired_ts:
            detected_tuples.append((asset, int(ts)))
    if not detected_tuples:
        return pd.DataFrame(columns=["ml_prob", "pnl_pct"])
    detected_df = pd.DataFrame(detected_tuples, columns=["asset", "signal_ts"])
    merged = detected_df.merge(
        pre_oot_df_scored[["asset", "signal_ts", "ml_prob"]],
        on=["asset", "signal_ts"], how="left",
    )
    scored = merged.dropna(subset=["ml_prob"])

    # Filter by threshold
    filtered_masks = filter_masks_by_threshold(
        entry_masks, candles_by_asset, scored, ml_threshold
    )
    trades = build_trades_with_scores(
        filtered_masks, candles_by_asset, exit_params, scored
    )
    return trades


# =============================================================================
# MAIN
# =============================================================================


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Load model + small feature sample, exit without simulating.")
    ap.add_argument("--single", type=str, default=None,
                    help="Run only the named variant (e.g., baseline__thr070__fixed_10__trail_3).")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip variants whose results folder already exists.")
    args = ap.parse_args()

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

    # Load classifier
    print(f"[grid] loading ML model {ML_MODEL_PATH} ...")
    if not ML_MODEL_PATH.exists():
        raise FileNotFoundError(f"ML model not found: {ML_MODEL_PATH}")
    classifier = joblib.load(ML_MODEL_PATH)
    expected_features = list(classifier.feature_name_)
    if expected_features != ML_FEATURE_COLS:
        print("FATAL: classifier feature_name_ differs from ML_FEATURE_COLS constant.")
        print(f"  classifier: {expected_features}")
        print(f"  code:       {ML_FEATURE_COLS}")
        raise RuntimeError("feature list mismatch")
    print(f"[grid] classifier features OK ({len(expected_features)} cols)")

    # Load snapshot
    print(f"[grid] loading snapshot {SNAPSHOT_DATE} ...")
    candles, universe = load_snapshot(SNAPSHOT_DATE)
    if pd.api.types.is_datetime64_any_dtype(candles["timestamp"]):
        candles = candles.copy()
        candles["timestamp"] = candles["timestamp"].astype("int64") // 10**9
    print(f"[grid] candles: {len(candles):,} rows, universe: {len(universe)} pairs")

    universe_assets = apply_universe_filter(universe)
    print(f"[grid] tradable universe: {len(universe_assets)} pairs")
    candles = candles[candles["asset"].isin(universe_assets)].copy()
    candles_by_asset = {
        asset: g.sort_values("timestamp").reset_index(drop=True)
        for asset, g in candles.groupby("asset")
    }
    print(f"[grid] candles grouped: {len(candles_by_asset)} assets")

    # OOT split
    oot_start_ts, max_ts = compute_oot_split(candles, HOLD_OUT_MONTHS)
    oot_start_dt = datetime.fromtimestamp(oot_start_ts, tz=UTC)
    max_ts_dt = datetime.fromtimestamp(max_ts, tz=UTC)
    print(f"[grid] OOT window: {oot_start_dt.date()} -> {max_ts_dt.date()}")

    # Load ML features from DB
    print("[grid] loading ML features from postgres volume_breakouts ...")
    engine = create_engine(DB_URL)
    pre_oot_df, oot_df = load_db_features(engine, oot_start_ts, max_ts)
    print(f"[grid] pre-OOT DB rows: {len(pre_oot_df):,}  OOT DB rows: {len(oot_df):,}")

    if len(oot_df) == 0:
        raise RuntimeError("no OOT feature rows found in volume_breakouts — cannot score")

    # Score OOT + pre-OOT rows with classifier (cache to avoid re-scoring per variant)
    print("[grid] scoring signals with classifier ...")
    oot_df["ml_prob"] = score_with_classifier(classifier, oot_df)
    pre_oot_df["ml_prob"] = score_with_classifier(classifier, pre_oot_df)
    print(f"[grid] OOT ml_prob range: [{oot_df['ml_prob'].min():.3f}, "
          f"{oot_df['ml_prob'].max():.3f}]  mean: {oot_df['ml_prob'].mean():.3f}")

    if args.dry_run:
        print("[grid] --dry-run: model + features OK. Exiting without simulation.")
        return

    # Snapshot shas for write_results
    manifest_path = _SNAPSHOTS_DIR / "MANIFEST.md"
    candles_sha, universe_sha = _parse_manifest_row(
        manifest_path.read_text(), SNAPSHOT_DATE
    )

    # Pre-detect signal masks + scored per signal variant (avoid redundant work)
    print("[grid] pre-detecting signal masks per variant ...")
    signal_cache: dict[str, dict] = {}
    for sig in SIGNALS:
        t0 = _time.monotonic()
        entry_masks, scored_signals, n_det, n_match = detect_and_score_signals(
            sig, candles_by_asset, oot_df, oot_start_ts, max_ts
        )
        dt = _time.monotonic() - t0
        print(f"  {sig}: {n_det} detected, {n_match} matched to DB "
              f"({100*n_match/max(1,n_det):.1f}%)  [{dt:.1f}s]")
        signal_cache[sig] = {
            "entry_masks": entry_masks,
            "scored_signals": scored_signals,
            "n_detected": n_det, "n_matched": n_match,
        }

    # Build grid
    grid = build_grid()
    if args.single:
        grid = [
            g for g in grid
            if variant_label(g["signal"], g["ml_threshold"], g["sizing"], g["exit"]) == args.single
        ]
        if not grid:
            raise ValueError(f"no variant matches --single {args.single}")
        print(f"[grid] --single: running 1 variant: {args.single}")
    print(f"[grid] running {len(grid)} variants ...")

    base_run_id_prefix = f"backtest_bigmover_ml_grid_v1_{_git_sha()[:8]}_" \
                        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"

    # Cache: trades_df per (signal, ml_threshold, exit_name) — sizing is post-proc
    sim_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    kelly_cache: dict[tuple[str, float, str], pd.DataFrame] = {}

    summary_rows: list[dict] = []
    results_root = Path("results")
    results_root.mkdir(exist_ok=True)

    for i, cell in enumerate(grid, 1):
        signal = cell["signal"]
        thr = cell["ml_threshold"]
        sizing = cell["sizing"]
        exit_name = cell["exit"]
        label = variant_label(signal, thr, sizing, exit_name)
        run_id = f"{base_run_id_prefix}__{label}"
        run_dir = results_root / run_id
        if args.skip_existing and run_dir.exists():
            print(f"  [{i}/{len(grid)}] {label}: SKIP (exists)")
            continue

        v_start = _time.monotonic()

        # Apply ML threshold filter to entry masks
        cached = signal_cache[signal]
        filtered_masks = filter_masks_by_threshold(
            cached["entry_masks"], candles_by_asset,
            cached["scored_signals"], thr,
        )

        # Simulate (cache across sizing methods)
        sim_key = (signal, thr, exit_name)
        if sim_key not in sim_cache:
            exit_params = EXIT_CONFIGS[exit_name]
            trades = build_trades_with_scores(
                filtered_masks, candles_by_asset, exit_params,
                cached["scored_signals"],
            )
            sim_cache[sim_key] = trades
        trades = sim_cache[sim_key].copy()

        # Kelly table calibration (pre-OOT at same threshold)
        kelly_table = None
        if sizing == "kelly_cap15":
            kelly_key = (signal, thr, exit_name)
            if kelly_key not in kelly_cache:
                pre_trades = simulate_pre_oot_for_kelly(
                    signal, candles_by_asset, pre_oot_df,
                    oot_start_ts, thr, EXIT_CONFIGS[exit_name],
                )
                if len(pre_trades) == 0:
                    kt = build_kelly_table(
                        pd.DataFrame({"ml_prob": [], "pnl_pct": []})
                    )
                else:
                    kt = build_kelly_table(pre_trades[["ml_prob", "pnl_pct"]])
                kelly_cache[kelly_key] = kt
            kelly_table = kelly_cache[kelly_key]

        # Apply sizing (post-proc)
        trades = apply_sizing(trades, sizing, kelly_table)

        # Metrics
        metrics_pct = compute_metrics(trades)
        mc = monte_carlo_pf(trades)
        metrics_usd = compute_usd_metrics(trades)

        winners = trades[trades["pnl_pct"] > 0]["pnl_pct"]
        avg_winner = float(winners.mean()) if len(winners) > 0 else 0.0
        median_winner = float(winners.median()) if len(winners) > 0 else 0.0

        metrics = {
            **metrics_pct,
            "pf_p5": mc["p5"], "pf_p50": mc["p50"], "pf_p95": mc["p95"],
            **metrics_usd,
            "avg_winner_pct": avg_winner,
            "median_winner_pct": median_winner,
            "n_matched_to_db": cached["n_matched"],
            "n_detected_in_snapshot": cached["n_detected"],
        }

        params = {
            "signal": signal, "ml_threshold": thr,
            "sizing": sizing, "exit": exit_name,
            "SNAPSHOT_DATE": SNAPSHOT_DATE,
            "HOLD_OUT_MONTHS": HOLD_OUT_MONTHS,
            "oot_start_ts": oot_start_ts,
            "VOL_SPIKE": VOL_SPIKE, "PRICE_MOVE_THRESH": PRICE_MOVE_THRESH,
            "COOLDOWN_BARS": COOLDOWN_BARS,
            "MAX_HOLD_BARS": MAX_HOLD_BARS,
            "MAX_CONCURRENT_POSITIONS": MAX_CONCURRENT_POSITIONS,
            "START_EQUITY_USD": START_EQUITY_USD,
            "LEVERAGE": LEVERAGE, "ROUND_TRIP_FEE": ROUND_TRIP_FEE,
            "exit_params": EXIT_CONFIGS[exit_name],
            "KELLY_BIN_EDGES": KELLY_BIN_EDGES,
            "KELLY_FRACTION_FLOOR": KELLY_FRACTION_FLOOR,
            "KELLY_FRACTION_CAP": KELLY_FRACTION_CAP,
        }

        write_results(
            run_id=run_id, trades_df=trades, metrics=metrics, params=params,
            snapshot_date=SNAPSHOT_DATE,
            snapshot_candles_sha256=candles_sha,
            snapshot_universe_sha256=universe_sha,
            git_dirty=git_dirty_at_start,
            wall_time_seconds=_time.monotonic() - v_start,
        )

        v_elapsed = _time.monotonic() - v_start
        print(
            f"  [{i}/{len(grid)}] {label}: "
            f"trades={metrics['trades']:4d} wr={metrics['wr']:.2f} "
            f"pf={metrics['pf']:.2f} p5={mc['p5']:.2f} "
            f"ret={metrics_usd['total_return_pct']:+.1f}% dd={metrics_usd['max_drawdown_pct']:.1f}% "
            f"({v_elapsed:.1f}s)"
        )

        summary_rows.append({
            "signal": signal, "ml_threshold": thr,
            "sizing": sizing, "exit": exit_name,
            "trades": metrics["trades"], "wr": metrics["wr"], "pf": metrics["pf"],
            "pf_p5": mc["p5"], "pf_p50": mc["p50"], "pf_p95": mc["p95"],
            "avg_pnl_pct": metrics["avg_pnl_pct"],
            "total_pnl_pct": metrics["total_pnl_pct"],
            "final_equity_usd": metrics_usd["final_equity_usd"],
            "total_return_pct": metrics_usd["total_return_pct"],
            "max_drawdown_pct": metrics_usd["max_drawdown_pct"],
            "sharpe": metrics_usd["sharpe"],
            "n_matched": cached["n_matched"],
            "n_detected": cached["n_detected"],
            "run_id": run_id,
        })

    if args.single:
        print("[grid] --single complete, skipping summary CSV.")
        return

    # Summary CSV
    summary_df = pd.DataFrame(summary_rows).sort_values(
        "total_return_pct", ascending=False
    )
    sha8 = _git_sha()[:8]
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = results_root / f"_grid_summary_bigmover_ml_v1_{sha8}_{ts_str}.csv"
    summary_df.to_csv(summary_path, index=False)

    total_elapsed = _time.monotonic() - overall_start
    print(f"\n[grid] done in {total_elapsed:.0f}s ({total_elapsed/60:.1f}m)")
    print(f"[grid] summary: {summary_path}")
    print("\nTop 10 by total_return_pct:")
    print(summary_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
