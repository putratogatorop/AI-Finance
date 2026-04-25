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

# --- PARAMS — predictor gate ---
BTC_ASSET = "BTCUSDT"
PREDICTOR_PATH = "models/continuation_v1.joblib"
PREDICTOR_META_PATH = "models/continuation_v1_meta.json"
# The threshold is locked at run-time as the 67th percentile of training-set
# predicted probabilities (matches Gate-A's selectivity ~33% kept). The
# *quantile* is the locked decision; the resulting probability cutoff is
# recomputed from training data on every run (deterministic given data + seed).
TRAIN_THRESHOLD_QUANTILE = 2.0 / 3.0

# --- PARAMS — OOT split per the 8-point standard ---
OOT_SPLIT_DATE = "2026-01-01"  # train ends here; OOT = (OOT_SPLIT_DATE, SNAPSHOT_DATE)

# --- PARAMS — ship gate (the 8-point standard) ---
SHIP_PF_FLOOR = 1.30
SHIP_OOT_N_FLOOR = 200
WF_FOLD_MONTHS = 3
WF_STEP_MONTHS = 1
MC_N_RESAMPLES = 5000

MIN_BARS_PER_ASSET = VOL_MA_PERIOD + max(PRICE_LOOKBACK_B, 96) + 10

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


def _run_strategy_b_with_predictor(candles: pd.DataFrame) -> pd.DataFrame:
    """Detect Strategy B entries, score each with the trained predictor,
    simulate, and return the full trade table tagged with `proba`.

    Filtering by the predictor threshold happens *after* this function so
    we can compute predictor-on / predictor-off metrics in the same run.
    """
    import joblib

    # Ensure services/python/ is on sys.path so we can import src.ml.* when
    # the script is invoked as `python scripts/backtest_strat_b_with_predictor.py`.
    services_python_root = str(Path(__file__).resolve().parents[1])
    if services_python_root not in sys.path:
        sys.path.insert(0, services_python_root)

    from src.ml.continuation.features import FEATURE_NAMES, FeatureContext

    print("[1/3] loading predictor + building feature panels...", flush=True)
    predictor_path = Path(__file__).resolve().parents[1] / PREDICTOR_PATH
    if not predictor_path.exists():
        raise FileNotFoundError(
            f"Predictor artifact missing: {predictor_path}. "
            f"Run scripts/train_continuation_classifier.py first."
        )
    model = joblib.load(predictor_path)
    ctx = FeatureContext(candles, btc_asset_key=BTC_ASSET)

    print("[2/3] running Strategy B per asset, scoring + simulating...",
          flush=True)
    rows: list[dict] = []
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
            # timestamps[eb] is already tz-aware (UTC) from the snapshot parquet.
            entry_ts = pd.Timestamp(timestamps[eb])
            if entry_ts.tzinfo is None:
                entry_ts = entry_ts.tz_localize("UTC")
            t = _simulate_short_trade(close, high, low, eb)
            if t is None:
                continue
            feats = ctx._build_for_typed(asset, entry_ts)
            X_one = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=float)
            if np.isnan(X_one).any():
                proba = float("nan")
            else:
                proba = float(model.predict_proba(X_one)[0, 1])

            t["direction"] = "short"
            t["symbol"] = asset
            t["entry_time"] = str(entry_ts)
            t["proba"] = proba
            rows.append(t)

        if (idx + 1) % 25 == 0:
            print(f"  [{idx + 1}/{len(assets)}] trades-collected={len(rows):,}",
                  flush=True)

    print("[3/3] done detecting + scoring", flush=True)
    if rows:
        return pd.DataFrame(rows)
    return pd.DataFrame(columns=["direction", "pnl_pct", "bars_held"])


def _pf(pnls: np.ndarray) -> float:
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return float("nan")
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses == 0:
        return float("inf")
    return float(wins / losses)


def _walkforward(trades_df: pd.DataFrame) -> dict:
    """Rolling 3-month folds (step=1mo) over the predictor-on trades.

    Returns a dict with per-fold rows + roll-up percentiles.
    """
    from dateutil.relativedelta import relativedelta

    if len(trades_df) == 0:
        return {"n_folds": 0, "folds": [], "rollup_pf": {}}

    et = pd.to_datetime(trades_df["entry_time"], utc=True)
    start = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    month = relativedelta(months=1)
    fold_window = relativedelta(months=WF_FOLD_MONTHS)
    folds: list[dict] = []
    s = start
    while s + fold_window <= end_global:
        e = s + fold_window
        sub = trades_df[(et >= s) & (et < e)]
        folds.append({
            "fold_start": str(s.date()),
            "fold_end": str(e.date()),
            "trades": int(len(sub)),
            "wr": float((sub["pnl_pct"] > 0).mean()) if len(sub) else 0.0,
            "pf": _pf(sub["pnl_pct"].to_numpy()),
            "total_pnl_pct": float(sub["pnl_pct"].sum()),
        })
        s += month

    pfs = np.array([
        f["pf"] for f in folds
        if f["trades"] >= 30 and np.isfinite(f["pf"])
    ])
    if len(pfs):
        rollup = {
            "n_folds_used": int(len(pfs)),
            "p5": float(np.percentile(pfs, 5)),
            "p25": float(np.percentile(pfs, 25)),
            "p50": float(np.percentile(pfs, 50)),
            "p75": float(np.percentile(pfs, 75)),
            "p95": float(np.percentile(pfs, 95)),
            "p_lt_1": float((pfs < 1.0).mean()),
            "p_gte_130": float((pfs >= 1.30).mean()),
        }
    else:
        rollup = {"n_folds_used": 0}
    return {"n_folds": len(folds), "folds": folds, "rollup_pf": rollup}


def _monte_carlo(pnls: np.ndarray, n: int = MC_N_RESAMPLES) -> dict:
    """Bootstrap PF distribution by resampling trades with replacement."""
    if len(pnls) == 0:
        return {"n_resamples": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    pfs = np.empty(n, dtype=float)
    for k in range(n):
        sample = rng.choice(pnls, size=len(pnls), replace=True)
        pfs[k] = _pf(sample)
    finite = pfs[np.isfinite(pfs)]
    return {
        "n_resamples": int(n),
        "p5": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
        "p50": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
        "p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
        "point_pf": _pf(pnls),
        "p_pf_lt_1": float((finite < 1.0).mean()) if len(finite) else float("nan"),
        "p_pf_gte_130": float((finite >= 1.30).mean()) if len(finite) else float("nan"),
    }


def main() -> None:
    import json
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

    trades_df = _run_strategy_b_with_predictor(candles)

    # Tag train vs OOT.
    et = pd.to_datetime(trades_df["entry_time"], utc=True)
    boundary = pd.Timestamp(OOT_SPLIT_DATE, tz="UTC")
    trades_df["period"] = np.where(et < boundary, "train", "oot")

    # Lock predictor threshold from training-set scores (top 1/3).
    train_with_score = trades_df[(trades_df["period"] == "train") & trades_df["proba"].notna()]
    if len(train_with_score) == 0:
        raise RuntimeError("No training-period trades with predictor scores — "
                           "feature warmup failed for the entire training set.")
    threshold = float(train_with_score["proba"].quantile(TRAIN_THRESHOLD_QUANTILE))
    trades_df["predictor_on"] = trades_df["proba"] >= threshold

    print(f"\nlocked predictor threshold (q{TRAIN_THRESHOLD_QUANTILE:.4f} of train scores): "
          f"{threshold:.4f}", flush=True)

    # 4-cell metrics.
    cell = {}
    for period in ("train", "oot"):
        for gate in ("off", "on"):
            mask = trades_df["period"] == period
            if gate == "on":
                mask = mask & trades_df["predictor_on"]
            cell[f"{period}_{gate}"] = compute_metrics(
                trades_df[mask] if mask.any() else trades_df.iloc[0:0]
            )
            print(f"  {period}_{gate}: trades={cell[f'{period}_{gate}']['trades']:>6} "
                  f"PF={cell[f'{period}_{gate}']['pf']:.3f} "
                  f"WR={cell[f'{period}_{gate}']['wr']*100:.1f}% "
                  f"avg={cell[f'{period}_{gate}']['avg_pnl_pct']*100:+.3f}%",
                  flush=True)

    # Walk-forward across the full timeline (predictor_on only).
    print("\nwalk-forward (predictor_on only, 3-month rolling, 1-month step)...", flush=True)
    wf = _walkforward(trades_df[trades_df["predictor_on"]])
    print(f"  folds: {wf['n_folds']}  used: {wf['rollup_pf'].get('n_folds_used', 0)}", flush=True)
    if wf["rollup_pf"].get("n_folds_used", 0) > 0:
        print(f"  p5={wf['rollup_pf']['p5']:.3f}  p50={wf['rollup_pf']['p50']:.3f}  "
              f"p95={wf['rollup_pf']['p95']:.3f}  P(loss)={wf['rollup_pf']['p_lt_1']*100:.1f}%",
              flush=True)

    # Monte Carlo on OOT predictor_on trades.
    print("\nmonte-carlo bootstrap on OOT predictor_on trades...", flush=True)
    oot_on_pnls = trades_df[(trades_df["period"] == "oot") & trades_df["predictor_on"]][
        "pnl_pct"
    ].to_numpy()
    mc = _monte_carlo(oot_on_pnls)
    print(f"  point PF={mc.get('point_pf', float('nan')):.3f}  "
          f"p5={mc.get('p5', float('nan')):.3f}  "
          f"p50={mc.get('p50', float('nan')):.3f}  "
          f"p95={mc.get('p95', float('nan')):.3f}", flush=True)

    # Ship gate.
    oot_on = cell["oot_on"]
    oot_n_ok = oot_on["trades"] >= SHIP_OOT_N_FLOOR
    oot_pf_ok = bool(np.isfinite(oot_on["pf"]) and oot_on["pf"] >= SHIP_PF_FLOOR)
    wf_p5_ok = bool(wf["rollup_pf"].get("p5", -1) > 1.0)
    mc_p5_ok = bool(np.isfinite(mc.get("p5", float("nan"))) and mc.get("p5", -1) > 1.0)
    ship_gate = {
        "oot_pf_ge_130": oot_pf_ok,
        "oot_n_ge_200": oot_n_ok,
        "walkforward_p5_gt_1": wf_p5_ok,
        "mc_p5_gt_1": mc_p5_ok,
        "overall": oot_pf_ok and oot_n_ok and wf_p5_ok and mc_p5_ok,
    }

    metrics = {
        "predictor_threshold": threshold,
        "predictor_threshold_quantile": TRAIN_THRESHOLD_QUANTILE,
        "by_cell": cell,
        "walkforward": wf,
        "monte_carlo_oot_on": mc,
        "ship_gate": ship_gate,
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
    out_dir = write_results(
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

    # Bonus per-artifact summaries (mc, wf) so reviewers can read fast.
    (out_dir / "mc_summary.json").write_text(json.dumps(mc, indent=2, default=str))
    (out_dir / "walkforward_summary.json").write_text(json.dumps(wf, indent=2, default=str))

    print(f"\nrun_id: {run_id}")
    print("ship_gate (8-point standard):")
    for k, v in ship_gate.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  → overall: {'SHIP' if ship_gate['overall'] else 'DO NOT SHIP'}")


if __name__ == "__main__":
    main()
