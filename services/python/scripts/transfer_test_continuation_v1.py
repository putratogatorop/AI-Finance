"""Transfer test: apply continuation_v1.joblib (no retraining) to Strategy A and C trades.

Plan reference (Phase 5): "same continuation_v1.joblib (no retraining) on
Strategy A and C trades; report whether the predictor generalizes."

What we measure: 4-cell PF/WR table per strategy, plus the ship-gate's OOT
PF and MC bootstrap p5 (skip walk-forward — that's a transfer-test, not a
deploy decision). Threshold is locked at the value Phase 4 produced
(meta.json's training-quantile threshold, recomputed deterministically).

Run from services/python/:
    python scripts/transfer_test_continuation_v1.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.continuation.features import FEATURE_NAMES, FeatureContext
from src.ml.continuation.labels import filter_labelable

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42

CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
MODEL_PATH = REPO_ROOT / "services/python/models" / "continuation_v1.joblib"

# Same B run that trained the predictor, used to lock the threshold deterministically.
B_TRADES_PATH = (
    REPO_ROOT / "services/python/results"
    / "backtest_3strat_protocol_B_28975aa0_20260424T093815Z" / "trades.csv"
)
A_TRADES_PATH = (
    REPO_ROOT / "services/python/results"
    / "backtest_3strat_protocol_A_28975aa0_20260424T093815Z" / "trades.csv"
)
C_TRADES_PATH = (
    REPO_ROOT / "services/python/results"
    / "backtest_3strat_protocol_C_28975aa0_20260424T093815Z" / "trades.csv"
)

THRESHOLD_QUANTILE = 2.0 / 3.0
LABEL_MAX_BARS = 192
MC_N_RESAMPLES = 5000


def _pf(pnls: np.ndarray) -> float:
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return float("nan")
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    if losses == 0:
        return float("inf")
    return float(wins / losses)


def _score_trades(trades: pd.DataFrame, model, ctx: FeatureContext) -> pd.Series:
    """Score every trade with the predictor; NaN where any feature is missing."""
    rows = []
    for row in trades.itertuples(index=False):
        feats = ctx._build_for_typed(row.symbol, row.entry_time)
        rows.append([feats[n] for n in FEATURE_NAMES])
    X = np.asarray(rows, dtype=float)
    finite = np.isfinite(X).all(axis=1)
    proba = np.full(len(trades), np.nan, dtype=float)
    if finite.any():
        proba[finite] = model.predict_proba(X[finite])[:, 1]
    return pd.Series(proba, index=trades.index, name="proba")


def _cell_metrics(trades: pd.DataFrame) -> dict:
    """Return {trades, wr, pf, avg_pnl_pct} for the slice."""
    if len(trades) == 0:
        return {"trades": 0, "wr": 0.0, "pf": float("nan"), "avg_pnl_pct": 0.0}
    pnl = trades["pnl_pct"].to_numpy(dtype=float)
    return {
        "trades": int(len(pnl)),
        "wr": float((pnl > 0).mean()),
        "pf": _pf(pnl),
        "avg_pnl_pct": float(pnl.mean()),
    }


def _bootstrap_p5(pnl: np.ndarray, n: int = MC_N_RESAMPLES) -> tuple[float, float, float]:
    if len(pnl) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(RANDOM_SEED)
    pfs = np.empty(n, dtype=float)
    for k in range(n):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        pfs[k] = _pf(sample)
    finite = pfs[np.isfinite(pfs)]
    if len(finite) == 0:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.percentile(finite, 5)),
        float(np.percentile(finite, 50)),
        float(np.percentile(finite, 95)),
    )


def _evaluate_strategy(
    label: str,
    trades_path: Path,
    model,
    ctx: FeatureContext,
    candles_max_ts: pd.Timestamp,
    threshold: float,
) -> dict:
    print(f"\n=== Transfer test: Strategy {label} ===", flush=True)
    trades = pd.read_csv(trades_path)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    print(f"  total trades in source: {len(trades):,}", flush=True)

    # Restrict to trades whose features can plausibly be built (label horizon
    # is irrelevant for scoring, but warmup is). We keep everything; missing
    # features yield NaN proba and are excluded from predictor_on by definition.
    trades = trades.sort_values("entry_time").reset_index(drop=True)

    print("  scoring...", flush=True)
    trades["proba"] = _score_trades(trades, model, ctx)
    n_scored = int(trades["proba"].notna().sum())
    print(f"  scored: {n_scored} / {len(trades)} (NaN where features incomplete)", flush=True)

    trades["period"] = np.where(trades["entry_time"] < OOT_BOUNDARY, "train", "oot")
    trades["predictor_on"] = trades["proba"] >= threshold

    cell = {
        f"{period}_{gate}": _cell_metrics(
            trades[(trades["period"] == period) & (trades["predictor_on"] if gate == "on" else ~trades["predictor_on"].isna() & ~False)]
            if False else (
                trades[(trades["period"] == period) & trades["predictor_on"]]
                if gate == "on"
                else trades[trades["period"] == period]
            )
        )
        for period in ("train", "oot")
        for gate in ("off", "on")
    }
    for k, v in cell.items():
        print(f"  {k}: trades={v['trades']:>6} PF={v['pf']:.3f} "
              f"WR={v['wr']*100:.1f}% avg={v['avg_pnl_pct']*100:+.3f}%", flush=True)

    # Monte-Carlo on OOT predictor_on
    oot_on_pnl = trades[(trades["period"] == "oot") & trades["predictor_on"]]["pnl_pct"].to_numpy()
    p5, p50, p95 = _bootstrap_p5(oot_on_pnl)
    print(f"  MC bootstrap (OOT predictor_on, n={len(oot_on_pnl)}): "
          f"p5={p5:.3f}  p50={p50:.3f}  p95={p95:.3f}", flush=True)

    return {
        "strategy": label,
        "n_total_trades": int(len(trades)),
        "n_scored": n_scored,
        "by_cell": cell,
        "mc_oot_on": {"p5": p5, "p50": p50, "p95": p95, "n": int(len(oot_on_pnl))},
    }


def main() -> None:
    started = time.monotonic()
    print(f"loading model from {MODEL_PATH}...", flush=True)
    model = joblib.load(MODEL_PATH)

    print(f"loading candles from {CANDLES_PATH}...", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    candles_max_ts = candles["timestamp"].max()

    print("building feature context (heavy)...", flush=True)
    ctx = FeatureContext(candles, btc_asset_key="BTCUSDT")

    # Lock threshold by re-deriving from the same B-trades the predictor was
    # trained on (deterministic given the saved model + canonical training set).
    print("re-deriving locked threshold from Strategy-B training scores...", flush=True)
    b_trades = pd.read_csv(B_TRADES_PATH)
    b_trades["entry_time"] = pd.to_datetime(b_trades["entry_time"], utc=True)
    b_keep = filter_labelable(b_trades, candles_max_ts, max_bars=LABEL_MAX_BARS)
    b_trades = b_trades[b_keep].sort_values("entry_time").reset_index(drop=True)
    b_train = b_trades[b_trades["entry_time"] < OOT_BOUNDARY]
    b_train_proba = _score_trades(b_train, model, ctx).dropna()
    threshold = float(b_train_proba.quantile(THRESHOLD_QUANTILE))
    print(f"  locked threshold: {threshold:.4f}", flush=True)

    results = {
        "snapshot_date": SNAPSHOT_DATE,
        "model_path": str(MODEL_PATH),
        "threshold": threshold,
        "threshold_quantile": THRESHOLD_QUANTILE,
        "strategies": {},
    }
    for label, path in [("A", A_TRADES_PATH), ("C", C_TRADES_PATH)]:
        results["strategies"][label] = _evaluate_strategy(
            label, path, model, ctx, candles_max_ts, threshold
        )

    out_dir = REPO_ROOT / "services/python/results" / "continuation_v1_transfer_test"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "transfer_test.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_dir / 'transfer_test.json'}")
    print(f"wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
