"""3-year v8 BALANCED classifier evaluation — Filter-mode vs Sizing-mode.

Trains HistGradientBoostingClassifier on 2023Q2-2025Q4, evaluates on 2026Q1
holdout. Compares THREE modes side-by-side per detector:

  A. BASELINE     — take every signal at size 1.0
  B. FILTER-MODE  — skip signals below locked train-PF-optimal threshold
  C. SIZING-MODE  — take every signal, size = rank-based mapping (mean = 1.0)

Sizing-mode is capital-neutral (mean(size) = 1.0 over the train distribution),
so it's an apples-to-apples comparison vs baseline. By construction, if the
classifier has any positive AUC, sizing-mode produces a higher weighted return
than baseline (Cov(pnl, size) > 0).

Per-quarter PnL breakdown reports actual cumulative PnL — not just PF. This
addresses the realistic concern that a high-PF/low-trade-count strategy may
underperform a low-PF/high-trade-count strategy in absolute dollars.

Usage from services/python/:
    .venv/bin/python scripts/train_v8_classifier_3y_sizing.py
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

TRAINING_PARQUET = (
    Path(__file__).resolve().parents[1]
    / "data" / "training" / "v8_trades_with_features_2026-04-01.parquet"
)
MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "v8_classifier_3y"
HOLD_OUT_START = pd.Timestamp("2026-01-01", tz="UTC")  # 2026Q1+ is hold-out
RANDOM_SEED = 42
MIN_TRADES_PER_DETECTOR = 100

FEATURES = [
    "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
    "close_to_high50_atr", "close_to_low50_atr",
    "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
    "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
    "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
    "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
    "breadth_up", "breadth_down", "signals_same_15m_same_detector",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]


def make_classifier() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=0.1,
        random_state=RANDOM_SEED,
    )


def pf(pnls: np.ndarray) -> float:
    if len(pnls) == 0:
        return 0.0
    w = pnls[pnls > 0].sum(); l = -pnls[pnls <= 0].sum()
    return float(w / l) if l > 0 else float("inf")


def compounded_eq(pnls: np.ndarray, sizes: np.ndarray | None = None,
                  base_size: float = 0.01, start: float = 100) -> float:
    """Sequential compounding. base_size = capital fraction per nominal trade.
    sizes: optional per-trade size multiplier (mean ~ 1.0 = capital-neutral)."""
    if sizes is None:
        sizes = np.ones_like(pnls)
    eq = start
    for p, s in zip(pnls, sizes):
        eq *= 1 + base_size * s * p
    return eq


def max_drawdown(pnls: np.ndarray, sizes: np.ndarray | None = None,
                 base_size: float = 0.01, start: float = 100) -> float:
    if sizes is None:
        sizes = np.ones_like(pnls)
    eq = start; peak = start; mdd = 0.0
    for p, s in zip(pnls, sizes):
        eq *= 1 + base_size * s * p
        peak = max(peak, eq)
        mdd = min(mdd, eq / peak - 1)
    return mdd * 100


def rank_size_mapping(train_scores: np.ndarray, low: float = 0.5,
                      high: float = 1.5) -> tuple[np.ndarray, np.ndarray]:
    """Build a rank-based score → size lookup from train scores.
    Returns (sorted_train_scores, mapped_sizes_per_train) so we can use
    np.searchsorted for fast holdout lookup.
    """
    sorted_scores = np.sort(train_scores)
    n = len(sorted_scores)
    # Each sorted score's rank → size
    ranks = np.linspace(0, 1, n)
    sizes = low + (high - low) * ranks
    return sorted_scores, sizes


def apply_size_mapping(scores: np.ndarray, sorted_train: np.ndarray,
                       train_sizes: np.ndarray) -> np.ndarray:
    """For each score, find its rank in the train distribution and return the size."""
    # searchsorted gives rank position; map via train_sizes interpolation
    n_train = len(sorted_train)
    pos = np.searchsorted(sorted_train, scores, side="right")
    pos_clipped = np.clip(pos, 1, n_train) - 1
    return train_sizes[pos_clipped]


def evaluate_modes(d_df: pd.DataFrame, model: HistGradientBoostingClassifier,
                   det: str) -> dict:
    """Evaluate baseline / filter / sizing modes on this detector's full data."""
    d = d_df.sort_values("entry_time").reset_index(drop=True)
    tr = d[d.entry_time < HOLD_OUT_START].reset_index(drop=True)
    ho = d[d.entry_time >= HOLD_OUT_START].reset_index(drop=True)
    if len(tr) < 30 or len(ho) < 5:
        return {}

    X_tr = tr[FEATURES].values; y_tr = tr["label"].values; pnl_tr = tr["pnl_pct"].values
    X_ho = ho[FEATURES].values; y_ho = ho["label"].values; pnl_ho = ho["pnl_pct"].values

    train_probs = model.predict_proba(X_tr)[:, 1]
    holdout_probs = model.predict_proba(X_ho)[:, 1]

    # Filter mode: pick threshold maximizing TRAIN compounded equity
    best_thr_eq, best_eq = 0.0, compounded_eq(pnl_tr)
    for thr in np.arange(0.30, 0.71, 0.025):
        m = train_probs >= thr
        if m.sum() < 50:
            continue
        eq = compounded_eq(pnl_tr[m])
        if eq > best_eq:
            best_eq, best_thr_eq = eq, float(thr)

    # Sizing-mode: lock rank→size mapping on TRAIN scores
    sorted_tr, tr_sizes = rank_size_mapping(train_probs)

    # Now compute all metrics
    def block(pnls, sizes=None, label=""):
        return {
            "label": label,
            "n": int(len(pnls)),
            "pf": round(pf(pnls), 3),
            "wr_pct": round(100 * (pnls > 0).mean() if len(pnls) else 0.0, 1),
            "total_pnl_pct": round(float(pnls.sum() * 100), 1),
            "weighted_return_pct": round(
                float((pnls * (sizes if sizes is not None else 1.0)).sum() * 100), 1
            ),
            "compounded_eq": round(compounded_eq(pnls, sizes), 2),
            "max_dd_pct": round(max_drawdown(pnls, sizes), 2),
        }

    # Train-set sizes from rank mapping (using train probs themselves)
    tr_sizes_for_train = apply_size_mapping(train_probs, sorted_tr, tr_sizes)
    ho_sizes = apply_size_mapping(holdout_probs, sorted_tr, tr_sizes)

    train_modes = {
        "baseline": block(pnl_tr, label="train baseline"),
        "filter": block(
            pnl_tr[train_probs >= best_thr_eq],
            label=f"train filter @ thr={best_thr_eq:.3f}",
        ),
        "sizing": block(pnl_tr, sizes=tr_sizes_for_train, label="train sizing"),
    }
    holdout_modes = {
        "baseline": block(pnl_ho, label="holdout baseline"),
        "filter": block(
            pnl_ho[holdout_probs >= best_thr_eq],
            label=f"holdout filter @ thr={best_thr_eq:.3f}",
        ),
        "sizing": block(pnl_ho, sizes=ho_sizes, label="holdout sizing"),
    }

    auc_tr = roc_auc_score(y_tr, train_probs) if len(set(y_tr)) >= 2 else None
    auc_ho = roc_auc_score(y_ho, holdout_probs) if len(set(y_ho)) >= 2 else None

    return {
        "detector": det,
        "auc_train": float(auc_tr) if auc_tr is not None else None,
        "auc_holdout": float(auc_ho) if auc_ho is not None else None,
        "filter_threshold_locked": float(best_thr_eq),
        "train": train_modes,
        "holdout": holdout_modes,
        "n_train": int(len(tr)),
        "n_holdout": int(len(ho)),
        "ho_sizes_mean": round(float(ho_sizes.mean()), 3),
        "ho_sizes_std": round(float(ho_sizes.std()), 3),
        "tr_sizes_mean_check": round(float(tr_sizes_for_train.mean()), 3),
    }


def per_quarter_breakdown(d_df: pd.DataFrame, model: HistGradientBoostingClassifier) -> pd.DataFrame:
    """Quarter-by-quarter PnL for each mode."""
    d = d_df.sort_values("entry_time").reset_index(drop=True).copy()
    d["quarter"] = d["entry_time"].dt.to_period("Q").astype(str)
    tr = d[d.entry_time < HOLD_OUT_START]
    if len(tr) < 30:
        return pd.DataFrame()

    train_probs = model.predict_proba(tr[FEATURES].values)[:, 1]
    sorted_tr, tr_sizes_arr = rank_size_mapping(train_probs)

    # Find filter threshold (same as evaluate_modes)
    pnl_tr = tr["pnl_pct"].values
    best_thr_eq, best_eq = 0.0, compounded_eq(pnl_tr)
    for thr in np.arange(0.30, 0.71, 0.025):
        m = train_probs >= thr
        if m.sum() < 50:
            continue
        eq = compounded_eq(pnl_tr[m])
        if eq > best_eq:
            best_eq, best_thr_eq = eq, float(thr)

    rows = []
    for q in sorted(d["quarter"].unique()):
        qd = d[d["quarter"] == q]
        if len(qd) == 0:
            continue
        probs = model.predict_proba(qd[FEATURES].values)[:, 1]
        sizes = apply_size_mapping(probs, sorted_tr, tr_sizes_arr)
        pnls = qd["pnl_pct"].values
        # Three modes
        base_pnl = pnls.sum() * 100
        filt_mask = probs >= best_thr_eq
        filt_pnl = pnls[filt_mask].sum() * 100
        sz_pnl = (pnls * sizes).sum() * 100
        rows.append({
            "quarter": q,
            "n": len(qd),
            "base_pnl%": round(base_pnl, 1),
            "filter_n": int(filt_mask.sum()),
            "filter_pnl%": round(filt_pnl, 1),
            "filter_kept_pct": round(100 * filt_mask.sum() / len(qd), 1),
            "sizing_pnl%": round(sz_pnl, 1),
            "sz_minus_base": round(sz_pnl - base_pnl, 1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    assert TRAINING_PARQUET.exists(), f"missing parquet: {TRAINING_PARQUET}"
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(TRAINING_PARQUET)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)

    print(f"Total trades: {len(df):,}")
    print(f"Time range:   {df.entry_time.min()} → {df.entry_time.max()}")
    print(f"Holdout cutoff: {HOLD_OUT_START}  (everything ≥ this date is holdout)")
    print()
    print("Per-detector counts:")
    print(df.groupby("detector").size().to_string())

    results: dict = {}
    print()
    print("=" * 116)
    print("WALK-FORWARD CV ON TRAIN PERIOD (per detector)")
    print("=" * 116)
    for det in sorted(df["detector"].unique()):
        d = df[df.detector == det]
        if len(d) < MIN_TRADES_PER_DETECTOR:
            print(f"\n=== {det}: SKIP — only {len(d)} trades ===")
            continue

        d = d.sort_values("entry_time").reset_index(drop=True)
        train_only = d[d.entry_time < HOLD_OUT_START].reset_index(drop=True)
        if len(train_only) < 200:
            print(f"\n=== {det}: SKIP — only {len(train_only)} train trades ===")
            continue

        print(f"\n=== {det} (total={len(d):,}, train={len(train_only):,}) ===")
        X = train_only[FEATURES].values
        y = train_only["label"].values
        pnls = train_only["pnl_pct"].values
        cv = TimeSeriesSplit(n_splits=5)
        cv_rows = []
        for fold_i, (tr, te) in enumerate(cv.split(X)):
            m = make_classifier()
            m.fit(X[tr], y[tr])
            probs = m.predict_proba(X[te])[:, 1]
            auc = float(roc_auc_score(y[te], probs)) if len(set(y[te])) >= 2 else float("nan")
            # Sizing-mode CV PnL
            sorted_tr_fold, tr_sizes_fold = rank_size_mapping(m.predict_proba(X[tr])[:, 1])
            te_sizes = apply_size_mapping(probs, sorted_tr_fold, tr_sizes_fold)
            base_pnl_pct = pnls[te].sum() * 100
            sized_pnl_pct = (pnls[te] * te_sizes).sum() * 100
            cv_rows.append({
                "fold": fold_i,
                "n_train": len(tr), "n_test": len(te),
                "auc": round(auc, 3),
                "base_pnl_pct": round(base_pnl_pct, 1),
                "sized_pnl_pct": round(sized_pnl_pct, 1),
                "lift_pct": round(sized_pnl_pct - base_pnl_pct, 1),
            })
        print(pd.DataFrame(cv_rows).to_string(index=False))

    # Train final + evaluate
    print()
    print("=" * 116)
    print("FINAL TRAIN + LOCKED HOLDOUT EVAL — three modes side by side")
    print("=" * 116)
    for det in sorted(df["detector"].unique()):
        d = df[df.detector == det]
        if len(d) < MIN_TRADES_PER_DETECTOR:
            continue
        d = d.sort_values("entry_time").reset_index(drop=True)
        train_only = d[d.entry_time < HOLD_OUT_START]
        if len(train_only) < 200:
            continue

        # Final fit on full train
        model = make_classifier()
        model.fit(train_only[FEATURES].values, train_only["label"].values)
        # Save
        joblib.dump(model, MODELS_DIR / f"{det}_histgbm_3y_v1.joblib")

        res = evaluate_modes(d, model, det)
        results[det] = res

        print(f"\n=== {det} (n_train={res['n_train']:,}, n_holdout={res['n_holdout']:,}) ===")
        print(f"  AUC train:   {res['auc_train']:.3f}")
        print(f"  AUC holdout: {res['auc_holdout']:.3f}")
        print(f"  Filter threshold (locked on train compounded equity): {res['filter_threshold_locked']:.3f}")
        print(f"  Holdout size distribution: mean={res['ho_sizes_mean']:.3f}, std={res['ho_sizes_std']:.3f}")
        print()
        print("  HOLDOUT — three modes (apples-to-apples):")
        print(f"    {'mode':10s} {'n':>7s} {'pf':>6s} {'wr%':>5s} "
              f"{'sum_pnl%':>10s} {'wt_return%':>11s} {'comp_eq':>9s} {'maxDD%':>7s}")
        for mode in ["baseline", "filter", "sizing"]:
            b = res["holdout"][mode]
            print(f"    {mode:10s} {b['n']:>7d} {b['pf']:>6.2f} {b['wr_pct']:>4.1f}% "
                  f"{b['total_pnl_pct']:>+9.1f}% {b['weighted_return_pct']:>+10.1f}% "
                  f"{b['compounded_eq']:>9.2f} {b['max_dd_pct']:>+6.2f}%")
        # Compute the locked train-score quantile lookup that live classifier
        # needs to reproduce the rank→size mapping deterministically.
        train_probs_full = model.predict_proba(train_only[FEATURES].values)[:, 1]
        n_q = 200
        q_idx = np.linspace(0, len(train_probs_full) - 1, n_q).astype(int)
        train_score_quantiles = np.sort(train_probs_full)[q_idx].tolist()

        # Save meta
        meta = {
            **res,
            "feature_names": FEATURES,
            "snapshot_date": "2026-04-01",
            "trained_at_utc": datetime.now(UTC).isoformat(),
            "model_class": "sklearn.ensemble.HistGradientBoostingClassifier",
            # Sizing-mode lookup
            "size_low": 0.5,
            "size_high": 1.5,
            "train_score_quantiles": train_score_quantiles,  # 200 evenly-spaced
        }
        with open(MODELS_DIR / f"{det}_histgbm_3y_v1_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)

    # Per-quarter breakdown for the main detector
    print()
    print("=" * 116)
    print("PER-QUARTER PnL BREAKDOWN")
    print("=" * 116)
    for det in sorted(df["detector"].unique()):
        d = df[df.detector == det]
        if len(d) < MIN_TRADES_PER_DETECTOR:
            continue
        train_only = d[d.entry_time < HOLD_OUT_START]
        if len(train_only) < 200:
            continue
        model = joblib.load(MODELS_DIR / f"{det}_histgbm_3y_v1.joblib")
        pq = per_quarter_breakdown(d, model)
        print(f"\n=== {det} ===")
        print(pq.to_string(index=False))

    print()
    print("=" * 116)
    print("PORTFOLIO SUMMARY (sum across all detectors above ship-floor)")
    print("=" * 116)
    # Sum holdout PnL across detectors
    rows = []
    for det, res in results.items():
        ho = res["holdout"]
        rows.append({
            "detector": det,
            "auc_holdout": res["auc_holdout"],
            "n_holdout": ho["baseline"]["n"],
            "base_pnl%": ho["baseline"]["total_pnl_pct"],
            "base_eq": ho["baseline"]["compounded_eq"],
            "filter_pnl%": ho["filter"]["total_pnl_pct"],
            "filter_eq": ho["filter"]["compounded_eq"],
            "sizing_pnl%": ho["sizing"]["weighted_return_pct"],
            "sizing_eq": ho["sizing"]["compounded_eq"],
            "sizing_lift_pnl": round(ho["sizing"]["weighted_return_pct"] - ho["baseline"]["total_pnl_pct"], 1),
            "sizing_lift_eq": round(ho["sizing"]["compounded_eq"] - ho["baseline"]["compounded_eq"], 2),
        })
    summary = pd.DataFrame(rows)
    print(summary.to_string(index=False))

    print()
    print(f"=== DONE — models saved to {MODELS_DIR} ===")


if __name__ == "__main__":
    main()
