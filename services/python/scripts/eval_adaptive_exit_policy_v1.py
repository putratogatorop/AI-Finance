"""Shared helpers for adaptive-exit-policy agents.

Loads the per-trade × per-cell PnL matrix produced by
`build_macd_pullback_long_3y_per_trade_matrix_v1.py` and offers utilities to
evaluate a routing "policy" (a per-trade choice of which cell to use).

A policy is just a function `policy(row) -> cell_col_name`. Helpers:
  - load_matrix(path) -> (df, cells, hold_mask)
  - eval_policy(df, choose_cell_fn, hold_mask) -> metrics dict + per-trade pnl
  - eval_threshold_rule(df, thresholds, cells) -> metrics dict
  - eval_bucket_policy(df, buckets, cells) -> metrics dict
  - eval_ml_router(df, predicted_cells) -> metrics dict
  - cell_usage_histogram(picks) -> dict

Metrics: PF, WR, total_pnl_pct, MDD%, all on full-period AND hold-out subset.

Run as a script for a quick smoke test:
    .venv/bin/python scripts/eval_adaptive_exit_policy_v1.py <matrix.parquet>
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def load_matrix(parquet_path: str | Path) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Return (df, cell_cols, hold_mask). df is sorted by entry_time."""
    df = pd.read_parquet(parquet_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df = df.sort_values("entry_time").reset_index(drop=True)
    cell_cols = [c for c in df.columns if c.startswith("pnl_")]
    hold_mask = df["is_hold"].astype(bool)
    return df, cell_cols, hold_mask


def _pf(p: pd.Series) -> float:
    p = p.dropna()
    w = float(p[p > 0].sum())
    l = float(-p[p < 0].sum())
    return float("inf") if l == 0 else w / l


def _max_drawdown_pct(pnl: pd.Series) -> float:
    p = pnl.dropna()
    if len(p) == 0:
        return 0.0
    cum = p.cumsum() * 100.0
    return float((cum.cummax() - cum).max())


def _metrics(pnl: pd.Series) -> dict:
    p = pnl.dropna()
    return {
        "n": int(len(p)),
        "pf": round(_pf(p), 4),
        "wr_pct": round(float((p > 0).mean() * 100), 2) if len(p) else 0.0,
        "total_pnl_pct": round(float(p.sum() * 100), 2),
        "mdd_pct": round(_max_drawdown_pct(p), 2),
    }


def cell_usage_histogram(picks: pd.Series) -> dict:
    counts = Counter(picks.dropna())
    total = sum(counts.values()) or 1
    return {cell: {"n": int(c), "pct": round(100 * c / total, 2)} for cell, c in counts.most_common()}


def eval_policy(df: pd.DataFrame, picks: pd.Series, hold_mask: pd.Series) -> dict:
    """Evaluate a policy where `picks[i]` is the cell column to use for trade i.

    Returns a metrics dict with full-period + hold-out subsets + cell histogram.
    """
    pnl = pd.Series(np.nan, index=df.index, name="pnl")
    for i, cell in enumerate(picks):
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            continue
        if cell not in df.columns:
            continue
        v = df.iloc[i][cell]
        if pd.notna(v):
            pnl.iloc[i] = v
    pnl_train = pnl[~hold_mask]
    pnl_hold = pnl[hold_mask]
    return {
        "all": _metrics(pnl),
        "train": _metrics(pnl_train),
        "hold": _metrics(pnl_hold),
        "cell_usage": cell_usage_histogram(picks),
        "_pnl_series": pnl,  # keep for caller's month-by-month aggregation
    }


def eval_threshold_rule(df: pd.DataFrame, thresholds: list[float], cells: list[str],
                         hold_mask: pd.Series) -> dict:
    """Coarse rule: cls_score < t1 -> cells[0], t1<=score<t2 -> cells[1], ..."""
    if len(cells) != len(thresholds) + 1:
        raise ValueError("len(cells) must equal len(thresholds) + 1")
    picks = pd.Series([None] * len(df), index=df.index)
    score = df["cls_score"].fillna(-1.0)  # missing score -> first cell (defensive)
    bins = np.array([-np.inf] + sorted(thresholds) + [np.inf])
    idx = np.searchsorted(bins, score.to_numpy(), side="right") - 1
    idx = np.clip(idx, 0, len(cells) - 1)
    picks = pd.Series([cells[i] for i in idx], index=df.index)
    return eval_policy(df, picks, hold_mask)


def eval_bucket_policy(df: pd.DataFrame, buckets: list[tuple[float, float]],
                       cells: list[str], hold_mask: pd.Series) -> dict:
    """Bucket policy: list of (lo, hi) score brackets paired with a cell each."""
    if len(buckets) != len(cells):
        raise ValueError("buckets and cells must have same length")
    score = df["cls_score"]
    picks = pd.Series([None] * len(df), index=df.index, dtype="object")
    for (lo, hi), cell in zip(buckets, cells, strict=True):
        mask = (score >= lo) & (score < hi)
        picks[mask] = cell
    # Assign last bucket cell to any score == 1.0 (closed boundary)
    if buckets:
        lo_max, hi_max = buckets[-1]
        picks[score == hi_max] = cells[-1]
    return eval_policy(df, picks, hold_mask)


def eval_predicted_cells(df: pd.DataFrame, predicted_cells: list[str],
                         hold_mask: pd.Series) -> dict:
    """ML-router policy: a per-trade list of cell column names."""
    if len(predicted_cells) != len(df):
        raise ValueError("predicted_cells length must match df")
    picks = pd.Series(predicted_cells, index=df.index)
    return eval_policy(df, picks, hold_mask)


def best_single_cell(df: pd.DataFrame, cell_cols: list[str], hold_mask: pd.Series,
                     by: str = "pf") -> tuple[str, dict]:
    """Find the single cell that scores best under the given metric on TRAIN."""
    train = df[~hold_mask]
    best = None
    best_metric = None
    for c in cell_cols:
        s = train[c].dropna()
        if len(s) == 0:
            continue
        m = _metrics(s)
        v = m.get(by, 0.0)
        if best_metric is None or v > best_metric:
            best, best_metric = c, v
    if best is None:
        raise RuntimeError("no valid cells found")
    picks = pd.Series([best] * len(df), index=df.index)
    return best, eval_policy(df, picks, hold_mask)


def month_by_month(pnl: pd.Series, df: pd.DataFrame) -> pd.DataFrame:
    """Group a per-trade pnl series by entry_time month, return DataFrame
    with month + total_pnl_pct + n_trades per month."""
    s = pnl.copy()
    s.index = df["entry_time"].values
    g = pd.DataFrame({
        "month": pd.to_datetime(s.index).strftime("%Y-%m"),
        "pnl_pct": s.values,
    })
    by = g.groupby("month")["pnl_pct"].agg([("n_trades", "count"),
                                            ("total_pnl_pct", lambda x: float(x.sum() * 100)),
                                            ("avg_pnl_pct", lambda x: float(x.mean() * 100) if len(x) else 0.0)])
    return by.reset_index()


# --- Smoke test --------------------------------------------------------

def _smoke(matrix_path: str) -> None:
    print(f"[smoke] loading {matrix_path}")
    df, cells, hold_mask = load_matrix(matrix_path)
    print(f"[smoke] entries={len(df)} cells={len(cells)} hold={int(hold_mask.sum())}")
    print(f"[smoke] cls_score range: [{df['cls_score'].min():.3f}, {df['cls_score'].max():.3f}]")

    # Best single cell
    best, m = best_single_cell(df, cells, hold_mask, by="pf")
    print(f"\n[smoke] best single cell by train PF: {best}")
    print(f"  all:   PF={m['all']['pf']}  total_pnl%={m['all']['total_pnl_pct']}  MDD%={m['all']['mdd_pct']}")
    print(f"  train: PF={m['train']['pf']}  total_pnl%={m['train']['total_pnl_pct']}  MDD%={m['train']['mdd_pct']}")
    print(f"  hold:  PF={m['hold']['pf']}  total_pnl%={m['hold']['total_pnl_pct']}  MDD%={m['hold']['mdd_pct']}")

    # User example: cls_score < 0.1 -> 15m SL=2 TP=4 (2:1); cls_score >= 0.5 -> 4h SL=1 TP=4 (4:1)
    # We don't have SL=1; use SL=1.5. So: < 0.1 -> 15m_2_4 (2:1), 0.1-0.5 -> 15m_2_6 (3:1), >= 0.5 -> 4h_1.5_6 (4:1)
    user_rule = eval_threshold_rule(
        df, thresholds=[0.1, 0.5],
        cells=["pnl_15m_2_4", "pnl_15m_2_6", "pnl_4h_1.5_6"],
        hold_mask=hold_mask,
    )
    print(f"\n[smoke] user-style rule (<0.1 -> 15m 2:1, 0.1-0.5 -> 15m 3:1, >=0.5 -> 4h 4:1):")
    print(f"  all:   PF={user_rule['all']['pf']}  total_pnl%={user_rule['all']['total_pnl_pct']}  MDD%={user_rule['all']['mdd_pct']}")
    print(f"  train: PF={user_rule['train']['pf']}  total_pnl%={user_rule['train']['total_pnl_pct']}  MDD%={user_rule['train']['mdd_pct']}")
    print(f"  hold:  PF={user_rule['hold']['pf']}  total_pnl%={user_rule['hold']['total_pnl_pct']}  MDD%={user_rule['hold']['mdd_pct']}")
    print(f"  cell usage: {json.dumps(user_rule['cell_usage'], indent=2)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: eval_adaptive_exit_policy_v1.py <matrix.parquet>")
        sys.exit(2)
    _smoke(sys.argv[1])
