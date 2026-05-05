"""v_new_1.6 — Cohort momentum feature (data-driven narrative).

For each (symbol, timestamp), computes:
- cohort_top10_momentum_7d: mean 7d close-to-close return of the 10 coins
  most correlated with this symbol over the prior 30 days
- cohort_top10_corr_mean: diagnostic — average pairwise correlation of the
  cohort (how tight is the tribe?)

LEAKAGE GUARANTEE:
- Cohort identification: rolling 30-day correlation matrix on 4h returns,
  computed at WEEKLY anchor timestamps using data strictly < anchor.
- Cohort returns measurement: 7-day window ending at t (strictly past).

Outputs:
    services/python/data/v_new_1_v2/cohort_momentum.parquet
    columns: symbol, timestamp, cohort_top10_momentum_7d, cohort_top10_corr_mean

USAGE:
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_6_cohort_momentum.py
"""
from __future__ import annotations

import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                    str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))

CORR_LOOKBACK_BARS = 180     # 30 days × 6 bars/day at 4h
RETURN_WINDOW_BARS = 42      # 7 days
TOP_N_COHORT = 10
MIN_OBS_FOR_CORR = 100       # require at least 100 valid bars in the window


def _ts(label: str = "") -> str:
    now = time.strftime("%H:%M:%S")
    return f"[{now}]{(' ' + label) if label else ''}"


def _build_returns_matrix(features_path: pathlib.Path) -> pd.DataFrame:
    """Returns a wide DF: index=timestamp, columns=symbols, values=4h log returns.
    Uses the 'close' column from features_full (or from labels if features lacks close)."""
    print(f"{_ts()} Loading features for close prices...")
    feats = pd.read_parquet(features_path, columns=["symbol", "timestamp", "close"])
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    print(f"{_ts()} Pivoting close prices to wide form...")
    wide_close = feats.pivot_table(index="timestamp", columns="symbol", values="close",
                                    aggfunc="first")
    wide_close = wide_close.sort_index()
    print(f"  wide_close shape: {wide_close.shape}")

    print(f"{_ts()} Computing 4h log returns...")
    returns = np.log(wide_close / wide_close.shift(1))
    return returns


def _weekly_anchors(returns: pd.DataFrame) -> list[pd.Timestamp]:
    """Pick weekly anchor timestamps inside the returns index."""
    if returns.empty:
        return []
    start = returns.index[0].normalize()
    end = returns.index[-1].normalize()
    anchors = pd.date_range(start, end, freq="W-MON", tz="UTC")
    # Snap each anchor to the nearest available 4h bar at or before that day
    snapped = []
    for a in anchors:
        avail = returns.index[returns.index <= a]
        if len(avail) > 0:
            snapped.append(avail[-1])
    return list(dict.fromkeys(snapped))  # dedupe preserving order


def _compute_anchor_cohorts(
    returns: pd.DataFrame, anchor_ts: pd.Timestamp,
) -> tuple[dict[str, list[str]], dict[str, float]]:
    """At anchor_ts, compute correlation matrix on the prior 30-day window
    and return per-symbol top-10 cohort + average cohort correlation."""
    window_end_idx = returns.index.get_loc(anchor_ts)
    window_start_idx = max(0, window_end_idx - CORR_LOOKBACK_BARS)
    window = returns.iloc[window_start_idx:window_end_idx]
    if len(window) < MIN_OBS_FOR_CORR:
        return {}, {}

    # Drop columns with too many NaN in the window
    valid_cols = window.columns[window.notna().sum() >= MIN_OBS_FOR_CORR]
    if len(valid_cols) < TOP_N_COHORT + 1:
        return {}, {}
    sub = window[valid_cols]
    corr = sub.corr(min_periods=MIN_OBS_FOR_CORR)

    cohorts: dict[str, list[str]] = {}
    cohort_corr_means: dict[str, float] = {}
    for s in valid_cols:
        if s not in corr.columns:
            continue
        s_corr = corr[s].drop(labels=[s], errors="ignore").dropna()
        if len(s_corr) < TOP_N_COHORT:
            continue
        top = s_corr.nlargest(TOP_N_COHORT)
        cohorts[s] = list(top.index)
        cohort_corr_means[s] = float(top.mean())
    return cohorts, cohort_corr_means


def _compute_cohort_features_for_window(
    returns: pd.DataFrame,
    cohorts: dict[str, list[str]],
    cohort_corr: dict[str, float],
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> pd.DataFrame:
    """For every timestamp in [window_start, window_end), compute each symbol's
    cohort_top10_momentum_7d using the cohort defined at the anchor (cohorts dict).
    Returns long-form DF: symbol, timestamp, cohort_top10_momentum_7d,
    cohort_top10_corr_mean."""
    mask = (returns.index >= window_start) & (returns.index < window_end)
    ts_in_window = returns.index[mask]
    if len(ts_in_window) == 0:
        return pd.DataFrame(columns=[
            "symbol", "timestamp",
            "cohort_top10_momentum_7d", "cohort_top10_corr_mean",
        ])

    # 7-day rolling cumulative return per symbol (close-to-close = sum of log returns)
    rolling_7d = returns.rolling(RETURN_WINDOW_BARS, min_periods=10).sum()
    # Convert from log to simple return space for averaging:
    rolling_7d_simple = np.exp(rolling_7d) - 1.0

    rows = []
    for ts in ts_in_window:
        if ts not in rolling_7d_simple.index:
            continue
        snap = rolling_7d_simple.loc[ts]
        for sym, cohort in cohorts.items():
            cohort_returns = snap.reindex(cohort).dropna()
            if len(cohort_returns) < TOP_N_COHORT // 2:
                continue
            rows.append({
                "symbol": sym,
                "timestamp": ts,
                "cohort_top10_momentum_7d": float(cohort_returns.mean()),
                "cohort_top10_corr_mean": cohort_corr.get(sym, np.nan),
            })
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_1.6 cohort momentum — start")
    print(f"  DATA: {DATA}")
    features_path = DATA / "features_full.parquet"
    if not features_path.exists():
        print(f"  ERROR: {features_path} missing")
        sys.exit(1)

    returns = _build_returns_matrix(features_path)
    print(f"  returns: {returns.shape}  "
          f"range {returns.index.min().date()} → {returns.index.max().date()}")

    anchors = _weekly_anchors(returns)
    print(f"  weekly anchors: {len(anchors)}")

    all_rows: list[pd.DataFrame] = []
    for i, anchor in enumerate(anchors):
        cohorts, cohort_corr = _compute_anchor_cohorts(returns, anchor)
        if not cohorts:
            continue
        # Apply this anchor's cohorts to all timestamps in [anchor, next_anchor)
        next_anchor = anchors[i + 1] if i + 1 < len(anchors) else returns.index[-1] + pd.Timedelta("8h")
        df = _compute_cohort_features_for_window(returns, cohorts, cohort_corr, anchor, next_anchor)
        if not df.empty:
            all_rows.append(df)
        if (i + 1) % 26 == 0:
            elapsed = time.time() - t0
            print(f"  {_ts()} anchor {i+1}/{len(anchors)} ({anchor.date()})  "
                  f"rows so far: {sum(len(r) for r in all_rows):,}  "
                  f"elapsed: {elapsed:.1f}s")

    if not all_rows:
        print(f"{_ts()} ERROR: no cohort rows computed")
        sys.exit(1)

    out = pd.concat(all_rows, ignore_index=True)
    out = out.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    out_path = DATA / "cohort_momentum.parquet"
    out.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\n{_ts()} DONE in {elapsed:.1f}s")
    print(f"  wrote {out_path} ({len(out):,} rows, "
          f"{out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"  feature stats:")
    print(f"    cohort_top10_momentum_7d:  "
          f"mean={out['cohort_top10_momentum_7d'].mean():.4f}  "
          f"std={out['cohort_top10_momentum_7d'].std():.4f}  "
          f"NaN%={out['cohort_top10_momentum_7d'].isna().mean()*100:.1f}")
    print(f"    cohort_top10_corr_mean:    "
          f"mean={out['cohort_top10_corr_mean'].mean():.4f}  "
          f"std={out['cohort_top10_corr_mean'].std():.4f}")


if __name__ == "__main__":
    main()
