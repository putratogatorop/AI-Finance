"""v_new_1.5 — augment features_full with v_new_1 sequence features.

Reads OOF (train_fit), CAL (cal_fit), and OOS (4 holdout years) score parquets
produced by v_new_1_phase3_train.py, builds a per-(symbol,timestamp) score
timeline, derives 12 lagged + rolling features, and merges them into
features_full.parquet → features_full_v1_5.parquet.

LEAKAGE GUARANTEE:
    - train_fit rows use OOF preds (each row's score came from a fold where
      that row was held out).
    - cal_fit rows use raw final-model preds (final model was trained on
      train_fit only — it never saw cal_fit).
    - OOS year rows use the saved oos_scored parquets (already non-leaky).

NaN policy:
    - Lag features are NaN if the prior timestamp had no score (early-series
      head, embargoed rows, gaps between OOS years and train period).
    - LightGBM handles NaN natively — no fill applied.

USAGE:
    services/python/.venv/bin/python3 \\
        services/python/scripts/v_new_1_5_augment_features.py
"""

from __future__ import annotations

import pathlib
import sys
import time

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "services" / "python"
sys.path.insert(0, str(PYTHON_ROOT))

DATA_DIR = PYTHON_ROOT / "data" / "v_new_1"

LAG_BARS = [1, 2, 6]                 # 4h, 8h, 24h ago
ROLL_WINDOW_BARS = 18                # 18 × 4h = 3 days
CHANGE_LAG_BARS = 6                  # 24h delta

NEW_FEATURE_COLS = [
    "score_long_t1", "score_long_t2", "score_long_t6",
    "score_long_roll_mean_3d", "score_long_roll_std_3d",
    "score_long_change_1d",
    "score_short_t1", "score_short_t2", "score_short_t6",
    "score_short_roll_mean_3d", "score_short_roll_std_3d",
    "score_short_change_1d",
]


def _ts(label: str = "") -> str:
    now = time.strftime("%H:%M:%S")
    return f"[{now}] {label}".rstrip()


def _load_score_timeline(direction: str) -> pd.DataFrame:
    """Return a long-form (symbol, timestamp, score) frame stitched from
    OOF + CAL + OOS parquets for one direction."""
    parts: list[pd.DataFrame] = []

    # OOF (train_fit, may have NaN rows from first slice + embargo)
    oof_path = DATA_DIR / f"oof_scored_{direction}.parquet"
    if oof_path.exists():
        oof = pd.read_parquet(oof_path, columns=["symbol", "timestamp", "score_raw"])
        oof = oof.rename(columns={"score_raw": "score"})
        oof["source"] = "oof"
        parts.append(oof)
        print(f"  {direction} OOF rows: {len(oof):,} "
              f"({(~oof['score'].isna()).sum():,} non-NaN)")
    else:
        raise FileNotFoundError(f"Missing OOF parquet: {oof_path}")

    # CAL (cal_fit, all valid)
    cal_path = DATA_DIR / f"cal_scored_{direction}.parquet"
    if cal_path.exists():
        cal = pd.read_parquet(cal_path, columns=["symbol", "timestamp", "score_raw"])
        cal = cal.rename(columns={"score_raw": "score"})
        cal["source"] = "cal"
        parts.append(cal)
        print(f"  {direction} CAL rows: {len(cal):,}")
    else:
        raise FileNotFoundError(f"Missing CAL parquet: {cal_path}")

    # OOS (4 holdout years, calibrated — but for tree-meta features, monotonic
    # rescaling is irrelevant. Use the 'score' col directly.)
    oos_path = DATA_DIR / f"oos_scored_{direction}.parquet"
    if oos_path.exists():
        oos = pd.read_parquet(oos_path, columns=["symbol", "timestamp", "score"])
        oos["source"] = "oos"
        parts.append(oos)
        print(f"  {direction} OOS rows: {len(oos):,}")
    else:
        raise FileNotFoundError(f"Missing OOS parquet: {oos_path}")

    timeline = pd.concat(parts, ignore_index=True)
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], utc=True)

    # Deduplicate (rare overlap if any). Keep the most-trusted source per row.
    # Priority: oos > cal > oof (closest to "held out from any model").
    priority = {"oos": 0, "cal": 1, "oof": 2}
    timeline["_pri"] = timeline["source"].map(priority).fillna(99).astype(int)
    timeline = (
        timeline.sort_values(["symbol", "timestamp", "_pri"])
        .drop_duplicates(subset=["symbol", "timestamp"], keep="first")
        .drop(columns=["_pri"])
        .reset_index(drop=True)
    )
    print(f"  {direction} stitched timeline: {len(timeline):,} rows "
          f"(non-NaN: {(~timeline['score'].isna()).sum():,})")
    return timeline


def _build_lag_roll_features(
    timeline: pd.DataFrame, direction: str,
) -> pd.DataFrame:
    """Per-symbol shift + rolling. Output keyed by (symbol, timestamp)."""
    df = timeline.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)["score"]

    out = pd.DataFrame({
        "symbol": df["symbol"].values,
        "timestamp": df["timestamp"].values,
    })

    for lag in LAG_BARS:
        out[f"score_{direction}_t{lag}"] = g.shift(lag).values

    # Rolling stats on the lagged-by-1 series so the window does not include
    # the current bar (no look-ahead).
    rolled = g.shift(1)
    out[f"score_{direction}_roll_mean_3d"] = (
        rolled.groupby(df["symbol"]).rolling(ROLL_WINDOW_BARS, min_periods=1)
        .mean().reset_index(level=0, drop=True).values
    )
    out[f"score_{direction}_roll_std_3d"] = (
        rolled.groupby(df["symbol"]).rolling(ROLL_WINDOW_BARS, min_periods=2)
        .std().reset_index(level=0, drop=True).values
    )

    # 24h change: score_t1 − score_t7  (use t1 not t0 to avoid look-ahead)
    score_t1 = g.shift(1).values
    score_t1_plus_lag = g.shift(1 + CHANGE_LAG_BARS).values
    out[f"score_{direction}_change_1d"] = score_t1 - score_t1_plus_lag

    return out


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_1.5 augment features — start")

    # ── 1. Load score timelines ──────────────────────────────────────────────
    print(f"\n{_ts()} Loading LONG score timeline...")
    timeline_long = _load_score_timeline("long")
    print(f"\n{_ts()} Loading SHORT score timeline...")
    timeline_short = _load_score_timeline("short")

    # ── 2. Build lag/roll features ───────────────────────────────────────────
    print(f"\n{_ts()} Building LONG lag/roll features...")
    feats_long = _build_lag_roll_features(timeline_long, "long")
    print(f"  LONG features: {feats_long.shape}")

    print(f"\n{_ts()} Building SHORT lag/roll features...")
    feats_short = _build_lag_roll_features(timeline_short, "short")
    print(f"  SHORT features: {feats_short.shape}")

    # ── 3. Merge long + short on (symbol, timestamp) ─────────────────────────
    print(f"\n{_ts()} Joining LONG + SHORT sequence features...")
    seq_feats = feats_long.merge(feats_short, on=["symbol", "timestamp"], how="outer")
    seq_feats = seq_feats.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    print(f"  combined sequence-feature frame: {seq_feats.shape}")
    for col in NEW_FEATURE_COLS:
        if col not in seq_feats.columns:
            seq_feats[col] = np.nan
    nan_pct = {
        col: round(100.0 * seq_feats[col].isna().mean(), 1)
        for col in NEW_FEATURE_COLS
    }
    print(f"  NaN% per new feature: {nan_pct}")

    # ── 4. Merge into features_full ──────────────────────────────────────────
    feats_path = DATA_DIR / "features_full.parquet"
    print(f"\n{_ts()} Loading features_full from {feats_path} ...")
    features_full = pd.read_parquet(feats_path)
    features_full["timestamp"] = pd.to_datetime(features_full["timestamp"], utc=True)
    print(f"  features_full: {features_full.shape}")

    # Drop new feature cols if they pre-exist (idempotent reruns).
    pre_existing = [c for c in NEW_FEATURE_COLS if c in features_full.columns]
    if pre_existing:
        print(f"  dropping {len(pre_existing)} pre-existing new-feature cols (idempotent rerun)")
        features_full = features_full.drop(columns=pre_existing)

    print(f"\n{_ts()} Left-joining sequence features onto features_full ...")
    out = features_full.merge(
        seq_feats[["symbol", "timestamp"] + NEW_FEATURE_COLS],
        on=["symbol", "timestamp"],
        how="left",
    )
    print(f"  result: {out.shape}")

    nan_pct_after_merge = {
        col: round(100.0 * out[col].isna().mean(), 1) for col in NEW_FEATURE_COLS
    }
    print(f"  NaN% in merged frame: {nan_pct_after_merge}")

    out_path = DATA_DIR / "features_full_v1_5.parquet"
    print(f"\n{_ts()} Writing {out_path} ...")
    out.to_parquet(out_path, index=False)
    sz_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"  wrote {sz_mb:.1f} MB")

    elapsed = time.time() - t0
    print(f"\n{_ts()} v_new_1.5 augment features — DONE in {elapsed:.1f}s")
    print(f"\nNew features: {NEW_FEATURE_COLS}")


if __name__ == "__main__":
    main()
