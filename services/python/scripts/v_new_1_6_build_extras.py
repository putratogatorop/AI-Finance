"""v_new_1.6 — Build the additive narrative-features parquet.

Combines:
  1. cohort_momentum.parquet  (from v_new_1_6_cohort_momentum.py)
  2. coingecko_categories.json (from v_new_1_6_coingecko_fetch.py)

into one wide parquet keyed by (symbol, timestamp) that can be merged onto
either features_full.parquet (if v_new_1.5 fails) or features_full_v1_5.parquet
(if v_new_1.5 passes) for v_new_1.6 training.

For each CoinGecko category we compute a per-timestamp momentum series
(mean 7d close-to-close return across category members in our universe), then
expose it per-symbol: cat_<label>_momentum_7d is the category's momentum if
the symbol is a member of that category, NaN otherwise.

Output:
    services/python/data/v_new_1_v2/features_full_v1_6_extras.parquet
    Columns:
      symbol, timestamp,
      cohort_top10_momentum_7d, cohort_top10_corr_mean,
      cat_memes_momentum_7d, cat_rwa_momentum_7d, cat_ai_tokens_momentum_7d,
      cat_depin_momentum_7d, cat_gamefi_momentum_7d, cat_defi_momentum_7d,
      cat_layer1_momentum_7d, cat_layer2_momentum_7d,
      cat_solana_eco_momentum_7d, cat_eth_eco_momentum_7d,
      cat_membership_count
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                    str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))

CAT_LABELS = [
    "memes", "rwa", "ai_tokens", "depin", "gamefi", "defi",
    "layer1", "layer2", "solana_eco", "eth_eco",
]
RETURN_WINDOW_BARS = 42  # 7 days × 6 bars/day


def _ts(label: str = "") -> str:
    now = time.strftime("%H:%M:%S")
    return f"[{now}]{(' ' + label) if label else ''}"


def _build_returns_matrix(features_path: pathlib.Path) -> pd.DataFrame:
    feats = pd.read_parquet(features_path, columns=["symbol", "timestamp", "close"])
    feats["timestamp"] = pd.to_datetime(feats["timestamp"], utc=True)
    wide_close = feats.pivot_table(index="timestamp", columns="symbol", values="close",
                                    aggfunc="first").sort_index()
    return np.log(wide_close / wide_close.shift(1))


def _per_category_momentum(
    returns: pd.DataFrame, members: list[str],
) -> pd.Series:
    """7-day cumulative log return for the category, computed as the mean of
    member-coin 7d returns (in simple-return space). Returns a Series indexed
    by timestamp."""
    valid = [m for m in members if m in returns.columns]
    if not valid:
        return pd.Series(np.nan, index=returns.index, name="cat_momentum")
    sub = returns[valid]
    rolling_log = sub.rolling(RETURN_WINDOW_BARS, min_periods=10).sum()
    rolling_simple = np.exp(rolling_log) - 1.0
    return rolling_simple.mean(axis=1, skipna=True)


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_1.6 build extras — start")

    # 1. Load cohort parquet
    cohort_path = DATA / "cohort_momentum.parquet"
    if not cohort_path.exists():
        print(f"  ERROR: {cohort_path} missing. Run v_new_1_6_cohort_momentum.py first.")
        sys.exit(1)
    cohort = pd.read_parquet(cohort_path)
    cohort["timestamp"] = pd.to_datetime(cohort["timestamp"], utc=True)
    print(f"  cohort: {cohort.shape}")

    # 2. Load CoinGecko categories
    cg_path = DATA / "coingecko_categories.json"
    if not cg_path.exists():
        print(f"  ERROR: {cg_path} missing. Run v_new_1_6_coingecko_fetch.py first.")
        sys.exit(1)
    with open(cg_path) as f:
        categories = json.load(f)
    print(f"  categories loaded: {list(categories.keys())}")

    # 3. Build returns matrix from features_full
    features_path = DATA / "features_full.parquet"
    print(f"\n{_ts()} Building returns matrix from {features_path} ...")
    returns = _build_returns_matrix(features_path)
    print(f"  returns: {returns.shape}")

    # 4. Per-category 7d momentum series
    print(f"\n{_ts()} Computing per-category momentum series...")
    cat_series: dict[str, pd.Series] = {}
    for label in CAT_LABELS:
        members = categories.get(label, [])
        if not members:
            cat_series[label] = pd.Series(np.nan, index=returns.index)
            print(f"  {label}: no members, all-NaN")
            continue
        cat_series[label] = _per_category_momentum(returns, members)
        n_in_universe = len([m for m in members if m in returns.columns])
        print(f"  {label}: {n_in_universe}/{len(members)} members in universe; "
              f"non-NaN {(~cat_series[label].isna()).sum():,}")

    # 5. Build per-symbol membership map
    sym_membership: dict[str, set[str]] = {}
    for label, members in categories.items():
        if label not in CAT_LABELS:
            continue
        for sym in members:
            sym_membership.setdefault(sym, set()).add(label)

    # 6. Build extras frame: every (symbol, timestamp) in the cohort frame gets
    #    cohort cols (already present) + cat_*_momentum cols (NaN unless symbol
    #    is a member of that category) + cat_membership_count
    print(f"\n{_ts()} Composing extras frame...")
    out = cohort.copy()
    # Pre-extract per-timestamp lookup arrays for each category
    for label in CAT_LABELS:
        ser = cat_series[label]
        # Map each row to its category momentum (if symbol is member)
        col_name = f"cat_{label}_momentum_7d"
        # Vectorise: build a per-row value as ser.reindex(timestamps); then NaN-out non-members
        ser_lookup = ser.reindex(out["timestamp"]).values
        is_member = out["symbol"].isin([s for s, cats in sym_membership.items() if label in cats]).values
        out[col_name] = np.where(is_member, ser_lookup, np.nan)

    # Membership count = how many of the CAT_LABELS the symbol belongs to
    out["cat_membership_count"] = out["symbol"].apply(
        lambda s: len(sym_membership.get(s, set()) & set(CAT_LABELS))
    ).astype(np.int8)

    # Sanity stats
    print(f"  extras shape: {out.shape}")
    print(f"  feature-non-NaN counts:")
    for c in [col for col in out.columns if col.startswith(("cohort_", "cat_"))]:
        nn = (~out[c].isna()).sum()
        pct = 100.0 * nn / len(out)
        print(f"    {c:36s}: {nn:>9,} non-NaN ({pct:5.1f}%)")

    out_path = DATA / "features_full_v1_6_extras.parquet"
    out.to_parquet(out_path, index=False)
    elapsed = time.time() - t0
    print(f"\n{_ts()} DONE in {elapsed:.1f}s")
    print(f"  wrote {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
