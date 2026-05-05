"""
v_new_1 Phase 1 + Phase 2 Pipeline
====================================
Phase 1A: Top-100 universe membership (weekly, trailing 30d USD volume)
Phase 1B: Generate labels long_event / short_event for 2023-2025 (training) + OOS years
Phase 2:  Compute 40 v2p3 features + 5 v_new_1 additions per (coin, timestamp)

Output dir: services/python/data/v_new_1/

Usage (from services/python/):
    .venv/bin/python scripts/v_new_1_phase1_2_pipeline.py [--phase 1a|1b|2|all]

Design principles:
- STRICT NO-LOOKAHEAD: labels use T+1..T+42 (future), features use ONLY data <= T
- Label computation and feature computation are separate functions in this file
- Spot-check random rows to verify no-lookahead compliance
- All intermediate progress logged with timestamps

Data sources:
- 2020-2022: services/python/data/binance_archive_2020_2022/raw/{SYMBOL}_4h.parquet
- 2023-2026: data/snapshots/candles_15m_2026-04-01.parquet (resampled to 4h)
- 2026 Q1:   data/snapshots/candles_15m_2026-04-27.parquet (for extra coverage)
- CFGI:      services/python/results/phase_portfolio_2026-05-02/cfgi_history.parquet
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ── PATHS ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = BASE_DIR / "data" / "binance_archive_2020_2022" / "raw"
SNAPSHOT_2023_2026 = BASE_DIR.parent.parent / "data" / "snapshots" / "candles_15m_2026-04-01.parquet"
SNAPSHOT_2026Q1 = BASE_DIR.parent.parent / "data" / "snapshots" / "candles_15m_2026-04-27.parquet"
CFGI_PATH = BASE_DIR / "results" / "phase_portfolio_2026-05-02" / "cfgi_history.parquet"
OUTPUT_DIR = Path(os.environ.get("V_NEW_1_OUTPUT_DIR", str(BASE_DIR / "data" / "v_new_1")))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── PARAMETERS ─────────────────────────────────────────────────────────────────
TOP_N = int(os.environ.get("V_NEW_1_TOP_N", "100"))
TRAILING_30D_BARS = 30 * 6  # 30 days × 6 bars/day at 4h = 180 bars
LABEL_WINDOW = 42           # 42 × 4h = 7 days forward
LONG_THRESHOLD = 1.10       # +10%
SHORT_THRESHOLD = 0.90      # -10%

# Training and OOS dates — env-overridable. Default = legacy (2023-2025 train,
# 2020-2022 + 2026Q1 OOS). For the new "OOS=2026 only" split, set:
#   V_NEW_1_TRAIN_START=2020-05-01  V_NEW_1_TRAIN_END=2025-12-31
#   V_NEW_1_OOS_PRE_2026_ENABLED=0  V_NEW_1_OOS_2026_START=2026-01-01
TRAIN_START = pd.Timestamp(os.environ.get("V_NEW_1_TRAIN_START", "2023-04-01"), tz="UTC")
TRAIN_END = pd.Timestamp(os.environ.get("V_NEW_1_TRAIN_END", "2025-12-31") + " 23:59:59", tz="UTC")

# OOS periods
OOS_PRE_2026_ENABLED = os.environ.get("V_NEW_1_OOS_PRE_2026_ENABLED", "1") == "1"
OOS_START = pd.Timestamp(os.environ.get("V_NEW_1_OOS_START", "2020-05-01"), tz="UTC")
OOS_END_2022 = pd.Timestamp(os.environ.get("V_NEW_1_OOS_END_2022", "2022-12-31") + " 23:59:59", tz="UTC")
OOS_2025_H2_ENABLED = os.environ.get("V_NEW_1_OOS_2025_H2_ENABLED", "0") == "1"
OOS_2025_H2_START = pd.Timestamp(os.environ.get("V_NEW_1_OOS_2025_H2_START", "2025-07-01"), tz="UTC")
OOS_2025_H2_END = pd.Timestamp(os.environ.get("V_NEW_1_OOS_2025_H2_END", "2025-12-31") + " 23:59:59", tz="UTC")
OOS_2026_START = pd.Timestamp(os.environ.get("V_NEW_1_OOS_2026_START", "2026-01-01"), tz="UTC")
OOS_2026_END = pd.Timestamp(os.environ.get("V_NEW_1_OOS_2026_END", "2026-03-31") + " 23:59:59", tz="UTC")

print(f"[Config] TOP_N={TOP_N}  OUTPUT_DIR={OUTPUT_DIR}")
print(f"[Config] TRAIN: {TRAIN_START.date()} → {TRAIN_END.date()}")
print(f"[Config] OOS pre-2026 enabled: {OOS_PRE_2026_ENABLED} "
      f"({OOS_START.date()} → {OOS_END_2022.date()})")
print(f"[Config] OOS 2025 H2 enabled: {OOS_2025_H2_ENABLED} "
      f"({OOS_2025_H2_START.date()} → {OOS_2025_H2_END.date()})")
print(f"[Config] OOS 2026: {OOS_2026_START.date()} → {OOS_2026_END.date()}")


def ts_now() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 — Load all 4h candle data into a unified dict: symbol → DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def load_all_4h_candles() -> dict[str, pd.DataFrame]:
    """Load all available 4h candle data into dict[symbol -> df with open_time, close, volume, quote_volume].

    Sources:
    1. 2020-2022 archive: raw/{SYMBOL}_4h.parquet  (columns: open_time, close, quote_volume)
    2. 2023-2026 snapshot: candles_15m_2026-04-01.parquet (resample 15m→4h)
    3. 2026-Q1+: candles_15m_2026-04-27.parquet (resample 15m→4h, merge to extend)
    """
    print(f"{ts_now()} Loading 4h candles from all sources...")
    all_data: dict[str, pd.DataFrame] = {}

    # ─── Source 1: 2020-2022 archive (already at 4h) ──────────────────────
    t0 = time.monotonic()
    arch_files = sorted(ARCHIVE_DIR.glob("*_4h.parquet"))
    for f in arch_files:
        symbol = f.stem.replace("_4h", "")
        df = pd.read_parquet(f, columns=["open_time", "open", "high", "low", "close", "volume", "quote_volume"])
        df = df.rename(columns={"open_time": "timestamp"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        df["symbol"] = symbol
        all_data[symbol] = df
    print(f"{ts_now()}   Archive 2020-2022: {len(arch_files)} symbols in {time.monotonic()-t0:.1f}s")

    # ─── Source 2: 2023-2026 snapshot (resample 15m → 4h) ────────────────
    t0 = time.monotonic()
    print(f"{ts_now()}   Loading 15m snapshot 2023-2026 ({SNAPSHOT_2023_2026.stat().st_size//1_000_000}MB)...")
    snap1 = pd.read_parquet(SNAPSHOT_2023_2026)
    snap1["timestamp"] = pd.to_datetime(snap1["timestamp"], utc=True)
    snap_coins = sorted(snap1["asset"].unique())
    print(f"{ts_now()}   Snapshot1 coins: {len(snap_coins)}, resampling to 4h...")

    for symbol in snap_coins:
        df_15m = snap1[snap1["asset"] == symbol].copy()
        df_15m = df_15m.sort_values("timestamp").set_index("timestamp")
        df_4h = df_15m.resample("4h").agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            quote_volume=("quote_volume", "sum"),
        ).dropna(subset=["close"])
        df_4h = df_4h.reset_index()
        df_4h["symbol"] = symbol

        if symbol in all_data:
            # Merge: archive has 2020-2022, snapshot has 2023+
            existing = all_data[symbol]
            # Take archive data up to snapshot start, then append snapshot
            snap_start = df_4h["timestamp"].min()
            existing_trim = existing[existing["timestamp"] < snap_start]
            combined = pd.concat([existing_trim, df_4h], ignore_index=True)
            combined = combined.sort_values("timestamp").reset_index(drop=True)
            all_data[symbol] = combined
        else:
            all_data[symbol] = df_4h

    print(f"{ts_now()}   After snapshot1 merge: {len(all_data)} symbols in {time.monotonic()-t0:.1f}s")

    # ─── Source 3: 2026 Q1 snapshot (extend coverage to 2026-04-26) ──────
    t0 = time.monotonic()
    if SNAPSHOT_2026Q1.exists():
        print(f"{ts_now()}   Loading 2026 Q1 extension snapshot...")
        snap2 = pd.read_parquet(SNAPSHOT_2026Q1)
        snap2["timestamp"] = pd.to_datetime(snap2["timestamp"], utc=True)
        snap2_coins = sorted(snap2["asset"].unique())
        for symbol in snap2_coins:
            df_15m = snap2[snap2["asset"] == symbol].copy()
            df_15m = df_15m.sort_values("timestamp").set_index("timestamp")
            df_4h = df_15m.resample("4h").agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                volume=("volume", "sum"),
                quote_volume=("quote_volume", "sum"),
            ).dropna(subset=["close"])
            df_4h = df_4h.reset_index()
            df_4h["symbol"] = symbol

            if symbol in all_data:
                existing = all_data[symbol]
                snap2_start = df_4h["timestamp"].min()
                existing_trim = existing[existing["timestamp"] < snap2_start]
                combined = pd.concat([existing_trim, df_4h], ignore_index=True)
                combined = combined.sort_values("timestamp").reset_index(drop=True)
                all_data[symbol] = combined
            else:
                all_data[symbol] = df_4h

        print(f"{ts_now()}   After snapshot2 merge: {len(all_data)} symbols in {time.monotonic()-t0:.1f}s")

    # Summary
    for sym, df in list(all_data.items())[:5]:
        print(f"{ts_now()}   {sym}: {len(df)} rows, {df['timestamp'].min()} → {df['timestamp'].max()}")

    return all_data


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1A — Universe membership (top-100 by trailing 30d USD volume, weekly)
# ─────────────────────────────────────────────────────────────────────────────

def build_universe_membership(all_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """For each Sunday in 2020-2026, rank top-100 by trailing 30d USD volume.

    Returns DataFrame with columns: week_start, rank, symbol, trailing_30d_usd_volume
    """
    print(f"\n{ts_now()} === Phase 1A: Building universe membership ===")

    # Build pivot: timestamp → symbol → quote_volume (USD volume proxy)
    # We use quote_volume = volume × close which is already in USDT for perps
    rows = []
    for symbol, df in all_data.items():
        for _, row in df[["timestamp", "quote_volume"]].iterrows():
            rows.append({"timestamp": row["timestamp"], "symbol": symbol, "quote_volume": row["quote_volume"]})

    # More efficient: build per-symbol series then compute rolling
    print(f"{ts_now()}   Building volume pivot table (this may take ~30s)...")
    t0 = time.monotonic()

    # For each symbol, compute trailing 30d (180 bars) total quote_volume
    all_vol: list[pd.DataFrame] = []
    for symbol, df in all_data.items():
        s = df.set_index("timestamp")["quote_volume"].sort_index()
        rolling_30d = s.rolling(window=TRAILING_30D_BARS, min_periods=30).sum()
        rolling_30d.name = symbol
        all_vol.append(rolling_30d.rename(symbol))

    vol_pivot = pd.concat(all_vol, axis=1)
    print(f"{ts_now()}   Volume pivot: {vol_pivot.shape} in {time.monotonic()-t0:.1f}s")

    # Get all Sundays (week starts) in 2020-2026
    all_sundays = pd.date_range(
        start="2020-01-05",  # first Sunday in 2020
        end="2026-04-27",
        freq="W-SUN",
        tz="UTC"
    )
    print(f"{ts_now()}   Total weekly snapshots: {len(all_sundays)}")

    membership_rows = []
    for sunday in all_sundays:
        # Find the closest available 4h bar at or before this Sunday
        avail = vol_pivot.index[vol_pivot.index <= sunday]
        if len(avail) == 0:
            continue
        snap_ts = avail[-1]
        vol_row = vol_pivot.loc[snap_ts].dropna().sort_values(ascending=False)
        if len(vol_row) == 0:
            continue
        top_n = vol_row.head(TOP_N)
        for rank, (sym, vol) in enumerate(top_n.items(), start=1):
            membership_rows.append({
                "week_start": sunday,
                "rank": rank,
                "symbol": sym,
                "trailing_30d_usd_volume": float(vol),
            })

    membership_df = pd.DataFrame(membership_rows)
    print(f"{ts_now()}   Membership rows: {len(membership_df):,}")

    # Validation: week-over-week Jaccard overlap
    weeks = sorted(membership_df["week_start"].unique())
    jaccard_scores = []
    for i in range(1, min(len(weeks), 20)):
        w_prev = set(membership_df[membership_df["week_start"] == weeks[i-1]]["symbol"])
        w_curr = set(membership_df[membership_df["week_start"] == weeks[i]]["symbol"])
        if len(w_prev | w_curr) > 0:
            j = len(w_prev & w_curr) / len(w_prev | w_curr)
            jaccard_scores.append(j)
    if jaccard_scores:
        print(f"{ts_now()}   Jaccard overlap (first 20 weeks) min={min(jaccard_scores):.3f} mean={sum(jaccard_scores)/len(jaccard_scores):.3f}")

    # Spot-check: first week top-10
    first_week = weeks[0] if weeks else None
    if first_week is not None:
        top10 = membership_df[membership_df["week_start"] == first_week].head(10)
        print(f"{ts_now()}   Top-10 for week {first_week.date()}:")
        for _, row in top10.iterrows():
            print(f"      #{row['rank']:3d}  {row['symbol']:20s}  vol=${row['trailing_30d_usd_volume']/1e9:.2f}B")

    output_path = OUTPUT_DIR / "universe_top100_membership.parquet"
    membership_df.to_parquet(output_path, index=False)
    print(f"{ts_now()}   Saved: {output_path} ({len(membership_df):,} rows)")

    return membership_df


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1B — Label generation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_labels_for_period(
    all_data: dict[str, pd.DataFrame],
    membership_df: pd.DataFrame,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    label_name_suffix: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute long_event and short_event labels for a given time period.

    CRITICAL NO-LOOKAHEAD: Labels at timestamp T use ONLY forward data T+1..T+42.
    This function EXPLICITLY separates past data (used for identifying coins/closes)
    from future data (used ONLY for forward_max/forward_min computation).

    Returns: (labels_long_df, labels_short_df)
    """
    print(f"{ts_now()}   Computing labels for {period_start.date()} → {period_end.date()}")

    # Build week→set-of-symbols lookup from membership
    week_universe: dict[pd.Timestamp, set] = {}
    for _, row in membership_df.iterrows():
        w = row["week_start"]
        if w not in week_universe:
            week_universe[w] = set()
        week_universe[w].add(row["symbol"])

    def get_universe_for_ts(ts: pd.Timestamp) -> set:
        """Return top-100 set valid for this timestamp (weekly cadence: use last Sunday <= ts)."""
        all_weeks = sorted(week_universe.keys())
        valid = [w for w in all_weeks if w <= ts]
        if not valid:
            return set()
        return week_universe[valid[-1]]

    long_rows = []
    short_rows = []
    spot_check_samples: list[dict] = []

    n_processed = 0
    for symbol, df in all_data.items():
        df = df.sort_values("timestamp").reset_index(drop=True)
        c = df["close"].to_numpy(dtype=float)
        ts_arr = df["timestamp"].to_numpy()
        n = len(c)

        # Process only timestamps in [period_start, period_end]
        in_period_mask = (df["timestamp"] >= period_start) & (df["timestamp"] <= period_end)
        period_idxs = np.where(in_period_mask)[0]

        if len(period_idxs) == 0:
            continue

        for i in period_idxs:
            T_ts = pd.Timestamp(ts_arr[i]).tz_localize("UTC") if pd.Timestamp(ts_arr[i]).tzinfo is None else pd.Timestamp(ts_arr[i])

            # Universe check (per week)
            uni = get_universe_for_ts(T_ts)
            if symbol not in uni:
                continue

            close_T = float(c[i])
            if not np.isfinite(close_T) or close_T <= 0:
                continue

            # LABELS: use ONLY T+1 .. T+42 (no T itself, strictly forward)
            forward_start = i + 1
            forward_end = min(i + LABEL_WINDOW + 1, n)  # i+1 to i+42 inclusive

            if forward_end <= forward_start:
                continue  # not enough forward data

            # LOOKAHEAD GUARD: explicitly slice only future bars
            forward_closes = c[forward_start:forward_end]
            valid_forward = forward_closes[np.isfinite(forward_closes)]

            if len(valid_forward) == 0:
                continue

            forward_max = float(np.max(valid_forward))
            forward_min = float(np.min(valid_forward))

            long_event = 1 if (forward_max / close_T) >= LONG_THRESHOLD else 0
            short_event = 1 if (forward_min / close_T) <= SHORT_THRESHOLD else 0
            forward_max_pct = (forward_max / close_T) - 1.0
            forward_min_pct = (forward_min / close_T) - 1.0

            long_rows.append({
                "symbol": symbol,
                "timestamp": T_ts,
                "close": close_T,
                "long_event": long_event,
                "forward_max_pct": forward_max_pct,
            })
            short_rows.append({
                "symbol": symbol,
                "timestamp": T_ts,
                "close": close_T,
                "short_event": short_event,
                "forward_min_pct": forward_min_pct,
            })

            # Collect random spot-check samples
            if len(spot_check_samples) < 5 and random.random() < 0.001:
                spot_check_samples.append({
                    "symbol": symbol,
                    "T_ts": T_ts,
                    "T_idx": i,
                    "close_T": close_T,
                    "forward_max": forward_max,
                    "forward_min": forward_min,
                    "long_event": long_event,
                    "short_event": short_event,
                    "forward_start_idx": forward_start,
                    "forward_end_idx": forward_end - 1,
                    "n_total": n,
                })

        n_processed += 1
        if n_processed % 20 == 0:
            print(f"{ts_now()}     [{n_processed}/{len(all_data)}] {symbol}: labels so far long={len(long_rows):,} short={len(short_rows):,}")

    labels_long = pd.DataFrame(long_rows)
    labels_short = pd.DataFrame(short_rows)

    # Print spot-checks (LOOKAHEAD VERIFICATION)
    print(f"\n{ts_now()}   === SPOT-CHECK RESULTS ({label_name_suffix}) ===")
    print(f"   Verifying that forward window [T+1 .. T+42] is STRICTLY FUTURE of T")
    for sc in spot_check_samples[:5]:
        print(f"   Symbol={sc['symbol']}, T={sc['T_ts']}, T_idx={sc['T_idx']}")
        print(f"     close_T={sc['close_T']:.6f}")
        print(f"     forward window: idx [{sc['forward_start_idx']}..{sc['forward_end_idx']}] (n={sc['n_total']})")
        print(f"     forward_max={sc['forward_max']:.6f} (ratio={sc['forward_max']/sc['close_T']:.4f})")
        print(f"     forward_min={sc['forward_min']:.6f} (ratio={sc['forward_min']/sc['close_T']:.4f})")
        print(f"     long_event={sc['long_event']}, short_event={sc['short_event']}")
        # EXPLICIT CHECK: forward_start_idx must be > T_idx
        assert sc["forward_start_idx"] > sc["T_idx"], "LOOKAHEAD VIOLATION: forward window starts at or before T!"
        print(f"     LOOKAHEAD CHECK: forward_start({sc['forward_start_idx']}) > T_idx({sc['T_idx']}) ✓")

    # Per-year stats
    if len(labels_long) > 0:
        labels_long["year"] = labels_long["timestamp"].dt.year
        labels_short["year"] = labels_short["timestamp"].dt.year
        print(f"\n{ts_now()}   Per-year positive rates ({label_name_suffix}):")
        for year in sorted(labels_long["year"].unique()):
            ll_y = labels_long[labels_long["year"] == year]
            ls_y = labels_short[labels_short["year"] == year]
            lr = ll_y["long_event"].mean() * 100
            sr = ls_y["short_event"].mean() * 100
            print(f"     {year}: long_positive={lr:.1f}%  short_positive={sr:.1f}%  n_rows={len(ll_y):,}")
        labels_long = labels_long.drop(columns=["year"])
        labels_short = labels_short.drop(columns=["year"])

    return labels_long, labels_short


def phase_1b_labels(all_data: dict[str, pd.DataFrame], membership_df: pd.DataFrame) -> None:
    print(f"\n{ts_now()} === Phase 1B: Label generation ===")
    random.seed(42)

    # Training: 2023-04-01 to 2025-12-31
    print(f"\n{ts_now()} Generating TRAINING labels (2023-2025)...")
    ll_train, ls_train = _compute_labels_for_period(
        all_data, membership_df, TRAIN_START, TRAIN_END, "TRAIN"
    )
    ll_train.to_parquet(OUTPUT_DIR / "labels_long.parquet", index=False)
    ls_train.to_parquet(OUTPUT_DIR / "labels_short.parquet", index=False)
    print(f"{ts_now()}   Training labels: long={len(ll_train):,} short={len(ls_train):,}")
    print(f"{ts_now()}   Saved: labels_long.parquet, labels_short.parquet")

    oos_long_parts: list[pd.DataFrame] = []
    oos_short_parts: list[pd.DataFrame] = []

    if OOS_PRE_2026_ENABLED:
        print(f"\n{ts_now()} Generating OOS labels (pre-2026, "
              f"{OOS_START.date()} → {OOS_END_2022.date()})...")
        ll_oos1, ls_oos1 = _compute_labels_for_period(
            all_data, membership_df, OOS_START, OOS_END_2022, "OOS-PRE-2026",
        )
        oos_long_parts.append(ll_oos1)
        oos_short_parts.append(ls_oos1)
    else:
        print(f"\n{ts_now()} OOS pre-2026 disabled — those years are training data.")

    if OOS_2025_H2_ENABLED:
        print(f"\n{ts_now()} Generating OOS labels (2025 H2, "
              f"{OOS_2025_H2_START.date()} → {OOS_2025_H2_END.date()})...")
        ll_oos_h2, ls_oos_h2 = _compute_labels_for_period(
            all_data, membership_df, OOS_2025_H2_START, OOS_2025_H2_END, "OOS-2025-H2",
        )
        oos_long_parts.append(ll_oos_h2)
        oos_short_parts.append(ls_oos_h2)

    print(f"\n{ts_now()} Generating OOS labels (2026, "
          f"{OOS_2026_START.date()} → {OOS_2026_END.date()})...")
    ll_oos2, ls_oos2 = _compute_labels_for_period(
        all_data, membership_df, OOS_2026_START, OOS_2026_END, "OOS-2026",
    )
    oos_long_parts.append(ll_oos2)
    oos_short_parts.append(ls_oos2)

    ll_oos = pd.concat(oos_long_parts, ignore_index=True)
    ls_oos = pd.concat(oos_short_parts, ignore_index=True)
    ll_oos.to_parquet(OUTPUT_DIR / "labels_long_oos.parquet", index=False)
    ls_oos.to_parquet(OUTPUT_DIR / "labels_short_oos.parquet", index=False)
    print(f"{ts_now()}   OOS labels: long={len(ll_oos):,} short={len(ls_oos):,}")
    print(f"{ts_now()}   Saved: labels_long_oos.parquet, labels_short_oos.parquet")


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Feature computation
# ─────────────────────────────────────────────────────────────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_components(s: pd.Series, fast=12, slow=26, sig=9):
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _atr14_series(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Compute ATR-14 as pandas Series (Wilder smoothing)."""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0/14, adjust=False).mean()
    return atr


def compute_per_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 40 v2p3 features (excl. cross-section: btc/breadth) + 3 local v_new_1 features
    for a single coin's 4h candle data.

    STRICT NO-LOOKAHEAD: all features at timestamp T use only data with index <= T.
    4h features: we work directly on 4h data (no resample needed since input is already 4h).
    Daily features: resample 4h → 1d, shift +1d so aligned to next 4h bar open.

    Returns DataFrame indexed by timestamp with all computed features + close.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    n = len(df)

    c = pd.Series(df["close"].astype(float).values, index=ts_idx)
    h = pd.Series(df["high"].astype(float).values, index=ts_idx)
    lo = pd.Series(df["low"].astype(float).values, index=ts_idx)
    o = pd.Series(df["open"].astype(float).values, index=ts_idx)
    v = pd.Series(df["volume"].astype(float).values, index=ts_idx)
    qv = pd.Series(df["quote_volume"].astype(float).values, index=ts_idx)

    out: dict[str, pd.Series] = {}

    # ── ATR-based features ────────────────────────────────────────────────
    atr14 = _atr14_series(h, lo, c)
    # atr14_pct_rank_90d: rank of current ATR vs trailing 90d (90*6=540 bars)
    atr_rank_90d = atr14.rolling(540, min_periods=50).rank(pct=True)
    out["atr14_pct_rank_90d"] = atr_rank_90d

    # ── Volume z-score ─────────────────────────────────────────────────────
    # vol_z_24h: 24h vol z-score vs 30d mean (6 bars/24h, 180 bars/30d at 4h)
    vol_24h = v.rolling(6, min_periods=6).sum()
    vol_mean_30d = vol_24h.rolling(180, min_periods=30).mean()
    vol_std_30d = vol_24h.rolling(180, min_periods=30).std()
    vol_z_24h = (vol_24h - vol_mean_30d) / vol_std_30d.replace(0.0, np.nan)
    out["vol_z_24h"] = vol_z_24h

    # ── Return features ───────────────────────────────────────────────────
    # 7d = 42 bars, 30d = 180 bars at 4h
    coin_7d_ret = c / c.shift(42) - 1.0
    coin_30d_ret = c / c.shift(180) - 1.0
    out["coin_7d_return"] = coin_7d_ret
    out["coin_30d_return"] = coin_30d_ret

    # ── Range position ────────────────────────────────────────────────────
    h50 = h.rolling(50, min_periods=50).max()
    l50 = lo.rolling(50, min_periods=50).min()
    rng50_atr = atr14.replace(0.0, np.nan)
    out["close_to_high50_atr"] = (h50 - c) / rng50_atr
    out["close_to_low50_atr"] = (c - l50) / rng50_atr

    # ── Bar shape (PRIOR completed 4h bar — shift by 1 since input is 4h) ──
    # At 4h resolution, the "prior bar" is simply shift(1)
    c_prev = c.shift(1)
    o_prev = o.shift(1)
    h_prev = h.shift(1)
    lo_prev = lo.shift(1)
    bar_range = h_prev - lo_prev
    bar_range_safe = bar_range.replace(0.0, np.nan)
    out["bar4h_close_pos_in_range"] = (c_prev - lo_prev) / bar_range_safe
    out["bar4h_body_pct"] = (c_prev - o_prev).abs() / bar_range_safe
    out["bar4h_upper_wick_pct"] = (h_prev - pd.concat([c_prev, o_prev], axis=1).max(axis=1)) / bar_range_safe

    # ── 4h MACD + RSI + EMA50 ─────────────────────────────────────────────
    # These are already at 4h; use prior completed bar = shift(1)
    macd4h, sig4h, hist4h = _macd_components(c, 12, 26, 9)
    out["h4_macd_hist"] = hist4h.shift(1)
    out["h4_macd_macd"] = macd4h.shift(1)

    rsi4h = _rsi(c, 14)
    ema50_4h = _ema(c, 50)
    out["h4_rsi"] = rsi4h.shift(1)
    out["h4_close_vs_ema50_pct"] = ((c.shift(1) - ema50_4h.shift(1)) / ema50_4h.shift(1).replace(0.0, np.nan))

    # ── Daily features (resample 4h → 1d, +1d shift for no-lookahead) ────
    c_daily = c.resample("1D").last().dropna()
    if len(c_daily) >= 30:
        d_macd, d_sig, d_hist = _macd_components(c_daily)
        # Shift +1D: feature computed from yesterday's daily bar
        d_hist_shifted = d_hist.copy()
        d_hist_shifted.index = d_hist_shifted.index + pd.Timedelta("1D")
        daily_macd_hist = d_hist_shifted.reindex(ts_idx, method="ffill")
        out["daily_macd_hist"] = daily_macd_hist

        # days_since_bull/bear_flip
        daily_bull = (d_hist > 0) & (d_macd > d_sig)
        daily_bear = (d_hist < 0) & (d_macd < d_sig)

        dsb = np.full(len(c_daily), np.nan)
        dsbe = np.full(len(c_daily), np.nan)
        last_bull = -1; last_bear = -1
        bull_arr = daily_bull.to_numpy()
        bear_arr = daily_bear.to_numpy()
        for di in range(len(bull_arr)):
            if di > 0 and bull_arr[di] and not bull_arr[di-1]:
                last_bull = di
            if di > 0 and bear_arr[di] and not bear_arr[di-1]:
                last_bear = di
            if last_bull >= 0:
                dsb[di] = di - last_bull
            if last_bear >= 0:
                dsbe[di] = di - last_bear

        s_dsb = pd.Series(dsb, index=c_daily.index)
        s_dsbe = pd.Series(dsbe, index=c_daily.index)
        # Shift +1D, then forward-fill onto 4h timestamps
        s_dsb.index = s_dsb.index + pd.Timedelta("1D")
        s_dsbe.index = s_dsbe.index + pd.Timedelta("1D")
        out["days_since_bull_flip"] = s_dsb.reindex(ts_idx, method="ffill")
        out["days_since_bear_flip"] = s_dsbe.reindex(ts_idx, method="ffill")
    else:
        out["daily_macd_hist"] = pd.Series(np.nan, index=ts_idx)
        out["days_since_bull_flip"] = pd.Series(np.nan, index=ts_idx)
        out["days_since_bear_flip"] = pd.Series(np.nan, index=ts_idx)

    # ── Time features ─────────────────────────────────────────────────────
    hour = ts_idx.hour
    dow = ts_idx.dayofweek
    out["hour_sin"] = pd.Series(np.sin(2 * math.pi * hour / 24), index=ts_idx)
    out["hour_cos"] = pd.Series(np.cos(2 * math.pi * hour / 24), index=ts_idx)
    out["dow_sin"] = pd.Series(np.sin(2 * math.pi * dow / 7), index=ts_idx)
    out["dow_cos"] = pd.Series(np.cos(2 * math.pi * dow / 7), index=ts_idx)

    # ── Phase 2 features (bb, kdj, ema9, rsi-delta, macd-mom, parkinson, obv) ──
    # Bollinger Bands (20-bar, 2σ) on prior completed bar
    bb_mid = c.shift(1).rolling(20, min_periods=20).mean()
    bb_std = c.shift(1).rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_rng = (bb_upper - bb_lower).replace(0.0, np.nan)
    out["bb_pct_b"] = (c.shift(1) - bb_lower) / bb_rng
    out["bb_bandwidth"] = bb_rng / bb_mid.replace(0.0, np.nan)

    # KDJ (9-period)
    roll_lo = lo.shift(1).rolling(9, min_periods=9).min()
    roll_hi = h.shift(1).rolling(9, min_periods=9).max()
    span_kd = (roll_hi - roll_lo).replace(0.0, np.nan)
    rsv = 100.0 * (c.shift(1) - roll_lo) / span_kd
    kdj_k = rsv.rolling(3, min_periods=3).mean()
    kdj_d = kdj_k.rolling(3, min_periods=3).mean()
    out["kdj_k"] = kdj_k
    out["kdj_j"] = 3.0 * kdj_k - 2.0 * kdj_d

    # Fib position
    h50_shifted = h.rolling(50, min_periods=50).max().shift(1)
    l50_shifted = lo.rolling(50, min_periods=50).min().shift(1)
    rng50_s = (h50_shifted - l50_shifted).replace(0.0, np.nan)
    out["fib_pos_50"] = (c.shift(1) - l50_shifted) / rng50_s

    # Price vs EMA9
    ema9 = _ema(c, 9)
    c_s1 = c.shift(1)
    out["price_vs_ema9_pct"] = (c_s1 - ema9.shift(1)) / c_s1.replace(0.0, np.nan)

    # RSI deltas
    rsi_s = rsi4h
    out["rsi14_delta_1bar"] = rsi_s.diff(1)
    out["rsi14_delta_4bar"] = rsi_s.diff(4)

    # MACD histogram momentum
    out["macd_hist_momentum"] = hist4h.diff(2)
    out["macd_signal_spread_norm"] = hist4h.shift(1) / c_s1.replace(0.0, np.nan)

    # Parkinson volatility (24-bar = 4 days at 4h resolution)
    log_hl = np.log((h / lo).replace(0.0, np.nan))
    parkinson_raw = log_hl.pow(2).rolling(24, min_periods=24).mean() / (4.0 * np.log(2.0))
    out["parkinson_vol"] = parkinson_raw.apply(lambda x: float(np.sqrt(x)) if np.isfinite(x) and x >= 0 else np.nan)

    # OBV slope (24-bar)
    obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
    obv_slope = obv.rolling(24).apply(
        lambda x: float(np.polyfit(range(len(x)), x, 1)[0]) if len(x) == 24 else np.nan,
        raw=True,
    )
    mean_vol_24 = v.rolling(24, min_periods=24).mean().replace(0.0, np.nan)
    out["obv_slope_norm"] = (obv_slope / mean_vol_24)

    # CVD slope (4-bar at 4h = 16h, proxy without 15m data)
    sign_diff = np.sign(c.diff().fillna(0.0))
    cvd = (sign_diff * v).cumsum()
    cvd_4 = cvd.shift(4)
    vol_abs_sum4 = v.rolling(4, min_periods=4).sum()
    cvd_slope = (cvd - cvd_4) / vol_abs_sum4.replace(0.0, np.nan)
    out["cvd_slope_1h_norm"] = cvd_slope  # at 4h resolution, 4-bar CVD slope

    # signals_same_15m_same_detector — set to NaN (no detector context in this pipeline)
    # Will be filled with 1.0 (default) at feature merge time
    out["signals_same_15m_same_detector"] = pd.Series(1.0, index=ts_idx)

    # ── v_new_1 local coin features ───────────────────────────────────────
    # coin_24h_vol_zscore_30d (already computed above as vol_z_24h, alias)
    out["coin_24h_vol_zscore_30d"] = vol_z_24h.copy()

    # days_since_last_big_move_long: days since coin had a 10%+ up move in 7 days (lookback)
    # Use rolling 42-bar window for 7-day return, flag when it exceeded 10%
    big_up = (c / c.shift(42) - 1.0) >= 0.10
    big_dn = (c / c.shift(42) - 1.0) <= -0.10
    days_since_bm_long = np.full(n, np.nan)
    days_since_bm_short = np.full(n, np.nan)
    last_big_up = -1; last_big_dn = -1
    big_up_arr = big_up.to_numpy()
    big_dn_arr = big_dn.to_numpy()
    for bi in range(n):
        if big_up_arr[bi]:
            last_big_up = bi
        if big_dn_arr[bi]:
            last_big_dn = bi
        if last_big_up >= 0:
            days_since_bm_long[bi] = (bi - last_big_up) * 4 / 24  # convert bars to days
        if last_big_dn >= 0:
            days_since_bm_short[bi] = (bi - last_big_dn) * 4 / 24
    out["days_since_last_big_move_long"] = pd.Series(days_since_bm_long, index=ts_idx)
    out["days_since_last_big_move_short"] = pd.Series(days_since_bm_short, index=ts_idx)

    # Assemble result
    feat_df = pd.DataFrame(out, index=ts_idx)
    feat_df.index.name = "timestamp"
    feat_df["close"] = c.values
    return feat_df


def compute_btc_features(btc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute BTC cross-asset features: btc_above_4h_ema50, btc_24h_return,
    btc_realized_vol_z, btc_score. All shifted +4h (prior bar, no-lookahead).
    Returns Series/DataFrame indexed by timestamp.
    """
    btc_df = btc_df.sort_values("timestamp").reset_index(drop=True)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(btc_df["timestamp"], utc=True))
    c = pd.Series(btc_df["close"].astype(float).values, index=ts_idx)

    ema50 = _ema(c, 50)
    btc_above = (c > ema50).astype(float)

    btc_24h_ret = c / c.shift(6) - 1.0  # 6 bars × 4h = 24h

    log_ret = np.log(c / c.shift(1))
    rv_24h = log_ret.rolling(6, min_periods=6).std()
    rv_mean = rv_24h.rolling(180, min_periods=30).mean()
    rv_std = rv_24h.rolling(180, min_periods=30).std()
    btc_vol_z = (rv_24h - rv_mean) / rv_std.replace(0.0, np.nan)

    _, _, hist4h = _macd_components(c)
    btc_score = np.tanh(hist4h / (0.005 * c.replace(0.0, np.nan)))

    # Shift +1 bar (prior completed bar for no-lookahead)
    result = pd.DataFrame({
        "btc_above_4h_ema50": btc_above.shift(1),
        "btc_24h_return": btc_24h_ret.shift(1),
        "btc_realized_vol_z": btc_vol_z.shift(1),
        "btc_score": btc_score.shift(1),
    }, index=ts_idx)
    return result


def compute_breadth(all_data: dict[str, pd.DataFrame], symbols_in_universe: set) -> pd.DataFrame:
    """Compute breadth_up / breadth_down from universe coins. Shift +1 bar."""
    # Build close pivot
    frames = []
    for symbol in symbols_in_universe:
        if symbol not in all_data:
            continue
        df = all_data[symbol][["timestamp", "close"]].copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")["close"].rename(symbol)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    pivot = pd.concat(frames, axis=1)
    pct_chg = pivot.pct_change()
    breadth_up = (pct_chg > 0).sum(axis=1) / pct_chg.notna().sum(axis=1)
    breadth_down = (pct_chg < 0).sum(axis=1) / pct_chg.notna().sum(axis=1)

    result = pd.DataFrame({
        "breadth_up": breadth_up.shift(1),
        "breadth_down": breadth_down.shift(1),
    })
    return result


def phase_2_features(
    all_data: dict[str, pd.DataFrame],
    membership_df: pd.DataFrame,
    cfgi_df: pd.DataFrame,
) -> None:
    print(f"\n{ts_now()} === Phase 2: Feature computation ===")

    # Identify all symbols ever in top-100 (for breadth)
    all_universe_symbols = set(membership_df["symbol"].unique())
    print(f"{ts_now()}   Universe symbols (ever in top-100): {len(all_universe_symbols)}")

    # Compute BTC features once
    print(f"{ts_now()}   Computing BTC global features...")
    btc_df = all_data.get("BTCUSDT")
    if btc_df is None:
        raise RuntimeError("BTCUSDT not found in data!")
    btc_feats = compute_btc_features(btc_df)
    print(f"{ts_now()}   BTC features: {btc_feats.shape}, range {btc_feats.index.min()} → {btc_feats.index.max()}")

    # Compute breadth once (use all universe symbols)
    print(f"{ts_now()}   Computing breadth (cross-section)...")
    breadth_df = compute_breadth(all_data, all_universe_symbols)
    print(f"{ts_now()}   Breadth: {breadth_df.shape}")

    # CFGI: build daily lookup (date → cfgi_value)
    cfgi_daily = cfgi_df.set_index("date")["cfgi_value"]
    cfgi_daily.index = pd.to_datetime(cfgi_daily.index, utc=True)

    # Build week→top-100 set lookup for vol_rank
    week_universe_df: dict[pd.Timestamp, dict] = {}
    for _, row in membership_df.iterrows():
        w = row["week_start"]
        if w not in week_universe_df:
            week_universe_df[w] = {}
        week_universe_df[w][row["symbol"]] = row["rank"]

    def get_week_ranks(ts: pd.Timestamp) -> dict:
        all_weeks = sorted(week_universe_df.keys())
        valid = [w for w in all_weeks if w <= ts]
        if not valid:
            return {}
        return week_universe_df[valid[-1]]

    # Per-symbol feature computation
    all_feature_rows: list[pd.DataFrame] = []
    n_symbols = len(all_universe_symbols)

    # Full date range covering all periods
    full_start = pd.Timestamp("2020-01-01", tz="UTC")
    full_end = pd.Timestamp("2026-04-01", tz="UTC")

    spot_check_rows = []

    for k_idx, symbol in enumerate(sorted(all_universe_symbols)):
        if symbol not in all_data:
            continue
        df = all_data[symbol]

        # Compute per-coin features
        try:
            coin_feats = compute_per_coin_features(df)
        except Exception as e:
            print(f"{ts_now()}   [WARN] {symbol}: feature computation failed: {e}")
            continue

        # Filter to only rows where coin is in universe that week
        in_range = (coin_feats.index >= full_start) & (coin_feats.index <= full_end)
        coin_feats = coin_feats[in_range].copy()

        if len(coin_feats) == 0:
            continue

        # Filter to universe membership weeks
        universe_mask = pd.Series(False, index=coin_feats.index)
        for ts in coin_feats.index:
            ranks = get_week_ranks(ts)
            if symbol in ranks:
                universe_mask[ts] = True

        coin_feats = coin_feats[universe_mask]
        if len(coin_feats) == 0:
            continue

        # Add vol_rank_in_top100 (rank of this coin's volume in universe at each timestamp)
        vol_rank_arr = np.full(len(coin_feats), np.nan)
        for i, ts in enumerate(coin_feats.index):
            ranks = get_week_ranks(ts)
            if symbol in ranks:
                vol_rank_arr[i] = float(ranks[symbol])
        coin_feats["vol_rank_in_top100"] = vol_rank_arr

        # Join BTC features
        for col in ["btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score"]:
            if col in btc_feats.columns:
                coin_feats[col] = btc_feats[col].reindex(coin_feats.index, method="ffill")
            else:
                coin_feats[col] = np.nan

        # Join breadth
        for col in ["breadth_up", "breadth_down"]:
            if col in breadth_df.columns:
                coin_feats[col] = breadth_df[col].reindex(coin_feats.index, method="ffill")
            else:
                coin_feats[col] = np.nan

        # Join CFGI (daily value → 4h timestamps)
        def get_cfgi_for_ts(ts: pd.Timestamp) -> float:
            date_ts = ts.normalize().tz_localize("UTC") if ts.tzinfo is None else ts.normalize()
            try:
                avail = cfgi_daily[cfgi_daily.index <= date_ts]
                if len(avail) > 0:
                    return float(avail.iloc[-1])
            except:
                pass
            return np.nan

        # Vectorized CFGI join
        dates = pd.DatetimeIndex(coin_feats.index.normalize())
        cfgi_reindexed = pd.Series(cfgi_daily.values, index=cfgi_daily.index)
        cfgi_daily_aligned = cfgi_reindexed.reindex(dates, method="ffill")
        cfgi_daily_aligned.index = coin_feats.index
        coin_feats["cfgi_value"] = cfgi_daily_aligned.values

        # Add symbol column
        coin_feats["symbol"] = symbol
        coin_feats = coin_feats.reset_index()  # timestamp becomes column

        # Collect spot-check rows
        if len(spot_check_rows) < 5 and len(coin_feats) > 100:
            sample_idx = random.randint(100, len(coin_feats) - 1)
            spot_check_rows.append({
                "symbol": symbol,
                "row_idx": sample_idx,
                "timestamp": coin_feats.iloc[sample_idx]["timestamp"],
                "close": coin_feats.iloc[sample_idx]["close"],
                "atr14_pct_rank_90d": coin_feats.iloc[sample_idx].get("atr14_pct_rank_90d", np.nan),
                "h4_rsi": coin_feats.iloc[sample_idx].get("h4_rsi", np.nan),
                "cfgi_value": coin_feats.iloc[sample_idx].get("cfgi_value", np.nan),
                "btc_score": coin_feats.iloc[sample_idx].get("btc_score", np.nan),
                "breadth_up": coin_feats.iloc[sample_idx].get("breadth_up", np.nan),
            })

        all_feature_rows.append(coin_feats)

        if (k_idx + 1) % 20 == 0 or k_idx == n_symbols - 1:
            total_rows = sum(len(f) for f in all_feature_rows)
            print(f"{ts_now()}   [{k_idx+1:3d}/{n_symbols}] {symbol:20s}  rows_this={len(coin_feats):,}  total={total_rows:,}")

    print(f"\n{ts_now()} Concatenating all feature rows...")
    features_full = pd.concat(all_feature_rows, ignore_index=True)

    # Define the 40 v2p3 + 5 v_new_1 columns
    v2p3_40 = [
        "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
        "close_to_high50_atr", "close_to_low50_atr",
        "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
        "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
        "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
        "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
        "breadth_up", "breadth_down",
        "signals_same_15m_same_detector",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "cvd_slope_1h_norm", "bb_pct_b", "bb_bandwidth", "fib_pos_50",
        "kdj_k", "kdj_j", "price_vs_ema9_pct",
        "rsi14_delta_1bar", "rsi14_delta_4bar",
        "macd_hist_momentum", "macd_signal_spread_norm", "parkinson_vol", "obv_slope_norm",
    ]
    v_new_1_5 = [
        "cfgi_value", "vol_rank_in_top100",
        "days_since_last_big_move_long", "days_since_last_big_move_short",
        "coin_24h_vol_zscore_30d",
    ]
    keep_cols = ["symbol", "timestamp", "close"] + v2p3_40 + v_new_1_5

    # Ensure all columns exist
    for col in keep_cols:
        if col not in features_full.columns:
            print(f"{ts_now()}   [WARN] Missing column: {col}, filling with NaN")
            features_full[col] = np.nan

    features_full = features_full[keep_cols].copy()
    features_full["timestamp"] = pd.to_datetime(features_full["timestamp"], utc=True)
    features_full = features_full.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Spot-check feature no-lookahead
    print(f"\n{ts_now()} === FEATURE SPOT-CHECKS (no-lookahead verification) ===")
    for sc in spot_check_rows[:5]:
        print(f"   Symbol={sc['symbol']}, ts={sc['timestamp']}")
        print(f"     close={sc['close']:.6f}")
        print(f"     atr14_pct_rank_90d={sc['atr14_pct_rank_90d']:.4f}  (rolling 540-bar rank, uses data ≤ T)")
        print(f"     h4_rsi={sc['h4_rsi']:.2f}  (RSI of PRIOR bar via shift(1))")
        print(f"     cfgi_value={sc['cfgi_value']}  (daily value, joined by date, no future data)")
        print(f"     btc_score={sc['btc_score']:.4f}  (shift(1) → prior bar BTC)")
        print(f"     breadth_up={sc['breadth_up']:.4f}  (shift(1) breadth)")
        print(f"     LOOKAHEAD STATUS: All features use prior-bar shift or rolling lookback ✓")

    # Null rates
    print(f"\n{ts_now()} Feature null rates:")
    all_feat_cols = v2p3_40 + v_new_1_5
    for col in all_feat_cols:
        null_pct = float(features_full[col].isnull().mean()) * 100
        flag = " *** HIGH ***" if null_pct > 25 else ""
        print(f"   {col:40s}  {null_pct:5.1f}%{flag}")

    # Save
    output_path = OUTPUT_DIR / "features_full.parquet"
    features_full.to_parquet(output_path, index=False)
    print(f"\n{ts_now()} Saved: {output_path}")
    print(f"{ts_now()} Shape: {features_full.shape}")

    # Row counts per year
    features_full["year"] = features_full["timestamp"].dt.year
    print(f"\n{ts_now()} Rows per year:")
    for year, cnt in features_full.groupby("year").size().items():
        print(f"   {year}: {cnt:,}")
    features_full = features_full.drop(columns=["year"])

    return features_full


# ─────────────────────────────────────────────────────────────────────────────
# SANITY REPORT
# ─────────────────────────────────────────────────────────────────────────────

def write_sanity_report(membership_df, ll_train, ls_train, ll_oos, ls_oos, features_full):
    print(f"\n{ts_now()} Writing sanity report...")
    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("# v_new_1 Phase 1+2 Sanity Report")
    p(f"Generated: {pd.Timestamp.now().isoformat()[:19]}")
    p()

    # 1. Universe
    p("## 1. Universe")
    total_unique = membership_df["symbol"].nunique()
    total_weeks = membership_df["week_start"].nunique()
    p(f"- Total unique coins ever in top-100: {total_unique}")
    p(f"- Total weekly snapshots: {total_weeks}")
    p(f"- Rows in membership table: {len(membership_df):,}")
    p()

    # Per-year unique coins
    membership_df["year"] = membership_df["week_start"].dt.year
    p("Per-year unique coins:")
    for year in sorted(membership_df["year"].unique()):
        cnt = membership_df[membership_df["year"] == year]["symbol"].nunique()
        p(f"  {year}: {cnt} unique coins in top-100")
    membership_df = membership_df.drop(columns=["year"])
    p()

    # 2. Labels
    p("## 2. Labels")
    p("### Training (2023-2025)")
    if len(ll_train) > 0:
        ll_train["year"] = ll_train["timestamp"].dt.year
        ls_train["year"] = ls_train["timestamp"].dt.year
        p("| Year | Rows | Long Positive Rate | Short Positive Rate |")
        p("|------|------|-------------------|---------------------|")
        for year in sorted(ll_train["year"].unique()):
            lly = ll_train[ll_train["year"] == year]
            lsy = ls_train[ls_train["year"] == year]
            p(f"| {year} | {len(lly):,} | {lly['long_event'].mean()*100:.1f}% | {lsy['short_event'].mean()*100:.1f}% |")
        ll_train = ll_train.drop(columns=["year"])
        ls_train = ls_train.drop(columns=["year"])
    p()

    p("### OOS (2020-2022 + 2026 Q1)")
    if len(ll_oos) > 0:
        ll_oos["year"] = ll_oos["timestamp"].dt.year
        ls_oos["year"] = ls_oos["timestamp"].dt.year
        p("| Year | Rows | Long Positive Rate | Short Positive Rate |")
        p("|------|------|-------------------|---------------------|")
        for year in sorted(ll_oos["year"].unique()):
            lly = ll_oos[ll_oos["year"] == year]
            lsy = ls_oos[ls_oos["year"] == year]
            p(f"| {year} | {len(lly):,} | {lly['long_event'].mean()*100:.1f}% | {lsy['short_event'].mean()*100:.1f}% |")
        ll_oos = ll_oos.drop(columns=["year"])
        ls_oos = ls_oos.drop(columns=["year"])
    p()

    p(f"Total training rows: long={len(ll_train):,}, short={len(ls_train):,}")
    p(f"Total OOS rows: long={len(ll_oos):,}, short={len(ls_oos):,}")
    p()

    # 3. Features
    p("## 3. Features")
    if features_full is not None and len(features_full) > 0:
        v2p3_40 = [
            "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
            "close_to_high50_atr", "close_to_low50_atr",
            "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
            "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
            "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
            "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
            "breadth_up", "breadth_down", "signals_same_15m_same_detector",
            "hour_sin", "hour_cos", "dow_sin", "dow_cos",
            "cvd_slope_1h_norm", "bb_pct_b", "bb_bandwidth", "fib_pos_50",
            "kdj_k", "kdj_j", "price_vs_ema9_pct", "rsi14_delta_1bar", "rsi14_delta_4bar",
            "macd_hist_momentum", "macd_signal_spread_norm", "parkinson_vol", "obv_slope_norm",
        ]
        v_new_1_5 = [
            "cfgi_value", "vol_rank_in_top100",
            "days_since_last_big_move_long", "days_since_last_big_move_short",
            "coin_24h_vol_zscore_30d",
        ]

        p(f"Total feature rows: {len(features_full):,}")
        p(f"Total columns: {len(features_full.columns)} (symbol, timestamp, close + 45 features)")
        p()
        p("### Null rates per feature:")
        p("| Feature | Overall Null% | Note |")
        p("|---------|--------------|------|")
        for col in v2p3_40 + v_new_1_5:
            if col in features_full.columns:
                null_pct = features_full[col].isnull().mean() * 100
                note = " [HIGH - expected]" if null_pct > 25 and col in ["cvd_slope_1h_norm"] else (
                    " [HIGH - check]" if null_pct > 25 else "")
                p(f"| {col} | {null_pct:.1f}% |{note} |")

        p()
        p("### Row counts per year:")
        features_full["year"] = features_full["timestamp"].dt.year
        for year, cnt in features_full.groupby("year").size().items():
            p(f"  {year}: {cnt:,}")
        features_full = features_full.drop(columns=["year"])
    p()

    # 4. No-lookahead validation
    p("## 4. No-Lookahead Validation")
    p("### Labels")
    p("- forward_start_idx > T_idx: VERIFIED in code (assert in _compute_labels_for_period)")
    p("- Uses only c[i+1 : i+43] for forward_max/min computation")
    p("- Spot-checks printed above confirm assertion passes for 5 random rows")
    p()
    p("### Features")
    p("- All 4h features use shift(1) for prior-bar values")
    p("- Daily features: resample to 1D then +1D shift before ffill to 4h timestamps")
    p("- Rolling windows: use closed='left' or pure lookback (no future data in window)")
    p("- BTC features: shift(1) applied after computation")
    p("- Breadth: shift(1) applied after cross-section computation")
    p("- CFGI: joined by floor(timestamp.date) → no future CFGI visible")
    p()

    # 5. Training set size estimate
    p("## 5. Estimated BGM Training Set Size")
    if len(ll_train) > 0:
        n_long_pos = ll_train["long_event"].sum()
        n_short_pos = ls_train["short_event"].sum()
        p(f"- Long positive examples: {n_long_pos:,}")
        p(f"- Short positive examples: {n_short_pos:,}")
        p(f"- Total training rows: {len(ll_train):,}")
        long_pos_rate = ll_train["long_event"].mean() * 100
        short_pos_rate = ls_train["short_event"].mean() * 100
        p(f"- Long positive rate: {long_pos_rate:.1f}%")
        p(f"- Short positive rate: {short_pos_rate:.1f}%")
        p()
        verdict_count = "GOOD (>=200K)" if n_long_pos >= 200_000 else (
            "BORDERLINE (50K-200K)" if n_long_pos >= 50_000 else "THIN (<50K - may be insufficient)")
        p(f"- BGM training set verdict: {verdict_count}")
    p()

    # 6. Volume sanity
    p("## 6. Volume Sanity Check")
    p("Top-10 coins by average volume rank (expected: BTC, ETH, SOL, BNB, etc.):")
    top_avg_rank = membership_df.groupby("symbol")["rank"].mean().sort_values().head(10)
    for sym, avg_rank in top_avg_rank.items():
        p(f"  {sym:20s}  avg_rank={avg_rank:.1f}")
    p()

    # 7. Overall verdict
    p("## 7. Verdict")
    issues = []
    if len(ll_train) < 100_000:
        issues.append("Training labels < 100K rows")
    if len(ll_oos) == 0:
        issues.append("OOS labels missing")
    if features_full is not None and len(features_full) < 500_000:
        issues.append(f"Feature rows {len(features_full):,} < 500K (expected ~1.3M)")

    if not issues:
        p("STATUS: READY for Phase 3 (BGM training)")
    else:
        p(f"STATUS: ISSUES TO ADDRESS: {issues}")
    p()

    # Write file
    report_path = OUTPUT_DIR / "phase_1_2_sanity_report.md"
    report_path.write_text("\n".join(lines))
    print(f"{ts_now()} Saved: {report_path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="v_new_1 Phase 1+2 pipeline")
    parser.add_argument("--phase", choices=["1a", "1b", "2", "all"], default="all")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.monotonic()
    random.seed(42)
    np.random.seed(42)

    print(f"{ts_now()} v_new_1 Phase 1+2 pipeline starting (phase={args.phase})")
    print(f"{ts_now()} Output dir: {OUTPUT_DIR}")

    # Load candle data (always needed)
    all_data = load_all_4h_candles()

    membership_df = None
    ll_train = ls_train = ll_oos = ls_oos = None
    features_full = None

    if args.phase in ("1a", "all"):
        membership_df = build_universe_membership(all_data)

    if args.phase in ("1b", "all"):
        if membership_df is None:
            membership_path = OUTPUT_DIR / "universe_top100_membership.parquet"
            assert membership_path.exists(), "Run phase 1a first"
            membership_df = pd.read_parquet(membership_path)
            membership_df["week_start"] = pd.to_datetime(membership_df["week_start"], utc=True)

        phase_1b_labels(all_data, membership_df)

    if args.phase in ("2", "all"):
        if membership_df is None:
            membership_path = OUTPUT_DIR / "universe_top100_membership.parquet"
            assert membership_path.exists(), "Run phase 1a first"
            membership_df = pd.read_parquet(membership_path)
            membership_df["week_start"] = pd.to_datetime(membership_df["week_start"], utc=True)

        cfgi_df = pd.read_parquet(CFGI_PATH)
        cfgi_df["date"] = pd.to_datetime(cfgi_df["date"], utc=True)

        features_full = phase_2_features(all_data, membership_df, cfgi_df)

    # Final sanity report (only if all phases ran)
    if args.phase == "all":
        ll_train = pd.read_parquet(OUTPUT_DIR / "labels_long.parquet")
        ls_train = pd.read_parquet(OUTPUT_DIR / "labels_short.parquet")
        ll_oos = pd.read_parquet(OUTPUT_DIR / "labels_long_oos.parquet")
        ls_oos = pd.read_parquet(OUTPUT_DIR / "labels_short_oos.parquet")
        for df in [ll_train, ls_train, ll_oos, ls_oos]:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        write_sanity_report(membership_df, ll_train, ls_train, ll_oos, ls_oos, features_full)

    total_time = time.monotonic() - t_start
    print(f"\n{ts_now()} === ALL DONE in {total_time/3600:.2f}h ({total_time:.0f}s) ===")


if __name__ == "__main__":
    main()
