"""Build per-trade × per-cell PnL matrix for `macd_early_trend_short` (3-year window).

Clone of build_macd_pullback_short_3y_per_trade_matrix_v1.py with the
early-trend restriction: pullback_short signals are only kept within the
first ≤10 days after the daily MACD bear-flip (EARLY_TREND_FRESH_BARS = 10×96).

Output:
  results/<run_id>/per_trade_matrix.parquet — one row per entry, columns:
    asset, entry_time, entry_idx, atr15m, atr1h, atr4h, cls_score, btc_score,
    is_hold (entry_time >= 2026-01-01),
    pnl_<atr_tf>_<sl>_<tp> for each of 30 candidate cells.

Run from services/python/:
    .venv/bin/python scripts/build_macd_early_trend_short_3y_per_trade_matrix_v1.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- Config ---------------------------------------------------------------

SNAPSHOT_DATE = "2026-04-01"
TOP_N_COINS = 100
MIN_QUOTE_VOLUME_24H = 500_000.0
COOLDOWN_BARS = 96
ATR_PERIOD = 14
TIMEOUT_BARS = 1344
FEE_PCT = 0.0006
WORST_FILL_BUFFER = 0.005
HOLD_OUT_START = "2026-01-01"
EARLY_TREND_FRESH_BARS = 10 * 96  # ≤10d since bear-flip

DETECTOR = "macd_early_trend_short"
DIRECTION = "short"

# Candidate cell menu — 30 cells across 3 ATR timeframes
CELLS: list[tuple[str, float, float]] = (
    [("15m", 1.5, tp) for tp in (2.0, 3.0, 4.0, 5.0, 6.0)]
    + [("15m", 2.0, tp) for tp in (2.0, 3.0, 4.0, 5.0, 6.0)]
    + [("1h",  1.5, tp) for tp in (3.0, 4.0, 5.0, 6.0, 8.0)]
    + [("1h",  2.0, tp) for tp in (3.0, 4.0, 5.0, 6.0, 8.0)]
    + [("4h",  1.5, tp) for tp in (4.0, 5.0, 6.0, 8.0, 10.0)]
    + [("4h",  2.0, tp) for tp in (4.0, 5.0, 6.0, 8.0, 10.0)]
)


def cell_col(tf: str, sl: float, tp: float) -> str:
    return f"pnl_{tf}_{sl:g}_{tp:g}"


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "v8_classifier_3y"
MODEL_STEM = "macd_early_trend_short_histgbm_3y_v1"
FEATURES_PARQUET = (
    Path(__file__).resolve().parents[1] / "data" / "training" /
    f"v8_trades_with_features_{SNAPSHOT_DATE}.parquet"
)
FEATURE_COLS = [
    "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
    "close_to_high50_atr", "close_to_low50_atr",
    "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
    "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
    "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
    "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
    "breadth_up", "breadth_down", "signals_same_15m_same_detector",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
_SNAPSHOTS_DIR = Path(__file__).resolve().parents[3] / "data" / "snapshots"


# --- Helpers --------------------------------------------------------------

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
        if len(parts) < 5 or not _DATE_RE.fullmatch(parts[0]):
            continue
        if parts[0] != date:
            continue
        return parts[2], parts[4]
    raise ValueError(f"date {date} not in MANIFEST.md")


def load_snapshot(date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = _SNAPSHOTS_DIR / "MANIFEST.md"
    candles_path = _SNAPSHOTS_DIR / f"candles_15m_{date}.parquet"
    universe_path = _SNAPSHOTS_DIR / f"universe_{date}.parquet"
    expected_c, expected_u = _parse_manifest_row(manifest.read_text(), date)
    actual_c = _sha256_file(candles_path)
    actual_u = _sha256_file(universe_path)
    if actual_c != expected_c:
        raise ValueError("candles sha mismatch")
    if actual_u != expected_u:
        raise ValueError("universe sha mismatch")
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)


def _atr14(high, low, close):
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.full(n, np.nan)
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    return atr


def _macd(close, fast=12, slow=26, signal=9):
    ef = close.ewm(span=fast, adjust=False).mean()
    es = close.ewm(span=slow, adjust=False).mean()
    macd = ef - es
    sig = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "signal": sig, "histogram": macd - sig})


def _apply_cooldown(idxs, cooldown_bars):
    if not idxs:
        return []
    kept = [idxs[0]]
    for idx in idxs[1:]:
        if idx - kept[-1] >= cooldown_bars:
            kept.append(idx)
    return kept


def simulate_e2_exit(high, low, close, entry_idx, atr14, sl_mult, tp_mult, direction):
    n = len(close)
    if entry_idx >= n - 1:
        return None
    ep = float(close[entry_idx])
    if not np.isfinite(ep) or ep <= 0 or not np.isfinite(atr14) or atr14 <= 0:
        return None
    if direction == "long":
        sl = max(ep - sl_mult * atr14, 0.0)
        tp = ep + tp_mult * atr14
    else:
        sl = ep + sl_mult * atr14
        tp = max(ep - tp_mult * atr14, 0.0)
    end = min(entry_idx + 1 + TIMEOUT_BARS, n)
    exit_idx = exit_price = exit_reason = None
    for j in range(entry_idx + 1, end):
        hi = float(high[j]); lo = float(low[j])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        if direction == "long":
            if lo <= sl:
                exit_idx = j
                exit_price = max(sl - WORST_FILL_BUFFER * atr14, 0.0)
                exit_reason = "stop_loss"; break
            if hi >= tp:
                exit_idx = j; exit_price = tp; exit_reason = "take_profit"; break
        else:
            if hi >= sl:
                exit_idx = j
                exit_price = sl + WORST_FILL_BUFFER * atr14
                exit_reason = "stop_loss"; break
            if lo <= tp:
                exit_idx = j; exit_price = tp; exit_reason = "take_profit"; break
    if exit_idx is None:
        exit_idx = end - 1
        exit_price = float(close[exit_idx])
        if not np.isfinite(exit_price):
            return None
    pnl_gross = (exit_price - ep) / ep if direction == "long" else (ep - exit_price) / ep
    return float(pnl_gross - 2 * FEE_PCT)


def score_to_size(meta, score):
    quantiles = np.asarray(meta["train_score_quantiles"], dtype=float)
    n_q = len(quantiles)
    pos = int(np.searchsorted(quantiles, score, side="right"))
    rank = max(0.0, min(1.0, pos / n_q))
    size_low = float(meta.get("size_low", 0.5))
    size_high = float(meta.get("size_high", 1.5))
    return size_low + (size_high - size_low) * rank


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _git_dirty() -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"]).strip())
    except Exception:
        return False


def build_run_id() -> str:
    sha = _git_sha()
    short_sha = sha[:8] if sha != "unknown" else "nogit"
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"adaptive_exit_per_trade_matrix_early_trend_short_v1_{short_sha}_{ts}"


# --- Main -----------------------------------------------------------------

def main() -> None:
    started = time.monotonic()
    git_dirty_at_start = _git_dirty()
    if git_dirty_at_start:
        print("WARNING: git working tree is dirty.", file=sys.stderr)

    print(f"[{time.monotonic() - started:6.1f}s] loading snapshot {SNAPSHOT_DATE}...")
    candles_all, universe = load_snapshot(SNAPSHOT_DATE)
    print(f"  candles: {len(candles_all):,} rows, {candles_all['asset'].nunique()} assets")

    universe = universe.copy()
    universe["asset_key"] = universe["symbol"].str.replace("_", "", regex=False)
    eligible_uni = universe[
        (universe["is_leveraged"] == False)  # noqa: E712
        & (universe["in_delisting"] == False)  # noqa: E712
        & (universe["quote_volume_24h"] >= MIN_QUOTE_VOLUME_24H)
    ].nlargest(TOP_N_COINS, "quote_volume_24h")
    eligible_assets = set(eligible_uni["asset_key"].astype(str))
    candles = candles_all[candles_all["asset"].isin(eligible_assets)].copy()
    print(f"[{time.monotonic() - started:6.1f}s] eligible: {candles['asset'].nunique()} assets")

    print(f"[{time.monotonic() - started:6.1f}s] loading classifier + features...")
    model = joblib.load(MODEL_DIR / f"{MODEL_STEM}.joblib")
    with open(MODEL_DIR / f"{MODEL_STEM}_meta.json") as f:
        meta = json.load(f)
    feat = pd.read_parquet(FEATURES_PARQUET)
    feat = feat[feat["detector"] == DETECTOR].copy()
    feat["entry_time"] = pd.to_datetime(feat["entry_time"], utc=True)
    feat_keyed = feat.set_index(["asset", "entry_time"])

    candles_by_asset = {a: g.sort_values("timestamp").reset_index(drop=True)
                        for a, g in candles.groupby("asset")}
    asset_list = sorted(eligible_assets)

    # --- Detect entries + classify ---
    print(f"[{time.monotonic() - started:6.1f}s] detect entries + score classifier...")
    entries: list[dict] = []
    asset_arrays: dict[str, dict] = {}
    asset_atrs: dict[str, dict] = {}
    n_features_missing = 0

    for k_idx, asset in enumerate(asset_list):
        df = candles_by_asset.get(asset)
        if df is None or len(df) < 4 * 96:
            continue
        ts = df["timestamp"].to_numpy()
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        n = len(c)
        ts_idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
        s_close = pd.Series(c, index=ts_idx)

        # ATRs at 3 timeframes
        atr15m = _atr14(h, lo, c)

        df_h1 = pd.DataFrame({
            "high": pd.Series(h, index=ts_idx),
            "low": pd.Series(lo, index=ts_idx),
            "close": pd.Series(c, index=ts_idx),
        }).resample("1h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(df_h1) >= ATR_PERIOD + 1:
            atr_h1 = _atr14(df_h1["high"].to_numpy(), df_h1["low"].to_numpy(), df_h1["close"].to_numpy())
            atr1h = (pd.Series(atr_h1, index=df_h1.index).shift(1)
                     .reindex(ts_idx, method="ffill").to_numpy())
        else:
            atr1h = np.full(n, np.nan)

        df_h4 = pd.DataFrame({
            "high": pd.Series(h, index=ts_idx),
            "low": pd.Series(lo, index=ts_idx),
            "close": pd.Series(c, index=ts_idx),
        }).resample("4h").agg({"high": "max", "low": "min", "close": "last"}).dropna()
        if len(df_h4) >= ATR_PERIOD + 1:
            atr_h4 = _atr14(df_h4["high"].to_numpy(), df_h4["low"].to_numpy(), df_h4["close"].to_numpy())
            atr4h = (pd.Series(atr_h4, index=df_h4.index).shift(1)
                     .reindex(ts_idx, method="ffill").to_numpy())
        else:
            atr4h = np.full(n, np.nan)

        # Daily MACD bear
        d_close = s_close.resample("1D").last().dropna()
        if len(d_close) >= 30:
            dm = _macd(d_close)
            dbear_arr = ((dm["histogram"] < 0) & (dm["macd"] < dm["signal"]))\
                .reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
        else:
            dbear_arr = np.zeros(n, dtype=bool)

        # 4h MACD cross-down ≥2 prior above
        h4_close = s_close.resample("4h").last().dropna()
        if len(h4_close) >= 30:
            hm = _macd(h4_close)
            h4_above = (hm["macd"] > hm["signal"]).astype(bool)
            h4_above_prev = h4_above.shift(1).fillna(False)
            h4_above_prev2 = h4_above.shift(2).fillna(False)
            cross_down = (~h4_above) & h4_above_prev & h4_above_prev2
            cd_full = cross_down.reindex(ts_idx, method="ffill").fillna(False)
            cd_prev = cd_full.shift(1).fillna(False)
            h4_cross_down_15m = (cd_full & ~cd_prev).to_numpy(dtype=bool)
        else:
            h4_cross_down_15m = np.zeros(n, dtype=bool)

        # Early-trend restriction: ≤10d (EARLY_TREND_FRESH_BARS bars) since daily MACD bear-flip
        flip = np.zeros(n, dtype=bool)
        if n > 0:
            flip[0] = dbear_arr[0]
            flip[1:] = dbear_arr[1:] & ~dbear_arr[:-1]
        flip_idx = np.where(flip)[0]
        fresh = np.zeros(n, dtype=bool)
        for start in flip_idx:
            end = min(n, start + EARLY_TREND_FRESH_BARS)
            fresh[start:end] = True

        raw = [i for i in range(n) if h4_cross_down_15m[i] and dbear_arr[i] and fresh[i]]
        ent_idxs = _apply_cooldown(raw, COOLDOWN_BARS)
        if not ent_idxs:
            continue
        asset_arrays[asset] = {"h": h, "l": lo, "c": c, "ts": ts}
        asset_atrs[asset] = {"15m": atr15m, "1h": atr1h, "4h": atr4h}

        for ei in ent_idxs:
            atr15 = float(atr15m[ei]) if ei < len(atr15m) and np.isfinite(atr15m[ei]) else float("nan")
            if not np.isfinite(atr15) or atr15 <= 0:
                continue
            atr1 = float(atr1h[ei]) if ei < len(atr1h) and np.isfinite(atr1h[ei]) else float("nan")
            atr4 = float(atr4h[ei]) if ei < len(atr4h) and np.isfinite(atr4h[ei]) else float("nan")
            entry_time = (pd.Timestamp(ts[ei]).tz_localize("UTC")
                          if pd.Timestamp(ts[ei]).tzinfo is None else pd.Timestamp(ts[ei]))
            try:
                fr = feat_keyed.loc[(asset, entry_time)]
                if isinstance(fr, pd.DataFrame):
                    fr = fr.iloc[0]
                x = fr[FEATURE_COLS].to_numpy(dtype=float).reshape(1, -1)
                if not np.all(np.isfinite(x)):
                    cls_score = float("nan")
                else:
                    cls_score = float(model.predict_proba(x)[0, 1])
                btc_score_v = float(fr["btc_score"]) if "btc_score" in fr.index else float("nan")
            except KeyError:
                n_features_missing += 1
                cls_score = float("nan")
                btc_score_v = float("nan")

            entries.append({
                "asset": asset,
                "entry_idx": ei,
                "entry_time": entry_time,
                "atr15m": atr15,
                "atr1h": atr1,
                "atr4h": atr4,
                "cls_score": cls_score,
                "btc_score": btc_score_v,
            })

        if (k_idx + 1) % 20 == 0 or k_idx == len(asset_list) - 1:
            print(f"[{time.monotonic() - started:6.1f}s] {k_idx + 1}/{len(asset_list)}  "
                  f"entries={len(entries)} feat_miss={n_features_missing}")

    df_entries = pd.DataFrame(entries)
    cutoff = pd.Timestamp(HOLD_OUT_START, tz="UTC")
    df_entries["is_hold"] = df_entries["entry_time"] >= cutoff
    print(f"[{time.monotonic() - started:6.1f}s] entries: {len(df_entries)} "
          f"(hold-out: {int(df_entries['is_hold'].sum())}), feat_miss: {n_features_missing}")

    # --- Sweep cells, fill matrix ---
    print(f"[{time.monotonic() - started:6.1f}s] sweeping {len(CELLS)} cells × {len(df_entries)} entries...")
    matrix = np.full((len(df_entries), len(CELLS)), np.nan)

    atr_lookup = {"15m": "atr15m", "1h": "atr1h", "4h": "atr4h"}
    for ci, (tf, sl_mult, tp_mult) in enumerate(CELLS):
        atr_col = atr_lookup[tf]
        for i, e in enumerate(df_entries.itertuples()):
            arrs = asset_arrays.get(e.asset)
            if arrs is None:
                continue
            atr_val = getattr(e, atr_col)
            if not np.isfinite(atr_val) or atr_val <= 0:
                continue
            pnl = simulate_e2_exit(arrs["h"], arrs["l"], arrs["c"], e.entry_idx,
                                   atr_val, sl_mult, tp_mult, DIRECTION)
            if pnl is not None:
                matrix[i, ci] = pnl
        col_pnl_sum = np.nansum(matrix[:, ci]) * 100
        col_n = int(np.sum(~np.isnan(matrix[:, ci])))
        print(f"[{time.monotonic() - started:6.1f}s]  cell {tf} SL={sl_mult} TP={tp_mult}  "
              f"n={col_n} sum_pnl%={col_pnl_sum:.0f}")

    # Build output dataframe
    out = df_entries.copy()
    for ci, (tf, sl_mult, tp_mult) in enumerate(CELLS):
        out[cell_col(tf, sl_mult, tp_mult)] = matrix[:, ci]

    # Add cls_mult column for sizing reference
    def s2s(s):
        if not np.isfinite(s):
            return 1.0
        quantiles = np.asarray(meta["train_score_quantiles"], dtype=float)
        n_q = len(quantiles)
        pos = int(np.searchsorted(quantiles, s, side="right"))
        rank = max(0.0, min(1.0, pos / n_q))
        size_low = float(meta.get("size_low", 0.5))
        size_high = float(meta.get("size_high", 1.5))
        return size_low + (size_high - size_low) * rank

    out["cls_mult"] = out["cls_score"].apply(s2s)

    # Output
    run_id = build_run_id()
    out_dir = Path(__file__).resolve().parents[1] / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True)
    out.to_parquet(out_dir / "per_trade_matrix.parquet", index=False)
    metadata = {
        "snapshot_date": SNAPSHOT_DATE,
        "detector": DETECTOR,
        "direction": DIRECTION,
        "n_entries": int(len(out)),
        "n_features_missing": int(n_features_missing),
        "n_cells": len(CELLS),
        "cells": [{"tf": tf, "sl": sl, "tp": tp, "col": cell_col(tf, sl, tp)} for tf, sl, tp in CELLS],
        "hold_out_start": HOLD_OUT_START,
        "early_trend_fresh_bars": EARLY_TREND_FRESH_BARS,
        "wall_time_seconds": time.monotonic() - started,
        "git_sha": _git_sha(),
        "git_dirty": git_dirty_at_start,
    }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    sanity_col = cell_col("15m", 2.0, 6.0)
    sanity_sum = float(out[sanity_col].sum() * 100)
    sanity_n = int(out[sanity_col].notna().sum())
    print(f"\nSANITY: live cell {sanity_col}  n={sanity_n}  total_pnl%={sanity_sum:.1f}")

    print(f"\n=== DONE wall_time {time.monotonic() - started:.1f}s ===")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
