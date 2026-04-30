"""Recent-95d backtest — v8 BALANCED stack with proposed Filters A + B.

Adds two research-driven filters to the v8 BALANCED 4-detector stack and
runs side-by-side vs the unfiltered baseline (which matches the live
deployed code).

Filters
-------
A — BREADTH CAP (per-detector)
    If a single 15m bar produces more than BREADTH_CAP_PCT × eligible_universe
    fires of the SAME detector, drop the entire bar's signals. Addresses the
    "30 alts crossed up at the same 4h, then all SL'd together" concentration
    cluster observed live on 2026-04-29 12:00 UTC.

B — MIN SL FLOOR
    Replace the E2 stop with sl = max(2× ATR, MIN_SL_PCT × entry). When ATR
    is small (low-vol pre-pump regime), the 2× ATR stop becomes < 1% wide
    and gets noise-stopped. The TP arm (6× ATR) is unchanged so the R:R
    floor in noisy regimes shifts from ~3:1 down to ~2:1 — but we trade
    fewer noise-stops.

Each (detector × filter-config) is reported with the same monthly + train/
hold-out breakdown as backtest_v8_balanced_recent_v1.py.

Usage from services/python/:
    .venv/bin/python scripts/backtest_v8_balanced_recent_filters_v1.py
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

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# --- CONSTANTS -------------------------------------------------------------
SNAPSHOT_DATE = "2026-04-27"
HOLD_OUT_START = "2026-03-28"
RANDOM_SEED = 42

TOP_N_COINS = 100
MIN_QUOTE_VOLUME_24H = 500_000.0

COOLDOWN_BARS = 96
ATR_PERIOD = 14
EARLY_TREND_FRESH_BARS = 10 * 96
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
EMA_TREND_SPAN = 50

E2_ATR_SL_MULT = 2.0
E2_ATR_TP_MULT = 6.0
TIMEOUT_BARS = 1344
FEE_PCT = 0.0006
WORST_FILL_BUFFER = 0.005

SHIP_FLOOR_PF = 1.30

# --- FILTER PARAMS ---------------------------------------------------------
FILTER_A_BREADTH_CAP_PCT = 0.15  # drop 15m bar if > 15% of eligible universe fires same detector
FILTER_B_MIN_SL_PCT = 0.015      # SL floor = 1.5% of entry price


# --- SNAPSHOT LOADER (same as v1) -----------------------------------------
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
    if not manifest.exists():
        raise FileNotFoundError(f"missing MANIFEST.md at {manifest}")
    expected_c, expected_u = _parse_manifest_row(manifest.read_text(), date)
    if _sha256_file(candles_path) != expected_c:
        raise ValueError(f"candles sha mismatch")
    if _sha256_file(universe_path) != expected_u:
        raise ValueError(f"universe sha mismatch")
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)


# --- INDICATORS (same as v1) -----------------------------------------------

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


def _macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "signal": sig, "histogram": macd - sig})


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def _apply_cooldown(idxs, cooldown_bars):
    if not idxs:
        return []
    kept = [idxs[0]]
    for idx in idxs[1:]:
        if idx - kept[-1] >= cooldown_bars:
            kept.append(idx)
    return kept


def detect_macd_pullback_long(n, db_arr, h4_cross_up_15m):
    return [i for i in range(n) if h4_cross_up_15m[i] and db_arr[i]]


def detect_macd_pullback_short(n, dbear_arr, h4_cross_down_15m):
    return [i for i in range(n) if h4_cross_down_15m[i] and dbear_arr[i]]


def detect_macd_early_trend_short(n, dbear_arr, h4_cross_down_15m):
    flip = np.zeros(n, dtype=bool)
    if n > 0:
        flip[0] = dbear_arr[0]
        flip[1:] = dbear_arr[1:] & ~dbear_arr[:-1]
    flip_idx = np.where(flip)[0]
    fresh = np.zeros(n, dtype=bool)
    for start in flip_idx:
        end = min(n, start + EARLY_TREND_FRESH_BARS)
        fresh[start:end] = True
    return [i for i in range(n) if h4_cross_down_15m[i] and dbear_arr[i] and fresh[i]]


def detect_rsi_recovery_long(n, rsi_recovery_15m):
    return [i for i in range(n) if rsi_recovery_15m[i]]


def simulate_e2_exit(high, low, close, entry_idx, atr14_at_entry, direction,
                     min_sl_pct: float = 0.0):
    """Standard E2 (2× ATR SL, 6× ATR TP). With min_sl_pct, the SL is widened
    to max(2× ATR, min_sl_pct × entry) — Filter B."""
    n = len(close)
    if entry_idx >= n - 1:
        return None
    ep = float(close[entry_idx])
    if not np.isfinite(ep) or ep <= 0:
        return None
    if not np.isfinite(atr14_at_entry) or atr14_at_entry <= 0:
        return None

    sl_dist_atr = E2_ATR_SL_MULT * atr14_at_entry
    sl_dist_pct = min_sl_pct * ep
    sl_dist = max(sl_dist_atr, sl_dist_pct)

    if direction == "long":
        sl = max(ep - sl_dist, 0.0)
        tp = ep + E2_ATR_TP_MULT * atr14_at_entry
    else:
        sl = ep + sl_dist
        tp = max(ep - E2_ATR_TP_MULT * atr14_at_entry, 0.0)

    exit_idx = exit_price = exit_reason = None
    end = min(entry_idx + 1 + TIMEOUT_BARS, n)
    for j in range(entry_idx + 1, end):
        hi = float(high[j]); lo = float(low[j])
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        if direction == "long":
            if lo <= sl:
                exit_idx = j
                exit_price = max(sl - WORST_FILL_BUFFER * atr14_at_entry, 0.0)
                exit_reason = "stop_loss"; break
            if hi >= tp:
                exit_idx = j; exit_price = tp; exit_reason = "take_profit"; break
        else:
            if hi >= sl:
                exit_idx = j
                exit_price = sl + WORST_FILL_BUFFER * atr14_at_entry
                exit_reason = "stop_loss"; break
            if lo <= tp:
                exit_idx = j; exit_price = tp; exit_reason = "take_profit"; break
    if exit_idx is None:
        exit_idx = end - 1
        exit_price = float(close[exit_idx])
        if not np.isfinite(exit_price):
            return None
        exit_reason = "timeout"
    pnl_gross = (exit_price - ep) / ep if direction == "long" else (ep - exit_price) / ep
    return {
        "entry_idx": int(entry_idx), "exit_idx": int(exit_idx),
        "entry_price": float(ep), "exit_price": float(exit_price),
        "sl_price": float(sl), "tp_price": float(tp),
        "sl_dist_pct": float((ep - sl) / ep) if direction == "long" else float((sl - ep) / ep),
        "exit_reason": exit_reason,
        "pnl_pct": float(pnl_gross - 2 * FEE_PCT),
        "bars_held": int(exit_idx - entry_idx),
        "atr14_at_entry": float(atr14_at_entry),
    }


# --- METRICS ---------------------------------------------------------------

def _pf(x):
    w = x[x > 0].sum(); l = -x[x < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def detector_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "total_pnl_pct": 0.0, "avg_hold_bars": 0}
    return {
        "trades": int(len(df)),
        "pf": round(_pf(df["pnl_pct"]), 4),
        "wr": round(float((df["pnl_pct"] > 0).sum() / len(df)), 4),
        "total_pnl_pct": round(float(df["pnl_pct"].sum() * 100), 3),
        "avg_pnl_pct": round(float(df["pnl_pct"].mean() * 100), 4),
        "avg_hold_bars": int(df["bars_held"].median()),
    }


def split_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {}
    cutoff = pd.Timestamp(HOLD_OUT_START, tz="UTC")
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    out = {}
    for det in df["detector"].unique():
        d = df[df["detector"] == det]
        td = d[d["entry_time"] < cutoff]
        hd = d[d["entry_time"] >= cutoff]
        out[det] = {
            "all": detector_metrics(d),
            "train": detector_metrics(td),
            "hold_out": detector_metrics(hd),
            "passes_ship_floor_holdout": detector_metrics(hd)["pf"] >= SHIP_FLOOR_PF,
        }
    return out


def by_month_metrics(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return pd.DataFrame()
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["month"] = df["entry_time"].dt.strftime("%Y-%m")
    rows = []
    for (month, det), g in df.groupby(["month", "detector"]):
        rows.append({
            "month": month, "detector": det,
            "trades": len(g),
            "pf": round(_pf(g["pnl_pct"]), 3),
            "wr": round(float((g["pnl_pct"] > 0).sum() / len(g)) * 100, 1),
            "total_pnl_pct": round(g["pnl_pct"].sum() * 100, 2),
        })
    return pd.DataFrame(rows).sort_values(["month", "detector"]).reset_index(drop=True)


# --- DETECT + SIMULATE for one filter config ------------------------------

def run_one_config(candles_by_asset, asset_list, *, breadth_cap_pct: float, min_sl_pct: float):
    eligible_n = len(asset_list)
    breadth_cap = int(breadth_cap_pct * eligible_n) if breadth_cap_pct > 0 else 99999
    print(f"  config: breadth_cap={breadth_cap} ({breadth_cap_pct*100:.0f}% of {eligible_n}), min_sl_pct={min_sl_pct*100:.2f}%")

    # Pass 1: collect all signals across assets, keyed by 15m timestamp + detector
    raw = []  # list of dicts: detector, asset, entry_idx, ts (pd.Timestamp), direction, atr_at_entry, ... (for re-running simulate)

    for k_idx, asset in enumerate(asset_list):
        df = candles_by_asset.get(asset)
        if df is None or len(df) < 4 * 96:
            continue
        ts = df["timestamp"].to_numpy()
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        n = len(c)
        atr14 = _atr14(h, lo, c)
        ts_idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
        s_close = pd.Series(c, index=ts_idx)

        d_close = s_close.resample("1D").last().dropna()
        if len(d_close) >= 30:
            d_macd_df = _macd(d_close, 12, 26, 9)
            daily_bull = (d_macd_df["histogram"] > 0) & (d_macd_df["macd"] > d_macd_df["signal"])
            daily_bear = (d_macd_df["histogram"] < 0) & (d_macd_df["macd"] < d_macd_df["signal"])
            db_arr = daily_bull.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
            dbear_arr = daily_bear.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
        else:
            db_arr = np.zeros(n, dtype=bool); dbear_arr = np.zeros(n, dtype=bool)

        h4_close = s_close.resample("4h").last().dropna()
        if len(h4_close) >= 30:
            h4_macd_df = _macd(h4_close, 12, 26, 9)
            h4_above = (h4_macd_df["macd"] > h4_macd_df["signal"]).astype(bool)
            h4_below_prev = ~h4_above.shift(1).fillna(False)
            h4_below_prev2 = ~h4_above.shift(2).fillna(False)
            h4_cross_up = h4_above & h4_below_prev & h4_below_prev2
            h4_above_prev = h4_above.shift(1).fillna(False)
            h4_above_prev2 = h4_above.shift(2).fillna(False)
            h4_cross_down = (~h4_above) & h4_above_prev & h4_above_prev2
            cu_full = h4_cross_up.reindex(ts_idx, method="ffill").fillna(False)
            h4_cross_up_15m = (cu_full & ~cu_full.shift(1).fillna(False)).to_numpy(dtype=bool)
            cd_full = h4_cross_down.reindex(ts_idx, method="ffill").fillna(False)
            h4_cross_down_15m = (cd_full & ~cd_full.shift(1).fillna(False)).to_numpy(dtype=bool)
        else:
            h4_cross_up_15m = np.zeros(n, dtype=bool); h4_cross_down_15m = np.zeros(n, dtype=bool)

        if len(h4_close) >= RSI_PERIOD + EMA_TREND_SPAN:
            h4_rsi = _rsi(h4_close, RSI_PERIOD)
            h4_ema50 = _ema(h4_close, EMA_TREND_SPAN)
            rsi_cross_h4 = (
                (h4_rsi > RSI_OVERSOLD)
                & (h4_rsi.shift(1) <= RSI_OVERSOLD).fillna(False)
                & (h4_close > h4_ema50)
            )
            rsi_full = rsi_cross_h4.reindex(ts_idx, method="ffill").fillna(False)
            rsi_recovery_15m = (rsi_full & ~rsi_full.shift(1).fillna(False)).to_numpy(dtype=bool)
        else:
            rsi_recovery_15m = np.zeros(n, dtype=bool)

        e_pb_short = _apply_cooldown(detect_macd_pullback_short(n, dbear_arr, h4_cross_down_15m), COOLDOWN_BARS)
        e_et_short = _apply_cooldown(detect_macd_early_trend_short(n, dbear_arr, h4_cross_down_15m), COOLDOWN_BARS)
        e_pb_long  = _apply_cooldown(detect_macd_pullback_long(n, db_arr, h4_cross_up_15m), COOLDOWN_BARS)
        e_rsi_long = _apply_cooldown(detect_rsi_recovery_long(n, rsi_recovery_15m), COOLDOWN_BARS)

        for det_name, entries, direction in (
            ("macd_pullback_short", e_pb_short, "short"),
            ("macd_early_trend_short", e_et_short, "short"),
            ("macd_pullback_long", e_pb_long, "long"),
            ("rsi_recovery_long", e_rsi_long, "long"),
        ):
            for ei in entries:
                atr_e = float(atr14[ei]) if ei < len(atr14) and np.isfinite(atr14[ei]) else float("nan")
                raw.append({
                    "detector": det_name, "asset": asset, "direction": direction,
                    "entry_idx": int(ei), "entry_ts": ts_idx[ei],
                    "atr14_at_entry": atr_e,
                    "_h": h, "_l": lo, "_c": c,
                })

    print(f"  raw signals: {len(raw)} across {eligible_n} assets")

    # Filter A: per-detector breadth cap by 15m timestamp
    if breadth_cap_pct > 0 and raw:
        df_raw = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in raw])
        counts = df_raw.groupby(["entry_ts", "detector"]).size().reset_index(name="n")
        over_cap = counts[counts["n"] > breadth_cap]
        drop_keys = set(zip(over_cap["entry_ts"], over_cap["detector"]))
        n_dropped = sum(1 for r in raw if (r["entry_ts"], r["detector"]) in drop_keys)
        kept = [r for r in raw if (r["entry_ts"], r["detector"]) not in drop_keys]
        print(f"  Filter A: dropped {len(over_cap)} (ts, det) buckets — total {n_dropped} signals removed")
    else:
        kept = raw

    # Pass 2: simulate exit per kept signal
    all_trades = []
    for r in kept:
        h, lo, c = r["_h"], r["_l"], r["_c"]
        trade = simulate_e2_exit(h, lo, c, r["entry_idx"], r["atr14_at_entry"], r["direction"],
                                 min_sl_pct=min_sl_pct)
        if trade is None:
            continue
        ts_idx_local = pd.DatetimeIndex(pd.to_datetime(
            np.array([r["entry_ts"]] + [None]*0), utc=True
        ))
        trade.update({
            "detector": r["detector"], "asset": r["asset"], "direction": r["direction"],
            "entry_time": r["entry_ts"].isoformat(),
            # exit_time looked up via entry+offset:
            "_exit_offset_bars": trade["bars_held"],
        })
        all_trades.append(trade)
    print(f"  trades after exits: {len(all_trades)}")
    return pd.DataFrame(all_trades)


# --- WRITE-OUT helpers -----------------------------------------------------

def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def build_run_id():
    sha = _git_sha()
    short_sha = sha[:8] if sha != "unknown" else "nogit"
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"backtest_v8_balanced_recent_filters_v1_{short_sha}_{ts}"


def main() -> None:
    started = time.monotonic()
    np.random.seed(RANDOM_SEED)

    print(f"[{time.monotonic() - started:6.1f}s] loading snapshot {SNAPSHOT_DATE}...")
    candles_all, universe = load_snapshot(SNAPSHOT_DATE)

    universe = universe.copy()
    universe["asset_key"] = universe["symbol"].str.replace("_", "", regex=False)
    eligible_uni = universe[
        (universe["is_leveraged"] == False)  # noqa: E712
        & (universe["in_delisting"] == False)  # noqa: E712
        & (universe["quote_volume_24h"] >= MIN_QUOTE_VOLUME_24H)
    ].nlargest(TOP_N_COINS, "quote_volume_24h")
    eligible_assets = set(eligible_uni["asset_key"].astype(str))
    candles = candles_all[candles_all["asset"].isin(eligible_assets)].copy()
    candles_by_asset = {a: g.sort_values("timestamp").reset_index(drop=True)
                        for a, g in candles.groupby("asset")}
    asset_list = sorted(eligible_assets)
    print(f"[{time.monotonic() - started:6.1f}s] eligible: {len(asset_list)} assets, {len(candles):,} candles")

    # === BASELINE (filters OFF) ===
    print(f"\n[{time.monotonic() - started:6.1f}s] === BASELINE (no filters) ===")
    baseline = run_one_config(candles_by_asset, asset_list, breadth_cap_pct=0.0, min_sl_pct=0.0)

    # === FILTERED (A + B) ===
    print(f"\n[{time.monotonic() - started:6.1f}s] === FILTER A + B ===")
    filtered = run_one_config(candles_by_asset, asset_list,
                              breadth_cap_pct=FILTER_A_BREADTH_CAP_PCT,
                              min_sl_pct=FILTER_B_MIN_SL_PCT)

    # === COMPARISON ===
    print(f"\n[{time.monotonic() - started:6.1f}s] === COMPARISON ===\n")

    base_split = split_metrics(baseline)
    filt_split = split_metrics(filtered)
    base_month = by_month_metrics(baseline)
    filt_month = by_month_metrics(filtered)

    print("=" * 110)
    print("PER-DETECTOR — full 95-day window")
    print("=" * 110)
    hdr = f"{'Detector':30s} | {'BASELINE n':>10s} {'PF':>6s} {'hold PF':>7s} | {'A+B n':>6s} {'PF':>6s} {'hold PF':>7s} | {'Δ PF':>6s} {'Δ hold':>7s}"
    print(hdr); print("-" * 110)
    detectors = ["macd_pullback_short", "macd_early_trend_short", "macd_pullback_long", "rsi_recovery_long"]
    for det in detectors:
        b = base_split.get(det, {})
        f = filt_split.get(det, {})
        b_n = b.get("all", {}).get("trades", 0)
        b_pf = b.get("all", {}).get("pf", 0.0)
        b_h = b.get("hold_out", {}).get("pf", 0.0)
        f_n = f.get("all", {}).get("trades", 0)
        f_pf = f.get("all", {}).get("pf", 0.0)
        f_h = f.get("hold_out", {}).get("pf", 0.0)
        d_pf = f_pf - b_pf
        d_h = f_h - b_h
        print(f"{det:30s} | {b_n:>10d} {b_pf:>6.2f} {b_h:>7.2f} | {f_n:>6d} {f_pf:>6.2f} {f_h:>7.2f} | {d_pf:>+6.2f} {d_h:>+7.2f}")

    # Pullback long monthly comparison (the focus)
    print()
    print("=" * 110)
    print("macd_pullback_long — month-by-month")
    print("=" * 110)
    print(f"{'month':10s} | {'BASELINE n':>10s} {'PF':>6s} {'WR':>5s} {'pnl%':>7s} | {'A+B n':>6s} {'PF':>6s} {'WR':>5s} {'pnl%':>7s}")
    print("-" * 110)
    bm = base_month[base_month["detector"] == "macd_pullback_long"].set_index("month")
    fm = filt_month[filt_month["detector"] == "macd_pullback_long"].set_index("month")
    months = sorted(set(bm.index).union(fm.index))
    for m in months:
        b = bm.loc[m] if m in bm.index else None
        f = fm.loc[m] if m in fm.index else None
        b_n = int(b["trades"]) if b is not None else 0
        b_pf = float(b["pf"]) if b is not None else 0.0
        b_wr = float(b["wr"]) if b is not None else 0.0
        b_pnl = float(b["total_pnl_pct"]) if b is not None else 0.0
        f_n = int(f["trades"]) if f is not None else 0
        f_pf = float(f["pf"]) if f is not None else 0.0
        f_wr = float(f["wr"]) if f is not None else 0.0
        f_pnl = float(f["total_pnl_pct"]) if f is not None else 0.0
        print(f"{m:10s} | {b_n:>10d} {b_pf:>6.2f} {b_wr:>4.1f}% {b_pnl:>+7.2f} | {f_n:>6d} {f_pf:>6.2f} {f_wr:>4.1f}% {f_pnl:>+7.2f}")

    # Save artifacts
    run_id = build_run_id()
    out_dir = Path(__file__).resolve().parents[1] / "results" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    if len(baseline):
        baseline.to_csv(out_dir / "baseline_trades.csv", index=False)
        base_month.to_csv(out_dir / "baseline_by_month.csv", index=False)
    if len(filtered):
        filtered.to_csv(out_dir / "filtered_trades.csv", index=False)
        filt_month.to_csv(out_dir / "filtered_by_month.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "snapshot_date": SNAPSHOT_DATE,
            "hold_out_start": HOLD_OUT_START,
            "filters": {
                "A_breadth_cap_pct": FILTER_A_BREADTH_CAP_PCT,
                "B_min_sl_pct": FILTER_B_MIN_SL_PCT,
            },
            "baseline": base_split,
            "filtered": filt_split,
        }, f, indent=2)

    print()
    print(f"=== DONE — wall_time {time.monotonic() - started:.1f}s ===")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
