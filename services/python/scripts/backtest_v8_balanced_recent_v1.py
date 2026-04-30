"""Recent-90d backtest for the v8 BALANCED 4-detector stack — month-by-month.

Validates the *deployed* live detector code against the latest snapshot
(2026-04-27 → covers ~2026-01-22 → 2026-04-26). The 90-day "recent" run
52dc5d18 only covered macd_pullback_long; this one extends to all four
detectors deployed in paper-v8 + adds calendar-month aggregation.

Detectors (deploy = research, all E2 exits):
  1. macd_pullback_short_e2   — daily MACD bear + 4h MACD cross-DOWN ≥2 prior above
  2. macd_early_trend_short_e2 — pullback_short restricted to first 10d after bear-flip
  3. macd_pullback_long_e2     — daily MACD bull + 4h MACD cross-UP ≥2 prior below
  4. rsi_recovery_long_e2      — 4h RSI(14) cross from ≤30 to >30 + close > 4h EMA(50)

Exit (E2 only): SL = entry ± 2× ATR, TP = entry ± 6× ATR, timeout 1344 bars (14d).
Fees: 0.06% × 2 per trade. Slippage: WORST_FILL_BUFFER 0.005 past SL.

Universe: top 100 USDT futures by quote_volume_24h, no leveraged tokens.
Cooldown: 96 bars (24h) per (symbol, detector).

Outputs to results/<run_id>/:
  - trades.csv          — every trade with entry/exit
  - metrics.json        — overall + per-detector + per-month breakdown
  - by_month.csv        — month × detector grid (PF, WR, n, total_pnl_pct)

Usage from services/python/:
    .venv/bin/python scripts/backtest_v8_balanced_recent_v1.py
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
HOLD_OUT_START = "2026-03-28"  # last 30d as hold-out for ship-floor check
RANDOM_SEED = 42

# Universe filter
TOP_N_COINS = 100
MIN_QUOTE_VOLUME_24H = 500_000.0

# Detector params (locked, deploy = research)
COOLDOWN_BARS = 96  # 24h
ATR_PERIOD = 14
EARLY_TREND_FRESH_BARS = 10 * 96  # ≤10d since bear-flip
RSI_PERIOD = 14
RSI_OVERSOLD = 30.0
EMA_TREND_SPAN = 50

# Exit params (E2 only, locked)
E2_ATR_SL_MULT = 2.0
E2_ATR_TP_MULT = 6.0
TIMEOUT_BARS = 1344  # 14d
FEE_PCT = 0.0006
WORST_FILL_BUFFER = 0.005

# Ship floor
SHIP_FLOOR_PF = 1.30


# --- SNAPSHOT LOADER (canonical, copied from backtest_template.py) --------
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
    if not candles_path.exists():
        raise FileNotFoundError(f"snapshot candles missing: {candles_path}")
    if not universe_path.exists():
        raise FileNotFoundError(f"snapshot universe missing: {universe_path}")
    expected_c, expected_u = _parse_manifest_row(manifest.read_text(), date)
    actual_c = _sha256_file(candles_path)
    actual_u = _sha256_file(universe_path)
    if actual_c != expected_c:
        raise ValueError(f"candles sha mismatch: expected {expected_c}, got {actual_c}")
    if actual_u != expected_u:
        raise ValueError(f"universe sha mismatch: expected {expected_u}, got {actual_u}")
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)


# --- INDICATORS ------------------------------------------------------------

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


# --- DETECTORS (mirror live scanner code bit-for-bit) --------------------

def detect_macd_pullback_long(n, db_arr, h4_cross_up_15m):
    return [i for i in range(n) if h4_cross_up_15m[i] and db_arr[i]]


def detect_macd_pullback_short(n, dbear_arr, h4_cross_down_15m):
    return [i for i in range(n) if h4_cross_down_15m[i] and dbear_arr[i]]


def detect_macd_early_trend_short(n, dbear_arr, h4_cross_down_15m):
    """Pullback short restricted to ≤10d since bear-flip."""
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


def simulate_e2_exit(high, low, close, entry_idx, atr14_at_entry, direction):
    n = len(close)
    if entry_idx >= n - 1:
        return None
    ep = float(close[entry_idx])
    if not np.isfinite(ep) or ep <= 0:
        return None
    if not np.isfinite(atr14_at_entry) or atr14_at_entry <= 0:
        return None
    if direction == "long":
        sl = max(ep - E2_ATR_SL_MULT * atr14_at_entry, 0.0)
        tp = ep + E2_ATR_TP_MULT * atr14_at_entry
    else:
        sl = ep + E2_ATR_SL_MULT * atr14_at_entry
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
        "exit_reason": exit_reason,
        "pnl_pct": float(pnl_gross - 2 * FEE_PCT),
        "bars_held": int(exit_idx - entry_idx),
        "atr14_at_entry": float(atr14_at_entry),
    }


# --- METRICS ---------------------------------------------------------------

def _pf(x):
    w = x[x > 0].sum(); l = -x[x < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _wr(x):
    if len(x) == 0:
        return 0.0
    return float((x > 0).sum() / len(x))


def detector_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "total_pnl_pct": 0.0}
    return {
        "trades": int(len(df)),
        "pf": round(_pf(df["pnl_pct"]), 4),
        "wr": round(_wr(df["pnl_pct"]), 4),
        "total_pnl_pct": round(float(df["pnl_pct"].sum() * 100), 3),
        "avg_pnl_pct": round(float(df["pnl_pct"].mean() * 100), 4),
        "median_hold_bars": int(df["bars_held"].median()) if "bars_held" in df.columns else 0,
    }


def split_metrics(df: pd.DataFrame) -> dict:
    cutoff = pd.Timestamp(HOLD_OUT_START, tz="UTC")
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    train_df = df[df["entry_time"] < cutoff]
    hold_df = df[df["entry_time"] >= cutoff]
    out = {}
    for det in df["detector"].unique():
        td = train_df[train_df["detector"] == det]
        hd = hold_df[hold_df["detector"] == det]
        ad = df[df["detector"] == det]
        out[det] = {
            "all": detector_metrics(ad),
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
            "month": month,
            "detector": det,
            "trades": len(g),
            "pf": round(_pf(g["pnl_pct"]), 3),
            "wr": round(_wr(g["pnl_pct"]) * 100, 1),
            "total_pnl_pct": round(g["pnl_pct"].sum() * 100, 2),
            "avg_pnl_pct": round(g["pnl_pct"].mean() * 100, 3),
            "median_hold_bars": int(g["bars_held"].median()),
        })
    bm = pd.DataFrame(rows)
    return bm.sort_values(["month", "detector"]).reset_index(drop=True)


# --- WRITE-OUT -------------------------------------------------------------

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
    return f"backtest_v8_balanced_recent_v1_{short_sha}_{ts}"


# --- MAIN ------------------------------------------------------------------

def main() -> None:
    started = time.monotonic()
    np.random.seed(RANDOM_SEED)
    git_dirty_at_start = _git_dirty()
    if git_dirty_at_start:
        print("WARNING: git working tree is dirty.", file=sys.stderr)

    print(f"[{time.monotonic() - started:6.1f}s] loading snapshot {SNAPSHOT_DATE}...")
    candles_all, universe = load_snapshot(SNAPSHOT_DATE)
    print(f"  candles: {len(candles_all):,} rows, {candles_all['asset'].nunique()} assets")

    # Universe filter (top 100 by 24h quote volume, no leveraged, not delisting)
    universe = universe.copy()
    universe["asset_key"] = universe["symbol"].str.replace("_", "", regex=False)
    eligible_uni = universe[
        (universe["is_leveraged"] == False)  # noqa: E712
        & (universe["in_delisting"] == False)  # noqa: E712
        & (universe["quote_volume_24h"] >= MIN_QUOTE_VOLUME_24H)
    ].nlargest(TOP_N_COINS, "quote_volume_24h")
    eligible_assets = set(eligible_uni["asset_key"].astype(str))
    candles = candles_all[candles_all["asset"].isin(eligible_assets)].copy()
    n_assets = candles["asset"].nunique()
    print(f"[{time.monotonic() - started:6.1f}s] eligible: {n_assets} assets ({len(candles):,} candles)")

    candles_by_asset = {a: g.sort_values("timestamp").reset_index(drop=True)
                        for a, g in candles.groupby("asset")}
    asset_list = sorted(eligible_assets)
    print(f"[{time.monotonic() - started:6.1f}s] running 4 detectors on {len(asset_list)} assets...")

    funnel = {
        "macd_pullback_short": 0,
        "macd_early_trend_short": 0,
        "macd_pullback_long": 0,
        "rsi_recovery_long": 0,
    }
    all_trades: list[dict] = []

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

        # --- Daily MACD (bull / bear) ---
        d_close = s_close.resample("1D").last().dropna()
        if len(d_close) >= 30:
            d_macd_df = _macd(d_close, 12, 26, 9)
            daily_bull_d = (d_macd_df["histogram"] > 0) & (d_macd_df["macd"] > d_macd_df["signal"])
            daily_bear_d = (d_macd_df["histogram"] < 0) & (d_macd_df["macd"] < d_macd_df["signal"])
            db_arr = daily_bull_d.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
            dbear_arr = daily_bear_d.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
        else:
            db_arr = np.zeros(n, dtype=bool)
            dbear_arr = np.zeros(n, dtype=bool)

        # --- 4H MACD cross-up / cross-down (≥2 prior bars opposite) ---
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
            cu_prev = cu_full.shift(1).fillna(False)
            h4_cross_up_15m = (cu_full & ~cu_prev).to_numpy(dtype=bool)
            cd_full = h4_cross_down.reindex(ts_idx, method="ffill").fillna(False)
            cd_prev = cd_full.shift(1).fillna(False)
            h4_cross_down_15m = (cd_full & ~cd_prev).to_numpy(dtype=bool)
        else:
            h4_cross_up_15m = np.zeros(n, dtype=bool)
            h4_cross_down_15m = np.zeros(n, dtype=bool)

        # --- 4H RSI cross-up + EMA50 trend filter (rsi_recovery_long) ---
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

        # --- Detect entries (with cooldown) ---
        e_pb_short = _apply_cooldown(detect_macd_pullback_short(n, dbear_arr, h4_cross_down_15m), COOLDOWN_BARS)
        e_et_short = _apply_cooldown(detect_macd_early_trend_short(n, dbear_arr, h4_cross_down_15m), COOLDOWN_BARS)
        e_pb_long = _apply_cooldown(detect_macd_pullback_long(n, db_arr, h4_cross_up_15m), COOLDOWN_BARS)
        e_rsi_long = _apply_cooldown(detect_rsi_recovery_long(n, rsi_recovery_15m), COOLDOWN_BARS)

        funnel["macd_pullback_short"] += len(e_pb_short)
        funnel["macd_early_trend_short"] += len(e_et_short)
        funnel["macd_pullback_long"] += len(e_pb_long)
        funnel["rsi_recovery_long"] += len(e_rsi_long)

        for det_name, entries, direction in (
            ("macd_pullback_short", e_pb_short, "short"),
            ("macd_early_trend_short", e_et_short, "short"),
            ("macd_pullback_long", e_pb_long, "long"),
            ("rsi_recovery_long", e_rsi_long, "long"),
        ):
            for ei in entries:
                atr_e = float(atr14[ei]) if ei < len(atr14) and np.isfinite(atr14[ei]) else float("nan")
                trade = simulate_e2_exit(h, lo, c, ei, atr_e, direction)
                if trade is None:
                    continue
                trade.update({
                    "detector": det_name,
                    "asset": asset,
                    "direction": direction,
                    "entry_time": pd.Timestamp(ts[ei]).isoformat(),
                    "exit_time": pd.Timestamp(ts[trade["exit_idx"]]).isoformat(),
                })
                all_trades.append(trade)

        if (k_idx + 1) % 10 == 0 or k_idx == len(asset_list) - 1:
            print(f"[{time.monotonic() - started:6.1f}s] {k_idx + 1}/{len(asset_list)}  "
                  f"pb_s={funnel['macd_pullback_short']:>4d} "
                  f"et_s={funnel['macd_early_trend_short']:>4d} "
                  f"pb_l={funnel['macd_pullback_long']:>4d} "
                  f"rsi_l={funnel['rsi_recovery_long']:>4d}  "
                  f"trades={len(all_trades)}")

    trades_df = pd.DataFrame(all_trades)
    print(f"[{time.monotonic() - started:6.1f}s] total trades: {len(trades_df)}")

    # --- Metrics ---
    by_strat = split_metrics(trades_df) if len(trades_df) else {}
    by_month_df = by_month_metrics(trades_df) if len(trades_df) else pd.DataFrame()

    metrics = {
        "snapshot_date": SNAPSHOT_DATE,
        "hold_out_start": HOLD_OUT_START,
        "ship_floor_pf": SHIP_FLOOR_PF,
        "n_eligible_assets": len(asset_list),
        "funnel": funnel,
        "by_detector": by_strat,
        "by_month": by_month_df.to_dict(orient="records") if not by_month_df.empty else [],
    }

    # --- Write results ---
    run_id = build_run_id()
    results_dir = Path(__file__).resolve().parents[1] / "results" / run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    if len(trades_df):
        trades_df.to_csv(results_dir / "trades.csv", index=False)
    if not by_month_df.empty:
        by_month_df.to_csv(results_dir / "by_month.csv", index=False)
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(results_dir / "run_metadata.json", "w") as f:
        json.dump({
            "run_id": run_id,
            "git_sha": _git_sha(),
            "git_dirty": git_dirty_at_start,
            "snapshot_date": SNAPSHOT_DATE,
            "wall_time_seconds": time.monotonic() - started,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }, f, indent=2)

    # --- Console output ---
    print()
    print("=" * 92)
    print(f"FUNNEL  (raw signals across {len(asset_list)} assets, ~95-day window)")
    print("=" * 92)
    for k, v in funnel.items():
        print(f"  {k:30s}  {v:>5d}")

    print()
    print("=" * 92)
    print(f"PER-DETECTOR — split by train (pre-{HOLD_OUT_START}) vs hold-out (last 30d)")
    print("=" * 92)
    print(f"{'Detector':30s}  {'all PF':>6}  {'all n':>5}  {'train PF':>8}  {'train n':>7}  {'hold PF':>7}  {'hold n':>6}  {'ship?':>5}")
    print("-" * 92)
    for det, b in by_strat.items():
        print(f"{det:30s}  "
              f"{b['all']['pf']:>6.2f}  {b['all']['trades']:>5d}  "
              f"{b['train']['pf']:>8.2f}  {b['train']['trades']:>7d}  "
              f"{b['hold_out']['pf']:>7.2f}  {b['hold_out']['trades']:>6d}  "
              f"{'YES' if b['passes_ship_floor_holdout'] else 'no':>5s}")

    if not by_month_df.empty:
        print()
        print("=" * 92)
        print("BY MONTH × DETECTOR")
        print("=" * 92)
        pivoted_pf = by_month_df.pivot(index="month", columns="detector", values="pf").fillna(0.0)
        pivoted_n = by_month_df.pivot(index="month", columns="detector", values="trades").fillna(0).astype(int)
        cols = [c for c in [
            "macd_pullback_short", "macd_early_trend_short",
            "macd_pullback_long", "rsi_recovery_long",
        ] if c in pivoted_pf.columns]
        print(f"{'month':10s}  " + "  ".join(f"{c[:18]:>18s}" for c in cols))
        for month, row in pivoted_pf.iterrows():
            cells = []
            for c in cols:
                pf_val = row.get(c, 0.0)
                n_val = pivoted_n.loc[month, c] if c in pivoted_n.columns else 0
                cells.append(f"PF{pf_val:>5.2f} n{int(n_val):>3d}".rjust(18))
            print(f"{month:10s}  " + "  ".join(cells))

    print()
    print(f"=== DONE — wall_time {time.monotonic() - started:.1f}s ===")
    print(f"output: {results_dir}")


if __name__ == "__main__":
    main()
