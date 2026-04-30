"""Recent-95d backtest — v8 BALANCED stack with Filters C, D, E, F.

Tests four pre-entry filters individually + combined, vs the unfiltered
baseline (which matches the live-deployed detector code).

Filters
-------
C — PRE-ENTRY PULLBACK
    For LONG: require the 4h bar that just closed (the cross-up bar) to have
    close < high − PULLBACK_PCT × range. I.e., the bar didn't close in the
    top PULLBACK_PCT of its range. Skips "buy at the top of the pump bar"
    setups. SHORT: mirror — require close > low + PULLBACK_PCT × range.

D — 4H BTC EMA(50) REGIME GATE
    For LONG: require BTC 4h close > BTC 4h EMA(50) at signal time.
    For SHORT: require BTC 4h close < BTC 4h EMA(50). Catches faster
    regime turns than the daily-MACD gate already in the detector.

E — CONFIRMATION DELAY
    Defer entry by CONFIRM_BARS × 15min. Only enter at the deferred bar if
    price hasn't reverted: LONG requires close[i+N] >= close[i] (price held
    or moved up); SHORT requires close[i+N] <= close[i]. The actual entry
    fill is at close[i+N], not close[i].

F — 7-DAY MOMENTUM
    For LONG: require 7d return ≥ 0 (no severe decline last week).
    For SHORT: require 7d return ≤ 0 (confirmed downtrend).
    SKIPPED for rsi_recovery_long (it explicitly fires on oversold —
    7d return is structurally negative there; would kill all signals).

Each filter is also configurable to be off.

Usage from services/python/:
    .venv/bin/python scripts/backtest_v8_balanced_recent_filters_cdef_v1.py
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
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


@dataclass
class FilterConfig:
    name: str
    # Filter C: skip if close is in top PULLBACK_PCT of bar range (long) — None = off
    pullback_pct: float | None = None
    # Filter D: BTC 4h EMA(50) regime gate — None = off
    btc_ema_gate: bool = False
    # Filter E: confirmation delay in 15m bars (e.g., 2 = wait 30 min) — None = off
    confirm_bars: int | None = None
    # Filter F: 7d momentum gate — None = off
    momentum_7d_gate: bool = False


# --- SNAPSHOT LOADER ------------------------------------------------------
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
    expected_c, expected_u = _parse_manifest_row(manifest.read_text(), date)
    if _sha256_file(candles_path) != expected_c:
        raise ValueError("candles sha mismatch")
    if _sha256_file(universe_path) != expected_u:
        raise ValueError("universe sha mismatch")
    return pd.read_parquet(candles_path), pd.read_parquet(universe_path)


# --- INDICATORS ------------------------------------------------------------

def _atr14(high, low, close):
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
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


def simulate_e2_exit(high, low, close, entry_idx, atr14_at_entry, direction, *,
                     entry_price_override: float | None = None):
    n = len(close)
    if entry_idx >= n - 1:
        return None
    ep = float(entry_price_override) if entry_price_override is not None else float(close[entry_idx])
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


def detector_metrics(df: pd.DataFrame) -> dict:
    if len(df) == 0:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "total_pnl_pct": 0.0}
    return {
        "trades": int(len(df)),
        "pf": round(_pf(df["pnl_pct"]), 4),
        "wr": round(float((df["pnl_pct"] > 0).sum() / len(df)), 4),
        "total_pnl_pct": round(float(df["pnl_pct"].sum() * 100), 3),
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
        out[det] = {
            "all": detector_metrics(d),
            "train": detector_metrics(d[d.entry_time < cutoff]),
            "hold_out": detector_metrics(d[d.entry_time >= cutoff]),
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


# --- RUN ONE CONFIG --------------------------------------------------------

def run_one_config(candles_by_asset, asset_list, btc_above_ema_15m: pd.Series, cfg: FilterConfig):
    """Run one filter config across all assets, return trades DataFrame.

    btc_above_ema_15m: pd.Series indexed at 4h boundaries with bool — does BTC
    close > BTC 4h EMA(50). Caller has already shifted so that 15m bar at time
    t inherits the value from the prior 4h period.
    """
    print(f"  config: {cfg.name}")
    print(f"    pullback_pct={cfg.pullback_pct}  btc_gate={cfg.btc_ema_gate}  confirm_bars={cfg.confirm_bars}  mom7d={cfg.momentum_7d_gate}")

    all_trades: list[dict] = []
    skip_counts = {"C": 0, "D": 0, "E": 0, "F": 0}

    for asset in asset_list:
        df = candles_by_asset.get(asset)
        if df is None or len(df) < 4 * 96 + 7 * 96:
            continue
        ts = df["timestamp"].to_numpy()
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        n = len(c)
        atr14 = _atr14(h, lo, c)
        ts_idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
        s_close = pd.Series(c, index=ts_idx)
        s_high = pd.Series(h, index=ts_idx)
        s_low = pd.Series(lo, index=ts_idx)

        # Daily MACD
        d_close = s_close.resample("1D").last().dropna()
        if len(d_close) >= 30:
            d_macd_df = _macd(d_close, 12, 26, 9)
            daily_bull = (d_macd_df["histogram"] > 0) & (d_macd_df["macd"] > d_macd_df["signal"])
            daily_bear = (d_macd_df["histogram"] < 0) & (d_macd_df["macd"] < d_macd_df["signal"])
            db_arr = daily_bull.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
            dbear_arr = daily_bear.reindex(ts_idx, method="ffill").fillna(False).to_numpy(dtype=bool)
        else:
            db_arr = np.zeros(n, dtype=bool); dbear_arr = np.zeros(n, dtype=bool)

        # 4h MACD + 4h H/L/C (used for Filter C)
        h4_close = s_close.resample("4h").last().dropna()
        h4_high = s_high.resample("4h").max().reindex(h4_close.index)
        h4_low = s_low.resample("4h").min().reindex(h4_close.index)
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

        # 4h RSI for rsi_recovery_long
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

        # Prior-4h H / L / C aligned to 15m grid (for Filter C)
        # The 4h labeled (entry_ts.floor("4h") - 4h) is the cross-bar.
        # Shift the 4h series forward by 4h so the 15m bar at 12:00 inherits the 08:00 4h value.
        h4_high_shifted = h4_high.copy(); h4_high_shifted.index = h4_high_shifted.index + pd.Timedelta("4h")
        h4_low_shifted = h4_low.copy(); h4_low_shifted.index = h4_low_shifted.index + pd.Timedelta("4h")
        h4_close_shifted = h4_close.copy(); h4_close_shifted.index = h4_close_shifted.index + pd.Timedelta("4h")
        prior_h4_high = h4_high_shifted.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        prior_h4_low = h4_low_shifted.reindex(ts_idx, method="ffill").to_numpy(dtype=float)
        prior_h4_close = h4_close_shifted.reindex(ts_idx, method="ffill").to_numpy(dtype=float)

        # BTC EMA gate aligned to this asset's 15m grid
        btc_above_15m_arr = btc_above_ema_15m.reindex(ts_idx, method="ffill").to_numpy(dtype=bool)

        # Detect entries (with cooldown)
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
                # ── Filter C: pre-entry pullback ──
                if cfg.pullback_pct is not None:
                    H = prior_h4_high[ei]; L = prior_h4_low[ei]; C = prior_h4_close[ei]
                    if not (np.isfinite(H) and np.isfinite(L) and np.isfinite(C)) or H <= L:
                        skip_counts["C"] += 1
                        continue
                    rng = H - L
                    if direction == "long":
                        if not (C < H - cfg.pullback_pct * rng):
                            skip_counts["C"] += 1
                            continue
                    else:
                        if not (C > L + cfg.pullback_pct * rng):
                            skip_counts["C"] += 1
                            continue

                # ── Filter D: 4h BTC EMA(50) regime ──
                if cfg.btc_ema_gate:
                    btc_above = bool(btc_above_15m_arr[ei]) if ei < len(btc_above_15m_arr) else False
                    if direction == "long" and not btc_above:
                        skip_counts["D"] += 1; continue
                    if direction == "short" and btc_above:
                        skip_counts["D"] += 1; continue

                # ── Filter F: 7d momentum ──  (skip rsi_recovery_long)
                if cfg.momentum_7d_gate and det_name != "rsi_recovery_long":
                    lookback = 7 * 96
                    if ei < lookback:
                        skip_counts["F"] += 1; continue
                    p_now = c[ei]; p_prev = c[ei - lookback]
                    if not (np.isfinite(p_now) and np.isfinite(p_prev) and p_prev > 0):
                        skip_counts["F"] += 1; continue
                    m7 = (p_now - p_prev) / p_prev
                    if direction == "long" and m7 < 0:
                        skip_counts["F"] += 1; continue
                    if direction == "short" and m7 > 0:
                        skip_counts["F"] += 1; continue

                # ── Filter E: confirmation delay ──
                effective_entry_idx = ei
                effective_entry_price = float(c[ei])
                if cfg.confirm_bars is not None and cfg.confirm_bars > 0:
                    cidx = ei + cfg.confirm_bars
                    if cidx >= n:
                        skip_counts["E"] += 1; continue
                    p_signal = float(c[ei])
                    p_confirm = float(c[cidx])
                    if not (np.isfinite(p_signal) and np.isfinite(p_confirm)):
                        skip_counts["E"] += 1; continue
                    if direction == "long" and p_confirm < p_signal:
                        skip_counts["E"] += 1; continue
                    if direction == "short" and p_confirm > p_signal:
                        skip_counts["E"] += 1; continue
                    effective_entry_idx = cidx
                    effective_entry_price = p_confirm

                # Simulate exit
                atr_e = float(atr14[ei]) if ei < len(atr14) and np.isfinite(atr14[ei]) else float("nan")
                trade = simulate_e2_exit(
                    h, lo, c, effective_entry_idx, atr_e, direction,
                    entry_price_override=effective_entry_price,
                )
                if trade is None:
                    continue
                trade.update({
                    "detector": det_name, "asset": asset, "direction": direction,
                    "entry_time": pd.Timestamp(ts[effective_entry_idx]).isoformat(),
                    "signal_time": pd.Timestamp(ts[ei]).isoformat(),
                })
                all_trades.append(trade)

    print(f"    skip counts: {skip_counts}")
    print(f"    trades: {len(all_trades)}")
    return pd.DataFrame(all_trades)


# --- MAIN ------------------------------------------------------------------

def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


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

    # Compute BTC 4h EMA(50) regime gate, aligned to global 15m timeline
    btc = candles_all[candles_all["asset"] == "BTCUSDT"].sort_values("timestamp")
    btc_close = btc.set_index("timestamp")["close"].astype(float)
    btc_h4 = btc_close.resample("4h").last().dropna()
    btc_h4_ema50 = btc_h4.ewm(span=EMA_TREND_SPAN, adjust=False).mean()
    btc_h4_above = (btc_h4 > btc_h4_ema50)
    # Shift forward by 4h so the 15m bar at 12:00 inherits the 08:00 4h value
    btc_h4_above_shifted = btc_h4_above.copy()
    btc_h4_above_shifted.index = btc_h4_above_shifted.index + pd.Timedelta("4h")
    btc_above_ema_15m = btc_h4_above_shifted  # caller will reindex per-asset
    print(f"[{time.monotonic() - started:6.1f}s] eligible: {len(asset_list)} assets, {len(candles):,} candles, BTC EMA gate ready ({btc_h4_above.sum()}/{len(btc_h4_above)} 4h bars above EMA50)")

    # Configs to run
    configs = [
        FilterConfig(name="baseline"),
        FilterConfig(name="C only — pullback ≥30%", pullback_pct=0.30),
        FilterConfig(name="D only — BTC 4h EMA50 gate", btc_ema_gate=True),
        FilterConfig(name="E only — 30min confirm", confirm_bars=2),
        FilterConfig(name="F only — 7d momentum gate", momentum_7d_gate=True),
        FilterConfig(name="C+D+E+F (all on)",
                     pullback_pct=0.30, btc_ema_gate=True, confirm_bars=2, momentum_7d_gate=True),
    ]

    results: dict[str, pd.DataFrame] = {}
    for cfg in configs:
        print(f"\n[{time.monotonic() - started:6.1f}s] === {cfg.name} ===")
        results[cfg.name] = run_one_config(candles_by_asset, asset_list, btc_above_ema_15m, cfg)

    # === COMPARISON ===
    print(f"\n[{time.monotonic() - started:6.1f}s] === COMPARISON: macd_pullback_long ===\n")
    print("=" * 116)
    print(f"{'config':30s} | {'n':>6s} {'PF':>6s} {'WR':>5s} {'pnl%':>8s} | {'train n':>7s} {'tr PF':>6s} | {'hold n':>6s} {'hold PF':>7s} | {'ship?':>5s}")
    print("-" * 116)
    for name in [c.name for c in configs]:
        df = results[name]
        d = df[df.detector == "macd_pullback_long"] if len(df) else df
        if len(d) == 0:
            print(f"{name:30s} | {'0':>6s} (no trades)")
            continue
        sm = split_metrics(df).get("macd_pullback_long", {})
        all_b = sm.get("all", {}); tr = sm.get("train", {}); ho = sm.get("hold_out", {})
        ship = "YES" if ho.get("pf", 0) >= SHIP_FLOOR_PF else "no"
        print(f"{name:30s} | "
              f"{all_b.get('trades',0):>6d} {all_b.get('pf',0):>6.2f} {all_b.get('wr',0)*100:>4.1f}% "
              f"{all_b.get('total_pnl_pct',0):>+8.1f} | "
              f"{tr.get('trades',0):>7d} {tr.get('pf',0):>6.2f} | "
              f"{ho.get('trades',0):>6d} {ho.get('pf',0):>7.2f} | {ship:>5s}")

    print()
    print("=" * 116)
    print("PER-DETECTOR HOLD-OUT PF (last 30d) — does each pass ship floor 1.30?")
    print("=" * 116)
    detectors = ["macd_pullback_short", "macd_early_trend_short", "macd_pullback_long", "rsi_recovery_long"]
    print(f"{'config':30s} | " + " | ".join(f"{d[:18]:>18s}" for d in detectors))
    print("-" * 116)
    for name in [c.name for c in configs]:
        df = results[name]
        sm = split_metrics(df)
        cells = []
        for det in detectors:
            ho = sm.get(det, {}).get("hold_out", {})
            pf = ho.get("pf", 0); n = ho.get("trades", 0)
            mark = "✓" if pf >= SHIP_FLOOR_PF else " "
            cells.append(f"{mark}PF{pf:>5.2f} n{n:>3d}")
        print(f"{name:30s} | " + " | ".join(c.rjust(18) for c in cells))

    # Monthly comparison for pullback_long across configs
    print()
    print("=" * 116)
    print("macd_pullback_long — month × config grid (PF | n)")
    print("=" * 116)
    months = sorted({m for df in results.values() for m in pd.to_datetime(df.entry_time, utc=True, errors='coerce').dt.strftime("%Y-%m") if pd.notna(m) and m != 'NaT'} - {'NaT'} if results else [])
    months = sorted({m for df in results.values()
                     if len(df) > 0
                     for m in pd.to_datetime(df['entry_time'], utc=True).dt.strftime("%Y-%m").unique()})
    print(f"{'config':30s} | " + " | ".join(f"{m:>11s}" for m in months))
    print("-" * 116)
    for name in [c.name for c in configs]:
        df = results[name]
        if len(df) == 0:
            continue
        d = df[df.detector == "macd_pullback_long"].copy()
        if len(d) == 0:
            print(f"{name:30s} | " + " | ".join("    -    " for _ in months)); continue
        d["entry_time"] = pd.to_datetime(d["entry_time"], utc=True)
        d["month"] = d["entry_time"].dt.strftime("%Y-%m")
        cells = []
        for m in months:
            g = d[d.month == m]
            if len(g) == 0:
                cells.append("   - / 0   ")
            else:
                pf = round(_pf(g["pnl_pct"]), 2)
                cells.append(f"{pf:>5.2f}/{len(g):>4d}")
        print(f"{name:30s} | " + " | ".join(c.rjust(11) for c in cells))

    # Save artifacts
    sha = _git_sha()
    short_sha = sha[:8] if sha != "unknown" else "nogit"
    out_dir = Path(__file__).resolve().parents[1] / "results" / f"backtest_v8_balanced_recent_filters_cdef_v1_{short_sha}_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for cfg in configs:
        df = results[cfg.name]
        sm = split_metrics(df)
        summary[cfg.name] = sm
        if len(df):
            slug = re.sub(r"[^a-z0-9]+", "_", cfg.name.lower()).strip("_")
            df.to_csv(out_dir / f"trades_{slug}.csv", index=False)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "snapshot_date": SNAPSHOT_DATE,
            "hold_out_start": HOLD_OUT_START,
            "configs": [{"name": c.name, "pullback_pct": c.pullback_pct,
                         "btc_ema_gate": c.btc_ema_gate, "confirm_bars": c.confirm_bars,
                         "momentum_7d_gate": c.momentum_7d_gate} for c in configs],
            "summary": summary,
        }, f, indent=2)

    print()
    print(f"=== DONE — wall_time {time.monotonic() - started:.1f}s ===")
    print(f"output: {out_dir}")


if __name__ == "__main__":
    main()
