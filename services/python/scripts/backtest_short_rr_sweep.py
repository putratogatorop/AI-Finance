# scripts/backtest_short_rr_sweep.py
"""R:R Sweep for Short Strategy.

Scans all coins ONCE for entry signals (same logic as blind short backtest),
then simulates each signal across all (stop, tp, timeout) combos.
This avoids re-scanning for each combo.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from scripts.backtest_momentum_scanner import load_coin  # noqa: E402

# ── Paths ────────────────────────────────────────────────────────────
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

# ── Entry params (same as blind backtest) ────────────────────────────
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
BTC_EMA_PERIOD = 30
COOLDOWN_BARS = 96

# ── R:R combos to sweep ─────────────────────────────────────────────
RR_COMBOS = [
    (0.03, 0.09, "3%/9% (3:1)"),
    (0.03, 0.06, "3%/6% (2:1)"),
    (0.03, 0.03, "3%/3% (1:1)"),
    (0.05, 0.15, "5%/15% (3:1) CURRENT"),
    (0.05, 0.10, "5%/10% (2:1)"),
    (0.05, 0.05, "5%/5% (1:1)"),
    (0.08, 0.08, "8%/8% (1:1)"),
    (0.05, 0.03, "5%/3% (tight TP)"),
]

TIMEOUTS = [96, 192, 384, 672]  # 24h, 48h, 96h, 1 week

# Round-trip friction per trade (Gate.io fees + spread/slippage on altcoins)
FRICTION_PCT = 0.0015


def load_btc():
    """Load BTC 15m data and compute 30-bar EMA."""
    df = load_coin(RAW_DIR / "BTCUSDT")
    if df is None or len(df) < BTC_EMA_PERIOD:
        raise RuntimeError("Cannot load BTC data")
    close = df["close"].values.astype(float)
    times = df["open_time"].values
    ema = pd.Series(close).ewm(span=BTC_EMA_PERIOD, adjust=False).mean().values
    return times, close, ema


def scan_and_collect_entries(symbol, df, btc_times, btc_close, btc_ema):
    """Same entry logic as blind backtest, returns list of entry bar indices."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["open_time"].values

    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    entries = []
    cooldown_until = 0

    for i in range(VOL_MA_PERIOD + PRICE_LOOKBACK, len(close)):
        if i < cooldown_until:
            continue

        # Gate 1: Volume spike
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
            continue

        # Gate 2: Price already DOWN from rolling high
        rolling_high = np.max(high[i - PRICE_LOOKBACK : i])
        if rolling_high <= 0:
            continue
        price_drop = (rolling_high - close[i]) / rolling_high
        if price_drop < PRICE_MOVE_THRESH:
            continue

        # Gate 3: BTC trend filter — only short when BTC BELOW EMA
        btc_idx = np.searchsorted(btc_times, times[i])
        btc_idx = min(btc_idx, len(btc_close) - 1)
        if btc_close[btc_idx] >= btc_ema[btc_idx]:
            continue

        entries.append({
            "bar_idx": i,
            "signal_time": pd.Timestamp(times[i]),
            "close": close,
            "high": high,
            "low": low,
        })
        cooldown_until = i + COOLDOWN_BARS

    return entries


def simulate_short(close, high, low, entry_bar, stop_pct, tp_pct, max_bars):
    """Short trade sim. Returns (pnl_pct, exit_reason)."""
    if entry_bar + 1 >= len(close):
        return None, None

    entry_price = close[entry_bar]
    if entry_price <= 0:
        return None, None

    sl_price = entry_price * (1 + stop_pct)
    tp_price = entry_price * (1 - tp_pct)

    last_bar = min(entry_bar + 1 + max_bars, len(close))

    for bar in range(entry_bar + 1, last_bar):
        if high[bar] >= sl_price:
            return -stop_pct - FRICTION_PCT, "stop_loss"
        if low[bar] <= tp_price:
            return tp_pct - FRICTION_PCT, "take_profit"

    # Timeout
    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (entry_price - exit_price) / entry_price - FRICTION_PCT
    return pnl_pct, "timeout"


def run_sweep():
    """Scan once, simulate all combos."""
    print("Loading BTC data for trend filter...")
    btc_times, btc_close, btc_ema = load_btc()
    print(f"  BTC bars: {len(btc_close)}")

    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(symbol_dirs)} coin directories")

    # ── Pass 1: collect all entry signals ────────────────────────────
    print("\n=== PASS 1: Scanning for entry signals ===")
    t0 = time.time()
    all_entries = []

    for idx, sym_dir in enumerate(symbol_dirs):
        symbol = sym_dir.name
        if symbol == "BTCUSDT":
            continue

        df = load_coin(sym_dir)
        if df is None or len(df) < 500:
            continue

        entries = scan_and_collect_entries(symbol, df, btc_times, btc_close, btc_ema)
        for e in entries:
            e["symbol"] = symbol
        all_entries.extend(entries)

        if (idx + 1) % 20 == 0:
            print(
                f"  {idx + 1}/{len(symbol_dirs)} coins, "
                f"{len(all_entries)} entries so far"
            )

    scan_time = time.time() - t0
    print(f"\nPass 1 complete: {len(all_entries)} entries in {scan_time:.1f}s")

    if not all_entries:
        print("No entries found. Exiting.")
        return

    # ── Pass 2: simulate all combos ──────────────────────────────────
    print("\n=== PASS 2: Simulating all R:R + timeout combos ===")
    t1 = time.time()

    combos = []
    for stop_pct, tp_pct, label in RR_COMBOS:
        for timeout in TIMEOUTS:
            combos.append((stop_pct, tp_pct, timeout, label))

    print(f"  {len(combos)} combos x {len(all_entries)} entries "
          f"= {len(combos) * len(all_entries):,} simulations")

    results = []

    for stop_pct, tp_pct, timeout, label in combos:
        pnls = []
        exits = {"take_profit": 0, "stop_loss": 0, "timeout": 0}

        for entry in all_entries:
            pnl, reason = simulate_short(
                entry["close"], entry["high"], entry["low"],
                entry["bar_idx"], stop_pct, tp_pct, timeout,
            )
            if pnl is not None:
                pnls.append(pnl)
                exits[reason] += 1

        if not pnls:
            continue

        pnl_arr = np.array(pnls)
        n = len(pnl_arr)
        wins = (pnl_arr > 0).sum()
        gross_profit = pnl_arr[pnl_arr > 0].sum()
        gross_loss = abs(pnl_arr[pnl_arr <= 0].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        timeout_h = timeout * 15 / 60

        results.append({
            "label": label,
            "timeout_h": f"{timeout_h:.0f}h",
            "timeout_bars": timeout,
            "trades": n,
            "wr": wins / n * 100,
            "pf": pf,
            "avg_pnl": pnl_arr.mean() * 100,
            "total_pnl": pnl_arr.sum() * 100,
            "tp_pct": exits["take_profit"] / n * 100,
            "sl_pct": exits["stop_loss"] / n * 100,
            "to_pct": exits["timeout"] / n * 100,
            "stop": stop_pct,
            "tp": tp_pct,
        })

    sim_time = time.time() - t1
    print(f"Pass 2 complete: {len(results)} combos in {sim_time:.1f}s")

    # ── Print results grid ───────────────────────────────────────────
    print("\n")
    print("=" * 105)
    print("R:R SWEEP RESULTS (sorted by Profit Factor)")
    print("=" * 105)
    print(
        f"{'Label':<24} {'Timeout':>7} {'Trades':>7} {'WR%':>7} "
        f"{'PF':>7} {'AvgPnL':>8} {'TotPnL':>9} {'TP%':>6} {'SL%':>6} {'TO%':>6}"
    )
    print("-" * 105)

    results.sort(key=lambda r: r["pf"], reverse=True)

    for r in results:
        print(
            f"{r['label']:<24} {r['timeout_h']:>7} {r['trades']:>7} "
            f"{r['wr']:>6.1f}% {r['pf']:>7.2f} {r['avg_pnl']:>+7.2f}% "
            f"{r['total_pnl']:>+8.1f}% {r['tp_pct']:>5.1f}% "
            f"{r['sl_pct']:>5.1f}% {r['to_pct']:>5.1f}%"
        )

    # ── Best combo ───────────────────────────────────────────────────
    print("\n" + "=" * 105)
    viable = [r for r in results if r["trades"] >= 1000]
    if viable:
        best = max(viable, key=lambda r: r["pf"])
        print(
            f"BEST COMBO (PF with >1000 trades): {best['label']} @ {best['timeout_h']} "
            f"— PF={best['pf']:.2f}, WR={best['wr']:.1f}%, "
            f"AvgPnL={best['avg_pnl']:+.2f}%, {best['trades']} trades"
        )
    else:
        print("No combo has >1000 trades.")

    # Also show best by avg_pnl
    if viable:
        best_pnl = max(viable, key=lambda r: r["avg_pnl"])
        print(
            f"BEST BY AVG PNL:                   {best_pnl['label']} @ {best_pnl['timeout_h']} "
            f"— AvgPnL={best_pnl['avg_pnl']:+.2f}%, PF={best_pnl['pf']:.2f}, "
            f"WR={best_pnl['wr']:.1f}%, {best_pnl['trades']} trades"
        )
    print("=" * 105)

    total_time = time.time() - t0
    print(f"\nTotal time: {total_time:.1f}s "
          f"(scan {scan_time:.1f}s + sim {sim_time:.1f}s)")


if __name__ == "__main__":
    run_sweep()
