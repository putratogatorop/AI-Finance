"""Exit-regime sweep: hold entries constant, vary exits across 6 configs × 3 strategies.

Strategies: all shorts. PnL for short = (entry - exit) / entry - FRICTION.
Timeout bars are 15m bars (672 = 1 week, 96 = 24h).

Sampling: 10K entries per strategy uniformly (seed=42) → 30K entries × 6 configs = 180K sims.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/putratogatorop/Desktop/AI Finance")
FRICTION = 0.0015
SAMPLE_N = 10_000
SEED = 42

RESULTS = {
    "A": ROOT / "services/python/results/backtest_3strat_protocol_A_28975aa0_20260424T093815Z/trades.csv",
    "B": ROOT / "services/python/results/backtest_3strat_protocol_B_28975aa0_20260424T093815Z/trades.csv",
    "C": ROOT / "services/python/results/backtest_3strat_protocol_C_28975aa0_20260424T093815Z/trades.csv",
}
PARQUET = ROOT / "data/snapshots/candles_15m_2026-04-01.parquet"


def simulate_config(
    entries: pd.DataFrame,  # columns: symbol, entry_time, entry_idx (offset into per-symbol array)
    sym_arrays: dict,        # symbol -> dict with open/high/low/close (np.ndarray) + len
    config: str,
) -> pd.DataFrame:
    """Simulate a specific exit config across all entries. Returns per-trade pnl_pct & bars_held."""
    n = len(entries)
    pnl = np.full(n, np.nan, dtype=np.float64)
    bars = np.full(n, -1, dtype=np.int64)

    # Per-config path-dependent loop. Numpy vector ops per trade on small bar windows.
    syms = entries["symbol"].to_numpy()
    ent_idxs = entries["entry_idx"].to_numpy()

    for i in range(n):
        sym = syms[i]
        ei = ent_idxs[i]
        arr = sym_arrays.get(sym)
        if arr is None:
            continue
        o = arr["open"]; h = arr["high"]; l = arr["low"]; c = arr["close"]; L = arr["len"]
        # entry executes at close of entry bar (standard); exits start from bar ei+1
        start = ei + 1
        if start >= L:
            continue
        entry_px = c[ei]
        if entry_px <= 0 or not np.isfinite(entry_px):
            continue

        # Determine timeout horizon
        if config == "E4":
            tmax = 96
        else:
            tmax = 672
        end = min(start + tmax, L)
        if end <= start:
            continue
        hi = h[start:end]
        lo = l[start:end]
        op = o[start:end]
        cl = c[start:end]

        exit_px = None
        exit_bar = None

        if config == "E1":  # SL 3% TP 5% (short: SL if high >= entry*1.03, TP if low <= entry*0.95)
            sl_lvl = entry_px * 1.03
            tp_lvl = entry_px * 0.95
            sl_hit = hi >= sl_lvl
            tp_hit = lo <= tp_lvl
            sl_first = np.argmax(sl_hit) if sl_hit.any() else 10**9
            tp_first = np.argmax(tp_hit) if tp_hit.any() else 10**9
            if sl_first == tp_first == 10**9:
                exit_bar = len(cl) - 1
                exit_px = cl[-1]
            elif sl_first <= tp_first:
                exit_bar = sl_first
                exit_px = sl_lvl
            else:
                exit_bar = tp_first
                exit_px = tp_lvl
        elif config == "E2":  # SL 2% TP 10%
            sl_lvl = entry_px * 1.02
            tp_lvl = entry_px * 0.90
            sl_hit = hi >= sl_lvl
            tp_hit = lo <= tp_lvl
            sl_first = np.argmax(sl_hit) if sl_hit.any() else 10**9
            tp_first = np.argmax(tp_hit) if tp_hit.any() else 10**9
            if sl_first == tp_first == 10**9:
                exit_bar = len(cl) - 1
                exit_px = cl[-1]
            elif sl_first <= tp_first:
                exit_bar = sl_first
                exit_px = sl_lvl
            else:
                exit_bar = tp_first
                exit_px = tp_lvl
        elif config == "E3":  # SL 5% TP 5%
            sl_lvl = entry_px * 1.05
            tp_lvl = entry_px * 0.95
            sl_hit = hi >= sl_lvl
            tp_hit = lo <= tp_lvl
            sl_first = np.argmax(sl_hit) if sl_hit.any() else 10**9
            tp_first = np.argmax(tp_hit) if tp_hit.any() else 10**9
            if sl_first == tp_first == 10**9:
                exit_bar = len(cl) - 1
                exit_px = cl[-1]
            elif sl_first <= tp_first:
                exit_bar = sl_first
                exit_px = sl_lvl
            else:
                exit_bar = tp_first
                exit_px = tp_lvl
        elif config == "E4":  # SL 5% TP 15%, timeout 96
            sl_lvl = entry_px * 1.05
            tp_lvl = entry_px * 0.85
            sl_hit = hi >= sl_lvl
            tp_hit = lo <= tp_lvl
            sl_first = np.argmax(sl_hit) if sl_hit.any() else 10**9
            tp_first = np.argmax(tp_hit) if tp_hit.any() else 10**9
            if sl_first == tp_first == 10**9:
                exit_bar = len(cl) - 1
                exit_px = cl[-1]
            elif sl_first <= tp_first:
                exit_bar = sl_first
                exit_px = sl_lvl
            else:
                exit_bar = tp_first
                exit_px = tp_lvl
        elif config == "E5":  # trailing stop 3% from best (short: best = running low, stop = best*1.03)
            # Path-dependent: iterate bars
            best = entry_px  # running low since entry
            exit_bar = len(cl) - 1
            exit_px = cl[-1]
            for b in range(len(cl)):
                # update best BEFORE checking stop so low this bar can lower best
                if lo[b] < best:
                    best = lo[b]
                stop_lvl = best * 1.03
                if hi[b] >= stop_lvl:
                    exit_bar = b
                    exit_px = stop_lvl
                    break
        elif config == "E6":  # SL 5% TP 15%, but exit on first green 15m bar (close>open) after entry
            sl_lvl = entry_px * 1.05
            tp_lvl = entry_px * 0.85
            exit_bar = len(cl) - 1
            exit_px = cl[-1]
            for b in range(len(cl)):
                # SL / TP priority within bar (same convention as others)
                if hi[b] >= sl_lvl:
                    exit_bar = b
                    exit_px = sl_lvl
                    break
                if lo[b] <= tp_lvl:
                    exit_bar = b
                    exit_px = tp_lvl
                    break
                if cl[b] > op[b]:
                    exit_bar = b
                    exit_px = cl[b]
                    break
        else:
            raise ValueError(config)

        # Short pnl
        raw = (entry_px - exit_px) / entry_px
        pnl[i] = raw - FRICTION
        bars[i] = exit_bar + 1  # bars held = exit_bar index (0-based) + 1

    out = entries.copy()
    out["pnl_pct"] = pnl
    out["bars_held"] = bars
    return out


def summarize(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["pnl_pct"])
    n = len(d)
    if n == 0:
        return dict(trades=0, wr=np.nan, pf=np.nan, avg=np.nan, total=np.nan, median_bars=np.nan)
    wins = d["pnl_pct"] > 0
    wr = 100.0 * wins.sum() / n
    gross_win = d.loc[wins, "pnl_pct"].sum()
    gross_loss = -d.loc[~wins, "pnl_pct"].sum()
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf
    return dict(
        trades=n,
        wr=wr,
        pf=pf,
        avg=100.0 * d["pnl_pct"].mean(),
        total=100.0 * d["pnl_pct"].sum(),
        median_bars=float(d["bars_held"].median()),
    )


def main():
    t0 = time.time()
    print("Loading parquet...", flush=True)
    candles = pd.read_parquet(PARQUET)
    candles = candles.rename(columns={"asset": "symbol"})
    candles = candles.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    # Convert to per-symbol arrays with timestamp index for lookup.
    print(f"  rows={len(candles):,}  elapsed={time.time()-t0:.1f}s", flush=True)

    # Load all trades
    trades_by = {}
    rng = np.random.default_rng(SEED)
    for s, path in RESULTS.items():
        tr = pd.read_csv(path, usecols=["symbol", "entry_time"])
        tr["entry_time"] = pd.to_datetime(tr["entry_time"], utc=True)
        if len(tr) > SAMPLE_N:
            idx = rng.choice(len(tr), size=SAMPLE_N, replace=False)
            tr = tr.iloc[idx].reset_index(drop=True)
        trades_by[s] = tr
        print(f"  strategy {s}: sampled {len(tr):,}", flush=True)

    # Build entry_idx per (symbol, entry_time) using per-symbol timestamp arrays.
    print("Building per-symbol timestamp index...", flush=True)
    sym_arrays = {}
    sym_ts_index = {}
    for sym, grp in candles.groupby("symbol", sort=False):
        ts = grp["timestamp"].to_numpy()
        sym_arrays[sym] = {
            "open": grp["open"].to_numpy(np.float64),
            "high": grp["high"].to_numpy(np.float64),
            "low": grp["low"].to_numpy(np.float64),
            "close": grp["close"].to_numpy(np.float64),
            "len": len(grp),
        }
        sym_ts_index[sym] = pd.Series(np.arange(len(ts)), index=pd.DatetimeIndex(ts))
    print(f"  symbols={len(sym_arrays):,}  elapsed={time.time()-t0:.1f}s", flush=True)

    # Resolve entry_idx for each trade
    for s in "ABC":
        tr = trades_by[s]
        idxs = np.full(len(tr), -1, dtype=np.int64)
        for i, (sym, et) in enumerate(zip(tr["symbol"].to_numpy(), tr["entry_time"].to_numpy())):
            sidx = sym_ts_index.get(sym)
            if sidx is None:
                continue
            # Use searchsorted; entry_time must exactly match a bar timestamp
            pos = sidx.index.searchsorted(et)
            if pos < len(sidx.index) and sidx.index[pos] == et:
                idxs[i] = sidx.iloc[pos]
        tr["entry_idx"] = idxs
        before = len(tr)
        tr = tr[tr["entry_idx"] >= 0].reset_index(drop=True)
        after = len(tr)
        if before != after:
            print(f"  strategy {s}: dropped {before-after} unmatched entries", flush=True)
        trades_by[s] = tr

    configs = ["E1", "E2", "E3", "E4", "E5", "E6"]
    rows = []
    for s in "ABC":
        tr = trades_by[s]
        for cfg in configs:
            t1 = time.time()
            sim = simulate_config(tr, sym_arrays, cfg)
            stats = summarize(sim)
            stats["strategy"] = s
            stats["config"] = cfg
            rows.append(stats)
            print(
                f"  {s}/{cfg}: trades={stats['trades']} WR={stats['wr']:.1f}% "
                f"PF={stats['pf']:.3f} avg={stats['avg']:.3f}% total={stats['total']:.1f}% "
                f"med_bars={stats['median_bars']:.0f} ({time.time()-t1:.1f}s)",
                flush=True,
            )

    out = pd.DataFrame(rows)[["strategy", "config", "trades", "wr", "pf", "avg", "total", "median_bars"]]
    print()
    print(out.to_string(index=False))

    out_path = ROOT / "services/python/results/exit_regime_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved → {out_path}")
    print(f"Total elapsed: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
