# Short Layered Exit Simulation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the simple MFE/MAE exit sim with a bar-by-bar simulation engine using layered SL/TP/trailing stops, then grid-sweep to find optimal parameters.

**Architecture:** Single Python script loads SHORT signals + raw 15m CSVs, walks bar-by-bar with 3-layer exit logic, grid-sweeps parameters in two phases (per-layer then cross-layer), writes best results to `ml_backtest_trades` table. No frontend changes.

**Tech Stack:** Python 3.12, pandas, numpy, joblib, SQLAlchemy, PostgreSQL

---

### Task 1: CSV Loader — Load Raw 15m Bars After a Signal

**Files:**
- Create: `services/python/scripts/backtest_short_layered.py`

This task creates the script skeleton with the CSV loading function that extracts the next 96 bars after a signal timestamp for a given symbol.

- [ ] **Step 1: Create script with CSV loader function**

```python
"""Layered exit backtest: SHORT-only, bar-by-bar sim with trailing stops."""
import sys
sys.path.insert(0, ".")

import logging
import glob
import numpy as np
import pandas as pd
import joblib
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
DATA_DIR = "../../data/raw/15m"
THRESHOLD = 0.90
BARS_24H = 96  # 24h of 15m candles

# ---------- CSV loader ----------

def load_symbol_bars(symbol: str) -> pd.DataFrame:
    """Load all 15m CSVs for a symbol into a single DataFrame, sorted by time."""
    pattern = f"{DATA_DIR}/{symbol}/{symbol}-15m-*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        return pd.DataFrame()
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="us", utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    return df


def get_bars_after_signal(bars_df: pd.DataFrame, signal_time: pd.Timestamp,
                          n_bars: int = BARS_24H) -> np.ndarray | None:
    """Return numpy array of (high, low, close) for the next n_bars after signal_time.

    Returns None if not enough data.
    """
    # Signal time is the candle open — we want bars AFTER this candle
    idx = bars_df["open_time"].searchsorted(signal_time, side="right")
    if idx + n_bars > len(bars_df):
        return None
    chunk = bars_df.iloc[idx:idx + n_bars]
    return chunk[["high", "low", "close"]].to_numpy(dtype=np.float64)
```

- [ ] **Step 2: Add a quick smoke test at the bottom and run it**

Add this temporary test block at the bottom of the file:

```python
if __name__ == "__main__":
    # Smoke test: load one symbol and check bars after a known signal
    bars = load_symbol_bars("ENJUSDT")
    logger.info(f"ENJUSDT: {len(bars)} bars, {bars['open_time'].min()} to {bars['open_time'].max()}")
    test_time = bars["open_time"].iloc[100]
    chunk = get_bars_after_signal(bars, test_time, 96)
    logger.info(f"Chunk shape: {chunk.shape if chunk is not None else 'None'}")
    logger.info(f"First bar H/L/C: {chunk[0] if chunk is not None else 'None'}")
```

Run: `cd services/python && python scripts/backtest_short_layered.py`
Expected: prints bar count, chunk shape (96, 3), and first bar values.

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_short_layered.py
git commit -m "feat(backtest): add CSV loader for bar-by-bar short sim"
```

---

### Task 2: Single-Trade Layered Sim Function

**Files:**
- Modify: `services/python/scripts/backtest_short_layered.py`

Add the core simulation function that takes an entry price, a numpy array of bars, and a layer config, and returns the weighted PnL + per-layer exit details.

- [ ] **Step 1: Add layer config dataclass and sim function**

Insert after the CSV loader functions, before `if __name__`:

```python
from dataclasses import dataclass


@dataclass
class LayerConfig:
    weight: float   # fraction of position, e.g. 0.25
    sl: float       # stop loss % as decimal, e.g. 0.02
    tp: float       # take profit % as decimal, e.g. 0.06
    trail_act: float  # trailing activation % as decimal
    trail_dist: float  # trailing distance % as decimal


def sim_trade(entry: float, bars: np.ndarray, layers: list[LayerConfig]) -> tuple[float, str, str]:
    """Simulate a SHORT trade bar-by-bar with layered exits.

    Args:
        entry: entry price (close of signal candle)
        bars: numpy array of shape (N, 3) with columns [high, low, close]
        layers: list of LayerConfig defining each position layer

    Returns:
        (total_pnl, exit_reason, exit_detail)
        - total_pnl: weighted sum of layer PnLs as a decimal (e.g. 0.035 = 3.5%)
        - exit_reason: dominant exit type (tp_hit, sl_hit, trail_hit, time_exit)
        - exit_detail: per-layer breakdown string
    """
    n_layers = len(layers)
    layer_pnl = np.zeros(n_layers)
    layer_exit = ["time_exit"] * n_layers
    layer_closed = [False] * n_layers
    best_favorable = np.zeros(n_layers)  # best unrealized gain seen per layer

    for bar_idx in range(len(bars)):
        bar_high, bar_low, bar_close = bars[bar_idx]

        for li in range(n_layers):
            if layer_closed[li]:
                continue

            lc = layers[li]
            adverse = (bar_high - entry) / entry      # price UP = bad for short
            favorable = (entry - bar_low) / entry      # price DOWN = good for short

            # 1. SL check FIRST (conservative — assumes SL hunter hits before TP)
            if adverse >= lc.sl:
                layer_pnl[li] = -lc.sl
                layer_exit[li] = "sl_hit"
                layer_closed[li] = True
                continue

            # 2. Trailing stop check
            if best_favorable[li] >= lc.trail_act:
                # Trail is active — check if price pulled back too far from best
                # Current favorable from close (not low, since we need to check retracement)
                current_fav = (entry - bar_high) / entry  # worst point this bar for favorable
                retracement = best_favorable[li] - current_fav
                if retracement >= lc.trail_dist:
                    # Stopped out by trailing stop
                    layer_pnl[li] = best_favorable[li] - lc.trail_dist
                    layer_exit[li] = "trail_hit"
                    layer_closed[li] = True
                    continue

            # 3. TP check
            if favorable >= lc.tp:
                layer_pnl[li] = lc.tp
                layer_exit[li] = "tp_hit"
                layer_closed[li] = True
                continue

            # 4. Update best favorable for trailing
            if favorable > best_favorable[li]:
                best_favorable[li] = favorable

        # Early exit if all layers closed
        if all(layer_closed):
            break

    # Time exit: close remaining layers at last bar's close
    last_close = bars[-1, 2]
    for li in range(n_layers):
        if not layer_closed[li]:
            layer_pnl[li] = (entry - last_close) / entry
            layer_exit[li] = "time_exit"

    # Weighted PnL
    total_pnl = sum(layers[li].weight * layer_pnl[li] for li in range(n_layers))

    # Dominant exit reason (by weight)
    exit_weights = {}
    for li in range(n_layers):
        ex = layer_exit[li]
        exit_weights[ex] = exit_weights.get(ex, 0) + layers[li].weight
    exit_reason = max(exit_weights, key=exit_weights.get)

    # Detail string
    parts = []
    for li in range(n_layers):
        pnl_str = f"{layer_pnl[li]*100:+.1f}"
        parts.append(f"L{li+1}:{layer_exit[li]}{pnl_str}")
    exit_detail = " ".join(parts)

    return total_pnl, exit_reason, exit_detail
```

- [ ] **Step 2: Replace the smoke test with a sim_trade test**

Replace the `if __name__` block:

```python
if __name__ == "__main__":
    # Test sim_trade with synthetic bars
    entry = 100.0
    # Bars: price goes up 0.5% (SL hunt), then drops 5%, then retraces 2%
    bars = np.array([
        [100.5, 99.5, 99.8],   # bar 1: slight dip, slight spike
        [100.3, 98.0, 98.5],   # bar 2: drops to 98 (2% favorable)
        [99.0, 95.0, 95.5],    # bar 3: drops to 95 (5% favorable)
        [97.0, 94.5, 96.0],    # bar 4: bounces to 97 from 94.5
        [98.0, 95.5, 97.5],    # bar 5: retraces to 98
    ])

    layers = [
        LayerConfig(weight=0.25, sl=0.02, tp=0.03, trail_act=0.02, trail_dist=0.01),
        LayerConfig(weight=0.50, sl=0.03, tp=0.06, trail_act=0.04, trail_dist=0.02),
        LayerConfig(weight=0.25, sl=0.05, tp=0.10, trail_act=0.06, trail_dist=0.03),
    ]

    pnl, reason, detail = sim_trade(entry, bars, layers)
    logger.info(f"PnL: {pnl*100:+.2f}%  Reason: {reason}  Detail: {detail}")
    # Expected: L1 hits TP at 3% on bar 2, L2 hits TP at 6% on bar 3 or trail, L3 trails/time exits
```

Run: `cd services/python && python scripts/backtest_short_layered.py`
Expected: prints PnL, reason, and per-layer detail showing different exit types.

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_short_layered.py
git commit -m "feat(backtest): add layered sim_trade function with SL/TP/trailing"
```

---

### Task 3: Signal Loading + Bar Preloading

**Files:**
- Modify: `services/python/scripts/backtest_short_layered.py`

Load signals from DB (same filtering as `backtest_short_only.py`), preload all CSV data per symbol to avoid repeated disk reads, and attach bar arrays to each signal.

- [ ] **Step 1: Add signal loading and bar preloading**

Replace the `if __name__` block with the full main pipeline. Insert before `if __name__`:

```python
def load_signals(engine, model) -> pd.DataFrame:
    """Load SHORT signals with ML filter and whitelist, same as backtest_short_only.py."""
    # Whitelist: coins with >= 20 historical trades
    with engine.connect() as conn:
        wl_rows = conn.execute(text("""
            SELECT symbol FROM ml_backtest_trades
            GROUP BY symbol HAVING COUNT(*) >= 20
        """)).fetchall()
    whitelist = {r[0] for r in wl_rows}
    logger.info(f"Whitelist: {len(whitelist)} coins")

    df = pd.read_sql(
        "SELECT * FROM volume_breakouts WHERE fwd_max_gain_24h IS NOT NULL AND fwd_max_loss_24h IS NOT NULL",
        engine,
    )
    logger.info(f"Loaded {len(df)} volume_breakouts rows")

    # Engineer features (same as backtest_short_only.py)
    df['avg_trade_size'] = df['quote_volume'] / df['trades'].clip(lower=1)
    df['vol_x_buyrat'] = df['vol_ratio'] * df['buy_ratio']
    df['candle_quality'] = df['body_pct'] - df['upper_wick_pct']
    df['wick_ratio'] = df['upper_wick_pct'] / (df['lower_wick_pct'] + 0.001)
    df['dollar_imbalance'] = (2 * df['taker_buy_quote'] - df['quote_volume']) / df['quote_volume'].clip(lower=1)
    df['exec_vs_close'] = (df['quote_volume'] / df['volume'].clip(lower=1e-10)) / df['close'].clip(lower=1e-10)
    df['vol_ratio_adj'] = df['vol_ratio'] / (df['volatility_20'] + 0.001)
    df['trend_aligned'] = (df['direction'] * df['price_vs_ema_50']).clip(lower=0)
    df['btc_aligned'] = (df['direction'] * df['btc_ret_24bar']).clip(lower=0)
    df['dist_range_pos'] = df['dist_from_20_high'] / (df['dist_from_20_high'].abs() + df['dist_from_20_low'].abs() + 0.001)
    df['tf_numeric'] = df['timeframe'].map({'15m': 0, '1h': 1, '4h': 2})
    df['signal_time'] = pd.to_datetime(df['signal_time'], utc=True)

    coin_stats = df[df['is_big_mover'] == True].groupby('symbol').agg(
        gainers=('move_type', lambda x: (x == 'GAINER').sum()),
        losers=('move_type', lambda x: (x == 'LOSER').sum()),
    ).reset_index()
    coin_stats['coin_gainer_ratio'] = coin_stats['gainers'] / (coin_stats['losers'] + 1)
    df['coin_gainer_ratio'] = df['symbol'].map(
        dict(zip(coin_stats['symbol'], coin_stats['coin_gainer_ratio']))
    ).fillna(1.5)

    FEATURE_COLS = [
        'vol_ratio', 'buy_ratio', 'price_change_1bar', 'price_change_2bar',
        'bar_range_pct', 'upper_wick_pct', 'lower_wick_pct', 'body_pct',
        'atr_14', 'rsi_14', 'volatility_20',
        'dist_from_20_high', 'dist_from_20_low', 'price_vs_ema_50',
        'btc_ret_24bar', 'btc_ret_96bar', 'btc_vol_ratio',
        'direction', 'tf_numeric',
        'avg_trade_size', 'vol_x_buyrat', 'candle_quality', 'wick_ratio',
        'dollar_imbalance', 'exec_vs_close', 'vol_ratio_adj',
        'trend_aligned', 'btc_aligned', 'dist_range_pos',
        'coin_gainer_ratio',
    ]

    X = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    proba = model.predict_proba(X)[:, 1]
    df['ml_proba'] = proba

    picks = df[
        (df['ml_proba'] >= THRESHOLD) & (df['direction'] == -1) & (df['symbol'].isin(whitelist))
    ].copy()
    logger.info(f"SHORT-only picks after ML+whitelist: {len(picks)}")
    return picks


def preload_bars(symbols: list[str]) -> dict[str, pd.DataFrame]:
    """Preload all 15m CSVs for the given symbols. Returns {symbol: DataFrame}."""
    cache = {}
    for sym in symbols:
        df = load_symbol_bars(sym)
        if len(df) > 0:
            cache[sym] = df
    logger.info(f"Preloaded bars for {len(cache)} symbols")
    return cache


def attach_bar_arrays(picks: pd.DataFrame, bar_cache: dict[str, pd.DataFrame]) -> list[dict]:
    """For each signal, extract the next 96 bars. Returns list of trade dicts with bars attached."""
    trades = []
    skipped = 0
    for _, row in picks.iterrows():
        sym = row['symbol']
        if sym not in bar_cache:
            skipped += 1
            continue
        bars = get_bars_after_signal(bar_cache[sym], row['signal_time'], BARS_24H)
        if bars is None:
            skipped += 1
            continue
        trades.append({
            'symbol': sym,
            'timeframe': row['timeframe'],
            'signal_time': row['signal_time'],
            'ml_proba': row['ml_proba'],
            'close': row['close'],
            'vol_ratio': row['vol_ratio'],
            'buy_ratio': row['buy_ratio'],
            'direction': int(row['direction']),
            'is_big_mover': bool(row['is_big_mover']),
            'move_type': row['move_type'],
            'fwd_max_gain_24h': row['fwd_max_gain_24h'],
            'fwd_max_loss_24h': row['fwd_max_loss_24h'],
            'fwd_ret_24h': row['fwd_ret_24h'],
            'fwd_ret_48h': row.get('fwd_ret_48h'),
            'fwd_ret_1w': row.get('fwd_ret_1w'),
            'rsi_14': row['rsi_14'],
            'atr_14': row['atr_14'],
            'btc_ret_24bar': row['btc_ret_24bar'],
            'coin_gainer_ratio': row['coin_gainer_ratio'],
            'bars': bars,
        })
    logger.info(f"Attached bars: {len(trades)} trades ({skipped} skipped — no bar data)")
    return trades
```

- [ ] **Step 2: Write the main block that loads signals and preloads bars**

Replace the `if __name__` block:

```python
if __name__ == "__main__":
    engine = create_engine(DB_URL)
    model = joblib.load("models/bigmover_classifier.joblib")

    # 1. Load signals
    picks = load_signals(engine, model)

    # 2. Preload bar data
    symbols = picks['symbol'].unique().tolist()
    bar_cache = preload_bars(symbols)

    # 3. Attach bar arrays
    trades = attach_bar_arrays(picks, bar_cache)
    logger.info(f"Ready for simulation: {len(trades)} trades with bar data")
```

Run: `cd services/python && python scripts/backtest_short_layered.py`
Expected: loads signals, preloads bars, prints "Ready for simulation: ~4000-5000 trades with bar data".

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_short_layered.py
git commit -m "feat(backtest): add signal loading and bar preloading for layered sim"
```

---

### Task 4: Grid Sweep — Phase 1 (Per-Layer)

**Files:**
- Modify: `services/python/scripts/backtest_short_layered.py`

Add the Phase 1 grid sweep: for each of the 3 layers independently, sweep 81 parameter combos while using mid-range defaults for the other 2 layers. Pick top 5 per layer by profit factor.

- [ ] **Step 1: Add grid sweep constants and Phase 1 function**

Insert after `attach_bar_arrays`, before `if __name__`:

```python
from itertools import product

# Grid ranges per layer (as decimals)
LAYER_GRIDS = [
    {  # Layer 1 (25% — tight)
        'sl':        [0.01, 0.015, 0.02],
        'tp':        [0.02, 0.03, 0.04],
        'trail_act': [0.015, 0.02, 0.03],
        'trail_dist': [0.005, 0.01, 0.015],
    },
    {  # Layer 2 (50% — core)
        'sl':        [0.02, 0.03, 0.04],
        'tp':        [0.05, 0.06, 0.08],
        'trail_act': [0.03, 0.04, 0.05],
        'trail_dist': [0.01, 0.015, 0.02],
    },
    {  # Layer 3 (25% — runner)
        'sl':        [0.04, 0.05, 0.07],
        'tp':        [0.08, 0.10, 0.13],
        'trail_act': [0.05, 0.07, 0.09],
        'trail_dist': [0.02, 0.03, 0.04],
    },
]

LAYER_WEIGHTS = [0.25, 0.50, 0.25]

# Mid-range defaults (index 1 of each range)
DEFAULT_LAYERS = [
    LayerConfig(weight=0.25, sl=0.015, tp=0.03, trail_act=0.02, trail_dist=0.01),
    LayerConfig(weight=0.50, sl=0.03, tp=0.06, trail_act=0.04, trail_dist=0.015),
    LayerConfig(weight=0.25, sl=0.05, tp=0.10, trail_act=0.07, trail_dist=0.03),
]


def eval_config(trades: list[dict], layers: list[LayerConfig]) -> dict:
    """Run sim_trade on all trades with given layer config. Return stats on 1-trade/day subset."""
    results = []
    for t in trades:
        pnl, reason, detail = sim_trade(t['close'], t['bars'], layers)
        results.append({
            'signal_time': t['signal_time'],
            'ml_proba': t['ml_proba'],
            'pnl': pnl,
            'reason': reason,
        })

    df = pd.DataFrame(results)
    df['date'] = df['signal_time'].dt.date

    # 1 trade/day: pick best ML proba per day
    daily = df.sort_values('ml_proba', ascending=False).groupby('date').first().reset_index()

    n = len(daily)
    if n == 0:
        return {'n': 0, 'wr': 0, 'pf': 0, 'total_pnl': 0, 'avg_pnl': 0}

    wins = (daily['pnl'] > 0).sum()
    gross_profit = daily.loc[daily['pnl'] > 0, 'pnl'].sum()
    gross_loss = abs(daily.loc[daily['pnl'] <= 0, 'pnl'].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.0

    return {
        'n': n,
        'wr': wins / n * 100,
        'pf': round(pf, 2),
        'total_pnl': round(daily['pnl'].sum() * 100, 1),
        'avg_pnl': round(daily['pnl'].mean() * 100, 2),
    }


def sweep_phase1(trades: list[dict]) -> list[list[LayerConfig]]:
    """Phase 1: sweep each layer independently, return top 5 configs per layer."""
    top_per_layer = []

    for li in range(3):
        grid = LAYER_GRIDS[li]
        combos = list(product(grid['sl'], grid['tp'], grid['trail_act'], grid['trail_dist']))
        logger.info(f"Phase 1 — Layer {li+1}: sweeping {len(combos)} combos...")

        results = []
        for sl, tp, ta, td in combos:
            # Build layer config: use defaults for other layers, sweep this one
            layers = [LayerConfig(weight=lw, sl=dl.sl, tp=dl.tp, trail_act=dl.trail_act, trail_dist=dl.trail_dist)
                      for lw, dl in zip(LAYER_WEIGHTS, DEFAULT_LAYERS)]
            layers[li] = LayerConfig(weight=LAYER_WEIGHTS[li], sl=sl, tp=tp, trail_act=ta, trail_dist=td)

            stats = eval_config(trades, layers)
            results.append((stats, LayerConfig(weight=LAYER_WEIGHTS[li], sl=sl, tp=tp, trail_act=ta, trail_dist=td)))

        # Sort by PF (primary), total_pnl (secondary)
        results.sort(key=lambda x: (x[0]['pf'], x[0]['total_pnl']), reverse=True)

        top5 = [r[1] for r in results[:5]]
        logger.info(f"  Top 5 for Layer {li+1}:")
        for rank, (stats, cfg) in enumerate(results[:5]):
            logger.info(f"    #{rank+1}: SL={cfg.sl*100:.1f}% TP={cfg.tp*100:.1f}% "
                        f"Trail={cfg.trail_act*100:.1f}%/{cfg.trail_dist*100:.1f}% "
                        f"→ PF={stats['pf']} WR={stats['wr']:.1f}% Total={stats['total_pnl']:+.0f}%")
        top_per_layer.append(top5)

    return top_per_layer
```

- [ ] **Step 2: Wire Phase 1 into main block**

Add to the `if __name__` block after `trades = attach_bar_arrays(...)`:

```python
    # 4. Phase 1: per-layer sweep
    top_per_layer = sweep_phase1(trades)
```

Run: `cd services/python && python scripts/backtest_short_layered.py`
Expected: prints top 5 configs per layer with PF/WR/total stats. Should complete in 1-3 minutes.

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_short_layered.py
git commit -m "feat(backtest): add Phase 1 per-layer grid sweep"
```

---

### Task 5: Grid Sweep — Phase 2 (Cross-Layer) + Best Config Selection

**Files:**
- Modify: `services/python/scripts/backtest_short_layered.py`

Add Phase 2: combine top 5 from each layer into 125 cross-layer combos, evaluate, select the best.

- [ ] **Step 1: Add Phase 2 sweep function**

Insert after `sweep_phase1`, before `if __name__`:

```python
def sweep_phase2(trades: list[dict], top_per_layer: list[list[LayerConfig]]) -> tuple[list[LayerConfig], dict]:
    """Phase 2: cross-layer sweep of top 5 × 5 × 5. Returns (best_layers, best_stats)."""
    combos = list(product(top_per_layer[0], top_per_layer[1], top_per_layer[2]))
    logger.info(f"Phase 2 — Cross-layer: sweeping {len(combos)} combos...")

    best_layers = None
    best_stats = {'pf': 0}

    for i, (l1, l2, l3) in enumerate(combos):
        layers = [l1, l2, l3]
        stats = eval_config(trades, layers)

        if (stats['pf'], stats['total_pnl']) > (best_stats['pf'], best_stats.get('total_pnl', 0)):
            best_stats = stats
            best_layers = layers

        if (i + 1) % 25 == 0:
            logger.info(f"  {i+1}/{len(combos)} done... best PF so far: {best_stats['pf']}")

    logger.info(f"\n{'='*60}")
    logger.info(f"BEST CONFIG (1 trade/day)")
    logger.info(f"{'='*60}")
    for li, lc in enumerate(best_layers):
        logger.info(f"  Layer {li+1} ({lc.weight*100:.0f}%): "
                     f"SL={lc.sl*100:.1f}% TP={lc.tp*100:.1f}% "
                     f"Trail={lc.trail_act*100:.1f}%/{lc.trail_dist*100:.1f}%")
    logger.info(f"  Trades: {best_stats['n']}")
    logger.info(f"  Win Rate: {best_stats['wr']:.1f}%")
    logger.info(f"  Profit Factor: {best_stats['pf']}")
    logger.info(f"  Total PnL: {best_stats['total_pnl']:+.0f}%")
    logger.info(f"  Avg PnL/trade: {best_stats['avg_pnl']:+.2f}%")

    return best_layers, best_stats
```

- [ ] **Step 2: Wire Phase 2 into main block**

Add to `if __name__` after Phase 1:

```python
    # 5. Phase 2: cross-layer sweep
    best_layers, best_stats = sweep_phase2(trades, top_per_layer)
```

Run: `cd services/python && python scripts/backtest_short_layered.py`
Expected: prints best config with PF, WR, total PnL. Phase 2 should take ~1 minute.

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_short_layered.py
git commit -m "feat(backtest): add Phase 2 cross-layer sweep + best config selection"
```

---

### Task 6: Save Results to ml_backtest_trades

**Files:**
- Modify: `services/python/scripts/backtest_short_layered.py`

Re-run the best config on all trades, save to `ml_backtest_trades` table (same schema + `exit_detail` column), print final all-trades + 1/day stats.

- [ ] **Step 1: Add save function**

Insert after `sweep_phase2`, before `if __name__`:

```python
def run_and_save(trades: list[dict], layers: list[LayerConfig], engine):
    """Run best config on all trades and save to ml_backtest_trades."""
    results = []
    for t in trades:
        pnl, reason, detail = sim_trade(t['close'], t['bars'], layers)
        status = 'won' if pnl > 0 else 'lost'
        results.append({
            'symbol': t['symbol'], 'timeframe': t['timeframe'],
            'signal_time': t['signal_time'].to_pydatetime(),
            'ml_proba': float(t['ml_proba']), 'close_price': float(t['close']),
            'vol_ratio': float(t['vol_ratio']),
            'buy_ratio': float(t['buy_ratio']) if pd.notna(t['buy_ratio']) else None,
            'direction': t['direction'], 'is_big_mover': t['is_big_mover'],
            'move_type': t['move_type'] if t['move_type'] != 'NONE' else None,
            'pnl_pct': float(pnl), 'exit_reason': reason, 'status': status,
            'exit_detail': detail,
            'fwd_max_gain_24h': float(t['fwd_max_gain_24h']) if pd.notna(t['fwd_max_gain_24h']) else None,
            'fwd_max_loss_24h': float(t['fwd_max_loss_24h']) if pd.notna(t['fwd_max_loss_24h']) else None,
            'fwd_ret_24h': float(t['fwd_ret_24h']) if pd.notna(t['fwd_ret_24h']) else None,
            'fwd_ret_48h': float(t['fwd_ret_48h']) if pd.notna(t['fwd_ret_48h']) else None,
            'fwd_ret_1w': float(t['fwd_ret_1w']) if pd.notna(t['fwd_ret_1w']) else None,
            'rsi_14': float(t['rsi_14']) if pd.notna(t['rsi_14']) else None,
            'atr_14': float(t['atr_14']) if pd.notna(t['atr_14']) else None,
            'btc_ret_24bar': float(t['btc_ret_24bar']) if pd.notna(t['btc_ret_24bar']) else None,
            'coin_gainer_ratio': float(t['coin_gainer_ratio']),
        })

    logger.info(f"Saving {len(results)} trades to ml_backtest_trades...")
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS ml_backtest_trades"))
        conn.execute(text("""
            CREATE TABLE ml_backtest_trades (
                id SERIAL PRIMARY KEY, symbol VARCHAR(20), timeframe VARCHAR(4),
                signal_time TIMESTAMPTZ, ml_proba DOUBLE PRECISION, close_price DOUBLE PRECISION,
                vol_ratio DOUBLE PRECISION, buy_ratio DOUBLE PRECISION, direction SMALLINT,
                is_big_mover BOOLEAN, move_type VARCHAR(10), pnl_pct DOUBLE PRECISION,
                exit_reason VARCHAR(20), status VARCHAR(10), exit_detail VARCHAR(100),
                fwd_max_gain_24h DOUBLE PRECISION, fwd_max_loss_24h DOUBLE PRECISION,
                fwd_ret_24h DOUBLE PRECISION, fwd_ret_48h DOUBLE PRECISION, fwd_ret_1w DOUBLE PRECISION,
                rsi_14 DOUBLE PRECISION, atr_14 DOUBLE PRECISION, btc_ret_24bar DOUBLE PRECISION,
                coin_gainer_ratio DOUBLE PRECISION
            )
        """))
        conn.commit()

    Session = sessionmaker(bind=engine)
    session = Session()
    for i in range(0, len(results), 1000):
        session.execute(text("""
            INSERT INTO ml_backtest_trades (symbol,timeframe,signal_time,ml_proba,close_price,vol_ratio,buy_ratio,
                direction,is_big_mover,move_type,pnl_pct,exit_reason,status,exit_detail,fwd_max_gain_24h,
                fwd_max_loss_24h,fwd_ret_24h,fwd_ret_48h,fwd_ret_1w,rsi_14,atr_14,btc_ret_24bar,coin_gainer_ratio)
            VALUES (:symbol,:timeframe,:signal_time,:ml_proba,:close_price,:vol_ratio,:buy_ratio,
                :direction,:is_big_mover,:move_type,:pnl_pct,:exit_reason,:status,:exit_detail,:fwd_max_gain_24h,
                :fwd_max_loss_24h,:fwd_ret_24h,:fwd_ret_48h,:fwd_ret_1w,:rsi_14,:atr_14,:btc_ret_24bar,:coin_gainer_ratio)
        """), results[i:i+1000])
        session.commit()
    session.close()
    logger.info("Saved!")

    # Print final stats
    with engine.connect() as conn:
        s = conn.execute(text("""
            SELECT COUNT(*)::int, COUNT(*) FILTER (WHERE status='won')::int,
                ROUND(COUNT(*) FILTER (WHERE status='won')::numeric/GREATEST(COUNT(*),1)*100,1),
                ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2),
                ROUND(AVG(pnl_pct)::numeric*100,2),
                ROUND(SUM(pnl_pct)::numeric*100,0),
                COUNT(*) FILTER (WHERE exit_reason='tp_hit')::int,
                COUNT(*) FILTER (WHERE exit_reason='sl_hit')::int,
                COUNT(*) FILTER (WHERE exit_reason='trail_hit')::int,
                COUNT(*) FILTER (WHERE exit_reason='time_exit')::int
            FROM ml_backtest_trades
        """)).fetchone()
        logger.info(f"\n{'='*60}")
        logger.info(f"ALL TRADES (layered exit)")
        logger.info(f"{'='*60}")
        logger.info(f"Trades: {s[0]}")
        logger.info(f"W/L: {s[1]}W / {s[0]-s[1]}L")
        logger.info(f"Win Rate: {s[2]}%")
        logger.info(f"Profit Factor: {s[3]}")
        logger.info(f"Avg PnL: {float(s[4]):+.2f}%")
        logger.info(f"Total PnL: {float(s[5]):+.0f}%")
        logger.info(f"Exits: TP={s[6]}, SL={s[7]}, Trail={s[8]}, Time={s[9]}")

        d = conn.execute(text("""
            WITH daily AS (
                SELECT *, ROW_NUMBER() OVER (PARTITION BY signal_time::date ORDER BY ml_proba DESC) as rn
                FROM ml_backtest_trades
            )
            SELECT COUNT(*)::int,
                ROUND(COUNT(*) FILTER (WHERE status='won')::numeric/GREATEST(COUNT(*),1)*100,1),
                ROUND(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct>0),0)::numeric/ABS(NULLIF(SUM(pnl_pct) FILTER (WHERE pnl_pct<=0),0))::numeric,2),
                ROUND(AVG(pnl_pct)::numeric*100,2),
                ROUND(SUM(pnl_pct)::numeric*100,0)
            FROM daily WHERE rn = 1
        """)).fetchone()
        logger.info(f"\n1 TRADE/DAY:")
        logger.info(f"Trades: {d[0]}")
        logger.info(f"Win Rate: {d[1]}%")
        logger.info(f"Profit Factor: {d[2]}")
        logger.info(f"Avg PnL: {float(d[3]):+.2f}%")
        logger.info(f"Total PnL: {float(d[4]):+.0f}%")
        if d[0] > 0:
            logger.info(f"Monthly: {float(d[4])/max(d[0]/30,1):+.1f}%")
```

- [ ] **Step 2: Wire save into main block**

Add to `if __name__` after Phase 2:

```python
    # 6. Save best config results
    run_and_save(trades, best_layers, engine)
```

Run: `cd services/python && python scripts/backtest_short_layered.py`
Expected: full pipeline runs — loads signals, sweeps, saves to DB, prints all-trades + 1/day stats with trail_hit as a new exit type.

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_short_layered.py
git commit -m "feat(backtest): save layered exit results to ml_backtest_trades"
```

---

### Task 7: Update Scanner-Short Page to Show Exit Detail

**Files:**
- Modify: `services/nextjs/src/app/scanner-short/page.tsx`

Add the `exit_detail` column to the simulation table and add `trail_hit` as a new exit badge color.

- [ ] **Step 1: Add exit_detail to sim query and trail_hit badge**

In `services/nextjs/src/app/scanner-short/page.tsx`, find the `simTrades` query (the `ordered` CTE around line 92) and add `exit_detail` to the SELECT:

Change the SELECT in the `ordered` CTE from:
```sql
SELECT trade_num, symbol, timeframe, signal_time, ml_proba, close_price,
  vol_ratio, buy_ratio, pnl_pct, exit_reason, status, is_big_mover,
  fwd_max_gain_24h,
```

To:
```sql
SELECT trade_num, symbol, timeframe, signal_time, ml_proba, close_price,
  vol_ratio, buy_ratio, pnl_pct, exit_reason, exit_detail, status, is_big_mover,
  fwd_max_gain_24h,
```

- [ ] **Step 2: Add exit_detail column to the sim table header**

Find the sim table `<thead>` (around line 218). After the `Exit` header:
```html
<th className="px-3 py-2 text-center text-slate-500">Exit</th>
```

Add:
```html
<th className="px-3 py-2 text-left text-slate-500">Layers</th>
```

- [ ] **Step 3: Add exit_detail cell and trail_hit badge color in sim table body**

Find the exit badge `<td>` in the sim table body (around line 251). Update the badge to include `trail_hit`:

```tsx
<td className="px-3 py-1.5 text-center">
  <span className={`text-[10px] px-1.5 py-0.5 rounded ${
    t.exit_reason === "tp_hit" ? "bg-green-900/50 text-green-400" :
    t.exit_reason === "sl_hit" ? "bg-red-900/50 text-red-400" :
    t.exit_reason === "trail_hit" ? "bg-yellow-900/50 text-yellow-400" :
    "bg-slate-700 text-slate-400"
  }`}>{t.exit_reason}</span>
</td>
```

After this `<td>`, add the exit_detail cell:
```tsx
<td className="px-3 py-1.5 text-left text-[10px] font-mono text-slate-500 whitespace-nowrap">
  {t.exit_detail || "—"}
</td>
```

- [ ] **Step 4: Also update trail_hit badge color in the All Trades table**

Find the exit badge in the All Trades table body (around line 400). Update the same way:

```tsx
<span className={`text-[10px] px-1.5 py-0.5 rounded ${
  t.exit_reason === "tp_hit" ? "bg-green-900/50 text-green-400" :
  t.exit_reason === "sl_hit" ? "bg-red-900/50 text-red-400" :
  t.exit_reason === "trail_hit" ? "bg-yellow-900/50 text-yellow-400" :
  "bg-slate-700 text-slate-400"
}`}>{t.exit_reason}</span>
```

- [ ] **Step 5: Verify the page loads**

Run: `cd services/nextjs && npm run dev`
Open: `http://localhost:3001/scanner-short`
Expected: page loads with exit_detail column visible in sim table, trail_hit badges show in yellow.

- [ ] **Step 6: Commit**

```bash
git add services/nextjs/src/app/scanner-short/page.tsx
git commit -m "feat(scanner-short): show exit_detail and trail_hit badge"
```
