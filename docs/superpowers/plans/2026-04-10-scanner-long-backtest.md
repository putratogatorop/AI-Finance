# Scanner Long Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a backtest script that compares 3 entry strategies for long trend-following trades on big movers, using fixed 3:1 R:R exits.

**Architecture:** Single script loads big_movers from DB as ground truth, loads 15m CSV data per coin, runs 3 independent strategy detectors over each data window, simulates trades bar-by-bar, and prints a comparison summary.

**Tech Stack:** Python 3.12, numpy, pandas, sqlalchemy, PostgreSQL

---

## File Structure

- **Create:** `services/python/scripts/backtest_scanner_long.py` — main backtest script (~350 lines)
  - `load_big_movers()` — query DB for upward big movers
  - `load_coin()` — reuse from `backtest_momentum_scanner.py`
  - `compute_vol_ma()` — 20-bar rolling volume average
  - `detect_strategy_a()` — volume spike + momentum confirmation
  - `detect_strategy_b()` — rolling price scan + volume check
  - `detect_strategy_c()` — piggyback on volume_breakouts table
  - `simulate_trade()` — bar-by-bar SL/TP/timeout sim
  - `run_backtest()` — orchestrator
  - `print_summary()` — comparison table
  - `save_to_db()` — persist results

No test files — this is a research/analysis script, not library code.

---

### Task 1: Script Skeleton + Data Loading

**Files:**
- Create: `services/python/scripts/backtest_scanner_long.py`

- [ ] **Step 1: Create script with imports, constants, and load_big_movers()**

```python
"""Scanner Long Backtest — compare 3 entry strategies for trend-following longs.

Loads big_movers (move_pct > 20%, upward) as ground truth,
runs 3 strategies on 15m candle data, fixed 3:1 R:R exits.

Run from services/python/:
    python scripts/backtest_scanner_long.py
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine, text
from scripts.backtest_momentum_scanner import load_coin

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

# Strategy parameters
VOL_SPIKE_A = 3.0       # Strategy A: 3x volume spike
PRICE_MOVE_THRESH = 0.05 # 5% price move confirmation (all strategies)
VOL_MA_PERIOD = 20       # Rolling volume average period
PRICE_LOOKBACK_A = 96    # Strategy A: 96 bars (24h) for rolling low
PRICE_LOOKBACK_B = 32    # Strategy B: 32 bars (8h) for price scan
VOL_SPIKE_B = 2.0        # Strategy B: 2x volume confirmation
VOL_SPIKE_C = 2.0        # Strategy C: min vol_ratio from volume_breakouts
PRICE_LOOKBACK_C = 96    # Strategy C: 96 bars for price confirmation

# Exit parameters
STOP_LOSS_PCT = 0.05     # -5% stop
TAKE_PROFIT_PCT = 0.15   # +15% target (3:1 R:R)
TIMEOUT_BARS = 672       # 1 week in 15m bars
COOLDOWN_BARS = 96       # 24h cooldown per symbol per strategy

engine = create_engine(DB_URL)


def load_big_movers() -> pd.DataFrame:
    """Load upward big movers (move_pct > 20%) from DB."""
    query = """
        SELECT symbol, move_start, move_peak, start_price, peak_price,
               move_pct, duration_bars, max_vol_ratio_during, avg_buy_ratio_during
        FROM big_movers
        WHERE move_pct > 0.2 AND peak_price > start_price AND direction = 1
        ORDER BY symbol, move_start
    """
    df = pd.read_sql(query, engine)
    print(f"Loaded {len(df)} upward big movers across {df['symbol'].nunique()} coins")
    return df
```

- [ ] **Step 2: Run to verify DB connection and data loading**

Run from `services/python/`:
```bash
cd services/python && python -c "
import sys; sys.path.insert(0, '.')
from scripts.backtest_scanner_long import load_big_movers
df = load_big_movers()
print(df.head())
print(f'Symbols: {df[\"symbol\"].nunique()}, Records: {len(df)}')
"
```

Expected: prints ~6961 records across 100+ coins.

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): add script skeleton with data loading"
```

---

### Task 2: Trade Simulator

**Files:**
- Modify: `services/python/scripts/backtest_scanner_long.py`

- [ ] **Step 1: Add simulate_trade() function**

Append after `load_big_movers()`:

```python
def simulate_trade(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    entry_bar: int,
    stop_pct: float = STOP_LOSS_PCT,
    tp_pct: float = TAKE_PROFIT_PCT,
    max_bars: int = TIMEOUT_BARS,
) -> dict | None:
    """Simulate a long trade bar-by-bar from entry_bar.

    Entry at next bar's open (approximated as close[entry_bar]).
    Returns dict with pnl_pct, exit_reason, bars_held, or None if not enough data.
    """
    if entry_bar + 1 >= len(close):
        return None

    entry_price = close[entry_bar]  # next bar open ~ current close
    if entry_price <= 0:
        return None

    sl_price = entry_price * (1 - stop_pct)
    tp_price = entry_price * (1 + tp_pct)

    for offset in range(1, min(max_bars, len(close) - entry_bar)):
        bar_idx = entry_bar + offset

        # Check stop loss (hit if low touches SL)
        if low[bar_idx] <= sl_price:
            return {
                "pnl_pct": -stop_pct,
                "exit_reason": "stop_loss",
                "bars_held": offset,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }

        # Check take profit (hit if high touches TP)
        if high[bar_idx] >= tp_price:
            return {
                "pnl_pct": tp_pct,
                "exit_reason": "take_profit",
                "bars_held": offset,
                "entry_price": entry_price,
                "exit_price": tp_price,
            }

    # Timeout — exit at last bar's close
    last_bar = min(entry_bar + max_bars, len(close) - 1)
    exit_price = close[last_bar]
    pnl = (exit_price - entry_price) / entry_price
    return {
        "pnl_pct": pnl,
        "exit_reason": "timeout",
        "bars_held": last_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }
```

- [ ] **Step 2: Quick smoke test**

```bash
cd services/python && python -c "
import sys; sys.path.insert(0, '.')
import numpy as np
from scripts.backtest_scanner_long import simulate_trade
# Simulated data: price goes from 100 to 120 in 10 bars
close = np.linspace(100, 120, 20)
high = close * 1.01
low = close * 0.99
result = simulate_trade(close, high, low, entry_bar=0)
print(result)  # should hit TP at +15%
"
```

Expected: `exit_reason: 'take_profit'`, `pnl_pct: 0.15`

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): add bar-by-bar trade simulator"
```

---

### Task 3: Strategy A — Volume Spike + Momentum Confirmation

**Files:**
- Modify: `services/python/scripts/backtest_scanner_long.py`

- [ ] **Step 1: Add detect_strategy_a()**

```python
def detect_strategy_a(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> list[int]:
    """Strategy A: Volume spike >= 3x avg + price already up 5% from 24h low.

    Returns list of bar indices where entry signal fires.
    """
    n = len(close)
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    entries = []
    cooldown = 0

    for i in range(max(start_idx, VOL_MA_PERIOD + PRICE_LOOKBACK_A), min(end_idx, n)):
        if cooldown > 0:
            cooldown -= 1
            continue

        # Gate 1: Volume spike
        if np.isnan(vol_ma[i]) or vol_ma[i] <= 0:
            continue
        vol_ratio = volume[i] / vol_ma[i]
        if vol_ratio < VOL_SPIKE_A:
            continue

        # Gate 2: Price already up 5% from rolling low
        lookback_low = np.min(low[i - PRICE_LOOKBACK_A : i])
        if lookback_low <= 0:
            continue
        price_move = (close[i] - lookback_low) / lookback_low
        if price_move < PRICE_MOVE_THRESH:
            continue

        entries.append(i)
        cooldown = COOLDOWN_BARS

    return entries
```

- [ ] **Step 2: Commit**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): add Strategy A — volume spike + momentum"
```

---

### Task 4: Strategy B — Rolling Price Scan

**Files:**
- Modify: `services/python/scripts/backtest_scanner_long.py`

- [ ] **Step 1: Add detect_strategy_b()**

```python
def detect_strategy_b(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    volume: np.ndarray,
    start_idx: int,
    end_idx: int,
) -> list[int]:
    """Strategy B: Price up 5% in 32 bars + at least one 2x volume bar.

    Returns list of bar indices where entry signal fires.
    """
    n = len(close)
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    entries = []
    cooldown = 0

    for i in range(max(start_idx, VOL_MA_PERIOD + PRICE_LOOKBACK_B), min(end_idx, n)):
        if cooldown > 0:
            cooldown -= 1
            continue

        # Gate 1: Price moved +5% in last 32 bars
        lookback_low = np.min(low[i - PRICE_LOOKBACK_B : i])
        if lookback_low <= 0:
            continue
        price_move = (close[i] - lookback_low) / lookback_low
        if price_move < PRICE_MOVE_THRESH:
            continue

        # Gate 2: At least one 2x volume bar in the lookback window
        has_vol_spike = False
        for j in range(i - PRICE_LOOKBACK_B, i + 1):
            if not np.isnan(vol_ma[j]) and vol_ma[j] > 0:
                if volume[j] / vol_ma[j] >= VOL_SPIKE_B:
                    has_vol_spike = True
                    break
        if not has_vol_spike:
            continue

        entries.append(i)
        cooldown = COOLDOWN_BARS

    return entries
```

- [ ] **Step 2: Commit**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): add Strategy B — rolling price scan"
```

---

### Task 5: Strategy C — Piggyback on Volume Breakouts

**Files:**
- Modify: `services/python/scripts/backtest_scanner_long.py`

- [ ] **Step 1: Add load_volume_breakouts() and detect_strategy_c()**

```python
def load_volume_breakouts() -> pd.DataFrame:
    """Load long volume breakouts with vol_ratio >= 2 from DB."""
    query = """
        SELECT symbol, signal_time, vol_ratio, close as signal_close
        FROM volume_breakouts
        WHERE direction = 1 AND vol_ratio >= 2.0 AND timeframe = '15m'
        ORDER BY symbol, signal_time
    """
    df = pd.read_sql(query, engine)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    print(f"Loaded {len(df)} long volume breakouts")
    return df


def detect_strategy_c(
    close: np.ndarray,
    low: np.ndarray,
    times: np.ndarray,
    vb_times: list,
    start_idx: int,
    end_idx: int,
) -> list[int]:
    """Strategy C: Volume breakout signal + price already up 5% in 96 bars.

    vb_times: list of signal_time timestamps from volume_breakouts for this symbol.
    Returns list of bar indices where entry signal fires.
    """
    if not vb_times:
        return []

    entries = []
    cooldown = 0
    vb_set = set(vb_times)

    for i in range(max(start_idx, PRICE_LOOKBACK_C), min(end_idx, len(close))):
        if cooldown > 0:
            cooldown -= 1
            continue

        # Gate 1: This bar matches a volume breakout signal
        bar_time = pd.Timestamp(times[i])
        if bar_time not in vb_set:
            continue

        # Gate 2: Price already up 5% from 96-bar low
        lookback_low = np.min(low[i - PRICE_LOOKBACK_C : i])
        if lookback_low <= 0:
            continue
        price_move = (close[i] - lookback_low) / lookback_low
        if price_move < PRICE_MOVE_THRESH:
            continue

        entries.append(i)
        cooldown = COOLDOWN_BARS

    return entries
```

- [ ] **Step 2: Commit**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): add Strategy C — piggyback on volume breakouts"
```

---

### Task 6: Orchestrator + Summary Printer

**Files:**
- Modify: `services/python/scripts/backtest_scanner_long.py`

- [ ] **Step 1: Add run_backtest() and print_summary()**

```python
def run_backtest():
    """Run all 3 strategies across all big movers and compare."""
    big_movers = load_big_movers()
    vb_df = load_volume_breakouts()

    # Group by symbol
    bm_by_symbol = big_movers.groupby("symbol")
    vb_by_symbol = vb_df.groupby("symbol")

    results = {"A": [], "B": [], "C": []}
    coins_processed = 0
    coins_skipped = 0

    symbols = big_movers["symbol"].unique()
    print(f"\nProcessing {len(symbols)} coins...")

    for symbol in symbols:
        sym_dir = RAW_DIR / symbol
        df = load_coin(sym_dir)
        if df is None or len(df) < PRICE_LOOKBACK_A + TIMEOUT_BARS:
            coins_skipped += 1
            continue

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        times = df["open_time"].values

        # Get this coin's big mover windows
        bm_rows = bm_by_symbol.get_group(symbol)

        for _, bm in bm_rows.iterrows():
            # Find bar indices for the window: move_start - 48 to move_peak + 96
            move_start_ts = pd.Timestamp(bm["move_start"]).tz_localize(None)
            move_peak_ts = pd.Timestamp(bm["move_peak"]).tz_localize(None)

            times_naive = pd.DatetimeIndex(times).tz_localize(None)
            start_idx = max(0, np.searchsorted(times_naive, move_start_ts) - 48)
            end_idx = min(len(close), np.searchsorted(times_naive, move_peak_ts) + 96)

            if end_idx - start_idx < 50:
                continue

            # Strategy A
            for entry_bar in detect_strategy_a(close, high, low, volume, start_idx, end_idx):
                trade = simulate_trade(close, high, low, entry_bar)
                if trade:
                    trade["symbol"] = symbol
                    trade["strategy"] = "A"
                    trade["signal_time"] = str(times[entry_bar])
                    results["A"].append(trade)

            # Strategy B
            for entry_bar in detect_strategy_b(close, high, low, volume, start_idx, end_idx):
                trade = simulate_trade(close, high, low, entry_bar)
                if trade:
                    trade["symbol"] = symbol
                    trade["strategy"] = "B"
                    trade["signal_time"] = str(times[entry_bar])
                    results["B"].append(trade)

            # Strategy C
            sym_vb_times = []
            if symbol in vb_by_symbol.groups:
                sym_vb = vb_by_symbol.get_group(symbol)
                sym_vb_times = sym_vb["signal_time"].tolist()
            for entry_bar in detect_strategy_c(close, low, times, sym_vb_times, start_idx, end_idx):
                trade = simulate_trade(close, high, low, entry_bar)
                if trade:
                    trade["symbol"] = symbol
                    trade["strategy"] = "C"
                    trade["signal_time"] = str(times[entry_bar])
                    results["C"].append(trade)

        coins_processed += 1
        if coins_processed % 20 == 0:
            total_trades = sum(len(v) for v in results.values())
            print(f"  {coins_processed}/{len(symbols)} coins, {total_trades} trades so far...")

    print(f"\nDone: {coins_processed} coins processed, {coins_skipped} skipped (no CSV)")
    return results


def print_summary(results: dict):
    """Print comparison table for all strategies."""
    print("\n" + "=" * 80)
    print("SCANNER LONG BACKTEST — STRATEGY COMPARISON")
    print("=" * 80)
    print(f"Exit rules: SL={STOP_LOSS_PCT:.0%} | TP={TAKE_PROFIT_PCT:.0%} | Timeout={TIMEOUT_BARS} bars (1 week)")
    print()

    header = f"{'Strategy':<12} {'Trades':>7} {'WinRate':>8} {'AvgPnL':>8} {'PF':>8} {'AvgBars':>8} {'TP%':>6} {'SL%':>6} {'TO%':>6}"
    print(header)
    print("-" * len(header))

    best_pf = -1
    best_strategy = None

    for name in ["A", "B", "C"]:
        trades = results[name]
        n = len(trades)
        if n == 0:
            print(f"  {name:<12} {'NO TRADES':>7}")
            continue

        pnls = [t["pnl_pct"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        bars = [t["bars_held"] for t in trades]

        win_rate = len(wins) / n
        avg_pnl = np.mean(pnls)
        gross_profit = sum(wins) if wins else 0
        gross_loss = abs(sum(losses)) if losses else 1
        pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        avg_bars = np.mean(bars)

        tp_count = sum(1 for t in trades if t["exit_reason"] == "take_profit")
        sl_count = sum(1 for t in trades if t["exit_reason"] == "stop_loss")
        to_count = sum(1 for t in trades if t["exit_reason"] == "timeout")

        label = f"  {name}: "
        if name == "A":
            label += "Vol+Momentum"
        elif name == "B":
            label += "PriceScan"
        elif name == "C":
            label += "VB Piggyback"

        print(f"{label:<12} {n:>7} {win_rate:>7.1%} {avg_pnl:>7.2%} {pf:>7.2f} {avg_bars:>7.0f} {tp_count/n:>5.0%} {sl_count/n:>5.0%} {to_count/n:>5.0%}")

        if pf > best_pf:
            best_pf = pf
            best_strategy = name

    print()
    print(f">>> WINNER: Strategy {best_strategy} (PF={best_pf:.2f})")
    print()

    # Per-strategy breakdown by exit reason
    for name in ["A", "B", "C"]:
        trades = results[name]
        if not trades:
            continue
        df = pd.DataFrame(trades)
        print(f"\n--- Strategy {name} breakdown ---")
        print(f"  Unique coins: {df['symbol'].nunique()}")
        by_exit = df.groupby("exit_reason").agg(
            count=("pnl_pct", "count"),
            avg_pnl=("pnl_pct", "mean"),
            avg_bars=("bars_held", "mean"),
        )
        print(by_exit.to_string(float_format="%.3f"))
```

- [ ] **Step 2: Add `__main__` block**

```python
if __name__ == "__main__":
    results = run_backtest()
    print_summary(results)
```

- [ ] **Step 3: Commit**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): add orchestrator and summary printer"
```

---

### Task 7: DB Save + Run Full Backtest

**Files:**
- Modify: `services/python/scripts/backtest_scanner_long.py`

- [ ] **Step 1: Add save_to_db()**

```python
def save_to_db(results: dict):
    """Save all trades to scanner_long_backtest table."""
    all_trades = []
    for strategy, trades in results.items():
        all_trades.extend(trades)

    if not all_trades:
        print("No trades to save")
        return

    df = pd.DataFrame(all_trades)

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_long_backtest"))
        conn.commit()

    df.to_sql("scanner_long_backtest", engine, if_exists="replace", index=False)
    print(f"Saved {len(df)} trades to scanner_long_backtest table")
```

- [ ] **Step 2: Update `__main__` to include save**

```python
if __name__ == "__main__":
    results = run_backtest()
    print_summary(results)
    save_to_db(results)
```

- [ ] **Step 3: Run the full backtest**

```bash
cd services/python && python scripts/backtest_scanner_long.py
```

Expected: processes ~100+ coins, prints strategy comparison table with trades/WR/PF for each.

- [ ] **Step 4: Commit**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): add DB save and run full backtest"
```

---

### Task 8: Analyze Results + Declare Winner

**Files:**
- No new files — this is analysis of the backtest output

- [ ] **Step 1: Review backtest output**

Look at the printed summary. Key metrics to evaluate:
- Which strategy has the highest profit factor?
- Which has the most trades (statistical significance)?
- What % hit take profit vs stop loss vs timeout?
- Are timeout trades net positive or negative?

- [ ] **Step 2: Run quick DB analysis on the winner**

```bash
cd services/python && python -c "
import pandas as pd
from sqlalchemy import create_engine
engine = create_engine('postgresql://postgres:MySQL100%25@localhost:5432/market')
df = pd.read_sql('SELECT * FROM scanner_long_backtest', engine)
print('=== MONTHLY BREAKDOWN (BEST STRATEGY) ===')
best = df.groupby('strategy').apply(lambda x: x['pnl_pct'].sum()).idxmax()
winner = df[df['strategy'] == best].copy()
winner['month'] = pd.to_datetime(winner['signal_time']).dt.to_period('M')
monthly = winner.groupby('month').agg(
    trades=('pnl_pct', 'count'),
    total_pnl=('pnl_pct', 'sum'),
    win_rate=('pnl_pct', lambda x: (x > 0).mean()),
)
print(monthly.to_string(float_format='%.3f'))
"
```

- [ ] **Step 3: Commit final results**

```bash
git add services/python/scripts/backtest_scanner_long.py
git commit -m "feat(scanner-long): complete backtest with 3-strategy comparison"
```
