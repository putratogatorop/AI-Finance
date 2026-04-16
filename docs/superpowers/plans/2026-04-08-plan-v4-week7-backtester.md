# V4 Week 7-8: Honest Backtester Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a vectorized backtester that simulates trading the LightGBM OOS predictions with realistic fees, funding costs, and slippage, producing the metrics needed to validate against spec success criteria.

**Architecture:** The backtester loads OOS predictions (from walk-forward training) and 15m price data, generates trades when conviction exceeds a threshold, simulates position management (stop-loss, take-profit, time-exit), deducts fees and funding costs, and computes Sharpe, profit factor, win rate, drawdown, and other metrics. It tests multiple conviction thresholds to find the optimal operating point.

**Tech Stack:** Python 3.12, pandas, numpy

---

## Key Data Facts

- OOS predictions: 155K rows per coin, conviction range [-0.33, +0.38], 98.8% predicted FLAT
- The 0.35 threshold from spec produces almost no trades — backtester must sweep thresholds
- Fees: 0.03% round-trip average (maker entry + maker exit)
- Funding: ~0.01% every 8 hours on open positions
- 15m candles available for intra-bar stop/TP simulation using high/low

## File Structure

| File | Responsibility |
|---|---|
| `src/ml/backtester_v4.py` | Core backtester engine: trade simulation, position management, metrics |
| `tests/test_backtester_v4.py` | Unit tests for backtester logic |
| `scripts/run_backtest_v4.py` | Load data, run backtest across coins and thresholds, save report |

---

### Task 1: Core Backtester Engine

**Files:**
- Create: `services/python/src/ml/backtester_v4.py`
- Create: `services/python/tests/test_backtester_v4.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_backtester_v4.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import pytest


def make_trades_df(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Make a simple OOS predictions + price DataFrame for testing."""
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2024-06-01", periods=n, freq="15min", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))

    conviction = rng.normal(0, 0.1, n)
    # Inject a few strong signals
    conviction[10] = 0.25   # LONG signal
    conviction[50] = -0.25  # SHORT signal

    return pd.DataFrame({
        "timestamp": ts,
        "close": close,
        "high": high,
        "low": low,
        "conviction": conviction,
    })


def test_backtest_basic():
    from src.ml.backtester_v4 import backtest_coin

    df = make_trades_df(500)
    result = backtest_coin(df, conviction_threshold=0.20, initial_capital=10000.0)

    assert "trades" in result
    assert "equity_curve" in result
    assert "metrics" in result
    assert len(result["trades"]) > 0
    assert result["metrics"]["total_trades"] > 0


def test_backtest_no_trades_high_threshold():
    from src.ml.backtester_v4 import backtest_coin

    df = make_trades_df(500)
    # Set threshold so high no signals trigger
    result = backtest_coin(df, conviction_threshold=0.99, initial_capital=10000.0)

    assert result["metrics"]["total_trades"] == 0
    assert result["metrics"]["total_return_pct"] == 0.0


def test_backtest_fees_reduce_returns():
    from src.ml.backtester_v4 import backtest_coin

    df = make_trades_df(1000, seed=123)
    # Same data, with and without fees
    r_fees = backtest_coin(df, conviction_threshold=0.15, fee_rate=0.0003)
    r_no_fees = backtest_coin(df, conviction_threshold=0.15, fee_rate=0.0)

    # Fees should reduce total return
    assert r_fees["metrics"]["total_return_pct"] <= r_no_fees["metrics"]["total_return_pct"]


def test_stop_loss_triggers():
    from src.ml.backtester_v4 import backtest_coin

    # Create data where price drops 5% immediately after signal
    n = 100
    close = np.full(n, 100.0)
    close[12:] = 94.0  # 6% drop after entry at bar 10
    high = close + 0.1
    low = close - 0.1
    low[12] = 93.0  # Ensure low is below stop

    conviction = np.zeros(n)
    conviction[10] = 0.30  # Strong long signal

    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "close": close, "high": high, "low": low,
        "conviction": conviction,
    })

    result = backtest_coin(df, conviction_threshold=0.20, stop_loss_pct=0.03)
    trades = result["trades"]
    assert len(trades) >= 1
    assert trades[0]["exit_reason"] == "stop_loss"
    assert trades[0]["pnl_pct"] < 0


def test_compute_metrics():
    from src.ml.backtester_v4 import compute_metrics

    trades = [
        {"pnl_pct": 0.02, "hold_bars": 16, "fee_paid": 0.3},
        {"pnl_pct": -0.01, "hold_bars": 8, "fee_paid": 0.3},
        {"pnl_pct": 0.03, "hold_bars": 32, "fee_paid": 0.3},
        {"pnl_pct": 0.015, "hold_bars": 12, "fee_paid": 0.3},
    ]
    equity = pd.Series([10000, 10200, 10100, 10400, 10550])

    m = compute_metrics(trades, equity)

    assert m["total_trades"] == 4
    assert m["win_rate"] == 0.75  # 3 wins out of 4
    assert m["profit_factor"] > 1.0
    assert "sharpe_ratio" in m
    assert "max_drawdown_pct" in m
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd services/python && python -m pytest tests/test_backtester_v4.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement backtester_v4.py**

```python
# src/ml/backtester_v4.py
"""V4 Vectorized Backtester.

Simulates trading LightGBM OOS predictions with realistic costs:
- Entry/exit fees (maker 0.015%, round-trip ~0.03%)
- Funding costs (~0.01% per 8h on open positions)
- Stop-loss using intra-bar high/low
- Take-profit and time-based exits
- No look-ahead bias: decisions use only current-bar data
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def backtest_coin(
    df: pd.DataFrame,
    conviction_threshold: float = 0.20,
    initial_capital: float = 10_000.0,
    leverage: float = 3.0,
    fee_rate: float = 0.0003,       # 0.03% round-trip
    funding_rate_8h: float = 0.0001, # 0.01% per 8h
    stop_loss_pct: float = 0.025,    # 2.5%
    take_profit_pct: float = 0.05,   # 5%
    max_hold_bars: int = 96,         # 24h at 15min
    risk_per_trade: float = 0.02,    # 2% of equity per trade
) -> dict:
    """Run backtest on a single coin's OOS predictions.

    Parameters
    ----------
    df : DataFrame with columns: timestamp, close, high, low, conviction
    conviction_threshold : minimum |conviction| to enter a trade
    initial_capital : starting equity in USD
    leverage : position leverage multiplier
    fee_rate : round-trip fee as fraction (entry + exit)
    funding_rate_8h : funding cost per 8 hours as fraction
    stop_loss_pct : stop-loss distance as fraction of entry price
    take_profit_pct : take-profit distance as fraction of entry price
    max_hold_bars : maximum bars to hold before time exit
    risk_per_trade : fraction of equity risked per trade

    Returns
    -------
    dict with keys: trades, equity_curve, metrics
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    conviction = df["conviction"].values.astype(float)
    n = len(df)

    equity = initial_capital
    trades = []
    equity_curve = [equity]

    # Position state
    in_position = False
    entry_price = 0.0
    entry_bar = 0
    direction = 0  # 1=long, -1=short
    position_size = 0.0

    for i in range(1, n):
        if in_position:
            bars_held = i - entry_bar

            # Check stop-loss using intra-bar low/high
            if direction == 1:
                stop_price = entry_price * (1 - stop_loss_pct)
                tp_price = entry_price * (1 + take_profit_pct)
                hit_stop = low[i] <= stop_price
                hit_tp = high[i] >= tp_price
            else:
                stop_price = entry_price * (1 + stop_loss_pct)
                tp_price = entry_price * (1 - take_profit_pct)
                hit_stop = high[i] >= stop_price
                hit_tp = low[i] <= tp_price

            time_exit = bars_held >= max_hold_bars

            # Determine exit
            exit_price = None
            exit_reason = None

            if hit_stop:
                exit_price = stop_price
                exit_reason = "stop_loss"
            elif hit_tp:
                exit_price = tp_price
                exit_reason = "take_profit"
            elif time_exit:
                exit_price = close[i]
                exit_reason = "time_exit"

            # Signal flip: conviction reverses strongly
            if exit_reason is None:
                if direction == 1 and conviction[i] < -conviction_threshold:
                    exit_price = close[i]
                    exit_reason = "signal_flip"
                elif direction == -1 and conviction[i] > conviction_threshold:
                    exit_price = close[i]
                    exit_reason = "signal_flip"

            if exit_price is not None:
                # Calculate PnL
                if direction == 1:
                    raw_return = (exit_price - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price) / entry_price

                # Fees: half on entry, half on exit
                fee_cost = fee_rate * position_size
                # Funding: every 32 bars (8h at 15min)
                funding_periods = bars_held / 32
                funding_cost = funding_rate_8h * position_size * funding_periods

                gross_pnl = raw_return * position_size
                net_pnl = gross_pnl - fee_cost - funding_cost

                equity += net_pnl

                trades.append({
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "hold_bars": bars_held,
                    "raw_return": raw_return,
                    "pnl_pct": net_pnl / position_size if position_size > 0 else 0,
                    "pnl_usd": net_pnl,
                    "fee_paid": fee_cost,
                    "funding_paid": funding_cost,
                    "exit_reason": exit_reason,
                    "position_size": position_size,
                })

                in_position = False

        # Entry logic (only if not in position)
        if not in_position:
            if abs(conviction[i]) >= conviction_threshold:
                direction = 1 if conviction[i] > 0 else -1
                entry_price = close[i]
                entry_bar = i
                # Position size: risk_per_trade * equity * leverage
                position_size = equity * risk_per_trade * leverage / stop_loss_pct
                position_size = min(position_size, equity * leverage)  # Cap at max leverage
                in_position = True

        equity_curve.append(equity)

    # Close any open position at end
    if in_position:
        exit_price = close[-1]
        if direction == 1:
            raw_return = (exit_price - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price) / entry_price

        bars_held = n - 1 - entry_bar
        fee_cost = fee_rate * position_size
        funding_cost = funding_rate_8h * position_size * (bars_held / 32)
        net_pnl = raw_return * position_size - fee_cost - funding_cost
        equity += net_pnl

        trades.append({
            "entry_bar": entry_bar, "exit_bar": n - 1,
            "direction": direction, "entry_price": entry_price,
            "exit_price": exit_price, "hold_bars": bars_held,
            "raw_return": raw_return,
            "pnl_pct": net_pnl / position_size if position_size > 0 else 0,
            "pnl_usd": net_pnl, "fee_paid": fee_cost,
            "funding_paid": funding_cost, "exit_reason": "end_of_data",
            "position_size": position_size,
        })
        equity_curve[-1] = equity

    eq_series = pd.Series(equity_curve)
    metrics = compute_metrics(trades, eq_series) if trades else _empty_metrics()

    return {"trades": trades, "equity_curve": eq_series, "metrics": metrics}


def compute_metrics(trades: list[dict], equity_curve: pd.Series) -> dict:
    """Compute performance metrics from trade list and equity curve."""
    n_trades = len(trades)
    if n_trades == 0:
        return _empty_metrics()

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / n_trades if n_trades > 0 else 0
    avg_win = np.mean(wins) if wins else 0
    avg_loss = abs(np.mean(losses)) if losses else 0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    # Sharpe from equity curve returns
    eq_returns = equity_curve.pct_change().dropna()
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        # Annualize: 96 bars/day * 365 days
        sharpe = eq_returns.mean() / eq_returns.std() * np.sqrt(96 * 365)
    else:
        sharpe = 0.0

    # Max drawdown
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_dd = abs(drawdown.min())

    # Hold time stats
    hold_bars = [t["hold_bars"] for t in trades]
    avg_hold_hours = np.mean(hold_bars) * 0.25  # 15min = 0.25h

    # Fee drag
    total_fees = sum(t["fee_paid"] + t.get("funding_paid", 0) for t in trades)
    gross_pnl = sum(t["raw_return"] * t["position_size"] for t in trades)
    fee_drag = total_fees / gross_pnl if gross_pnl > 0 else 0

    total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    # Profitable months
    # Approximate: count how many 30-day periods are positive
    eq_monthly = equity_curve.iloc[::2880]  # Sample every ~month
    monthly_returns = eq_monthly.pct_change().dropna()
    profitable_months_pct = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0

    return {
        "total_trades": n_trades,
        "total_return_pct": float(total_return_pct),
        "win_rate": float(win_rate),
        "avg_win_pct": float(avg_win),
        "avg_loss_pct": float(avg_loss),
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "avg_hold_hours": float(avg_hold_hours),
        "total_fees": float(total_fees),
        "fee_drag_pct": float(fee_drag),
        "profitable_months_pct": float(profitable_months_pct),
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0,
        "total_return_pct": 0.0,
        "win_rate": 0.0,
        "avg_win_pct": 0.0,
        "avg_loss_pct": 0.0,
        "profit_factor": 0.0,
        "sharpe_ratio": 0.0,
        "max_drawdown_pct": 0.0,
        "avg_hold_hours": 0.0,
        "total_fees": 0.0,
        "fee_drag_pct": 0.0,
        "profitable_months_pct": 0.0,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd services/python && python -m pytest tests/test_backtester_v4.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add services/python/src/ml/backtester_v4.py services/python/tests/test_backtester_v4.py
git commit -m "feat: add v4 backtester engine with fees, funding, and stop-loss"
```

---

### Task 2: Backtest Runner Script

**Files:**
- Create: `services/python/scripts/run_backtest_v4.py`

- [ ] **Step 1: Implement the runner script**

```python
# scripts/run_backtest_v4.py
"""Run v4 backtest across all coins and conviction thresholds.

Loads OOS predictions from walk-forward training + 15m price data,
runs backtests at multiple conviction thresholds, and produces a
summary report with spec success criteria validation.

Outputs: models/v4/backtest_report.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.ml.backtester_v4 import backtest_coin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("C:/Users/togat/Desktop/AI-Finance/models/v4")
DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"

V4_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

# Spec success criteria
SPEC_CRITERIA = {
    "sharpe_ratio": 1.5,
    "profit_factor": 1.4,
    "win_rate": 0.48,
    "max_drawdown_pct": 0.25,
    "profitable_months_pct": 0.60,
    "min_trades_per_week": 5,
    "max_trades_per_week": 15,
}


def load_price_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, open, high, low, close
        FROM asset_prices_15m
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"asset": asset})


def load_oos_predictions(asset: str) -> pd.DataFrame:
    path = MODEL_DIR / asset / "oos_predictions.parquet"
    return pd.read_parquet(path)


def merge_predictions_with_prices(prices: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    """Merge OOS predictions with price data by timestamp."""
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)

    merged = prices.merge(preds[["timestamp", "conviction"]], on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def run_asset_backtest(engine, asset: str) -> dict:
    """Run backtest for one asset at multiple thresholds."""
    logger.info(f"  {asset}: loading data...")
    prices = load_price_data(engine, asset)
    preds = load_oos_predictions(asset)
    merged = merge_predictions_with_prices(prices, preds)
    logger.info(f"  {asset}: {len(merged):,} merged rows, "
                f"conviction range [{merged['conviction'].min():.3f}, {merged['conviction'].max():.3f}]")

    results = {}
    for threshold in THRESHOLDS:
        r = backtest_coin(merged, conviction_threshold=threshold)
        m = r["metrics"]
        n_trades = m["total_trades"]

        # Calculate trades per week (total OOS period in weeks)
        total_bars = len(merged)
        total_weeks = total_bars / (96 * 7)  # 96 bars/day * 7 days
        trades_per_week = n_trades / total_weeks if total_weeks > 0 else 0

        m["trades_per_week"] = float(trades_per_week)
        results[f"{threshold:.2f}"] = m

        logger.info(f"  {asset} @ {threshold:.2f}: "
                    f"{n_trades} trades ({trades_per_week:.1f}/wk) "
                    f"ret={m['total_return_pct']:.1%} "
                    f"sharpe={m['sharpe_ratio']:.2f} "
                    f"wr={m['win_rate']:.1%} "
                    f"dd={m['max_drawdown_pct']:.1%}")

    return results


def check_spec_criteria(metrics: dict) -> dict:
    """Check metrics against spec success criteria."""
    checks = {}
    checks["sharpe_ratio"] = metrics.get("sharpe_ratio", 0) >= SPEC_CRITERIA["sharpe_ratio"]
    checks["profit_factor"] = metrics.get("profit_factor", 0) >= SPEC_CRITERIA["profit_factor"]
    checks["win_rate"] = metrics.get("win_rate", 0) >= SPEC_CRITERIA["win_rate"]
    checks["max_drawdown"] = metrics.get("max_drawdown_pct", 1) <= SPEC_CRITERIA["max_drawdown_pct"]
    checks["profitable_months"] = metrics.get("profitable_months_pct", 0) >= SPEC_CRITERIA["profitable_months_pct"]
    tpw = metrics.get("trades_per_week", 0)
    checks["trade_frequency"] = SPEC_CRITERIA["min_trades_per_week"] <= tpw <= SPEC_CRITERIA["max_trades_per_week"]
    checks["all_pass"] = all(checks.values())
    return checks


def main():
    engine = create_engine(DB_URL)
    start = time.time()

    logger.info("V4 Backtest Runner")
    logger.info(f"Assets: {V4_ASSETS}")
    logger.info(f"Thresholds: {THRESHOLDS}")

    report = {"assets": {}, "best_per_asset": {}}

    for asset in V4_ASSETS:
        logger.info(f"\nBacktesting {asset}...")
        asset_results = run_asset_backtest(engine, asset)
        report["assets"][asset] = asset_results

        # Find best threshold by Sharpe (with min 10 trades)
        best_threshold = None
        best_sharpe = -999
        for thresh_str, m in asset_results.items():
            if m["total_trades"] >= 10 and m["sharpe_ratio"] > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_threshold = thresh_str

        if best_threshold:
            best_m = asset_results[best_threshold]
            spec_check = check_spec_criteria(best_m)
            report["best_per_asset"][asset] = {
                "threshold": best_threshold,
                "metrics": best_m,
                "spec_compliance": spec_check,
            }

    # Save report
    report_path = MODEL_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"\nBest results per asset:")
    for asset, info in report.get("best_per_asset", {}).items():
        m = info["metrics"]
        spec = info["spec_compliance"]
        status = "PASS" if spec["all_pass"] else "FAIL"
        logger.info(f"  {asset} @ {info['threshold']}: "
                    f"sharpe={m['sharpe_ratio']:.2f} "
                    f"pf={m['profit_factor']:.2f} "
                    f"wr={m['win_rate']:.1%} "
                    f"dd={m['max_drawdown_pct']:.1%} "
                    f"tpw={m['trades_per_week']:.1f} "
                    f"[{status}]")

    engine.dispose()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add services/python/scripts/run_backtest_v4.py
git commit -m "feat: add v4 backtest runner with threshold sweep and spec validation"
```

- [ ] **Step 3: Run the backtest**

Run: `cd services/python && python scripts/run_backtest_v4.py`
Expected: Tests 8 thresholds × 7 assets = 56 backtests. Takes ~1-5 minutes. Outputs `models/v4/backtest_report.json`.

- [ ] **Step 4: Review results against spec criteria**

Check the output for:
- Sharpe > 1.5
- Profit factor > 1.4
- Win rate > 48%
- Max drawdown < 25%
- 5-15 trades per week
- Red flag: Sharpe > 3.0 or win rate > 65%
