import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def make_trades_df(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    ts = pd.date_range("2024-06-01", periods=n, freq="15min", tz="UTC")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    high = close * (1 + rng.uniform(0, 0.005, n))
    low = close * (1 - rng.uniform(0, 0.005, n))
    conviction = rng.normal(0, 0.1, n)
    conviction[60] = 0.25   # After ATR warmup (56 bars)
    conviction[120] = -0.25
    return pd.DataFrame({
        "timestamp": ts, "close": close, "high": high, "low": low,
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
    result = backtest_coin(df, conviction_threshold=0.99, initial_capital=10000.0)
    assert result["metrics"]["total_trades"] == 0
    assert result["metrics"]["total_return_pct"] == 0.0


def test_backtest_fees_reduce_returns():
    from src.ml.backtester_v4 import backtest_coin
    df = make_trades_df(1000, seed=123)
    r_fees = backtest_coin(df, conviction_threshold=0.15, fee_rate=0.0003)
    r_no_fees = backtest_coin(df, conviction_threshold=0.15, fee_rate=0.0)
    assert r_fees["metrics"]["total_return_pct"] <= r_no_fees["metrics"]["total_return_pct"]


def test_stop_loss_triggers():
    from src.ml.backtester_v4 import backtest_coin
    n = 100
    close = np.full(n, 100.0)
    close[72:] = 94.0  # 6% drop after entry at bar 70 (after ATR warmup)
    high = close + 0.1
    low = close - 0.1
    low[72] = 93.0  # Low pierces stop
    conviction = np.zeros(n)
    conviction[70] = 0.30  # Signal after ATR warmup (56 bars)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC"),
        "close": close, "high": high, "low": low, "conviction": conviction,
    })
    # Use fixed stop so test is deterministic
    result = backtest_coin(df, conviction_threshold=0.20, stop_loss_pct=0.03)
    trades = result["trades"]
    assert len(trades) >= 1
    assert trades[0]["exit_reason"] == "stop_loss"
    assert trades[0]["pnl_pct"] < 0


def test_compute_metrics():
    from src.ml.backtester_v4 import compute_metrics
    trades = [
        {"pnl_pct": 0.02, "hold_bars": 16, "fee_paid": 0.3, "funding_paid": 0,
         "raw_return": 0.02, "position_size": 1000},
        {"pnl_pct": -0.01, "hold_bars": 8, "fee_paid": 0.3, "funding_paid": 0,
         "raw_return": -0.01, "position_size": 1000},
        {"pnl_pct": 0.03, "hold_bars": 32, "fee_paid": 0.3, "funding_paid": 0,
         "raw_return": 0.03, "position_size": 1000},
        {"pnl_pct": 0.015, "hold_bars": 12, "fee_paid": 0.3, "funding_paid": 0,
         "raw_return": 0.015, "position_size": 1000},
    ]
    equity = pd.Series([10000, 10200, 10100, 10400, 10550])
    m = compute_metrics(trades, equity)
    assert m["total_trades"] == 4
    assert m["win_rate"] == 0.75
    assert m["profit_factor"] > 1.0
    assert "sharpe_ratio" in m
    assert "max_drawdown_pct" in m
