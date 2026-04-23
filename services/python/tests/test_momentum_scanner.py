# tests/test_momentum_scanner.py
import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def make_coin_data(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic 4h OHLCV with a breakout at bar 100."""
    rng = np.random.RandomState(seed)
    close = 1.0 * np.exp(np.cumsum(rng.normal(0, 0.005, n)))
    # Inject breakout at bar 100: price jumps 8%, volume spikes 5x
    close[100:] *= 1.08
    volume = rng.uniform(1000, 5000, n)
    volume[100] = 25000  # 5x spike
    volume[101] = 20000  # Confirms
    high = close * (1 + rng.uniform(0, 0.02, n))
    low = close * (1 - rng.uniform(0, 0.02, n))

    return pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": close * (1 + rng.normal(0, 0.005, n)),
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


def test_detect_breakouts():
    from scripts.backtest_momentum_scanner import detect_breakouts

    df = make_coin_data(200)
    signals = detect_breakouts(
        df, vol_mult=3.0, price_thresh=0.05, lookback=20
    )

    # Should detect the breakout at bar 100-101
    assert len(signals) > 0
    assert any(99 <= s["bar"] <= 102 for s in signals)
    assert signals[0]["direction"] == 1  # Long (price went up)


def test_no_breakout_in_quiet_data():
    from scripts.backtest_momentum_scanner import detect_breakouts

    rng = np.random.RandomState(99)
    n = 200
    close = 1.0 + np.cumsum(rng.normal(0, 0.001, n))
    df = pd.DataFrame({
        "open_time": pd.date_range("2024-01-01", periods=n, freq="4h"),
        "open": close, "high": close + 0.01, "low": close - 0.01,
        "close": close,
        "volume": rng.uniform(1000, 2000, n),  # No spikes
    })

    signals = detect_breakouts(df, vol_mult=3.0, price_thresh=0.05, lookback=20)
    assert len(signals) == 0


def test_simulate_trade():
    from scripts.backtest_momentum_scanner import simulate_pullback_trade

    # Price goes up 8%, pulls back 3%, then continues up 15%
    n = 20
    close = np.array([1.0]*5 + [1.08]*3 + [1.05]*2 + [1.15]*5 + [1.20]*5)
    high = close + 0.01
    low = close - 0.01

    volume = np.full(n, 5000.0)
    trade = simulate_pullback_trade(
        close=close, high=high, low=low,
        volume=volume,
        signal_bar=5, direction=1,
        pullback_pct=0.03, atr_stop_mult=2.0,
        atr_value=0.02, max_hold_bars=12,
    )

    assert trade is not None
    assert trade["direction"] == 1
    assert trade["pnl_pct"] > 0  # Should be profitable
