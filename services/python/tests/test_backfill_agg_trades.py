# tests/test_backfill_agg_trades.py
import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def test_compute_hourly_trade_metrics():
    from scripts.backfill_agg_trades import compute_hourly_trade_metrics

    n = 1000
    rng = np.random.RandomState(42)
    ts = pd.date_range("2024-06-01", periods=n, freq="5s", tz="UTC")
    qty = rng.exponential(1.0, n)  # Most trades small, few large
    is_buyer = rng.choice([True, False], n)

    df = pd.DataFrame({
        "timestamp": ts,
        "quantity": qty,
        "is_buyer_maker": is_buyer,
    })

    result = compute_hourly_trade_metrics(df)

    assert "large_trade_ratio" in result.columns
    assert "large_trade_imbalance" in result.columns
    assert "trade_count_ratio" in result.columns
    assert len(result) >= 1

    # large_trade_ratio should be between 0 and 1
    assert result["large_trade_ratio"].dropna().min() >= 0
    assert result["large_trade_ratio"].dropna().max() <= 1

    # imbalance should be between -1 and 1
    valid = result["large_trade_imbalance"].dropna()
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01
