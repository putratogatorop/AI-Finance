# tests/test_backfill_contract_stats.py
import sys

sys.path.insert(0, ".")

from datetime import datetime

SAMPLE_RESPONSE = [
    {
        "time": 1717200000, "open_interest_usd": 4000000000.0,
        "long_liq_usd": 50000.0, "short_liq_usd": 30000.0,
        "lsr_taker": 1.05, "lsr_account": 0.95,
        "top_lsr_size": 1.10, "top_lsr_account": 0.88,
        "open_interest": 50000, "mark_price": 70000.0,
        "long_liq_size": 5, "short_liq_size": 3,
        "long_liq_amount": 50000, "short_liq_amount": 30000,
    },
    {
        "time": 1717200300, "open_interest_usd": 4010000000.0,
        "long_liq_usd": 10000.0, "short_liq_usd": 5000.0,
        "lsr_taker": 1.02, "lsr_account": 0.97,
        "top_lsr_size": 1.08, "top_lsr_account": 0.90,
        "open_interest": 50100, "mark_price": 70050.0,
        "long_liq_size": 1, "short_liq_size": 0,
        "long_liq_amount": 10000, "short_liq_amount": 5000,
    },
]


def test_parse_contract_stats():
    from scripts.backfill_contract_stats import parse_contract_stats

    rows = parse_contract_stats("BTC", SAMPLE_RESPONSE)
    assert len(rows) == 2
    assert rows[0]["asset"] == "BTC"
    assert rows[0]["open_interest_usd"] == 4000000000.0
    assert rows[0]["lsr_taker"] == 1.05
    assert isinstance(rows[0]["timestamp"], datetime)


def test_aggregate_to_1h():
    import numpy as np
    import pandas as pd

    from scripts.backfill_contract_stats import aggregate_5min_to_1h

    # 12 records = 1 hour of 5-min data
    ts = pd.date_range("2024-06-01 00:00", periods=12, freq="5min", tz="UTC")
    data = []
    for i, t in enumerate(ts):
        data.append({
            "asset": "BTC", "timestamp": t,
            "open_interest_usd": 4e9 + i * 1e6,
            "long_liq_usd": 10000.0 + i * 1000,
            "short_liq_usd": 5000.0 + i * 500,
            "lsr_taker": 1.0 + i * 0.01,
            "lsr_account": 0.9 + i * 0.01,
            "top_lsr_size": 1.1, "top_lsr_account": 0.88,
        })

    df = pd.DataFrame(data)
    result = aggregate_5min_to_1h(df)

    assert len(result) == 1
    # OI should be last value of the hour
    assert result.iloc[0]["open_interest_usd"] == 4e9 + 11 * 1e6
    # Liquidations should be summed
    expected_liq_long = sum(10000.0 + i * 1000 for i in range(12))
    assert abs(result.iloc[0]["long_liq_usd"] - expected_liq_long) < 0.01
    # LSR should be mean
    expected_lsr = np.mean([1.0 + i * 0.01 for i in range(12)])
    assert abs(result.iloc[0]["lsr_taker"] - expected_lsr) < 0.001
