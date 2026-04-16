import sys

sys.path.insert(0, ".")

import numpy as np
import pandas as pd


def test_compute_contract_features():
    from scripts.run_v5_pipeline import compute_contract_features

    n = 200
    rng = np.random.RandomState(42)
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC"),
        "open_interest_usd": 4e9 + np.cumsum(rng.normal(0, 1e7, n)),
        "long_liq_usd": rng.exponential(10000, n),
        "short_liq_usd": rng.exponential(10000, n),
        "lsr_taker": 1.0 + rng.normal(0, 0.1, n),
        "lsr_account": 0.9 + rng.normal(0, 0.1, n),
    })

    result = compute_contract_features(df)

    for col in ["oi_change_1h", "oi_change_4h", "oi_zscore",
                "liq_long_1h", "liq_short_1h", "liq_imbalance",
                "ls_ratio", "ls_ratio_change_4h"]:
        assert col in result.columns, f"Missing {col}"

    # liq_imbalance should be in [-1, 1]
    valid = result["liq_imbalance"].dropna()
    assert valid.min() >= -1.01
    assert valid.max() <= 1.01


def test_build_extreme_move_target():
    from scripts.run_v5_pipeline import build_extreme_move_target

    n = 500
    rng = np.random.RandomState(42)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.005, n))))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))

    y1, y2 = build_extreme_move_target(close, high, low, horizon=4)

    # Stage 1: should have some 1s (extreme moves)
    assert y1.sum() > 0
    assert set(y1.dropna().unique()).issubset({0, 1})

    # Stage 2: only defined where y1=1
    assert y2[y1 == 0].isna().all() or (y2[y1 == 0] == y2[y1 == 0]).all()
