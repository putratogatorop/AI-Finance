"""Tests for RLDataLoader."""

import numpy as np
import pandas as pd

from src.rl.data_loader import RLDataLoader


def _make_ohlcv_csv(path, symbol, n_rows=300, start_ts=1640995200000, step=14400000):
    """Create a fake OHLCV CSV file with realistic-ish data."""
    rng = np.random.default_rng(42)
    base_price = 100.0
    prices = base_price + np.cumsum(rng.normal(0, 0.5, n_rows))
    prices = np.abs(prices) + 1  # ensure positive

    df = pd.DataFrame({
        "open_time": [start_ts + i * step for i in range(n_rows)],
        "open": prices,
        "high": prices * (1 + rng.uniform(0, 0.02, n_rows)),
        "low": prices * (1 - rng.uniform(0, 0.02, n_rows)),
        "close": prices + rng.normal(0, 0.3, n_rows),
        "volume": rng.uniform(100, 10000, n_rows),
        "close_time": [start_ts + (i + 1) * step - 1 for i in range(n_rows)],
        "quote_volume": rng.uniform(1e6, 1e8, n_rows),
        "trades": rng.integers(100, 5000, n_rows),
        "taker_buy_base": rng.uniform(50, 5000, n_rows),
        "taker_buy_quote": rng.uniform(1e5, 1e7, n_rows),
        "ignore": 0,
    })
    # Ensure close is positive
    df["close"] = df["close"].abs() + 1

    token_dir = path / symbol
    token_dir.mkdir(parents=True, exist_ok=True)
    csv_path = token_dir / f"{symbol}-4h-2022-01.csv"
    df.to_csv(csv_path, index=False)
    return df


def _make_manifest(path, symbols_months):
    """Create manifest.csv with symbol,months columns."""
    df = pd.DataFrame(symbols_months, columns=["symbol", "months"])
    df.to_csv(path / "manifest.csv", index=False)


class TestLoadTokenOhlcv:
    def test_loads_and_deduplicates(self, tmp_path):
        """Load CSVs for a token: concatenates, deduplicates, sorts."""
        rng = np.random.default_rng(0)
        base_ts = 1640995200000
        step = 14400000

        # File 1: rows 0-9
        token_dir = tmp_path / "AAAUSDT"
        token_dir.mkdir()
        rows_a = pd.DataFrame({
            "open_time": [base_ts + i * step for i in range(10)],
            "open": rng.uniform(10, 20, 10),
            "high": rng.uniform(20, 30, 10),
            "low": rng.uniform(5, 10, 10),
            "close": rng.uniform(10, 20, 10),
            "volume": rng.uniform(100, 1000, 10),
        })
        rows_a.to_csv(token_dir / "AAAUSDT-4h-2022-01.csv", index=False)

        # File 2: rows 8-14 (overlap on 8,9)
        rows_b = pd.DataFrame({
            "open_time": [base_ts + i * step for i in range(8, 15)],
            "open": rng.uniform(10, 20, 7),
            "high": rng.uniform(20, 30, 7),
            "low": rng.uniform(5, 10, 7),
            "close": rng.uniform(10, 20, 7),
            "volume": rng.uniform(100, 1000, 7),
        })
        rows_b.to_csv(token_dir / "AAAUSDT-4h-2022-02.csv", index=False)

        loader = RLDataLoader(raw_dir=tmp_path)
        df = loader._load_token_ohlcv("AAAUSDT")

        assert df is not None
        # 10 + 7 - 2 duplicates = 15 unique rows
        assert len(df) == 15
        # Sorted by open_time
        assert df["open_time"].is_monotonic_increasing
        # No duplicates
        assert df["open_time"].nunique() == 15

    def test_returns_none_for_missing(self, tmp_path):
        loader = RLDataLoader(raw_dir=tmp_path)
        assert loader._load_token_ohlcv("NONEXISTENT") is None


class TestGetAltSymbols:
    def test_excludes_btc_eth(self, tmp_path):
        """BTC and ETH should be excluded from alt symbols."""
        _make_manifest(tmp_path, [
            ("BTCUSDT", 50),
            ("ETHUSDT", 50),
            ("DOGEUSDT", 40),
            ("SOLUSDT", 35),
            ("ADAUSDT", 30),
        ])

        loader = RLDataLoader(raw_dir=tmp_path, max_tokens=10)
        alts = loader._get_alt_symbols()

        assert "BTCUSDT" not in alts
        assert "ETHUSDT" not in alts
        assert "DOGEUSDT" in alts
        assert "SOLUSDT" in alts
        assert "ADAUSDT" in alts

    def test_respects_max_tokens(self, tmp_path):
        _make_manifest(tmp_path, [
            ("DOGEUSDT", 40),
            ("SOLUSDT", 35),
            ("ADAUSDT", 30),
            ("XRPUSDT", 25),
        ])

        loader = RLDataLoader(raw_dir=tmp_path, max_tokens=2)
        alts = loader._get_alt_symbols()
        assert len(alts) == 2

    def test_sorted_by_months_desc(self, tmp_path):
        _make_manifest(tmp_path, [
            ("ADAUSDT", 10),
            ("DOGEUSDT", 40),
            ("SOLUSDT", 25),
        ])

        loader = RLDataLoader(raw_dir=tmp_path, max_tokens=10)
        alts = loader._get_alt_symbols()
        assert alts == ["DOGEUSDT", "SOLUSDT", "ADAUSDT"]


class TestBuildDataset:
    def test_returns_correct_structure(self, tmp_path):
        """build_dataset returns dict with correct keys, shapes, dtypes."""
        n = 300
        _make_ohlcv_csv(tmp_path, "BTCUSDT", n_rows=n)
        _make_ohlcv_csv(tmp_path, "ETHUSDT", n_rows=n)
        _make_ohlcv_csv(tmp_path, "DOGEUSDT", n_rows=n)

        _make_manifest(tmp_path, [
            ("BTCUSDT", 50),
            ("ETHUSDT", 50),
            ("DOGEUSDT", 40),
        ])

        loader = RLDataLoader(raw_dir=tmp_path, max_tokens=10)
        ds = loader.build_dataset()

        # Required keys
        assert set(ds.keys()) == {
            "alt_features", "indicator_features", "prices",
            "alt_names", "timestamps",
        }

        # alt_features is 3D float32
        af = ds["alt_features"]
        assert af.ndim == 3
        assert af.dtype == np.float32
        n_ts, n_alts, n_feat = af.shape
        assert n_alts == 1  # only DOGE (BTC/ETH excluded)
        assert n_feat > 0

        # indicator_features is 2D float32
        ind = ds["indicator_features"]
        assert ind.ndim == 2
        assert ind.dtype == np.float32
        assert ind.shape[0] == n_ts

        # prices is 3D with 4 columns (OHLC)
        pr = ds["prices"]
        assert pr.ndim == 3
        assert pr.shape == (n_ts, n_alts, 4)
        assert pr.dtype == np.float32

        # alt_names excludes BTC/ETH
        assert "BTC" not in ds["alt_names"]
        assert "ETH" not in ds["alt_names"]
        assert "DOGE" in ds["alt_names"]

        # timestamps length matches
        assert len(ds["timestamps"]) == n_ts

        # No NaN in outputs
        assert not np.isnan(af).any()
        assert not np.isnan(ind).any()
        assert not np.isnan(pr).any()
