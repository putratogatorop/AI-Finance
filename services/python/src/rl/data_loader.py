"""Data loader: raw 4h CSVs to RL-ready numpy arrays."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.features_v2 import FeatureEngineerV2
from src.rl.config import RLConfig
from src.rl.features import build_cross_token_features, build_indicator_features

logger = logging.getLogger(__name__)


class RLDataLoader:
    """Load raw 4h crypto CSVs and produce structured numpy arrays for the gym."""

    def __init__(self, raw_dir: Path, max_tokens: int = 200):
        self.raw_dir = Path(raw_dir)
        self.max_tokens = max_tokens
        self.config = RLConfig()

    # ── Private helpers ──────────────────────────────────────────

    def _load_token_ohlcv(self, symbol: str) -> pd.DataFrame | None:
        """Load all CSV files for one token into a single DataFrame.

        Files live at raw_dir/{symbol}/*.csv with columns:
        open_time, open, high, low, close, volume (plus extras we ignore).
        Sort by open_time, deduplicate, return None if no files found.
        """
        token_dir = self.raw_dir / symbol
        if not token_dir.is_dir():
            return None

        csv_files = sorted(token_dir.glob("*.csv"))
        if not csv_files:
            return None

        frames: list[pd.DataFrame] = []
        for f in csv_files:
            try:
                chunk = pd.read_csv(f)
                frames.append(chunk)
            except Exception:
                logger.warning("Failed to read %s, skipping", f)

        if not frames:
            return None

        df = pd.concat(frames, ignore_index=True)
        df = df.sort_values("open_time").drop_duplicates(subset="open_time").reset_index(drop=True)

        # Keep only the columns we need
        keep = ["open_time", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        return df

    def _get_alt_symbols(self) -> list[str]:
        """Get tradeable alt symbols, excluding indicator assets (BTC, ETH).

        Reads manifest.csv if it exists (columns: symbol, months), sorts by
        months descending, takes top max_tokens. Filters out BTCUSDT/ETHUSDT.
        """
        indicator_symbols = {
            f"{a}USDT" for a in self.config.INDICATOR_ASSETS
        }

        manifest_path = self.raw_dir / "manifest.csv"
        if manifest_path.is_file():
            manifest = pd.read_csv(manifest_path)
            manifest = manifest.sort_values("months", ascending=False)
            symbols = [
                s for s in manifest["symbol"].tolist()
                if s not in indicator_symbols
            ]
            return symbols[: self.max_tokens]

        # Fallback: list directories
        dirs = sorted(
            d.name for d in self.raw_dir.iterdir()
            if d.is_dir() and d.name not in indicator_symbols
        )
        return dirs[: self.max_tokens]

    # ── Public API ───────────────────────────────────────────────

    def build_dataset(self) -> dict:
        """Build complete RL dataset from raw CSVs.

        Returns
        -------
        dict with keys:
            alt_features   : (n_timestamps, n_alts, n_alt_features) float32
            indicator_features : (n_timestamps, n_indicator_features) float32
            prices         : (n_timestamps, n_alts, 4) float32  — OHLC
            alt_names      : list[str]  — asset names without USDT suffix
            timestamps     : np.ndarray
        """
        # 1. Load BTC and ETH
        btc_df = self._load_token_ohlcv("BTCUSDT")
        eth_df = self._load_token_ohlcv("ETHUSDT")
        if btc_df is None or eth_df is None:
            raise FileNotFoundError("BTCUSDT and ETHUSDT data are required")

        # 2. Common timestamps between BTC and ETH
        common_ts = np.intersect1d(btc_df["open_time"].values, eth_df["open_time"].values)
        btc_df = btc_df[btc_df["open_time"].isin(common_ts)].reset_index(drop=True)
        eth_df = eth_df[eth_df["open_time"].isin(common_ts)].reset_index(drop=True)
        logger.info("BTC/ETH common timestamps: %d", len(common_ts))

        # 3. Build indicator features from BTC/ETH
        indicator_df = build_indicator_features(btc_df, eth_df)

        # 4. Process alt tokens
        alt_symbols = self._get_alt_symbols()
        logger.info("Processing %d alt candidates", len(alt_symbols))

        fe = FeatureEngineerV2()
        alt_feature_arrays: list[np.ndarray] = []
        alt_price_arrays: list[np.ndarray] = []
        alt_names: list[str] = []
        min_len = len(common_ts)

        for symbol in alt_symbols:
            alt_df = self._load_token_ohlcv(symbol)
            if alt_df is None:
                continue

            # Align to common timestamps
            alt_df = alt_df[alt_df["open_time"].isin(common_ts)].reset_index(drop=True)
            if len(alt_df) < 200:
                logger.debug("Skipping %s — only %d rows after alignment", symbol, len(alt_df))
                continue

            # Feature engineering
            asset_name = symbol.replace("USDT", "")
            feat_df = fe.build_features(alt_df, asset=asset_name)
            feature_cols = fe.get_feature_cols()

            # Cross-token features (BTC-alt)
            # We need to align btc_close to the same rows that survived build_features
            n_feat = len(feat_df)
            btc_close_aligned = btc_df["close"].iloc[-n_feat:].reset_index(drop=True)
            cross_df = build_cross_token_features(feat_df["_close"], btc_close_aligned)

            # Combine feature columns + cross-token columns
            all_feat_cols = feature_cols + list(cross_df.columns)
            combined = pd.concat([feat_df[feature_cols], cross_df], axis=1)

            alt_feature_arrays.append(combined.values)
            alt_price_arrays.append(
                feat_df[["_close"]].assign(
                    _open=alt_df["open"].iloc[-n_feat:].values,
                    _high=alt_df["high"].iloc[-n_feat:].values,
                    _low=alt_df["low"].iloc[-n_feat:].values,
                )[["_open", "_high", "_low", "_close"]].values
            )
            alt_names.append(asset_name)
            min_len = min(min_len, n_feat)
            logger.debug("Loaded %s: %d rows, %d features", symbol, n_feat, len(all_feat_cols))

        if not alt_names:
            raise ValueError("No alt tokens loaded — check data directory")

        # 5. Align all to shortest common length
        alt_features = np.stack(
            [arr[-min_len:] for arr in alt_feature_arrays], axis=1
        ).astype(np.float32)

        prices = np.stack(
            [arr[-min_len:] for arr in alt_price_arrays], axis=1
        ).astype(np.float32)

        indicator_features = indicator_df.iloc[-min_len:].values.astype(np.float32)

        timestamps = common_ts[-min_len:]

        # 6. Fill NaN with 0
        alt_features = np.nan_to_num(alt_features, nan=0.0)
        indicator_features = np.nan_to_num(indicator_features, nan=0.0)
        prices = np.nan_to_num(prices, nan=0.0)

        logger.info(
            "Dataset built: %d timestamps, %d alts, alt_features=%s, indicator=%s",
            min_len, len(alt_names), alt_features.shape, indicator_features.shape,
        )

        return {
            "alt_features": alt_features,
            "indicator_features": indicator_features,
            "prices": prices,
            "alt_names": alt_names,
            "timestamps": timestamps,
        }
