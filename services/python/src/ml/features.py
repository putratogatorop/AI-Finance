from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:

    def compute_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        close = out["close"]
        high = out["high"]
        low = out["low"]
        volume = out["volume"]

        for period in (5, 10, 20, 50):
            out[f"sma_{period}"] = close.rolling(window=period).mean()
        for period in (5, 10, 20, 50):
            out[f"ema_{period}"] = close.ewm(
                span=period, adjust=False
            ).mean()

        out["rsi_14"] = self._compute_rsi(close, period=14)

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        out["macd"] = ema_12 - ema_26
        out["macd_signal"] = out["macd"].ewm(
            span=9, adjust=False
        ).mean()
        out["macd_histogram"] = out["macd"] - out["macd_signal"]

        bb_middle = close.rolling(window=20).mean()
        bb_std = close.rolling(window=20).std()
        out["bb_upper"] = bb_middle + 2 * bb_std
        out["bb_middle"] = bb_middle
        out["bb_lower"] = bb_middle - 2 * bb_std
        out["bb_width"] = (
            (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]
        )

        out["atr_14"] = self._compute_atr(high, low, close, period=14)

        for days, label in [(7, "7d"), (14, "14d"), (28, "28d")]:
            out[f"momentum_{label}"] = close.pct_change(periods=days)

        out["volume_sma_20"] = volume.rolling(window=20).mean()
        out["volume_ratio"] = volume / out["volume_sma_20"]
        daily_returns = close.pct_change()
        out["volatility_20d"] = daily_returns.rolling(window=20).std()
        out["price_to_sma_20"] = close / out["sma_20"] - 1
        out["price_to_sma_50"] = close / out["sma_50"] - 1
        out["daily_return"] = daily_returns

        return out

    def _compute_rsi(
        self, close: pd.Series, period: int = 14
    ) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = (-delta).where(delta < 0, 0.0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _compute_atr(
        self, high: pd.Series, low: pd.Series,
        close: pd.Series, period: int = 14
    ) -> pd.Series:
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def compute_fundamental_features(
        self, fundamentals: pd.DataFrame, asset: str
    ) -> dict[str, float]:
        row = fundamentals[fundamentals["asset"] == asset]
        if row.empty:
            return {
                "market_cap_rank": 0.0,
                "volume_rank": 0.0,
                "market_cap_log": 0.0,
                "volume_24h_log": 0.0,
            }

        row = row.iloc[0]
        fundamentals_sorted = fundamentals.sort_values(
            "total_volume_24h", ascending=False
        ).reset_index(drop=True)
        volume_rank_idx = fundamentals_sorted[
            fundamentals_sorted["asset"] == asset
        ].index
        volume_rank = (
            int(volume_rank_idx[0]) + 1
            if len(volume_rank_idx) > 0 else 0
        )

        market_cap_val = row.get("market_cap", 0) or 0
        volume_val = row.get("total_volume_24h", 0) or 0

        return {
            "market_cap_rank": float(
                row.get("market_cap_rank", 0) or 0
            ),
            "volume_rank": float(volume_rank),
            "market_cap_log": float(np.log1p(market_cap_val)),
            "volume_24h_log": float(np.log1p(volume_val)),
        }

    def compute_market_context_features(
        self, market_context: pd.DataFrame
    ) -> pd.DataFrame:
        out = market_context.copy()
        out["total_market_cap_change_7d"] = out[
            "total_market_cap"
        ].pct_change(periods=7)
        out["btc_dominance_change_7d"] = out[
            "btc_dominance"
        ].diff(periods=7)
        out["fear_greed_sma_7"] = out[
            "fear_greed_index"
        ].rolling(window=7).mean()
        return out

    def build_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df[["date", "close"]].copy()
        close = out["close"]
        for days, label in [
            (7, "7d"), (14, "14d"), (28, "28d"), (42, "42d")
        ]:
            out[f"target_{label}"] = close.shift(-days) / close - 1
        return out

    def build_feature_matrix(
        self,
        daily_prices: pd.DataFrame,
        fundamentals: pd.DataFrame,
        market_context: pd.DataFrame,
        asset: str,
    ) -> pd.DataFrame:
        tech = self.compute_technical_features(daily_prices)
        ctx = self.compute_market_context_features(market_context)

        ctx_cols = [
            "date", "btc_dominance", "fear_greed_index",
            "total_market_cap_change_7d",
            "btc_dominance_change_7d", "fear_greed_sma_7",
        ]
        merged = tech.merge(ctx[ctx_cols], on="date", how="left")

        fund_feats = self.compute_fundamental_features(
            fundamentals, asset
        )
        for k, v in fund_feats.items():
            merged[k] = v

        drop_cols = [
            "date", "asset", "open", "high",
            "low", "close", "volume",
        ]
        feature_cols = [
            c for c in merged.columns if c not in drop_cols
        ]
        result = merged[feature_cols].copy()
        result = result.dropna().reset_index(drop=True)
        return result

    def feature_names(self) -> list[str]:
        return [
            "sma_5", "sma_10", "sma_20", "sma_50",
            "ema_5", "ema_10", "ema_20", "ema_50",
            "rsi_14",
            "macd", "macd_signal", "macd_histogram",
            "bb_upper", "bb_middle", "bb_lower", "bb_width",
            "atr_14",
            "momentum_7d", "momentum_14d", "momentum_28d",
            "volume_sma_20", "volume_ratio",
            "volatility_20d",
            "price_to_sma_20", "price_to_sma_50",
            "daily_return",
            "btc_dominance", "fear_greed_index",
            "total_market_cap_change_7d",
            "btc_dominance_change_7d",
            "fear_greed_sma_7",
            "market_cap_rank", "volume_rank",
            "market_cap_log", "volume_24h_log",
        ]
