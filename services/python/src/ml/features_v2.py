"""Feature engineering v2 — stationary features only.

Expert-informed redesign:
- NO raw price levels (SMA, EMA, BB bands) — non-stationary
- NO fake market_context (np.random) — actively harmful
- ALL features are ratios, returns, oscillators, or normalized values
- Designed for 4h candles with multi-timeframe support
- Works across 200+ tokens with a universal model
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineerV2:
    """Stationary feature engineering for crypto trading."""

    # ── Feature columns output by build_features() ──
    FEATURE_COLS: list[str] = []  # Set dynamically after first call

    def build_features(self, df: pd.DataFrame, asset: str = "BTC") -> pd.DataFrame:
        """Build all features from OHLCV data. Input must have: open, high, low, close, volume.

        Returns DataFrame with only feature columns (no raw prices), NaN rows dropped.
        """
        out = df.copy()
        close = out["close"].astype(float)
        high = out["high"].astype(float)
        low = out["low"].astype(float)
        open_ = out["open"].astype(float)
        volume = out["volume"].astype(float)

        # ═══════════════════════════════════════════
        # 1. MOMENTUM (log returns, stationary)
        # ═══════════════════════════════════════════
        for period in [3, 6, 12, 24, 42, 84, 168]:
            # 4h candles: 6=1d, 42=7d, 84=14d, 168=28d
            out[f"ret_{period}"] = np.log(close / close.shift(period))

        # Rate of change (same as returns but for cross-reference)
        out["roc_6"] = close.pct_change(6)    # 1-day
        out["roc_42"] = close.pct_change(42)  # 7-day

        # ═══════════════════════════════════════════
        # 2. PRICE POSITION (relative to moving averages)
        # ═══════════════════════════════════════════
        for period in [20, 50, 100, 200]:
            sma = close.rolling(period).mean()
            out[f"price_to_sma_{period}"] = close / sma - 1

        # EMA ratios (fast/slow)
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        ema_50 = close.ewm(span=50, adjust=False).mean()
        out["ema_12_26_ratio"] = ema_12 / ema_26 - 1
        out["ema_12_50_ratio"] = ema_12 / ema_50 - 1

        # ═══════════════════════════════════════════
        # 3. RSI
        # ═══════════════════════════════════════════
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        out["rsi_14"] = 100 - (100 / (1 + rs))
        # Normalize to [-1, 1] for model
        out["rsi_norm"] = (out["rsi_14"] - 50) / 50

        # ═══════════════════════════════════════════
        # 4. MACD (normalized by price)
        # ═══════════════════════════════════════════
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        out["macd_hist_norm"] = (macd - macd_signal) / close

        # ═══════════════════════════════════════════
        # 5. BOLLINGER BANDS (only stationary derivatives)
        # ═══════════════════════════════════════════
        bb_mid = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        out["bb_width"] = (bb_upper - bb_lower) / bb_mid
        out["bb_pct"] = (close - bb_lower) / (bb_upper - bb_lower)

        # ═══════════════════════════════════════════
        # 6. VOLATILITY
        # ═══════════════════════════════════════════
        daily_ret = close.pct_change()

        # Realized volatility at multiple scales
        out["vol_6"] = daily_ret.rolling(6).std()    # ~1 day
        out["vol_24"] = daily_ret.rolling(24).std()   # ~4 days
        out["vol_72"] = daily_ret.rolling(72).std()   # ~12 days
        out["vol_168"] = daily_ret.rolling(168).std()  # ~28 days

        # Volatility ratio (compression/expansion)
        out["vol_ratio_short_long"] = out["vol_6"] / out["vol_72"].replace(0, np.nan)

        # Parkinson volatility (uses high/low, more efficient)
        hl_log = np.log(high / low)
        out["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * (hl_log ** 2).rolling(24).mean()
        )

        # ATR normalized by price
        prev_close = close.shift(1)
        tr = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean()
        out["atr_norm"] = atr_14 / close

        # ═══════════════════════════════════════════
        # 7. VOLUME FEATURES
        # ═══════════════════════════════════════════
        vol_sma_20 = volume.rolling(20).mean()
        out["volume_ratio"] = volume / vol_sma_20.replace(0, np.nan)
        out["volume_change_6"] = np.log(
            volume.rolling(6).mean() / vol_sma_20.replace(0, np.nan)
        )

        # OBV slope (normalized)
        obv = (np.sign(close.diff()) * volume).cumsum()
        obv_slope = obv.rolling(24).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 24 else 0,
            raw=True,
        )
        out["obv_slope_norm"] = obv_slope / volume.rolling(24).mean().replace(0, np.nan)

        # ═══════════════════════════════════════════
        # 8. MICROSTRUCTURE (from OHLCV)
        # ═══════════════════════════════════════════
        bar_range = high - low
        _body = (close - open_).abs()

        # Close location value: where close sits in the bar
        out["clv"] = (2 * close - high - low) / bar_range.replace(0, np.nan)

        # Upper/lower shadow ratio
        upper_shadow = high - pd.concat([close, open_], axis=1).max(axis=1)
        lower_shadow = pd.concat([close, open_], axis=1).min(axis=1) - low
        out["shadow_ratio"] = upper_shadow / lower_shadow.replace(0, np.nan)
        out["shadow_ratio"] = out["shadow_ratio"].clip(-5, 5)  # Cap outliers

        # Bar range / ATR (normalized bar size)
        out["bar_range_norm"] = bar_range / atr_14.replace(0, np.nan)

        # ═══════════════════════════════════════════
        # 9. DISTRIBUTION FEATURES
        # ═══════════════════════════════════════════
        out["skew_24"] = daily_ret.rolling(24).skew()
        out["kurt_24"] = daily_ret.rolling(24).kurt()

        # Consecutive direction (streak)
        direction = np.sign(close.diff())
        streak = direction.copy()
        for i in range(1, len(streak)):
            if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
                streak.iloc[i] = streak.iloc[i - 1] + direction.iloc[i]
            else:
                streak.iloc[i] = direction.iloc[i]
        out["streak"] = streak

        # Drawdown from rolling high
        rolling_high = close.rolling(168).max()  # 28-day high
        out["drawdown"] = close / rolling_high - 1

        # Return autocorrelation (momentum vs mean-reversion regime)
        out["ret_autocorr_6"] = daily_ret.rolling(42).apply(
            lambda x: x.autocorr(lag=6) if len(x) >= 7 else 0, raw=False
        )

        # ═══════════════════════════════════════════
        # 10. REGIME INDICATORS (rule-based, binary)
        # ═══════════════════════════════════════════
        sma_50 = close.rolling(50).mean()
        sma_200 = close.rolling(200).mean()
        out["above_sma50"] = (close > sma_50).astype(float)
        out["above_sma200"] = (close > sma_200).astype(float)
        out["golden_cross"] = (sma_50 > sma_200).astype(float)

        # ═══════════════════════════════════════════
        # 11. TIME FEATURES
        # ═══════════════════════════════════════════
        if "timestamp" in out.columns:
            ts = pd.to_datetime(out["timestamp"])
        elif "date" in out.columns:
            ts = pd.to_datetime(out["date"])
        elif "open_time" in out.columns:
            ts = pd.to_datetime(out["open_time"], unit="ms")
        else:
            ts = out.index.to_series()

        out["hour_sin"] = np.sin(2 * np.pi * ts.dt.hour / 24)
        out["hour_cos"] = np.cos(2 * np.pi * ts.dt.hour / 24)
        out["dow_sin"] = np.sin(2 * np.pi * ts.dt.dayofweek / 7)
        out["dow_cos"] = np.cos(2 * np.pi * ts.dt.dayofweek / 7)

        # ═══════════════════════════════════════════
        # 12. RAW VALUES (for regime filter / stops, NOT features)
        # ═══════════════════════════════════════════
        out["_close"] = close
        out["_atr_14"] = atr_14
        out["_sma_50"] = sma_50
        out["_roc_20_candles"] = close.pct_change(20)

        # ═══════════════════════════════════════════
        # Collect feature columns and drop NaN
        # ═══════════════════════════════════════════
        meta_cols = {"open", "high", "low", "close", "volume", "date",
                     "timestamp", "asset", "open_time", "close_time",
                     "quote_volume", "trades", "taker_buy_base",
                     "taker_buy_quote", "ignore", "id"}
        raw_cols = {c for c in out.columns if c.startswith("_")}
        feature_cols = [c for c in out.columns if c not in meta_cols and c not in raw_cols]

        # Drop NaN rows (warmup period)
        valid = out[feature_cols].notna().all(axis=1)
        result = out[valid].reset_index(drop=True)

        self.__class__.FEATURE_COLS = feature_cols
        return result

    def build_target(
        self,
        df: pd.DataFrame,
        horizon: int = 30,  # 30 × 4h = 5 days
        threshold: float = 0.02,  # 2% move
    ) -> pd.Series:
        """Classification target: did price go up > threshold in next N candles?

        Returns Series of 0/1. NaN for last `horizon` rows.
        """
        close = df["_close"] if "_close" in df.columns else df["close"]
        forward_ret = close.shift(-horizon) / close - 1
        target = (forward_ret > threshold).astype(float)
        target[forward_ret.isna()] = np.nan
        return target

    def get_feature_cols(self) -> list[str]:
        """Return the list of feature column names (after build_features called)."""
        return self.FEATURE_COLS
