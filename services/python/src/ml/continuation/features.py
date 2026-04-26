"""Feature library for the continuation predictor (short side, v1).

CONTRACT — strict no-lookahead:
  Every feature for trade `t` reads ONLY bars whose timestamp `<=` `entry_time(t)`.
  The entry bar itself is "available" because the signal fires at its close.
  Future bars are off-limits even for normalisation. tests/test_continuation_features.py
  enforces this via a recorded-access fixture.

The 12 features (see plan):
  Asset-local (5)
   - asset_4h_ret           pct change of asset close over last 16 bars
   - asset_24h_ret          pct change of asset close over last 96 bars
   - prior_bar_wick_ratio   (high-close)/(high-low) of the bar BEFORE entry
   - atr_norm_drop          (rolling_high32 - close) / ATR14   at the entry bar
   - bars_since_high        bars since the 32-bar rolling-high (incl. current)
  BTC (3)
   - btc_4h_ret             BTC pct change over last 16 bars at entry
   - btc_above_ema50_4h     bool — BTC 4h close > 4h EMA-50 (resampled)
   - btc_realized_vol_24h   stdev of BTC hourly returns over the last 24h
  Cross-sectional (2)
   - vol_decile             cross-sectional decile of asset 24h-volume vs. universe
   - concurrent_down_breadth fraction of universe with 4h-ret < -2% at entry
  OHLCV order-flow proxies (2)
   - body_to_range_ratio    mean(|close-open|/(high-low)) over last 8 bars (incl entry)
   - vol_zscore_8           z-score of entry-bar volume vs. prior 8 bars (excl entry)

Usage:

    ctx = FeatureContext(candles)            # builds all panels once
    rows = [ctx.build_for(trade_row) for trade_row in trades.to_dict("records")]
    X = pd.DataFrame(rows)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# ---- knobs (kept here so they show up in feature_meta.json) ------------------

VOL_DECILE_LOOKBACK = 96     # 96 15m bars of volume sum approximates "24h volume"
BREADTH_LOOKBACK_4H = 16     # bars (= 4h)
BREADTH_4H_DOWN_THRESH = -0.02
BTC_VOL_LOOKBACK_24H = 96    # 24h on 15m
BTC_4H_LOOKBACK = 16
ASSET_4H_LOOKBACK = 16
ASSET_24H_LOOKBACK = 96
ATR_LOOKBACK = 14
ROLLING_HIGH_LOOKBACK = 32
BODY_RATIO_LOOKBACK = 8
VOL_ZSCORE_LOOKBACK = 8
BTC_MACRO_TIMEFRAME = "4h"
BTC_MACRO_EMA_SPAN = 50

FEATURE_NAMES = [
    "asset_4h_ret",
    "asset_24h_ret",
    "prior_bar_wick_ratio",
    "atr_norm_drop",
    "bars_since_high",
    "btc_4h_ret",
    "btc_above_ema50_4h",
    "btc_realized_vol_24h",
    "vol_decile",
    "concurrent_down_breadth",
    "body_to_range_ratio",
    "vol_zscore_8",
]


@dataclass
class _AssetView:
    """Per-asset numpy arrays for fast feature lookup."""
    ts: np.ndarray            # datetime64[ns, UTC]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    # Pre-computed rolling features (length matches above):
    atr14: np.ndarray
    rolling_high32: np.ndarray
    rolling_argmax32: np.ndarray  # for bars_since_high
    body_to_range_8: np.ndarray   # mean over last 8 bars including current
    vol_mean_prior8: np.ndarray   # rolling mean of volume over PRIOR 8 bars
    vol_std_prior8: np.ndarray


def _atr_array(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int) -> np.ndarray:
    """ATR with simple moving average over `n` true ranges. NaN before warmup."""
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return pd.Series(tr).rolling(n, min_periods=n).mean().to_numpy()


def _argmax_in_window(arr: np.ndarray, window: int) -> np.ndarray:
    """For each i, position (within last `window` bars including i) of the max value.

    Returned offset is `i - argmax_global`, so 0 means current bar is the max.
    NaN before warmup.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = arr[i - window + 1 : i + 1]
        out[i] = float((window - 1) - int(np.argmax(seg)))  # bars-since
    return out


def _build_asset_view(sub: pd.DataFrame) -> _AssetView:
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    # Strip tz info to land on plain datetime64[ns] so np.searchsorted works
    # with tz-naive comparisons. UTC is the convention everywhere — this is a
    # representational change only.
    ts = (
        pd.to_datetime(sub["timestamp"], utc=True)
        .dt.tz_convert(None)
        .to_numpy()
    )
    o = sub["open"].to_numpy(dtype=float)
    h = sub["high"].to_numpy(dtype=float)
    lo = sub["low"].to_numpy(dtype=float)
    c = sub["close"].to_numpy(dtype=float)
    v = sub["volume"].to_numpy(dtype=float)

    atr14 = _atr_array(h, lo, c, ATR_LOOKBACK)

    rolling_high32 = (
        pd.Series(h)
        .rolling(ROLLING_HIGH_LOOKBACK, min_periods=ROLLING_HIGH_LOOKBACK)
        .max()
        .to_numpy()
    )
    rolling_argmax32 = _argmax_in_window(h, ROLLING_HIGH_LOOKBACK)

    range_ = h - lo
    range_safe = np.where(range_ > 0, range_, np.nan)
    body_ratio_per_bar = np.abs(c - o) / range_safe  # NaN where range=0
    body_to_range_8 = (
        pd.Series(body_ratio_per_bar)
        .rolling(BODY_RATIO_LOOKBACK, min_periods=1)
        .mean()
        .to_numpy()
    )

    # Volume z-score uses ONLY prior bars (not current), so shift by 1.
    vol_mean_prior8 = (
        pd.Series(v).rolling(VOL_ZSCORE_LOOKBACK, min_periods=VOL_ZSCORE_LOOKBACK)
        .mean().shift(1).to_numpy()
    )
    vol_std_prior8 = (
        pd.Series(v).rolling(VOL_ZSCORE_LOOKBACK, min_periods=VOL_ZSCORE_LOOKBACK)
        .std(ddof=0).shift(1).to_numpy()
    )

    return _AssetView(
        ts=ts, open=o, high=h, low=lo, close=c, volume=v,
        atr14=atr14,
        rolling_high32=rolling_high32,
        rolling_argmax32=rolling_argmax32,
        body_to_range_8=body_to_range_8,
        vol_mean_prior8=vol_mean_prior8,
        vol_std_prior8=vol_std_prior8,
    )


@dataclass
class FeatureContext:
    """All snapshot-wide panels needed to build features for any trade.

    Construction is the heavy step (BTC resample, vol-decile pivot, breadth
    panel). Per-trade lookup is then O(1).
    """
    candles: pd.DataFrame
    btc_asset_key: str = "BTCUSDT"
    universe_listed_since: dict[str, pd.Timestamp] = field(default_factory=dict)

    _btc_view: _AssetView | None = field(init=False, default=None)
    _btc_above_ema50_4h_15m: pd.Series | None = field(init=False, default=None)
    _btc_realized_vol_24h: pd.Series | None = field(init=False, default=None)
    _btc_4h_ret_15m: pd.Series | None = field(init=False, default=None)
    _vol_decile_panel: pd.DataFrame | None = field(init=False, default=None)
    _breadth_4h_panel: pd.Series | None = field(init=False, default=None)
    _per_asset: dict[str, _AssetView] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # Pre-build BTC view (used by 3 features) and the asset cache will fill lazily.
        btc_sub = self.candles[self.candles["asset"] == self.btc_asset_key]
        if len(btc_sub) == 0:
            raise ValueError(
                f"No candles for BTC key {self.btc_asset_key!r}; "
                f"set FeatureContext(candles, btc_asset_key=...) to the right symbol."
            )
        self._btc_view = _build_asset_view(btc_sub)

        # BTC 4h panel: resample to 4h, EMA-50, then forward-fill back to 15m grid.
        btc_15m_close = (
            btc_sub.sort_values("timestamp").set_index("timestamp")["close"].astype(float)
        )
        btc_4h = btc_15m_close.resample(BTC_MACRO_TIMEFRAME, label="right", closed="right").last()
        ema = btc_4h.ewm(span=BTC_MACRO_EMA_SPAN, adjust=False).mean()
        # Shift by 1 so the 4h EMA at any 15m timestamp uses only the most recent
        # CLOSED 4h bar. This is critical for no-lookahead at the boundary.
        above_4h = (btc_4h > ema).shift(1)
        self._btc_above_ema50_4h_15m = (
            above_4h.reindex(btc_15m_close.index, method="ffill").fillna(False)
        )

        # BTC 4h pct change at every 15m bar: ((btc[t]/btc[t-16]) - 1).
        self._btc_4h_ret_15m = btc_15m_close.pct_change(BTC_4H_LOOKBACK)

        # BTC realized 24h vol: stdev of 1h returns over last 24 bars (1h granularity).
        btc_1h_close = btc_15m_close.resample("1h", label="right", closed="right").last()
        btc_1h_ret = btc_1h_close.pct_change()
        rv = btc_1h_ret.rolling(24, min_periods=24).std(ddof=0)
        # Reindex back to 15m grid.
        self._btc_realized_vol_24h = rv.reindex(btc_15m_close.index, method="ffill")

        # Volume-decile panel: per-asset 24h-rolling-volume (sum over last 96 15m bars),
        # then cross-sectional rank per timestamp. Decile 0..9.
        vol_panel = (
            self.candles.sort_values(["asset", "timestamp"])
            .assign(
                vol24h=lambda df: df.groupby("asset")["volume"].transform(
                    lambda s: s.rolling(VOL_DECILE_LOOKBACK, min_periods=VOL_DECILE_LOOKBACK).sum()
                )
            )
            .pivot(index="timestamp", columns="asset", values="vol24h")
        )
        # Survivorship: drop assets that haven't been listed yet at each timestamp.
        # If `universe_listed_since` is provided, mask future-dated entries.
        if self.universe_listed_since:
            mask = pd.DataFrame(False, index=vol_panel.index, columns=vol_panel.columns)
            for sym in vol_panel.columns:
                listed = self.universe_listed_since.get(sym)
                if listed is not None:
                    mask[sym] = vol_panel.index >= listed
                else:
                    # Not in universe parquet: keep all (older listings predate the parquet).
                    mask[sym] = True
            vol_panel = vol_panel.where(mask)
        pct_rank = vol_panel.rank(axis=1, pct=True)
        self._vol_decile_panel = (
            (pct_rank * 10.0 - 1e-9).clip(lower=0).apply(np.floor).astype("Int64")
        )

        # Concurrent 4h-down breadth: fraction of universe with 4h-ret < -2% at each timestamp.
        ret_4h_panel = (
            self.candles.sort_values(["asset", "timestamp"])
            .assign(
                ret4h=lambda df: df.groupby("asset")["close"].transform(
                    lambda s: s.pct_change(BREADTH_LOOKBACK_4H)
                )
            )
            .pivot(index="timestamp", columns="asset", values="ret4h")
        )
        if self.universe_listed_since:
            ret_4h_panel = ret_4h_panel.where(mask)
        # Avoid global mean across NaN (those assets haven't been listed yet).
        is_down = (ret_4h_panel < BREADTH_4H_DOWN_THRESH)
        n_listed = ret_4h_panel.notna().sum(axis=1)
        n_down = is_down.sum(axis=1)
        self._breadth_4h_panel = (
            (n_down / n_listed.where(n_listed > 0)).rename("concurrent_down_breadth")
        )

    # --- per-asset cache -----------------------------------------------------

    def _asset_view(self, symbol: str) -> _AssetView | None:
        if symbol in self._per_asset:
            return self._per_asset[symbol]
        sub = self.candles[self.candles["asset"] == symbol]
        if len(sub) == 0:
            return None
        view = _build_asset_view(sub)
        self._per_asset[symbol] = view
        return view

    # --- public API ---------------------------------------------------------

    def build_for(self, trade_row: dict[str, Any]) -> dict[str, float]:
        """Compute features for one trade. Returns dict of FEATURE_NAMES → float.

        Trade row must have at least: 'symbol', 'entry_time'.
        Missing data (e.g. warmup not satisfied) yields NaN per-feature; the
        caller is expected to drop rows with any NaN before training.
        """
        sym = trade_row["symbol"]
        raw_ts = pd.Timestamp(trade_row["entry_time"])
        entry_ts = (
            raw_ts.tz_convert("UTC") if raw_ts.tzinfo
            else pd.Timestamp(trade_row["entry_time"], tz="UTC")
        )
        return self._build_for_typed(sym, entry_ts)

    def _build_for_typed(self, symbol: str, entry_ts: pd.Timestamp) -> dict[str, float]:
        out: dict[str, float] = {n: float("nan") for n in FEATURE_NAMES}

        view = self._asset_view(symbol)
        if view is None:
            return out

        # Locate entry bar: last bar with timestamp <= entry_ts.
        # Convert tz-aware entry_ts to tz-naive datetime64 to match view.ts.
        entry_naive = (
            entry_ts.tz_convert(None) if entry_ts.tzinfo is not None else entry_ts
        )
        ts_arr = view.ts
        i = int(np.searchsorted(ts_arr, np.datetime64(entry_naive), side="right")) - 1
        if i < 0:
            return out

        # asset_4h_ret, asset_24h_ret
        if i - ASSET_4H_LOOKBACK >= 0 and view.close[i - ASSET_4H_LOOKBACK] > 0:
            out["asset_4h_ret"] = float(view.close[i] / view.close[i - ASSET_4H_LOOKBACK] - 1.0)
        if i - ASSET_24H_LOOKBACK >= 0 and view.close[i - ASSET_24H_LOOKBACK] > 0:
            out["asset_24h_ret"] = float(view.close[i] / view.close[i - ASSET_24H_LOOKBACK] - 1.0)

        # prior_bar_wick_ratio at i-1
        if i >= 1:
            h, lo, c = view.high[i - 1], view.low[i - 1], view.close[i - 1]
            rng = h - lo
            if rng > 0:
                out["prior_bar_wick_ratio"] = float((h - c) / rng)

        # atr_norm_drop at i
        atr = view.atr14[i] if i < len(view.atr14) else np.nan
        rh32 = view.rolling_high32[i] if i < len(view.rolling_high32) else np.nan
        if np.isfinite(atr) and atr > 0 and np.isfinite(rh32):
            out["atr_norm_drop"] = float((rh32 - view.close[i]) / atr)

        # bars_since_high at i
        bsh = view.rolling_argmax32[i] if i < len(view.rolling_argmax32) else np.nan
        if np.isfinite(bsh):
            out["bars_since_high"] = float(bsh)

        # body_to_range_ratio at i
        br8 = view.body_to_range_8[i] if i < len(view.body_to_range_8) else np.nan
        if np.isfinite(br8):
            out["body_to_range_ratio"] = float(br8)

        # vol_zscore_8 at i (uses PRIOR 8 bars only — not current)
        vmu = view.vol_mean_prior8[i] if i < len(view.vol_mean_prior8) else np.nan
        vsd = view.vol_std_prior8[i] if i < len(view.vol_std_prior8) else np.nan
        if np.isfinite(vmu) and np.isfinite(vsd) and vsd > 0:
            out["vol_zscore_8"] = float((view.volume[i] - vmu) / vsd)

        # BTC features at entry_ts
        if self._btc_view is not None:
            btc_ts = self._btc_view.ts
            bi = int(np.searchsorted(btc_ts, np.datetime64(entry_naive), side="right")) - 1
            if bi >= BTC_4H_LOOKBACK and self._btc_view.close[bi - BTC_4H_LOOKBACK] > 0:
                out["btc_4h_ret"] = float(
                    self._btc_view.close[bi] / self._btc_view.close[bi - BTC_4H_LOOKBACK] - 1.0
                )

        # btc_above_ema50_4h, btc_realized_vol_24h via reindexed series
        if self._btc_above_ema50_4h_15m is not None:
            try:
                # asof lookup — last value at-or-before entry_ts.
                v = self._btc_above_ema50_4h_15m.asof(entry_ts)
                if pd.notna(v):
                    out["btc_above_ema50_4h"] = float(bool(v))
            except KeyError:
                pass
        if self._btc_realized_vol_24h is not None:
            v = self._btc_realized_vol_24h.asof(entry_ts)
            if pd.notna(v):
                out["btc_realized_vol_24h"] = float(v)

        # vol_decile (cross-sectional)
        if self._vol_decile_panel is not None:
            try:
                d = self._vol_decile_panel.at[entry_ts, symbol]
                if pd.notna(d):
                    out["vol_decile"] = float(int(d))
            except KeyError:
                pass

        # concurrent_down_breadth
        if self._breadth_4h_panel is not None:
            v = self._breadth_4h_panel.asof(entry_ts)
            if pd.notna(v):
                out["concurrent_down_breadth"] = float(v)

        return out
