"""Feature library for the bigmover combined-with-indicators predictor (v1).

This is the SYNTHESIZED feature set after the Phase-3 5-specialist team
(EMA / RSI / MACD / KDJ / BTC-trend) ranked candidate features by marginal
AUC and walk-forward stability. See services/python/src/ml/bigmover_combined/
SYNTHESIS.md for the per-specialist findings and the rejection rationale
for dropped features.

CONTRACT — strict no-lookahead:
  Every feature for trade `t` reads ONLY bars whose timestamp `<=` `entry_time(t)`.
  Verified by tests/test_bigmover_combined_features.py via future-permutation
  invariance.

The 16 predictive features (+ is_short metadata):

Bigmover detector internals (5)
  - vol_ratio                 volume / 20-bar volume MA at entry bar
  - price_drop_pct            (rolling96_high - close) / rolling96_high
  - price_rise_pct            (close - rolling96_low) / rolling96_low
  - accel_atr_norm            (close[i] - close[i-1]) / ATR14
  - bars_since_high32         bars since the 32-bar high

EMA (2; from EMA specialist's top picks)
  - price_vs_ema9_pct         (close - EMA9) / close
  - price_vs_ema50_pct        (close - EMA50) / close

RSI (2; from RSI specialist's top picks — both *delta* versions, level dropped)
  - rsi14_delta_1bar          RSI14[i] - RSI14[i-1]      (15m momentum)
  - rsi14_delta_4bar          RSI14[i] - RSI14[i-4]      (1h momentum)

MACD (2; from MACD specialist's top picks)
  - macd_signal_spread_norm   (macd - signal) / close   (= macd_hist / close)
  - macd_hist_momentum        macd_histogram[i] - macd_histogram[i-2] (jerk)

KDJ (2; from KDJ specialist's top picks — raw levels beat derived spreads)
  - kdj_j                     J line value
  - kdj_k                     K line value

BTC (2)
  - btc_trend_score           Candidate B: clip((macd-signal)_4h_lag1 / close_4h_lag1,
                              ±k=0.005255) / k. Range [-1, +1]. Sign-aware:
                              positive = bull, negative = bear, magnitude = strength.
                              Walk-forward winner over daily-SMA50 placeholder.
  - btc_realized_vol_24h      stdev of BTC 1h returns over last 24h

Cross-sectional (1)
  - concurrent_dir_breadth    fraction of universe whose 4h return matches
                              the trade's direction (favorable = down for short,
                              up for long)

Plus 1 metadata (NOT a feature; used to dispatch direction-aware combos):
  - is_short                  1 if direction == 'short' else 0
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from src.ml.indicators import atr as _atr
from src.ml.indicators import ema as _ema
from src.ml.indicators import kdj as _kdj
from src.ml.indicators import macd as _macd
from src.ml.indicators import rsi as _rsi

# --- knobs (also serialized into feature_meta.json) -------------------------

VOL_MA_PERIOD = 20
PRICE_LOOKBACK_96 = 96
PRICE_LOOKBACK_32 = 32
ATR_LOOKBACK = 14
EMA_FAST = 9
EMA_SLOW = 50
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH = 9, 3, 3
BREADTH_4H_LOOKBACK = 16
BTC_VOL_LOOKBACK_24H_HOURS = 24
# Candidate B BTC trend score parameters (locked by Phase-3 specialist).
# k is the 90th percentile of |MACD-signal spread / close| on the 4h BTC
# timeline of the 2026-04-01 snapshot.
BTC_TREND_K = 0.005255
BTC_TREND_TIMEFRAME = "4h"
BTC_TREND_MACD_FAST = 12
BTC_TREND_MACD_SLOW = 26
BTC_TREND_MACD_SIGNAL = 9

FEATURE_NAMES = [
    # bigmover internals
    "vol_ratio",
    "price_drop_pct",
    "price_rise_pct",
    "accel_atr_norm",
    "bars_since_high32",
    # EMA (synthesized: 9 + 50)
    "price_vs_ema9_pct",
    "price_vs_ema50_pct",
    # RSI (synthesized: deltas only)
    "rsi14_delta_1bar",
    "rsi14_delta_4bar",
    # MACD (synthesized: spread + jerk)
    "macd_signal_spread_norm",
    "macd_hist_momentum",
    # KDJ (synthesized: raw J + K)
    "kdj_j",
    "kdj_k",
    # BTC
    "btc_trend_score",
    "btc_realized_vol_24h",
    # cross-sectional
    "concurrent_dir_breadth",
]


# --- per-asset cache --------------------------------------------------------


@dataclass
class _AssetView:
    ts: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    vol_ma20: np.ndarray
    rolling_high96: np.ndarray
    rolling_low96: np.ndarray
    bars_since_high32: np.ndarray
    atr14: np.ndarray
    ema9: np.ndarray
    ema50: np.ndarray
    rsi14: np.ndarray
    macd: np.ndarray
    macd_sig: np.ndarray
    macd_hist: np.ndarray
    kdj_k: np.ndarray
    kdj_d: np.ndarray
    kdj_j: np.ndarray


def _argmax_in_window_to_offset(arr: np.ndarray, window: int) -> np.ndarray:
    """For each i, position (within last `window` bars including i) of the max value.
    Returns offset = (window-1) - argmax_local, so 0 means current bar is the max.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = arr[i - window + 1 : i + 1]
        out[i] = float((window - 1) - int(np.argmax(seg)))
    return out


def _build_asset_view(sub: pd.DataFrame, ts_col: str = "timestamp") -> _AssetView:
    sub = sub.sort_values(ts_col).reset_index(drop=True)
    ts = (
        pd.to_datetime(sub[ts_col], utc=True)
        .dt.tz_convert(None)
        .to_numpy()
    )
    o = sub["open"].to_numpy(dtype=float)
    h = sub["high"].to_numpy(dtype=float)
    lo = sub["low"].to_numpy(dtype=float)
    c = sub["close"].to_numpy(dtype=float)
    v = sub["volume"].to_numpy(dtype=float)
    vol_ma20 = pd.Series(v).rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().to_numpy()
    rolling_high96 = (
        pd.Series(h).rolling(PRICE_LOOKBACK_96, min_periods=PRICE_LOOKBACK_96).max().to_numpy()
    )
    rolling_low96 = (
        pd.Series(lo).rolling(PRICE_LOOKBACK_96, min_periods=PRICE_LOOKBACK_96).min().to_numpy()
    )
    bars_since_high32 = _argmax_in_window_to_offset(h, PRICE_LOOKBACK_32)
    atr14 = _atr(pd.Series(h), pd.Series(lo), pd.Series(c), ATR_LOOKBACK).to_numpy()
    ema9 = _ema(pd.Series(c), EMA_FAST).to_numpy()
    ema50 = _ema(pd.Series(c), EMA_SLOW).to_numpy()
    rsi14 = _rsi(pd.Series(c), RSI_PERIOD).to_numpy()
    md = _macd(pd.Series(c), MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_arr = md["macd"].to_numpy()
    macd_sig_arr = md["signal"].to_numpy()
    macd_hist_arr = md["histogram"].to_numpy()
    k = _kdj(
        pd.Series(h), pd.Series(lo), pd.Series(c),
        n=KDJ_N, k_smooth=KDJ_K_SMOOTH, d_smooth=KDJ_D_SMOOTH,
    )
    return _AssetView(
        ts=ts, open=o, high=h, low=lo, close=c, volume=v,
        vol_ma20=vol_ma20,
        rolling_high96=rolling_high96, rolling_low96=rolling_low96,
        bars_since_high32=bars_since_high32, atr14=atr14,
        ema9=ema9, ema50=ema50,
        rsi14=rsi14,
        macd=macd_arr, macd_sig=macd_sig_arr, macd_hist=macd_hist_arr,
        kdj_k=k["k"].to_numpy(), kdj_d=k["d"].to_numpy(), kdj_j=k["j"].to_numpy(),
    )


# --- snapshot-wide panels ---------------------------------------------------


@dataclass
class FeatureContext:
    """All snapshot-wide panels needed to build features for any trade.

    Heavy step at construction: per-asset rolling indicators (cached lazily),
    BTC 4h MACD trend score (Candidate B), cross-sectional 4h-return panel.
    Per-trade lookup is then O(1).
    """
    candles: pd.DataFrame
    btc_asset_key: str = "BTCUSDT"
    universe_listed_since: dict[str, pd.Timestamp] = field(default_factory=dict)

    _btc_view: _AssetView | None = field(init=False, default=None)
    _btc_trend_score_15m: pd.Series | None = field(init=False, default=None)
    _btc_realized_vol_24h_15m: pd.Series | None = field(init=False, default=None)
    _ret_4h_panel: pd.DataFrame | None = field(init=False, default=None)
    _per_asset: dict[str, _AssetView] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        # BTC view (per-asset rolling).
        btc_sub = self.candles[self.candles["asset"] == self.btc_asset_key]
        if len(btc_sub) == 0:
            raise ValueError(
                f"No candles for BTC key {self.btc_asset_key!r}; "
                "set FeatureContext(candles, btc_asset_key=...) to the right symbol."
            )
        self._btc_view = _build_asset_view(btc_sub)

        # BTC trend score (Candidate B):
        # Resample BTC 15m close to 4h, compute MACD(12,26,9), take
        # (macd-signal) / close, shift by 1 4h-bar (no lookahead at boundary),
        # clip to ±BTC_TREND_K, divide by k → range [-1, +1]. Forward-fill to
        # 15m grid.
        btc_15m_close = (
            btc_sub.sort_values("timestamp").set_index("timestamp")["close"].astype(float)
        )
        btc_4h = btc_15m_close.resample(BTC_TREND_TIMEFRAME, label="right", closed="right").last()
        m4h = _macd(btc_4h, BTC_TREND_MACD_FAST, BTC_TREND_MACD_SLOW, BTC_TREND_MACD_SIGNAL)
        spread_4h = (m4h["macd"] - m4h["signal"]).shift(1) / btc_4h.shift(1)
        score_4h = spread_4h.clip(-BTC_TREND_K, BTC_TREND_K) / BTC_TREND_K
        self._btc_trend_score_15m = score_4h.reindex(btc_15m_close.index, method="ffill")

        # BTC realized 24h vol from 1h returns.
        btc_1h_close = btc_15m_close.resample("1h", label="right", closed="right").last()
        btc_1h_ret = btc_1h_close.pct_change()
        rv = btc_1h_ret.rolling(
            window=BTC_VOL_LOOKBACK_24H_HOURS, min_periods=BTC_VOL_LOOKBACK_24H_HOURS
        ).std(ddof=0)
        self._btc_realized_vol_24h_15m = rv.reindex(btc_15m_close.index, method="ffill")

        # Cross-sectional 4h-return panel (timestamp x asset).
        ret_4h_panel = (
            self.candles.sort_values(["asset", "timestamp"])
            .assign(
                ret4h=lambda df: df.groupby("asset")["close"].transform(
                    lambda s: s.pct_change(BREADTH_4H_LOOKBACK)
                )
            )
            .pivot(index="timestamp", columns="asset", values="ret4h")
        )
        if self.universe_listed_since:
            mask = pd.DataFrame(False, index=ret_4h_panel.index, columns=ret_4h_panel.columns)
            for sym in ret_4h_panel.columns:
                listed = self.universe_listed_since.get(sym)
                mask[sym] = (ret_4h_panel.index >= listed) if listed is not None else True
            ret_4h_panel = ret_4h_panel.where(mask)
        self._ret_4h_panel = ret_4h_panel

    # --- per-asset cache lookup ---------------------------------------------

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
        """Build features for one trade. Returns dict of FEATURE_NAMES → float (+ is_short)."""
        sym = trade_row["symbol"]
        direction = str(trade_row.get("direction", "short")).lower()
        ts = trade_row["entry_time"]
        if not isinstance(ts, pd.Timestamp):
            ts = pd.Timestamp(ts)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return self._build_for_typed(sym, ts, direction)

    def _build_for_typed(
        self, symbol: str, entry_ts: pd.Timestamp, direction: str
    ) -> dict[str, float]:
        out: dict[str, float] = {n: float("nan") for n in FEATURE_NAMES}
        out["is_short"] = float(direction == "short")

        view = self._asset_view(symbol)
        if view is None:
            return out

        entry_naive = (
            entry_ts.tz_convert(None) if entry_ts.tzinfo is not None else entry_ts
        )
        i = int(np.searchsorted(view.ts, np.datetime64(entry_naive), side="right")) - 1
        if i < 0:
            return out

        c, _h, _l, _o, vol = view.close, view.high, view.low, view.open, view.volume

        # --- bigmover internals ----------------------------------------------
        if i < len(view.vol_ma20) and view.vol_ma20[i] > 0:
            out["vol_ratio"] = float(vol[i] / view.vol_ma20[i])
        if (
            i < len(view.rolling_high96)
            and np.isfinite(view.rolling_high96[i])
            and view.rolling_high96[i] > 0
        ):
            out["price_drop_pct"] = float((view.rolling_high96[i] - c[i]) / view.rolling_high96[i])
        if (
            i < len(view.rolling_low96)
            and np.isfinite(view.rolling_low96[i])
            and view.rolling_low96[i] > 0
        ):
            out["price_rise_pct"] = float((c[i] - view.rolling_low96[i]) / view.rolling_low96[i])
        if i >= 1 and i < len(view.atr14) and np.isfinite(view.atr14[i]) and view.atr14[i] > 0:
            out["accel_atr_norm"] = float((c[i] - c[i - 1]) / view.atr14[i])
        if i < len(view.bars_since_high32) and np.isfinite(view.bars_since_high32[i]):
            out["bars_since_high32"] = float(view.bars_since_high32[i])

        # --- EMA (9, 50) -----------------------------------------------------
        if i < len(view.ema9) and np.isfinite(view.ema9[i]) and c[i] > 0:
            out["price_vs_ema9_pct"] = float((c[i] - view.ema9[i]) / c[i])
        if i < len(view.ema50) and np.isfinite(view.ema50[i]) and c[i] > 0:
            out["price_vs_ema50_pct"] = float((c[i] - view.ema50[i]) / c[i])

        # --- RSI deltas (1-bar = 15m, 4-bar = 1h) ----------------------------
        if (
            i >= 1 and i < len(view.rsi14)
            and np.isfinite(view.rsi14[i]) and np.isfinite(view.rsi14[i - 1])
        ):
            out["rsi14_delta_1bar"] = float(view.rsi14[i] - view.rsi14[i - 1])
        if (
            i >= 4 and i < len(view.rsi14)
            and np.isfinite(view.rsi14[i]) and np.isfinite(view.rsi14[i - 4])
        ):
            out["rsi14_delta_4bar"] = float(view.rsi14[i] - view.rsi14[i - 4])

        # --- MACD (signal-spread normalised + 2-bar histogram momentum) ------
        if (
            i < len(view.macd) and i < len(view.macd_sig)
            and np.isfinite(view.macd[i]) and np.isfinite(view.macd_sig[i]) and c[i] > 0
        ):
            out["macd_signal_spread_norm"] = float(
                (view.macd[i] - view.macd_sig[i]) / c[i]
            )
        if (
            i >= 2 and i < len(view.macd_hist)
            and np.isfinite(view.macd_hist[i]) and np.isfinite(view.macd_hist[i - 2])
        ):
            out["macd_hist_momentum"] = float(view.macd_hist[i] - view.macd_hist[i - 2])

        # --- KDJ raw levels --------------------------------------------------
        if i < len(view.kdj_j) and np.isfinite(view.kdj_j[i]):
            out["kdj_j"] = float(view.kdj_j[i])
        if i < len(view.kdj_k) and np.isfinite(view.kdj_k[i]):
            out["kdj_k"] = float(view.kdj_k[i])

        # --- BTC trend score (Candidate B) -----------------------------------
        if self._btc_trend_score_15m is not None:
            v = self._btc_trend_score_15m.asof(entry_ts)
            if pd.notna(v):
                out["btc_trend_score"] = float(v)

        # --- BTC realized 24h vol --------------------------------------------
        if self._btc_realized_vol_24h_15m is not None:
            v = self._btc_realized_vol_24h_15m.asof(entry_ts)
            if pd.notna(v):
                out["btc_realized_vol_24h"] = float(v)

        # --- concurrent_dir_breadth ------------------------------------------
        if self._ret_4h_panel is not None:
            try:
                row = self._ret_4h_panel.loc[entry_ts]
            except KeyError:
                row = None
            if row is not None and row.notna().any():
                if direction == "short":
                    matches = (row < 0).sum()
                else:
                    matches = (row > 0).sum()
                out["concurrent_dir_breadth"] = float(matches / row.notna().sum())

        return out
