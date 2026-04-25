"""Feature library for the bigmover combined-with-indicators predictor (v1).

CONTRACT — strict no-lookahead:
  Every feature for trade `t` reads ONLY bars whose timestamp `<=` `entry_time(t)`.
  The bar at exact entry_time IS available (the signal fires at its close).
  Verified by tests/test_bigmover_combined_features.py via future-permutation
  invariance.

The 19 features (see plan):

Bigmover detector internals (5)
  - vol_ratio                 volume / 20-bar volume MA at entry bar
  - price_drop_pct            (rolling96_high - close) / rolling96_high
  - price_rise_pct            (close - rolling96_low) / rolling96_low
  - accel_atr_norm            (close - close[i-1]) / ATR14 (signed)
  - bars_since_high32         bars since the 32-bar high

EMA (3)
  - price_vs_ema20_pct        (close - EMA20) / close
  - price_vs_ema50_pct        (close - EMA50) / close
  - ema20_minus_ema50_atr     (EMA20 - EMA50) / ATR14

RSI (2)
  - rsi14                     14-bar Wilder/SMA RSI
  - rsi14_delta               RSI change over last 1 bar

MACD (3)
  - macd_norm                 macd / close
  - macd_signal_spread_norm   (macd - signal) / close
  - macd_hist_sign_change     1 if histogram crossed zero in last 4 bars else 0

KDJ (3)
  - kdj_k                     K line value
  - kdj_j_minus_k             J - K spread
  - kdj_k_above_d             1 if K > D else 0 (golden/death proxy)

BTC trend (2)
  - btc_trend_score           continuous score in [-1, 1], sign = direction of BTC
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

# --- knobs (mirror feature_meta.json) ---------------------------------------

VOL_MA_PERIOD = 20
PRICE_LOOKBACK_96 = 96
PRICE_LOOKBACK_32 = 32
ATR_LOOKBACK = 14
EMA_FAST = 20
EMA_SLOW = 50
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_N, KDJ_K, KDJ_D = 9, 3, 3
BREADTH_4H_LOOKBACK = 16
BTC_TREND_SMA = 50
BTC_TREND_CAP = 0.10  # |close - sma50| / sma50 above 10% saturates the score
BTC_VOL_LOOKBACK_24H_HOURS = 24
BTC_MACD_LOOKBACK = 4

FEATURE_NAMES = [
    # bigmover internals
    "vol_ratio",
    "price_drop_pct",
    "price_rise_pct",
    "accel_atr_norm",
    "bars_since_high32",
    # EMA
    "price_vs_ema20_pct",
    "price_vs_ema50_pct",
    "ema20_minus_ema50_atr",
    # RSI
    "rsi14",
    "rsi14_delta",
    # MACD
    "macd_norm",
    "macd_signal_spread_norm",
    "macd_hist_sign_change",
    # KDJ
    "kdj_k",
    "kdj_j_minus_k",
    "kdj_k_above_d",
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
    # rolling features
    vol_ma20: np.ndarray
    rolling_high96: np.ndarray
    rolling_low96: np.ndarray
    bars_since_high32: np.ndarray
    atr14: np.ndarray
    ema20: np.ndarray
    ema50: np.ndarray
    rsi14: np.ndarray
    macd: np.ndarray
    macd_sig: np.ndarray
    macd_hist: np.ndarray
    macd_hist_sign_change: np.ndarray
    kdj_k: np.ndarray
    kdj_d: np.ndarray
    kdj_j: np.ndarray


def _argmax_in_window_to_offset(arr: np.ndarray, window: int) -> np.ndarray:
    """For each i, position (within last `window` bars including i) of the max value.
    Returns offset = (window-1) - argmax_local, so 0 means current bar is the max.
    NaN before warmup.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = arr[i - window + 1 : i + 1]
        out[i] = float((window - 1) - int(np.argmax(seg)))
    return out


def _build_asset_view(sub: pd.DataFrame, ts_col: str = "timestamp") -> _AssetView:
    sub = sub.sort_values(ts_col).reset_index(drop=True)
    # Strip tz info → tz-naive datetime64[ns] for fast np.searchsorted lookup.
    ts = (
        pd.to_datetime(sub[ts_col], utc=True)
        .dt.tz_convert(None)
        .to_numpy()
    )
    o = sub["open"].to_numpy(dtype=float)
    h = sub["high"].to_numpy(dtype=float)
    l = sub["low"].to_numpy(dtype=float)
    c = sub["close"].to_numpy(dtype=float)
    v = sub["volume"].to_numpy(dtype=float)
    vol_ma20 = pd.Series(v).rolling(VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().to_numpy()
    rolling_high96 = (
        pd.Series(h).rolling(PRICE_LOOKBACK_96, min_periods=PRICE_LOOKBACK_96).max().to_numpy()
    )
    rolling_low96 = (
        pd.Series(l).rolling(PRICE_LOOKBACK_96, min_periods=PRICE_LOOKBACK_96).min().to_numpy()
    )
    bars_since_high32 = _argmax_in_window_to_offset(h, PRICE_LOOKBACK_32)
    atr14 = _atr(pd.Series(h), pd.Series(l), pd.Series(c), ATR_LOOKBACK).to_numpy()
    ema20 = _ema(pd.Series(c), EMA_FAST).to_numpy()
    ema50 = _ema(pd.Series(c), EMA_SLOW).to_numpy()
    rsi14 = _rsi(pd.Series(c), RSI_PERIOD).to_numpy()
    md = _macd(pd.Series(c), MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_arr = md["macd"].to_numpy()
    macd_sig_arr = md["signal"].to_numpy()
    macd_hist_arr = md["histogram"].to_numpy()
    # macd histogram sign change in last `BTC_MACD_LOOKBACK` bars (incl. current).
    sign = np.sign(macd_hist_arr)
    sign_change = np.zeros(len(sign), dtype=float)
    for i in range(BTC_MACD_LOOKBACK, len(sign)):
        window_signs = sign[i - BTC_MACD_LOOKBACK + 1 : i + 1]
        # 1 if both +1 and -1 present in window (i.e., crossed zero)
        if np.any(window_signs > 0) and np.any(window_signs < 0):
            sign_change[i] = 1.0
    sign_change[:BTC_MACD_LOOKBACK] = np.nan

    k = _kdj(pd.Series(h), pd.Series(l), pd.Series(c), n=KDJ_N, k_smooth=KDJ_K, d_smooth=KDJ_D)
    kdj_k = k["k"].to_numpy()
    kdj_d = k["d"].to_numpy()
    kdj_j = k["j"].to_numpy()

    return _AssetView(
        ts=ts, open=o, high=h, low=l, close=c, volume=v,
        vol_ma20=vol_ma20,
        rolling_high96=rolling_high96, rolling_low96=rolling_low96,
        bars_since_high32=bars_since_high32, atr14=atr14,
        ema20=ema20, ema50=ema50,
        rsi14=rsi14,
        macd=macd_arr, macd_sig=macd_sig_arr, macd_hist=macd_hist_arr,
        macd_hist_sign_change=sign_change,
        kdj_k=kdj_k, kdj_d=kdj_d, kdj_j=kdj_j,
    )


# --- snapshot-wide panels ---------------------------------------------------


@dataclass
class FeatureContext:
    """All snapshot-wide panels needed to build features for any trade.

    Construction is the heavy step (per-asset rolling indicators, BTC trend
    panel, cross-sectional breadth panel). Per-trade lookup is then O(1).
    """
    candles: pd.DataFrame
    btc_asset_key: str = "BTCUSDT"
    universe_listed_since: dict[str, pd.Timestamp] = field(default_factory=dict)

    _btc_view: _AssetView | None = field(init=False, default=None)
    _btc_trend_score_15m: pd.Series | None = field(init=False, default=None)
    _btc_realized_vol_24h_15m: pd.Series | None = field(init=False, default=None)
    _ret_4h_panel: pd.DataFrame | None = field(init=False, default=None)
    _per_asset: dict[str, _AssetView] = field(init=False, default_factory=dict)

    def __post_init__(self):
        # BTC view
        btc_sub = self.candles[self.candles["asset"] == self.btc_asset_key]
        if len(btc_sub) == 0:
            raise ValueError(
                f"No candles for BTC key {self.btc_asset_key!r}; "
                "set FeatureContext(candles, btc_asset_key=...) to the right symbol."
            )
        self._btc_view = _build_asset_view(btc_sub)

        # BTC trend score: continuous, signed, in [-1, 1].
        # Score = clip((close - SMA50_daily) / SMA50_daily, -BTC_TREND_CAP, +BTC_TREND_CAP)
        #         / BTC_TREND_CAP. So +1 = BTC at +10% above SMA50 daily (strong bull);
        # -1 = BTC at -10% below SMA50 daily (strong bear); 0 = at SMA50.
        btc_15m_close = (
            btc_sub.sort_values("timestamp").set_index("timestamp")["close"].astype(float)
        )
        btc_daily = btc_15m_close.resample("1D", label="right", closed="right").last()
        btc_sma50 = btc_daily.rolling(window=BTC_TREND_SMA, min_periods=BTC_TREND_SMA).mean()
        # No-lookahead: at any 15m timestamp, use the most recent CLOSED daily bar's
        # SMA50 (= shift by 1 daily bar).
        score_daily = ((btc_daily - btc_sma50) / btc_sma50).shift(1)
        score_daily = score_daily.clip(-BTC_TREND_CAP, BTC_TREND_CAP) / BTC_TREND_CAP
        self._btc_trend_score_15m = score_daily.reindex(btc_15m_close.index, method="ffill")

        # BTC realized 24h vol
        btc_1h_close = btc_15m_close.resample("1h", label="right", closed="right").last()
        btc_1h_ret = btc_1h_close.pct_change()
        rv = btc_1h_ret.rolling(window=BTC_VOL_LOOKBACK_24H_HOURS, min_periods=BTC_VOL_LOOKBACK_24H_HOURS).std(ddof=0)
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

        c, h, l, o, vol = view.close, view.high, view.low, view.open, view.volume

        # vol_ratio
        if i < len(view.vol_ma20) and view.vol_ma20[i] > 0:
            out["vol_ratio"] = float(vol[i] / view.vol_ma20[i])

        # price_drop_pct, price_rise_pct
        if i < len(view.rolling_high96) and np.isfinite(view.rolling_high96[i]) and view.rolling_high96[i] > 0:
            out["price_drop_pct"] = float((view.rolling_high96[i] - c[i]) / view.rolling_high96[i])
        if i < len(view.rolling_low96) and np.isfinite(view.rolling_low96[i]) and view.rolling_low96[i] > 0:
            out["price_rise_pct"] = float((c[i] - view.rolling_low96[i]) / view.rolling_low96[i])

        # accel_atr_norm
        if i >= 1 and i < len(view.atr14) and np.isfinite(view.atr14[i]) and view.atr14[i] > 0:
            out["accel_atr_norm"] = float((c[i] - c[i - 1]) / view.atr14[i])

        # bars_since_high32
        if i < len(view.bars_since_high32) and np.isfinite(view.bars_since_high32[i]):
            out["bars_since_high32"] = float(view.bars_since_high32[i])

        # EMA features
        if i < len(view.ema20) and np.isfinite(view.ema20[i]) and c[i] > 0:
            out["price_vs_ema20_pct"] = float((c[i] - view.ema20[i]) / c[i])
        if i < len(view.ema50) and np.isfinite(view.ema50[i]) and c[i] > 0:
            out["price_vs_ema50_pct"] = float((c[i] - view.ema50[i]) / c[i])
        if (
            i < len(view.ema20) and i < len(view.ema50) and i < len(view.atr14)
            and np.isfinite(view.ema20[i]) and np.isfinite(view.ema50[i])
            and np.isfinite(view.atr14[i]) and view.atr14[i] > 0
        ):
            out["ema20_minus_ema50_atr"] = float((view.ema20[i] - view.ema50[i]) / view.atr14[i])

        # RSI features
        if i < len(view.rsi14) and np.isfinite(view.rsi14[i]):
            out["rsi14"] = float(view.rsi14[i])
            if i >= 1 and np.isfinite(view.rsi14[i - 1]):
                out["rsi14_delta"] = float(view.rsi14[i] - view.rsi14[i - 1])

        # MACD features
        if i < len(view.macd) and np.isfinite(view.macd[i]) and c[i] > 0:
            out["macd_norm"] = float(view.macd[i] / c[i])
        if (
            i < len(view.macd) and i < len(view.macd_sig)
            and np.isfinite(view.macd[i]) and np.isfinite(view.macd_sig[i]) and c[i] > 0
        ):
            out["macd_signal_spread_norm"] = float(
                (view.macd[i] - view.macd_sig[i]) / c[i]
            )
        if i < len(view.macd_hist_sign_change) and np.isfinite(view.macd_hist_sign_change[i]):
            out["macd_hist_sign_change"] = float(view.macd_hist_sign_change[i])

        # KDJ features
        if i < len(view.kdj_k) and np.isfinite(view.kdj_k[i]):
            out["kdj_k"] = float(view.kdj_k[i])
            if np.isfinite(view.kdj_j[i]):
                out["kdj_j_minus_k"] = float(view.kdj_j[i] - view.kdj_k[i])
            if np.isfinite(view.kdj_d[i]):
                out["kdj_k_above_d"] = float(view.kdj_k[i] > view.kdj_d[i])

        # BTC trend score (asof at entry_ts so we get the most recent value <= ts)
        if self._btc_trend_score_15m is not None:
            v = self._btc_trend_score_15m.asof(entry_ts)
            if pd.notna(v):
                out["btc_trend_score"] = float(v)

        # BTC realized 24h vol
        if self._btc_realized_vol_24h_15m is not None:
            v = self._btc_realized_vol_24h_15m.asof(entry_ts)
            if pd.notna(v):
                out["btc_realized_vol_24h"] = float(v)

        # concurrent_dir_breadth: fraction of universe whose 4h ret matches trade direction
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
