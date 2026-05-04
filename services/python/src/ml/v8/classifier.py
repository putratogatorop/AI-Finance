"""V8 BALANCED — live trade-quality classifier (sizing-mode).

Loads the per-detector HistGradientBoosting models trained in
scripts/train_v8_classifier_3y_sizing.py (output at models/v8_classifier_3y/).

For each incoming v8 signal, computes 27 pre-entry features from:
  - The asset's recent 15m candle history (DB query, ~30 days)
  - BTC's recent candle history (cached, refreshed every 15m)
  - Universe breadth (cached, refreshed every 15m)
  - Time-of-week (from signal_time)

Then scores via the detector's classifier and returns a sizing multiplier
in [SIZE_LOW, SIZE_HIGH] (default [0.5, 1.5]), capital-neutral on the
classifier's TRAIN distribution. Multiplier of 1.0 means average confidence;
< 1.0 = below-average, > 1.0 = above-average.

Public API:
    cls = V8Classifier(engine)
    multiplier = cls.score_size(detector, asset, entry_time, signal_row)
    # In paper_executor, multiply v6 sizing's notional by `multiplier`.

Failure modes (all return 1.0 — no size adjustment):
    - Detector not trained (e.g., rsi_recovery_long, n=3)
    - Insufficient candle history for feature computation
    - Any exception during scoring (logged, not raised)
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgbm  # noqa: F401 (needed to unpickle CalibratedLGBM from v2p2 models)
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression  # noqa: F401
from sqlalchemy import text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

# --- CONFIG ----------------------------------------------------------------

MODELS_DIR = Path(__file__).resolve().parents[3] / "models" / "v8_classifier_3y"

# Feature schema MUST match prepare_v8_training_data_3y.py exactly
FEATURES: list[str] = [
    "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
    "close_to_high50_atr", "close_to_low50_atr",
    "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
    "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
    "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
    "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
    "breadth_up", "breadth_down", "signals_same_15m_same_detector",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

# Detector → live signal table (used for breadth + same-15m count queries)
DETECTOR_SIGNAL_TABLES: dict[str, str] = {
    "v8_macd_pullback_long_e2": "scanner_signals_macd_pullback_long",
    "v8_macd_pullback_short_e2": "scanner_signals_macd_pullback_short",
    "v8_macd_early_trend_short_e2": "scanner_signals_macd_early_trend_short",
    "v8_rsi_recovery_long_e2": "scanner_signals_rsi_recovery_long",
}

# Detector → trained-model filename stem
DETECTOR_MODEL_STEMS: dict[str, str] = {
    "v8_macd_pullback_long_e2": "macd_pullback_long_histgbm_3y_v1",
    "v8_macd_pullback_short_e2": "macd_pullback_short_histgbm_3y_v1",
    "v8_macd_early_trend_short_e2": "macd_early_trend_short_histgbm_3y_v1",
    # rsi_recovery_long not trained — only 3 trades in 3y data
}

# Indicator params (must match training)
ATR_PERIOD = 14
EMA_TREND_SPAN = 50
RSI_PERIOD = 14

# Cache TTL — refresh BTC + breadth every 15 minutes (one bar)
CACHE_TTL_SECONDS = 15 * 60

# ── v2p2 model constants (parallel A/B scoring path, does not affect live trades) ──

MODELS_DIR_V2P2 = Path(__file__).resolve().parents[3] / "models" / "v8_classifier_lgbm_v2p2"

# Optimal threshold from Phase 2 training (shifted from v1's 0.40 due to isotonic calibration)
CLS_THRESHOLD_V2P2: dict[str, float] = {
    "v8_macd_pullback_long_e2": 0.475,
}

DETECTOR_MODEL_STEMS_V2P2: dict[str, str] = {
    "v8_macd_pullback_long_e2": "macd_pullback_long_lgbm_v2p2",
}

# 40 features: 27 P1 + 13 P2 (cvd_divergence_4h pruned: SHAP=0)
FEATURES_V2P2: list[str] = [
    # P1
    "atr14_pct_rank_90d", "vol_z_24h", "coin_7d_return", "coin_30d_return",
    "close_to_high50_atr", "close_to_low50_atr",
    "bar4h_close_pos_in_range", "bar4h_body_pct", "bar4h_upper_wick_pct",
    "h4_macd_hist", "h4_macd_macd", "h4_rsi", "h4_close_vs_ema50_pct",
    "daily_macd_hist", "days_since_bull_flip", "days_since_bear_flip",
    "btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score",
    "breadth_up", "breadth_down", "signals_same_15m_same_detector",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # P2
    "cvd_slope_1h_norm", "bb_pct_b", "bb_bandwidth", "fib_pos_50",
    "kdj_k", "kdj_j", "price_vs_ema9_pct",
    "rsi14_delta_1bar", "rsi14_delta_4bar",
    "macd_hist_momentum", "macd_signal_spread_norm",
    "parkinson_vol", "obv_slope_norm",
]


# --- INDICATOR HELPERS (same math as training) --------------------------

def _atr14(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    if n < 2:
        return np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= ATR_PERIOD:
        atr[ATR_PERIOD - 1] = tr[:ATR_PERIOD].mean()
        for i in range(ATR_PERIOD, n):
            atr[i] = (atr[i - 1] * (ATR_PERIOD - 1) + tr[i]) / ATR_PERIOD
    return atr


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"macd": macd, "signal": sig, "histogram": macd - sig})


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


# --- CALIBRATED LGBM WRAPPER (for joblib unpickling v2p2 models) -----------

class CalibratedLGBM:
    """LGBMClassifier + IsotonicRegression calibrator from Phase 2 training."""

    def __init__(self, base_model: lgbm.LGBMClassifier, calibrator: IsotonicRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])


# --- LIVE FEATURE BUILDER --------------------------------------------------

@dataclass
class _Caches:
    btc_above_ema_at: pd.Series | None = None
    btc_24h_ret_at: pd.Series | None = None
    btc_rvz_at: pd.Series | None = None
    btc_score_at: pd.Series | None = None
    breadth_up_at: pd.Series | None = None
    breadth_down_at: pd.Series | None = None
    refreshed_at: float = 0.0


class V8Classifier:
    def __init__(self, engine: Engine, models_dir: Path | None = None) -> None:
        self.engine = engine
        self.models_dir = models_dir or MODELS_DIR
        self._models: dict[str, Any] = {}
        self._metas: dict[str, dict] = {}
        self._models_v2: dict[str, Any] = {}
        self._metas_v2: dict[str, dict] = {}
        self._cache = _Caches()
        self._load_models()
        self._load_v2_models()

    def _load_models(self) -> None:
        for det, stem in DETECTOR_MODEL_STEMS.items():
            model_path = self.models_dir / f"{stem}.joblib"
            meta_path = self.models_dir / f"{stem}_meta.json"
            if not model_path.exists() or not meta_path.exists():
                logger.warning(
                    "v8 classifier model missing for %s at %s — that detector will use multiplier=1.0",
                    det, model_path,
                )
                continue
            self._models[det] = joblib.load(model_path)
            with open(meta_path) as f:
                self._metas[det] = json.load(f)
            logger.info(
                "v8 classifier loaded: %s (auc_holdout=%.3f, n_train=%d, "
                "size_range=[%.2f, %.2f])",
                det,
                self._metas[det].get("auc_holdout", float("nan")),
                self._metas[det].get("n_train", 0),
                self._metas[det].get("size_low", 0.5),
                self._metas[det].get("size_high", 1.5),
            )

    def _load_v2_models(self) -> None:
        for det, stem in DETECTOR_MODEL_STEMS_V2P2.items():
            model_path = MODELS_DIR_V2P2 / f"{stem}.joblib"
            meta_path = MODELS_DIR_V2P2 / f"{stem}_meta.json"
            if not model_path.exists() or not meta_path.exists():
                logger.info("v8 v2p2 classifier not found for %s — A/B scoring disabled", det)
                continue
            self._models_v2[det] = joblib.load(model_path)
            with open(meta_path) as f:
                self._metas_v2[det] = json.load(f)
            logger.info("v8 v2p2 classifier loaded: %s (n_features=%d, thr=%.3f)",
                        det, self._metas_v2[det].get("n_features", 0),
                        CLS_THRESHOLD_V2P2.get(det, 0.475))

    # --- public API --------------------------------------------------------

    def score_size(
        self,
        strategy: str,
        symbol: str,
        entry_time: datetime,
        signal_row: dict[str, Any],
    ) -> tuple[float, dict[str, Any]]:
        """Return (size_multiplier, info_dict) for the given v8 signal.

        Multiplier in [size_low, size_high] (default [0.5, 1.5]) — multiply
        v6 sizing's notional_usd by this. info_dict contains {score, mode}
        for audit logging.

        Returns (1.0, {'mode': '...'}) if scoring is unavailable for any
        reason — never raises.
        """
        if strategy not in self._models:
            return 1.0, {"mode": "no_model", "reason": f"detector {strategy} not trained"}

        try:
            features = self._build_features(strategy, symbol, entry_time, signal_row)
        except Exception as e:
            logger.warning("v8 classifier feature build failed for %s/%s: %s", strategy, symbol, e)
            return 1.0, {"mode": "feature_error", "reason": str(e)[:200]}

        # Validate
        x = np.array([features[f] for f in FEATURES], dtype=float)
        if not np.all(np.isfinite(x)):
            missing = [FEATURES[i] for i in range(len(FEATURES)) if not np.isfinite(x[i])]
            logger.warning(
                "v8 classifier feature NaN for %s/%s: %s — using multiplier=1.0",
                strategy, symbol, missing[:5],
            )
            return 1.0, {"mode": "nan_features", "missing": missing[:10]}

        try:
            score = float(self._models[strategy].predict_proba(x.reshape(1, -1))[0, 1])
        except Exception as e:
            logger.warning("v8 classifier predict failed for %s/%s: %s", strategy, symbol, e)
            return 1.0, {"mode": "predict_error", "reason": str(e)[:200]}

        size = self._score_to_size(strategy, score)
        return size, {"mode": "sizing", "score": round(score, 4), "size": round(size, 3)}

    # --- score → size mapping (locked from training) ----------------------

    def _score_to_size(self, strategy: str, score: float) -> float:
        meta = self._metas[strategy]
        quantiles = np.asarray(meta["train_score_quantiles"], dtype=float)
        n_q = len(quantiles)
        pos = int(np.searchsorted(quantiles, score, side="right"))
        rank = max(0.0, min(1.0, pos / n_q))
        size_low = float(meta.get("size_low", 0.5))
        size_high = float(meta.get("size_high", 1.5))
        return size_low + (size_high - size_low) * rank

    def score_size_v2(
        self,
        strategy: str,
        symbol: str,
        entry_time: datetime,
        signal_row: dict[str, Any],
    ) -> tuple[float | None, dict[str, Any]]:
        """Return (cls_score_v2, info_dict) using the v2p2 LightGBM model.

        This is a parallel A/B scoring path — result is logged for comparison
        but does NOT affect live trading decisions. Returns (None, info) if the
        v2p2 model is not loaded or feature computation fails.
        """
        if strategy not in self._models_v2:
            return None, {"mode": "no_v2_model"}

        try:
            p1_feats = self._build_features(strategy, symbol, entry_time, signal_row)
            p2_feats = self._compute_p2_extra_features(symbol, entry_time)
        except Exception as e:
            logger.debug("v8 v2p2 feature build failed for %s/%s: %s", strategy, symbol, e)
            return None, {"mode": "feature_error_v2", "reason": str(e)[:120]}

        x = np.array([{**p1_feats, **p2_feats}[f] for f in FEATURES_V2P2], dtype=float)
        if not np.all(np.isfinite(x)):
            return None, {"mode": "nan_features_v2"}

        try:
            score = float(self._models_v2[strategy].predict_proba(x.reshape(1, -1))[0, 1])
        except Exception as e:
            logger.debug("v8 v2p2 predict failed for %s/%s: %s", strategy, symbol, e)
            return None, {"mode": "predict_error_v2", "reason": str(e)[:120]}

        return score, {"mode": "v2p2", "score_v2": round(score, 4),
                       "thr_v2": CLS_THRESHOLD_V2P2.get(strategy, 0.475)}

    def _compute_p2_extra_features(self, symbol: str, entry_time: datetime) -> dict[str, float]:
        """Compute the 13 P2 features from live 15m candles.

        Uses the same asset_prices_15m query as _compute_per_asset_features.
        All features are OHLCV-derived with no lookahead.
        """
        bars_needed = 90 * 96
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT timestamp, high, low, close, volume
                    FROM asset_prices_15m
                    WHERE asset = :asset AND timestamp <= :t
                    ORDER BY timestamp DESC
                    LIMIT :lim
                """),
                {"asset": symbol, "t": entry_time, "lim": bars_needed},
            ).fetchall()
        if len(rows) < 200:
            raise ValueError(f"insufficient candles for {symbol} ({len(rows)} bars)")

        df = pd.DataFrame(rows, columns=["timestamp", "high", "low", "close", "volume"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        c = df["close"].astype(float).to_numpy()
        h = df["high"].astype(float).to_numpy()
        lo = df["low"].astype(float).to_numpy()
        v = df["volume"].astype(float).to_numpy()
        n = len(c)

        s_close = pd.Series(c, index=ts_idx)
        s_high = pd.Series(h, index=ts_idx)
        s_low = pd.Series(lo, index=ts_idx)
        s_vol = pd.Series(v, index=ts_idx)

        # ── 15m features ──────────────────────────────────────────────────
        sign_diff = np.sign(s_close.diff().fillna(0.0))
        cvd = (sign_diff * s_vol).cumsum()
        cvd_4 = float(cvd.iloc[-5]) if n >= 5 else float("nan")
        vol_abs_sum4 = float(s_vol.iloc[-4:].sum())
        cvd_slope_1h_norm = (float(cvd.iloc[-1]) - cvd_4) / vol_abs_sum4 \
            if vol_abs_sum4 > 0 and np.isfinite(cvd_4) else float("nan")

        h50 = float(s_close.iloc[-50:].max()) if n >= 50 else float("nan")
        l50 = float(s_close.iloc[-50:].min()) if n >= 50 else float("nan")
        fib_pos_50 = (c[-1] - l50) / (h50 - l50) if np.isfinite(h50) and (h50 - l50) > 0 \
            else float("nan")

        if n >= 96:
            log_hl = np.log((s_high.iloc[-96:] / s_low.iloc[-96:]).replace(0.0, np.nan))
            parkinson_sq_mean = float((log_hl ** 2).mean())
            parkinson_vol = float(np.sqrt(parkinson_sq_mean / (4.0 * np.log(2.0)))) \
                if parkinson_sq_mean >= 0 else float("nan")
        else:
            parkinson_vol = float("nan")

        if n >= 24:
            obv = (sign_diff * s_vol).cumsum()
            obv_24 = obv.iloc[-24:].to_numpy()
            x_vals = np.arange(24, dtype=float)
            obv_slope = float(np.polyfit(x_vals, obv_24, 1)[0])
            mean_vol_24 = float(s_vol.iloc[-24:].mean())
            obv_slope_norm = obv_slope / mean_vol_24 if mean_vol_24 > 0 else float("nan")
        else:
            obv_slope_norm = float("nan")

        # ── 4h features (prior completed bar) ────────────────────────────
        h4_close = s_close.resample("4h").last().dropna()
        h4_high = s_high.resample("4h").max().reindex(h4_close.index)
        h4_low = s_low.resample("4h").min().reindex(h4_close.index)

        prior_4h_label = pd.Timestamp(entry_time).floor("4h") - pd.Timedelta("4h")
        if prior_4h_label not in h4_close.index:
            candidates = h4_close.index[h4_close.index < pd.Timestamp(entry_time)]
            if len(candidates) == 0:
                nan_feat: dict[str, float] = {
                    f: float("nan") for f in [
                        "bb_pct_b", "bb_bandwidth", "kdj_k", "kdj_j",
                        "price_vs_ema9_pct", "rsi14_delta_1bar", "rsi14_delta_4bar",
                        "macd_hist_momentum", "macd_signal_spread_norm",
                    ]
                }
                return {"cvd_slope_1h_norm": cvd_slope_1h_norm, "fib_pos_50": fib_pos_50,
                        "parkinson_vol": parkinson_vol, "obv_slope_norm": obv_slope_norm,
                        **nan_feat}
            prior_4h_label = candidates[-1]

        C4 = float(h4_close.loc[prior_4h_label])
        prior_pos = int(h4_close.index.get_loc(prior_4h_label))

        def _nan_if_not_enough(series: pd.Series, idx: int, min_pos: int) -> float:
            return float(series.iloc[idx]) if idx >= min_pos else float("nan")

        # Bollinger Bands (20-bar)
        bb_mid_s = h4_close.rolling(20, min_periods=20).mean()
        bb_std_s = h4_close.rolling(20, min_periods=20).std()
        bb_mid = _nan_if_not_enough(bb_mid_s, prior_pos, 19)
        bb_std = _nan_if_not_enough(bb_std_s, prior_pos, 19)
        if np.isfinite(bb_mid) and np.isfinite(bb_std) and bb_std > 0:
            bb_upper = bb_mid + 2.0 * bb_std
            bb_lower = bb_mid - 2.0 * bb_std
            bb_rng = bb_upper - bb_lower
            bb_pct_b = (C4 - bb_lower) / bb_rng
            bb_bandwidth = bb_rng / bb_mid if bb_mid > 0 else float("nan")
        else:
            bb_pct_b = bb_bandwidth = float("nan")

        # KDJ (9-period RSV, 3-smooth K, 3-smooth D)
        roll_lo_s = h4_low.rolling(9, min_periods=9).min()
        roll_hi_s = h4_high.rolling(9, min_periods=9).max()
        rsv_s = 100.0 * (h4_close - roll_lo_s) / (roll_hi_s - roll_lo_s).replace(0.0, np.nan)
        kdj_k_s = rsv_s.rolling(3, min_periods=3).mean()
        kdj_d_s = kdj_k_s.rolling(3, min_periods=3).mean()
        kdj_j_s = 3.0 * kdj_k_s - 2.0 * kdj_d_s
        kdj_k = _nan_if_not_enough(kdj_k_s, prior_pos, 10)
        kdj_j = _nan_if_not_enough(kdj_j_s, prior_pos, 12)

        # EMA9 → price_vs_ema9_pct
        ema9_4h = h4_close.ewm(span=9, adjust=False).mean()
        ema9_at = _nan_if_not_enough(ema9_4h, prior_pos, 0)
        price_vs_ema9_pct = (C4 - ema9_at) / C4 if C4 > 0 and np.isfinite(ema9_at) \
            else float("nan")

        # RSI14 deltas
        delta = h4_close.diff()
        avg_gain = delta.clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean()
        avg_loss = (-delta).clip(lower=0).ewm(alpha=1.0 / 14, adjust=False).mean()
        rsi_s = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss.replace(0.0, np.nan))
        rsi_at = _nan_if_not_enough(rsi_s, prior_pos, 13)
        rsi_1b = _nan_if_not_enough(rsi_s, prior_pos - 1, 13) if prior_pos >= 1 else float("nan")
        rsi_4b = _nan_if_not_enough(rsi_s, prior_pos - 4, 13) if prior_pos >= 4 else float("nan")
        rsi14_delta_1bar = (rsi_at - rsi_1b) if np.isfinite(rsi_at) and np.isfinite(rsi_1b) \
            else float("nan")
        rsi14_delta_4bar = (rsi_at - rsi_4b) if np.isfinite(rsi_at) and np.isfinite(rsi_4b) \
            else float("nan")

        # MACD histogram momentum + spread norm
        ema_fast = h4_close.ewm(span=12, adjust=False).mean()
        ema_slow = h4_close.ewm(span=26, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        macd_sig = macd_line.ewm(span=9, adjust=False).mean()
        hist_s = macd_line - macd_sig
        hist_at = _nan_if_not_enough(hist_s, prior_pos, 25)
        hist_2b = _nan_if_not_enough(hist_s, prior_pos - 2, 25) if prior_pos >= 2 \
            else float("nan")
        macd_hist_momentum = (hist_at - hist_2b) if np.isfinite(hist_at) and np.isfinite(hist_2b) \
            else float("nan")
        macd_signal_spread_norm = hist_at / C4 if C4 > 0 and np.isfinite(hist_at) else float("nan")

        return {
            "cvd_slope_1h_norm": cvd_slope_1h_norm,
            "bb_pct_b": bb_pct_b,
            "bb_bandwidth": bb_bandwidth,
            "fib_pos_50": fib_pos_50,
            "kdj_k": kdj_k,
            "kdj_j": kdj_j,
            "price_vs_ema9_pct": price_vs_ema9_pct,
            "rsi14_delta_1bar": rsi14_delta_1bar,
            "rsi14_delta_4bar": rsi14_delta_4bar,
            "macd_hist_momentum": macd_hist_momentum,
            "macd_signal_spread_norm": macd_signal_spread_norm,
            "parkinson_vol": parkinson_vol,
            "obv_slope_norm": obv_slope_norm,
        }

    # --- feature build ----------------------------------------------------

    def _build_features(
        self,
        strategy: str,
        symbol: str,
        entry_time: datetime,
        signal_row: dict[str, Any],
    ) -> dict[str, float]:
        # Refresh BTC + breadth caches if stale
        self._refresh_caches_if_stale()

        # Per-asset features (ATR rank, 7d/30d returns, bar shape, MACD/RSI, etc.)
        per_asset = self._compute_per_asset_features(symbol, entry_time)

        # BTC features at entry_time (from cache)
        btc_above = self._lookup_at(self._cache.btc_above_ema_at, entry_time)
        btc_24h = self._lookup_at(self._cache.btc_24h_ret_at, entry_time)
        btc_rvz = self._lookup_at(self._cache.btc_rvz_at, entry_time)
        btc_score = self._lookup_at(self._cache.btc_score_at, entry_time)

        # Breadth at entry_time (from cache)
        breadth_up = self._lookup_at(self._cache.breadth_up_at, entry_time)
        breadth_down = self._lookup_at(self._cache.breadth_down_at, entry_time)

        # signals_same_15m_same_detector: count this detector's signals at this exact 15m
        sig_table = DETECTOR_SIGNAL_TABLES.get(strategy)
        if sig_table:
            with self.engine.connect() as conn:
                row = conn.execute(
                    text(f"SELECT COUNT(*) FROM {sig_table} WHERE signal_time = :t"),  # noqa: S608
                    {"t": entry_time},
                ).fetchone()
            n_same = int(row[0]) if row else 1
        else:
            n_same = 1

        # Time of week (cyclical)
        hour = entry_time.hour
        dow = entry_time.weekday()
        hour_sin = math.sin(2 * math.pi * hour / 24)
        hour_cos = math.cos(2 * math.pi * hour / 24)
        dow_sin = math.sin(2 * math.pi * dow / 7)
        dow_cos = math.cos(2 * math.pi * dow / 7)

        # Use signal_row values where available (atr14, btc_score, etc.)
        # but prefer recomputed values for consistency with training
        return {
            **per_asset,
            "btc_above_4h_ema50": float(btc_above) if btc_above is not None else float("nan"),
            "btc_24h_return": float(btc_24h) if btc_24h is not None else float("nan"),
            "btc_realized_vol_z": float(btc_rvz) if btc_rvz is not None else float("nan"),
            "btc_score": float(btc_score) if btc_score is not None else float(signal_row.get("btc_score", float("nan"))),
            "breadth_up": float(breadth_up) if breadth_up is not None else float("nan"),
            "breadth_down": float(breadth_down) if breadth_down is not None else float("nan"),
            "signals_same_15m_same_detector": float(n_same),
            "hour_sin": hour_sin, "hour_cos": hour_cos,
            "dow_sin": dow_sin, "dow_cos": dow_cos,
        }

    def _compute_per_asset_features(self, symbol: str, entry_time: datetime) -> dict[str, float]:
        """Query asset's last ~35 days of 15m candles, compute features at the
        entry-time bar. Returns dict for the 16 per-asset feature columns."""
        # 35 days × 96 bars/day = 3360. Pull 90 days for the ATR percentile.
        bars_needed = 90 * 96
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT timestamp, open, high, low, close, volume
                    FROM asset_prices_15m
                    WHERE asset = :asset AND timestamp <= :t
                    ORDER BY timestamp DESC
                    LIMIT :lim
                """),
                {"asset": symbol, "t": entry_time, "lim": bars_needed},
            ).fetchall()
        if len(rows) < 4 * 96 + 30 * 96:
            raise ValueError(f"insufficient candles for {symbol} ({len(rows)} bars)")

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.sort_values("timestamp").reset_index(drop=True)

        ts = df["timestamp"].to_numpy()
        h = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        o = df["open"].to_numpy(dtype=float)
        c = df["close"].to_numpy(dtype=float)
        v = df["volume"].to_numpy(dtype=float)
        n = len(c)

        ts_idx = pd.DatetimeIndex(pd.to_datetime(ts, utc=True))
        s_close = pd.Series(c, index=ts_idx)
        s_high = pd.Series(h, index=ts_idx)
        s_low = pd.Series(lo, index=ts_idx)
        s_open = pd.Series(o, index=ts_idx)
        s_vol = pd.Series(v, index=ts_idx)

        atr14 = _atr14(h, lo, c)

        # ATR pct rank — 90 days
        s_atr = pd.Series(atr14)
        atr_pct_rank = float(s_atr.rolling(8640, min_periods=500).rank(pct=True).iloc[-1])

        # Volume z-score
        vol_24h = s_vol.rolling(96, min_periods=96).sum()
        vol_24h_mean_30d = vol_24h.rolling(2880, min_periods=720).mean()
        vol_24h_std_30d = vol_24h.rolling(2880, min_periods=720).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            vz = (vol_24h - vol_24h_mean_30d) / vol_24h_std_30d
        vol_z = float(vz.iloc[-1])

        # 7d / 30d returns
        coin_7d = float(c[-1] / c[-7 * 96] - 1.0) if n > 7 * 96 else float("nan")
        coin_30d = float(c[-1] / c[-30 * 96] - 1.0) if n > 30 * 96 else float("nan")

        # 50-bar high / low normalized by ATR
        h50 = float(h[-50:].max()) if n >= 50 else float("nan")
        l50 = float(lo[-50:].min()) if n >= 50 else float("nan")
        atr_now = float(atr14[-1])
        close_to_high50_atr = (h50 - c[-1]) / atr_now if atr_now > 0 else float("nan")
        close_to_low50_atr = (c[-1] - l50) / atr_now if atr_now > 0 else float("nan")

        # 4H bar shape — use the PRIOR-completed 4H bar (the one before the entry bar's 4H period)
        # The entry bar is the FIRST 15m of a new 4H period, so we want the 4H bar that just
        # closed (the one labeled at entry_time - 4h in the resampled left-labeled series).
        h4_close = s_close.resample("4h").last().dropna()
        h4_open = s_open.resample("4h").first().reindex(h4_close.index)
        h4_high = s_high.resample("4h").max().reindex(h4_close.index)
        h4_low = s_low.resample("4h").min().reindex(h4_close.index)
        # The 4H bar that just closed before entry_time has label = entry_time - 4h
        prior_4h_label = pd.Timestamp(entry_time).floor("4h") - pd.Timedelta("4h")
        if prior_4h_label not in h4_close.index:
            # Fall back to the latest 4H bar before entry_time
            prior_4h_label = h4_close.index[h4_close.index < pd.Timestamp(entry_time)][-1]
        H = float(h4_high.loc[prior_4h_label])
        L = float(h4_low.loc[prior_4h_label])
        O = float(h4_open.loc[prior_4h_label])
        C = float(h4_close.loc[prior_4h_label])
        rng = H - L
        bar4h_close_pos = (C - L) / rng if rng > 0 else float("nan")
        bar4h_body_pct = abs(C - O) / rng if rng > 0 else float("nan")
        bar4h_upper_wick = (H - max(O, C)) / rng if rng > 0 else float("nan")

        # 4H MACD (latest closed 4h)
        h4_macd_df = _macd(h4_close, 12, 26, 9)
        h4_macd_macd = float(h4_macd_df["macd"].loc[prior_4h_label])
        h4_macd_hist = float(h4_macd_df["histogram"].loc[prior_4h_label])

        # 4H RSI + EMA50
        h4_rsi_series = _rsi(h4_close, RSI_PERIOD)
        h4_ema50 = _ema(h4_close, EMA_TREND_SPAN)
        h4_rsi = float(h4_rsi_series.loc[prior_4h_label])
        ema50_now = float(h4_ema50.loc[prior_4h_label])
        h4_close_vs_ema50 = (C - ema50_now) / ema50_now if ema50_now > 0 else float("nan")

        # Daily MACD + flip recency
        d_close = s_close.resample("1D").last().dropna()
        d_macd_df = _macd(d_close, 12, 26, 9)
        # Use the day BEFORE entry_time's day (no look-ahead)
        prior_day_label = pd.Timestamp(entry_time).floor("D") - pd.Timedelta("1D")
        if prior_day_label not in d_close.index:
            prior_day_label = d_close.index[d_close.index < pd.Timestamp(entry_time)][-1]
        daily_macd_hist = float(d_macd_df["histogram"].loc[prior_day_label])

        # Days since most recent bull/bear flip (counted as days from prior_day_label backwards)
        d_macd_arr = d_macd_df["macd"].values
        d_sig_arr = d_macd_df["signal"].values
        d_hist_arr = d_macd_df["histogram"].values
        d_bull = (d_hist_arr > 0) & (d_macd_arr > d_sig_arr)
        d_bear = (d_hist_arr < 0) & (d_macd_arr < d_sig_arr)
        prior_day_pos = int(d_close.index.get_loc(prior_day_label))
        last_bull_flip = -1
        last_bear_flip = -1
        for i in range(prior_day_pos + 1):
            if i > 0 and d_bull[i] and not d_bull[i - 1]:
                last_bull_flip = i
            if i > 0 and d_bear[i] and not d_bear[i - 1]:
                last_bear_flip = i
        days_since_bull = float(prior_day_pos - last_bull_flip) if last_bull_flip >= 0 else float("nan")
        days_since_bear = float(prior_day_pos - last_bear_flip) if last_bear_flip >= 0 else float("nan")

        return {
            "atr14_pct_rank_90d": atr_pct_rank,
            "vol_z_24h": vol_z,
            "coin_7d_return": coin_7d,
            "coin_30d_return": coin_30d,
            "close_to_high50_atr": close_to_high50_atr,
            "close_to_low50_atr": close_to_low50_atr,
            "bar4h_close_pos_in_range": bar4h_close_pos,
            "bar4h_body_pct": bar4h_body_pct,
            "bar4h_upper_wick_pct": bar4h_upper_wick,
            "h4_macd_hist": h4_macd_hist,
            "h4_macd_macd": h4_macd_macd,
            "h4_rsi": h4_rsi,
            "h4_close_vs_ema50_pct": h4_close_vs_ema50,
            "daily_macd_hist": daily_macd_hist,
            "days_since_bull_flip": days_since_bull,
            "days_since_bear_flip": days_since_bear,
        }

    # --- BTC + breadth cache ---------------------------------------------

    def _refresh_caches_if_stale(self) -> None:
        import time
        now = time.time()
        if self._cache.refreshed_at and now - self._cache.refreshed_at < CACHE_TTL_SECONDS:
            return
        self._refresh_btc_cache()
        self._refresh_breadth_cache()
        self._cache.refreshed_at = now

    def _refresh_btc_cache(self) -> None:
        # Pull last 35 days of BTC 15m candles
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT timestamp, close
                    FROM asset_prices_15m
                    WHERE asset = 'BTCUSDT'
                    ORDER BY timestamp DESC
                    LIMIT :lim
                """),
                {"lim": 35 * 96},
            ).fetchall()
        if not rows:
            return
        df = pd.DataFrame(rows, columns=["timestamp", "close"]).sort_values("timestamp").reset_index(drop=True)
        ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
        c = df["close"].astype(float).to_numpy()
        s_close = pd.Series(c, index=ts_idx)

        h4 = s_close.resample("4h").last().dropna()
        h4_ema50 = _ema(h4, EMA_TREND_SPAN)
        btc_above_ema = (h4 > h4_ema50).astype(float)
        btc_above_ema_shifted = btc_above_ema.copy()
        btc_above_ema_shifted.index = btc_above_ema_shifted.index + pd.Timedelta("4h")

        btc_24h_ret = pd.Series(np.nan, index=ts_idx)
        if len(c) > 96:
            btc_24h_ret.iloc[96:] = c[96:] / c[:-96] - 1.0

        log_ret = np.concatenate([[np.nan], np.log(c[1:] / c[:-1])])
        s_logret = pd.Series(log_ret, index=ts_idx)
        rv = s_logret.rolling(96, min_periods=96).std()
        rv_mean = rv.rolling(2880, min_periods=720).mean()
        rv_std = rv.rolling(2880, min_periods=720).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            btc_rvz = (rv - rv_mean) / rv_std

        h4_macd_df = _macd(h4)
        btc_score = np.tanh(h4_macd_df["histogram"] / (0.005 * h4.replace(0, np.nan)))
        btc_score_shifted = btc_score.copy()
        btc_score_shifted.index = btc_score_shifted.index + pd.Timedelta("4h")

        self._cache.btc_above_ema_at = btc_above_ema_shifted
        self._cache.btc_24h_ret_at = btc_24h_ret
        self._cache.btc_rvz_at = btc_rvz
        self._cache.btc_score_at = btc_score_shifted

    def _refresh_breadth_cache(self) -> None:
        # 4h returns across the eligible universe — fetch last 35 days of all assets
        with self.engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT timestamp, asset, close
                    FROM asset_prices_15m
                    WHERE timestamp >= NOW() - INTERVAL '35 days'
                """),
            ).fetchall()
        if not rows:
            return
        df = pd.DataFrame(rows, columns=["timestamp", "asset", "close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        pivot = df.pivot_table(
            index="timestamp", columns="asset", values="close", aggfunc="last"
        )
        h4 = pivot.resample("4h").last()
        h4_ret = h4.pct_change()
        breadth_up = (h4_ret > 0).sum(axis=1) / h4_ret.notna().sum(axis=1)
        breadth_down = (h4_ret < 0).sum(axis=1) / h4_ret.notna().sum(axis=1)
        breadth_up = breadth_up.replace([np.inf, -np.inf], np.nan)
        breadth_down = breadth_down.replace([np.inf, -np.inf], np.nan)
        breadth_up.index = breadth_up.index + pd.Timedelta("4h")
        breadth_down.index = breadth_down.index + pd.Timedelta("4h")
        self._cache.breadth_up_at = breadth_up
        self._cache.breadth_down_at = breadth_down

    @staticmethod
    def _lookup_at(s: pd.Series | None, t: datetime) -> float | None:
        if s is None or len(s) == 0:
            return None
        idx = pd.Timestamp(t)
        # ffill: take the latest value at or before t
        sub = s[s.index <= idx]
        if len(sub) == 0:
            return None
        return float(sub.iloc[-1])
