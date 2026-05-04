"""v_new_1 Shadow Scoring Service — runs every 4h via cron.

Scores every coin in the current top-100 universe using the v_new_1_long and
v_new_1_short Platt-calibrated models and writes top-K signals to a daily
Parquet log. NO actual trading — pure observation mode.

Cron entry (install once):
    5 */4 * * * /opt/ai-finance/services/python/.venv/bin/python3 \
        /opt/ai-finance/services/python/scripts/v_new_1_shadow_score.py \
        >> /opt/ai-finance/services/python/results/v_new_1_shadow_log/cron.log 2>&1

Log output:
    /opt/ai-finance/services/python/results/v_new_1_shadow_log/v_new_1_signals_YYYY-MM-DD.parquet

USAGE (from /opt/ai-finance/):
    services/python/.venv/bin/python3 \
        services/python/scripts/v_new_1_shadow_score.py
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import sys
import time
import warnings
from datetime import UTC, datetime, timedelta
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sqlalchemy import create_engine, text


# ── CalibratedLGBM stub — must match the class used when model was pickled ────
# The joblib file was serialised with CalibratedLGBM defined in phase3_train.py.
# We must define an identical class here so pickle can reconstruct the object.
class CalibratedLGBM:
    """LGBMClassifier + IsotonicRegression calibrator (matches phase3_train.py)."""

    def __init__(self, base_model, calibrator: IsotonicRegression) -> None:
        self.base_model = base_model
        self.calibrator = calibrator

    def predict_proba(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        raw = self.base_model.predict_proba(X)[:, 1]
        cal = self.calibrator.predict(raw)
        return np.column_stack([1.0 - cal, cal])

    def predict(self, X: np.ndarray) -> np.ndarray:  # noqa: N803
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

warnings.filterwarnings("ignore", category=UserWarning)

# ── libomp workaround for macOS (no-op on Linux) ─────────────────────────────
_SKLEARN_DYLIBS = (
    pathlib.Path(__file__).resolve().parents[1]
    / ".venv"
    / "lib"
    / "python3.12"
    / "site-packages"
    / "sklearn"
    / ".dylibs"
)
if _SKLEARN_DYLIBS.is_dir():
    os.environ.setdefault("DYLD_LIBRARY_PATH", str(_SKLEARN_DYLIBS))

# ── paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PYTHON_ROOT = REPO_ROOT / "services" / "python"
MODELS_DIR = PYTHON_ROOT / "models"
LOG_DIR = PYTHON_ROOT / "results" / "v_new_1_shadow_log"

# ── policy (from Phase 4 best_policy.json) ───────────────────────────────────
TOP_K_LONG = 13
TOP_K_SHORT = 14
SCORE_THRESHOLD_LONG = 0.317
SCORE_THRESHOLD_SHORT = 0.468
CFGI_LONG_GATE = 60     # only take longs when CFGI > 60 (greed regime)
CFGI_SHORT_GATE = 50    # only take shorts when CFGI < 50 (fear regime)

# ── DB ────────────────────────────────────────────────────────────────────────
# On VPS: postgres runs as Docker container, exposed on localhost:5432
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://aifinance:6bnn9pcfRpf9xW-CO4wsMmVtyA094nbK1XUQ72Tyi1U@localhost:5432/aifinance",
)

# ── Universe ──────────────────────────────────────────────────────────────────
TOP_N_UNIVERSE = 100
TRAILING_30D_BARS = 180          # 30 days × 6 bars/day at 4h = 180 bars
MIN_VOLUME_USD = 500_000         # minimum daily USD volume to qualify

# ── Candles config ───────────────────────────────────────────────────────────
LOOKBACK_4H_BARS = 220           # ~37 days of 4h candles (needs ~180 for features + warmup)
MIN_BARS_REQUIRED = 80           # minimum bars after resampling to attempt scoring

# ── Resource limit ────────────────────────────────────────────────────────────
# Prevent this script from hogging all cores during its run
MAX_LGBM_THREADS = 2


def _ts(label: str = "") -> str:
    t = datetime.now(UTC).strftime("%H:%M:%S")
    return f"[{t}]{' ' + label if label else ''}"


# ── Feature helpers (from v_new_1_phase1_2_pipeline.py) ──────────────────────

def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(s: pd.Series, period: int = 14) -> pd.Series:
    delta = s.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def _macd_components(s: pd.Series, fast: int = 12, slow: int = 26, sig: int = 9):
    ema_f = s.ewm(span=fast, adjust=False).mean()
    ema_s = s.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    signal = macd.ewm(span=sig, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _atr14_series(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / 14, adjust=False).mean()


def compute_per_coin_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 44 v_new_1 features for a single coin's 4h candle history.

    Matches feature computation in v_new_1_phase1_2_pipeline.py exactly.
    Returns a DataFrame indexed by timestamp.

    STRICT NO-LOOKAHEAD: all features at T use only data ≤ T.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True))
    n = len(df)

    c = pd.Series(df["close"].astype(float).values, index=ts_idx)
    h = pd.Series(df["high"].astype(float).values, index=ts_idx)
    lo = pd.Series(df["low"].astype(float).values, index=ts_idx)
    o = pd.Series(df["open"].astype(float).values, index=ts_idx)
    v = pd.Series(df["volume"].astype(float).values, index=ts_idx)

    out: dict[str, pd.Series] = {}

    # ATR-based
    atr14 = _atr14_series(h, lo, c)
    out["atr14_pct_rank_90d"] = atr14.rolling(540, min_periods=50).rank(pct=True)

    # Volume z-score
    vol_24h = v.rolling(6, min_periods=6).sum()
    vol_mean_30d = vol_24h.rolling(180, min_periods=30).mean()
    vol_std_30d = vol_24h.rolling(180, min_periods=30).std()
    vol_z_24h = (vol_24h - vol_mean_30d) / vol_std_30d.replace(0.0, np.nan)
    out["vol_z_24h"] = vol_z_24h
    out["coin_24h_vol_zscore_30d"] = vol_z_24h.copy()

    # Return features
    out["coin_7d_return"] = c / c.shift(42) - 1.0
    out["coin_30d_return"] = c / c.shift(180) - 1.0

    # Range position
    h50 = h.rolling(50, min_periods=50).max()
    l50 = lo.rolling(50, min_periods=50).min()
    rng50_atr = atr14.replace(0.0, np.nan)
    out["close_to_high50_atr"] = (h50 - c) / rng50_atr
    out["close_to_low50_atr"] = (c - l50) / rng50_atr

    # Bar shape (prior bar)
    c_prev = c.shift(1)
    o_prev = o.shift(1)
    h_prev = h.shift(1)
    lo_prev = lo.shift(1)
    bar_range = h_prev - lo_prev
    bar_range_safe = bar_range.replace(0.0, np.nan)
    out["bar4h_close_pos_in_range"] = (c_prev - lo_prev) / bar_range_safe
    out["bar4h_body_pct"] = (c_prev - o_prev).abs() / bar_range_safe
    out["bar4h_upper_wick_pct"] = (
        h_prev - pd.concat([c_prev, o_prev], axis=1).max(axis=1)
    ) / bar_range_safe

    # 4h MACD + RSI + EMA50
    macd4h, sig4h, hist4h = _macd_components(c, 12, 26, 9)
    out["h4_macd_hist"] = hist4h.shift(1)
    out["h4_macd_macd"] = macd4h.shift(1)
    rsi4h = _rsi(c, 14)
    ema50_4h = _ema(c, 50)
    out["h4_rsi"] = rsi4h.shift(1)
    out["h4_close_vs_ema50_pct"] = (
        (c.shift(1) - ema50_4h.shift(1)) / ema50_4h.shift(1).replace(0.0, np.nan)
    )

    # Daily features (+1d shift for no-lookahead)
    c_daily = c.resample("1D").last().dropna()
    if len(c_daily) >= 30:
        d_macd, d_sig, d_hist = _macd_components(c_daily)
        d_hist_shifted = d_hist.copy()
        d_hist_shifted.index = d_hist_shifted.index + pd.Timedelta("1D")
        out["daily_macd_hist"] = d_hist_shifted.reindex(ts_idx, method="ffill")

        daily_bull = (d_hist > 0) & (d_macd > d_sig)
        daily_bear = (d_hist < 0) & (d_macd < d_sig)
        dsb = np.full(len(c_daily), np.nan)
        dsbe = np.full(len(c_daily), np.nan)
        last_bull = -1; last_bear = -1
        bull_arr = daily_bull.to_numpy()
        bear_arr = daily_bear.to_numpy()
        for di in range(len(bull_arr)):
            if di > 0 and bull_arr[di] and not bull_arr[di - 1]:
                last_bull = di
            if di > 0 and bear_arr[di] and not bear_arr[di - 1]:
                last_bear = di
            if last_bull >= 0:
                dsb[di] = di - last_bull
            if last_bear >= 0:
                dsbe[di] = di - last_bear

        s_dsb = pd.Series(dsb, index=c_daily.index)
        s_dsbe = pd.Series(dsbe, index=c_daily.index)
        s_dsb.index = s_dsb.index + pd.Timedelta("1D")
        s_dsbe.index = s_dsbe.index + pd.Timedelta("1D")
        out["days_since_bull_flip"] = s_dsb.reindex(ts_idx, method="ffill")
        out["days_since_bear_flip"] = s_dsbe.reindex(ts_idx, method="ffill")
    else:
        out["daily_macd_hist"] = pd.Series(np.nan, index=ts_idx)
        out["days_since_bull_flip"] = pd.Series(np.nan, index=ts_idx)
        out["days_since_bear_flip"] = pd.Series(np.nan, index=ts_idx)

    # Time features
    hour = ts_idx.hour
    dow = ts_idx.dayofweek
    out["hour_sin"] = pd.Series(np.sin(2 * math.pi * hour / 24), index=ts_idx)
    out["hour_cos"] = pd.Series(np.cos(2 * math.pi * hour / 24), index=ts_idx)
    out["dow_sin"] = pd.Series(np.sin(2 * math.pi * dow / 7), index=ts_idx)
    out["dow_cos"] = pd.Series(np.cos(2 * math.pi * dow / 7), index=ts_idx)

    # Bollinger Bands
    bb_mid = c.shift(1).rolling(20, min_periods=20).mean()
    bb_std = c.shift(1).rolling(20, min_periods=20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_rng = (bb_upper - bb_lower).replace(0.0, np.nan)
    out["bb_pct_b"] = (c.shift(1) - bb_lower) / bb_rng
    out["bb_bandwidth"] = bb_rng / bb_mid.replace(0.0, np.nan)

    # KDJ (9-period)
    roll_lo = lo.shift(1).rolling(9, min_periods=9).min()
    roll_hi = h.shift(1).rolling(9, min_periods=9).max()
    span_kd = (roll_hi - roll_lo).replace(0.0, np.nan)
    rsv = 100.0 * (c.shift(1) - roll_lo) / span_kd
    kdj_k = rsv.rolling(3, min_periods=3).mean()
    kdj_d = kdj_k.rolling(3, min_periods=3).mean()
    out["kdj_k"] = kdj_k
    out["kdj_j"] = 3.0 * kdj_k - 2.0 * kdj_d

    # Fib position
    h50_shifted = h.rolling(50, min_periods=50).max().shift(1)
    l50_shifted = lo.rolling(50, min_periods=50).min().shift(1)
    rng50_s = (h50_shifted - l50_shifted).replace(0.0, np.nan)
    out["fib_pos_50"] = (c.shift(1) - l50_shifted) / rng50_s

    # Price vs EMA9
    ema9 = _ema(c, 9)
    c_s1 = c.shift(1)
    out["price_vs_ema9_pct"] = (c_s1 - ema9.shift(1)) / c_s1.replace(0.0, np.nan)

    # RSI deltas
    out["rsi14_delta_1bar"] = rsi4h.diff(1)
    out["rsi14_delta_4bar"] = rsi4h.diff(4)

    # MACD histogram momentum
    out["macd_hist_momentum"] = hist4h.diff(2)
    out["macd_signal_spread_norm"] = hist4h.shift(1) / c_s1.replace(0.0, np.nan)

    # Parkinson volatility
    log_hl = np.log((h / lo).replace(0.0, np.nan))
    parkinson_raw = log_hl.pow(2).rolling(24, min_periods=24).mean() / (4.0 * math.log(2.0))
    out["parkinson_vol"] = parkinson_raw.apply(
        lambda x: float(np.sqrt(x)) if np.isfinite(x) and x >= 0 else np.nan
    )

    # OBV slope
    obv = (np.sign(c.diff().fillna(0.0)) * v).cumsum()
    obv_slope = obv.rolling(24).apply(
        lambda x: float(np.polyfit(range(len(x)), x, 1)[0]) if len(x) == 24 else np.nan,
        raw=True,
    )
    mean_vol_24 = v.rolling(24, min_periods=24).mean().replace(0.0, np.nan)
    out["obv_slope_norm"] = obv_slope / mean_vol_24

    # CVD slope
    sign_diff = np.sign(c.diff().fillna(0.0))
    cvd = (sign_diff * v).cumsum()
    cvd_4 = cvd.shift(4)
    vol_abs_sum4 = v.rolling(4, min_periods=4).sum()
    out["cvd_slope_1h_norm"] = (cvd - cvd_4) / vol_abs_sum4.replace(0.0, np.nan)

    # days_since_last_big_move
    big_up = (c / c.shift(42) - 1.0) >= 0.10
    big_dn = (c / c.shift(42) - 1.0) <= -0.10
    days_bm_long = np.full(n, np.nan)
    days_bm_short = np.full(n, np.nan)
    last_big_up = -1; last_big_dn = -1
    big_up_arr = big_up.to_numpy()
    big_dn_arr = big_dn.to_numpy()
    for bi in range(n):
        if big_up_arr[bi]:
            last_big_up = bi
        if big_dn_arr[bi]:
            last_big_dn = bi
        if last_big_up >= 0:
            days_bm_long[bi] = (bi - last_big_up) * 4 / 24
        if last_big_dn >= 0:
            days_bm_short[bi] = (bi - last_big_dn) * 4 / 24
    out["days_since_last_big_move_long"] = pd.Series(days_bm_long, index=ts_idx)
    out["days_since_last_big_move_short"] = pd.Series(days_bm_short, index=ts_idx)

    feat_df = pd.DataFrame(out, index=ts_idx)
    feat_df.index.name = "timestamp"
    feat_df["close"] = c.values
    return feat_df


def compute_btc_features(btc_df: pd.DataFrame) -> pd.DataFrame:
    """Compute BTC cross-asset features. All shifted +1 bar (no-lookahead)."""
    btc_df = btc_df.sort_values("timestamp").reset_index(drop=True)
    ts_idx = pd.DatetimeIndex(pd.to_datetime(btc_df["timestamp"], utc=True))
    c = pd.Series(btc_df["close"].astype(float).values, index=ts_idx)

    ema50 = _ema(c, 50)
    btc_above = (c > ema50).astype(float)
    btc_24h_ret = c / c.shift(6) - 1.0

    log_ret = np.log(c / c.shift(1))
    rv_24h = log_ret.rolling(6, min_periods=6).std()
    rv_mean = rv_24h.rolling(180, min_periods=30).mean()
    rv_std = rv_24h.rolling(180, min_periods=30).std()
    btc_vol_z = (rv_24h - rv_mean) / rv_std.replace(0.0, np.nan)

    _, _, hist4h = _macd_components(c)
    btc_score = np.tanh(hist4h / (0.005 * c.replace(0.0, np.nan)))

    return pd.DataFrame({
        "btc_above_4h_ema50": btc_above.shift(1),
        "btc_24h_return": btc_24h_ret.shift(1),
        "btc_realized_vol_z": btc_vol_z.shift(1),
        "btc_score": btc_score.shift(1),
    }, index=ts_idx)


def compute_breadth(close_pivot: pd.DataFrame) -> pd.DataFrame:
    """Compute breadth_up / breadth_down from close pivot. Shift +1 bar."""
    pct_chg = close_pivot.pct_change()
    n_valid = pct_chg.notna().sum(axis=1)
    breadth_up = (pct_chg > 0).sum(axis=1) / n_valid
    breadth_down = (pct_chg < 0).sum(axis=1) / n_valid
    return pd.DataFrame({
        "breadth_up": breadth_up.shift(1),
        "breadth_down": breadth_down.shift(1),
    })


# ── CFGI fetch (alternative.me API + DB fallback) ────────────────────────────

def get_cfgi_value(engine) -> float:
    """Fetch current CFGI value. Try alternative.me API first; fall back to DB."""
    # Try alternative.me API
    try:
        req = Request(
            "https://api.alternative.me/fng/?limit=1&format=json",
            headers={"User-Agent": "ai-finance-shadow-scorer/1.0"},
        )
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        cfgi = float(data["data"][0]["value"])
        print(f"{_ts('CFGI')} API value: {cfgi:.0f}")
        return cfgi
    except (HTTPError, URLError, KeyError, ValueError, TimeoutError) as e:
        print(f"{_ts('CFGI')} API failed ({e}), falling back to DB ...")

    # Fall back to DB (most recent value in fear_greed_index table)
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT value FROM fear_greed_index ORDER BY date DESC LIMIT 1")
            ).fetchone()
        if row:
            cfgi = float(row[0])
            print(f"{_ts('CFGI')} DB value: {cfgi:.0f}")
            return cfgi
    except Exception as e:
        print(f"{_ts('CFGI')} DB fallback failed ({e})")

    # Last resort: neutral 50
    print(f"{_ts('CFGI')} Using neutral fallback: 50")
    return 50.0


# ── Universe construction ─────────────────────────────────────────────────────

def get_top100_universe(engine) -> list[str]:
    """Determine top-100 coins by trailing 30-day USD volume from Postgres 15m candles.

    Returns list of symbol strings (e.g. ['BTCUSDT', 'ETHUSDT', ...]).
    """
    lookback_start = datetime.now(UTC) - timedelta(days=35)

    with engine.connect() as conn:
        df = pd.read_sql(
            text(
                """
                SELECT asset AS symbol,
                       SUM(quote_volume) AS total_usd_volume
                FROM asset_prices_15m
                WHERE timestamp >= :start
                GROUP BY asset
                HAVING SUM(quote_volume) >= :min_vol
                ORDER BY total_usd_volume DESC
                LIMIT :top_n
                """
            ),
            conn,
            params={
                "start": lookback_start,
                "min_vol": MIN_VOLUME_USD * 24 * 4 * 30,  # 30-day minimum total
                "top_n": TOP_N_UNIVERSE,
            },
        )

    symbols = df["symbol"].tolist()
    print(f"{_ts('Universe')} Top-{len(symbols)} coins by 30d volume")
    if symbols:
        print(f"  Top-5: {symbols[:5]}")
    return symbols


def fetch_4h_candles(engine, symbol: str, n_bars: int = LOOKBACK_4H_BARS) -> pd.DataFrame | None:
    """Fetch last n_bars×4h worth of 15m candles from Postgres and resample to 4h.

    Returns DataFrame with columns: timestamp, open, high, low, close, volume, quote_volume
    or None if insufficient data.
    """
    # Fetch enough 15m bars (n_bars * 16 = 15m bars for n_bars 4h candles)
    lookback_start = datetime.now(UTC) - timedelta(hours=n_bars * 4 + 8)

    try:
        with engine.connect() as conn:
            df = pd.read_sql(
                text(
                    """
                    SELECT timestamp, open, high, low, close, volume, quote_volume
                    FROM asset_prices_15m
                    WHERE asset = :symbol AND timestamp >= :start
                    ORDER BY timestamp ASC
                    """
                ),
                conn,
                params={"symbol": symbol, "start": lookback_start},
            )
    except Exception as e:
        print(f"  [WARN] {symbol}: DB fetch failed: {e}")
        return None

    if df.empty or len(df) < 16:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()

    # Resample 15m → 4h
    df_4h = df.resample("4h").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        quote_volume=("quote_volume", "sum"),
    ).dropna(subset=["close"])

    if len(df_4h) < MIN_BARS_REQUIRED:
        return None

    df_4h = df_4h.reset_index()
    df_4h["symbol"] = symbol
    return df_4h


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models() -> dict:
    """Load base LightGBM models, Platt calibrators, and feature name lists."""
    models = {}
    for direction in ["long", "short"]:
        dir_label = f"v_new_1_{direction}"
        model_dir = MODELS_DIR / dir_label

        # Base model
        base_path = model_dir / f"{dir_label}.joblib"
        if not base_path.exists():
            raise FileNotFoundError(f"Model not found: {base_path}")
        calibrated = joblib.load(base_path)
        base_model = calibrated.base_model

        # Platt calibrator
        platt_path = model_dir / "platt_calibrator.joblib"
        if not platt_path.exists():
            raise FileNotFoundError(
                f"Platt calibrator not found: {platt_path}\n"
                f"Run v_new_1_platt_fit.py first."
            )
        platt: LogisticRegression = joblib.load(platt_path)

        # Feature names from meta
        meta_path = model_dir / f"{dir_label}_meta.json"
        with open(meta_path) as f:
            meta = json.load(f)
        feature_names = [
            f for f in meta["feature_names"]
            if f != "signals_same_15m_same_detector"
        ]

        # Cap LightGBM threads to avoid hogging VPS cores
        try:
            base_model.set_params(num_threads=MAX_LGBM_THREADS, n_jobs=MAX_LGBM_THREADS)
        except Exception:
            pass

        models[direction] = {
            "base_model": base_model,
            "platt": platt,
            "feature_names": feature_names,
        }
        print(f"{_ts('Load')} {dir_label}: {len(feature_names)} features")

    return models


# ── Score one coin at the latest 4h bar ──────────────────────────────────────

def score_coin(
    symbol: str,
    df_4h: pd.DataFrame,
    btc_feats: pd.DataFrame,
    breadth_df: pd.DataFrame,
    cfgi_value: float,
    vol_rank: float,
    models: dict,
) -> dict | None:
    """Compute features and score a single coin at the latest bar.

    Returns a dict with raw + platt scores for both directions, or None on failure.
    """
    try:
        coin_feats = compute_per_coin_features(df_4h)
    except Exception as e:
        print(f"  [WARN] {symbol}: feature computation failed: {e}")
        return None

    if len(coin_feats) == 0:
        return None

    latest_ts = coin_feats.index[-1]
    row = coin_feats.iloc[[-1]].copy()  # latest bar only

    # Join BTC features (forward-fill)
    for col in ["btc_above_4h_ema50", "btc_24h_return", "btc_realized_vol_z", "btc_score"]:
        if col in btc_feats.columns:
            row[col] = btc_feats[col].reindex([latest_ts], method="ffill").values[0]
        else:
            row[col] = np.nan

    # Join breadth
    for col in ["breadth_up", "breadth_down"]:
        if col in breadth_df.columns:
            row[col] = breadth_df[col].reindex([latest_ts], method="ffill").values[0]
        else:
            row[col] = np.nan

    # CFGI and vol_rank (cross-sectional)
    row["cfgi_value"] = cfgi_value
    row["vol_rank_in_top100"] = vol_rank

    close_at_signal = float(row["close"].values[0]) if "close" in row.columns else np.nan

    result = {
        "symbol": symbol,
        "timestamp": latest_ts,
        "market_close_at_signal": close_at_signal,
        "cfgi_value": cfgi_value,
    }

    for direction in ["long", "short"]:
        m = models[direction]
        feat_names = m["feature_names"]
        available = [f for f in feat_names if f in row.columns]
        X = np.nan_to_num(row[available].values, nan=0.0)  # noqa: N806

        raw_prob = float(m["base_model"].predict_proba(X)[:, 1][0])
        platt_prob = float(m["platt"].predict_proba(np.array([[raw_prob]]))[:, 1][0])

        result[f"score_raw_{direction}"] = raw_prob
        result[f"score_platt_{direction}"] = platt_prob

    return result


# ── Main scoring loop ─────────────────────────────────────────────────────────

def run_shadow_scoring() -> None:
    t_start = time.monotonic()
    run_ts = datetime.now(UTC)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    date_str = run_ts.strftime("%Y-%m-%d")
    log_path = LOG_DIR / f"v_new_1_signals_{date_str}.parquet"

    print(f"\n{'='*65}")
    print(f"{_ts()} v_new_1 Shadow Scoring — {run_ts.isoformat()}")
    print(f"{'='*65}")
    print(f"  TOP_K_LONG={TOP_K_LONG}  TOP_K_SHORT={TOP_K_SHORT}")
    print(f"  SCORE_THRESHOLD_LONG={SCORE_THRESHOLD_LONG:.3f}  SHORT={SCORE_THRESHOLD_SHORT:.3f}")
    print(f"  CFGI_LONG_GATE={CFGI_LONG_GATE} (long only when CFGI > gate)")
    print(f"  CFGI_SHORT_GATE={CFGI_SHORT_GATE} (short only when CFGI < gate)")
    print(f"  Log: {log_path}")

    # ── Connect to DB ─────────────────────────────────────────────────────────
    engine = create_engine(DB_URL, pool_pre_ping=True)

    # ── Load models ───────────────────────────────────────────────────────────
    print(f"\n{_ts()} Loading models ...")
    models = load_models()

    # ── Get CFGI ─────────────────────────────────────────────────────────────
    cfgi = get_cfgi_value(engine)
    longs_allowed = cfgi > CFGI_LONG_GATE
    shorts_allowed = cfgi < CFGI_SHORT_GATE
    print(f"{_ts('CFGI')} value={cfgi:.0f}  longs_allowed={longs_allowed}  shorts_allowed={shorts_allowed}")

    # ── Get universe ──────────────────────────────────────────────────────────
    print(f"\n{_ts()} Determining top-{TOP_N_UNIVERSE} universe ...")
    universe = get_top100_universe(engine)
    if len(universe) < 10:
        print(f"{_ts()} ERROR: Universe too small ({len(universe)} coins). Check DB connectivity.")
        return

    # ── Fetch BTC candles once for cross-asset features ───────────────────────
    btc_symbol = "BTCUSDT"
    if btc_symbol not in universe:
        universe_with_btc = [btc_symbol] + universe
    else:
        universe_with_btc = universe

    print(f"\n{_ts()} Fetching BTC candles for cross-asset features ...")
    btc_df = fetch_4h_candles(engine, btc_symbol, n_bars=LOOKBACK_4H_BARS)
    if btc_df is None:
        print(f"{_ts()} ERROR: Could not fetch BTC candles — aborting.")
        return
    btc_feats = compute_btc_features(btc_df)
    print(f"  BTC 4h bars: {len(btc_df)}  last: {btc_feats.index[-1]}")

    # ── Fetch all coins and build close pivot for breadth ─────────────────────
    print(f"\n{_ts()} Fetching candles for {len(universe)} coins ...")
    coin_candles: dict[str, pd.DataFrame] = {}
    failed_symbols = []

    for i, symbol in enumerate(universe):
        df_4h = fetch_4h_candles(engine, symbol, n_bars=LOOKBACK_4H_BARS)
        if df_4h is None:
            failed_symbols.append(symbol)
            continue
        coin_candles[symbol] = df_4h
        if (i + 1) % 25 == 0:
            print(f"  Fetched {i+1}/{len(universe)} coins ({len(coin_candles)} OK, {len(failed_symbols)} failed)")

    print(f"{_ts()} Candles: {len(coin_candles)} OK, {len(failed_symbols)} failed")
    if failed_symbols:
        print(f"  Failed: {failed_symbols[:10]}{'...' if len(failed_symbols) > 10 else ''}")

    # ── Build breadth pivot ───────────────────────────────────────────────────
    print(f"{_ts()} Computing breadth ...")
    close_frames = []
    for symbol, df in coin_candles.items():
        df_ts = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)))["close"]
        close_frames.append(df_ts.rename(symbol))

    if close_frames:
        close_pivot = pd.concat(close_frames, axis=1)
        breadth_df = compute_breadth(close_pivot)
    else:
        breadth_df = pd.DataFrame()

    # ── Score each coin ───────────────────────────────────────────────────────
    print(f"\n{_ts()} Scoring {len(coin_candles)} coins ...")
    raw_scores: list[dict] = []

    for rank_i, (symbol, df_4h) in enumerate(coin_candles.items()):
        vol_rank = float(rank_i + 1)  # approximate rank from universe order
        result = score_coin(
            symbol=symbol,
            df_4h=df_4h,
            btc_feats=btc_feats,
            breadth_df=breadth_df,
            cfgi_value=cfgi,
            vol_rank=vol_rank,
            models=models,
        )
        if result is not None:
            raw_scores.append(result)

    print(f"{_ts()} Scored {len(raw_scores)} / {len(coin_candles)} coins successfully")

    if not raw_scores:
        print(f"{_ts()} ERROR: No scores computed — aborting.")
        return

    scores_df = pd.DataFrame(raw_scores)

    # ── Apply filters and build signal rows ──────────────────────────────────
    signal_rows: list[dict] = []

    for direction in ["long", "short"]:
        score_col_raw = f"score_raw_{direction}"
        score_col_platt = f"score_platt_{direction}"
        threshold = SCORE_THRESHOLD_LONG if direction == "long" else SCORE_THRESHOLD_SHORT
        top_k = TOP_K_LONG if direction == "long" else TOP_K_SHORT
        cfgi_blocked = (direction == "long" and not longs_allowed) or \
                       (direction == "short" and not shorts_allowed)

        # Sort by Platt score descending for ranking
        dir_df = scores_df[[
            "symbol", "timestamp", "market_close_at_signal", "cfgi_value",
            score_col_raw, score_col_platt,
        ]].copy().sort_values(score_col_platt, ascending=False).reset_index(drop=True)

        for i, row in dir_df.iterrows():
            platt_score = float(row[score_col_platt])
            raw_score = float(row[score_col_raw])
            rank = int(i) + 1  # 1-indexed rank within direction

            if cfgi_blocked:
                action = f"skipped_cfgi_{'long' if direction == 'long' else 'short'}"
            elif platt_score < threshold:
                action = "skipped_score"
            elif rank > top_k:
                action = "skipped_topk"
            else:
                action = "taken"

            signal_rows.append({
                "timestamp": run_ts,
                "symbol": str(row["symbol"]),
                "direction": direction,
                "score_raw": round(raw_score, 6),
                "score_platt": round(platt_score, 6),
                "rank": rank,
                "cfgi_value": float(row["cfgi_value"]),
                "action": action,
                "market_close_at_signal": float(row["market_close_at_signal"])
                if pd.notna(row["market_close_at_signal"]) else None,
            })

    signals_df = pd.DataFrame(signal_rows)

    # ── Append to daily log parquet ───────────────────────────────────────────
    if log_path.exists():
        existing = pd.read_parquet(log_path)
        # Deduplicate: drop rows with same (timestamp, symbol, direction)
        combined = pd.concat([existing, signals_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp", "symbol", "direction"], keep="last")
        combined.to_parquet(log_path, index=False)
        print(f"\n{_ts()} Appended to {log_path} ({len(combined)} total rows today)")
    else:
        signals_df.to_parquet(log_path, index=False)
        print(f"\n{_ts()} Created {log_path} ({len(signals_df)} rows)")

    # ── Summary ───────────────────────────────────────────────────────────────
    taken = signals_df[signals_df["action"] == "taken"]
    taken_long = taken[taken["direction"] == "long"]
    taken_short = taken[taken["direction"] == "short"]

    top1_long = taken_long.sort_values("score_platt", ascending=False).head(1)
    top1_short = taken_short.sort_values("score_platt", ascending=False).head(1)

    elapsed = time.monotonic() - t_start
    print(f"\n{'='*65}")
    print(f"SHADOW SCORING SUMMARY — {run_ts.strftime('%Y-%m-%d %H:%M UTC')}")
    print(f"{'='*65}")
    print(f"  CFGI: {cfgi:.0f}  |  longs_allowed={longs_allowed}  shorts_allowed={shorts_allowed}")
    print(f"  Universe: {len(universe)} coins | Scored: {len(raw_scores)}")
    print(f"  Longs TAKEN:  {len(taken_long):2d} / {TOP_K_LONG}  "
          f"top-1: {top1_long['symbol'].values[0] if len(top1_long) else 'none'} "
          f"({top1_long['score_platt'].values[0]:.3f} if len(top1_long) else '')".rstrip("'"))
    print(f"  Shorts TAKEN: {len(taken_short):2d} / {TOP_K_SHORT}  "
          f"top-1: {top1_short['symbol'].values[0] if len(top1_short) else 'none'} "
          f"({top1_short['score_platt'].values[0]:.3f} if len(top1_short) else '')".rstrip("'"))
    print(f"  Runtime: {elapsed:.1f}s")
    print(f"  Log: {log_path}")

    # Show top-5 taken signals
    if len(taken) > 0:
        print(f"\n  Top taken signals:")
        top5 = taken.sort_values("score_platt", ascending=False).head(5)
        for _, r in top5.iterrows():
            print(f"    [{r['direction'].upper():5s}] {r['symbol']:12s} "
                  f"platt={r['score_platt']:.3f}  raw={r['score_raw']:.3f}  "
                  f"rank={r['rank']}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_shadow_scoring()
