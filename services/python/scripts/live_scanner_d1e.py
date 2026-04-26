"""Live scanner — D1e detector (Phase 17 locked) with v4 classifier scoring.

Polls 15min candles for top-N USDT-perp pairs, detects D1e SHORT signals,
attaches the 16 ML features, scores with v4 classifier, and writes scored
signals to `scanner_signals_d1e`. The paper executor reads this table and
routes signals to the four S×E paper accounts (S1×E1, S1×E2, S4×E1, S4×E2).

D1e detector (SHORT-only, no long side — long failed ship gate):
    (rolling_high_96 - close) / ATR_14 >= 2.5
    AND vol / vol_MA_20 >= 2.0
    AND close < open                                  (red bar)
    AND universe_down_breadth_4h >= 0.6
    AND btc_trend_score < 0                           (sign-locked)

v4 classifier (16 features + is_short, threshold 0.5010 = TRAIN q50):
    score >= 0.5010 -> cls_kept=True (paper executor gates on this)
    Below threshold rows are still written (cls_kept=False) for diagnostics.

Cooldown per symbol: 96 bars (24h), matching backtest.

Architectural notes:
  - The detector + features use FeatureContext from
    `src.ml.bigmover_combined.features` to keep live math bit-for-bit identical
    to the research pipeline (no separate live re-implementation).
  - FeatureContext is rebuilt on every candle refresh (every 15 min), since
    bars only change at 15m boundaries. Per-asset rolling indicators are
    cached lazily inside FeatureContext.
  - Cross-sectional breadth_4h panel is built once per refresh; the detector
    looks up the value at the current bar timestamp.
  - Signals are emitted with `cls_kept` as an advisory flag; paper executor
    can apply its own per-account gating on top.

Usage from services/python/:
    python scripts/live_scanner_d1e.py
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from src import db_adapters  # noqa: F401  registers numpy->psycopg2 adapters
from src.ml.bigmover_combined.features import FEATURE_NAMES, FeatureContext

# ── Config ───────────────────────────────────────────────────────────
DB_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:MySQL100%25@localhost:5432/market",
)
GATEIO_BASE = "https://api.gateio.ws/api/v4"

SCANNER_TOP_N = int(os.environ.get("SCANNER_TOP_N", "100"))
SCAN_INTERVAL = 60
CANDLE_INTERVAL = 900  # 15 min — bar boundary
HISTORY_BARS = 2100  # ~22 days for breadth + indicator warmup
MIN_VOLUME_USD = 500_000

# D1e detector params (locked Phase 17.A)
ATR_DROP_MULT = 2.5
VOL_RATIO_MIN = 2.0
BREADTH_MIN = 0.6
PRICE_LOOKBACK_96 = 96
ATR_PERIOD = 14
COOLDOWN_SECONDS = 96 * 900  # 24h per symbol

# v4 classifier paths
MODEL_PATH = Path("models/d1e_short_v4.joblib")
META_PATH = Path("models/d1e_short_v4_meta.json")

Path("logs").mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/live_scanner_d1e.log", mode="a"),
    ],
)
logger = logging.getLogger("scanner_d1e")


# ── Gate.io tickers ──────────────────────────────────────────────────


def fetch_json(url: str, retries: int = 2):
    for attempt in range(retries):
        try:
            req = Request(url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"})
            resp = urlopen(req, timeout=10)
            return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_all_tickers() -> dict[str, dict]:
    data = fetch_json(f"{GATEIO_BASE}/futures/usdt/tickers")
    if not data:
        return {}
    out: dict[str, dict] = {}
    for t in data:
        pair = t.get("contract", "")
        if not pair.endswith("_USDT"):
            continue
        try:
            out[pair] = {
                "last": float(t.get("last", 0)),
                "quote_volume": float(t.get("volume_24h_quote", 0)),
            }
        except (ValueError, TypeError):
            continue
    return out


# ── Candle fetch from DB ─────────────────────────────────────────────

_db_engine = None


def get_db_engine():
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(DB_URL)
    return _db_engine


def fetch_candles(pair: str, limit: int = HISTORY_BARS) -> pd.DataFrame | None:
    asset = pair.replace("_", "")
    eng = get_db_engine()
    with eng.connect() as conn:
        rows = conn.execute(
            text("""
                SELECT timestamp, open, high, low, close, volume
                FROM asset_prices_15m WHERE asset = :a
                ORDER BY timestamp DESC LIMIT :lim
            """),
            {"a": asset, "lim": limit},
        ).fetchall()
    if not rows or len(rows) < PRICE_LOOKBACK_96 + ATR_PERIOD + 5:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["asset"] = asset
    return df.sort_values("timestamp").reset_index(drop=True)


def build_universe_candles(candle_cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-pair caches into a single multi-asset DataFrame for FeatureContext."""
    frames = []
    for df in candle_cache.values():
        if df is None or df.empty:
            continue
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["timestamp", "asset", "open", "high", "low", "close", "volume"])
    return pd.concat(frames, ignore_index=True)


# ── DB schema ────────────────────────────────────────────────────────


def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(
            text("""
                CREATE TABLE IF NOT EXISTS scanner_signals_d1e (
                    id SERIAL PRIMARY KEY,
                    symbol VARCHAR(20) NOT NULL,
                    direction VARCHAR(5) NOT NULL DEFAULT 'short',
                    signal_time TIMESTAMPTZ NOT NULL,
                    status VARCHAR(10) DEFAULT 'active',
                    -- detector internals
                    drop_atr_norm DOUBLE PRECISION,
                    vol_ratio DOUBLE PRECISION,
                    breadth_4h DOUBLE PRECISION,
                    rolling_high_96 DOUBLE PRECISION,
                    atr14_at_entry DOUBLE PRECISION,
                    btc_score DOUBLE PRECISION,
                    -- 16 features stored as JSONB (FeatureContext output)
                    features JSONB,
                    -- v4 classifier
                    cls_score DOUBLE PRECISION,
                    cls_kept BOOLEAN,
                    -- entry context
                    entry_price DOUBLE PRECISION,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE (symbol, direction, signal_time)
                )
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_d1e_signal_time
                ON scanner_signals_d1e (signal_time DESC)
            """)
        )
        conn.execute(
            text("""
                CREATE INDEX IF NOT EXISTS idx_d1e_active
                ON scanner_signals_d1e (status, signal_time)
                WHERE status = 'active'
            """)
        )


def write_signal(
    engine,
    *,
    symbol: str,
    direction: str,
    signal_time: datetime,
    drop_atr_norm: float,
    vol_ratio: float,
    breadth_4h: float,
    rolling_high_96: float,
    atr14_at_entry: float,
    btc_score: float,
    features: dict,
    cls_score: float,
    cls_kept: bool,
    entry_price: float,
) -> bool:
    try:
        with engine.begin() as conn:
            conn.execute(
                text("""
                    INSERT INTO scanner_signals_d1e
                    (symbol, direction, signal_time, status,
                     drop_atr_norm, vol_ratio, breadth_4h, rolling_high_96,
                     atr14_at_entry, btc_score, features, cls_score, cls_kept,
                     entry_price)
                    VALUES (:symbol, :direction, :sig_t, 'active',
                            :drop, :vr, :br, :rh,
                            :atr, :bs, CAST(:feats AS JSONB), :cs, :kept,
                            :ep)
                    ON CONFLICT (symbol, direction, signal_time) DO NOTHING
                """),
                {
                    "symbol": symbol, "direction": direction, "sig_t": signal_time,
                    "drop": drop_atr_norm, "vr": vol_ratio, "br": breadth_4h,
                    "rh": rolling_high_96, "atr": atr14_at_entry, "bs": btc_score,
                    "feats": json.dumps(features, default=str),
                    "cs": cls_score, "kept": cls_kept, "ep": entry_price,
                },
            )
        return True
    except Exception as e:
        logger.error(f"Failed to insert D1e signal for {symbol}: {e}")
        return False


# ── D1e detector ─────────────────────────────────────────────────────


def detect_d1e_short(
    ctx: FeatureContext,
    symbol: str,
    df: pd.DataFrame,
    entry_time: pd.Timestamp | None = None,
) -> dict | None:
    """Evaluate D1e SHORT detector at `entry_time` (defaults to latest bar of df).

    `df` is only used as a sanity check on history length — the actual feature
    + indicator lookups go through `ctx` (FeatureContext) which holds the
    canonical per-asset view + cross-sectional panels. Passing `entry_time`
    explicitly lets this function be called against historical bars (e.g. the
    smoke test); the live scanner passes the latest closed bar.

    Returns a dict with detector internals + features + entry context, or None
    if any gate fails.
    """
    if len(df) < PRICE_LOOKBACK_96 + ATR_PERIOD + 5:
        return None
    view = ctx._asset_view(symbol)  # re-uses cached rolling indicators
    if view is None or len(view.close) == 0:
        return None

    if entry_time is None:
        # Live mode: use the most recent bar in the cached df
        entry_time = pd.Timestamp(df["timestamp"].iloc[-1])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    else:
        entry_time = entry_time.tz_convert("UTC")
    entry_naive = entry_time.tz_convert(None)
    i = int(np.searchsorted(view.ts, np.datetime64(entry_naive), side="right")) - 1
    if i < PRICE_LOOKBACK_96 + ATR_PERIOD:
        return None

    c = float(view.close[i])
    o = float(view.open[i])
    if not (np.isfinite(c) and np.isfinite(o)):
        return None
    # Red bar: close < open (SHORT-only entry)
    if c >= o:
        return None

    # ATR-normalized drop from rolling 96-high
    if i >= len(view.rolling_high96) or not np.isfinite(view.rolling_high96[i]):
        return None
    rh = float(view.rolling_high96[i])
    if i >= len(view.atr14) or not np.isfinite(view.atr14[i]) or view.atr14[i] <= 0:
        return None
    atr = float(view.atr14[i])
    drop_atr_norm = (rh - c) / atr
    if drop_atr_norm < ATR_DROP_MULT:
        return None

    # Volume gate
    if i >= len(view.vol_ma20) or not np.isfinite(view.vol_ma20[i]) or view.vol_ma20[i] <= 0:
        return None
    vol_ratio = float(view.volume[i] / view.vol_ma20[i])
    if vol_ratio < VOL_RATIO_MIN:
        return None

    entry_ts = entry_time

    # Cross-sectional breadth: fraction of universe with negative 4h ret at this bar.
    # Panel index is TZ-aware UTC (matches FeatureContext.build_for usage), so we
    # look up with the TZ-aware entry_ts directly.
    breadth_4h = float("nan")
    if ctx._ret_4h_panel is not None:
        try:
            row = ctx._ret_4h_panel.loc[entry_ts]
        except KeyError:
            row = None
        if row is not None and row.notna().any():
            breadth_4h = float((row < 0).sum() / row.notna().sum())
    if not np.isfinite(breadth_4h) or breadth_4h < BREADTH_MIN:
        return None

    # BTC trend score sign-lock: SHORT only when BTC trending down
    btc_score = float("nan")
    if ctx._btc_trend_score_15m is not None:
        v = ctx._btc_trend_score_15m.asof(entry_ts)
        if pd.notna(v):
            btc_score = float(v)
    if not np.isfinite(btc_score) or btc_score >= 0:
        return None

    # Build the 16-feature row (+ is_short) for v4 classifier
    feats = ctx._build_for_typed(symbol, entry_ts, "short")
    if any(not np.isfinite(feats.get(k, float("nan"))) for k in FEATURE_NAMES):
        return None

    return {
        "symbol": symbol,
        "direction": "short",
        "signal_time": entry_ts.to_pydatetime(),
        "drop_atr_norm": drop_atr_norm,
        "vol_ratio": vol_ratio,
        "breadth_4h": breadth_4h,
        "rolling_high_96": rh,
        "atr14_at_entry": atr,
        "btc_score": btc_score,
        "features": feats,
        "entry_price": c,
    }


# ── Classifier ───────────────────────────────────────────────────────


def load_classifier():
    if not MODEL_PATH.exists() or not META_PATH.exists():
        raise FileNotFoundError(f"Missing v4 model at {MODEL_PATH} or meta at {META_PATH}")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    feat_order = list(meta["feature_names"])
    return model, threshold, feat_order


def score_signal(model, feat_order: list[str], features: dict) -> float:
    x = np.array([features.get(k, np.nan) for k in feat_order], dtype=float).reshape(1, -1)
    if not np.all(np.isfinite(x)):
        return float("nan")
    return float(model.predict_proba(x)[0, 1])


# ── Main loop ────────────────────────────────────────────────────────


def main() -> None:
    logger.info("=" * 60)
    logger.info("LIVE SCANNER — D1e (Phase 17 locked, SHORT-only) + v4 classifier")
    logger.info("=" * 60)
    logger.info(
        f"Detector: drop>={ATR_DROP_MULT}*ATR vol>={VOL_RATIO_MIN}x red-bar "
        f"breadth>={BREADTH_MIN} btc_score<0 cooldown={COOLDOWN_SECONDS//3600}h"
    )

    engine = get_db_engine()
    ensure_table(engine)

    model, threshold, feat_order = load_classifier()
    logger.info(f"v4 classifier loaded (threshold={threshold:.4f}, "
                f"{len(feat_order)} features)")

    tickers = fetch_all_tickers()
    if not tickers:
        logger.error("No tickers fetched. Exiting.")
        return
    logger.info(f"Found {len(tickers)} USDT futures perps")

    candle_cache: dict[str, pd.DataFrame] = {}
    cooldowns: dict[str, float] = {}
    feature_ctx: FeatureContext | None = None
    last_candle_update = 0.0

    # Pre-load BTC + top-N
    btc_df = fetch_candles("BTC_USDT")
    if btc_df is not None:
        candle_cache["BTC_USDT"] = btc_df
    sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True)
    loaded = 0
    for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
        if pair == "BTC_USDT":
            continue
        df = fetch_candles(pair)
        if df is not None:
            candle_cache[pair] = df
            loaded += 1
        time.sleep(0.05)
    last_candle_update = time.time()
    logger.info(f"Pre-loaded {loaded} pairs (+ BTC)")

    # Build initial FeatureContext
    universe_df = build_universe_candles(candle_cache)
    if not universe_df.empty:
        try:
            feature_ctx = FeatureContext(candles=universe_df, btc_asset_key="BTCUSDT")
            logger.info("FeatureContext built (BTC trend score + breadth panel ready)")
        except Exception as e:
            logger.exception(f"Failed to build FeatureContext: {e}")
            feature_ctx = None

    logger.info("Scanner started. Ctrl+C to stop.")
    cycle = 0
    while True:
        try:
            cycle += 1
            cycle_start = time.time()
            now = cycle_start

            tickers = fetch_all_tickers()
            if not tickers:
                logger.warning(f"Cycle {cycle}: No tickers, skipping")
                time.sleep(SCAN_INTERVAL)
                continue

            # Refresh candles every 15 min
            if now - last_candle_update > CANDLE_INTERVAL:
                logger.info(f"Cycle {cycle}: refreshing candles...")
                btc_new = fetch_candles("BTC_USDT")
                if btc_new is not None:
                    candle_cache["BTC_USDT"] = btc_new
                sorted_pairs = sorted(
                    tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True
                )
                updated = 0
                for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
                    if pair == "BTC_USDT":
                        continue
                    df = fetch_candles(pair)
                    if df is not None:
                        candle_cache[pair] = df
                        updated += 1
                    time.sleep(0.05)
                last_candle_update = now
                # Rebuild FeatureContext on candle refresh
                universe_df = build_universe_candles(candle_cache)
                try:
                    feature_ctx = FeatureContext(candles=universe_df, btc_asset_key="BTCUSDT")
                    logger.info(f"  refreshed {updated} pairs, FeatureContext rebuilt")
                except Exception as e:
                    logger.exception(f"FeatureContext rebuild failed: {e}")

            if feature_ctx is None:
                logger.warning(f"Cycle {cycle}: no FeatureContext, skipping detection")
                time.sleep(SCAN_INTERVAL)
                continue

            # Scan
            signals_written = 0
            kept = 0
            scanned = 0
            for pair, ticker in tickers.items():
                if pair == "BTC_USDT":
                    continue
                if ticker["quote_volume"] < MIN_VOLUME_USD:
                    continue
                df = candle_cache.get(pair)
                if df is None:
                    continue
                symbol = pair.replace("_", "")
                scanned += 1

                # Cooldown
                if now - cooldowns.get(symbol, 0.0) < COOLDOWN_SECONDS:
                    continue

                hit = detect_d1e_short(feature_ctx, symbol, df)
                if hit is None:
                    continue

                cls_score = score_signal(model, feat_order, hit["features"])
                cls_kept = bool(np.isfinite(cls_score) and cls_score >= threshold)
                if cls_kept:
                    kept += 1

                ok = write_signal(
                    engine,
                    symbol=symbol,
                    direction=hit["direction"],
                    signal_time=hit["signal_time"],
                    drop_atr_norm=hit["drop_atr_norm"],
                    vol_ratio=hit["vol_ratio"],
                    breadth_4h=hit["breadth_4h"],
                    rolling_high_96=hit["rolling_high_96"],
                    atr14_at_entry=hit["atr14_at_entry"],
                    btc_score=hit["btc_score"],
                    features=hit["features"],
                    cls_score=cls_score if np.isfinite(cls_score) else None,
                    cls_kept=cls_kept,
                    entry_price=hit["entry_price"],
                )
                if ok:
                    cooldowns[symbol] = now
                    signals_written += 1

            elapsed = time.time() - cycle_start
            if signals_written > 0 or cycle % 5 == 0:
                logger.info(
                    f"Cycle {cycle}: scanned={scanned} fired={signals_written} "
                    f"v4_kept={kept} ({elapsed:.1f}s)"
                )

            sleep_time = max(0.0, SCAN_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Interrupted — shutting down.")
            break
        except Exception as e:
            logger.exception(f"Cycle {cycle} error: {e}")
            time.sleep(SCAN_INTERVAL)


if __name__ == "__main__":
    main()
