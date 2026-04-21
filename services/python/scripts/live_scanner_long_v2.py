"""Live Scanner Long v2 — Long the Successful Pullback.

Mirror of live_scanner_v2.py with inverted logic:
  1. Detect pump (vol spike + price up from 96-bar low + BTC bullish regime)
  2. Wait for pullback (price retraces >= 2% from pump high)
  3. Enter on reclaim (price reclaims 3% above pullback low)

Uses ML model at threshold 0.80 to filter signals.

Run alongside live_scanner_v2.py (separate terminal) for dual-direction coverage.
"""

import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from src import db_adapters  # noqa: F401  # registers numpy→psycopg2 adapters

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
GATEIO_BASE = "https://api.gateio.ws/api/v4"
MODEL_PATH = Path("models/scanner_long_v2_ml.joblib")
SCANNER_TOP_N = int(os.environ.get("SCANNER_TOP_N", "100"))

VOL_SPIKE = 3.0
PRICE_RISE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
PULLBACK_MIN_PCT = float(os.environ.get("PULLBACK_MIN_PCT", "0.02"))
RECLAIM_PCT = float(os.environ.get("RECLAIM_PCT", "0.03"))
MAX_BARS_AFTER_PUMP = 96
COOLDOWN_SECONDS = 96 * 900

ML_THRESHOLD = 0.80
SCAN_INTERVAL = 60
CANDLE_INTERVAL = 900
CANDLE_INTERVAL_STR = "15m"
HISTORY_BARS = 2100  # ~22 days — enough for 20-day EMA regime filter
MIN_VOLUME_USD = 500_000
REGIME_CACHE_SECONDS = 3600

FEATURE_COLS = [
    "vol_ratio", "price_move", "price_change_1bar", "price_change_4bar",
    "bar_range_norm", "upper_wick_pct", "lower_wick_pct", "body_pct",
    "vol_trend", "price_trend_4bar", "price_trend_24bar",
    "volatility_20", "rsi_14",
    "btc_ret_4bar", "btc_ret_24bar",
    "hour_of_day", "day_of_week",
    "bars_since_last_spike", "atr_14",
    "pullback_pct", "pullback_bars", "pullback_vol_ratio", "reclaim_speed",
]


@dataclass
class CoinState:
    phase: str = "idle"  # idle, watching_pullback, watching_reclaim
    pump_time: datetime | None = None
    pump_close: float = 0.0
    pump_vol_ratio: float = 0.0
    pullback_low: float = 0.0
    pullback_low_bar: int = 0
    bars_since_pump: int = 0
    pump_bar_vol_ma: float = 0.0


logger = logging.getLogger("scanner_long_v2")


def setup_logging():
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/live_scanner_long_v2.log", mode="a"),
        ],
    )


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


def fetch_all_tickers():
    data = fetch_json(f"{GATEIO_BASE}/spot/tickers")
    if not data:
        return {}
    tickers = {}
    for t in data:
        pair = t.get("currency_pair", "")
        if not pair.endswith("_USDT"):
            continue
        try:
            tickers[pair] = {
                "last": float(t.get("last", 0)),
                "quote_volume": float(t.get("quote_volume", 0)),
            }
        except (ValueError, TypeError):
            continue
    return tickers


_db_engine = None


def get_db_engine():
    global _db_engine
    if _db_engine is None:
        _db_engine = create_engine(DB_URL)
    return _db_engine


def fetch_candles(pair: str, interval: str = CANDLE_INTERVAL_STR, limit: int = HISTORY_BARS):
    """Load 15min candles from DB (source of truth). pair 'BTC_USDT' -> asset 'BTCUSDT'."""
    asset = pair.replace("_", "")
    eng = get_db_engine()
    with eng.connect() as conn:
        result = conn.execute(text("""
            SELECT timestamp, open, high, low, close, volume
            FROM asset_prices_15m WHERE asset = :a
            ORDER BY timestamp DESC LIMIT :lim
        """), {"a": asset, "lim": limit}).fetchall()
    if not result or len(result) < 10:
        return None
    df = pd.DataFrame(result, columns=["timestamp", "open", "high", "low", "close", "volume"])
    return df.sort_values("timestamp").reset_index(drop=True)


def compute_regime(candle_cache, btc_df):
    """Bullish regime: BTC daily close > 20d EMA AND breadth > 40%."""
    if len(btc_df) < 20 * 96:
        return False
    btc_daily = btc_df.set_index("timestamp").resample("1D").agg({"close": "last"}).dropna()
    if len(btc_daily) < 20:
        return False
    btc_ema20 = btc_daily["close"].ewm(span=20, adjust=False).mean()
    btc_bullish = float(btc_daily["close"].iloc[-1]) > float(btc_ema20.iloc[-1])
    if not btc_bullish:
        return False

    above_count = 0
    total_count = 0
    for pair, df in candle_cache.items():
        if pair == "BTC_USDT" or len(df) < 20 * 96:
            continue
        coin_daily = df.set_index("timestamp").resample("1D").agg({"close": "last"}).dropna()
        if len(coin_daily) < 20:
            continue
        ema20 = coin_daily["close"].ewm(span=20, adjust=False).mean()
        total_count += 1
        if float(coin_daily["close"].iloc[-1]) > float(ema20.iloc[-1]):
            above_count += 1
    breadth = above_count / total_count if total_count > 0 else 0.5
    return breadth > 0.40


def check_pump(df, vol_ma):
    i = len(df) - 1
    if i < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return None
    close = df["close"].values
    low = df["low"].values
    volume = df["volume"].values
    if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
        return None
    rolling_low = np.min(low[max(0, i - PRICE_LOOKBACK):i])
    if rolling_low <= 0:
        return None
    price_rise = (close[i] - rolling_low) / rolling_low
    if price_rise < PRICE_RISE_THRESH:
        return None
    return {
        "vol_ratio": volume[i] / vol_ma[i],
        "price_rise": price_rise,
        "pump_close": float(close[i]),
        "vol_ma_at_pump": float(vol_ma[i]),
    }


def update_coin_state(state, df):
    close = df["close"].values
    i = len(df) - 1
    state.bars_since_pump += 1

    if state.bars_since_pump > MAX_BARS_AFTER_PUMP:
        state.phase = "idle"
        return None

    if state.phase == "watching_pullback":
        if close[i] < state.pullback_low:
            state.pullback_low = float(close[i])
            state.pullback_low_bar = state.bars_since_pump
        pullback_pct = (state.pump_close - state.pullback_low) / state.pump_close
        if pullback_pct >= PULLBACK_MIN_PCT:
            if close[i] > state.pullback_low * (1 + RECLAIM_PCT):
                state.phase = "idle"
                return "SIGNAL"
            if close[i] > state.pullback_low * 1.01:
                state.phase = "watching_reclaim"
        return None

    if state.phase == "watching_reclaim":
        if close[i] < state.pullback_low:
            state.pullback_low = float(close[i])
            state.pullback_low_bar = state.bars_since_pump
            state.phase = "watching_pullback"
            return None
        if close[i] > state.pullback_low * (1 + RECLAIM_PCT):
            state.phase = "idle"
            return "SIGNAL"
        return None
    return None


def extract_features(df, state, btc_df):
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_arr = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    i = len(df) - 1
    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values

    vol_ratio = volume[i] / vol_ma[i] if i >= VOL_MA_PERIOD and vol_ma[i] > 0 else 0

    # price_move for long: rise from 96-bar rolling LOW
    if i >= PRICE_LOOKBACK:
        rolling_low = np.min(low[i - PRICE_LOOKBACK:i])
        price_move = (close[i] - rolling_low) / rolling_low if rolling_low > 0 else 0
    else:
        price_move = 0

    price_change_1bar = (close[i] - close[i - 1]) / close[i - 1] if i >= 1 and close[i - 1] > 0 else 0
    price_change_4bar = (close[i] - close[i - 4]) / close[i - 4] if i >= 4 and close[i - 4] > 0 else 0

    bar_range = high[i] - low[i]
    bar_range_norm = bar_range / close[i] if close[i] > 0 else 0
    if bar_range > 0:
        upper_wick_pct = (high[i] - max(open_arr[i], close[i])) / bar_range
        lower_wick_pct = (min(open_arr[i], close[i]) - low[i]) / bar_range
        body_pct = abs(close[i] - open_arr[i]) / bar_range
    else:
        upper_wick_pct = lower_wick_pct = body_pct = 0

    vol_trend = vol_ma[i] / vol_ma[i - 4] if i >= 4 + VOL_MA_PERIOD and vol_ma[i - 4] > 0 else 1.0

    price_trend_4bar = (close[i] - close[i - 4]) / close[i - 4] if i >= 4 and close[i - 4] > 0 else 0
    price_trend_24bar = (close[i] - close[i - 24]) / close[i - 24] if i >= 24 and close[i - 24] > 0 else 0

    if i >= 20:
        rets = np.diff(close[i - 20:i + 1]) / close[i - 20:i]
        rets = rets[np.isfinite(rets)]
        volatility_20 = float(np.std(rets)) if len(rets) > 0 else 0
    else:
        volatility_20 = 0

    if i >= 14:
        changes = np.diff(close[i - 14:i + 1])
        gains = np.mean(changes[changes > 0]) if np.any(changes > 0) else 0
        losses_abs = np.mean(np.abs(changes[changes < 0])) if np.any(changes < 0) else 0
        rsi_14 = 100 - (100 / (1 + gains / losses_abs)) if losses_abs > 0 else 100
    else:
        rsi_14 = 50

    btc_ret_4bar = 0.0
    btc_ret_24bar = 0.0
    if btc_df is not None and len(btc_df) > 24:
        btc_c = btc_df["close"].values.astype(float)
        btc_n = len(btc_c)
        if btc_n >= 5 and btc_c[btc_n - 5] > 0:
            btc_ret_4bar = (btc_c[-1] - btc_c[btc_n - 5]) / btc_c[btc_n - 5]
        if btc_n >= 25 and btc_c[btc_n - 25] > 0:
            btc_ret_24bar = (btc_c[-1] - btc_c[btc_n - 25]) / btc_c[btc_n - 25]

    ts = df["timestamp"].iloc[i]
    hour_of_day = ts.hour if hasattr(ts, "hour") else 0
    day_of_week = ts.weekday() if hasattr(ts, "weekday") else 0

    bars_since_last_spike = PRICE_LOOKBACK
    for j in range(i - 1, max(i - PRICE_LOOKBACK, 0), -1):
        if j < VOL_MA_PERIOD:
            break
        if vol_ma[j] > 0 and volume[j] / vol_ma[j] >= 2.0:
            bars_since_last_spike = i - j
            break

    if i >= 15:
        tr_arr = np.maximum(
            high[i - 14:i] - low[i - 14:i],
            np.maximum(
                np.abs(high[i - 14:i] - close[i - 15:i - 1]),
                np.abs(low[i - 14:i] - close[i - 15:i - 1]),
            ),
        )
        atr_14 = float(np.mean(tr_arr) / close[i]) if close[i] > 0 else 0
    else:
        atr_14 = 0

    pullback_pct = (state.pump_close - state.pullback_low) / state.pump_close
    pullback_bars = state.pullback_low_bar

    pump_bar_approx = max(0, i - state.bars_since_pump)
    pullback_peak_approx = pump_bar_approx + state.pullback_low_bar
    if pullback_peak_approx <= pump_bar_approx or pullback_peak_approx >= len(volume):
        pullback_vol_ratio = 0.0
    else:
        pullback_slice = volume[pump_bar_approx + 1:pullback_peak_approx + 1]
        vm = state.pump_bar_vol_ma if state.pump_bar_vol_ma > 0 else 1.0
        pullback_vol_ratio = float(np.mean(pullback_slice) / vm) if len(pullback_slice) > 0 else 0.0

    reclaim_speed = state.bars_since_pump - state.pullback_low_bar

    return {
        "vol_ratio": vol_ratio, "price_move": price_move,
        "price_change_1bar": price_change_1bar, "price_change_4bar": price_change_4bar,
        "bar_range_norm": bar_range_norm,
        "upper_wick_pct": upper_wick_pct, "lower_wick_pct": lower_wick_pct, "body_pct": body_pct,
        "vol_trend": vol_trend,
        "price_trend_4bar": price_trend_4bar, "price_trend_24bar": price_trend_24bar,
        "volatility_20": volatility_20, "rsi_14": rsi_14,
        "btc_ret_4bar": btc_ret_4bar, "btc_ret_24bar": btc_ret_24bar,
        "hour_of_day": hour_of_day, "day_of_week": day_of_week,
        "bars_since_last_spike": bars_since_last_spike, "atr_14": atr_14,
        "pullback_pct": pullback_pct, "pullback_bars": pullback_bars,
        "pullback_vol_ratio": pullback_vol_ratio, "reclaim_speed": reclaim_speed,
    }


def ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals_long_v2 (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                direction SMALLINT NOT NULL DEFAULT 1,
                ml_prob DOUBLE PRECISION,
                vol_ratio DOUBLE PRECISION,
                price_rise DOUBLE PRECISION,
                pullback_pct DOUBLE PRECISION,
                signal_time TIMESTAMPTZ NOT NULL,
                status VARCHAR(10) DEFAULT 'active',
                entry_price DOUBLE PRECISION,
                pnl_pct DOUBLE PRECISION,
                exit_reason VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))


def write_signal(engine, symbol, signal_data, ml_prob):
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO scanner_signals_long_v2
                (symbol, direction, ml_prob, vol_ratio, price_rise, pullback_pct,
                 signal_time, status, entry_price)
                VALUES (:symbol, 1, :ml_prob, :vol_ratio, :price_rise, :pullback_pct,
                        :signal_time, 'active', :entry_price)
            """),
            {
                "symbol": symbol, "ml_prob": ml_prob,
                "vol_ratio": signal_data["vol_ratio"],
                "price_rise": signal_data["price_rise"],
                "pullback_pct": signal_data["pullback_pct"],
                "signal_time": datetime.now(timezone.utc),
                "entry_price": signal_data["entry_price"],
            },
        )


def main():
    setup_logging()
    logger.info("=" * 60)
    logger.info("LIVE SCANNER LONG V2 — LONG THE SUCCESSFUL PULLBACK")
    logger.info("=" * 60)
    logger.info(
        f"Params: vol_spike={VOL_SPIKE} rise_thresh={PRICE_RISE_THRESH} "
        f"pullback_min={PULLBACK_MIN_PCT} reclaim={RECLAIM_PCT} ml_thresh={ML_THRESHOLD}"
    )

    engine = create_engine(DB_URL)
    ensure_table(engine)

    model = None
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded ML model from {MODEL_PATH}")
    else:
        logger.warning(f"No ML model at {MODEL_PATH} — running without ML filter")

    coin_states: dict[str, CoinState] = {}
    candle_cache: dict[str, pd.DataFrame] = {}
    recent_signals: dict[str, float] = {}
    btc_df = None
    regime_allowed = False
    last_regime_check = 0.0
    last_candle_update = 0.0

    tickers = fetch_all_tickers()
    logger.info(f"Found {len(tickers)} USDT pairs on Gate.io")

    btc_df = fetch_candles("BTC_USDT")
    if btc_df is not None:
        candle_cache["BTC_USDT"] = btc_df
        logger.info(f"BTC history loaded: {len(btc_df)} candles")

    sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True)
    loaded = 0
    for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
        if pair == "BTC_USDT":
            continue
        df = fetch_candles(pair)
        if df is not None:
            candle_cache[pair] = df
            loaded += 1
        time.sleep(0.1)
    last_candle_update = time.time()
    logger.info(f"Pre-loaded {loaded} pairs' candle history")

    if btc_df is not None:
        regime_allowed = compute_regime(candle_cache, btc_df)
        last_regime_check = time.time()
        logger.info(f"Initial regime: {'ALLOWED' if regime_allowed else 'BLOCKED'}")

    cycle = 0
    logger.info("Scanner started. Press Ctrl+C to stop.")

    while True:
        try:
            cycle += 1
            cycle_start = time.time()
            now = cycle_start

            tickers = fetch_all_tickers()
            if not tickers:
                time.sleep(SCAN_INTERVAL)
                continue

            if now - last_candle_update > CANDLE_INTERVAL:
                btc_new = fetch_candles("BTC_USDT")
                if btc_new is not None:
                    btc_df = btc_new
                    candle_cache["BTC_USDT"] = btc_df
                sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True)
                for pair, _ in sorted_pairs[:SCANNER_TOP_N]:
                    if pair == "BTC_USDT":
                        continue
                    df = fetch_candles(pair)
                    if df is not None:
                        candle_cache[pair] = df
                    time.sleep(0.1)
                last_candle_update = now

            if now - last_regime_check > REGIME_CACHE_SECONDS:
                if btc_df is not None:
                    regime_allowed = compute_regime(candle_cache, btc_df)
                    last_regime_check = now
                    logger.info(f"Cycle {cycle}: Regime -> {'ALLOWED' if regime_allowed else 'BLOCKED'}")

            if not regime_allowed:
                for s in coin_states.values():
                    s.phase = "idle"
                if cycle % 15 == 0:
                    logger.info(f"Cycle {cycle}: Regime BLOCKED (BTC bearish or breadth < 40%)")
                time.sleep(max(0, SCAN_INTERVAL - (time.time() - cycle_start)))
                continue

            signals_written = 0
            for pair, ticker in tickers.items():
                if pair == "BTC_USDT" or pair not in candle_cache:
                    continue
                if ticker["quote_volume"] < MIN_VOLUME_USD / 96:
                    continue
                df = candle_cache[pair]
                if len(df) < PRICE_LOOKBACK + VOL_MA_PERIOD + 5:
                    continue

                vol_ma = pd.Series(df["volume"].values).rolling(VOL_MA_PERIOD).mean().values
                symbol = pair.replace("_", "")

                if symbol in recent_signals and now - recent_signals[symbol] < COOLDOWN_SECONDS:
                    continue

                if symbol not in coin_states or coin_states[symbol].phase == "idle":
                    pump = check_pump(df, vol_ma)
                    if pump:
                        coin_states[symbol] = CoinState(
                            phase="watching_pullback",
                            pump_time=datetime.now(timezone.utc),
                            pump_close=pump["pump_close"],
                            pump_vol_ratio=pump["vol_ratio"],
                            pullback_low=pump["pump_close"],
                            bars_since_pump=0,
                            pump_bar_vol_ma=pump["vol_ma_at_pump"],
                        )
                        logger.info(
                            f"  PUMP: {pair} vol={pump['vol_ratio']:.1f}x "
                            f"rise={pump['price_rise']:.1%}"
                        )
                    continue

                result = update_coin_state(coin_states[symbol], df)
                if result == "SIGNAL":
                    state = coin_states[symbol]
                    pullback_pct = (state.pump_close - state.pullback_low) / state.pump_close
                    entry_price = float(df["close"].iloc[-1])

                    features = extract_features(df, state, btc_df)

                    ml_prob = 0.0
                    if model is not None:
                        X = pd.DataFrame(
                            [[features.get(col, 0) for col in FEATURE_COLS]],
                            columns=FEATURE_COLS,
                        )
                        X = X.fillna(0).replace([np.inf, -np.inf], 0)
                        try:
                            ml_prob = float(model.predict_proba(X)[0][1])
                        except Exception as e:
                            logger.error(f"ML error for {pair}: {e}")
                            continue
                    signal_data = {
                        "vol_ratio": state.pump_vol_ratio,
                        "price_rise": (entry_price - state.pullback_low) / state.pullback_low,
                        "pullback_pct": pullback_pct,
                        "entry_price": entry_price,
                    }
                    write_signal(engine, symbol, signal_data, ml_prob)
                    recent_signals[symbol] = now
                    signals_written += 1

                    tag = "SIGNAL" if ml_prob >= ML_THRESHOLD else "SIGNAL(low-ml)"
                    logger.info(
                        f"  {tag}: {pair} LONG | ml={ml_prob:.2f} "
                        f"vol={state.pump_vol_ratio:.1f}x pullback={pullback_pct:.1%} "
                        f"entry=${entry_price:.4f}"
                    )

            active_watches = sum(1 for s in coin_states.values() if s.phase != "idle")
            elapsed = time.time() - cycle_start
            if cycle % 5 == 0 or signals_written > 0:
                logger.info(
                    f"Cycle {cycle}: {len(tickers)} pairs, {active_watches} watching, "
                    f"{signals_written} signals | regime=ALLOWED ({elapsed:.1f}s)"
                )

            if cycle % 60 == 0:
                for sym in [s for s, st in coin_states.items() if st.phase == "idle"]:
                    del coin_states[sym]

            time.sleep(max(0, SCAN_INTERVAL - elapsed))

        except KeyboardInterrupt:
            logger.info("Scanner long v2 stopped by user.")
            break
        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
            time.sleep(SCAN_INTERVAL)

    engine.dispose()


if __name__ == "__main__":
    main()
