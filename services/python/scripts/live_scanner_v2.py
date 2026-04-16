"""Live Scanner v2 — Short the Failed Bounce.

3-phase entry:
  1. Detect dump (vol spike + price drop + BTC bearish + regime)
  2. Wait for bounce (price recovers +2% from dump low)
  3. Enter on rejection (price gives back 3% of bounce = lower high)

Uses ML model at threshold 0.70 to filter signals.

Usage: python scripts/live_scanner_v2.py
"""

import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
GATEIO_BASE = "https://api.gateio.ws/api/v4"
MODEL_PATH = Path("models/scanner_short_v2_ml.joblib")

# Scanner params (match backtest_short_v2.py)
VOL_SPIKE = 3.0
PRICE_DROP_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
BTC_EMA_PERIOD = 30
BOUNCE_MIN_PCT = 0.02       # Bounce must recover at least 2%
REJECTION_PCT = 0.03         # Bounce gives back 3% = rejection
MAX_BARS_AFTER_DUMP = 96     # 24h window for bounce+rejection
COOLDOWN_SECONDS = 96 * 900  # 96 bars * 15min = 24h

ML_THRESHOLD = 0.70
SCAN_INTERVAL = 60
CANDLE_INTERVAL = 900        # 15 min in seconds
CANDLE_INTERVAL_STR = "15m"
HISTORY_BARS = 2100  # ~22 days — enough for 20-day EMA regime filter
MIN_VOLUME_USD = 500_000
REGIME_CACHE_SECONDS = 3600  # Recompute regime once per hour

FEATURE_COLS = [
    "vol_ratio", "price_move", "price_change_1bar", "price_change_4bar",
    "bar_range_norm", "upper_wick_pct", "lower_wick_pct", "body_pct",
    "vol_trend", "price_trend_4bar", "price_trend_24bar",
    "volatility_20", "rsi_14",
    "btc_ret_4bar", "btc_ret_24bar",
    "hour_of_day", "day_of_week",
    "bars_since_last_spike", "atr_14",
    "bounce_pct", "bounce_bars", "bounce_vol_ratio", "rejection_speed",
]


@dataclass
class CoinState:
    """Track the 3-phase entry state for one coin."""

    phase: str = "idle"  # idle, watching_bounce, watching_rejection
    dump_time: datetime | None = None
    dump_close: float = 0.0
    dump_vol_ratio: float = 0.0
    bounce_high: float = 0.0
    bounce_high_bar: int = 0   # how many bars after dump
    bars_since_dump: int = 0
    dump_bar_vol_ma: float = 0.0  # vol_ma at dump bar for bounce_vol_ratio


# ── Logging setup (deferred until main() creates logs/ dir) ─────────
logger = logging.getLogger("scanner_v2")


def setup_logging():
    """Configure logging after ensuring logs/ directory exists."""
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/live_scanner_v2.log", mode="a"),
        ],
    )


# ── Gate.io API helpers (same as live_scanner.py) ───────────────────


def fetch_json(url: str, retries: int = 2) -> dict | list | None:
    """Fetch JSON from URL with retries."""
    for attempt in range(retries):
        try:
            req = Request(
                url, headers={"Accept": "application/json", "User-Agent": "Mozilla/5.0"}
            )
            resp = urlopen(req, timeout=10)
            return json.loads(resp.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as e:
            logger.warning(f"API error (attempt {attempt + 1}): {e}")
            if attempt < retries - 1:
                time.sleep(2)
    return None


def fetch_all_tickers() -> dict[str, dict]:
    """Fetch all Gate.io spot tickers. Returns {pair: {last, vol, ...}}."""
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
                "base_volume": float(t.get("base_volume", 0)),
                "quote_volume": float(t.get("quote_volume", 0)),
                "high_24h": float(t.get("high_24h", 0)),
                "low_24h": float(t.get("low_24h", 0)),
                "change_pct": float(t.get("change_percentage", 0)),
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


def fetch_candles(
    pair: str, interval: str = CANDLE_INTERVAL_STR, limit: int = HISTORY_BARS
) -> pd.DataFrame | None:
    """Load 15min candles from DB (source of truth). pair e.g. 'BTC_USDT' -> asset 'BTCUSDT'."""
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


def list_db_assets() -> list[str]:
    """List all assets in DB (USDT-suffixed)."""
    eng = get_db_engine()
    with eng.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT asset FROM asset_prices_15m")).fetchall()
    return [r[0] for r in result]


# ── Regime computation ──────────────────────────────────────────────


def compute_regime(candle_cache: dict[str, pd.DataFrame], btc_df: pd.DataFrame) -> bool:
    """Check if regime allows shorting.

    Returns True if BTC daily close < 20-day EMA AND < 60% of coins above
    their 20-day EMA.
    """
    # BTC check: resample to daily, compute 20-day EMA
    if len(btc_df) < 20 * 96:  # need ~20 days of 15m data
        return False

    btc_daily = (
        btc_df.set_index("timestamp")
        .resample("1D")
        .agg({"close": "last"})
        .dropna()
    )
    if len(btc_daily) < 20:
        return False

    btc_ema20 = btc_daily["close"].ewm(span=20, adjust=False).mean()
    btc_bearish = float(btc_daily["close"].iloc[-1]) < float(btc_ema20.iloc[-1])

    if not btc_bearish:
        return False

    # Breadth check: % of coins above their 20-day EMA
    above_count = 0
    total_count = 0
    for pair, df in candle_cache.items():
        if pair == "BTC_USDT" or len(df) < 20 * 96:
            continue
        coin_daily = (
            df.set_index("timestamp")
            .resample("1D")
            .agg({"close": "last"})
            .dropna()
        )
        if len(coin_daily) < 20:
            continue
        ema20 = coin_daily["close"].ewm(span=20, adjust=False).mean()
        total_count += 1
        if float(coin_daily["close"].iloc[-1]) > float(ema20.iloc[-1]):
            above_count += 1

    breadth = above_count / total_count if total_count > 0 else 0.5
    return breadth < 0.60


# ── Dump detection ──────────────────────────────────────────────────


def check_dump(df: pd.DataFrame, vol_ma: np.ndarray) -> dict | None:
    """Check if the latest bar is a dump signal.

    Returns dict with vol_ratio, price_drop, dump_close, or None.
    """
    i = len(df) - 1
    if i < PRICE_LOOKBACK + VOL_MA_PERIOD:
        return None

    close = df["close"].values
    high = df["high"].values
    volume = df["volume"].values

    # Vol spike check
    if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
        return None

    # Price drop from 96-bar high
    rolling_high = np.max(high[max(0, i - PRICE_LOOKBACK):i])
    if rolling_high <= 0:
        return None
    price_drop = (rolling_high - close[i]) / rolling_high
    if price_drop < PRICE_DROP_THRESH:
        return None

    return {
        "vol_ratio": volume[i] / vol_ma[i],
        "price_drop": price_drop,
        "dump_close": float(close[i]),
        "vol_ma_at_dump": float(vol_ma[i]),
    }


# ── State machine update ───────────────────────────────────────────


def update_coin_state(state: CoinState, df: pd.DataFrame) -> str | None:
    """Update state machine for one coin.

    Returns "SIGNAL" if entry triggered, else None.
    """
    close = df["close"].values
    i = len(df) - 1

    state.bars_since_dump += 1

    # Expire if too many bars since dump
    if state.bars_since_dump > MAX_BARS_AFTER_DUMP:
        state.phase = "idle"
        return None

    if state.phase == "watching_bounce":
        # Track bounce high
        if close[i] > state.bounce_high:
            state.bounce_high = float(close[i])
            state.bounce_high_bar = state.bars_since_dump

        # Check if bounce is meaningful (>= 2% recovery from dump close)
        bounce_pct = (state.bounce_high - state.dump_close) / state.dump_close
        if bounce_pct >= BOUNCE_MIN_PCT:
            # Check if price now rejecting (giving back 3% of bounce high)
            if close[i] < state.bounce_high * (1 - REJECTION_PCT):
                state.phase = "idle"
                return "SIGNAL"
            # Has the bounce peaked? (current close < bounce high by ~1%)
            if close[i] < state.bounce_high * 0.99:
                state.phase = "watching_rejection"
        return None

    if state.phase == "watching_rejection":
        # Still track if bounce goes higher
        if close[i] > state.bounce_high:
            state.bounce_high = float(close[i])
            state.bounce_high_bar = state.bars_since_dump
            state.phase = "watching_bounce"
            return None

        # Check rejection
        if close[i] < state.bounce_high * (1 - REJECTION_PCT):
            state.phase = "idle"
            return "SIGNAL"
        return None

    return None


# ── Feature extraction ──────────────────────────────────────────────


def extract_features(
    df: pd.DataFrame, state: CoinState, btc_df: pd.DataFrame | None
) -> dict:
    """Extract all 23 features (19 standard + 4 bounce-specific) at signal bar.

    Matches FEATURE_COLS from backtest_short_v2.py.
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_arr = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    i = len(df) - 1

    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values

    # vol_ratio
    vol_ratio = volume[i] / vol_ma[i] if i >= VOL_MA_PERIOD and vol_ma[i] > 0 else 0

    # price_move: drop from 96-bar rolling high
    if i >= PRICE_LOOKBACK:
        rolling_high = np.max(high[i - PRICE_LOOKBACK:i])
        price_move = (rolling_high - close[i]) / rolling_high if rolling_high > 0 else 0
    else:
        price_move = 0

    # price_change_1bar
    price_change_1bar = (
        (close[i] - close[i - 1]) / close[i - 1] if i >= 1 and close[i - 1] > 0 else 0
    )

    # price_change_4bar
    price_change_4bar = (
        (close[i] - close[i - 4]) / close[i - 4] if i >= 4 and close[i - 4] > 0 else 0
    )

    # Candle shape
    bar_range = high[i] - low[i]
    bar_range_norm = bar_range / close[i] if close[i] > 0 else 0
    if bar_range > 0:
        upper_wick_pct = (high[i] - max(open_arr[i], close[i])) / bar_range
        lower_wick_pct = (min(open_arr[i], close[i]) - low[i]) / bar_range
        body_pct = abs(close[i] - open_arr[i]) / bar_range
    else:
        upper_wick_pct = lower_wick_pct = body_pct = 0

    # vol_trend
    vol_trend = (
        vol_ma[i] / vol_ma[i - 4]
        if i >= 4 + VOL_MA_PERIOD and vol_ma[i - 4] > 0
        else 1.0
    )

    # price_trend_4bar, price_trend_24bar
    price_trend_4bar = (
        (close[i] - close[i - 4]) / close[i - 4] if i >= 4 and close[i - 4] > 0 else 0
    )
    price_trend_24bar = (
        (close[i] - close[i - 24]) / close[i - 24] if i >= 24 and close[i - 24] > 0 else 0
    )

    # volatility_20
    if i >= 20:
        rets = np.diff(close[i - 20:i + 1]) / close[i - 20:i]
        rets = rets[np.isfinite(rets)]
        volatility_20 = float(np.std(rets)) if len(rets) > 0 else 0
    else:
        volatility_20 = 0

    # rsi_14
    if i >= 14:
        changes = np.diff(close[i - 14:i + 1])
        gains = np.mean(changes[changes > 0]) if np.any(changes > 0) else 0
        losses_abs = np.mean(np.abs(changes[changes < 0])) if np.any(changes < 0) else 0
        rsi_14 = 100 - (100 / (1 + gains / losses_abs)) if losses_abs > 0 else 100
    else:
        rsi_14 = 50

    # BTC context
    btc_ret_4bar = 0.0
    btc_ret_24bar = 0.0
    if btc_df is not None and len(btc_df) > 24:
        btc_c = btc_df["close"].values.astype(float)
        btc_n = len(btc_c)
        if btc_n >= 5 and btc_c[btc_n - 5] > 0:
            btc_ret_4bar = (btc_c[-1] - btc_c[btc_n - 5]) / btc_c[btc_n - 5]
        if btc_n >= 25 and btc_c[btc_n - 25] > 0:
            btc_ret_24bar = (btc_c[-1] - btc_c[btc_n - 25]) / btc_c[btc_n - 25]

    # Time features
    ts = df["timestamp"].iloc[i]
    if hasattr(ts, "hour"):
        hour_of_day = ts.hour
        day_of_week = ts.weekday()
    else:
        hour_of_day = 0
        day_of_week = 0

    # bars_since_last_spike
    bars_since_last_spike = PRICE_LOOKBACK
    for j in range(i - 1, max(i - PRICE_LOOKBACK, 0), -1):
        if j < VOL_MA_PERIOD:
            break
        if vol_ma[j] > 0 and volume[j] / vol_ma[j] >= 2.0:
            bars_since_last_spike = i - j
            break

    # atr_14
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

    # Bounce-specific features
    bounce_pct = (state.bounce_high - state.dump_close) / state.dump_close
    bounce_bars = state.bounce_high_bar  # bars from dump to bounce peak

    # bounce_vol_ratio: avg vol during bounce relative to vol_ma at dump
    dump_bar_approx = max(0, i - state.bars_since_dump)
    bounce_peak_approx = dump_bar_approx + state.bounce_high_bar
    if bounce_peak_approx <= dump_bar_approx or bounce_peak_approx >= len(volume):
        bounce_vol_ratio = 0.0
    else:
        bounce_slice = volume[dump_bar_approx + 1:bounce_peak_approx + 1]
        vm = state.dump_bar_vol_ma if state.dump_bar_vol_ma > 0 else 1.0
        bounce_vol_ratio = float(np.mean(bounce_slice) / vm) if len(bounce_slice) > 0 else 0.0

    # rejection_speed: bars from bounce peak to now (entry)
    rejection_speed = state.bars_since_dump - state.bounce_high_bar

    return {
        "vol_ratio": vol_ratio,
        "price_move": price_move,
        "price_change_1bar": price_change_1bar,
        "price_change_4bar": price_change_4bar,
        "bar_range_norm": bar_range_norm,
        "upper_wick_pct": upper_wick_pct,
        "lower_wick_pct": lower_wick_pct,
        "body_pct": body_pct,
        "vol_trend": vol_trend,
        "price_trend_4bar": price_trend_4bar,
        "price_trend_24bar": price_trend_24bar,
        "volatility_20": volatility_20,
        "rsi_14": rsi_14,
        "btc_ret_4bar": btc_ret_4bar,
        "btc_ret_24bar": btc_ret_24bar,
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "bars_since_last_spike": bars_since_last_spike,
        "atr_14": atr_14,
        "bounce_pct": bounce_pct,
        "bounce_bars": bounce_bars,
        "bounce_vol_ratio": bounce_vol_ratio,
        "rejection_speed": rejection_speed,
    }


# ── DB helpers ──────────────────────────────────────────────────────


def ensure_table(engine):
    """Create scanner_signals_v2 table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS scanner_signals_v2 (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                direction SMALLINT NOT NULL DEFAULT -1,
                ml_prob DOUBLE PRECISION,
                vol_ratio DOUBLE PRECISION,
                price_drop DOUBLE PRECISION,
                bounce_pct DOUBLE PRECISION,
                signal_time TIMESTAMPTZ NOT NULL,
                status VARCHAR(10) DEFAULT 'active',
                entry_price DOUBLE PRECISION,
                pnl_pct DOUBLE PRECISION,
                exit_reason VARCHAR(20),
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
        """))


def write_signal(engine, symbol: str, signal_data: dict, ml_prob: float):
    """Write signal to scanner_signals_v2 table."""
    with engine.begin() as conn:
        conn.execute(
            text("""
                INSERT INTO scanner_signals_v2
                (symbol, direction, ml_prob, vol_ratio, price_drop, bounce_pct,
                 signal_time, status, entry_price)
                VALUES (:symbol, -1, :ml_prob, :vol_ratio, :price_drop, :bounce_pct,
                        :signal_time, 'active', :entry_price)
            """),
            {
                "symbol": symbol,
                "ml_prob": ml_prob,
                "vol_ratio": signal_data["vol_ratio"],
                "price_drop": signal_data["price_drop"],
                "bounce_pct": signal_data["bounce_pct"],
                "signal_time": datetime.now(timezone.utc),
                "entry_price": signal_data["entry_price"],
            },
        )


# ── Main loop ───────────────────────────────────────────────────────


def main():
    setup_logging()

    logger.info("=" * 60)
    logger.info("LIVE SCANNER V2 — SHORT THE FAILED BOUNCE")
    logger.info("=" * 60)
    logger.info(
        f"Params: vol_spike={VOL_SPIKE} drop_thresh={PRICE_DROP_THRESH} "
        f"bounce_min={BOUNCE_MIN_PCT} rejection={REJECTION_PCT} ml_thresh={ML_THRESHOLD}"
    )
    logger.info(f"Scan interval: {SCAN_INTERVAL}s")

    engine = create_engine(DB_URL)
    ensure_table(engine)

    # Load ML model
    model = None
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        logger.info(f"Loaded ML model from {MODEL_PATH}")
    else:
        logger.warning(
            f"No ML model found at {MODEL_PATH} — running without ML filter "
            "(all rejection signals will fire)"
        )

    coin_states: dict[str, CoinState] = {}
    candle_cache: dict[str, pd.DataFrame] = {}
    recent_signals: dict[str, float] = {}  # {symbol: epoch_time}
    btc_df: pd.DataFrame | None = None

    # Regime cache
    regime_allowed = False
    last_regime_check = 0.0

    # Candle refresh timing
    last_candle_update = 0.0

    cycle = 0
    logger.info("Fetching initial candle history...")

    # Get initial ticker list
    tickers = fetch_all_tickers()
    logger.info(f"Found {len(tickers)} USDT pairs on Gate.io")

    # Fetch BTC candles first
    btc_df = fetch_candles("BTC_USDT")
    if btc_df is not None:
        candle_cache["BTC_USDT"] = btc_df
        logger.info(f"BTC history loaded: {len(btc_df)} candles")

    # Pre-load top pairs by volume
    sorted_pairs = sorted(tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True)
    loaded = 0
    for pair, _ in sorted_pairs[:100]:
        if pair == "BTC_USDT":
            continue
        df = fetch_candles(pair)
        if df is not None:
            candle_cache[pair] = df
            loaded += 1
        time.sleep(0.1)  # Rate limit
    last_candle_update = time.time()
    logger.info(f"Pre-loaded {loaded} pairs' candle history")

    # Initial regime check
    if btc_df is not None:
        regime_allowed = compute_regime(candle_cache, btc_df)
        last_regime_check = time.time()
        logger.info(f"Initial regime: {'ALLOWED' if regime_allowed else 'BLOCKED'}")

    logger.info("Scanner started. Press Ctrl+C to stop.")
    logger.info("=" * 60)

    while True:
        try:
            cycle += 1
            cycle_start = time.time()
            now = cycle_start

            # Fetch tickers
            tickers = fetch_all_tickers()
            if not tickers:
                logger.warning(f"Cycle {cycle}: Failed to fetch tickers, skipping")
                time.sleep(SCAN_INTERVAL)
                continue

            # Update candle history every 15 minutes
            if now - last_candle_update > CANDLE_INTERVAL:
                logger.info(f"Cycle {cycle}: Updating 15min candle history...")

                # Update BTC
                btc_new = fetch_candles("BTC_USDT")
                if btc_new is not None:
                    btc_df = btc_new
                    candle_cache["BTC_USDT"] = btc_df

                # Update top pairs by volume
                sorted_pairs = sorted(
                    tickers.items(), key=lambda x: x[1]["quote_volume"], reverse=True
                )
                updated = 0
                for pair, _ in sorted_pairs[:100]:
                    if pair == "BTC_USDT":
                        continue
                    df = fetch_candles(pair)
                    if df is not None:
                        candle_cache[pair] = df
                        updated += 1
                    time.sleep(0.1)

                last_candle_update = now
                logger.info(f"  Updated {updated} pairs' candle history")

            # Recompute regime once per hour (expensive)
            if now - last_regime_check > REGIME_CACHE_SECONDS:
                if btc_df is not None:
                    regime_allowed = compute_regime(candle_cache, btc_df)
                    last_regime_check = now
                    logger.info(
                        f"Cycle {cycle}: Regime recomputed -> "
                        f"{'ALLOWED' if regime_allowed else 'BLOCKED'}"
                    )

            if not regime_allowed:
                # Clear all active watches — regime changed
                active = sum(1 for s in coin_states.values() if s.phase != "idle")
                if active > 0:
                    for s in coin_states.values():
                        s.phase = "idle"
                if cycle % 15 == 0:
                    logger.info(
                        f"Cycle {cycle}: Regime BLOCKED (BTC bullish or breadth > 60%), "
                        f"cleared {active} watches"
                    )
                sleep_time = max(0, SCAN_INTERVAL - (time.time() - cycle_start))
                time.sleep(sleep_time)
                continue

            # For each coin: check dump or update state
            signals_written = 0
            for pair, ticker in tickers.items():
                if pair == "BTC_USDT":
                    continue
                if pair not in candle_cache:
                    continue
                if ticker["quote_volume"] < MIN_VOLUME_USD / 96:
                    continue

                df = candle_cache[pair]
                if len(df) < PRICE_LOOKBACK + VOL_MA_PERIOD + 5:
                    continue

                vol_ma = pd.Series(df["volume"].values).rolling(VOL_MA_PERIOD).mean().values
                symbol = pair.replace("_", "")

                # Cooldown check
                if symbol in recent_signals:
                    if now - recent_signals[symbol] < COOLDOWN_SECONDS:
                        continue

                # If coin is idle, check for dump
                if symbol not in coin_states or coin_states[symbol].phase == "idle":
                    dump = check_dump(df, vol_ma)
                    if dump:
                        coin_states[symbol] = CoinState(
                            phase="watching_bounce",
                            dump_time=datetime.now(timezone.utc),
                            dump_close=dump["dump_close"],
                            dump_vol_ratio=dump["vol_ratio"],
                            bounce_high=dump["dump_close"],
                            bars_since_dump=0,
                            dump_bar_vol_ma=dump["vol_ma_at_dump"],
                        )
                        logger.info(
                            f"  DUMP: {pair} vol={dump['vol_ratio']:.1f}x "
                            f"drop={dump['price_drop']:.1%}"
                        )
                    continue

                # Update state machine for coins already being tracked
                result = update_coin_state(coin_states[symbol], df)

                if result == "SIGNAL":
                    state = coin_states[symbol]
                    bounce_pct = (state.bounce_high - state.dump_close) / state.dump_close
                    entry_price = float(df["close"].iloc[-1])

                    # Extract features
                    features = extract_features(df, state, btc_df)

                    # ML filter
                    ml_prob = 0.0
                    if model is not None:
                        X = pd.DataFrame(
                            [[features.get(col, 0) for col in FEATURE_COLS]],
                            columns=FEATURE_COLS,
                        )
                        X = X.fillna(0).replace([np.inf, -np.inf], 0)
                        try:
                            if hasattr(model, "predict_proba"):
                                ml_prob = float(model.predict_proba(X)[0][1])
                            else:
                                ml_prob = float(model.predict(X)[0])
                        except Exception as e:
                            logger.error(f"ML prediction error for {pair}: {e}")
                            continue

                        if ml_prob < ML_THRESHOLD:
                            logger.info(
                                f"  REJECT: {pair} ml={ml_prob:.2f} < {ML_THRESHOLD}"
                            )
                            continue

                    # Write signal
                    signal_data = {
                        "vol_ratio": state.dump_vol_ratio,
                        "price_drop": (state.bounce_high - entry_price) / state.bounce_high,
                        "bounce_pct": bounce_pct,
                        "entry_price": entry_price,
                    }
                    write_signal(engine, symbol, signal_data, ml_prob)
                    recent_signals[symbol] = now
                    signals_written += 1

                    logger.info(
                        f"  SIGNAL: {pair} SHORT | ml={ml_prob:.2f} "
                        f"vol={state.dump_vol_ratio:.1f}x bounce={bounce_pct:.1%} "
                        f"entry=${entry_price:.4f}"
                    )

            # Cycle summary
            active_watches = sum(1 for s in coin_states.values() if s.phase != "idle")
            elapsed = time.time() - cycle_start
            if cycle % 5 == 0 or signals_written > 0:
                logger.info(
                    f"Cycle {cycle}: {len(tickers)} pairs, "
                    f"{active_watches} watching, "
                    f"{signals_written} signals "
                    f"| regime=ALLOWED ({elapsed:.1f}s)"
                )

            # Clean up expired states to prevent memory leak
            if cycle % 60 == 0:
                expired = [
                    sym for sym, s in coin_states.items()
                    if s.phase == "idle"
                ]
                for sym in expired:
                    del coin_states[sym]

            sleep_time = max(0, SCAN_INTERVAL - elapsed)
            time.sleep(sleep_time)

        except KeyboardInterrupt:
            logger.info("Scanner v2 stopped by user.")
            break
        except Exception as e:
            logger.error(f"Cycle {cycle} error: {e}", exc_info=True)
            time.sleep(SCAN_INTERVAL)

    engine.dispose()
    logger.info("Scanner v2 shutdown complete.")


if __name__ == "__main__":
    main()
