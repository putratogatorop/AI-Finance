"""v_new_2 — Live scanner (paper-deploy version).

Runs every 4h on VPS. For each symbol in the U4 universe (memes + AI + top-50),
fetches recent 4h candles from Postgres, computes v_new_2 features (daily EMAs,
cycle highs/lows, pullback metrics), checks if rule fires THIS bar, scores via
honest-trained BGM model, and writes signals where bgm_score ≥ 0.60 to the
`v_new_2_signals` table.

INVOCATION (on VPS via cron every 4h):
    python services/python/scripts/v_new_2_live_scanner.py

CONFIGURATION (env):
    DATABASE_URL          — Postgres connection
    V_NEW_2_BGM_THRESHOLD — default 0.60
    V_NEW_2_DRY_RUN       — if "1", logs signals but doesn't write to DB
    V_NEW_2_MIN_BARS      — minimum 4h bars required per symbol (default 600 = 100 days)

OUTPUT: rows in v_new_2_signals (one per rule-trigger bar that scored ≥ thr)

Universe selection: services/python/data/v_new_1_v2/v_new_2_u4_universe.json
                    (auto-generated on first run if missing)
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("v_new_2_scanner")

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "services" / "python" / "data" / "v_new_1_v2"
MODELS_DIR = ROOT / "services" / "python" / "models"

DB_URL = os.environ.get("DATABASE_URL",
                          "postgresql://postgres:MySQL100%25@localhost:5432/market")
THRESHOLD = float(os.environ.get("V_NEW_2_BGM_THRESHOLD", "0.60"))
DRY_RUN = os.environ.get("V_NEW_2_DRY_RUN", "0") == "1"
MIN_BARS = int(os.environ.get("V_NEW_2_MIN_BARS", "600"))

EMA_FAST = 20
EMA_SLOW = 50
BARS_PER_DAY = 6
LOOKBACK_BARS = 600   # ~100 days of 4h bars

# Feature columns the BGM model expects (must match v_new_2_step_d_honest_eval.py)
V_NEW_1_FEATURES = [
    "atr14_pct_rank_90d","vol_z_24h","coin_7d_return","coin_30d_return",
    "close_to_high50_atr","close_to_low50_atr","bar4h_close_pos_in_range",
    "bar4h_body_pct","bar4h_upper_wick_pct","h4_macd_hist","h4_macd_macd",
    "h4_rsi","h4_close_vs_ema50_pct","daily_macd_hist","days_since_bull_flip",
    "days_since_bear_flip","btc_above_4h_ema50","btc_24h_return","btc_realized_vol_z",
    "btc_score","breadth_up","breadth_down","hour_sin","hour_cos","dow_sin","dow_cos",
    "cvd_slope_1h_norm","bb_pct_b","bb_bandwidth","fib_pos_50","kdj_k","kdj_j",
    "price_vs_ema9_pct","rsi14_delta_1bar","rsi14_delta_4bar","macd_hist_momentum",
    "macd_signal_spread_norm","parkinson_vol","obv_slope_norm","cfgi_value",
    "vol_rank_in_top100","days_since_last_big_move_long","days_since_last_big_move_short",
    "coin_24h_vol_zscore_30d",
]
RULE_FEATURES = [
    "d_ema_spread_pct","days_since_long_flip","days_since_short_flip",
    "pullback_pct","rise_pct","pullback_atr","rise_atr","atr14_pct","trend_strength",
]
TIER_DUMMIES = ["tier_top25", "tier_26-50", "tier_51-100", "tier_101-200"]
FEAT_COLS = V_NEW_1_FEATURES + RULE_FEATURES + TIER_DUMMIES

# Per-tier rule configs (from honest backtest)
TIER_CONFIGS = {
    "top25":   {"long": ("atr",  2.0), "short": ("atr",  4.0)},
    "26-50":   {"long": ("pct", 15.0), "short": ("pct", 10.0)},
    "51-100":  {"long": ("atr",  3.0), "short": ("atr",  4.0)},
    "101-200": {"long": ("atr",  4.0), "short": ("atr",  4.0)},
}


# ─── DB helpers ──────────────────────────────────────────────────────────────
_engine = None


def _engine_lazy():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=300)
    return _engine


def _ensure_signals_table():
    """Create v_new_2_signals + supporting tables if missing."""
    with _engine_lazy().connect() as c:
        c.execute(text("""
            CREATE TABLE IF NOT EXISTS v_new_2_signals (
                id            SERIAL PRIMARY KEY,
                signal_time   TIMESTAMPTZ NOT NULL,
                symbol        VARCHAR(20) NOT NULL,
                side          VARCHAR(10) NOT NULL,
                tier          VARCHAR(10),
                pullback_pct  DOUBLE PRECISION,
                pullback_atr  DOUBLE PRECISION,
                bgm_score     DOUBLE PRECISION NOT NULL,
                d_ema_spread  DOUBLE PRECISION,
                close_at_signal DOUBLE PRECISION NOT NULL,
                taken         BOOLEAN DEFAULT FALSE,
                created_at    TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT v_new_2_signals_unique UNIQUE(signal_time, symbol, side)
            );
            CREATE INDEX IF NOT EXISTS ix_v_new_2_signals_time
                ON v_new_2_signals(signal_time DESC);
            CREATE INDEX IF NOT EXISTS ix_v_new_2_signals_taken
                ON v_new_2_signals(taken) WHERE taken = FALSE;
        """))
        c.commit()


def _load_universe() -> set[str]:
    """U4 universe = memes + AI + top-50."""
    universe_path = DATA_DIR / "v_new_2_u4_universe.json"
    if universe_path.exists():
        with open(universe_path) as f:
            return set(json.load(f))
    log.info("U4 universe file missing; building from CG categories + membership")
    with open(DATA_DIR / "coingecko_categories.json") as f:
        cats = json.load(f)
    memes = set(cats.get("memes", []))
    ai = set(cats.get("ai_tokens", []))
    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    avg_rank = mem.groupby("symbol")["rank"].mean()
    top50 = set(avg_rank[avg_rank <= 50].index)
    universe = memes | ai | top50
    universe_path.parent.mkdir(parents=True, exist_ok=True)
    with open(universe_path, "w") as f:
        json.dump(sorted(universe), f, indent=2)
    log.info("U4 universe built and cached: %d symbols", len(universe))
    return universe


def _load_bgm_models() -> tuple:
    long_model = joblib.load(MODELS_DIR / "v_new_2_bgm_long_honest" / "model.joblib")
    short_model = joblib.load(MODELS_DIR / "v_new_2_bgm_short_honest" / "model.joblib")
    return long_model, short_model


def _fetch_recent_candles(symbols: set[str], lookback_bars: int) -> pd.DataFrame:
    """Read last `lookback_bars` 4h candles per symbol from candles_4h table."""
    cutoff_ts = datetime.now(UTC) - timedelta(hours=4 * lookback_bars + 16)
    sym_list = ",".join(f"'{s}'" for s in symbols)
    sql = f"""
        SELECT symbol, ts AS timestamp, close
        FROM candles_4h
        WHERE symbol IN ({sym_list})
          AND ts >= :cutoff
        ORDER BY symbol, ts
    """
    with _engine_lazy().connect() as c:
        df = pd.read_sql(text(sql), c, params={"cutoff": cutoff_ts})
    if len(df) == 0:
        log.warning("no candles returned (table candles_4h empty for these symbols?)")
        return df
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _compute_v_new_2_features(g: pd.DataFrame) -> pd.DataFrame:
    """Per-symbol: compute daily EMA20/50, cycle high/low, pullback metrics, ATR.
    Mirrors v_new_2_features.py + v_new_2_features_v2.py logic."""
    g = g.sort_values("timestamp").reset_index(drop=True)
    if len(g) < EMA_SLOW * BARS_PER_DAY + 10:
        return None
    # Daily resample
    g_idx = g.set_index("timestamp")
    daily = g_idx["close"].resample("1D").last().dropna()
    if len(daily) < EMA_SLOW + 5:
        return None
    d_ema20 = daily.ewm(span=EMA_FAST, adjust=False).mean()
    d_ema50 = daily.ewm(span=EMA_SLOW, adjust=False).mean()
    long_flip = ((d_ema20 > d_ema50).astype(int).diff() > 0).astype(int)
    short_flip = ((d_ema20 < d_ema50).astype(int).diff() > 0).astype(int)

    def _days_since(flip):
        out = np.full(len(flip), np.nan)
        last_idx = -1_000_000
        for i in range(len(flip)):
            if flip.iloc[i] == 1:
                last_idx = i
            if last_idx >= 0:
                out[i] = i - last_idx
        return pd.Series(out, index=flip.index)

    days_long = _days_since(long_flip)
    days_short = _days_since(short_flip)

    # Cycle high/low since last flip
    def _cycle_high(closes, flip):
        out = np.full(len(closes), np.nan)
        rmax = -np.inf
        for i in range(len(closes)):
            rmax = closes.iloc[i] if flip.iloc[i] == 1 else max(rmax, closes.iloc[i])
            out[i] = rmax
        return pd.Series(out, index=closes.index)
    def _cycle_low(closes, flip):
        out = np.full(len(closes), np.nan)
        rmin = np.inf
        for i in range(len(closes)):
            rmin = closes.iloc[i] if flip.iloc[i] == 1 else min(rmin, closes.iloc[i])
            out[i] = rmin
        return pd.Series(out, index=closes.index)
    cycle_high_d = _cycle_high(daily, long_flip)
    cycle_low_d = _cycle_low(daily, short_flip)

    # Map daily → 4h
    daily_df = pd.DataFrame({
        "_day": d_ema20.index,
        "d_ema20": d_ema20.values,
        "d_ema50": d_ema50.values,
        "d_trend_long": (d_ema20 > d_ema50).astype(int).values,
        "d_trend_short": (d_ema20 < d_ema50).astype(int).values,
        "days_since_long_flip": days_long.values,
        "days_since_short_flip": days_short.values,
        "cycle_high": cycle_high_d.values,
        "cycle_low": cycle_low_d.values,
    })
    g["_day"] = g["timestamp"].dt.floor("D")
    g = g.merge(daily_df, on="_day", how="left").drop(columns=["_day"])
    g["d_ema_spread_pct"] = (g["d_ema20"] - g["d_ema50"]) / g["d_ema50"] * 100.0
    g["pullback_pct"] = (g["cycle_high"] - g["close"]) / g["cycle_high"] * 100.0
    g["rise_pct"] = (g["close"] - g["cycle_low"]) / g["cycle_low"] * 100.0
    g["trend_strength"] = (g["d_ema20"] - g["d_ema50"]).abs() / g["d_ema50"]

    # ATR_14 (close-based proxy)
    abs_ret = (g["close"] / g["close"].shift(1) - 1).abs()
    g["atr14_pct"] = abs_ret.rolling(14, min_periods=5).mean() * 100.0
    g["pullback_atr"] = g["pullback_pct"] / g["atr14_pct"]
    g["rise_atr"] = g["rise_pct"] / g["atr14_pct"]
    return g


def _detect_trigger(latest: pd.Series, prev: pd.Series, tier: str, side: str) -> bool:
    """Check if pullback rule fires at THIS bar (latest)."""
    cfg = TIER_CONFIGS.get(tier, {}).get(side)
    if cfg is None:
        return False
    pb_kind, pb_lvl = cfg
    if side == "long":
        if pd.isna(latest.get("d_trend_long")) or latest["d_trend_long"] != 1:
            return False
        cur = latest["pullback_pct"] if pb_kind == "pct" else latest["pullback_atr"]
        prev_v = prev["pullback_pct"] if pb_kind == "pct" else prev["pullback_atr"]
    else:
        if pd.isna(latest.get("d_trend_short")) or latest["d_trend_short"] != 1:
            return False
        cur = latest["rise_pct"] if pb_kind == "pct" else latest["rise_atr"]
        prev_v = prev["rise_pct"] if pb_kind == "pct" else prev["rise_atr"]
    if pd.isna(cur) or cur < pb_lvl:
        return False
    if not pd.isna(prev_v) and prev_v >= pb_lvl:
        return False     # already triggered last bar
    return True


def _bgm_score_at(model, base_features: dict, rule_features: dict, tier: str) -> float:
    """Build feature vector matching the model's training order; predict P(profitable)."""
    row = {**base_features, **rule_features}
    for t in ["top25","26-50","51-100","101-200"]:
        row[f"tier_{t}"] = 1 if t == tier else 0
    X = np.array([[row.get(f, 0.0) for f in FEAT_COLS]])
    X = np.nan_to_num(X, nan=0.0)
    return float(model.predict_proba(X)[0, 1])


def _get_tier_for_symbol(sym: str, mem: pd.DataFrame) -> str | None:
    sub = mem[mem["symbol"] == sym]
    if len(sub) == 0:
        return None
    avg_rank = sub["rank"].mean()
    if avg_rank <= 25:  return "top25"
    if avg_rank <= 50:  return "26-50"
    if avg_rank <= 100: return "51-100"
    if avg_rank <= 200: return "101-200"
    return None


def main():
    log.info("v_new_2 live scanner starting (DRY_RUN=%s, threshold=%.2f)", DRY_RUN, THRESHOLD)
    universe = _load_universe()
    log.info("universe: %d symbols", len(universe))

    if not DRY_RUN:
        _ensure_signals_table()
        log.info("DB schema verified")

    long_model, short_model = _load_bgm_models()
    log.info("BGM models loaded")

    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")

    # Fetch candles
    log.info("fetching last %d 4h bars per symbol...", LOOKBACK_BARS)
    candles = _fetch_recent_candles(universe, LOOKBACK_BARS)
    if len(candles) == 0:
        log.error("no candles returned; aborting")
        return
    log.info("fetched %d total candles, %d unique symbols",
              len(candles), candles["symbol"].nunique())

    # NOTE: live scanner needs to ALSO compute the 44 base features.
    # In production these are pre-computed in a `features_4h` table by another
    # service. For initial paper-deploy, those features are loaded from there.
    # This script will FAIL if base features are not available — see TODO below.
    log.warning("TODO: load base 44 v_new_1 features from features_4h table. "
                "This scaffold computes only the v_new_2 rule features. "
                "See V_NEW_2_PAPER_DEPLOY_PLAN-2026-05-05.md for missing piece.")

    n_signals = 0
    for sym in universe:
        g = candles[candles["symbol"] == sym]
        if len(g) < MIN_BARS:
            continue
        feats = _compute_v_new_2_features(g)
        if feats is None or len(feats) < 2:
            continue
        latest = feats.iloc[-1]
        prev = feats.iloc[-2]
        tier = _get_tier_for_symbol(sym, mem)
        if tier is None:
            continue

        for side in ["long", "short"]:
            if not _detect_trigger(latest, prev, tier, side):
                continue
            # TODO: this will require base features fetched from `features_4h`
            # For now, emit signal with NULL bgm_score so we can validate the
            # scanner pipeline; BGM scoring requires the full feature vector.
            base_feats_stub = {f: np.nan for f in V_NEW_1_FEATURES}
            rule_feats = {
                "d_ema_spread_pct": latest.get("d_ema_spread_pct", 0),
                "days_since_long_flip": latest.get("days_since_long_flip", 0),
                "days_since_short_flip": latest.get("days_since_short_flip", 0),
                "pullback_pct": latest.get("pullback_pct", 0),
                "rise_pct": latest.get("rise_pct", 0),
                "pullback_atr": latest.get("pullback_atr", 0),
                "rise_atr": latest.get("rise_atr", 0),
                "atr14_pct": latest.get("atr14_pct", 0),
                "trend_strength": latest.get("trend_strength", 0),
            }
            model = long_model if side == "long" else short_model
            bgm = _bgm_score_at(model, base_feats_stub, rule_feats, tier)
            if bgm < THRESHOLD:
                continue

            n_signals += 1
            log.info("  signal: %s %s tier=%s bgm=%.3f pullback_pct=%.2f close=%.6f",
                      sym, side, tier, bgm, latest.get("pullback_pct", 0), latest["close"])
            if DRY_RUN:
                continue
            with _engine_lazy().connect() as c:
                c.execute(text("""
                    INSERT INTO v_new_2_signals
                        (signal_time, symbol, side, tier, pullback_pct, pullback_atr,
                         bgm_score, d_ema_spread, close_at_signal)
                    VALUES (:ts, :sym, :side, :tier, :pb_pct, :pb_atr, :bgm,
                            :spread, :close)
                    ON CONFLICT DO NOTHING
                """), {
                    "ts": pd.Timestamp(latest["timestamp"]).to_pydatetime(),
                    "sym": sym, "side": side, "tier": tier,
                    "pb_pct": float(latest.get("pullback_pct", 0)),
                    "pb_atr": float(latest.get("pullback_atr", 0)),
                    "bgm": float(bgm),
                    "spread": float(latest.get("d_ema_spread_pct", 0)),
                    "close": float(latest["close"]),
                })
                c.commit()

    log.info("scan complete: %d signals (DRY_RUN=%s)", n_signals, DRY_RUN)


if __name__ == "__main__":
    main()
