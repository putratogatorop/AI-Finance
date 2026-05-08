"""v_new_2 — Production live scanner (paper-deploy).

Reuses feature computation from v_new_1_shadow_score.py (44 base features)
and layers v_new_2 specifics on top:
  - Daily EMA20/50 + cycle high/low (per coin)
  - Pullback / rise (% and ATR-normalized)
  - Per-tier rule trigger detection (with meme 20% override)
  - BGM scoring with v_new_2 honest models
  - Write rule-trigger bars where bgm_score >= 0.60 to v_new_2_signals

Cron (every 4h, 5min after candle close):
    5 */4 * * * /opt/ai-finance/services/python/.venv/bin/python3 \
        /opt/ai-finance/services/python/scripts/v_new_2_live_scanner_prod.py

Env config:
    DATABASE_URL          — Postgres connection (default localhost)
    V_NEW_2_BGM_THRESHOLD — default 0.60
    V_NEW_2_DRY_RUN       — if "1", logs only, no DB writes
"""
from __future__ import annotations

import json
import logging
import os
import pathlib
import sys
import time
from datetime import UTC, datetime, timedelta

import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

# Reuse all heavy lifting from shadow_score (44 base features, BTC, breadth, CFGI)
sys.path.insert(0, "/opt/ai-finance/services/python/scripts")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
try:
    from v_new_1_shadow_score import (
        compute_per_coin_features,
        compute_btc_features,
        compute_breadth,
        get_cfgi_value,
        fetch_4h_candles,
        _ema as _ema_series,
        CalibratedLGBM,  # noqa: F401  (needed for joblib unpickle of v_new_1 models)
    )
except ImportError as e:
    print(f"FATAL: cannot import from v_new_1_shadow_score: {e}")
    sys.exit(1)

logging.basicConfig(level=logging.INFO,
                     format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("v_new_2_scanner")

ROOT = pathlib.Path("/opt/ai-finance")
if not ROOT.exists():
    ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "services" / "python" / "data" / "v_new_1_v2"
MODELS_DIR = ROOT / "services" / "python" / "models"

DB_URL = os.environ.get("DATABASE_URL",
                          "postgresql://aifinance:6bnn9pcfRpf9xW-CO4wsMmVtyA094nbK1XUQ72Tyi1U@localhost:5432/aifinance")
THRESHOLD = float(os.environ.get("V_NEW_2_BGM_THRESHOLD", "0.60"))
DRY_RUN = os.environ.get("V_NEW_2_DRY_RUN", "0") == "1"

# Tier-aware BGM floors (override `THRESHOLD` per tier/side bucket).
# Validated by 2020-2026 historical sweep over 13,783 long trades:
#   top25  @ 0.50: PF 7.92, WR 67%   (current default — strong)
#   51-100 @ 0.50: PF 4.98, WR 64%   (weaker — lower-liquidity names)
#   51-100 @ 0.70: PF 10.54, WR 76%  (beats top25 default)
#   meme   @ 0.75: PF 26.88, WR 80%  (memes need the strongest filter)
TIER_FLOORS = {
    "meme":     0.75,
    "top25":    0.50,
    "26-50":    0.60,
    "51-100":   0.70,
    "101-200":  0.75,
}


def _tier_floor(tier: str, is_meme: bool) -> float:
    """Resolve effective BGM floor for a (tier, is_meme) bucket."""
    if is_meme:
        return TIER_FLOORS["meme"]
    return TIER_FLOORS.get(tier, THRESHOLD)

# LSTM regime — dynamic threshold adjustment per regime
LSTM_ENABLED = os.environ.get("V_NEW_2_LSTM_ENABLED", "1") == "1"
LSTM_DIR = ROOT / "services" / "python" / "models" / "v_new_2_lstm"
LSTM_SEQ_LEN = 30
# BGM threshold by regime (bear=tighter, bull_mania=more permissive)
REGIME_THRESHOLD = {"bear": 0.70, "neutral": 0.60, "bull_mania": 0.50}
# ATR trail multiplier override for executor (written to signal for executor to read)
REGIME_MEME_TRAIL   = {"bear": 3.0, "neutral": 3.0,  "bull_mania": 5.0}
REGIME_NONMEME_TRAIL = {"bear": 8.0, "neutral": 10.0, "bull_mania": 15.0}
REGIME_NAMES = ["bear", "neutral", "bull_mania"]

EMA_FAST = 20
EMA_SLOW = 50
BARS_PER_DAY = 6
LOOKBACK_BARS = 600   # ~100 days of 4h bars

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
FEAT_COLS_BGM = V_NEW_1_FEATURES + RULE_FEATURES + TIER_DUMMIES

# Per-tier rule configs (from research)
TIER_CONFIGS = {
    "top25":   {"long": ("atr",  2.0), "short": ("atr",  4.0)},
    "26-50":   {"long": ("pct", 15.0), "short": ("pct", 10.0)},
    "51-100":  {"long": ("atr",  3.0), "short": ("atr",  4.0)},
    "101-200": {"long": ("atr",  4.0), "short": ("atr",  4.0)},
}
MEME_PULLBACK = ("pct", 20.0)


# ─── DB helpers ─────────────────────────────────────────────────────────────
_engine = None


def _engine_lazy():
    global _engine
    if _engine is None:
        _engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=300)
    return _engine


# ─── Universe + tier helpers ────────────────────────────────────────────────
def _load_u4_universe() -> set[str]:
    """Load the active universe set. Prefers u5 (memes+ai+defi+top75); falls
    back to legacy u4 (memes+ai+top50) until the scanner regenerates u5."""
    p_u5 = DATA_DIR / "v_new_2_u5_universe.json"
    p_u4 = DATA_DIR / "v_new_2_u4_universe.json"
    p = p_u5 if p_u5.exists() else p_u4
    with open(p) as f:
        return set(json.load(f))


def _load_meme_set() -> set[str]:
    with open(DATA_DIR / "coingecko_categories.json") as f:
        cats = json.load(f)
    return set(cats.get("memes", []))


def _load_tier_map() -> dict[str, str]:
    """Returns symbol → tier label."""
    mem = pd.read_parquet(DATA_DIR / "universe_top100_membership.parquet")
    avg_rank = mem.groupby("symbol")["rank"].mean()
    out = {}
    for sym, rank in avg_rank.items():
        if rank <= 25: out[sym] = "top25"
        elif rank <= 50: out[sym] = "26-50"
        elif rank <= 100: out[sym] = "51-100"
        elif rank <= 200: out[sym] = "101-200"
    return out


# ─── v_new_2 specific feature computation ───────────────────────────────────
def _compute_v_new_2_rule_features(df_4h: pd.DataFrame) -> pd.DataFrame:
    """Add daily EMA20/50, cycle high/low, pullback/rise (% and ATR), trend_strength.
    Input: 4h candle df with timestamp (DatetimeIndex or column) and close column.
    Output: same df with v_new_2 columns appended."""
    df = df_4h.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.DatetimeIndex(pd.to_datetime(df["timestamp"], utc=True)))
        else:
            raise ValueError("df_4h needs timestamp index or column")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")

    daily = df["close"].resample("1D").last().dropna()
    if len(daily) < EMA_SLOW + 5:
        # Not enough history — return df with NaN columns
        for c in ["d_ema20","d_ema50","d_trend_long","d_trend_short",
                  "days_since_long_flip","days_since_short_flip",
                  "cycle_high","cycle_low","d_ema_spread_pct",
                  "pullback_pct","rise_pct","pullback_atr","rise_atr",
                  "trend_strength","atr14_pct"]:
            df[c] = np.nan
        return df

    d_ema20 = daily.ewm(span=EMA_FAST, adjust=False).mean()
    d_ema50 = daily.ewm(span=EMA_SLOW, adjust=False).mean()
    long_flip = ((d_ema20 > d_ema50).astype(int).diff() > 0).astype(int)
    short_flip = ((d_ema20 < d_ema50).astype(int).diff() > 0).astype(int)

    # Days since each flip
    def _days_since(flip):
        out = np.full(len(flip), np.nan)
        last = -1_000_000
        for i in range(len(flip)):
            if flip.iloc[i] == 1: last = i
            if last >= 0: out[i] = i - last
        return pd.Series(out, index=flip.index)

    days_long = _days_since(long_flip)
    days_short = _days_since(short_flip)

    # Cycle high/low (running since last flip)
    def _cycle(closes, flip, op):
        out = np.full(len(closes), np.nan)
        running = closes.iloc[0]
        for i in range(len(closes)):
            running = closes.iloc[i] if flip.iloc[i] == 1 else (
                op(running, closes.iloc[i])
            )
            out[i] = running
        return pd.Series(out, index=closes.index)

    cycle_high_d = _cycle(daily, long_flip, max)
    cycle_low_d = _cycle(daily, short_flip, min)

    # Daily-level dataframe
    daily_df = pd.DataFrame({
        "_day": d_ema20.index.date,
        "d_ema20": d_ema20.values,
        "d_ema50": d_ema50.values,
        "d_trend_long": (d_ema20 > d_ema50).astype(int).values,
        "d_trend_short": (d_ema20 < d_ema50).astype(int).values,
        "days_since_long_flip": days_long.values,
        "days_since_short_flip": days_short.values,
        "cycle_high": cycle_high_d.values,
        "cycle_low": cycle_low_d.values,
    })
    df["_day"] = df.index.date
    saved_idx = df.index
    df = df.merge(daily_df, on="_day", how="left").drop(columns=["_day"])
    df.index = saved_idx

    df["d_ema_spread_pct"] = (df["d_ema20"] - df["d_ema50"]) / df["d_ema50"] * 100.0
    df["pullback_pct"] = (df["cycle_high"] - df["close"]) / df["cycle_high"] * 100.0
    df["rise_pct"] = (df["close"] - df["cycle_low"]) / df["cycle_low"] * 100.0
    df["trend_strength"] = (df["d_ema20"] - df["d_ema50"]).abs() / df["d_ema50"]

    abs_ret = (df["close"] / df["close"].shift(1) - 1).abs()
    df["atr14_pct"] = abs_ret.rolling(14, min_periods=5).mean() * 100.0
    df["pullback_atr"] = df["pullback_pct"] / df["atr14_pct"]
    df["rise_atr"] = df["rise_pct"] / df["atr14_pct"]

    return df


def _detect_trigger(latest, prev, tier, side, is_meme) -> bool:
    """Returns True if rule fires THIS bar."""
    if is_meme:
        pb_kind, pb_lvl = MEME_PULLBACK
    else:
        cfg = TIER_CONFIGS.get(tier, {}).get(side)
        if cfg is None: return False
        pb_kind, pb_lvl = cfg

    if side == "long":
        if pd.isna(latest.get("d_trend_long")) or latest["d_trend_long"] != 1:
            return False
        cur = latest.get("pullback_pct") if pb_kind == "pct" else latest.get("pullback_atr")
        prev_v = prev.get("pullback_pct") if pb_kind == "pct" else prev.get("pullback_atr")
    else:
        if pd.isna(latest.get("d_trend_short")) or latest["d_trend_short"] != 1:
            return False
        cur = latest.get("rise_pct") if pb_kind == "pct" else latest.get("rise_atr")
        prev_v = prev.get("rise_pct") if pb_kind == "pct" else prev.get("rise_atr")
    if pd.isna(cur) or cur < pb_lvl:
        return False
    if not pd.isna(prev_v) and prev_v >= pb_lvl:
        return False
    return True


# ─── BGM scoring ────────────────────────────────────────────────────────────
# ─── LSTM regime loader ──────────────────────────────────────────────────────
_lstm_cache: dict | None = None


def _load_lstm_regime():
    global _lstm_cache
    if _lstm_cache is not None:
        return _lstm_cache
    try:
        import torch
        import torch.nn as nn

        class _LSTMRegime(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                                    batch_first=True,
                                    dropout=dropout if num_layers > 1 else 0.0)
                self.head = nn.Sequential(
                    nn.Linear(hidden_size, 32), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(32, 3),
                )
            def forward(self, x):
                out, _ = self.lstm(x)
                return self.head(out[:, -1, :])

        with open(LSTM_DIR / "meta.json") as f:
            cfg = json.load(f)
        scaler = joblib.load(LSTM_DIR / "scaler.pkl")
        model = _LSTMRegime(cfg["n_features"], cfg["hidden_size"],
                             cfg["num_layers"], 0.0)
        model.load_state_dict(torch.load(LSTM_DIR / "regime_lstm.pt",
                                          map_location="cpu"))
        model.eval()
        _lstm_cache = {"model": model, "scaler": scaler,
                       "n_features": cfg["n_features"], "torch": torch}
        log.info("LSTM regime model loaded (hidden=%d layers=%d)",
                  cfg["hidden_size"], cfg["num_layers"])
    except Exception as e:
        log.warning("LSTM regime load failed (%s) — using static threshold", e)
        _lstm_cache = {}
    return _lstm_cache


def _predict_regime(v2_df: pd.DataFrame, btc_feats: pd.DataFrame,
                    side: str) -> tuple[str, list[float]]:
    """Build 30-bar sequence for this symbol and return (regime_name, probs[3]).

    Features (6 per bar, matching training):
      close_ret, atr14_pct, d_ema_spread_pct, trend_strength, btc_ret_4h, side_sign
    """
    lstm = _load_lstm_regime()
    if not lstm:
        return "neutral", [0.0, 1.0, 0.0]
    try:
        import torch
        seq_df = v2_df.tail(LSTM_SEQ_LEN).copy()
        side_sign = 1.0 if side == "long" else -1.0

        cr  = seq_df["close"].pct_change().fillna(0.0).values
        atr = seq_df["atr14_pct"].fillna(0.02).values
        ema = seq_df["d_ema_spread_pct"].fillna(0.0).values
        ts_str = seq_df["trend_strength"].fillna(0.0).values

        # BTC 4h return aligned to sequence timestamps
        btc_ret = btc_feats["btc_24h_return"].reindex(
            seq_df.index, method="ffill").fillna(0.0).values / 4.0  # daily→4h approx

        side_arr = np.full(len(seq_df), side_sign)
        seq = np.stack([cr, atr, ema, ts_str, btc_ret, side_arr], axis=1).astype(np.float32)

        # Pad if shorter than SEQ_LEN
        if len(seq) < LSTM_SEQ_LEN:
            pad = LSTM_SEQ_LEN - len(seq)
            seq = np.pad(seq, ((pad, 0), (0, 0)), mode="constant")

        seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale + predict
        _n_f = lstm["n_features"]
        seq_scaled = lstm["scaler"].transform(seq.reshape(-1, _n_f)).reshape(1, LSTM_SEQ_LEN, _n_f)
        with torch.no_grad():
            logits = lstm["model"](torch.tensor(seq_scaled, dtype=torch.float32))
            probs = torch.softmax(logits, dim=1).numpy()[0].tolist()

        regime_idx = int(np.argmax(probs))
        return REGIME_NAMES[regime_idx], probs
    except Exception as e:
        log.debug("LSTM regime predict error: %s", e)
        return "neutral", [0.0, 1.0, 0.0]


_bgm_models_cache = None


def _load_bgm_models():
    global _bgm_models_cache
    if _bgm_models_cache is None:
        long_model = joblib.load(MODELS_DIR / "v_new_2_bgm_long_honest" / "model.joblib")
        short_model = joblib.load(MODELS_DIR / "v_new_2_bgm_short_honest" / "model.joblib")
        _bgm_models_cache = {"long": long_model, "short": short_model}
    return _bgm_models_cache


def _score_bgm(row_dict: dict, tier: str, side: str) -> float:
    """Build feature vector matching v_new_2 BGM training; return P(profitable)."""
    feats = {f: row_dict.get(f, 0.0) for f in V_NEW_1_FEATURES + RULE_FEATURES}
    for t in ["top25","26-50","51-100","101-200"]:
        feats[f"tier_{t}"] = 1 if t == tier else 0
    X = np.array([[feats.get(f, 0.0) for f in FEAT_COLS_BGM]])
    X = np.nan_to_num(X, nan=0.0)
    model = _load_bgm_models()[side]
    return float(model.predict_proba(X)[0, 1])


# ─── Main scanner loop ──────────────────────────────────────────────────────
def main():
    log.info("v_new_2 live scanner — start (DRY_RUN=%s threshold=%.2f)", DRY_RUN, THRESHOLD)
    universe = _load_u4_universe()
    memes = _load_meme_set()
    tier_map = _load_tier_map()
    log.info("universe: %d syms, memes: %d", len(universe), len(memes & universe))

    engine = _engine_lazy()

    # Per-coin features need BTC features + breadth + CFGI (cross-sectional)
    log.info("fetching BTC + breadth + CFGI context...")
    btc_df = fetch_4h_candles(engine, "BTCUSDT", n_bars=LOOKBACK_BARS)
    if btc_df is None or len(btc_df) < 200:
        log.error("could not fetch BTC candles; aborting")
        return
    btc_feats = compute_btc_features(btc_df)
    cfgi_value = get_cfgi_value(engine)

    # Build close pivot for breadth (we need many coins' close history)
    log.info("fetching close history for breadth...")
    close_pivot_parts = []
    for sym in universe:
        try:
            d = fetch_4h_candles(engine, sym, n_bars=LOOKBACK_BARS)
            if d is None or len(d) < 50: continue
            ts = pd.DatetimeIndex(pd.to_datetime(d["timestamp"], utc=True))
            s = pd.Series(d["close"].astype(float).values, index=ts, name=sym)
            close_pivot_parts.append(s)
        except Exception:
            continue
    if close_pivot_parts:
        close_pivot = pd.concat(close_pivot_parts, axis=1).sort_index()
        breadth_df = compute_breadth(close_pivot)
    else:
        log.error("no close data for breadth; aborting")
        return

    # Per-coin scanning
    n_signals = 0
    n_processed = 0
    n_trend_long = 0
    n_trend_short = 0
    n_triggers_long = 0
    n_triggers_short = 0
    n_bgm_pass = 0
    for sym in sorted(universe):
        tier = tier_map.get(sym)
        if tier is None: continue
        is_meme = sym in memes
        try:
            df_4h = fetch_4h_candles(engine, sym, n_bars=LOOKBACK_BARS)
            if df_4h is None or len(df_4h) < EMA_SLOW * BARS_PER_DAY + 10:
                continue
            n_processed += 1

            # Compute 44 base features
            base_feats = compute_per_coin_features(df_4h)
            if len(base_feats) < 2:
                continue

            # Add v_new_2 rule features
            v2_df = _compute_v_new_2_rule_features(df_4h)
            # Align indexes (base_feats may have different index)
            v2_df = v2_df.reindex(base_feats.index, method="ffill")
            for c in RULE_FEATURES + ["d_trend_long", "d_trend_short", "close",
                                        "cycle_high", "cycle_low"]:
                if c in v2_df.columns:
                    base_feats[c] = v2_df[c].values

            # Cross-sectional features at latest timestamp
            latest_ts = base_feats.index[-1]
            base_feats["btc_above_4h_ema50"] = btc_feats["btc_above_4h_ema50"].reindex([latest_ts], method="ffill").values[0]
            base_feats["btc_24h_return"] = btc_feats["btc_24h_return"].reindex([latest_ts], method="ffill").values[0]
            base_feats["btc_realized_vol_z"] = btc_feats["btc_realized_vol_z"].reindex([latest_ts], method="ffill").values[0]
            base_feats["btc_score"] = btc_feats["btc_score"].reindex([latest_ts], method="ffill").values[0]
            base_feats["breadth_up"] = breadth_df["breadth_up"].reindex([latest_ts], method="ffill").values[0]
            base_feats["breadth_down"] = breadth_df["breadth_down"].reindex([latest_ts], method="ffill").values[0]
            base_feats["cfgi_value"] = cfgi_value

            # Latest two rows for trigger detection
            latest = base_feats.iloc[-1]
            prev = base_feats.iloc[-2]

            if latest.get("d_trend_long") == 1: n_trend_long += 1
            if latest.get("d_trend_short") == 1: n_trend_short += 1

            # LSTM regime: run once per symbol (shared for long+short)
            if LSTM_ENABLED:
                regime, regime_probs = _predict_regime(v2_df, btc_feats, "long")
            else:
                regime, regime_probs = "neutral", [0.0, 1.0, 0.0]

            for side in ["long", "short"]:
                if not _detect_trigger(latest, prev, tier, side, is_meme):
                    continue
                if side == "long": n_triggers_long += 1
                else: n_triggers_short += 1

                # Effective threshold = max(tier floor, regime floor).
                # Tier floor protects against lower-liquidity false positives;
                # regime floor tightens during bear and loosens in bull_mania.
                tier_floor = _tier_floor(tier, is_meme)
                regime_floor = REGIME_THRESHOLD.get(regime, THRESHOLD) if LSTM_ENABLED else THRESHOLD
                eff_threshold = max(tier_floor, regime_floor)

                bgm = _score_bgm(latest.to_dict(), tier, side)
                if bgm >= eff_threshold: n_bgm_pass += 1
                if bgm < eff_threshold:
                    log.info("  near-miss: %s %s tier=%s meme=%s bgm=%.3f thr=%.2f (tier=%.2f regime=%s/%.2f) pb=%.2f",
                              sym, side, tier, is_meme, bgm, eff_threshold,
                              tier_floor, regime, regime_floor,
                              float(latest.get("pullback_pct" if side == "long" else "rise_pct", 0) or 0))
                    continue
                n_signals += 1

                # Regime-adjusted ATR trail multipliers (for executor to read)
                regime_meme_trail    = REGIME_MEME_TRAIL.get(regime, 3.0)
                regime_nonmeme_trail = REGIME_NONMEME_TRAIL.get(regime, 10.0)

                log.info("  SIGNAL: %s %s tier=%s meme=%s bgm=%.3f regime=%s(%.2f/%.2f/%.2f) "
                          "thr=%.2f trail=%.0f/%.0f close=%.6f",
                          sym, side, tier, is_meme, bgm, regime,
                          regime_probs[0], regime_probs[1], regime_probs[2],
                          eff_threshold, regime_meme_trail, regime_nonmeme_trail,
                          float(latest.get("close", 0)))
                if DRY_RUN:
                    continue
                with engine.connect() as c:
                    c.execute(text("""
                        INSERT INTO v_new_2_signals
                            (signal_time, symbol, side, tier, is_meme,
                             pullback_pct, pullback_atr, bgm_score,
                             d_ema_spread, close_at_signal,
                             lstm_regime, lstm_p_bear, lstm_p_neutral, lstm_p_bull,
                             regime_meme_trail, regime_nonmeme_trail)
                        VALUES (:ts, :sym, :side, :tier, :ismeme,
                                :pb_pct, :pb_atr, :bgm, :spread, :close,
                                :regime, :p_bear, :p_neutral, :p_bull,
                                :meme_trail, :nonmeme_trail)
                        ON CONFLICT (signal_time, symbol, side) DO NOTHING
                    """), {
                        "ts": pd.Timestamp(latest_ts).to_pydatetime(),
                        "sym": sym, "side": side, "tier": tier, "ismeme": is_meme,
                        "pb_pct": float(latest.get("pullback_pct", 0) or 0),
                        "pb_atr": float(latest.get("pullback_atr", 0) or 0),
                        "bgm": float(bgm),
                        "spread": float(latest.get("d_ema_spread_pct", 0) or 0),
                        "close": float(latest.get("close", 0)),
                        "regime": regime,
                        "p_bear": float(regime_probs[0]),
                        "p_neutral": float(regime_probs[1]),
                        "p_bull": float(regime_probs[2]),
                        "meme_trail": float(regime_meme_trail),
                        "nonmeme_trail": float(regime_nonmeme_trail),
                    })
                    c.commit()
        except Exception as e:
            log.warning("  %s: scan failed: %s", sym, e)
            continue

    log.info("scan complete: %d processed, %d signals (DRY_RUN=%s)",
              n_processed, n_signals, DRY_RUN)
    log.info("  diag: trend_long=%d trend_short=%d triggers_long=%d triggers_short=%d bgm_pass=%d",
              n_trend_long, n_trend_short, n_triggers_long, n_triggers_short, n_bgm_pass)


if __name__ == "__main__":
    main()
