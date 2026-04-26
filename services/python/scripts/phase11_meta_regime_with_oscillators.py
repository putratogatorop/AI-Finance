"""Phase 11 — meta-regime classifier with BTC oscillator features.

Adds 5 BTC oscillator features to the Phase 10.2 trend-only set, motivated
by the user's observation that the catastrophic months span TWO opposite
sub-regimes that the trend features can't separate:

  - Strong-bear CONTINUATION (best months: 2024-08 PF 3.32, 2025-10 PF 2.66)
  - Strong-bear WHIPSAW (worst months: 2023-09 PF 0.30)

Both have similar BTC trend / momentum profiles. Oscillator extremes
(RSI < 20, KDJ J < -10, BB %B < 0.05, capitulation volume) are
classical signals that distinguish "exhausted, snap-back coming" from
"trending, room to run."

This script:
  1. Computes 12 BTC features (7 trend + 5 oscillator) at every month-start.
  2. Trains a v2 meta-regime LR classifier on those 12 features.
  3. Applies the gate to v2 SHORT trades, recomputes ship-gate metrics.
  4. Also tests 2 hand-crafted whipsaw guards as a sanity check.

Run from services/python/:
    python scripts/phase11_meta_regime_with_oscillators.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.indicators import ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
V2_RUN_DIR = (
    REPO_ROOT / "services/python/results"
    / "backtest_bigmover_combined_v2_outcome_0582b799_20260426T033736Z"
)
TRADES_CSV = V2_RUN_DIR / "trades.csv"
OUT_PATH = V2_RUN_DIR / "phase11_meta_regime_with_oscillators.json"
MODELS_DIR = REPO_ROOT / "services/python/models"
MODEL_OUT = MODELS_DIR / "meta_regime_v2_oscillators.joblib"
META_OUT = MODELS_DIR / "meta_regime_v2_oscillators_meta.json"

TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
WINDOW_MONTHS = 3
LR_C = 0.1
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3

# Trend features (from Phase 10.2)
TREND_FEATURES = [
    "btc_30d_return",
    "btc_90d_return",
    "btc_180d_return",
    "btc_30d_realized_vol",
    "btc_vs_sma200_pct",
    "btc_vs_ema26_weekly_pct",
    "btc_macd_4h_signal_spread_norm",
]

# NEW: oscillator features for whipsaw vs continuation discrimination
OSCILLATOR_FEATURES = [
    "btc_rsi14_daily",
    "btc_rsi14_daily_5bar_delta",
    "btc_kdj_j_daily",
    "btc_bbands_pct_b_daily",
    "btc_volume_5d_zscore",
]

FEATURE_NAMES = TREND_FEATURES + OSCILLATOR_FEATURES


# --- Helpers ----------------------------------------------------------------


def _pf(pnls: np.ndarray) -> float:
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _build_btc_panels(candles: pd.DataFrame) -> dict[str, pd.Series]:
    """Pre-compute all BTC feature series at 15m granularity (forward-filled)."""
    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    close_15m = btc["close"].astype(float)
    high_15m = btc["high"].astype(float)
    low_15m = btc["low"].astype(float)
    volume_15m = btc["volume"].astype(float)
    btc_daily_close = close_15m.resample("1D", label="right", closed="right").last().dropna()
    btc_daily_high = high_15m.resample("1D", label="right", closed="right").max().dropna()
    btc_daily_low = low_15m.resample("1D", label="right", closed="right").min().dropna()
    btc_daily_volume = volume_15m.resample("1D", label="right", closed="right").sum().dropna()

    panels: dict[str, pd.Series] = {}

    # Trend features (Phase 10.2 set, all shifted by 1 daily/4h/weekly bar)
    panels["btc_30d_return"] = (
        btc_daily_close.pct_change(30).shift(1).reindex(close_15m.index, method="ffill")
    )
    panels["btc_90d_return"] = (
        btc_daily_close.pct_change(90).shift(1).reindex(close_15m.index, method="ffill")
    )
    panels["btc_180d_return"] = (
        btc_daily_close.pct_change(180).shift(1).reindex(close_15m.index, method="ffill")
    )
    btc_logret = np.log(btc_daily_close / btc_daily_close.shift(1))
    panels["btc_30d_realized_vol"] = (
        btc_logret.rolling(30, min_periods=30).std(ddof=0).shift(1).reindex(close_15m.index, method="ffill")
    )
    sma200 = btc_daily_close.rolling(200, min_periods=200).mean().shift(1)
    panels["btc_vs_sma200_pct"] = (
        ((btc_daily_close - sma200) / sma200).reindex(close_15m.index, method="ffill")
    )
    btc_weekly = close_15m.resample("1W", label="right", closed="right").last()
    weekly_ema26 = _ema(btc_weekly, 26).shift(1)
    panels["btc_vs_ema26_weekly_pct"] = (
        ((btc_weekly - weekly_ema26) / weekly_ema26).reindex(close_15m.index, method="ffill")
    )
    btc_4h = close_15m.resample("4h", label="right", closed="right").last()
    m4h = _macd(btc_4h, 12, 26, 9)
    spread_4h = ((m4h["macd"] - m4h["signal"]).shift(1) / btc_4h.shift(1))
    panels["btc_macd_4h_signal_spread_norm"] = (
        spread_4h.reindex(close_15m.index, method="ffill")
    )

    # NEW oscillator features
    rsi_daily = _rsi(btc_daily_close, 14).shift(1)
    panels["btc_rsi14_daily"] = rsi_daily.reindex(close_15m.index, method="ffill")
    panels["btc_rsi14_daily_5bar_delta"] = (
        (rsi_daily - rsi_daily.shift(5)).reindex(close_15m.index, method="ffill")
    )

    kdj_daily = _kdj(btc_daily_high, btc_daily_low, btc_daily_close, n=9, k_smooth=3, d_smooth=3)
    j_daily = kdj_daily["j"].shift(1)
    panels["btc_kdj_j_daily"] = j_daily.reindex(close_15m.index, method="ffill")

    bb_period = 20
    bb_std = 2.0
    sma20 = btc_daily_close.rolling(bb_period, min_periods=bb_period).mean()
    std20 = btc_daily_close.rolling(bb_period, min_periods=bb_period).std(ddof=0)
    bb_upper = (sma20 + bb_std * std20).shift(1)
    bb_lower = (sma20 - bb_std * std20).shift(1)
    bb_close = btc_daily_close.shift(1)
    bb_pct_b = (bb_close - bb_lower) / (bb_upper - bb_lower).where(
        (bb_upper - bb_lower) > 0
    )
    panels["btc_bbands_pct_b_daily"] = bb_pct_b.reindex(close_15m.index, method="ffill")

    # Volume 5-day z-score: ratio of last-5-day volume sum vs prior 30-day rolling
    vol_5d_sum = btc_daily_volume.rolling(5, min_periods=5).sum()
    vol_5d_mu = vol_5d_sum.rolling(30, min_periods=30).mean()
    vol_5d_sd = vol_5d_sum.rolling(30, min_periods=30).std(ddof=0)
    vol_z = ((vol_5d_sum - vol_5d_mu) / vol_5d_sd.where(vol_5d_sd > 0)).shift(1)
    panels["btc_volume_5d_zscore"] = vol_z.reindex(close_15m.index, method="ffill")

    return panels


def _features_at(panels: dict, ts: pd.Timestamp) -> dict:
    out = {}
    for name in FEATURE_NAMES:
        s = panels[name]
        try:
            v = s.asof(ts)
        except KeyError:
            v = float("nan")
        out[name] = float(v) if pd.notna(v) else float("nan")
    return out


def _walkforward_pf(short_trades, start_ts, end_ts):
    et = pd.to_datetime(short_trades["entry_time"], utc=True)
    sub = short_trades[(et >= start_ts) & (et < end_ts)]
    if len(sub) < 30:
        return None
    return _pf(sub["sized_pnl"].to_numpy())


def _walkforward_rollup(df, pnl_col="sized_pnl"):
    if len(df) == 0:
        return {"n_folds": 0, "rollup_pf": {}}
    et = pd.to_datetime(df["entry_time"], utc=True)
    s = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    fold_w = relativedelta(months=WF_FOLD_MONTHS)
    month = relativedelta(months=1)
    folds = []
    while s + fold_w <= end_global:
        sub = df[(et >= s) & (et < s + fold_w)]
        folds.append({
            "fold_start": str(s.date()),
            "trades": int(len(sub)),
            "pf": _pf(sub[pnl_col].to_numpy()),
        })
        s += month
    pfs = np.array([f["pf"] for f in folds if f["trades"] >= 30 and np.isfinite(f["pf"])])
    rollup = {}
    if len(pfs):
        rollup = {
            "n_folds_used": int(len(pfs)),
            "p5": float(np.percentile(pfs, 5)),
            "p50": float(np.percentile(pfs, 50)),
            "p95": float(np.percentile(pfs, 95)),
            "p_lt_1": float((pfs < 1.0).mean()),
        }
    return {"n_folds": len(folds), "rollup_pf": rollup}


def _monte_carlo(pnls, n=MC_N_RESAMPLES):
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {"n": 0}
    rng = np.random.default_rng(RANDOM_SEED)
    pfs = np.array([_pf(rng.choice(pnls, len(pnls), replace=True)) for _ in range(n)])
    finite = pfs[np.isfinite(pfs)]
    return {
        "n": int(len(pnls)),
        "point_pf": _pf(pnls),
        "p5": float(np.percentile(finite, 5)) if len(finite) else float("nan"),
        "p50": float(np.percentile(finite, 50)) if len(finite) else float("nan"),
        "p95": float(np.percentile(finite, 95)) if len(finite) else float("nan"),
    }


# --- Main -------------------------------------------------------------------


def main() -> None:
    started = time.monotonic()

    print(f"loading v2 trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    threshold_predictor = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    short_full = df[
        (df["direction"] == "short")
        & (df["proba"] >= threshold_predictor)
        & (df["pos_scale"] > 0)
    ].copy()
    print(f"  SHORT ml_on+sized entries: {len(short_full):,}", flush=True)

    print(f"\nloading BTC candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    print("building BTC trend + oscillator panels (12 features)...", flush=True)
    panels = _build_btc_panels(candles)

    # ---- Window-level dataset ---------------------------------------------
    start = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    window_w = relativedelta(months=WINDOW_MONTHS)
    month = relativedelta(months=1)
    rows = []
    s = start
    while s + window_w <= end_global:
        feats = _features_at(panels, s)
        fwd_pf = _walkforward_pf(short_full, s, s + window_w)
        rows.append({
            "window_start": str(s.date()),
            **feats,
            "fwd_pf": fwd_pf,
        })
        s += month
    win_df = pd.DataFrame(rows)
    finite = win_df[FEATURE_NAMES].notna().all(axis=1) & win_df["fwd_pf"].notna()
    print(f"  windows total: {len(win_df)}  with complete features+label: {int(finite.sum())}",
          flush=True)
    win_df = win_df[finite].reset_index(drop=True)
    win_df["label"] = (win_df["fwd_pf"] >= 1.0).astype(int)
    pos_rate = float((win_df["label"] == 1).mean())
    print(f"  positive rate (PF >= 1.0): {pos_rate:.1%}", flush=True)

    win_df["window_start_ts"] = pd.to_datetime(win_df["window_start"]).dt.tz_localize("UTC")
    is_train = win_df["window_start_ts"] < TRAIN_OOT_BOUNDARY
    is_oot = ~is_train
    train_X = win_df.loc[is_train, FEATURE_NAMES].to_numpy(dtype=float)
    train_y = win_df.loc[is_train, "label"].to_numpy(dtype=int)
    oot_X = win_df.loc[is_oot, FEATURE_NAMES].to_numpy(dtype=float)
    oot_y = win_df.loc[is_oot, "label"].to_numpy(dtype=int)
    print(f"  train windows: {len(train_X)}  oot windows: {len(oot_X)}", flush=True)

    # ---- Train v2 LR with strong L2 ---------------------------------------
    print(f"\n=== Training meta-regime v2 LR (C={LR_C}, 12 features) ===")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced",
        )),
    ])
    pipe.fit(train_X, train_y)
    train_proba = pipe.predict_proba(train_X)[:, 1]
    train_auc = roc_auc_score(train_y, train_proba) if len(np.unique(train_y)) >= 2 else float("nan")
    if len(oot_X) > 0 and len(np.unique(oot_y)) >= 2:
        oot_proba = pipe.predict_proba(oot_X)[:, 1]
        oot_auc = roc_auc_score(oot_y, oot_proba)
    else:
        oot_auc = float("nan")
    print(f"  train AUC: {train_auc:.4f}    oot AUC: {oot_auc:.4f}", flush=True)
    coef_dict = {n: float(c) for n, c in zip(FEATURE_NAMES, pipe.named_steps["clf"].coef_[0])}
    print("  feature coefficients (scaled):", flush=True)
    for n in FEATURE_NAMES:
        print(f"    {n:<40}  {coef_dict[n]:+.3f}", flush=True)

    joblib.dump(pipe, MODEL_OUT)
    META_OUT.write_text(json.dumps({
        "model": "logistic_regression",
        "feature_names": FEATURE_NAMES,
        "label": "fwd_3mo_short_only_pf_ge_1.0",
        "C": LR_C,
        "train_auc": float(train_auc) if np.isfinite(train_auc) else None,
        "oot_auc": float(oot_auc) if np.isfinite(oot_auc) else None,
        "train_n_windows": int(len(train_X)),
        "oot_n_windows": int(len(oot_X)),
        "feature_coefficients_scaled": coef_dict,
        "snapshot_date": SNAPSHOT_DATE,
    }, indent=2, default=str))

    # ---- Apply ML gate to v2 SHORT trades --------------------------------
    threshold_meta = 0.5
    short_full["meta_features_ts"] = (
        short_full["entry_time"].dt.tz_convert("UTC").dt.to_period("M").dt.start_time
        .dt.tz_localize("UTC")
    )
    feat_dicts = [_features_at(panels, ts) for ts in short_full["meta_features_ts"]]
    feat_arr = np.array([[d[n] for n in FEATURE_NAMES] for d in feat_dicts], dtype=float)
    finite_mask = np.isfinite(feat_arr).all(axis=1)
    proba = np.full(len(short_full), np.nan, dtype=float)
    if finite_mask.any():
        proba[finite_mask] = pipe.predict_proba(feat_arr[finite_mask])[:, 1]
    short_full["meta_proba"] = proba
    short_full["meta_pass"] = short_full["meta_proba"] >= threshold_meta

    ml_gated = short_full[short_full["meta_pass"]]
    train_g = ml_gated[ml_gated["period"] == "train"]
    oot_g = ml_gated[ml_gated["period"] == "oot"]
    wf = _walkforward_rollup(ml_gated)
    mc = _monte_carlo(oot_g["sized_pnl"].to_numpy())

    print("\n=== Ship gate (v2 SHORT + ml + sizing + ML meta-gate v2) ===")
    print(f"  TRAIN gated: n={len(train_g):>5}  PF={_pf(train_g['sized_pnl'].to_numpy()):.3f}",
          flush=True)
    print(f"  OOT gated:   n={len(oot_g):>5}  PF={_pf(oot_g['sized_pnl'].to_numpy()):.3f}",
          flush=True)
    print(f"  WF p5={wf['rollup_pf'].get('p5', float('nan')):.3f}  "
          f"p50={wf['rollup_pf'].get('p50', float('nan')):.3f}  "
          f"P(loss)={wf['rollup_pf'].get('p_lt_1', float('nan'))*100:.1f}%",
          flush=True)
    print(f"  MC p5={mc.get('p5', float('nan')):.3f}  "
          f"p50={mc.get('p50', float('nan')):.3f}", flush=True)

    oot_pf = _pf(oot_g["sized_pnl"].to_numpy())
    ml_ship = {
        "oot_pf_ge_130": bool(np.isfinite(oot_pf) and oot_pf >= 1.30),
        "oot_n_ge_200": bool(len(oot_g) >= 200),
        "wf_p5_gt_1": bool(wf["rollup_pf"].get("p5", -1) > 1.0),
        "mc_p5_gt_1": bool(np.isfinite(mc.get("p5", float("nan"))) and mc.get("p5", -1) > 1.0),
    }
    ml_ship["overall"] = all(ml_ship.values())
    print(f"  → {'SHIP' if ml_ship['overall'] else 'DO NOT SHIP'}", flush=True)

    # ---- Hand-crafted whipsaw guards (sanity check) ----------------------
    print("\n=== Hand-crafted whipsaw guards on v2 SHORT trades ===")
    rules = {}

    def _eval_rule(rule_name, mask):
        gated = short_full[mask].copy()
        train_g = gated[gated["period"] == "train"]
        oot_g = gated[gated["period"] == "oot"]
        wf_r = _walkforward_rollup(gated)
        mc_r = _monte_carlo(oot_g["sized_pnl"].to_numpy())
        result = {
            "train_n": int(len(train_g)), "train_pf": _pf(train_g["sized_pnl"].to_numpy()),
            "oot_n": int(len(oot_g)), "oot_pf": _pf(oot_g["sized_pnl"].to_numpy()),
            "wf_p5": wf_r["rollup_pf"].get("p5"),
            "wf_p50": wf_r["rollup_pf"].get("p50"),
            "wf_p_lt_1": wf_r["rollup_pf"].get("p_lt_1"),
            "mc_p5": mc_r.get("p5"),
        }
        rules[rule_name] = result
        wf_p5 = result["wf_p5"]
        mc_p5 = result["mc_p5"]
        wf_p5_str = f"{wf_p5:.3f}" if wf_p5 is not None and np.isfinite(wf_p5) else "nan"
        mc_p5_str = f"{mc_p5:.3f}" if mc_p5 is not None and np.isfinite(mc_p5) else "nan"
        oot_pf_str = f"{result['oot_pf']:.3f}" if np.isfinite(result['oot_pf']) else "nan"
        print(f"  {rule_name:<60}  TRAIN n={result['train_n']:>5}  "
              f"OOT n={result['oot_n']:>5}  OOT PF={oot_pf_str}  "
              f"WF p5={wf_p5_str}  MC p5={mc_p5_str}",
              flush=True)

    # Compute per-trade BTC oscillator features for hand-crafted rules
    short_full["btc_rsi14_daily"] = panels["btc_rsi14_daily"].reindex(
        short_full["entry_time"]
    ).to_numpy()
    short_full["btc_kdj_j_daily"] = panels["btc_kdj_j_daily"].reindex(
        short_full["entry_time"]
    ).to_numpy()
    short_full["btc_bbands_pct_b_daily"] = panels["btc_bbands_pct_b_daily"].reindex(
        short_full["entry_time"]
    ).to_numpy()
    short_full["btc_volume_5d_zscore"] = panels["btc_volume_5d_zscore"].reindex(
        short_full["entry_time"]
    ).to_numpy()

    # Baseline (no rule) — for reference
    _eval_rule("R0_baseline_no_rule", pd.Series([True] * len(short_full), index=short_full.index))
    # Rule 1: skip when BTC RSI < 20 (extreme oversold)
    _eval_rule(
        "R1_skip_rsi_lt_20",
        ~(short_full["btc_rsi14_daily"] < 20),
    )
    # Rule 2: skip when BTC KDJ J < -10 (extreme oversold)
    _eval_rule(
        "R2_skip_kdj_j_lt_neg10",
        ~(short_full["btc_kdj_j_daily"] < -10),
    )
    # Rule 3: skip when BTC %B < 0.05 (touching lower band)
    _eval_rule(
        "R3_skip_bb_pct_b_lt_0p05",
        ~(short_full["btc_bbands_pct_b_daily"] < 0.05),
    )
    # Rule 4: skip when BTC volume 5d z-score > 2.5 (capitulation spike)
    _eval_rule(
        "R4_skip_vol_z_gt_2p5",
        ~(short_full["btc_volume_5d_zscore"] > 2.5),
    )
    # Rule 5: combined whipsaw guard (any of the above)
    whipsaw = (
        (short_full["btc_rsi14_daily"] < 20)
        | (short_full["btc_kdj_j_daily"] < -10)
        | (short_full["btc_bbands_pct_b_daily"] < 0.05)
        | (short_full["btc_volume_5d_zscore"] > 2.5)
    )
    _eval_rule("R5_skip_any_whipsaw_signal", ~whipsaw)
    # Rule 6: stricter combo (RSI<25 AND %B<0.10) — softer but stacked
    softer = (short_full["btc_rsi14_daily"] < 25) & (short_full["btc_bbands_pct_b_daily"] < 0.10)
    _eval_rule("R6_skip_rsi_lt_25_AND_bb_lt_0p10", ~softer)
    # Rule 7: stack the Phase 10.1 BTC-90d gate AND R5 whipsaw guard
    btc_90d = panels["btc_90d_return"].reindex(short_full["entry_time"]).to_numpy()
    momentum_block = btc_90d > 0.10
    full_block = momentum_block | whipsaw
    _eval_rule("R7_phase10.1_T0.10_AND_whipsaw_guard", ~full_block)

    # ---- Save -------------------------------------------------------------
    out = {
        "feature_names": FEATURE_NAMES,
        "trend_features": TREND_FEATURES,
        "oscillator_features": OSCILLATOR_FEATURES,
        "ml_meta_regime_v2": {
            "model": "logistic_regression",
            "C": LR_C,
            "train_auc": float(train_auc) if np.isfinite(train_auc) else None,
            "oot_auc": float(oot_auc) if np.isfinite(oot_auc) else None,
            "train_n_windows": int(len(train_X)),
            "oot_n_windows": int(len(oot_X)),
            "feature_coefficients_scaled": coef_dict,
            "threshold": threshold_meta,
            "n_train_gated": int(len(train_g)),
            "n_oot_gated": int(len(oot_g)),
            "ship_gate": ml_ship,
        },
        "hand_crafted_rules": rules,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
