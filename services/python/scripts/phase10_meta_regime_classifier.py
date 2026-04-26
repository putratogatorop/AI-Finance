"""Phase 10.2 — ML meta-regime classifier on BTC-only macro features.

Predicts at every entry: "is the next 3 months short-friendly?"
Built on top of v2 SHORT-only ml_on+sized trades. Tiny window-level
sample (~34 monthly samples) so we use strong L2 regularization and
report all the relevant generalization metrics.

Run from services/python/:
    python scripts/phase10_meta_regime_classifier.py
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

from src.ml.indicators import ema as _ema, macd as _macd

# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_DATE = "2026-04-01"
CANDLES_PATH = REPO_ROOT / "data/snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
V2_RUN_DIR = (
    REPO_ROOT / "services/python/results"
    / "backtest_bigmover_combined_v2_outcome_0582b799_20260426T033736Z"
)
TRADES_CSV = V2_RUN_DIR / "trades.csv"
OUT_PATH = V2_RUN_DIR / "phase10_meta_regime_classifier.json"
MODELS_DIR = REPO_ROOT / "services/python/models"
MODEL_OUT = MODELS_DIR / "meta_regime_v1.joblib"
META_OUT = MODELS_DIR / "meta_regime_v1_meta.json"

TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
WINDOW_MONTHS = 3
LR_C = 0.1     # strong L2 (small C) given tiny sample
RANDOM_PURGE_MONTHS = 3   # purge gap between train and val folds
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3

FEATURE_NAMES = [
    "btc_30d_return",
    "btc_90d_return",
    "btc_180d_return",
    "btc_30d_realized_vol",
    "btc_vs_sma200_pct",
    "btc_vs_ema26_weekly_pct",
    "btc_macd_4h_signal_spread_norm",
]


# --- Helpers ----------------------------------------------------------------


def _pf(pnls: np.ndarray) -> float:
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _build_btc_panels(candles: pd.DataFrame) -> dict:
    """Pre-compute all BTC feature series at 15m granularity."""
    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")["close"]
        .astype(float)
    )
    btc_daily = btc.resample("1D", label="right", closed="right").last().dropna()
    # All shifts are by 1 daily/4h/weekly bar to avoid boundary-of-period leak.
    panels = {}
    panels["btc_30d_return"] = btc_daily.pct_change(30).shift(1).reindex(btc.index, method="ffill")
    panels["btc_90d_return"] = btc_daily.pct_change(90).shift(1).reindex(btc.index, method="ffill")
    panels["btc_180d_return"] = btc_daily.pct_change(180).shift(1).reindex(btc.index, method="ffill")
    btc_30d_logret = np.log(btc_daily / btc_daily.shift(1))
    panels["btc_30d_realized_vol"] = (
        btc_30d_logret.rolling(window=30, min_periods=30).std(ddof=0).shift(1)
        .reindex(btc.index, method="ffill")
    )
    sma200 = btc_daily.rolling(window=200, min_periods=200).mean().shift(1)
    panels["btc_vs_sma200_pct"] = (
        ((btc_daily - sma200) / sma200).reindex(btc.index, method="ffill")
    )
    btc_weekly = btc.resample("1W", label="right", closed="right").last()
    weekly_ema26 = _ema(btc_weekly, 26).shift(1)
    panels["btc_vs_ema26_weekly_pct"] = (
        ((btc_weekly - weekly_ema26) / weekly_ema26).reindex(btc.index, method="ffill")
    )
    btc_4h = btc.resample("4h", label="right", closed="right").last()
    m4h = _macd(btc_4h, 12, 26, 9)
    spread_4h = ((m4h["macd"] - m4h["signal"]).shift(1) / btc_4h.shift(1))
    panels["btc_macd_4h_signal_spread_norm"] = (
        spread_4h.reindex(btc.index, method="ffill")
    )
    return panels


def _features_at(panels: dict, ts: pd.Timestamp) -> dict:
    """Look up each feature series at timestamp ts (returns dict, NaN if missing)."""
    out = {}
    for name in FEATURE_NAMES:
        s = panels[name]
        try:
            v = s.asof(ts)
        except KeyError:
            v = float("nan")
        out[name] = float(v) if pd.notna(v) else float("nan")
    return out


def _walkforward_pf_per_window(
    short_trades: pd.DataFrame, start_ts: pd.Timestamp, end_ts: pd.Timestamp,
) -> float | None:
    """SHORT-only PF in [start, end). None if < 30 trades."""
    et = pd.to_datetime(short_trades["entry_time"], utc=True)
    sub = short_trades[(et >= start_ts) & (et < end_ts)]
    if len(sub) < 30:
        return None
    return _pf(sub["sized_pnl"].to_numpy())


# --- Main ------------------------------------------------------------------


def main() -> None:
    started = time.monotonic()

    print(f"loading v2 trades from {TRADES_CSV}", flush=True)
    df = pd.read_csv(TRADES_CSV)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)

    # Restrict to SHORT ml_on+sized
    threshold_predictor = float(
        df[(df["period"] == "train") & df["proba"].notna()]["proba"].quantile(0.5)
    )
    short_full = df[
        (df["direction"] == "short")
        & (df["proba"] >= threshold_predictor)
        & (df["pos_scale"] > 0)
    ].copy()
    print(f"  SHORT ml_on+sized entries: {len(short_full):,}", flush=True)

    # ---- Build BTC feature panels --------------------------------------
    print(f"\nloading BTC candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    print("building BTC feature panels...", flush=True)
    panels = _build_btc_panels(candles)

    # ---- Build window-level training set ------------------------------
    # For each month-start ts in [2023-04 → 2026-01], compute features at ts
    # and label = (forward-3-month PF >= 1.0) on existing v2 SHORT trades.
    print("building window-level training set (monthly cadence)...", flush=True)
    start = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    window_w = relativedelta(months=WINDOW_MONTHS)
    month = relativedelta(months=1)

    rows = []
    s = start
    while s + window_w <= end_global:
        feats = _features_at(panels, s)
        fwd_pf = _walkforward_pf_per_window(short_full, s, s + window_w)
        rows.append({
            "window_start": str(s.date()),
            "window_end": str((s + window_w).date()),
            **feats,
            "fwd_pf": fwd_pf,
        })
        s += month
    win_df = pd.DataFrame(rows)
    # Drop rows with any NaN feature (warmup) or undefined label
    finite_mask = win_df[FEATURE_NAMES].notna().all(axis=1) & win_df["fwd_pf"].notna()
    print(f"  total windows: {len(win_df)}", flush=True)
    print(f"  windows with complete features + label: {int(finite_mask.sum())}",
          flush=True)
    win_df = win_df[finite_mask].reset_index(drop=True)
    win_df["label"] = (win_df["fwd_pf"] >= 1.0).astype(int)
    print(f"  positive rate (windows with PF >= 1.0): "
          f"{(win_df['label']==1).mean():.1%}", flush=True)

    # ---- Train/OOT split ---------------------------------------------------
    win_df["window_start_ts"] = pd.to_datetime(win_df["window_start"]).dt.tz_localize("UTC")
    is_train = win_df["window_start_ts"] < TRAIN_OOT_BOUNDARY
    is_oot = ~is_train
    train_X = win_df.loc[is_train, FEATURE_NAMES].to_numpy(dtype=float)
    train_y = win_df.loc[is_train, "label"].to_numpy(dtype=int)
    oot_X = win_df.loc[is_oot, FEATURE_NAMES].to_numpy(dtype=float)
    oot_y = win_df.loc[is_oot, "label"].to_numpy(dtype=int)
    print(f"\n  train: {is_train.sum()} windows  oot: {is_oot.sum()} windows", flush=True)

    # ---- Train LR ---------------------------------------------------------
    print(f"\n=== Training LR (C={LR_C}, strong L2) ===")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C, max_iter=2000, random_state=RANDOM_SEED, class_weight="balanced",
        )),
    ])
    pipe.fit(train_X, train_y)
    train_proba = pipe.predict_proba(train_X)[:, 1]
    oot_proba = pipe.predict_proba(oot_X)[:, 1] if len(oot_X) > 0 else np.array([])

    train_auc = (
        roc_auc_score(train_y, train_proba) if len(np.unique(train_y)) >= 2 else float("nan")
    )
    oot_auc = (
        roc_auc_score(oot_y, oot_proba) if len(np.unique(oot_y)) >= 2 else float("nan")
    )
    print(f"  train AUC: {train_auc:.4f}", flush=True)
    print(f"  oot   AUC: {oot_auc:.4f}", flush=True)
    print(f"  train n: {len(train_X)}  oot n: {len(oot_X)}", flush=True)
    print("  feature coefficients (scaled): "
          + ", ".join(f"{n}={c:+.3f}" for n, c in zip(
              FEATURE_NAMES, pipe.named_steps["clf"].coef_[0]
          )),
          flush=True)

    # Save model + meta
    joblib.dump(pipe, MODEL_OUT)
    coef_dict = {n: float(c) for n, c in zip(FEATURE_NAMES, pipe.named_steps["clf"].coef_[0])}
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
        "window_months": WINDOW_MONTHS,
        "snapshot_date": SNAPSHOT_DATE,
    }, indent=2, default=str))
    print(f"  saved {MODEL_OUT}, {META_OUT}", flush=True)

    # ---- Apply gate to v2 SHORT trades ---------------------------------
    print("\n=== Applying meta-regime gate to v2 SHORT trades ===")
    # For each entry_time, find the window-start for the 3-month period
    # CONTAINING entry_time, look up feature panel at that window-start,
    # and predict P(short-friendly).
    threshold_meta = 0.5
    print(f"  meta-gate threshold (lock at 0.5 of train-fold scores): {threshold_meta:.3f}",
          flush=True)
    short_full["meta_features_ts"] = short_full["entry_time"].dt.to_period("M").dt.start_time
    if short_full["meta_features_ts"].dt.tz is None:
        short_full["meta_features_ts"] = short_full["meta_features_ts"].dt.tz_localize("UTC")
    feat_dicts = [_features_at(panels, ts) for ts in short_full["meta_features_ts"]]
    feat_arr = np.array([[d[n] for n in FEATURE_NAMES] for d in feat_dicts], dtype=float)
    finite = np.isfinite(feat_arr).all(axis=1)
    proba = np.full(len(short_full), np.nan, dtype=float)
    if finite.any():
        proba[finite] = pipe.predict_proba(feat_arr[finite])[:, 1]
    short_full["meta_proba"] = proba
    short_full["meta_pass"] = short_full["meta_proba"] >= threshold_meta

    # ---- Evaluate ship gate at meta_pass ----------------------------------
    gated = short_full[short_full["meta_pass"]]
    train_gated = gated[gated["period"] == "train"]
    oot_gated = gated[gated["period"] == "oot"]

    # Walk-forward
    et = pd.to_datetime(gated["entry_time"], utc=True)
    s = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    fold_w = relativedelta(months=WF_FOLD_MONTHS)
    folds = []
    while s + fold_w <= end_global:
        sub = gated[(et >= s) & (et < s + fold_w)]
        folds.append({
            "fold_start": str(s.date()),
            "fold_end": str((s + fold_w).date()),
            "trades": int(len(sub)),
            "pf": _pf(sub["sized_pnl"].to_numpy()),
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

    # MC bootstrap on OOT
    oot_pnls = oot_gated["sized_pnl"].to_numpy(dtype=float)
    if len(oot_pnls) > 0:
        rng = np.random.default_rng(RANDOM_SEED)
        mc_pfs = np.array([
            _pf(rng.choice(oot_pnls, size=len(oot_pnls), replace=True))
            for _ in range(MC_N_RESAMPLES)
        ])
        mc_finite = mc_pfs[np.isfinite(mc_pfs)]
        mc = {
            "n": int(len(oot_pnls)),
            "point_pf": _pf(oot_pnls),
            "p5": float(np.percentile(mc_finite, 5)) if len(mc_finite) else float("nan"),
            "p50": float(np.percentile(mc_finite, 50)) if len(mc_finite) else float("nan"),
            "p95": float(np.percentile(mc_finite, 95)) if len(mc_finite) else float("nan"),
        }
    else:
        mc = {"n": 0}

    print(f"\n  TRAIN gated: n={len(train_gated)}  PF={_pf(train_gated['sized_pnl'].to_numpy()):.3f}",
          flush=True)
    print(f"  OOT gated:   n={len(oot_gated)}  PF={_pf(oot_pnls):.3f}",
          flush=True)
    print(f"  WF p5: {rollup.get('p5', float('nan')):.3f}  "
          f"p50: {rollup.get('p50', float('nan')):.3f}", flush=True)
    print(f"  MC p5: {mc.get('p5', float('nan')):.3f}  "
          f"p50: {mc.get('p50', float('nan')):.3f}", flush=True)

    # Ship gate
    oot_pf = _pf(oot_pnls)
    ship_gate = {
        "oot_pf_ge_130": bool(np.isfinite(oot_pf) and oot_pf >= 1.30),
        "oot_n_ge_200": bool(len(oot_pnls) >= 200),
        "walkforward_p5_gt_1": bool(rollup.get("p5", -1) > 1.0),
        "mc_p5_gt_1": bool(np.isfinite(mc.get("p5", float("nan"))) and mc.get("p5", -1) > 1.0),
    }
    ship_gate["overall"] = all(ship_gate.values())
    print("\n=== Ship gate (SHORT + ml + sizing + ML meta-gate) ===")
    for k, v in ship_gate.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(f"  → overall: {'SHIP' if ship_gate['overall'] else 'DO NOT SHIP YET'}")

    # ---- Save -------------------------------------------------------------
    out = {
        "feature_names": FEATURE_NAMES,
        "label": "fwd_3mo_short_only_pf_ge_1.0",
        "C": LR_C,
        "train_auc": float(train_auc) if np.isfinite(train_auc) else None,
        "oot_auc": float(oot_auc) if np.isfinite(oot_auc) else None,
        "train_n_windows": int(len(train_X)),
        "oot_n_windows": int(len(oot_X)),
        "feature_coefficients_scaled": coef_dict,
        "meta_gate_threshold": threshold_meta,
        "n_short_with_meta": int(short_full["meta_proba"].notna().sum()),
        "n_train_gated": int(len(train_gated)),
        "n_oot_gated": int(len(oot_gated)),
        "train_pf_gated": _pf(train_gated["sized_pnl"].to_numpy()),
        "oot_pf_gated": _pf(oot_pnls),
        "walkforward": rollup,
        "monte_carlo_oot": mc,
        "ship_gate": ship_gate,
    }
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
