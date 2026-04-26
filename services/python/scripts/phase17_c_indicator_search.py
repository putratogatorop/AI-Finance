"""Phase 17.C — Add 5 indicator features and re-train the D1 SHORT classifier.

PRE-COMMITTED DECISION RULE (locked at top, before any data is touched):
  Augmented (21-feature + is_short) classifier "wins" iff ALL THREE hold:
    (1) Classifier-gated OOT PF > 2.607 (current v3 = 16-feature locked baseline).
    (2) WF p5 > 0.719 (current v3 = 16-feature locked baseline).
    (3) No individual added feature has CV-AUC delta worse than -0.005 when
        dropped from the bundle (i.e. each added feature must not be
        destructive — the bundle has no "wolves").
  If win → save d1_short_v3_plus.{joblib, _meta.json}; print "ADOPT v3-plus".
  Else   → do NOT save model files (or save under "rejected_" prefix);
            print "DROP — v3 holds".

5 NEW FEATURES (each computed at entry_time using ONLY bars whose timestamp
<= entry_time, mirrored from src/ml/bigmover_combined/features.py:_build_asset_view):
  bb_pct_b         — Bollinger %B on coin's 15m close (period=20, k=2).
                       %B = (close - lower) / (upper - lower)
                       upper = sma20 + 2*std20, lower = sma20 - 2*std20.
  bb_width_norm    — (upper - lower) / close.
  bb_squeeze_flag  — 1 if bb_width is <= 10th-pct of bb_width over the last
                     96 bars at entry, else 0.
  atr_pct_close    — ATR_14 / close (use indicators.atr; ATR period=14).
  ema21_cross_ema50— sign(EMA21 - EMA50) at entry. Returns -1, 0, +1.

HARD CONSTRAINTS:
  - LR only. No LightGBM.
  - Same TRAIN/OOT split @ 2026-01-01 UTC.
  - Same hyperparams: C=1.0, max_iter=2000, class_weight='balanced', seed=42.
  - Same purge=192, k=5 purged-time-series CV.
  - No live changes. No git commits.
  - Do NOT modify the existing classifier file or features.py.
  - Reproducibility: train+evaluate twice end-to-end, confirm identical
    OOT AUC and WF p5 to 4 decimal places. Print PASS / FAIL.

Run from services/python/:
    uv run python scripts/phase17_c_indicator_search.py
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

from src.ml.bigmover_combined.features import FEATURE_NAMES
from src.ml.indicators import atr as _atr, ema as _ema  # noqa: F401

# --- Paths / constants ------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3

DATA_DIR = SERVICES_PY / "data"
MODELS_DIR = SERVICES_PY / "models"
RESULTS_DIR = SERVICES_PY / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
TRADES_WITH_FEATS_PATH = DATA_DIR / "d1_short_trades_with_features.csv"

# Existing v3 baseline (locked numbers from Phase 16.2 / Phase 17 description).
V3_OOT_AUC = 0.5623
V3_OOT_PF = 2.607  # classifier-gated (D1 + v3 LR)
V3_WF_P5 = 0.719

# Augmented model output paths
V3_PLUS_MODEL_PATH = MODELS_DIR / "d1_short_v3_plus.joblib"
V3_PLUS_META_PATH = MODELS_DIR / "d1_short_v3_plus_meta.json"
RESULTS_OUT = RESULTS_DIR / "phase17_c_indicator_search.json"

# Hyperparams (locked, same as v3)
LR_C = 1.0
LR_MAX_ITER = 2000
PURGE_BARS = 192
K_FOLDS = 5

# 5 new features
NEW_FEATURE_NAMES = [
    "bb_pct_b",
    "bb_width_norm",
    "bb_squeeze_flag",
    "atr_pct_close",
    "ema21_cross_ema50",
]

# BB / ATR / EMA params for new features
BB_PERIOD = 20
BB_K = 2.0
BB_SQUEEZE_LOOKBACK = 96
BB_SQUEEZE_PCT = 0.10
ATR_PERIOD_NEW = 14
EMA_FAST_NEW = 21
EMA_SLOW_NEW = 50


# ----------------------------------------------------------------------------
# Per-asset view for the 5 new features (mirrors features.py:_build_asset_view).
# ----------------------------------------------------------------------------


def _build_new_feat_view(sub: pd.DataFrame) -> dict:
    """Build per-asset arrays + timestamp index for the 5 new features.
    All values at index i are computable using ONLY bars [0..i].
    """
    sub = sub.sort_values("timestamp").reset_index(drop=True)
    ts = (
        pd.to_datetime(sub["timestamp"], utc=True).dt.tz_convert(None).to_numpy()
    )
    h = sub["high"].to_numpy(dtype=float)
    l = sub["low"].to_numpy(dtype=float)
    c = sub["close"].to_numpy(dtype=float)

    # Bollinger Bands (period=20, k=2) on close
    sma20 = pd.Series(c).rolling(BB_PERIOD, min_periods=BB_PERIOD).mean().to_numpy()
    std20 = pd.Series(c).rolling(BB_PERIOD, min_periods=BB_PERIOD).std(ddof=0).to_numpy()
    upper = sma20 + BB_K * std20
    lower = sma20 - BB_K * std20
    width = upper - lower
    # %B
    denom = (upper - lower)
    with np.errstate(divide="ignore", invalid="ignore"):
        pct_b = np.where(denom > 0, (c - lower) / denom, np.nan)
        bb_width_norm = np.where(c > 0, width / c, np.nan)

    # bb_squeeze_flag: width <= 10th-pct of width over last 96 bars (incl. current)
    width_series = pd.Series(width)
    p10_width = width_series.rolling(BB_SQUEEZE_LOOKBACK, min_periods=BB_SQUEEZE_LOOKBACK).quantile(BB_SQUEEZE_PCT).to_numpy()
    with np.errstate(invalid="ignore"):
        bb_squeeze = np.where(
            np.isfinite(width) & np.isfinite(p10_width),
            (width <= p10_width).astype(float),
            np.nan,
        )

    # ATR/close
    atr14 = _atr(pd.Series(h), pd.Series(l), pd.Series(c), ATR_PERIOD_NEW).to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = np.where((c > 0) & np.isfinite(atr14), atr14 / c, np.nan)

    # EMA21 / EMA50 cross sign
    ema21 = _ema(pd.Series(c), EMA_FAST_NEW).to_numpy()
    ema50 = _ema(pd.Series(c), EMA_SLOW_NEW).to_numpy()
    diff = ema21 - ema50
    cross = np.where(np.isfinite(diff), np.sign(diff), np.nan)

    return {
        "ts": ts,
        "bb_pct_b": pct_b,
        "bb_width_norm": bb_width_norm,
        "bb_squeeze_flag": bb_squeeze,
        "atr_pct_close": atr_pct,
        "ema21_cross_ema50": cross,
    }


# ----------------------------------------------------------------------------
# CV helpers (copied from train_d1_classifier_v3.py — DO NOT EDIT).
# ----------------------------------------------------------------------------


def _purged_kfold_indices(n: int, k: int = 5, purge: int = 192):
    fold_size = n // k
    folds = []
    for i in range(k):
        val_start = i * fold_size
        val_end = (i + 1) * fold_size if i < k - 1 else n
        train_end = max(0, val_start - purge)
        if train_end <= 0:
            continue
        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        folds.append((train_idx, val_idx))
    return folds


def _make_pipe():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C, max_iter=LR_MAX_ITER, random_state=RANDOM_SEED, class_weight="balanced",
        )),
    ])


# ----------------------------------------------------------------------------
# Backtest metric helpers (mirrored from backtest_d1_with_v3_classifier.py).
# ----------------------------------------------------------------------------


def _pf(pnls):
    p = np.asarray(pnls, dtype=float)
    if len(p) == 0:
        return float("nan")
    w = p[p > 0].sum()
    l = -p[p < 0].sum()
    return float("inf") if l == 0 else float(w / l)


def _cell(pnls):
    pnls = np.asarray(pnls, dtype=float)
    if len(pnls) == 0:
        return {"trades": 0, "wr": 0.0, "pf": float("nan"),
                "avg_pnl_pct": 0.0, "total_pnl_pct": 0.0}
    return {
        "trades": int(len(pnls)),
        "wr": float((pnls > 0).mean()),
        "pf": _pf(pnls),
        "avg_pnl_pct": float(pnls.mean()),
        "total_pnl_pct": float(pnls.sum()),
    }


def _walkforward(df, pnl_col="sized_pnl"):
    if len(df) == 0:
        return {"n_folds_used": 0}
    et = pd.to_datetime(df["entry_time"], utc=True)
    s = pd.Timestamp("2023-04-01", tz="UTC")
    end_global = pd.Timestamp(SNAPSHOT_DATE, tz="UTC")
    fold_w = relativedelta(months=WF_FOLD_MONTHS)
    month = relativedelta(months=1)
    pfs = []
    s_ = s
    while s_ + fold_w <= end_global:
        sub = df[(et >= s_) & (et < s_ + fold_w)]
        if len(sub) >= 30:
            p = _pf(sub[pnl_col].to_numpy())
            if np.isfinite(p):
                pfs.append(p)
        s_ += month
    pfs = np.array(pfs)
    if len(pfs) == 0:
        return {"n_folds_used": 0}
    return {
        "n_folds_used": int(len(pfs)),
        "p5": float(np.percentile(pfs, 5)),
        "p50": float(np.percentile(pfs, 50)),
        "p95": float(np.percentile(pfs, 95)),
        "p_lt_1": float((pfs < 1.0).mean()),
        "p_gte_130": float((pfs >= 1.30).mean()),
    }


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


def _summarize(df: pd.DataFrame, label: str) -> dict:
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["period"] = np.where(df["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
    train_pnls = df[df["period"] == "train"]["sized_pnl"].to_numpy()
    oot_pnls = df[df["period"] == "oot"]["sized_pnl"].to_numpy()
    train_m = _cell(train_pnls)
    oot_m = _cell(oot_pnls)
    wf = _walkforward(df, "sized_pnl")
    mc = _monte_carlo(oot_pnls)
    print(
        f"  {label:<26} TRAIN n={train_m['trades']:>5} PF={train_m['pf']:.3f}  "
        f"OOT n={oot_m['trades']:>5} PF={oot_m['pf']:.3f}  "
        f"WF p5={wf.get('p5', float('nan')):.3f}  P(loss)={wf.get('p_lt_1', float('nan'))*100:.1f}%  "
        f"MC p5={mc.get('p5', float('nan')):.3f}",
        flush=True,
    )
    return {
        "train_metrics": train_m,
        "oot_metrics": oot_m,
        "walkforward": wf,
        "monte_carlo_oot": mc,
    }


# ----------------------------------------------------------------------------
# 5-feature build pass for every trade (per-asset cache of the new view).
# ----------------------------------------------------------------------------


def build_new_features_for_trades(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Building 5 new features per trade ===", flush=True)
    cache: dict[str, dict] = {}
    out = {n: np.full(len(trades), np.nan) for n in NEW_FEATURE_NAMES}
    assets = trades["asset"].to_numpy()
    entry_times = pd.to_datetime(trades["entry_time"], utc=True)
    entry_naive = entry_times.dt.tz_convert(None).to_numpy().astype("datetime64[ns]")

    unique_assets = sorted(set(assets.tolist()))
    print(f"  unique assets: {len(unique_assets)}", flush=True)

    # Pre-build per-asset views.
    for idx, asset in enumerate(unique_assets):
        sub = candles[candles["asset"] == asset]
        if len(sub) == 0:
            continue
        cache[asset] = _build_new_feat_view(sub)
        if (idx + 1) % 50 == 0:
            print(f"    {idx + 1}/{len(unique_assets)} asset views built", flush=True)

    # Now look up per-trade.
    for k, asset in enumerate(assets):
        view = cache.get(asset)
        if view is None:
            continue
        ts_arr = view["ts"]
        i = int(np.searchsorted(ts_arr, entry_naive[k], side="right")) - 1
        if i < 0:
            continue
        for n in NEW_FEATURE_NAMES:
            arr = view[n]
            if i < len(arr):
                v = arr[i]
                if np.isfinite(v):
                    out[n][k] = float(v)
        if (k + 1) % 20000 == 0:
            print(f"    trade lookups: {k + 1}/{len(trades)}", flush=True)

    return pd.DataFrame(out)


# ----------------------------------------------------------------------------
# End-to-end train+eval (the function we will call twice for reproducibility).
# ----------------------------------------------------------------------------


def train_and_eval(
    X_full: np.ndarray, y: np.ndarray, is_train: np.ndarray,
    feature_cols: list, df_for_bt: pd.DataFrame,
) -> dict:
    """Train LR on TRAIN, lock threshold @ TRAIN q0.5, gate trades by score, return all metrics."""
    train_X = X_full[is_train]
    train_y = y[is_train]
    oot_X = X_full[~is_train]
    oot_y = y[~is_train]

    # CV
    cv_aucs = []
    for fold_i, (tr_idx, va_idx) in enumerate(_purged_kfold_indices(len(train_X), k=K_FOLDS, purge=PURGE_BARS)):
        if len(np.unique(train_y[tr_idx])) < 2 or len(np.unique(train_y[va_idx])) < 2:
            continue
        pipe = _make_pipe()
        pipe.fit(train_X[tr_idx], train_y[tr_idx])
        proba_val = pipe.predict_proba(train_X[va_idx])[:, 1]
        auc = roc_auc_score(train_y[va_idx], proba_val)
        cv_aucs.append(float(auc))

    pipe = _make_pipe()
    pipe.fit(train_X, train_y)
    train_scores = pipe.predict_proba(train_X)[:, 1]
    oot_scores = pipe.predict_proba(oot_X)[:, 1]
    train_auc = float(roc_auc_score(train_y, train_scores))
    oot_auc = (
        float(roc_auc_score(oot_y, oot_scores))
        if len(np.unique(oot_y)) >= 2 else float("nan")
    )
    threshold = float(np.percentile(train_scores, 50.0))

    # Score the full ledger with this fitted pipe and gate
    all_scores = pipe.predict_proba(X_full)[:, 1]
    df_scored = df_for_bt.copy()
    df_scored["score"] = all_scores
    gated = df_scored[df_scored["score"] >= threshold].reset_index(drop=True)

    # Backtest classifier-gated metrics
    raw = _summarize(df_scored, "raw (no gate)")
    classified = _summarize(gated, "classifier-gated")

    return {
        "pipeline": pipe,
        "feature_cols": feature_cols,
        "cv_aucs": cv_aucs,
        "train_auc": train_auc,
        "oot_auc": oot_auc,
        "threshold": threshold,
        "raw": raw,
        "classifier_gated": classified,
        "kept_pct": float(len(gated) / max(len(df_scored), 1)),
    }


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main():
    started = time.monotonic()
    np.random.seed(RANDOM_SEED)

    # 1) Load the prebuilt feature ledger (already has 16 features + is_short
    #    + sized_pnl etc.) — produced by train_d1_classifier_v3.py. We only
    #    need to bolt on the 5 new features.
    print(f"loading {TRADES_WITH_FEATS_PATH}", flush=True)
    df = pd.read_csv(TRADES_WITH_FEATS_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"  {len(df):,} feature-complete trades loaded", flush=True)

    # Minimal sanity: existing 16 + is_short cols all present.
    for col in FEATURE_NAMES + ["is_short"]:
        if col not in df.columns:
            raise RuntimeError(f"missing existing feature column: {col}")

    # 2) Compute label
    df["label"] = (df["pnl_pct"] > 0).astype(int)
    print(f"  outcome-label positive rate: {df['label'].mean():.1%}", flush=True)

    # 3) Load candles (only needed for new-feature build; the existing 16 are
    #    already cached in the CSV).
    print(f"\nloading candles {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)
    print(f"  {len(candles):,} 15m bars loaded", flush=True)

    # 4) Build the 5 new features per trade.
    new_feats = build_new_features_for_trades(df, candles)
    df_aug = pd.concat([df.reset_index(drop=True), new_feats.reset_index(drop=True)], axis=1)

    # 5) Drop NaN rows in the 21-feature + is_short matrix (some assets too
    #    new for BB/ATR warmup will get dropped here).
    feat_cols_full = FEATURE_NAMES + NEW_FEATURE_NAMES + ["is_short"]
    finite_mask = df_aug[feat_cols_full].notna().all(axis=1).to_numpy()
    n_before = len(df_aug)
    df_aug = df_aug[finite_mask].reset_index(drop=True)
    print(f"  full-21-feature rows: {len(df_aug):,}/{n_before:,} "
          f"(dropped {n_before - len(df_aug):,} NaN)", flush=True)

    # 6) TRAIN / OOT split
    is_train = (df_aug["entry_time"] < TRAIN_OOT_BOUNDARY).to_numpy()
    print(f"  train: {is_train.sum():,}  oot: {(~is_train).sum():,}", flush=True)

    X_full = df_aug[feat_cols_full].to_numpy(dtype=float)
    y = df_aug["label"].to_numpy(dtype=int)

    # 7) End-to-end train and evaluate (run #1)
    print("\n=== RUN 1 — full augmented (21-feature + is_short) ===", flush=True)
    run1 = train_and_eval(X_full, y, is_train, feat_cols_full, df_aug)
    print(f"  CV AUCs: {[f'{x:.4f}' for x in run1['cv_aucs']]}")
    print(f"  TRAIN AUC: {run1['train_auc']:.4f}    OOT AUC: {run1['oot_auc']:.4f}    "
          f"gap: {run1['train_auc'] - run1['oot_auc']:.4f}")
    print(f"  threshold @ TRAIN q0.5: {run1['threshold']:.4f}  "
          f"(kept {run1['kept_pct']*100:.1f}% of trades)")

    # 8) Reproducibility: train and evaluate twice end-to-end.
    print("\n=== RUN 2 — reproducibility check (re-train end-to-end) ===", flush=True)
    run2 = train_and_eval(X_full, y, is_train, feat_cols_full, df_aug)
    print(f"  TRAIN AUC: {run2['train_auc']:.4f}    OOT AUC: {run2['oot_auc']:.4f}")

    repro_oot_auc_eq = round(run1["oot_auc"], 4) == round(run2["oot_auc"], 4)
    repro_wf_p5_eq = round(
        run1["classifier_gated"]["walkforward"].get("p5", float("nan")), 4
    ) == round(
        run2["classifier_gated"]["walkforward"].get("p5", float("nan")), 4
    )
    repro_pass = bool(repro_oot_auc_eq and repro_wf_p5_eq)
    print(f"\nREPRODUCIBILITY: {'PASS' if repro_pass else 'FAIL'}")
    print(f"  run1 OOT AUC={run1['oot_auc']:.4f}  run2 OOT AUC={run2['oot_auc']:.4f}")
    print(f"  run1 WF p5 ={run1['classifier_gated']['walkforward'].get('p5', float('nan')):.4f}  "
          f"run2 WF p5 ={run2['classifier_gated']['walkforward'].get('p5', float('nan')):.4f}")

    # 9) Per-feature ablation: drop ONE new feature at a time, re-train, compare CV-AUC.
    print("\n=== Ablation — drop 1 new feature at a time (20 + is_short) ===", flush=True)
    full_cv_mean = float(np.mean(run1["cv_aucs"]))
    full_oot_auc = run1["oot_auc"]
    ablation = {}
    any_destructive = False
    for drop_name in NEW_FEATURE_NAMES:
        kept_cols = [c for c in feat_cols_full if c != drop_name]
        X_drop = df_aug[kept_cols].to_numpy(dtype=float)
        # CV
        cv_aucs_drop = []
        for fold_i, (tr_idx, va_idx) in enumerate(
            _purged_kfold_indices(int(is_train.sum()), k=K_FOLDS, purge=PURGE_BARS)
        ):
            tr_X = X_drop[is_train][tr_idx]; tr_y = y[is_train][tr_idx]
            va_X = X_drop[is_train][va_idx]; va_y = y[is_train][va_idx]
            if len(np.unique(tr_y)) < 2 or len(np.unique(va_y)) < 2:
                continue
            pipe = _make_pipe()
            pipe.fit(tr_X, tr_y)
            cv_aucs_drop.append(float(roc_auc_score(va_y, pipe.predict_proba(va_X)[:, 1])))
        cv_mean_drop = float(np.mean(cv_aucs_drop)) if cv_aucs_drop else float("nan")
        # Full-fit OOT
        pipe = _make_pipe()
        pipe.fit(X_drop[is_train], y[is_train])
        oot_auc_drop = (
            float(roc_auc_score(y[~is_train], pipe.predict_proba(X_drop[~is_train])[:, 1]))
            if len(np.unique(y[~is_train])) >= 2 else float("nan")
        )
        # Helpfulness: positive delta = feature was helpful
        cv_delta_when_dropped = full_cv_mean - cv_mean_drop  # full minus drop
        oot_delta_when_dropped = full_oot_auc - oot_auc_drop
        # Destructive condition: feature is destructive when its CV-AUC delta is
        # WORSE than -0.005, i.e. dropping it IMPROVES CV-AUC by >= 0.005.
        # cv_delta_when_dropped = full_cv_mean - cv_mean_drop. If drop improves
        # mean: cv_mean_drop > full_cv_mean → cv_delta_when_dropped < 0.
        # Destructive ⇔ cv_delta_when_dropped <= -0.005.
        is_destructive = cv_delta_when_dropped <= -0.005
        if is_destructive:
            any_destructive = True
        ablation[drop_name] = {
            "cv_aucs_drop": cv_aucs_drop,
            "cv_mean_drop": cv_mean_drop,
            "cv_delta_when_dropped": cv_delta_when_dropped,
            "oot_auc_drop": oot_auc_drop,
            "oot_delta_when_dropped": oot_delta_when_dropped,
            "is_destructive": bool(is_destructive),
        }
        print(f"  drop {drop_name:<22} CV-mean={cv_mean_drop:.4f}  "
              f"OOT-AUC={oot_auc_drop:.4f}  "
              f"ΔCV(full-drop)={cv_delta_when_dropped:+.4f}  "
              f"ΔOOT(full-drop)={oot_delta_when_dropped:+.4f}  "
              f"destructive={is_destructive}")

    # 10) Decision rule
    cls_oot_pf = run1["classifier_gated"]["oot_metrics"]["pf"]
    cls_wf_p5 = run1["classifier_gated"]["walkforward"].get("p5", float("nan"))
    cls_mc_p5 = run1["classifier_gated"]["monte_carlo_oot"].get("p5", float("nan"))
    cls_oot_n = run1["classifier_gated"]["oot_metrics"]["trades"]

    win_pf = bool(np.isfinite(cls_oot_pf) and cls_oot_pf > V3_OOT_PF)
    win_wf = bool(np.isfinite(cls_wf_p5) and cls_wf_p5 > V3_WF_P5)
    win_no_destructive = bool(not any_destructive)
    overall_win = bool(win_pf and win_wf and win_no_destructive)

    print("\n=== Decision check (vs locked v3) ===")
    print(f"  v3 (16-feat) baseline: OOT PF={V3_OOT_PF}, WF p5={V3_WF_P5}, OOT AUC={V3_OOT_AUC}")
    print(f"  v3-plus (21-feat):     OOT PF={cls_oot_pf:.3f}, WF p5={cls_wf_p5:.3f}, "
          f"OOT AUC={run1['oot_auc']:.4f}, MC p5={cls_mc_p5:.3f}, OOT n={cls_oot_n}")
    print(f"  win_pf={win_pf}  win_wf={win_wf}  win_no_destructive={win_no_destructive}")
    decision = "ADOPT v3-plus" if overall_win else "DROP — v3 holds"
    print(f"  DECISION: {decision}")

    # 11) Save model files only if win.
    saved_paths = {}
    if overall_win:
        joblib.dump(run1["pipeline"], V3_PLUS_MODEL_PATH)
        meta = {
            "model": "logistic_regression",
            "phase": "17.C",
            "parent_model": "d1_short_v3",
            "feature_names": feat_cols_full,
            "added_features": NEW_FEATURE_NAMES,
            "label_kind": "outcome",
            "lr_C": LR_C,
            "lr_max_iter": LR_MAX_ITER,
            "purge_bars": PURGE_BARS,
            "k_folds": K_FOLDS,
            "cv_aucs": run1["cv_aucs"],
            "train_auc": run1["train_auc"],
            "oot_auc": run1["oot_auc"],
            "threshold_train_q50": run1["threshold"],
            "trained_on_n": int(is_train.sum()),
            "trained_on_pos_rate": float(np.mean(y[is_train])),
            "snapshot_date": SNAPSHOT_DATE,
            "decision": "adopt",
        }
        V3_PLUS_META_PATH.write_text(json.dumps(meta, indent=2, default=str))
        saved_paths = {
            "model": str(V3_PLUS_MODEL_PATH),
            "meta": str(V3_PLUS_META_PATH),
        }
        print(f"\nADOPT v3-plus — saved {V3_PLUS_MODEL_PATH}, {V3_PLUS_META_PATH}")
    else:
        print("\nDROP — not saving v3-plus model (rule failed). v3 holds as production.")

    # 12) Write JSON summary
    summary = {
        "phase": "17.C — indicator search (5 new features, LR only)",
        "snapshot": SNAPSHOT_DATE,
        "decision_rule": {
            "win_iff": "OOT PF > 2.607 AND WF p5 > 0.719 AND no individual added feature has CV-AUC delta worse than -0.005 when dropped",
            "v3_baseline": {"oot_pf": V3_OOT_PF, "wf_p5": V3_WF_P5, "oot_auc": V3_OOT_AUC},
        },
        "n_trades_full_21_feat": int(len(df_aug)),
        "n_train": int(is_train.sum()),
        "n_oot": int((~is_train).sum()),
        "added_features": NEW_FEATURE_NAMES,
        "augmented_full": {
            "cv_aucs": run1["cv_aucs"],
            "cv_auc_mean": float(np.mean(run1["cv_aucs"])),
            "train_auc": run1["train_auc"],
            "oot_auc": run1["oot_auc"],
            "train_oot_gap": run1["train_auc"] - run1["oot_auc"],
            "threshold_train_q50": run1["threshold"],
            "kept_pct": run1["kept_pct"],
            "raw_summary": run1["raw"],
            "classifier_gated_summary": run1["classifier_gated"],
        },
        "reproducibility": {
            "run1_oot_auc": run1["oot_auc"],
            "run2_oot_auc": run2["oot_auc"],
            "run1_wf_p5": run1["classifier_gated"]["walkforward"].get("p5", float("nan")),
            "run2_wf_p5": run2["classifier_gated"]["walkforward"].get("p5", float("nan")),
            "pass": repro_pass,
        },
        "ablation": ablation,
        "decision": "adopt" if overall_win else "drop",
        "saved_paths": saved_paths,
        "wall_seconds": time.monotonic() - started,
    }
    RESULTS_OUT.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {RESULTS_OUT}")
    print(f"total wall: {time.monotonic() - started:.1f}s")
    print(f"\nFINAL DECISION: {decision}")


if __name__ == "__main__":
    main()
