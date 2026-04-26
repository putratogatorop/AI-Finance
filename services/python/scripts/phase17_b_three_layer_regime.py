"""Phase 17.B — 3-LAYER regime cascade for D1 SHORT entries.

Phase 16 locked the production config: D1 detector + sign-locked Candidate-B
BTC trend score sizing + v3 LR classifier (threshold = TRAIN q0.5 = 0.4991) +
rapid-rally exit. OOT PF = 2.607, WF p5 = 0.719, MC p5 = 2.371. The remaining
weakness is WF p5: more than 5% of 3-month folds still have PF < 1.0.

Phase 17.B replaces the SINGLE-layer BTC trend score (which is *one* coarse
"is BTC bullish-ish" sizer) with a THREE-LAYER cascade that fires a hard veto
when the trend at multiple scales says "this market is clearly going up":
  L1 — BTC 4h regime (ALL-BULLISH = bad for SHORT)
  L2 — coin daily regime (ALL-BULLISH = bad for SHORT)
  L3 — coin 1h regime  (ALL-BULLISH = bad for SHORT)

Each layer's stack is the canonical phase14 panel:
  EMA9>EMA21>EMA50  AND  RSI(14)>=50  AND  MACD>signal  AND  KDJ K>D
all on shifted-by-1-bar close (no lookahead), then forward-filled to the 15m grid.

3 cascade variants:
  C1  strict-AND     admit iff NONE of {L1, L2, L3} is ALL-BULLISH
  C2  L1 hard-veto   admit iff L1 is NOT ALL-BULLISH (ignore L2, L3)
  C3  soft 2-of-3    admit iff at least 2 of {L1, L2, L3} are NOT ALL-BULLISH

PRE-COMMITTED DECISION RULE:
  Cascade variant V "wins" iff
    (a) WF p5 > 0.719  AND
    (b) OOT n >= 1500  AND
    (c) NO metric (OOT PF, MC p5) drops more than 5% relative to current
        locked config (OOT PF 2.607, MC p5 2.371) under D1 + v3 classifier.
  Print PASS/FAIL per gate condition for each variant and final decision:
    "STACK <variant>"  or  "DROP - current config holds"

HARD CONSTRAINTS:
  - Reuse v3 classifier as-is. NO retraining.
  - TRAIN_OOT_BOUNDARY = 2026-01-01 UTC, WF_FOLD_MONTHS = 3,
    MC_N_RESAMPLES = 5000, RANDOM_SEED = 42.
  - Reproducibility: re-run metrics computation twice, confirm identical numbers.

Run from services/python/:
    uv run python scripts/phase17_b_three_layer_regime.py
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ml.bigmover_combined.features import FEATURE_NAMES
from src.ml.indicators import ema as _ema, kdj as _kdj, macd as _macd, rsi as _rsi


# --- Paths ------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]
SERVICES_PY = Path(__file__).resolve().parents[1]
SNAPSHOT_DATE = "2026-04-01"
TRAIN_OOT_BOUNDARY = pd.Timestamp("2026-01-01", tz="UTC")
RANDOM_SEED = 42
MC_N_RESAMPLES = 5000
WF_FOLD_MONTHS = 3

CANDLES_PATH = REPO_ROOT / "data" / "snapshots" / f"candles_15m_{SNAPSHOT_DATE}.parquet"
DATA_DIR = SERVICES_PY / "data"
MODELS_DIR = SERVICES_PY / "models"
RESULTS_DIR = SERVICES_PY / "results"
TRADES_PATH = DATA_DIR / "d1_short_trades_with_features.csv"
MODEL_PATH = MODELS_DIR / "d1_short_v3.joblib"
META_PATH = MODELS_DIR / "d1_short_v3_meta.json"
OUT_PATH = RESULTS_DIR / "phase17_b_three_layer_regime.json"

# Indicator stack params (canonical phase14)
EMA_FAST = 9
EMA_MID = 21
EMA_SLOW = 50
RSI_PERIOD = 14
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
KDJ_N, KDJ_K_SMOOTH, KDJ_D_SMOOTH = 9, 3, 3

# Locked Phase-16 reference numbers (current production)
LOCKED_OOT_PF = 2.607
LOCKED_MC_P5 = 2.371
LOCKED_WF_P5 = 0.719

# Decision-rule constants
WF_P5_FLOOR = 0.719
OOT_N_FLOOR = 1500
DROP_TOLERANCE = 0.05  # 5% relative drop allowed


# ============================================================================
# Helpers (copied verbatim from phase16_3 / phase14)
# ============================================================================

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
    """Compute TRAIN/OOT/WF/MC metrics for a trade subset."""
    df = df.copy()
    if len(df) == 0:
        empty = {"train_metrics": _cell([]), "oot_metrics": _cell([]),
                 "walkforward": {"n_folds_used": 0},
                 "monte_carlo_oot": {"n": 0}}
        print(f"  {label:<32} (no trades)", flush=True)
        return empty
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["period"] = np.where(df["entry_time"] < TRAIN_OOT_BOUNDARY, "train", "oot")
    train_pnls = df[df["period"] == "train"]["sized_pnl"].to_numpy()
    oot_pnls = df[df["period"] == "oot"]["sized_pnl"].to_numpy()
    train_m = _cell(train_pnls)
    oot_m = _cell(oot_pnls)
    wf = _walkforward(df, "sized_pnl")
    mc = _monte_carlo(oot_pnls)
    print(
        f"  {label:<32} TRAIN n={train_m['trades']:>5} PF={train_m['pf']:.3f}  "
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


# ============================================================================
# Cascade panels
# ============================================================================

def _all_bullish_panel(open_, high, low, close, *, include_ema_slow=True):
    """Boolean Series: True at bar i iff stack is ALL-BULLISH at i (using shift-1).

    Stack:
      EMA9 > EMA21 > EMA50   (close.shift(1) > ema_fast.shift(1) > ema_mid.shift(1) > ema_slow.shift(1))
      RSI(14) >= 50          (shift 1)
      MACD > signal          (shift 1)
      KDJ K > D              (shift 1)
    """
    ema_fast = _ema(close, EMA_FAST).shift(1)
    ema_mid = _ema(close, EMA_MID).shift(1)
    ema_cond = (close.shift(1) > ema_fast) & (ema_fast > ema_mid)
    if include_ema_slow:
        ema_slow = _ema(close, EMA_SLOW).shift(1)
        ema_cond = ema_cond & (ema_mid > ema_slow)
    rsi_cond = _rsi(close, RSI_PERIOD).shift(1) >= 50.0
    md = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    macd_cond = md["macd"].shift(1) > md["signal"].shift(1)
    kdj_v = _kdj(high, low, close, n=KDJ_N, k_smooth=KDJ_K_SMOOTH, d_smooth=KDJ_D_SMOOTH)
    kdj_cond = kdj_v["k"].shift(1) > kdj_v["d"].shift(1)
    return (ema_cond & rsi_cond & macd_cond & kdj_cond).fillna(False)


def _resample_ohlc(s_ohlc: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample OHLC to a coarser TF (label='right', closed='right')."""
    return s_ohlc.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()


def _build_btc_4h_bull_panel_15m(candles: pd.DataFrame) -> pd.Series:
    """L1 — BTC 4h ALL-BULLISH boolean reindexed to BTC 15m grid (tz-naive index)."""
    btc = (
        candles[candles["asset"] == "BTCUSDT"]
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    btc_15m = btc[["open", "high", "low", "close"]].astype(float)
    btc_4h = _resample_ohlc(btc_15m, "4h")
    bull_4h = _all_bullish_panel(
        btc_4h["open"], btc_4h["high"], btc_4h["low"], btc_4h["close"],
        include_ema_slow=True,
    )
    panel_15m = bull_4h.reindex(btc_15m.index, method="ffill").fillna(False).astype(bool)
    # Convert to tz-naive internal storage (matches phase14/15)
    panel_15m.index = pd.to_datetime(panel_15m.index, utc=True).tz_convert(None)
    return panel_15m


def _build_coin_panels_at_entry(
    candles: pd.DataFrame, trades_df: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """For each row in trades_df, compute (L2_daily_bull, L3_1h_bull) booleans.

    Build L2/L3 per asset once, cache on tz-naive 15m grid, then reindex via
    np.searchsorted (last <= entry_time -> ffill).
    """
    # Pre-group entry times per asset for vectorized lookup
    trades_df = trades_df.reset_index(drop=True).copy()
    trades_df["_idx"] = np.arange(len(trades_df))

    out_l2 = np.zeros(len(trades_df), dtype=bool)
    out_l3 = np.zeros(len(trades_df), dtype=bool)

    cs = candles.sort_values(["asset", "timestamp"])
    assets_in_trades = set(trades_df["asset"].unique())

    n_done = 0
    for asset, sub in cs.groupby("asset", sort=False):
        if asset not in assets_in_trades:
            continue
        sub = sub.sort_values("timestamp")
        ts = pd.to_datetime(sub["timestamp"], utc=True)
        s_close = pd.Series(sub["close"].astype(float).to_numpy(), index=ts)
        s_open = pd.Series(sub["open"].astype(float).to_numpy(), index=ts)
        s_high = pd.Series(sub["high"].astype(float).to_numpy(), index=ts)
        s_low = pd.Series(sub["low"].astype(float).to_numpy(), index=ts)
        df_15m = pd.DataFrame(
            {"open": s_open, "high": s_high, "low": s_low, "close": s_close}
        )

        # L2 — daily
        df_d = _resample_ohlc(df_15m, "1D")
        bull_d = _all_bullish_panel(
            df_d["open"], df_d["high"], df_d["low"], df_d["close"],
            include_ema_slow=True,
        )
        l2_15m = bull_d.reindex(df_15m.index, method="ffill").fillna(False).astype(bool)
        l2_15m.index = pd.to_datetime(l2_15m.index, utc=True).tz_convert(None)

        # L3 — 1h
        df_h = _resample_ohlc(df_15m, "1h")
        bull_h = _all_bullish_panel(
            df_h["open"], df_h["high"], df_h["low"], df_h["close"],
            include_ema_slow=True,
        )
        l3_15m = bull_h.reindex(df_15m.index, method="ffill").fillna(False).astype(bool)
        l3_15m.index = pd.to_datetime(l3_15m.index, utc=True).tz_convert(None)

        # Lookup at this asset's trades
        rows = trades_df[trades_df["asset"] == asset]
        if len(rows) == 0:
            continue
        entry_naive = pd.to_datetime(rows["entry_time"], utc=True).dt.tz_convert(None)
        idx_dt = pd.DatetimeIndex(entry_naive)
        l2_vals = l2_15m.reindex(idx_dt, method="ffill").to_numpy()
        l3_vals = l3_15m.reindex(idx_dt, method="ffill").to_numpy()
        # NaNs become False (early entries before warmup)
        l2_vals = np.where(pd.isna(l2_vals), False, l2_vals).astype(bool)
        l3_vals = np.where(pd.isna(l3_vals), False, l3_vals).astype(bool)
        out_l2[rows["_idx"].to_numpy()] = l2_vals
        out_l3[rows["_idx"].to_numpy()] = l3_vals

        n_done += 1
        if n_done % 25 == 0:
            print(f"    L2/L3 panels built for {n_done} assets...", flush=True)

    print(f"    L2/L3 panels built for {n_done} assets total", flush=True)
    return out_l2, out_l3


# ============================================================================
# Decision rule
# ============================================================================

def _decision_per_variant(name: str, m: dict) -> tuple[bool, dict]:
    """Apply the pre-committed PASS/FAIL gate to a (D1+cls+Cx) variant."""
    oot_n = m["oot_metrics"]["trades"]
    oot_pf = m["oot_metrics"]["pf"]
    wf_p5 = m["walkforward"].get("p5", float("nan"))
    mc_p5 = m["monte_carlo_oot"].get("p5", float("nan"))

    cond_wf = bool(np.isfinite(wf_p5) and wf_p5 > WF_P5_FLOOR)
    cond_n = bool(oot_n >= OOT_N_FLOOR)
    # No more than 5% drop relative to LOCKED_OOT_PF / LOCKED_MC_P5
    oot_pf_drop_ratio = (LOCKED_OOT_PF - oot_pf) / LOCKED_OOT_PF if np.isfinite(oot_pf) else 1.0
    mc_p5_drop_ratio = (LOCKED_MC_P5 - mc_p5) / LOCKED_MC_P5 if np.isfinite(mc_p5) else 1.0
    cond_oot_pf_no_drop = bool(oot_pf_drop_ratio <= DROP_TOLERANCE)
    cond_mc_p5_no_drop = bool(mc_p5_drop_ratio <= DROP_TOLERANCE)

    overall = cond_wf and cond_n and cond_oot_pf_no_drop and cond_mc_p5_no_drop
    detail = {
        "oot_n": int(oot_n),
        "oot_pf": float(oot_pf) if np.isfinite(oot_pf) else None,
        "wf_p5": float(wf_p5) if np.isfinite(wf_p5) else None,
        "mc_p5": float(mc_p5) if np.isfinite(mc_p5) else None,
        "cond_wf_p5_gt_floor": cond_wf,
        "cond_oot_n_ge_1500": cond_n,
        "cond_oot_pf_no_5pct_drop": cond_oot_pf_no_drop,
        "cond_mc_p5_no_5pct_drop": cond_mc_p5_no_drop,
        "oot_pf_drop_ratio": float(oot_pf_drop_ratio),
        "mc_p5_drop_ratio": float(mc_p5_drop_ratio),
        "overall": bool(overall),
    }
    print(f"\n  {name} decision gates:")
    print(f"    cond_wf_p5_gt_{WF_P5_FLOOR} ({wf_p5:.3f} > {WF_P5_FLOOR}):  "
          f"{'PASS' if cond_wf else 'FAIL'}")
    print(f"    cond_oot_n_ge_{OOT_N_FLOOR}  ({oot_n} >= {OOT_N_FLOOR}):  "
          f"{'PASS' if cond_n else 'FAIL'}")
    print(f"    cond_oot_pf_no_drop ({oot_pf:.3f} vs {LOCKED_OOT_PF}, drop {oot_pf_drop_ratio*100:.2f}%):  "
          f"{'PASS' if cond_oot_pf_no_drop else 'FAIL'}")
    print(f"    cond_mc_p5_no_drop  ({mc_p5:.3f} vs {LOCKED_MC_P5}, drop {mc_p5_drop_ratio*100:.2f}%):  "
          f"{'PASS' if cond_mc_p5_no_drop else 'FAIL'}")
    return overall, detail


# ============================================================================
# Main
# ============================================================================

def main():
    started = time.monotonic()

    print(f"loading trades from {TRADES_PATH}", flush=True)
    df = pd.read_csv(TRADES_PATH)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    print(f"  {len(df):,} trades loaded", flush=True)

    print(f"loading candles from {CANDLES_PATH}", flush=True)
    candles = pd.read_parquet(CANDLES_PATH)
    candles["timestamp"] = pd.to_datetime(candles["timestamp"], utc=True)

    # ----- L1 panel (BTC 4h ALL-BULLISH) ----------------------------------
    print("\nbuilding L1 (BTC 4h ALL-BULLISH) panel...", flush=True)
    l1_panel = _build_btc_4h_bull_panel_15m(candles)
    print(f"  L1 active rate over BTC 15m timeline: {l1_panel.mean()*100:.2f}%", flush=True)

    # Look up L1 at each trade's entry_time
    entry_naive = pd.to_datetime(df["entry_time"], utc=True).dt.tz_convert(None)
    df["L1_bull"] = l1_panel.reindex(
        pd.DatetimeIndex(entry_naive), method="ffill"
    ).fillna(False).to_numpy().astype(bool)
    print(f"  L1 active at entry: {df['L1_bull'].mean()*100:.2f}% of trades", flush=True)

    # ----- L2 + L3 panels per asset ---------------------------------------
    print("\nbuilding L2 (coin daily) + L3 (coin 1h) ALL-BULLISH panels per asset...",
          flush=True)
    l2_arr, l3_arr = _build_coin_panels_at_entry(candles, df)
    df["L2_bull"] = l2_arr
    df["L3_bull"] = l3_arr
    print(f"  L2 active at entry: {df['L2_bull'].mean()*100:.2f}% of trades")
    print(f"  L3 active at entry: {df['L3_bull'].mean()*100:.2f}% of trades")

    # ----- Cascade admit masks --------------------------------------------
    # SHORT entry: layer ALL-BULLISH = trend-fight risk = BAD. Admit by *avoiding* bull layers.
    bull_count = (
        df["L1_bull"].astype(int) + df["L2_bull"].astype(int) + df["L3_bull"].astype(int)
    )
    df["admit_C1"] = (bull_count == 0)            # NONE bullish -> admit
    df["admit_C2"] = (~df["L1_bull"])             # L1 not bullish -> admit
    df["admit_C3"] = (bull_count <= 1)            # >=2 NOT bullish -> admit (i.e. bull_count <= 1)
    print(f"  C1 admit rate: {df['admit_C1'].mean()*100:.2f}%")
    print(f"  C2 admit rate: {df['admit_C2'].mean()*100:.2f}%")
    print(f"  C3 admit rate: {df['admit_C3'].mean()*100:.2f}%")

    # ----- Classifier -----------------------------------------------------
    if not (MODEL_PATH.exists() and META_PATH.exists()):
        raise FileNotFoundError("v3 classifier missing — required for decision rule.")
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    threshold = float(meta["threshold_train_q50"])
    feat_cols = FEATURE_NAMES + ["is_short"]
    X = df[feat_cols].to_numpy(dtype=float)
    df["score"] = model.predict_proba(X)[:, 1]
    print(f"\nv3 classifier loaded; threshold (TRAIN q0.5) = {threshold:.4f}")

    # ----- 8 metric sweeps -------------------------------------------------
    print("\n=== Variant comparison ===")

    def _run_sweep():
        out = {}
        out["d1_raw"] = _summarize(df, "D1 raw")
        out["d1_c1"] = _summarize(df[df["admit_C1"]].reset_index(drop=True),
                                  "D1 + C1 (strict AND)")
        out["d1_c2"] = _summarize(df[df["admit_C2"]].reset_index(drop=True),
                                  "D1 + C2 (L1 hard veto)")
        out["d1_c3"] = _summarize(df[df["admit_C3"]].reset_index(drop=True),
                                  "D1 + C3 (soft 2-of-3)")
        cls_mask = df["score"] >= threshold
        out["d1_cls"] = _summarize(df[cls_mask].reset_index(drop=True),
                                   "D1 + cls (current locked)")
        out["d1_cls_c1"] = _summarize(
            df[cls_mask & df["admit_C1"]].reset_index(drop=True), "D1 + cls + C1"
        )
        out["d1_cls_c2"] = _summarize(
            df[cls_mask & df["admit_C2"]].reset_index(drop=True), "D1 + cls + C2"
        )
        out["d1_cls_c3"] = _summarize(
            df[cls_mask & df["admit_C3"]].reset_index(drop=True), "D1 + cls + C3"
        )
        return out

    sweep_pass1 = _run_sweep()

    print("\n=== Reproducibility check (rerun) ===")
    sweep_pass2 = _run_sweep()

    def _reproduce_ok(a: dict, b: dict) -> bool:
        for k in a:
            ma, mb = a[k], b[k]
            for sec in ("train_metrics", "oot_metrics", "walkforward", "monte_carlo_oot"):
                if ma[sec] != mb[sec]:
                    # numeric comparisons need allclose for safety; use isclose on each scalar
                    da, db_ = ma[sec], mb[sec]
                    if set(da.keys()) != set(db_.keys()):
                        return False
                    for kk in da:
                        va, vb = da[kk], db_[kk]
                        if isinstance(va, float) and isinstance(vb, float):
                            if np.isnan(va) and np.isnan(vb):
                                continue
                            if not np.isclose(va, vb, atol=1e-12, rtol=0):
                                return False
                        elif va != vb:
                            return False
        return True

    repro_ok = _reproduce_ok(sweep_pass1, sweep_pass2)
    print(f"REPRODUCIBILITY: {'PASS' if repro_ok else 'FAIL'}")

    # ----- Decision per cascade variant (on D1 + cls + Cx) ----------------
    print("\n=== Decision rule per (D1 + cls + Cx) variant ===")
    decisions = {}
    decisions["C1"] = _decision_per_variant("D1 + cls + C1", sweep_pass1["d1_cls_c1"])
    decisions["C2"] = _decision_per_variant("D1 + cls + C2", sweep_pass1["d1_cls_c2"])
    decisions["C3"] = _decision_per_variant("D1 + cls + C3", sweep_pass1["d1_cls_c3"])

    winners = [k for k, (ok, _) in decisions.items() if ok]
    if len(winners) == 0:
        final_decision = "DROP - current config holds"
    elif len(winners) == 1:
        final_decision = f"STACK {winners[0]}"
    else:
        # Tie-break: highest WF p5 wins
        best = max(winners, key=lambda k: decisions[k][1]["wf_p5"] or float("-inf"))
        final_decision = f"STACK {best}"

    print(f"\nFINAL DECISION: {final_decision}")

    # ----- JSON out --------------------------------------------------------
    out = {
        "phase": "17.B - 3-layer regime cascade (BTC 4h, coin daily, coin 1h)",
        "snapshot": SNAPSHOT_DATE,
        "train_oot_boundary": str(TRAIN_OOT_BOUNDARY),
        "wf_fold_months": WF_FOLD_MONTHS,
        "mc_n_resamples": MC_N_RESAMPLES,
        "random_seed": RANDOM_SEED,
        "v3_threshold": threshold,
        "locked_reference": {
            "oot_pf": LOCKED_OOT_PF,
            "mc_p5": LOCKED_MC_P5,
            "wf_p5": LOCKED_WF_P5,
        },
        "decision_rule": {
            "wf_p5_floor": WF_P5_FLOOR,
            "oot_n_floor": OOT_N_FLOOR,
            "drop_tolerance_rel": DROP_TOLERANCE,
        },
        "layer_active_rates_at_entry": {
            "L1_btc_4h": float(df["L1_bull"].mean()),
            "L2_coin_daily": float(df["L2_bull"].mean()),
            "L3_coin_1h": float(df["L3_bull"].mean()),
        },
        "cascade_admit_rates": {
            "C1": float(df["admit_C1"].mean()),
            "C2": float(df["admit_C2"].mean()),
            "C3": float(df["admit_C3"].mean()),
        },
        "sweep": sweep_pass1,
        "reproducibility_pass": bool(repro_ok),
        "per_variant_decision": {k: v[1] for k, v in decisions.items()},
        "final_decision": final_decision,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {OUT_PATH}")
    print(f"total wall: {time.monotonic() - started:.1f}s")


if __name__ == "__main__":
    main()
