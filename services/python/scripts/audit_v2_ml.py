"""Audit v2+ML scanner for data-dredging bias and fee impact.

Produces a strict out-of-time (OOT) evaluation using the last 3 calendar months
as a held-out test set, threshold selected only on an 80/20 split of the
pre-OOT data (not on OOT), and net-of-fee PF.

Run from services/python/:
    python scripts/audit_v2_ml.py
"""

import sys

sys.path.insert(0, ".")

import os
from datetime import timezone

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sqlalchemy import create_engine, text

DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
ROUND_TRIP_FEE = 0.0012  # Gate.io 0.06% taker x 2 legs

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

MIN_TRADES = 30

LGBM_PARAMS = dict(
    n_estimators=200,
    num_leaves=15,
    min_child_samples=30,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    class_weight="balanced",
    verbose=-1,
    random_state=42,
)


def compute_metrics(pnls: np.ndarray, fee: float = 0.0) -> dict:
    """Return trades/WR/PF gross and net of fee."""
    if len(pnls) == 0:
        return {"trades": 0, "wr": 0.0, "pf_gross": 0.0, "pf_net": 0.0, "avg_pnl": 0.0}
    net = pnls - fee
    wins_g = pnls[pnls > 0]
    loss_g = pnls[pnls <= 0]
    wins_n = net[net > 0]
    loss_n = net[net <= 0]
    pf_gross = wins_g.sum() / abs(loss_g.sum()) if loss_g.sum() != 0 else float("inf")
    pf_net = wins_n.sum() / abs(loss_n.sum()) if loss_n.sum() != 0 else float("inf")
    wr = (pnls > 0).mean()
    return {
        "trades": len(pnls),
        "wr": wr,
        "pf_gross": pf_gross,
        "pf_net": pf_net,
        "avg_pnl": float(pnls.mean()),
    }


def sweep_thresholds(pnls: np.ndarray, proba: np.ndarray,
                     label: str = "") -> tuple[float, dict]:
    """Sweep [0.50..0.75] and pick best PF on the given split.

    Returns (best_thresh, metrics_at_best_thresh).
    """
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]
    best_pf = 0.0
    best_thresh = 0.50
    best_m: dict = {}
    for t in thresholds:
        mask = proba >= t
        m = compute_metrics(pnls[mask])
        if m["trades"] >= MIN_TRADES and m["pf_gross"] > best_pf:
            best_pf = m["pf_gross"]
            best_thresh = t
            best_m = m
    if label:
        print(f"  [{label}] best threshold sweep: {best_thresh:.2f} => PF_gross={best_pf:.2f} "
              f"({best_m.get('trades', 0)} trades)")
    return best_thresh, best_m


def main():
    engine = create_engine(DB_URL)

    # ── Load data ────────────────────────────────────────────────────
    print("Loading scanner_short_v2 variant=A ...")
    df = pd.read_sql(
        "SELECT * FROM scanner_short_v2 WHERE variant = 'A' ORDER BY signal_time",
        engine,
    )
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    df = df.sort_values("signal_time").reset_index(drop=True)
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    print(f"  {len(df)} trades, {df['signal_time'].min().date()} -> {df['signal_time'].max().date()}")

    # ── OOT cutoff: last 3 calendar months ──────────────────────────
    max_date = df["signal_time"].max()
    oot_start = (max_date - pd.DateOffset(months=3)).floor("D")
    pre_oot = df[df["signal_time"] < oot_start].copy().reset_index(drop=True)
    oot = df[df["signal_time"] >= oot_start].copy().reset_index(drop=True)
    print(f"\nOOT split: < {oot_start.date()} => pre-OOT {len(pre_oot)}, "
          f"OOT {len(oot)} ({oot_start.date()} -> {max_date.date()})")

    # ── Split pre-OOT into train (80%) + val (20%) ──────────────────
    n_pre = len(pre_oot)
    val_start = int(n_pre * 0.80)
    train_df = pre_oot.iloc[:val_start].copy().reset_index(drop=True)
    val_df = pre_oot.iloc[val_start:].copy().reset_index(drop=True)
    print(f"  Train: {len(train_df)} trades  Val: {len(val_df)} trades")

    # ── Train LightGBM on train split ───────────────────────────────
    print("\nTraining LightGBM on train split ...")
    X_train = train_df[FEATURE_COLS].values
    y_train = (train_df["pnl_pct"] > 0).astype(int).values
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    # ── Select threshold on VALIDATION set (not OOT) ────────────────
    val_proba = model.predict_proba(val_df[FEATURE_COLS].values)[:, 1]
    val_pnls = val_df["pnl_pct"].values
    print("\nThreshold sweep on VALIDATION set (this is the only allowed dredge):")
    best_thresh, _ = sweep_thresholds(val_pnls, val_proba, label="VAL")

    # ── Evaluate on OOT (out-of-time, never touched during selection) ─
    oot_proba = model.predict_proba(oot[FEATURE_COLS].values)[:, 1]
    oot_pnls = oot["pnl_pct"].values

    oot_raw = compute_metrics(oot_pnls, fee=ROUND_TRIP_FEE)
    oot_mask = oot_proba >= best_thresh
    oot_filtered_pnls = oot_pnls[oot_mask]
    oot_filtered = compute_metrics(oot_filtered_pnls, fee=ROUND_TRIP_FEE)

    # ── Load dashboard numbers (dredged) ────────────────────────────
    dash_df = pd.read_sql("SELECT pnl_pct FROM scanner_short_v2_ml_filtered", engine)
    dash_pnls = dash_df["pnl_pct"].values
    dash = compute_metrics(dash_pnls, fee=ROUND_TRIP_FEE)

    # ── Print comparison table ───────────────────────────────────────
    hdr = f"\n{'':45} {'Trades':>7} {'WR':>7} {'PF (gross)':>11} {'PF (net fees)':>14}"
    sep = "-" * 85
    print("\n" + "=" * 85)
    print("AUDIT COMPARISON TABLE")
    print("=" * 85)
    print(hdr)
    print(sep)

    def row(label, m):
        pf_g = f"{m['pf_gross']:.2f}" if m['pf_gross'] != float('inf') else "inf"
        pf_n = f"{m['pf_net']:.2f}" if m['pf_net'] != float('inf') else "inf"
        wr = f"{m['wr']:.1%}"
        return f"{label:<45} {m['trades']:>7} {wr:>7} {pf_g:>11} {pf_n:>14}"

    print(row("Dashboard (dredged, OOS)", dash))
    print(row("OOT raw (no filter)", oot_raw))
    print(row(f"OOT ML>={best_thresh:.2f} (val-selected threshold)", oot_filtered))
    print(sep)

    print(f"\nFee drag per trade: {ROUND_TRIP_FEE*100:.2f}%  "
          f"(avg gross PnL OOT filtered: {oot_filtered['avg_pnl']*100:.2f}%)")

    dredge_ratio = dash["pf_gross"] / max(oot_filtered["pf_gross"], 0.01)
    print(f"Dredge inflation: dashboard PF {dash['pf_gross']:.2f} vs OOT PF "
          f"{oot_filtered['pf_gross']:.2f}  => {dredge_ratio:.1f}x inflation")

    print("\nDone.")


if __name__ == "__main__":
    main()
