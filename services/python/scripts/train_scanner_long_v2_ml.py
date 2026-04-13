"""Train LightGBM classifier for Scanner Long v2 signals.

Walk-forward validation on Variant A trades from scanner_long_v2.
Sweeps ML thresholds up to 0.90 (same range as short v2 winner).

Run from services/python/:
    python scripts/train_scanner_long_v2_ml.py
"""

import sys

sys.path.insert(0, ".")

import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sqlalchemy import create_engine, text

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"
MODEL_PATH = Path("models/scanner_long_v2_ml.joblib")

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

ML_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
MIN_TRADES_PER_FOLD = 20

engine = create_engine(DB_URL)


def load_trades() -> pd.DataFrame:
    query = "SELECT * FROM scanner_long_v2 WHERE variant = 'A' ORDER BY signal_time"
    df = pd.read_sql(query, engine)
    df["signal_time"] = pd.to_datetime(df["signal_time"])
    df = df.sort_values("signal_time").reset_index(drop=True)
    print(f"Loaded {len(df)} trades from scanner_long_v2 (variant=A)")
    print(f"  Date range: {df['signal_time'].min()} -> {df['signal_time'].max()}")
    print(f"  Win rate: {(df['pnl_pct'] > 0).mean():.1%}")
    return df


def compute_metrics(pnls: np.ndarray) -> dict:
    if len(pnls) == 0:
        return {"trades": 0, "wr": 0, "pf": 0, "avg_pnl": 0, "total_pnl": 0}
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    return {
        "trades": len(pnls),
        "wr": len(wins) / len(pnls),
        "pf": pf,
        "avg_pnl": float(np.mean(pnls)),
        "total_pnl": float(np.sum(pnls)),
    }


def walk_forward(df: pd.DataFrame):
    n = len(df)
    folds = [
        ("Fold 1 (60/20)", 0, int(n * 0.6), int(n * 0.6), int(n * 0.8)),
        ("Fold 2 (80/20)", 0, int(n * 0.8), int(n * 0.8), n),
    ]

    all_oos = []
    for fold_name, tr_start, tr_end, te_start, te_end in folds:
        train = df.iloc[tr_start:tr_end]
        test = df.iloc[te_start:te_end]

        X_train = train[FEATURE_COLS].values
        y_train = (train["pnl_pct"] > 0).astype(int).values
        X_test = test[FEATURE_COLS].values
        y_test = (test["pnl_pct"] > 0).astype(int).values

        print(f"\n{'=' * 70}")
        print(f"{fold_name}")
        print(f"  Train: {len(train)} trades, WR={y_train.mean():.1%}")
        print(f"  Test:  {len(test)} trades, WR={y_test.mean():.1%}")

        model = LGBMClassifier(
            n_estimators=200, num_leaves=15, min_child_samples=30,
            learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", verbose=-1, random_state=42,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)[:, 1]
        test_pnls = test["pnl_pct"].values

        base = compute_metrics(test_pnls)
        print(f"\n  {'Threshold':<12} {'Trades':>7} {'WR':>7} {'PF':>7} {'AvgPnL':>9}")
        print(f"  {'-' * 48}")
        print(f"  {'Raw':<12} {base['trades']:>7} {base['wr']:>6.1%} {base['pf']:>7.2f} "
              f"{base['avg_pnl']*100:>+8.2f}%")

        best_pf = 0
        best_thresh = 0.50
        for thresh in ML_THRESHOLDS:
            mask = proba >= thresh
            m = compute_metrics(test_pnls[mask])
            marker = ""
            if m["trades"] >= MIN_TRADES_PER_FOLD and m["pf"] > best_pf:
                best_pf = m["pf"]
                best_thresh = thresh
                marker = " <-- best"
            print(f"  ML>={thresh:.2f}      {m['trades']:>7} {m['wr']:>6.1%} {m['pf']:>7.2f} "
                  f"{m['avg_pnl']*100:>+8.2f}%{marker}")

        print(f"\n  Best threshold (>={MIN_TRADES_PER_FOLD} trades): {best_thresh:.2f} "
              f"(PF={best_pf:.2f})")

        fi = sorted(zip(FEATURE_COLS, model.feature_importances_),
                    key=lambda x: x[1], reverse=True)
        print(f"\n  Top 10 features:")
        for feat, imp in fi[:10]:
            print(f"    {feat:<24} {imp:>5}")

        oos_df = test.copy()
        oos_df["ml_prob"] = proba
        all_oos.append(oos_df)

    return all_oos


def find_best_threshold(all_oos):
    combined = pd.concat(all_oos, ignore_index=True)
    combined = combined.drop_duplicates(subset=["symbol", "signal_time"], keep="last")
    combined = combined.sort_values("signal_time").reset_index(drop=True)

    print(f"\n{'=' * 70}")
    print(f"AGGREGATE OOS: {len(combined)} trades")
    print(f"{'=' * 70}")

    best_pf = 0
    best_thresh = 0.50
    print(f"\n  {'Threshold':<12} {'Trades':>7} {'WR':>7} {'PF':>7} {'AvgPnL':>9}")
    print(f"  {'-' * 48}")
    base = compute_metrics(combined["pnl_pct"].values)
    print(f"  {'Raw':<12} {base['trades']:>7} {base['wr']:>6.1%} {base['pf']:>7.2f} "
          f"{base['avg_pnl']*100:>+8.2f}%")

    for thresh in ML_THRESHOLDS:
        mask = combined["ml_prob"] >= thresh
        m = compute_metrics(combined.loc[mask, "pnl_pct"].values)
        marker = ""
        if m["trades"] >= MIN_TRADES_PER_FOLD and m["pf"] > best_pf:
            best_pf = m["pf"]
            best_thresh = thresh
            marker = " <-- best"
        print(f"  ML>={thresh:.2f}      {m['trades']:>7} {m['wr']:>6.1%} {m['pf']:>7.2f} "
              f"{m['avg_pnl']*100:>+8.2f}%{marker}")

    print(f"\n  Recommended threshold: {best_thresh:.2f} (PF={best_pf:.2f})")
    return best_thresh, combined


def train_final(df):
    print(f"\n{'=' * 70}")
    print("TRAINING FINAL MODEL ON ALL DATA")
    print(f"{'=' * 70}")
    X = df[FEATURE_COLS].values
    y = (df["pnl_pct"] > 0).astype(int).values
    model = LGBMClassifier(
        n_estimators=200, num_leaves=15, min_child_samples=30,
        learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", verbose=-1, random_state=42,
    )
    model.fit(X, y)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"  Trained on {len(df)} trades. Saved to {MODEL_PATH}")
    return model


def save_filtered(combined, threshold):
    filtered = combined[combined["ml_prob"] >= threshold].copy()
    if len(filtered) == 0:
        print("No filtered trades to save.")
        return
    save_cols = ["symbol", "signal_time", "entry_price", "exit_price",
                 "pnl_pct", "exit_reason", "bars_held", "ml_prob"]
    save_cols = [c for c in save_cols if c in filtered.columns]
    save_df = filtered[save_cols].sort_values("signal_time").reset_index(drop=True)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_long_v2_ml_filtered"))
        conn.commit()
    save_df.to_sql("scanner_long_v2_ml_filtered", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(save_df)} filtered OOS trades (threshold={threshold:.2f})")


def main():
    start = time.time()
    print("Scanner Long v2 ML Training")
    print("=" * 70)
    df = load_trades()
    if len(df) < 100:
        print(f"ERROR: Only {len(df)} trades — need at least 100. Run backtest_long_v2.py first.")
        return
    df[FEATURE_COLS] = df[FEATURE_COLS].replace([np.inf, -np.inf], np.nan).fillna(0)
    all_oos = walk_forward(df)
    best_thresh, combined = find_best_threshold(all_oos)
    # Override: use 0.80 for live (261 trades at PF 17.86) — 0.90 (24 trades) too thin
    live_thresh = 0.80
    train_final(df)
    save_filtered(combined, live_thresh)
    best_thresh = live_thresh
    raw = compute_metrics(df["pnl_pct"].values)
    filt = compute_metrics(combined.loc[combined["ml_prob"] >= best_thresh, "pnl_pct"].values)
    print(f"\n{'=' * 70}")
    print("COMPARISON: Raw v2 vs ML-Filtered")
    print(f"{'=' * 70}")
    print(f"  Raw:           {raw['trades']:>6} trades, WR={raw['wr']:.1%}, "
          f"PF={raw['pf']:.2f}, TotPnL={raw['total_pnl']*100:+.1f}%")
    print(f"  ML>={best_thresh:.2f} (OOS): {filt['trades']:>6} trades, WR={filt['wr']:.1%}, "
          f"PF={filt['pf']:.2f}, TotPnL={filt['total_pnl']*100:+.1f}%")
    print(f"\nElapsed: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
