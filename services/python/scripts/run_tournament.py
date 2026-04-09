"""ML Tournament: Compare trained models vs baselines.

Uses existing XGBoost/LightGBM predictions (.npz) + original parquet
with _forward_ret to compute real trading metrics. Computes 3 baselines
from the raw feature data. Outputs tournament_summary.json for the dashboard.
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path("C:/Users/togat/Desktop/AI-Finance")
DATA_PATH = PROJECT / "data/features/tournament_data.parquet"
RESULTS_DIR = PROJECT / "models/results"
OUTPUT_PATH = RESULTS_DIR / "tournament_summary.json"

FEE_RATE = 0.002  # 0.2% round-trip fee
HOLD_PERIOD = 30   # 30 × 4h = 5 days (forward return horizon)
MAX_RETURN = 0.50  # Cap at +50% per trade
MIN_RETURN = -0.25 # Cap at -25% (simulates stop-loss)
TRADES_PER_YEAR = 365 / 5  # ~73 non-overlapping 5-day trades per year


def load_data():
    """Load parquet with forward returns."""
    df = pd.read_parquet(DATA_PATH)
    valid = df["_target"].notna()
    df = df.loc[valid].reset_index(drop=True)
    df["_forward_ret"] = df["_forward_ret"].clip(MIN_RETURN, MAX_RETURN)
    return df


def get_rebalance_dates(df):
    """Get shared rebalancing dates: every 30 candles from the global timeline."""
    all_ts = sorted(df["_timestamp"].unique())
    return all_ts[::HOLD_PERIOD]


def sample_at_rebalance(df, rebalance_dates):
    """Keep only rows at shared rebalance dates (all tokens evaluated together)."""
    return df[df["_timestamp"].isin(set(rebalance_dates))].copy()


def compute_trading_metrics(timestamps, returns, signals, name, tier):
    """Compute trading metrics from non-overlapping trades.

    signals: 1=long, -1=short, 0=flat (cash)
    For longs: profit when price goes up  (trade_ret = forward_ret - fee)
    For shorts: profit when price drops   (trade_ret = -forward_ret - fee)
    """
    active_mask = signals != 0
    n_trades = int(active_mask.sum())

    if n_trades == 0:
        return {
            "name": name, "tier": tier, "sharpe": 0.0,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "win_rate": 0.0, "profit_factor": 0.0, "trades": 0,
            "avg_return_pct": 0.0, "equity_curve": [],
            "long_trades": 0, "short_trades": 0,
        }

    # Directional returns: long=+ret, short=-ret, both pay fee
    dir_returns = signals * returns  # 1*ret for long, -1*ret for short, 0 for flat
    trade_rets = dir_returns[active_mask] - FEE_RATE

    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets <= 0]
    win_rate = float(len(wins) / n_trades) * 100
    gross_profit = float(wins.sum()) if len(wins) > 0 else 0.0
    gross_loss = float(np.abs(losses.sum())) if len(losses) > 0 else 1e-9
    profit_factor = gross_profit / gross_loss
    avg_ret = float(trade_rets.mean()) * 100

    # Portfolio-level: equal-weight average return per period
    tmp = pd.DataFrame({"ts": timestamps, "dir_ret": dir_returns, "active": active_mask})
    active = tmp[tmp["active"]].copy()
    active["net_ret"] = active["dir_ret"] - FEE_RATE
    period_ret = active.groupby("ts")["net_ret"].mean().sort_index()

    all_ts = sorted(tmp["ts"].unique())
    period_ret = period_ret.reindex(all_ts, fill_value=0.0)

    # Compound equity
    equity = (1 + period_ret).cumprod().values

    # Sharpe (annualized)
    pr = period_ret.values
    mean_p = pr.mean()
    std_p = pr.std()
    sharpe = (mean_p / std_p) * np.sqrt(TRADES_PER_YEAR) if std_p > 0 else 0.0

    total_return = float(equity[-1] - 1.0) * 100

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / running_max
    max_dd = float(np.min(drawdowns)) * 100

    step = max(1, len(equity) // 200)
    eq_sampled = [round(float(v), 4) for v in equity[::step]]

    return {
        "name": name, "tier": tier,
        "sharpe": round(sharpe, 3),
        "total_return_pct": round(total_return, 2),
        "max_drawdown_pct": round(max_dd, 2),
        "win_rate": round(win_rate, 2),
        "profit_factor": round(profit_factor, 3),
        "trades": n_trades,
        "avg_return_pct": round(avg_ret, 4),
        "equity_curve": eq_sampled,
        "long_trades": int((signals == 1).sum()),
        "short_trades": int((signals == -1).sum()),
    }


def baseline_buy_and_hold(df):
    """B0: Always long (no shorting ability)."""
    signals = np.ones(len(df))
    return compute_trading_metrics(
        df["_timestamp"].values, df["_forward_ret"].values,
        signals, "Buy & Hold", "baseline",
    )


def baseline_sma50_long_short(df):
    """B1: Long above SMA50, short below."""
    above = df["above_sma50"].values
    signals = np.where(above > 0.5, 1, -1).astype(int)
    return compute_trading_metrics(
        df["_timestamp"].values, df["_forward_ret"].values,
        signals, "SMA50 L/S", "baseline",
    )


def baseline_momentum_long_short(df):
    """B2: Long when momentum > 0, short when < 0."""
    mom = df["ret_42"].values
    signals = np.where(mom > 0, 1, -1).astype(int)
    return compute_trading_metrics(
        df["_timestamp"].values, df["_forward_ret"].values,
        signals, "Momentum L/S", "baseline",
    )


def ml_model_from_predictions(df_full, rebalance_dates, model_name, display_name):
    """Load ML predictions, attach to full data, then sample at rebalance dates."""
    pred_path = RESULTS_DIR / f"{model_name}_predictions.npz"
    if not pred_path.exists():
        print(f"  Skipping {model_name}: no predictions file found")
        return None

    data = np.load(pred_path)
    preds_all = data["preds"]
    probs_all = data["probs"]

    # Predictions cover the LAST N rows of full data (walk-forward starts late)
    n_preds = len(preds_all)
    offset = len(df_full) - n_preds

    # Attach predictions to the full dataframe rows they correspond to
    df_pred = df_full.iloc[offset:].copy().reset_index(drop=True)
    df_pred["_ml_signal"] = preds_all
    df_pred["_ml_prob"] = probs_all

    # Sample at shared rebalance dates
    df_sampled = df_pred[df_pred["_timestamp"].isin(set(rebalance_dates))].copy()

    probs_sampled = df_sampled["_ml_prob"].values
    returns = df_sampled["_forward_ret"].values
    timestamps = df_sampled["_timestamp"].values

    # Long/short signals: high prob → long, low prob → short, middle → flat
    signals = np.where(probs_sampled >= 0.55, 1, np.where(probs_sampled < 0.45, -1, 0)).astype(int)

    metrics = compute_trading_metrics(
        timestamps, returns, signals, display_name, "ml"
    )

    # Classification stats from saved JSON
    metrics_path = RESULTS_DIR / f"{model_name}_metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            m = json.load(f)
        metrics["auc"] = round(m.get("auc", 0), 4)
        metrics["accuracy_pct"] = round(m.get("accuracy", 0) * 100, 2)
        metrics["precision_pct"] = round(m.get("precision", 0) * 100, 2)
        metrics["signal_rate_pct"] = round(float(preds_all.mean()) * 100, 2)

    # Confidence distribution
    buckets = [(0.5, 0.55), (0.55, 0.6), (0.6, 0.65), (0.65, 0.7),
               (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
    conf_dist = []
    for lo, hi in buckets:
        mask = (probs_all >= lo) & (probs_all < hi)
        count = int(mask.sum())
        if count > 0:
            actuals = data["actuals"][mask[:len(data["actuals"])]]
            hit_rate = round(float(actuals.mean()) * 100, 1) if len(actuals) > 0 else 0.0
        else:
            hit_rate = 0.0
        conf_dist.append({
            "bucket": f"{lo:.0%}-{hi:.0%}",
            "count": count,
            "hit_rate": hit_rate,
        })
    metrics["confidence_distribution"] = conf_dist

    return metrics


def main():
    print("=" * 70)
    print("ML TOURNAMENT — Scoring from existing predictions")
    print("=" * 70)

    print("\nLoading data...")
    df_full = load_data()
    rebalance_dates = get_rebalance_dates(df_full)
    df = sample_at_rebalance(df_full, rebalance_dates)
    n_tokens = df["_asset"].nunique()
    ts_min = str(df["_timestamp"].min())
    ts_max = str(df["_timestamp"].max())
    print(f"  Full: {len(df_full):,} rows")
    print(f"  Rebalance dates: {len(rebalance_dates)} (every {HOLD_PERIOD} candles = ~5 days)")
    print(f"  Sampled rows: {len(df):,} ({n_tokens} tokens)")
    print(f"  Period: {ts_min} to {ts_max}")

    results = []

    # Baselines (on sampled non-overlapping data)
    print("\nComputing baselines...")
    for fn in [baseline_buy_and_hold, baseline_sma50_long_short, baseline_momentum_long_short]:
        r = fn(df)
        print(f"  {r['name']}: Sharpe={r['sharpe']}, Return={r['total_return_pct']}%, "
              f"MaxDD={r['max_drawdown_pct']}%, WR={r['win_rate']}%, Trades={r['trades']:,}")
        results.append(r)

    # ML models (from saved .npz predictions — uses full data then samples)
    print("\nScoring ML models from saved predictions...")
    ml_models = [
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
    ]
    for model_id, display in ml_models:
        r = ml_model_from_predictions(df_full, rebalance_dates, model_id, display)
        if r:
            print(f"  {r['name']}: Sharpe={r['sharpe']}, Return={r['total_return_pct']}%, "
                  f"MaxDD={r['max_drawdown_pct']}%, WR={r['win_rate']}%, "
                  f"Trades={r['trades']:,}, AUC={r.get('auc', 'n/a')}")
            results.append(r)

    # Sort by Sharpe
    results.sort(key=lambda x: x["sharpe"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_samples": len(df_full),
        "total_tokens": n_tokens,
        "period": f"{ts_min} to {ts_max}",
        "fee_rate_pct": FEE_RATE * 100,
        "models": results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    # Print leaderboard
    print(f"\nSaved to {OUTPUT_PATH}")
    print("\n" + "=" * 90)
    print("TOURNAMENT LEADERBOARD (ranked by Sharpe)")
    print("=" * 90)
    print(f"{'#':<3} {'Model':<20} {'Tier':<10} {'Sharpe':>8} {'Return%':>10} "
          f"{'MaxDD%':>9} {'WinRate%':>10} {'PF':>8} {'Trades':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r['rank']:<3} {r['name']:<20} {r['tier']:<10} {r['sharpe']:>8.3f} "
              f"{r['total_return_pct']:>9.2f}% {r['max_drawdown_pct']:>8.2f}% "
              f"{r['win_rate']:>9.2f}% {r['profit_factor']:>8.3f} {r['trades']:>10,}")
    print("=" * 90)


if __name__ == "__main__":
    main()
