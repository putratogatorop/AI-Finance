# scripts/run_backtest_v4.py
"""Run v4 backtest across all coins and conviction thresholds.

Loads OOS predictions from walk-forward training + 15m price data,
runs backtests at multiple conviction thresholds, and produces a
summary report with spec success criteria validation.

Outputs: models/v4/backtest_report.json
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")
from src.ml.backtester_v4 import backtest_coin

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = Path("C:/Users/togat/Desktop/AI-Finance/models/v4")
DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"

V4_ASSETS = ["BTC", "ETH", "SOL", "DOGE", "XRP", "AVAX", "LINK"]
THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30]

SPEC_CRITERIA = {
    "sharpe_ratio": 1.5,
    "profit_factor": 1.4,
    "win_rate": 0.48,
    "max_drawdown_pct": 0.25,
    "min_trades_per_week": 5,
    "max_trades_per_week": 15,
    "profitable_months_pct": 0.60,
}


def load_price_data(engine, asset: str) -> pd.DataFrame:
    query = text("""
        SELECT timestamp, open, high, low, close
        FROM asset_prices_15m
        WHERE asset = :asset
        ORDER BY timestamp
    """)
    with engine.connect() as conn:
        return pd.read_sql(query, conn, params={"asset": asset})


def load_oos_predictions(asset: str) -> pd.DataFrame:
    path = MODEL_DIR / asset / "oos_predictions.parquet"
    return pd.read_parquet(path)


def merge_predictions_with_prices(prices: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
    prices["timestamp"] = pd.to_datetime(prices["timestamp"], utc=True)
    preds["timestamp"] = pd.to_datetime(preds["timestamp"], utc=True)
    merged = prices.merge(preds[["timestamp", "conviction"]], on="timestamp", how="inner")
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


def run_asset_backtest(engine, asset: str) -> dict:
    logger.info(f"  {asset}: loading data...")
    prices = load_price_data(engine, asset)
    preds = load_oos_predictions(asset)
    merged = merge_predictions_with_prices(prices, preds)
    logger.info(f"  {asset}: {len(merged):,} merged rows, "
                f"conviction range [{merged['conviction'].min():.3f}, {merged['conviction'].max():.3f}]")

    results = {}
    for threshold in THRESHOLDS:
        r = backtest_coin(merged, conviction_threshold=threshold)
        m = r["metrics"]
        n_trades = m["total_trades"]

        total_bars = len(merged)
        total_weeks = total_bars / (96 * 7)
        trades_per_week = n_trades / total_weeks if total_weeks > 0 else 0

        m["trades_per_week"] = float(trades_per_week)
        results[f"{threshold:.2f}"] = m

        logger.info(f"  {asset} @ {threshold:.2f}: "
                    f"{n_trades} trades ({trades_per_week:.1f}/wk) "
                    f"ret={m['total_return_pct']:.1%} "
                    f"sharpe={m['sharpe_ratio']:.2f} "
                    f"wr={m['win_rate']:.1%} "
                    f"dd={m['max_drawdown_pct']:.1%}")

    return results


def check_spec_criteria(metrics: dict) -> dict:
    checks = {}
    checks["sharpe_ratio"] = metrics.get("sharpe_ratio", 0) >= SPEC_CRITERIA["sharpe_ratio"]
    checks["profit_factor"] = metrics.get("profit_factor", 0) >= SPEC_CRITERIA["profit_factor"]
    checks["win_rate"] = metrics.get("win_rate", 0) >= SPEC_CRITERIA["win_rate"]
    checks["max_drawdown"] = metrics.get("max_drawdown_pct", 1) <= SPEC_CRITERIA["max_drawdown_pct"]
    checks["profitable_months"] = metrics.get("profitable_months_pct", 0) >= SPEC_CRITERIA["profitable_months_pct"]
    tpw = metrics.get("trades_per_week", 0)
    checks["trade_frequency"] = SPEC_CRITERIA["min_trades_per_week"] <= tpw <= SPEC_CRITERIA["max_trades_per_week"]
    checks["all_pass"] = all(checks.values())
    return checks


def main():
    engine = create_engine(DB_URL)
    start = time.time()

    logger.info("V4 Backtest Runner")
    logger.info(f"Assets: {V4_ASSETS}")
    logger.info(f"Thresholds: {THRESHOLDS}")

    report = {"assets": {}, "best_per_asset": {}}

    for asset in V4_ASSETS:
        logger.info(f"\nBacktesting {asset}...")
        asset_results = run_asset_backtest(engine, asset)
        report["assets"][asset] = asset_results

        best_threshold = None
        best_sharpe = -999
        for thresh_str, m in asset_results.items():
            if m["total_trades"] >= 10 and m["sharpe_ratio"] > best_sharpe:
                best_sharpe = m["sharpe_ratio"]
                best_threshold = thresh_str

        if best_threshold:
            best_m = asset_results[best_threshold]
            spec_check = check_spec_criteria(best_m)
            report["best_per_asset"][asset] = {
                "threshold": best_threshold,
                "metrics": best_m,
                "spec_compliance": spec_check,
            }

    report_path = MODEL_DIR / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    elapsed = time.time() - start
    logger.info("\n" + "=" * 60)
    logger.info(f"DONE in {elapsed:.0f}s!")
    logger.info(f"\nBest results per asset:")
    for asset, info in report.get("best_per_asset", {}).items():
        m = info["metrics"]
        spec = info["spec_compliance"]
        status = "PASS" if spec["all_pass"] else "FAIL"
        logger.info(f"  {asset} @ {info['threshold']}: "
                    f"sharpe={m['sharpe_ratio']:.2f} "
                    f"pf={m['profit_factor']:.2f} "
                    f"wr={m['win_rate']:.1%} "
                    f"dd={m['max_drawdown_pct']:.1%} "
                    f"tpw={m['trades_per_week']:.1f} "
                    f"[{status}]")

    engine.dispose()


if __name__ == "__main__":
    main()
