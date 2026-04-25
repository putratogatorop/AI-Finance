"""Leaderboard for the bigmover ML grid.

Reads a `_grid_summary_bigmover_ml_v1_*.csv` produced by
`backtest_bigmover_ml_grid_v1.py` and prints the top candidates after applying:
  - trades >= MIN_TRADES         (default 100 — per CLAUDE.md realistic-floor rule)
  - pf_p5   >= MIN_PF_P5          (default 1.30 — Monte Carlo robustness gate)

Sorts by total_return_pct DESC. Biggest PnL among strategies that survive the
5th-percentile bootstrap PF floor.

Usage:
    python scripts/rank_bigmover_ml_grid.py results/_grid_summary_bigmover_ml_v1_*.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

MIN_TRADES = 100
MIN_PF_P5 = 1.30
TOP_N = 10


def rank(csv_path: Path, min_trades: int, min_pf_p5: float, top_n: int) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "signal", "ml_threshold", "sizing", "exit",
        "trades", "wr", "pf", "pf_p5", "total_return_pct",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in {csv_path}: {missing}")

    filt = df[(df["trades"] >= min_trades) & (df["pf_p5"] >= min_pf_p5)].copy()
    filt = filt.sort_values("total_return_pct", ascending=False).head(top_n)
    return filt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv_path", type=Path, help="path to _grid_summary_bigmover_ml_v1_*.csv")
    ap.add_argument("--min-trades", type=int, default=MIN_TRADES)
    ap.add_argument("--min-pf-p5", type=float, default=MIN_PF_P5)
    ap.add_argument("--top", type=int, default=TOP_N)
    args = ap.parse_args()

    if not args.csv_path.exists():
        print(f"file not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    result = rank(args.csv_path, args.min_trades, args.min_pf_p5, args.top)
    if result.empty:
        print(
            f"no variants passed filter (trades >= {args.min_trades} "
            f"AND pf_p5 >= {args.min_pf_p5})"
        )
        return

    cols = [
        "signal", "ml_threshold", "sizing", "exit",
        "trades", "wr", "pf", "pf_p5", "pf_p50", "pf_p95",
        "avg_pnl_pct", "total_return_pct",
        "max_drawdown_pct", "sharpe", "run_id",
    ]
    cols = [c for c in cols if c in result.columns]
    print(f"\nTop {len(result)} variants (trades >= {args.min_trades}, "
          f"pf_p5 >= {args.min_pf_p5}):\n")
    print(result[cols].to_string(index=False))


if __name__ == "__main__":
    main()
