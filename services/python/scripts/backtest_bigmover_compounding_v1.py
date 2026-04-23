"""Compounding scenario for the 2 shortlisted bigmover accounts.

Reads trades from the sizing-sweep run (multi_bar_confirm LONG + SHORT @ 15%/5x/SL7%),
re-simulates them with COMPOUNDING equity (vs the flat sizing used in the sweep).

Two independent accounts:
  - multi_bar_confirm__long__15pct_05x_sl07   (started with $100, compounds independently)
  - multi_bar_confirm__short__15pct_05x_sl07  (started with $100, compounds independently)

Sizing per trade: notional = equity_at_entry * POSITION_PCT * LEVERAGE
  - equity_at_entry = START_EQUITY_USD + sum of realized PnL of all trades closed
                      strictly before this trade's entry_time
  - Once a trade closes, its pnl_usd lands in realized equity for later trades
  - Open trades don't reduce equity-available (margin model, not cash model)

Output:
  - results/<run_id>__<label>/{metrics,params,run_metadata}.json
  - results/<run_id>__<label>/trades_compounded.csv
  - results/<run_id>__<label>/monthly.csv      <- the key output the user wants
  - results/_compounding_summary_<sha8>_<ts>.csv
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time as _time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# =============================================================================
# CONFIG
# =============================================================================

SOURCE_RUN_PREFIX = (
    "backtest_bigmover_sizing_sweep_v1_abfe8b06_20260423T145954Z__"
)
ACCOUNTS = [
    {
        "label": "multi_bar_confirm_long_15_5_7",
        "source_folder": (
            SOURCE_RUN_PREFIX + "multi_bar_confirm__long__15pct_05x_sl07"
        ),
        "position_pct": 0.15, "leverage": 5, "sl_pct": 0.07,
        "direction": "long",
    },
    {
        "label": "multi_bar_confirm_short_15_5_7",
        "source_folder": (
            SOURCE_RUN_PREFIX + "multi_bar_confirm__short__15pct_05x_sl07"
        ),
        "position_pct": 0.15, "leverage": 5, "sl_pct": 0.07,
        "direction": "short",
    },
]

START_EQUITY_USD = 100.0
ROUND_TRIP_FEE = 0.0012  # 0.06% x 2

# Realistic max-notional caps for compounding scenarios. Uncapped compounding
# produces absurd numbers because a +edge strategy compounded across thousands
# of trades blows up exponentially. Real crypto futures have per-position
# liquidity limits; Gate.io tiered limits for top-100 alts range from ~$50k
# (small alts) to ~$5M (BTC/ETH).
#   uncapped:  no cap — pure math (for reference)
#   1M:        $1,000,000 max notional per trade — large personal account
#   100k:      $100,000 max notional per trade — mid-size account
#   10k:       $10,000 max notional per trade — small account / first 6 months live
POSITION_CAPS_USD = [None, 1_000_000, 100_000, 10_000]


# =============================================================================
# HELPERS
# =============================================================================


def _git_sha():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()


def _git_dirty():
    return bool(
        subprocess.check_output(["git", "status", "--porcelain"]).decode().strip()
    )


def compound_equity(
    trades_df, position_pct, leverage,
    start_equity=START_EQUITY_USD, notional_cap=None,
):
    """Process trades in time order, sizing each by current realized equity.

    If `notional_cap` is provided, each trade's notional is clipped to that value.
    This models real exchange per-position liquidity limits and prevents the
    strategy's positive expectancy from compounding into unrealistic numbers.

    Returns trades df with columns: equity_at_entry, notional_usd, pnl_pct_net,
    pnl_usd, equity_after_exit, was_capped (bool).
    """
    df = trades_df.reset_index(drop=True).copy()
    n = len(df)
    if n == 0:
        empty = []
        return df.assign(
            equity_at_entry=empty, notional_usd=empty, pnl_pct_net=empty,
            pnl_usd=empty, equity_after_exit=empty, was_capped=empty,
        )

    events = []
    for i, row in df.iterrows():
        events.append((int(row["entry_time"]), 1, i, "open"))
        events.append((int(row["exit_time"]), 0, i, "close"))
    events.sort(key=lambda e: (e[0], e[1]))

    equity = float(start_equity)
    equity_at_entry = np.zeros(n)
    notional_arr = np.zeros(n)
    pnl_net_arr = np.zeros(n)
    pnl_usd_arr = np.zeros(n)
    equity_after = np.zeros(n)
    was_capped = np.zeros(n, dtype=bool)

    for ts, _, idx, kind in events:
        if kind == "open":
            eq = max(equity, 1.0)
            desired_notional = eq * position_pct * leverage
            if notional_cap is not None and desired_notional > notional_cap:
                notional = float(notional_cap)
                was_capped[idx] = True
            else:
                notional = desired_notional
            equity_at_entry[idx] = eq
            notional_arr[idx] = notional
            pnl_pct = float(df.at[idx, "pnl_pct"])
            pnl_net = pnl_pct - ROUND_TRIP_FEE
            pnl_net_arr[idx] = pnl_net
            pnl_usd_arr[idx] = notional * pnl_net
        else:
            equity += pnl_usd_arr[idx]
            equity_after[idx] = equity

    df["equity_at_entry"] = equity_at_entry
    df["notional_usd"] = notional_arr
    df["pnl_pct_net"] = pnl_net_arr
    df["pnl_usd"] = pnl_usd_arr
    df["equity_after_exit"] = equity_after
    df["was_capped"] = was_capped
    return df


def monthly_aggregate(df, start_equity=START_EQUITY_USD):
    """Group by entry-month. Compute trades, wr, pf, total_pnl_usd, DD within month,
    and running equity at month-end (taken as last close within the month).
    """
    if len(df) == 0:
        return pd.DataFrame(columns=[
            "month", "trades", "wr", "pf", "avg_pnl_pct", "total_pnl_usd",
            "equity_end", "month_return_pct", "drawdown_in_month_pct",
            "avg_notional_usd",
        ])
    work = df.copy()
    work["entry_dt"] = pd.to_datetime(work["entry_time"], unit="s", utc=True)
    work["exit_dt"] = pd.to_datetime(work["exit_time"], unit="s", utc=True)
    work["month"] = work["entry_dt"].dt.strftime("%Y-%m")
    work_sorted = work.sort_values("exit_time").reset_index(drop=True)

    rows = []
    prev_equity_end = start_equity
    for month, grp in work.groupby("month", sort=True):
        m_sorted = grp.sort_values("entry_time").reset_index(drop=True)
        # Running equity within month (based on close order)
        closed_in_month = grp.sort_values("exit_time").reset_index(drop=True)
        if len(closed_in_month) == 0:
            equity_end = prev_equity_end
        else:
            equity_end = float(closed_in_month["equity_after_exit"].iloc[-1])
        wins = grp.loc[grp["pnl_pct"] > 0, "pnl_pct"]
        losses = -grp.loc[grp["pnl_pct"] < 0, "pnl_pct"]
        wins_sum = float(wins.sum())
        losses_sum = float(losses.sum())
        pf = float("inf") if losses_sum == 0 else wins_sum / losses_sum
        # Drawdown inside the month (on the equity_after_exit sequence)
        eq_series = closed_in_month["equity_after_exit"].values
        if len(eq_series) > 0:
            running_max = np.maximum.accumulate(
                np.concatenate([[prev_equity_end], eq_series])
            )[1:]
            dd_series = (eq_series - running_max) / running_max
            dd_in_month = float(dd_series.min()) if len(dd_series) > 0 else 0.0
        else:
            dd_in_month = 0.0
        rows.append({
            "month": month,
            "trades": int(len(grp)),
            "wr": float((grp["pnl_pct"] > 0).mean()),
            "pf": pf,
            "avg_pnl_pct": float(grp["pnl_pct"].mean()),
            "total_pnl_usd": float(grp["pnl_usd"].sum()),
            "equity_end": equity_end,
            "month_return_pct": float(
                (equity_end - prev_equity_end) / prev_equity_end * 100
            ),
            "drawdown_in_month_pct": float(dd_in_month * 100),
            "avg_notional_usd": float(grp["notional_usd"].mean()),
        })
        prev_equity_end = equity_end
    return pd.DataFrame(rows)


def portfolio_metrics(df, start_equity=START_EQUITY_USD):
    if len(df) == 0:
        return {
            "trades": 0, "wr": 0.0, "pf": 0.0,
            "final_equity_usd": start_equity,
            "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
            "avg_notional_usd": 0.0, "final_notional_usd": 0.0,
        }
    closed = df.sort_values("exit_time").reset_index(drop=True)
    eq = closed["equity_after_exit"].values
    running_max = np.maximum.accumulate(np.concatenate([[start_equity], eq]))[1:]
    dd = (eq - running_max) / running_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    wins = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    losses = -df.loc[df["pnl_pct"] < 0, "pnl_pct"].sum()
    pf = float("inf") if losses == 0 else float(wins / losses)
    final_eq = float(closed["equity_after_exit"].iloc[-1])
    return {
        "trades": int(len(df)),
        "wr": float((df["pnl_pct"] > 0).mean()),
        "pf": pf,
        "final_equity_usd": final_eq,
        "total_return_pct": float((final_eq - start_equity) / start_equity * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "avg_notional_usd": float(df["notional_usd"].mean()),
        "final_notional_usd": float(df["notional_usd"].iloc[-1]),
    }


# =============================================================================
# MAIN
# =============================================================================


def main():
    if _git_dirty():
        print("WARNING: git dirty.", file=sys.stderr)

    results_root = Path("results")
    base_run_id = (
        f"backtest_bigmover_compounding_v1_{_git_sha()[:8]}_"
        f"{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    )

    summary_rows = []
    for acct in ACCOUNTS:
        src = results_root / acct["source_folder"] / "trades.csv"
        if not src.exists():
            raise FileNotFoundError(
                f"Source trades.csv missing: {src}. "
                f"Run backtest_bigmover_sizing_sweep_v1.py first."
            )
        trades = pd.read_csv(src)

        for cap in POSITION_CAPS_USD:
            cap_label = "uncapped" if cap is None else f"cap{int(cap):_}".replace("_", "k" if cap < 1_000_000 else "M")
            # Prettify: 10000 -> 10k, 100000 -> 100k, 1000000 -> 1M
            if cap is None:
                cap_label = "uncapped"
            elif cap >= 1_000_000:
                cap_label = f"cap{int(cap/1_000_000)}M"
            else:
                cap_label = f"cap{int(cap/1000)}k"

            label = f"{acct['label']}__{cap_label}"
            print(f"\n[compound] {label}: {len(trades)} source trades")
            t0 = _time.monotonic()
            compounded = compound_equity(
                trades, acct["position_pct"], acct["leverage"],
                START_EQUITY_USD, notional_cap=cap,
            )
            metrics = portfolio_metrics(compounded, START_EQUITY_USD)
            monthly = monthly_aggregate(compounded, START_EQUITY_USD)
            capped_count = int(compounded["was_capped"].sum()) if "was_capped" in compounded.columns else 0
            dt = _time.monotonic() - t0

            run_dir = results_root / f"{base_run_id}__{label}"
            run_dir.mkdir(parents=True, exist_ok=False)
            compounded.to_csv(run_dir / "trades_compounded.csv", index=False)
            monthly.to_csv(run_dir / "monthly.csv", index=False)
            with open(run_dir / "metrics.json", "w") as f:
                json.dump({**metrics, "trades_capped": capped_count},
                          f, indent=2, sort_keys=True, default=str)
            with open(run_dir / "params.json", "w") as f:
                json.dump({
                    **{k: v for k, v in acct.items() if k != "source_folder"},
                    "source_run": acct["source_folder"],
                    "start_equity_usd": START_EQUITY_USD,
                    "round_trip_fee": ROUND_TRIP_FEE,
                    "notional_cap_usd": cap,
                    "compounding": True,
                }, f, indent=2, sort_keys=True, default=str)
            with open(run_dir / "run_metadata.json", "w") as f:
                json.dump({
                    "run_id": run_dir.name, "git_sha": _git_sha(),
                    "git_dirty": _git_dirty(),
                    "timestamp_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "wall_time_seconds": float(dt),
                }, f, indent=2, sort_keys=True, default=str)

            print(f"  trades={metrics['trades']} capped={capped_count} "
                  f"final=${metrics['final_equity_usd']:,.0f} "
                  f"return={metrics['total_return_pct']:+,.0f}% "
                  f"dd={metrics['max_drawdown_pct']:.1f}%")

            summary_rows.append({
                "account": acct["label"],
                "direction": acct["direction"],
                "cap_label": cap_label,
                "notional_cap_usd": cap if cap else "uncapped",
                "position_pct": acct["position_pct"],
                "leverage": acct["leverage"],
                "sl_pct": acct["sl_pct"],
                "trades": metrics["trades"],
                "trades_capped": capped_count,
                "wr": metrics["wr"],
                "pf": metrics["pf"],
                "final_equity_usd": metrics["final_equity_usd"],
                "total_return_pct": metrics["total_return_pct"],
                "max_drawdown_pct": metrics["max_drawdown_pct"],
                "avg_notional_usd": metrics["avg_notional_usd"],
                "final_notional_usd": metrics["final_notional_usd"],
                "months": len(monthly),
                "run_dir": run_dir.name,
            })

    sha8 = _git_sha()[:8]
    ts_str = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    summary_path = results_root / f"_compounding_summary_{sha8}_{ts_str}.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\n[compound] summary: {summary_path}")
    print("\n=== COMPOUNDING RESULTS ===")
    for row in summary_rows:
        print(f"  {row['account']}: final=${row['final_equity_usd']:,.0f} "
              f"({row['total_return_pct']:+.0f}%), DD={row['max_drawdown_pct']:.1f}%, "
              f"avg notional=${row['avg_notional_usd']:,.0f}")


if __name__ == "__main__":
    main()
