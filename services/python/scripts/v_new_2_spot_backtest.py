"""v_new_2 — Spot portfolio backtest (long-only, no leverage, cap=1/2 focus).

Question: with a strict cap of 1 or 2 concurrent long-only positions and no leverage
(spot), what's the best PnL config? Does it beat BTC buy-and-hold?

Source: v_new_2_trades_scored_honest.parquet (43,191 trades 2020-2026, with funding+slippage
in pnl_pct_honest, OOS-scored bgm_honest).

Splits per CLAUDE.md (only post-2025 is honestly OOS):
  - train       2020-2024  (in-sample for BGM — INFORMATIONAL only)
  - thr_select  2025-H1    (threshold-selection slice — pick config here)
  - cal         2025-H2    (calibration slice — confirm)
  - holdout     2026-Q1    (locked OOS — final test)

Sweeps:
  cap            ∈ {1, 2, 3, 5}
  bgm_threshold  ∈ {0.50, 0.55, 0.60, 0.65, 0.70}
  sizing_mode    ∈ {full, half, fixed_1pct}
                    full       = 1.0 / cap   per slot (fully invested at cap)
                    half       = 0.5 / cap   per slot
                    fixed_1pct = 0.01         per slot (small, for risk floor)

Selection: chronological (FIFO). Skip new signals when at cap. No early-close.
Exits: as already realized in pnl_pct_honest (existing trail+timeout from rules backtest).

Compares to BTC HODL on the same window per split.

Outputs:
  services/python/results/v_new_2/spot_backtest_results.csv
  services/python/results/v_new_2/spot_backtest_verdict.md
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "services" / "python" / "data"
TRADES_PATH = DATA_DIR / "v_new_1_v2" / "v_new_2_trades_scored_honest.parquet"
BTC_PATH = DATA_DIR / "binance_archive_pilot" / "BTCUSDT_4h.parquet"
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
RESULTS.mkdir(parents=True, exist_ok=True)

CAPS = [1, 2, 3, 5]
THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
SIZING_MODES = ["full", "half", "fixed_1pct"]
SPLIT_ORDER = ["train", "thr_select", "cal", "holdout"]


@dataclass
class OpenPosition:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    pnl_pct: float
    bgm_score: float
    side: str
    size_pct: float


def _portfolio_sim(trades_df: pd.DataFrame, cap: int, size_pct: float) -> dict:
    """Spot portfolio sim: long-only assumed by caller, no leverage, FIFO with cap."""
    if len(trades_df) == 0:
        return {
            "n_total": 0, "n_taken": 0, "n_skipped": 0,
            "final_equity": 1.0, "compd_pct": 0.0,
            "max_dd_pct": 0.0, "sharpe": 0.0,
        }

    df = trades_df.sort_values("entry_time").reset_index(drop=True)
    equity = 1.0
    open_positions: list[OpenPosition] = []
    equity_curve: list[tuple[pd.Timestamp, float]] = [
        (df.iloc[0]["entry_time"] - pd.Timedelta("1h"), 1.0)
    ]
    n_taken = 0
    n_skipped = 0

    for _, row in df.iterrows():
        et, xt = row["entry_time"], row["exit_time"]
        if pd.isna(et) or pd.isna(xt):
            continue
        # Close exited positions first
        new_open: list[OpenPosition] = []
        for pos in open_positions:
            if pos.exit_time <= et:
                trade_eq_change = pos.pnl_pct * pos.size_pct  # NO leverage (spot)
                equity *= (1.0 + trade_eq_change)
                equity = max(equity, 1e-6)
                equity_curve.append((pos.exit_time, equity))
            else:
                new_open.append(pos)
        open_positions = new_open

        if len(open_positions) < cap:
            new_pos = OpenPosition(
                symbol=row["symbol"], entry_time=et, exit_time=xt,
                pnl_pct=float(row["pnl_pct_honest"]),
                bgm_score=float(row.get("bgm_honest", 0.0)),
                side=row["side"], size_pct=size_pct,
            )
            open_positions.append(new_pos)
            n_taken += 1
        else:
            n_skipped += 1

    # Close remaining at their natural exits
    for pos in open_positions:
        trade_eq_change = pos.pnl_pct * pos.size_pct
        equity *= (1.0 + trade_eq_change)
        equity = max(equity, 1e-6)
        equity_curve.append((pos.exit_time, equity))

    eq_df = pd.DataFrame(equity_curve, columns=["t", "equity"]).sort_values("t")
    eq_arr = eq_df["equity"].values
    rm = np.maximum.accumulate(eq_arr)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (eq_arr - rm) / safe
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    final_equity = float(eq_arr[-1])

    eq_df["t"] = pd.to_datetime(eq_df["t"], utc=True)
    eq_df = eq_df.set_index("t").sort_index()
    daily_eq = eq_df["equity"].resample("1D").last().ffill()
    daily_returns = daily_eq.pct_change().dropna()
    if len(daily_returns) > 5 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365))
    else:
        sharpe = 0.0

    return {
        "n_total": int(len(df)),
        "n_taken": int(n_taken),
        "n_skipped": int(n_skipped),
        "final_equity": final_equity,
        "compd_pct": (final_equity - 1.0) * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
    }


def _btc_hodl(btc_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    sub = btc_df[(btc_df["t"] >= start) & (btc_df["t"] <= end)].sort_values("t").reset_index(drop=True)
    if len(sub) < 2:
        return {"return_pct": 0.0, "max_dd_pct": 0.0}
    eq = sub["close"].values / sub["close"].iloc[0]
    rm = np.maximum.accumulate(eq)
    dd = (eq - rm) / rm
    return {
        "return_pct": float((eq[-1] - 1.0) * 100),
        "max_dd_pct": float(dd.min() * 100),
    }


def _resolve_size_pct(mode: str, cap: int) -> float:
    if mode == "full":
        return 1.0 / cap
    if mode == "half":
        return 0.5 / cap
    if mode == "fixed_1pct":
        return 0.01
    raise ValueError(f"unknown sizing mode: {mode}")


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading trades...")
    trades = pd.read_parquet(TRADES_PATH)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)

    long = trades[trades["side"] == "long"].copy()
    print(f"  total={len(trades):,}  long-only={len(long):,}")
    print(f"  long by split: {long['split'].value_counts().to_dict()}")

    print(f"[{time.strftime('%H:%M:%S')}] loading BTC 4h prices...")
    btc = pd.read_parquet(BTC_PATH)
    btc = btc.rename(columns={"open_time": "t"})
    btc["t"] = pd.to_datetime(btc["t"], utc=True)
    btc = btc[["t", "close"]].sort_values("t").drop_duplicates(subset=["t"]).reset_index(drop=True)
    print(f"  BTC bars={len(btc):,}  range={btc['t'].min()} → {btc['t'].max()}")

    # Compute split windows from data
    split_windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for split_name in SPLIT_ORDER:
        sub = long[long["split"] == split_name]
        if len(sub) == 0:
            continue
        split_windows[split_name] = (sub["entry_time"].min(), sub["exit_time"].max())
    print("  split windows:")
    for k, (s, e) in split_windows.items():
        print(f"    {k:11s} {s} → {e}  ({(e-s).days}d)")

    print(f"[{time.strftime('%H:%M:%S')}] running spot backtest sweep...")
    rows = []
    for split_name, (win_start, win_end) in split_windows.items():
        hodl = _btc_hodl(btc, win_start, win_end)
        for thr in THRESHOLDS:
            for cap in CAPS:
                for mode in SIZING_MODES:
                    size = _resolve_size_pct(mode, cap)
                    sub = long[
                        (long["split"] == split_name) &
                        (long["bgm_honest"] >= thr)
                    ]
                    res = _portfolio_sim(sub, cap=cap, size_pct=size)
                    rows.append({
                        "split": split_name,
                        "threshold": thr,
                        "cap": cap,
                        "sizing_mode": mode,
                        "size_pct": size,
                        "n_total": res["n_total"],
                        "n_taken": res["n_taken"],
                        "n_skipped": res["n_skipped"],
                        "compd_pct": res["compd_pct"],
                        "max_dd_pct": res["max_dd_pct"],
                        "sharpe": res["sharpe"],
                        "btc_hodl_pct": hodl["return_pct"],
                        "btc_dd_pct": hodl["max_dd_pct"],
                        "vs_hodl_pp": res["compd_pct"] - hodl["return_pct"],
                        "win_start": win_start,
                        "win_end": win_end,
                    })

    df_out = pd.DataFrame(rows)
    out_csv = RESULTS / "spot_backtest_results.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv.relative_to(ROOT)} ({len(df_out)} rows)")

    # Verdict
    md: list[str] = []
    md.append(f"# v_new_2 Spot Portfolio Backtest — {time.strftime('%Y-%m-%d')}\n\n")
    md.append("**Spec.** Long-only, no leverage, FIFO selection, BGM-thresholded.\n")
    md.append("Exits as realized in `pnl_pct_honest` (existing rules-backtest trail/timeout, "
              "with funding+slippage applied).\n\n")
    md.append(f"**Source.** `{TRADES_PATH.relative_to(ROOT)}` "
              f"({len(trades):,} total, {len(long):,} long).\n\n")
    md.append("**OOS caveat per CLAUDE.md.** Only `cal` + `holdout` are honestly OOS. `train` "
              "(2020-2024) is in-sample for BGM — informational only. `thr_select` (2025-H1) "
              "is the legitimate slice for picking config; `holdout` (2026-Q1) is the locked test.\n\n")

    for split_name in SPLIT_ORDER:
        sub = df_out[df_out["split"] == split_name]
        if len(sub) == 0:
            continue
        hodl_pct = sub["btc_hodl_pct"].iloc[0]
        hodl_dd = sub["btc_dd_pct"].iloc[0]
        win_start = sub["win_start"].iloc[0]
        win_end = sub["win_end"].iloc[0]
        is_oos = split_name in ("thr_select", "cal", "holdout")
        marker = "[OOS]" if is_oos else "[IN-SAMPLE]"
        md.append(f"## {split_name} {marker}\n\n")
        md.append(f"Window: {win_start.date()} → {win_end.date()} "
                  f"({(win_end - win_start).days}d). "
                  f"BTC HODL **{hodl_pct:+.1f}%** (max DD {hodl_dd:.1f}%).\n\n")
        md.append("Top 10 by compd PnL:\n\n")
        md.append("| thr | cap | sizing | size_pct | n_taken | compd | DD | sharpe | vs HODL |\n")
        md.append("|---|---|---|---|---|---|---|---|---|\n")
        for _, r in sub.sort_values("compd_pct", ascending=False).head(10).iterrows():
            md.append(
                f"| {r['threshold']:.2f} | {int(r['cap'])} | {r['sizing_mode']} "
                f"| {r['size_pct']:.3f} | {int(r['n_taken'])} "
                f"| {r['compd_pct']:+.1f}% | {r['max_dd_pct']:+.1f}% "
                f"| {r['sharpe']:.2f} | {r['vs_hodl_pp']:+.1f}pp |\n"
            )
        md.append("\nCap=1 / Cap=2 focused (the user's question — best PnL by sizing/threshold):\n\n")
        md.append("| thr | cap | sizing | n_taken | compd | DD | sharpe | vs HODL |\n")
        md.append("|---|---|---|---|---|---|---|---|\n")
        focus = sub[sub["cap"].isin([1, 2])].sort_values(
            ["cap", "compd_pct"], ascending=[True, False]
        )
        for _, r in focus.iterrows():
            md.append(
                f"| {r['threshold']:.2f} | {int(r['cap'])} | {r['sizing_mode']} "
                f"| {int(r['n_taken'])} | {r['compd_pct']:+.1f}% "
                f"| {r['max_dd_pct']:+.1f}% | {r['sharpe']:.2f} "
                f"| {r['vs_hodl_pp']:+.1f}pp |\n"
            )
        md.append("\n")

    # Cross-split robustness for cap ∈ {1,2}: which (thr, mode) is best across splits?
    md.append("## Cross-split robustness — cap=1 and cap=2 only\n\n")
    md.append("Sum of `compd_pct` across the OOS splits (`thr_select` + `cal` + `holdout`). "
              "Higher is better — but a cap/thr/sizing that wins one split and loses another is "
              "less robust than one that wins all three.\n\n")
    oos = df_out[df_out["split"].isin(["thr_select", "cal", "holdout"])]
    for cap in (1, 2):
        sub = oos[oos["cap"] == cap]
        if len(sub) == 0:
            continue
        agg = sub.groupby(["threshold", "sizing_mode"]).agg(
            sum_compd=("compd_pct", "sum"),
            min_split=("compd_pct", "min"),
            max_split=("compd_pct", "max"),
            mean_dd=("max_dd_pct", "mean"),
            sum_n_taken=("n_taken", "sum"),
        ).reset_index().sort_values("sum_compd", ascending=False)
        md.append(f"### Cap = {cap}\n\n")
        md.append("| thr | sizing | sum_compd | min_split | max_split | mean_DD | sum_n |\n")
        md.append("|---|---|---|---|---|---|---|\n")
        for _, r in agg.head(8).iterrows():
            md.append(
                f"| {r['threshold']:.2f} | {r['sizing_mode']} | {r['sum_compd']:+.1f}% "
                f"| {r['min_split']:+.1f}% | {r['max_split']:+.1f}% "
                f"| {r['mean_dd']:.1f}% | {int(r['sum_n_taken'])} |\n"
            )
        md.append("\n")

    out_md = RESULTS / "spot_backtest_verdict.md"
    out_md.write_text("".join(md))
    print(f"  saved: {out_md.relative_to(ROOT)}")
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
