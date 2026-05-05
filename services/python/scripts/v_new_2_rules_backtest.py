"""v_new_2 — Rules-only backtest (Layer 1 baseline).

Strategy:
  LONG entry:
    - d_trend_long == 1 (daily EMA20 > EMA50)
    - pullback_pct CROSSES threshold X (was below, now ≥X)
  LONG exit:
    - d_trend_long flips to 0 (trend ended), OR
    - hold timeout (e.g., 90 days)

  SHORT entry/exit: mirror.

Sweeps: pullback ∈ {5,10,15,20,25}%  ×  hold timeout ∈ {30, 60, 90} days

Multi-OOS evaluation: 2025_h2 (2025-07 → 2025-12) AND 2026_q1 (2026-01 → 2026-03)
plus full-history reference. Each variant produces:
  - n_trades, win_rate, profit_factor, sum_pnl, compounded@1×/3×/5×, max_dd

Output:
  services/python/results/v_new_2/rules_backtest.csv
  services/python/results/v_new_2/rules_verdict.md
"""
from __future__ import annotations

import os
import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = pathlib.Path(os.environ.get("V_NEW_1_DATA_DIR",
                                        str(ROOT / "services" / "python" / "data" / "v_new_1_v2")))
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
RESULTS.mkdir(parents=True, exist_ok=True)

PULLBACKS = [5.0, 10.0, 15.0, 20.0, 25.0]    # in pct (matches pullback_pct/rise_pct units)
TIMEOUTS_DAYS = [30, 60, 90]
BARS_PER_DAY = 6
FRICTION = 0.001  # 0.1% round-trip per trade

OOS_SLICES = {
    "2025_h2": ("2025-07-01", "2025-12-31"),
    "2026_q1": ("2026-01-01", "2026-03-31"),
    "full":    ("2020-05-01", "2026-04-01"),
}


@dataclass
class Trade:
    symbol: str
    side: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_price: float
    exit_price: float
    bars_held: int
    exit_reason: str
    pnl_pct: float


def _ts() -> str:
    return f"[{time.strftime('%H:%M:%S')}]"


def _backtest_one(
    df: pd.DataFrame, side: str, pullback_thr: float, timeout_bars: int,
) -> list[Trade]:
    """One symbol's series — returns list of trades for one variant."""
    trades: list[Trade] = []
    n = len(df)
    if n < 50:
        return trades

    closes = df["close"].values
    timestamps = df["timestamp"].reset_index(drop=True)

    if side == "long":
        trend_arr = df["d_trend_long"].values
        pullback_arr = df["pullback_pct"].values
        thr_col = pullback_arr
    else:
        trend_arr = df["d_trend_short"].values
        pullback_arr = df["rise_pct"].values
        thr_col = pullback_arr

    # Find triggers: bar where threshold is met AND prev bar was below threshold
    # AND we're currently in the matching trend regime
    in_position = False
    i = 0
    while i < n - 1:
        if in_position:
            i += 1
            continue
        if pd.isna(trend_arr[i]) or pd.isna(thr_col[i]):
            i += 1
            continue
        if trend_arr[i] != 1:
            i += 1
            continue
        # Threshold-crossing trigger: now ≥ thr, prev < thr (or NaN)
        if thr_col[i] < pullback_thr:
            i += 1
            continue
        prev = thr_col[i-1] if i > 0 else float("nan")
        if not pd.isna(prev) and prev >= pullback_thr:
            i += 1
            continue
        # Trigger fires — enter at close[i]
        entry_idx = i
        entry_price = closes[entry_idx]
        # Walk forward looking for exit
        exit_idx = None
        exit_reason = None
        max_idx = min(n - 1, entry_idx + timeout_bars)
        for k in range(entry_idx + 1, max_idx + 1):
            t_now = trend_arr[k]
            if pd.isna(t_now):
                continue
            if t_now != 1:
                exit_idx = k
                exit_reason = "trend_flip"
                break
        if exit_idx is None:
            exit_idx = max_idx
            exit_reason = "timeout"
        exit_price = closes[exit_idx]
        if side == "long":
            pnl = (exit_price / entry_price) - 1.0
        else:
            pnl = (entry_price / exit_price) - 1.0
        pnl -= FRICTION
        trades.append(Trade(
            symbol=df["symbol"].iloc[0],
            side=side,
            entry_time=timestamps.iloc[entry_idx],
            exit_time=timestamps.iloc[exit_idx],
            entry_price=float(entry_price),
            exit_price=float(exit_price),
            bars_held=int(exit_idx - entry_idx),
            exit_reason=exit_reason,
            pnl_pct=float(pnl),
        ))
        # After exit, jump to exit bar + 1 (cooldown=1 bar)
        i = exit_idx + 1
    return trades


def _summary(pnls: np.ndarray, leverage: float):
    if len(pnls) == 0:
        return {"n":0}
    p = pnls * leverage
    eq = np.cumprod(1.0 + np.clip(p, -0.99, 10.0))
    rm = np.maximum.accumulate(eq)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (eq - rm) / safe
    wins = p[p > 0]
    losses = p[p < 0]
    return {
        "n": int(len(p)),
        "wr": float((p > 0).mean()),
        "sum_pnl_pct": float(p.sum() * 100.0),
        "compounded_pct": float((eq[-1] - 1.0) * 100.0),
        "max_dd_pct": float(dd.min() * 100.0),
        "avg_win_pct": float(wins.mean() * 100) if len(wins) else 0.0,
        "avg_loss_pct": float(losses.mean() * 100) if len(losses) else 0.0,
        "pf": float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf"),
    }


def _filter_trades_to_window(trades: list[Trade], start: str, end: str) -> list[Trade]:
    s = pd.Timestamp(start, tz="UTC")
    e = pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)
    return [t for t in trades if s <= t.entry_time < e]


def main() -> None:
    t0 = time.time()
    print(f"{_ts()} v_new_2 rules-only backtest — start")
    df = pd.read_parquet(DATA_DIR / "features_v_new_2.parquet",
                          columns=["symbol","timestamp","close",
                                   "d_trend_long","d_trend_short",
                                   "pullback_pct","rise_pct"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values(["symbol","timestamp"]).reset_index(drop=True)
    print(f"  loaded {len(df):,} rows  ({df['symbol'].nunique()} symbols)")

    # Per-symbol grouping for backtest
    syms = df["symbol"].unique()
    rows = []
    for side in ["long", "short"]:
        for pb in PULLBACKS:
            for tdays in TIMEOUTS_DAYS:
                t_var = time.time()
                all_trades: list[Trade] = []
                for sym in syms:
                    g = df[df["symbol"] == sym]
                    trades = _backtest_one(g, side, pb, tdays * BARS_PER_DAY)
                    all_trades.extend(trades)
                # Per OOS slice + full
                slice_summaries = {}
                for slice_name, (s, e) in OOS_SLICES.items():
                    tr = _filter_trades_to_window(all_trades, s, e)
                    pnls = np.array([t.pnl_pct for t in tr])
                    slice_summaries[slice_name] = _summary(pnls, 1.0)
                    if slice_name == "full":
                        slice_summaries["full_3x"] = _summary(pnls, 3.0)
                        slice_summaries["full_5x"] = _summary(pnls, 5.0)

                row = {
                    "side": side, "pullback_pct": pb, "timeout_days": tdays,
                    "n_total": slice_summaries["full"].get("n", 0),
                    "full_wr": slice_summaries["full"].get("wr", 0)*100,
                    "full_pf": slice_summaries["full"].get("pf", 0),
                    "full_avg_win_pct": slice_summaries["full"].get("avg_win_pct", 0),
                    "full_avg_loss_pct": slice_summaries["full"].get("avg_loss_pct", 0),
                    "full_1x_compd_pct": slice_summaries["full"].get("compounded_pct", 0),
                    "full_3x_compd_pct": slice_summaries["full_3x"].get("compounded_pct", 0),
                    "full_5x_compd_pct": slice_summaries["full_5x"].get("compounded_pct", 0),
                    "full_3x_dd_pct": slice_summaries["full_3x"].get("max_dd_pct", 0),
                    "full_5x_dd_pct": slice_summaries["full_5x"].get("max_dd_pct", 0),
                    "h2_n": slice_summaries["2025_h2"].get("n", 0),
                    "h2_wr": slice_summaries["2025_h2"].get("wr", 0)*100,
                    "h2_pf": slice_summaries["2025_h2"].get("pf", 0),
                    "h2_1x_compd_pct": slice_summaries["2025_h2"].get("compounded_pct", 0),
                    "q1_n": slice_summaries["2026_q1"].get("n", 0),
                    "q1_wr": slice_summaries["2026_q1"].get("wr", 0)*100,
                    "q1_pf": slice_summaries["2026_q1"].get("pf", 0),
                    "q1_1x_compd_pct": slice_summaries["2026_q1"].get("compounded_pct", 0),
                }
                rows.append(row)
                print(f"  {side} pb={pb:5.1f}% timeout={tdays}d  "
                      f"n={row['n_total']:>5}  full_PF={row['full_pf']:5.2f}  "
                      f"full_1x={row['full_1x_compd_pct']:+8.0f}%  "
                      f"H2_PF={row['h2_pf']:4.2f}({row['h2_n']:>4})  "
                      f"Q1_PF={row['q1_pf']:4.2f}({row['q1_n']:>3})  "
                      f"3x_DD={row['full_3x_dd_pct']:+5.0f}%  "
                      f"({time.time()-t_var:.1f}s)")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(RESULTS / "rules_backtest.csv", index=False)
    print(f"\n  saved: {RESULTS / 'rules_backtest.csv'}")

    # Verdict
    md = []
    md.append(f"# v_new_2 — Rules-Only Backtest Verdict ({time.strftime('%Y-%m-%d')})\n\n")
    md.append("Per-coin daily EMA20/50 trend + pullback entry. Exit on trend flip OR timeout.\n\n")
    for side in ["long","short"]:
        md.append(f"## {side.upper()} — sorted by full-history PF\n\n")
        md.append("| pullback | timeout | n | WR | PF | avg_win | avg_loss | 1× | 3× | 5× | 3× DD |\n")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|\n")
        sub = df_out[df_out["side"]==side].sort_values("full_pf", ascending=False)
        for _, r in sub.iterrows():
            md.append(f"| {r['pullback_pct']:.0f}% | {int(r['timeout_days'])}d | "
                      f"{int(r['n_total'])} | {r['full_wr']:.1f}% | {r['full_pf']:.2f} | "
                      f"{r['full_avg_win_pct']:+.1f}% | {r['full_avg_loss_pct']:+.1f}% | "
                      f"{r['full_1x_compd_pct']:+.0f}% | {r['full_3x_compd_pct']:+.0f}% | "
                      f"{r['full_5x_compd_pct']:+.0f}% | {r['full_3x_dd_pct']:+.0f}% |\n")
        md.append("\n")
        md.append(f"### {side.upper()} — OOS comparison (top 5 by full PF)\n\n")
        md.append("| pullback | timeout | 2025 H2 (n / WR / PF / 1× compd) | 2026 Q1 |\n")
        md.append("|---|---|---|---|\n")
        for _, r in sub.head(5).iterrows():
            md.append(f"| {r['pullback_pct']:.0f}% | {int(r['timeout_days'])}d | "
                      f"{int(r['h2_n'])}/{r['h2_wr']:.0f}%/{r['h2_pf']:.2f}/{r['h2_1x_compd_pct']:+.0f}% | "
                      f"{int(r['q1_n'])}/{r['q1_wr']:.0f}%/{r['q1_pf']:.2f}/{r['q1_1x_compd_pct']:+.0f}% |\n")
        md.append("\n")
    md.append("## Caveats\n")
    md.append("- Full-history is ~6 years; CAGR ~ (1+full_compounded/100)^(1/6)-1 (annualised).\n")
    md.append("- 5× DD = -100% means equity wiped at some point in compounding sequence.\n")
    md.append("- Real deployment requires concurrent-position cap and per-trade risk sizing.\n")
    md.append("- These are TRADE-LEVEL stats (sequential). Multiple triggers in same trend would compound — assumption is each trade is independent.\n")

    (RESULTS / "rules_verdict.md").write_text("".join(md))
    print(f"  saved: {RESULTS / 'rules_verdict.md'}")
    print(f"\n{_ts()} DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
