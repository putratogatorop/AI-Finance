"""v_new_2 — Spot tiered profit-take backtest (long-only, post-recoup slot freeing).

Question: with cap=1 or 2 and a moon-bag profit ladder (sell N% at +X%, sell again at +Y%, let
the rest ride), what's the best PnL config?

Differences vs v_new_2_spot_backtest.py (flat):
  - Per-bar simulation using features parquet (symbol, timestamp, close, atr14_pct)
  - Tracks running peak gain per trade; fires tier sales when peak crosses thresholds
  - Slot frees after the FIRST tier-sale (post-recoup) — moon bag stays in inventory but does
    not block new entries
  - Final exit: trail catches OR hard-timeout at end of horizon (use the original trade's
    exit_time as a backstop ceiling — we never hold beyond what the rules-backtest held)

Sizing sweep:
  initial_size_pct ∈ {0.25, 0.33, 0.50, 0.75, 1.0/cap}

Tier ladder sweep:
  AGG     +50/+200/+500   (sell 50% at each)
  STD     +100/+400/+900  (sell 50% at each — classic moon-bag)
  CONS    +100/+300/INF   (2-tier with infinite final ride)
  HEAVY   +200/+500/+1000 (slow taker, big moon bag)
  TWO     +100/+500       (2-tier only, sell 50% / 50%, then full)

Cap ∈ {1, 2}. Threshold locked at 0.65 (winner from flat backtest).

Output:
  services/python/results/v_new_2/spot_tiered_backtest_results.csv
  services/python/results/v_new_2/spot_tiered_backtest_verdict.md
"""
from __future__ import annotations

import pathlib
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "services" / "python" / "data" / "v_new_1_v2"
FEATURES_PATH = DATA_DIR / "features_v_new_2_v2.parquet"
TRADES_PATH = DATA_DIR / "v_new_2_trades_scored_honest.parquet"
BTC_PATH = ROOT / "services" / "python" / "data" / "binance_archive_pilot" / "BTCUSDT_4h.parquet"
RESULTS = ROOT / "services" / "python" / "results" / "v_new_2"
RESULTS.mkdir(parents=True, exist_ok=True)

BGM_THRESHOLD = 0.65
CAPS = [1, 2]
INITIAL_SIZES = [0.25, 0.33, 0.50, 0.75]   # spot: max 1.0/cap to avoid over-allocation
SPLITS = ["thr_select", "cal", "holdout"]   # OOS only

# Tier ladders. Each tier = (peak_threshold_pct, fraction_of_remaining_to_sell)
# Threshold is gain percentage; e.g. 100 = +100% (2x). Fraction is how much of currently-held
# position to sell at that trigger. INF = never trigger (kept-forever moon bag).
LADDERS: dict[str, list[tuple[float, float]]] = {
    "AGG":   [(50.0, 0.50), (200.0, 0.50), (500.0, 0.50)],
    "STD":   [(100.0, 0.50), (400.0, 0.50), (900.0, 0.50)],
    "CONS":  [(100.0, 0.50), (300.0, 0.50)],   # final tier missing → ride forever
    "HEAVY": [(200.0, 0.50), (500.0, 0.50), (1000.0, 0.50)],
    "TWO":   [(100.0, 0.50), (500.0, 0.50)],
    "FLAT":  [],   # baseline: no tier-sales (matches v_new_2_spot_backtest.py)
}

# Final exit on the moon bag (after all tiers fired). If the price falls more than this %
# from running peak, sell the moon bag. None = ride to original trade's exit_time.
MOON_BAG_TRAIL_PCT = 50.0   # 50% drop from peak triggers moon-bag exit


@dataclass
class TradeOutcome:
    symbol: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp        # actual exit (may differ from input exit if moon-bag trail catches)
    realized_pnl_on_size: float    # final realized return on the initial sized capital (e.g. 1.5 = +150%)
    tier_fires: list[float]        # thresholds that fired during this trade (e.g. [100, 400])
    moon_bag_remaining: float      # fraction of original position still held when trade closed
    slot_free_time: pd.Timestamp   # when the slot becomes free (entry_time if no tier fired,
                                   # else first-tier-fire time)


def _simulate_trade(close_series: pd.Series, entry_time: pd.Timestamp, exit_time: pd.Timestamp,
                    ladder: list[tuple[float, float]],
                    moon_bag_trail_pct: float | None) -> TradeOutcome:
    """Simulate one trade bar-by-bar with tiered profit-taking.

    close_series indexed by timestamp, ascending. Slice from entry_time inclusive to
    exit_time inclusive.

    Returns TradeOutcome with the realized return per unit of sized capital.
    """
    # Slice between entry and exit (inclusive). The features parquet's first timestamp at
    # entry_time IS the entry bar (we buy at its close).
    hold = close_series.loc[entry_time:exit_time]
    if len(hold) < 2:
        # Degenerate — fall back to "no movement"
        return TradeOutcome(
            symbol="", entry_time=entry_time, exit_time=exit_time,
            realized_pnl_on_size=0.0, tier_fires=[], moon_bag_remaining=1.0,
            slot_free_time=entry_time,
        )

    entry_price = float(hold.iloc[0])
    if entry_price <= 0 or np.isnan(entry_price):
        return TradeOutcome(
            symbol="", entry_time=entry_time, exit_time=exit_time,
            realized_pnl_on_size=0.0, tier_fires=[], moon_bag_remaining=1.0,
            slot_free_time=entry_time,
        )

    realized = 0.0           # cash collected from tier-sales (as fraction of initial position)
    held_fraction = 1.0      # how much of the initial position is still held
    tier_idx = 0             # next ladder tier to check
    tier_fires: list[float] = []
    slot_free_time: pd.Timestamp | None = None
    running_peak_gain_pct = 0.0
    running_peak_price = entry_price

    final_price = entry_price
    final_ts = entry_time
    moon_bag_exited_early = False

    for ts, price in hold.items():
        if np.isnan(price):
            continue
        gain_pct = (price / entry_price - 1.0) * 100.0
        if price > running_peak_price:
            running_peak_price = float(price)
            running_peak_gain_pct = gain_pct

        # Fire any tiers we crossed (in order)
        while tier_idx < len(ladder) and running_peak_gain_pct >= ladder[tier_idx][0]:
            trigger_pct, sell_frac = ladder[tier_idx]
            sell_amount = held_fraction * sell_frac
            # Sell at trigger_pct gain price (approximation — we crossed the threshold this bar)
            sell_price_factor = 1.0 + trigger_pct / 100.0
            realized += sell_amount * sell_price_factor   # cash equivalent in entry-units
            held_fraction -= sell_amount
            tier_fires.append(trigger_pct)
            if slot_free_time is None:
                slot_free_time = ts   # first tier fired → slot frees here
            tier_idx += 1

        # Moon-bag trail check (only if all ladders fired and there's something left)
        if tier_idx >= len(ladder) and held_fraction > 1e-9 and moon_bag_trail_pct is not None:
            drop_from_peak = (price - running_peak_price) / running_peak_price * 100.0
            if drop_from_peak <= -moon_bag_trail_pct:
                # Sell moon bag at current price
                realized += held_fraction * (price / entry_price)
                final_price = float(price)
                final_ts = ts
                held_fraction = 0.0
                moon_bag_exited_early = True
                break

        final_price = float(price)
        final_ts = ts

    # If we still hold something at exit_time, sell at final price
    if held_fraction > 1e-9:
        realized += held_fraction * (final_price / entry_price)
        held_fraction = 0.0

    if slot_free_time is None:
        slot_free_time = final_ts   # slot was held to natural exit

    realized_pnl_on_size = realized - 1.0   # -1 because the initial position cost 1.0 units
    return TradeOutcome(
        symbol="", entry_time=entry_time,
        exit_time=final_ts if moon_bag_exited_early else exit_time,
        realized_pnl_on_size=float(realized_pnl_on_size),
        tier_fires=tier_fires,
        moon_bag_remaining=0.0,
        slot_free_time=slot_free_time,
    )


@dataclass
class OpenPosition:
    symbol: str
    entry_time: pd.Timestamp
    slot_free_time: pd.Timestamp    # when the slot becomes available again (post-recoup)
    realized_pnl_on_size: float
    size_pct: float                  # fraction of equity allocated at entry
    closed_eq_change: float | None = None   # final equity multiplier change (computed at close)
    moon_close_time: pd.Timestamp | None = None
    pending_close_time: pd.Timestamp | None = None  # original exit_time, when remaining moon bag finishes


def _portfolio_sim(trades_df: pd.DataFrame, prices: dict[str, pd.Series],
                   cap: int, initial_size_pct: float,
                   ladder: list[tuple[float, float]],
                   moon_bag_trail_pct: float | None) -> dict:
    """Run the spot tiered portfolio sim.

    Selection: chronological. When new signal fires, free any slots whose `slot_free_time`
    has passed. If under cap, take it. Otherwise skip (no preemption).

    PnL accounting: we assume the realized_pnl_on_size from _simulate_trade is realized at
    the trade's `exit_time` (or moon-bag exit if earlier). So equity compounds at exit
    chronology, but the slot frees at `slot_free_time` (post-recoup logic).
    """
    if len(trades_df) == 0:
        return _empty_result()

    df = trades_df.sort_values("entry_time").reset_index(drop=True)
    equity = 1.0
    open_positions: list[OpenPosition] = []
    closed_events: list[tuple[pd.Timestamp, float]] = []   # (exit_time, eq_change_amount)
    n_taken = 0
    n_skipped = 0
    tier_fires_global: list[int] = [0, 0, 0]   # count per tier index
    pnl_per_trade: list[float] = []

    for _, row in df.iterrows():
        et = row["entry_time"]
        # Apply any closes whose exit_time <= et
        future_open: list[OpenPosition] = []
        for pos in open_positions:
            if pos.pending_close_time is not None and pos.pending_close_time <= et:
                # The close already happened — its equity change must be applied now
                if pos.closed_eq_change is not None:
                    equity *= (1.0 + pos.closed_eq_change)
                    closed_events.append((pos.pending_close_time, equity))
            else:
                future_open.append(pos)
        open_positions = future_open

        # Free slots whose slot_free_time is past
        active_slots = sum(1 for p in open_positions if p.slot_free_time > et)
        if active_slots >= cap:
            n_skipped += 1
            continue

        # Take the trade
        sym = row["symbol"]
        if sym not in prices:
            n_skipped += 1
            continue
        outcome = _simulate_trade(prices[sym], et, row["exit_time"], ladder, moon_bag_trail_pct)
        # Equity change = realized_pnl_on_size × initial_size_pct  (no leverage)
        eq_change = outcome.realized_pnl_on_size * initial_size_pct
        for tp in outcome.tier_fires:
            for idx, (thr, _) in enumerate(ladder):
                if abs(tp - thr) < 1e-6 and idx < len(tier_fires_global):
                    tier_fires_global[idx] += 1

        # Add to open positions; the close will be applied at outcome.exit_time
        pos = OpenPosition(
            symbol=sym, entry_time=et,
            slot_free_time=outcome.slot_free_time,
            realized_pnl_on_size=outcome.realized_pnl_on_size,
            size_pct=initial_size_pct,
            closed_eq_change=eq_change,
            pending_close_time=outcome.exit_time,
        )
        open_positions.append(pos)
        n_taken += 1
        pnl_per_trade.append(outcome.realized_pnl_on_size)

    # Apply remaining closes
    for pos in sorted(open_positions, key=lambda p: p.pending_close_time or p.slot_free_time):
        if pos.closed_eq_change is not None:
            equity *= (1.0 + pos.closed_eq_change)
            closed_events.append((pos.pending_close_time, equity))

    if not closed_events:
        return _empty_result(n_total=len(df))

    eq_df = pd.DataFrame(closed_events, columns=["t", "equity"]).sort_values("t").reset_index(drop=True)
    eq_arr = eq_df["equity"].values
    rm = np.maximum.accumulate(eq_arr)
    safe = np.where(rm > 1e-10, rm, 1e-10)
    dd = (eq_arr - rm) / safe
    max_dd = float(dd.min())
    final_equity = float(eq_arr[-1])

    eq_df["t"] = pd.to_datetime(eq_df["t"], utc=True)
    eq_df = eq_df.set_index("t").sort_index()
    daily_eq = eq_df["equity"].resample("1D").last().ffill()
    daily_returns = daily_eq.pct_change().dropna()
    if len(daily_returns) > 5 and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(365))
    else:
        sharpe = 0.0

    win_rate = float(sum(1 for p in pnl_per_trade if p > 0) / max(len(pnl_per_trade), 1))

    return {
        "n_total": int(len(df)),
        "n_taken": int(n_taken),
        "n_skipped": int(n_skipped),
        "final_equity": final_equity,
        "compd_pct": (final_equity - 1.0) * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
        "win_rate": win_rate,
        "tier1_fires": tier_fires_global[0] if len(tier_fires_global) > 0 else 0,
        "tier2_fires": tier_fires_global[1] if len(tier_fires_global) > 1 else 0,
        "tier3_fires": tier_fires_global[2] if len(tier_fires_global) > 2 else 0,
        "median_pnl_per_trade": float(np.median(pnl_per_trade)) if pnl_per_trade else 0.0,
        "max_pnl_per_trade": float(np.max(pnl_per_trade)) if pnl_per_trade else 0.0,
    }


def _empty_result(n_total: int = 0) -> dict:
    return {
        "n_total": n_total, "n_taken": 0, "n_skipped": 0,
        "final_equity": 1.0, "compd_pct": 0.0, "max_dd_pct": 0.0,
        "sharpe": 0.0, "win_rate": 0.0,
        "tier1_fires": 0, "tier2_fires": 0, "tier3_fires": 0,
        "median_pnl_per_trade": 0.0, "max_pnl_per_trade": 0.0,
    }


def _btc_hodl(btc_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    sub = btc_df[(btc_df["t"] >= start) & (btc_df["t"] <= end)].sort_values("t").reset_index(drop=True)
    if len(sub) < 2:
        return {"return_pct": 0.0, "max_dd_pct": 0.0}
    eq = sub["close"].values / sub["close"].iloc[0]
    rm = np.maximum.accumulate(eq)
    dd = (eq - rm) / rm
    return {"return_pct": float((eq[-1] - 1.0) * 100), "max_dd_pct": float(dd.min() * 100)}


def main():
    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading trades + features...")
    trades = pd.read_parquet(TRADES_PATH)
    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True)
    long = trades[
        (trades["side"] == "long") &
        (trades["split"].isin(SPLITS)) &
        (trades["bgm_honest"] >= BGM_THRESHOLD)
    ].copy()
    print(f"  long+OOS+bgm>={BGM_THRESHOLD}: {len(long):,} trades")

    feat = pd.read_parquet(FEATURES_PATH, columns=["symbol", "timestamp", "close"])
    feat["timestamp"] = pd.to_datetime(feat["timestamp"], utc=True)
    needed_syms = set(long["symbol"].unique())
    feat = feat[feat["symbol"].isin(needed_syms)].copy()
    print(f"  features rows for needed symbols: {len(feat):,}")

    # Build per-symbol close series for fast slicing
    print(f"[{time.strftime('%H:%M:%S')}] building per-symbol close series...")
    prices: dict[str, pd.Series] = {}
    for sym, grp in feat.groupby("symbol"):
        ser = grp.set_index("timestamp")["close"].sort_index()
        # de-dup index in case
        ser = ser[~ser.index.duplicated(keep="first")]
        prices[sym] = ser
    print(f"  {len(prices)} symbol series ready")

    btc = pd.read_parquet(BTC_PATH).rename(columns={"open_time": "t"})
    btc["t"] = pd.to_datetime(btc["t"], utc=True)
    btc = btc[["t", "close"]].sort_values("t").reset_index(drop=True)

    # Split windows from data
    split_windows = {}
    for sn in SPLITS:
        sub = long[long["split"] == sn]
        if len(sub) > 0:
            split_windows[sn] = (sub["entry_time"].min(), sub["exit_time"].max())

    print(f"[{time.strftime('%H:%M:%S')}] running tiered sweep...")
    rows = []
    for split_name, (win_start, win_end) in split_windows.items():
        hodl = _btc_hodl(btc, win_start, win_end)
        sub = long[long["split"] == split_name]
        for cap in CAPS:
            for size in INITIAL_SIZES:
                # spot constraint: size × cap ≤ 1.0 (no over-allocation)
                if size * cap > 1.0 + 1e-9:
                    continue
                for ladder_name, ladder in LADDERS.items():
                    res = _portfolio_sim(
                        sub, prices, cap=cap, initial_size_pct=size,
                        ladder=ladder, moon_bag_trail_pct=MOON_BAG_TRAIL_PCT,
                    )
                    rows.append({
                        "split": split_name,
                        "cap": cap,
                        "size_pct": size,
                        "ladder": ladder_name,
                        "n_total": res["n_total"],
                        "n_taken": res["n_taken"],
                        "n_skipped": res["n_skipped"],
                        "compd_pct": res["compd_pct"],
                        "max_dd_pct": res["max_dd_pct"],
                        "sharpe": res["sharpe"],
                        "win_rate": res["win_rate"],
                        "tier1": res["tier1_fires"],
                        "tier2": res["tier2_fires"],
                        "tier3": res["tier3_fires"],
                        "median_pnl": res["median_pnl_per_trade"],
                        "max_pnl": res["max_pnl_per_trade"],
                        "btc_hodl_pct": hodl["return_pct"],
                        "btc_dd_pct": hodl["max_dd_pct"],
                        "vs_hodl_pp": res["compd_pct"] - hodl["return_pct"],
                    })

    df_out = pd.DataFrame(rows)
    out_csv = RESULTS / "spot_tiered_backtest_results.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"  saved: {out_csv.relative_to(ROOT)} ({len(df_out)} rows)")

    # Verdict
    md: list[str] = []
    md.append(f"# v_new_2 Spot Tiered-Profit-Take Backtest — {time.strftime('%Y-%m-%d')}\n\n")
    md.append("**Spec.** Long-only, no leverage, post-recoup slot freeing, BGM-thresholded "
              f"@ {BGM_THRESHOLD}, FIFO selection.\n")
    md.append("Per-bar simulation using `features_v_new_2_v2.parquet` (close prices). "
              f"Moon bag trail: {MOON_BAG_TRAIL_PCT}% drop from peak triggers exit.\n\n")

    for split_name in SPLITS:
        sub = df_out[df_out["split"] == split_name]
        if len(sub) == 0: continue
        hodl_pct = sub["btc_hodl_pct"].iloc[0]
        hodl_dd = sub["btc_dd_pct"].iloc[0]
        md.append(f"## {split_name} [OOS]\n\n")
        md.append(f"BTC HODL **{hodl_pct:+.1f}%** (DD {hodl_dd:.1f}%).\n\n")
        md.append("Top 12 by compd PnL (all caps/sizes/ladders):\n\n")
        md.append("| cap | size | ladder | n_taken | compd | DD | sharpe | WR | tier1/2/3 | vs HODL |\n")
        md.append("|---|---|---|---|---|---|---|---|---|---|\n")
        for _, r in sub.sort_values("compd_pct", ascending=False).head(12).iterrows():
            md.append(
                f"| {int(r['cap'])} | {r['size_pct']:.2f} | {r['ladder']} "
                f"| {int(r['n_taken'])} | {r['compd_pct']:+.1f}% | {r['max_dd_pct']:+.1f}% "
                f"| {r['sharpe']:.2f} | {r['win_rate']*100:.0f}% "
                f"| {int(r['tier1'])}/{int(r['tier2'])}/{int(r['tier3'])} "
                f"| {r['vs_hodl_pp']:+.1f}pp |\n"
            )
        md.append("\n")

    # Cross-OOS robustness
    md.append("## Cross-OOS robustness (sum across thr_select + cal + holdout)\n\n")
    agg = df_out.groupby(["cap", "size_pct", "ladder"]).agg(
        sum_compd=("compd_pct", "sum"),
        min_split=("compd_pct", "min"),
        max_split=("compd_pct", "max"),
        mean_dd=("max_dd_pct", "mean"),
        mean_sharpe=("sharpe", "mean"),
        sum_taken=("n_taken", "sum"),
        sum_tier1=("tier1", "sum"),
        sum_tier3=("tier3", "sum"),
    ).reset_index().sort_values("sum_compd", ascending=False)

    md.append("Top 15 configs by sum across the 3 OOS splits:\n\n")
    md.append("| cap | size | ladder | sum_compd | min_split | max_split | mean_DD | sharpe | n_taken | tier1/tier3 |\n")
    md.append("|---|---|---|---|---|---|---|---|---|---|\n")
    for _, r in agg.head(15).iterrows():
        md.append(
            f"| {int(r['cap'])} | {r['size_pct']:.2f} | {r['ladder']} "
            f"| {r['sum_compd']:+.1f}% | {r['min_split']:+.1f}% | {r['max_split']:+.1f}% "
            f"| {r['mean_dd']:.1f}% | {r['mean_sharpe']:.2f} | {int(r['sum_taken'])} "
            f"| {int(r['sum_tier1'])}/{int(r['sum_tier3'])} |\n"
        )
    md.append("\n")

    # Best per cap (so user can compare cap=1 vs cap=2 directly)
    md.append("## Best config per cap\n\n")
    for cap in CAPS:
        cap_agg = agg[agg["cap"] == cap].head(5)
        md.append(f"### Cap = {cap} — top 5\n\n")
        md.append("| size | ladder | sum_compd | min_split | max_split | mean_DD |\n")
        md.append("|---|---|---|---|---|---|\n")
        for _, r in cap_agg.iterrows():
            md.append(
                f"| {r['size_pct']:.2f} | {r['ladder']} | {r['sum_compd']:+.1f}% "
                f"| {r['min_split']:+.1f}% | {r['max_split']:+.1f}% | {r['mean_dd']:.1f}% |\n"
            )
        md.append("\n")

    out_md = RESULTS / "spot_tiered_backtest_verdict.md"
    out_md.write_text("".join(md))
    print(f"  saved: {out_md.relative_to(ROOT)}")
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
