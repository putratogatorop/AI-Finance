# scripts/backtest_short_regime.py
"""Short backtest with higher-timeframe regime filter.

Replaces the weak BTC 15m 30-EMA filter with:
1. BTC daily trend: BTC daily close < 20-day EMA (resampled from 15m)
2. Altcoin breadth: < 60% of coins above their own 20-day EMA (daily)

Runs two variants:
  V1: Regime + 5%/15% SL/TP, 672-bar timeout
  V2: Regime + 5%/5%  SL/TP, 192-bar timeout
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from scripts.backtest_momentum_scanner import load_coin  # noqa: E402

# ── Database ─────────────────────────────────────────────────────────
DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"

# ── Paths ────────────────────────────────────────────────────────────
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

# ── Strategy params ──────────────────────────────────────────────────
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
COOLDOWN_BARS = 96

# ── Variant configs ──────────────────────────────────────────────────
VARIANTS = {
    "v1_regime_5_15": {"stop_pct": 0.05, "tp_pct": 0.15, "timeout": 672},
    "v2_regime_5_5":  {"stop_pct": 0.05, "tp_pct": 0.05, "timeout": 192},
}

# ── ML Feature columns ──────────────────────────────────────────────
FEATURE_COLS = [
    "vol_ratio", "price_move", "price_change_1bar", "price_change_4bar",
    "bar_range_norm", "upper_wick_pct", "lower_wick_pct", "body_pct",
    "vol_trend", "price_trend_4bar", "price_trend_24bar",
    "volatility_20", "rsi_14",
    "btc_ret_4bar", "btc_ret_24bar",
    "hour_of_day", "day_of_week",
    "bars_since_last_spike", "atr_14",
]

engine = create_engine(DB_URL)


# ── Regime computation ───────────────────────────────────────────────

def compute_daily_regime(all_coin_data: dict, btc_df: pd.DataFrame) -> dict:
    """Precompute daily regime signals.

    Returns dict: {date -> {"btc_bearish": bool, "breadth": float,
                             "breadth_low": bool, "allowed": bool}}
    """
    # BTC daily close and 20-day EMA
    btc_daily = btc_df.set_index("open_time").resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    btc_daily["ema20"] = btc_daily["close"].ewm(span=20, adjust=False).mean()
    btc_daily["bearish"] = btc_daily["close"] < btc_daily["ema20"]

    # Altcoin breadth: for each day, what % of coins have close > 20-day EMA
    coin_above_ema: dict = {}  # date -> [count_above, count_total]

    for symbol, df in all_coin_data.items():
        if symbol == "BTCUSDT":
            continue
        coin_daily = df.set_index("open_time").resample("1D").agg({"close": "last"}).dropna()
        coin_daily["ema20"] = coin_daily["close"].ewm(span=20, adjust=False).mean()
        coin_daily["above"] = coin_daily["close"] > coin_daily["ema20"]

        for date, row in coin_daily.iterrows():
            d = date.date()
            if d not in coin_above_ema:
                coin_above_ema[d] = [0, 0]
            coin_above_ema[d][1] += 1
            if row["above"]:
                coin_above_ema[d][0] += 1

    # Build regime dict
    regime = {}
    for date in sorted(coin_above_ema.keys()):
        above, total = coin_above_ema[date]
        breadth = above / total if total > 0 else 0.5

        btc_row = btc_daily[btc_daily.index.date == date]
        btc_bear = bool(btc_row["bearish"].iloc[0]) if len(btc_row) > 0 else False

        allowed = btc_bear and breadth < 0.60
        regime[date] = {
            "btc_bearish": btc_bear,
            "breadth": breadth,
            "breadth_low": breadth < 0.60,
            "allowed": allowed,
        }

    return regime


# ── Trade simulation ─────────────────────────────────────────────────

def simulate_trade(close, high, low, entry_bar, stop_pct, tp_pct, max_bars):
    """Simulate a short trade from entry_bar."""
    if entry_bar + 1 >= len(close):
        return None

    entry_price = close[entry_bar]
    if entry_price <= 0:
        return None

    sl_price = entry_price * (1 + stop_pct)
    tp_price = entry_price * (1 - tp_pct)

    last_bar = min(entry_bar + 1 + max_bars, len(close))

    for bar in range(entry_bar + 1, last_bar):
        if high[bar] >= sl_price:
            return {
                "pnl_pct": -stop_pct,
                "exit_reason": "stop_loss",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }
        if low[bar] <= tp_price:
            return {
                "pnl_pct": tp_pct,
                "exit_reason": "take_profit",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": tp_price,
            }

    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (entry_price - exit_price) / entry_price
    return {
        "pnl_pct": pnl_pct,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


# ── Coin scanner ─────────────────────────────────────────────────────

def scan_coin(symbol, df, btc_times, btc_close, regime):
    """Scan all bars of one coin for SHORT entries with regime filter.

    Returns list of signal dicts (no trade sim yet — just features + bar index).
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_arr = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["open_time"].values

    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    signals = []
    cooldown_until = 0

    for i in range(VOL_MA_PERIOD + PRICE_LOOKBACK, len(close)):
        if i < cooldown_until:
            continue

        # Gate 1: Volume spike
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
            continue

        # Gate 2: Price already DOWN from rolling high
        rolling_high = np.max(high[i - PRICE_LOOKBACK: i])
        if rolling_high <= 0:
            continue
        price_drop = (rolling_high - close[i]) / rolling_high
        if price_drop < PRICE_MOVE_THRESH:
            continue

        # Gate 3: Daily regime filter
        bar_date = pd.Timestamp(times[i]).date()
        if bar_date not in regime or not regime[bar_date]["allowed"]:
            continue

        # ── Extract ML features at signal bar i ──────────────────
        vol_ratio = volume[i] / vol_ma[i]
        price_move = price_drop

        price_change_1bar = (
            (close[i] - close[i - 1]) / close[i - 1] if close[i - 1] > 0 else 0
        )
        price_change_4bar = (
            (close[i] - close[i - 4]) / close[i - 4]
            if i >= 4 and close[i - 4] > 0 else 0
        )

        bar_range = high[i] - low[i]
        bar_range_norm = bar_range / close[i] if close[i] > 0 else 0
        if bar_range > 0:
            upper_wick_pct = (high[i] - max(open_arr[i], close[i])) / bar_range
            lower_wick_pct = (min(open_arr[i], close[i]) - low[i]) / bar_range
            body_pct = abs(close[i] - open_arr[i]) / bar_range
        else:
            upper_wick_pct = lower_wick_pct = body_pct = 0

        vol_trend = (
            vol_ma[i] / vol_ma[i - 4] if i >= 4 and vol_ma[i - 4] > 0 else 1.0
        )

        price_trend_4bar = (
            (close[i] - close[i - 4]) / close[i - 4]
            if i >= 4 and close[i - 4] > 0 else 0
        )
        price_trend_24bar = (
            (close[i] - close[i - 24]) / close[i - 24]
            if i >= 24 and close[i - 24] > 0 else 0
        )

        if i >= 20:
            rets = np.diff(close[i - 20: i + 1]) / close[i - 20: i]
            volatility_20 = float(np.std(rets))
        else:
            volatility_20 = 0

        if i >= 14:
            changes = np.diff(close[i - 14: i + 1])
            gains = np.mean(changes[changes > 0]) if np.any(changes > 0) else 0
            losses_abs = (
                np.mean(np.abs(changes[changes < 0])) if np.any(changes < 0) else 0
            )
            rsi_14 = (
                100 - (100 / (1 + gains / losses_abs)) if losses_abs > 0 else 100
            )
        else:
            rsi_14 = 50

        btc_idx = np.searchsorted(btc_times, times[i])
        btc_idx = min(btc_idx, len(btc_close) - 1)
        btc_ret_4bar = (
            (btc_close[btc_idx] - btc_close[max(0, btc_idx - 4)])
            / btc_close[max(0, btc_idx - 4)]
            if btc_close[max(0, btc_idx - 4)] > 0 else 0
        )
        btc_ret_24bar = (
            (btc_close[btc_idx] - btc_close[max(0, btc_idx - 24)])
            / btc_close[max(0, btc_idx - 24)]
            if btc_close[max(0, btc_idx - 24)] > 0 else 0
        )

        ts = pd.Timestamp(times[i])
        hour_of_day = ts.hour
        day_of_week = ts.dayofweek

        bars_since_last_spike = PRICE_LOOKBACK
        for j in range(i - 1, max(i - PRICE_LOOKBACK, 0), -1):
            if vol_ma[j] > 0 and volume[j] / vol_ma[j] >= 2.0:
                bars_since_last_spike = i - j
                break

        if i >= 14:
            tr_arr = np.maximum(
                high[i - 14: i] - low[i - 14: i],
                np.abs(high[i - 14: i] - close[i - 15: i - 1]),
                np.abs(low[i - 14: i] - close[i - 15: i - 1]),
            )
            atr_14 = float(np.mean(tr_arr) / close[i]) if close[i] > 0 else 0
        else:
            atr_14 = 0

        signals.append({
            "symbol": symbol,
            "signal_time": pd.Timestamp(times[i]),
            "bar_idx": i,
            "vol_ratio": vol_ratio,
            "price_move": price_move,
            "price_change_1bar": price_change_1bar,
            "price_change_4bar": price_change_4bar,
            "bar_range_norm": bar_range_norm,
            "upper_wick_pct": upper_wick_pct,
            "lower_wick_pct": lower_wick_pct,
            "body_pct": body_pct,
            "vol_trend": vol_trend,
            "price_trend_4bar": price_trend_4bar,
            "price_trend_24bar": price_trend_24bar,
            "volatility_20": volatility_20,
            "rsi_14": rsi_14,
            "btc_ret_4bar": btc_ret_4bar,
            "btc_ret_24bar": btc_ret_24bar,
            "hour_of_day": hour_of_day,
            "day_of_week": day_of_week,
            "bars_since_last_spike": bars_since_last_spike,
            "atr_14": atr_14,
        })
        cooldown_until = i + COOLDOWN_BARS

    return signals


# ── Main backtest ────────────────────────────────────────────────────

def run_backtest():
    """Load all data once, compute regime, scan each coin, sim trades for both variants."""
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(symbol_dirs)} coin directories")

    # ── Step 1: Load ALL coin data ───────────────────────────────────
    print("Loading all coin data...")
    all_coin_data: dict[str, pd.DataFrame] = {}
    for sym_dir in symbol_dirs:
        df = load_coin(sym_dir)
        if df is not None and len(df) >= 500:
            all_coin_data[sym_dir.name] = df
    print(f"  Loaded {len(all_coin_data)} coins with >= 500 bars")

    if "BTCUSDT" not in all_coin_data:
        raise RuntimeError("BTCUSDT data not found — needed for regime filter")

    btc_df = all_coin_data["BTCUSDT"]
    btc_times = btc_df["open_time"].values
    btc_close = btc_df["close"].values.astype(float)

    # ── Step 2: Compute daily regime ─────────────────────────────────
    print("Computing daily regime (BTC daily EMA + altcoin breadth)...")
    regime = compute_daily_regime(all_coin_data, btc_df)
    print(f"  Regime computed for {len(regime)} days")

    # ── Step 3: Scan all coins ───────────────────────────────────────
    print("Scanning coins for entry signals...")
    all_signals: list[dict] = []
    # Keep coin data arrays for trade sim
    coin_arrays: dict[str, tuple] = {}

    for idx, (symbol, df) in enumerate(all_coin_data.items()):
        if symbol == "BTCUSDT":
            continue

        signals = scan_coin(symbol, df, btc_times, btc_close, regime)
        if signals:
            all_signals.extend(signals)
            coin_arrays[symbol] = (
                df["close"].values.astype(float),
                df["high"].values.astype(float),
                df["low"].values.astype(float),
            )

        if (idx + 1) % 20 == 0:
            print(f"  {idx + 1}/{len(all_coin_data)} coins, "
                  f"{len(all_signals)} signals so far")

    print(f"\nScan complete: {len(all_signals)} signals across all coins")

    # ── Step 4: Simulate trades for each variant ─────────────────────
    variant_trades: dict[str, list] = {}
    for vname, vparams in VARIANTS.items():
        print(f"\nSimulating trades for {vname} "
              f"(SL={vparams['stop_pct']:.0%} TP={vparams['tp_pct']:.0%} "
              f"TO={vparams['timeout']} bars)...")
        trades = []
        for sig in all_signals:
            symbol = sig["symbol"]
            if symbol not in coin_arrays:
                continue
            close, high, low = coin_arrays[symbol]
            result = simulate_trade(
                close, high, low, sig["bar_idx"],
                vparams["stop_pct"], vparams["tp_pct"], vparams["timeout"],
            )
            if result:
                trade = {**sig, **result}
                del trade["bar_idx"]
                trade["variant"] = vname
                trades.append(trade)
        variant_trades[vname] = trades
        print(f"  {vname}: {len(trades)} trades")

    return variant_trades, regime


# ── Reporting ────────────────────────────────────────────────────────

def print_variant_summary(trades, label, stop_pct, tp_pct, timeout):
    """Print stats for one variant."""
    if not trades:
        print(f"\n  {label}: No trades found.")
        return

    df = pd.DataFrame(trades)
    n = len(df)
    wins = (df["pnl_pct"] > 0).sum()
    losses = (df["pnl_pct"] <= 0).sum()
    wr = wins / n * 100
    avg_pnl = df["pnl_pct"].mean() * 100
    total_pnl = df["pnl_pct"].sum() * 100

    gross_profit = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
    gross_loss = abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    tp_n = (df["exit_reason"] == "take_profit").sum()
    sl_n = (df["exit_reason"] == "stop_loss").sum()
    to_n = (df["exit_reason"] == "timeout").sum()
    avg_bars = df["bars_held"].mean()

    print(f"\n  {label}")
    print(f"  Exit: SL={stop_pct:.0%}  TP={tp_pct:.0%}  "
          f"Timeout={timeout} bars ({timeout * 15 / 60:.0f}h)")
    print(f"  Trades:       {n}")
    print(f"  Wins/Losses:  {wins}/{losses}")
    print(f"  Win Rate:     {wr:.1f}%")
    print(f"  Profit Factor:{pf:>7.2f}")
    print(f"  Avg PnL:      {avg_pnl:+.2f}%")
    print(f"  Total PnL:    {total_pnl:+.1f}%")
    print(f"  Avg Bars:     {avg_bars:.0f} ({avg_bars * 15 / 60:.1f}h)")
    print(f"  TP/SL/TO:     {tp_n} ({tp_n/n*100:.1f}%) / "
          f"{sl_n} ({sl_n/n*100:.1f}%) / "
          f"{to_n} ({to_n/n*100:.1f}%)")
    print(f"  Unique coins: {df['symbol'].nunique()}")

    # Monthly breakdown
    print(f"\n  {'MONTHLY BREAKDOWN':}")
    df["month"] = pd.to_datetime(df["signal_time"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades=("pnl_pct", "size"),
        wins=("pnl_pct", lambda x: (x > 0).sum()),
        total_pnl=("pnl_pct", "sum"),
    )
    monthly["wr"] = monthly["wins"] / monthly["trades"] * 100

    print(f"  {'Month':<10} {'Trades':>7} {'Wins':>5} "
          f"{'WR%':>6} {'TotPnL':>8}")
    print("  " + "-" * 45)
    for period, row in monthly.iterrows():
        print(f"  {str(period):<10} {int(row['trades']):>7} {int(row['wins']):>5} "
              f"{row['wr']:>5.1f}% {row['total_pnl']*100:>+7.1f}%")


def print_regime_stats(regime):
    """Print regime filter statistics."""
    print("\nREGIME FILTER STATISTICS")
    print("=" * 65)

    allowed_days = sum(1 for v in regime.values() if v["allowed"])
    blocked_days = sum(1 for v in regime.values() if not v["allowed"])
    total_days = len(regime)
    btc_bear_days = sum(1 for v in regime.values() if v["btc_bearish"])
    breadth_low_days = sum(1 for v in regime.values() if v["breadth_low"])

    print(f"  Total days:          {total_days}")
    print(f"  Allowed (both):      {allowed_days} ({allowed_days/total_days*100:.1f}%)")
    print(f"  Blocked:             {blocked_days} ({blocked_days/total_days*100:.1f}%)")
    print(f"  BTC bearish days:    {btc_bear_days} ({btc_bear_days/total_days*100:.1f}%)")
    print(f"  Breadth < 60% days:  {breadth_low_days} "
          f"({breadth_low_days/total_days*100:.1f}%)")

    # Monthly breakdown of allowed/blocked
    print(f"\n  {'MONTHLY REGIME BREAKDOWN':}")
    monthly: dict = {}
    for date, v in regime.items():
        m = date.strftime("%Y-%m")
        if m not in monthly:
            monthly[m] = {"allowed": 0, "blocked": 0, "total": 0}
        monthly[m]["total"] += 1
        if v["allowed"]:
            monthly[m]["allowed"] += 1
        else:
            monthly[m]["blocked"] += 1

    print(f"  {'Month':<10} {'Total':>6} {'Allowed':>8} {'Blocked':>8} {'Allow%':>7}")
    print("  " + "-" * 45)
    for m in sorted(monthly.keys()):
        r = monthly[m]
        pct = r["allowed"] / r["total"] * 100 if r["total"] > 0 else 0
        print(f"  {m:<10} {r['total']:>6} {r['allowed']:>8} {r['blocked']:>8} "
              f"{pct:>6.1f}%")


def print_summary(variant_trades, regime):
    """Print full report."""
    print("\n" + "=" * 65)
    print("REGIME FILTER BACKTEST RESULTS")
    print("Regime: BTC daily close < 20-day EMA AND < 60% coins above 20-EMA")
    print("Entry:  vol spike >= 3x + price drop >= 5%")
    print("=" * 65)

    print_regime_stats(regime)

    for vname, vparams in VARIANTS.items():
        print("\n" + "-" * 65)
        print_variant_summary(
            variant_trades[vname], vname,
            vparams["stop_pct"], vparams["tp_pct"], vparams["timeout"],
        )

    # Comparison line
    print("\n" + "=" * 65)
    print("COMPARISON")
    print("-" * 65)
    for vname, trades in variant_trades.items():
        if not trades:
            continue
        df = pd.DataFrame(trades)
        n = len(df)
        wr = (df["pnl_pct"] > 0).sum() / n * 100
        gp = df.loc[df["pnl_pct"] > 0, "pnl_pct"].sum()
        gl = abs(df.loc[df["pnl_pct"] <= 0, "pnl_pct"].sum())
        pf = gp / gl if gl > 0 else float("inf")
        print(f"  {vname:<25} {n:>6} trades, WR {wr:.1f}%, PF {pf:.2f}")
    print(f"  {'Original (BTC 15m EMA)':<25} 59545 trades, WR ~50%, PF 1.09")
    print("=" * 65)


def save_to_db(variant_trades):
    """Save all trades to scanner_short_regime table."""
    all_trades = []
    for trades in variant_trades.values():
        all_trades.extend(trades)

    if not all_trades:
        print("No trades to save.")
        return

    df = pd.DataFrame(all_trades)
    if "month" in df.columns:
        df = df.drop(columns=["month"])

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_short_regime"))
        conn.commit()
    df.to_sql("scanner_short_regime", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} trades to scanner_short_regime table.")


if __name__ == "__main__":
    variant_trades, regime = run_backtest()
    print_summary(variant_trades, regime)
    save_to_db(variant_trades)
