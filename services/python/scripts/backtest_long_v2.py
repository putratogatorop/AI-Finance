# scripts/backtest_long_v2.py
"""Scanner Long v2 — Delayed Entry + Regime Filter (mirror of short v2).

Combines:
1. Regime gate: BTC daily close > 20-day EMA AND altcoin breadth > 40%
2. Phase 1 — Detect pump: vol spike >= 3x + price up 5% from 96-bar low
3. Phase 2 — Wait for pullback: price pulls back >= 2% from pump high within 96 bars
4. Phase 3 — Enter on reclaim: close > pullback_low * 1.03

Exit variants:
  A: 5% stop / 5% target / 192-bar timeout (1:1, 48h)
  B: 5% stop / 10% target / 384-bar timeout (2:1, 96h)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

sys.path.insert(0, ".")

from scripts.backtest_momentum_scanner import load_coin  # noqa: E402

import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
COOLDOWN_BARS = 96

PULLBACK_WINDOW = 96
PULLBACK_MIN_PCT = 0.02
RECLAIM_PCT = 0.03

VARIANT_A = {"stop_pct": 0.05, "tp_pct": 0.05, "max_bars": 192, "label": "A (5%/5% 48h)"}
VARIANT_B = {"stop_pct": 0.05, "tp_pct": 0.10, "max_bars": 384, "label": "B (5%/10% 96h)"}

# Round-trip friction per trade: ~0.10% fees (Gate.io spot 0.1% maker/taker round-trip)
# + 0.05% spread/slippage on altcoins = 0.15% total. Subtracted from pnl on every trade,
# winners and losers alike.
FRICTION_PCT = 0.0015

FEATURE_COLS = [
    "vol_ratio", "price_move", "price_change_1bar", "price_change_4bar",
    "bar_range_norm", "upper_wick_pct", "lower_wick_pct", "body_pct",
    "vol_trend", "price_trend_4bar", "price_trend_24bar",
    "volatility_20", "rsi_14",
    "btc_ret_4bar", "btc_ret_24bar",
    "hour_of_day", "day_of_week",
    "bars_since_last_spike", "atr_14",
]

EXTRA_FEATURES = [
    "pullback_pct",
    "pullback_bars",
    "pullback_vol_ratio",
    "reclaim_speed",
]

engine = create_engine(DB_URL)


def compute_daily_regime(all_coin_data: dict, btc_df: pd.DataFrame) -> dict:
    """Bullish regime: BTC close > 20d EMA AND breadth > 40%."""
    btc_daily = btc_df.set_index("open_time").resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    btc_daily["ema20"] = btc_daily["close"].ewm(span=20, adjust=False).mean()
    btc_daily["bullish"] = btc_daily["close"] > btc_daily["ema20"]

    coin_above_ema: dict = {}
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

    regime = {}
    for date in sorted(coin_above_ema.keys()):
        above, total = coin_above_ema[date]
        breadth = above / total if total > 0 else 0.5

        btc_row = btc_daily[btc_daily.index.date == date]
        btc_bull = bool(btc_row["bullish"].iloc[0]) if len(btc_row) > 0 else False

        allowed = btc_bull and breadth > 0.40
        regime[date] = {
            "btc_bullish": btc_bull,
            "breadth": breadth,
            "breadth_high": breadth > 0.40,
            "allowed": allowed,
        }
    return regime


def simulate_long(close, high, low, entry_bar, stop_pct, tp_pct, max_bars):
    """Long trade. SL: price DOWN by stop_pct. TP: price UP by tp_pct."""
    if entry_bar + 1 >= len(close):
        return None

    entry_price = close[entry_bar]
    if entry_price <= 0:
        return None

    sl_price = entry_price * (1 - stop_pct)
    tp_price = entry_price * (1 + tp_pct)

    last_bar = min(entry_bar + 1 + max_bars, len(close))

    for bar in range(entry_bar + 1, last_bar):
        if low[bar] <= sl_price:
            return {
                "pnl_pct": -stop_pct - FRICTION_PCT,
                "exit_reason": "stop_loss",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }
        if high[bar] >= tp_price:
            return {
                "pnl_pct": tp_pct - FRICTION_PCT,
                "exit_reason": "take_profit",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": tp_price,
            }

    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (exit_price - entry_price) / entry_price - FRICTION_PCT
    return {
        "pnl_pct": pnl_pct,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


def extract_features(
    i, close, high, low, open_arr, volume, vol_ma, times,
    btc_times, btc_close,
    pullback_pct, pullback_bars, pullback_vol_ratio, reclaim_speed,
):
    """Extract 19 standard + 4 pullback-specific features at bar i."""
    vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0

    rolling_low = np.min(low[i - PRICE_LOOKBACK: i])
    price_move = (close[i] - rolling_low) / rolling_low if rolling_low > 0 else 0

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

    return {
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
        "pullback_pct": pullback_pct,
        "pullback_bars": pullback_bars,
        "pullback_vol_ratio": pullback_vol_ratio,
        "reclaim_speed": reclaim_speed,
    }


def scan_coin(symbol, df, btc_times, btc_close, regime):
    """Three-phase long entry: pump → pullback → reclaim."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    open_arr = df["open"].values.astype(float)
    volume = df["volume"].values.astype(float)
    times = df["open_time"].values

    vol_ma = pd.Series(volume).rolling(VOL_MA_PERIOD).mean().values
    trades_a = []
    trades_b = []
    cooldown_until = 0

    for i in range(VOL_MA_PERIOD + PRICE_LOOKBACK, len(close) - PULLBACK_WINDOW):
        if i < cooldown_until:
            continue

        # Phase 1: Detect pump
        if vol_ma[i] <= 0 or volume[i] / vol_ma[i] < VOL_SPIKE:
            continue

        rolling_low = np.min(low[i - PRICE_LOOKBACK: i])
        if rolling_low <= 0:
            continue
        price_rise = (close[i] - rolling_low) / rolling_low
        if price_rise < PRICE_MOVE_THRESH:
            continue

        bar_date = pd.Timestamp(times[i]).date()
        if bar_date not in regime or not regime[bar_date]["allowed"]:
            continue

        pump_close = close[i]
        pump_bar = i

        # Phase 2: Wait for pullback (-2% from pump close lowest low)
        pullback_low = pump_close
        pullback_bar = None
        for j in range(pump_bar + 1, min(pump_bar + PULLBACK_WINDOW + 1, len(close))):
            if close[j] < pullback_low:
                pullback_low = close[j]
                pullback_bar = j

        if pullback_bar is None or (pump_close - pullback_low) / pump_close < PULLBACK_MIN_PCT:
            continue

        # Phase 3: Enter on reclaim
        reclaim_level = pullback_low * (1 + RECLAIM_PCT)
        entry_bar = None
        for j in range(pullback_bar + 1, min(pump_bar + PULLBACK_WINDOW + 1, len(close))):
            if close[j] > reclaim_level:
                entry_bar = j
                break

        if entry_bar is None:
            continue

        pullback_pct = (pump_close - pullback_low) / pump_close
        pullback_bars = pullback_bar - pump_bar
        pullback_slice = volume[pump_bar + 1: pullback_bar + 1]
        vol_ma_at_pb = vol_ma[pullback_bar] if vol_ma[pullback_bar] > 0 else 1.0
        pullback_vol_ratio = (
            float(np.mean(pullback_slice) / vol_ma_at_pb)
            if len(pullback_slice) > 0 else 0.0
        )
        reclaim_speed = entry_bar - pullback_bar

        features = extract_features(
            entry_bar, close, high, low, open_arr, volume, vol_ma, times,
            btc_times, btc_close,
            pullback_pct, pullback_bars, pullback_vol_ratio, reclaim_speed,
        )

        entry_time = pd.Timestamp(times[entry_bar])
        pump_time = pd.Timestamp(times[pump_bar])

        base = {
            "symbol": symbol,
            "signal_time": entry_time,
            "pump_time": pump_time,
            "pullback_pct": pullback_pct,
            "pullback_bars": pullback_bars,
            "pullback_vol_ratio": pullback_vol_ratio,
            "reclaim_speed": reclaim_speed,
            **features,
        }

        trade_a = simulate_long(
            close, high, low, entry_bar,
            VARIANT_A["stop_pct"], VARIANT_A["tp_pct"], VARIANT_A["max_bars"],
        )
        if trade_a:
            trade_a = {**base, **trade_a, "variant": "A"}
            trades_a.append(trade_a)

        trade_b = simulate_long(
            close, high, low, entry_bar,
            VARIANT_B["stop_pct"], VARIANT_B["tp_pct"], VARIANT_B["max_bars"],
        )
        if trade_b:
            trade_b = {**base, **trade_b, "variant": "B"}
            trades_b.append(trade_b)

        cooldown_until = entry_bar + COOLDOWN_BARS

    return trades_a, trades_b


def run_backtest():
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    print(f"Found {len(symbol_dirs)} coin directories")

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

    print("Computing daily regime (BTC > 20d EMA + breadth > 40%)...")
    regime = compute_daily_regime(all_coin_data, btc_df)
    allowed_days = sum(1 for v in regime.values() if v["allowed"])
    print(f"  Allowed days: {allowed_days} / {len(regime)} "
          f"({allowed_days / len(regime) * 100:.1f}%)")

    print("Scanning coins for pump -> pullback -> reclaim signals...")
    all_a: list[dict] = []
    all_b: list[dict] = []

    for idx, (symbol, df) in enumerate(all_coin_data.items()):
        if symbol == "BTCUSDT":
            continue
        trades_a, trades_b = scan_coin(symbol, df, btc_times, btc_close, regime)
        all_a.extend(trades_a)
        all_b.extend(trades_b)
        if (idx + 1) % 20 == 0:
            print(
                f"  {idx + 1}/{len(all_coin_data)} coins, "
                f"A={len(all_a)} B={len(all_b)} trades"
            )

    print(f"\nScan complete: A={len(all_a)}, B={len(all_b)} trades")
    return all_a, all_b, regime


def print_variant_summary(trades, label):
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
    print(f"\n  Variant {label}:")
    print(f"    Trades:       {n}")
    print(f"    Wins/Losses:  {wins}/{losses}")
    print(f"    Win Rate:     {wr:.1f}%")
    print(f"    Profit Factor:{pf:>7.2f}")
    print(f"    Avg PnL:      {avg_pnl:+.2f}%")
    print(f"    Total PnL:    {total_pnl:+.1f}%")
    print(f"    Avg Bars Held:{avg_bars:.0f} ({avg_bars * 15 / 60:.1f}h)")
    print(f"    TP/SL/TO:     {tp_n} ({tp_n/n*100:.1f}%) / "
          f"{sl_n} ({sl_n/n*100:.1f}%) / "
          f"{to_n} ({to_n/n*100:.1f}%)")
    print(f"    Unique coins: {df['symbol'].nunique()}")


def save_to_db(trades_a, trades_b):
    all_trades = trades_a + trades_b
    if not all_trades:
        print("No trades to save.")
        return
    df = pd.DataFrame(all_trades)
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_long_v2"))
        conn.commit()
    df.to_sql("scanner_long_v2", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} trades to scanner_long_v2 table.")


if __name__ == "__main__":
    trades_a, trades_b, regime = run_backtest()
    print("\n" + "=" * 65)
    print("SCANNER LONG V2 — Delayed Entry + Regime Filter")
    print("=" * 65)
    print_variant_summary(trades_a, VARIANT_A["label"])
    print_variant_summary(trades_b, VARIANT_B["label"])
    save_to_db(trades_a, trades_b)
