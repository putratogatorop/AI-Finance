# scripts/backtest_short_v2.py
"""Scanner Short v2 — Delayed Entry + Regime Filter.

Combines both improvements:
1. Regime gate: BTC daily close < 20-day EMA AND altcoin breadth < 60%
2. Phase 1 — Detect dump: vol spike >= 3x + price down 5% from 96-bar high
3. Phase 2 — Wait for bounce: price bounces >= 2% from dump low within 96 bars
4. Phase 3 — Enter on re-rejection: close < bounce_high * 0.97

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

# ── Database ─────────────────────────────────────────────────────────
import os
DB_URL = os.environ.get("DATABASE_URL", "postgresql://postgres:MySQL100%25@localhost:5432/market")

# ── Paths ────────────────────────────────────────────────────────────
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")

# ── Strategy params ──────────────────────────────────────────────────
VOL_SPIKE = 3.0
PRICE_MOVE_THRESH = 0.05
VOL_MA_PERIOD = 20
PRICE_LOOKBACK = 96
COOLDOWN_BARS = 96

# ── Bounce detection params ──────────────────────────────────────────
BOUNCE_WINDOW = 96          # max bars after dump to find bounce + rejection
BOUNCE_MIN_PCT = 0.02       # bounce must be at least +2% from dump close
REJECTION_PCT = 0.03        # price gives back 3% from bounce high

# ── Variant params ───────────────────────────────────────────────────
VARIANT_A = {"stop_pct": 0.05, "tp_pct": 0.05, "max_bars": 192, "label": "A (5%/5% 48h)"}
VARIANT_B = {"stop_pct": 0.05, "tp_pct": 0.10, "max_bars": 384, "label": "B (5%/10% 96h)"}

# Round-trip friction per trade: ~0.10% fees (Gate.io spot 0.1% maker/taker round-trip)
# + 0.05% spread/slippage on altcoins = 0.15% total. Subtracted from pnl on every trade,
# winners and losers alike.
FRICTION_PCT = 0.0015

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

EXTRA_FEATURES = [
    "bounce_pct",           # How much did the bounce recover (%)
    "bounce_bars",          # How many bars did the bounce take
    "bounce_vol_ratio",     # Avg vol during bounce vs vol_ma
    "rejection_speed",      # Bars from bounce peak to entry
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

def simulate_short(close, high, low, entry_bar, stop_pct, tp_pct, max_bars):
    """Simulate a short trade from entry_bar.

    Entry at close[entry_bar].
    SL: price goes UP by stop_pct.
    TP: price goes DOWN by tp_pct.
    """
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
                "pnl_pct": -stop_pct - FRICTION_PCT,
                "exit_reason": "stop_loss",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": sl_price,
            }
        if low[bar] <= tp_price:
            return {
                "pnl_pct": tp_pct - FRICTION_PCT,
                "exit_reason": "take_profit",
                "bars_held": bar - entry_bar,
                "entry_price": entry_price,
                "exit_price": tp_price,
            }

    # Timeout
    exit_bar = last_bar - 1
    exit_price = close[exit_bar]
    pnl_pct = (entry_price - exit_price) / entry_price - FRICTION_PCT
    return {
        "pnl_pct": pnl_pct,
        "exit_reason": "timeout",
        "bars_held": exit_bar - entry_bar,
        "entry_price": entry_price,
        "exit_price": exit_price,
    }


# ── Feature extraction ───────────────────────────────────────────────

def extract_features(
    i, close, high, low, open_arr, volume, vol_ma, times,
    btc_times, btc_close,
    bounce_pct, bounce_bars, bounce_vol_ratio, rejection_speed,
):
    """Extract 19 standard + 4 bounce-specific features at bar i."""
    vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0

    rolling_high = np.max(high[i - PRICE_LOOKBACK: i])
    price_move = (rolling_high - close[i]) / rolling_high if rolling_high > 0 else 0

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
        # Bounce-specific features
        "bounce_pct": bounce_pct,
        "bounce_bars": bounce_bars,
        "bounce_vol_ratio": bounce_vol_ratio,
        "rejection_speed": rejection_speed,
    }


# ── Coin scanner ─────────────────────────────────────────────────────

def scan_coin(symbol, df, btc_times, btc_close, regime):
    """Scan one coin for delayed-entry short signals with regime filter.

    Three-phase entry:
      Phase 1: Detect dump (vol spike + price drop + regime gate)
      Phase 2: Wait for bounce (+2% from dump close)
      Phase 3: Enter on re-rejection (close < bounce_high * 0.97)

    Returns (trades_a, trades_b) lists.
    """
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

    for i in range(VOL_MA_PERIOD + PRICE_LOOKBACK, len(close) - BOUNCE_WINDOW):
        if i < cooldown_until:
            continue

        # ── Phase 1: Detect dump (watchlist trigger) ─────────────
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

        # Gate 3: Daily regime filter (BTC bearish + breadth low)
        bar_date = pd.Timestamp(times[i]).date()
        if bar_date not in regime or not regime[bar_date]["allowed"]:
            continue

        dump_close = close[i]
        dump_bar = i

        # ── Phase 2: Wait for bounce (+2% from dump close) ──────
        bounce_high = dump_close
        bounce_bar = None
        for j in range(dump_bar + 1, min(dump_bar + BOUNCE_WINDOW + 1, len(close))):
            if close[j] > bounce_high:
                bounce_high = close[j]
                bounce_bar = j

        if bounce_bar is None or (bounce_high - dump_close) / dump_close < BOUNCE_MIN_PCT:
            continue  # No meaningful bounce

        # ── Phase 3: Enter on re-rejection ───────────────────────
        rejection_level = bounce_high * (1 - REJECTION_PCT)
        entry_bar = None
        for j in range(bounce_bar + 1, min(dump_bar + BOUNCE_WINDOW + 1, len(close))):
            if close[j] < rejection_level:
                entry_bar = j
                break

        if entry_bar is None:
            continue  # No rejection within window

        # ── Compute bounce-specific features ─────────────────────
        bounce_pct = (bounce_high - dump_close) / dump_close
        bounce_bars = bounce_bar - dump_bar
        # Avg volume during bounce relative to vol_ma
        bounce_slice = volume[dump_bar + 1: bounce_bar + 1]
        vol_ma_at_bounce = vol_ma[bounce_bar] if vol_ma[bounce_bar] > 0 else 1.0
        bounce_vol_ratio = (
            float(np.mean(bounce_slice) / vol_ma_at_bounce)
            if len(bounce_slice) > 0 else 0.0
        )
        rejection_speed = entry_bar - bounce_bar

        # ── Extract ML features at entry bar ─────────────────────
        features = extract_features(
            entry_bar, close, high, low, open_arr, volume, vol_ma, times,
            btc_times, btc_close,
            bounce_pct, bounce_bars, bounce_vol_ratio, rejection_speed,
        )

        entry_time = pd.Timestamp(times[entry_bar])
        dump_time = pd.Timestamp(times[dump_bar])

        base = {
            "symbol": symbol,
            "signal_time": entry_time,
            "dump_time": dump_time,
            "bounce_pct": bounce_pct,
            "bounce_bars": bounce_bars,
            "bounce_vol_ratio": bounce_vol_ratio,
            "rejection_speed": rejection_speed,
            **features,
        }

        # Simulate Variant A
        trade_a = simulate_short(
            close, high, low, entry_bar,
            VARIANT_A["stop_pct"], VARIANT_A["tp_pct"], VARIANT_A["max_bars"],
        )
        if trade_a:
            trade_a = {**base, **trade_a, "variant": "A"}
            trades_a.append(trade_a)

        # Simulate Variant B
        trade_b = simulate_short(
            close, high, low, entry_bar,
            VARIANT_B["stop_pct"], VARIANT_B["tp_pct"], VARIANT_B["max_bars"],
        )
        if trade_b:
            trade_b = {**base, **trade_b, "variant": "B"}
            trades_b.append(trade_b)

        cooldown_until = entry_bar + COOLDOWN_BARS

    return trades_a, trades_b


# ── Main backtest ────────────────────────────────────────────────────

def run_backtest():
    """Load all data, compute regime, scan each coin with delayed entry."""
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

    allowed_days = sum(1 for v in regime.values() if v["allowed"])
    print(f"  Allowed days: {allowed_days} / {len(regime)} "
          f"({allowed_days / len(regime) * 100:.1f}%)")

    # ── Step 3: Scan all coins (delayed entry + regime) ──────────────
    print("Scanning coins for delayed-entry + regime signals...")
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


# ── Reporting ────────────────────────────────────────────────────────

def print_variant_summary(trades, label):
    """Print summary for one variant."""
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
    avg_bounce = df["bounce_pct"].mean() * 100
    avg_rej_speed = df["rejection_speed"].mean()

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
    print(f"    Avg Bounce:   {avg_bounce:.1f}% before rejection")
    print(f"    Avg Rej Speed:{avg_rej_speed:.0f} bars from bounce peak")
    print(f"    Unique coins: {df['symbol'].nunique()}")


def print_top_coins(trades, label, n=10):
    """Print top coins by trade count."""
    if not trades:
        return
    df = pd.DataFrame(trades)
    coin_stats = df.groupby("symbol").agg(
        trades=("pnl_pct", "size"),
        wins=("pnl_pct", lambda x: (x > 0).sum()),
        total_pnl=("pnl_pct", "sum"),
    )
    coin_stats["wr"] = coin_stats["wins"] / coin_stats["trades"] * 100
    coin_stats = coin_stats.sort_values("trades", ascending=False).head(n)

    print(f"\n  TOP {n} COINS — Variant {label}:")
    print(f"  {'Symbol':<15} {'Trades':>7} {'Wins':>5} "
          f"{'WR%':>6} {'TotPnL':>8}")
    print("  " + "-" * 48)
    for symbol, row in coin_stats.iterrows():
        print(f"  {symbol:<15} {int(row['trades']):>7} {int(row['wins']):>5} "
              f"{row['wr']:>5.1f}% {row['total_pnl']*100:>+7.1f}%")


def print_monthly(trades, label):
    """Print monthly breakdown."""
    if not trades:
        return
    df = pd.DataFrame(trades)
    df["month"] = pd.to_datetime(df["signal_time"]).dt.to_period("M")
    monthly = df.groupby("month").agg(
        trades=("pnl_pct", "size"),
        wins=("pnl_pct", lambda x: (x > 0).sum()),
        total_pnl=("pnl_pct", "sum"),
        avg_pnl=("pnl_pct", "mean"),
    )
    monthly["wr"] = monthly["wins"] / monthly["trades"] * 100

    print(f"\n  MONTHLY BREAKDOWN — Variant {label}:")
    print(f"  {'Month':<10} {'Trades':>7} {'Wins':>5} "
          f"{'WR%':>6} {'TotPnL':>8} {'AvgPnL':>8}")
    print("  " + "-" * 55)
    for period, row in monthly.iterrows():
        print(f"  {str(period):<10} {int(row['trades']):>7} {int(row['wins']):>5} "
              f"{row['wr']:>5.1f}% "
              f"{row['total_pnl']*100:>+7.1f}% {row['avg_pnl']*100:>+7.2f}%")


def print_summary(trades_a, trades_b):
    """Print full comparison report."""
    print("\n" + "=" * 65)
    print("SCANNER SHORT V2 — Delayed Entry + Regime Filter")
    print("=" * 65)
    print("Regime:  BTC daily close < 20-day EMA AND breadth < 60%")
    print("Phase 1: vol spike >= 3x + price drop >= 5% from 96-bar high")
    print(f"Phase 2: wait for bounce >= {BOUNCE_MIN_PCT:.0%} from dump close")
    print(f"Phase 3: enter when close < bounce_high * {1 - REJECTION_PCT:.2f}")
    print(f"Cooldown: {COOLDOWN_BARS} bars after entry")
    print("=" * 65)

    print_variant_summary(trades_a, VARIANT_A["label"])
    print_variant_summary(trades_b, VARIANT_B["label"])

    # Comparison with previous backtests
    print("\n" + "-" * 65)
    print("COMPARISON WITH PREVIOUS BACKTESTS")
    print("-" * 65)
    print("  vs Original:            59545 trades, 32% WR, PF 1.09")
    print("  vs Delayed only:        27443 trades, 65% WR, PF 2.05")
    print("  vs Regime only:         36135 trades, 35% WR, PF 1.26")

    if trades_a:
        df_a = pd.DataFrame(trades_a)
        n_a = len(df_a)
        wr_a = (df_a["pnl_pct"] > 0).sum() / n_a * 100
        gp_a = df_a.loc[df_a["pnl_pct"] > 0, "pnl_pct"].sum()
        gl_a = abs(df_a.loc[df_a["pnl_pct"] <= 0, "pnl_pct"].sum())
        pf_a = gp_a / gl_a if gl_a > 0 else float("inf")
        print(f"  v2 Variant A:           {n_a} trades, "
              f"{wr_a:.0f}% WR, PF {pf_a:.2f}")

    if trades_b:
        df_b = pd.DataFrame(trades_b)
        n_b = len(df_b)
        wr_b = (df_b["pnl_pct"] > 0).sum() / n_b * 100
        gp_b = df_b.loc[df_b["pnl_pct"] > 0, "pnl_pct"].sum()
        gl_b = abs(df_b.loc[df_b["pnl_pct"] <= 0, "pnl_pct"].sum())
        pf_b = gp_b / gl_b if gl_b > 0 else float("inf")
        print(f"  v2 Variant B:           {n_b} trades, "
              f"{wr_b:.0f}% WR, PF {pf_b:.2f}")

    # Monthly and top coins for Variant A
    print_monthly(trades_a, VARIANT_A["label"])
    print_top_coins(trades_a, VARIANT_A["label"])

    # Monthly and top coins for Variant B
    print_monthly(trades_b, VARIANT_B["label"])
    print_top_coins(trades_b, VARIANT_B["label"])

    print("\n" + "=" * 65)
    print("This is a BLIND backtest — no big_movers lookahead.")
    print("Delayed entry + regime filter for maximum selectivity.")
    print("=" * 65)


def save_to_db(trades_a, trades_b):
    """Save all trades to scanner_short_v2 table."""
    all_trades = trades_a + trades_b
    if not all_trades:
        print("No trades to save.")
        return

    df = pd.DataFrame(all_trades)
    if "month" in df.columns:
        df = df.drop(columns=["month"])

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS scanner_short_v2"))
        conn.commit()
    df.to_sql("scanner_short_v2", engine, if_exists="replace", index=False)
    print(f"\nSaved {len(df)} trades to scanner_short_v2 table.")


if __name__ == "__main__":
    trades_a, trades_b, regime = run_backtest()
    print_summary(trades_a, trades_b)
    save_to_db(trades_a, trades_b)
