"""BTC Swing Trading Backtest — Expert-Informed v3.

Changes from v1/v2 based on ML Expert + Quant + Risk advisor consensus:
1. Classification target: P(BTC up >3% in 14d) instead of raw return regression
2. Stationary features only (ratios, returns, oscillators — no raw price levels)
3. Removed fake market_context features (were np.random)
4. ATR-based stops: 2x ATR(14), floor 5%, ceiling 10%
5. Regime filter gates: trend (>50-SMA), volatility, momentum, model confidence
6. Purged walk-forward with embargo gap
7. Reduced feature set (~18 features for 762 rows)
8. XGBoost only (LSTM overfits at this data size, LGB too similar to XGB)
9. Simple LGB as diversity model with different hyperparams
"""

import logging
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, ".")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DB_URL = "postgresql://postgres:MySQL100%25@localhost:5432/market"

# ── Strategy parameters ──
TARGET_RETURN_PCT = 0.03       # Predict: will BTC go up >3% in 14 days?
TARGET_HORIZON = 14            # days
CONFIDENCE_GATE = 0.60         # min predicted probability to act
COOLDOWN_DAYS = 14             # min days between entries
MAX_POSITIONS = 3
ATR_MULTIPLIER = 2.0           # stop distance = 2x ATR
ATR_STOP_FLOOR = 0.05          # min 5% stop
ATR_STOP_CEILING = 0.10        # max 10% stop — reject if wider
MOMENTUM_GATE = -0.12          # reject if 20d ROC < -12%
MAX_HOLD_DAYS = 42             # force exit after 6 weeks
STAGNATION_DAYS = 21           # exit if flat after 3 weeks
STAGNATION_BAND = 0.03         # "flat" = within +/-3%
RETRAIN_EVERY = 60             # retrain every 60 days
EMBARGO_DAYS = 21              # purge gap = 1.5x target horizon
MIN_TRAIN = 180                # min training samples


def load_data():
    engine = create_engine(DB_URL)
    daily = pd.read_sql(
        "SELECT date, asset, open, high, low, close, volume "
        "FROM asset_prices_daily WHERE asset='BTC' ORDER BY date", engine)
    daily["date"] = pd.to_datetime(daily["date"])
    engine.dispose()
    logger.info(f"Loaded {len(daily)} daily rows ({daily.date.iloc[0].date()} -> {daily.date.iloc[-1].date()})")
    return daily


def build_features(daily):
    """Build stationary features only. No raw price levels."""
    df = daily.copy()
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # ── Price-relative features (stationary) ──
    for p in [10, 20, 50]:
        sma = close.rolling(p).mean()
        df[f"price_to_sma_{p}"] = close / sma - 1

    # EMA ratios
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    df["ema_12_26_ratio"] = ema_12 / ema_26 - 1

    # ── Momentum (log returns at multiple scales) ──
    for d in [3, 7, 14, 28]:
        df[f"ret_{d}d"] = np.log(close / close.shift(d))

    # ── RSI ──
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta).where(delta < 0, 0.0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # ── MACD histogram ──
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    df["macd_hist"] = macd - macd_signal
    # Normalize by price to make stationary
    df["macd_hist_norm"] = df["macd_hist"] / close

    # ── Bollinger Band width (stationary) ──
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["bb_width"] = (2 * bb_std) / bb_mid
    df["bb_pct"] = (close - (bb_mid - 2 * bb_std)) / (4 * bb_std)  # %B

    # ── ATR normalized by price ──
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_14 = tr.rolling(14).mean()
    df["atr_norm"] = atr_14 / close

    # ── Volume features ──
    vol_sma_20 = volume.rolling(20).mean()
    df["volume_ratio"] = volume / vol_sma_20
    df["volume_change_5d"] = np.log(volume.rolling(5).mean() / vol_sma_20)

    # ── Volatility ──
    daily_ret = close.pct_change()
    df["volatility_10d"] = daily_ret.rolling(10).std()
    df["volatility_30d"] = daily_ret.rolling(30).std()
    df["vol_ratio"] = df["volatility_10d"] / df["volatility_30d"]  # vol compression/expansion

    # ── Regime indicators (rule-based, not ML) ──
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    df["trend_50"] = (close > sma_50).astype(float)
    df["trend_200"] = (close > sma_200).astype(float)
    df["sma_50_200_cross"] = (sma_50 > sma_200).astype(float)

    # ── Day-of-week (crypto has calendar effects) ──
    df["day_of_week"] = df["date"].dt.dayofweek / 6.0  # normalize 0-1

    # Keep raw values needed for regime filter / stops
    df["raw_close"] = close
    df["raw_atr_14"] = atr_14
    df["raw_sma_50"] = sma_50
    df["roc_20"] = close.pct_change(20)

    # ── Feature columns (only stationary ones) ──
    feature_cols = [
        "price_to_sma_10", "price_to_sma_20", "price_to_sma_50",
        "ema_12_26_ratio",
        "ret_3d", "ret_7d", "ret_14d", "ret_28d",
        "rsi_14", "macd_hist_norm",
        "bb_width", "bb_pct",
        "atr_norm",
        "volume_ratio", "volume_change_5d",
        "volatility_10d", "volatility_30d", "vol_ratio",
        "trend_50", "trend_200", "sma_50_200_cross",
        "day_of_week",
    ]

    # Drop warmup rows
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    logger.info(f"Features: {len(feature_cols)} columns, {len(df)} samples after warmup")
    return df, feature_cols


def build_target(df):
    """Classification target: did BTC go up >3% in the next 14 days?"""
    forward_ret = df["raw_close"].shift(-TARGET_HORIZON) / df["raw_close"] - 1
    target = (forward_ret > TARGET_RETURN_PCT).astype(int)

    # Mark rows without target (last TARGET_HORIZON rows)
    valid = ~forward_ret.isna()

    pct_positive = target[valid].mean() * 100
    logger.info(f"Target: P(up >{TARGET_RETURN_PCT*100:.0f}% in {TARGET_HORIZON}d) — "
                f"{pct_positive:.1f}% positive class rate")
    return target, valid


def regime_filter(row):
    """4-gate regime filter. All must pass for a BUY signal."""
    reasons = []

    # Gate 1: Trend — price must be above 50-SMA
    if row["raw_close"] < row["raw_sma_50"]:
        reasons.append("below 50-SMA")

    # Gate 2: Volatility — stop would be too wide
    stop_pct = ATR_MULTIPLIER * row["raw_atr_14"] / row["raw_close"]
    if stop_pct > ATR_STOP_CEILING:
        reasons.append(f"vol too high ({stop_pct:.1%})")

    # Gate 3: Momentum — no catching falling knives
    if row["roc_20"] < MOMENTUM_GATE:
        reasons.append(f"ROC20={row['roc_20']:.1%}")

    return len(reasons) == 0, reasons


def walk_forward(df, feature_cols, target, valid):
    """Purged walk-forward with embargo gap."""
    n = len(df)
    all_signals = []
    last_signal_idx = -999
    models_trained = 0

    current = MIN_TRAIN
    while current < n:
        # Define windows with embargo
        train_end = current - EMBARGO_DAYS
        test_end = min(current + RETRAIN_EVERY, n)

        if train_end < 100:
            current += RETRAIN_EVERY
            continue

        # Train set: all valid rows before embargo
        train_mask = valid.iloc[:train_end]
        X_train = df[feature_cols].iloc[:train_end][train_mask]
        y_train = target.iloc[:train_end][train_mask]

        if len(X_train) < 80 or y_train.sum() < 10:
            current += RETRAIN_EVERY
            continue

        # Validation: between train_end and current (with embargo)
        val_start = train_end + EMBARGO_DAYS
        if val_start >= current:
            val_start = train_end
        val_mask = valid.iloc[val_start:current]
        X_val = df[feature_cols].iloc[val_start:current][val_mask]
        y_val = target.iloc[val_start:current][val_mask]

        test_dates = df["date"].iloc[current:test_end]
        logger.info(f"Window: train={len(X_train)} (pos={int(y_train.sum())}), "
                     f"test={test_end - current} "
                     f"({df.date.iloc[current].date()} -> {df.date.iloc[min(test_end-1, n-1)].date()})")

        # ── Train XGBoost classifier ──
        xgb = XGBClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7,
            min_child_weight=5, reg_alpha=0.5, reg_lambda=2.0,
            scale_pos_weight=len(y_train[y_train==0]) / max(len(y_train[y_train==1]), 1),
            random_state=42, n_jobs=-1, verbosity=0,
            eval_metric="logloss",
        )
        xgb.fit(X_train, y_train)

        # ── Train LightGBM classifier (diversity) ──
        lgb = LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.03,
            num_leaves=15, subsample=0.7, colsample_bytree=0.6,
            min_child_samples=10, reg_alpha=1.0, reg_lambda=3.0,
            is_unbalance=True,
            random_state=123, n_jobs=-1, verbose=-1,
        )
        lgb.fit(X_train, y_train)

        # Validation metrics
        if len(X_val) > 5:
            xgb_proba_val = xgb.predict_proba(X_val)[:, 1]
            lgb_proba_val = lgb.predict_proba(X_val)[:, 1]
            try:
                xgb_auc = roc_auc_score(y_val, xgb_proba_val)
                lgb_auc = roc_auc_score(y_val, lgb_proba_val)
                logger.info(f"  Val AUC: XGB={xgb_auc:.3f}, LGB={lgb_auc:.3f}")
            except ValueError:
                pass

        models_trained += 1

        # ── Generate signals on test chunk ──
        for i in range(current, test_end):
            if not valid.iloc[i]:
                continue

            # Cooldown check
            if (i - last_signal_idx) < COOLDOWN_DAYS:
                continue

            row = df.iloc[i]

            # Regime filter (4 gates)
            passes, reasons = regime_filter(row)
            if not passes:
                continue

            # Model predictions
            X_i = df[feature_cols].iloc[[i]]
            xgb_prob = xgb.predict_proba(X_i)[0][1]
            lgb_prob = lgb.predict_proba(X_i)[0][1]

            # Ensemble: average probability
            avg_prob = 0.55 * xgb_prob + 0.45 * lgb_prob

            # Gate 4: Confidence
            if avg_prob < CONFIDENCE_GATE:
                continue

            # ATR-based stop
            atr = row["raw_atr_14"]
            entry_price = row["raw_close"]
            stop_pct = max(ATR_MULTIPLIER * atr / entry_price, ATR_STOP_FLOOR)

            all_signals.append({
                "idx": i,
                "date": row["date"],
                "asset": "BTC",
                "action": "BUY",
                "confidence": avg_prob,
                "suggested_hold_days": TARGET_HORIZON,
                "stop_loss_pct": -stop_pct * 100,
                "expected_return_pct": TARGET_RETURN_PCT * 100,
                "model_agreement": f"prob={avg_prob:.2f}",
                "entry_price": entry_price,
                "atr_stop_price": entry_price * (1 - stop_pct),
            })
            last_signal_idx = i

        current = test_end

    logger.info(f"Walk-forward: {models_trained} retrains, {len(all_signals)} signals")
    return all_signals


def backtest(daily, signals):
    """Custom backtest with ATR stops, trailing stops, time exits."""
    if not signals:
        return None

    # Build price lookup
    prices = daily.set_index("date")[["open", "high", "low", "close"]].copy()

    trades = []
    open_positions = []

    for sig in signals:
        sig_date = sig["date"]
        # Entry on next trading day's open
        future = prices.index[prices.index > sig_date]
        if len(future) == 0:
            continue
        entry_date = future[0]
        entry_price = float(prices.loc[entry_date, "open"])
        stop_price = entry_price * (1 + sig["stop_loss_pct"] / 100)
        max_price = entry_price

        # Check position limit
        active = [p for p in open_positions if p["exit_date"] is None]
        if len(active) >= MAX_POSITIONS:
            continue

        pos = {
            "entry_date": entry_date,
            "entry_price": entry_price,
            "stop_price": stop_price,
            "max_price": max_price,
            "trailing_active": False,
            "exit_date": None,
            "exit_price": None,
            "exit_reason": None,
            "confidence": sig["confidence"],
        }
        open_positions.append(pos)

    # Simulate day by day
    all_dates = sorted(prices.index)
    for date in all_dates:
        for pos in open_positions:
            if pos["exit_date"] is not None:
                continue
            if date <= pos["entry_date"]:
                continue

            day_high = float(prices.loc[date, "high"])
            day_low = float(prices.loc[date, "low"])
            day_close = float(prices.loc[date, "close"])
            days_held = (date - pos["entry_date"]).days

            # Update max price for trailing stop
            if day_high > pos["max_price"]:
                pos["max_price"] = day_high

            # Trailing stop: activate after +8%, trail at -6% from high
            unrealized = (pos["max_price"] - pos["entry_price"]) / pos["entry_price"]
            if unrealized >= 0.08:
                pos["trailing_active"] = True
                new_trail = pos["max_price"] * 0.94  # 6% from high
                pos["stop_price"] = max(pos["stop_price"], new_trail)

            # Check stop-loss
            if day_low <= pos["stop_price"]:
                pos["exit_date"] = date
                pos["exit_price"] = pos["stop_price"]
                pos["exit_reason"] = "trailing_stop" if pos["trailing_active"] else "stop_loss"
                continue

            # Stagnation exit: flat after 21 days
            pnl_pct = (day_close - pos["entry_price"]) / pos["entry_price"]
            if days_held >= STAGNATION_DAYS and abs(pnl_pct) < STAGNATION_BAND:
                pos["exit_date"] = date
                pos["exit_price"] = day_close
                pos["exit_reason"] = "stagnation"
                continue

            # Max hold exit
            if days_held >= MAX_HOLD_DAYS:
                pos["exit_date"] = date
                pos["exit_price"] = day_close
                pos["exit_reason"] = "max_hold"
                continue

    # Close any remaining positions at last price
    last_date = all_dates[-1]
    last_close = float(prices.loc[last_date, "close"])
    for pos in open_positions:
        if pos["exit_date"] is None:
            pos["exit_date"] = last_date
            pos["exit_price"] = last_close
            pos["exit_reason"] = "end_of_data"

    # Build trade records
    for pos in open_positions:
        pnl_pct = (pos["exit_price"] - pos["entry_price"]) / pos["entry_price"] * 100
        pnl_idr = int(1_000_000 * pnl_pct / 100)
        hold_days = (pos["exit_date"] - pos["entry_date"]).days
        trades.append({
            "entry_date": pos["entry_date"],
            "exit_date": pos["exit_date"],
            "entry_price": pos["entry_price"],
            "exit_price": pos["exit_price"],
            "pnl_pct": pnl_pct,
            "pnl_idr": pnl_idr,
            "hold_days": hold_days,
            "exit_reason": pos["exit_reason"],
            "confidence": pos["confidence"],
        })

    return trades


def print_results(trades, daily):
    """Print comprehensive results."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS — Expert-Informed v3")
    print(f"Target: P(BTC up >{TARGET_RETURN_PCT*100:.0f}% in {TARGET_HORIZON}d)")
    print(f"Gates: 50-SMA trend + ATR vol + ROC20 momentum + {CONFIDENCE_GATE:.0%} confidence")
    print(f"Stops: {ATR_MULTIPLIER}x ATR (floor {ATR_STOP_FLOOR:.0%}, ceiling {ATR_STOP_CEILING:.0%})")
    print(f"Trailing: activate at +8%, trail at -6% from high")
    print("=" * 70)

    if not trades:
        print("\nNo trades executed.")
        return

    total_pnl = sum(t["pnl_idr"] for t in trades)
    total_invested = len(trades) * 1_000_000
    total_ret = total_pnl / total_invested * 100 if total_invested > 0 else 0

    winners = [t for t in trades if t["pnl_pct"] > 0]
    losers = [t for t in trades if t["pnl_pct"] <= 0]
    win_rate = len(winners) / len(trades) if trades else 0

    avg_win = np.mean([t["pnl_pct"] for t in winners]) if winners else 0
    avg_loss = np.mean([t["pnl_pct"] for t in losers]) if losers else 0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # Sharpe (trade-level)
    rets = [t["pnl_pct"] for t in trades]
    sharpe = np.mean(rets) / np.std(rets) * np.sqrt(26) if len(rets) > 1 and np.std(rets) > 0 else 0

    # Max drawdown (cumulative P&L)
    cum_pnl = np.cumsum([t["pnl_idr"] for t in trades])
    peak = np.maximum.accumulate(cum_pnl)
    dd = np.where(peak > 0, (cum_pnl - peak) / peak * 100, 0)
    max_dd = float(np.min(dd)) if len(dd) > 0 else 0

    print(f"\n{'Total Trades:':<30} {len(trades)}")
    print(f"{'Winners / Losers:':<30} {len(winners)} / {len(losers)}")
    print(f"{'Win Rate:':<30} {win_rate:.1%}")
    print(f"{'Total Return (IDR):':<30} Rp {total_pnl:+,.0f}")
    print(f"{'Total Return (%):':<30} {total_ret:+.2f}%")
    print(f"{'Avg Win:':<30} {avg_win:+.2f}%")
    print(f"{'Avg Loss:':<30} {avg_loss:+.2f}%")
    print(f"{'Reward/Risk Ratio:':<30} {rr_ratio:.2f}")
    print(f"{'Sharpe (annualized):':<30} {sharpe:.2f}")
    print(f"{'Max Drawdown:':<30} {max_dd:.2f}%")
    avg_hold = np.mean([t["hold_days"] for t in trades])
    print(f"{'Avg Hold Days:':<30} {avg_hold:.1f}")

    # By exit reason
    reasons = {}
    for t in trades:
        r = t["exit_reason"]
        reasons.setdefault(r, {"count": 0, "pnl": 0})
        reasons[r]["count"] += 1
        reasons[r]["pnl"] += t["pnl_idr"]
    print(f"\nEXIT REASONS:")
    for r, v in sorted(reasons.items()):
        print(f"  {r:<20} {v['count']:>3} trades, Rp {v['pnl']:>+12,.0f}")

    # Monthly
    monthly = {}
    for t in trades:
        m = t["exit_date"].strftime("%Y-%m")
        monthly.setdefault(m, {"pnl": 0, "trades": 0})
        monthly[m]["pnl"] += t["pnl_idr"]
        monthly[m]["trades"] += 1

    if monthly:
        print(f"\nMONTHLY:")
        print(f"{'Month':<12} {'P&L (IDR)':>15} {'Trades':>8}")
        print("-" * 38)
        for m in sorted(monthly):
            pnl = monthly[m]["pnl"]
            print(f"{m:<12} Rp {pnl:>+12,.0f} {monthly[m]['trades']:>8}")

    # Trade log
    print(f"\nTRADE LOG:")
    print(f"{'Entry':<12} {'Exit':<12} {'Entry$':>10} {'Exit$':>10} {'P&L%':>8} {'Days':>5} {'Conf':>6} {'Reason':<15}")
    print("-" * 82)
    for t in trades:
        ed = t["entry_date"].strftime("%Y-%m-%d")
        xd = t["exit_date"].strftime("%Y-%m-%d")
        print(f"{ed:<12} {xd:<12} ${t['entry_price']:>9,.0f} ${t['exit_price']:>9,.0f} "
              f"{t['pnl_pct']:>+7.2f}% {t['hold_days']:>5} {t['confidence']:>5.0%} {t['exit_reason']:<15}")

    # Compare to SMA200 baseline
    print(f"\n{'BENCHMARK COMPARISON:'}")
    print(f"  SMA200 baseline: +14.2% return, -31.0% max DD")
    print(f"  Buy & Hold:      +17.1% return, -49.5% max DD")
    print(f"  This strategy:   {total_ret:+.1f}% return, {max_dd:.1f}% max DD")

    print("\n" + "=" * 70)


def main():
    daily = load_data()
    df, feature_cols = build_features(daily)
    target, valid = build_target(df)

    signals = walk_forward(df, feature_cols, target, valid)

    if not signals:
        logger.warning("No signals generated.")
        return

    logger.info("Running backtest...")
    trades = backtest(daily, signals)

    if trades:
        print_results(trades, daily)
    else:
        logger.warning("No trades executed.")


if __name__ == "__main__":
    main()
