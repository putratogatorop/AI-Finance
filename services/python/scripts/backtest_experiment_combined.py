# scripts/backtest_experiment_combined.py
"""Combined Experiment: Volatility Filter + Adaptive ATR Stops + ML Filter.

Stacks three independently-validated improvements:
  1. Volatility filter:   only trade coins with ATR 0.3%–2.0%
  2. Adaptive ATR stops:  all levels in ATR multiples (tiered SL, widening trail)
  3. ML filter:           LightGBM walk-forward to separate real vs fake breakouts

Walk-forward: train 18 months, test 6 months.
Reports per-window AND aggregate OOS at thresholds 0.50 / 0.55 / 0.60 / 0.65 / 0.70.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

from scripts.backtest_momentum_scanner import (
    compute_atr,
    detect_breakouts,
    load_coin,
)
from scripts.backtest_scanner_ml_filter import (
    FEATURE_COLS,
    build_walk_forward_splits,
    extract_features,
)
from scripts.backtest_experiment_adaptive import simulate_adaptive_trade

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("LightGBM not installed. Run: pip install lightgbm")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_DIR = Path("C:/Users/togat/Desktop/AI-Finance/data/raw/15m")
MIN_VOLUME_USD = 1_000_000
MIN_BARS = 200

# ── Scanner params ─────────────────────────────────────────────────────────────
VOL_MULT = 4.0
PRICE_THRESH = 0.035
LOOKBACK = 20
MAX_DAILY_MOVE = 0.30
BTC_TREND_PERIOD = 30  # span for EMA trend filter

# ── Volatility filter ──────────────────────────────────────────────────────────
VOL_FILTER_LOW = 0.003   # ATR < 0.3% → skip (ultra-low-vol, no edge)
VOL_FILTER_HIGH = 0.02   # ATR > 2.0% → skip (high-vol, erratic stops)

# ── Trade params ───────────────────────────────────────────────────────────────
MAX_HOLD_BARS = 192   # 48h safety net
MAX_LEVERAGE = 3.0

# ── ML params ──────────────────────────────────────────────────────────────────
ML_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]
TRAIN_MONTHS = 18
TEST_MONTHS = 6

ML_PARAMS = {
    "objective": "binary",
    "num_leaves": 23,
    "min_child_samples": 25,
    "learning_rate": 0.03,
    "n_estimators": 400,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 3,
    "verbose": -1,
    "random_state": 42,
}


# ── Metrics helper ─────────────────────────────────────────────────────────────

def compute_metrics(trades: pd.DataFrame) -> dict:
    if len(trades) == 0:
        return {"n": 0, "wr": 0, "pf": 0, "exp": 0, "avg_win": 0, "avg_loss": 0}
    pnls = trades["pnl_pct"].values
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = len(wins) / len(pnls)
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(abs(np.mean(losses))) if len(losses) > 0 else 0.0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float("inf")
    exp = float(np.mean(pnls))
    return {"n": len(trades), "wr": wr, "pf": pf, "exp": exp,
            "avg_win": avg_win, "avg_loss": avg_loss}


def fmt(m: dict) -> str:
    if m["n"] == 0:
        return "trades=0 (no data)"
    pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "inf"
    return (f"trades={m['n']} WR={m['wr']:.1%} PF={pf_str} exp={m['exp']:+.2%}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    logger.info("Combined Experiment: Vol-Filter + Adaptive-ATR + ML-Filter")
    logger.info(f"  Vol filter:  ATR [{VOL_FILTER_LOW:.1%} – {VOL_FILTER_HIGH:.1%}]")
    logger.info("  Adaptive ATR: Tier1=-2xATR(50%), Tier2=-3xATR(100%) | "
                "Trail: +3x->2xATR, +6x->3xATR, +10x->4xATR | Partial 33% @+5xATR")
    logger.info(f"  ML walk-forward: train={TRAIN_MONTHS}mo test={TEST_MONTHS}mo "
                f"thresholds={ML_THRESHOLDS}")

    # ── Load BTC ───────────────────────────────────────────────────────────────
    btc_df = load_coin(RAW_DIR / "BTCUSDT")
    if btc_df is None:
        logger.error("Cannot load BTC data!")
        return
    btc_close = btc_df["close"].values.astype(float)
    btc_volume = btc_df["volume"].values.astype(float)
    btc_ema = pd.Series(btc_close).ewm(span=BTC_TREND_PERIOD).mean().values
    btc_times = btc_df["open_time"].values
    logger.info(f"BTC loaded: {len(btc_df)} bars")

    # ── Scan all coins ─────────────────────────────────────────────────────────
    symbol_dirs = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()])
    logger.info(f"Scanning {len(symbol_dirs)} coins...")

    all_signals: list[dict] = []
    n_skipped_low_vol = 0
    n_skipped_high_vol = 0
    coins_with_signals = 0

    for sym_dir in symbol_dirs:
        symbol = sym_dir.name
        df = load_coin(sym_dir)
        if df is None or len(df) < MIN_BARS:
            continue

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)
        times = df["open_time"].values

        # Liquidity filter
        avg_vol = np.nanmean(volume[-100:]) if len(volume) >= 100 else np.nanmean(volume)
        avg_price = np.nanmean(close[-100:]) if len(close) >= 100 else np.nanmean(close)
        if avg_vol * avg_price < MIN_VOLUME_USD / 96:
            continue

        # Detect breakouts
        signals = detect_breakouts(df, vol_mult=VOL_MULT,
                                   price_thresh=PRICE_THRESH, lookback=LOOKBACK)
        if not signals:
            continue

        coins_with_signals += 1

        # Pre-compute ATR arrays once per coin
        atr_14 = compute_atr(high, low, close, period=14)
        atr_96 = compute_atr(high, low, close, period=96)

        for sig in signals:
            bar = sig["bar"]
            atr_val = atr_14[bar] if bar < len(atr_14) else np.nan
            if np.isnan(atr_val) or atr_val == 0:
                continue

            # ── 1. Volatility filter ───────────────────────────────────────────
            if atr_val < VOL_FILTER_LOW:
                n_skipped_low_vol += 1
                continue
            if atr_val > VOL_FILTER_HIGH:
                n_skipped_high_vol += 1
                continue

            # ── BTC trend filter ───────────────────────────────────────────────
            sig_time = times[bar]
            btc_idx = int(np.searchsorted(btc_times, sig_time))
            btc_idx = min(btc_idx, len(btc_close) - 1)
            if sig["direction"] == 1 and btc_close[btc_idx] < btc_ema[btc_idx]:
                continue
            if sig["direction"] == -1 and btc_close[btc_idx] > btc_ema[btc_idx]:
                continue

            # ── 2. Adaptive ATR trade simulation ──────────────────────────────
            trade = simulate_adaptive_trade(
                close=close, high=high, low=low, volume=volume,
                signal_bar=bar, direction=sig["direction"],
                atr_value=atr_val, max_hold=MAX_HOLD_BARS,
            )
            if trade is None:
                continue

            # ── 3. Extract ML features ────────────────────────────────────────
            feats = extract_features(
                df=df,
                bar=bar,
                btc_close=btc_close,
                btc_volume=btc_volume,
                btc_times=btc_times,
                sig_vol_ratio=sig["volume_ratio"],
                sig_price_change=sig["price_change"],
                atr_14=atr_14,
                atr_96=atr_96,
            )
            if feats is None:
                continue

            label = 1 if trade["pnl_pct"] > 0 else 0

            record = {
                **feats,
                "label": label,
                "pnl_pct": trade["pnl_pct"],
                "exit_reason": trade["exit_reason"],
                "symbol": symbol,
                "signal_time": str(times[bar]),
                "direction": sig["direction"],
                "atr_value": atr_val,
            }
            all_signals.append(record)

    logger.info(
        f"Scan done in {time.time()-t0:.0f}s | "
        f"coins_with_signals={coins_with_signals} | "
        f"signals_total={len(all_signals)} | "
        f"skipped_low_vol={n_skipped_low_vol} skipped_high_vol={n_skipped_high_vol}"
    )

    if not all_signals:
        logger.error("No signals collected after vol filter!")
        return

    # ── Build DataFrame ────────────────────────────────────────────────────────
    signals_df = pd.DataFrame(all_signals)
    signals_df["signal_time"] = pd.to_datetime(signals_df["signal_time"])
    signals_df = signals_df.sort_values("signal_time").reset_index(drop=True)

    # Fill num_concurrent: count signals in the same 15min window
    signals_df["time_bucket"] = signals_df["signal_time"].dt.floor("15min")
    concurrent_counts = signals_df.groupby("time_bucket").size()
    signals_df["num_concurrent"] = signals_df["time_bucket"].map(concurrent_counts)
    signals_df = signals_df.drop(columns=["time_bucket"])

    # Sanitise features
    signals_df[FEATURE_COLS] = (
        signals_df[FEATURE_COLS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
    )

    label_dist = signals_df["label"].value_counts()
    logger.info(
        f"Labels: {label_dist.get(1, 0)} winners / {label_dist.get(0, 0)} losers | "
        f"WR={label_dist.get(1, 0)/len(signals_df):.1%}"
    )

    # ── Walk-forward ML ────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("WALK-FORWARD ML FILTER  (combined: vol-filter + adaptive ATR + ML)")
    print("=" * 68)

    splits = build_walk_forward_splits(signals_df, TRAIN_MONTHS, TEST_MONTHS)
    logger.info(f"Walk-forward windows: {len(splits)}")

    if not splits:
        logger.error("Not enough data for walk-forward splits!")
        return

    oos_chunks: list[pd.DataFrame] = []   # test data with ml_proba attached

    for i, (train_df, test_df) in enumerate(splits):
        t_start = pd.to_datetime(train_df["signal_time"]).min().date()
        t_end   = pd.to_datetime(train_df["signal_time"]).max().date()
        v_start = pd.to_datetime(test_df["signal_time"]).min().date()
        v_end   = pd.to_datetime(test_df["signal_time"]).max().date()

        print(f"\nWindow {i+1}: train={t_start}->{t_end} ({len(train_df)}) "
              f"test={v_start}->{v_end} ({len(test_df)})")

        X_train = train_df[FEATURE_COLS].values
        y_train = train_df["label"].values
        X_test  = test_df[FEATURE_COLS].values

        n_pos = int(y_train.sum())
        n_neg = int(len(y_train) - n_pos)

        if n_pos < 10 or n_neg < 10:
            logger.warning("  Insufficient class balance, skipping window")
            # Still add test data without proba so OOS unfiltered is complete
            test_df = test_df.copy()
            test_df["ml_proba"] = np.nan
            oos_chunks.append(test_df)
            continue

        model = LGBMClassifier(**ML_PARAMS)
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]

        test_df = test_df.copy()
        test_df["ml_proba"] = proba

        # Unfiltered window metrics
        m_uf = compute_metrics(test_df)
        print(f"  Unfiltered:  {fmt(m_uf)}")

        # Per-threshold
        for thresh in ML_THRESHOLDS:
            filt = test_df[proba > thresh]
            m = compute_metrics(filt)
            pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "inf"
            print(f"  @{thresh:.2f}:       trades={m['n']} WR={m['wr']:.1%} "
                  f"PF={pf_str} exp={m['exp']:+.2%}")

        # Feature importance (top 5)
        fi = dict(zip(FEATURE_COLS, model.feature_importances_))
        top5 = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("  Top features: " + ", ".join(f"{k}={v:.0f}" for k, v in top5))

        oos_chunks.append(test_df)

    # ── Aggregate OOS ──────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("AGGREGATE OOS:")
    print("=" * 68)

    all_oos = pd.concat(oos_chunks, ignore_index=True)
    # Only rows where we have proba (skipped windows have NaN)
    scored_oos = all_oos.dropna(subset=["ml_proba"])

    if len(scored_oos) == 0:
        logger.error("No scored OOS trades!")
        return

    # Time span for monthly rate calculation
    oos_days = (
        pd.to_datetime(scored_oos["signal_time"]).max()
        - pd.to_datetime(scored_oos["signal_time"]).min()
    ).days
    oos_months = max(oos_days / 30, 1)

    for thresh in ML_THRESHOLDS:
        filt = scored_oos[scored_oos["ml_proba"] > thresh]
        m = compute_metrics(filt)
        if m["n"] == 0:
            print(f"  @{thresh:.2f}: trades=0")
            continue
        tpm = m["n"] / oos_months
        monthly_ret = m["exp"] * tpm * MAX_LEVERAGE
        pf_str = f"{m['pf']:.2f}" if m["pf"] != float("inf") else "inf"
        print(
            f"  @{thresh:.2f}: trades={m['n']:4d}  WR={m['wr']:.1%}  "
            f"PF={pf_str}  exp={m['exp']:+.2%}  "
            f"avg_win={m['avg_win']:.2%}  avg_loss={m['avg_loss']:.2%}  "
            f"~{monthly_ret:.1%}/mo(3x)"
        )

    # ── Last 10 trades @ 0.65 ──────────────────────────────────────────────────
    thresh_last = 0.65
    filt_65 = scored_oos[scored_oos["ml_proba"] > thresh_last].sort_values("signal_time")
    print(f"\nLAST 10 TRADES @ {thresh_last}:")
    last_10 = filt_65.tail(10)
    for _, t in last_10.iterrows():
        direction = "LONG" if t["direction"] == 1 else "SHORT"
        result = "WIN " if t["pnl_pct"] > 0 else "LOSS"
        ts = str(t["signal_time"])[:19]
        print(
            f"  {ts} | {t['symbol']:>14s} | {direction} | "
            f"pnl={t['pnl_pct']:+6.2%} | {t['exit_reason']:>15s} | "
            f"ml={t['ml_proba']:.3f} | atr={t['atr_value']:.3%} | {result}"
        )

    # ── Monthly breakdown for OOS @ 0.60 ──────────────────────────────────────
    filt_60 = scored_oos[scored_oos["ml_proba"] > 0.60].copy()
    if len(filt_60) > 0:
        filt_60["month"] = pd.to_datetime(filt_60["signal_time"]).dt.to_period("M")
        monthly = filt_60.groupby("month")["pnl_pct"].agg(["sum", "count", "mean"])
        monthly["leveraged"] = monthly["sum"] * MAX_LEVERAGE
        print(f"\nOOS @0.60 monthly returns (3x leverage):")
        for month, row in monthly.iterrows():
            print(f"  {month}:  {row['leveraged']:+.1%}  ({int(row['count'])} trades)")
        profitable = (monthly["leveraged"] > 0).sum()
        print(f"  Profitable months: {profitable}/{len(monthly)} ({profitable/len(monthly):.0%})")

    # ── Exit reason breakdown (unfiltered OOS) ─────────────────────────────────
    print(f"\nOOS unfiltered exit reasons:")
    for reason, count in scored_oos["exit_reason"].value_counts().items():
        print(f"  {reason}: {count} ({count/len(scored_oos):.1%})")

    print(f"\nTotal elapsed: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
