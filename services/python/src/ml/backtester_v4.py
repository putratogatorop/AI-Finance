"""V4 Vectorized Backtester.

Simulates trading LightGBM OOS predictions with realistic costs:
- Entry/exit fees (maker 0.015%, round-trip ~0.03%)
- Funding costs (~0.01% per 8h on open positions)
- Stop-loss using intra-bar high/low
- Take-profit and time-based exits
- No look-ahead bias: decisions use only current-bar data
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 56) -> np.ndarray:
    """Compute ATR over `period` bars. ATR(56) on 15min = ~14h of volatility."""
    n = len(close)
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i],
                     abs(high[i] - close[i - 1]),
                     abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        for i in range(period, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def backtest_coin(
    df: pd.DataFrame,
    conviction_threshold: float = 0.20,
    initial_capital: float = 10_000.0,
    leverage: float = 3.0,
    fee_rate: float = 0.0003,
    funding_rate_8h: float = 0.0001,
    stop_loss_pct: float = 0.0,        # 0 = use ATR-based stop
    take_profit_pct: float = 0.0,      # 0 = use ATR-based TP
    atr_stop_mult: float = 2.5,        # stop = entry +/- atr_stop_mult * ATR
    atr_tp_mult: float = 5.0,          # TP = entry +/- atr_tp_mult * ATR
    max_hold_bars: int = 192,           # 48h at 15min
    risk_per_trade: float = 0.01,       # 1% of equity per trade
) -> dict:
    """Run backtest on a single coin's OOS predictions.

    Changes from v1:
    - ATR-based stops/TP (default) instead of fixed percentage
    - Fixed position sizing: risk / (stop_distance * leverage) — not * leverage
    - Lower risk per trade (1% default)
    - Longer max hold (48h)
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    conviction = df["conviction"].values.astype(float)
    n = len(df)

    # Precompute ATR for dynamic stops
    atr = _compute_atr(high, low, close, period=56)

    equity = initial_capital
    trades = []
    equity_curve = [equity]

    in_position = False
    entry_price = 0.0
    entry_bar = 0
    direction = 0
    position_size = 0.0
    stop_price = 0.0
    tp_price = 0.0

    for i in range(1, n):
        if in_position:
            bars_held = i - entry_bar

            # Check stops
            if direction == 1:
                hit_stop = low[i] <= stop_price
                hit_tp = high[i] >= tp_price
            else:
                hit_stop = high[i] >= stop_price
                hit_tp = low[i] <= tp_price

            time_exit = bars_held >= max_hold_bars

            exit_price_val = None
            exit_reason = None

            if hit_stop:
                exit_price_val = stop_price
                exit_reason = "stop_loss"
            elif hit_tp:
                exit_price_val = tp_price
                exit_reason = "take_profit"
            elif time_exit:
                exit_price_val = close[i]
                exit_reason = "time_exit"

            # Signal flip exit
            if exit_reason is None:
                if direction == 1 and conviction[i] < -conviction_threshold:
                    exit_price_val = close[i]
                    exit_reason = "signal_flip"
                elif direction == -1 and conviction[i] > conviction_threshold:
                    exit_price_val = close[i]
                    exit_reason = "signal_flip"

            if exit_price_val is not None:
                if direction == 1:
                    raw_return = (exit_price_val - entry_price) / entry_price
                else:
                    raw_return = (entry_price - exit_price_val) / entry_price

                fee_cost = fee_rate * position_size
                funding_periods = bars_held / 32
                funding_cost = funding_rate_8h * position_size * funding_periods

                gross_pnl = raw_return * position_size
                net_pnl = gross_pnl - fee_cost - funding_cost
                equity = max(equity + net_pnl, 1.0)  # Floor at $1 to prevent negative

                trades.append({
                    "entry_bar": entry_bar, "exit_bar": i,
                    "direction": direction, "entry_price": entry_price,
                    "exit_price": exit_price_val, "hold_bars": bars_held,
                    "raw_return": raw_return,
                    "pnl_pct": net_pnl / position_size if position_size > 0 else 0,
                    "pnl_usd": net_pnl, "fee_paid": fee_cost,
                    "funding_paid": funding_cost, "exit_reason": exit_reason,
                    "position_size": position_size,
                })
                in_position = False

        # Entry logic
        if not in_position and not np.isnan(atr[i]):
            if abs(conviction[i]) >= conviction_threshold:
                direction = 1 if conviction[i] > 0 else -1
                entry_price = close[i]
                entry_bar = i

                # ATR-based or fixed stops
                if stop_loss_pct > 0:
                    stop_dist = entry_price * stop_loss_pct
                else:
                    stop_dist = atr[i] * atr_stop_mult

                if take_profit_pct > 0:
                    tp_dist = entry_price * take_profit_pct
                else:
                    tp_dist = atr[i] * atr_tp_mult

                if direction == 1:
                    stop_price = entry_price - stop_dist
                    tp_price = entry_price + tp_dist
                else:
                    stop_price = entry_price + stop_dist
                    tp_price = entry_price - tp_dist

                # FIXED: position_size = risk / (stop_distance * leverage)
                stop_pct = stop_dist / entry_price
                position_size = (equity * risk_per_trade) / stop_pct
                position_size = min(position_size, equity * leverage)
                in_position = True

        equity_curve.append(equity)

    # Close any open position at end
    if in_position:
        exit_price_val = close[-1]
        if direction == 1:
            raw_return = (exit_price_val - entry_price) / entry_price
        else:
            raw_return = (entry_price - exit_price_val) / entry_price
        bars_held = n - 1 - entry_bar
        fee_cost = fee_rate * position_size
        funding_cost = funding_rate_8h * position_size * (bars_held / 32)
        net_pnl = raw_return * position_size - fee_cost - funding_cost
        equity = max(equity + net_pnl, 1.0)
        trades.append({
            "entry_bar": entry_bar, "exit_bar": n - 1,
            "direction": direction, "entry_price": entry_price,
            "exit_price": exit_price_val, "hold_bars": bars_held,
            "raw_return": raw_return,
            "pnl_pct": net_pnl / position_size if position_size > 0 else 0,
            "pnl_usd": net_pnl, "fee_paid": fee_cost,
            "funding_paid": funding_cost, "exit_reason": "end_of_data",
            "position_size": position_size,
        })
        equity_curve[-1] = equity

    eq_series = pd.Series(equity_curve)
    metrics = compute_metrics(trades, eq_series) if trades else _empty_metrics()
    return {"trades": trades, "equity_curve": eq_series, "metrics": metrics}


def compute_metrics(trades: list[dict], equity_curve: pd.Series) -> dict:
    n_trades = len(trades)
    if n_trades == 0:
        return _empty_metrics()

    pnls = [t["pnl_pct"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    win_rate = len(wins) / n_trades
    profit_factor = (sum(wins) / abs(sum(losses))) if losses and sum(losses) != 0 else float("inf")

    eq_returns = equity_curve.pct_change().dropna()
    if len(eq_returns) > 1 and eq_returns.std() > 0:
        sharpe = eq_returns.mean() / eq_returns.std() * np.sqrt(96 * 365)
    else:
        sharpe = 0.0

    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    max_dd = abs(drawdown.min())

    hold_bars = [t["hold_bars"] for t in trades]
    avg_hold_hours = np.mean(hold_bars) * 0.25

    total_fees = sum(t["fee_paid"] + t.get("funding_paid", 0) for t in trades)
    gross_pnl = sum(t["raw_return"] * t["position_size"] for t in trades)
    fee_drag = total_fees / gross_pnl if gross_pnl > 0 else 0

    total_return_pct = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1

    eq_monthly = equity_curve.iloc[::2880]
    monthly_returns = eq_monthly.pct_change().dropna()
    profitable_months_pct = (monthly_returns > 0).mean() if len(monthly_returns) > 0 else 0

    return {
        "total_trades": n_trades,
        "total_return_pct": float(total_return_pct),
        "win_rate": float(win_rate),
        "avg_win_pct": float(np.mean(wins)) if wins else 0.0,
        "avg_loss_pct": float(abs(np.mean(losses))) if losses else 0.0,
        "profit_factor": float(profit_factor),
        "sharpe_ratio": float(sharpe),
        "max_drawdown_pct": float(max_dd),
        "avg_hold_hours": float(avg_hold_hours),
        "total_fees": float(total_fees),
        "fee_drag_pct": float(fee_drag),
        "profitable_months_pct": float(profitable_months_pct),
    }


def _empty_metrics() -> dict:
    return {
        "total_trades": 0, "total_return_pct": 0.0, "win_rate": 0.0,
        "avg_win_pct": 0.0, "avg_loss_pct": 0.0, "profit_factor": 0.0,
        "sharpe_ratio": 0.0, "max_drawdown_pct": 0.0, "avg_hold_hours": 0.0,
        "total_fees": 0.0, "fee_drag_pct": 0.0, "profitable_months_pct": 0.0,
    }
