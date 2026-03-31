"""Backtesting engine for validating the trading system against historical data.

Simulates trades exactly as the live system would: signals, entries, exits,
stop-losses, and position limits. Uses no lookahead bias -- the engine only
sees data available at the time of each decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class BacktestConfig:
    """Configuration for a backtest run."""

    position_size_idr: int = 1_000_000
    stop_loss_pct: float = -8.0
    max_positions: int = 8
    max_single_asset_pct: float = 25.0
    confidence_threshold: float = 0.7
    total_capital_idr: int = 10_000_000


@dataclass
class BacktestTrade:
    """Record of a single completed trade in the backtest."""

    asset: str
    signal_date: datetime
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    entry_amount_idr: int
    exit_amount_idr: int
    pnl_idr: int
    pnl_pct: float
    hold_days: int
    exit_reason: str  # "target_hold", "stop_loss", "signal_exit"


@dataclass
class BacktestResult:
    """Complete results of a backtest run."""

    total_return_idr: int
    total_return_pct: float
    win_rate: float
    reward_risk_ratio: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_trades: int
    max_concurrent_positions: int
    trades: list[BacktestTrade]
    monthly_breakdown: list[dict[str, Any]]
    per_asset_breakdown: list[dict[str, Any]]


@dataclass
class _OpenPosition:
    """Internal tracking of an open position during backtesting."""

    asset: str
    entry_date: datetime
    entry_price: float
    entry_amount_idr: int
    quantity: float
    stop_loss_price: float
    target_exit_date: datetime
    signal_date: datetime


class BacktestEngine:
    """Simulates the trading system against historical data."""

    def __init__(self, config: BacktestConfig | None = None) -> None:
        self._config = config or BacktestConfig()

    def run(
        self,
        price_data: pd.DataFrame,
        signals: pd.DataFrame,
    ) -> BacktestResult:
        """Run the backtest simulation."""
        if signals.empty:
            return BacktestResult(
                total_return_idr=0,
                total_return_pct=0.0,
                win_rate=0.0,
                reward_risk_ratio=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                max_concurrent_positions=0,
                trades=[],
                monthly_breakdown=[],
                per_asset_breakdown=[],
            )

        price_data = price_data.copy()
        price_data["date"] = pd.to_datetime(price_data["date"])
        signals = signals.copy()
        signals["date"] = pd.to_datetime(signals["date"])

        asset_prices: dict[str, pd.DataFrame] = {}
        for asset in price_data["asset"].unique():
            asset_df = (
                price_data[price_data["asset"] == asset]
                .sort_values("date")
                .set_index("date")
            )
            asset_prices[asset] = asset_df

        all_dates = sorted(price_data["date"].unique())

        open_positions: list[_OpenPosition] = []
        completed_trades: list[BacktestTrade] = []
        max_concurrent = 0
        portfolio_value_history: list[float] = []

        signals_by_date: dict[Any, pd.DataFrame] = {}
        for date, group in signals.groupby("date"):
            signals_by_date[date] = group

        for date in all_dates:
            # 1. Check stop-losses and target exits
            positions_to_close: list[tuple[int, str]] = []
            for i, pos in enumerate(open_positions):
                if pos.asset not in asset_prices:
                    continue
                if date not in asset_prices[pos.asset].index:
                    continue

                current_low = asset_prices[pos.asset].loc[date, "low"]
                current_close = asset_prices[pos.asset].loc[date, "close"]

                if isinstance(current_low, pd.Series):
                    current_low = current_low.iloc[0]
                if isinstance(current_close, pd.Series):
                    current_close = current_close.iloc[0]

                if current_low <= pos.stop_loss_price:
                    positions_to_close.append((i, "stop_loss"))
                    continue

                if date >= pos.target_exit_date:
                    positions_to_close.append((i, "target_hold"))
                    continue

            # Close positions (reverse order to maintain indices)
            for idx, reason in sorted(positions_to_close, reverse=True):
                pos = open_positions.pop(idx)
                if (
                    pos.asset in asset_prices
                    and date in asset_prices[pos.asset].index
                ):
                    exit_price_raw = asset_prices[pos.asset].loc[date, "close"]
                    if isinstance(exit_price_raw, pd.Series):
                        exit_price_raw = exit_price_raw.iloc[0]

                    if reason == "stop_loss":
                        exit_price = float(pos.stop_loss_price)
                    else:
                        exit_price = float(exit_price_raw)

                    exit_amount = int(pos.quantity * exit_price)
                    pnl_idr = exit_amount - pos.entry_amount_idr
                    pnl_pct = (
                        (exit_price - pos.entry_price) / pos.entry_price
                    ) * 100

                    completed_trades.append(
                        BacktestTrade(
                            asset=pos.asset,
                            signal_date=pos.signal_date,
                            entry_date=pos.entry_date,
                            exit_date=date,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            entry_amount_idr=pos.entry_amount_idr,
                            exit_amount_idr=exit_amount,
                            pnl_idr=pnl_idr,
                            pnl_pct=pnl_pct,
                            hold_days=(date - pos.entry_date).days,
                            exit_reason=reason,
                        )
                    )

            # 2. Process new signals
            if date in signals_by_date:
                for _, sig in signals_by_date[date].iterrows():
                    if sig["action"] != "BUY":
                        continue
                    if sig["confidence"] < self._config.confidence_threshold:
                        continue

                    if len(open_positions) >= self._config.max_positions:
                        continue

                    asset_exposure = sum(
                        p.entry_amount_idr
                        for p in open_positions
                        if p.asset == sig["asset"]
                    )
                    max_allowed = self._config.total_capital_idr * (
                        self._config.max_single_asset_pct / 100
                    )
                    if (
                        asset_exposure + self._config.position_size_idr
                        > max_allowed
                    ):
                        continue

                    asset = sig["asset"]
                    if asset not in asset_prices:
                        continue

                    future_dates = asset_prices[asset].index[
                        asset_prices[asset].index > date
                    ]
                    if len(future_dates) == 0:
                        continue

                    entry_date = future_dates[0]
                    entry_price_raw = asset_prices[asset].loc[
                        entry_date, "open"
                    ]
                    if isinstance(entry_price_raw, pd.Series):
                        entry_price_raw = entry_price_raw.iloc[0]
                    entry_price = float(entry_price_raw)

                    quantity = self._config.position_size_idr / entry_price
                    stop_loss_price = entry_price * (
                        1 + self._config.stop_loss_pct / 100
                    )
                    hold_days = int(sig.get("suggested_hold_days", 14))
                    target_exit = entry_date + pd.Timedelta(days=hold_days)

                    open_positions.append(
                        _OpenPosition(
                            asset=asset,
                            entry_date=entry_date,
                            entry_price=entry_price,
                            entry_amount_idr=self._config.position_size_idr,
                            quantity=quantity,
                            stop_loss_price=stop_loss_price,
                            target_exit_date=target_exit,
                            signal_date=date,
                        )
                    )

            max_concurrent = max(max_concurrent, len(open_positions))

            total_value = float(self._config.total_capital_idr)
            for pos in open_positions:
                if (
                    pos.asset in asset_prices
                    and date in asset_prices[pos.asset].index
                ):
                    cp = asset_prices[pos.asset].loc[date, "close"]
                    if isinstance(cp, pd.Series):
                        cp = cp.iloc[0]
                    total_value += (
                        pos.quantity * float(cp) - pos.entry_amount_idr
                    )
            portfolio_value_history.append(total_value)

        # Force-close remaining
        for pos in open_positions:
            if pos.asset in asset_prices:
                dates_available = asset_prices[pos.asset].index
                if len(dates_available) > 0:
                    last_avail = dates_available[-1]
                    exit_price_raw = asset_prices[pos.asset].loc[
                        last_avail, "close"
                    ]
                    if isinstance(exit_price_raw, pd.Series):
                        exit_price_raw = exit_price_raw.iloc[0]
                    exit_price = float(exit_price_raw)
                    exit_amount = int(pos.quantity * exit_price)
                    pnl_idr = exit_amount - pos.entry_amount_idr
                    pnl_pct = (
                        (exit_price - pos.entry_price) / pos.entry_price
                    ) * 100
                    completed_trades.append(
                        BacktestTrade(
                            asset=pos.asset,
                            signal_date=pos.signal_date,
                            entry_date=pos.entry_date,
                            exit_date=last_avail,
                            entry_price=pos.entry_price,
                            exit_price=exit_price,
                            entry_amount_idr=pos.entry_amount_idr,
                            exit_amount_idr=exit_amount,
                            pnl_idr=pnl_idr,
                            pnl_pct=pnl_pct,
                            hold_days=(last_avail - pos.entry_date).days,
                            exit_reason="backtest_end",
                        )
                    )

        return self._compute_metrics(
            completed_trades, max_concurrent, portfolio_value_history
        )

    def _compute_metrics(
        self,
        trades: list[BacktestTrade],
        max_concurrent: int,
        portfolio_value_history: list[float],
    ) -> BacktestResult:
        total_trades = len(trades)

        if total_trades == 0:
            return BacktestResult(
                total_return_idr=0,
                total_return_pct=0.0,
                win_rate=0.0,
                reward_risk_ratio=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                total_trades=0,
                max_concurrent_positions=max_concurrent,
                trades=[],
                monthly_breakdown=[],
                per_asset_breakdown=[],
            )

        total_pnl = sum(t.pnl_idr for t in trades)
        total_invested = sum(t.entry_amount_idr for t in trades)
        total_return_pct = (
            (total_pnl / total_invested * 100) if total_invested > 0 else 0.0
        )

        winning_trades = [t for t in trades if t.pnl_idr > 0]
        win_rate = len(winning_trades) / total_trades

        avg_win = (
            np.mean([t.pnl_idr for t in winning_trades])
            if winning_trades
            else 0.0
        )
        losing_trades = [t for t in trades if t.pnl_idr < 0]
        avg_loss = (
            abs(np.mean([t.pnl_idr for t in losing_trades]))
            if losing_trades
            else 1.0
        )
        reward_risk_ratio = (
            float(avg_win / avg_loss) if avg_loss > 0 else 0.0
        )

        daily_returns = self._compute_daily_returns(portfolio_value_history)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe_ratio = float(
                np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            )
        else:
            sharpe_ratio = 0.0

        max_drawdown_pct = self._compute_max_drawdown(portfolio_value_history)
        monthly_breakdown = self._compute_monthly_breakdown(trades)
        per_asset_breakdown = self._compute_per_asset_breakdown(trades)

        return BacktestResult(
            total_return_idr=total_pnl,
            total_return_pct=total_return_pct,
            win_rate=win_rate,
            reward_risk_ratio=reward_risk_ratio,
            sharpe_ratio=sharpe_ratio,
            max_drawdown_pct=max_drawdown_pct,
            total_trades=total_trades,
            max_concurrent_positions=max_concurrent,
            trades=trades,
            monthly_breakdown=monthly_breakdown,
            per_asset_breakdown=per_asset_breakdown,
        )

    def _compute_daily_returns(
        self, portfolio_values: list[float]
    ) -> list[float]:
        if len(portfolio_values) < 2:
            return []
        values = np.array(portfolio_values)
        returns = np.diff(values) / values[:-1]
        return returns.tolist()

    def _compute_max_drawdown(self, portfolio_values: list[float]) -> float:
        if not portfolio_values:
            return 0.0
        values = np.array(portfolio_values)
        peak = np.maximum.accumulate(values)
        drawdowns = (values - peak) / peak * 100
        return float(np.min(drawdowns))

    def _compute_monthly_breakdown(
        self, trades: list[BacktestTrade]
    ) -> list[dict[str, Any]]:
        monthly: dict[str, dict[str, Any]] = {}
        for trade in trades:
            month_key = trade.exit_date.strftime("%Y-%m")
            if month_key not in monthly:
                monthly[month_key] = {
                    "month": month_key,
                    "pnl_idr": 0,
                    "trades": 0,
                }
            monthly[month_key]["pnl_idr"] += trade.pnl_idr
            monthly[month_key]["trades"] += 1
        return sorted(monthly.values(), key=lambda x: x["month"])

    def _compute_per_asset_breakdown(
        self, trades: list[BacktestTrade]
    ) -> list[dict[str, Any]]:
        by_asset: dict[str, list[BacktestTrade]] = {}
        for trade in trades:
            by_asset.setdefault(trade.asset, []).append(trade)

        result: list[dict[str, Any]] = []
        for asset, asset_trades in sorted(by_asset.items()):
            total_pnl = sum(t.pnl_idr for t in asset_trades)
            wins = sum(1 for t in asset_trades if t.pnl_idr > 0)
            win_rate = wins / len(asset_trades) if asset_trades else 0.0
            result.append({
                "asset": asset,
                "total_pnl_idr": total_pnl,
                "win_rate": win_rate,
                "trades": len(asset_trades),
                "avg_hold_days": np.mean(
                    [t.hold_days for t in asset_trades]
                ),
            })
        return result
