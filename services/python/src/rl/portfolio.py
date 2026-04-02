"""Portfolio state tracker for RL trading environment."""

from dataclasses import dataclass, field

from src.rl.config import RLConfig


@dataclass
class Position:
    """A single open position."""

    asset: str
    side: int  # 1=long, -1=short
    size_idr: float
    quantity: float
    entry_price: float
    sl_price: float
    tp_price: float
    windows_held: int = 0


class Portfolio:
    """Tracks cash, open positions, and computes portfolio metrics."""

    def __init__(self, cfg: RLConfig | None = None):
        self.cfg = cfg or RLConfig()
        self.cash: float = self.cfg.STARTING_CAPITAL
        self.positions: dict[str, Position] = {}
        self._peak_equity: float = self.cfg.STARTING_CAPITAL

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def n_positions(self) -> int:
        return len(self.positions)

    @property
    def equity(self) -> float:
        """Cash + sum of position sizes (at entry)."""
        return self.cash + sum(p.size_idr for p in self.positions.values())

    def equity_at_prices(self, current_prices: dict[str, float]) -> float:
        """Mark-to-market equity using current prices."""
        mtm = self.cash
        for asset, pos in self.positions.items():
            price = current_prices.get(asset, pos.entry_price)
            pnl = pos.side * (price - pos.entry_price) * pos.quantity
            mtm += pos.size_idr + pnl
        return mtm

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def open_position(
        self,
        asset: str,
        side: int,
        size_pct: float,
        entry_price: float,
        sl_pct: float,
        tp_pct: float,
    ) -> bool:
        """Open a position. Returns False if rejected by constraints."""
        # Constraint: duplicate asset
        if asset in self.positions:
            return False

        # Constraint: max positions
        if self.n_positions >= self.cfg.MAX_POSITIONS:
            return False

        # Size in IDR based on current equity
        size_idr = self.equity * size_pct

        # Constraint: max single position
        if size_pct > self.cfg.MAX_SINGLE_POSITION_PCT:
            return False

        # Constraint: max exposure
        total_exposure = sum(p.size_idr for p in self.positions.values()) + size_idr
        if total_exposure / self.equity > self.cfg.MAX_EXPOSURE_PCT:
            return False

        # Constraint: insufficient cash (need size + fees)
        fee_rate = self.cfg.TAKER_FEE + self.cfg.SLIPPAGE
        total_cost = size_idr + size_idr * fee_rate
        if total_cost > self.cash:
            return False

        # Compute SL/TP prices
        if side == 1:  # long
            sl_price = entry_price * (1 - sl_pct)
            tp_price = entry_price * (1 + tp_pct)
        else:  # short
            sl_price = entry_price * (1 + sl_pct)
            tp_price = entry_price * (1 - tp_pct)

        quantity = size_idr / entry_price

        pos = Position(
            asset=asset,
            side=side,
            size_idr=size_idr,
            quantity=quantity,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
        )

        self.positions[asset] = pos
        self.cash -= total_cost
        return True

    def close_position(self, asset: str, exit_price: float) -> float:
        """Close a position. Returns net PnL in IDR after fees."""
        pos = self.positions.pop(asset)
        fee_rate = self.cfg.TAKER_FEE + self.cfg.SLIPPAGE

        gross_pnl = pos.side * (exit_price - pos.entry_price) * pos.quantity
        exit_value = pos.size_idr + gross_pnl
        exit_fee = exit_value * fee_rate

        net_pnl = gross_pnl - exit_fee
        # Return size + net pnl to cash (we already deducted size+open_fee on open)
        self.cash += pos.size_idr + net_pnl
        return net_pnl

    def check_sl_tp(
        self, candle_data: dict[str, dict[str, float]]
    ) -> dict[str, dict[str, float | str]]:
        """Check SL/TP for all open positions. Returns fills."""
        fills: dict[str, dict[str, float | str]] = {}
        assets_to_close = []

        for asset, pos in self.positions.items():
            if asset not in candle_data:
                continue
            high = candle_data[asset]["high"]
            low = candle_data[asset]["low"]

            triggered = False
            reason = ""
            exit_price = 0.0

            if pos.side == 1:  # long
                if low <= pos.sl_price:
                    triggered = True
                    reason = "sl"
                    exit_price = pos.sl_price
                elif high >= pos.tp_price:
                    triggered = True
                    reason = "tp"
                    exit_price = pos.tp_price
            else:  # short
                if high >= pos.sl_price:
                    triggered = True
                    reason = "sl"
                    exit_price = pos.sl_price
                elif low <= pos.tp_price:
                    triggered = True
                    reason = "tp"
                    exit_price = pos.tp_price

            if triggered:
                assets_to_close.append((asset, exit_price, reason))

        for asset, exit_price, reason in assets_to_close:
            pnl = self.close_position(asset, exit_price)
            fills[asset] = {"reason": reason, "pnl": pnl}

        return fills

    def deduct_funding(self, n_8h_periods: int = 1) -> None:
        """Deduct funding rate from cash for all open positions."""
        for pos in self.positions.values():
            funding = pos.size_idr * self.cfg.FUNDING_RATE_8H * n_8h_periods
            self.cash -= funding

    def get_state_vector(
        self, current_prices: dict[str, float]
    ) -> list[float]:
        """Return 5 floats for the RL observation."""
        eq = self.equity_at_prices(current_prices)
        if eq <= 0:
            eq = 1e-9  # avoid division by zero

        # cash_ratio
        cash_ratio = max(0.0, min(1.0, self.cash / eq))

        # positions_count_ratio
        pos_ratio = self.n_positions / self.cfg.MAX_POSITIONS

        # portfolio_pnl_pct
        portfolio_pnl_pct = (eq - self.cfg.STARTING_CAPITAL) / self.cfg.STARTING_CAPITAL

        # avg_holding_time_normalized (normalize by EPISODE_WINDOWS)
        if self.n_positions > 0:
            avg_hold = sum(
                p.windows_held for p in self.positions.values()
            ) / self.n_positions
            avg_hold_norm = avg_hold / self.cfg.EPISODE_WINDOWS
        else:
            avg_hold_norm = 0.0

        # max_position_drawdown
        max_dd = 0.0
        for asset, pos in self.positions.items():
            price = current_prices.get(asset, pos.entry_price)
            pnl_pct = pos.side * (price - pos.entry_price) / pos.entry_price
            if pnl_pct < max_dd:
                max_dd = pnl_pct

        return [cash_ratio, pos_ratio, portfolio_pnl_pct, avg_hold_norm, max_dd]

    def increment_hold_times(self) -> None:
        """Increment windows_held for all open positions."""
        for pos in self.positions.values():
            pos.windows_held += 1
