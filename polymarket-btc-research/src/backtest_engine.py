"""
Backtesting Engine for Polymarket BTC Strategies

Simulates trading strategies on historical (or synthetic) market data
to validate edge detection and profitability.
"""

import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from scipy import stats


class Side(Enum):
    """Trade side."""
    BUY = "buy"
    SELL = "sell"


class Token(Enum):
    """Token type."""
    UP = "up"
    DOWN = "down"


@dataclass
class Trade:
    """Single trade execution."""
    timestamp: str
    market_id: str
    side: Side
    token: Token
    price: float  # Execution price (after spread)
    size: float  # Position size
    btc_price_at_entry: float
    btc_threshold: float
    seconds_until_close: int
    theoretical_prob: float  # Our calculated probability
    market_prob: float  # Market-implied probability
    edge_bps: float  # Edge in basis points


@dataclass
class Position:
    """Open position."""
    market_id: str
    token: Token
    entry_price: float
    size: float
    entry_time: str
    btc_price_at_entry: float
    btc_threshold: float


@dataclass
class ClosedPosition:
    """Closed position with P&L."""
    market_id: str
    token: Token
    entry_price: float
    exit_price: float
    size: float
    entry_time: str
    exit_time: str
    btc_price_at_entry: float
    btc_price_at_exit: float
    btc_threshold: float
    pnl: float
    return_pct: float
    win: bool


@dataclass
class BacktestResults:
    """Complete backtest results."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    total_volume: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    closed_positions: List[ClosedPosition] = field(default_factory=list)
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)

    def summary(self) -> str:
        """Generate summary report."""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║                   BACKTEST RESULTS                            ║
╚══════════════════════════════════════════════════════════════╝

📊 PERFORMANCE METRICS
├─ Total Trades:        {self.total_trades:,}
├─ Winning Trades:      {self.winning_trades:,}
├─ Losing Trades:       {self.losing_trades:,}
├─ Win Rate:            {self.win_rate:.1f}%
├─ Total P&L:           ${self.total_pnl:,.2f}
├─ Total Volume:        ${self.total_volume:,.2f}
├─ Avg Win:             ${self.avg_win:,.2f}
├─ Avg Loss:            ${self.avg_loss:,.2f}
├─ Sharpe Ratio:        {self.sharpe_ratio:.2f}
└─ Max Drawdown:        {self.max_drawdown:.1f}%

🎯 EDGE DETECTION
├─ Avg Edge (winners):  {self._avg_edge_winners():.1f} bps
├─ Avg Edge (losers):   {self._avg_edge_losers():.1f} bps
└─ Edge Predictive:     {self._edge_predictive():.1f}%

⏰ TIMING ANALYSIS
├─ Avg Entry Time:      {self._avg_entry_time():.1f}s before close
├─ Avg Exit Time:       {self._avg_exit_time():.1f}s before close
└─ Avg Hold Time:       {self._avg_hold_time():.1f}s

💰 RISK METRICS
├─ Best Trade:          ${self._best_trade():,.2f}
├─ Worst Trade:         ${self._worst_trade():,.2f}
├─ Avg Trade Size:      ${self._avg_trade_size():,.2f}
└─ ROI:                 {self._roi():.2f}%
"""

    def _avg_edge_winners(self) -> float:
        winners = [t for t in self.trades if self._trade_won(t)]
        if not winners:
            return 0.0
        return np.mean([t.edge_bps for t in winners])

    def _avg_edge_losers(self) -> float:
        losers = [t for t in self.trades if not self._trade_won(t)]
        if not losers:
            return 0.0
        return np.mean([t.edge_bps for t in losers])

    def _edge_predictive(self) -> float:
        """% of trades where positive edge led to win."""
        if not self.trades:
            return 0.0
        positive_edge_trades = [t for t in self.trades if t.edge_bps > 0]
        if not positive_edge_trades:
            return 0.0
        wins = sum(1 for t in positive_edge_trades if self._trade_won(t))
        return (wins / len(positive_edge_trades)) * 100

    def _trade_won(self, trade: Trade) -> bool:
        """Check if a trade was profitable."""
        for cp in self.closed_positions:
            if cp.entry_time == trade.timestamp and cp.market_id == trade.market_id:
                return cp.win
        return False

    def _avg_entry_time(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.seconds_until_close for t in self.trades])

    def _avg_exit_time(self) -> float:
        if not self.closed_positions:
            return 0.0
        times = []
        for cp in self.closed_positions:
            # Parse timestamps and calculate
            entry = datetime.fromisoformat(cp.entry_time.replace("Z", "+00:00"))
            exit = datetime.fromisoformat(cp.exit_time.replace("Z", "+00:00"))
            # Estimate seconds before close at exit
            times.append(5 * 60 - (exit - entry).total_seconds())
        return np.mean(times)

    def _avg_hold_time(self) -> float:
        if not self.closed_positions:
            return 0.0
        times = []
        for cp in self.closed_positions:
            entry = datetime.fromisoformat(cp.entry_time.replace("Z", "+00:00"))
            exit = datetime.fromisoformat(cp.exit_time.replace("Z", "+00:00"))
            times.append((exit - entry).total_seconds())
        return np.mean(times)

    def _best_trade(self) -> float:
        if not self.closed_positions:
            return 0.0
        return max(cp.pnl for cp in self.closed_positions)

    def _worst_trade(self) -> float:
        if not self.closed_positions:
            return 0.0
        return min(cp.pnl for cp in self.closed_positions)

    def _avg_trade_size(self) -> float:
        if not self.trades:
            return 0.0
        return np.mean([t.price * t.size for t in self.trades])

    def _roi(self) -> float:
        if self.total_volume == 0:
            return 0.0
        return (self.total_pnl / self.total_volume) * 100


class BacktestEngine:
    """Backtesting engine for Polymarket strategies."""

    def __init__(
        self,
        initial_capital: float = 10000.0,
        position_size: float = 100.0,
        min_edge_bps: float = 50.0,  # Minimum edge to trade
        min_seconds_before_close: int = 30,  # Don't enter too late
        max_seconds_before_close: int = 240,  # Don't enter too early
        btc_volatility: float = 0.5  # Annualized BTC volatility
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting capital
            position_size: Size per trade
            min_edge_bps: Minimum edge in bps to take a trade
            min_seconds_before_close: Don't enter if < this many seconds left
            max_seconds_before_close: Don't enter if > this many seconds left
            btc_volatility: Annualized BTC volatility for pricing
        """
        self.initial_capital = initial_capital
        self.position_size = position_size
        self.min_edge_bps = min_edge_bps
        self.min_seconds_before_close = min_seconds_before_close
        self.max_seconds_before_close = max_seconds_before_close
        self.btc_volatility = btc_volatility

        # State
        self.capital = initial_capital
        self.positions: Dict[str, Position] = {}
        self.results = BacktestResults()

    def calculate_theoretical_price(
        self,
        btc_price: float,
        threshold: float,
        seconds_until_close: int
    ) -> float:
        """Calculate theoretical fair price using Black-Scholes digital option."""
        if seconds_until_close <= 0:
            return 1.0 if btc_price >= threshold else 0.0

        years_to_expiry = seconds_until_close / (365.25 * 24 * 3600)

        if years_to_expiry < 1e-6:
            return 1.0 if btc_price >= threshold else 0.0

        log_moneyness = np.log(btc_price / threshold)
        vol_time = self.btc_volatility * np.sqrt(years_to_expiry)

        if vol_time < 1e-6:
            return 1.0 if btc_price >= threshold else 0.0

        d = log_moneyness / vol_time
        fair_price = stats.norm.cdf(d)

        return max(0.01, min(0.99, fair_price))

    def should_trade(
        self,
        theoretical_price: float,
        market_price: float,
        seconds_until_close: int
    ) -> Tuple[bool, Optional[Side], float]:
        """
        Determine if we should trade based on edge.

        Returns:
            (should_trade, side, edge_bps)
        """
        # Check timing window
        if seconds_until_close < self.min_seconds_before_close:
            return False, None, 0.0
        if seconds_until_close > self.max_seconds_before_close:
            return False, None, 0.0

        # Skip if market price is invalid
        if market_price <= 0.01 or market_price >= 0.99:
            return False, None, 0.0

        # Calculate edge
        edge = theoretical_price - market_price
        edge_bps = (edge / market_price) * 10000

        # Buy if market is underpricing (we think it should be higher)
        if edge_bps > self.min_edge_bps:
            return True, Side.BUY, edge_bps

        # Sell if market is overpricing (we think it should be lower)
        if edge_bps < -self.min_edge_bps:
            return True, Side.SELL, edge_bps

        return False, None, edge_bps

    def execute_trade(
        self,
        snapshot: Dict,
        side: Side,
        edge_bps: float
    ) -> Optional[Trade]:
        """Execute a trade with realistic fills."""
        market_id = snapshot["market_id"]

        # Don't trade if already have position
        if market_id in self.positions:
            return None

        # Get order book
        up_bids = snapshot["up_token_bids"]
        up_asks = snapshot["up_token_asks"]

        if not up_bids or not up_asks:
            return None

        # Determine execution price
        if side == Side.BUY:
            # Buy at the ask (pay the spread)
            execution_price = up_asks[0]["price"]
        else:
            # Sell at the bid
            execution_price = up_bids[0]["price"]

        # Calculate theoretical price
        theoretical_price = self.calculate_theoretical_price(
            snapshot["btc_price"],
            snapshot["btc_threshold"],
            snapshot["seconds_until_close"]
        )

        # Create trade
        trade = Trade(
            timestamp=snapshot["timestamp"],
            market_id=market_id,
            side=side,
            token=Token.UP,
            price=execution_price,
            size=self.position_size,
            btc_price_at_entry=snapshot["btc_price"],
            btc_threshold=snapshot["btc_threshold"],
            seconds_until_close=snapshot["seconds_until_close"],
            theoretical_prob=theoretical_price,
            market_prob=snapshot["up_token_mid"],
            edge_bps=edge_bps
        )

        # Open position
        self.positions[market_id] = Position(
            market_id=market_id,
            token=Token.UP,
            entry_price=execution_price,
            size=self.position_size,
            entry_time=snapshot["timestamp"],
            btc_price_at_entry=snapshot["btc_price"],
            btc_threshold=snapshot["btc_threshold"]
        )

        # Update capital
        cost = execution_price * self.position_size
        self.capital -= cost
        self.results.total_volume += cost

        return trade

    def close_position(
        self,
        market_id: str,
        snapshot: Dict,
        resolution: float  # 0.0 or 1.0
    ) -> Optional[ClosedPosition]:
        """Close position at market resolution."""
        if market_id not in self.positions:
            return None

        position = self.positions[market_id]

        # Exit price is resolution value
        exit_price = resolution

        # Calculate P&L
        pnl = (exit_price - position.entry_price) * position.size

        # Update capital
        self.capital += (exit_price * position.size)

        # Create closed position
        closed = ClosedPosition(
            market_id=market_id,
            token=position.token,
            entry_price=position.entry_price,
            exit_price=exit_price,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=snapshot["timestamp"],
            btc_price_at_entry=position.btc_price_at_entry,
            btc_price_at_exit=snapshot["btc_price"],
            btc_threshold=position.btc_threshold,
            pnl=pnl,
            return_pct=(pnl / (position.entry_price * position.size)) * 100,
            win=pnl > 0
        )

        # Remove position
        del self.positions[market_id]

        # Update results
        self.results.total_pnl += pnl
        self.results.closed_positions.append(closed)

        if closed.win:
            self.results.winning_trades += 1
        else:
            self.results.losing_trades += 1

        return closed

    def run(self, data_path: str) -> BacktestResults:
        """
        Run backtest on dataset.

        Args:
            data_path: Path to JSON data file

        Returns:
            BacktestResults
        """
        print("=" * 80)
        print("RUNNING BACKTEST")
        print("=" * 80)
        print(f"Data: {data_path}")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Position Size: ${self.position_size:,.2f}")
        print(f"Min Edge: {self.min_edge_bps:.1f} bps")
        print(f"Entry Window: {self.min_seconds_before_close}-{self.max_seconds_before_close}s before close")
        print()

        # Load data
        with open(data_path, "r") as f:
            data = json.load(f)

        total_markets = len(data)
        print(f"Loaded {total_markets:,} markets")

        # Process each market
        for i, (market_id, snapshots) in enumerate(data.items(), 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{total_markets} markets...")

            # Process snapshots chronologically
            for snapshot in snapshots:
                # Calculate theoretical price
                theoretical_price = self.calculate_theoretical_price(
                    snapshot["btc_price"],
                    snapshot["btc_threshold"],
                    snapshot["seconds_until_close"]
                )

                market_price = snapshot["up_token_mid"]

                # Check if we should trade
                should_trade, side, edge_bps = self.should_trade(
                    theoretical_price,
                    market_price,
                    snapshot["seconds_until_close"]
                )

                if should_trade and side:
                    trade = self.execute_trade(snapshot, side, edge_bps)
                    if trade:
                        self.results.trades.append(trade)

            # Close position at market end
            final_snapshot = snapshots[-1]
            resolution = 1.0 if final_snapshot["btc_price"] >= final_snapshot["btc_threshold"] else 0.0

            closed = self.close_position(market_id, final_snapshot, resolution)

            # Track equity
            self.results.equity_curve.append((
                final_snapshot["timestamp"],
                self.capital
            ))

        # Calculate final metrics
        self._calculate_metrics()

        print(f"\n✅ Backtest complete!")
        print(f"   Processed {total_markets:,} markets")
        print(f"   Executed {len(self.results.trades):,} trades")
        print(f"   Final Capital: ${self.capital:,.2f}")
        print(f"   Total P&L: ${self.results.total_pnl:,.2f}")

        return self.results

    def _calculate_metrics(self):
        """Calculate summary metrics."""
        self.results.total_trades = len(self.results.closed_positions)

        if self.results.total_trades > 0:
            self.results.win_rate = (self.results.winning_trades / self.results.total_trades) * 100

            # Average win/loss
            wins = [cp.pnl for cp in self.results.closed_positions if cp.win]
            losses = [cp.pnl for cp in self.results.closed_positions if not cp.win]

            self.results.avg_win = np.mean(wins) if wins else 0.0
            self.results.avg_loss = np.mean(losses) if losses else 0.0

            # Sharpe ratio
            returns = [cp.return_pct for cp in self.results.closed_positions]
            if len(returns) > 1:
                self.results.sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
            else:
                self.results.sharpe_ratio = 0.0

            # Max drawdown
            equity = [self.initial_capital] + [eq[1] for eq in self.results.equity_curve]
            peak = equity[0]
            max_dd = 0.0
            for value in equity:
                if value > peak:
                    peak = value
                dd = ((peak - value) / peak) * 100
                if dd > max_dd:
                    max_dd = dd
            self.results.max_drawdown = max_dd


def main():
    """Run backtest example."""
    import os

    data_file = "data/synthetic_markets_7d.json"

    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("   Run: .venv/bin/python src/synthetic_data_generator.py")
        return

    # Create backtest engine
    engine = BacktestEngine(
        initial_capital=10000.0,
        position_size=100.0,
        min_edge_bps=50.0,  # Only trade with 50+ bps edge
        min_seconds_before_close=30,  # Enter at least 30s before
        max_seconds_before_close=180,  # Enter at most 3min before
        btc_volatility=0.5
    )

    # Run backtest
    results = engine.run(data_file)

    # Print summary
    print("\n" + results.summary())

    # Save results
    results_file = "data/backtest_results.json"
    with open(results_file, "w") as f:
        json.dump({
            "summary": {
                "total_trades": results.total_trades,
                "winning_trades": results.winning_trades,
                "losing_trades": results.losing_trades,
                "win_rate": results.win_rate,
                "total_pnl": results.total_pnl,
                "sharpe_ratio": results.sharpe_ratio,
                "max_drawdown": results.max_drawdown
            },
            "trades": [
                {
                    "timestamp": t.timestamp,
                    "market_id": t.market_id,
                    "side": t.side.value,
                    "price": t.price,
                    "edge_bps": t.edge_bps
                }
                for t in results.trades[:100]  # First 100 trades
            ]
        }, f, indent=2)

    print(f"\n✅ Results saved to {results_file}")


if __name__ == "__main__":
    main()
