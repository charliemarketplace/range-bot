"""
Immutable data types for OP buyback analysis.

All types are frozen dataclasses - no mutation allowed.
"""
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence


# =============================================================================
# Scope A: Sequencer Revenue
# =============================================================================

@dataclass(frozen=True)
class DailyRevenue:
    """Daily sequencer revenue from Optimism."""
    date: str                    # YYYY-MM-DD
    l1_fee_eth: Decimal          # L1 data fees
    l2_fee_eth: Decimal          # L2 execution fees
    tx_count: int                # Transaction count

    @property
    def total_revenue_eth(self) -> Decimal:
        return self.l1_fee_eth + self.l2_fee_eth

    @property
    def buyback_allocation_eth(self) -> Decimal:
        """50% of revenue allocated to buybacks."""
        return self.total_revenue_eth / 2


# =============================================================================
# Scope B: Pool Metadata
# =============================================================================

@dataclass(frozen=True)
class PoolMeta:
    """Metadata for an OP/ETH pool."""
    pool_address: str
    project: str                 # "velodrome" or "uniswap_v3"
    token0: str                  # Token0 address
    token1: str                  # Token1 address
    token0_symbol: str           # e.g., "OP" or "WETH"
    token1_symbol: str           # e.g., "WETH" or "OP"
    token0_decimals: int
    token1_decimals: int

    @property
    def op_is_token0(self) -> bool:
        return self.token0_symbol == "OP"


@dataclass(frozen=True)
class PoolComparison:
    """Pool comparison for selecting deepest pool."""
    project: str
    pool_address: str
    total_volume_eth: Decimal
    trade_count: int
    avg_daily_volume_eth: Decimal


# =============================================================================
# Scope C: Swap Events
# =============================================================================

@dataclass(frozen=True)
class Swap:
    """Raw swap event from pool."""
    block_number: int
    tx_hash: str
    log_index: int
    amount0: int                 # Signed, raw (no decimal adjustment)
    amount1: int                 # Signed, raw (no decimal adjustment)
    timestamp: int               # Unix timestamp (joined from blocks table)


@dataclass(frozen=True)
class PricedSwap:
    """Swap with derived price and decimal-adjusted amounts."""
    block_number: int
    tx_hash: str
    log_index: int
    timestamp: int
    amount_op: Decimal           # OP amount (positive = bought, negative = sold)
    amount_eth: Decimal          # ETH amount (positive = received, negative = spent)
    price_eth_per_op: Decimal    # ETH cost per 1 OP (lower = better for buyer)
    is_buy: bool                 # True if OP was bought (ETH spent)


# =============================================================================
# Derived: Daily Aggregates
# =============================================================================

@dataclass(frozen=True)
class DailyAggregate:
    """Daily trading aggregates for a pool."""
    date: str                    # YYYY-MM-DD
    buy_volume_eth: Decimal      # ETH spent buying OP
    sell_volume_eth: Decimal     # ETH received selling OP
    buy_volume_op: Decimal       # OP bought
    sell_volume_op: Decimal      # OP sold
    open_price: Decimal          # First swap price (OP/ETH)
    close_price: Decimal         # Last swap price
    high_price: Decimal          # Max price
    low_price: Decimal           # Min price
    twap: Decimal                # Time-weighted average price
    vwap: Decimal                # Volume-weighted average price
    trade_count: int             # Number of swaps

    @property
    def net_buy_volume_eth(self) -> Decimal:
        return self.buy_volume_eth - self.sell_volume_eth

    @property
    def net_buy_volume_op(self) -> Decimal:
        return self.buy_volume_op - self.sell_volume_op


# =============================================================================
# Strategy Results
# =============================================================================

@dataclass(frozen=True)
class DailyBuyback:
    """Result of a single day's buyback."""
    date: str
    eth_spent: Decimal           # ETH used for buyback
    op_acquired: Decimal         # OP tokens acquired
    avg_price: Decimal           # Effective price paid


@dataclass(frozen=True)
class StrategyResult:
    """Complete result of a buyback strategy simulation."""
    strategy_name: str
    daily_buybacks: tuple[DailyBuyback, ...]
    total_eth_spent: Decimal
    total_op_acquired: Decimal
    avg_price_paid: Decimal

    @property
    def num_days(self) -> int:
        return len(self.daily_buybacks)


@dataclass(frozen=True)
class StrategyComparison:
    """Comparison of multiple strategies."""
    strategies: tuple[StrategyResult, ...]
    best_strategy: str
    worst_strategy: str
    best_vs_worst_op_diff: Decimal
    best_vs_worst_pct_diff: Decimal


# =============================================================================
# Analysis Outputs
# =============================================================================

@dataclass(frozen=True)
class AnalysisResult:
    """Complete analysis output."""
    period_start: str
    period_end: str
    pool: PoolMeta
    total_sequencer_revenue_eth: Decimal
    total_buyback_allocation_eth: Decimal
    strategy_comparison: StrategyComparison
    daily_aggregates: tuple[DailyAggregate, ...]
    daily_revenues: tuple[DailyRevenue, ...]
