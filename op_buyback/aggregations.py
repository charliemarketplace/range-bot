"""
Aggregation functions - convert raw swaps to daily aggregates.

All functions are pure - no side effects, no mutation.
"""
from decimal import Decimal
from datetime import date
from typing import Sequence
from collections import defaultdict

from .types import (
    Swap,
    PricedSwap,
    PoolMeta,
    DailyAggregate,
)


# =============================================================================
# Price Derivation
# =============================================================================

def derive_price_from_swap(
    swap: Swap,
    pool_meta: PoolMeta
) -> PricedSwap:
    """
    Derive price from swap amounts.

    Price = abs(amount_eth) / abs(amount_op) with decimal adjustment

    Returns PricedSwap with human-readable amounts and price.
    """
    decimals0 = pool_meta.token0_decimals
    decimals1 = pool_meta.token1_decimals

    # Convert raw amounts to decimal-adjusted
    amount0_adj = Decimal(swap.amount0) / Decimal(10 ** decimals0)
    amount1_adj = Decimal(swap.amount1) / Decimal(10 ** decimals1)

    # Determine which is OP and which is ETH
    if pool_meta.op_is_token0:
        amount_op = amount0_adj
        amount_eth = amount1_adj
    else:
        amount_op = amount1_adj
        amount_eth = amount0_adj

    # Calculate price (ETH per OP) - how much ETH to buy 1 OP
    # Lower price = better for buyer
    abs_op = abs(amount_op)
    abs_eth = abs(amount_eth)

    if abs_op > 0:
        price_eth_per_op = abs_eth / abs_op
    else:
        price_eth_per_op = Decimal("0")

    # Determine if this is a buy (OP acquired, ETH spent)
    is_buy = amount_op > 0 and amount_eth < 0

    return PricedSwap(
        block_number=swap.block_number,
        tx_hash=swap.tx_hash,
        log_index=swap.log_index,
        timestamp=swap.timestamp,
        amount_op=amount_op,
        amount_eth=amount_eth,
        price_eth_per_op=price_eth_per_op,
        is_buy=is_buy,
    )


def price_swaps(
    swaps: Sequence[Swap],
    pool_meta: PoolMeta
) -> tuple[PricedSwap, ...]:
    """Convert all swaps to priced swaps."""
    return tuple(derive_price_from_swap(s, pool_meta) for s in swaps)


# =============================================================================
# Daily Grouping
# =============================================================================

def timestamp_to_date(timestamp: int) -> str:
    """Convert Unix timestamp to YYYY-MM-DD string."""
    return date.fromtimestamp(timestamp).isoformat()


def group_swaps_by_date(
    swaps: Sequence[PricedSwap]
) -> dict[str, list[PricedSwap]]:
    """Group swaps by calendar date."""
    grouped: dict[str, list[PricedSwap]] = defaultdict(list)

    for swap in swaps:
        day = timestamp_to_date(swap.timestamp)
        grouped[day].append(swap)

    return dict(grouped)


# =============================================================================
# OHLC Calculation
# =============================================================================

def compute_ohlc(
    swaps: Sequence[PricedSwap]
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """
    Compute OHLC from swaps.

    Returns (open, high, low, close)
    """
    if not swaps:
        zero = Decimal("0")
        return (zero, zero, zero, zero)

    # Sort by timestamp then log_index for determinism
    sorted_swaps = sorted(swaps, key=lambda s: (s.timestamp, s.log_index))

    prices = [s.price_eth_per_op for s in sorted_swaps if s.price_eth_per_op > 0]

    if not prices:
        zero = Decimal("0")
        return (zero, zero, zero, zero)

    open_price = prices[0]
    close_price = prices[-1]
    high_price = max(prices)
    low_price = min(prices)

    return (open_price, high_price, low_price, close_price)


# =============================================================================
# TWAP Calculation
# =============================================================================

def compute_twap(
    swaps: Sequence[PricedSwap]
) -> Decimal:
    """
    Compute time-weighted average price.

    Each price is weighted by the time until the next swap.
    """
    if not swaps:
        return Decimal("0")

    sorted_swaps = sorted(swaps, key=lambda s: (s.timestamp, s.log_index))
    valid_swaps = [s for s in sorted_swaps if s.price_eth_per_op > 0]

    if not valid_swaps:
        return Decimal("0")

    if len(valid_swaps) == 1:
        return valid_swaps[0].price_eth_per_op

    # Calculate time-weighted sum
    total_time = Decimal("0")
    weighted_sum = Decimal("0")

    for i in range(len(valid_swaps) - 1):
        current = valid_swaps[i]
        next_swap = valid_swaps[i + 1]

        time_delta = Decimal(next_swap.timestamp - current.timestamp)
        if time_delta > 0:
            weighted_sum += current.price_eth_per_op * time_delta
            total_time += time_delta

    # Add last swap (use 1 second as minimum weight)
    if total_time > 0:
        return weighted_sum / total_time
    else:
        # All swaps at same timestamp - simple average
        return sum(s.price_eth_per_op for s in valid_swaps) / len(valid_swaps)


# =============================================================================
# VWAP Calculation
# =============================================================================

def compute_vwap(
    swaps: Sequence[PricedSwap]
) -> Decimal:
    """
    Compute volume-weighted average price.

    VWAP = Σ(price × volume) / Σ(volume)
    Using ETH volume as weight.
    """
    if not swaps:
        return Decimal("0")

    valid_swaps = [s for s in swaps if s.price_eth_per_op > 0]

    if not valid_swaps:
        return Decimal("0")

    total_volume = Decimal("0")
    weighted_sum = Decimal("0")

    for swap in valid_swaps:
        volume = abs(swap.amount_eth)
        weighted_sum += swap.price_eth_per_op * volume
        total_volume += volume

    if total_volume > 0:
        return weighted_sum / total_volume
    else:
        return Decimal("0")


# =============================================================================
# Volume Calculation
# =============================================================================

def compute_volumes(
    swaps: Sequence[PricedSwap]
) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    """
    Compute buy and sell volumes.

    Returns (buy_volume_eth, sell_volume_eth, buy_volume_op, sell_volume_op)
    """
    buy_eth = Decimal("0")
    sell_eth = Decimal("0")
    buy_op = Decimal("0")
    sell_op = Decimal("0")

    for swap in swaps:
        if swap.is_buy:
            # Buying OP: ETH spent (negative), OP received (positive)
            buy_eth += abs(swap.amount_eth)
            buy_op += abs(swap.amount_op)
        else:
            # Selling OP: ETH received (positive), OP spent (negative)
            sell_eth += abs(swap.amount_eth)
            sell_op += abs(swap.amount_op)

    return (buy_eth, sell_eth, buy_op, sell_op)


# =============================================================================
# Daily Aggregate Construction
# =============================================================================

def compute_daily_aggregate(
    day: str,
    swaps: Sequence[PricedSwap]
) -> DailyAggregate:
    """Compute all aggregates for a single day."""
    open_p, high_p, low_p, close_p = compute_ohlc(swaps)
    twap = compute_twap(swaps)
    vwap = compute_vwap(swaps)
    buy_eth, sell_eth, buy_op, sell_op = compute_volumes(swaps)

    return DailyAggregate(
        date=day,
        buy_volume_eth=buy_eth,
        sell_volume_eth=sell_eth,
        buy_volume_op=buy_op,
        sell_volume_op=sell_op,
        open_price=open_p,
        close_price=close_p,
        high_price=high_p,
        low_price=low_p,
        twap=twap,
        vwap=vwap,
        trade_count=len(swaps),
    )


def aggregate_swaps_to_daily(
    swaps: Sequence[PricedSwap]
) -> tuple[DailyAggregate, ...]:
    """
    Aggregate swaps into daily OHLC/TWAP/VWAP.

    Returns sorted tuple of DailyAggregate.
    """
    grouped = group_swaps_by_date(swaps)

    aggregates = [
        compute_daily_aggregate(day, day_swaps)
        for day, day_swaps in sorted(grouped.items())
    ]

    return tuple(aggregates)


# =============================================================================
# Full Pipeline
# =============================================================================

def process_swaps(
    raw_swaps: Sequence[Swap],
    pool_meta: PoolMeta
) -> tuple[DailyAggregate, ...]:
    """
    Full pipeline: raw swaps → priced swaps → daily aggregates.
    """
    priced = price_swaps(raw_swaps, pool_meta)
    return aggregate_swaps_to_daily(priced)
