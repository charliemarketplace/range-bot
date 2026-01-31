"""
Buyback strategy simulations.

All functions are pure - given the same inputs, produce the same outputs.
"""
import random
import math
from decimal import Decimal
from typing import Sequence, Callable

from .types import (
    DailyRevenue,
    DailyAggregate,
    DailyBuyback,
    StrategyResult,
    StrategyComparison,
    PricedSwap,
)


# =============================================================================
# Strategy Type
# =============================================================================

# A strategy is a function that takes daily revenue and daily aggregate
# and returns the effective price for that day's buyback
StrategyFn = Callable[[DailyRevenue, DailyAggregate], Decimal]


# =============================================================================
# Individual Strategies
# =============================================================================

def best_possible_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
    """
    Best possible: buy at daily low price.

    This is the theoretical optimum if you had perfect foresight.
    """
    return aggregate.low_price if aggregate.low_price > 0 else aggregate.vwap


def worst_possible_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
    """
    Worst possible: buy at daily high price.

    This is the theoretical worst case.
    """
    return aggregate.high_price if aggregate.high_price > 0 else aggregate.vwap


def twap_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
    """
    TWAP: time-weighted average price.

    Execute evenly across time throughout the day.
    """
    return aggregate.twap if aggregate.twap > 0 else aggregate.vwap


def vwap_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
    """
    VWAP: volume-weighted average price.

    Execute proportional to market volume.
    """
    return aggregate.vwap if aggregate.vwap > 0 else aggregate.twap


def open_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
    """
    Open: buy at daily open price.

    Simple strategy - execute immediately at market open.
    """
    return aggregate.open_price if aggregate.open_price > 0 else aggregate.vwap


def close_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
    """
    Close: buy at daily close price.

    Execute at end of day.
    """
    return aggregate.close_price if aggregate.close_price > 0 else aggregate.vwap


def midpoint_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
    """
    Midpoint: average of high and low.

    Simple range midpoint strategy.
    """
    if aggregate.high_price > 0 and aggregate.low_price > 0:
        return (aggregate.high_price + aggregate.low_price) / 2
    return aggregate.vwap


# =============================================================================
# Random Strategy (Monte Carlo)
# =============================================================================

def create_random_strategy(seed: int = 42) -> StrategyFn:
    """
    Create a random strategy that picks a random price within the day's range.

    Returns a new strategy function with fixed seed for reproducibility.
    """
    rng = random.Random(seed)

    def random_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
        if aggregate.high_price > 0 and aggregate.low_price > 0:
            # Random price within the day's range
            low = float(aggregate.low_price)
            high = float(aggregate.high_price)
            price = rng.uniform(low, high)
            return Decimal(str(price))
        return aggregate.vwap

    return random_price


# =============================================================================
# Bayesian Strategy (Carlos Bayes - placeholder)
# =============================================================================

def create_bayesian_strategy(
    prior_window: int = 10,
    confidence: float = 0.9
) -> StrategyFn:
    """
    Bayesian strategy using rolling VWAP as prior.

    Similar to the range-bot approach:
    1. Use recent VWAP as prior center
    2. Use recent volatility to set prior width
    3. Update with current day's data
    4. Execute at posterior expected value

    This is a simplified version - full implementation would use
    the bayesian engine from the range-bot module.
    """
    history: list[DailyAggregate] = []

    def bayesian_price(revenue: DailyRevenue, aggregate: DailyAggregate) -> Decimal:
        nonlocal history

        # Add current day to history
        history.append(aggregate)

        # Keep only recent history
        if len(history) > prior_window:
            history = history[-prior_window:]

        # If not enough history, use VWAP
        if len(history) < 3:
            return aggregate.vwap

        # Compute rolling VWAP as prior center
        vwaps = [d.vwap for d in history[:-1] if d.vwap > 0]
        if not vwaps:
            return aggregate.vwap

        prior_center = sum(vwaps) / len(vwaps)

        # Current day's VWAP as likelihood center
        likelihood_center = aggregate.vwap

        # Simple weighted combination (Bayesian-ish)
        # Prior weight decreases with volatility
        recent_volatility = _compute_volatility(history[:-1])
        prior_weight = Decimal(str(1 / (1 + float(recent_volatility) * 10)))

        posterior = prior_weight * prior_center + (1 - prior_weight) * likelihood_center

        return posterior

    return bayesian_price


def _compute_volatility(aggregates: Sequence[DailyAggregate]) -> Decimal:
    """Compute volatility from daily price ranges."""
    if not aggregates:
        return Decimal("0.01")

    ranges = []
    for agg in aggregates:
        if agg.high_price > 0 and agg.low_price > 0 and agg.vwap > 0:
            daily_range = (agg.high_price - agg.low_price) / agg.vwap
            ranges.append(float(daily_range))

    if not ranges:
        return Decimal("0.01")

    avg_range = sum(ranges) / len(ranges)
    return Decimal(str(avg_range))


# =============================================================================
# Strategy Execution
# =============================================================================

def execute_strategy(
    strategy_name: str,
    strategy_fn: StrategyFn,
    revenues: Sequence[DailyRevenue],
    aggregates: Sequence[DailyAggregate]
) -> StrategyResult:
    """
    Execute a strategy over the full period.

    Matches revenues to aggregates by date and computes buybacks.
    """
    # Create lookup by date
    agg_by_date = {a.date: a for a in aggregates}

    daily_buybacks = []
    total_eth = Decimal("0")
    total_op = Decimal("0")

    for revenue in revenues:
        aggregate = agg_by_date.get(revenue.date)

        if aggregate is None or aggregate.vwap <= 0:
            # No trading data for this day - skip
            continue

        eth_to_spend = revenue.buyback_allocation_eth
        effective_price = strategy_fn(revenue, aggregate)

        if effective_price <= 0:
            continue

        # Price is ETH/OP, so OP = ETH / price
        op_acquired = eth_to_spend / effective_price

        daily_buybacks.append(DailyBuyback(
            date=revenue.date,
            eth_spent=eth_to_spend,
            op_acquired=op_acquired,
            avg_price=effective_price,
        ))

        total_eth += eth_to_spend
        total_op += op_acquired

    # Average price paid = total ETH / total OP
    avg_price = total_eth / total_op if total_op > 0 else Decimal("0")

    return StrategyResult(
        strategy_name=strategy_name,
        daily_buybacks=tuple(daily_buybacks),
        total_eth_spent=total_eth,
        total_op_acquired=total_op,
        avg_price_paid=avg_price,
    )


# =============================================================================
# Strategy Comparison
# =============================================================================

def compare_strategies(
    results: Sequence[StrategyResult]
) -> StrategyComparison:
    """Compare multiple strategy results."""
    if not results:
        raise ValueError("No strategy results to compare")

    # Find best and worst by total OP acquired
    sorted_by_op = sorted(results, key=lambda r: r.total_op_acquired, reverse=True)
    best = sorted_by_op[0]
    worst = sorted_by_op[-1]

    op_diff = best.total_op_acquired - worst.total_op_acquired
    pct_diff = (op_diff / worst.total_op_acquired * 100) if worst.total_op_acquired > 0 else Decimal("0")

    return StrategyComparison(
        strategies=tuple(results),
        best_strategy=best.strategy_name,
        worst_strategy=worst.strategy_name,
        best_vs_worst_op_diff=op_diff,
        best_vs_worst_pct_diff=pct_diff,
    )


# =============================================================================
# Run All Strategies
# =============================================================================

def run_all_strategies(
    revenues: Sequence[DailyRevenue],
    aggregates: Sequence[DailyAggregate],
    random_seed: int = 42
) -> StrategyComparison:
    """
    Run all defined strategies and return comparison.
    """
    strategies = [
        ("best_possible", best_possible_price),
        ("worst_possible", worst_possible_price),
        ("twap", twap_price),
        ("vwap", vwap_price),
        ("open", open_price),
        ("close", close_price),
        ("midpoint", midpoint_price),
        ("random", create_random_strategy(random_seed)),
        ("bayesian", create_bayesian_strategy()),
    ]

    results = [
        execute_strategy(name, fn, revenues, aggregates)
        for name, fn in strategies
    ]

    return compare_strategies(results)


# =============================================================================
# Monte Carlo Simulation
# =============================================================================

def monte_carlo_random_strategy(
    revenues: Sequence[DailyRevenue],
    aggregates: Sequence[DailyAggregate],
    num_simulations: int = 1000,
    base_seed: int = 42
) -> tuple[StrategyResult, ...]:
    """
    Run many random strategy simulations.

    Returns distribution of outcomes for statistical analysis.
    """
    results = []

    for i in range(num_simulations):
        strategy_fn = create_random_strategy(seed=base_seed + i)
        result = execute_strategy(
            f"random_{i}",
            strategy_fn,
            revenues,
            aggregates
        )
        results.append(result)

    return tuple(results)


def analyze_monte_carlo(
    results: Sequence[StrategyResult]
) -> dict:
    """
    Analyze Monte Carlo results.

    Returns statistics about the random strategy distribution.
    """
    op_totals = [float(r.total_op_acquired) for r in results]

    if not op_totals:
        return {}

    op_totals_sorted = sorted(op_totals)
    n = len(op_totals_sorted)

    return {
        "min_op": Decimal(str(min(op_totals))),
        "max_op": Decimal(str(max(op_totals))),
        "mean_op": Decimal(str(sum(op_totals) / n)),
        "median_op": Decimal(str(op_totals_sorted[n // 2])),
        "p5_op": Decimal(str(op_totals_sorted[int(n * 0.05)])),
        "p95_op": Decimal(str(op_totals_sorted[int(n * 0.95)])),
        "std_op": Decimal(str(_std(op_totals))),
    }


def _std(values: list[float]) -> float:
    """Compute standard deviation."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(variance)
