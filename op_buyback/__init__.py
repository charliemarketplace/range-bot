"""
OP Buyback Analysis Module

Hypothetical retroactive analysis of OP token buybacks
using Optimism sequencer revenue.

Pure functional design - all functions are side-effect free.
"""

from .types import (
    DailyRevenue,
    PoolMeta,
    PoolComparison,
    Swap,
    PricedSwap,
    DailyAggregate,
    DailyBuyback,
    StrategyResult,
    StrategyComparison,
    AnalysisResult,
)

from .aggregations import (
    derive_price_from_swap,
    price_swaps,
    compute_ohlc,
    compute_twap,
    compute_vwap,
    compute_volumes,
    aggregate_swaps_to_daily,
    process_swaps,
)

from .strategies import (
    best_possible_price,
    worst_possible_price,
    twap_price,
    vwap_price,
    execute_strategy,
    compare_strategies,
    run_all_strategies,
    monte_carlo_random_strategy,
    analyze_monte_carlo,
)

from .analysis import (
    run_analysis,
    format_full_report,
)

__all__ = [
    # Types
    "DailyRevenue",
    "PoolMeta",
    "PoolComparison",
    "Swap",
    "PricedSwap",
    "DailyAggregate",
    "DailyBuyback",
    "StrategyResult",
    "StrategyComparison",
    "AnalysisResult",
    # Aggregations
    "derive_price_from_swap",
    "price_swaps",
    "compute_ohlc",
    "compute_twap",
    "compute_vwap",
    "compute_volumes",
    "aggregate_swaps_to_daily",
    "process_swaps",
    # Strategies
    "best_possible_price",
    "worst_possible_price",
    "twap_price",
    "vwap_price",
    "execute_strategy",
    "compare_strategies",
    "run_all_strategies",
    "monte_carlo_random_strategy",
    "analyze_monte_carlo",
    # Analysis
    "run_analysis",
    "format_full_report",
]
