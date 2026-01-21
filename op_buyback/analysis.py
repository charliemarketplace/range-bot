"""
Full analysis pipeline - ties all modules together.

Pure functions for running complete analysis.
"""
from decimal import Decimal
from typing import Sequence

from .types import (
    DailyRevenue,
    DailyAggregate,
    PoolMeta,
    PoolComparison,
    AnalysisResult,
    StrategyComparison,
    Swap,
)
from .data_stubs import (
    generate_daily_revenues,
    get_pool_comparisons,
    select_deepest_pool,
    get_pool_meta,
    generate_swaps,
)
from .aggregations import process_swaps
from .strategies import run_all_strategies, monte_carlo_random_strategy, analyze_monte_carlo


# =============================================================================
# Full Analysis Pipeline
# =============================================================================

def run_analysis(
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42
) -> AnalysisResult:
    """
    Run complete retroactive buyback analysis.

    Steps:
    1. Load sequencer revenue (Scope A)
    2. Select deepest pool (Scope B)
    3. Load swap data (Scope C)
    4. Aggregate to daily data
    5. Run strategy simulations
    6. Return results
    """
    # Scope A: Sequencer Revenue
    revenues = generate_daily_revenues(start_date, end_date, seed)

    # Scope B: Pool Selection
    pool_comparisons = get_pool_comparisons(seed)
    selected_pool = select_deepest_pool(pool_comparisons)
    pool_meta = get_pool_meta(selected_pool.pool_address)

    # Scope C: Swap Data
    raw_swaps = generate_swaps(pool_meta, start_date, end_date, seed=seed)

    # Aggregate
    daily_aggregates = process_swaps(raw_swaps, pool_meta)

    # Strategy Comparison
    strategy_comparison = run_all_strategies(revenues, daily_aggregates, seed)

    # Totals
    total_revenue = sum(r.total_revenue_eth for r in revenues)
    total_allocation = sum(r.buyback_allocation_eth for r in revenues)

    return AnalysisResult(
        period_start=start_date,
        period_end=end_date,
        pool=pool_meta,
        total_sequencer_revenue_eth=total_revenue,
        total_buyback_allocation_eth=total_allocation,
        strategy_comparison=strategy_comparison,
        daily_aggregates=daily_aggregates,
        daily_revenues=revenues,
    )


# =============================================================================
# Reporting Functions
# =============================================================================

def format_strategy_summary(comparison: StrategyComparison) -> str:
    """Format strategy comparison as text report."""
    lines = [
        "=" * 70,
        "STRATEGY COMPARISON",
        "=" * 70,
        "",
    ]

    # Sort by OP acquired
    sorted_strategies = sorted(
        comparison.strategies,
        key=lambda s: s.total_op_acquired,
        reverse=True
    )

    lines.append(f"{'Strategy':<20} {'ETH Spent':>15} {'OP Acquired':>18} {'Avg Price':>12}")
    lines.append("-" * 70)

    for strat in sorted_strategies:
        lines.append(
            f"{strat.strategy_name:<20} "
            f"{float(strat.total_eth_spent):>15,.2f} "
            f"{float(strat.total_op_acquired):>18,.2f} "
            f"{float(strat.avg_price_paid):>12,.4f}"
        )

    lines.append("-" * 70)
    lines.append(f"\nBest Strategy: {comparison.best_strategy}")
    lines.append(f"Worst Strategy: {comparison.worst_strategy}")
    lines.append(f"Difference: {float(comparison.best_vs_worst_op_diff):,.2f} OP ({float(comparison.best_vs_worst_pct_diff):.2f}%)")

    return "\n".join(lines)


def format_revenue_summary(revenues: Sequence[DailyRevenue]) -> str:
    """Format revenue summary."""
    total_l1 = sum(r.l1_fee_eth for r in revenues)
    total_l2 = sum(r.l2_fee_eth for r in revenues)
    total = sum(r.total_revenue_eth for r in revenues)
    total_buyback = sum(r.buyback_allocation_eth for r in revenues)
    total_tx = sum(r.tx_count for r in revenues)

    lines = [
        "=" * 70,
        "SEQUENCER REVENUE SUMMARY",
        "=" * 70,
        "",
        f"Period: {revenues[0].date} to {revenues[-1].date}",
        f"Days: {len(revenues)}",
        "",
        f"L1 Fees:          {float(total_l1):>15,.2f} ETH",
        f"L2 Fees:          {float(total_l2):>15,.2f} ETH",
        f"Total Revenue:    {float(total):>15,.2f} ETH",
        f"Buyback (50%):    {float(total_buyback):>15,.2f} ETH",
        f"Transactions:     {total_tx:>15,}",
        "",
        f"Avg Daily Revenue: {float(total / len(revenues)):>14,.2f} ETH",
        f"Avg Daily Buyback: {float(total_buyback / len(revenues)):>14,.2f} ETH",
    ]

    return "\n".join(lines)


def format_trading_summary(aggregates: Sequence[DailyAggregate]) -> str:
    """Format trading data summary."""
    total_buy_eth = sum(a.buy_volume_eth for a in aggregates)
    total_sell_eth = sum(a.sell_volume_eth for a in aggregates)
    total_trades = sum(a.trade_count for a in aggregates)

    prices = [float(a.vwap) for a in aggregates if a.vwap > 0]
    avg_price = sum(prices) / len(prices) if prices else 0

    lines = [
        "=" * 70,
        "TRADING DATA SUMMARY",
        "=" * 70,
        "",
        f"Days with data: {len(aggregates)}",
        f"Total trades: {total_trades:,}",
        "",
        f"Total Buy Volume:  {float(total_buy_eth):>15,.2f} ETH",
        f"Total Sell Volume: {float(total_sell_eth):>15,.2f} ETH",
        f"Net Buy Volume:    {float(total_buy_eth - total_sell_eth):>15,.2f} ETH",
        "",
        f"Avg Daily Price:   {avg_price:>15,.6f} ETH/OP",
    ]

    return "\n".join(lines)


def format_full_report(result: AnalysisResult) -> str:
    """Format complete analysis report."""
    sections = [
        "=" * 70,
        "HYPOTHETICAL OP BUYBACK ANALYSIS",
        f"Period: {result.period_start} to {result.period_end}",
        f"Pool: {result.pool.project} ({result.pool.pool_address[:10]}...)",
        "=" * 70,
        "",
        format_revenue_summary(result.daily_revenues),
        "",
        format_trading_summary(result.daily_aggregates),
        "",
        format_strategy_summary(result.strategy_comparison),
    ]

    return "\n".join(sections)


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Run analysis and print report."""
    print("Running OP buyback analysis...")
    print("(Using mock data - replace with real RPC/Dune data for production)")
    print()

    result = run_analysis()
    report = format_full_report(result)
    print(report)

    # Monte Carlo analysis
    print()
    print("=" * 70)
    print("MONTE CARLO ANALYSIS (Random Strategy)")
    print("=" * 70)
    print("\nRunning 1000 random simulations...")

    mc_results = monte_carlo_random_strategy(
        result.daily_revenues,
        result.daily_aggregates,
        num_simulations=1000
    )

    mc_stats = analyze_monte_carlo(mc_results)
    print(f"\nRandom Strategy Distribution:")
    print(f"  Min OP:    {float(mc_stats['min_op']):>15,.2f}")
    print(f"  P5 OP:     {float(mc_stats['p5_op']):>15,.2f}")
    print(f"  Median OP: {float(mc_stats['median_op']):>15,.2f}")
    print(f"  Mean OP:   {float(mc_stats['mean_op']):>15,.2f}")
    print(f"  P95 OP:    {float(mc_stats['p95_op']):>15,.2f}")
    print(f"  Max OP:    {float(mc_stats['max_op']):>15,.2f}")
    print(f"  Std Dev:   {float(mc_stats['std_op']):>15,.2f}")


if __name__ == "__main__":
    main()
