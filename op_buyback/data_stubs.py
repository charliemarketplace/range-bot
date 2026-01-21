"""
Data loading stubs - mock data for development.

These functions return realistic mock data. Replace with actual
RPC/Dune calls for production.

All functions are pure - no side effects, deterministic with seed.
"""
import random
from decimal import Decimal
from datetime import date, timedelta
from typing import Sequence

from .types import (
    DailyRevenue,
    PoolMeta,
    PoolComparison,
    Swap,
)


# =============================================================================
# Scope A: Sequencer Revenue (Mock Dune Data)
# =============================================================================

def generate_daily_revenues(
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    seed: int = 42
) -> tuple[DailyRevenue, ...]:
    """
    Generate mock daily sequencer revenue data.

    Real implementation: Dune query on optimism.transactions
    """
    random.seed(seed)

    revenues = []
    current = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    # Base revenue with some growth and weekly seasonality
    base_l1 = 15.0  # ETH per day
    base_l2 = 25.0  # ETH per day

    day_num = 0
    while current <= end:
        # Weekly seasonality (lower on weekends)
        weekday_mult = 0.7 if current.weekday() >= 5 else 1.0

        # Gradual growth over year
        growth_mult = 1.0 + (day_num / 365) * 0.3

        # Random noise
        noise_l1 = random.gauss(1.0, 0.15)
        noise_l2 = random.gauss(1.0, 0.2)

        l1_fee = Decimal(str(round(base_l1 * weekday_mult * growth_mult * noise_l1, 4)))
        l2_fee = Decimal(str(round(base_l2 * weekday_mult * growth_mult * noise_l2, 4)))

        # Tx count correlates with fees
        tx_count = int((float(l1_fee) + float(l2_fee)) * 50000 * random.gauss(1.0, 0.1))

        revenues.append(DailyRevenue(
            date=current.isoformat(),
            l1_fee_eth=max(Decimal("0.1"), l1_fee),
            l2_fee_eth=max(Decimal("0.1"), l2_fee),
            tx_count=max(100000, tx_count)
        ))

        current += timedelta(days=1)
        day_num += 1

    return tuple(revenues)


# =============================================================================
# Scope B: Pool Selection (Mock Dune Data)
# =============================================================================

def get_pool_comparisons(seed: int = 42) -> tuple[PoolComparison, ...]:
    """
    Get comparison of OP/ETH pools.

    Real implementation: Dune query comparing Velodrome vs Uniswap v3
    """
    return (
        PoolComparison(
            project="velodrome",
            pool_address="0x0df083de449f75691fc5a36477a6f3284c269108",
            total_volume_eth=Decimal("125000.5"),
            trade_count=89000,
            avg_daily_volume_eth=Decimal("342.5")
        ),
        PoolComparison(
            project="uniswap_v3",
            pool_address="0x68f5c0a2de713a54991e01858fd27a3832401849",
            total_volume_eth=Decimal("85000.3"),
            trade_count=52000,
            avg_daily_volume_eth=Decimal("232.9")
        ),
    )


def select_deepest_pool(comparisons: Sequence[PoolComparison]) -> PoolComparison:
    """Select pool with highest total volume."""
    return max(comparisons, key=lambda p: p.total_volume_eth)


def get_pool_meta(pool_address: str) -> PoolMeta:
    """
    Get metadata for a pool.

    Real implementation: RPC calls to get token info
    """
    # Mock data for known pools
    pools = {
        "0x0df083de449f75691fc5a36477a6f3284c269108": PoolMeta(
            pool_address="0x0df083de449f75691fc5a36477a6f3284c269108",
            project="velodrome",
            token0="0x4200000000000000000000000000000000000042",  # OP
            token1="0x4200000000000000000000000000000000000006",  # WETH
            token0_symbol="OP",
            token1_symbol="WETH",
            token0_decimals=18,
            token1_decimals=18,
        ),
        "0x68f5c0a2de713a54991e01858fd27a3832401849": PoolMeta(
            pool_address="0x68f5c0a2de713a54991e01858fd27a3832401849",
            project="uniswap_v3",
            token0="0x4200000000000000000000000000000000000042",  # OP
            token1="0x4200000000000000000000000000000000000006",  # WETH
            token0_symbol="OP",
            token1_symbol="WETH",
            token0_decimals=18,
            token1_decimals=18,
        ),
    }
    return pools.get(pool_address, pools[list(pools.keys())[0]])


# =============================================================================
# Scope C: Swap Events (Mock RPC Data)
# =============================================================================

def generate_swaps(
    pool_meta: PoolMeta,
    start_date: str = "2025-01-01",
    end_date: str = "2025-12-31",
    swaps_per_day: int = 250,
    seed: int = 42
) -> tuple[Swap, ...]:
    """
    Generate mock swap events.

    Real implementation: RPC eth_getLogs for Swap events

    Price model:
    - OP starts around 1.5 USD, ETH around 3500 USD
    - OP/ETH ratio ~= 0.00043 (1 ETH = ~2300 OP)
    - Random walk with mean reversion
    """
    random.seed(seed)

    swaps = []

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    current = start

    # Starting price: ~2300 OP per ETH
    op_per_eth = 2300.0

    block_number = 115000000  # Starting block (approximate for 2025)
    log_index = 0

    while current <= end:
        # Timestamp for start of day
        day_timestamp = int((current - date(1970, 1, 1)).total_seconds())

        # Number of swaps varies by day
        day_swaps = int(swaps_per_day * random.gauss(1.0, 0.3))
        day_swaps = max(50, day_swaps)

        for i in range(day_swaps):
            # Time within day (spread across 24 hours)
            swap_timestamp = day_timestamp + int(86400 * i / day_swaps)

            # Price random walk with mean reversion
            price_change = random.gauss(0, 0.01) + 0.001 * (2300 - op_per_eth)
            op_per_eth = max(1500, min(3500, op_per_eth * (1 + price_change)))

            # Random trade size (in ETH)
            # Most trades small, some large
            if random.random() < 0.05:
                eth_amount = random.uniform(10, 100)  # Large trade
            else:
                eth_amount = random.uniform(0.1, 5)  # Normal trade

            op_amount = eth_amount * op_per_eth

            # Randomly buy or sell OP
            is_buy = random.random() < 0.5

            if pool_meta.op_is_token0:
                if is_buy:
                    # Buying OP: spend ETH (negative), receive OP (positive)
                    amount0 = int(op_amount * 10**18)
                    amount1 = -int(eth_amount * 10**18)
                else:
                    # Selling OP: spend OP (negative), receive ETH (positive)
                    amount0 = -int(op_amount * 10**18)
                    amount1 = int(eth_amount * 10**18)
            else:
                if is_buy:
                    amount0 = -int(eth_amount * 10**18)
                    amount1 = int(op_amount * 10**18)
                else:
                    amount0 = int(eth_amount * 10**18)
                    amount1 = -int(op_amount * 10**18)

            swaps.append(Swap(
                block_number=block_number,
                tx_hash=f"0x{random.randbytes(32).hex()}",
                log_index=log_index,
                amount0=amount0,
                amount1=amount1,
                timestamp=swap_timestamp,
            ))

            block_number += random.randint(1, 5)
            log_index = (log_index + 1) % 100

        current += timedelta(days=1)

    return tuple(swaps)


# =============================================================================
# Data Loading Interface (to be replaced with real implementations)
# =============================================================================

def load_revenues_from_csv(path: str) -> tuple[DailyRevenue, ...]:
    """Load revenues from CSV file. Stub - returns mock data."""
    return generate_daily_revenues()


def load_swaps_from_sqlite(path: str, pool_address: str) -> tuple[Swap, ...]:
    """Load swaps from SQLite. Stub - returns mock data."""
    pool_meta = get_pool_meta(pool_address)
    return generate_swaps(pool_meta)


def load_pool_meta_from_sqlite(path: str, pool_address: str) -> PoolMeta:
    """Load pool metadata from SQLite. Stub - returns mock data."""
    return get_pool_meta(pool_address)
