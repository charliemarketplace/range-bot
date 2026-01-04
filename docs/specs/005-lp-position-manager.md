# 005: LP Position Manager - Uniswap v3 Execution

## Overview

Module for managing Uniswap v3 LP positions on-chain. Handles position creation, modification, fee collection, and rebalancing. This is the execution layer that translates Bayesian recommendations into on-chain actions.

## Core Concept

```
Bayesian Engine outputs: RangeRecommendation(lower_tick, upper_tick, ...)
Position Manager:
  1. Reads current position state
  2. Computes delta (what needs to change)
  3. Executes transactions (burn old, mint new)
  4. Handles gas, slippage, MEV protection

Constraints:
  - Minimize gas costs (don't rebalance for tiny improvements)
  - Minimize slippage (use appropriate deadline, slippage tolerance)
  - MEV protection (private mempool or flashbots where available)
```

## Module Structure

```
src/
  position/
    __init__.py
    types.py           # Position and transaction types
    reader.py          # Read on-chain position state
    calculator.py      # Compute liquidity amounts, fees
    executor.py        # Execute transactions
    rebalancer.py      # Decide when/how to rebalance
    gas.py             # Gas estimation and optimization
    simulation.py      # Dry-run / simulation mode
```

## Data Types

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Sequence
from enum import Enum

@dataclass(frozen=True)
class PoolConfig:
    """Uniswap v3 pool configuration."""
    chain_id: int
    pool_address: str
    token0: str                    # e.g., WETH address
    token1: str                    # e.g., USDC address
    fee_tier: int                  # 500, 3000, or 10000 (bps)
    tick_spacing: int              # 10, 60, or 200

@dataclass(frozen=True)
class PositionState:
    """Current state of an LP position."""
    token_id: int                  # NFT token ID (0 if no position)
    owner: str                     # Owner address
    pool: PoolConfig
    tick_lower: int
    tick_upper: int
    liquidity: int                 # Current liquidity units
    tokens_owed_0: Decimal         # Uncollected fees in token0
    tokens_owed_1: Decimal         # Uncollected fees in token1
    token0_amount: Decimal         # Current token0 in position
    token1_amount: Decimal         # Current token1 in position

    @property
    def is_in_range(self) -> bool:
        """Is current price within position range?"""
        ...

    @property
    def total_value_usd(self) -> Decimal:
        """Total value including uncollected fees."""
        ...

@dataclass(frozen=True)
class PoolState:
    """Current state of the pool."""
    sqrt_price_x96: int
    tick: int
    liquidity: int                 # Active liquidity
    fee_growth_global_0: int
    fee_growth_global_1: int

    @property
    def price(self) -> Decimal:
        """Current price derived from sqrtPriceX96."""
        ...

@dataclass(frozen=True)
class RebalanceAction:
    """Describes a rebalance operation."""
    action_type: str              # "mint", "burn", "burn_and_mint", "collect_only", "none"
    current_position: Optional[PositionState]
    target_tick_lower: int
    target_tick_upper: int
    target_liquidity: int
    estimated_gas: int
    estimated_slippage_bps: int
    reason: str                   # Why this rebalance is recommended

@dataclass(frozen=True)
class TransactionParams:
    """Parameters for executing a transaction."""
    to: str
    data: bytes
    value: int                    # ETH value to send
    gas_limit: int
    max_fee_per_gas: int
    max_priority_fee_per_gas: int
    deadline: int                 # Unix timestamp

@dataclass(frozen=True)
class TransactionResult:
    """Result of a transaction execution."""
    success: bool
    tx_hash: Optional[str]
    block_number: Optional[int]
    gas_used: Optional[int]
    error: Optional[str]
    # Position state changes
    liquidity_delta: int
    token0_delta: Decimal
    token1_delta: Decimal

class RebalanceDecision(Enum):
    REBALANCE = "rebalance"
    HOLD = "hold"
    COLLECT_ONLY = "collect_only"
```

## Core Functions

### Position Reading

```python
async def get_position_state(
    token_id: int,
    pool: PoolConfig,
    provider: Web3Provider
) -> PositionState:
    """
    Read current position state from chain.

    Calls:
    - NonfungiblePositionManager.positions(tokenId)
    - Computes tokens owed from fee growth
    """
    ...

async def get_pool_state(
    pool: PoolConfig,
    provider: Web3Provider
) -> PoolState:
    """
    Read current pool state.

    Calls:
    - Pool.slot0() for price and tick
    - Pool.liquidity() for active liquidity
    """
    ...

async def get_all_positions(
    owner: str,
    pool: PoolConfig,
    provider: Web3Provider
) -> Sequence[PositionState]:
    """Get all positions owned by address in this pool."""
    ...
```

### Liquidity Calculations

```python
def calculate_liquidity_for_amounts(
    sqrt_price_x96: int,
    tick_lower: int,
    tick_upper: int,
    amount0_desired: int,
    amount1_desired: int
) -> int:
    """
    Calculate liquidity from desired token amounts.

    Uses Uniswap v3 liquidity math.
    Pure function.
    """
    ...

def calculate_amounts_for_liquidity(
    sqrt_price_x96: int,
    tick_lower: int,
    tick_upper: int,
    liquidity: int
) -> tuple[int, int]:
    """
    Calculate token amounts from liquidity.

    Returns (amount0, amount1).
    Pure function.
    """
    ...

def calculate_optimal_liquidity(
    available_token0: Decimal,
    available_token1: Decimal,
    sqrt_price_x96: int,
    tick_lower: int,
    tick_upper: int
) -> tuple[int, Decimal, Decimal]:
    """
    Calculate maximum liquidity given available tokens.

    Returns (liquidity, token0_used, token1_used).
    May leave some of one token unused depending on price/range.
    """
    ...
```

### Rebalancing Logic

```python
def should_rebalance(
    current: PositionState,
    recommendation: RangeRecommendation,
    pool_state: PoolState,
    config: RebalanceConfig
) -> tuple[RebalanceDecision, str]:
    """
    Decide whether to rebalance.

    Factors:
    1. Is current range significantly different from recommended?
    2. Is the cost (gas + slippage) worth the improvement?
    3. Is there urgency (price near edge, high fees uncollected)?

    Returns decision and reason.
    """
    # Don't rebalance for tiny improvements
    tick_delta = abs(current.tick_lower - recommendation.lower_tick) + \
                 abs(current.tick_upper - recommendation.upper_tick)

    if tick_delta < config.min_tick_delta:
        return RebalanceDecision.HOLD, f"Tick delta {tick_delta} below threshold"

    # Check if price is about to leave range
    ticks_to_lower = pool_state.tick - current.tick_lower
    ticks_to_upper = current.tick_upper - pool_state.tick

    if ticks_to_lower < config.urgency_tick_buffer or \
       ticks_to_upper < config.urgency_tick_buffer:
        return RebalanceDecision.REBALANCE, "Price near range boundary"

    # Check gas economics
    estimated_improvement = estimate_fee_improvement(current, recommendation, pool_state)
    estimated_cost = estimate_rebalance_cost(current, recommendation, pool_state)

    if estimated_improvement > estimated_cost * config.min_improvement_ratio:
        return RebalanceDecision.REBALANCE, f"Improvement {estimated_improvement} > cost {estimated_cost}"

    # Check if fees worth collecting
    if current.tokens_owed_0 + current.tokens_owed_1 > config.min_collect_value:
        return RebalanceDecision.COLLECT_ONLY, "Collecting accumulated fees"

    return RebalanceDecision.HOLD, "No action needed"

@dataclass(frozen=True)
class RebalanceConfig:
    """Configuration for rebalancing decisions."""
    min_tick_delta: int = 20           # Minimum tick change to rebalance
    urgency_tick_buffer: int = 50      # Rebalance if price within N ticks of edge
    min_improvement_ratio: float = 2.0 # Improvement must be 2x cost
    min_collect_value: Decimal = Decimal("10")  # Minimum fees to trigger collect
    max_slippage_bps: int = 50         # Maximum allowed slippage
    deadline_seconds: int = 300        # Transaction deadline
```

### Transaction Building

```python
def build_mint_params(
    pool: PoolConfig,
    tick_lower: int,
    tick_upper: int,
    amount0_desired: int,
    amount1_desired: int,
    amount0_min: int,
    amount1_min: int,
    recipient: str,
    deadline: int
) -> dict:
    """Build parameters for NonfungiblePositionManager.mint()."""
    ...

def build_increase_liquidity_params(
    token_id: int,
    amount0_desired: int,
    amount1_desired: int,
    amount0_min: int,
    amount1_min: int,
    deadline: int
) -> dict:
    """Build parameters for increaseLiquidity()."""
    ...

def build_decrease_liquidity_params(
    token_id: int,
    liquidity: int,
    amount0_min: int,
    amount1_min: int,
    deadline: int
) -> dict:
    """Build parameters for decreaseLiquidity()."""
    ...

def build_collect_params(
    token_id: int,
    recipient: str,
    amount0_max: int = 2**128 - 1,
    amount1_max: int = 2**128 - 1
) -> dict:
    """Build parameters for collect()."""
    ...

def build_multicall(calls: Sequence[bytes], deadline: int) -> bytes:
    """
    Combine multiple calls into single multicall.

    Useful for: decreaseLiquidity + collect + mint in one tx.
    """
    ...
```

### Transaction Execution

```python
async def execute_rebalance(
    action: RebalanceAction,
    wallet: Wallet,
    provider: Web3Provider,
    config: ExecutionConfig
) -> TransactionResult:
    """
    Execute a rebalance action.

    Steps:
    1. Build transaction calldata
    2. Estimate gas
    3. Submit transaction (with MEV protection if configured)
    4. Wait for confirmation
    5. Verify state change
    """
    ...

async def simulate_rebalance(
    action: RebalanceAction,
    provider: Web3Provider
) -> SimulationResult:
    """
    Simulate rebalance without executing.

    Uses eth_call to preview results.
    Returns expected state changes.
    """
    ...

@dataclass(frozen=True)
class ExecutionConfig:
    """Configuration for transaction execution."""
    use_flashbots: bool = False        # Use Flashbots for MEV protection
    max_gas_price_gwei: int = 100      # Circuit breaker
    confirmation_blocks: int = 2       # Blocks to wait for confirmation
    simulation_required: bool = True   # Always simulate first
```

### Gas Optimization

```python
async def estimate_gas(
    tx_params: TransactionParams,
    provider: Web3Provider
) -> int:
    """Estimate gas for transaction."""
    ...

async def get_optimal_gas_price(
    provider: Web3Provider,
    urgency: str = "medium"  # "low", "medium", "high"
) -> tuple[int, int]:
    """
    Get optimal (max_fee, priority_fee) for current conditions.

    Uses EIP-1559 fee estimation.
    """
    ...

def estimate_rebalance_cost(
    current: PositionState,
    recommendation: RangeRecommendation,
    pool_state: PoolState
) -> Decimal:
    """
    Estimate total cost of rebalancing in USD.

    Includes:
    - Gas cost
    - Expected slippage
    - Price impact
    """
    ...
```

## Safety Features

### Circuit Breakers

```python
def check_safety_conditions(
    action: RebalanceAction,
    pool_state: PoolState,
    config: SafetyConfig
) -> tuple[bool, Optional[str]]:
    """
    Check if it's safe to execute.

    Conditions:
    - Gas price below maximum
    - Slippage within tolerance
    - Pool liquidity sufficient
    - No unusual price movement

    Returns (is_safe, reason_if_not).
    """
    ...

@dataclass(frozen=True)
class SafetyConfig:
    max_gas_price_gwei: int = 100
    max_slippage_bps: int = 100
    min_pool_liquidity: int = 1_000_000
    max_price_change_1h_pct: float = 5.0
```

### Simulation Mode

```python
class SimulationExecutor:
    """
    Executor that simulates but doesn't submit transactions.

    Useful for:
    - Backtesting
    - Dry runs
    - Testing
    """

    async def execute(self, action: RebalanceAction) -> TransactionResult:
        """Simulate execution, return expected result."""
        ...
```

## Testing Requirements

### Unit Tests
- [ ] Liquidity calculations match reference implementation
- [ ] Tick/price conversions are correct
- [ ] Multicall encoding is correct
- [ ] Slippage calculations are accurate
- [ ] Gas estimation is reasonable

### Integration Tests
- [ ] Read position state from mainnet fork
- [ ] Simulate rebalance on fork
- [ ] Full rebalance on testnet
- [ ] Multicall execution works

### Property-Based Tests
- [ ] Liquidity math: amounts → liquidity → amounts roundtrips
- [ ] Tick bounds are always valid (lower < upper, on spacing)

### Safety Tests
- [ ] Circuit breakers trigger on high gas
- [ ] Slippage protection works
- [ ] Deadline enforcement works

## Dependencies

```python
web3 = "^6.0"            # Ethereum interaction
eth-abi = "^4.0"         # ABI encoding
eth-account = "^0.9"     # Transaction signing

# Optional MEV protection
flashbots = "^2.0"       # Flashbots bundle submission
```

## Contract Addresses

```python
ADDRESSES = {
    "ethereum": {
        "position_manager": "0xC36442b4a4522E871399CD717aBDD847Ab11FE88",
        "factory": "0x1F98431c8aD98523631AE4a59f267346ea31F984",
        "quoter_v2": "0x61fFE014bA17989E743c5F6cB21bF9697530B21e",
    },
    "base": {
        "position_manager": "0x03a520b32C04BF3bEEf7BEb72E919cf822Ed34f1",
        "factory": "0x33128a8fC17869897dcE68Ed026d694621f6FDfD",
        "quoter_v2": "0x3d4e44Eb1374240CE5F1B871ab261CD16335B76a",
    }
}
```

## Acceptance Criteria

- [ ] Position state reading from both chains
- [ ] Liquidity calculations match Uniswap SDK
- [ ] Rebalance decision logic with configurable thresholds
- [ ] Transaction building for all operations
- [ ] Simulation mode for testing
- [ ] Gas estimation and optimization
- [ ] Safety circuit breakers
- [ ] 90%+ test coverage
- [ ] Successful testnet rebalance

## References

- [Uniswap v3 Periphery](https://github.com/Uniswap/v3-periphery)
- [NonfungiblePositionManager](https://docs.uniswap.org/contracts/v3/reference/periphery/NonfungiblePositionManager)
- [Liquidity Math](https://uniswap.org/whitepaper-v3.pdf) (Section 6)
