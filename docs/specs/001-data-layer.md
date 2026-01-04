# 001: Data Layer - Historical OHLC & On-Chain Data Storage

## Overview

Module for fetching, storing, and retrieving historical on-chain data for backtesting and live operation.

## Data Sources

| Chain | Pool | Fee Tier |
|-------|------|----------|
| Ethereum | ETH/USDC | 0.05% |
| Ethereum | ETH/USDC | 0.3% |
| Base | ETH/USDC | 0.05% |
| Base | ETH/USDC | 0.3% |

Full history since each pool's launch block.

## Data Points Required

### Per-Block Data
- `block_number`: uint64
- `timestamp`: uint64
- `sqrt_price_x96`: uint256
- `tick`: int24
- `liquidity`: uint128
- `token0_balance`: uint256 (pool's ETH balance)
- `token1_balance`: uint256 (pool's USDC balance)

### Aggregated OHLC
- `period_start`: timestamp
- `open`: decimal
- `high`: decimal
- `low`: decimal
- `close`: decimal
- `volume_token0`: decimal
- `volume_token1`: decimal
- `vwap`: decimal
- `num_swaps`: uint32

Aggregation periods: 1 block, 1 minute, 5 minute, 1 hour

### LP Events
- Mint (liquidity added)
- Burn (liquidity removed)
- Swap (trades)
- Collect (fee collection)

Each with: `block_number`, `tx_hash`, `tick_lower`, `tick_upper`, `amount0`, `amount1`, `liquidity_delta`

### Tick Liquidity Snapshots
- Liquidity available at each initialized tick
- Snapshotted hourly or on significant change

## Module Structure

```
src/
  data/
    types.py          # Immutable dataclasses for all data types
    fetchers/
      __init__.py
      base.py         # Abstract fetcher protocol
      rpc.py          # Direct RPC fetcher (eth_call, eth_getLogs)
    storage/
      __init__.py
      base.py         # Abstract storage protocol
      sqlite.py       # Local SQLite for dev/testing
      json_store.py   # Simple JSON file storage
      s3.py           # S3 for production cold storage
    aggregators.py    # Pure functions: raw events -> OHLC
    backfill.py       # Historical data backfill orchestration
```

## Functional Programming Requirements

### Pure Functions
All data transformation functions must be pure:
```python
# Good - pure function
def aggregate_to_ohlc(swaps: Sequence[Swap], period: timedelta) -> Sequence[OHLC]:
    ...

# Bad - side effects
def aggregate_to_ohlc(swaps: Sequence[Swap], period: timedelta) -> Sequence[OHLC]:
    logger.info("Aggregating...")  # Side effect
    ...
```

### Immutable Data Structures
Use frozen dataclasses or NamedTuples:
```python
from dataclasses import dataclass

@dataclass(frozen=True)
class OHLC:
    period_start: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    vwap: Decimal
```

### Result Types for Error Handling
No exceptions for expected failures:
```python
from typing import Union
from dataclasses import dataclass

@dataclass(frozen=True)
class FetchError:
    source: str
    message: str
    retryable: bool

Result = Union[T, FetchError]

def fetch_block_data(block: int) -> Result[BlockData]:
    ...
```

### Separation of I/O
Fetchers and storage are the only modules that perform I/O. All other modules are pure transformations.

## Storage Requirements

### Schema Design
- Partitioned by chain + pool address
- Indexed by block_number (primary) and timestamp (secondary)
- Support efficient range queries

### Query Patterns
| Query | Expected Latency |
|-------|------------------|
| Latest N candles | < 50ms |
| Block range (1000 blocks) | < 100ms |
| Full day OHLC | < 200ms |
| Tick liquidity snapshot | < 100ms |

### Backfill Strategy
1. Parallel fetch by block range chunks (10,000 blocks per worker)
2. Idempotent writes (re-running backfill is safe)
3. Progress checkpointing (resume from failure)
4. Validation checksums against known block data

## Testing Requirements

### Unit Tests
- [ ] OHLC aggregation from mock swap events
- [ ] VWAP calculation correctness
- [ ] Tick liquidity computation
- [ ] Data type serialization/deserialization
- [ ] Error handling paths

### Integration Tests
- [ ] Fetcher against historical block (deterministic)
- [ ] Storage round-trip (write → read → compare)
- [ ] Backfill for small block range

### Property-Based Tests
- [ ] OHLC: high >= max(open, close), low <= min(open, close)
- [ ] VWAP is within [low, high]
- [ ] Aggregation is associative (chunk then merge == single pass)

## Dependencies

```
# Core
python = "^3.11"

# Data
requests = "^2.32"      # HTTP for RPC calls

# Storage (optional, for production)
boto3 = "^1.28"         # S3
```

## Acceptance Criteria

- [x] All data types defined as frozen dataclasses
- [x] RPC fetcher implemented (see poc/simple_poc.py)
- [ ] JSON storage adapter working for local dev
- [ ] Backfill script can populate 10,000 blocks
- [ ] 90%+ unit test coverage on pure functions
- [ ] Integration test passes against mainnet historical data

## References

- [Uniswap v3 Core](https://github.com/Uniswap/v3-core)
- [onchain-pricing repo](https://github.com/charliemarketplace/onchain-pricing)
