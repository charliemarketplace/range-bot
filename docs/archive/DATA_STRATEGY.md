# Data Collection Strategy

**No external paid services** - we collect and store our own data.

## Data Sources (Ranked by Reliability)

### 1. Direct RPC (Primary)

**Best for**: Real-time data, block-level granularity, full control

| Provider | Free Tier | Rate Limit | Archive Access |
|----------|-----------|------------|----------------|
| [PublicNode](https://ethereum-rpc.publicnode.com) | Unlimited | ~50 RPS shared | Limited |
| [Ankr](https://rpc.ankr.com/eth) | Unlimited | ~30 RPS | No |
| [DRPC](https://eth.drpc.org) | Generous | ~100 RPS | Partial |
| [LlamaNodes](https://eth.llamarpc.com) | Unlimited | Variable | No |
| [Alchemy](https://www.alchemy.com) | 300M CU/month | Varies | Yes (paid) |
| [Infura](https://infura.io) | 100K req/day | 10 RPS | Yes (paid) |

**RPC Methods We Need:**
```
eth_call          # Read pool state (slot0, liquidity)
eth_getLogs       # Fetch Swap events
eth_blockNumber   # Current block
eth_getBlockByNumber  # Block timestamps
```

**Limitations:**
- Free tiers don't have archive nodes (historical blocks)
- Need archive access for blocks > ~128 back
- **Solution**: Use Alchemy/Infura free tier for backfill, public RPCs for live

### 2. Uniswap Subgraph (Secondary)

**Best for**: Historical aggregated data, OHLC, easier queries

**Endpoints:**
- Decentralized: `https://gateway.thegraph.com/api/[KEY]/subgraphs/id/5zvR82...`
- Hosted (deprecated but works): `https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3`

**Available Data:**
| Entity | Fields | Use Case |
|--------|--------|----------|
| `pool` | sqrtPrice, tick, liquidity, TVL | Current state |
| `poolHourData` | open, high, low, close, volume | Hourly OHLC |
| `poolDayData` | Same as hourly | Daily OHLC |
| `swaps` | amounts, sqrtPriceX96, tick | VWAP calculation |
| `ticks` | liquidityGross, liquidityNet | Liquidity distribution |

**Cost:**
- Hosted: Free (being deprecated)
- Decentralized: Requires GRT tokens (~$0.0001 per query)

**Note**: Subgraph data has ~10-30 second delay from chain.

### 3. Dune Analytics (Backup)

**Best for**: One-time historical backfills, complex queries

- Free tier: 2,500 credits/month
- Can export CSVs for historical data
- Has pre-built Uniswap tables

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      Data Collection Flow                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  BACKFILL (One-time)                                            │
│  ══════════════════                                             │
│                                                                  │
│  Alchemy/Infura (archive) ──► Historical Swaps ──► SQLite/S3   │
│                                 (eth_getLogs)                    │
│                                                                  │
│  Subgraph ──► poolHourData ──► OHLC Candles ──► SQLite/S3      │
│               poolDayData                                        │
│                                                                  │
│  ─────────────────────────────────────────────────────────────  │
│                                                                  │
│  LIVE (Ongoing)                                                 │
│  ═════════════                                                  │
│                                                                  │
│  PublicNode/Ankr ──► Current Pool State ──► Lambda             │
│  (hourly poll)       (slot0, liquidity)                         │
│                                                                  │
│  Subgraph ──► Last 24h OHLC ──► Lambda                         │
│  (hourly poll)                                                   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Backfill Historical Data

```python
# Strategy: Fetch swap events in chunks, compute OHLC ourselves

CHUNK_SIZE = 2000  # Blocks per getLogs call
START_BLOCK = 12369621  # Uniswap v3 deployment
END_BLOCK = current_block

for from_block in range(START_BLOCK, END_BLOCK, CHUNK_SIZE):
    to_block = min(from_block + CHUNK_SIZE - 1, END_BLOCK)

    # Fetch swap events
    swaps = eth_getLogs(pool, SWAP_TOPIC, from_block, to_block)

    # Store raw events
    storage.save_swaps(swaps)

    # Rate limit: ~1 request per second to stay under limits
    time.sleep(1)
```

**Estimated backfill time:**
- Ethereum mainnet: ~20M blocks / 2000 = 10,000 requests
- At 1 RPS = ~3 hours
- At 10 RPS = ~17 minutes

### Phase 2: Compute Aggregates

```python
# After backfill, compute OHLC from raw swaps

def aggregate_swaps_to_ohlc(swaps: list[Swap], period: timedelta) -> list[OHLC]:
    """Group swaps by time period and compute OHLC."""
    grouped = group_by_period(swaps, period)

    ohlc_data = []
    for period_start, period_swaps in grouped.items():
        prices = [swap_to_price(s) for s in period_swaps]
        volumes = [abs(s.amount0) for s in period_swaps]

        ohlc_data.append(OHLC(
            period_start=period_start,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=sum(volumes),
            vwap=compute_vwap(prices, volumes),
            num_swaps=len(period_swaps)
        ))

    return ohlc_data
```

### Phase 3: Live Data Collection

```python
# Lambda runs hourly

async def collect_live_data():
    # 1. Current pool state via RPC (fastest)
    pool_state = await get_pool_state_rpc(PUBLIC_RPC)

    # 2. Last 24h OHLC via subgraph (pre-aggregated)
    ohlc_24h = await get_pool_hour_data(SUBGRAPH, hours=24)

    # 3. Store new data points
    await storage.append_pool_state(pool_state)
    await storage.update_ohlc(ohlc_24h)

    return pool_state, ohlc_24h
```

## Storage Schema

### SQLite (Local Development)

```sql
CREATE TABLE swaps (
    id TEXT PRIMARY KEY,
    block_number INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    tx_hash TEXT NOT NULL,
    amount0 TEXT NOT NULL,
    amount1 TEXT NOT NULL,
    sqrt_price_x96 TEXT NOT NULL,
    liquidity TEXT NOT NULL,
    tick INTEGER NOT NULL,
    pool_address TEXT NOT NULL,
    chain TEXT NOT NULL
);

CREATE INDEX idx_swaps_block ON swaps(block_number);
CREATE INDEX idx_swaps_pool_time ON swaps(pool_address, timestamp);

CREATE TABLE ohlc (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    period_start INTEGER NOT NULL,
    period_seconds INTEGER NOT NULL,  -- 3600 for hourly, 86400 for daily
    open TEXT NOT NULL,
    high TEXT NOT NULL,
    low TEXT NOT NULL,
    close TEXT NOT NULL,
    volume TEXT NOT NULL,
    vwap TEXT NOT NULL,
    num_swaps INTEGER NOT NULL,
    pool_address TEXT NOT NULL,
    chain TEXT NOT NULL,
    UNIQUE(pool_address, period_start, period_seconds)
);

CREATE INDEX idx_ohlc_pool_time ON ohlc(pool_address, period_start);

CREATE TABLE pool_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    block_number INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    sqrt_price_x96 TEXT NOT NULL,
    tick INTEGER NOT NULL,
    liquidity TEXT NOT NULL,
    pool_address TEXT NOT NULL,
    chain TEXT NOT NULL,
    UNIQUE(pool_address, block_number)
);
```

### S3 (Production Cold Storage)

```
s3://range-bot-data/
├── swaps/
│   ├── ethereum/
│   │   ├── 0x88e6a0c2.../              # Pool address
│   │   │   ├── 2024/01/swaps.parquet
│   │   │   ├── 2024/02/swaps.parquet
│   │   │   └── ...
│   └── base/
│       └── ...
├── ohlc/
│   ├── ethereum/
│   │   └── 0x88e6a0c2.../
│   │       ├── hourly.parquet
│   │       └── daily.parquet
│   └── base/
└── snapshots/
    └── ...
```

## Cost Estimates

### Free Tier Limits

| Service | Free Limit | Our Usage | Sufficient? |
|---------|------------|-----------|-------------|
| Alchemy | 300M CU/month | ~50K calls for backfill | ✓ |
| Infura | 100K req/day | ~10K/day backfill | ✓ |
| PublicNode | Unlimited | ~24 calls/day live | ✓ |
| The Graph | Need GRT | ~24 calls/day | ✓ ($0.01/month) |

### Storage Costs

| Storage | Data Size | Monthly Cost |
|---------|-----------|--------------|
| S3 (raw swaps) | ~5GB | $0.12 |
| S3 (OHLC) | ~100MB | $0.00 |
| DynamoDB (hot state) | <1GB | $0.25 |
| **Total** | | **<$1/month** |

## Fallback Strategy

If primary source fails:

1. **RPC fails** → Fall back to subgraph
2. **Subgraph fails** → Use cached data (up to 1 hour stale)
3. **All fail** → Alert, skip this run, use last known state

```python
async def get_data_with_fallback():
    try:
        return await get_from_rpc(PUBLIC_RPC)
    except Exception:
        pass

    try:
        return await get_from_subgraph(SUBGRAPH)
    except Exception:
        pass

    # Use cache
    cached = await storage.get_latest_cached()
    if cached and (now - cached.timestamp) < timedelta(hours=1):
        return cached

    raise DataUnavailableError("All sources failed, cache too stale")
```

## Testing the Strategy

Run the POC locally:

```bash
uv run python poc/simple_poc.py
```

This fetches real swap data from Ethereum mainnet, stores locally, and runs Bayesian analysis.

Output files:
- `poc/data/swaps.json` - Raw swap events
- `poc/data/ohlc.json` - Aggregated candles
- `poc/data/results.json` - Analysis results

### Visualize the Data

Open `poc/viewer.html` in a browser to see:
- OHLC candlestick chart
- Individual swap prices (scatter)
- VWAP line overlay
- Recommended 90% range band
- Current price marker

## Next Steps

1. [x] Test RPC endpoints (PublicNode works)
2. [x] Fetch real swap data
3. [x] Compute OHLC and VWAP
4. [x] Run Bayesian analysis
5. [ ] Set up SQLite schema for persistent storage
6. [ ] Implement fallback logic
7. [ ] Add Base chain support

## Retroactive LP Assessment

For retroactive analysis of hypothetical LP positions over historical swaps, see:
- **R implementation**: [charliemarketplace/uniswap](https://github.com/charliemarketplace/uniswap) - Uniswap v3 math in R for optimal position identification

### Data Requirements for Retroactive Assessment

To calculate fees earned by a hypothetical position across N swaps:

1. **Swap data with tick**: Each swap includes the `tick` at execution
2. **Position range**: `[tick_lower, tick_upper]` defining the LP position
3. **Liquidity at position**: Your liquidity vs total liquidity in that tick range

**Fee calculation per swap:**
```
if tick_lower <= swap.tick <= tick_upper:
    fee_amount = swap.volume * fee_tier * (your_liquidity / total_liquidity_in_range)
else:
    fee_amount = 0  # Position out of range, no fees earned
```

The R repo's `ethwbtc_trade_history` dataset demonstrates this structure with complete swap records including tick data.
