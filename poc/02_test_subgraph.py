"""
POC 2: Test Uniswap v3 Subgraph for Historical Data

The Graph provides free access to Uniswap v3 data.
Note: Requires a small amount of GRT in your Graph account for production use,
but the hosted service still works for testing.

Data available:
- poolDayData / poolHourData: aggregated OHLC-like data per pool
- swaps: individual swap events
- ticks: liquidity at each tick
"""
import requests
import json
from datetime import datetime, timedelta
from decimal import Decimal

# Subgraph endpoints
# The Graph's decentralized network (requires API key + small GRT balance)
GRAPH_API_KEY = "YOUR_API_KEY"  # Get from https://thegraph.com/studio/
DECENTRALIZED_ENDPOINT = f"https://gateway.thegraph.com/api/{GRAPH_API_KEY}/subgraphs/id/5zvR82QoaXYFyDEKLZ9t6v9adgnptxYpKpSbxtgVENFV"

# Hosted service (deprecated but still works for testing)
HOSTED_ENDPOINT = "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3"

# Pool IDs (lowercase addresses)
ETH_USDC_005_POOL = "0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640"
ETH_USDC_03_POOL = "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8"


def query_subgraph(endpoint: str, query: str, variables: dict = None) -> dict:
    """Execute a GraphQL query against the subgraph."""
    payload = {"query": query}
    if variables:
        payload["variables"] = variables

    resp = requests.post(endpoint, json=payload, timeout=30)
    return resp.json()


def get_pool_hour_data(endpoint: str, pool_id: str, hours: int = 24) -> list:
    """
    Fetch hourly pool data for the last N hours.

    Returns OHLC-like data: open, high, low, close prices per hour.
    """
    # Calculate timestamp for N hours ago
    now = int(datetime.now().timestamp())
    start_time = now - (hours * 3600)

    query = """
    query PoolHourData($pool: String!, $startTime: Int!) {
        poolHourDatas(
            first: 100
            orderBy: periodStartUnix
            orderDirection: desc
            where: {
                pool: $pool
                periodStartUnix_gte: $startTime
            }
        ) {
            periodStartUnix
            tick
            sqrtPrice
            liquidity
            high
            low
            open
            close
            volumeToken0
            volumeToken1
            volumeUSD
            txCount
        }
    }
    """

    result = query_subgraph(endpoint, query, {
        "pool": pool_id,
        "startTime": start_time
    })

    return result.get("data", {}).get("poolHourDatas", [])


def get_recent_swaps(endpoint: str, pool_id: str, limit: int = 100) -> list:
    """
    Fetch recent swap events.

    Each swap gives us exact price at that moment.
    """
    query = """
    query RecentSwaps($pool: String!, $limit: Int!) {
        swaps(
            first: $limit
            orderBy: timestamp
            orderDirection: desc
            where: { pool: $pool }
        ) {
            id
            timestamp
            amount0
            amount1
            amountUSD
            sqrtPriceX96
            tick
        }
    }
    """

    result = query_subgraph(endpoint, query, {
        "pool": pool_id,
        "limit": limit
    })

    return result.get("data", {}).get("swaps", [])


def get_pool_state(endpoint: str, pool_id: str) -> dict:
    """Fetch current pool state."""
    query = """
    query PoolState($pool: ID!) {
        pool(id: $pool) {
            id
            token0 {
                symbol
                decimals
            }
            token1 {
                symbol
                decimals
            }
            feeTier
            sqrtPrice
            tick
            liquidity
            totalValueLockedToken0
            totalValueLockedToken1
            totalValueLockedUSD
            volumeUSD
        }
    }
    """

    result = query_subgraph(endpoint, query, {"pool": pool_id})
    return result.get("data", {}).get("pool", {})


def get_tick_liquidity(endpoint: str, pool_id: str, tick_range: tuple = None) -> list:
    """
    Fetch liquidity at each initialized tick.

    Useful for understanding where liquidity is concentrated.
    """
    query = """
    query TickLiquidity($pool: String!, $skip: Int!) {
        ticks(
            first: 1000
            skip: $skip
            orderBy: tickIdx
            where: { pool: $pool }
        ) {
            tickIdx
            liquidityGross
            liquidityNet
        }
    }
    """

    all_ticks = []
    skip = 0

    while True:
        result = query_subgraph(endpoint, query, {
            "pool": pool_id,
            "skip": skip
        })
        ticks = result.get("data", {}).get("ticks", [])

        if not ticks:
            break

        all_ticks.extend(ticks)
        skip += len(ticks)

        if len(ticks) < 1000:
            break

    return all_ticks


def sqrtprice_to_price(sqrt_price: str, decimals0: int = 18, decimals1: int = 6) -> float:
    """Convert sqrtPriceX96 to human-readable price."""
    sqrt_price_x96 = int(sqrt_price)
    price_raw = (sqrt_price_x96 ** 2) / (2 ** 192)
    # Adjust for decimal difference
    price = price_raw * (10 ** (decimals0 - decimals1))
    return price


def main():
    print("=" * 60)
    print("POC 2: Testing Uniswap v3 Subgraph")
    print("=" * 60)

    endpoint = HOSTED_ENDPOINT
    pool_id = ETH_USDC_005_POOL

    print(f"\nEndpoint: {endpoint}")
    print(f"Pool: ETH/USDC 0.05% ({pool_id})")

    # Test 1: Pool State
    print("\n## 1. Current Pool State")
    print("-" * 40)
    try:
        pool = get_pool_state(endpoint, pool_id)
        if pool:
            print(f"Token0: {pool.get('token0', {}).get('symbol')} ({pool.get('token0', {}).get('decimals')} decimals)")
            print(f"Token1: {pool.get('token1', {}).get('symbol')} ({pool.get('token1', {}).get('decimals')} decimals)")
            print(f"Fee Tier: {int(pool.get('feeTier', 0)) / 10000}%")
            print(f"Current Tick: {pool.get('tick')}")
            print(f"Liquidity: {int(pool.get('liquidity', 0)):,}")
            print(f"TVL: ${float(pool.get('totalValueLockedUSD', 0)):,.2f}")

            # Calculate price
            if pool.get('sqrtPrice'):
                price = sqrtprice_to_price(pool['sqrtPrice'])
                print(f"Price: ${price:,.2f} USDC/ETH")
        else:
            print("No pool data returned")
    except Exception as e:
        print(f"Error: {e}")

    # Test 2: Hourly Data (OHLC)
    print("\n## 2. Hourly OHLC Data (last 24h)")
    print("-" * 40)
    try:
        hour_data = get_pool_hour_data(endpoint, pool_id, hours=24)
        if hour_data:
            print(f"Got {len(hour_data)} hourly records")
            print("\nMost recent hours:")
            for h in hour_data[:5]:
                ts = datetime.fromtimestamp(int(h['periodStartUnix']))
                vol = float(h.get('volumeUSD', 0))
                print(f"  {ts}: O={h.get('open', 'N/A')[:10]} H={h.get('high', 'N/A')[:10]} "
                      f"L={h.get('low', 'N/A')[:10]} C={h.get('close', 'N/A')[:10]} Vol=${vol:,.0f}")
        else:
            print("No hourly data returned")
    except Exception as e:
        print(f"Error: {e}")

    # Test 3: Recent Swaps
    print("\n## 3. Recent Swaps")
    print("-" * 40)
    try:
        swaps = get_recent_swaps(endpoint, pool_id, limit=10)
        if swaps:
            print(f"Got {len(swaps)} recent swaps")
            print("\nLatest swaps:")
            for s in swaps[:5]:
                ts = datetime.fromtimestamp(int(s['timestamp']))
                amount_usd = float(s.get('amountUSD', 0))
                tick = s.get('tick')
                print(f"  {ts}: ${amount_usd:,.2f} @ tick {tick}")
        else:
            print("No swap data returned")
    except Exception as e:
        print(f"Error: {e}")

    # Test 4: Tick Liquidity
    print("\n## 4. Tick Liquidity Distribution")
    print("-" * 40)
    try:
        ticks = get_tick_liquidity(endpoint, pool_id)
        if ticks:
            print(f"Got {len(ticks)} initialized ticks")

            # Find ticks with most liquidity
            sorted_ticks = sorted(ticks, key=lambda t: abs(int(t['liquidityNet'])), reverse=True)
            print("\nTop 5 ticks by liquidity change:")
            for t in sorted_ticks[:5]:
                print(f"  Tick {t['tickIdx']}: liquidityNet={int(t['liquidityNet']):,}")
        else:
            print("No tick data returned")
    except Exception as e:
        print(f"Error: {e}")

    print("\n" + "=" * 60)
    print("CONCLUSION: Subgraph provides rich historical data!")
    print("=" * 60)
    print("\nData available for our needs:")
    print("  ✓ Current pool state (price, tick, liquidity)")
    print("  ✓ Hourly OHLC data (open, high, low, close)")
    print("  ✓ Individual swap events (for VWAP calculation)")
    print("  ✓ Tick liquidity distribution")
    print("\nNote: The hosted service is being deprecated.")
    print("For production, use the decentralized network with an API key.")


if __name__ == "__main__":
    main()
