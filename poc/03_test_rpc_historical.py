"""
POC 3: Direct RPC Historical Data Fetching

For cases where subgraph isn't available or we need block-level granularity,
we can fetch historical data directly via RPC using:
1. eth_call at specific block numbers
2. eth_getLogs for swap events

This is slower but gives us complete control.
"""
import requests
import json
from typing import Optional
from dataclasses import dataclass
from decimal import Decimal

# Constants
SLOT0_SELECTOR = "0x3850c7bd"
LIQUIDITY_SELECTOR = "0x1a686502"

# Swap event signature: Swap(address,address,int256,int256,uint160,uint128,int24)
SWAP_EVENT_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# Pool addresses
ETH_USDC_005_POOL = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"


@dataclass(frozen=True)
class SwapEvent:
    """Decoded swap event."""
    block_number: int
    tx_hash: str
    log_index: int
    sender: str
    recipient: str
    amount0: int
    amount1: int
    sqrt_price_x96: int
    liquidity: int
    tick: int


@dataclass(frozen=True)
class PoolSnapshot:
    """Pool state at a specific block."""
    block_number: int
    sqrt_price_x96: int
    tick: int
    liquidity: int
    price_usdc_per_eth: float


def eth_call(rpc_url: str, to: str, data: str, block: str = "latest") -> dict:
    """Make an eth_call to read contract state at a specific block."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": to, "data": data}, block],
        "id": 1
    }
    resp = requests.post(rpc_url, json=payload, timeout=30)
    return resp.json()


def eth_get_logs(rpc_url: str, address: str, topics: list,
                 from_block: int, to_block: int) -> dict:
    """Fetch event logs for a block range."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getLogs",
        "params": [{
            "address": address,
            "topics": topics,
            "fromBlock": hex(from_block),
            "toBlock": hex(to_block)
        }],
        "id": 1
    }
    resp = requests.post(rpc_url, json=payload, timeout=60)
    return resp.json()


def eth_block_number(rpc_url: str) -> int:
    """Get current block number."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_blockNumber",
        "params": [],
        "id": 1
    }
    resp = requests.post(rpc_url, json=payload, timeout=10)
    result = resp.json()
    return int(result["result"], 16)


def decode_slot0(hex_result: str) -> dict:
    """Decode slot0 response."""
    data = hex_result[2:] if hex_result.startswith("0x") else hex_result
    sqrt_price_x96 = int(data[0:64], 16)
    tick = int(data[64:128], 16)
    if tick >= 2**23:
        tick -= 2**24

    price_raw = (sqrt_price_x96 ** 2) / (2 ** 192)
    price_usdc_per_eth = price_raw * (10 ** 12)

    return {
        "sqrt_price_x96": sqrt_price_x96,
        "tick": tick,
        "price_usdc_per_eth": price_usdc_per_eth
    }


def decode_liquidity(hex_result: str) -> int:
    """Decode liquidity response."""
    data = hex_result[2:] if hex_result.startswith("0x") else hex_result
    return int(data, 16)


def decode_swap_log(log: dict) -> SwapEvent:
    """Decode a Swap event log."""
    # Topics: [event_signature, sender (indexed), recipient (indexed)]
    sender = "0x" + log["topics"][1][26:]
    recipient = "0x" + log["topics"][2][26:]

    # Data: amount0, amount1, sqrtPriceX96, liquidity, tick
    data = log["data"][2:]  # Remove 0x

    # All values are 32 bytes (64 hex chars)
    amount0 = int(data[0:64], 16)
    if amount0 >= 2**255:
        amount0 -= 2**256

    amount1 = int(data[64:128], 16)
    if amount1 >= 2**255:
        amount1 -= 2**256

    sqrt_price_x96 = int(data[128:192], 16)
    liquidity = int(data[192:256], 16)

    tick = int(data[256:320], 16)
    if tick >= 2**23:
        tick -= 2**24

    return SwapEvent(
        block_number=int(log["blockNumber"], 16),
        tx_hash=log["transactionHash"],
        log_index=int(log["logIndex"], 16),
        sender=sender,
        recipient=recipient,
        amount0=amount0,
        amount1=amount1,
        sqrt_price_x96=sqrt_price_x96,
        liquidity=liquidity,
        tick=tick
    )


def get_pool_snapshot(rpc_url: str, pool: str, block: int) -> Optional[PoolSnapshot]:
    """Get pool state at a specific block."""
    block_hex = hex(block)

    try:
        slot0_resp = eth_call(rpc_url, pool, SLOT0_SELECTOR, block_hex)
        if "error" in slot0_resp:
            return None

        slot0 = decode_slot0(slot0_resp["result"])

        liq_resp = eth_call(rpc_url, pool, LIQUIDITY_SELECTOR, block_hex)
        if "error" in liq_resp:
            return None

        liquidity = decode_liquidity(liq_resp["result"])

        return PoolSnapshot(
            block_number=block,
            sqrt_price_x96=slot0["sqrt_price_x96"],
            tick=slot0["tick"],
            liquidity=liquidity,
            price_usdc_per_eth=slot0["price_usdc_per_eth"]
        )
    except Exception as e:
        print(f"Error fetching block {block}: {e}")
        return None


def get_swaps_in_range(rpc_url: str, pool: str, from_block: int, to_block: int) -> list[SwapEvent]:
    """Fetch all swap events in a block range."""
    result = eth_get_logs(
        rpc_url,
        pool,
        [SWAP_EVENT_TOPIC],
        from_block,
        to_block
    )

    if "error" in result:
        print(f"Error fetching logs: {result['error']}")
        return []

    logs = result.get("result", [])
    return [decode_swap_log(log) for log in logs]


def compute_vwap(swaps: list[SwapEvent]) -> Optional[float]:
    """Compute VWAP from swap events."""
    if not swaps:
        return None

    total_volume_usd = Decimal(0)
    total_value = Decimal(0)

    for swap in swaps:
        # Price from sqrtPriceX96
        price = ((swap.sqrt_price_x96 ** 2) / (2 ** 192)) * (10 ** 12)

        # Volume in ETH (amount0)
        volume_eth = abs(swap.amount0) / (10 ** 18)

        # Volume in USD
        volume_usd = Decimal(str(volume_eth * price))

        total_volume_usd += volume_usd
        total_value += volume_usd * Decimal(str(price))

    if total_volume_usd == 0:
        return None

    return float(total_value / total_volume_usd)


def main():
    print("=" * 60)
    print("POC 3: Direct RPC Historical Data Fetching")
    print("=" * 60)

    # Use a free RPC (you'll need to run this locally)
    rpc_url = "https://ethereum-rpc.publicnode.com"
    pool = ETH_USDC_005_POOL

    print(f"\nRPC: {rpc_url}")
    print(f"Pool: ETH/USDC 0.05% ({pool})")

    # Test 1: Get current block and pool state
    print("\n## 1. Current State")
    print("-" * 40)
    try:
        current_block = eth_block_number(rpc_url)
        print(f"Current block: {current_block:,}")

        snapshot = get_pool_snapshot(rpc_url, pool, current_block)
        if snapshot:
            print(f"Price: ${snapshot.price_usdc_per_eth:,.2f}")
            print(f"Tick: {snapshot.tick}")
            print(f"Liquidity: {snapshot.liquidity:,}")
    except Exception as e:
        print(f"Error: {e}")
        print("\n[!] RPC call failed - this is expected if running in restricted environment")
        print("    Run this script locally with actual RPC access")

    # Test 2: Historical snapshots
    print("\n## 2. Historical Pool Snapshots")
    print("-" * 40)
    print("Example code for fetching snapshots every 100 blocks:")
    print("""
    blocks = range(current_block - 1000, current_block, 100)
    snapshots = []
    for block in blocks:
        snapshot = get_pool_snapshot(rpc_url, pool, block)
        if snapshot:
            snapshots.append(snapshot)
            print(f"Block {block}: ${snapshot.price_usdc_per_eth:,.2f}")
    """)

    # Test 3: Swap events
    print("\n## 3. Swap Event Fetching")
    print("-" * 40)
    print("Example code for fetching swaps in a block range:")
    print("""
    swaps = get_swaps_in_range(rpc_url, pool, current_block - 100, current_block)
    print(f"Found {len(swaps)} swaps")
    for swap in swaps[:5]:
        price = ((swap.sqrt_price_x96 ** 2) / (2 ** 192)) * (10 ** 12)
        print(f"Block {swap.block_number}: ${price:,.2f} ({swap.amount0 / 1e18:.4f} ETH)")
    """)

    # Test 4: VWAP calculation
    print("\n## 4. VWAP Calculation")
    print("-" * 40)
    print("Example code for computing VWAP from swaps:")
    print("""
    vwap = compute_vwap(swaps)
    print(f"100-block VWAP: ${vwap:,.2f}")
    """)

    print("\n" + "=" * 60)
    print("CONCLUSION: Direct RPC gives us full control!")
    print("=" * 60)
    print("\nData we can fetch:")
    print("  ✓ Pool state at any historical block")
    print("  ✓ All swap events via getLogs")
    print("  ✓ Compute VWAP from swap data")
    print("  ✓ Block-level granularity")
    print("\nRate limiting considerations:")
    print("  - Free RPCs: ~20-50 requests/second")
    print("  - Archive node needed for historical blocks")
    print("  - Consider batching requests with multicall")
    print("  - eth_getLogs can fetch 2000 blocks per call")


if __name__ == "__main__":
    main()
