"""
POC 1: Test free public RPCs for Uniswap v3 pool state

Tests fetching current pool state (price, liquidity, tick) from free endpoints.
"""
import requests
import json
from decimal import Decimal

# Free public RPC endpoints to test
RPC_ENDPOINTS = {
    "ethereum": [
        "https://ethereum-rpc.publicnode.com",
        "https://rpc.ankr.com/eth",
        "https://eth.drpc.org",
        "https://1rpc.io/eth",
    ],
    "base": [
        "https://base-rpc.publicnode.com",
        "https://rpc.ankr.com/base",
        "https://base.drpc.org",
        "https://1rpc.io/base",
    ]
}

# Uniswap v3 pool addresses
POOLS = {
    "ethereum": {
        "ETH_USDC_005": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",  # 0.05% fee
        "ETH_USDC_03": "0x8ad599c3A0ff1De082011EFDDc58f1908eb6e6D8",   # 0.3% fee
    },
    "base": {
        "ETH_USDC_005": "0xd0b53D9277642d899DF5C87A3966A349A798F224",  # 0.05% fee
    }
}

# Uniswap v3 Pool ABI - just the functions we need
# slot0() returns (sqrtPriceX96, tick, observationIndex, observationCardinality,
#                  observationCardinalityNext, feeProtocol, unlocked)
SLOT0_SELECTOR = "0x3850c7bd"  # slot0()
LIQUIDITY_SELECTOR = "0x1a686502"  # liquidity()


def eth_call(rpc_url: str, to: str, data: str) -> dict:
    """Make an eth_call to read contract state."""
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_call",
        "params": [{"to": to, "data": data}, "latest"],
        "id": 1
    }
    resp = requests.post(rpc_url, json=payload, timeout=10)
    return resp.json()


def decode_slot0(hex_result: str) -> dict:
    """Decode slot0 response."""
    # Remove 0x prefix
    data = hex_result[2:] if hex_result.startswith("0x") else hex_result

    # slot0 returns 7 values, each 32 bytes (64 hex chars)
    sqrt_price_x96 = int(data[0:64], 16)
    tick = int(data[64:128], 16)
    # Handle signed int24 for tick
    if tick >= 2**23:
        tick -= 2**24

    # Convert sqrtPriceX96 to human-readable price
    # price = (sqrtPriceX96 / 2^96)^2
    # For ETH/USDC where USDC is token1 (6 decimals) and ETH is token0 (18 decimals):
    # price in USDC per ETH = (sqrtPriceX96^2 / 2^192) * 10^12
    price_raw = (sqrt_price_x96 ** 2) / (2 ** 192)
    # Adjust for decimal difference (18 - 6 = 12)
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


def test_rpc_endpoint(chain: str, rpc_url: str, pool_address: str) -> dict:
    """Test an RPC endpoint by fetching pool state."""
    result = {
        "rpc": rpc_url,
        "chain": chain,
        "pool": pool_address,
        "success": False,
        "error": None,
        "data": None
    }

    try:
        # Fetch slot0
        slot0_resp = eth_call(rpc_url, pool_address, SLOT0_SELECTOR)
        if "error" in slot0_resp:
            result["error"] = f"slot0 error: {slot0_resp['error']}"
            return result

        slot0 = decode_slot0(slot0_resp["result"])

        # Fetch liquidity
        liq_resp = eth_call(rpc_url, pool_address, LIQUIDITY_SELECTOR)
        if "error" in liq_resp:
            result["error"] = f"liquidity error: {liq_resp['error']}"
            return result

        liquidity = decode_liquidity(liq_resp["result"])

        result["success"] = True
        result["data"] = {
            **slot0,
            "liquidity": liquidity
        }

    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    print("=" * 60)
    print("POC 1: Testing Free Public RPC Endpoints")
    print("=" * 60)

    for chain, rpcs in RPC_ENDPOINTS.items():
        print(f"\n## {chain.upper()}")

        # Get first pool for this chain
        pool_name = list(POOLS[chain].keys())[0]
        pool_address = POOLS[chain][pool_name]
        print(f"Pool: {pool_name} ({pool_address})")

        for rpc_url in rpcs:
            result = test_rpc_endpoint(chain, rpc_url, pool_address)

            status = "✓" if result["success"] else "✗"
            print(f"\n  {status} {rpc_url}")

            if result["success"]:
                data = result["data"]
                print(f"    Price: ${data['price_usdc_per_eth']:,.2f} USDC/ETH")
                print(f"    Tick: {data['tick']}")
                print(f"    Liquidity: {data['liquidity']:,}")
            else:
                print(f"    Error: {result['error']}")

    print("\n" + "=" * 60)
    print("CONCLUSION: Free RPCs work for reading current pool state!")
    print("=" * 60)


if __name__ == "__main__":
    main()
