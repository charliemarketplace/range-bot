"""
POC 5: Real Data - Fetch 1000 blocks of swaps and analyze

Fetches actual swap events from Ethereum mainnet using free public RPC,
then runs VWAP + Bayesian analysis on real data.
"""
import requests
import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional
import statistics

# Free public RPCs to try
RPCS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://eth.drpc.org",
    "https://1rpc.io/eth",
    "https://eth.llamarpc.com",
]

# ETH/USDC 0.05% pool
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

# Swap event topic
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"


@dataclass(frozen=True)
class Swap:
    block_number: int
    tx_hash: str
    amount0: int  # ETH (can be negative)
    amount1: int  # USDC (can be negative)
    sqrt_price_x96: int
    liquidity: int
    tick: int
    price_usdc_per_eth: float


@dataclass(frozen=True)
class OHLC:
    block_start: int
    block_end: int
    open: float
    high: float
    low: float
    close: float
    volume_eth: float
    vwap: float
    num_swaps: int


@dataclass(frozen=True)
class Distribution:
    center: float
    prices: tuple[float, ...]
    probabilities: tuple[float, ...]

    def expected_value(self) -> float:
        return sum(p * prob for p, prob in zip(self.prices, self.probabilities))

    def std_dev(self) -> float:
        ev = self.expected_value()
        variance = sum(prob * (p - ev) ** 2 for p, prob in zip(self.prices, self.probabilities))
        return math.sqrt(variance)

    def probability_in_range(self, low: float, high: float) -> float:
        return sum(prob for p, prob in zip(self.prices, self.probabilities) if low <= p <= high)


def rpc_call(rpc_url: str, method: str, params: list, timeout: int = 30) -> dict:
    """Make a JSON-RPC call."""
    payload = {
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1
    }
    resp = requests.post(rpc_url, json=payload, timeout=timeout)
    return resp.json()


def find_working_rpc() -> Optional[str]:
    """Find a working RPC endpoint."""
    for rpc in RPCS:
        try:
            result = rpc_call(rpc, "eth_blockNumber", [], timeout=5)
            if "result" in result:
                print(f"✓ {rpc} is working")
                return rpc
        except Exception as e:
            print(f"✗ {rpc}: {e}")
    return None


def get_block_number(rpc: str) -> int:
    """Get current block number."""
    result = rpc_call(rpc, "eth_blockNumber", [])
    return int(result["result"], 16)


def get_logs(rpc: str, address: str, topics: list, from_block: int, to_block: int) -> list:
    """Fetch event logs."""
    result = rpc_call(rpc, "eth_getLogs", [{
        "address": address,
        "topics": topics,
        "fromBlock": hex(from_block),
        "toBlock": hex(to_block)
    }], timeout=60)

    if "error" in result:
        raise Exception(f"RPC error: {result['error']}")

    return result.get("result", [])


def decode_swap_log(log: dict) -> Swap:
    """Decode a Swap event log."""
    data = log["data"][2:]  # Remove 0x

    # Decode signed int256 values
    def decode_int256(hex_str: str) -> int:
        val = int(hex_str, 16)
        if val >= 2**255:
            val -= 2**256
        return val

    def decode_int24(hex_str: str) -> int:
        val = int(hex_str, 16)
        if val >= 2**23:
            val -= 2**24
        return val

    amount0 = decode_int256(data[0:64])
    amount1 = decode_int256(data[64:128])
    sqrt_price_x96 = int(data[128:192], 16)
    liquidity = int(data[192:256], 16)
    tick = decode_int24(data[256:320])

    # Calculate price: USDC per ETH
    # sqrtPriceX96 = sqrt(price) * 2^96
    # price = (sqrtPriceX96 / 2^96)^2
    # Adjust for decimals: ETH has 18, USDC has 6
    price_raw = (sqrt_price_x96 ** 2) / (2 ** 192)
    price_usdc_per_eth = price_raw * (10 ** 12)  # Adjust for decimal difference

    return Swap(
        block_number=int(log["blockNumber"], 16),
        tx_hash=log["transactionHash"],
        amount0=amount0,
        amount1=amount1,
        sqrt_price_x96=sqrt_price_x96,
        liquidity=liquidity,
        tick=tick,
        price_usdc_per_eth=price_usdc_per_eth
    )


def fetch_swaps(rpc: str, num_blocks: int = 1000, chunk_size: int = 500) -> list[Swap]:
    """Fetch swaps from the last N blocks in chunks."""
    current_block = get_block_number(rpc)
    start_block = current_block - num_blocks

    print(f"Fetching swaps from block {start_block:,} to {current_block:,}")

    all_swaps = []

    for from_block in range(start_block, current_block, chunk_size):
        to_block = min(from_block + chunk_size - 1, current_block)

        print(f"  Fetching blocks {from_block:,} - {to_block:,}...", end=" ")

        try:
            logs = get_logs(rpc, POOL_ADDRESS, [SWAP_TOPIC], from_block, to_block)
            swaps = [decode_swap_log(log) for log in logs]
            all_swaps.extend(swaps)
            print(f"got {len(swaps)} swaps")
        except Exception as e:
            print(f"error: {e}")

        # Rate limiting - be nice to free RPCs
        time.sleep(0.5)

    return sorted(all_swaps, key=lambda s: (s.block_number, s.tx_hash))


def aggregate_to_ohlc(swaps: list[Swap], blocks_per_candle: int = 100) -> list[OHLC]:
    """Aggregate swaps into OHLC candles."""
    if not swaps:
        return []

    # Group by block range
    min_block = min(s.block_number for s in swaps)
    max_block = max(s.block_number for s in swaps)

    candles = []

    for block_start in range(min_block, max_block + 1, blocks_per_candle):
        block_end = block_start + blocks_per_candle - 1

        period_swaps = [s for s in swaps if block_start <= s.block_number <= block_end]

        if not period_swaps:
            continue

        prices = [s.price_usdc_per_eth for s in period_swaps]
        volumes = [abs(s.amount0) / 1e18 for s in period_swaps]  # ETH volume

        # VWAP
        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        vwap = total_value / total_volume if total_volume > 0 else prices[-1]

        candles.append(OHLC(
            block_start=block_start,
            block_end=block_end,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume_eth=total_volume,
            vwap=vwap,
            num_swaps=len(period_swaps)
        ))

    return candles


def compute_rolling_vwap(candles: list[OHLC], window: int = 10) -> tuple[float, float]:
    """Compute median VWAP and std dev over window."""
    if len(candles) < window:
        window = len(candles)

    recent = candles[-window:]
    vwaps = [c.vwap for c in recent]

    median_vwap = statistics.median(vwaps)
    std_dev = statistics.stdev(vwaps) if len(vwaps) > 1 else median_vwap * 0.01

    return median_vwap, std_dev


def build_laplace_prior(center: float, scale: float, num_points: int = 101) -> Distribution:
    """Build Laplace prior distribution."""
    half_range = scale * 4
    min_price = center - half_range
    max_price = center + half_range

    prices = []
    probs = []

    for i in range(num_points):
        p = min_price + (max_price - min_price) * i / (num_points - 1)
        prices.append(p)

        # Laplace PDF
        prob = (1 / (2 * scale)) * math.exp(-abs(p - center) / scale)
        probs.append(prob)

    # Normalize
    total = sum(probs)
    probs = [p / total for p in probs]

    return Distribution(center=center, prices=tuple(prices), probabilities=tuple(probs))


def build_likelihood(candles: list[OHLC], num_points: int = 101) -> Distribution:
    """Build likelihood from recent OHLC."""
    all_lows = [c.low for c in candles]
    all_highs = [c.high for c in candles]

    min_price = min(all_lows) * 0.995
    max_price = max(all_highs) * 1.005
    center = (min_price + max_price) / 2

    prices = []
    probs = [0.0] * num_points

    for i in range(num_points):
        p = min_price + (max_price - min_price) * i / (num_points - 1)
        prices.append(p)

    # Add probability mass from each candle (more recent = more weight)
    for idx, candle in enumerate(candles):
        weight = 0.9 ** (len(candles) - 1 - idx)
        for i, p in enumerate(prices):
            if candle.low <= p <= candle.high:
                probs[i] += weight

    # Normalize
    total = sum(probs)
    if total > 0:
        probs = [p / total for p in probs]
    else:
        probs = [1 / num_points] * num_points

    return Distribution(center=center, prices=tuple(prices), probabilities=tuple(probs))


def bayesian_update(prior: Distribution, likelihood: Distribution) -> Distribution:
    """Compute posterior = prior × likelihood."""
    # Use prior's price grid
    new_probs = []

    for price, prior_prob in zip(prior.prices, prior.probabilities):
        # Interpolate likelihood
        lik_prob = interpolate(likelihood, price)
        new_probs.append(prior_prob * lik_prob)

    # Normalize
    total = sum(new_probs)
    if total > 0:
        new_probs = [p / total for p in new_probs]
    else:
        new_probs = list(prior.probabilities)

    return Distribution(center=prior.center, prices=prior.prices, probabilities=tuple(new_probs))


def interpolate(dist: Distribution, target: float) -> float:
    """Interpolate probability at target price."""
    prices = dist.prices

    for i in range(len(prices) - 1):
        if prices[i] <= target <= prices[i + 1]:
            t = (target - prices[i]) / (prices[i + 1] - prices[i]) if prices[i + 1] != prices[i] else 0
            return (1 - t) * dist.probabilities[i] + t * dist.probabilities[i + 1]

    if target < prices[0]:
        return dist.probabilities[0]
    return dist.probabilities[-1]


def optimize_range(posterior: Distribution, target_coverage: float = 0.9) -> dict:
    """Find optimal range for target coverage."""
    n = len(posterior.prices)
    best_range = None
    best_width = float('inf')

    for i in range(n):
        cumsum = 0.0
        for j in range(i, n):
            cumsum += posterior.probabilities[j]
            if cumsum >= target_coverage:
                width = posterior.prices[j] - posterior.prices[i]
                if width < best_width:
                    best_width = width
                    best_range = (i, j, cumsum)
                break

    if best_range is None:
        best_range = (0, n - 1, 1.0)

    i, j, coverage = best_range
    lower = posterior.prices[i]
    upper = posterior.prices[j]

    return {
        "lower": lower,
        "upper": upper,
        "center": (lower + upper) / 2,
        "coverage": coverage,
        "width_pct": (upper - lower) / ((lower + upper) / 2) * 100
    }


def main():
    print("=" * 70)
    print("POC 5: Real Data Analysis - Last 1000 Blocks")
    print("=" * 70)

    # Find working RPC
    print("\n## 1. Finding Working RPC")
    print("-" * 50)
    rpc = find_working_rpc()

    if not rpc:
        print("\nNo working RPC found. Try running locally.")
        return

    # Fetch swaps
    print("\n## 2. Fetching Swap Events")
    print("-" * 50)
    swaps = fetch_swaps(rpc, num_blocks=1000, chunk_size=500)

    if not swaps:
        print("No swaps found!")
        return

    print(f"\nTotal swaps: {len(swaps)}")
    print(f"Block range: {swaps[0].block_number:,} - {swaps[-1].block_number:,}")
    print(f"Price range: ${min(s.price_usdc_per_eth for s in swaps):,.2f} - ${max(s.price_usdc_per_eth for s in swaps):,.2f}")

    # Aggregate to OHLC
    print("\n## 3. Aggregate to 100-Block Candles")
    print("-" * 50)
    candles = aggregate_to_ohlc(swaps, blocks_per_candle=100)
    print(f"Generated {len(candles)} candles")

    print("\nRecent candles:")
    for c in candles[-5:]:
        print(f"  Blocks {c.block_start}-{c.block_end}: "
              f"O=${c.open:,.2f} H=${c.high:,.2f} L=${c.low:,.2f} C=${c.close:,.2f} "
              f"VWAP=${c.vwap:,.2f} Vol={c.volume_eth:.2f}ETH")

    # VWAP Prior
    print("\n## 4. Compute Rolling VWAP Prior")
    print("-" * 50)
    median_vwap, std_dev = compute_rolling_vwap(candles, window=10)
    print(f"10-candle median VWAP: ${median_vwap:,.2f}")
    print(f"Standard deviation: ${std_dev:,.2f}")

    prior = build_laplace_prior(median_vwap, std_dev * 2)
    print(f"Prior EV: ${prior.expected_value():,.2f}")
    print(f"Prior StdDev: ${prior.std_dev():,.2f}")

    # Likelihood
    print("\n## 5. Build Likelihood from Recent Data")
    print("-" * 50)
    likelihood = build_likelihood(candles[-10:])
    print(f"Likelihood EV: ${likelihood.expected_value():,.2f}")

    # Bayesian Update
    print("\n## 6. Bayesian Update")
    print("-" * 50)
    posterior = bayesian_update(prior, likelihood)
    print(f"Posterior EV: ${posterior.expected_value():,.2f}")
    print(f"Posterior StdDev: ${posterior.std_dev():,.2f}")

    # Optimal Range
    print("\n## 7. Optimal LP Range (90% Coverage)")
    print("-" * 50)
    rec = optimize_range(posterior, target_coverage=0.90)
    print(f"Lower: ${rec['lower']:,.2f}")
    print(f"Upper: ${rec['upper']:,.2f}")
    print(f"Center: ${rec['center']:,.2f}")
    print(f"Coverage: {rec['coverage']*100:.1f}%")
    print(f"Range Width: {rec['width_pct']:.2f}%")

    # Current price comparison
    current_price = swaps[-1].price_usdc_per_eth
    in_range = rec['lower'] <= current_price <= rec['upper']
    print(f"\nCurrent Price: ${current_price:,.2f}")
    print(f"In Recommended Range: {'✓ Yes' if in_range else '✗ No'}")

    # ASCII chart
    print("\n## 8. Posterior Distribution")
    print("-" * 50)
    max_prob = max(posterior.probabilities)
    step = max(1, len(posterior.prices) // 20)

    for i in range(0, len(posterior.prices), step):
        price = posterior.prices[i]
        prob = posterior.probabilities[i]
        bar_len = int(prob / max_prob * 40) if max_prob > 0 else 0
        marker = "█" if rec['lower'] <= price <= rec['upper'] else "░"
        bar = marker * bar_len
        print(f"${price:>10,.0f} | {bar}")

    print("\n" + "=" * 70)
    print("COMPLETE: Real Ethereum mainnet data analyzed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
