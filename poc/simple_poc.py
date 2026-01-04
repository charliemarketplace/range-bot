"""
Simple POC: Fetch Uniswap v3 swaps, store locally, analyze with Bayesian model.
"""
import requests
import json
import math
import statistics
from pathlib import Path

RPC = "https://ethereum-rpc.publicnode.com"
POOL = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # ETH/USDC 0.05%
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
DATA_DIR = Path(__file__).parent / "data"


def rpc(method: str, params: list) -> dict:
    r = requests.post(RPC, json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1}, timeout=60)
    return r.json()


def get_block() -> int:
    return int(rpc("eth_blockNumber", [])["result"], 16)


def get_logs(from_block: int, to_block: int) -> list:
    return rpc("eth_getLogs", [{
        "address": POOL,
        "topics": [SWAP_TOPIC],
        "fromBlock": hex(from_block),
        "toBlock": hex(to_block)
    }])["result"]


def decode_swap(log: dict) -> dict:
    data = log["data"][2:]

    def int256(h): v = int(h, 16); return v - 2**256 if v >= 2**255 else v
    def int24(h): v = int(h, 16); return v - 2**24 if v >= 2**23 else v

    sqrt_price = int(data[128:192], 16)
    # Pool has USDC=token0 (6 dec), WETH=token1 (18 dec)
    # price_raw = token1/token0 in base units, invert for USDC/ETH
    price = 1e12 / ((sqrt_price ** 2) / (2 ** 192))

    return {
        "block": int(log["blockNumber"], 16),
        "tx": log["transactionHash"],
        "amount0": int256(data[0:64]),
        "amount1": int256(data[64:128]),
        "sqrt_price": sqrt_price,
        "liquidity": int(data[192:256], 16),
        "tick": int24(data[256:320]),
        "price": price
    }


def fetch_swaps(num_blocks: int = 1000) -> list:
    """Fetch swaps from last N blocks."""
    end = get_block()
    start = end - num_blocks

    swaps = []
    for i in range(start, end, 500):
        logs = get_logs(i, min(i + 499, end))
        swaps.extend(decode_swap(log) for log in logs)
        print(f"  {i}-{min(i+499, end)}: {len(logs)} swaps")

    return sorted(swaps, key=lambda s: s["block"])


def to_ohlc(swaps: list, blocks: int = 100) -> list:
    """Aggregate swaps to OHLC candles."""
    if not swaps:
        return []

    candles = []
    min_b, max_b = swaps[0]["block"], swaps[-1]["block"]

    for b in range(min_b, max_b + 1, blocks):
        ps = [s for s in swaps if b <= s["block"] < b + blocks]
        if not ps:
            continue

        prices = [s["price"] for s in ps]
        vols = [abs(s["amount0"]) / 1e18 for s in ps]
        vwap = sum(p*v for p,v in zip(prices, vols)) / sum(vols) if sum(vols) > 0 else prices[-1]

        candles.append({
            "block": b,
            "o": prices[0], "h": max(prices), "l": min(prices), "c": prices[-1],
            "vol": sum(vols), "vwap": vwap, "n": len(ps)
        })

    return candles


def laplace_dist(center: float, scale: float, n: int = 101) -> tuple:
    """Build Laplace distribution."""
    half = scale * 4
    prices = [center - half + (2*half) * i / (n-1) for i in range(n)]
    probs = [math.exp(-abs(p - center) / scale) / (2*scale) for p in prices]
    total = sum(probs)
    return prices, [p/total for p in probs]


def likelihood_dist(candles: list, n: int = 101) -> tuple:
    """Build likelihood from OHLC."""
    lo, hi = min(c["l"] for c in candles) * 0.995, max(c["h"] for c in candles) * 1.005
    prices = [lo + (hi-lo) * i / (n-1) for i in range(n)]
    probs = [0.0] * n

    for idx, c in enumerate(candles):
        w = 0.9 ** (len(candles) - 1 - idx)
        for i, p in enumerate(prices):
            if c["l"] <= p <= c["h"]:
                probs[i] += w

    total = sum(probs)
    return prices, [p/total for p in probs] if total > 0 else [1/n]*n


def bayesian_update(prior_prices, prior_probs, lik_prices, lik_probs) -> tuple:
    """Posterior = prior * likelihood."""
    def interp(prices, probs, target):
        for i in range(len(prices)-1):
            if prices[i] <= target <= prices[i+1]:
                t = (target - prices[i]) / (prices[i+1] - prices[i])
                return (1-t) * probs[i] + t * probs[i+1]
        return probs[0] if target < prices[0] else probs[-1]

    post = [p * interp(lik_prices, lik_probs, pr) for pr, p in zip(prior_prices, prior_probs)]
    total = sum(post)
    return prior_prices, [p/total for p in post] if total > 0 else prior_probs


def optimal_range(prices, probs, coverage: float = 0.9) -> dict:
    """Find tightest range with target coverage."""
    best = None
    for i in range(len(prices)):
        cumsum = 0.0
        for j in range(i, len(prices)):
            cumsum += probs[j]
            if cumsum >= coverage:
                width = prices[j] - prices[i]
                if best is None or width < best[2]:
                    best = (i, j, width, cumsum)
                break

    if not best:
        return {"lower": prices[0], "upper": prices[-1], "coverage": 1.0}

    return {
        "lower": prices[best[0]],
        "upper": prices[best[1]],
        "coverage": best[3]
    }


def main():
    DATA_DIR.mkdir(exist_ok=True)

    # 1. Fetch swaps
    print("Fetching swaps...")
    swaps = fetch_swaps(1000)
    print(f"Got {len(swaps)} swaps")

    # Save raw
    with open(DATA_DIR / "swaps.json", "w") as f:
        json.dump(swaps, f)
    print(f"Saved to {DATA_DIR / 'swaps.json'}")

    # 2. Build OHLC
    candles = to_ohlc(swaps, 100)
    with open(DATA_DIR / "ohlc.json", "w") as f:
        json.dump(candles, f)
    print(f"Built {len(candles)} candles -> {DATA_DIR / 'ohlc.json'}")

    # 3. VWAP prior
    vwaps = [c["vwap"] for c in candles[-10:]]
    median_vwap = statistics.median(vwaps)
    std = statistics.stdev(vwaps) if len(vwaps) > 1 else median_vwap * 0.01

    prior_prices, prior_probs = laplace_dist(median_vwap, std * 2)
    print(f"Prior: center=${median_vwap:,.2f}, scale=${std*2:,.2f}")

    # 4. Likelihood
    lik_prices, lik_probs = likelihood_dist(candles[-10:])

    # 5. Posterior
    post_prices, post_probs = bayesian_update(prior_prices, prior_probs, lik_prices, lik_probs)
    ev = sum(p*pr for p, pr in zip(post_prices, post_probs))
    print(f"Posterior EV: ${ev:,.2f}")

    # 6. Optimal range
    rec = optimal_range(post_prices, post_probs, 0.9)
    print(f"Range: ${rec['lower']:,.2f} - ${rec['upper']:,.2f} ({rec['coverage']*100:.1f}% coverage)")

    current = swaps[-1]["price"]
    in_range = rec["lower"] <= current <= rec["upper"]
    print(f"Current: ${current:,.2f} {'[IN]' if in_range else '[OUT]'}")

    # Save results
    results = {
        "median_vwap": median_vwap,
        "posterior_ev": ev,
        "range": rec,
        "current_price": current,
        "in_range": in_range
    }
    with open(DATA_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results -> {DATA_DIR / 'results.json'}")


if __name__ == "__main__":
    main()
