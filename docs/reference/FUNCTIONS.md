# Function Reference

All key functions used in the range-bot system with complete implementations.

---

## Price Conversion

### `sqrt_price_to_price`

Converts Uniswap v3's sqrtPriceX96 to human-readable USD/ETH price.

```python
def sqrt_price_to_price(sqrt_price_x96: int) -> float:
    """
    Convert sqrtPriceX96 to price.

    Uniswap v3 stores price as sqrt(price) × 2^96 for precision.

    For ETH/USDC pool:
    - token0 = WETH (18 decimals)
    - token1 = USDC (6 decimals)
    - Native price = USDC per WETH (needs decimal adjustment)

    Formula:
        price_raw = (sqrtPriceX96)^2 / 2^192
        price_adjusted = 10^12 / price_raw  (for 18-6 decimal diff)

    Args:
        sqrt_price_x96: The sqrtPriceX96 value from pool slot0

    Returns:
        Price in USD per ETH (e.g., 3200.50)
    """
    price = 1e12 / ((sqrt_price_x96 ** 2) / (2 ** 192))
    return price
```

---

## Data Aggregation

### `to_ohlc`

Aggregates raw swaps into OHLC candles.

```python
def to_ohlc(swaps: list, blocks_per_candle: int = 50) -> list:
    """
    Aggregate swaps to OHLC candles by block periods.

    Args:
        swaps: List of swap dicts, each with:
            - block: int (block number)
            - price: float (execution price)
            - amount1: int (USDC amount, can be negative)
        blocks_per_candle: Blocks per candle (50 blocks ≈ 10 min)

    Returns:
        List of candle dicts, each with:
            - block_start: int
            - o: float (open price)
            - h: float (high price)
            - l: float (low price)
            - c: float (close price)
            - vol: float (total USDC volume)
            - vwap: float (volume-weighted average price)
            - n: int (number of swaps)
    """
    if not swaps:
        return []

    candles = []
    min_block = swaps[0]["block"]
    max_block = swaps[-1]["block"]

    # Align to period boundaries
    period_start = (min_block // blocks_per_candle) * blocks_per_candle

    while period_start <= max_block:
        period_end = period_start + blocks_per_candle
        ps = [s for s in swaps if period_start <= s["block"] < period_end]

        if ps:
            prices = [s["price"] for s in ps]
            vols = [abs(s["amount1"]) / 1e6 for s in ps]  # USDC is 6 decimals
            total_vol = sum(vols)
            vwap = sum(p * v for p, v in zip(prices, vols)) / total_vol if total_vol > 0 else prices[-1]

            candles.append({
                "block_start": period_start,
                "o": prices[0],
                "h": max(prices),
                "l": min(prices),
                "c": prices[-1],
                "vol": total_vol,
                "vwap": vwap,
                "n": len(ps)
            })

        period_start = period_end

    return candles
```

---

## Bayesian Model

### `laplace_dist`

Builds a Laplace (double-exponential) prior distribution.

```python
def laplace_dist(center: float, scale: float, n: int = 101) -> tuple[list, list]:
    """
    Build discretized Laplace distribution.

    The Laplace distribution has fatter tails than Gaussian,
    better matching observed price return distributions.

    PDF: f(x) = (1/2b) × exp(-|x - μ| / b)

    Args:
        center: Location parameter (μ), typically median VWAP
        scale: Scale parameter (b), typically 2 × std(VWAP)
        n: Number of discrete points

    Returns:
        (prices, probabilities) tuple where:
            - prices: List of n price points spanning ±4 scales
            - probabilities: Normalized probabilities summing to 1
    """
    import math

    half = scale * 4  # Cover ±4 scale parameters
    prices = [center - half + (2 * half) * i / (n - 1) for i in range(n)]
    probs = [math.exp(-abs(p - center) / scale) / (2 * scale) for p in prices]
    total = sum(probs)
    return prices, [p / total for p in probs]
```

### `likelihood_kde`

Builds KDE-based likelihood from OHLC candles.

```python
def likelihood_kde(candles: list, n: int = 101) -> tuple[list, list]:
    """
    Build likelihood distribution using Kernel Density Estimation.

    Uses candle midpoints with exponential decay (recent = more weight).
    Bandwidth selected using Silverman's rule.

    Args:
        candles: List of OHLC candles with 'h' (high) and 'l' (low)
        n: Number of discrete points

    Returns:
        (prices, probabilities) tuple
    """
    import math
    import statistics

    if not candles:
        return [0], [1]

    # Price range with 0.5% buffer
    lo = min(c["l"] for c in candles) * 0.995
    hi = max(c["h"] for c in candles) * 1.005
    prices = [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    # Collect midpoints with decay weights
    points = []
    weights = []
    for idx, c in enumerate(candles):
        w = 0.9 ** (len(candles) - 1 - idx)  # Recent candles weighted more
        mid = (c["h"] + c["l"]) / 2
        points.append(mid)
        weights.append(w)

    if not points:
        return prices, [1 / n] * n

    # Silverman bandwidth
    std = statistics.stdev(points) if len(points) > 1 else (hi - lo) * 0.1
    bw = 1.06 * std * (len(points) ** -0.2)

    # Evaluate KDE at each price point
    probs = [0.0] * n
    for i, p in enumerate(prices):
        for pt, w in zip(points, weights):
            probs[i] += w * math.exp(-0.5 * ((p - pt) / bw) ** 2)

    total = sum(probs)
    return prices, [p / total for p in probs] if total > 0 else [1 / n] * n
```

### `bayesian_update`

Combines prior and likelihood to produce posterior.

```python
def bayesian_update(
    prior_prices: list,
    prior_probs: list,
    lik_prices: list,
    lik_probs: list
) -> tuple[list, list]:
    """
    Compute posterior = prior × likelihood (normalized).

    Uses linear interpolation to align distributions.

    Args:
        prior_prices: Price points for prior
        prior_probs: Prior probabilities
        lik_prices: Price points for likelihood
        lik_probs: Likelihood probabilities

    Returns:
        (prices, posterior_probabilities) tuple
    """
    def interp(prices, probs, target):
        """Linear interpolation for probability at target price."""
        for i in range(len(prices) - 1):
            if prices[i] <= target <= prices[i + 1]:
                t = (target - prices[i]) / (prices[i + 1] - prices[i])
                return (1 - t) * probs[i] + t * probs[i + 1]
        return probs[0] if target < prices[0] else probs[-1]

    # Posterior ∝ Prior × Likelihood
    post = [p * interp(lik_prices, lik_probs, pr) for pr, p in zip(prior_prices, prior_probs)]
    total = sum(post)
    return prior_prices, [p / total for p in post] if total > 0 else prior_probs
```

### `optimal_range`

Finds the tightest credible interval.

```python
def optimal_range(prices: list, probs: list, coverage: float = 0.9) -> dict:
    """
    Find tightest interval containing target coverage.

    Sweeps all possible [i, j] ranges and finds the one with
    minimum width that still contains >= coverage probability.

    Args:
        prices: Discrete price points
        probs: Probability at each point
        coverage: Target coverage (e.g., 0.9 for 90%)

    Returns:
        Dict with 'lower', 'upper', 'coverage' keys
    """
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
```

---

## Stability Detection

### `stability_score`

Measures market stability (trending vs choppy).

```python
def stability_score(candles: list) -> float:
    """
    Compute stability score: 0 = trending, 1 = choppy/stable.

    Uses path efficiency: ratio of net move to total path length.

    Intuition:
    - Trending market: price moves in one direction (net ≈ path)
    - Choppy market: price oscillates (net << path)

    Args:
        candles: List of candles with 'vwap' field

    Returns:
        Score from 0 (trending/unstable) to 1 (choppy/stable)
    """
    if len(candles) < 2:
        return 1.0

    vwaps = [c["vwap"] for c in candles]

    # Net move: how far from start to end
    net_move = abs(vwaps[-1] - vwaps[0])

    # Total path: sum of all movements
    total_path = sum(abs(vwaps[i + 1] - vwaps[i]) for i in range(len(vwaps) - 1))

    if total_path == 0:
        return 1.0

    # Efficiency: 0 = choppy, 1 = straight line
    efficiency = net_move / total_path

    return 1 - efficiency
```

### `extract_features`

Extracts early warning features from candles.

```python
def extract_features(candles: list) -> dict:
    """
    Extract features that predict catastrophic range breaches.

    Args:
        candles: Recent OHLC candles (typically last 10)

    Returns:
        Dict with:
            - stability_trend: Change in stability (negative = getting worse)
            - range_expansion: Recent range / earlier range
            - price_velocity: % price change in last candle
            - vol_spike: Recent volume / average volume
    """
    import statistics

    if len(candles) < 6:
        return None

    vwaps = [c["vwap"] for c in candles]
    vols = [c["vol"] for c in candles]
    ranges = [c["h"] - c["l"] for c in candles]

    mid = len(candles) // 2

    # Stability trend
    def calc_stab(v):
        if len(v) < 2:
            return 1.0
        net = abs(v[-1] - v[0])
        path = sum(abs(v[i + 1] - v[i]) for i in range(len(v) - 1))
        return 1 - (net / path) if path > 0 else 1.0

    early_stab = calc_stab(vwaps[:mid])
    late_stab = calc_stab(vwaps[mid:])
    stability_trend = late_stab - early_stab  # Negative = getting less stable

    # Range expansion
    early_ranges = ranges[:mid]
    late_ranges = ranges[mid:]
    range_expansion = (
        statistics.mean(late_ranges) / statistics.mean(early_ranges)
        if statistics.mean(early_ranges) > 0 else 1
    )

    # Price velocity
    price_velocity = (
        abs(vwaps[-1] - vwaps[-2]) / vwaps[-2] * 100
        if len(vwaps) >= 2 else 0
    )

    # Volume spike
    vol_spike = (
        vols[-1] / statistics.mean(vols[:-1])
        if len(vols) >= 2 and statistics.mean(vols[:-1]) > 0 else 1
    )

    return {
        "stability_trend": stability_trend,
        "range_expansion": range_expansion,
        "price_velocity": price_velocity,
        "vol_spike": vol_spike,
    }
```

### `should_alert`

Checks if any alert threshold is breached.

```python
def should_alert(features: dict, sensitivity: float = 4.0) -> tuple[bool, list]:
    """
    Check if early warning thresholds are breached.

    Args:
        features: Output from extract_features()
        sensitivity: Threshold multiplier
            1.0 = strict (20% participation)
            4.0 = aggressive (52% participation)
            5.0 = lenient (58% participation)

    Returns:
        (should_alert: bool, triggered_features: list)
    """
    # Base thresholds (learned from training set)
    base = {
        "stability_trend": -0.0522,
        "range_expansion": 1.0940,
        "price_velocity": 0.1472,
        "vol_spike": 0.7026,
    }

    # Adjusted thresholds
    thresholds = {
        "stability_trend": base["stability_trend"] * sensitivity,
        "range_expansion": base["range_expansion"] + (sensitivity - 1) * 0.1,
        "price_velocity": base["price_velocity"] * sensitivity,
        "vol_spike": base["vol_spike"] * sensitivity,
    }

    triggered = []

    if features["stability_trend"] < thresholds["stability_trend"]:
        triggered.append("stability_trend")
    if features["range_expansion"] > thresholds["range_expansion"]:
        triggered.append("range_expansion")
    if features["price_velocity"] > thresholds["price_velocity"]:
        triggered.append("price_velocity")
    if features["vol_spike"] > thresholds["vol_spike"]:
        triggered.append("vol_spike")

    return len(triggered) > 0, triggered
```

---

## Momentum Direction

### `get_direction`

Predicts price direction based on momentum.

```python
def get_direction(lookback_swaps: list, threshold: float = 0.0018) -> str:
    """
    Predict direction based on 30-min momentum.

    94% accurate on catastrophic epochs (validated out-of-sample).

    Args:
        lookback_swaps: Recent swaps with 'price' field
            (~150 swaps ≈ 30 min on ETH/USDC)
        threshold: Momentum threshold (0.18% = 0.0018)

    Returns:
        "ETH": Go long ETH (88.8% accurate)
        "USDC": Go long USDC (97.9% accurate)
        "HOLD": Stay 50/50 (22.6% of cases)
    """
    prices = [s["price"] for s in lookback_swaps]

    if len(prices) < 150:
        return "HOLD"

    # 30-minute momentum
    momentum = (prices[-1] - prices[-150]) / prices[-150]

    if momentum > threshold:
        return "ETH"
    elif momentum < -threshold:
        return "USDC"
    else:
        return "HOLD"
```

---

## Constants

```python
# Pool configuration
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # ETH/USDC 0.05%
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# Timing
BLOCKS_PER_CANDLE = 50      # ~10 minutes at 12s/block
LOOKBACK_BLOCKS = 1000      # ~3.3 hours
LOOKAHEAD_BLOCKS = 100      # ~20 minutes

# Thresholds (aggressive profile)
ALERT_THRESHOLDS = {
    "stability_trend": -0.21,
    "range_expansion": 1.39,
    "price_velocity": 0.59,
    "vol_spike": 2.81,
}

MOMENTUM_THRESHOLD = 0.0018  # 0.18%
```
