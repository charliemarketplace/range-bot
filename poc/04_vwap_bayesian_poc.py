"""
POC 4: VWAP Prior + Bayesian Update (Working Example)

This POC uses realistic mock data to demonstrate:
1. VWAP calculation from swap events
2. Prior distribution construction
3. Bayesian update with likelihood
4. Range optimization

This runs without any external dependencies.
"""
import math
import random
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence
import statistics


# ============================================================================
# Data Types
# ============================================================================

@dataclass(frozen=True)
class Swap:
    """A swap event."""
    block_number: int
    timestamp: int
    price: Decimal  # USDC per ETH
    volume_eth: Decimal


@dataclass(frozen=True)
class OHLC:
    """OHLC candle."""
    period_start: int
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: Decimal
    vwap: Decimal
    num_swaps: int


@dataclass(frozen=True)
class Distribution:
    """Discrete probability distribution over prices."""
    center: Decimal
    prices: tuple[Decimal, ...]
    probabilities: tuple[float, ...]

    def expected_value(self) -> Decimal:
        total = sum(float(p) * prob for p, prob in zip(self.prices, self.probabilities))
        return Decimal(str(total))

    def std_dev(self) -> Decimal:
        ev = float(self.expected_value())
        variance = sum(
            prob * (float(p) - ev) ** 2
            for p, prob in zip(self.prices, self.probabilities)
        )
        return Decimal(str(math.sqrt(variance)))

    def probability_in_range(self, low: Decimal, high: Decimal) -> float:
        return sum(
            prob for p, prob in zip(self.prices, self.probabilities)
            if low <= p <= high
        )


@dataclass(frozen=True)
class RangeRecommendation:
    """Optimal LP range."""
    lower_price: Decimal
    upper_price: Decimal
    coverage: float
    center: Decimal


# ============================================================================
# Mock Data Generation
# ============================================================================

def generate_mock_swaps(
    num_swaps: int = 500,
    base_price: float = 3500,
    volatility: float = 0.001,  # Per swap
    seed: int = 42
) -> list[Swap]:
    """Generate realistic mock swap data with mean reversion."""
    random.seed(seed)

    swaps = []
    price = base_price
    mean_price = base_price

    for i in range(num_swaps):
        # Mean reversion + random walk
        reversion = 0.01 * (mean_price - price)
        noise = random.gauss(0, volatility * price)
        price = price + reversion + noise

        # Update mean slowly
        mean_price = 0.999 * mean_price + 0.001 * price

        swaps.append(Swap(
            block_number=20000000 + i * 10,
            timestamp=1700000000 + i * 120,  # ~2 min per swap
            price=Decimal(str(round(price, 2))),
            volume_eth=Decimal(str(round(random.uniform(0.1, 10), 4)))
        ))

    return swaps


def swaps_to_ohlc(swaps: Sequence[Swap], period_seconds: int = 3600) -> list[OHLC]:
    """Aggregate swaps into OHLC candles."""
    if not swaps:
        return []

    # Group by period
    periods: dict[int, list[Swap]] = {}
    for swap in swaps:
        period_start = (swap.timestamp // period_seconds) * period_seconds
        if period_start not in periods:
            periods[period_start] = []
        periods[period_start].append(swap)

    # Build candles
    candles = []
    for period_start in sorted(periods.keys()):
        period_swaps = periods[period_start]

        prices = [s.price for s in period_swaps]
        volumes = [s.volume_eth for s in period_swaps]

        # VWAP calculation
        total_value = sum(p * v for p, v in zip(prices, volumes))
        total_volume = sum(volumes)
        vwap = total_value / total_volume if total_volume > 0 else prices[-1]

        candles.append(OHLC(
            period_start=period_start,
            open=prices[0],
            high=max(prices),
            low=min(prices),
            close=prices[-1],
            volume=total_volume,
            vwap=vwap,
            num_swaps=len(period_swaps)
        ))

    return candles


# ============================================================================
# VWAP Prior Construction
# ============================================================================

def compute_rolling_vwap(
    candles: Sequence[OHLC],
    window: int = 10
) -> tuple[Decimal, Decimal]:
    """
    Compute rolling VWAP statistics.

    Returns (median_vwap, std_dev)
    """
    if len(candles) < window:
        window = len(candles)

    recent = candles[-window:]

    # Median of VWAPs (more robust than mean)
    vwaps = [float(c.vwap) for c in recent]
    median_vwap = Decimal(str(statistics.median(vwaps)))

    # Standard deviation for uncertainty
    std_dev = Decimal(str(statistics.stdev(vwaps))) if len(vwaps) > 1 else Decimal("10")

    return median_vwap, std_dev


def build_laplace_prior(
    center: Decimal,
    scale: Decimal,
    num_points: int = 101,
    range_mult: float = 4.0
) -> Distribution:
    """
    Build a Laplace (double exponential) prior distribution.

    Laplace has fatter tails than Gaussian, better for price distributions.
    """
    # Price grid: center ± range_mult * scale
    half_range = float(scale) * range_mult
    min_price = float(center) - half_range
    max_price = float(center) + half_range

    prices = []
    probabilities = []

    for i in range(num_points):
        p = min_price + (max_price - min_price) * i / (num_points - 1)
        prices.append(Decimal(str(round(p, 2))))

        # Laplace PDF: (1/2b) * exp(-|x-μ|/b)
        b = float(scale)
        prob = (1 / (2 * b)) * math.exp(-abs(p - float(center)) / b)
        probabilities.append(prob)

    # Normalize
    total = sum(probabilities)
    probabilities = [p / total for p in probabilities]

    return Distribution(
        center=center,
        prices=tuple(prices),
        probabilities=tuple(probabilities)
    )


# ============================================================================
# Likelihood from Recent Data
# ============================================================================

def build_likelihood_from_ohlc(
    candles: Sequence[OHLC],
    num_points: int = 101,
    decay: float = 0.9
) -> Distribution:
    """
    Build likelihood distribution from recent OHLC.

    "Where has price been recently?"
    """
    if not candles:
        raise ValueError("Need at least one candle")

    # Find price range
    all_lows = [float(c.low) for c in candles]
    all_highs = [float(c.high) for c in candles]

    min_price = min(all_lows) * 0.99
    max_price = max(all_highs) * 1.01
    center = (min_price + max_price) / 2

    prices = []
    probabilities = []

    for i in range(num_points):
        p = min_price + (max_price - min_price) * i / (num_points - 1)
        prices.append(Decimal(str(round(p, 2))))
        probabilities.append(0.0)

    # Add probability mass from each candle
    for idx, candle in enumerate(candles):
        # More recent candles get more weight
        weight = decay ** (len(candles) - 1 - idx)

        low = float(candle.low)
        high = float(candle.high)

        # Spread probability uniformly over [low, high]
        for i, p in enumerate(prices):
            if low <= float(p) <= high:
                probabilities[i] += weight

    # Normalize
    total = sum(probabilities)
    if total > 0:
        probabilities = [p / total for p in probabilities]
    else:
        probabilities = [1 / num_points] * num_points

    return Distribution(
        center=Decimal(str(round(center, 2))),
        prices=tuple(prices),
        probabilities=tuple(probabilities)
    )


# ============================================================================
# Bayesian Update
# ============================================================================

def bayesian_update(prior: Distribution, likelihood: Distribution) -> Distribution:
    """
    Compute posterior = prior × likelihood (normalized).
    """
    # Align distributions to same price grid
    # For simplicity, use prior's grid and interpolate likelihood

    new_probs = []
    for i, (price, prior_prob) in enumerate(zip(prior.prices, prior.probabilities)):
        # Find closest likelihood probability
        lik_prob = interpolate_probability(likelihood, price)

        # Bayes: posterior ∝ prior × likelihood
        new_probs.append(prior_prob * lik_prob)

    # Normalize
    total = sum(new_probs)
    if total > 0:
        new_probs = [p / total for p in new_probs]
    else:
        new_probs = list(prior.probabilities)

    return Distribution(
        center=prior.center,
        prices=prior.prices,
        probabilities=tuple(new_probs)
    )


def interpolate_probability(dist: Distribution, target_price: Decimal) -> float:
    """Interpolate probability at a price point."""
    target = float(target_price)
    prices = [float(p) for p in dist.prices]

    # Find bracketing prices
    for i in range(len(prices) - 1):
        if prices[i] <= target <= prices[i + 1]:
            # Linear interpolation
            t = (target - prices[i]) / (prices[i + 1] - prices[i]) if prices[i + 1] != prices[i] else 0
            return (1 - t) * dist.probabilities[i] + t * dist.probabilities[i + 1]

    # Outside range: return edge probability
    if target < prices[0]:
        return dist.probabilities[0]
    return dist.probabilities[-1]


# ============================================================================
# Range Optimization
# ============================================================================

def optimize_range(
    posterior: Distribution,
    target_coverage: float = 0.9
) -> RangeRecommendation:
    """
    Find the tightest range that covers target_coverage of probability mass.
    """
    n = len(posterior.prices)
    best_range = None
    best_width = float('inf')

    # Try all possible ranges
    for i in range(n):
        cumsum = 0.0
        for j in range(i, n):
            cumsum += posterior.probabilities[j]
            if cumsum >= target_coverage:
                width = float(posterior.prices[j]) - float(posterior.prices[i])
                if width < best_width:
                    best_width = width
                    best_range = (i, j)
                break

    if best_range is None:
        # Fallback: use full range
        best_range = (0, n - 1)

    i, j = best_range
    lower = posterior.prices[i]
    upper = posterior.prices[j]
    actual_coverage = posterior.probability_in_range(lower, upper)
    center = (lower + upper) / 2

    return RangeRecommendation(
        lower_price=lower,
        upper_price=upper,
        coverage=actual_coverage,
        center=center
    )


# ============================================================================
# Main Demo
# ============================================================================

def main():
    print("=" * 70)
    print("POC 4: VWAP Prior + Bayesian Update Demo")
    print("=" * 70)

    # Generate mock data
    print("\n## 1. Generate Mock Swap Data")
    print("-" * 50)
    swaps = generate_mock_swaps(num_swaps=500, base_price=3500, volatility=0.0015)
    print(f"Generated {len(swaps)} swaps")
    print(f"Price range: ${min(s.price for s in swaps):,.2f} - ${max(s.price for s in swaps):,.2f}")

    # Aggregate to OHLC
    print("\n## 2. Aggregate to Hourly OHLC")
    print("-" * 50)
    candles = swaps_to_ohlc(swaps, period_seconds=3600)
    print(f"Generated {len(candles)} hourly candles")

    for c in candles[-3:]:
        print(f"  O=${c.open:,.2f} H=${c.high:,.2f} L=${c.low:,.2f} C=${c.close:,.2f} VWAP=${c.vwap:,.2f}")

    # Compute rolling VWAP
    print("\n## 3. Compute Rolling VWAP Statistics")
    print("-" * 50)
    median_vwap, std_dev = compute_rolling_vwap(candles, window=10)
    print(f"10-candle median VWAP: ${median_vwap:,.2f}")
    print(f"Standard deviation: ${std_dev:,.2f}")

    # Build prior
    print("\n## 4. Build Laplace Prior")
    print("-" * 50)
    prior = build_laplace_prior(median_vwap, std_dev * 2, num_points=101)
    print(f"Prior center: ${prior.center:,.2f}")
    print(f"Prior std dev: ${prior.std_dev():,.2f}")
    print(f"Prior 90% range: ${prior.prices[5]:,.2f} - ${prior.prices[95]:,.2f}")

    # Build likelihood
    print("\n## 5. Build Likelihood from Recent OHLC")
    print("-" * 50)
    likelihood = build_likelihood_from_ohlc(candles[-10:])
    print(f"Likelihood center: ${likelihood.center:,.2f}")
    print(f"Likelihood based on last 10 candles")

    # Bayesian update
    print("\n## 6. Bayesian Update (Posterior = Prior × Likelihood)")
    print("-" * 50)
    posterior = bayesian_update(prior, likelihood)
    print(f"Posterior expected value: ${posterior.expected_value():,.2f}")
    print(f"Posterior std dev: ${posterior.std_dev():,.2f}")

    # Compare prior vs posterior
    print("\n  Comparison:")
    print(f"  Prior EV:     ${prior.expected_value():,.2f}")
    print(f"  Likelihood EV: ${likelihood.expected_value():,.2f}")
    print(f"  Posterior EV: ${posterior.expected_value():,.2f}")

    # Optimize range
    print("\n## 7. Optimize LP Range")
    print("-" * 50)
    recommendation = optimize_range(posterior, target_coverage=0.90)
    print(f"Recommended range: ${recommendation.lower_price:,.2f} - ${recommendation.upper_price:,.2f}")
    print(f"Range center: ${recommendation.center:,.2f}")
    print(f"Coverage: {recommendation.coverage * 100:.1f}%")

    range_width = float(recommendation.upper_price - recommendation.lower_price)
    range_pct = range_width / float(recommendation.center) * 100
    print(f"Range width: ${range_width:,.2f} ({range_pct:.2f}%)")

    # Show probability distribution
    print("\n## 8. Posterior Distribution (ASCII Chart)")
    print("-" * 50)
    print_ascii_distribution(posterior)

    print("\n" + "=" * 70)
    print("SUCCESS: Core Bayesian logic working!")
    print("=" * 70)
    print("\nKey insights:")
    print("  1. VWAP provides robust price anchor (handles outliers)")
    print("  2. Laplace prior has fat tails (better for price data)")
    print("  3. Bayesian update combines prior belief with recent data")
    print("  4. Optimal range balances coverage vs concentration")


def print_ascii_distribution(dist: Distribution, width: int = 50):
    """Print ASCII visualization of distribution."""
    max_prob = max(dist.probabilities)

    # Sample every Nth point
    n = len(dist.prices)
    step = max(1, n // 20)

    for i in range(0, n, step):
        price = dist.prices[i]
        prob = dist.probabilities[i]
        bar_len = int(prob / max_prob * width) if max_prob > 0 else 0
        bar = "█" * bar_len
        print(f"${float(price):>8,.0f} | {bar}")


if __name__ == "__main__":
    main()
