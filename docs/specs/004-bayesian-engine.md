# 004: Bayesian Engine - Prior × Likelihood → Posterior

## Overview

Core module that combines multiple prior inputs (VWAP statistics, Opus predictions) with likelihood functions (recent OHLC data) to produce posterior distributions over future price. The posterior directly informs optimal LP range positioning.

## Core Concept

```
Bayes' Theorem:
  P(price | data) ∝ P(data | price) × P(price)
  posterior      ∝ likelihood      × prior

Our instantiation:
  prior       = VWAP-based distribution + Opus adjustment
  likelihood  = Recent OHLC data (where has price been?)
  posterior   = Combined belief about where price will be

The posterior's optimal range → LP tick boundaries
```

## Module Structure

```
src/
  bayesian/
    __init__.py
    types.py           # Distribution types
    prior.py           # Prior combination
    likelihood.py      # Likelihood from OHLC
    posterior.py       # Bayesian update
    optimizer.py       # Range optimization from posterior
    discretization.py  # Continuous ↔ discrete transformations
```

## Data Types

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable, Sequence
import numpy as np

@dataclass(frozen=True)
class DiscreteDistribution:
    """
    Discretized probability distribution over price levels.

    Prices are represented as indices into a fixed grid.
    Grid is defined relative to a reference price.
    """
    reference_price: Decimal          # Center of the grid
    grid_step_bps: int                # Grid spacing in basis points
    num_points: int                   # Total grid points (odd, centered)
    probabilities: tuple[float, ...]  # P(price at grid point i)

    @property
    def price_grid(self) -> tuple[Decimal, ...]:
        """Compute actual prices for each grid point."""
        ...

    def probability_in_range(self, low: Decimal, high: Decimal) -> float:
        """Sum probability mass within [low, high]."""
        ...

    def expected_value(self) -> Decimal:
        """E[price] under this distribution."""
        ...

    def variance(self) -> Decimal:
        """Var[price] under this distribution."""
        ...

    def percentile(self, p: float) -> Decimal:
        """Price at percentile p (0-1)."""
        ...

@dataclass(frozen=True)
class ContinuousDistribution:
    """
    Parametric continuous distribution.
    Used for analytical calculations before discretization.
    """
    family: str                       # "laplace", "gaussian", "mixture"
    params: dict                      # Family-specific parameters

    def pdf(self, x: Decimal) -> float:
        """Probability density at x."""
        ...

    def cdf(self, x: Decimal) -> float:
        """Cumulative distribution at x."""
        ...

    def discretize(
        self,
        reference: Decimal,
        grid_step_bps: int,
        num_points: int
    ) -> DiscreteDistribution:
        """Convert to discrete distribution."""
        ...

@dataclass(frozen=True)
class RangeRecommendation:
    """
    Optimal LP range derived from posterior.
    """
    lower_price: Decimal
    upper_price: Decimal
    lower_tick: int                   # Uniswap v3 tick
    upper_tick: int
    coverage_probability: float       # P(price in range)
    expected_fee_capture: float       # Relative to full range
    concentration_factor: float       # How much more concentrated than full range
    confidence: float                 # Confidence in recommendation

@dataclass(frozen=True)
class BayesianState:
    """
    Complete state of the Bayesian engine at a point in time.
    Immutable snapshot for reproducibility.
    """
    timestamp: int
    block_number: int
    current_price: Decimal
    prior: DiscreteDistribution
    likelihood: DiscreteDistribution
    posterior: DiscreteDistribution
    recommendation: RangeRecommendation
    inputs: dict                      # Raw inputs for debugging
```

## Core Functions

### Prior Combination

```python
def combine_priors(
    vwap_prior: DiscreteDistribution,
    opus_contribution: OpusPriorContribution,
) -> DiscreteDistribution:
    """
    Combine VWAP-based prior with Opus adjustment.

    Method:
    1. Shift VWAP prior center by opus center_adjustment
    2. Scale width by opus scale_multiplier
    3. Weight combination by opus weight

    Result: weighted mixture of original and adjusted priors.
    """
    ...

def build_vwap_prior(
    rolling_vwap: RollingVWAP,
    reference_price: Decimal,
    grid_step_bps: int = 10,
    num_points: int = 201
) -> DiscreteDistribution:
    """
    Construct prior from VWAP statistics.

    Laplace distribution centered on median VWAP,
    scale derived from rolling std dev.
    """
    ...
```

### Likelihood Construction

```python
def build_likelihood_from_ohlc(
    recent_ohlc: Sequence[OHLC],
    reference_price: Decimal,
    grid_step_bps: int = 10,
    num_points: int = 201,
    decay_factor: float = 0.95
) -> DiscreteDistribution:
    """
    Construct likelihood from recent OHLC data.

    Method:
    - For each candle, spread probability mass over [low, high]
    - Weight recent candles more heavily (exponential decay)
    - Normalize to sum to 1

    Interpretation: "Where has price been recently?"
    This is the likelihood P(observed data | true price level).
    """
    ...

def build_volume_weighted_likelihood(
    recent_ohlc: Sequence[OHLC],
    reference_price: Decimal,
    grid_step_bps: int = 10,
    num_points: int = 201
) -> DiscreteDistribution:
    """
    Likelihood weighted by volume.

    Higher volume at a price level = stronger evidence
    that this is a "fair" price.
    """
    ...
```

### Bayesian Update

```python
def bayesian_update(
    prior: DiscreteDistribution,
    likelihood: DiscreteDistribution
) -> DiscreteDistribution:
    """
    Compute posterior via Bayes' theorem.

    posterior[i] ∝ prior[i] × likelihood[i]

    Then normalize so sum = 1.

    Pure function, no side effects.
    """
    if prior.reference_price != likelihood.reference_price:
        raise ValueError("Prior and likelihood must share reference price")
    if prior.num_points != likelihood.num_points:
        raise ValueError("Prior and likelihood must share grid size")

    unnormalized = tuple(
        p * l for p, l in zip(prior.probabilities, likelihood.probabilities)
    )
    total = sum(unnormalized)

    if total == 0:
        # Fallback: return prior if likelihood is zero everywhere
        return prior

    normalized = tuple(p / total for p in unnormalized)

    return DiscreteDistribution(
        reference_price=prior.reference_price,
        grid_step_bps=prior.grid_step_bps,
        num_points=prior.num_points,
        probabilities=normalized
    )

def sequential_update(
    prior: DiscreteDistribution,
    likelihoods: Sequence[DiscreteDistribution]
) -> DiscreteDistribution:
    """
    Apply multiple likelihood updates sequentially.

    Equivalent to single update with product of likelihoods,
    but more numerically stable.
    """
    from functools import reduce
    return reduce(bayesian_update, likelihoods, prior)
```

### Range Optimization

```python
def optimize_range(
    posterior: DiscreteDistribution,
    target_coverage: float = 0.90,
    min_range_bps: int = 100,        # Minimum 1% range
    max_range_bps: int = 2000        # Maximum 20% range
) -> RangeRecommendation:
    """
    Find optimal LP range given posterior distribution.

    Objective: Find the tightest range that captures target_coverage
    of the probability mass.

    This maximizes fee capture efficiency:
    - Tighter range = more liquidity concentration = more fees when in range
    - But must cover enough probability to be in range often enough

    Algorithm:
    1. For each possible range width (min to max)
    2. Find optimal placement (slide window to maximize coverage)
    3. Select tightest range achieving target_coverage
    """
    ...

def optimize_range_with_fee_model(
    posterior: DiscreteDistribution,
    current_liquidity: dict[int, Decimal],  # tick -> liquidity
    target_coverage: float = 0.90,
) -> RangeRecommendation:
    """
    Optimize range considering existing liquidity.

    Insight: Positioning where liquidity is thin increases fee share.

    Modified objective:
    - Maximize: P(in range) × fee_share(range)
    - Where fee_share depends on our liquidity vs total at each tick
    """
    ...

def price_to_tick(price: Decimal, tick_spacing: int = 10) -> int:
    """Convert price to Uniswap v3 tick (rounded to spacing)."""
    import math
    raw_tick = math.floor(math.log(float(price)) / math.log(1.0001))
    return (raw_tick // tick_spacing) * tick_spacing

def tick_to_price(tick: int) -> Decimal:
    """Convert Uniswap v3 tick to price."""
    return Decimal(str(1.0001 ** tick))
```

### Confidence Estimation

```python
def estimate_recommendation_confidence(
    posterior: DiscreteDistribution,
    prior: DiscreteDistribution,
    recommendation: RangeRecommendation
) -> float:
    """
    Estimate confidence in the range recommendation.

    Factors:
    1. Posterior concentration (low variance = high confidence)
    2. Prior-posterior agreement (similar = stable belief)
    3. Coverage robustness (small range changes don't hurt much)

    Returns 0-1 confidence score.
    """
    ...

def compute_range_sensitivity(
    posterior: DiscreteDistribution,
    recommendation: RangeRecommendation,
    perturbation_bps: int = 50
) -> dict[str, float]:
    """
    How sensitive is coverage to small range changes?

    Returns:
    - coverage_if_tighter: Coverage if range shrinks by perturbation
    - coverage_if_wider: Coverage if range expands by perturbation
    - coverage_if_shifted_up: Coverage if range shifts up by perturbation
    - coverage_if_shifted_down: Coverage if range shifts down by perturbation
    """
    ...
```

## Mathematical Details

### Grid Discretization

Price grid is geometric (constant % steps), not arithmetic:

```
price[i] = reference × (1 + grid_step_bps/10000)^(i - center_index)

For grid_step_bps=10, num_points=201:
- Center index: 100
- Price[100] = reference (current price)
- Price[101] = reference × 1.001 (0.1% higher)
- Price[0] = reference × 0.905 (~10% lower)
- Price[200] = reference × 1.105 (~10% higher)
```

### Numerical Stability

Log-space computation for products:
```python
def bayesian_update_log(prior: DiscreteDistribution, likelihood: DiscreteDistribution):
    log_prior = np.log(np.array(prior.probabilities) + 1e-10)
    log_likelihood = np.log(np.array(likelihood.probabilities) + 1e-10)
    log_posterior = log_prior + log_likelihood
    log_posterior -= np.max(log_posterior)  # Prevent overflow
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()
    return posterior
```

### Prior Combination Weights

When combining VWAP prior with Opus contribution:
```
combined = (1 - opus_weight) × vwap_prior + opus_weight × adjusted_prior

where adjusted_prior = shift(scale(vwap_prior, scale_multiplier), center_adjustment)
```

## Testing Requirements

### Unit Tests
- [ ] Distribution normalization (always sums to 1)
- [ ] Bayesian update with uniform prior returns normalized likelihood
- [ ] Bayesian update with uniform likelihood returns prior
- [ ] Range optimization respects min/max bounds
- [ ] Price/tick conversions are inverses
- [ ] Log-space and linear-space updates match

### Property-Based Tests
- [ ] Posterior is always valid distribution (non-negative, sums to 1)
- [ ] Sequential updates commute (order doesn't matter)
- [ ] Tighter coverage target → tighter range
- [ ] Zero likelihood at all points → returns prior

### Integration Tests
- [ ] Full pipeline: OHLC → prior → likelihood → posterior → range
- [ ] Recommendation changes appropriately with market conditions
- [ ] Backtesting against historical positions

### Numerical Tests
- [ ] No NaN/Inf in outputs
- [ ] Handles extreme prices (very high/low)
- [ ] Handles zero/near-zero probabilities
- [ ] Deterministic (same inputs → same outputs)

## Dependencies

```python
numpy = "^1.24"          # Numerical operations
scipy = "^1.11"          # Statistical distributions, optimization
```

## Acceptance Criteria

- [ ] All distribution types with full method implementations
- [ ] Bayesian update working in log-space for stability
- [ ] Range optimizer finds optimal coverage ranges
- [ ] Confidence estimation provides meaningful scores
- [ ] Full pipeline produces sensible recommendations
- [ ] 95%+ test coverage
- [ ] Documentation with mathematical derivations
- [ ] Benchmarks showing <10ms for full update cycle

## References

- [Bayesian inference](https://en.wikipedia.org/wiki/Bayesian_inference)
- [Uniswap v3 tick math](https://docs.uniswap.org/contracts/v3/reference/core/libraries/TickMath)
- [Concentrated liquidity](https://uniswap.org/whitepaper-v3.pdf)
