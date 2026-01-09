# 002: VWAP Prior Module

## Overview

Module for computing rolling Volume-Weighted Average Price (VWAP) and deriving statistical priors for price distribution. This is the foundation of the "Prior 1" insight: the 100-block median VWAP is revisited 90%+ of the time within the next 1000 blocks.

## Core Concept

```
Price tends to mean-revert to recent volume-weighted levels.

Given: 100-block rolling median VWAP at block N
Observation: Price crosses this level again within blocks N+1 to N+1000
             with >90% probability (validated blocks 15M-16M on mainnet)

This statistical regularity informs our Bayesian prior for where to position
LP ranges.
```

## Module Structure

```
src/
  prior/
    __init__.py
    types.py           # Prior distribution types
    vwap.py            # VWAP calculation functions
    statistics.py      # Rolling statistics, median, percentiles
    distribution.py    # Prior distribution construction
    validation.py      # Backtesting prior accuracy
```

## Data Types

```python
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence

@dataclass(frozen=True)
class VWAP:
    """Single VWAP computation result."""
    block_start: int
    block_end: int
    value: Decimal
    total_volume: Decimal
    num_swaps: int

@dataclass(frozen=True)
class RollingVWAP:
    """Rolling VWAP over a window."""
    current_block: int
    window_blocks: int
    vwap: Decimal
    median_vwap: Decimal      # Median of per-block VWAPs in window
    std_dev: Decimal          # Standard deviation of price in window

@dataclass(frozen=True)
class PriorDistribution:
    """
    Prior belief about where price will be.
    Represented as a discretized probability distribution over price levels.
    """
    center: Decimal           # Expected value (median VWAP)
    std_dev: Decimal          # Uncertainty width
    price_levels: tuple[Decimal, ...]  # Discrete price points
    probabilities: tuple[float, ...]   # P(price near level) for each

    def probability_in_range(self, low: Decimal, high: Decimal) -> float:
        """Probability that price falls within [low, high]."""
        ...

    def optimal_range(self, coverage: float = 0.9) -> tuple[Decimal, Decimal]:
        """Return tightest range containing `coverage` probability mass."""
        ...

@dataclass(frozen=True)
class PriorValidation:
    """Results of backtesting prior accuracy."""
    block_range: tuple[int, int]
    predictions: int
    hits: int                 # Times price revisited predicted level
    hit_rate: float
    mean_blocks_to_hit: float
    false_positive_rate: float
```

## Core Functions

### VWAP Calculation

```python
def compute_vwap(swaps: Sequence[Swap]) -> VWAP:
    """
    Compute VWAP from a sequence of swaps.

    VWAP = Σ(price_i × volume_i) / Σ(volume_i)

    Pure function, no side effects.
    """
    ...

def compute_rolling_vwap(
    swaps: Sequence[Swap],
    window_blocks: int,
    current_block: int
) -> RollingVWAP:
    """
    Compute rolling VWAP statistics over a block window.

    Returns median VWAP (more robust than mean) plus standard deviation
    for uncertainty estimation.
    """
    ...
```

### Prior Construction

```python
def build_prior_from_vwap(
    rolling_vwap: RollingVWAP,
    historical_volatility: Decimal,
    confidence_window_blocks: int = 1000
) -> PriorDistribution:
    """
    Construct a prior distribution centered on median VWAP.

    The width of the distribution is informed by:
    - Recent volatility (std_dev from rolling window)
    - Historical hit rate calibration
    - Confidence window (how far ahead we're predicting)

    Returns a discretized distribution suitable for Bayesian updating.
    """
    ...

def adjust_prior_for_volatility_regime(
    base_prior: PriorDistribution,
    current_volatility: Decimal,
    baseline_volatility: Decimal
) -> PriorDistribution:
    """
    Widen or narrow prior based on current vs baseline volatility.

    High volatility regime -> wider, less confident prior
    Low volatility regime -> tighter, more confident prior
    """
    ...
```

### Validation

```python
def validate_prior_accuracy(
    historical_data: Sequence[OHLC],
    window_blocks: int = 100,
    forward_blocks: int = 1000
) -> PriorValidation:
    """
    Backtest the VWAP-based prior against historical data.

    For each block N:
    1. Compute 100-block median VWAP
    2. Check if price touches that level in blocks N+1 to N+1000
    3. Record hit/miss

    Returns aggregate statistics.
    """
    ...

def compute_calibration_curve(
    historical_data: Sequence[OHLC],
    confidence_levels: Sequence[float] = (0.5, 0.7, 0.9, 0.95)
) -> dict[float, float]:
    """
    Check if prior confidence intervals are well-calibrated.

    A well-calibrated 90% interval should contain the actual price
    90% of the time.

    Returns {target_coverage: actual_coverage} mapping.
    """
    ...
```

## Statistical Foundations

### Why Median VWAP?

- **Median** is robust to outliers (flash crashes, large single trades)
- **VWAP** weights by volume, so large trades matter more
- Together: a robust estimate of "fair value" according to market activity

### Prior Distribution Shape

Default: **Laplace distribution** (double exponential)
- Fatter tails than Gaussian
- Better matches observed price return distributions
- Simple parameterization: location (median) + scale (volatility)

```python
def laplace_prior(center: Decimal, scale: Decimal, num_points: int = 100) -> PriorDistribution:
    """Construct a Laplace-distributed prior."""
    ...
```

### Volatility Adjustment

Volatility is not constant. The prior should adapt:

```
adjusted_scale = base_scale × (current_vol / historical_vol)^α

where α ∈ [0.5, 1.0] controls sensitivity to vol changes
```

## Functional Requirements

### Purity
All functions in this module must be pure:
- No I/O
- No random state (PRNGs must be seeded externally)
- No mutation

### Composition
Functions should compose cleanly:
```python
# Pipeline style
prior = pipe(
    swaps,
    partial(compute_rolling_vwap, window_blocks=100, current_block=block),
    partial(build_prior_from_vwap, historical_volatility=vol),
    partial(adjust_prior_for_volatility_regime, current_volatility=curr_vol, baseline_volatility=base_vol)
)
```

## Testing Requirements

### Unit Tests
- [ ] VWAP calculation matches manual computation
- [ ] Median VWAP is within [min, max] of individual VWAPs
- [ ] Prior distribution sums to 1.0
- [ ] `optimal_range(0.9)` contains ~90% of probability mass
- [ ] Volatility adjustment widens prior when vol increases

### Property-Based Tests
- [ ] VWAP is always within [low, high] of swaps
- [ ] Rolling statistics are invariant to swap ordering within same block
- [ ] Prior is symmetric around center (for symmetric underlying distribution)

### Validation Tests
- [ ] Run `validate_prior_accuracy` on blocks 15M-16M
- [ ] Confirm >90% hit rate for 100-block window, 1000-block forward
- [ ] Calibration curve shows reasonable calibration

## Dependencies

```python
# Core
numpy = "^1.24"          # Numerical operations
scipy = "^1.11"          # Statistical distributions

# Optional
numba = "^0.58"          # JIT compilation for hot paths
```

## Acceptance Criteria

- [ ] All core functions implemented as pure functions
- [ ] VWAP calculation tested against known values
- [ ] Prior distribution type with probability queries
- [ ] Validation script reproduces 90%+ hit rate on historical data
- [ ] Calibration check shows no severe miscalibration
- [ ] Documentation with mathematical foundations
- [ ] 95%+ test coverage

## References

- [VWAP Wikipedia](https://en.wikipedia.org/wiki/Volume-weighted_average_price)
- [Laplace Distribution](https://en.wikipedia.org/wiki/Laplace_distribution)
- Original research: blocks 15M-16M mainnet validation
