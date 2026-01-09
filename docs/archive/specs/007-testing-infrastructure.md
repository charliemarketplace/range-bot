# 007: Testing Infrastructure

## Overview

Comprehensive testing infrastructure for the range-bot system. Emphasizes functional programming principles: pure functions are easy to test, I/O is isolated and mockable. Includes unit tests, property-based tests, integration tests, and a backtesting framework.

## Testing Philosophy

```
Functional Programming Testing Advantages:
1. Pure functions → deterministic tests, no setup/teardown
2. Immutable data → no hidden state mutations
3. Explicit I/O → clear mock boundaries
4. Composition → test small pieces, compose with confidence

Testing Pyramid:
                    ┌───────────┐
                    │  E2E (5%) │  Testnet deployment
                   ─┼───────────┼─
                  ┌─┴───────────┴─┐
                  │Integration(15%)│  Mocked external services
                 ─┼───────────────┼─
               ┌──┴───────────────┴──┐
               │  Property-Based (20%)│  Invariants, edge cases
              ─┼─────────────────────┼─
            ┌──┴─────────────────────┴──┐
            │     Unit Tests (60%)       │  Pure function tests
            └────────────────────────────┘
```

## Module Structure

```
tests/
  __init__.py
  conftest.py              # Shared fixtures
  fixtures/
    __init__.py
    ohlc.py                # OHLC data generators
    positions.py           # Position state generators
    pools.py               # Pool state generators
    responses.py           # API response mocks
  mocks/
    __init__.py
    providers.py           # Web3 provider mocks
    anthropic.py           # Anthropic API mocks
    storage.py             # Storage backend mocks
  strategies/
    __init__.py
    data.py                # Hypothesis strategies for data types
  unit/
    test_vwap.py
    test_bayesian.py
    test_liquidity_math.py
    test_tick_math.py
    ...
  integration/
    test_pipeline.py
    test_execution.py
    test_data_fetch.py
    ...
  property/
    test_distributions.py
    test_math_invariants.py
    ...
  backtesting/
    __init__.py
    framework.py           # Backtesting engine
    scenarios.py           # Test scenarios
    test_historical.py     # Backtest runs
```

## Fixtures

### OHLC Data Fixtures

```python
# tests/fixtures/ohlc.py
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence
import json
from pathlib import Path

@dataclass(frozen=True)
class OHLCFixture:
    """Pre-computed OHLC data for testing."""
    name: str
    description: str
    data: tuple[OHLC, ...]
    expected_vwap: Decimal
    expected_volatility: Decimal

def load_fixture(name: str) -> OHLCFixture:
    """Load fixture from JSON file."""
    path = Path(__file__).parent / "data" / f"{name}.json"
    with open(path) as f:
        raw = json.load(f)
    return OHLCFixture(
        name=raw["name"],
        description=raw["description"],
        data=tuple(OHLC(**d) for d in raw["data"]),
        expected_vwap=Decimal(raw["expected_vwap"]),
        expected_volatility=Decimal(raw["expected_volatility"])
    )

# Standard fixtures
STABLE_MARKET = load_fixture("stable_market")      # Low vol, trending up
VOLATILE_MARKET = load_fixture("volatile_market")  # High vol, choppy
TRENDING_DOWN = load_fixture("trending_down")      # Clear downtrend
FLASH_CRASH = load_fixture("flash_crash")          # Sudden drop, recovery
RANGE_BOUND = load_fixture("range_bound")          # Oscillating in range

def generate_random_ohlc(
    num_candles: int,
    start_price: Decimal = Decimal("2000"),
    volatility: float = 0.02,
    seed: int = 42
) -> tuple[OHLC, ...]:
    """Generate random but realistic OHLC data."""
    import random
    random.seed(seed)

    candles = []
    price = float(start_price)

    for i in range(num_candles):
        change = random.gauss(0, volatility)
        open_price = price
        close_price = price * (1 + change)

        high = max(open_price, close_price) * (1 + abs(random.gauss(0, volatility/2)))
        low = min(open_price, close_price) * (1 - abs(random.gauss(0, volatility/2)))

        volume = random.uniform(100, 1000) * price

        candles.append(OHLC(
            period_start=i * 3600,  # Hourly
            open=Decimal(str(round(open_price, 2))),
            high=Decimal(str(round(high, 2))),
            low=Decimal(str(round(low, 2))),
            close=Decimal(str(round(close_price, 2))),
            volume=Decimal(str(round(volume, 2))),
            vwap=Decimal(str(round((high + low + close_price) / 3, 2))),
            num_swaps=random.randint(10, 100)
        ))

        price = close_price

    return tuple(candles)
```

### Position Fixtures

```python
# tests/fixtures/positions.py

def create_position_state(
    tick_lower: int = -100,
    tick_upper: int = 100,
    liquidity: int = 1_000_000,
    **overrides
) -> PositionState:
    """Create a PositionState with sensible defaults."""
    defaults = {
        "token_id": 12345,
        "owner": "0x" + "1" * 40,
        "pool": DEFAULT_POOL_CONFIG,
        "tick_lower": tick_lower,
        "tick_upper": tick_upper,
        "liquidity": liquidity,
        "tokens_owed_0": Decimal("0"),
        "tokens_owed_1": Decimal("0"),
        "token0_amount": Decimal("1.5"),
        "token1_amount": Decimal("3000"),
    }
    defaults.update(overrides)
    return PositionState(**defaults)

# Common position scenarios
POSITION_IN_RANGE = create_position_state()
POSITION_OUT_OF_RANGE_LOW = create_position_state(tick_lower=-200, tick_upper=-100)
POSITION_OUT_OF_RANGE_HIGH = create_position_state(tick_lower=100, tick_upper=200)
POSITION_WIDE_RANGE = create_position_state(tick_lower=-1000, tick_upper=1000)
POSITION_NARROW_RANGE = create_position_state(tick_lower=-10, tick_upper=10)
POSITION_WITH_FEES = create_position_state(
    tokens_owed_0=Decimal("0.1"),
    tokens_owed_1=Decimal("200")
)
```

## Mocks

### Web3 Provider Mock

```python
# tests/mocks/providers.py
from typing import Any, Callable
from unittest.mock import AsyncMock

class MockWeb3Provider:
    """
    Mock Web3 provider for testing.

    Allows configuring responses for specific calls.
    """

    def __init__(self):
        self._responses: dict[str, Any] = {}
        self._call_log: list[tuple[str, tuple]] = []

    def set_response(self, method: str, response: Any) -> None:
        """Configure response for a method."""
        self._responses[method] = response

    def set_response_fn(self, method: str, fn: Callable) -> None:
        """Configure dynamic response function."""
        self._responses[method] = fn

    async def eth_call(self, tx: dict, block: str = "latest") -> bytes:
        """Mock eth_call."""
        self._call_log.append(("eth_call", (tx, block)))
        key = f"eth_call:{tx.get('to')}:{tx.get('data')[:10]}"
        if key in self._responses:
            resp = self._responses[key]
            return resp(tx, block) if callable(resp) else resp
        return b""

    async def eth_get_logs(self, filter_params: dict) -> list[dict]:
        """Mock eth_getLogs."""
        self._call_log.append(("eth_get_logs", (filter_params,)))
        return self._responses.get("eth_get_logs", [])

    def get_call_log(self) -> list[tuple[str, tuple]]:
        """Get log of all calls made."""
        return self._call_log.copy()

    def reset(self) -> None:
        """Reset mock state."""
        self._responses.clear()
        self._call_log.clear()


def create_mock_provider_with_pool_state(pool_state: PoolState) -> MockWeb3Provider:
    """Create provider pre-configured with pool state responses."""
    provider = MockWeb3Provider()

    # slot0 response
    provider.set_response(
        f"eth_call:{pool_state.pool_address}:0x3850c7bd",
        encode_slot0_response(pool_state)
    )

    # liquidity response
    provider.set_response(
        f"eth_call:{pool_state.pool_address}:0x1a686502",
        encode_liquidity_response(pool_state.liquidity)
    )

    return provider
```

### Anthropic API Mock

```python
# tests/mocks/anthropic.py
from typing import Optional
import json

class MockAnthropicClient:
    """Mock Anthropic client for testing Opus integration."""

    def __init__(self):
        self._responses: list[dict] = []
        self._default_response: Optional[dict] = None
        self._call_count = 0

    def set_response(self, response: dict) -> None:
        """Set next response."""
        self._responses.append(response)

    def set_default_response(self, response: dict) -> None:
        """Set default response when queue is empty."""
        self._default_response = response

    def set_prediction_response(
        self,
        direction: str = "sideways",
        confidence: str = "medium",
        expected_range_percent: float = 1.5,
        reasoning: str = "Test response"
    ) -> None:
        """Convenience method for setting prediction response."""
        self.set_response({
            "content": [{
                "type": "text",
                "text": json.dumps({
                    "direction": direction,
                    "confidence": confidence,
                    "expected_range_percent": expected_range_percent,
                    "support_level": None,
                    "resistance_level": None,
                    "pattern_identified": None,
                    "reasoning": reasoning
                })
            }]
        })

    async def messages_create(self, **kwargs) -> dict:
        """Mock messages.create endpoint."""
        self._call_count += 1

        if self._responses:
            return self._responses.pop(0)
        if self._default_response:
            return self._default_response

        raise ValueError("No mock response configured")

    @property
    def call_count(self) -> int:
        return self._call_count
```

## Property-Based Testing

### Hypothesis Strategies

```python
# tests/strategies/data.py
from hypothesis import strategies as st
from hypothesis.strategies import composite
from decimal import Decimal

@composite
def ohlc_strategy(draw, min_price=100, max_price=10000):
    """Generate valid OHLC data."""
    open_price = draw(st.decimals(min_value=min_price, max_value=max_price, places=2))
    close_price = draw(st.decimals(min_value=min_price, max_value=max_price, places=2))

    low = min(open_price, close_price) * Decimal("0.95")
    high = max(open_price, close_price) * Decimal("1.05")

    return OHLC(
        period_start=draw(st.integers(min_value=0, max_value=2**32)),
        open=open_price,
        high=draw(st.decimals(min_value=float(max(open_price, close_price)), max_value=float(high), places=2)),
        low=draw(st.decimals(min_value=float(low), max_value=float(min(open_price, close_price)), places=2)),
        close=close_price,
        volume=draw(st.decimals(min_value=0, max_value=1000000, places=2)),
        vwap=draw(st.decimals(min_value=float(low), max_value=float(high), places=2)),
        num_swaps=draw(st.integers(min_value=0, max_value=1000))
    )

@composite
def distribution_strategy(draw, num_points=101):
    """Generate valid probability distribution."""
    weights = draw(st.lists(
        st.floats(min_value=0.001, max_value=1.0),
        min_size=num_points,
        max_size=num_points
    ))
    total = sum(weights)
    probabilities = tuple(w / total for w in weights)

    return DiscreteDistribution(
        reference_price=draw(st.decimals(min_value=100, max_value=10000, places=2)),
        grid_step_bps=draw(st.sampled_from([5, 10, 20, 50])),
        num_points=num_points,
        probabilities=probabilities
    )

@composite
def tick_range_strategy(draw, spacing=10):
    """Generate valid tick range."""
    lower = draw(st.integers(min_value=-887220, max_value=887210))
    lower = (lower // spacing) * spacing
    width = draw(st.integers(min_value=spacing, max_value=10000))
    width = (width // spacing) * spacing
    upper = lower + width
    return (lower, upper)
```

### Property Tests

```python
# tests/property/test_distributions.py
from hypothesis import given, settings
from tests.strategies.data import distribution_strategy, ohlc_strategy

@given(distribution_strategy())
def test_distribution_sums_to_one(dist: DiscreteDistribution):
    """Distribution probabilities must sum to 1."""
    total = sum(dist.probabilities)
    assert abs(total - 1.0) < 1e-9

@given(distribution_strategy(), distribution_strategy())
def test_bayesian_update_produces_valid_distribution(prior, likelihood):
    """Bayesian update always produces valid distribution."""
    # Align distributions
    aligned_likelihood = align_distribution(likelihood, prior.reference_price, prior.grid_step_bps)

    posterior = bayesian_update(prior, aligned_likelihood)

    assert abs(sum(posterior.probabilities) - 1.0) < 1e-9
    assert all(p >= 0 for p in posterior.probabilities)

@given(st.lists(ohlc_strategy(), min_size=10, max_size=100))
def test_vwap_within_price_range(candles):
    """VWAP must be within the price range of candles."""
    vwap = compute_vwap_from_ohlc(candles)

    all_lows = [c.low for c in candles]
    all_highs = [c.high for c in candles]

    assert min(all_lows) <= vwap <= max(all_highs)

@given(tick_range_strategy())
def test_tick_price_roundtrip(tick_range):
    """Converting tick to price and back should be stable."""
    lower, upper = tick_range

    lower_price = tick_to_price(lower)
    upper_price = tick_to_price(upper)

    recovered_lower = price_to_tick(lower_price, tick_spacing=10)
    recovered_upper = price_to_tick(upper_price, tick_spacing=10)

    # Allow for rounding to tick spacing
    assert abs(recovered_lower - lower) <= 10
    assert abs(recovered_upper - upper) <= 10
```

## Backtesting Framework

```python
# tests/backtesting/framework.py
from dataclasses import dataclass
from decimal import Decimal
from typing import Sequence, Optional
import json

@dataclass(frozen=True)
class BacktestConfig:
    """Configuration for a backtest run."""
    start_block: int
    end_block: int
    chain: str
    pool_address: str
    initial_capital_usd: Decimal
    rebalance_config: RebalanceConfig
    bayesian_config: dict  # target_coverage, etc.

@dataclass(frozen=True)
class BacktestStep:
    """Single step in backtest."""
    block_number: int
    timestamp: int
    price: Decimal
    position_lower_tick: int
    position_upper_tick: int
    position_value_usd: Decimal
    fees_earned_usd: Decimal
    in_range: bool
    action: str  # "hold", "rebalance", "collect"
    gas_cost_usd: Decimal

@dataclass(frozen=True)
class BacktestResult:
    """Results of a backtest run."""
    config: BacktestConfig
    steps: tuple[BacktestStep, ...]
    total_fees_earned: Decimal
    total_gas_spent: Decimal
    final_value: Decimal
    roi_percent: float
    time_in_range_percent: float
    num_rebalances: int
    sharpe_ratio: Optional[float]

    def to_json(self) -> str:
        """Serialize for storage/analysis."""
        ...

class BacktestEngine:
    """
    Engine for running historical backtests.

    Uses stored historical data to simulate the full pipeline.
    """

    def __init__(
        self,
        data_loader: DataLoader,
        config: BacktestConfig
    ):
        self.data_loader = data_loader
        self.config = config

    def run(self) -> BacktestResult:
        """
        Run backtest over configured block range.

        For each hour:
        1. Load historical data up to that point
        2. Run Bayesian pipeline (without Opus - use historical vol only)
        3. Make rebalance decision
        4. Simulate execution
        5. Track P&L
        """
        steps = []
        position = self._initialize_position()

        for block in range(self.config.start_block, self.config.end_block, BLOCKS_PER_HOUR):
            step = self._simulate_step(block, position)
            steps.append(step)

            if step.action == "rebalance":
                position = self._update_position(position, step)

        return self._compute_results(steps)

    def _simulate_step(self, block: int, position: PositionState) -> BacktestStep:
        """Simulate a single pipeline step."""
        # Load historical data
        ohlc = self.data_loader.get_ohlc_at_block(block, lookback_hours=24)
        pool_state = self.data_loader.get_pool_state_at_block(block)

        # Run pipeline (without Opus)
        vwap_prior = build_vwap_prior(compute_rolling_vwap(ohlc, 100), pool_state.price)
        likelihood = build_likelihood_from_ohlc(ohlc, pool_state.price)
        posterior = bayesian_update(vwap_prior, likelihood)
        recommendation = optimize_range(posterior)

        # Decision
        decision, _ = should_rebalance(position, recommendation, pool_state, self.config.rebalance_config)

        # Compute metrics
        in_range = position.tick_lower <= pool_state.tick <= position.tick_upper
        fees = self._estimate_fees_earned(position, pool_state, in_range)
        gas_cost = self._estimate_gas_cost(decision, block)

        return BacktestStep(
            block_number=block,
            timestamp=self.data_loader.get_timestamp(block),
            price=pool_state.price,
            position_lower_tick=position.tick_lower,
            position_upper_tick=position.tick_upper,
            position_value_usd=self._compute_position_value(position, pool_state),
            fees_earned_usd=fees,
            in_range=in_range,
            action=decision.value,
            gas_cost_usd=gas_cost
        )

    def _compute_results(self, steps: Sequence[BacktestStep]) -> BacktestResult:
        """Aggregate step results into final metrics."""
        total_fees = sum(s.fees_earned_usd for s in steps)
        total_gas = sum(s.gas_cost_usd for s in steps)
        time_in_range = sum(1 for s in steps if s.in_range) / len(steps) * 100
        num_rebalances = sum(1 for s in steps if s.action == "rebalance")

        returns = [
            float((steps[i].position_value_usd - steps[i-1].position_value_usd) / steps[i-1].position_value_usd)
            for i in range(1, len(steps))
        ]

        sharpe = self._compute_sharpe(returns) if len(returns) > 10 else None

        return BacktestResult(
            config=self.config,
            steps=tuple(steps),
            total_fees_earned=total_fees,
            total_gas_spent=total_gas,
            final_value=steps[-1].position_value_usd,
            roi_percent=float((steps[-1].position_value_usd - self.config.initial_capital_usd) / self.config.initial_capital_usd * 100),
            time_in_range_percent=time_in_range,
            num_rebalances=num_rebalances,
            sharpe_ratio=sharpe
        )
```

### Backtest Scenarios

```python
# tests/backtesting/scenarios.py

SCENARIOS = {
    "stable_2023_q4": BacktestConfig(
        start_block=18_000_000,
        end_block=18_500_000,
        chain="ethereum",
        pool_address="0x...",
        initial_capital_usd=Decimal("10000"),
        rebalance_config=RebalanceConfig(),
        bayesian_config={"target_coverage": 0.9}
    ),
    "volatile_2024_q1": BacktestConfig(
        start_block=19_000_000,
        end_block=19_500_000,
        ...
    ),
    "base_launch": BacktestConfig(
        chain="base",
        ...
    )
}

def run_all_scenarios() -> dict[str, BacktestResult]:
    """Run all defined scenarios and return results."""
    results = {}
    for name, config in SCENARIOS.items():
        engine = BacktestEngine(DataLoader(config.chain), config)
        results[name] = engine.run()
    return results
```

## Test Configuration

### pytest Configuration

```ini
# pytest.ini
[pytest]
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short --strict-markers
markers =
    unit: Unit tests (fast, no I/O)
    integration: Integration tests (slower, may use mocks)
    property: Property-based tests (hypothesis)
    backtest: Backtesting tests (slow, uses historical data)
    slow: Slow tests (skip with -m "not slow")

filterwarnings =
    ignore::DeprecationWarning
```

### Coverage Configuration

```ini
# .coveragerc
[run]
source = src
omit =
    */tests/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise NotImplementedError
    if TYPE_CHECKING:

fail_under = 85
```

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/unit -v --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3

  property-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/property -v --hypothesis-seed=42

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration -v

  backtest:
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: pip install -e ".[dev]"
      - run: pytest tests/backtesting -v -m "not slow"
```

## Dependencies

```python
# Dev dependencies
pytest = "^7.4"
pytest-cov = "^4.1"
pytest-asyncio = "^0.21"
hypothesis = "^6.88"
pytest-xdist = "^3.3"     # Parallel test execution
```

## Acceptance Criteria

- [ ] Fixture generators for all data types
- [ ] Mocks for Web3 provider and Anthropic client
- [ ] Hypothesis strategies for property tests
- [ ] Backtest framework with historical data loading
- [ ] At least 3 backtest scenarios defined
- [ ] pytest configuration with markers
- [ ] Coverage configuration targeting 85%+
- [ ] CI/CD pipeline running all test types
- [ ] Documentation on running tests locally
- [ ] Example test for each module

## References

- [pytest documentation](https://docs.pytest.org/)
- [Hypothesis](https://hypothesis.readthedocs.io/)
- [Property-based testing intro](https://hypothesis.works/articles/what-is-hypothesis/)
