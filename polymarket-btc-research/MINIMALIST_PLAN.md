# Minimalist Trading Plan

## Core Insight
Polymarket 5-min BTC markets resolve based on Chainlink price, but market-implied probabilities (token prices) can misprice during the window. Buy mispriced tokens, hold to resolution.

## Data Sources

### 1. Historical (Backtesting)
- **Polymarket API**: Market start/end times, resolutions (free, proven)
- **Base cbBTC/USDC Uniswap V3**: Swap events for historical BTC pricing (on-chain, free)

### 2. Live (Production)
- **Chainlink BTC/USD**: Current price via Polymarket RTDS or on-chain (real-time)
- **Polymarket CLOB**: UP/DOWN token prices (order book, real-time)

## Strategy (Functional)

```
Input: market, current_btc_price, up_token_price, down_token_price, seconds_remaining

Step 1: Calculate fair probability
  bayesian_prob_up = f(current_btc, market_start_btc, volatility, seconds_remaining)

Step 2: Identify mispricing
  market_implied_prob_up = up_token_price
  edge = bayesian_prob_up - market_implied_prob_up

Step 3: Position sizing (Kelly)
  if edge > threshold:
    kelly_fraction = edge / (edge + 1)  # Simplified Kelly
    position_size = bankroll * kelly_fraction * safety_factor

Step 4: Execution decision
  if edge > min_edge_threshold AND position_size > min_position:
    buy_token = UP if bayesian_prob_up > market_implied_prob_up else DOWN
    amount = position_size
    EXECUTE: place_order(buy_token, amount)

Step 5: Exit strategy (choose ONE)
  Option A: Hold to resolution (simplest, no gas waste)
  Option B: Stop loss if edge reverses
  Option C: Take profit at 15s before end if target hit
```

## Implementation (Minimalist Modules)

### Module 1: Price Oracle (`oracle.py`)
```python
def get_btc_price_from_base():
    """Query Base cbBTC/USDC pool for current BTC price."""
    # On-chain call to Uniswap V3 pool
    # Return: float (BTC price in USD)

def get_historical_btc_prices(start_ts, end_ts):
    """Get historical BTC prices from Base for backtesting."""
    # Query swap events, build price series
    # Return: [(timestamp, price), ...]
```

### Module 2: Market Data (`markets.py`)
```python
def get_active_markets():
    """Get currently active 5-min BTC markets from Polymarket."""
    # Polymarket Gamma API
    # Return: [market_dict, ...]

def get_token_prices(up_token_id, down_token_id):
    """Get current UP/DOWN token prices from order book."""
    # Polymarket CLOB API
    # Return: (up_price, down_price)
```

### Module 3: Model (`model.py`)
```python
def estimate_probability_up(current_btc, start_btc, volatility, seconds_remaining):
    """Bayesian estimation of P(BTC_end >= BTC_start)."""
    # Brownian motion model
    # Return: float (probability 0-1)

def calculate_kelly_size(edge, bankroll, safety_factor=0.25):
    """Kelly criterion position sizing."""
    # Kelly fraction with safety adjustment
    # Return: float (position size in USDC)
```

### Module 4: Execution (`execute.py`)
```python
def place_order(token_id, side, amount):
    """Place order on Polymarket CLOB."""
    # Sign order, submit via API
    # Return: order_id

def check_position():
    """Check if position should be closed early."""
    # Evaluate stop loss / take profit conditions
    # Return: bool (should_exit)
```

### Module 5: Main Loop (`main.py`)
```python
def run_strategy():
    while True:
        markets = get_active_markets()

        for market in markets:
            # Get current prices
            btc_price = get_btc_price_from_base()
            up_price, down_price = get_token_prices(market)

            # Calculate edge
            prob_up = estimate_probability_up(btc_price, market.start_btc, ...)
            edge = prob_up - up_price

            # Execute if edge exists
            if abs(edge) > MIN_EDGE:
                size = calculate_kelly_size(edge, BANKROLL)
                token = UP if edge > 0 else DOWN
                place_order(token, 'BUY', size)

        sleep(POLL_INTERVAL)
```

## Execution Options (Ranked by Simplicity)

### Option 1: Hold to Resolution (Simplest) ✅ RECOMMEND
- Buy mispriced token
- Do nothing until market resolves
- No stop loss, no early exit
- Minimal gas, minimal complexity
- **Risk**: Full loss if wrong

### Option 2: Time-Based Take Profit
- Buy mispriced token
- If profit target hit AND >15s before end: sell
- Otherwise hold to resolution
- **Complexity**: +1 transaction per exit

### Option 3: Dynamic Stop Loss
- Buy mispriced token
- Monitor edge continuously
- If edge reverses beyond threshold: exit
- Otherwise hold to resolution
- **Complexity**: Continuous monitoring + conditional exits

**START WITH**: Option 1 (hold to resolution)

## Minimal Viable Backtest

### Data Needed
1. 10-20 historical markets with:
   - Start/end timestamps
   - Actual resolution (Up/Down)
   - Historical BTC prices from Base (at start, middle, end of each market)

### Test Logic
```python
for market in historical_markets:
    # At market midpoint (2.5 min in)
    btc_current = get_historical_btc_price(market.start_ts + 150)

    # Estimate probability
    prob_up = model.estimate_probability_up(
        btc_current,
        market.start_btc,
        volatility=default_vol,
        seconds_remaining=150
    )

    # Simulate: if we bought at this probability
    edge = prob_up - 0.5  # Assume market was 50/50

    # Check actual outcome
    actual_win = (market.resolution == "Up" and prob_up > 0.5) or \
                 (market.resolution == "Down" and prob_up < 0.5)

    # Track: win rate, average edge, Kelly performance
```

**Success Metric**: Win rate > 55% with consistent edge > 2%

## File Structure (Minimalist)

```
strategy/
├── oracle.py         # Base cbBTC/USDC price feed
├── markets.py        # Polymarket data (markets, order books)
├── model.py          # Bayesian probability + Kelly sizing
├── execute.py        # Order placement (if needed)
├── backtest.py       # Historical validation
└── main.py           # Live strategy loop

data/
├── historical_markets.json
└── backtest_results.json

config.yaml           # Bankroll, thresholds, safety factors
```

## Next Immediate Steps

1. **Implement `oracle.py`** - Get Base cbBTC/USDC working (historical + live)
2. **Validate model** - Run backtest on 10 historical markets
3. **If backtest shows edge** → implement live execution
4. **If no edge** → document and stop

## Critical Constraints

- **Functional style**: Pure functions, no classes unless necessary
- **Minimal dependencies**: web3, requests, numpy/scipy only
- **No over-engineering**: Start with Option 1 (hold to resolution)
- **Fast validation**: Test on 10 markets before building execution

## Risk Parameters (Example)

```yaml
bankroll: 1000  # Total USDC allocated
min_edge_threshold: 0.02  # 2% minimum edge to trade
kelly_safety_factor: 0.25  # Use 25% of Kelly (conservative)
max_position_size: 100  # Max $100 per market
poll_interval: 2  # Check markets every 2 seconds
```

## Success Criteria

**Backtest (10 markets)**:
- Win rate > 55%
- Average edge > 2%
- Kelly returns > baseline

**Live (50 markets)**:
- Actual win rate matches backtest ±5%
- Profitable after gas costs
- No catastrophic losses

If criteria not met → document findings, archive project
