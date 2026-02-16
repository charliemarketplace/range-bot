# Immediate Action Plan

## Goal
Validate edge hypothesis on 10 historical markets using Base cbBTC/USDC as price oracle.

## Steps (Linear, No Skipping)

### Step 1: Get Base cbBTC/USDC Price Feed Working (1-2 hours)
**File**: `strategy/oracle.py`

```python
# Implement:
def get_cbbtc_usdc_pool_address():
    """Find cbBTC/USDC Uniswap V3 pool on Base."""

def get_current_price():
    """Get current BTC price from pool."""

def get_price_at_timestamp(timestamp):
    """Get historical BTC price from swap events."""
```

**Validation**: Print BTC price from 10 random timestamps, compare to known values.

---

### Step 2: Fetch 10 Historical Markets (30 min)
**File**: `strategy/markets.py`

```python
# Use existing knowledge:
timestamps = [1771125900, 1771125600, ...]  # 10 recent markets

for ts in timestamps:
    market = fetch_market_data(ts)
    save_to_json(market)  # Include: start_ts, end_ts, resolution
```

**Output**: `data/historical_markets.json` with 10 markets

---

### Step 3: Run Minimal Backtest (1 hour)
**File**: `strategy/backtest.py`

```python
# For each historical market:
# 1. Get BTC price at market midpoint (2.5 min in)
# 2. Estimate probability using Bayesian model
# 3. Compare to actual resolution
# 4. Track: wins, losses, average edge

# Report:
# - Win rate (% correct predictions)
# - Average edge when model was confident
# - Would Kelly strategy have been profitable?
```

**Success Criteria**:
- Win rate > 55%
- Average edge > 2%

---

### Step 4: Decision Point
**If backtest shows edge** → Proceed to Step 5
**If no edge** → Document findings, stop project

---

### Step 5: Implement Live Monitoring (2-3 hours)
**File**: `strategy/main.py`

```python
# Minimal live loop:
while True:
    markets = get_active_markets()

    for market in markets:
        btc_price = get_current_price()
        up_price, down_price = get_token_prices(market)

        prob_up = estimate_probability_up(btc_price, market.start_btc, ...)
        edge = prob_up - up_price

        if abs(edge) > MIN_EDGE:
            print(f"EDGE DETECTED: {edge:.2%} on {market.title}")
            # Manual execution for now (copy/paste to Polymarket UI)

    sleep(2)
```

**Output**: Live edge signals, manual execution

---

### Step 6: Automate Execution (Future)
Only implement if manual execution proves profitable over 20-50 markets.

---

## Timeline

**Today**: Steps 1-3 (3-4 hours total)
**Decision**: Edge exists or not
**If yes**: Steps 5-6 (another day)

## Files to Create

```
strategy/
├── oracle.py         ← Step 1
├── markets.py        ← Step 2 (mostly done, reuse existing)
├── model.py          ← Reuse existing src/bayesian_model.py
└── backtest.py       ← Step 3

data/
└── historical_markets.json  ← Step 2 output
```

## No Over-Engineering

- No classes (use functions)
- No database (use JSON)
- No complicated execution (hold to resolution)
- No premature optimization

Get to YES/NO on edge as fast as possible.
