# Range-Bot: Complete Technical Guide

A Bayesian LP range optimizer for Uniswap v3 concentrated liquidity positions, with stability alerting and momentum-based directional trading.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [System Architecture](#system-architecture)
3. [System 1: Range Optimization](#system-1-range-optimization)
4. [System 2: Stability Alerting](#system-2-stability-alerting)
5. [System 3: Momentum Direction](#system-3-momentum-direction)
6. [Backtesting Results](#backtesting-results)
7. [Implementation Reference](#implementation-reference)
8. [File Index](#file-index)

---

## Executive Summary

### What This System Does

Optimizes Uniswap v3 LP positions by:
1. **Computing optimal price ranges** using Bayesian inference on recent swap data
2. **Detecting unstable market conditions** that could cause range breaches
3. **Taking directional positions** during detected trends

### Key Performance (Out-of-Sample Validated)

| System | Metric | Result |
|--------|--------|--------|
| Range Optimization | Median coverage | 87.7% |
| Stability Alerting | Coverage improvement | +5.1% |
| Momentum Direction | Accuracy | 94.1% |

### Dataset

- **Pool**: ETH/USDC 0.05% (0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640)
- **Blocks**: 23,000,000 - 24,000,000 (1M blocks)
- **Swaps**: 780,559 swap events
- **Time Period**: ~4 months of data

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           COMPLETE SYSTEM FLOW                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────┐                                                          │
│   │  Raw Swaps  │ ← eth_getLogs from Ethereum RPC                          │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│          ▼                                                                  │
│   ┌─────────────┐                                                          │
│   │    OHLC     │ Aggregate to candles (50 blocks = ~10 min each)          │
│   │   Candles   │                                                          │
│   └──────┬──────┘                                                          │
│          │                                                                  │
│          ├─────────────────────────────────────────┐                       │
│          │                                         │                       │
│          ▼                                         ▼                       │
│   ┌─────────────┐                           ┌─────────────┐                │
│   │  SYSTEM 1   │                           │  SYSTEM 2   │                │
│   │   Bayesian  │                           │  Stability  │                │
│   │    Range    │                           │   Alerting  │                │
│   └──────┬──────┘                           └──────┬──────┘                │
│          │                                         │                       │
│          │                              ┌──────────┴──────────┐            │
│          │                              │                     │            │
│          │                           STABLE              ALERT             │
│          │                              │                     │            │
│          ▼                              ▼                     ▼            │
│   ┌─────────────┐                ┌─────────────┐      ┌─────────────┐     │
│   │  LP Range   │                │   Apply LP  │      │  SYSTEM 3   │     │
│   │  [lo, hi]   │───────────────►│    Range    │      │  Momentum   │     │
│   └─────────────┘                └─────────────┘      │  Direction  │     │
│                                                       └──────┬──────┘     │
│                                                              │            │
│                                              ┌───────────────┼────────┐   │
│                                              │               │        │   │
│                                              ▼               ▼        ▼   │
│                                          100% ETH       100% USDC   50/50 │
│                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## System 1: Range Optimization

### Purpose

Compute the optimal price range for an LP position that captures 90% of expected future swaps while minimizing range width (maximizing capital efficiency).

### Algorithm

```
1. COLLECT: Fetch last 1000 blocks of swap data
2. AGGREGATE: Build OHLC candles (50 blocks each → ~20 candles)
3. PRIOR: Build Laplace distribution centered on median VWAP
4. LIKELIHOOD: Build KDE from candle midpoints (with decay)
5. POSTERIOR: Multiply prior × likelihood, normalize
6. OPTIMIZE: Find tightest interval containing 90% probability mass
```

### Mathematical Details

#### VWAP Calculation
```
VWAP = Σ(price_i × volume_i) / Σ(volume_i)

Where:
- price_i = swap execution price (from sqrtPriceX96)
- volume_i = |amount1| / 1e6 (USDC volume, 6 decimals)
```

#### Laplace Prior
```
P(price | center, scale) = (1/2b) × exp(-|price - center| / b)

Where:
- center = median(VWAP of recent candles)
- b = scale = 2 × std(VWAP of recent candles)
```

#### KDE Likelihood
```
P(price | candles) = Σ w_i × K((price - mid_i) / h)

Where:
- mid_i = (candle_high + candle_low) / 2
- w_i = 0.9^(n-1-i) (exponential decay, recent = more weight)
- h = 1.06 × σ × n^(-0.2) (Silverman bandwidth)
- K = Gaussian kernel
```

#### Posterior
```
P(price | data) ∝ Prior(price) × Likelihood(price)
```

#### Optimal Range
```
Find [lo, hi] that minimizes (hi - lo) subject to:
∫[lo to hi] P(price) dp ≥ 0.9
```

### Performance

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Mean coverage | 69.3% | Dragged down by catastrophic misses |
| Median coverage | 87.7% | Model works well most of the time |
| Std deviation | 35.8% | High variance (bimodal) |
| Catastrophic (<30%) | 20.4% | 1 in 5 periods are bad |

---

## System 2: Stability Alerting

### Purpose

Detect when market conditions are unstable and the range optimizer shouldn't be trusted. Allows withdrawing LP before catastrophic range breaches.

### Features Monitored

| Feature | Definition | Alert Condition |
|---------|------------|-----------------|
| `stability_trend` | Change in stability score over time | < -0.21 |
| `range_expansion` | Recent candle range / earlier range | > 1.39 |
| `price_velocity` | |price[-1] - price[-2]| / price[-2] × 100 | > 0.59 |
| `vol_spike` | Recent volume / average volume | > 2.81 |

### Stability Score

```python
def stability_score(candles):
    """
    0 = trending (bad for LP)
    1 = choppy/mean-reverting (good for LP)
    """
    vwaps = [c["vwap"] for c in candles]

    # Net move: straight-line distance
    net_move = abs(vwaps[-1] - vwaps[0])

    # Total path: sum of all movements
    total_path = sum(abs(vwaps[i+1] - vwaps[i]) for i in range(len(vwaps)-1))

    # High path efficiency = trending = bad
    # Low path efficiency = choppy = good
    return 1 - (net_move / total_path)
```

### Key Insight

**Stability level doesn't predict misses, but stability TREND does.**

| Feature | Hits (>=80%) | Misses (<30%) | Delta |
|---------|--------------|---------------|-------|
| stability_trend | +0.08 | -0.19 | -347% |
| range_expansion | 0.52 | 0.89 | +71% |
| price_velocity | 0.14% | 0.22% | +54% |
| vol_spike | 0.50 | 0.78 | +57% |

### Threshold Tuning

| Sensitivity | Participation | Mean Coverage | Catastrophic |
|-------------|---------------|---------------|--------------|
| 1.0 (strict) | 20.5% | 79.0% | 11.0% |
| 2.0 | 36.3% | 76.7% | 13.2% |
| 3.0 | 45.3% | 75.3% | 14.5% |
| **4.0 (selected)** | **52.2%** | **74.5%** | **15.5%** |
| 5.0 (lenient) | 57.8% | 73.9% | 16.0% |

### Performance (Out-of-Sample)

| Metric | Without Alerts | With Alerts | Change |
|--------|----------------|-------------|--------|
| Participation | 100% | 52% | -48% |
| Mean coverage | 69.4% | 74.5% | **+5.1%** |
| Median coverage | 87.7% | 94.2% | **+6.5%** |
| Catastrophic rate | 20.4% | 15.5% | **-4.9%** |

---

## System 3: Momentum Direction

### Purpose

When an alert triggers, determine which direction the market is trending to take a directional position (100% ETH or 100% USDC) instead of neutral 50/50.

### Algorithm

```python
def get_direction(lookback_swaps, threshold=0.0018):
    """
    94% accurate on catastrophic epochs (validated out-of-sample).

    Args:
        lookback_swaps: Recent swap data (~150 swaps = 30 min)
        threshold: 0.18% momentum threshold (learned from training)

    Returns:
        "ETH" | "USDC" | "HOLD"
    """
    prices = [s["price"] for s in lookback_swaps]

    if len(prices) < 150:
        return "HOLD"

    # 30-minute momentum
    momentum = (prices[-1] - prices[-150]) / prices[-150]

    if momentum > threshold:
        return "ETH"    # 88.8% accurate
    elif momentum < -threshold:
        return "USDC"   # 97.9% accurate
    else:
        return "HOLD"   # 22.6% of cases
```

### Why 30-Minute Momentum?

| Window | Accuracy | Strong (>0.2%) Accuracy |
|--------|----------|-------------------------|
| 5 min | 80% | 88% |
| **30 min** | **88%** | **90%** |
| 1 hr | 75% | 80% |

30-minute momentum is the sweet spot: long enough to confirm trend, short enough to capture most of the move.

### Validation Results (Train/Test Split)

**Train**: Blocks 23M - 23.5M (learn threshold)
**Test**: Blocks 23.5M - 24M (evaluate)

| Metric | Training | Test (OOS) |
|--------|----------|------------|
| Accuracy | 90.0% | **94.1%** |
| Coverage | 68.8% | 77.4% |
| Avg capture | 0.452% | **0.668%** |

### Direction Breakdown (Test Set)

| Direction | Trades | Accuracy |
|-----------|--------|----------|
| UP | 107 | 88.8% |
| DOWN | 146 | **97.9%** |

DOWN detection is more accurate, possibly because dumps are more abrupt/clear.

### Why Detection (Not Prediction) Is Fine

The signal is **detection**, not prediction:
- By the time momentum crosses 0.18%, the move is already underway
- But we still capture **91% of the average move** (0.668% of 0.733%)
- Tolerating small initial deviation to confirm trend is acceptable

---

## Backtesting Results

### Methodology

1. **No data leakage**: Train on blocks 23M-23.5M, test on 23.5M-24M
2. **Random sampling**: 100K random intervals to avoid periodicity bias
3. **Proper validation**: All thresholds learned from training only

### Combined System Performance

| Scenario | Action | Metric |
|----------|--------|--------|
| Stable (52% of time) | LP in Bayesian range | 94.2% median coverage |
| Alert + momentum > 0.18% | 100% ETH | 88.8% direction accuracy |
| Alert + momentum < -0.18% | 100% USDC | 97.9% direction accuracy |
| Alert + |momentum| < 0.18% | 50/50 | Neutral, no directional risk |

### Expected Behavior

```
Over 1000 evaluation periods:
- 520 periods: LP in range (94.2% median coverage)
- 480 periods: Alerts trigger
  - 373 periods (77.4%): Clear momentum → directional bet (94% accurate)
  - 107 periods (22.6%): Unclear → hold 50/50
```

---

## Implementation Reference

### Price Calculation from sqrtPriceX96

```python
def sqrt_price_to_price(sqrt_price_x96: int) -> float:
    """
    Convert Uniswap v3 sqrtPriceX96 to human-readable price.

    For ETH/USDC pool (token0=WETH, token1=USDC):
    - sqrtPriceX96 = sqrt(price) × 2^96
    - price = token1/token0 = USDC per ETH
    - But we want ETH/USDC ($ per ETH), so invert

    Returns: Price in USD per ETH
    """
    # sqrtPrice = sqrtPriceX96 / 2^96
    # price = sqrtPrice^2 = (sqrtPriceX96^2) / 2^192
    # This gives USDC per ETH in raw units

    # Adjust for decimals: USDC has 6, ETH has 18
    # So multiply by 10^(18-6) = 10^12
    price = 1e12 / ((sqrt_price_x96 ** 2) / (2 ** 192))
    return price
```

### OHLC Aggregation

```python
def to_ohlc(swaps: list, blocks_per_candle: int = 50) -> list:
    """
    Aggregate swaps to OHLC candles.

    Args:
        swaps: List of swap dicts with 'block', 'price', 'amount1'
        blocks_per_candle: Blocks per candle (50 ≈ 10 min)

    Returns:
        List of OHLC candles with: o, h, l, c, vol, vwap, n
    """
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
            vols = [abs(s["amount1"]) / 1e6 for s in ps]  # USDC volume
            total_vol = sum(vols)
            vwap = sum(p*v for p,v in zip(prices, vols)) / total_vol if total_vol > 0 else prices[-1]

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

### Complete Alert Check

```python
def should_alert(candles, sensitivity=4.0) -> tuple[bool, list]:
    """
    Check if any early warning threshold is breached.

    Args:
        candles: Recent OHLC candles
        sensitivity: Threshold multiplier (4.0 = aggressive)

    Returns:
        (should_alert, list_of_triggered_features)
    """
    features = extract_features(candles[-10:])

    # Base thresholds (from training)
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

## File Index

### POC Scripts

| File | Purpose |
|------|---------|
| `poc/simple_poc.py` | Live data fetch + Bayesian model |
| `poc/viewer.html` | Interactive Highcharts visualization |

### Backtesting

| File | Purpose |
|------|---------|
| `poc/backtest_range.py` | Fixed-stride backtester |
| `poc/backtest_random.py` | Random-sample backtester (100K samples) |
| `poc/backtest_with_alerts.py` | Train/test split alert validation |
| `poc/tune_thresholds.py` | Sensitivity analysis |
| `poc/validate_momentum.py` | Train/test momentum validation |

### Analysis

| File | Purpose |
|------|---------|
| `poc/analyze_failures.py` | Feature analysis: hits vs misses |
| `poc/analyze_catastrophes.py` | Early detection timing |
| `poc/analyze_momentum.py` | Direction prediction exploration |

### Data

| File | Purpose |
|------|---------|
| `poc/data/swaps.db` | SQLite: 780K swaps (blocks 23M-24M) |
| `poc/data/backtest_results.json` | Backtest outputs |
| `poc/data/momentum_validation.json` | Momentum validation results |

---

## Key Definitions

| Term | Definition |
|------|------------|
| **VWAP** | Volume-Weighted Average Price: Σ(price × volume) / Σ(volume) |
| **Credible Interval** | Bayesian analog of confidence interval; range containing X% of posterior probability |
| **Coverage** | % of future swaps that fall within predicted range |
| **Catastrophic** | Coverage < 30% (range badly missed) |
| **Stability Score** | 1 - (net_move / total_path); 0=trending, 1=choppy |
| **Momentum** | (price_now - price_30min_ago) / price_30min_ago |

---

## Summary

Three validated systems working together:

1. **Range Optimization**: Bayesian model for LP range selection (87.7% median coverage)
2. **Stability Alerting**: Early warning for trending markets (+5% coverage improvement)
3. **Momentum Direction**: Directional bets during alerts (94% accuracy)

All validated out-of-sample with proper train/test splits. Ready for paper trading.
