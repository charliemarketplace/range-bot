# Stability Alert System

## Summary

This document captures the key findings from backtesting the Bayesian LP range optimizer on 1M blocks of historical Uniswap v3 ETH/USDC swap data (blocks 23M-24M).

## The Core Problem

The Bayesian model computes a 90% credible interval for future swap prices. Initial backtesting showed:

- **Mean coverage: 69%** (should be 90%)
- **Median coverage: 88%** (close to target!)

The gap between mean and median revealed the issue: **the model works well most of the time, but occasional catastrophic misses drag down the mean**.

## Key Insight: Regime Detection

Rather than fixing the model (which works fine in stable conditions), we should **detect when the model shouldn't be trusted**.

### Stability Score (First Attempt)

```python
def stability_score(candles):
    """0 = trending, 1 = stable (choppy)"""
    net_move = abs(vwaps[-1] - vwaps[0])
    total_path = sum(|vwaps[i+1] - vwaps[i]|)
    return 1 - (net_move / total_path)
```

Finding: The **level** of stability doesn't predict misses well. Both hits and misses had ~0.63 stability scores.

### Early Warning Features (Breakthrough)

Analyzing what distinguishes catastrophic misses from good predictions:

| Feature | Hits | Misses | Delta |
|---------|------|--------|-------|
| **stability_trend** | +0.08 | -0.19 | -347% |
| **range_expansion** | 0.52 | 0.89 | +71% |
| **price_velocity** | 0.14% | 0.22% | +54% |
| **vol_spike** | 0.50 | 0.78 | +57% |

**Key insight**: It's not *where* stability is, but *where it's going*. Before a miss:
- Stability is **decreasing** (becoming more directional)
- Candle ranges are **expanding**
- Price is **accelerating**
- Volume is **spiking**

## Train/Test Validation

Proper ML methodology with no data leakage:
- **Train**: Blocks 23M - 23.5M (learn thresholds)
- **Test**: Blocks 23.5M - 24M (evaluate)

### Learned Thresholds (from training set)

```python
thresholds = {
    "stability_trend": -0.0522,   # Alert if below (stability decreasing)
    "range_expansion": 1.0940,    # Alert if above
    "price_velocity": 0.1472,     # Alert if above
    "vol_spike": 0.7026,          # Alert if above
}
```

### Out-of-Sample Results

| Metric | LP Always | LP With Alerts | Change |
|--------|-----------|----------------|--------|
| Mean Coverage | 69.4% | 79.0% | +9.6% |
| Median Coverage | 87.7% | 97.9% | +10.2% |
| Catastrophic (<30%) | 20.4% | 11.0% | -9.4% |
| Good (>=80%) | 54.9% | 65.5% | +10.7% |

**The early warning signals generalize to unseen data.**

### Tradeoff

Current thresholds are conservative:
- Skip 79.5% of periods
- Only LP 20.5% of the time
- Skipped periods had 66.9% avg coverage (not terrible)

This is tunable based on risk appetite.

## Architecture Decision

Two-layer system:

```
┌─────────────────────────────────────────────────────────┐
│                    LAYER 1: ALERTING                    │
│                                                         │
│   Monitors: stability_trend, range_expansion,           │
│             price_velocity, vol_spike                   │
│                                                         │
│   Decision: Should we LP right now?                     │
│   - YES → proceed to Layer 2                            │
│   - NO  → withdraw liquidity, wait                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                 LAYER 2: RANGE OPTIMIZATION             │
│                                                         │
│   Bayesian model (assumes stable conditions):           │
│   - Prior: Laplace centered on median VWAP              │
│   - Likelihood: KDE over recent candle midpoints        │
│   - Posterior: Prior × Likelihood                       │
│   - Output: Optimal 90% credible interval               │
└─────────────────────────────────────────────────────────┘
```

## Files

- `backtest_range.py` - Fixed-stride backtester with stability segmentation
- `backtest_random.py` - Random-sample backtester (unbiased)
- `analyze_failures.py` - Feature analysis: hits vs misses
- `backtest_with_alerts.py` - Train/test split with early warning system

## Threshold Tuning Results

Tested sensitivity multipliers from 0.5 (strict) to 5.0 (lenient):

| Sensitivity | Participation | Mean Coverage | Improvement | Catastrophic |
|-------------|---------------|---------------|-------------|--------------|
| 1.0 (strict) | 20.5% | 79.0% | +9.6% | 11.0% |
| 2.0 | 36.3% | 76.7% | +7.4% | 13.2% |
| 3.0 | 45.3% | 75.3% | +5.9% | 14.5% |
| 4.0 | 52.2% | 74.5% | +5.1% | 15.5% |
| 5.0 (lenient) | 57.8% | 73.9% | +4.5% | 16.0% |

### Selected: Aggressive (Sensitivity 4.0)

```python
thresholds = {
    "stability_trend": -0.21,    # Base * 4.0
    "range_expansion": 1.39,     # Base + 0.30
    "price_velocity": 0.59,      # Base * 4.0
    "vol_spike": 2.81,           # Base * 4.0
}
```

This gives:
- **~52% participation** (LP more than half the time)
- **+5.1% coverage improvement** (74.5% vs 69.4% baseline)
- **15.5% catastrophic rate** (down from 20.4%)
- **94.2% median coverage** (up from 87.7%)

### Risk Profiles

| Profile | Sensitivity | Use Case |
|---------|-------------|----------|
| Conservative | 1.0 | Maximum protection, accept low participation |
| Balanced | 2.5 | Good tradeoff for most users |
| Aggressive | 4.0+ | Maximize fee earnings, accept more risk |

## Next Steps

1. **Implement real-time alerting** in the live system
2. **Add withdrawal/reentry logic** when alerts trigger
3. **Consider the "what to do during alerts"** question (hold stables? swap to trending asset?)

## Key Learnings

1. **Median vs Mean matters** - High variance systems need both metrics
2. **Trend > Level** - The direction of change is more predictive than absolute values
3. **Train/Test split is essential** - Features looked great in-sample, validated out-of-sample
4. **Conservative isn't always better** - 80% skip rate may leave money on the table
