# Range-Bot System Overview

## What We Built

Two separate systems that could work independently or together:

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM 1                                    │
│                   Range Optimization                                │
│                  (Well-Validated)                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: Recent swap data (1000 blocks lookback)                   │
│                                                                     │
│   Process:                                                          │
│   1. Aggregate to OHLC candles (50 blocks each)                    │
│   2. Build Laplace prior centered on median VWAP                   │
│   3. Build KDE likelihood from candle midpoints                    │
│   4. Bayesian update: Posterior = Prior × Likelihood               │
│   5. Find tightest 90% credible interval                           │
│                                                                     │
│   Output: [lower, upper] price range for LP position               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM 2                                    │
│                   Stability Alerting                                │
│                  (Validated with train/test split)                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: Recent candle features                                     │
│                                                                     │
│   Features monitored:                                               │
│   - stability_trend: Is market becoming more directional?          │
│   - range_expansion: Are candle ranges growing?                    │
│   - price_velocity: Is price accelerating?                         │
│   - vol_spike: Is volume spiking?                                  │
│                                                                     │
│   Output: ALERT (withdraw LP) or OK (continue LPing)               │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         SYSTEM 3                                    │
│                   Momentum Direction                                │
│                  (NEEDS MORE VALIDATION)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   Input: 30-min price momentum at alert time                       │
│                                                                     │
│   Output: Directional bet (ETH vs USDC)                            │
│                                                                     │
│   ⚠️  CAVEATS - SEE BELOW                                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## System 1: Range Optimization

### Backtesting Performance

**Dataset**: 780K swaps, blocks 23M-24M, ETH/USDC 0.05% pool

**Method**: 100K random samples, 1000-block lookback, 100-block lookahead

| Metric | Value |
|--------|-------|
| Mean coverage | 69.3% |
| Median coverage | 87.7% |
| Std deviation | 35.8% |
| P10-P90 range | [6.8%, 100%] |

### Interpretation

- **Median near target**: 87.7% median suggests the model works well most of the time
- **High variance**: Bimodal distribution - many ~100% hits, some catastrophic misses
- **Mean dragged down**: Catastrophic misses (~20% of periods) pull mean to 69%

### Key Insight

The model is well-calibrated for **stable, mean-reverting markets**. It fails during **trending regimes** where price moves directionally out of the predicted range.

---

## System 2: Stability Alerting

### Backtesting Performance

**Method**: Train on blocks 23M-23.5M, test on blocks 23.5M-24M (no leakage)

**Aggressive thresholds (sensitivity=4.0)**:

| Metric | Without Alerts | With Alerts | Change |
|--------|----------------|-------------|--------|
| Participation | 100% | 52% | -48% |
| Mean coverage | 69.4% | 74.5% | +5.1% |
| Median coverage | 87.7% | 94.2% | +6.5% |
| Catastrophic rate | 20.4% | 15.5% | -4.9% |

### Learned Thresholds

```python
# Aggressive profile (sensitivity=4.0)
thresholds = {
    "stability_trend": -0.21,   # Alert if below
    "range_expansion": 1.39,    # Alert if above
    "price_velocity": 0.59,     # Alert if above
    "vol_spike": 2.81,          # Alert if above
}
```

### Interpretation

- **Validated out-of-sample**: Thresholds learned from first half work on second half
- **Tradeoff is real**: More protection = less participation
- **Improvement modest but real**: +5% coverage, -5% catastrophes

---

## System 3: Momentum Direction (CAUTION)

### Reported Performance

| Momentum Window | Accuracy |
|-----------------|----------|
| 30 min | 88% |
| 30 min (strong >0.2%) | 90% |

### Why This Needs More Validation

**Potential issues:**

1. **Look-ahead bias**: We measure momentum at t=0 (decision point), but the "catastrophe" unfolds over the next 100 blocks. If price is already moving, we're detecting the move, not predicting it.

2. **No train/test split**: Unlike System 2, we didn't validate momentum thresholds out-of-sample.

3. **Survivorship in definition**: "Catastrophe" is defined by future coverage <30%. The momentum that creates the catastrophe is inherently correlated with the direction.

4. **Too good to be true**: 90% accuracy on direction in crypto markets is extraordinary and warrants skepticism.

### Proper Validation Needed

Before trusting System 3:

```python
# 1. Train/test split on momentum thresholds
# 2. Measure momentum BEFORE the alert triggers, not at trigger time
# 3. Test on completely different time periods (different market regimes)
# 4. Paper trade before real capital
```

### Conservative Approach

Until validated, treat alerts as:
- **Withdraw LP** ✓ (validated)
- **Go to 50/50 or stables** (safe default)
- **Directional bet** ✗ (not yet validated)

---

## File Index

### Core Scripts

| File | Purpose | Validation Level |
|------|---------|------------------|
| `simple_poc.py` | Live data fetch + Bayesian model | Working |
| `viewer.html` | Interactive visualization | Working |

### Backtesting

| File | Purpose | Validation Level |
|------|---------|------------------|
| `backtest_range.py` | Fixed-stride backtest | ✓ Complete |
| `backtest_random.py` | Random-sample backtest | ✓ Complete |
| `backtest_with_alerts.py` | Train/test alert validation | ✓ Complete |
| `tune_thresholds.py` | Threshold sensitivity analysis | ✓ Complete |

### Analysis

| File | Purpose | Validation Level |
|------|---------|------------------|
| `analyze_failures.py` | Feature analysis: hits vs misses | ✓ Complete |
| `analyze_catastrophes.py` | Early detection timing | ✓ Complete |
| `analyze_momentum.py` | Direction prediction | ⚠️ Needs validation |

### Documentation

| File | Purpose |
|------|---------|
| `stability_alert.md` | Alert system deep-dive |
| `SYSTEM_OVERVIEW.md` | This file |
| `DATA_STRATEGY.md` | Data architecture |

---

## Recommended Next Steps

### Immediate (Before Live)

1. **Validate momentum out-of-sample**: Proper train/test split
2. **Test on different time periods**: Bull market, bear market, sideways
3. **Paper trade**: Run system without real capital

### Architecture

1. **Separate the systems**: Run range optimization and alerting independently
2. **A/B test**: Compare "alert → 50/50" vs "alert → momentum bet"
3. **Measure regret**: Track what would have happened under each strategy

### Production

1. **Real-time data pipeline**: WebSocket to Uniswap events
2. **Alert delivery**: Push notifications, Telegram bot
3. **Execution**: Integration with LP management (rebalance, withdraw)

---

## Summary

| System | Status | Trust Level |
|--------|--------|-------------|
| Range Optimization | Complete | High (well-tested) |
| Stability Alerting | Complete | Medium-High (validated OOS) |
| Momentum Direction | Exploratory | Low (needs validation) |

**Bottom line**: We have a working range optimizer and a validated alert system. The momentum direction signal is intriguing but unproven - treat it as a hypothesis to test, not a strategy to deploy.
