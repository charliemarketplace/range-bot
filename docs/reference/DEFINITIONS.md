# Key Definitions & Concepts

---

## Uniswap v3 Concepts

### Concentrated Liquidity

Unlike v2 where liquidity is spread across all prices (0 to ∞), v3 allows LPs to concentrate liquidity in specific price ranges. This increases capital efficiency but requires active management.

```
v2: Liquidity spread 0 → ∞ (low efficiency, passive)
v3: Liquidity in [lo, hi] (high efficiency, active management)
```

### sqrtPriceX96

Uniswap v3's internal price representation. Stores `sqrt(price) × 2^96` as a uint160 for precision without floating point.

```
sqrtPriceX96 = √(token1/token0) × 2^96

To convert to human-readable price:
price = (sqrtPriceX96 / 2^96)^2
      = sqrtPriceX96^2 / 2^192
```

### Tick

Discrete price points in Uniswap v3. Each tick represents a 0.01% price change.

```
price = 1.0001^tick
tick = log(price) / log(1.0001)
```

### Pool Tokens

For ETH/USDC 0.05% pool:
- **token0**: WETH (18 decimals)
- **token1**: USDC (6 decimals)
- **Fee tier**: 0.05% (500 bps)

---

## Price Metrics

### VWAP (Volume-Weighted Average Price)

Average price weighted by trade volume. Large trades influence VWAP more than small trades.

```
VWAP = Σ(price_i × volume_i) / Σ(volume_i)
```

**Why VWAP?** Better represents "fair value" than simple average because it accounts for market activity.

### OHLC (Open, High, Low, Close)

Standard candle representation:
- **Open**: First price in period
- **High**: Maximum price in period
- **Low**: Minimum price in period
- **Close**: Last price in period

Additional fields we track:
- **Volume**: Total trading volume (in USDC)
- **VWAP**: Volume-weighted average price
- **N**: Number of swaps

---

## Statistical Concepts

### Laplace Distribution

Also called "double exponential" distribution. Used as our prior because:
1. Fatter tails than Gaussian (accounts for outliers)
2. Simple parameterization (location + scale)
3. Better matches observed price returns

```
PDF: f(x|μ,b) = (1/2b) × exp(-|x-μ|/b)

Parameters:
- μ (mu): Location (center/mean)
- b: Scale (controls spread)
```

### KDE (Kernel Density Estimation)

Non-parametric method to estimate probability density from data points. Places a "kernel" (e.g., Gaussian) at each data point and sums them.

```
KDE(x) = (1/n) × Σ K((x - x_i) / h)

Where:
- K: Kernel function (we use Gaussian)
- h: Bandwidth (controls smoothness)
- x_i: Data points
```

**Silverman's Rule** for bandwidth selection:
```
h = 1.06 × σ × n^(-0.2)
```

### Credible Interval

Bayesian analog of confidence interval. A 90% credible interval contains 90% of the posterior probability mass.

**Key difference from frequentist CI**: The credible interval directly gives the probability that the parameter lies within the interval.

---

## Model Metrics

### Coverage

Percentage of future swaps that fall within the predicted range.

```
coverage = (swaps inside range) / (total swaps)
```

| Coverage | Interpretation |
|----------|----------------|
| 100% | Perfect prediction |
| 90% | Target (matches 90% CI) |
| 60-80% | Acceptable |
| <30% | Catastrophic miss |

### Calibration

A model is "well-calibrated" if predicted probabilities match observed frequencies.

```
For a 90% CI: actual coverage should be ~90%
For a 50% CI: actual coverage should be ~50%
```

Our model's calibration (on test set):

| Target | Actual (100 blocks) |
|--------|---------------------|
| 50% | ~62% |
| 70% | ~70% |
| 90% | ~88% |

---

## Stability Metrics

### Stability Score

Measures whether price action is trending or mean-reverting.

```
stability = 1 - (net_move / total_path)

Where:
- net_move = |price_end - price_start|
- total_path = Σ|price[i+1] - price[i]|
```

| Score | Interpretation |
|-------|----------------|
| 0.0 | Pure trend (straight line) |
| 0.5 | Moderate trend |
| 1.0 | Pure oscillation (no net move) |

### Stability Trend

Change in stability score over time. More predictive than stability level.

```
stability_trend = stability(recent) - stability(earlier)

Negative trend = market becoming more directional (bad for LP)
Positive trend = market becoming choppier (good for LP)
```

### Range Expansion

Ratio of recent candle ranges to earlier ranges. Indicates volatility changes.

```
range_expansion = mean(recent_ranges) / mean(earlier_ranges)

> 1.0: Volatility increasing
< 1.0: Volatility decreasing
```

### Price Velocity

Rate of price change (as percentage).

```
velocity = |price[-1] - price[-2]| / price[-2] × 100
```

### Volume Spike

Ratio of recent volume to average volume.

```
vol_spike = volume[-1] / mean(volume[:-1])

> 1.0: Above average volume
< 1.0: Below average volume
```

---

## Momentum

### 30-Minute Momentum

Price change over ~30 minutes (150 swaps on ETH/USDC).

```
momentum = (price_now - price_30min_ago) / price_30min_ago

> +0.18%: Uptrend detected → favor ETH
< -0.18%: Downtrend detected → favor USDC
else: Unclear → stay neutral
```

### Why 0.18% Threshold?

Learned from training data as optimal tradeoff:
- Lower threshold: More false signals
- Higher threshold: Miss more moves
- 0.18%: Best accuracy (94%) while covering 77% of catastrophes

---

## Backtesting Terms

### Train/Test Split

Divide data into non-overlapping sets:
- **Training set**: Learn model parameters
- **Test set**: Evaluate performance

Our split:
- Train: Blocks 23M - 23.5M
- Test: Blocks 23.5M - 24M

### Out-of-Sample (OOS)

Performance on data not used for training. The true test of model validity.

### Data Leakage

When information from the test set influences training. Invalidates results. Examples:
- Using future data to set thresholds
- Overlapping train/test periods
- Feature engineering using test set statistics

### Sensitivity (Alert Thresholds)

Multiplier that adjusts alert strictness:

```
threshold_adjusted = threshold_base × sensitivity
```

| Sensitivity | Behavior |
|-------------|----------|
| 1.0 | Strict (many alerts, few LPs) |
| 4.0 | Aggressive (fewer alerts, more LPing) |
| 5.0 | Lenient (rare alerts, maximum LPing) |

---

## Performance Terms

### Participation Rate

Percentage of time the system is actively LPing (not in alert mode).

```
participation = (non-alert periods) / (total periods) × 100
```

### Catastrophic Rate

Percentage of periods with coverage < 30%.

```
catastrophic_rate = (periods with coverage < 30%) / (total periods) × 100

Baseline: ~20%
With alerts: ~15%
```

### Avg Capture

During trending periods, how much of the price move we capture by trading directionally.

```
avg_capture = mean(captured_moves)

Where captured_move = price_change × direction_accuracy
```

---

## Block/Time Conversions

Ethereum averages ~12 seconds per block.

| Blocks | Approximate Time |
|--------|------------------|
| 1 | 12 seconds |
| 5 | 1 minute |
| 50 | 10 minutes |
| 100 | 20 minutes |
| 300 | 1 hour |
| 1000 | 3.3 hours |
| 7200 | 1 day |

---

## Summary Table

| Term | Definition | Typical Value |
|------|------------|---------------|
| VWAP | Volume-weighted avg price | ~$3200 |
| Coverage | % swaps in range | 87.7% median |
| Stability | 0=trending, 1=choppy | 0.64 avg |
| Momentum | 30-min price change | ±0.5% |
| Participation | % time LPing | 52% |
| Catastrophic | Coverage < 30% | 15.5% rate |
