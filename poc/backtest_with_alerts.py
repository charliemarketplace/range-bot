"""
Backtest with early warning system using proper train/test split.

Train: blocks 23M - 23.5M (find optimal alert thresholds)
Test: blocks 23.5M - 24M (evaluate with those thresholds)
"""
import json
import math
import random
import sqlite3
import statistics
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CANDLE_BLOCKS = 50

# Train/Test split
TRAIN_START = 23_000_000
TRAIN_END = 23_500_000
TEST_START = 23_500_000
TEST_END = 24_000_000

LOOKBACK = 1000
LOOKAHEAD = 100


# -----------------------------------------------------------------------------
# Core functions (from backtest_random.py)
# -----------------------------------------------------------------------------

def get_swaps(conn, from_block, to_block):
    cursor = conn.execute("""
        SELECT block_number, price, amount1
        FROM swaps WHERE block_number BETWEEN ? AND ?
        ORDER BY block_number
    """, (from_block, to_block))
    return [{"block": row[0], "price": row[1], "amount1": int(row[2])} for row in cursor]


def swaps_to_candles(swaps):
    if not swaps:
        return []
    candles = []
    min_block, max_block = swaps[0]["block"], swaps[-1]["block"]
    period_start = (min_block // CANDLE_BLOCKS) * CANDLE_BLOCKS

    while period_start <= max_block:
        period_end = period_start + CANDLE_BLOCKS
        ps = [s for s in swaps if period_start <= s["block"] < period_end]
        if ps:
            prices = [s["price"] for s in ps]
            vols = [abs(s["amount1"]) / 1e6 for s in ps]
            total_vol = sum(vols)
            vwap = sum(p * v for p, v in zip(prices, vols)) / total_vol if total_vol > 0 else prices[-1]
            candles.append({"h": max(prices), "l": min(prices), "vwap": vwap, "vol": total_vol})
        period_start = period_end
    return candles


def laplace_dist(center, scale, n=101):
    half = scale * 4
    prices = [center - half + (2 * half) * i / (n - 1) for i in range(n)]
    probs = [math.exp(-abs(p - center) / scale) / (2 * scale) for p in prices]
    total = sum(probs)
    return prices, [p / total for p in probs]


def likelihood_kde(candles, n=101):
    if not candles:
        return [0], [1]
    lo = min(c["l"] for c in candles) * 0.995
    hi = max(c["h"] for c in candles) * 1.005
    prices = [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    points, weights = [], []
    for idx, c in enumerate(candles):
        w = 0.9 ** (len(candles) - 1 - idx)
        points.append((c["h"] + c["l"]) / 2)
        weights.append(w)

    if not points:
        return prices, [1 / n] * n

    std = statistics.stdev(points) if len(points) > 1 else (hi - lo) * 0.1
    bw = 1.06 * std * (len(points) ** -0.2)

    probs = [0.0] * n
    for i, p in enumerate(prices):
        for pt, w in zip(points, weights):
            probs[i] += w * math.exp(-0.5 * ((p - pt) / bw) ** 2)

    total = sum(probs)
    return prices, [p / total for p in probs] if total > 0 else [1 / n] * n


def bayesian_update(prior_prices, prior_probs, lik_prices, lik_probs):
    def interp(prices, probs, target):
        for i in range(len(prices) - 1):
            if prices[i] <= target <= prices[i + 1]:
                t = (target - prices[i]) / (prices[i + 1] - prices[i])
                return (1 - t) * probs[i] + t * probs[i + 1]
        return probs[0] if target < prices[0] else probs[-1]

    post = [p * interp(lik_prices, lik_probs, pr) for pr, p in zip(prior_prices, prior_probs)]
    total = sum(post)
    return prior_prices, [p / total for p in post] if total > 0 else prior_probs


def optimal_range(prices, probs, coverage=0.9):
    best = None
    for i in range(len(prices)):
        cumsum = 0.0
        for j in range(i, len(prices)):
            cumsum += probs[j]
            if cumsum >= coverage:
                width = prices[j] - prices[i]
                if best is None or width < best[2]:
                    best = (i, j, width, cumsum)
                break
    if not best:
        return prices[0], prices[-1]
    return prices[best[0]], prices[best[1]]


# -----------------------------------------------------------------------------
# Early Warning Features
# -----------------------------------------------------------------------------

def extract_features(candles):
    """Extract early warning features from lookback candles."""
    if len(candles) < 6:
        return None

    recent = candles[-10:] if len(candles) >= 10 else candles
    vwaps = [c["vwap"] for c in recent]
    vols = [c["vol"] for c in recent]
    ranges = [c["h"] - c["l"] for c in recent]

    # Stability trend (key signal from analysis)
    mid = len(recent) // 2
    early_vwaps = vwaps[:mid]
    late_vwaps = vwaps[mid:]

    def calc_stability(v):
        if len(v) < 2:
            return 1.0
        net = abs(v[-1] - v[0])
        path = sum(abs(v[i + 1] - v[i]) for i in range(len(v) - 1))
        return 1 - (net / path) if path > 0 else 1.0

    early_stab = calc_stability(early_vwaps)
    late_stab = calc_stability(late_vwaps)
    stability_trend = late_stab - early_stab  # Negative = getting less stable

    # Range expansion
    early_ranges = ranges[:len(ranges) // 2]
    late_ranges = ranges[len(ranges) // 2:]
    range_expansion = (
        statistics.mean(late_ranges) / statistics.mean(early_ranges)
        if statistics.mean(early_ranges) > 0 else 1
    )

    # Price velocity (% change in last candle)
    price_velocity = abs(vwaps[-1] - vwaps[-2]) / vwaps[-2] * 100 if len(vwaps) >= 2 else 0

    # Volume spike
    vol_spike = vols[-1] / statistics.mean(vols[:-1]) if len(vols) >= 2 and statistics.mean(vols[:-1]) > 0 else 1

    return {
        "stability_trend": stability_trend,
        "range_expansion": range_expansion,
        "price_velocity": price_velocity,
        "vol_spike": vol_spike,
    }


# -----------------------------------------------------------------------------
# Single Evaluation
# -----------------------------------------------------------------------------

@dataclass
class EvalResult:
    start_block: int
    coverage: float
    features: dict


def evaluate_single(conn, start_block):
    """Evaluate a single block interval, return features + coverage."""
    lookback_swaps = get_swaps(conn, start_block - LOOKBACK, start_block - 1)
    if len(lookback_swaps) < 10:
        return None

    candles = swaps_to_candles(lookback_swaps)
    if len(candles) < 6:
        return None

    features = extract_features(candles)
    if not features:
        return None

    recent = candles[-10:]
    vwaps = [c["vwap"] for c in recent]
    median_vwap = statistics.median(vwaps)
    std = statistics.stdev(vwaps) if len(vwaps) > 1 else median_vwap * 0.01

    prior_p, prior_prob = laplace_dist(median_vwap, std * 2)
    lik_p, lik_prob = likelihood_kde(recent)
    post_p, post_prob = bayesian_update(prior_p, prior_prob, lik_p, lik_prob)
    lower, upper = optimal_range(post_p, post_prob, 0.9)

    lookahead_swaps = get_swaps(conn, start_block, start_block + LOOKAHEAD - 1)
    if not lookahead_swaps:
        return None

    swaps_in = sum(1 for s in lookahead_swaps if lower <= s["price"] <= upper)
    coverage = swaps_in / len(lookahead_swaps)

    return EvalResult(start_block=start_block, coverage=coverage, features=features)


# -----------------------------------------------------------------------------
# Train: Find Optimal Thresholds
# -----------------------------------------------------------------------------

def run_train(conn, n_samples=20000, seed=42):
    """Run on train set, find optimal thresholds."""
    random.seed(seed)

    min_start = TRAIN_START + LOOKBACK
    max_start = TRAIN_END - LOOKAHEAD

    print(f"TRAINING on blocks {TRAIN_START:,} - {TRAIN_END:,}")
    print(f"  Sampling {n_samples:,} random intervals...")

    results = []
    for i in range(n_samples):
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1:,}/{n_samples:,}]")
        block = random.randint(min_start, max_start)
        r = evaluate_single(conn, block)
        if r:
            results.append(r)

    print(f"  Got {len(results):,} valid samples")

    # Find thresholds that separate hits from misses
    hits = [r for r in results if r.coverage >= 0.8]
    misses = [r for r in results if r.coverage < 0.3]

    print(f"\n  Hits (>=80%): {len(hits)}, Misses (<30%): {len(misses)}")

    # For each feature, find threshold that maximizes separation
    thresholds = {}

    for feature in ["stability_trend", "range_expansion", "price_velocity", "vol_spike"]:
        hit_vals = [r.features[feature] for r in hits]
        miss_vals = [r.features[feature] for r in misses]

        hit_mean = statistics.mean(hit_vals)
        miss_mean = statistics.mean(miss_vals)

        # Threshold = midpoint between means
        # For stability_trend: alert if BELOW threshold (negative = bad)
        # For others: alert if ABOVE threshold (higher = bad)
        threshold = (hit_mean + miss_mean) / 2
        thresholds[feature] = threshold

        print(f"  {feature}: hit={hit_mean:.4f}, miss={miss_mean:.4f}, threshold={threshold:.4f}")

    return thresholds, results


# -----------------------------------------------------------------------------
# Test: Evaluate with Learned Thresholds
# -----------------------------------------------------------------------------

def should_alert(features, thresholds):
    """Check if any early warning threshold is breached."""
    # stability_trend: alert if below (becoming less stable)
    if features["stability_trend"] < thresholds["stability_trend"]:
        return True, "stability_trend"

    # Others: alert if above
    if features["range_expansion"] > thresholds["range_expansion"]:
        return True, "range_expansion"

    if features["price_velocity"] > thresholds["price_velocity"]:
        return True, "price_velocity"

    if features["vol_spike"] > thresholds["vol_spike"]:
        return True, "vol_spike"

    return False, None


def run_test(conn, thresholds, n_samples=20000, seed=123):
    """Run on test set with learned thresholds."""
    random.seed(seed)

    min_start = TEST_START + LOOKBACK
    max_start = TEST_END - LOOKAHEAD

    print(f"\nTESTING on blocks {TEST_START:,} - {TEST_END:,}")
    print(f"  Using thresholds learned from training")
    print(f"  Sampling {n_samples:,} random intervals...")

    results = []
    for i in range(n_samples):
        if (i + 1) % 5000 == 0:
            print(f"  [{i+1:,}/{n_samples:,}]")
        block = random.randint(min_start, max_start)
        r = evaluate_single(conn, block)
        if r:
            results.append(r)

    print(f"  Got {len(results):,} valid samples")

    # Evaluate: compare "LP always" vs "LP with alerts"
    coverages_all = [r.coverage for r in results]

    alerted = []
    not_alerted = []
    alert_reasons = {"stability_trend": 0, "range_expansion": 0, "price_velocity": 0, "vol_spike": 0}

    for r in results:
        alert, reason = should_alert(r.features, thresholds)
        if alert:
            alerted.append(r)
            alert_reasons[reason] += 1
        else:
            not_alerted.append(r)

    coverages_filtered = [r.coverage for r in not_alerted]

    return {
        "all": coverages_all,
        "filtered": coverages_filtered,
        "alerted": alerted,
        "not_alerted": not_alerted,
        "alert_reasons": alert_reasons,
    }


def print_results(test_results):
    all_cov = test_results["all"]
    filt_cov = test_results["filtered"]
    alerted = test_results["alerted"]
    reasons = test_results["alert_reasons"]

    print("\n" + "=" * 70)
    print("TEST RESULTS (out-of-sample)")
    print("=" * 70)

    print(f"\nSAMPLES:")
    print(f"  Total: {len(all_cov):,}")
    print(f"  Alerted (would skip): {len(alerted):,} ({len(alerted)/len(all_cov)*100:.1f}%)")
    print(f"  Not alerted (would LP): {len(filt_cov):,} ({len(filt_cov)/len(all_cov)*100:.1f}%)")

    print(f"\nALERT TRIGGERS:")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count:,}")

    print(f"\nCOVERAGE (90% CI):")
    print(f"  {'Strategy':<25} {'Mean':>10} {'Median':>10} {'Std':>10}")
    print(f"  {'-'*55}")

    all_mean = statistics.mean(all_cov) * 100
    all_median = statistics.median(all_cov) * 100
    all_std = statistics.stdev(all_cov) * 100

    filt_mean = statistics.mean(filt_cov) * 100 if filt_cov else 0
    filt_median = statistics.median(filt_cov) * 100 if filt_cov else 0
    filt_std = statistics.stdev(filt_cov) * 100 if len(filt_cov) > 1 else 0

    print(f"  {'LP Always':<25} {all_mean:>9.1f}% {all_median:>9.1f}% {all_std:>9.1f}%")
    print(f"  {'LP With Alerts':<25} {filt_mean:>9.1f}% {filt_median:>9.1f}% {filt_std:>9.1f}%")

    improvement = filt_mean - all_mean
    print(f"\n  Improvement: {improvement:+.1f}% mean coverage")

    # Check coverage of skipped periods
    if alerted:
        alert_cov = [r.coverage for r in alerted]
        alert_mean = statistics.mean(alert_cov) * 100
        print(f"\n  Avg coverage of SKIPPED periods: {alert_mean:.1f}%")
        print(f"  (These would have been bad LP periods)")

    # Breakdown by coverage bucket
    print(f"\nCOVERAGE DISTRIBUTION:")
    buckets = [(0, 0.3, "Catastrophic (<30%)"), (0.3, 0.6, "Poor (30-60%)"),
               (0.6, 0.8, "Okay (60-80%)"), (0.8, 1.01, "Good (>=80%)")]

    print(f"  {'Bucket':<25} {'All':>12} {'Filtered':>12} {'Change':>12}")
    print(f"  {'-'*55}")

    for lo, hi, label in buckets:
        all_n = sum(1 for c in all_cov if lo <= c < hi)
        filt_n = sum(1 for c in filt_cov if lo <= c < hi)
        all_pct = all_n / len(all_cov) * 100
        filt_pct = filt_n / len(filt_cov) * 100 if filt_cov else 0
        change = filt_pct - all_pct
        print(f"  {label:<25} {all_pct:>11.1f}% {filt_pct:>11.1f}% {change:>+11.1f}%")


def main():
    conn = sqlite3.connect(str(DATA_DIR / "swaps.db"))

    # Train
    thresholds, train_results = run_train(conn, n_samples=30000)

    # Test
    test_results = run_test(conn, thresholds, n_samples=30000)

    conn.close()

    # Print results
    print_results(test_results)

    # Save
    output = {
        "thresholds": thresholds,
        "test_summary": {
            "n_total": len(test_results["all"]),
            "n_filtered": len(test_results["filtered"]),
            "n_alerted": len(test_results["alerted"]),
            "coverage_all_mean": statistics.mean(test_results["all"]),
            "coverage_filtered_mean": statistics.mean(test_results["filtered"]) if test_results["filtered"] else 0,
            "alert_reasons": test_results["alert_reasons"],
        }
    }

    with open(DATA_DIR / "backtest_alerts.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {DATA_DIR / 'backtest_alerts.json'}")


if __name__ == "__main__":
    main()
