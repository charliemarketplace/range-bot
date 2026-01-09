"""
Tune alert thresholds to find optimal risk/reward tradeoff.

Tests multiple threshold multipliers to find Pareto frontier between:
- Coverage improvement (want high)
- Participation rate (want high - LP more often)
"""
import json
import math
import random
import sqlite3
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CANDLE_BLOCKS = 50

TRAIN_START = 23_000_000
TRAIN_END = 23_500_000
TEST_START = 23_500_000
TEST_END = 24_000_000

LOOKBACK = 1000
LOOKAHEAD = 100


# -----------------------------------------------------------------------------
# Core functions (condensed from backtest_with_alerts.py)
# -----------------------------------------------------------------------------

def get_swaps(conn, from_block, to_block):
    cursor = conn.execute(
        "SELECT block_number, price, amount1 FROM swaps WHERE block_number BETWEEN ? AND ? ORDER BY block_number",
        (from_block, to_block))
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
        points.append((c["h"] + c["l"]) / 2)
        weights.append(0.9 ** (len(candles) - 1 - idx))
    if not points:
        return prices, [1 / n] * n
    std = statistics.stdev(points) if len(points) > 1 else (hi - lo) * 0.1
    bw = 1.06 * std * (len(points) ** -0.2)
    probs = [sum(w * math.exp(-0.5 * ((p - pt) / bw) ** 2) for pt, w in zip(points, weights)) for p in prices]
    total = sum(probs)
    return prices, [p / total for p in probs] if total > 0 else [1 / n] * n


def bayesian_update(prior_p, prior_prob, lik_p, lik_prob):
    def interp(prices, probs, target):
        for i in range(len(prices) - 1):
            if prices[i] <= target <= prices[i + 1]:
                t = (target - prices[i]) / (prices[i + 1] - prices[i])
                return (1 - t) * probs[i] + t * probs[i + 1]
        return probs[0] if target < prices[0] else probs[-1]
    post = [p * interp(lik_p, lik_prob, pr) for pr, p in zip(prior_p, prior_prob)]
    total = sum(post)
    return prior_p, [p / total for p in post] if total > 0 else prior_prob


def optimal_range(prices, probs, coverage=0.9):
    best = None
    for i in range(len(prices)):
        cumsum = 0.0
        for j in range(i, len(prices)):
            cumsum += probs[j]
            if cumsum >= coverage:
                width = prices[j] - prices[i]
                if best is None or width < best[2]:
                    best = (i, j, width)
                break
    if not best:
        return prices[0], prices[-1]
    return prices[best[0]], prices[best[1]]


def extract_features(candles):
    if len(candles) < 6:
        return None
    recent = candles[-10:] if len(candles) >= 10 else candles
    vwaps = [c["vwap"] for c in recent]
    vols = [c["vol"] for c in recent]
    ranges = [c["h"] - c["l"] for c in recent]

    mid = len(recent) // 2
    def calc_stab(v):
        if len(v) < 2: return 1.0
        net = abs(v[-1] - v[0])
        path = sum(abs(v[i+1] - v[i]) for i in range(len(v)-1))
        return 1 - (net / path) if path > 0 else 1.0

    stability_trend = calc_stab(vwaps[mid:]) - calc_stab(vwaps[:mid])
    early_r, late_r = ranges[:len(ranges)//2], ranges[len(ranges)//2:]
    range_expansion = statistics.mean(late_r) / statistics.mean(early_r) if statistics.mean(early_r) > 0 else 1
    price_velocity = abs(vwaps[-1] - vwaps[-2]) / vwaps[-2] * 100 if len(vwaps) >= 2 else 0
    vol_spike = vols[-1] / statistics.mean(vols[:-1]) if len(vols) >= 2 and statistics.mean(vols[:-1]) > 0 else 1

    return {"stability_trend": stability_trend, "range_expansion": range_expansion,
            "price_velocity": price_velocity, "vol_spike": vol_spike}


def evaluate_single(conn, start_block):
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
    return {"coverage": swaps_in / len(lookahead_swaps), "features": features}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    conn = sqlite3.connect(str(DATA_DIR / "swaps.db"))
    random.seed(42)

    # Collect training data
    print("Collecting training data...")
    train_results = []
    min_start, max_start = TRAIN_START + LOOKBACK, TRAIN_END - LOOKAHEAD
    for i in range(30000):
        if (i + 1) % 10000 == 0:
            print(f"  Train: {i+1}/30000")
        r = evaluate_single(conn, random.randint(min_start, max_start))
        if r:
            train_results.append(r)

    # Learn base thresholds from training (midpoint between hit/miss means)
    hits = [r for r in train_results if r["coverage"] >= 0.8]
    misses = [r for r in train_results if r["coverage"] < 0.3]

    base_thresholds = {}
    for feat in ["stability_trend", "range_expansion", "price_velocity", "vol_spike"]:
        hit_mean = statistics.mean([r["features"][feat] for r in hits])
        miss_mean = statistics.mean([r["features"][feat] for r in misses])
        base_thresholds[feat] = (hit_mean + miss_mean) / 2

    print(f"\nBase thresholds (midpoint):")
    for k, v in base_thresholds.items():
        print(f"  {k}: {v:.4f}")

    # Collect test data
    print("\nCollecting test data...")
    random.seed(123)
    test_results = []
    min_start, max_start = TEST_START + LOOKBACK, TEST_END - LOOKAHEAD
    for i in range(30000):
        if (i + 1) % 10000 == 0:
            print(f"  Test: {i+1}/30000")
        r = evaluate_single(conn, random.randint(min_start, max_start))
        if r:
            test_results.append(r)

    conn.close()

    baseline_cov = statistics.mean([r["coverage"] for r in test_results])
    baseline_median = statistics.median([r["coverage"] for r in test_results])
    baseline_catastrophic = sum(1 for r in test_results if r["coverage"] < 0.3) / len(test_results)

    print(f"\nBaseline (no alerts): {baseline_cov*100:.1f}% mean, {baseline_median*100:.1f}% median, {baseline_catastrophic*100:.1f}% catastrophic")

    # Test different threshold sensitivities
    # Multiplier > 1 = more lenient (alert less), < 1 = more strict (alert more)
    print("\n" + "=" * 90)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("=" * 90)
    print(f"{'Sensitivity':<12} {'Participation':>14} {'Mean Cov':>10} {'Median':>10} {'Catastrophic':>14} {'Improvement':>12}")
    print("-" * 90)

    results_table = []

    for sensitivity in [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0]:
        # Adjust thresholds based on sensitivity
        # For stability_trend (alert if below): more negative threshold = more lenient
        # For others (alert if above): higher threshold = more lenient
        adjusted = {
            "stability_trend": base_thresholds["stability_trend"] * sensitivity,
            "range_expansion": base_thresholds["range_expansion"] + (sensitivity - 1) * 0.1,
            "price_velocity": base_thresholds["price_velocity"] * sensitivity,
            "vol_spike": base_thresholds["vol_spike"] * sensitivity,
        }

        # Apply thresholds
        not_alerted = []
        for r in test_results:
            f = r["features"]
            alert = (
                f["stability_trend"] < adjusted["stability_trend"] or
                f["range_expansion"] > adjusted["range_expansion"] or
                f["price_velocity"] > adjusted["price_velocity"] or
                f["vol_spike"] > adjusted["vol_spike"]
            )
            if not alert:
                not_alerted.append(r)

        if not not_alerted:
            continue

        participation = len(not_alerted) / len(test_results) * 100
        mean_cov = statistics.mean([r["coverage"] for r in not_alerted]) * 100
        median_cov = statistics.median([r["coverage"] for r in not_alerted]) * 100
        catastrophic = sum(1 for r in not_alerted if r["coverage"] < 0.3) / len(not_alerted) * 100
        improvement = mean_cov - baseline_cov * 100

        print(f"{sensitivity:<12.2f} {participation:>13.1f}% {mean_cov:>9.1f}% {median_cov:>9.1f}% {catastrophic:>13.1f}% {improvement:>+11.1f}%")

        results_table.append({
            "sensitivity": sensitivity,
            "thresholds": adjusted,
            "participation_pct": participation,
            "mean_coverage_pct": mean_cov,
            "median_coverage_pct": median_cov,
            "catastrophic_pct": catastrophic,
            "improvement_pct": improvement,
        })

    # Find best tradeoff (maximize participation * improvement)
    print("\n" + "-" * 90)
    print("RECOMMENDATIONS:")

    # Best for max improvement
    best_improvement = max(results_table, key=lambda x: x["improvement_pct"])
    print(f"\n  Max Improvement: sensitivity={best_improvement['sensitivity']}")
    print(f"    +{best_improvement['improvement_pct']:.1f}% coverage, {best_improvement['participation_pct']:.1f}% participation")

    # Best balanced (high participation + good improvement)
    # Score = participation * (improvement + 5) to favor participation when improvements are similar
    for r in results_table:
        r["score"] = r["participation_pct"] * (r["improvement_pct"] + 5)

    best_balanced = max(results_table, key=lambda x: x["score"])
    print(f"\n  Balanced (recommended): sensitivity={best_balanced['sensitivity']}")
    print(f"    +{best_balanced['improvement_pct']:.1f}% coverage, {best_balanced['participation_pct']:.1f}% participation")
    print(f"    Thresholds:")
    for k, v in best_balanced["thresholds"].items():
        print(f"      {k}: {v:.4f}")

    # Best for high participation (>50%)
    high_participation = [r for r in results_table if r["participation_pct"] >= 50]
    if high_participation:
        best_hp = max(high_participation, key=lambda x: x["improvement_pct"])
        print(f"\n  High Participation (>=50%): sensitivity={best_hp['sensitivity']}")
        print(f"    +{best_hp['improvement_pct']:.1f}% coverage, {best_hp['participation_pct']:.1f}% participation")

    # Save results
    output = {
        "baseline": {
            "mean_coverage": baseline_cov,
            "median_coverage": baseline_median,
            "catastrophic_rate": baseline_catastrophic,
        },
        "base_thresholds": base_thresholds,
        "sensitivity_analysis": results_table,
        "recommended": best_balanced,
    }

    with open(DATA_DIR / "threshold_tuning.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {DATA_DIR / 'threshold_tuning.json'}")


if __name__ == "__main__":
    main()
