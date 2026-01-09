"""
Random-sample backtest: evaluate Bayesian range predictions on randomly sampled block intervals.

Instead of fixed stride, samples N random start blocks with replacement to avoid
periodicity bias and get more robust statistics.
"""
import argparse
import json
import math
import random
import sqlite3
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB = DATA_DIR / "swaps.db"

CANDLE_BLOCKS = 50  # ~10 min per candle


# -----------------------------------------------------------------------------
# Stability Detection
# -----------------------------------------------------------------------------

def stability_score(candles: list) -> float:
    """0 = trending hard, 1 = perfectly stable."""
    if len(candles) < 2:
        return 1.0

    vwaps = [c["vwap"] for c in candles]
    net_move = abs(vwaps[-1] - vwaps[0])
    total_path = sum(abs(vwaps[i + 1] - vwaps[i]) for i in range(len(vwaps) - 1))

    if total_path == 0:
        return 1.0

    return 1 - (net_move / total_path)


# -----------------------------------------------------------------------------
# Bayesian Model
# -----------------------------------------------------------------------------

def laplace_dist(center: float, scale: float, n: int = 101) -> tuple:
    half = scale * 4
    prices = [center - half + (2 * half) * i / (n - 1) for i in range(n)]
    probs = [math.exp(-abs(p - center) / scale) / (2 * scale) for p in prices]
    total = sum(probs)
    return prices, [p / total for p in probs]


def likelihood_kde(candles: list, n: int = 101) -> tuple:
    """KDE likelihood (better performer from prior tests)."""
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


def bayesian_update(prior_prices, prior_probs, lik_prices, lik_probs) -> tuple:
    def interp(prices, probs, target):
        for i in range(len(prices) - 1):
            if prices[i] <= target <= prices[i + 1]:
                t = (target - prices[i]) / (prices[i + 1] - prices[i])
                return (1 - t) * probs[i] + t * probs[i + 1]
        return probs[0] if target < prices[0] else probs[-1]

    post = [p * interp(lik_prices, lik_probs, pr) for pr, p in zip(prior_prices, prior_probs)]
    total = sum(post)
    return prior_prices, [p / total for p in post] if total > 0 else prior_probs


def optimal_range(prices, probs, coverage: float = 0.9) -> dict:
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
        return {"lower": prices[0], "upper": prices[-1]}

    return {"lower": prices[best[0]], "upper": prices[best[1]]}


# -----------------------------------------------------------------------------
# OHLC Aggregation
# -----------------------------------------------------------------------------

def swaps_to_candles(swaps: list) -> list:
    if not swaps:
        return []

    candles = []
    min_block = swaps[0]["block"]
    max_block = swaps[-1]["block"]
    period_start = (min_block // CANDLE_BLOCKS) * CANDLE_BLOCKS

    while period_start <= max_block:
        period_end = period_start + CANDLE_BLOCKS
        ps = [s for s in swaps if period_start <= s["block"] < period_end]

        if ps:
            prices = [s["price"] for s in ps]
            vols = [abs(s["amount1"]) / 1e6 for s in ps]
            total_vol = sum(vols)
            vwap = sum(p * v for p, v in zip(prices, vols)) / total_vol if total_vol > 0 else prices[-1]

            candles.append({
                "block_start": period_start,
                "o": prices[0],
                "h": max(prices),
                "l": min(prices),
                "c": prices[-1],
                "vwap": vwap,
            })

        period_start = period_end

    return candles


# -----------------------------------------------------------------------------
# Database
# -----------------------------------------------------------------------------

def get_swaps(conn: sqlite3.Connection, from_block: int, to_block: int) -> list:
    cursor = conn.execute("""
        SELECT block_number, price, amount1
        FROM swaps
        WHERE block_number BETWEEN ? AND ?
        ORDER BY block_number
    """, (from_block, to_block))
    return [{"block": row[0], "price": row[1], "amount1": int(row[2])} for row in cursor]


# -----------------------------------------------------------------------------
# Single Evaluation
# -----------------------------------------------------------------------------

@dataclass
class EvalResult:
    start_block: int
    stability: float
    range_lower: float
    range_upper: float
    range_width: float
    lookahead_swaps: int
    swaps_in_range: int
    coverage: float


def evaluate_single(conn: sqlite3.Connection, start_block: int, lookback: int, lookahead: int) -> EvalResult | None:
    """Evaluate a single random block interval."""

    # Get lookback swaps
    lookback_swaps = get_swaps(conn, start_block - lookback, start_block - 1)
    if len(lookback_swaps) < 10:
        return None

    # Build candles
    candles = swaps_to_candles(lookback_swaps)
    if len(candles) < 3:
        return None

    recent_candles = candles[-10:]

    # Stability
    stab = stability_score(recent_candles)

    # Prior from VWAP
    vwaps = [c["vwap"] for c in recent_candles]
    median_vwap = statistics.median(vwaps)
    std = statistics.stdev(vwaps) if len(vwaps) > 1 else median_vwap * 0.01
    prior_prices, prior_probs = laplace_dist(median_vwap, std * 2)

    # Likelihood (KDE only for speed)
    lik_prices, lik_probs = likelihood_kde(recent_candles)

    # Posterior
    post_prices, post_probs = bayesian_update(prior_prices, prior_probs, lik_prices, lik_probs)

    # 90% range
    rng = optimal_range(post_prices, post_probs, 0.9)
    lower, upper = rng["lower"], rng["upper"]

    # Evaluate lookahead
    lookahead_swaps = get_swaps(conn, start_block, start_block + lookahead - 1)
    if not lookahead_swaps:
        return None

    swaps_in = sum(1 for s in lookahead_swaps if lower <= s["price"] <= upper)

    return EvalResult(
        start_block=start_block,
        stability=stab,
        range_lower=lower,
        range_upper=upper,
        range_width=upper - lower,
        lookahead_swaps=len(lookahead_swaps),
        swaps_in_range=swaps_in,
        coverage=swaps_in / len(lookahead_swaps)
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def run_random_backtest(
    db_path: Path,
    start_block: int,
    end_block: int,
    lookback: int,
    lookahead: int,
    n_samples: int,
    seed: int | None = None
) -> dict:
    """Run random-sample backtest."""

    if seed is not None:
        random.seed(seed)

    conn = sqlite3.connect(str(db_path))

    # Valid range for sampling
    min_start = start_block + lookback
    max_start = end_block - lookahead

    print(f"Random Backtest: {n_samples:,} samples")
    print(f"  Block range: {start_block:,} to {end_block:,}")
    print(f"  Valid eval range: {min_start:,} to {max_start:,}")
    print(f"  Lookback: {lookback}, Lookahead: {lookahead}")
    print()

    results = []
    skipped = 0

    for i in range(n_samples):
        if (i + 1) % 10000 == 0 or i == 0:
            pct = 100 * (i + 1) / n_samples
            print(f"  [{i+1:,}/{n_samples:,}] ({pct:.1f}%) - {len(results):,} valid, {skipped:,} skipped")

        # Random start block
        eval_block = random.randint(min_start, max_start)

        result = evaluate_single(conn, eval_block, lookback, lookahead)
        if result:
            results.append(result)
        else:
            skipped += 1

    conn.close()

    print(f"\nCompleted: {len(results):,} valid evaluations, {skipped:,} skipped")

    # Aggregate stats
    if not results:
        return {"error": "No valid results"}

    coverages = [r.coverage for r in results]
    stabilities = [r.stability for r in results]
    widths = [r.range_width for r in results]

    # Split by stability
    stable = [r for r in results if r.stability >= 0.5]
    trending = [r for r in results if r.stability < 0.5]

    summary = {
        "n_samples": n_samples,
        "n_valid": len(results),
        "n_skipped": skipped,
        "lookback": lookback,
        "lookahead": lookahead,

        "overall": {
            "coverage_mean": statistics.mean(coverages),
            "coverage_std": statistics.stdev(coverages) if len(coverages) > 1 else 0,
            "coverage_median": statistics.median(coverages),
            "coverage_p10": sorted(coverages)[len(coverages) // 10],
            "coverage_p90": sorted(coverages)[9 * len(coverages) // 10],
            "range_width_mean": statistics.mean(widths),
            "stability_mean": statistics.mean(stabilities),
        },

        "stable": {
            "n": len(stable),
            "pct": len(stable) / len(results) * 100,
            "coverage_mean": statistics.mean([r.coverage for r in stable]) if stable else 0,
            "coverage_std": statistics.stdev([r.coverage for r in stable]) if len(stable) > 1 else 0,
        },

        "trending": {
            "n": len(trending),
            "pct": len(trending) / len(results) * 100,
            "coverage_mean": statistics.mean([r.coverage for r in trending]) if trending else 0,
            "coverage_std": statistics.stdev([r.coverage for r in trending]) if len(trending) > 1 else 0,
        },
    }

    return {
        "summary": summary,
        "sample_results": [asdict(r) for r in results[:1000]],  # First 1000 for inspection
    }


def print_summary(results: dict):
    s = results["summary"]

    print("\n" + "=" * 60)
    print("RANDOM BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nSamples: {s['n_valid']:,} valid / {s['n_samples']:,} attempted")
    print(f"Lookback: {s['lookback']} blocks, Lookahead: {s['lookahead']} blocks")

    o = s["overall"]
    print(f"\nOVERALL (90% CI targeting 90% coverage):")
    print(f"  Coverage: {o['coverage_mean']*100:.1f}% +/- {o['coverage_std']*100:.1f}%")
    print(f"  Median: {o['coverage_median']*100:.1f}%")
    print(f"  P10-P90: [{o['coverage_p10']*100:.1f}%, {o['coverage_p90']*100:.1f}%]")
    print(f"  Avg range width: ${o['range_width_mean']:.2f}")
    print(f"  Avg stability: {o['stability_mean']:.3f}")

    st = s["stable"]
    print(f"\nSTABLE PERIODS ({st['n']:,} = {st['pct']:.1f}%):")
    print(f"  Coverage: {st['coverage_mean']*100:.1f}% +/- {st['coverage_std']*100:.1f}%")

    tr = s["trending"]
    print(f"\nTRENDING PERIODS ({tr['n']:,} = {tr['pct']:.1f}%):")
    print(f"  Coverage: {tr['coverage_mean']*100:.1f}% +/- {tr['coverage_std']*100:.1f}%")

    delta = st['coverage_mean'] - tr['coverage_mean']
    print(f"\nStable vs Trending delta: {delta*100:+.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Random-sample backtest")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--start-block", type=int, default=23000000)
    parser.add_argument("--end-block", type=int, default=24000000)
    parser.add_argument("--lookback", type=int, default=1000)
    parser.add_argument("--lookahead", type=int, default=100)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "backtest_random.json")

    args = parser.parse_args()

    results = run_random_backtest(
        db_path=args.db,
        start_block=args.start_block,
        end_block=args.end_block,
        lookback=args.lookback,
        lookahead=args.lookahead,
        n_samples=args.samples,
        seed=args.seed
    )

    print_summary(results)

    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
