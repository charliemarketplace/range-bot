"""
Backtest Bayesian range predictions on historical swap data.

Evaluates how well 90% credible intervals predict future prices across
multiple time horizons (100, 500, 1000 blocks).
"""
import argparse
import json
import math
import sqlite3
import statistics
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"
DEFAULT_DB = DATA_DIR / "swaps.db"

# Block time ~12s, so 1000 blocks ~3.3 hours
CANDLE_BLOCKS = 50  # ~10 min per candle (50 blocks * 12s)


@dataclass
class WindowResult:
    window_blocks: int
    total_swaps: int
    swaps_in_range: int
    swap_coverage: float
    blocks_with_swaps: int
    blocks_in_range: int
    time_coverage: float


@dataclass
class EvalPoint:
    eval_block: int
    method: str
    range_lower: float
    range_upper: float
    range_width: float
    median_vwap: float
    results: dict = field(default_factory=dict)  # window -> WindowResult


# -----------------------------------------------------------------------------
# Bayesian Model (adapted from simple_poc.py)
# -----------------------------------------------------------------------------

def laplace_dist(center: float, scale: float, n: int = 101) -> tuple:
    """Build Laplace prior distribution."""
    half = scale * 4
    prices = [center - half + (2 * half) * i / (n - 1) for i in range(n)]
    probs = [math.exp(-abs(p - center) / scale) / (2 * scale) for p in prices]
    total = sum(probs)
    return prices, [p / total for p in probs]


def likelihood_custom(candles: list, n: int = 101) -> tuple:
    """Custom OHLC likelihood: candle range voting with decay."""
    if not candles:
        return [0], [1]

    lo = min(c["l"] for c in candles) * 0.995
    hi = max(c["h"] for c in candles) * 1.005
    prices = [lo + (hi - lo) * i / (n - 1) for i in range(n)]
    probs = [0.0] * n

    for idx, c in enumerate(candles):
        w = 0.9 ** (len(candles) - 1 - idx)  # Decay older candles
        for i, p in enumerate(prices):
            if c["l"] <= p <= c["h"]:
                probs[i] += w

    total = sum(probs)
    return prices, [p / total for p in probs] if total > 0 else [1 / n] * n


def likelihood_kde(candles: list, n: int = 101) -> tuple:
    """KDE likelihood: Gaussian kernel smoothing over candle midpoints."""
    if not candles:
        return [0], [1]

    lo = min(c["l"] for c in candles) * 0.995
    hi = max(c["h"] for c in candles) * 1.005
    prices = [lo + (hi - lo) * i / (n - 1) for i in range(n)]

    # Collect candle midpoints with decay weights
    points = []
    weights = []
    for idx, c in enumerate(candles):
        w = 0.9 ** (len(candles) - 1 - idx)
        mid = (c["h"] + c["l"]) / 2
        points.append(mid)
        weights.append(w)

    if not points:
        return prices, [1 / n] * n

    # Silverman bandwidth
    std = statistics.stdev(points) if len(points) > 1 else (hi - lo) * 0.1
    bw = 1.06 * std * (len(points) ** -0.2)

    probs = [0.0] * n
    for i, p in enumerate(prices):
        for pt, w in zip(points, weights):
            probs[i] += w * math.exp(-0.5 * ((p - pt) / bw) ** 2)

    total = sum(probs)
    return prices, [p / total for p in probs] if total > 0 else [1 / n] * n


def bayesian_update(prior_prices, prior_probs, lik_prices, lik_probs) -> tuple:
    """Posterior = prior * likelihood."""
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
    """Find tightest range with target coverage."""
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
        return {"lower": prices[0], "upper": prices[-1], "coverage": 1.0}

    return {
        "lower": prices[best[0]],
        "upper": prices[best[1]],
        "coverage": best[3]
    }


# -----------------------------------------------------------------------------
# OHLC Aggregation
# -----------------------------------------------------------------------------

def swaps_to_candles(swaps: list, blocks_per_candle: int = CANDLE_BLOCKS) -> list:
    """Aggregate swaps to OHLC candles by block periods."""
    if not swaps:
        return []

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
            vwap = sum(p * v for p, v in zip(prices, vols)) / total_vol if total_vol > 0 else prices[-1]

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


# -----------------------------------------------------------------------------
# Database Queries
# -----------------------------------------------------------------------------

def get_swaps(conn: sqlite3.Connection, from_block: int, to_block: int) -> list:
    """Fetch swaps in block range."""
    cursor = conn.execute("""
        SELECT block_number, price, amount1
        FROM swaps
        WHERE block_number BETWEEN ? AND ?
        ORDER BY block_number
    """, (from_block, to_block))

    return [{"block": row[0], "price": row[1], "amount1": int(row[2])} for row in cursor]


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_range(swaps: list, lower: float, upper: float, window_blocks: int) -> WindowResult:
    """Evaluate how many swaps/blocks fall within range."""
    if not swaps:
        return WindowResult(
            window_blocks=window_blocks,
            total_swaps=0,
            swaps_in_range=0,
            swap_coverage=0.0,
            blocks_with_swaps=0,
            blocks_in_range=0,
            time_coverage=0.0
        )

    swaps_in = sum(1 for s in swaps if lower <= s["price"] <= upper)

    # Group by block
    blocks = {}
    for s in swaps:
        b = s["block"]
        if b not in blocks:
            blocks[b] = {"in_range": False, "has_swap": True}
        if lower <= s["price"] <= upper:
            blocks[b]["in_range"] = True

    blocks_in = sum(1 for b in blocks.values() if b["in_range"])

    return WindowResult(
        window_blocks=window_blocks,
        total_swaps=len(swaps),
        swaps_in_range=swaps_in,
        swap_coverage=swaps_in / len(swaps) if swaps else 0,
        blocks_with_swaps=len(blocks),
        blocks_in_range=blocks_in,
        time_coverage=blocks_in / len(blocks) if blocks else 0
    )


def compute_range_for_coverage(prices, probs, coverage: float) -> tuple:
    """Compute optimal range for given coverage level."""
    result = optimal_range(prices, probs, coverage)
    return result["lower"], result["upper"]


# -----------------------------------------------------------------------------
# Main Backtest
# -----------------------------------------------------------------------------

def run_backtest(
    db_path: Path,
    start_block: int,
    end_block: int,
    stride: int,
    lookback: int,
    windows: list[int],
    coverages: list[float],
    methods: list[str]
) -> dict:
    """Run full backtest."""

    conn = sqlite3.connect(str(db_path))

    # Results storage
    eval_points = []

    # Calibration tracking: method -> window -> target_coverage -> [actual_coverages]
    calibration = {m: {w: {c: [] for c in coverages} for w in windows} for m in methods}

    # Coverage stats: method -> window -> [swap_coverages]
    swap_coverages = {m: {w: [] for w in windows} for m in methods}
    time_coverages = {m: {w: [] for w in windows} for m in methods}

    # Count eval points
    total_points = (end_block - start_block - lookback) // stride

    print(f"Running backtest: {total_points} evaluation points")
    print(f"  Blocks: {start_block:,} to {end_block:,}")
    print(f"  Stride: {stride}, Lookback: {lookback}")
    print(f"  Windows: {windows}")
    print(f"  Coverages: {coverages}")
    print(f"  Methods: {methods}")
    print()

    point_num = 0
    for eval_block in range(start_block + lookback, end_block - max(windows), stride):
        point_num += 1

        # Progress every 100 points
        if point_num % 100 == 0 or point_num == 1:
            pct = 100 * point_num / total_points
            print(f"  [{point_num}/{total_points}] Block {eval_block:,} ({pct:.1f}%)")

        # 1. Get lookback swaps
        lookback_swaps = get_swaps(conn, eval_block - lookback, eval_block - 1)
        if len(lookback_swaps) < 10:
            continue  # Skip if too few swaps

        # 2. Build OHLC candles
        candles = swaps_to_candles(lookback_swaps)
        if len(candles) < 3:
            continue  # Need enough candles

        # 3. Build prior from VWAP
        vwaps = [c["vwap"] for c in candles[-10:]]  # Last 10 candles
        median_vwap = statistics.median(vwaps)
        std = statistics.stdev(vwaps) if len(vwaps) > 1 else median_vwap * 0.01
        prior_prices, prior_probs = laplace_dist(median_vwap, std * 2)

        # 4. Get lookahead swaps for each window
        lookahead_swaps = {}
        for w in windows:
            lookahead_swaps[w] = get_swaps(conn, eval_block, eval_block + w - 1)

        # 5. Evaluate each method
        for method in methods:
            # Build likelihood
            if method == "custom":
                lik_prices, lik_probs = likelihood_custom(candles[-10:])
            else:  # kde
                lik_prices, lik_probs = likelihood_kde(candles[-10:])

            # Posterior
            post_prices, post_probs = bayesian_update(prior_prices, prior_probs, lik_prices, lik_probs)

            # For each coverage level and window, evaluate
            for target_cov in coverages:
                lower, upper = compute_range_for_coverage(post_prices, post_probs, target_cov)

                for w in windows:
                    result = evaluate_range(lookahead_swaps[w], lower, upper, w)

                    # Track calibration
                    if result.total_swaps > 0:
                        calibration[method][w][target_cov].append(result.swap_coverage)

                    # Track main metrics for 90% coverage
                    if target_cov == 0.9 and result.total_swaps > 0:
                        swap_coverages[method][w].append(result.swap_coverage)
                        time_coverages[method][w].append(result.time_coverage)

                        # Store eval point (only for 90%)
                        if w == windows[0]:  # Avoid duplicates
                            ep = EvalPoint(
                                eval_block=eval_block,
                                method=method,
                                range_lower=lower,
                                range_upper=upper,
                                range_width=upper - lower,
                                median_vwap=median_vwap,
                            )
                            eval_points.append(ep)

    conn.close()

    # Compute summary statistics
    summary = {"methods": {}}

    for method in methods:
        summary["methods"][method] = {"windows": {}}
        for w in windows:
            sc = swap_coverages[method][w]
            tc = time_coverages[method][w]

            summary["methods"][method]["windows"][str(w)] = {
                "swap_coverage_mean": statistics.mean(sc) if sc else 0,
                "swap_coverage_std": statistics.stdev(sc) if len(sc) > 1 else 0,
                "time_coverage_mean": statistics.mean(tc) if tc else 0,
                "time_coverage_std": statistics.stdev(tc) if len(tc) > 1 else 0,
                "n_points": len(sc)
            }

    # Calibration curves
    calibration_summary = {}
    for method in methods:
        calibration_summary[method] = {}
        for w in windows:
            calibration_summary[method][f"{w}_blocks"] = {}
            for target_cov in coverages:
                actual = calibration[method][w][target_cov]
                if actual:
                    calibration_summary[method][f"{w}_blocks"][str(int(target_cov * 100))] = (
                        statistics.mean(actual) * 100
                    )

    return {
        "summary": summary,
        "calibration": calibration_summary,
        "eval_points": [asdict(ep) for ep in eval_points[:100]],  # First 100 for inspection
        "params": {
            "start_block": start_block,
            "end_block": end_block,
            "stride": stride,
            "lookback": lookback,
            "windows": windows,
            "coverages": coverages,
            "methods": methods
        }
    }


def print_summary(results: dict):
    """Print formatted summary table."""
    print("\n" + "=" * 75)
    print("BACKTEST RESULTS")
    print("=" * 75)

    summary = results["summary"]

    # Header
    print(f"\n{'Method':<10} {'Window':<12} {'Swap Coverage':<18} {'Time Coverage':<18} {'Cal(90%)':<10}")
    print("-" * 75)

    for method, data in summary["methods"].items():
        for window, stats in data["windows"].items():
            sc_mean = stats["swap_coverage_mean"] * 100
            sc_std = stats["swap_coverage_std"] * 100
            tc_mean = stats["time_coverage_mean"] * 100
            tc_std = stats["time_coverage_std"] * 100

            # Get calibration for 90%
            cal_90 = results["calibration"].get(method, {}).get(f"{window}_blocks", {}).get("90", 0)

            print(f"{method:<10} {window + ' blk':<12} "
                  f"{sc_mean:5.1f}% +/- {sc_std:4.1f}%    "
                  f"{tc_mean:5.1f}% +/- {tc_std:4.1f}%    "
                  f"{cal_90:5.1f}%")

    print("\n" + "-" * 75)
    print("CALIBRATION CURVES")
    print("-" * 75)

    for method in results["calibration"]:
        print(f"\n{method.upper()}:")
        for window_key, cals in results["calibration"][method].items():
            cal_str = ", ".join(f"{k}%: {v:.1f}%" for k, v in sorted(cals.items(), key=lambda x: int(x[0])))
            print(f"  {window_key}: {cal_str}")


def main():
    parser = argparse.ArgumentParser(description="Backtest Bayesian range predictions")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="Path to swaps database")
    parser.add_argument("--start-block", type=int, default=23000000, help="Start block")
    parser.add_argument("--end-block", type=int, default=24000000, help="End block")
    parser.add_argument("--stride", type=int, default=1000, help="Blocks between eval points")
    parser.add_argument("--lookback", type=int, default=1000, help="Lookback blocks for model")
    parser.add_argument("--windows", type=str, default="100,500,1000", help="Lookahead windows")
    parser.add_argument("--coverages", type=str, default="0.5,0.7,0.9,0.95", help="Coverage levels")
    parser.add_argument("--methods", type=str, default="custom,kde", help="Likelihood methods")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "backtest_results.json")

    args = parser.parse_args()

    windows = [int(w) for w in args.windows.split(",")]
    coverages = [float(c) for c in args.coverages.split(",")]
    methods = args.methods.split(",")

    print(f"Backtest: {args.db}")
    print(f"Output: {args.output}")
    print()

    results = run_backtest(
        db_path=args.db,
        start_block=args.start_block,
        end_block=args.end_block,
        stride=args.stride,
        lookback=args.lookback,
        windows=windows,
        coverages=coverages,
        methods=methods
    )

    # Print summary
    print_summary(results)

    # Save results
    args.output.parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Save calibration separately for plotting
    cal_path = args.output.parent / "calibration.json"
    with open(cal_path, "w") as f:
        json.dump(results["calibration"], f, indent=2)
    print(f"Calibration saved to {cal_path}")


if __name__ == "__main__":
    main()
