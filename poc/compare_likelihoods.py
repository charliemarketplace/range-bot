"""
Compare likelihood methods: Custom OHLC-range vs Range-aware KDE

Run: uv run python poc/compare_likelihoods.py
"""
import json
import numpy as np
from pathlib import Path

# Optional: scipy for KDE (falls back to manual implementation if not available)
try:
    from scipy.stats import gaussian_kde
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("scipy not installed - using manual KDE implementation")

DATA_DIR = Path(__file__).parent / "data"


def load_ohlc():
    """Load OHLC data from JSON."""
    with open(DATA_DIR / "ohlc.json") as f:
        return json.load(f)


# =============================================================================
# METHOD 1: Custom OHLC Range Voting (current implementation)
# =============================================================================

def likelihood_custom(candles, n=101):
    """
    Custom method: Each candle "votes" for all prices in its [low, high] range.
    Recent candles weighted more heavily (0.9 decay).
    """
    lo = min(c["l"] for c in candles) * 0.995
    hi = max(c["h"] for c in candles) * 1.005
    prices = np.linspace(lo, hi, n)
    probs = np.zeros(n)

    for idx, c in enumerate(candles):
        weight = 0.9 ** (len(candles) - 1 - idx)
        mask = (prices >= c["l"]) & (prices <= c["h"])
        probs[mask] += weight

    probs /= probs.sum()
    return prices, probs


# =============================================================================
# METHOD 2: Range-aware KDE (theoretically grounded)
# =============================================================================

def likelihood_kde(candles, n=101, samples_per_candle=20):
    """
    Range-aware KDE: Treat each candle as uniform distribution over [low, high],
    sample points from that range, then apply Gaussian KDE.
    """
    points = []
    weights = []

    for idx, c in enumerate(candles):
        w = 0.9 ** (len(candles) - 1 - idx)
        # Sample uniformly across the candle's range
        candle_points = np.linspace(c["l"], c["h"], samples_per_candle)
        points.extend(candle_points)
        weights.extend([w] * samples_per_candle)

    points = np.array(points)
    weights = np.array(weights)

    # Define evaluation grid
    lo = min(c["l"] for c in candles) * 0.995
    hi = max(c["h"] for c in candles) * 1.005
    prices = np.linspace(lo, hi, n)

    if HAS_SCIPY:
        # Use scipy's weighted KDE
        kde = gaussian_kde(points, weights=weights)
        probs = kde(prices)
    else:
        # Manual Gaussian KDE implementation
        probs = manual_kde(points, weights, prices)

    probs /= probs.sum()
    return prices, probs


def manual_kde(points, weights, eval_points, bandwidth=None):
    """
    Manual weighted Gaussian KDE (if scipy unavailable).
    Uses Silverman's rule for bandwidth selection.
    """
    n = len(points)
    if bandwidth is None:
        # Silverman's rule of thumb
        std = np.std(points)
        bandwidth = 1.06 * std * n ** (-1/5)

    probs = np.zeros(len(eval_points))
    for i, x in enumerate(eval_points):
        # Gaussian kernel centered at each point
        kernel_vals = np.exp(-0.5 * ((x - points) / bandwidth) ** 2)
        probs[i] = np.sum(weights * kernel_vals)

    return probs


# =============================================================================
# METHOD 3: Close-only KDE (baseline - what standard KDE would do)
# =============================================================================

def likelihood_close_only(candles, n=101):
    """
    Standard KDE using only closing prices (baseline for comparison).
    """
    closes = np.array([c["c"] for c in candles])
    weights = np.array([0.9 ** (len(candles) - 1 - i) for i in range(len(candles))])

    lo = min(c["l"] for c in candles) * 0.995
    hi = max(c["h"] for c in candles) * 1.005
    prices = np.linspace(lo, hi, n)

    if HAS_SCIPY:
        kde = gaussian_kde(closes, weights=weights)
        probs = kde(prices)
    else:
        probs = manual_kde(closes, weights, prices)

    probs /= probs.sum()
    return prices, probs


# =============================================================================
# ANALYSIS: Compare methods
# =============================================================================

def find_90_range(prices, probs):
    """Find tightest interval containing 90% probability mass."""
    best = None
    n = len(prices)

    for i in range(n):
        cumsum = 0.0
        for j in range(i, n):
            cumsum += probs[j]
            if cumsum >= 0.90:
                width = prices[j] - prices[i]
                if best is None or width < best["width"]:
                    best = {
                        "lower": prices[i],
                        "upper": prices[j],
                        "width": width,
                        "coverage": cumsum
                    }
                break

    return best


def entropy(probs):
    """Shannon entropy - higher = more spread out."""
    p = probs[probs > 0]
    return -np.sum(p * np.log(p))


def compare_methods(candles):
    """Run all methods and compare results."""
    print(f"\nAnalyzing {len(candles)} candles...")
    print("=" * 70)

    results = {}

    # Run each method
    methods = [
        ("Custom (OHLC range voting)", likelihood_custom),
        ("Range-aware KDE", likelihood_kde),
        ("Close-only KDE (baseline)", likelihood_close_only),
    ]

    for name, func in methods:
        prices, probs = func(candles)
        range_90 = find_90_range(prices, probs)
        ent = entropy(probs)

        results[name] = {
            "prices": prices,
            "probs": probs,
            "range": range_90,
            "entropy": ent
        }

        print(f"\n{name}:")
        print(f"  90% Range: ${range_90['lower']:.2f} - ${range_90['upper']:.2f}")
        print(f"  Width:     ${range_90['width']:.2f}")
        print(f"  Entropy:   {ent:.3f} (higher = more spread)")

    return results


def print_ascii_comparison(results):
    """Print ASCII art comparison of distributions."""
    print("\n" + "=" * 70)
    print("DISTRIBUTION COMPARISON (ASCII)")
    print("=" * 70)

    # Get common price range
    all_prices = results["Custom (OHLC range voting)"]["prices"]
    min_p, max_p = all_prices[0], all_prices[-1]

    width = 60
    height = 10

    for name, data in results.items():
        prices = data["prices"]
        probs = data["probs"]
        range_90 = data["range"]

        # Normalize probs to height
        max_prob = probs.max()
        scaled = (probs / max_prob * height).astype(int)

        print(f"\n{name}:")
        print(f"  ${min_p:.0f}" + " " * (width - 12) + f"${max_p:.0f}")

        # Draw distribution (using ASCII-safe characters)
        for row in range(height, 0, -1):
            line = "  "
            for i in range(0, len(scaled), len(scaled) // width):
                if scaled[i] >= row:
                    # Check if in 90% range
                    if range_90["lower"] <= prices[i] <= range_90["upper"]:
                        line += "#"
                    else:
                        line += "."
                else:
                    line += " "
            print(line)

        print("  " + "-" * width)
        print(f"  [. = outside 90%]  [# = inside 90% range: ${range_90['lower']:.2f}-${range_90['upper']:.2f}]")


def save_comparison_data(results, candles):
    """Save comparison data for visualization in viewer."""
    output = {
        "candles_used": len(candles),
        "methods": {}
    }

    for name, data in results.items():
        output["methods"][name] = {
            "prices": data["prices"].tolist(),
            "probs": data["probs"].tolist(),
            "range_90": data["range"],
            "entropy": data["entropy"]
        }

    with open(DATA_DIR / "likelihood_comparison.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nComparison data saved to {DATA_DIR / 'likelihood_comparison.json'}")


def main():
    print("=" * 70)
    print("LIKELIHOOD METHOD COMPARISON")
    print("Custom OHLC-range voting vs Range-aware KDE vs Close-only KDE")
    print("=" * 70)

    # Load data
    ohlc = load_ohlc()
    recent = ohlc[-10:]  # Last 10 candles (same as used in viewer)

    print(f"\nUsing last 10 candles:")
    print(f"  Time range: {recent[0]['time_str']} - {recent[-1]['time_str']} UTC")
    print(f"  Price range: ${min(c['l'] for c in recent):.2f} - ${max(c['h'] for c in recent):.2f}")

    # Compare methods
    results = compare_methods(recent)

    # ASCII visualization
    print_ascii_comparison(results)

    # Analysis
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    custom_range = results["Custom (OHLC range voting)"]["range"]["width"]
    kde_range = results["Range-aware KDE"]["range"]["width"]
    close_range = results["Close-only KDE (baseline)"]["range"]["width"]

    print(f"""
90% Range Width Comparison:
  Custom OHLC:    ${custom_range:.2f}
  Range-aware KDE: ${kde_range:.2f}  ({(kde_range/custom_range - 1)*100:+.1f}% vs custom)
  Close-only KDE:  ${close_range:.2f}  ({(close_range/custom_range - 1)*100:+.1f}% vs custom)

Interpretation:
  - Narrower range = more confident (but riskier for LP)
  - Wider range = more conservative (safer for LP, lower fee density)
  - KDE produces smoother distributions (better for gradient-based optimization)
  - Custom method is "blockier" but directly interpretable
""")

    # Save for viewer
    save_comparison_data(results, recent)

    print("\nRecommendation:")
    if abs(kde_range - custom_range) / custom_range < 0.05:
        print("  -> Methods agree within 5% - either is fine for this data")
    elif kde_range < custom_range:
        print("  -> KDE is tighter - custom method may be more conservative (safer)")
    else:
        print("  -> Custom is tighter - KDE may be more conservative (safer)")


if __name__ == "__main__":
    main()
