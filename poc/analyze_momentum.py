"""
Check if signed price momentum predicts catastrophe direction.
"""
import random
import sqlite3
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CANDLE_BLOCKS = 50
LOOKBACK = 1000
LOOKAHEAD = 100


def get_swaps(conn, from_block, to_block):
    cursor = conn.execute(
        "SELECT block_number, price FROM swaps WHERE block_number BETWEEN ? AND ? ORDER BY block_number",
        (from_block, to_block))
    return [{"block": row[0], "price": row[1]} for row in cursor]


def analyze_momentum(conn, start_block):
    """Check momentum signals."""
    lookback_swaps = get_swaps(conn, start_block - LOOKBACK, start_block - 1)
    if len(lookback_swaps) < 100:
        return None

    lookahead_swaps = get_swaps(conn, start_block, start_block + LOOKAHEAD - 1)
    if len(lookahead_swaps) < 10:
        return None

    # Calculate lookback momentum (signed)
    lookback_prices = [s["price"] for s in lookback_swaps]

    # Multiple momentum windows
    momentum_5min = (lookback_prices[-1] - lookback_prices[-25]) / lookback_prices[-25] * 100 if len(lookback_prices) > 25 else 0
    momentum_30min = (lookback_prices[-1] - lookback_prices[-150]) / lookback_prices[-150] * 100 if len(lookback_prices) > 150 else 0
    momentum_1hr = (lookback_prices[-1] - lookback_prices[-300]) / lookback_prices[-300] * 100 if len(lookback_prices) > 300 else 0

    # Simple range for coverage calc
    recent_prices = lookback_prices[-200:]  # Last ~40 min
    median_price = statistics.median(recent_prices)
    std = statistics.stdev(recent_prices)
    lower = median_price - std * 3
    upper = median_price + std * 3

    # Coverage
    swaps_in = sum(1 for s in lookahead_swaps if lower <= s["price"] <= upper)
    coverage = swaps_in / len(lookahead_swaps)

    if coverage >= 0.3:  # Not catastrophic
        return None

    # Direction of catastrophe
    start_price = lookback_prices[-1]
    end_price = lookahead_swaps[-1]["price"]
    future_move = (end_price - start_price) / start_price * 100
    direction = "UP" if future_move > 0 else "DOWN"

    return {
        "momentum_5min": momentum_5min,
        "momentum_30min": momentum_30min,
        "momentum_1hr": momentum_1hr,
        "future_move": future_move,
        "direction": direction,
    }


def main():
    conn = sqlite3.connect(str(DATA_DIR / "swaps.db"))
    random.seed(42)

    print("Analyzing momentum signals in catastrophes...\n")

    catastrophes = []
    attempts = 0
    min_start = 23_000_000 + LOOKBACK
    max_start = 24_000_000 - LOOKAHEAD

    while len(catastrophes) < 300 and attempts < 100000:
        attempts += 1
        block = random.randint(min_start, max_start)
        result = analyze_momentum(conn, block)
        if result:
            catastrophes.append(result)

    conn.close()

    print(f"Found {len(catastrophes)} catastrophes\n")

    up_moves = [c for c in catastrophes if c["direction"] == "UP"]
    down_moves = [c for c in catastrophes if c["direction"] == "DOWN"]

    print("=" * 70)
    print("MOMENTUM VS FUTURE DIRECTION")
    print("=" * 70)

    print(f"\n{'Momentum Window':<20} {'UP mean':<12} {'DOWN mean':<12} {'Same Sign?':<15}")
    print("-" * 60)

    for mom_key, label in [("momentum_5min", "5 min"), ("momentum_30min", "30 min"), ("momentum_1hr", "1 hr")]:
        up_mom = [c[mom_key] for c in up_moves]
        down_mom = [c[mom_key] for c in down_moves]

        up_mean = statistics.mean(up_mom) if up_mom else 0
        down_mean = statistics.mean(down_mom) if down_mom else 0

        # Does positive momentum predict UP, negative predict DOWN?
        same_sign = "YES" if (up_mean > 0 and down_mean < 0) else "NO"

        print(f"{label:<20} {up_mean:>+10.3f}% {down_mean:>+10.3f}%   {same_sign:<15}")

    # More detailed: momentum sign accuracy
    print("\n" + "=" * 70)
    print("MOMENTUM SIGN AS PREDICTOR")
    print("=" * 70)

    for mom_key, label in [("momentum_5min", "5 min"), ("momentum_30min", "30 min"), ("momentum_1hr", "1 hr")]:
        # If momentum > 0, predict UP. If momentum < 0, predict DOWN.
        correct = 0
        total = 0
        for c in catastrophes:
            mom = c[mom_key]
            if abs(mom) < 0.01:  # Skip near-zero
                continue
            predicted = "UP" if mom > 0 else "DOWN"
            if predicted == c["direction"]:
                correct += 1
            total += 1

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"{label} momentum: {accuracy:.1f}% accuracy ({correct}/{total})")

    # Check if STRONG momentum is more predictive
    print("\n" + "=" * 70)
    print("STRONG MOMENTUM (|mom| > 0.2%) AS PREDICTOR")
    print("=" * 70)

    for mom_key, label in [("momentum_5min", "5 min"), ("momentum_30min", "30 min"), ("momentum_1hr", "1 hr")]:
        correct = 0
        total = 0
        for c in catastrophes:
            mom = c[mom_key]
            if abs(mom) < 0.2:  # Only strong momentum
                continue
            predicted = "UP" if mom > 0 else "DOWN"
            if predicted == c["direction"]:
                correct += 1
            total += 1

        accuracy = correct / total * 100 if total > 0 else 0
        print(f"{label} strong momentum: {accuracy:.1f}% accuracy ({correct}/{total})")

    # Correlation
    print("\n" + "=" * 70)
    print("CORRELATION: LOOKBACK MOMENTUM vs FUTURE MOVE")
    print("=" * 70)

    for mom_key, label in [("momentum_5min", "5 min"), ("momentum_30min", "30 min"), ("momentum_1hr", "1 hr")]:
        moms = [c[mom_key] for c in catastrophes]
        futures = [c["future_move"] for c in catastrophes]

        # Simple correlation
        mean_mom = statistics.mean(moms)
        mean_fut = statistics.mean(futures)

        cov = sum((m - mean_mom) * (f - mean_fut) for m, f in zip(moms, futures)) / len(moms)
        std_mom = statistics.stdev(moms)
        std_fut = statistics.stdev(futures)
        corr = cov / (std_mom * std_fut) if std_mom > 0 and std_fut > 0 else 0

        print(f"{label}: correlation = {corr:.3f}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print("\nMomentum-based direction prediction:")


if __name__ == "__main__":
    main()
