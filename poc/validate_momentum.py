"""
Proper train/test validation of momentum-based direction prediction.

Key questions:
1. Does momentum direction accuracy hold out-of-sample?
2. How much of the move can we capture by acting at detection time?
3. What's the optimal momentum threshold?
"""
import json
import random
import sqlite3
import statistics
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# Train/test split
TRAIN_START = 23_000_000
TRAIN_END = 23_500_000
TEST_START = 23_500_000
TEST_END = 24_000_000

LOOKBACK = 1000
LOOKAHEAD = 100


def get_swaps(conn, from_block, to_block):
    cursor = conn.execute(
        "SELECT block_number, price FROM swaps WHERE block_number BETWEEN ? AND ? ORDER BY block_number",
        (from_block, to_block))
    return [{"block": row[0], "price": row[1]} for row in cursor]


def is_catastrophic(lookback_swaps, lookahead_swaps):
    """Check if this period would be catastrophic for LP."""
    if len(lookback_swaps) < 100 or len(lookahead_swaps) < 10:
        return False, 0, 0

    # Simple range based on recent prices
    recent = [s["price"] for s in lookback_swaps[-200:]]
    median_price = statistics.median(recent)
    std = statistics.stdev(recent) if len(recent) > 1 else median_price * 0.01
    lower = median_price - std * 3
    upper = median_price + std * 3

    # Coverage
    swaps_in = sum(1 for s in lookahead_swaps if lower <= s["price"] <= upper)
    coverage = swaps_in / len(lookahead_swaps)

    # Price move
    start_price = lookback_swaps[-1]["price"]
    end_price = lookahead_swaps[-1]["price"]
    move_pct = (end_price - start_price) / start_price * 100

    return coverage < 0.3, coverage, move_pct


def get_momentum(swaps, window_swaps=150):
    """Get signed momentum over window (~30 min with 150 swaps)."""
    if len(swaps) < window_swaps:
        return 0
    return (swaps[-1]["price"] - swaps[-window_swaps]["price"]) / swaps[-window_swaps]["price"] * 100


def analyze_period(conn, start_block):
    """Analyze a single period for momentum/direction relationship."""
    lookback = get_swaps(conn, start_block - LOOKBACK, start_block - 1)
    lookahead = get_swaps(conn, start_block, start_block + LOOKAHEAD - 1)

    if len(lookback) < 200 or len(lookahead) < 10:
        return None

    is_cat, coverage, future_move = is_catastrophic(lookback, lookahead)

    if not is_cat:
        return None

    # Momentum at detection time (t=0)
    momentum = get_momentum(lookback)

    # How much of the move remains after detection?
    # If we act at t=0, we enter at lookback[-1] price
    # The move continues for LOOKAHEAD blocks
    entry_price = lookback[-1]["price"]
    exit_price = lookahead[-1]["price"]
    capturable_move = (exit_price - entry_price) / entry_price * 100

    # Direction
    direction = "UP" if future_move > 0 else "DOWN"
    predicted = "UP" if momentum > 0 else "DOWN"
    correct = direction == predicted

    return {
        "start_block": start_block,
        "momentum": momentum,
        "future_move": future_move,
        "capturable_move": capturable_move,
        "direction": direction,
        "predicted": predicted,
        "correct": correct,
    }


def collect_data(conn, start_block, end_block, n_samples, seed):
    """Collect catastrophic periods from a block range."""
    random.seed(seed)
    results = []
    min_start = start_block + LOOKBACK
    max_start = end_block - LOOKAHEAD

    attempts = 0
    while len(results) < n_samples and attempts < n_samples * 10:
        attempts += 1
        block = random.randint(min_start, max_start)
        r = analyze_period(conn, block)
        if r:
            results.append(r)

    return results


def find_optimal_threshold(train_data):
    """Find momentum threshold that maximizes accuracy on training data."""
    best_threshold = 0
    best_accuracy = 0

    # Test thresholds from 0 to 0.5%
    for threshold in [i * 0.02 for i in range(26)]:  # 0, 0.02, 0.04, ..., 0.5
        correct = 0
        total = 0

        for r in train_data:
            if abs(r["momentum"]) < threshold:
                continue  # Skip uncertain cases

            predicted = "UP" if r["momentum"] > 0 else "DOWN"
            if predicted == r["direction"]:
                correct += 1
            total += 1

        if total > 0:
            accuracy = correct / total
            # Penalize thresholds that skip too many (want >50% coverage)
            coverage = total / len(train_data)
            if coverage > 0.5 and accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold

    return best_threshold, best_accuracy


def evaluate(data, threshold):
    """Evaluate momentum strategy on data with given threshold."""
    results = {
        "total": len(data),
        "traded": 0,
        "correct": 0,
        "wrong": 0,
        "skipped": 0,
        "captured_moves": [],
        "missed_moves": [],
    }

    for r in data:
        if abs(r["momentum"]) < threshold:
            results["skipped"] += 1
            results["missed_moves"].append(r["capturable_move"])
            continue

        results["traded"] += 1
        predicted = "UP" if r["momentum"] > 0 else "DOWN"

        if predicted == r["direction"]:
            results["correct"] += 1
            # Captured move (positive if we got direction right)
            results["captured_moves"].append(abs(r["capturable_move"]))
        else:
            results["wrong"] += 1
            # Lost move (negative, we bet wrong)
            results["captured_moves"].append(-abs(r["capturable_move"]))

    results["accuracy"] = results["correct"] / results["traded"] if results["traded"] > 0 else 0
    results["coverage"] = results["traded"] / results["total"] if results["total"] > 0 else 0

    if results["captured_moves"]:
        results["avg_capture"] = statistics.mean(results["captured_moves"])
        results["total_capture"] = sum(results["captured_moves"])
    else:
        results["avg_capture"] = 0
        results["total_capture"] = 0

    return results


def main():
    conn = sqlite3.connect(str(DATA_DIR / "swaps.db"))

    print("=" * 70)
    print("MOMENTUM VALIDATION WITH TRAIN/TEST SPLIT")
    print("=" * 70)

    # Collect training data
    print(f"\nCollecting TRAINING data (blocks {TRAIN_START:,} - {TRAIN_END:,})...")
    train_data = collect_data(conn, TRAIN_START, TRAIN_END, 500, seed=42)
    print(f"  Found {len(train_data)} catastrophic periods")

    # Collect test data
    print(f"\nCollecting TEST data (blocks {TEST_START:,} - {TEST_END:,})...")
    test_data = collect_data(conn, TEST_START, TEST_END, 500, seed=123)
    print(f"  Found {len(test_data)} catastrophic periods")

    conn.close()

    # Find optimal threshold on training data
    print("\n" + "-" * 70)
    print("TRAINING: Finding optimal momentum threshold")
    print("-" * 70)

    optimal_threshold, train_accuracy = find_optimal_threshold(train_data)
    print(f"\nOptimal threshold: {optimal_threshold:.2f}%")
    print(f"Training accuracy at this threshold: {train_accuracy*100:.1f}%")

    # Evaluate on training data (in-sample)
    train_results = evaluate(train_data, optimal_threshold)
    print(f"\nTraining set performance:")
    print(f"  Accuracy: {train_results['accuracy']*100:.1f}%")
    print(f"  Coverage: {train_results['coverage']*100:.1f}% of catastrophes traded")
    print(f"  Avg capture: {train_results['avg_capture']:.3f}% per trade")

    # Evaluate on test data (out-of-sample)
    print("\n" + "-" * 70)
    print("TEST: Evaluating on held-out data")
    print("-" * 70)

    test_results = evaluate(test_data, optimal_threshold)
    print(f"\nTest set performance:")
    print(f"  Accuracy: {test_results['accuracy']*100:.1f}%")
    print(f"  Coverage: {test_results['coverage']*100:.1f}% of catastrophes traded")
    print(f"  Traded: {test_results['traded']} / {test_results['total']}")
    print(f"  Correct: {test_results['correct']}, Wrong: {test_results['wrong']}")

    print(f"\nMove capture analysis:")
    print(f"  Avg capture per trade: {test_results['avg_capture']:.3f}%")
    print(f"  Total captured: {test_results['total_capture']:.2f}%")

    # What if we didn't trade (just withdrew to 50/50)?
    all_moves = [abs(r["capturable_move"]) for r in test_data]
    avg_move = statistics.mean(all_moves)
    print(f"\n  Avg catastrophe move size: {avg_move:.3f}%")
    print(f"  If 50/50: capture ~0% (no directional exposure)")
    print(f"  With momentum: capture {test_results['avg_capture']:.3f}% avg")

    # Breakdown by direction
    print("\n" + "-" * 70)
    print("BREAKDOWN BY DIRECTION")
    print("-" * 70)

    up_trades = [r for r in test_data if r["direction"] == "UP" and abs(r["momentum"]) >= optimal_threshold]
    down_trades = [r for r in test_data if r["direction"] == "DOWN" and abs(r["momentum"]) >= optimal_threshold]

    up_correct = sum(1 for r in up_trades if r["correct"])
    down_correct = sum(1 for r in down_trades if r["correct"])

    print(f"\nUP moves:")
    print(f"  Total: {len(up_trades)}, Correct: {up_correct} ({up_correct/len(up_trades)*100:.1f}%)" if up_trades else "  No UP trades")

    print(f"\nDOWN moves:")
    print(f"  Total: {len(down_trades)}, Correct: {down_correct} ({down_correct/len(down_trades)*100:.1f}%)" if down_trades else "  No DOWN trades")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"""
Momentum threshold: {optimal_threshold:.2f}%
  - Learned from training set (blocks 23M-23.5M)
  - Applied to test set (blocks 23.5M-24M)

Out-of-sample results:
  - Direction accuracy: {test_results['accuracy']*100:.1f}%
  - Trade coverage: {test_results['coverage']*100:.1f}% of catastrophes
  - Avg capture: {test_results['avg_capture']:.3f}% per trade

Interpretation:
  - {'VALIDATED' if test_results['accuracy'] > 0.7 else 'NOT VALIDATED'}: Accuracy {'holds' if test_results['accuracy'] > 0.7 else 'does not hold'} out-of-sample
  - {'PROFITABLE' if test_results['avg_capture'] > 0 else 'NOT PROFITABLE'}: Avg capture is {'positive' if test_results['avg_capture'] > 0 else 'negative'}
""")

    # Save results
    output = {
        "optimal_threshold": optimal_threshold,
        "train": {
            "n_samples": len(train_data),
            "accuracy": train_results["accuracy"],
            "coverage": train_results["coverage"],
        },
        "test": {
            "n_samples": len(test_data),
            "accuracy": test_results["accuracy"],
            "coverage": test_results["coverage"],
            "avg_capture": test_results["avg_capture"],
            "total_capture": test_results["total_capture"],
        },
    }

    with open(DATA_DIR / "momentum_validation.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results saved to {DATA_DIR / 'momentum_validation.json'}")


if __name__ == "__main__":
    main()
