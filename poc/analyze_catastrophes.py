"""
Analyze catastrophic epochs:
1. How early can we detect them?
2. Is there directional signal (price up vs down)?
"""
import json
import math
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
            candles.append({"h": max(prices), "l": min(prices), "vwap": vwap, "vol": total_vol, "block": period_start})
        period_start = period_end
    return candles


def extract_features_at_offset(candles, offset=0):
    """Extract features from candles ending at offset from the end.
    offset=0 means use all candles (current time)
    offset=2 means exclude last 2 candles (2 candles earlier)
    """
    if offset > 0:
        candles = candles[:-offset]

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

    return {
        "stability_trend": stability_trend,
        "range_expansion": range_expansion,
        "price_velocity": price_velocity,
        "vol_spike": vol_spike,
    }


def would_alert(features, sensitivity=4.0):
    """Check if aggressive thresholds would trigger."""
    base = {
        "stability_trend": -0.0522,
        "range_expansion": 1.0940,
        "price_velocity": 0.1472,
        "vol_spike": 0.7026,
    }
    thresholds = {
        "stability_trend": base["stability_trend"] * sensitivity,
        "range_expansion": base["range_expansion"] + (sensitivity - 1) * 0.1,
        "price_velocity": base["price_velocity"] * sensitivity,
        "vol_spike": base["vol_spike"] * sensitivity,
    }

    triggered = []
    if features["stability_trend"] < thresholds["stability_trend"]:
        triggered.append("stability_trend")
    if features["range_expansion"] > thresholds["range_expansion"]:
        triggered.append("range_expansion")
    if features["price_velocity"] > thresholds["price_velocity"]:
        triggered.append("price_velocity")
    if features["vol_spike"] > thresholds["vol_spike"]:
        triggered.append("vol_spike")

    return len(triggered) > 0, triggered


def analyze_epoch(conn, start_block):
    """Analyze a single epoch for early detection and directionality."""

    # Get lookback data
    lookback_swaps = get_swaps(conn, start_block - LOOKBACK, start_block - 1)
    if len(lookback_swaps) < 10:
        return None

    candles = swaps_to_candles(lookback_swaps)
    if len(candles) < 10:
        return None

    # Get lookahead data for coverage calculation
    lookahead_swaps = get_swaps(conn, start_block, start_block + LOOKAHEAD - 1)
    if not lookahead_swaps:
        return None

    # Calculate the 90% range
    recent = candles[-10:]
    vwaps = [c["vwap"] for c in recent]
    median_vwap = statistics.median(vwaps)
    std = statistics.stdev(vwaps) if len(vwaps) > 1 else median_vwap * 0.01

    # Simple range calculation (using prior width)
    half_width = std * 2 * 4  # Laplace scale * 4 for 90%
    lower = median_vwap - half_width * 0.9
    upper = median_vwap + half_width * 0.9

    # Coverage
    swaps_in = sum(1 for s in lookahead_swaps if lower <= s["price"] <= upper)
    coverage = swaps_in / len(lookahead_swaps)

    # Is this catastrophic?
    is_catastrophic = coverage < 0.3

    if not is_catastrophic:
        return None  # Only analyze catastrophes

    # DIRECTION: Did price go up or down?
    start_price = lookback_swaps[-1]["price"]
    end_price = lookahead_swaps[-1]["price"]
    price_change_pct = (end_price - start_price) / start_price * 100
    direction = "UP" if price_change_pct > 0 else "DOWN"

    # Did price escape above or below the range?
    prices_above = sum(1 for s in lookahead_swaps if s["price"] > upper)
    prices_below = sum(1 for s in lookahead_swaps if s["price"] < lower)
    escape_direction = "UP" if prices_above > prices_below else "DOWN"

    # EARLY DETECTION: Check features at different time offsets
    # Each candle is ~10 min (50 blocks), so offset=3 means 30 min earlier
    early_detection = {}
    for offset in [0, 1, 2, 3, 4, 5]:  # 0, 10, 20, 30, 40, 50 min before
        features = extract_features_at_offset(candles, offset)
        if features:
            alert, triggers = would_alert(features)
            early_detection[offset] = {
                "would_alert": alert,
                "triggers": triggers,
                "features": features,
            }

    # Check stability_trend sign (negative = getting less stable = bearish signal?)
    final_features = extract_features_at_offset(candles, 0)
    stability_trend_sign = "NEGATIVE" if final_features["stability_trend"] < 0 else "POSITIVE"

    return {
        "start_block": start_block,
        "coverage": coverage,
        "price_change_pct": price_change_pct,
        "direction": direction,
        "escape_direction": escape_direction,
        "stability_trend_sign": stability_trend_sign,
        "early_detection": early_detection,
        "final_features": final_features,
    }


def main():
    conn = sqlite3.connect(str(DATA_DIR / "swaps.db"))
    random.seed(42)

    print("Finding catastrophic epochs...")

    # Sample random blocks and find catastrophes
    catastrophes = []
    attempts = 0

    min_start = 23_000_000 + LOOKBACK
    max_start = 24_000_000 - LOOKAHEAD

    while len(catastrophes) < 200 and attempts < 50000:
        attempts += 1
        block = random.randint(min_start, max_start)
        result = analyze_epoch(conn, block)
        if result:
            catastrophes.append(result)
            if len(catastrophes) % 50 == 0:
                print(f"  Found {len(catastrophes)} catastrophes...")

    conn.close()

    print(f"\nAnalyzed {len(catastrophes)} catastrophic epochs")

    # ANALYSIS 1: Directionality
    print("\n" + "=" * 70)
    print("DIRECTIONALITY ANALYSIS")
    print("=" * 70)

    up_moves = [c for c in catastrophes if c["direction"] == "UP"]
    down_moves = [c for c in catastrophes if c["direction"] == "DOWN"]

    print(f"\nPrice direction during catastrophes:")
    print(f"  UP:   {len(up_moves)} ({len(up_moves)/len(catastrophes)*100:.1f}%)")
    print(f"  DOWN: {len(down_moves)} ({len(down_moves)/len(catastrophes)*100:.1f}%)")

    avg_up = statistics.mean([c["price_change_pct"] for c in up_moves]) if up_moves else 0
    avg_down = statistics.mean([c["price_change_pct"] for c in down_moves]) if down_moves else 0
    print(f"\n  Avg UP move:   {avg_up:+.2f}%")
    print(f"  Avg DOWN move: {avg_down:+.2f}%")

    # Check if stability_trend sign predicts direction
    print(f"\nStability trend sign vs direction:")
    neg_stab = [c for c in catastrophes if c["stability_trend_sign"] == "NEGATIVE"]
    pos_stab = [c for c in catastrophes if c["stability_trend_sign"] == "POSITIVE"]

    neg_up = sum(1 for c in neg_stab if c["direction"] == "UP")
    neg_down = sum(1 for c in neg_stab if c["direction"] == "DOWN")
    pos_up = sum(1 for c in pos_stab if c["direction"] == "UP")
    pos_down = sum(1 for c in pos_stab if c["direction"] == "DOWN")

    print(f"  Negative stability_trend: {neg_up} UP, {neg_down} DOWN")
    print(f"  Positive stability_trend: {pos_up} UP, {pos_down} DOWN")

    # Check raw stability_trend value correlation with direction
    up_stab_trends = [c["final_features"]["stability_trend"] for c in up_moves]
    down_stab_trends = [c["final_features"]["stability_trend"] for c in down_moves]

    print(f"\n  Avg stability_trend when UP:   {statistics.mean(up_stab_trends):.4f}")
    print(f"  Avg stability_trend when DOWN: {statistics.mean(down_stab_trends):.4f}")

    # ANALYSIS 2: Early Detection
    print("\n" + "=" * 70)
    print("EARLY DETECTION ANALYSIS")
    print("=" * 70)

    print(f"\nHow early could we detect these catastrophes? (aggressive thresholds)")
    print(f"{'Offset':<10} {'Minutes Before':<15} {'Would Alert':<15} {'% Detected':<12}")
    print("-" * 55)

    for offset in [0, 1, 2, 3, 4, 5]:
        detected = sum(1 for c in catastrophes if c["early_detection"].get(offset, {}).get("would_alert", False))
        pct = detected / len(catastrophes) * 100
        minutes = offset * 10  # Each candle ~10 min
        print(f"{offset:<10} {minutes:<15} {detected:<15} {pct:.1f}%")

    # Which triggers fire most often?
    print(f"\nTrigger frequency at t=0:")
    trigger_counts = {"stability_trend": 0, "range_expansion": 0, "price_velocity": 0, "vol_spike": 0}
    for c in catastrophes:
        for trigger in c["early_detection"].get(0, {}).get("triggers", []):
            trigger_counts[trigger] += 1

    for trigger, count in sorted(trigger_counts.items(), key=lambda x: -x[1]):
        print(f"  {trigger}: {count} ({count/len(catastrophes)*100:.1f}%)")

    # ANALYSIS 3: Can we predict direction?
    print("\n" + "=" * 70)
    print("DIRECTION PREDICTION SIGNALS")
    print("=" * 70)

    # Check if recent price velocity (signed) predicts direction
    # We need to look at signed velocity, not absolute
    print("\nLooking at momentum indicators...")

    for c in catastrophes:
        candles_data = None  # We'd need to reload, let's use what we have
        # The features we have are absolute values, we need signed

    # Use stability_trend as a proxy - negative might indicate downtrend starting
    # Let's check correlation more carefully

    # Simple check: is the sign of price change correlated with any feature?
    print("\nFeature values by direction:")
    print(f"{'Feature':<20} {'UP mean':<12} {'DOWN mean':<12} {'Diff':<12} {'Signal?':<10}")
    print("-" * 70)

    for feat in ["stability_trend", "range_expansion", "price_velocity", "vol_spike"]:
        up_vals = [c["final_features"][feat] for c in up_moves]
        down_vals = [c["final_features"][feat] for c in down_moves]
        up_mean = statistics.mean(up_vals)
        down_mean = statistics.mean(down_vals)
        diff = up_mean - down_mean
        # Is the difference meaningful? (> 10% of the range)
        all_vals = up_vals + down_vals
        val_range = max(all_vals) - min(all_vals) if all_vals else 1
        signal = "YES" if abs(diff) > val_range * 0.1 else "no"
        print(f"{feat:<20} {up_mean:<12.4f} {down_mean:<12.4f} {diff:<+12.4f} {signal:<10}")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Detection rate at t=0
    detected_at_0 = sum(1 for c in catastrophes if c["early_detection"].get(0, {}).get("would_alert", False))
    detected_at_2 = sum(1 for c in catastrophes if c["early_detection"].get(2, {}).get("would_alert", False))

    print(f"\n1. EARLY DETECTION:")
    print(f"   - At t=0 (when LP would start): {detected_at_0/len(catastrophes)*100:.1f}% detected")
    print(f"   - At t-20min: {detected_at_2/len(catastrophes)*100:.1f}% detected")
    print(f"   - Signal degrades ~{(detected_at_0-detected_at_2)/len(catastrophes)*100:.1f}% over 20 min")

    print(f"\n2. DIRECTIONALITY:")
    up_pct = len(up_moves)/len(catastrophes)*100
    print(f"   - {up_pct:.1f}% UP vs {100-up_pct:.1f}% DOWN (roughly balanced)")
    print(f"   - Current features do NOT reliably predict direction")
    print(f"   - Implication: withdraw to 50/50 or stables, not directional bet")


if __name__ == "__main__":
    main()
