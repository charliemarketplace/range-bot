"""
Analyze what distinguishes catastrophic misses from good predictions.

Look at lookback characteristics that might predict regime breaks.
"""
import json
import sqlite3
import statistics
import math
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
CANDLE_BLOCKS = 50


def get_swaps(conn, from_block, to_block):
    cursor = conn.execute("""
        SELECT block_number, price, amount1
        FROM swaps
        WHERE block_number BETWEEN ? AND ?
        ORDER BY block_number
    """, (from_block, to_block))
    return [{"block": row[0], "price": row[1], "amount1": int(row[2])} for row in cursor]


def swaps_to_candles(swaps):
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
                "h": max(prices), "l": min(prices),
                "vwap": vwap, "vol": total_vol, "n": len(ps)
            })
        period_start = period_end
    return candles


def analyze_lookback(candles):
    """Extract features from lookback candles that might predict failure."""
    if len(candles) < 5:
        return None

    recent = candles[-10:] if len(candles) >= 10 else candles
    vwaps = [c["vwap"] for c in recent]
    vols = [c["vol"] for c in recent]
    ranges = [c["h"] - c["l"] for c in recent]

    # Basic stats
    vwap_std = statistics.stdev(vwaps) if len(vwaps) > 1 else 0
    vol_mean = statistics.mean(vols)

    # Stability score
    net_move = abs(vwaps[-1] - vwaps[0])
    total_path = sum(abs(vwaps[i+1] - vwaps[i]) for i in range(len(vwaps)-1))
    stability = 1 - (net_move / total_path) if total_path > 0 else 1

    # VOLATILITY ACCELERATION: is recent vol higher than earlier vol?
    if len(recent) >= 6:
        early_ranges = ranges[:len(ranges)//2]
        late_ranges = ranges[len(ranges)//2:]
        vol_accel = statistics.mean(late_ranges) / statistics.mean(early_ranges) if statistics.mean(early_ranges) > 0 else 1
    else:
        vol_accel = 1

    # STABILITY TREND: is stability decreasing?
    if len(recent) >= 6:
        mid = len(recent) // 2
        early_vwaps = vwaps[:mid]
        late_vwaps = vwaps[mid:]

        early_net = abs(early_vwaps[-1] - early_vwaps[0])
        early_path = sum(abs(early_vwaps[i+1] - early_vwaps[i]) for i in range(len(early_vwaps)-1))
        early_stab = 1 - (early_net / early_path) if early_path > 0 else 1

        late_net = abs(late_vwaps[-1] - late_vwaps[0])
        late_path = sum(abs(late_vwaps[i+1] - late_vwaps[i]) for i in range(len(late_vwaps)-1))
        late_stab = 1 - (late_net / late_path) if late_path > 0 else 1

        stability_trend = late_stab - early_stab  # Negative = getting less stable
    else:
        stability_trend = 0

    # VOLUME SPIKE: is recent volume much higher than average?
    if len(vols) >= 3:
        vol_spike = vols[-1] / statistics.mean(vols[:-1]) if statistics.mean(vols[:-1]) > 0 else 1
    else:
        vol_spike = 1

    # RANGE EXPANSION: is the candle range growing?
    if len(ranges) >= 3:
        range_expansion = ranges[-1] / statistics.mean(ranges[:-1]) if statistics.mean(ranges[:-1]) > 0 else 1
    else:
        range_expansion = 1

    # PRICE VELOCITY: how fast is price moving?
    if len(vwaps) >= 2:
        price_velocity = abs(vwaps[-1] - vwaps[-2]) / vwaps[-2] * 100  # % change
    else:
        price_velocity = 0

    return {
        "stability": stability,
        "stability_trend": stability_trend,
        "vol_accel": vol_accel,
        "vol_spike": vol_spike,
        "range_expansion": range_expansion,
        "price_velocity": price_velocity,
        "vwap_std": vwap_std,
    }


def main():
    # Load random backtest results
    with open(DATA_DIR / "backtest_random.json") as f:
        data = json.load(f)

    results = data["sample_results"]
    print(f"Analyzing {len(results)} sample results")

    # Split into hits (>= 80% coverage) and misses (< 30% coverage)
    hits = [r for r in results if r["coverage"] >= 0.8]
    misses = [r for r in results if r["coverage"] < 0.3]

    print(f"Hits (>=80%): {len(hits)}")
    print(f"Misses (<30%): {len(misses)}")

    # Sample some from each for detailed analysis
    import random
    random.seed(42)

    sample_hits = random.sample(hits, min(500, len(hits)))
    sample_misses = random.sample(misses, min(500, len(misses)))

    conn = sqlite3.connect(str(DATA_DIR / "swaps.db"))

    def analyze_group(samples, label):
        features = []
        for r in samples:
            swaps = get_swaps(conn, r["start_block"] - 1000, r["start_block"] - 1)
            candles = swaps_to_candles(swaps)
            f = analyze_lookback(candles)
            if f:
                features.append(f)

        if not features:
            return

        print(f"\n{label} ({len(features)} samples):")
        for key in features[0].keys():
            vals = [f[key] for f in features]
            mean = statistics.mean(vals)
            std = statistics.stdev(vals) if len(vals) > 1 else 0
            print(f"  {key:20s}: {mean:8.4f} +/- {std:.4f}")

        return features

    hit_features = analyze_group(sample_hits, "HITS (>=80% coverage)")
    miss_features = analyze_group(sample_misses, "MISSES (<30% coverage)")

    conn.close()

    # Compare
    if hit_features and miss_features:
        print("\n" + "=" * 60)
        print("DIFFERENCE (Miss - Hit):")
        print("=" * 60)
        for key in hit_features[0].keys():
            hit_mean = statistics.mean([f[key] for f in hit_features])
            miss_mean = statistics.mean([f[key] for f in miss_features])
            diff = miss_mean - hit_mean
            pct = (diff / hit_mean * 100) if hit_mean != 0 else 0
            signal = "***" if abs(pct) > 20 else "**" if abs(pct) > 10 else "*" if abs(pct) > 5 else ""
            print(f"  {key:20s}: {diff:+8.4f} ({pct:+.1f}%) {signal}")


if __name__ == "__main__":
    main()
