"""
Synthetic Data Generator for Polymarket BTC Markets

Generates realistic order book snapshots and BTC price movements
for testing the backtesting system.

Based on observed patterns from real Polymarket 5-minute BTC markets:
- Markets created every 5 minutes
- BTC threshold set at current price (rounded)
- Order books develop liquidity over time
- Spreads narrow as market end approaches
- Final 30 seconds see rapid price action
"""

import json
import random
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class SyntheticSnapshot:
    """Single order book snapshot."""
    timestamp: str
    market_id: str
    market_type: str
    market_end_time: str
    btc_price: float
    btc_threshold: float
    seconds_until_close: int
    up_token_mid: float
    down_token_mid: float
    up_token_bids: List[Dict[str, float]]
    up_token_asks: List[Dict[str, float]]
    spread_bps: float

    def to_dict(self) -> dict:
        return asdict(self)


class SyntheticDataGenerator:
    """Generates realistic synthetic Polymarket data."""

    def __init__(self, seed: int = 42):
        """Initialize with random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def generate_btc_price_path(
        self,
        start_price: float,
        num_minutes: int,
        volatility: float = 0.0002  # ~2bps per minute
    ) -> List[Tuple[datetime, float]]:
        """
        Generate realistic BTC price path using geometric Brownian motion.

        Args:
            start_price: Starting BTC price
            num_minutes: Number of minutes to simulate
            volatility: Per-minute volatility (std dev)

        Returns:
            List of (timestamp, price) tuples
        """
        prices = [start_price]
        start_time = datetime.now()

        for i in range(num_minutes):
            # Geometric Brownian motion: price changes are log-normal
            change = np.random.normal(0, volatility)
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)

        timestamps = [start_time + timedelta(minutes=i) for i in range(num_minutes + 1)]

        return list(zip(timestamps, prices))

    def calculate_fair_price(
        self,
        btc_price: float,
        threshold: float,
        seconds_until_close: int,
        volatility: float = 0.0002
    ) -> float:
        """
        Calculate fair price for UP token using Black-Scholes-like logic.

        Digital option pricing:
        P(UP) = N(d) where d = (ln(S/K)) / (sigma * sqrt(t))
        S = current BTC price
        K = threshold price
        t = time to expiration (years)
        sigma = volatility
        N = normal CDF

        Args:
            btc_price: Current BTC price
            threshold: Strike price for the market
            seconds_until_close: Seconds until market closes
            volatility: Annualized volatility (~0.5 for BTC)

        Returns:
            Fair price for UP token (0 to 1)
        """
        if seconds_until_close <= 0:
            # Market closed - binary outcome
            return 1.0 if btc_price >= threshold else 0.0

        # Time to expiration in years
        years_to_expiry = seconds_until_close / (365.25 * 24 * 3600)

        # Prevent division by zero
        if years_to_expiry < 1e-6:
            return 1.0 if btc_price >= threshold else 0.0

        # Calculate d (normalized distance to strike)
        log_moneyness = np.log(btc_price / threshold)
        vol_time = volatility * np.sqrt(years_to_expiry)

        if vol_time < 1e-6:
            # Very close to expiry - step function
            return 1.0 if btc_price >= threshold else 0.0

        d = log_moneyness / vol_time

        # Normal CDF (probability of being ITM)
        from scipy import stats
        fair_price = stats.norm.cdf(d)

        return max(0.01, min(0.99, fair_price))

    def generate_order_book(
        self,
        mid_price: float,
        seconds_until_close: int,
        market_depth: float = 1000.0
    ) -> Tuple[List[Dict], List[Dict], float]:
        """
        Generate realistic order book with bids and asks.

        Spread narrows as market close approaches.
        More liquidity near mid price.

        Args:
            mid_price: Fair mid price for the token
            seconds_until_close: Seconds until market closes
            market_depth: Total liquidity in the market

        Returns:
            (bids, asks, spread_bps)
        """
        # Spread narrows as we approach close
        # Wide spread early (50-100 bps), tight late (5-10 bps)
        if seconds_until_close > 240:  # >4 min
            base_spread_bps = random.uniform(50, 100)
        elif seconds_until_close > 120:  # 2-4 min
            base_spread_bps = random.uniform(20, 50)
        elif seconds_until_close > 30:  # 30s-2min
            base_spread_bps = random.uniform(10, 25)
        else:  # <30s
            base_spread_bps = random.uniform(5, 15)

        # Convert bps to price units
        spread = mid_price * (base_spread_bps / 10000)
        half_spread = spread / 2

        bid_price = mid_price - half_spread
        ask_price = mid_price + half_spread

        # Generate order book levels (3-5 levels each side)
        num_levels = random.randint(3, 5)

        bids = []
        asks = []

        # Distribute liquidity across levels (more near mid)
        for i in range(num_levels):
            # Price spacing increases away from mid
            level_offset = (i + 1) * (spread * random.uniform(0.5, 1.5))

            # Size decreases away from mid
            size_factor = np.exp(-i * 0.5)  # Exponential decay
            size = (market_depth / num_levels) * size_factor * random.uniform(0.5, 1.5)

            bid = {
                "price": round(max(0.01, bid_price - level_offset * i), 4),
                "size": round(size, 2)
            }

            ask = {
                "price": round(min(0.99, ask_price + level_offset * i), 4),
                "size": round(size, 2)
            }

            bids.append(bid)
            asks.append(ask)

        # Sort bids descending, asks ascending
        bids.sort(key=lambda x: x["price"], reverse=True)
        asks.sort(key=lambda x: x["price"])

        return bids, asks, base_spread_bps

    def generate_market(
        self,
        market_id: str,
        start_time: datetime,
        duration_minutes: int,
        btc_start_price: float,
        btc_threshold: float,
        snapshot_interval_seconds: int = 5
    ) -> List[SyntheticSnapshot]:
        """
        Generate complete market lifecycle with snapshots.

        Args:
            market_id: Unique market ID
            start_time: Market start time
            duration_minutes: Market duration (5 for 5m markets)
            btc_start_price: BTC price at market start
            btc_threshold: Threshold price for resolution
            snapshot_interval_seconds: Seconds between snapshots

        Returns:
            List of snapshots for the entire market
        """
        snapshots = []
        end_time = start_time + timedelta(minutes=duration_minutes)

        # Generate BTC price path (1-second granularity for realism)
        num_seconds = duration_minutes * 60
        btc_prices = []
        current_price = btc_start_price

        for i in range(num_seconds + 1):
            # Micro-movements each second
            change = np.random.normal(0, 0.0001)  # ~1bps per second
            current_price = current_price * (1 + change)
            timestamp = start_time + timedelta(seconds=i)
            btc_prices.append((timestamp, current_price))

        # Generate snapshots at specified interval
        for i in range(0, num_seconds + 1, snapshot_interval_seconds):
            timestamp, btc_price = btc_prices[i]
            seconds_until_close = (end_time - timestamp).total_seconds()

            # Calculate fair prices
            up_fair = self.calculate_fair_price(
                btc_price, btc_threshold, seconds_until_close
            )
            down_fair = 1.0 - up_fair

            # Add market inefficiency (noise in market-implied price)
            # This simulates markets being slightly mispriced relative to fair value
            # Noise decreases as market close approaches (more efficient)
            inefficiency_factor = min(1.0, seconds_until_close / 180.0)  # Max at 180s+
            noise_std = 0.02 * inefficiency_factor  # Up to 2% noise early, 0% at close
            market_noise = np.random.normal(0, noise_std)

            # Market-implied price = fair price + noise
            market_mid = max(0.02, min(0.98, up_fair + market_noise))

            # Generate order books around market price (not fair price)
            market_depth = random.uniform(500, 2000)  # Varies by market
            up_bids, up_asks, spread_bps = self.generate_order_book(
                market_mid, seconds_until_close, market_depth
            )

            snapshot = SyntheticSnapshot(
                timestamp=timestamp.isoformat(),
                market_id=market_id,
                market_type=f"{duration_minutes}m",
                market_end_time=end_time.isoformat(),
                btc_price=round(btc_price, 2),
                btc_threshold=round(btc_threshold, 2),
                seconds_until_close=int(seconds_until_close),
                up_token_mid=round(market_mid, 4),  # Market price (not fair)
                down_token_mid=round(1.0 - market_mid, 4),
                up_token_bids=up_bids,
                up_token_asks=up_asks,
                spread_bps=round(spread_bps, 2)
            )

            snapshots.append(snapshot)

        return snapshots

    def generate_dataset(
        self,
        num_days: int = 30,
        markets_per_day: int = 288,  # Every 5 minutes
        btc_start_price: float = 95000.0,
        save_path: str = "data/synthetic_markets.json"
    ) -> Dict[str, List[SyntheticSnapshot]]:
        """
        Generate complete dataset of markets.

        Args:
            num_days: Number of days to simulate
            markets_per_day: Markets per day (288 = every 5 min)
            btc_start_price: Starting BTC price
            save_path: Path to save JSON data

        Returns:
            Dict mapping market_id -> list of snapshots
        """
        print(f"Generating {num_days} days of synthetic data...")
        print(f"  Markets per day: {markets_per_day}")
        print(f"  Total markets: {num_days * markets_per_day}")

        all_markets = {}
        start_time = datetime.now() - timedelta(days=num_days)

        # Generate BTC price path for entire period
        minutes_total = num_days * 24 * 60
        btc_path = self.generate_btc_price_path(btc_start_price, minutes_total)

        # Generate markets at 5-minute intervals
        for day in range(num_days):
            for market_num in range(markets_per_day):
                market_idx = day * markets_per_day + market_num
                market_start = start_time + timedelta(minutes=market_idx * 5)

                # Get BTC price at market start
                path_idx = market_idx * 5  # 5-minute intervals
                if path_idx < len(btc_path):
                    _, btc_price = btc_path[path_idx]
                else:
                    btc_price = btc_path[-1][1]

                # Threshold is rounded BTC price
                btc_threshold = round(btc_price / 100) * 100  # Round to nearest $100

                # Generate market
                market_id = f"market_{market_idx:05d}"

                snapshots = self.generate_market(
                    market_id=market_id,
                    start_time=market_start,
                    duration_minutes=5,
                    btc_start_price=btc_price,
                    btc_threshold=btc_threshold,
                    snapshot_interval_seconds=5  # Snapshot every 5 seconds
                )

                all_markets[market_id] = snapshots

                if (market_idx + 1) % 100 == 0:
                    print(f"  Generated {market_idx + 1}/{num_days * markets_per_day} markets...")

        print(f"\n✅ Generated {len(all_markets)} markets")
        print(f"   Total snapshots: {sum(len(s) for s in all_markets.values())}")

        # Save to JSON
        print(f"\nSaving to {save_path}...")
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        json_data = {
            market_id: [s.to_dict() for s in snapshots]
            for market_id, snapshots in all_markets.items()
        }

        with open(save_path, "w") as f:
            json.dump(json_data, f, indent=2)

        print(f"✅ Saved to {save_path}")

        return all_markets


def main():
    """Generate synthetic dataset."""
    # Import scipy for normal CDF
    try:
        from scipy import stats
    except ImportError:
        print("Installing scipy for normal CDF calculation...")
        import subprocess
        subprocess.check_call([
            ".venv/bin/pip", "install", "scipy", "numpy"
        ])
        from scipy import stats

    generator = SyntheticDataGenerator(seed=42)

    # Generate 7 days for testing (keep it manageable)
    dataset = generator.generate_dataset(
        num_days=7,
        markets_per_day=288,  # Every 5 minutes
        btc_start_price=95000.0,
        save_path="data/synthetic_markets_7d.json"
    )

    # Print statistics
    print("\n" + "=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)

    total_snapshots = sum(len(s) for s in dataset.values())
    avg_snapshots = total_snapshots / len(dataset)

    print(f"Total markets: {len(dataset)}")
    print(f"Total snapshots: {total_snapshots:,}")
    print(f"Avg snapshots per market: {avg_snapshots:.1f}")

    # Sample market statistics
    sample_market_id = list(dataset.keys())[0]
    sample_snapshots = dataset[sample_market_id]

    print(f"\nSample market: {sample_market_id}")
    print(f"  Snapshots: {len(sample_snapshots)}")
    print(f"  Duration: {sample_snapshots[0].timestamp} to {sample_snapshots[-1].timestamp}")
    print(f"  BTC start: ${sample_snapshots[0].btc_price:,.2f}")
    print(f"  BTC end: ${sample_snapshots[-1].btc_price:,.2f}")
    print(f"  Threshold: ${sample_snapshots[0].btc_threshold:,.2f}")

    # Spread statistics
    spreads = [s.spread_bps for s in sample_snapshots]
    print(f"\n  Spread (bps):")
    print(f"    Min: {min(spreads):.1f}")
    print(f"    Max: {max(spreads):.1f}")
    print(f"    Avg: {np.mean(spreads):.1f}")
    print(f"    Final: {spreads[-1]:.1f}")


if __name__ == "__main__":
    main()
