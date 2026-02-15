"""
Market Data Collector
Collects and aligns data from Polymarket markets and Chainlink price feeds.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path

from polymarket_client import PolymarketClient
from chainlink_fetcher import ChainlinkFetcher


class MarketDataCollector:
    """Collects and aligns market and oracle data for analysis"""

    def __init__(
        self,
        rpc_url: str = "https://polygon-rpc.com",
        output_dir: str = "./data"
    ):
        """
        Initialize the data collector

        Args:
            rpc_url: Polygon RPC endpoint
            output_dir: Directory to save collected data
        """
        self.polymarket = PolymarketClient()
        self.chainlink = ChainlinkFetcher(rpc_url=rpc_url)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print("Market Data Collector initialized")

    def find_active_btc_market(self) -> Optional[Dict]:
        """
        Find an active BTC 5-minute market

        Returns:
            Market data dictionary or None
        """
        markets = self.polymarket.search_btc_markets(limit=200)

        # Filter for active markets
        now = datetime.utcnow()

        for market in markets:
            if market.get('active'):
                # Check if market is live (end date in future)
                end_date = market.get('end_date_iso')
                if end_date:
                    try:
                        end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                        if end_time > now:
                            return market
                    except:
                        continue

        return None

    def collect_market_window(
        self,
        market: Dict,
        monitor_duration: int = 30,
        sample_interval: float = 1.0
    ) -> Dict[str, any]:
        """
        Collect synchronized data for a market window

        Args:
            market: Market metadata
            monitor_duration: Seconds to monitor (e.g., final 30 seconds)
            sample_interval: Sampling frequency in seconds

        Returns:
            Dictionary with aligned market and oracle data
        """
        print(f"\nCollecting data for market: {market.get('question', 'Unknown')}")
        print(f"Monitor duration: {monitor_duration}s, Interval: {sample_interval}s")

        # Get token IDs for Up and Down outcomes
        tokens = market.get('tokens', [])
        if len(tokens) < 2:
            print("Error: Market doesn't have 2 outcome tokens")
            return {}

        # Typically: [No, Yes] or [Down, Up]
        # Need to identify which is which
        token_0 = tokens[0]
        token_1 = tokens[1]

        # Heuristic: "Yes" or outcome with higher token_id is often "Up"
        # You may need to adjust this based on actual market structure
        token_id_down = token_0.get('token_id')
        token_id_up = token_1.get('token_id')

        print(f"Token Down: {token_id_down}")
        print(f"Token Up: {token_id_up}")

        # Collect data
        data_points = []
        start_time = time.time()
        end_time = start_time + monitor_duration

        print("\nStarting data collection...")

        while time.time() < end_time:
            timestamp = datetime.utcnow()
            unix_time = time.time()

            # Collect Chainlink price
            chainlink_data = self.chainlink.get_latest_price()

            # Collect Polymarket order books
            book_up = self.polymarket.get_order_book(token_id_up)
            book_down = self.polymarket.get_order_book(token_id_down)

            # Construct data point
            data_point = {
                'timestamp_iso': timestamp.isoformat(),
                'timestamp_unix': unix_time,
                'chainlink': chainlink_data,
                'order_book_up': book_up,
                'order_book_down': book_down
            }

            # Calculate derived metrics
            if book_up and book_down:
                up_price = book_up.get('mid_price')
                down_price = book_down.get('mid_price')

                if up_price and down_price:
                    data_point['market_prob_up'] = up_price
                    data_point['market_prob_down'] = down_price
                    data_point['prob_sum'] = up_price + down_price
                    data_point['inefficiency'] = abs(1.0 - (up_price + down_price))

            data_points.append(data_point)

            # Progress indicator
            elapsed = time.time() - start_time
            remaining = monitor_duration - elapsed
            print(f"\rCollecting... {elapsed:.1f}s / {monitor_duration}s (remaining: {remaining:.1f}s)", end='')

            time.sleep(sample_interval)

        print(f"\n\nCollection complete! Collected {len(data_points)} data points.")

        # Package results
        result = {
            'market_metadata': market,
            'collection_start': datetime.utcfromtimestamp(start_time).isoformat(),
            'collection_end': datetime.utcfromtimestamp(end_time).isoformat(),
            'duration_seconds': monitor_duration,
            'sample_interval': sample_interval,
            'num_samples': len(data_points),
            'token_id_up': token_id_up,
            'token_id_down': token_id_down,
            'data_points': data_points
        }

        return result

    def save_collection(self, data: Dict, prefix: str = "market_data"):
        """
        Save collected data to JSON file

        Args:
            data: Collected data dictionary
            prefix: Filename prefix
        """
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"\nData saved to: {filepath}")
        return filepath

    def analyze_collection(self, data: Dict) -> Dict[str, any]:
        """
        Perform basic analysis on collected data

        Args:
            data: Collected market data

        Returns:
            Analysis results
        """
        data_points = data.get('data_points', [])

        if not data_points:
            return {'error': 'No data points to analyze'}

        # Extract time series
        timestamps = []
        btc_prices = []
        prob_up = []
        prob_down = []
        spreads_up = []
        spreads_down = []

        for point in data_points:
            timestamps.append(point['timestamp_unix'])

            # Chainlink price
            if point.get('chainlink'):
                btc_prices.append(point['chainlink'].get('price'))

            # Market probabilities
            if point.get('market_prob_up'):
                prob_up.append(point['market_prob_up'])
            if point.get('market_prob_down'):
                prob_down.append(point['market_prob_down'])

            # Spreads
            if point.get('order_book_up'):
                spreads_up.append(point['order_book_up'].get('spread', 0))
            if point.get('order_book_down'):
                spreads_down.append(point['order_book_down'].get('spread', 0))

        # Calculate summary statistics
        analysis = {
            'num_samples': len(data_points),
            'duration_seconds': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
        }

        # BTC price analysis
        if btc_prices:
            analysis['btc_price'] = {
                'start': btc_prices[0],
                'end': btc_prices[-1],
                'change': btc_prices[-1] - btc_prices[0],
                'change_pct': ((btc_prices[-1] - btc_prices[0]) / btc_prices[0]) * 100,
                'min': min(btc_prices),
                'max': max(btc_prices),
                'range': max(btc_prices) - min(btc_prices)
            }

        # Market probability analysis
        if prob_up:
            analysis['prob_up'] = {
                'start': prob_up[0],
                'end': prob_up[-1],
                'change': prob_up[-1] - prob_up[0],
                'min': min(prob_up),
                'max': max(prob_up),
                'mean': sum(prob_up) / len(prob_up)
            }

        if prob_down:
            analysis['prob_down'] = {
                'start': prob_down[0],
                'end': prob_down[-1],
                'change': prob_down[-1] - prob_down[0],
                'min': min(prob_down),
                'max': max(prob_down),
                'mean': sum(prob_down) / len(prob_down)
            }

        # Spread analysis
        if spreads_up:
            analysis['spread_up'] = {
                'mean': sum(spreads_up) / len(spreads_up),
                'min': min(spreads_up),
                'max': max(spreads_up)
            }

        if spreads_down:
            analysis['spread_down'] = {
                'mean': sum(spreads_down) / len(spreads_down),
                'min': min(spreads_down),
                'max': max(spreads_down)
            }

        return analysis


def main():
    """Example usage of MarketDataCollector"""
    print("=== Polymarket BTC Market Data Collector ===\n")

    # Initialize collector
    collector = MarketDataCollector(output_dir="./data")

    # Find an active market
    print("Searching for active BTC 5-minute market...")
    market = collector.find_active_btc_market()

    if not market:
        print("No active BTC markets found.")
        print("\nNote: BTC 5-minute markets may not be continuously active.")
        print("Try running this script when markets are known to be active.")
        return

    print(f"\nFound active market:")
    print(f"  Question: {market.get('question')}")
    print(f"  End time: {market.get('end_date_iso')}")

    # Collect data for final 30 seconds (or whatever duration you want)
    # In production, you'd want to time this to capture the final moments
    print("\nCollecting 30-second sample...")
    data = collector.collect_market_window(
        market=market,
        monitor_duration=30,
        sample_interval=1.0
    )

    if data:
        # Analyze the data
        print("\nAnalyzing collected data...")
        analysis = collector.analyze_collection(data)

        print("\n=== Analysis Results ===")
        print(json.dumps(analysis, indent=2))

        # Save to file
        collector.save_collection(data, prefix="btc_market")

        # Also save analysis
        collector.save_collection(analysis, prefix="btc_analysis")


if __name__ == "__main__":
    main()
