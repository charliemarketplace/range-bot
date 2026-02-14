"""
Chainlink Price Feed Fetcher
Fetches BTC-USD price data from Chainlink oracles on Polygon.
"""

from web3 import Web3
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import time
import json


class ChainlinkFetcher:
    """Fetches price data from Chainlink BTC-USD price feeds"""

    # Polygon Mainnet Chainlink BTC/USD Price Feed
    BTC_USD_FEED_ADDRESS = "0xc907E116054Ad103354f2D350FD2514433D57F6f"

    # Price Feed ABI (minimal interface)
    PRICE_FEED_ABI = [
        {
            "inputs": [],
            "name": "latestRoundData",
            "outputs": [
                {"name": "roundId", "type": "uint80"},
                {"name": "answer", "type": "int256"},
                {"name": "startedAt", "type": "uint256"},
                {"name": "updatedAt", "type": "uint256"},
                {"name": "answeredInRound", "type": "uint80"}
            ],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "decimals",
            "outputs": [{"name": "", "type": "uint8"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [],
            "name": "description",
            "outputs": [{"name": "", "type": "string"}],
            "stateMutability": "view",
            "type": "function"
        },
        {
            "inputs": [{"name": "roundId", "type": "uint80"}],
            "name": "getRoundData",
            "outputs": [
                {"name": "roundId", "type": "uint80"},
                {"name": "answer", "type": "int256"},
                {"name": "startedAt", "type": "uint256"},
                {"name": "updatedAt", "type": "uint256"},
                {"name": "answeredInRound", "type": "uint80"}
            ],
            "stateMutability": "view",
            "type": "function"
        }
    ]

    def __init__(self, rpc_url: str = "https://polygon-rpc.com"):
        """
        Initialize Chainlink fetcher

        Args:
            rpc_url: Polygon RPC endpoint (default: public RPC)
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))

        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to Polygon RPC: {rpc_url}")

        self.price_feed = self.w3.eth.contract(
            address=Web3.to_checksum_address(self.BTC_USD_FEED_ADDRESS),
            abi=self.PRICE_FEED_ABI
        )

        # Get feed metadata
        self.decimals = self.price_feed.functions.decimals().call()
        self.description = self.price_feed.functions.description().call()

        print(f"Connected to Chainlink feed: {self.description}")
        print(f"Decimals: {self.decimals}")

    def get_latest_price(self) -> Dict[str, any]:
        """
        Get the latest BTC-USD price from Chainlink

        Returns:
            Dictionary with price data
        """
        try:
            round_data = self.price_feed.functions.latestRoundData().call()

            round_id, answer, started_at, updated_at, answered_in_round = round_data

            # Convert price from int256 to float
            price = answer / (10 ** self.decimals)

            return {
                'round_id': round_id,
                'price': price,
                'started_at': started_at,
                'updated_at': updated_at,
                'updated_at_iso': datetime.utcfromtimestamp(updated_at).isoformat(),
                'answered_in_round': answered_in_round,
                'timestamp': datetime.utcnow().isoformat(),
                'feed': self.description
            }
        except Exception as e:
            print(f"Error fetching latest price: {e}")
            return None

    def get_round_data(self, round_id: int) -> Optional[Dict[str, any]]:
        """
        Get price data for a specific round

        Args:
            round_id: The round ID to query

        Returns:
            Dictionary with price data or None
        """
        try:
            round_data = self.price_feed.functions.getRoundData(round_id).call()

            round_id, answer, started_at, updated_at, answered_in_round = round_data

            price = answer / (10 ** self.decimals)

            return {
                'round_id': round_id,
                'price': price,
                'started_at': started_at,
                'updated_at': updated_at,
                'updated_at_iso': datetime.utcfromtimestamp(updated_at).isoformat(),
                'answered_in_round': answered_in_round
            }
        except Exception as e:
            print(f"Error fetching round {round_id}: {e}")
            return None

    def get_historical_prices(
        self,
        start_round: int,
        num_rounds: int = 100
    ) -> List[Dict[str, any]]:
        """
        Fetch historical price data for multiple rounds

        Args:
            start_round: Starting round ID
            num_rounds: Number of rounds to fetch

        Returns:
            List of price data dictionaries
        """
        prices = []

        print(f"Fetching {num_rounds} historical rounds starting from {start_round}...")

        for i in range(num_rounds):
            round_id = start_round + i
            data = self.get_round_data(round_id)

            if data:
                prices.append(data)
            else:
                print(f"Skipping round {round_id}")

            # Rate limiting
            if i % 10 == 0:
                time.sleep(0.1)

        print(f"Fetched {len(prices)} rounds successfully")
        return prices

    def monitor_price_updates(
        self,
        duration_seconds: int = 300,
        check_interval: float = 1.0
    ) -> List[Dict[str, any]]:
        """
        Monitor price updates in real-time

        Args:
            duration_seconds: How long to monitor (default 5 minutes)
            check_interval: How often to check for updates (seconds)

        Returns:
            List of price update events
        """
        updates = []
        start_time = time.time()
        end_time = start_time + duration_seconds

        last_round_id = None

        print(f"Monitoring Chainlink price updates for {duration_seconds} seconds...")
        print(f"Check interval: {check_interval} seconds\n")

        while time.time() < end_time:
            price_data = self.get_latest_price()

            if price_data:
                current_round_id = price_data['round_id']

                # Only record if this is a new round (price update)
                if current_round_id != last_round_id:
                    price_data['event'] = 'new_round'
                    price_data['local_timestamp'] = datetime.utcnow().isoformat()
                    updates.append(price_data)

                    print(f"Update detected - Round {current_round_id}: ${price_data['price']:.2f}")

                    last_round_id = current_round_id

            time.sleep(check_interval)

        print(f"\nMonitoring complete. Detected {len(updates)} price updates.")
        return updates

    def get_price_at_timestamp(
        self,
        target_timestamp: int,
        search_rounds: int = 100
    ) -> Optional[Dict[str, any]]:
        """
        Find the price closest to a target timestamp

        Args:
            target_timestamp: Unix timestamp to search for
            search_rounds: Number of recent rounds to search

        Returns:
            Price data closest to target timestamp
        """
        latest = self.get_latest_price()
        if not latest:
            return None

        latest_round = latest['round_id']

        # Search backwards from latest round
        best_match = None
        min_time_diff = float('inf')

        for i in range(search_rounds):
            round_id = latest_round - i
            data = self.get_round_data(round_id)

            if data:
                time_diff = abs(data['updated_at'] - target_timestamp)

                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    best_match = data
                    best_match['time_diff_seconds'] = time_diff

                # If we've gone too far back, stop
                if data['updated_at'] < target_timestamp - 3600:  # 1 hour buffer
                    break

        return best_match


def main():
    """Example usage of ChainlinkFetcher"""
    print("=== Chainlink BTC-USD Price Feed Demo ===\n")

    # Initialize fetcher
    try:
        fetcher = ChainlinkFetcher()
    except ConnectionError as e:
        print(f"Connection error: {e}")
        print("\nNote: You may need to use a dedicated RPC provider.")
        print("Free options: Alchemy, Infura, QuickNode")
        return

    print("\n1. Getting latest price...")
    latest = fetcher.get_latest_price()
    if latest:
        print(f"   BTC Price: ${latest['price']:.2f}")
        print(f"   Updated At: {latest['updated_at_iso']}")
        print(f"   Round ID: {latest['round_id']}")

    print("\n2. Monitoring for price updates (30 seconds)...")
    updates = fetcher.monitor_price_updates(duration_seconds=30, check_interval=1.0)

    if updates:
        print(f"\n   Detected {len(updates)} updates in 30 seconds")
        for update in updates:
            print(f"   - {update['updated_at_iso']}: ${update['price']:.2f}")
    else:
        print("   No updates detected (this is normal, updates occur every 1-5 minutes)")

    print("\n3. Testing historical price lookup...")
    target_time = int(time.time()) - 3600  # 1 hour ago
    historical = fetcher.get_price_at_timestamp(target_time, search_rounds=50)
    if historical:
        print(f"   Price ~1 hour ago: ${historical['price']:.2f}")
        print(f"   Actual timestamp: {historical['updated_at_iso']}")
        print(f"   Time difference: {historical.get('time_diff_seconds', 0)} seconds")


if __name__ == "__main__":
    main()
