"""
Polymarket API Client
Interfaces with Polymarket's CLOB API and Gamma API for market data.
"""

import requests
import time
from typing import Dict, List, Optional, Any
from datetime import datetime
import json


class PolymarketClient:
    """Client for interacting with Polymarket APIs"""

    CLOB_BASE_URL = "https://clob.polymarket.com"
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Research Bot)',
            'Accept': 'application/json'
        })

    def get_markets(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Fetch markets from Gamma API

        Args:
            limit: Number of markets to fetch
            offset: Offset for pagination

        Returns:
            List of market dictionaries
        """
        url = f"{self.GAMMA_BASE_URL}/markets"
        params = {
            'limit': limit,
            'offset': offset,
            'closed': 'false'
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching markets: {e}")
            return []

    def search_btc_markets(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for BTC 5-minute up/down markets

        Returns:
            List of BTC market dictionaries
        """
        markets = self.get_markets(limit=limit)

        # Filter for BTC up/down 5-minute markets
        btc_markets = []
        for market in markets:
            question = market.get('question', '').lower()
            if 'btc' in question or 'bitcoin' in question:
                if 'up or down' in question or '5:' in question:
                    btc_markets.append(market)

        return btc_markets

    def get_market_details(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed market information

        Args:
            condition_id: The condition ID for the market

        Returns:
            Market details dictionary or None
        """
        url = f"{self.GAMMA_BASE_URL}/markets/{condition_id}"

        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching market details: {e}")
            return None

    def get_order_book(self, token_id: str) -> Optional[Dict[str, Any]]:
        """
        Get order book for a specific outcome token

        Args:
            token_id: The ERC1155 token ID for the outcome

        Returns:
            Order book data with bids, asks, spread, etc.
        """
        url = f"{self.CLOB_BASE_URL}/book"
        params = {'token_id': token_id}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Parse order book
            order_book = {
                'timestamp': datetime.utcnow().isoformat(),
                'token_id': token_id,
                'bids': data.get('bids', []),
                'asks': data.get('asks', []),
                'market': data.get('market', ''),
                'asset_id': data.get('asset_id', '')
            }

            # Calculate spread and mid price
            if order_book['bids'] and order_book['asks']:
                best_bid = float(order_book['bids'][0]['price'])
                best_ask = float(order_book['asks'][0]['price'])
                order_book['best_bid'] = best_bid
                order_book['best_ask'] = best_ask
                order_book['spread'] = best_ask - best_bid
                order_book['mid_price'] = (best_bid + best_ask) / 2

            return order_book

        except requests.exceptions.RequestException as e:
            print(f"Error fetching order book: {e}")
            return None

    def get_last_trade_price(self, token_id: str) -> Optional[float]:
        """
        Get the last trade price for a token

        Args:
            token_id: The ERC1155 token ID

        Returns:
            Last trade price or None
        """
        url = f"{self.CLOB_BASE_URL}/last-trade-price"
        params = {'token_id': token_id}

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return float(response.json().get('price', 0))
        except requests.exceptions.RequestException as e:
            print(f"Error fetching last trade price: {e}")
            return None

    def get_market_trades(self, condition_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent trades for a market

        Args:
            condition_id: The condition ID
            limit: Number of trades to fetch

        Returns:
            List of trade dictionaries
        """
        url = f"{self.CLOB_BASE_URL}/trades"
        params = {
            'market': condition_id,
            'limit': limit
        }

        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching trades: {e}")
            return []

    def monitor_market_realtime(
        self,
        token_id_up: str,
        token_id_down: str,
        duration_seconds: int = 30,
        interval_seconds: float = 1.0
    ) -> List[Dict[str, Any]]:
        """
        Monitor a market's order book in real-time

        Args:
            token_id_up: Token ID for "Up" outcome
            token_id_down: Token ID for "Down" outcome
            duration_seconds: How long to monitor
            interval_seconds: Sampling interval

        Returns:
            List of snapshots with timestamps
        """
        snapshots = []
        start_time = time.time()
        end_time = start_time + duration_seconds

        print(f"Monitoring market for {duration_seconds} seconds...")

        while time.time() < end_time:
            snapshot = {
                'timestamp': datetime.utcnow().isoformat(),
                'unix_time': time.time()
            }

            # Get order books for both outcomes
            book_up = self.get_order_book(token_id_up)
            book_down = self.get_order_book(token_id_down)

            if book_up:
                snapshot['up_price'] = book_up.get('mid_price')
                snapshot['up_bid'] = book_up.get('best_bid')
                snapshot['up_ask'] = book_up.get('best_ask')
                snapshot['up_spread'] = book_up.get('spread')

            if book_down:
                snapshot['down_price'] = book_down.get('mid_price')
                snapshot['down_bid'] = book_down.get('best_bid')
                snapshot['down_ask'] = book_down.get('best_ask')
                snapshot['down_spread'] = book_down.get('spread')

            # Calculate implied probabilities
            if snapshot.get('up_price') and snapshot.get('down_price'):
                snapshot['implied_prob_up'] = snapshot['up_price']
                snapshot['implied_prob_down'] = snapshot['down_price']
                snapshot['prob_sum'] = snapshot['up_price'] + snapshot['down_price']

            snapshots.append(snapshot)

            # Wait for next interval
            time.sleep(interval_seconds)

        print(f"Collected {len(snapshots)} snapshots")
        return snapshots


def main():
    """Example usage of PolymarketClient"""
    client = PolymarketClient()

    print("=== Polymarket BTC Markets Research ===\n")

    # Search for BTC markets
    print("Searching for BTC 5-minute markets...")
    btc_markets = client.search_btc_markets(limit=200)

    print(f"Found {len(btc_markets)} BTC-related markets\n")

    # Display first few markets
    for i, market in enumerate(btc_markets[:5]):
        print(f"Market {i+1}:")
        print(f"  Question: {market.get('question', 'N/A')}")
        print(f"  Condition ID: {market.get('condition_id', 'N/A')}")
        print(f"  End Date: {market.get('end_date_iso', 'N/A')}")
        print(f"  Active: {market.get('active', 'N/A')}")
        print()

    # If we found active markets, demonstrate order book fetching
    if btc_markets:
        market = btc_markets[0]
        print(f"\nFetching order book for: {market.get('question', 'N/A')}")

        # Note: You'd need to get the actual token IDs from the market
        # This is just a demonstration structure
        tokens = market.get('tokens', [])
        if len(tokens) >= 2:
            token_id_0 = tokens[0].get('token_id')
            token_id_1 = tokens[1].get('token_id')

            if token_id_0:
                book = client.get_order_book(token_id_0)
                if book:
                    print(f"\nOutcome 0 Order Book:")
                    print(f"  Best Bid: {book.get('best_bid', 'N/A')}")
                    print(f"  Best Ask: {book.get('best_ask', 'N/A')}")
                    print(f"  Spread: {book.get('spread', 'N/A')}")
                    print(f"  Mid Price: {book.get('mid_price', 'N/A')}")


if __name__ == "__main__":
    main()
