"""
PolyBackTest API Client

Fetches historical Polymarket order book data for backtesting.
API Documentation: https://docs.polybacktest.com/
"""

import os
import time
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import requests
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OrderBookSnapshot:
    """Single order book snapshot with BTC price reference."""
    timestamp: datetime
    market_id: str
    market_type: str  # "5m", "15m", "1hr", "4hr", "24hr"
    btc_price: float
    up_token_bids: List[Dict[str, float]]  # [{"price": 0.55, "size": 100}, ...]
    up_token_asks: List[Dict[str, float]]
    down_token_bids: List[Dict[str, float]]
    down_token_asks: List[Dict[str, float]]

    @property
    def up_token_mid_price(self) -> float:
        """Calculate mid price for UP token."""
        if not self.up_token_bids or not self.up_token_asks:
            return 0.0
        best_bid = self.up_token_bids[0]["price"]
        best_ask = self.up_token_asks[0]["price"]
        return (best_bid + best_ask) / 2

    @property
    def down_token_mid_price(self) -> float:
        """Calculate mid price for DOWN token."""
        if not self.down_token_bids or not self.down_token_asks:
            return 0.0
        best_bid = self.down_token_bids[0]["price"]
        best_ask = self.down_token_asks[0]["price"]
        return (best_bid + best_ask) / 2

    @property
    def spread_bps(self) -> float:
        """Calculate spread in basis points for UP token."""
        if not self.up_token_bids or not self.up_token_asks:
            return 0.0
        best_bid = self.up_token_bids[0]["price"]
        best_ask = self.up_token_asks[0]["price"]
        mid = (best_bid + best_ask) / 2
        if mid == 0:
            return 0.0
        return ((best_ask - best_bid) / mid) * 10000


@dataclass
class Market:
    """Polymarket BTC market metadata."""
    market_id: str
    question: str
    market_type: str  # "5m", "15m", "1hr", "4hr", "24hr"
    start_time: datetime
    end_time: datetime
    resolution_time: Optional[datetime]
    resolved: bool
    btc_threshold: float  # BTC price threshold for resolution
    up_token_id: str
    down_token_id: str


class PolyBackTestClient:
    """Client for PolyBackTest API - Historical Polymarket data."""

    BASE_URL = "https://api.polybacktest.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize PolyBackTest client.

        Args:
            api_key: API key from polybacktest.com. If not provided,
                    will look for POLYBACKTEST_API_KEY env variable.
        """
        self.api_key = api_key or os.getenv("POLYBACKTEST_API_KEY")
        if not self.api_key:
            logger.warning(
                "No PolyBackTest API key provided. "
                "Sign up at https://polybacktest.com to get one."
            )

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"X-API-Key": self.api_key})

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make API request with retry logic."""
        url = f"{self.BASE_URL}{endpoint}"

        for attempt in range(3):
            try:
                response = self.session.request(method, url, **kwargs)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 401:
                    raise ValueError(
                        "Invalid API key. Sign up at https://polybacktest.com"
                    )
                elif e.response.status_code == 429:
                    # Rate limited, wait and retry
                    wait_time = 2 ** attempt
                    logger.warning(f"Rate limited. Waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise
            except requests.exceptions.RequestException as e:
                if attempt < 2:
                    logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                    time.sleep(1)
                    continue
                else:
                    raise

        raise Exception("Max retries exceeded")

    def get_markets(
        self,
        market_type: Optional[str] = None,
        active_only: bool = False,
        limit: int = 100
    ) -> List[Market]:
        """
        Get list of BTC markets.

        Args:
            market_type: Filter by type: "5m", "15m", "1hr", "4hr", "24hr"
            active_only: Only return active (unresolved) markets
            limit: Max number of markets to return

        Returns:
            List of Market objects
        """
        params = {"limit": limit}
        if market_type:
            params["market_type"] = market_type
        if active_only:
            params["active"] = "true"

        response = self._request("GET", "/markets", params=params)

        markets = []
        for item in response.get("markets", []):
            market = Market(
                market_id=item["market_id"],
                question=item["question"],
                market_type=item["market_type"],
                start_time=datetime.fromisoformat(item["start_time"].replace("Z", "+00:00")),
                end_time=datetime.fromisoformat(item["end_time"].replace("Z", "+00:00")),
                resolution_time=datetime.fromisoformat(item["resolution_time"].replace("Z", "+00:00")) if item.get("resolution_time") else None,
                resolved=item["resolved"],
                btc_threshold=float(item["btc_threshold"]),
                up_token_id=item["up_token_id"],
                down_token_id=item["down_token_id"]
            )
            markets.append(market)

        return markets

    def get_snapshots(
        self,
        market_id: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 10000
    ) -> List[OrderBookSnapshot]:
        """
        Get historical order book snapshots for a market.

        Args:
            market_id: Market ID to fetch snapshots for
            start_time: Start of time range (default: market start)
            end_time: End of time range (default: market end)
            limit: Max snapshots to return (default: 10000)

        Returns:
            List of OrderBookSnapshot objects, sorted by timestamp
        """
        params = {"limit": limit}
        if start_time:
            params["start_time"] = start_time.isoformat()
        if end_time:
            params["end_time"] = end_time.isoformat()

        response = self._request(
            "GET",
            f"/markets/{market_id}/snapshots",
            params=params
        )

        snapshots = []
        for item in response.get("snapshots", []):
            snapshot = OrderBookSnapshot(
                timestamp=datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
                market_id=market_id,
                market_type=item["market_type"],
                btc_price=float(item["btc_price"]),
                up_token_bids=item["up_token"]["bids"],
                up_token_asks=item["up_token"]["asks"],
                down_token_bids=item["down_token"]["bids"],
                down_token_asks=item["down_token"]["asks"]
            )
            snapshots.append(snapshot)

        return snapshots

    def download_market_data(
        self,
        market_type: str = "5m",
        days_back: int = 30,
        save_path: Optional[str] = None
    ) -> Dict[str, List[OrderBookSnapshot]]:
        """
        Download historical data for all markets of a given type.

        Args:
            market_type: Type of markets to download ("5m", "15m", etc.)
            days_back: How many days of history to download
            save_path: Optional path to save data as JSON

        Returns:
            Dict mapping market_id -> list of snapshots
        """
        logger.info(f"Fetching {market_type} markets from last {days_back} days...")

        # Get all markets of this type
        all_markets = self.get_markets(market_type=market_type, limit=1000)

        # Filter to markets from last N days
        cutoff = datetime.now() - timedelta(days=days_back)
        recent_markets = [
            m for m in all_markets
            if m.end_time >= cutoff
        ]

        logger.info(f"Found {len(recent_markets)} markets to download")

        # Download snapshots for each market
        all_data = {}
        for i, market in enumerate(recent_markets, 1):
            logger.info(
                f"[{i}/{len(recent_markets)}] Downloading {market.market_id} "
                f"({market.question[:50]}...)"
            )

            try:
                snapshots = self.get_snapshots(market.market_id)
                all_data[market.market_id] = snapshots
                logger.info(f"  -> {len(snapshots)} snapshots")

                # Be nice to the API
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"  -> Failed: {e}")
                continue

        if save_path:
            logger.info(f"Saving data to {save_path}...")
            import json

            # Convert to JSON-serializable format
            json_data = {}
            for market_id, snapshots in all_data.items():
                json_data[market_id] = [
                    {
                        "timestamp": s.timestamp.isoformat(),
                        "btc_price": s.btc_price,
                        "market_type": s.market_type,
                        "up_token": {
                            "bids": s.up_token_bids,
                            "asks": s.up_token_asks,
                            "mid_price": s.up_token_mid_price
                        },
                        "down_token": {
                            "bids": s.down_token_bids,
                            "asks": s.down_token_asks,
                            "mid_price": s.down_token_mid_price
                        },
                        "spread_bps": s.spread_bps
                    }
                    for s in snapshots
                ]

            with open(save_path, "w") as f:
                json.dump(json_data, f, indent=2)

            logger.info(f"Saved {len(all_data)} markets with {sum(len(s) for s in all_data.values())} total snapshots")

        return all_data


def main():
    """Example usage and testing."""
    logging.basicConfig(level=logging.INFO)

    client = PolyBackTestClient()

    # Test 1: Get active 5-minute markets
    print("\n=== Active 5-minute BTC Markets ===")
    markets = client.get_markets(market_type="5m", active_only=True, limit=5)
    for market in markets:
        print(f"  {market.question}")
        print(f"    Ends: {market.end_time}")
        print(f"    Threshold: ${market.btc_threshold:,.2f}")
        print()

    # Test 2: Get snapshots for most recent market
    if markets:
        print(f"\n=== Snapshots for: {markets[0].question} ===")
        snapshots = client.get_snapshots(markets[0].market_id, limit=10)

        for snapshot in snapshots[:5]:
            print(f"  {snapshot.timestamp}")
            print(f"    BTC: ${snapshot.btc_price:,.2f}")
            print(f"    UP mid: {snapshot.up_token_mid_price:.4f}")
            print(f"    DOWN mid: {snapshot.down_token_mid_price:.4f}")
            print(f"    Spread: {snapshot.spread_bps:.1f} bps")
            print()

    # Test 3: Download 7 days of 5m market data
    print("\n=== Downloading 7 days of 5m market data ===")
    data = client.download_market_data(
        market_type="5m",
        days_back=7,
        save_path="data/polybacktest_5m_7days.json"
    )

    print(f"\nDownloaded {len(data)} markets")
    print(f"Total snapshots: {sum(len(s) for s in data.values())}")


if __name__ == "__main__":
    main()
