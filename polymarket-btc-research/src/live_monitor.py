#!/usr/bin/env python3
"""
Live Market Monitor
Real-time monitoring of BTC markets with Bayesian edge detection.
"""

import time
import json
from datetime import datetime, timedelta
from typing import Optional, Dict
import os
from pathlib import Path

from polymarket_client import PolymarketClient
from chainlink_fetcher import ChainlinkFetcher
from bayesian_model import BayesianBTCModel


class LiveMarketMonitor:
    """Real-time market monitoring and edge detection"""

    def __init__(
        self,
        rpc_url: str = "https://polygon-rpc.com",
        min_edge_threshold: float = 0.05,
        output_dir: str = "./data"
    ):
        """
        Initialize live monitor

        Args:
            rpc_url: Polygon RPC endpoint
            min_edge_threshold: Minimum edge to flag opportunities (default 5%)
            output_dir: Directory to save logs
        """
        print("Initializing Live Market Monitor...")
        self.polymarket = PolymarketClient()
        self.chainlink = ChainlinkFetcher(rpc_url=rpc_url)
        self.model = BayesianBTCModel()
        self.min_edge_threshold = min_edge_threshold
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Price history for volatility/drift estimation
        self.price_history = []
        self.timestamp_history = []

        print("✓ Initialization complete\n")

    def find_active_market(self) -> Optional[Dict]:
        """Find an active BTC 5-minute market"""
        markets = self.polymarket.search_btc_markets(limit=200)

        now = datetime.utcnow()

        for market in markets:
            if not market.get('active'):
                continue

            end_date = market.get('end_date_iso')
            if not end_date:
                continue

            try:
                end_time = datetime.fromisoformat(end_date.replace('Z', '+00:00'))

                # Check if market is currently active
                if end_time > now:
                    time_remaining = (end_time - now).total_seconds()

                    # Only return if market has reasonable time left (>30 sec, <5 min)
                    if 30 < time_remaining < 320:
                        market['time_remaining'] = time_remaining
                        market['end_time'] = end_time
                        return market
            except:
                continue

        return None

    def monitor_market_final_seconds(
        self,
        market: Dict,
        monitor_duration: int = 30
    ):
        """
        Monitor market in final seconds with Bayesian edge detection

        Args:
            market: Market metadata
            monitor_duration: Seconds to monitor
        """
        print(f"\n{'='*60}")
        print(f"MONITORING MARKET: {market.get('question', 'Unknown')}")
        print(f"End time: {market.get('end_date_iso')}")
        print(f"Time remaining: {market.get('time_remaining', 0):.1f} seconds")
        print(f"{'='*60}\n")

        # Extract token IDs
        tokens = market.get('tokens', [])
        if len(tokens) < 2:
            print("Error: Market doesn't have 2 outcome tokens")
            return

        token_id_down = tokens[0].get('token_id')
        token_id_up = tokens[1].get('token_id')

        # Get opening price (from market metadata or first Chainlink reading)
        opening_price_data = self.chainlink.get_latest_price()
        if not opening_price_data:
            print("Error: Could not fetch opening BTC price")
            return

        opening_price = opening_price_data['price']
        print(f"Opening BTC Price: ${opening_price:.2f}\n")

        # Reset price history
        self.price_history = [opening_price]
        self.timestamp_history = [time.time()]

        # Monitoring loop
        opportunities = []
        start_time = time.time()
        end_time = start_time + monitor_duration

        print(f"{'Time':<12} {'BTC Price':<12} {'Up':<8} {'Down':<8} {'P(Up)':<8} {'Edge':<8} {'Action':<30}")
        print("-" * 100)

        while time.time() < end_time:
            timestamp = datetime.utcnow()
            unix_time = time.time()
            seconds_remaining = end_time - unix_time

            # Get current BTC price
            btc_data = self.chainlink.get_latest_price()
            if not btc_data:
                time.sleep(0.5)
                continue

            current_btc_price = btc_data['price']

            # Update price history
            self.price_history.append(current_btc_price)
            self.timestamp_history.append(unix_time)

            # Keep only recent history (last 60 seconds)
            cutoff_time = unix_time - 60
            while len(self.timestamp_history) > 1 and self.timestamp_history[0] < cutoff_time:
                self.price_history.pop(0)
                self.timestamp_history.pop(0)

            # Estimate volatility and drift
            volatility = self.model.estimate_volatility(self.price_history)
            drift = self.model.estimate_drift(self.price_history, self.timestamp_history)

            # Get order books
            book_up = self.polymarket.get_order_book(token_id_up)
            book_down = self.polymarket.get_order_book(token_id_down)

            if not book_up or not book_down:
                time.sleep(0.5)
                continue

            market_price_up = book_up.get('mid_price', 0)
            market_price_down = book_down.get('mid_price', 0)

            # Bayesian estimate
            estimate = self.model.estimate_probability_up(
                current_btc_price=current_btc_price,
                opening_btc_price=opening_price,
                seconds_remaining=seconds_remaining,
                recent_volatility=volatility,
                recent_drift=drift
            )

            # Edge calculation
            edge = self.model.calculate_edge(
                bayesian_prob_up=estimate['prob_up'],
                market_price_up=market_price_up,
                market_price_down=market_price_down
            )

            # Display current state
            time_str = f"{seconds_remaining:.1f}s"
            btc_str = f"${current_btc_price:.2f}"
            up_str = f"{market_price_up:.3f}"
            down_str = f"{market_price_down:.3f}"
            prob_str = f"{estimate['prob_up']:.3f}"

            # Determine if we have edge
            max_edge = max(edge['edge_up'], edge['edge_down'])
            edge_str = f"{max_edge*100:+.2f}%"

            action = edge['recommended_action']

            # Color coding for significant edge (just text, no actual colors in basic terminal)
            if max_edge > self.min_edge_threshold:
                marker = "🔥 "
            else:
                marker = "   "

            print(f"{marker}{time_str:<12} {btc_str:<12} {up_str:<8} {down_str:<8} {prob_str:<8} {edge_str:<8} {action:<30}")

            # Record opportunity
            if max_edge > self.min_edge_threshold:
                opportunity = {
                    'timestamp': timestamp.isoformat(),
                    'seconds_remaining': seconds_remaining,
                    'btc_price': current_btc_price,
                    'opening_price': opening_price,
                    'market_price_up': market_price_up,
                    'market_price_down': market_price_down,
                    'bayesian_prob_up': estimate['prob_up'],
                    'edge_up': edge['edge_up'],
                    'edge_down': edge['edge_down'],
                    'ev_up': edge['ev_bet_up'],
                    'ev_down': edge['ev_bet_down'],
                    'action': action
                }
                opportunities.append(opportunity)

            # Sample every 1 second
            time.sleep(1.0)

        print("\n" + "="*100)
        print(f"Monitoring complete. Found {len(opportunities)} opportunities with edge > {self.min_edge_threshold*100:.1f}%")

        if opportunities:
            print("\nOpportunities detected:")
            for opp in opportunities:
                print(f"  [{opp['seconds_remaining']:.1f}s remaining] {opp['action']}")

            # Save opportunities
            filename = f"opportunities_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = self.output_dir / filename

            with open(filepath, 'w') as f:
                json.dump({
                    'market': market,
                    'opening_price': opening_price,
                    'opportunities': opportunities
                }, f, indent=2)

            print(f"\nSaved opportunities to: {filepath}")

    def run_continuous(self, check_interval: int = 30):
        """
        Run continuous monitoring, checking for new markets

        Args:
            check_interval: Seconds between market checks
        """
        print("Starting continuous market monitoring...")
        print(f"Checking for new markets every {check_interval} seconds")
        print("Press Ctrl+C to stop\n")

        try:
            while True:
                market = self.find_active_market()

                if market:
                    # Found an active market
                    time_remaining = market.get('time_remaining', 0)

                    # If less than 60 seconds remain, start monitoring
                    if time_remaining < 60:
                        self.monitor_market_final_seconds(
                            market,
                            monitor_duration=min(30, int(time_remaining))
                        )
                    else:
                        print(f"Market found but {time_remaining:.0f}s remaining, waiting...")
                        time.sleep(check_interval)
                else:
                    print(f"No active markets found. Checking again in {check_interval}s...")
                    time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\n\nMonitoring stopped by user.")


def main():
    """Main entry point"""
    print("=== Polymarket BTC Live Market Monitor ===\n")

    # Load configuration from environment
    rpc_url = os.getenv('POLYGON_RPC_URL', 'https://polygon-rpc.com')
    min_edge = float(os.getenv('MIN_EDGE_THRESHOLD', '0.05'))

    # Initialize monitor
    monitor = LiveMarketMonitor(
        rpc_url=rpc_url,
        min_edge_threshold=min_edge
    )

    print("Choose mode:")
    print("1. Monitor next available market (one-shot)")
    print("2. Continuous monitoring (checks for new markets)")

    try:
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == '2':
            monitor.run_continuous()
        else:
            # One-shot mode
            print("\nSearching for active market...")
            market = monitor.find_active_market()

            if market:
                time_remaining = market.get('time_remaining', 0)
                monitor_duration = min(30, int(time_remaining))

                monitor.monitor_market_final_seconds(market, monitor_duration)
            else:
                print("No active markets found.")
                print("Markets run every 5 minutes. Try again later.")

    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
