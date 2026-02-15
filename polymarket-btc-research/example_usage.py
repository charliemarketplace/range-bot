#!/usr/bin/env python3
"""
Example Usage
Demonstrates the complete workflow for Polymarket BTC research.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from polymarket_client import PolymarketClient
from chainlink_fetcher import ChainlinkFetcher
from bayesian_model import BayesianBTCModel
import time


def example_1_basic_queries():
    """Example 1: Basic API queries"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Basic Queries")
    print("="*60 + "\n")

    # Initialize clients
    poly = PolymarketClient()
    chain = ChainlinkFetcher()

    # Get current BTC price from Chainlink
    print("1. Fetching BTC price from Chainlink...")
    btc_data = chain.get_latest_price()
    if btc_data:
        print(f"   BTC Price: ${btc_data['price']:.2f}")
        print(f"   Updated: {btc_data['updated_at_iso']}")
        print(f"   Round ID: {btc_data['round_id']}")

    # Search for BTC markets
    print("\n2. Searching for BTC 5-minute markets...")
    markets = poly.search_btc_markets(limit=50)
    print(f"   Found {len(markets)} BTC-related markets")

    if markets:
        print("\n   Sample market:")
        market = markets[0]
        print(f"   Question: {market.get('question', 'N/A')}")
        print(f"   Active: {market.get('active', 'N/A')}")
        print(f"   End time: {market.get('end_date_iso', 'N/A')}")


def example_2_bayesian_model():
    """Example 2: Bayesian probability estimation"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Bayesian Model")
    print("="*60 + "\n")

    model = BayesianBTCModel()

    # Scenario 1: Price is up, 15 seconds left
    print("Scenario 1: BTC up $50, 15 seconds remaining")
    print("-" * 40)
    estimate1 = model.estimate_probability_up(
        current_btc_price=95050.0,
        opening_btc_price=95000.0,
        seconds_remaining=15
    )
    print(f"P(Up):   {estimate1['prob_up']:.4f} ({estimate1['prob_up']*100:.2f}%)")
    print(f"P(Down): {estimate1['prob_down']:.4f} ({estimate1['prob_down']*100:.2f}%)")

    # Scenario 2: Price is down, 10 seconds left
    print("\nScenario 2: BTC down $30, 10 seconds remaining")
    print("-" * 40)
    estimate2 = model.estimate_probability_up(
        current_btc_price=94970.0,
        opening_btc_price=95000.0,
        seconds_remaining=10
    )
    print(f"P(Up):   {estimate2['prob_up']:.4f} ({estimate2['prob_up']*100:.2f}%)")
    print(f"P(Down): {estimate2['prob_down']:.4f} ({estimate2['prob_down']*100:.2f}%)")

    # Scenario 3: Price exactly at open, 5 seconds left
    print("\nScenario 3: BTC exactly at open, 5 seconds remaining")
    print("-" * 40)
    estimate3 = model.estimate_probability_up(
        current_btc_price=95000.0,
        opening_btc_price=95000.0,
        seconds_remaining=5
    )
    print(f"P(Up):   {estimate3['prob_up']:.4f} ({estimate3['prob_up']*100:.2f}%)")
    print(f"P(Down): {estimate3['prob_down']:.4f} ({estimate3['prob_down']*100:.2f}%)")


def example_3_edge_detection():
    """Example 3: Edge detection and trading signals"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Edge Detection")
    print("="*60 + "\n")

    model = BayesianBTCModel()

    # Scenario: BTC up $50, 10 seconds left
    opening_price = 95000.0
    current_price = 95050.0
    seconds_remaining = 10

    print(f"Market situation:")
    print(f"  Opening: ${opening_price:.2f}")
    print(f"  Current: ${current_price:.2f}")
    print(f"  Change: +${current_price - opening_price:.2f}")
    print(f"  Time left: {seconds_remaining}s")

    # Bayesian estimate
    estimate = model.estimate_probability_up(
        current_btc_price=current_price,
        opening_btc_price=opening_price,
        seconds_remaining=seconds_remaining
    )

    print(f"\nBayesian Estimate:")
    print(f"  P(Up): {estimate['prob_up']:.4f}")

    # Test different market scenarios
    scenarios = [
        {"name": "Efficient Market", "up": 0.65, "down": 0.35},
        {"name": "Underpriced Up", "up": 0.55, "down": 0.45},
        {"name": "Overpriced Up", "up": 0.75, "down": 0.25},
        {"name": "Inefficient (sum > 1)", "up": 0.60, "down": 0.50},
    ]

    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        print(f"  Market: Up={scenario['up']:.2f}, Down={scenario['down']:.2f}, Sum={scenario['up']+scenario['down']:.2f}")

        edge = model.calculate_edge(
            bayesian_prob_up=estimate['prob_up'],
            market_price_up=scenario['up'],
            market_price_down=scenario['down']
        )

        print(f"  Edge(Up): {edge['edge_up_pct']:+.2f}%")
        print(f"  Edge(Down): {edge['edge_down_pct']:+.2f}%")
        print(f"  EV(Up): {edge['ev_bet_up']:+.4f}")
        print(f"  EV(Down): {edge['ev_bet_down']:+.4f}")
        print(f"  → {edge['recommended_action']}")


def example_4_order_book():
    """Example 4: Real order book data"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Order Book Analysis")
    print("="*60 + "\n")

    poly = PolymarketClient()

    print("Searching for an active market with tokens...")
    markets = poly.search_btc_markets(limit=100)

    for market in markets:
        tokens = market.get('tokens', [])
        if len(tokens) >= 2 and market.get('active'):
            print(f"\nMarket: {market.get('question', 'N/A')}")

            token_0 = tokens[0].get('token_id')
            token_1 = tokens[1].get('token_id')

            if token_0:
                print(f"\nToken 0 ({tokens[0].get('outcome', 'Unknown')}):")
                book = poly.get_order_book(token_0)
                if book:
                    print(f"  Best Bid: {book.get('best_bid', 'N/A')}")
                    print(f"  Best Ask: {book.get('best_ask', 'N/A')}")
                    print(f"  Spread: {book.get('spread', 'N/A')}")
                    print(f"  Mid Price: {book.get('mid_price', 'N/A')}")

                    # Show top 3 bids/asks
                    if book.get('bids'):
                        print(f"  Top 3 Bids:")
                        for i, bid in enumerate(book['bids'][:3], 1):
                            print(f"    {i}. {bid.get('price')} @ {bid.get('size')}")

            # Only show one market as example
            break
    else:
        print("No active markets with tokens found.")


def example_5_price_monitoring():
    """Example 5: Monitor price updates"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Price Monitoring (30 seconds)")
    print("="*60 + "\n")

    chain = ChainlinkFetcher()

    print("Monitoring Chainlink BTC price for updates...")
    print("Note: Chainlink updates occur every 1-5 minutes on average\n")

    updates = chain.monitor_price_updates(duration_seconds=30, check_interval=2.0)

    if updates:
        print(f"\nDetected {len(updates)} price updates:")
        for update in updates:
            print(f"  {update['updated_at_iso']}: ${update['price']:.2f} (Round {update['round_id']})")
    else:
        print("\nNo price updates detected (this is normal - updates are infrequent)")

    # Show current price
    current = chain.get_latest_price()
    if current:
        print(f"\nCurrent BTC Price: ${current['price']:.2f}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" Polymarket BTC 5-Minute Markets - Example Usage")
    print("="*70)

    examples = [
        ("Basic Queries", example_1_basic_queries),
        ("Bayesian Model", example_2_bayesian_model),
        ("Edge Detection", example_3_edge_detection),
        ("Order Book", example_4_order_book),
        ("Price Monitoring", example_5_price_monitoring),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    print(f"  {len(examples)+1}. Run all examples")

    try:
        choice = input("\nEnter choice (1-6): ").strip()

        if choice == str(len(examples)+1):
            # Run all
            for name, func in examples:
                try:
                    func()
                    time.sleep(1)
                except Exception as e:
                    print(f"\nError in {name}: {e}")
                    import traceback
                    traceback.print_exc()
        elif choice.isdigit() and 1 <= int(choice) <= len(examples):
            # Run selected example
            name, func = examples[int(choice)-1]
            try:
                func()
            except Exception as e:
                print(f"\nError: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("Invalid choice")

        print("\n" + "="*70)
        print("Example complete!")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == "__main__":
    main()
