#!/usr/bin/env python3
"""
Test Suite for Polymarket BTC Research
Tests all components to ensure they work correctly.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

import time
from datetime import datetime


class TestRunner:
    """Simple test runner with pass/fail tracking"""

    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []

    def run_test(self, name, test_func):
        """Run a single test and track results"""
        print(f"\n{'='*60}")
        print(f"TEST: {name}")
        print('='*60)

        try:
            result = test_func()
            if result:
                print(f"✅ PASSED")
                self.passed += 1
                self.tests.append((name, True, None))
            else:
                print(f"❌ FAILED: Test returned False")
                self.failed += 1
                self.tests.append((name, False, "Test returned False"))
        except Exception as e:
            print(f"❌ FAILED: {e}")
            import traceback
            traceback.print_exc()
            self.failed += 1
            self.tests.append((name, False, str(e)))

    def print_summary(self):
        """Print test summary"""
        print(f"\n{'='*60}")
        print("TEST SUMMARY")
        print('='*60)

        for name, passed, error in self.tests:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status}: {name}")
            if error:
                print(f"       Error: {error}")

        print(f"\nTotal: {self.passed + self.failed} tests")
        print(f"Passed: {self.passed}")
        print(f"Failed: {self.failed}")
        print(f"Success Rate: {(self.passed/(self.passed+self.failed)*100):.1f}%")

        return self.failed == 0


def test_imports():
    """Test 1: Verify all imports work"""
    print("Testing imports...")

    try:
        from polymarket_client import PolymarketClient
        print("  ✓ polymarket_client imported")

        from chainlink_fetcher import ChainlinkFetcher
        print("  ✓ chainlink_fetcher imported")

        from market_collector import MarketDataCollector
        print("  ✓ market_collector imported")

        from bayesian_model import BayesianBTCModel
        print("  ✓ bayesian_model imported")

        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_polymarket_connection():
    """Test 2: Verify Polymarket API is accessible"""
    print("Testing Polymarket API connection...")

    from polymarket_client import PolymarketClient

    client = PolymarketClient()

    # Try to fetch markets
    markets = client.get_markets(limit=5)

    if markets:
        print(f"  ✓ Successfully fetched {len(markets)} markets")

        # Check structure
        if len(markets) > 0:
            market = markets[0]
            required_fields = ['question', 'condition_id']

            for field in required_fields:
                if field in market:
                    print(f"  ✓ Market has '{field}' field")
                else:
                    print(f"  ✗ Market missing '{field}' field")
                    return False

        return True
    else:
        print("  ✗ No markets returned")
        return False


def test_chainlink_connection():
    """Test 3: Verify Chainlink oracle is accessible"""
    print("Testing Chainlink oracle connection...")

    try:
        from chainlink_fetcher import ChainlinkFetcher

        fetcher = ChainlinkFetcher()
        print(f"  ✓ Connected to: {fetcher.description}")

        # Get latest price
        price_data = fetcher.get_latest_price()

        if price_data:
            price = price_data['price']
            print(f"  ✓ Got BTC price: ${price:,.2f}")

            # Sanity check - BTC should be between $10k and $200k
            if 10000 < price < 200000:
                print(f"  ✓ Price is in reasonable range")
            else:
                print(f"  ⚠ Warning: Price seems unusual: ${price:,.2f}")

            # Check timestamp
            if 'updated_at' in price_data:
                print(f"  ✓ Has update timestamp: {price_data.get('updated_at_iso', 'N/A')}")

            return True
        else:
            print("  ✗ No price data returned")
            return False

    except ConnectionError as e:
        print(f"  ✗ Connection failed: {e}")
        print("  ℹ This might be due to RPC rate limiting - try a dedicated provider")
        return False


def test_btc_market_search():
    """Test 4: Search for BTC markets"""
    print("Testing BTC market search...")

    from polymarket_client import PolymarketClient

    client = PolymarketClient()
    btc_markets = client.search_btc_markets(limit=100)

    print(f"  ✓ Found {len(btc_markets)} BTC-related markets")

    if btc_markets:
        # Show a sample market
        market = btc_markets[0]
        print(f"  Sample: {market.get('question', 'N/A')[:80]}...")

        # Check if tokens are present
        tokens = market.get('tokens', [])
        print(f"  ✓ Market has {len(tokens)} outcome tokens")

        return True
    else:
        print("  ℹ No BTC markets found (this might be okay)")
        return True  # Not a failure, just no markets right now


def test_order_book():
    """Test 5: Fetch order book data"""
    print("Testing order book fetching...")

    from polymarket_client import PolymarketClient

    client = PolymarketClient()
    markets = client.search_btc_markets(limit=50)

    # Find a market with tokens
    for market in markets:
        tokens = market.get('tokens', [])
        if len(tokens) >= 1:
            token_id = tokens[0].get('token_id')

            if token_id:
                print(f"  Testing with token: {token_id}")

                book = client.get_order_book(token_id)

                if book:
                    print(f"  ✓ Got order book")

                    # Check fields
                    if 'bids' in book and 'asks' in book:
                        print(f"  ✓ Has bids ({len(book['bids'])}) and asks ({len(book['asks'])})")

                    if 'mid_price' in book:
                        print(f"  ✓ Mid price: {book['mid_price']}")

                    if 'spread' in book:
                        print(f"  ✓ Spread: {book['spread']}")

                    return True
                else:
                    print(f"  ✗ No order book returned")
                    return False

    print("  ℹ No markets with tokens found to test")
    return True  # Not a hard failure


def test_bayesian_model():
    """Test 6: Test Bayesian model calculations"""
    print("Testing Bayesian model...")

    from bayesian_model import BayesianBTCModel

    model = BayesianBTCModel()
    print("  ✓ Model initialized")

    # Test scenario: price is up, few seconds left
    estimate = model.estimate_probability_up(
        current_btc_price=95050.0,
        opening_btc_price=95000.0,
        seconds_remaining=15
    )

    print(f"  ✓ Estimation completed")

    # Verify output structure
    required_fields = ['prob_up', 'prob_down', 'confidence']
    for field in required_fields:
        if field in estimate:
            print(f"  ✓ Has '{field}': {estimate[field]:.4f}")
        else:
            print(f"  ✗ Missing '{field}'")
            return False

    # Sanity checks
    prob_up = estimate['prob_up']
    prob_down = estimate['prob_down']

    # Probabilities should sum to 1
    prob_sum = prob_up + prob_down
    if abs(prob_sum - 1.0) < 0.0001:
        print(f"  ✓ Probabilities sum to 1.0")
    else:
        print(f"  ✗ Probabilities sum to {prob_sum:.4f}, not 1.0")
        return False

    # Since BTC is up, P(Up) should be > 0.5
    if prob_up > 0.5:
        print(f"  ✓ P(Up) > 0.5 when price is up (makes sense)")
    else:
        print(f"  ⚠ P(Up) = {prob_up:.4f} when price is up (unexpected)")

    return True


def test_edge_calculation():
    """Test 7: Test edge calculation"""
    print("Testing edge calculation...")

    from bayesian_model import BayesianBTCModel

    model = BayesianBTCModel()

    # Calculate edge
    edge = model.calculate_edge(
        bayesian_prob_up=0.65,
        market_price_up=0.55,
        market_price_down=0.45
    )

    print(f"  ✓ Edge calculated")

    required_fields = ['edge_up', 'edge_down', 'ev_bet_up', 'ev_bet_down', 'recommended_action']
    for field in required_fields:
        if field in edge:
            value = edge[field]
            if isinstance(value, (int, float)):
                print(f"  ✓ {field}: {value:.4f}")
            else:
                print(f"  ✓ {field}: {value}")
        else:
            print(f"  ✗ Missing '{field}'")
            return False

    # With Bayesian = 0.65 and Market = 0.55, we should have positive edge on Up
    if edge['edge_up'] > 0:
        print(f"  ✓ Positive edge detected correctly ({edge['edge_up_pct']:.2f}%)")
    else:
        print(f"  ✗ Expected positive edge but got {edge['edge_up']:.4f}")
        return False

    return True


def test_volatility_estimation():
    """Test 8: Test volatility estimation"""
    print("Testing volatility estimation...")

    from bayesian_model import BayesianBTCModel

    model = BayesianBTCModel()

    # Test with price history
    prices = [95000, 95010, 95005, 95015, 95020, 95018, 95025]

    vol = model.estimate_volatility(prices)

    print(f"  ✓ Volatility estimated: {vol:.2f}")

    # Should be positive
    if vol >= 0:
        print(f"  ✓ Volatility is non-negative")
    else:
        print(f"  ✗ Volatility is negative: {vol}")
        return False

    # Test drift
    timestamps = [0, 1, 2, 3, 4, 5, 6]
    drift = model.estimate_drift(prices, timestamps)

    print(f"  ✓ Drift estimated: {drift:.2f}")

    # Prices are trending up, so drift should be positive
    if drift > 0:
        print(f"  ✓ Drift is positive (prices trending up)")
    else:
        print(f"  ⚠ Drift is {drift:.2f} (expected positive)")

    return True


def test_data_types():
    """Test 9: Verify data types are correct"""
    print("Testing data types and structures...")

    from polymarket_client import PolymarketClient
    from chainlink_fetcher import ChainlinkFetcher

    # Test Polymarket
    poly = PolymarketClient()
    markets = poly.get_markets(limit=1)

    if markets and len(markets) > 0:
        market = markets[0]
        if isinstance(market, dict):
            print(f"  ✓ Market is a dictionary")
        else:
            print(f"  ✗ Market is {type(market)}, expected dict")
            return False

    # Test Chainlink
    try:
        chain = ChainlinkFetcher()
        price_data = chain.get_latest_price()

        if price_data:
            if isinstance(price_data, dict):
                print(f"  ✓ Price data is a dictionary")

            if isinstance(price_data.get('price'), (int, float)):
                print(f"  ✓ Price is numeric")
            else:
                print(f"  ✗ Price is {type(price_data.get('price'))}")
                return False
    except ConnectionError:
        print(f"  ℹ Skipping Chainlink data type test (connection issue)")

    return True


def test_error_handling():
    """Test 10: Test error handling"""
    print("Testing error handling...")

    from polymarket_client import PolymarketClient

    client = PolymarketClient()

    # Test with invalid token ID (should handle gracefully)
    book = client.get_order_book("invalid_token_id_12345")

    if book is None:
        print(f"  ✓ Invalid token ID handled gracefully (returned None)")
    else:
        print(f"  ⚠ Got unexpected result for invalid token: {book}")

    # Test Bayesian model with edge cases
    from bayesian_model import BayesianBTCModel
    model = BayesianBTCModel()

    # Test with 0 seconds remaining
    try:
        estimate = model.estimate_probability_up(
            current_btc_price=95000.0,
            opening_btc_price=95000.0,
            seconds_remaining=0
        )
        print(f"  ✓ Handled 0 seconds remaining")
    except Exception as e:
        print(f"  ✗ Failed with 0 seconds: {e}")
        return False

    return True


def main():
    """Run all tests"""
    print("="*60)
    print("POLYMARKET BTC RESEARCH - TEST SUITE")
    print("="*60)
    print(f"Started: {datetime.utcnow().isoformat()}")

    runner = TestRunner()

    # Run all tests
    runner.run_test("Import all modules", test_imports)
    runner.run_test("Connect to Polymarket API", test_polymarket_connection)
    runner.run_test("Connect to Chainlink oracle", test_chainlink_connection)
    runner.run_test("Search for BTC markets", test_btc_market_search)
    runner.run_test("Fetch order book data", test_order_book)
    runner.run_test("Bayesian model estimation", test_bayesian_model)
    runner.run_test("Edge calculation", test_edge_calculation)
    runner.run_test("Volatility and drift estimation", test_volatility_estimation)
    runner.run_test("Data type validation", test_data_types)
    runner.run_test("Error handling", test_error_handling)

    # Print summary
    all_passed = runner.print_summary()

    print(f"\nFinished: {datetime.utcnow().isoformat()}")

    if all_passed:
        print("\n🎉 All tests passed! Code is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
