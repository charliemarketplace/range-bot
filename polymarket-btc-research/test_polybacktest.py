#!/usr/bin/env python3
"""
Test PolyBackTest API connection and fetch sample data.
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from polybacktest_client import PolyBackTestClient
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_api_connection():
    """Test if PolyBackTest API is accessible."""
    print("=" * 80)
    print("TESTING POLYBACKTEST API CONNECTION")
    print("=" * 80)

    client = PolyBackTestClient()

    # Test 1: Try to fetch markets without API key
    print("\n[TEST 1] Fetching markets (checking if API key is required)...")
    try:
        markets = client.get_markets(limit=3)
        print(f"✅ SUCCESS: Fetched {len(markets)} markets")

        if markets:
            print("\nSample market:")
            m = markets[0]
            print(f"  Question: {m.question}")
            print(f"  Type: {m.market_type}")
            print(f"  Ends: {m.end_time}")
            print(f"  Threshold: ${m.btc_threshold:,.2f}")
            print(f"  Resolved: {m.resolved}")

        return True

    except ValueError as e:
        if "Invalid API key" in str(e):
            print("❌ API key required")
            print("\nTo get an API key:")
            print("  1. Go to https://polybacktest.com")
            print("  2. Sign up for free account")
            print("  3. Get your API key")
            print("  4. Add to .env: POLYBACKTEST_API_KEY=your_key_here")
            return False
        else:
            raise

    except Exception as e:
        print(f"❌ ERROR: {type(e).__name__}: {e}")
        print("\nPossible issues:")
        print("  - PolyBackTest API might not exist yet")
        print("  - API endpoint structure might be different")
        print("  - Service might be down")
        print("\nTrying alternative: Check if polybacktest.com is real...")
        return False


def test_alternative_sources():
    """Test alternative data sources if PolyBackTest doesn't work."""
    print("\n" + "=" * 80)
    print("TESTING ALTERNATIVE DATA SOURCES")
    print("=" * 80)

    import requests

    # Test 1: Check if polybacktest.com exists
    print("\n[TEST] Checking if polybacktest.com is accessible...")
    try:
        response = requests.get("https://polybacktest.com", timeout=10)
        print(f"✅ Website exists (status {response.status_code})")
    except Exception as e:
        print(f"❌ Website not accessible: {e}")
        print("\n⚠️  PolyBackTest might not be publicly available yet")
        print("    This was mentioned in search results but API may not be live")

    # Test 2: Try PolymarketData.co alternative
    print("\n[TEST] Checking polymarketdata.co alternative...")
    try:
        response = requests.get("https://www.polymarketdata.co", timeout=10)
        print(f"✅ PolymarketData.co exists (status {response.status_code})")
        print("    This could be an alternative data source")
    except Exception as e:
        print(f"❌ Not accessible: {e}")

    # Test 3: Check regular Polymarket API (we know this works)
    print("\n[TEST] Verifying standard Polymarket API works...")
    try:
        response = requests.get(
            "https://gamma-api.polymarket.com/markets?limit=1",
            timeout=10
        )
        data = response.json()
        print(f"✅ Standard Polymarket API works ({len(data)} markets)")
        print("    We can collect data ourselves if needed")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run all tests."""
    api_works = test_api_connection()

    if not api_works:
        test_alternative_sources()

        print("\n" + "=" * 80)
        print("RECOMMENDATION")
        print("=" * 80)
        print("""
If PolyBackTest API is not available:

OPTION 1: Collect our own data (Free, but takes time)
  - Use standard Polymarket APIs (CLOB + Gamma)
  - Run data collector for 1-2 weeks
  - Build our own historical dataset

OPTION 2: Use mock/synthetic data for testing (Immediate)
  - Generate realistic order book snapshots
  - Test backtesting system logic
  - Validate edge detection algorithm

OPTION 3: Contact PolyBackTest directly
  - Email them about API access
  - They might be in beta
  - Could provide early access

For now, let's proceed with OPTION 2 (synthetic data) to test
the backtesting system, then move to OPTION 1 (collect real data).
        """)

    else:
        print("\n" + "=" * 80)
        print("NEXT STEPS")
        print("=" * 80)
        print("""
✅ PolyBackTest API is working!

Next steps:
  1. Download 30 days of 5m market data
  2. Integrate into backtesting system
  3. Run backtests and validate model
  4. Generate performance reports
        """)


if __name__ == "__main__":
    main()
