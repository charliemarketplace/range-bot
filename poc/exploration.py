"""
Exploration script to understand the swaps.db schema and data.
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "swaps.db"


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Access columns by name

    print("=" * 70)
    print("SCHEMA")
    print("=" * 70)

    # Get table info
    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()

    for table in tables:
        table_name = table[0]
        print(f"\n{table_name}:")
        columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        for col in columns:
            print(f"  {col[1]:<20} {col[2]}")

    print("\n" + "=" * 70)
    print("SAMPLE ROWS (first 5 swaps)")
    print("=" * 70)

    rows = conn.execute("SELECT * FROM swaps LIMIT 5").fetchall()
    for i, row in enumerate(rows):
        print(f"\n--- Swap {i+1} ---")
        for key in row.keys():
            print(f"  {key:<20} {row[key]}")

    print("\n" + "=" * 70)
    print("BASIC STATS")
    print("=" * 70)

    # Count and block range
    stats = conn.execute("""
        SELECT
            COUNT(*) as total_swaps,
            MIN(block_number) as min_block,
            MAX(block_number) as max_block,
            MIN(price) as min_price,
            MAX(price) as max_price,
            AVG(price) as avg_price
        FROM swaps
    """).fetchone()

    print(f"\nTotal swaps:    {stats['total_swaps']:,}")
    print(f"Block range:    {stats['min_block']:,} - {stats['max_block']:,}")
    print(f"Price range:    ${stats['min_price']:,.2f} - ${stats['max_price']:,.2f}")
    print(f"Avg price:      ${stats['avg_price']:,.2f}")

    # Swaps per block stats
    swaps_per_block = conn.execute("""
        SELECT
            AVG(cnt) as avg_per_block,
            MAX(cnt) as max_per_block
        FROM (
            SELECT block_number, COUNT(*) as cnt
            FROM swaps
            GROUP BY block_number
        )
    """).fetchone()

    print(f"\nAvg swaps/block: {swaps_per_block['avg_per_block']:.2f}")
    print(f"Max swaps/block: {swaps_per_block['max_per_block']}")

    print("\n" + "=" * 70)
    print("VOLUME ANALYSIS")
    print("=" * 70)

    # Volume calculations (amount0 is USDC with 6 decimals)
    # Positive amount0 = USDC in (buying ETH), Negative = USDC out (selling ETH)
    volume = conn.execute("""
        SELECT
            SUM(ABS(CAST(amount0 AS REAL))) / 1e6 as total_usdc_volume,
            SUM(ABS(CAST(amount1 AS REAL))) / 1e18 as total_eth_volume,
            SUM(CASE WHEN CAST(amount0 AS REAL) > 0 THEN CAST(amount0 AS REAL) ELSE 0 END) / 1e6 as usdc_in,
            SUM(CASE WHEN CAST(amount0 AS REAL) < 0 THEN ABS(CAST(amount0 AS REAL)) ELSE 0 END) / 1e6 as usdc_out
        FROM swaps
    """).fetchone()

    print(f"\nTotal USDC volume: ${volume['total_usdc_volume']:,.0f}")
    print(f"Total ETH volume:  {volume['total_eth_volume']:,.2f} ETH")
    print(f"USDC in (buys):    ${volume['usdc_in']:,.0f}")
    print(f"USDC out (sells):  ${volume['usdc_out']:,.0f}")

    print("\n" + "=" * 70)
    print("PRICE DISTRIBUTION (by 1000-block buckets)")
    print("=" * 70)

    # Sample price over time (every 100k blocks)
    buckets = conn.execute("""
        SELECT
            (block_number / 100000) * 100000 as block_bucket,
            COUNT(*) as swap_count,
            AVG(price) as avg_price,
            MIN(price) as min_price,
            MAX(price) as max_price
        FROM swaps
        GROUP BY block_bucket
        ORDER BY block_bucket
    """).fetchall()

    print(f"\n{'Block Range':<25} {'Swaps':>10} {'Avg Price':>12} {'Range':>25}")
    print("-" * 75)
    for b in buckets:
        block_start = b['block_bucket']
        block_end = block_start + 99999
        price_range = f"${b['min_price']:,.0f} - ${b['max_price']:,.0f}"
        print(f"{block_start:,}-{block_end:,}  {b['swap_count']:>10,} ${b['avg_price']:>10,.2f} {price_range:>25}")

    print("\n" + "=" * 70)
    print("TICK DISTRIBUTION")
    print("=" * 70)

    # Tick stats (tick represents the price range in Uniswap v3)
    tick_stats = conn.execute("""
        SELECT
            MIN(tick) as min_tick,
            MAX(tick) as max_tick,
            AVG(tick) as avg_tick
        FROM swaps
    """).fetchone()

    print(f"\nTick range: {tick_stats['min_tick']:,} to {tick_stats['max_tick']:,}")
    print(f"Avg tick:   {tick_stats['avg_tick']:,.0f}")

    # Explain tick -> price relationship
    # For ETH/USDC: price = 1.0001^tick * (10^12 / 10^18) = 1.0001^tick * 10^-6
    # But we store computed price already, so just show correlation
    sample = conn.execute("""
        SELECT tick, price FROM swaps
        ORDER BY block_number
        LIMIT 5
    """).fetchall()

    print(f"\nTick vs Price samples:")
    for s in sample:
        print(f"  tick {s['tick']:>7} -> ${s['price']:,.2f}")

    conn.close()
    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
