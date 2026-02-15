"""
Batch fetcher for historical swap data.
Respectful rate limiting, progress tracking, SQLite storage.

Usage:
    uv run python poc/batch_fetcher.py
"""
import requests
import sqlite3
import time
import sys
from pathlib import Path
from datetime import datetime

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# Config
RPC_ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://eth.drpc.org",
]
POOL = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"  # ETH/USDC 0.05%
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"
DATA_DIR = Path(__file__).parent / "data"

# Batch settings
START_BLOCK = 23_000_000
END_BLOCK = 24_000_000
CHUNK_SIZE = 2_000  # blocks per request
RATE_LIMIT_DELAY = 1.0  # 1 second between requests
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
BACKOFF_BASE = 2


def get_db():
    """Initialize SQLite database with schema."""
    db_path = DATA_DIR / "swaps.db"
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS swaps (
            id TEXT PRIMARY KEY,
            block_number INTEGER NOT NULL,
            tx_hash TEXT NOT NULL,
            log_index INTEGER NOT NULL,
            amount0 TEXT NOT NULL,
            amount1 TEXT NOT NULL,
            sqrt_price_x96 TEXT NOT NULL,
            liquidity TEXT NOT NULL,
            tick INTEGER NOT NULL,
            price REAL NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_swaps_block ON swaps(block_number)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_progress (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            last_block INTEGER NOT NULL,
            total_swaps INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def rpc_call(endpoint: str, method: str, params: list) -> dict:
    """Make RPC call with timeout."""
    r = requests.post(
        endpoint,
        json={"jsonrpc": "2.0", "method": method, "params": params, "id": 1},
        timeout=REQUEST_TIMEOUT
    )
    return r.json()


def get_logs_with_retry(from_block: int, to_block: int) -> tuple[list, float]:
    """Fetch logs with exponential backoff and endpoint rotation."""
    last_error = None

    for attempt in range(MAX_RETRIES):
        endpoint = RPC_ENDPOINTS[attempt % len(RPC_ENDPOINTS)]
        try:
            start = time.time()
            result = rpc_call(endpoint, "eth_getLogs", [{
                "address": POOL,
                "topics": [SWAP_TOPIC],
                "fromBlock": hex(from_block),
                "toBlock": hex(to_block)
            }])
            elapsed = time.time() - start

            if "error" in result:
                error_msg = result["error"].get("message", str(result["error"]))
                last_error = error_msg
                if "rate" in error_msg.lower() or "limit" in error_msg.lower():
                    wait = BACKOFF_BASE ** attempt
                    print(f"  Rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                # Other RPC error, try next endpoint
                time.sleep(1)
                continue

            return result.get("result", []), elapsed

        except requests.exceptions.Timeout:
            last_error = f"Timeout on {endpoint}"
            print(f"  {last_error}, attempt {attempt + 1}/{MAX_RETRIES}")
            time.sleep(BACKOFF_BASE ** attempt)
        except requests.exceptions.RequestException as e:
            last_error = str(e)
            print(f"  Request error: {e}, attempt {attempt + 1}/{MAX_RETRIES}")
            time.sleep(BACKOFF_BASE ** attempt)

    raise Exception(f"Failed after {MAX_RETRIES} attempts: {last_error}")


def decode_swap(log: dict) -> dict:
    """Decode swap event from log."""
    data = log["data"][2:]

    def int256(h):
        v = int(h, 16)
        return v - 2**256 if v >= 2**255 else v

    def int24(h):
        v = int(h, 16)
        return v - 2**24 if v >= 2**23 else v

    sqrt_price = int(data[128:192], 16)
    price = 1e12 / ((sqrt_price ** 2) / (2 ** 192)) if sqrt_price > 0 else 0

    return {
        "id": f"{log['transactionHash']}-{log['logIndex']}",
        "block_number": int(log["blockNumber"], 16),
        "tx_hash": log["transactionHash"],
        "log_index": int(log["logIndex"], 16),
        "amount0": str(int256(data[0:64])),
        "amount1": str(int256(data[64:128])),
        "sqrt_price_x96": str(sqrt_price),
        "liquidity": str(int(data[192:256], 16)),
        "tick": int24(data[256:320]),
        "price": price
    }


def save_swaps(conn: sqlite3.Connection, swaps: list) -> int:
    """Insert swaps into SQLite, ignoring duplicates. Returns count inserted."""
    if not swaps:
        return 0
    cursor = conn.executemany("""
        INSERT OR IGNORE INTO swaps
        (id, block_number, tx_hash, log_index, amount0, amount1,
         sqrt_price_x96, liquidity, tick, price)
        VALUES (:id, :block_number, :tx_hash, :log_index, :amount0, :amount1,
                :sqrt_price_x96, :liquidity, :tick, :price)
    """, swaps)
    conn.commit()
    return cursor.rowcount


def update_progress(conn: sqlite3.Connection, last_block: int, total_swaps: int):
    """Update fetch progress checkpoint."""
    conn.execute("""
        INSERT OR REPLACE INTO fetch_progress (id, last_block, total_swaps, updated_at)
        VALUES (1, ?, ?, ?)
    """, (last_block, total_swaps, datetime.utcnow().isoformat()))
    conn.commit()


def get_progress(conn: sqlite3.Connection) -> tuple[int, int] | None:
    """Get last fetched block and total swaps from checkpoint."""
    row = conn.execute(
        "SELECT last_block, total_swaps FROM fetch_progress WHERE id = 1"
    ).fetchone()
    return (row[0], row[1]) if row else None


def fetch_range(start_block: int, end_block: int):
    """Fetch swap events for a block range with resume capability."""
    DATA_DIR.mkdir(exist_ok=True)
    conn = get_db()

    # Check for resume point
    progress = get_progress(conn)
    total_swaps = 0
    if progress and progress[0] >= start_block:
        start_block = progress[0] + 1
        total_swaps = progress[1]
        print(f"Resuming from block {start_block:,} ({total_swaps:,} swaps already)")

    total_blocks = end_block - start_block
    total_chunks = (total_blocks + CHUNK_SIZE - 1) // CHUNK_SIZE

    print(f"Fetching blocks {start_block:,} to {end_block:,}")
    print(f"Chunk size: {CHUNK_SIZE:,} blocks | Chunks: {total_chunks:,}")
    print(f"Rate limit: {RATE_LIMIT_DELAY}s between requests")
    print(f"Database: {DATA_DIR / 'swaps.db'}")
    print("=" * 70)

    errors = 0
    batch_start = time.time()
    chunk_times = []

    for chunk_idx in range(total_chunks):
        chunk_start = start_block + (chunk_idx * CHUNK_SIZE)
        chunk_end = min(chunk_start + CHUNK_SIZE - 1, end_block - 1)

        try:
            logs, elapsed = get_logs_with_retry(chunk_start, chunk_end)
            swaps = [decode_swap(log) for log in logs]
            inserted = save_swaps(conn, swaps)
            total_swaps += len(swaps)
            update_progress(conn, chunk_end, total_swaps)
            chunk_times.append(elapsed)

            # Progress output
            pct = (chunk_idx + 1) / total_chunks * 100
            avg_time = sum(chunk_times[-20:]) / len(chunk_times[-20:])  # Rolling avg
            remaining_chunks = total_chunks - chunk_idx - 1
            eta_seconds = remaining_chunks * (avg_time + RATE_LIMIT_DELAY)
            eta_str = f"{eta_seconds/3600:.1f}h" if eta_seconds > 3600 else f"{eta_seconds/60:.0f}m"

            print(f"[{chunk_idx + 1:>4}/{total_chunks}] "
                  f"{chunk_start:,}-{chunk_end:,} | "
                  f"{len(swaps):>4} swaps ({elapsed:>5.1f}s) | "
                  f"Total: {total_swaps:,} | "
                  f"ETA: {eta_str} | {pct:.1f}%")

        except Exception as e:
            errors += 1
            print(f"[{chunk_idx + 1:>4}/{total_chunks}] ERROR at {chunk_start:,}: {e}")
            if errors >= 10:
                print("Too many consecutive errors, stopping. Run again to resume.")
                break

        # Rate limiting
        if chunk_idx < total_chunks - 1:
            time.sleep(RATE_LIMIT_DELAY)

    batch_elapsed = time.time() - batch_start

    # Final summary
    print("=" * 70)
    print(f"Completed in {batch_elapsed/3600:.2f} hours ({batch_elapsed:.0f}s)")
    print(f"Total swaps fetched: {total_swaps:,}")
    print(f"Errors: {errors}")
    if chunk_times:
        print(f"Avg request time: {sum(chunk_times)/len(chunk_times):.2f}s")

    # DB stats
    count = conn.execute("SELECT COUNT(*) FROM swaps").fetchone()[0]
    min_block = conn.execute("SELECT MIN(block_number) FROM swaps").fetchone()[0]
    max_block = conn.execute("SELECT MAX(block_number) FROM swaps").fetchone()[0]
    print(f"Database: {count:,} swaps, blocks {min_block:,} - {max_block:,}")

    conn.close()


if __name__ == "__main__":
    print(f"Starting batch fetch at {datetime.utcnow().isoformat()}")
    print()
    fetch_range(START_BLOCK, END_BLOCK)
    print()
    print(f"Finished at {datetime.utcnow().isoformat()}")
