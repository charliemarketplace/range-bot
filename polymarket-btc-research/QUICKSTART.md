# Quick Start Guide

Get up and running with Polymarket BTC research in 5 minutes.

## Prerequisites

- Python 3.11+ installed
- UV package manager (recommended) or pip
- Polygon RPC access (free tier is fine)

## Installation

### 1. Install Dependencies

```bash
cd polymarket-btc-research

# Option A: Using UV (recommended)
uv sync

# Option B: Using pip
pip install -e ..
```

### 2. Configure Environment

```bash
# Copy template
cp .env.example .env

# Edit .env (optional - defaults work for testing)
nano .env
```

For basic testing, you can use the default public RPC. For production:
- Get free RPC from [Alchemy](https://www.alchemy.com/), [Infura](https://infura.io/), or [QuickNode](https://www.quicknode.com/)
- Add to `.env`: `POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY`

### 3. Create Data Directory

```bash
mkdir -p data
```

## Usage Examples

### Run Interactive Examples

```bash
python example_usage.py
```

Choose from:
1. Basic Queries - Test Polymarket and Chainlink APIs
2. Bayesian Model - See probability estimates
3. Edge Detection - Trading signal generation
4. Order Book - Real market data
5. Price Monitoring - Watch Chainlink updates
6. Run All - Execute all examples

### Test Individual Components

**Polymarket API:**
```bash
python src/polymarket_client.py
```

**Chainlink Oracle:**
```bash
python src/chainlink_fetcher.py
```

**Bayesian Model:**
```bash
python src/bayesian_model.py
```

**Market Collector:**
```bash
python src/market_collector.py
```

### Live Market Monitoring

```bash
python src/live_monitor.py
```

This will:
- Search for active BTC 5-minute markets
- Monitor price feeds in real-time
- Calculate Bayesian probabilities
- Detect edge opportunities
- Save opportunities to `data/` folder

## Expected Output

### Successful Test
```
Connected to Chainlink feed: BTC / USD
Decimals: 8
BTC Price: $95,234.50
Found 15 BTC-related markets
```

### If You See Errors

**"Connection error"**: RPC rate limited or unavailable
- Solution: Use a dedicated RPC provider (Alchemy/Infura)

**"No active markets"**: Markets aren't running right now
- Solution: BTC 5-min markets run continuously, try again in a few minutes

**"API timeout"**: Network issues or rate limiting
- Solution: Check internet connection, try dedicated RPC

## Next Steps

### For Research

1. **Collect Data**:
   ```bash
   python src/market_collector.py
   ```

2. **Analyze Results**:
   - Check `data/` folder for JSON files
   - Load into pandas/numpy for analysis

3. **Backtest Strategy**:
   - Collect 1-2 weeks of data
   - Test Bayesian model accuracy
   - Calculate risk-adjusted returns

### For Live Trading (Advanced)

⚠️ **Warning**: Only for experienced traders. Test thoroughly first.

1. Set up dedicated RPC (Alchemy/QuickNode)
2. Implement execution logic (not included)
3. Start with paper trading
4. Monitor for at least 1 week before live capital
5. Never risk more than you can afford to lose

## Common Commands

```bash
# Search for BTC markets
python -c "from src.polymarket_client import PolymarketClient; \
           c = PolymarketClient(); \
           m = c.search_btc_markets(); \
           print(f'Found {len(m)} markets')"

# Get current BTC price
python -c "from src.chainlink_fetcher import ChainlinkFetcher; \
           c = ChainlinkFetcher(); \
           p = c.get_latest_price(); \
           print(f'BTC: ${p[\"price\"]:.2f}')"

# Calculate edge
python -c "from src.bayesian_model import BayesianBTCModel; \
           m = BayesianBTCModel(); \
           e = m.estimate_probability_up(95050, 95000, 15); \
           print(f'P(Up): {e[\"prob_up\"]:.4f}')"
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd polymarket-btc-research

# Reinstall dependencies
uv sync --reinstall
```

### RPC Connection Issues
```bash
# Test RPC connection
python -c "from web3 import Web3; \
           w3 = Web3(Web3.HTTPProvider('https://polygon-rpc.com')); \
           print('Connected' if w3.is_connected() else 'Failed')"
```

### No Data Returned
- Markets may not be active right now
- API rate limits (use dedicated RPC)
- Check internet connection

## Getting Help

1. Check the main [README.md](README.md) for detailed documentation
2. Review [research.md](research.md) for methodology
3. Examine example scripts in `src/`
4. Review error messages carefully

## What's Next?

- ✅ You've run basic examples
- ⬜ Collect 1 week of market data
- ⬜ Analyze historical edge opportunities
- ⬜ Refine Bayesian model parameters
- ⬜ Backtest trading strategy
- ⬜ Paper trade for validation
- ⬜ (Optional) Deploy live with small capital

## Resources

- [Polymarket Docs](https://docs.polymarket.com)
- [Chainlink Data Streams](https://docs.chain.link/data-streams)
- [Web3.py Docs](https://web3py.readthedocs.io/)

---

**Disclaimer**: Research and educational purposes only. Not financial advice.
