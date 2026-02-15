# Polymarket BTC 5-Minute Markets Research

Research and analysis tools for Polymarket's BTC 5-minute up/down prediction markets using Bayesian modeling.

## Overview

This project explores edge opportunities in Polymarket's ultra-short BTC markets by:
1. Collecting high-frequency price data from Chainlink BTC-USD oracles
2. Monitoring real-time order book data from Polymarket
3. Using Bayesian inference to estimate fair probabilities in the final 15-30 seconds
4. Identifying mispricing between market-implied and model-estimated probabilities

## Market Structure

- **Market Type**: Binary prediction markets (Up/Down)
- **Duration**: 5 minutes per market
- **Resolution Source**: Chainlink BTC-USD Data Streams
- **Settlement**: Automated, instant
- **Chain**: Polygon
- **Framework**: Gnosis Conditional Token Framework (CTF)

## Installation

### Prerequisites
- Python 3.11+
- UV package manager (recommended) or pip

### Setup

```bash
# Clone the repository
cd polymarket-btc-research

# Install dependencies with UV
uv sync

# Or with pip
pip install -e .

# Copy environment template
cp .env.example .env

# Edit .env with your RPC URL and API keys
```

### Configuration

Edit `.env` file with your settings:
- `POLYGON_RPC_URL`: Polygon RPC endpoint (free: Alchemy, Infura, QuickNode)
- Optional: Chainlink Data Streams API credentials for historical data

## Project Structure

```
polymarket-btc-research/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── polymarket_client.py     # Polymarket API client
│   ├── chainlink_fetcher.py     # Chainlink oracle data fetcher
│   ├── market_collector.py      # Market data collector
│   ├── bayesian_model.py        # Bayesian probability model
│   └── live_monitor.py          # Live market monitoring
├── data/                        # Collected data (gitignored)
├── notebooks/                   # Analysis notebooks
├── research.md                  # Detailed research document
├── README.md                    # This file
└── .env.example                 # Environment template
```

## Usage

### 1. Test Individual Components

```bash
# Test Polymarket API client
python src/polymarket_client.py

# Test Chainlink price fetcher
python src/chainlink_fetcher.py

# Test Bayesian model
python src/bayesian_model.py
```

### 2. Collect Market Data

```bash
# Run market data collector
python src/market_collector.py
```

This will:
- Find an active BTC 5-minute market
- Collect synchronized price data from Chainlink and Polymarket
- Save data to `data/` directory
- Perform basic analysis

### 3. Live Monitoring Example

```bash
# Monitor a market in real-time (when available)
python src/live_monitor.py
```

### 4. Use as Library

```python
from polymarket_client import PolymarketClient
from chainlink_fetcher import ChainlinkFetcher
from bayesian_model import BayesianBTCModel

# Initialize clients
poly = PolymarketClient()
chain = ChainlinkFetcher()
model = BayesianBTCModel()

# Get current BTC price
btc_data = chain.get_latest_price()
print(f"BTC: ${btc_data['price']:.2f}")

# Find active markets
markets = poly.search_btc_markets()
print(f"Found {len(markets)} BTC markets")

# Estimate probability
estimate = model.estimate_probability_up(
    current_btc_price=95050,
    opening_btc_price=95000,
    seconds_remaining=15
)
print(f"P(Up): {estimate['prob_up']:.4f}")
```

## Core Components

### PolymarketClient
- Interfaces with Polymarket's CLOB and Gamma APIs
- Fetches market metadata, order books, and trade data
- Supports real-time monitoring

### ChainlinkFetcher
- Reads Chainlink BTC-USD price feed on Polygon
- Supports latest price, historical rounds, and monitoring
- Connects via Web3.py

### MarketDataCollector
- Orchestrates synchronized data collection
- Aligns Chainlink prices with Polymarket order books
- Handles time-series data storage

### BayesianBTCModel
- Estimates fair probability P(BTC_close >= BTC_open)
- Models BTC price as Brownian motion with drift
- Calculates edge and expected value vs market prices
- Provides Kelly criterion position sizing

## Research Questions

1. **Price Discovery Efficiency**: Do token prices efficiently reflect BTC price trajectory in final seconds?
2. **Volatility Impact**: Does BTC volatility create exploitable mispricing?
3. **Liquidity**: How do liquidity conditions affect spreads and execution?
4. **Information Asymmetry**: Is there latency arbitrage from direct oracle access?

## Data Collection Strategy

### Required Data Sources

1. **Chainlink BTC-USD Oracle**
   - Latest price updates
   - Sub-second precision (via Data Streams API)
   - Historical round data

2. **Polymarket Markets**
   - Market creation/close timestamps
   - Opening/closing BTC prices
   - Actual resolutions

3. **Order Book Data**
   - Up/Down token prices (1-second intervals)
   - Order book depth
   - Bid-ask spreads
   - Focus: Final 15-30 seconds

4. **External Context**
   - Spot exchange prices (validation)
   - Futures markets
   - Order flow indicators

## Analysis Pipeline

```
Data Collection → Alignment → Feature Engineering → Modeling → Backtesting
```

1. **Collection**: Gather Chainlink, Polymarket, and market data
2. **Alignment**: Synchronize timestamps across sources (UTC)
3. **Features**: Volatility, drift, momentum, liquidity, spreads
4. **Modeling**: Bayesian probability estimation
5. **Backtesting**: Simulate trading with execution costs

## Key Metrics

- **Edge**: Difference between Bayesian estimate and market price
- **Expected Value**: Profit expectation per dollar bet
- **Kelly Fraction**: Optimal position size
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades

## Risk Considerations

### Data Quality
- API rate limits
- Missing data points
- Timestamp synchronization errors

### Execution Risk
- Gas costs (minimal on Polygon)
- Order execution latency
- Slippage in thin markets
- Front-running

### Model Risk
- Overfitting on limited data
- Non-stationary volatility
- Regime changes
- Parameter sensitivity

## Cost-Benefit Analysis

### Costs
- RPC provider: $0-500/month
- Chainlink API: TBD (quote required)
- Infrastructure: $100-300/month
- Gas: ~$0.01-0.10 per trade
- Development: 6-8 weeks

### Potential Returns
- Hypothesized edge: 2-10% per trade
- Markets per day: ~288 (every 5 minutes)
- Selective entry: 10-50 trades/day
- Target win rate: 55-65%

## Next Steps

### Phase 1: Validation (Week 1-2)
- [ ] Verify Chainlink API access
- [ ] Collect 1-week sample dataset
- [ ] Validate data quality
- [ ] Assess edge existence

### Phase 2: Model Development (Week 3-4)
- [ ] Refine Bayesian model
- [ ] Incorporate volatility estimation
- [ ] Add momentum signals
- [ ] Calibrate on historical data

### Phase 3: Backtesting (Week 5-6)
- [ ] Build backtesting framework
- [ ] Account for execution costs
- [ ] Measure risk-adjusted returns
- [ ] Stress test scenarios

### Phase 4: Live Testing (Week 7-8)
- [ ] Paper trading implementation
- [ ] Real-time data feeds
- [ ] Monitor performance
- [ ] Refine entry/exit rules

## Resources

### Documentation
- [Polymarket Docs](https://docs.polymarket.com)
- [Chainlink Data Streams](https://docs.chain.link/data-streams)
- [Gnosis CTF](https://docs.gnosis.io/conditionaltokens/)

### Contracts (Polygon)
- CTF: `0x4d97dcd97ec945f40cf65f87097ace5ea0476045`
- CTF Exchange: `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e`
- Chainlink BTC-USD: `0xc907E116054Ad103354f2D350FD2514433D57F6f`

### APIs
- CLOB: `https://clob.polymarket.com`
- Gamma: `https://gamma-api.polymarket.com`
- RTDS: `wss://ws-live-data.polymarket.com`

## Contributing

This is a research project. Contributions, ideas, and feedback welcome.

## Disclaimer

This is for research and educational purposes only. Not financial advice. Trading prediction markets involves risk. Always do your own research and never risk more than you can afford to lose.

## License

MIT License - See LICENSE file for details
