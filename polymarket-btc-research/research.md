# Polymarket BTC 5-Minute Up/Down Markets Research

## Executive Summary

This research document explores the structure, technical implementation, and potential edge opportunities in Polymarket's 5-minute Bitcoin price prediction markets. These markets launched February 12, 2026, and use Chainlink's BTC-USD data streams for automated settlement. The hypothesis is that Bayesian models can estimate fair prices for market resolution tokens (up/down) in the final 15-30 seconds, potentially identifying mispricing opportunities.

---

## 1. Polymarket BTC Up/Down 5-Minute Markets

### 1.1 Market Structure

**Launch Date**: February 12, 2026

**Market Format**:
- Each market covers a discrete 5-minute period
- Users bet on binary outcomes: "Up" or "Down"
- "Up" wins if BTC price at interval end ≥ starting price
- "Down" wins if BTC price at interval end < starting price
- Markets run continuously with new 5-minute windows

**Settlement**:
- Automated instant settlement using Chainlink BTC-USD data streams
- No human oracle intervention required
- Sub-second settlement latency

**Technical Foundation**:
- Built on prior 15-minute crypto market infrastructure
- Uses Chainlink's high-frequency oracle infrastructure
- Represents compression of timeframes from 15 min → 5 min

**Future Roadmap**:
- 1-minute price prediction markets planned
- POLY token release planned

### 1.2 Smart Contract Architecture

#### Core Contracts

**1. Conditional Token Framework (CTF)**
- **Address**: `0x4d97dcd97ec945f40cf65f87097ace5ea0476045` (Polygon)
- **Purpose**: Core ERC-1155 system for all prediction markets
- **Functions**:
  - Condition setup
  - Token minting/splitting
  - Merging positions
  - Redemption after resolution

**2. CTF Exchange Contract**
- **Address**: `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e` (Polygon)
- **Purpose**: Atomic swaps between CTF ERC1155 assets and ERC20 collateral
- **Model**: Hybrid decentralized exchange
  - Offchain matching by operator
  - Onchain settlement (non-custodial)
- **GitHub**: https://github.com/Polymarket/ctf-exchange

**3. UMA CTF Adapter**
- **Purpose**: Oracle bridge between UMA Optimistic Oracle and CTF conditions
- **Workflow**:
  1. Market initialization with parameters stored onchain
  2. Proposer submits resolution answer with bond
  3. 2-hour challenge period
  4. Resolution executed and condition resolved
- **GitHub**: https://github.com/Polymarket/uma-ctf-adapter

#### Token Mechanics

**Binary Outcome Tokens**:
- Each market has YES and NO tokens (ERC1155)
- 1 USDC splits into 1 YES + 1 NO token
- Tokens are condition-specific and collateral-backed
- Winning tokens redeem for 1 USDC
- Losing tokens become worthless

**Condition Preparation**:
```
1. prepareCondition() called on CTF contract
2. Generates unique conditionId
3. Enables collateral splitting into position sets
4. Links condition to resolution oracle
```

### 1.3 Key Events for Onchain Monitoring

**Critical Event Logs**:

1. **ConditionPreparation**
   - Emitted: When new market created
   - Data: conditionId, oracle, questionId, outcomeSlotCount

2. **QuestionInitialized** (UMA Oracle)
   - Emitted: Market question initialized in UMA
   - Data: questionId, ancillaryData, requestTimestamp, reward

3. **PayoutReported** (Resolution)
   - Emitted: When resolution data available
   - Data: conditionId, payouts[]

4. **ConditionResolution**
   - Emitted: When market resolves
   - Data: conditionId, oracle, questionId, payouts[]

---

## 2. Polymarket API Structure

### 2.1 API Architecture (2026)

Polymarket operates three main API systems:

**1. Gamma API** (Market Metadata)
- **Purpose**: Market discovery and metadata
- **Endpoints**:
  - Market listings
  - Market details
  - Market parameters

**2. CLOB API** (Central Limit Order Book)
- **Base URL**: `https://clob.polymarket.com`
- **Purpose**: Trading operations
- **Key Endpoints**:
  - `GET /book` - Order book summary
  - `POST /order` - Create order
  - `DELETE /order` - Cancel order
  - Trade execution
  - Position management

**3. Data API**
- **Purpose**: User-specific data
- **Functions**:
  - Portfolio positions
  - Trade history
  - PnL tracking

### 2.2 Order Book API

**Endpoint**: `GET /book`
- **URL**: `https://clob.polymarket.com/book`
- **Parameters**:
  - `token_id` - Specific outcome token
- **Response Data**:
  - Bids array (price, size)
  - Asks array (price, size)
  - Spread
  - Market depth
  - Timestamp
  - Last trade price

**Market Depth Analysis**:
- Full L2 order book data available
- Real-time liquidity metrics
- Historical order book snapshots via timeseries API

### 2.3 Real-Time Data Systems

**RTDS (Real-Time Data Socket)**
- **WebSocket URL**: `wss://ws-live-data.polymarket.com`
- **Purpose**: Real-time market updates
- **Subscriptions**:
  - `crypto_prices` - Binance source price data
  - `crypto_prices_chainlink` - Chainlink oracle prices
  - Market updates
  - Comment feeds

**CLOB WebSocket**
- **Purpose**: Order book updates and trade execution
- **Use Case**: High-frequency trading bots
- **Data**:
  - Order fills
  - Book updates
  - Trade notifications

**Python SDK**: `polymarket-apis` (PyPI)
**TypeScript SDK**: https://github.com/Polymarket/real-time-data-client
**Rust SDK**: `polymarket-rtds` (crates.io)

### 2.4 Historical Data Access

**Timeseries API**:
- **Endpoint**: `/timeseries` (documented in CLOB section)
- **Data Available**:
  - Historical order book snapshots
  - Trade history
  - Price history
  - Volume data
- **Granularity**: Second-level precision available

---

## 3. Chainlink BTC-USD Oracle Structure

### 3.1 Standard Price Feeds vs Data Streams

**Standard Price Feed** (Polygon Mainnet)
- **Type**: Push-based oracle
- **URL**: https://data.chain.link/feeds/polygon/mainnet/btc-usd
- **Update Frequency**: Variable (deviation threshold triggered)
- **Typical Updates**: 1-5 minutes
- **Access**: Read directly from onchain contract
- **Use Case**: Standard DeFi applications

**Data Streams** (High-Frequency)
- **Type**: Pull-based oracle
- **URL**: https://data.chain.link/streams/btc-usd
- **Update Frequency**: Sub-second aggregation
- **Latency**: Sub-second
- **Access**: REST API, WebSocket, or SDK
- **Verification**: Cryptographic signatures
- **Use Case**: Low-latency applications like Polymarket 5-min markets

### 3.2 Data Streams Architecture

**Design Philosophy**:
- Pull-based: Retrieve reports on-demand
- Offchain aggregation with onchain verification
- Multiple data points per second available
- Cryptographic proof of data authenticity

**Access Methods**:

1. **REST API**
   - Endpoint: Data Streams REST API
   - Authentication: HMAC-based authentication required
   - Functions:
     - Latest reports
     - Historical reports
     - Stream metadata

2. **WebSocket**
   - Real-time streaming connection
   - Sub-second updates
   - Push notifications for new data

3. **SDK Integration**
   - Direct integration into applications
   - Simplified authentication handling

### 3.3 Data Structure

**Report Contents**:
- Mid price
- LWBA (Liquidity-Weighted Bid-Ask) price
- Timestamp (high precision)
- Volatility metrics
- Liquidity indicators
- Cryptographic signatures

### 3.4 Resolution Source for Polymarket

**Specific Feed**: BTC/USD Data Stream
- **URL**: https://data.chain.link/streams/btc-usd
- **Resolution Method**:
  1. Market opens at timestamp T0
  2. Starting price P0 recorded from Chainlink
  3. Market closes at T0 + 5 minutes
  4. Ending price P1 recorded from Chainlink
  5. If P1 ≥ P0 → "Up" wins
  6. If P1 < P0 → "Down" wins

**Critical Precision Points**:
- Exact timestamp alignment crucial
- Sub-second precision available
- Deterministic resolution (no human intervention)
- Same data source accessible to all participants

---

## 4. Historical Data Analysis Plan

### 4.1 Data Collection Strategy

**Objective**: Align high-frequency BTC-USD price data with market open/close times and up/down token prices at second-level granularity to identify edge opportunities.

#### Data Sources Required

**1. Chainlink Historical BTC-USD Data**
- **Primary Source**: Chainlink Data Streams REST API
- **Access Requirements**:
  - API credentials (HMAC authentication)
  - Historical reports endpoint
- **Data Points Needed**:
  - Timestamp (microsecond precision)
  - BTC-USD price
  - Bid/Ask spread
  - Liquidity metrics
  - Data quality indicators
- **Frequency**: Sub-second (all available data points)
- **Time Range**: Past 30-90 days of 5-min market history

**2. Polymarket Market Data**
- **Source**: Polymarket CLOB API + onchain events
- **Data Points**:
  - Market creation timestamp (exact)
  - Market close timestamp (exact)
  - Opening BTC price (from Chainlink)
  - Closing BTC price (from Chainlink)
  - Actual resolution (Up or Down)
- **Access**:
  - Query ConditionPreparation events
  - Query ConditionResolution events
  - API historical market endpoint

**3. Token Price Data (Order Book)**
- **Source**: Polymarket timeseries API + RTDS recordings
- **Data Points**:
  - Up token price (second-by-second)
  - Down token price (second-by-second)
  - Order book depth
  - Bid-ask spread
  - Volume
  - Liquidity
- **Critical Window**: Final 15-30 seconds before market close
- **Frequency**: 1-second granularity minimum

**4. External Market Context**
- **BTC Spot Exchanges**: Binance, Coinbase, Kraken tick data
- **BTC Futures**: CME, Binance perpetuals
- **Order Flow**: Large trades, liquidations
- **Market Microstructure**: Spread, depth, volatility

### 4.2 Data Pipeline Architecture

```
Step 1: Chainlink Data Collection
├── Authenticate with Data Streams API
├── Query historical BTC-USD reports
├── Parse timestamps to microsecond precision
├── Store: timestamp, price, spread, liquidity
└── Index by timestamp

Step 2: Market Identification
├── Query Polymarket onchain events
├── Filter for BTC 5-min up/down markets
├── Extract market open/close timestamps
├── Retrieve opening and closing BTC prices
├── Determine actual outcome (Up/Down)
└── Create market index table

Step 3: Token Price Collection
├── For each market in index:
│   ├── Query timeseries API for market period
│   ├── Focus on final 15-30 seconds
│   ├── Extract Up token prices (1-sec intervals)
│   ├── Extract Down token prices (1-sec intervals)
│   ├── Calculate implied probabilities
│   └── Record order book depth
└── Store in time-series database

Step 4: Data Alignment
├── Synchronize timestamps across sources
├── Account for timezone differences (all UTC)
├── Handle missing data points (interpolation)
├── Validate data consistency
└── Create unified dataset

Step 5: Feature Engineering
├── Calculate true probability (Bayesian prior)
├── Extract market-implied probability
├── Compute mispricing (edge)
├── Measure BTC volatility in window
├── Quantify liquidity conditions
├── Detect momentum signals
└── Engineer additional features
```

### 4.3 Timestamp Alignment Strategy

**Critical Requirements**:
- All timestamps must be in UTC
- Microsecond precision for Chainlink data
- Second precision minimum for token prices
- Account for blockchain confirmation delays

**Alignment Process**:
```
1. Market Open Time (T_open)
   └── ConditionPreparation event block timestamp

2. Market Close Time (T_close)
   └── T_open + exactly 5 minutes (300 seconds)

3. Chainlink Price at Open (P_open)
   └── Data Stream report closest to T_open (≤1sec tolerance)

4. Chainlink Price at Close (P_close)
   └── Data Stream report closest to T_close (≤1sec tolerance)

5. Token Price Time Series
   └── For T in [T_close - 30sec, T_close]:
       ├── Query Up token price at T
       ├── Query Down token price at T
       └── Store (T, P_up, P_down, depth)
```

### 4.4 Edge Detection Framework

**Hypothesis**: In the final 15-30 seconds, market participants may misprice tokens due to:
- Information latency
- Limited liquidity
- Behavioral biases
- Inefficient price discovery
- Oracle reading delays

**Bayesian Model Approach**:

**Inputs (Final 30 Seconds)**:
1. Current BTC price trajectory
2. Opening price P_open
3. Remaining time to close
4. BTC volatility (realized, implied)
5. Market microstructure (spread, depth)
6. Order flow imbalance
7. External market conditions

**Model Output**:
- P(Up) - Bayesian probability that P_close ≥ P_open
- P(Down) - Bayesian probability that P_close < P_open

**Edge Calculation**:
```
Market_Implied_P(Up) = P_up_token
Bayesian_P(Up) = Bayesian model output
Edge = Bayesian_P(Up) - Market_Implied_P(Up)

If Edge > threshold (e.g., 5%):
    → Potential profitable opportunity to buy Up token
If Edge < -threshold:
    → Potential profitable opportunity to buy Down token
```

### 4.5 Key Research Questions

**1. Price Discovery Efficiency**
- How efficiently do token prices converge to fair value?
- Is there systematic mispricing in final seconds?
- Do prices overreact or underreact to BTC movements?

**2. Volatility Impact**
- How does BTC volatility affect token pricing accuracy?
- Are high-volatility periods more exploitable?
- What's the relationship between realized vs implied vol?

**3. Liquidity Conditions**
- Does low liquidity create mispricing opportunities?
- How wide are spreads in final seconds?
- Can edge compensate for execution costs?

**4. Momentum Signals**
- Does recent BTC price momentum predict outcomes?
- Are market participants anchored to outdated probabilities?
- Can sub-second price changes be predictive?

**5. Information Asymmetry**
- Who has fastest access to Chainlink data?
- Is there latency arbitrage opportunity?
- Can direct oracle reading provide edge?

### 4.6 Implementation Roadmap

**Phase 1: Data Infrastructure (Week 1-2)**
- [ ] Obtain Chainlink Data Streams API credentials
- [ ] Set up Polygon node or use RPC provider
- [ ] Build event log indexer for Polymarket contracts
- [ ] Configure timeseries database (TimescaleDB/InfluxDB)
- [ ] Implement data collection scripts

**Phase 2: Historical Data Collection (Week 2-3)**
- [ ] Collect 30-90 days of Chainlink BTC-USD data
- [ ] Index all historical 5-min BTC markets
- [ ] Download token price timeseries for each market
- [ ] Validate data completeness and quality
- [ ] Build aligned dataset

**Phase 3: Exploratory Analysis (Week 3-4)**
- [ ] Analyze token price behavior in final 30 seconds
- [ ] Measure pricing efficiency
- [ ] Identify patterns of mispricing
- [ ] Correlate with volatility and liquidity
- [ ] Quantify potential edge magnitude

**Phase 4: Bayesian Model Development (Week 4-6)**
- [ ] Define prior distributions
- [ ] Build likelihood function for BTC movements
- [ ] Implement posterior probability calculation
- [ ] Calibrate model on historical data
- [ ] Validate out-of-sample performance

**Phase 5: Backtesting (Week 6-7)**
- [ ] Simulate trading strategy
- [ ] Account for execution costs (spread, gas)
- [ ] Measure risk-adjusted returns (Sharpe, Sortino)
- [ ] Stress test under various market conditions
- [ ] Optimize entry/exit timing

**Phase 6: Real-Time Prototype (Week 7-8)**
- [ ] Build live data feeds (Chainlink + Polymarket)
- [ ] Implement real-time Bayesian inference
- [ ] Create order execution logic
- [ ] Test on paper trading
- [ ] Monitor for edge opportunities

---

## 5. Technical Considerations

### 5.1 Challenges

**Data Access**:
- Chainlink Data Streams requires paid API access
- Historical data may have gaps
- Sub-second data storage requirements (large)
- Synchronization across multiple data sources

**Execution Risk**:
- Gas costs on Polygon (minimal but non-zero)
- Order execution latency
- Slippage in thin markets
- Front-running risk

**Model Risk**:
- Overfitting on limited historical data
- Regime changes in market behavior
- Non-stationarity of BTC volatility
- Parameter sensitivity

### 5.2 Mitigation Strategies

**Data Quality**:
- Multiple data source validation
- Outlier detection and filtering
- Gap-filling interpolation methods
- Redundant data collection

**Execution Optimization**:
- Pre-signed orders for speed
- MEV protection strategies
- Optimal order sizing
- Dynamic spread management

**Model Robustness**:
- Cross-validation techniques
- Walk-forward optimization
- Ensemble models
- Adaptive parameter updating
- Regular recalibration

### 5.3 Cost-Benefit Analysis

**Costs**:
- Chainlink Data Streams API: $X/month (requires quote)
- RPC provider (Polygon): $50-500/month depending on volume
- Infrastructure (server, database): $100-300/month
- Gas costs: ~$0.01-0.10 per trade on Polygon
- Development time: 6-8 weeks

**Potential Returns**:
- Edge per trade: 2-10% (hypothesis)
- Trades per day: 10-50 (288 markets per day, selective entry)
- Win rate: 55-65% (hypothesis)
- Position size: Limited by liquidity
- Risk-adjusted returns: To be determined by backtest

---

## 6. Key Contracts and Resources

### Smart Contracts (Polygon Mainnet)

- **CTF Contract**: `0x4d97dcd97ec945f40cf65f87097ace5ea0476045`
- **CTF Exchange**: `0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e`

### API Endpoints

- **CLOB API**: `https://clob.polymarket.com`
- **RTDS WebSocket**: `wss://ws-live-data.polymarket.com`
- **Chainlink BTC-USD Feed**: https://data.chain.link/feeds/polygon/mainnet/btc-usd
- **Chainlink Data Streams**: https://data.chain.link/streams/btc-usd

### GitHub Repositories

- **CTF Exchange**: https://github.com/Polymarket/ctf-exchange
- **UMA CTF Adapter**: https://github.com/Polymarket/uma-ctf-adapter
- **Real-Time Data Client**: https://github.com/Polymarket/real-time-data-client

### Documentation

- **Polymarket Docs**: https://docs.polymarket.com
- **Chainlink Data Streams**: https://docs.chain.link/data-streams
- **Chainlink Historical Data**: https://docs.chain.link/data-feeds/historical-data

---

## 7. Next Steps

1. **Validate Access**: Confirm ability to access Chainlink Data Streams historical data
2. **Prototype Collector**: Build minimal data collection script
3. **Sample Analysis**: Analyze 1-week sample of data manually
4. **Assess Feasibility**: Determine if edge exists before full buildout
5. **Iterate**: Refine hypothesis based on initial findings

---

## 8. References

### Polymarket Markets & Launch
- [Polymarket Debuts 5-Minute Bitcoin Prediction Markets With Instant Settlement | CoinMarketCap](https://coinmarketcap.com/academy/article/polymarket-debuts-5-minute-bitcoin-prediction-markets-with-instant-settlement)
- [RTDS Crypto Prices - Polymarket Documentation](https://docs.polymarket.com/developers/RTDS/RTDS-crypto-prices)
- [Polymarket Introduces 5-Minute Bitcoin Price Prediction Market](https://mlq.ai/news/polymarket-introduces-5-minute-bitcoin-price-prediction-market/)
- [Polymarket crypto markets launch 5-minute trading Chainlink](https://en.cryptonomist.ch/2026/02/13/polymarket-crypto-markets-5-minute/)

### Smart Contracts & Architecture
- [Overview - Polymarket Documentation](https://docs.polymarket.com/developers/CTF/overview)
- [Decoding the Digital Tea Leaves: A Guide to Analyzing Polymarket's On-Chain Order Data](https://yzc.me/x01Crypto/decoding-polymarket)
- [GitHub - Polymarket/ctf-exchange](https://github.com/Polymarket/ctf-exchange)
- [Polymarket: Conditional Tokens | PolygonScan](https://polygonscan.com/address/0x4d97dcd97ec945f40cf65f87097ace5ea0476045)
- [Polymarket: CTF Exchange | PolygonScan](https://polygonscan.com/address/0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e)

### API Documentation
- [Get order book summary - Polymarket Documentation](https://docs.polymarket.com/api-reference/orderbook/get-order-book-summary)
- [The Polymarket API: Architecture, Endpoints, and Use Cases | Medium](https://medium.com/@gwrx2005/the-polymarket-api-architecture-endpoints-and-use-cases-f1d88fa6c1bf)
- [Polymarket API - Get Prices, Trades & Market Data | Bitquery](https://docs.bitquery.io/docs/examples/polymarket-api/)

### Resolution & Oracles
- [How Are Prediction Markets Resolved? - Polymarket Documentation](https://docs.polymarket.com/polymarket-learn/markets/how-are-markets-resolved)
- [Polymarket + UMA](https://legacy-docs.polymarket.com/polymarket-+-uma)
- [Inside UMA Oracle | How Prediction Markets Resolution Works](https://rocknblock.io/blog/how-prediction-markets-resolution-works-uma-optimistic-oracle-polymarket)
- [GitHub - Polymarket/uma-ctf-adapter](https://github.com/Polymarket/uma-ctf-adapter)

### Chainlink Oracles
- [BTC / USD Price Feed | Chainlink](https://data.chain.link/feeds/polygon/mainnet/btc-usd)
- [BTC / USD Data Stream | Chainlink](https://data.chain.link/streams/btc-usd-cexprice-streams)
- [Chainlink Data Streams | Chainlink Documentation](https://docs.chain.link/data-streams)
- [Getting Historical Data | Chainlink Documentation](https://docs.chain.link/data-feeds/historical-data)
- [Data Streams REST API | Chainlink Documentation](https://docs.chain.link/data-streams/reference/data-streams-api/interface-api)

### Real-Time Data
- [Real Time Data Socket - Polymarket Documentation](https://docs.polymarket.com/developers/RTDS/RTDS-overview)
- [GitHub - Polymarket/real-time-data-client](https://github.com/Polymarket/real-time-data-client)
- [Historical Timeseries Data - Polymarket Documentation](https://docs.polymarket.com/developers/CLOB/timeseries)

---

**Document Version**: 1.0
**Last Updated**: February 14, 2026
**Author**: Research Team
**Status**: Initial Research Phase
