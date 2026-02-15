# API & Data Access Guide

Complete breakdown of all APIs, costs, authentication requirements, and historical data availability.

---

## 1. Polymarket APIs

### A. Gamma API (Market Metadata)

**Base URL**: `https://gamma-api.polymarket.com`

**Purpose**: Market discovery and metadata
- Get list of markets
- Market details (question, outcomes, end times)
- Market status (active/closed)

**Authentication**: ❌ **None required** (public API)

**Cost**: ✅ **Free**

**Rate Limits**: Unknown (generous for read-only)

**Historical Data**: ✅ **Yes**
- Can query closed markets
- Market metadata persists
- Resolution data available

**Example Endpoints**:
```
GET /markets?limit=100
GET /markets/{condition_id}
```

**Key Fields Returned**:
- `question` - Market question
- `condition_id` - Unique identifier
- `tokens` - Outcome tokens (Up/Down)
- `end_date_iso` - Market close time
- `active` - Is market currently active
- `closed` - Is market resolved

---

### B. CLOB API (Central Limit Order Book)

**Base URL**: `https://clob.polymarket.com`

**Purpose**: Trading and order book data
- Real-time order books (bids/asks)
- Market depth and liquidity
- Last trade prices
- Trade history

**Authentication**:
- ❌ **Read-only (order book)**: No auth required
- ✅ **Trading**: Requires API key + wallet signature

**Cost**: ✅ **Free** for reading data

**Rate Limits**: ~10 requests/second for public endpoints

**Historical Data**: ⚠️ **Limited**
- Order book snapshots: Only current
- Trade history: Recent trades available
- Timeseries endpoint: Available but not fully documented

**Example Endpoints**:
```
GET /book?token_id={token_id}        # Current order book
GET /last-trade-price?token_id=...   # Last trade
GET /trades?market={id}&limit=100    # Recent trades
GET /timeseries (documented but limited)
```

**Key Data Returned**:
```json
{
  "bids": [{"price": "0.55", "size": "100"}, ...],
  "asks": [{"price": "0.56", "size": "50"}, ...],
  "market": "condition_id",
  "timestamp": "..."
}
```

**Historical Limitation**:
- ⚠️ **No official historical order book snapshots**
- Need to collect in real-time yourself
- Or use third-party indexers (see below)

---

### C. Real-Time Data Socket (RTDS)

**WebSocket URL**: `wss://ws-live-data.polymarket.com`

**Purpose**: Real-time streaming data
- Live crypto prices (BTC, ETH, etc.)
- Chainlink oracle prices
- Market updates
- Comment feeds

**Authentication**: ❌ **None required**

**Cost**: ✅ **Free**

**Historical Data**: ❌ **No** (live streaming only)

**Channels**:
```javascript
// Subscribe to crypto prices
{
  "type": "subscribe",
  "channel": "crypto_prices"
}

// Subscribe to Chainlink prices
{
  "type": "subscribe",
  "channel": "crypto_prices_chainlink"
}
```

**SDKs**:
- Python: `pip install polymarket-rtds`
- TypeScript: https://github.com/Polymarket/real-time-data-client
- Rust: `cargo add polymarket-rtds`

---

## 2. Chainlink Price Feeds

### A. Standard On-Chain Price Feed (Polygon)

**Contract**: `0xc907E116054Ad103354f2D350FD2514433D57F6f` (BTC/USD on Polygon)

**Purpose**: On-chain BTC price data
- Latest price
- Round data
- Update timestamps

**Authentication**: ❌ **None** (public blockchain)

**Cost**:
- ✅ **Free** to read
- Requires Polygon RPC access (see below)

**Update Frequency**:
- Every 1-5 minutes typically
- Triggered by 0.5% deviation threshold
- Or 1 hour heartbeat

**Historical Data**: ✅ **Yes, via round IDs**

```python
# Get latest
latestRoundData() → (roundId, price, startedAt, updatedAt, answeredInRound)

# Get historical
getRoundData(roundId) → historical price at that round
```

**Limitations**:
- ⚠️ **Not sub-second precision** (updates every 1-5 min)
- ⚠️ **Must query round-by-round** (no bulk API)
- ⚠️ **Gap between rounds** (can miss exact timestamps)

**Access via**: Web3.py, ethers.js, or any Web3 library

---

### B. Chainlink Data Streams (High Frequency)

**URL**: https://data.chain.link/streams/btc-usd

**Purpose**: Sub-second BTC price data
- High-frequency updates (multiple per second)
- Low-latency (~100ms)
- Cryptographically signed reports
- **This is what Polymarket uses for 5-min markets**

**Authentication**: ✅ **Required**
- HMAC-based authentication
- API key + secret

**Cost**: 💰 **PAID SERVICE**
- Not publicly priced
- Must contact Chainlink for quote
- Likely $500-5000+/month depending on usage

**Access Methods**:
1. REST API (query on-demand)
2. WebSocket (streaming)
3. Pull oracle (fetch + verify on-chain)

**Historical Data**: ✅ **Yes**
- Historical reports available via REST API
- Query by timestamp range
- Sub-second precision maintained

**Example Request**:
```bash
# Requires authentication headers
GET https://api.data-streams.chain.link/reports/btc-usd?start=...&end=...
```

**Data Structure**:
```json
{
  "mid_price": 95234.50,
  "bid_price": 95233.00,
  "ask_price": 95236.00,
  "timestamp": 1707927384123456,  // microsecond precision
  "signature": "0x..."
}
```

**For This Project**:
- ⚠️ **Required for high-accuracy backtesting**
- ⚠️ **Not required for live trading** (can use standard feed)
- Alternative: Collect live data yourself via standard feed

---

## 3. Polygon RPC Access

**Purpose**: Read Chainlink oracle data on-chain

**Free Options** (with limits):
1. **Public RPC**: `https://polygon-rpc.com`
   - ✅ Free
   - ⚠️ Rate limited (~100 req/min)
   - ⚠️ Unreliable (often down)

2. **Alchemy**: https://www.alchemy.com/
   - ✅ Free tier: 300M compute units/month
   - Enough for ~100k-1M oracle reads
   - Reliable and fast

3. **Infura**: https://www.infura.io/
   - ✅ Free tier: 100k requests/day
   - Good reliability

4. **QuickNode**: https://www.quicknode.com/
   - ✅ Free tier: Limited
   - Very fast

**Authentication**: ✅ API key required (except public RPC)

**Cost**: ✅ Free tier sufficient for research/testing

**Recommendation**:
- Development: Alchemy free tier
- Production: Alchemy growth plan ($49/month) or QuickNode

---

## 4. Third-Party Data Providers (Optional)

### A. Bitquery (Polymarket Indexer)

**URL**: https://docs.bitquery.io/docs/examples/polymarket-api/

**Purpose**: Historical Polymarket data
- Market history
- Trade data
- Event logs indexed

**Authentication**: ✅ **Required** (API key)

**Cost**:
- Free tier: Limited queries
- Paid: $49-499/month

**Historical Data**: ✅ **Full historical access**

**Use Case**: Alternative to self-indexing if you need historical data quickly

---

### B. Dune Analytics

**URL**: https://dune.com/

**Purpose**: Query Polymarket smart contract events

**Authentication**: ❌ **None** for public dashboards, ✅ API key for queries

**Cost**:
- Free: View dashboards
- Paid: $39-399/month for API access

**Historical Data**: ✅ **Full blockchain history**

---

## 5. Self-Hosted Infrastructure (Alternative)

### Option: Run Your Own Polygon Archive Node

**Purpose**: Complete independence, full historical data

**Cost**: 💰 **$200-500/month** (VPS with 2TB+ storage)

**Pros**:
- Unlimited requests
- Full historical data
- No rate limits
- Complete privacy

**Cons**:
- Requires DevOps skills
- Maintenance burden
- Initial sync takes days

---

## Cost Summary by Use Case

### 🎯 Use Case 1: Research & Backtesting (Historical Analysis)

**Required**:
- ✅ Polymarket Gamma API: **Free**
- ✅ Polygon RPC (Alchemy free): **Free**
- ❌ Chainlink Data Streams: **$0-5000/month** (optional, can estimate)

**Alternative (Cheaper)**:
- Use standard Chainlink feed for rough estimates: **Free**
- Accept lower precision (1-5 min updates vs sub-second)
- Collect your own data going forward: **Free**

**Total**: **$0** (with limitations) or **$500-5000/month** (high precision)

---

### 🎯 Use Case 2: Live Monitoring (No Trading)

**Required**:
- ✅ Polymarket CLOB API: **Free**
- ✅ Polymarket RTDS WebSocket: **Free**
- ✅ Polygon RPC (Alchemy free): **Free**

**Total**: **$0/month**

---

### 🎯 Use Case 3: Live Trading (Full Production)

**Required**:
- ✅ Polymarket CLOB API: **Free**
- ✅ Polymarket trading API: **Free** (just gas costs)
- ✅ Polygon RPC (Alchemy Growth): **$49/month**
- ❌ Chainlink Data Streams: **Optional** ($500-5000/month)

**Gas Costs**:
- Polygon gas: **~$0.01-0.10 per trade**
- Very low compared to Ethereum

**Total**: **$49-5049/month** + gas fees

---

## Historical Data Availability Matrix

| Data Type | Source | Historical? | Cost | Precision |
|-----------|--------|-------------|------|-----------|
| Market metadata | Gamma API | ✅ Yes | Free | N/A |
| Order book snapshots | CLOB API | ⚠️ Limited | Free | 1-second (if you collect) |
| Trade history | CLOB API | ✅ Recent | Free | Transaction-level |
| BTC price (standard) | Chainlink on-chain | ✅ Yes (by round) | Free | 1-5 min updates |
| BTC price (high-freq) | Data Streams | ✅ Yes | Paid | Sub-second |
| Market resolutions | On-chain events | ✅ Yes | Free | Exact |
| Token prices (historical) | Self-collect or Bitquery | ⚠️ Must collect | Free or $49+/month | Depends |

---

## Recommended Setup by Budget

### 💰 Budget: $0/month (Hobbyist)

**What you get**:
- Live market monitoring ✅
- Current order books ✅
- Chainlink prices (1-5 min precision) ✅
- Can collect your own historical data going forward ✅

**What you miss**:
- Historical high-frequency BTC data ❌
- Historical order book snapshots ❌
- Must build historical dataset from scratch

**Can you trade profitably?**: Maybe
- Can detect edge in real-time
- Limited backtesting (rough estimates only)
- Risk: Model not validated on historical data

---

### 💰 Budget: $50/month (Serious Hobbyist)

**Includes**:
- Alchemy Growth plan ($49/month)
- Reliable RPC access
- Higher rate limits

**What you get**:
- Everything from $0 tier ✅
- Better reliability ✅
- Can collect data 24/7 without rate limits ✅

**What you miss**:
- Still no historical high-frequency data ❌
- Must collect forward-looking

**Can you trade profitably?**: Yes, if patient
- Collect 1-2 months of data first
- Validate model
- Then trade with confidence

---

### 💰 Budget: $500-5000/month (Professional)

**Includes**:
- Alchemy/QuickNode premium
- Chainlink Data Streams access
- Possibly Bitquery for historical data

**What you get**:
- Full historical high-frequency data ✅
- Immediate backtesting capability ✅
- Sub-second precision ✅
- Professional-grade infrastructure ✅

**Can you trade profitably?**: Best chance
- Validate model immediately on years of data
- Know exactly where edge exists
- Optimize before risking capital

---

## Practical Recommendations

### 🎯 For This Project (Polymarket BTC 5-min markets):

**Minimum Viable (Free)**:
1. Use Alchemy free tier RPC
2. Use Polymarket free APIs
3. Start collecting data NOW (forward-looking)
4. Use standard Chainlink feed (accept 1-5 min precision)
5. After 2-4 weeks, have enough data to validate model

**Optimal (Paid)**:
1. Get Chainlink Data Streams access (contact Chainlink)
2. Request historical data for past 90 days
3. Validate model immediately
4. Know edge before deploying capital

**Smart Middle Ground**:
1. Start with free tier
2. Collect 1 week of live data
3. Do preliminary analysis
4. If promising, pay for historical data
5. Validate thoroughly before scaling

---

## How to Get Started (Free Tier)

```bash
# 1. Sign up for Alchemy (free)
https://auth.alchemyapi.io/signup

# 2. Create Polygon app, get RPC URL
https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY

# 3. Update .env
POLYGON_RPC_URL=https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY

# 4. Start collecting data
.venv/bin/python polymarket-btc-research/src/market_collector.py

# 5. Run for 1-2 weeks, build dataset

# 6. Analyze and validate model
```

**Cost**: $0

**Time to tradeable model**: 1-2 weeks (data collection period)

---

## Authentication Setup Guide

### Alchemy (Free Tier - Recommended)

1. Go to https://www.alchemy.com/
2. Sign up (free)
3. Create app → Polygon → Mainnet
4. Copy API key
5. RPC URL: `https://polygon-mainnet.g.alchemy.com/v2/YOUR_KEY`

**Free Tier Limits**:
- 300M compute units/month
- ~1M+ oracle reads
- More than enough for research

---

### Chainlink Data Streams (Enterprise - Optional)

1. Go to https://chain.link/
2. Contact sales: "Interested in Data Streams API access for BTC/USD"
3. Explain use case
4. Get quote (likely $500-5000/month)
5. Receive API credentials (HMAC key + secret)

**Only needed if**:
- You want immediate backtesting
- Need sub-second historical data
- Have budget for professional-grade data

---

### Polymarket Trading (If You Want to Trade)

1. No API key needed for reading
2. To trade:
   - Connect wallet (MetaMask)
   - Sign message for session
   - Use CLOB API with signatures

**Cost**: Free (just gas fees)

---

## Bottom Line

**Can you use this project for free?** ✅ **YES**

**What's the catch?**
- No historical high-frequency data
- Must collect your own data going forward
- Takes 1-2 weeks to build dataset
- Lower precision (1-5 min vs sub-second)

**Is free tier enough to find edge?** ✅ **Probably**
- Standard Chainlink feed is same source Polymarket uses (eventually)
- Market inefficiency is at the order book level
- Main limitation is backtesting speed, not edge discovery

**Recommendation**:
- Start free
- Collect 2 weeks of data
- If model shows promise, invest in paid data
- Don't pay for data until you see evidence of edge

---

**Created**: 2026-02-14
**Status**: Complete API reference
