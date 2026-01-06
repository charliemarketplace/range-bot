# range-bot

Bayesian LP range optimization for Uniswap v3.

## Concept

Uses a Bayesian approach to optimize Uniswap v3 concentrated liquidity positions:

1. **Prior**: Rolling 100-block median VWAP provides a mean-reversion anchor (historically revisited 90%+ of the time within 1000 blocks)
2. **Likelihood**: Recent OHLC data shows where price has been
3. **AI Input**: Claude (Opus/Sonnet) analyzes chart images to contribute directional hints
4. **Posterior**: Bayesian update combines all inputs into a probability distribution
5. **Optimization**: Find the tightest LP range covering 90% of the posterior mass

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Hourly Pipeline                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Fetch Data ──► 2. VWAP Prior ──► 3. Opus Analysis       │
│        │                  │                   │              │
│        │                  ▼                   ▼              │
│        │          ┌─────────────────────────────┐           │
│        └────────► │    4. Bayesian Update       │           │
│                   │    prior × likelihood       │           │
│                   └─────────────────────────────┘           │
│                                │                             │
│                                ▼                             │
│                   ┌─────────────────────────────┐           │
│                   │  5. Range Optimization      │           │
│                   │  → (tick_lower, tick_upper) │           │
│                   └─────────────────────────────┘           │
│                                │                             │
│                                ▼                             │
│                   ┌─────────────────────────────┐           │
│                   │  6. Rebalance Decision      │           │
│                   │  cost vs improvement        │           │
│                   └─────────────────────────────┘           │
│                                │                             │
│                                ▼                             │
│                   ┌─────────────────────────────┐           │
│                   │  7. Execute (if needed)     │           │
│                   │  burn → mint                │           │
│                   └─────────────────────────────┘           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3.11+**: Core logic, Bayesian computations
- **AWS Lambda**: Hourly scheduled execution
- **DynamoDB/S3**: State and historical data storage
- **Anthropic API**: Chart analysis via Claude
- **web3.py/ethers**: Ethereum interaction

## Design Principles

- **Functional Programming**: Pure functions, immutable data, explicit I/O boundaries
- **Modular**: Each component is independent and testable
- **Unit Tested**: 85%+ coverage target before implementation

## Specifications

See `docs/specs/` for detailed module specifications:

| Spec | Module | Description |
|------|--------|-------------|
| [001](docs/specs/001-data-layer.md) | Data Layer | Historical OHLC & on-chain data storage |
| [002](docs/specs/002-vwap-prior.md) | VWAP Prior | Rolling VWAP calculation and prior construction |
| [003](docs/specs/003-opus-integration.md) | Opus Integration | AI chart analysis for directional hints |
| [004](docs/specs/004-bayesian-engine.md) | Bayesian Engine | Prior × likelihood → posterior computation |
| [005](docs/specs/005-lp-position-manager.md) | LP Position Manager | Uniswap v3 position execution |
| [006](docs/specs/006-scheduler-orchestration.md) | Scheduler | Lambda orchestration and event triggers |
| [007](docs/specs/007-testing-infrastructure.md) | Testing | Test framework, fixtures, backtesting |

## Supported Pools

- Ethereum: ETH/USDC 0.05%, ETH/USDC 0.3%
- Base: ETH/USDC 0.05%, ETH/USDC 0.3%

## References

- [On-chain Pricing Research](https://flipsidecrypto.beehiiv.com/p/onchain-pricing)
- [onchain-pricing repo](https://github.com/charliemarketplace/onchain-pricing)
- [Uniswap v3 Whitepaper](https://uniswap.org/whitepaper-v3.pdf)
- [Uniswap v3 R Package](https://github.com/charliemarketplace/uniswap) - Retroactive LP position optimization in R

## License

MIT
