# 006: Scheduler & Orchestration

## Overview

Module for coordinating the entire system: scheduled runs, event-driven triggers, state management, and observability. Runs on AWS Lambda with EventBridge scheduling and SNS/SQS for event handling.

## Core Concept

```
Primary Loop (Hourly):
  1. Fetch latest data
  2. Compute VWAP prior
  3. Query Opus for chart analysis
  4. Run Bayesian update
  5. Get range recommendation
  6. Decide: rebalance or hold
  7. Execute if needed
  8. Log everything

Event Triggers (Async):
  - Price exceeds current range bounds → urgent rebalance check
  - Volatility spike detected → recalculate with updated prior
  - Gas price drops significantly → opportunistic rebalance
```

## Module Structure

```
src/
  orchestration/
    __init__.py
    types.py           # Run state, event types
    scheduler.py       # Lambda handler, scheduling logic
    pipeline.py        # Main execution pipeline
    triggers.py        # Event trigger handlers
    state.py           # State persistence
    alerts.py          # Alerting and notifications
    metrics.py         # Metrics collection
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        AWS Infrastructure                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  EventBridge ──────► Lambda (hourly)                            │
│       │                   │                                      │
│       │                   ▼                                      │
│       │            ┌─────────────┐                              │
│       │            │  Pipeline   │                              │
│       │            │             │                              │
│       │            │ 1. Fetch    │◄────── RPC / Flipside        │
│       │            │ 2. VWAP     │                              │
│       │            │ 3. Opus     │◄────── Anthropic API         │
│       │            │ 4. Bayes    │                              │
│       │            │ 5. Decide   │                              │
│       │            │ 6. Execute  │──────► Ethereum / Base       │
│       │            └─────────────┘                              │
│       │                   │                                      │
│       │                   ▼                                      │
│       │            DynamoDB (state)                             │
│       │            S3 (historical)                              │
│       │            CloudWatch (metrics)                         │
│       │                                                          │
│  SNS ◄─────────────────────────────────────────────────────────┤
│   │                                                              │
│   └──► Lambda (alert handler) ──► PagerDuty / Telegram         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘

Event Sources:
  - Alchemy/Infura webhooks (price threshold)
  - CloudWatch alarms (volatility)
  - Manual trigger (API Gateway)
```

## Data Types

```python
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Sequence

class RunStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

class TriggerType(Enum):
    SCHEDULED = "scheduled"          # Hourly cron
    PRICE_BREACH = "price_breach"    # Price left range
    VOLATILITY = "volatility"        # Vol spike
    GAS_OPPORTUNITY = "gas"          # Low gas window
    MANUAL = "manual"                # API trigger

@dataclass(frozen=True)
class RunContext:
    """Context for a single pipeline run."""
    run_id: str                      # UUID
    trigger: TriggerType
    timestamp: datetime
    chain: str                       # "ethereum" or "base"
    pool_address: str
    position_token_id: int

@dataclass(frozen=True)
class RunResult:
    """Result of a pipeline run."""
    run_id: str
    status: RunStatus
    duration_ms: int
    trigger: TriggerType

    # Pipeline outputs
    vwap_prior_center: Optional[Decimal]
    opus_direction: Optional[str]
    opus_confidence: Optional[str]
    posterior_center: Optional[Decimal]
    recommended_lower_tick: Optional[int]
    recommended_upper_tick: Optional[int]

    # Decision
    decision: Optional[str]          # "rebalance", "hold", "collect"
    decision_reason: Optional[str]

    # Execution
    tx_hash: Optional[str]
    gas_used: Optional[int]
    execution_error: Optional[str]

    # Metrics
    position_value_before: Optional[Decimal]
    position_value_after: Optional[Decimal]
    fees_collected: Optional[Decimal]

@dataclass(frozen=True)
class SystemState:
    """Persisted system state."""
    last_run_id: str
    last_run_time: datetime
    last_successful_rebalance: Optional[datetime]
    current_position: Optional[PositionState]
    consecutive_failures: int
    total_fees_collected: Decimal
    total_gas_spent: Decimal

@dataclass(frozen=True)
class AlertEvent:
    """Event that triggers an alert."""
    severity: str                    # "info", "warning", "critical"
    title: str
    message: str
    metadata: dict
```

## Core Functions

### Lambda Handler

```python
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    """
    Main Lambda entry point.

    Handles both scheduled and event-triggered invocations.
    """
    trigger = determine_trigger_type(event)
    run_context = build_run_context(event, trigger)

    try:
        result = run_pipeline(run_context)
        persist_result(result)
        emit_metrics(result)

        if result.status == RunStatus.FAILED:
            send_alert(build_failure_alert(result))

        return {"statusCode": 200, "body": result.to_json()}

    except Exception as e:
        handle_fatal_error(run_context, e)
        raise

def determine_trigger_type(event: dict) -> TriggerType:
    """Determine what triggered this invocation."""
    if "source" in event and event["source"] == "aws.events":
        return TriggerType.SCHEDULED
    if "Records" in event:  # SNS/SQS
        return parse_event_trigger(event["Records"][0])
    if "httpMethod" in event:  # API Gateway
        return TriggerType.MANUAL
    return TriggerType.SCHEDULED
```

### Pipeline Execution

```python
async def run_pipeline(context: RunContext) -> RunResult:
    """
    Execute the full pipeline.

    Orchestrates all modules in sequence.
    """
    start_time = time.time()

    # 1. Load state
    state = await load_system_state(context.chain, context.pool_address)
    position = await get_position_state(context.position_token_id, ...)

    # 2. Fetch latest data
    ohlc_data = await fetch_recent_ohlc(
        context.pool_address,
        lookback_hours=24
    )
    pool_state = await get_pool_state(context.pool_address, ...)

    # 3. Compute VWAP prior
    rolling_vwap = compute_rolling_vwap(ohlc_data, window_blocks=100)
    vwap_prior = build_vwap_prior(rolling_vwap, pool_state.price)

    # 4. Query Opus (parallel with step 3 in practice)
    chart_image = generate_chart_image(ohlc_data)
    opus_prediction = await analyze_chart(ChartAnalysisRequest(
        image_base64=encode_chart_for_api(chart_image),
        current_price=pool_state.price,
        timeframe="1h",
        context=f"ETH/USDC {context.pool_address}"
    ))

    # 5. Combine into posterior
    opus_contribution = prediction_to_prior_contribution(opus_prediction)
    combined_prior = combine_priors(vwap_prior, opus_contribution)
    likelihood = build_likelihood_from_ohlc(ohlc_data, pool_state.price)
    posterior = bayesian_update(combined_prior, likelihood)

    # 6. Get recommendation
    recommendation = optimize_range(posterior, target_coverage=0.90)

    # 7. Decide
    decision, reason = should_rebalance(position, recommendation, pool_state, ...)

    # 8. Execute if needed
    tx_hash = None
    gas_used = None
    execution_error = None

    if decision == RebalanceDecision.REBALANCE:
        action = build_rebalance_action(position, recommendation)
        result = await execute_rebalance(action, ...)
        tx_hash = result.tx_hash
        gas_used = result.gas_used
        execution_error = result.error

    elif decision == RebalanceDecision.COLLECT_ONLY:
        result = await collect_fees(position, ...)
        tx_hash = result.tx_hash

    # 9. Build result
    duration_ms = int((time.time() - start_time) * 1000)

    return RunResult(
        run_id=context.run_id,
        status=RunStatus.COMPLETED if not execution_error else RunStatus.FAILED,
        duration_ms=duration_ms,
        trigger=context.trigger,
        vwap_prior_center=rolling_vwap.median_vwap,
        opus_direction=opus_prediction.direction.value,
        opus_confidence=opus_prediction.confidence.value,
        posterior_center=posterior.expected_value(),
        recommended_lower_tick=recommendation.lower_tick,
        recommended_upper_tick=recommendation.upper_tick,
        decision=decision.value,
        decision_reason=reason,
        tx_hash=tx_hash,
        gas_used=gas_used,
        execution_error=execution_error,
        position_value_before=position.total_value_usd,
        position_value_after=None,  # Computed after tx confirmation
        fees_collected=position.tokens_owed_0 + position.tokens_owed_1 if decision == RebalanceDecision.COLLECT_ONLY else None
    )
```

### Event Triggers

```python
async def handle_price_breach_event(event: dict) -> None:
    """
    Handle price breach trigger.

    Triggered when price moves outside current position range.
    """
    pool_address = event["pool_address"]
    current_tick = event["current_tick"]

    # Load position
    state = await load_system_state(...)
    position = state.current_position

    # Check if actually breached (event might be stale)
    if position.tick_lower <= current_tick <= position.tick_upper:
        logger.info("Price back in range, skipping")
        return

    # Run urgent pipeline
    context = RunContext(
        run_id=str(uuid.uuid4()),
        trigger=TriggerType.PRICE_BREACH,
        ...
    )
    await run_pipeline(context)

async def handle_volatility_event(event: dict) -> None:
    """
    Handle volatility spike trigger.

    Triggered when realized vol exceeds threshold.
    """
    # Run pipeline with widened prior
    ...

def setup_price_breach_monitor(
    pool_address: str,
    position: PositionState,
    provider: Web3Provider
) -> None:
    """
    Configure webhook/subscription for price breach alerts.

    Options:
    - Alchemy Custom Webhooks
    - Infura Web3 Notifications
    - Self-hosted event listener
    """
    ...
```

### State Management

```python
async def load_system_state(chain: str, pool: str) -> SystemState:
    """Load state from DynamoDB."""
    ...

async def save_system_state(state: SystemState) -> None:
    """Persist state to DynamoDB."""
    ...

async def persist_result(result: RunResult) -> None:
    """
    Persist run result.

    - DynamoDB for latest state
    - S3 for historical archive
    """
    ...
```

### Metrics & Alerting

```python
def emit_metrics(result: RunResult) -> None:
    """
    Emit CloudWatch metrics.

    Metrics:
    - run_duration_ms
    - decision (dimension)
    - position_value
    - fees_collected
    - gas_used
    - opus_confidence (dimension)
    """
    cloudwatch.put_metric_data(
        Namespace="RangeBot",
        MetricData=[
            {
                "MetricName": "RunDuration",
                "Value": result.duration_ms,
                "Unit": "Milliseconds",
                "Dimensions": [
                    {"Name": "Chain", "Value": result.chain},
                    {"Name": "Trigger", "Value": result.trigger.value}
                ]
            },
            ...
        ]
    )

def send_alert(alert: AlertEvent) -> None:
    """Send alert via SNS."""
    sns.publish(
        TopicArn=ALERT_TOPIC_ARN,
        Subject=f"[{alert.severity.upper()}] {alert.title}",
        Message=json.dumps({
            "title": alert.title,
            "message": alert.message,
            "metadata": alert.metadata
        })
    )

def build_failure_alert(result: RunResult) -> AlertEvent:
    """Build alert for failed run."""
    return AlertEvent(
        severity="critical" if result.consecutive_failures > 3 else "warning",
        title=f"Pipeline run failed: {result.run_id}",
        message=result.execution_error or "Unknown error",
        metadata={
            "run_id": result.run_id,
            "trigger": result.trigger.value,
            "duration_ms": result.duration_ms
        }
    )
```

## Infrastructure as Code

### Terraform / CDK Outline

```hcl
# Lambda function
resource "aws_lambda_function" "range_bot" {
  function_name = "range-bot-pipeline"
  runtime       = "python3.11"
  handler       = "orchestration.scheduler.lambda_handler"
  timeout       = 300  # 5 minutes max
  memory_size   = 512

  environment {
    variables = {
      ANTHROPIC_API_KEY = data.aws_secretsmanager_secret_version.anthropic.secret_string
      ETH_RPC_URL       = var.eth_rpc_url
      BASE_RPC_URL      = var.base_rpc_url
      WALLET_PRIVATE_KEY = data.aws_secretsmanager_secret_version.wallet.secret_string
    }
  }
}

# Hourly schedule
resource "aws_cloudwatch_event_rule" "hourly" {
  name                = "range-bot-hourly"
  schedule_expression = "rate(1 hour)"
}

resource "aws_cloudwatch_event_target" "lambda" {
  rule      = aws_cloudwatch_event_rule.hourly.name
  target_id = "range-bot-lambda"
  arn       = aws_lambda_function.range_bot.arn
}

# DynamoDB for state
resource "aws_dynamodb_table" "state" {
  name         = "range-bot-state"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }
  attribute {
    name = "sk"
    type = "S"
  }
}

# S3 for historical data
resource "aws_s3_bucket" "data" {
  bucket = "range-bot-data-${var.environment}"
}

# SNS for alerts
resource "aws_sns_topic" "alerts" {
  name = "range-bot-alerts"
}
```

## Configuration

```python
@dataclass(frozen=True)
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    # Scheduling
    run_interval_minutes: int = 60
    max_run_duration_seconds: int = 300

    # Chains to manage
    chains: tuple[str, ...] = ("ethereum", "base")

    # Thresholds for event triggers
    price_breach_buffer_ticks: int = 10
    volatility_spike_threshold: float = 2.0  # 2x normal vol
    gas_opportunity_threshold_gwei: int = 20

    # Alerting
    alert_on_failure: bool = True
    alert_on_rebalance: bool = True
    max_consecutive_failures: int = 5

    # Cost limits
    max_daily_gas_spend_usd: Decimal = Decimal("50")
    max_daily_opus_calls: int = 30
```

## Testing Requirements

### Unit Tests
- [ ] Trigger type detection
- [ ] Pipeline step sequencing
- [ ] State serialization/deserialization
- [ ] Metric emission format
- [ ] Alert building logic

### Integration Tests
- [ ] Full pipeline with mocked external services
- [ ] DynamoDB state round-trip
- [ ] S3 result archival
- [ ] CloudWatch metric emission

### End-to-End Tests
- [ ] Lambda invocation via test event
- [ ] Scheduled trigger simulation
- [ ] Event trigger simulation

## Dependencies

```python
boto3 = "^1.28"          # AWS SDK
aws-lambda-powertools = "^2.20"  # Lambda utilities
```

## Acceptance Criteria

- [ ] Lambda handler working with both scheduled and event triggers
- [ ] Full pipeline execution in < 60 seconds typical
- [ ] State persistence to DynamoDB
- [ ] Result archival to S3
- [ ] CloudWatch metrics emitting
- [ ] Alert notifications working
- [ ] Terraform/CDK for infrastructure
- [ ] Local testing mode (no AWS dependencies)
- [ ] 85%+ test coverage

## Monitoring Dashboard

Key metrics to display:
- Run success rate (24h rolling)
- Average run duration
- Decision distribution (rebalance vs hold)
- Position value over time
- Cumulative fees collected
- Gas spent
- Opus confidence distribution

## References

- [AWS Lambda Python](https://docs.aws.amazon.com/lambda/latest/dg/python-handler.html)
- [EventBridge Scheduler](https://docs.aws.amazon.com/scheduler/latest/UserGuide/what-is-scheduler.html)
- [Lambda Powertools](https://docs.powertools.aws.dev/lambda/python/latest/)
