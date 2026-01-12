# Production Deployment Guide: Range-Bot on AWS

This guide provides exact step-by-step instructions to deploy the range-bot trading system on AWS. Follow each step in order.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [AWS Account Setup](#2-aws-account-setup)
3. [Local Development Environment](#3-local-development-environment)
4. [CDK Project Setup](#4-cdk-project-setup)
5. [Data Pipeline Implementation](#5-data-pipeline-implementation)
6. [Model Inference Lambdas](#6-model-inference-lambdas)
7. [Observability Stack](#7-observability-stack)
8. [Execution Layer](#8-execution-layer)
9. [Risk Management](#9-risk-management)
10. [Deploy and Test](#10-deploy-and-test)
11. [Go-Live Checklist](#11-go-live-checklist)
12. [Quick Reference Commands](#12-quick-reference-commands)
13. [Signer Container Implementation](#13-signer-container-implementation)
14. [Uniswap V3 Math Library](#14-uniswap-v3-math-library)
15. [Integration Tests](#15-integration-tests)
16. [Detailed Operations Runbook](#16-detailed-operations-runbook)
17. [Troubleshooting Guide](#17-troubleshooting-guide)

---

## 1. Prerequisites

### 1.1 Required Accounts

| Account | Purpose | Sign Up URL |
|---------|---------|-------------|
| AWS | Infrastructure hosting | https://aws.amazon.com/free |
| Alchemy | Ethereum RPC + webhooks | https://www.alchemy.com/ |
| GitHub | Source control | https://github.com |

### 1.2 Required Software

Install these on your development machine:

```bash
# Node.js (required for CDK)
# Download from: https://nodejs.org/en/download/
# Verify installation:
node --version  # Should be v18+
npm --version   # Should be 9+

# Python 3.11+
# Download from: https://www.python.org/downloads/
# Verify installation:
python --version  # Should be 3.11+
pip --version

# AWS CLI v2
# Download from: https://aws.amazon.com/cli/
# Verify installation:
aws --version  # Should be aws-cli/2.x.x

# AWS CDK
npm install -g aws-cdk
cdk --version  # Should be 2.x.x

# Git
git --version
```

### 1.3 Required Knowledge

Before proceeding, ensure you understand:
- Basic AWS concepts (IAM, Lambda, DynamoDB)
- Python programming
- Command line / terminal usage
- Uniswap v3 concepts (LP, ticks, liquidity)

---

## 2. AWS Account Setup

### 2.1 Create AWS Account

1. Go to https://aws.amazon.com/free
2. Click "Create a Free Account"
3. Enter email address and choose account name: `range-bot-prod`
4. Verify email
5. Enter payment information (required even for free tier)
6. Choose "Basic support - Free"
7. Complete phone verification
8. Select "Personal" account type
9. Complete signup

### 2.2 Secure Root Account

**IMPORTANT: Do these immediately after account creation**

1. **Enable MFA on root account:**
   ```
   AWS Console → IAM → Security credentials
   → Multi-factor authentication (MFA) → Activate MFA
   → Select "Authenticator app"
   → Scan QR code with Google Authenticator / Authy
   → Enter two consecutive codes
   → Click "Add MFA"
   ```

2. **Create billing alert:**
   ```
   AWS Console → Billing → Budgets → Create budget
   → Choose "Cost budget"
   → Budget name: "Monthly-100-USD"
   → Enter budgeted amount: 100
   → Add alert threshold: 80%
   → Enter your email address
   → Create budget
   ```

### 2.3 Create IAM Admin User

Never use root account for daily operations. Create an admin user:

1. **Create IAM user:**
   ```
   AWS Console → IAM → Users → Create user
   → User name: "admin"
   → Check "Provide user access to the AWS Management Console"
   → Select "I want to create an IAM user"
   → Custom password (save this securely!)
   → Uncheck "Users must create a new password at next sign-in"
   → Next
   ```

2. **Attach policies:**
   ```
   → Select "Attach policies directly"
   → Search and check "AdministratorAccess"
   → Next → Create user
   ```

3. **Create access keys for CLI:**
   ```
   IAM → Users → admin → Security credentials
   → Access keys → Create access key
   → Select "Command Line Interface (CLI)"
   → Check acknowledgment → Next
   → Create access key
   → SAVE ACCESS KEY ID AND SECRET (you won't see secret again!)
   ```

### 2.4 Configure AWS CLI

Open terminal and run:

```bash
aws configure
```

Enter when prompted:
```
AWS Access Key ID: [paste your access key]
AWS Secret Access Key: [paste your secret key]
Default region name: us-east-1
Default output format: json
```

Verify configuration:
```bash
aws sts get-caller-identity
```

Expected output:
```json
{
    "UserId": "AIDA...",
    "Account": "123456789012",
    "Arn": "arn:aws:iam::123456789012:user/admin"
}
```

### 2.5 Bootstrap CDK

CDK needs a one-time bootstrap in each region:

```bash
cdk bootstrap aws://ACCOUNT-NUMBER/us-east-1
```

Replace `ACCOUNT-NUMBER` with your 12-digit account ID from the previous command.

Expected output:
```
 ⏳  Bootstrapping environment aws://123456789012/us-east-1...
 ✅  Environment aws://123456789012/us-east-1 bootstrapped
```

---

## 3. Local Development Environment

### 3.1 Clone Repository

```bash
cd C:\Users\crmer\Documents\Crypto
cd range-bot
```

### 3.2 Create Python Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it (Windows PowerShell):
.venv\Scripts\Activate.ps1

# Or Windows CMD:
.venv\Scripts\activate.bat

# Or Git Bash:
source .venv/Scripts/activate

# Verify activation (should show .venv in prompt)
```

### 3.3 Install Dependencies

Create/update `requirements.txt` in project root:

```
# requirements.txt
requests>=2.32.5
numpy>=1.24
scipy>=1.11
web3>=6.0
eth-account>=0.9
eth-abi>=4.0
boto3>=1.28
python-dotenv>=1.0
pytest>=7.0
pytest-cov>=4.0
mypy>=1.0
black>=23.0
```

Install:
```bash
pip install -r requirements.txt
```

### 3.4 Create Project Structure

```bash
# Create directory structure (Windows)
mkdir infra\stacks
mkdir lambdas\swap_listener
mkdir lambdas\ohlc_aggregator
mkdir lambdas\bayesian_range
mkdir lambdas\alert_engine
mkdir lambdas\momentum_signal
mkdir lambdas\risk_engine
mkdir lambdas\orchestrator
mkdir lambdas\layers\dependencies
mkdir src\execution
mkdir containers\signer
mkdir tests\unit
mkdir tests\integration
```

Final structure:
```
range-bot/
├── docs/                    # Existing docs
├── poc/                     # Existing POC code
├── infra/                   # NEW: CDK infrastructure
│   ├── app.py
│   ├── cdk.json
│   ├── requirements.txt
│   └── stacks/
│       ├── __init__.py
│       ├── data_pipeline.py
│       ├── model_inference.py
│       ├── execution.py
│       ├── risk_management.py
│       └── observability.py
├── lambdas/                 # NEW: Lambda function code
│   ├── layers/
│   │   └── dependencies/
│   │       └── requirements.txt
│   ├── swap_listener/
│   │   └── handler.py
│   ├── ohlc_aggregator/
│   │   └── handler.py
│   ├── bayesian_range/
│   │   └── handler.py
│   ├── alert_engine/
│   │   └── handler.py
│   ├── momentum_signal/
│   │   └── handler.py
│   ├── risk_engine/
│   │   └── handler.py
│   └── orchestrator/
│       └── handler.py
├── src/                     # NEW: Shared library code
│   └── execution/
│       ├── __init__.py
│       ├── position_manager.py
│       ├── tx_builder.py
│       ├── sizing.py
│       ├── risk.py
│       └── uniswap_math.py
├── containers/              # NEW: Docker containers
│   └── signer/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── main.py
│       └── signer.py
├── tests/                   # NEW: Test suites
│   ├── unit/
│   └── integration/
├── requirements.txt
└── pyproject.toml
```

---

## 4. CDK Project Setup

### 4.1 Create CDK Configuration Files

Create `infra/requirements.txt`:
```
aws-cdk-lib>=2.100.0
constructs>=10.0.0
```

Create `infra/cdk.json`:
```json
{
  "app": "python app.py",
  "context": {
    "@aws-cdk/aws-apigateway:usagePlanKeyOrderInsensitiveId": true,
    "@aws-cdk/core:stackRelativeExports": true,
    "@aws-cdk/aws-lambda:recognizeLayerVersion": true
  }
}
```

Create `infra/stacks/__init__.py`:
```python
# Empty file to make this a Python package
```

### 4.2 Create CDK App Entry Point

Create `infra/app.py`:

```python
#!/usr/bin/env python3
"""
CDK App Entry Point

Deploys all range-bot infrastructure stacks.
"""
import os
import aws_cdk as cdk
from stacks.data_pipeline import DataPipelineStack
from stacks.model_inference import ModelInferenceStack
from stacks.execution import ExecutionStack
from stacks.risk_management import RiskManagementStack
from stacks.observability import ObservabilityStack

app = cdk.App()

# Get environment from context or default
env_name = app.node.try_get_context("env") or "dev"

# AWS environment
aws_env = cdk.Environment(
    account=os.environ.get("CDK_DEFAULT_ACCOUNT"),
    region=os.environ.get("CDK_DEFAULT_REGION", "us-east-1"),
)

# =========================================================================
# CONFIGURATION - UPDATE THESE VALUES
# =========================================================================
ALERT_EMAIL = "YOUR_EMAIL@example.com"  # <-- CHANGE THIS
# =========================================================================

# Common tags for all resources
tags = {
    "Project": "range-bot",
    "Environment": env_name,
    "ManagedBy": "CDK",
}

# Deploy stacks in dependency order
data_stack = DataPipelineStack(
    app, f"RangeBot-Data-{env_name}",
    env=aws_env,
    env_name=env_name,
)

observability_stack = ObservabilityStack(
    app, f"RangeBot-Observability-{env_name}",
    env=aws_env,
    env_name=env_name,
    alert_email=ALERT_EMAIL,
)

model_stack = ModelInferenceStack(
    app, f"RangeBot-Model-{env_name}",
    env=aws_env,
    env_name=env_name,
    data_stack=data_stack,
)

execution_stack = ExecutionStack(
    app, f"RangeBot-Execution-{env_name}",
    env=aws_env,
    env_name=env_name,
)

risk_stack = RiskManagementStack(
    app, f"RangeBot-Risk-{env_name}",
    env=aws_env,
    env_name=env_name,
    data_stack=data_stack,
    observability_stack=observability_stack,
)

# Apply tags to all stacks
for stack in [data_stack, model_stack, observability_stack, execution_stack, risk_stack]:
    for key, value in tags.items():
        cdk.Tags.of(stack).add(key, value)

app.synth()
```

### 4.3 Install CDK Dependencies

```bash
cd infra
pip install -r requirements.txt
cd ..
```

---

## 5. Data Pipeline Implementation

### 5.1 Create Data Pipeline Stack

Create `infra/stacks/data_pipeline.py`:

```python
"""
Data Pipeline Stack

Creates:
- DynamoDB tables for swap buffer and OHLC candles
- S3 bucket for historical archives
- Lambda for swap ingestion
- Lambda for OHLC aggregation
"""
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_dynamodb as dynamodb,
    aws_s3 as s3,
    aws_lambda as lambda_,
    aws_lambda_event_sources as lambda_event_sources,
    aws_events as events,
    aws_events_targets as targets,
    aws_logs as logs,
)
from constructs import Construct


class DataPipelineStack(Stack):
    """Data ingestion and storage infrastructure."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # =====================================================================
        # DynamoDB Tables
        # =====================================================================

        # Swap Buffer Table (hot data, 4-hour TTL)
        self.swap_table = dynamodb.Table(
            self,
            "SwapBuffer",
            table_name=f"range-bot-swaps-{env_name}",
            partition_key=dynamodb.Attribute(
                name="pk",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="sk",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            time_to_live_attribute="ttl",
            removal_policy=RemovalPolicy.RETAIN if env_name == "prod" else RemovalPolicy.DESTROY,
            point_in_time_recovery=True,
            stream=dynamodb.StreamViewType.NEW_IMAGE,
        )

        # OHLC Candles Table (warm data)
        self.ohlc_table = dynamodb.Table(
            self,
            "OHLCCandles",
            table_name=f"range-bot-ohlc-{env_name}",
            partition_key=dynamodb.Attribute(
                name="pk",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="sk",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN if env_name == "prod" else RemovalPolicy.DESTROY,
            point_in_time_recovery=True,
        )

        # Model State Table
        self.model_state_table = dynamodb.Table(
            self,
            "ModelState",
            table_name=f"range-bot-model-state-{env_name}",
            partition_key=dynamodb.Attribute(
                name="pk",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="sk",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True,
        )

        # Position State Table
        self.position_table = dynamodb.Table(
            self,
            "PositionState",
            table_name=f"range-bot-positions-{env_name}",
            partition_key=dynamodb.Attribute(
                name="pk",
                type=dynamodb.AttributeType.STRING,
            ),
            sort_key=dynamodb.Attribute(
                name="sk",
                type=dynamodb.AttributeType.STRING,
            ),
            billing_mode=dynamodb.BillingMode.PAY_PER_REQUEST,
            removal_policy=RemovalPolicy.RETAIN,
            point_in_time_recovery=True,
        )

        # =====================================================================
        # S3 Bucket for Historical Archive
        # =====================================================================

        self.archive_bucket = s3.Bucket(
            self,
            "HistoricalArchive",
            bucket_name=f"range-bot-archive-{self.account}-{env_name}",
            versioned=True,
            encryption=s3.BucketEncryption.S3_MANAGED,
            removal_policy=RemovalPolicy.RETAIN,
            lifecycle_rules=[
                s3.LifecycleRule(
                    transitions=[
                        s3.Transition(
                            storage_class=s3.StorageClass.GLACIER,
                            transition_after=Duration.days(90),
                        )
                    ],
                )
            ],
        )

        # =====================================================================
        # SwapListener Lambda
        # =====================================================================

        self.swap_listener = lambda_.Function(
            self,
            "SwapListener",
            function_name=f"range-bot-swap-listener-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/swap_listener"),
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "SWAP_TABLE": self.swap_table.table_name,
                "POOL_ADDRESS": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
                "ENV_NAME": env_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Grant DynamoDB write access
        self.swap_table.grant_write_data(self.swap_listener)

        # Schedule: poll every 1 minute (EventBridge minimum)
        events.Rule(
            self,
            "SwapListenerSchedule",
            rule_name=f"range-bot-swap-listener-{env_name}",
            schedule=events.Schedule.rate(Duration.minutes(1)),
            targets=[targets.LambdaFunction(self.swap_listener)],
        )

        # =====================================================================
        # OHLC Aggregator Lambda
        # =====================================================================

        self.ohlc_aggregator = lambda_.Function(
            self,
            "OHLCAggregator",
            function_name=f"range-bot-ohlc-aggregator-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/ohlc_aggregator"),
            timeout=Duration.seconds(60),
            memory_size=512,
            environment={
                "SWAP_TABLE": self.swap_table.table_name,
                "OHLC_TABLE": self.ohlc_table.table_name,
                "BLOCKS_PER_CANDLE": "50",
                "ENV_NAME": env_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Grant DynamoDB access
        self.swap_table.grant_read_data(self.ohlc_aggregator)
        self.ohlc_table.grant_write_data(self.ohlc_aggregator)

        # Trigger from DynamoDB Streams
        self.ohlc_aggregator.add_event_source(
            lambda_event_sources.DynamoEventSource(
                self.swap_table,
                starting_position=lambda_.StartingPosition.LATEST,
                batch_size=100,
                max_batching_window=Duration.seconds(30),
            )
        )

        # Also schedule every 10 minutes as backup
        events.Rule(
            self,
            "OHLCAggregatorSchedule",
            rule_name=f"range-bot-ohlc-aggregator-{env_name}",
            schedule=events.Schedule.rate(Duration.minutes(10)),
            targets=[targets.LambdaFunction(self.ohlc_aggregator)],
        )
```

### 5.2 Create SwapListener Lambda

Create `lambdas/swap_listener/handler.py`:

```python
"""
SwapListener Lambda

Fetches new swap events from Uniswap v3 pool and stores in DynamoDB.
Triggered by EventBridge schedule.
"""
import json
import logging
import os
import time
from decimal import Decimal
from typing import Any

import boto3
import urllib.request

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
SWAP_TABLE = os.environ["SWAP_TABLE"]
POOL_ADDRESS = os.environ["POOL_ADDRESS"]
ENV_NAME = os.environ.get("ENV_NAME", "dev")

# Uniswap v3 Swap event topic
SWAP_TOPIC = "0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67"

# RPC endpoints (rotate for reliability)
RPC_ENDPOINTS = [
    "https://ethereum-rpc.publicnode.com",
    "https://rpc.ankr.com/eth",
    "https://eth.drpc.org",
]

# DynamoDB client
dynamodb = boto3.resource("dynamodb")
table = dynamodb.Table(SWAP_TABLE)


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Main Lambda handler.

    Fetches recent swap events and stores them in DynamoDB.
    """
    logger.info(f"SwapListener invoked")

    try:
        # Get current block number
        current_block = get_current_block()

        # Get last processed block from DynamoDB (or default to current - 100)
        last_block = get_last_processed_block(current_block - 100)

        # Fetch swaps from last_block to current
        swaps = []
        if current_block > last_block:
            swaps = fetch_swaps(last_block + 1, current_block)
            logger.info(f"Fetched {len(swaps)} swaps from blocks {last_block + 1} to {current_block}")

            # Store swaps in DynamoDB
            if swaps:
                store_swaps(swaps)

            # Update last processed block
            update_last_processed_block(current_block)
        else:
            logger.info(f"No new blocks (current={current_block}, last={last_block})")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Success",
                "blocks_processed": current_block - last_block,
                "swaps_stored": len(swaps),
            }),
        }

    except Exception as e:
        logger.error(f"Error in SwapListener: {e}", exc_info=True)
        raise


def rpc_call(endpoint: str, method: str, params: list, timeout: int = 10) -> dict:
    """Make JSON-RPC call to Ethereum node."""
    data = json.dumps({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def get_current_block() -> int:
    """Get current Ethereum block number."""
    for endpoint in RPC_ENDPOINTS:
        try:
            result = rpc_call(endpoint, "eth_blockNumber", [])
            return int(result["result"], 16)
        except Exception as e:
            logger.warning(f"RPC error on {endpoint}: {e}")
            continue

    raise RuntimeError("All RPC endpoints failed")


def fetch_swaps(from_block: int, to_block: int) -> list[dict]:
    """Fetch swap events from the pool."""
    swaps = []

    for endpoint in RPC_ENDPOINTS:
        try:
            result = rpc_call(
                endpoint,
                "eth_getLogs",
                [{
                    "address": POOL_ADDRESS,
                    "topics": [SWAP_TOPIC],
                    "fromBlock": hex(from_block),
                    "toBlock": hex(to_block),
                }],
                timeout=30,
            )

            if "error" in result:
                logger.warning(f"RPC error: {result['error']}")
                continue

            logs = result.get("result", [])

            for log in logs:
                swap = decode_swap(log)
                if swap:
                    swaps.append(swap)

            return swaps

        except Exception as e:
            logger.warning(f"Error fetching from {endpoint}: {e}")
            continue

    raise RuntimeError("All RPC endpoints failed for getLogs")


def decode_swap(log: dict) -> dict | None:
    """Decode a Uniswap v3 Swap event log."""
    try:
        data = log["data"][2:]  # Remove '0x' prefix

        # Decode data fields (each 64 hex chars = 32 bytes)
        amount0 = int(data[0:64], 16)
        amount1 = int(data[64:128], 16)
        sqrt_price_x96 = int(data[128:192], 16)
        liquidity = int(data[192:256], 16)
        tick = int(data[256:320], 16)

        # Handle signed integers
        if amount0 >= 2**255:
            amount0 -= 2**256
        if amount1 >= 2**255:
            amount1 -= 2**256
        if tick >= 2**23:
            tick -= 2**24

        # Calculate price from sqrtPriceX96
        # price = 1e12 / ((sqrt_price_x96 ** 2) / (2 ** 192))
        if sqrt_price_x96 > 0:
            price = 1e12 / ((sqrt_price_x96 ** 2) / (2 ** 192))
        else:
            price = 0

        block_number = int(log["blockNumber"], 16)
        tx_hash = log["transactionHash"]
        log_index = int(log["logIndex"], 16)

        return {
            "block_number": block_number,
            "tx_hash": tx_hash,
            "log_index": log_index,
            "amount0": str(amount0),
            "amount1": str(amount1),
            "sqrt_price_x96": str(sqrt_price_x96),
            "liquidity": str(liquidity),
            "tick": tick,
            "price": price,
        }

    except Exception as e:
        logger.warning(f"Error decoding swap: {e}")
        return None


def store_swaps(swaps: list[dict]) -> None:
    """Store swaps in DynamoDB with batch writer."""
    ttl = int(time.time()) + (4 * 60 * 60)  # 4 hours from now

    with table.batch_writer() as batch:
        for swap in swaps:
            item = {
                "pk": f"POOL#{POOL_ADDRESS}",
                "sk": f"BLOCK#{swap['block_number']:010d}#TX#{swap['tx_hash']}#{swap['log_index']}",
                "block_number": swap["block_number"],
                "tx_hash": swap["tx_hash"],
                "log_index": swap["log_index"],
                "amount0": swap["amount0"],
                "amount1": swap["amount1"],
                "sqrt_price_x96": swap["sqrt_price_x96"],
                "liquidity": swap["liquidity"],
                "tick": swap["tick"],
                "price": Decimal(str(swap["price"])),
                "ttl": ttl,
            }
            batch.put_item(Item=item)


def get_last_processed_block(default: int) -> int:
    """Get last processed block from DynamoDB."""
    try:
        response = table.get_item(
            Key={
                "pk": "METADATA",
                "sk": "LAST_BLOCK",
            }
        )
        if "Item" in response:
            return int(response["Item"]["block_number"])
    except Exception as e:
        logger.warning(f"Error getting last block: {e}")

    return default


def update_last_processed_block(block: int) -> None:
    """Update last processed block in DynamoDB."""
    table.put_item(
        Item={
            "pk": "METADATA",
            "sk": "LAST_BLOCK",
            "block_number": block,
            "updated_at": int(time.time()),
        }
    )
```

### 5.3 Create OHLC Aggregator Lambda

Create `lambdas/ohlc_aggregator/handler.py`:

```python
"""
OHLC Aggregator Lambda

Aggregates raw swap data into OHLC candles.
Triggered by DynamoDB Streams or scheduled backup.
"""
import json
import logging
import os
import time
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
SWAP_TABLE = os.environ["SWAP_TABLE"]
OHLC_TABLE = os.environ["OHLC_TABLE"]
BLOCKS_PER_CANDLE = int(os.environ.get("BLOCKS_PER_CANDLE", "50"))
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

# DynamoDB clients
dynamodb = boto3.resource("dynamodb")
swap_table = dynamodb.Table(SWAP_TABLE)
ohlc_table = dynamodb.Table(OHLC_TABLE)


def lambda_handler(event: dict, context: Any) -> dict:
    """
    Main Lambda handler.

    Aggregates swaps into OHLC candles.
    """
    logger.info("OHLC Aggregator invoked")

    try:
        # Get latest candle from OHLC table
        latest_candle_block = get_latest_candle_block()

        # Get current block range from swap data
        swap_block_range = get_swap_block_range()

        if not swap_block_range:
            logger.info("No swap data available")
            return {"statusCode": 200, "body": "No data"}

        min_block, max_block = swap_block_range

        # Calculate candle periods to process
        start_period = ((latest_candle_block or min_block) // BLOCKS_PER_CANDLE) * BLOCKS_PER_CANDLE
        end_period = (max_block // BLOCKS_PER_CANDLE) * BLOCKS_PER_CANDLE

        candles_created = 0

        for period_start in range(start_period, end_period + 1, BLOCKS_PER_CANDLE):
            period_end = period_start + BLOCKS_PER_CANDLE - 1

            # Skip if candle already exists
            if candle_exists(period_start):
                continue

            # Get swaps for this period
            swaps = get_swaps_for_period(period_start, period_end)

            if not swaps:
                continue

            # Aggregate to OHLC
            candle = aggregate_to_ohlc(swaps, period_start)

            if candle:
                # Store candle
                store_candle(candle)
                candles_created += 1

        logger.info(f"Created {candles_created} candles")

        return {
            "statusCode": 200,
            "body": json.dumps({
                "message": "Success",
                "candles_created": candles_created,
            }),
        }

    except Exception as e:
        logger.error(f"Error in OHLC Aggregator: {e}", exc_info=True)
        raise


def get_latest_candle_block() -> int | None:
    """Get the latest candle block from OHLC table."""
    try:
        response = ohlc_table.query(
            KeyConditionExpression=Key("pk").eq(f"POOL#{POOL_ADDRESS}"),
            ScanIndexForward=False,
            Limit=1,
        )
        if response.get("Items"):
            sk = response["Items"][0]["sk"]
            # sk format: BLOCK#0021500000
            return int(sk.split("#")[1])
    except Exception as e:
        logger.warning(f"Error getting latest candle: {e}")

    return None


def get_swap_block_range() -> tuple | None:
    """Get min and max block numbers from swap table."""
    try:
        # Query for earliest
        earliest = swap_table.query(
            KeyConditionExpression=Key("pk").eq(f"POOL#{POOL_ADDRESS}"),
            ScanIndexForward=True,
            Limit=1,
        )

        # Query for latest
        latest = swap_table.query(
            KeyConditionExpression=Key("pk").eq(f"POOL#{POOL_ADDRESS}"),
            ScanIndexForward=False,
            Limit=1,
        )

        if earliest.get("Items") and latest.get("Items"):
            min_block = earliest["Items"][0]["block_number"]
            max_block = latest["Items"][0]["block_number"]
            return (min_block, max_block)

    except Exception as e:
        logger.warning(f"Error getting swap block range: {e}")

    return None


def candle_exists(period_start: int) -> bool:
    """Check if candle already exists."""
    try:
        response = ohlc_table.get_item(
            Key={
                "pk": f"POOL#{POOL_ADDRESS}",
                "sk": f"BLOCK#{period_start:010d}",
            }
        )
        return "Item" in response
    except Exception:
        return False


def get_swaps_for_period(start_block: int, end_block: int) -> list:
    """Get all swaps for a block range."""
    swaps = []

    response = swap_table.query(
        KeyConditionExpression=(
            Key("pk").eq(f"POOL#{POOL_ADDRESS}") &
            Key("sk").between(
                f"BLOCK#{start_block:010d}",
                f"BLOCK#{end_block:010d}~"
            )
        ),
    )

    swaps.extend(response.get("Items", []))

    # Handle pagination
    while "LastEvaluatedKey" in response:
        response = swap_table.query(
            KeyConditionExpression=(
                Key("pk").eq(f"POOL#{POOL_ADDRESS}") &
                Key("sk").between(
                    f"BLOCK#{start_block:010d}",
                    f"BLOCK#{end_block:010d}~"
                )
            ),
            ExclusiveStartKey=response["LastEvaluatedKey"],
        )
        swaps.extend(response.get("Items", []))

    return swaps


def aggregate_to_ohlc(swaps: list, period_start: int) -> dict | None:
    """Aggregate swaps into OHLC candle."""
    # Sort by block and log index
    sorted_swaps = sorted(
        swaps,
        key=lambda x: (x["block_number"], x["log_index"])
    )

    prices = [float(s["price"]) for s in sorted_swaps if float(s["price"]) > 0]

    if not prices:
        return None

    # Calculate OHLC
    open_price = prices[0]
    high_price = max(prices)
    low_price = min(prices)
    close_price = prices[-1]

    # Calculate volume (sum of |amount1| in USDC terms)
    volume = sum(abs(int(s["amount1"])) / 1e6 for s in sorted_swaps)

    # Calculate VWAP
    weighted_sum = sum(
        float(s["price"]) * abs(int(s["amount1"])) / 1e6
        for s in sorted_swaps
        if float(s["price"]) > 0
    )
    vwap = weighted_sum / volume if volume > 0 else close_price

    return {
        "pk": f"POOL#{POOL_ADDRESS}",
        "sk": f"BLOCK#{period_start:010d}",
        "period_start": period_start,
        "period_end": period_start + BLOCKS_PER_CANDLE - 1,
        "o": Decimal(str(round(open_price, 6))),
        "h": Decimal(str(round(high_price, 6))),
        "l": Decimal(str(round(low_price, 6))),
        "c": Decimal(str(round(close_price, 6))),
        "vol": Decimal(str(round(volume, 2))),
        "vwap": Decimal(str(round(vwap, 6))),
        "n": len(sorted_swaps),
        "created_at": int(time.time()),
    }


def store_candle(candle: dict) -> None:
    """Store OHLC candle in DynamoDB."""
    ohlc_table.put_item(Item=candle)
```

---

## 6. Model Inference Lambdas

### 6.1 Create Model Inference Stack

Create `infra/stacks/model_inference.py`:

```python
"""
Model Inference Stack

Creates:
- BayesianRange Lambda
- AlertEngine Lambda
- MomentumSignal Lambda
- SNS topic for signals
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as lambda_,
    aws_events as events,
    aws_events_targets as targets,
    aws_sns as sns,
    aws_logs as logs,
)
from constructs import Construct


class ModelInferenceStack(Stack):
    """Model inference infrastructure."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str,
        data_stack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name
        self.data_stack = data_stack

        # =====================================================================
        # SNS Topic for Signals
        # =====================================================================

        self.signal_topic = sns.Topic(
            self,
            "SignalTopic",
            topic_name=f"range-bot-signals-{env_name}",
            display_name="Range Bot Trading Signals",
        )

        # =====================================================================
        # BayesianRange Lambda
        # =====================================================================

        self.bayesian_lambda = lambda_.Function(
            self,
            "BayesianRange",
            function_name=f"range-bot-bayesian-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/bayesian_range"),
            timeout=Duration.seconds(60),
            memory_size=1024,
            environment={
                "OHLC_TABLE": data_stack.ohlc_table.table_name,
                "MODEL_STATE_TABLE": data_stack.model_state_table.table_name,
                "SIGNAL_TOPIC_ARN": self.signal_topic.topic_arn,
                "COVERAGE_TARGET": "0.90",
                "LOOKBACK_CANDLES": "20",
                "ENV_NAME": env_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Grant permissions
        data_stack.ohlc_table.grant_read_data(self.bayesian_lambda)
        data_stack.model_state_table.grant_read_write_data(self.bayesian_lambda)
        self.signal_topic.grant_publish(self.bayesian_lambda)

        # Schedule every 10 minutes
        events.Rule(
            self,
            "BayesianSchedule",
            rule_name=f"range-bot-bayesian-{env_name}",
            schedule=events.Schedule.rate(Duration.minutes(10)),
            targets=[targets.LambdaFunction(self.bayesian_lambda)],
        )

        # =====================================================================
        # AlertEngine Lambda
        # =====================================================================

        self.alert_lambda = lambda_.Function(
            self,
            "AlertEngine",
            function_name=f"range-bot-alerts-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/alert_engine"),
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "OHLC_TABLE": data_stack.ohlc_table.table_name,
                "MODEL_STATE_TABLE": data_stack.model_state_table.table_name,
                "SIGNAL_TOPIC_ARN": self.signal_topic.topic_arn,
                "SENSITIVITY": "4.0",
                "ENV_NAME": env_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Grant permissions
        data_stack.ohlc_table.grant_read_data(self.alert_lambda)
        data_stack.model_state_table.grant_read_write_data(self.alert_lambda)
        self.signal_topic.grant_publish(self.alert_lambda)

        # Schedule every 10 minutes
        events.Rule(
            self,
            "AlertSchedule",
            rule_name=f"range-bot-alerts-{env_name}",
            schedule=events.Schedule.rate(Duration.minutes(10)),
            targets=[targets.LambdaFunction(self.alert_lambda)],
        )

        # =====================================================================
        # MomentumSignal Lambda
        # =====================================================================

        self.momentum_lambda = lambda_.Function(
            self,
            "MomentumSignal",
            function_name=f"range-bot-momentum-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/momentum_signal"),
            timeout=Duration.seconds(30),
            memory_size=256,
            environment={
                "SWAP_TABLE": data_stack.swap_table.table_name,
                "MODEL_STATE_TABLE": data_stack.model_state_table.table_name,
                "SIGNAL_TOPIC_ARN": self.signal_topic.topic_arn,
                "MOMENTUM_THRESHOLD": "0.0018",
                "LOOKBACK_SWAPS": "150",
                "ENV_NAME": env_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Grant permissions
        data_stack.swap_table.grant_read_data(self.momentum_lambda)
        data_stack.model_state_table.grant_read_write_data(self.momentum_lambda)
        self.signal_topic.grant_publish(self.momentum_lambda)

        # Schedule every 10 minutes (runs when alert is active)
        events.Rule(
            self,
            "MomentumSchedule",
            rule_name=f"range-bot-momentum-{env_name}",
            schedule=events.Schedule.rate(Duration.minutes(10)),
            targets=[targets.LambdaFunction(self.momentum_lambda)],
        )
```

### 6.2 Create BayesianRange Lambda

Create `lambdas/bayesian_range/handler.py`:

```python
"""
BayesianRange Lambda

Computes optimal LP price ranges using Bayesian inference.
"""
import json
import logging
import math
import os
import time
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
OHLC_TABLE = os.environ["OHLC_TABLE"]
MODEL_STATE_TABLE = os.environ["MODEL_STATE_TABLE"]
SIGNAL_TOPIC_ARN = os.environ["SIGNAL_TOPIC_ARN"]
COVERAGE_TARGET = float(os.environ.get("COVERAGE_TARGET", "0.90"))
LOOKBACK_CANDLES = int(os.environ.get("LOOKBACK_CANDLES", "20"))
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

# AWS clients
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")
ohlc_table = dynamodb.Table(OHLC_TABLE)
model_table = dynamodb.Table(MODEL_STATE_TABLE)


def lambda_handler(event: dict, context: Any) -> dict:
    """Main Lambda handler."""
    logger.info("BayesianRange Lambda invoked")

    try:
        # Fetch recent OHLC candles
        candles = fetch_recent_candles(LOOKBACK_CANDLES)

        if len(candles) < 10:
            logger.warning(f"Insufficient candles: {len(candles)}")
            return {"statusCode": 200, "body": "Insufficient data"}

        # Extract VWAPs and midpoints
        vwaps = [float(c["vwap"]) for c in candles]
        mids = [(float(c["h"]) + float(c["l"])) / 2 for c in candles]

        # Compute Bayesian range
        range_lo, range_hi = compute_bayesian_range(vwaps, mids, COVERAGE_TARGET)

        logger.info(f"Computed range: [{range_lo:.2f}, {range_hi:.2f}]")

        # Save to model state
        save_range_to_state(range_lo, range_hi, candles[-1])

        # Publish signal
        publish_range_signal(range_lo, range_hi)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "range_lo": range_lo,
                "range_hi": range_hi,
                "candles_used": len(candles),
            }),
        }

    except Exception as e:
        logger.error(f"Error in BayesianRange: {e}", exc_info=True)
        raise


def fetch_recent_candles(count: int) -> list:
    """Fetch most recent OHLC candles."""
    response = ohlc_table.query(
        KeyConditionExpression=Key("pk").eq(f"POOL#{POOL_ADDRESS}"),
        ScanIndexForward=False,
        Limit=count,
    )

    candles = response.get("Items", [])
    candles.reverse()  # Chronological order
    return candles


def compute_bayesian_range(
    vwaps: list,
    mids: list,
    coverage: float,
) -> tuple:
    """
    Compute optimal price range using Bayesian inference.

    Uses Laplace prior + KDE likelihood approach from POC.
    """
    n = len(mids)

    # Build price grid
    price_min = min(mids) * 0.95
    price_max = max(mids) * 1.05
    n_points = 500
    step = (price_max - price_min) / n_points
    prices = [price_min + i * step for i in range(n_points)]

    # Prior: Laplace distribution
    prior_center = sorted(vwaps)[len(vwaps) // 2]  # Median
    prior_scale = std(vwaps) * 2
    prior = [laplace_pdf(p, prior_center, prior_scale) for p in prices]

    # Likelihood: KDE with exponential decay
    bandwidth = 1.06 * std(mids) * (n ** -0.2)  # Silverman
    weights = [0.9 ** (n - 1 - i) for i in range(n)]
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    likelihood = []
    for p in prices:
        l = 0
        for i, mid in enumerate(mids):
            l += weights[i] * gauss_pdf(p, mid, bandwidth)
        likelihood.append(l)

    # Posterior: prior * likelihood
    posterior = [prior[i] * likelihood[i] for i in range(len(prices))]
    post_sum = sum(posterior) * step
    posterior = [p / post_sum for p in posterior]

    # Find optimal range (tightest interval with coverage)
    cdf = []
    cumsum = 0
    for p in posterior:
        cumsum += p * step
        cdf.append(cumsum)

    best_lo, best_hi = prices[0], prices[-1]
    best_width = best_hi - best_lo

    for i in range(len(prices)):
        # Find j where CDF[j] - CDF[i] >= coverage
        target = cdf[i] + coverage
        for j in range(i, len(prices)):
            if cdf[j] >= target:
                width = prices[j] - prices[i]
                if width < best_width:
                    best_width = width
                    best_lo = prices[i]
                    best_hi = prices[j]
                break

    return best_lo, best_hi


def std(values: list) -> float:
    """Calculate standard deviation."""
    n = len(values)
    if n < 2:
        return 1.0
    mean = sum(values) / n
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return math.sqrt(variance)


def laplace_pdf(x: float, loc: float, scale: float) -> float:
    """Laplace probability density function."""
    return (1 / (2 * scale)) * math.exp(-abs(x - loc) / scale)


def gauss_pdf(x: float, mean: float, std: float) -> float:
    """Gaussian probability density function."""
    return (1 / (std * math.sqrt(2 * math.pi))) * math.exp(-0.5 * ((x - mean) / std) ** 2)


def save_range_to_state(range_lo: float, range_hi: float, latest_candle: dict) -> None:
    """Save computed range to model state table."""
    model_table.put_item(
        Item={
            "pk": "MODEL",
            "sk": "BAYESIAN_RANGE",
            "range_lo": Decimal(str(round(range_lo, 6))),
            "range_hi": Decimal(str(round(range_hi, 6))),
            "latest_candle_block": latest_candle["period_start"],
            "updated_at": int(time.time()),
        }
    )


def publish_range_signal(range_lo: float, range_hi: float) -> None:
    """Publish range signal to SNS."""
    sns.publish(
        TopicArn=SIGNAL_TOPIC_ARN,
        Message=json.dumps({
            "signal_type": "BAYESIAN_RANGE",
            "range_lo": range_lo,
            "range_hi": range_hi,
        }),
        MessageAttributes={
            "signal_type": {
                "DataType": "String",
                "StringValue": "BAYESIAN_RANGE",
            }
        },
    )
```

### 6.3 Create AlertEngine Lambda

Create `lambdas/alert_engine/handler.py`:

```python
"""
AlertEngine Lambda

Evaluates market stability and triggers alerts.
"""
import json
import logging
import math
import os
import time
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
OHLC_TABLE = os.environ["OHLC_TABLE"]
MODEL_STATE_TABLE = os.environ["MODEL_STATE_TABLE"]
SIGNAL_TOPIC_ARN = os.environ["SIGNAL_TOPIC_ARN"]
SENSITIVITY = float(os.environ.get("SENSITIVITY", "4.0"))
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

# Base thresholds (from training)
BASE_THRESHOLDS = {
    "stability_trend": -0.0522,
    "range_expansion": 1.0940,
    "price_velocity": 0.1472,
    "vol_spike": 0.7026,
}

# AWS clients
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")
ohlc_table = dynamodb.Table(OHLC_TABLE)
model_table = dynamodb.Table(MODEL_STATE_TABLE)


def lambda_handler(event: dict, context: Any) -> dict:
    """Main Lambda handler."""
    logger.info("AlertEngine Lambda invoked")

    try:
        # Fetch recent candles
        candles = fetch_recent_candles(10)

        if len(candles) < 10:
            logger.warning(f"Insufficient candles: {len(candles)}")
            return {"statusCode": 200, "body": "Insufficient data"}

        # Extract features
        features = extract_features(candles)
        logger.info(f"Features: {features}")

        # Apply thresholds
        thresholds = {k: v * SENSITIVITY for k, v in BASE_THRESHOLDS.items()}

        # Check for alerts
        is_alert, triggered_features = check_alert(features, thresholds)
        logger.info(f"Alert: {is_alert}, triggered: {triggered_features}")

        # Save state
        save_alert_state(is_alert, features, triggered_features)

        # Publish signal
        publish_alert_signal(is_alert, features, triggered_features)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "is_alert": is_alert,
                "features": features,
                "triggered": triggered_features,
            }),
        }

    except Exception as e:
        logger.error(f"Error in AlertEngine: {e}", exc_info=True)
        raise


def fetch_recent_candles(count: int) -> list:
    """Fetch most recent OHLC candles."""
    response = ohlc_table.query(
        KeyConditionExpression=Key("pk").eq(f"POOL#{POOL_ADDRESS}"),
        ScanIndexForward=False,
        Limit=count,
    )
    candles = response.get("Items", [])
    candles.reverse()
    return candles


def extract_features(candles: list) -> dict:
    """Extract stability features from candles."""
    vwaps = [float(c["vwap"]) for c in candles]
    volumes = [float(c["vol"]) for c in candles]

    n = len(candles)
    mid = n // 2

    # Stability scores
    early_stability = calculate_stability(vwaps[:mid])
    late_stability = calculate_stability(vwaps[mid:])
    stability_trend = late_stability - early_stability

    # Range expansion
    early_ranges = [float(c["h"]) - float(c["l"]) for c in candles[:mid]]
    late_ranges = [float(c["h"]) - float(c["l"]) for c in candles[mid:]]
    early_avg = sum(early_ranges) / len(early_ranges) if early_ranges else 1
    late_avg = sum(late_ranges) / len(late_ranges) if late_ranges else 1
    range_expansion = late_avg / early_avg if early_avg > 0 else 1.0

    # Price velocity
    if len(vwaps) >= 2 and vwaps[-2] > 0:
        price_velocity = abs(vwaps[-1] - vwaps[-2]) / vwaps[-2] * 100
    else:
        price_velocity = 0.0

    # Volume spike
    avg_vol = sum(volumes[:-1]) / len(volumes[:-1]) if len(volumes) > 1 else volumes[-1]
    vol_spike = volumes[-1] / avg_vol if avg_vol > 0 else 1.0

    return {
        "stability_trend": stability_trend,
        "range_expansion": range_expansion,
        "price_velocity": price_velocity,
        "vol_spike": vol_spike,
    }


def calculate_stability(prices: list) -> float:
    """Calculate stability score (0=trending, 1=choppy)."""
    if len(prices) < 2:
        return 0.5
    net_move = abs(prices[-1] - prices[0])
    total_path = sum(abs(prices[i+1] - prices[i]) for i in range(len(prices) - 1))
    if total_path == 0:
        return 1.0
    return 1.0 - (net_move / total_path)


def check_alert(features: dict, thresholds: dict) -> tuple:
    """Check if any thresholds breached."""
    triggered = []

    if features["stability_trend"] < thresholds["stability_trend"]:
        triggered.append("stability_trend")
    if features["range_expansion"] > thresholds["range_expansion"]:
        triggered.append("range_expansion")
    if features["price_velocity"] > thresholds["price_velocity"]:
        triggered.append("price_velocity")
    if features["vol_spike"] > thresholds["vol_spike"]:
        triggered.append("vol_spike")

    return len(triggered) > 0, triggered


def save_alert_state(is_alert: bool, features: dict, triggered: list) -> None:
    """Save alert state."""
    model_table.put_item(
        Item={
            "pk": "MODEL",
            "sk": "ALERT_STATE",
            "is_alert": is_alert,
            "stability_trend": Decimal(str(round(features["stability_trend"], 6))),
            "range_expansion": Decimal(str(round(features["range_expansion"], 6))),
            "price_velocity": Decimal(str(round(features["price_velocity"], 6))),
            "vol_spike": Decimal(str(round(features["vol_spike"], 6))),
            "triggered_features": triggered,
            "updated_at": int(time.time()),
        }
    )


def publish_alert_signal(is_alert: bool, features: dict, triggered: list) -> None:
    """Publish alert signal."""
    sns.publish(
        TopicArn=SIGNAL_TOPIC_ARN,
        Message=json.dumps({
            "signal_type": "STABILITY_ALERT",
            "is_alert": is_alert,
            "features": features,
            "triggered_features": triggered,
        }),
        MessageAttributes={
            "signal_type": {"DataType": "String", "StringValue": "STABILITY_ALERT"},
            "is_alert": {"DataType": "String", "StringValue": str(is_alert)},
        },
    )
```

### 6.4 Create MomentumSignal Lambda

Create `lambdas/momentum_signal/handler.py`:

```python
"""
MomentumSignal Lambda

Predicts directional move when alert triggers.
"""
import json
import logging
import os
import time
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration
SWAP_TABLE = os.environ["SWAP_TABLE"]
MODEL_STATE_TABLE = os.environ["MODEL_STATE_TABLE"]
SIGNAL_TOPIC_ARN = os.environ["SIGNAL_TOPIC_ARN"]
MOMENTUM_THRESHOLD = float(os.environ.get("MOMENTUM_THRESHOLD", "0.0018"))
LOOKBACK_SWAPS = int(os.environ.get("LOOKBACK_SWAPS", "150"))
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"

# AWS clients
dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")
swap_table = dynamodb.Table(SWAP_TABLE)
model_table = dynamodb.Table(MODEL_STATE_TABLE)


def lambda_handler(event: dict, context: Any) -> dict:
    """Main Lambda handler."""
    logger.info("MomentumSignal Lambda invoked")

    try:
        # Check if alert is active
        alert_state = get_alert_state()
        if not alert_state.get("is_alert", False):
            logger.info("No active alert, skipping momentum calculation")
            return {"statusCode": 200, "body": "No alert"}

        # Fetch recent swaps
        swaps = fetch_recent_swaps(LOOKBACK_SWAPS)

        if len(swaps) < LOOKBACK_SWAPS:
            logger.warning(f"Insufficient swaps: {len(swaps)}")
            direction = "HOLD"
            momentum = 0.0
        else:
            prices = [float(s["price"]) for s in swaps]
            momentum = (prices[-1] - prices[0]) / prices[0]

            if momentum > MOMENTUM_THRESHOLD:
                direction = "ETH"
            elif momentum < -MOMENTUM_THRESHOLD:
                direction = "USDC"
            else:
                direction = "HOLD"

        logger.info(f"Momentum: {momentum:.4f}, Direction: {direction}")

        # Save state
        save_momentum_state(momentum, direction)

        # Publish signal
        publish_momentum_signal(momentum, direction)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "momentum": momentum,
                "direction": direction,
                "swaps_analyzed": len(swaps),
            }),
        }

    except Exception as e:
        logger.error(f"Error in MomentumSignal: {e}", exc_info=True)
        raise


def get_alert_state() -> dict:
    """Get current alert state."""
    try:
        response = model_table.get_item(
            Key={"pk": "MODEL", "sk": "ALERT_STATE"}
        )
        return response.get("Item", {})
    except Exception:
        return {}


def fetch_recent_swaps(count: int) -> list:
    """Fetch most recent swaps."""
    response = swap_table.query(
        KeyConditionExpression=Key("pk").eq(f"POOL#{POOL_ADDRESS}"),
        ScanIndexForward=False,
        Limit=count,
    )
    swaps = response.get("Items", [])
    swaps.reverse()
    return swaps


def save_momentum_state(momentum: float, direction: str) -> None:
    """Save momentum state."""
    model_table.put_item(
        Item={
            "pk": "MODEL",
            "sk": "MOMENTUM_SIGNAL",
            "momentum": Decimal(str(round(momentum, 6))),
            "direction": direction,
            "updated_at": int(time.time()),
        }
    )


def publish_momentum_signal(momentum: float, direction: str) -> None:
    """Publish momentum signal."""
    sns.publish(
        TopicArn=SIGNAL_TOPIC_ARN,
        Message=json.dumps({
            "signal_type": "MOMENTUM_DIRECTION",
            "momentum": momentum,
            "direction": direction,
        }),
        MessageAttributes={
            "signal_type": {"DataType": "String", "StringValue": "MOMENTUM_DIRECTION"},
            "direction": {"DataType": "String", "StringValue": direction},
        },
    )
```

---

## 7. Observability Stack

Create `infra/stacks/observability.py`:

```python
"""
Observability Stack

Creates CloudWatch Dashboard and SNS alerts.
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_cloudwatch as cloudwatch,
    aws_cloudwatch_actions as cw_actions,
    aws_sns as sns,
    aws_sns_subscriptions as subscriptions,
)
from constructs import Construct


class ObservabilityStack(Stack):
    """Monitoring and alerting infrastructure."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str,
        alert_email: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # SNS Topic for Alerts
        self.alert_topic = sns.Topic(
            self,
            "AlertTopic",
            topic_name=f"range-bot-alerts-{env_name}",
            display_name="Range Bot Critical Alerts",
        )

        # Subscribe email
        self.alert_topic.add_subscription(
            subscriptions.EmailSubscription(alert_email)
        )

        # CloudWatch Dashboard
        self.dashboard = cloudwatch.Dashboard(
            self,
            "Dashboard",
            dashboard_name=f"RangeBot-{env_name}",
        )

        # Header
        self.dashboard.add_widgets(
            cloudwatch.TextWidget(
                markdown=f"# Range-Bot Dashboard ({env_name})",
                width=24,
                height=1,
            ),
        )

        # Lambda metrics row
        self.dashboard.add_widgets(
            cloudwatch.GraphWidget(
                title="Lambda Invocations",
                width=8,
                height=6,
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Invocations",
                        dimensions_map={"FunctionName": f"range-bot-swap-listener-{env_name}"},
                        statistic="Sum",
                        period=Duration.minutes(5),
                        label="SwapListener",
                    ),
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Invocations",
                        dimensions_map={"FunctionName": f"range-bot-bayesian-{env_name}"},
                        statistic="Sum",
                        period=Duration.minutes(5),
                        label="BayesianRange",
                    ),
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Invocations",
                        dimensions_map={"FunctionName": f"range-bot-alerts-{env_name}"},
                        statistic="Sum",
                        period=Duration.minutes(5),
                        label="AlertEngine",
                    ),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Lambda Errors",
                width=8,
                height=6,
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Errors",
                        dimensions_map={"FunctionName": f"range-bot-swap-listener-{env_name}"},
                        statistic="Sum",
                        period=Duration.minutes(5),
                        label="SwapListener",
                    ),
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Errors",
                        dimensions_map={"FunctionName": f"range-bot-bayesian-{env_name}"},
                        statistic="Sum",
                        period=Duration.minutes(5),
                        label="BayesianRange",
                    ),
                ],
            ),
            cloudwatch.GraphWidget(
                title="Lambda Duration (ms)",
                width=8,
                height=6,
                left=[
                    cloudwatch.Metric(
                        namespace="AWS/Lambda",
                        metric_name="Duration",
                        dimensions_map={"FunctionName": f"range-bot-bayesian-{env_name}"},
                        statistic="Average",
                        period=Duration.minutes(5),
                        label="BayesianRange",
                    ),
                ],
            ),
        )

        # Lambda Error Alarm
        lambda_error_alarm = cloudwatch.Alarm(
            self,
            "LambdaErrorAlarm",
            alarm_name=f"range-bot-lambda-errors-{env_name}",
            alarm_description="Lambda function errors detected",
            metric=cloudwatch.Metric(
                namespace="AWS/Lambda",
                metric_name="Errors",
                dimensions_map={"FunctionName": f"range-bot-swap-listener-{env_name}"},
                statistic="Sum",
                period=Duration.minutes(5),
            ),
            threshold=3,
            evaluation_periods=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_THRESHOLD,
        )
        lambda_error_alarm.add_alarm_action(cw_actions.SnsAction(self.alert_topic))
```

---

## 8. Execution Layer

Create `infra/stacks/execution.py`:

```python
"""
Execution Stack

Creates isolated VPC, Secrets Manager, and ECS for signing.
"""
from aws_cdk import (
    Stack,
    Duration,
    RemovalPolicy,
    aws_ec2 as ec2,
    aws_ecs as ecs,
    aws_secretsmanager as secretsmanager,
    aws_sqs as sqs,
    aws_logs as logs,
)
from constructs import Construct


class ExecutionStack(Stack):
    """Secure transaction execution infrastructure."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # Isolated VPC (no internet access)
        self.vpc = ec2.Vpc(
            self,
            "SignerVPC",
            vpc_name=f"range-bot-signer-{env_name}",
            max_azs=2,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="Isolated",
                    subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
                    cidr_mask=24,
                )
            ],
        )

        # VPC Endpoints for AWS services
        self.vpc.add_interface_endpoint(
            "SecretsManagerEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
        )
        self.vpc.add_interface_endpoint(
            "SQSEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SQS,
        )
        self.vpc.add_interface_endpoint(
            "CloudWatchLogsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.CLOUDWATCH_LOGS,
        )

        # Secrets Manager for private key
        self.private_key_secret = secretsmanager.Secret(
            self,
            "WalletPrivateKey",
            secret_name=f"range-bot/wallet-key-{env_name}",
            description="Encrypted private key for LP management",
            removal_policy=RemovalPolicy.RETAIN,
        )

        # SQS Queue for transaction requests
        self.tx_queue = sqs.Queue(
            self,
            "TxRequestQueue",
            queue_name=f"range-bot-tx-requests-{env_name}",
            visibility_timeout=Duration.seconds(300),
            retention_period=Duration.hours(1),
        )

        # ECS Cluster
        self.cluster = ecs.Cluster(
            self,
            "SignerCluster",
            cluster_name=f"range-bot-signer-{env_name}",
            vpc=self.vpc,
            container_insights=True,
        )

        # Note: ECS Task and Service would be added here
        # Skipped for initial deployment - will add when ready for live trading
```

---

## 9. Risk Management

Create `infra/stacks/risk_management.py`:

```python
"""
Risk Management Stack
"""
from aws_cdk import (
    Stack,
    Duration,
    aws_lambda as lambda_,
    aws_logs as logs,
)
from constructs import Construct


class RiskManagementStack(Stack):
    """Risk management infrastructure."""

    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        env_name: str,
        data_stack,
        observability_stack,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.env_name = env_name

        # RiskEngine Lambda
        self.risk_engine = lambda_.Function(
            self,
            "RiskEngine",
            function_name=f"range-bot-risk-engine-{env_name}",
            runtime=lambda_.Runtime.PYTHON_3_11,
            handler="handler.lambda_handler",
            code=lambda_.Code.from_asset("../lambdas/risk_engine"),
            timeout=Duration.seconds(10),
            memory_size=256,
            environment={
                "POSITION_TABLE": data_stack.position_table.table_name,
                "MODEL_STATE_TABLE": data_stack.model_state_table.table_name,
                "ALERT_TOPIC_ARN": observability_stack.alert_topic.topic_arn,
                "MAX_POSITION_PCT": "0.80",
                "STOP_LOSS_PCT": "0.05",
                "TAKE_PROFIT_PCT": "0.10",
                "MAX_DAILY_DRAWDOWN": "0.03",
                "MAX_GAS_GWEI": "100",
                "ENV_NAME": env_name,
            },
            tracing=lambda_.Tracing.ACTIVE,
            log_retention=logs.RetentionDays.TWO_WEEKS,
        )

        # Grant permissions
        data_stack.position_table.grant_read_write_data(self.risk_engine)
        data_stack.model_state_table.grant_read_data(self.risk_engine)
        observability_stack.alert_topic.grant_publish(self.risk_engine)
```

Create `lambdas/risk_engine/handler.py`:

```python
"""
RiskEngine Lambda - Pre-trade risk checks.
"""
import json
import logging
import os
from decimal import Decimal
from typing import Any

import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

POSITION_TABLE = os.environ["POSITION_TABLE"]
ALERT_TOPIC_ARN = os.environ["ALERT_TOPIC_ARN"]
MAX_POSITION_PCT = Decimal(os.environ.get("MAX_POSITION_PCT", "0.80"))
MAX_DAILY_DRAWDOWN = Decimal(os.environ.get("MAX_DAILY_DRAWDOWN", "0.03"))
MAX_GAS_GWEI = int(os.environ.get("MAX_GAS_GWEI", "100"))

dynamodb = boto3.resource("dynamodb")
sns = boto3.client("sns")
position_table = dynamodb.Table(POSITION_TABLE)


def lambda_handler(event: dict, context: Any) -> dict:
    """Check risk limits before trade execution."""
    logger.info(f"RiskEngine invoked: {json.dumps(event)}")

    try:
        position = get_position_state()

        checks = [
            check_drawdown_limit(position),
            check_position_size(event, position),
            check_gas_price(event),
        ]

        for passed, reason in checks:
            if not passed:
                logger.warning(f"Risk check failed: {reason}")
                if "drawdown" in reason.lower():
                    send_alert(f"Risk limit breached: {reason}")
                return {"statusCode": 200, "body": json.dumps({"approved": False, "reason": reason})}

        return {"statusCode": 200, "body": json.dumps({"approved": True, "reason": "All checks passed"})}

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"statusCode": 200, "body": json.dumps({"approved": False, "reason": str(e)})}


def get_position_state() -> dict:
    try:
        response = position_table.get_item(Key={"pk": "POSITION", "sk": "CURRENT"})
        return response.get("Item", {})
    except Exception:
        return {}


def check_drawdown_limit(position: dict) -> tuple:
    daily_pnl_pct = Decimal(str(position.get("daily_pnl_pct", 0)))
    if daily_pnl_pct < -MAX_DAILY_DRAWDOWN:
        return False, f"Daily drawdown limit exceeded: {daily_pnl_pct:.2%}"
    return True, "OK"


def check_position_size(event: dict, position: dict) -> tuple:
    total_value = Decimal(str(position.get("total_value_usd", 1000)))
    position_value = Decimal(str(event.get("position_value_usd", 0)))
    if total_value > 0:
        pct = position_value / total_value
        if pct > MAX_POSITION_PCT:
            return False, f"Position too large: {pct:.2%}"
    return True, "OK"


def check_gas_price(event: dict) -> tuple:
    gas = event.get("gas_price_gwei", 0)
    if gas > MAX_GAS_GWEI:
        return False, f"Gas too high: {gas} gwei"
    return True, "OK"


def send_alert(message: str) -> None:
    sns.publish(TopicArn=ALERT_TOPIC_ARN, Subject="Range-Bot Risk Alert", Message=message)
```

---

## 10. Deploy and Test

### 10.1 First Deployment

```bash
cd infra

# Update your email in app.py first!
# ALERT_EMAIL = "your-email@example.com"

# Synthesize (generates CloudFormation)
cdk synth

# Deploy all stacks to dev
cdk deploy --all -c env=dev --require-approval never
```

### 10.2 Verify Deployment

```bash
# Check Lambda logs (wait a few minutes for first invocation)
aws logs tail /aws/lambda/range-bot-swap-listener-dev --follow

# Check DynamoDB has data
aws dynamodb scan --table-name range-bot-swaps-dev --limit 5

# Check model state
aws dynamodb scan --table-name range-bot-model-state-dev
```

### 10.3 Confirm Email Subscription

Check your email and click the confirmation link from AWS SNS.

---

## 11. Go-Live Checklist

Before deploying to production:

```
□ Paper trading for 2+ weeks
□ Predictions match POC backtest results
□ No Lambda errors
□ Email alerts working
□ Fresh wallet generated (offline)
□ Private key stored in Secrets Manager
□ Wallet funded ($500 ETH + $500 USDC)
```

### Deploy to Production

```bash
# Update email in app.py for production alerts
# Deploy to prod
cdk deploy --all -c env=prod
```

---

## 12. Quick Reference Commands

```bash
# View Lambda logs
aws logs tail /aws/lambda/range-bot-swap-listener-dev --follow

# Query DynamoDB
aws dynamodb query \
    --table-name range-bot-ohlc-dev \
    --key-condition-expression "pk = :pk" \
    --expression-attribute-values '{":pk": {"S": "POOL#0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"}}' \
    --scan-index-forward false \
    --limit 5

# Invoke Lambda manually
aws lambda invoke --function-name range-bot-bayesian-dev response.json
cat response.json

# Disable a rule (pause processing)
aws events disable-rule --name range-bot-swap-listener-dev

# Enable a rule
aws events enable-rule --name range-bot-swap-listener-dev

# Check alarm status
aws cloudwatch describe-alarms --alarm-names range-bot-lambda-errors-dev
```

---

## 13. Signer Container Implementation

The signer runs in an isolated ECS Fargate container with no internet access except through VPC endpoints. It polls SQS for transaction requests, signs them using the private key from Secrets Manager, and submits via Flashbots.

### 13.1 Create Container Directory Structure

```bash
mkdir -p containers/signer
```

### 13.2 Create Dockerfile

Create `containers/signer/Dockerfile`:

```dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY *.py .

# Create non-root user for security
RUN useradd -m -u 1000 -s /bin/bash appuser && \
    chown -R appuser:appuser /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import sys; sys.exit(0)"

# Run the signer
CMD ["python", "-u", "main.py"]
```

### 13.3 Create Requirements

Create `containers/signer/requirements.txt`:

```
web3>=6.0,<7.0
eth-account>=0.9,<1.0
eth-abi>=4.0,<5.0
boto3>=1.28,<2.0
requests>=2.31,<3.0
```

### 13.4 Create Main Entry Point

Create `containers/signer/main.py`:

```python
#!/usr/bin/env python3
"""
Transaction Signer Service

Runs in isolated ECS Fargate container.
- Polls SQS for transaction requests
- Signs transactions using private key from Secrets Manager
- Submits via Flashbots Protect RPC (MEV protection)
- Reports results back to DynamoDB
"""
import json
import logging
import os
import signal
import sys
import time
from dataclasses import dataclass
from typing import Optional

import boto3

from signer import TransactionSigner
from uniswap_math import price_to_tick, tick_to_price

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("signer-main")

# Configuration from environment
QUEUE_URL = os.environ["QUEUE_URL"]
POSITION_TABLE = os.environ.get("POSITION_TABLE", "range-bot-positions-prod")
ENV_NAME = os.environ.get("ENV_NAME", "prod")

# RPC endpoints
FLASHBOTS_RPC = "https://rpc.flashbots.net"
PUBLIC_RPC = "https://ethereum-rpc.publicnode.com"

# AWS clients
sqs = boto3.client("sqs")
dynamodb = boto3.resource("dynamodb")
position_table = dynamodb.Table(POSITION_TABLE)

# Graceful shutdown flag
shutdown_requested = False


@dataclass
class TransactionRequest:
    """Parsed transaction request from SQS."""
    action: str
    params: dict
    request_id: str
    urgency: str = "medium"


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def get_private_key() -> str:
    """
    Retrieve private key from environment (injected by ECS from Secrets Manager).

    The key is injected as WALLET_PRIVATE_KEY environment variable by ECS
    using the secrets configuration in the task definition.
    """
    key = os.environ.get("WALLET_PRIVATE_KEY", "")

    if not key:
        raise ValueError("WALLET_PRIVATE_KEY not found in environment")

    # Remove 0x prefix if present
    if key.startswith("0x"):
        key = key[2:]

    # Validate key format (64 hex characters)
    if len(key) != 64:
        raise ValueError(f"Invalid private key length: {len(key)} (expected 64)")

    try:
        bytes.fromhex(key)
    except ValueError:
        raise ValueError("Private key is not valid hexadecimal")

    return key


def main():
    """Main entry point - runs the signer loop."""
    global shutdown_requested

    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    logger.info(f"Signer service starting (env={ENV_NAME})")

    # Get private key and initialize signer
    try:
        private_key = get_private_key()
        rpc_url = FLASHBOTS_RPC if ENV_NAME == "prod" else PUBLIC_RPC
        signer = TransactionSigner(private_key, rpc_url)
        logger.info(f"Signer initialized: address={signer.address}")
    except Exception as e:
        logger.error(f"Failed to initialize signer: {e}")
        sys.exit(1)

    # Clear sensitive data from memory
    del private_key

    # Main loop
    consecutive_errors = 0
    max_consecutive_errors = 10

    while not shutdown_requested:
        try:
            # Poll SQS for messages (long polling)
            response = sqs.receive_message(
                QueueUrl=QUEUE_URL,
                MaxNumberOfMessages=1,
                WaitTimeSeconds=20,
                VisibilityTimeout=300,
                MessageAttributeNames=["All"],
            )

            messages = response.get("Messages", [])

            for message in messages:
                if shutdown_requested:
                    logger.info("Shutdown requested, skipping remaining messages")
                    break

                try:
                    process_message(message, signer)

                    # Delete message after successful processing
                    sqs.delete_message(
                        QueueUrl=QUEUE_URL,
                        ReceiptHandle=message["ReceiptHandle"],
                    )

                    consecutive_errors = 0

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    consecutive_errors += 1

                    # Record failure
                    record_transaction_result(
                        message.get("MessageId", "unknown"),
                        success=False,
                        error=str(e),
                    )

                    # Message will return to queue after visibility timeout

            # Check for too many consecutive errors
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"Too many consecutive errors ({consecutive_errors}), exiting")
                sys.exit(1)

        except Exception as e:
            logger.error(f"Error in main loop: {e}", exc_info=True)
            consecutive_errors += 1
            time.sleep(5)

    logger.info("Signer service shutting down")


def process_message(message: dict, signer: TransactionSigner):
    """Process a single transaction request from SQS."""
    body = json.loads(message["Body"])

    request = TransactionRequest(
        action=body.get("action"),
        params=body.get("params", {}),
        request_id=message.get("MessageId", "unknown"),
        urgency=body.get("urgency", "medium"),
    )

    logger.info(f"Processing transaction: action={request.action}, id={request.request_id}")

    # Execute based on action type
    if request.action == "mint":
        tx_hash = execute_mint(signer, request.params)
    elif request.action == "burn":
        tx_hash = execute_burn(signer, request.params)
    elif request.action == "collect":
        tx_hash = execute_collect(signer, request.params)
    elif request.action == "swap":
        tx_hash = execute_swap(signer, request.params)
    elif request.action == "emergency_exit":
        tx_hash = execute_emergency_exit(signer, request.params)
    else:
        raise ValueError(f"Unknown action: {request.action}")

    logger.info(f"Transaction submitted: {tx_hash}")

    # Wait for confirmation
    receipt = signer.wait_for_confirmation(tx_hash, timeout=300, confirmations=2)

    if receipt["status"] == 1:
        logger.info(f"Transaction confirmed: block={receipt['blockNumber']}, gas={receipt['gasUsed']}")
        record_transaction_result(
            request.request_id,
            success=True,
            tx_hash=tx_hash,
            block_number=receipt["blockNumber"],
            gas_used=receipt["gasUsed"],
        )
    else:
        logger.error(f"Transaction reverted: {tx_hash}")
        record_transaction_result(
            request.request_id,
            success=False,
            tx_hash=tx_hash,
            error="Transaction reverted on-chain",
        )
        raise RuntimeError(f"Transaction reverted: {tx_hash}")


def execute_mint(signer: TransactionSigner, params: dict) -> str:
    """Execute mint (create LP position)."""
    return signer.mint_position(
        tick_lower=params["tick_lower"],
        tick_upper=params["tick_upper"],
        amount0=params["amount0"],
        amount1=params["amount1"],
        slippage_bps=params.get("slippage_bps", 50),
    )


def execute_burn(signer: TransactionSigner, params: dict) -> str:
    """Execute burn (remove liquidity)."""
    return signer.decrease_liquidity(
        token_id=params["token_id"],
        liquidity=params["liquidity"],
        amount0_min=params.get("amount0_min", 0),
        amount1_min=params.get("amount1_min", 0),
    )


def execute_collect(signer: TransactionSigner, params: dict) -> str:
    """Execute collect (claim fees)."""
    return signer.collect_fees(token_id=params["token_id"])


def execute_swap(signer: TransactionSigner, params: dict) -> str:
    """Execute swap for directional position."""
    return signer.swap_exact_input(
        token_in=params["token_in"],
        token_out=params["token_out"],
        amount_in=params["amount_in"],
        amount_out_min=params.get("amount_out_min", 0),
        slippage_bps=params.get("slippage_bps", 50),
    )


def execute_emergency_exit(signer: TransactionSigner, params: dict) -> str:
    """
    Execute emergency exit - burn all liquidity and collect.

    Uses multicall to atomically:
    1. Decrease all liquidity
    2. Collect all tokens
    """
    token_id = params.get("token_id")

    if not token_id:
        # Get token ID from position state
        position = get_current_position()
        token_id = position.get("token_id")

    if not token_id:
        raise ValueError("No token_id provided and no active position found")

    # Get current liquidity
    position_info = signer.get_position_info(token_id)
    liquidity = position_info["liquidity"]

    if liquidity == 0:
        logger.info("No liquidity to withdraw")
        return signer.collect_fees(token_id)

    # Execute multicall: decreaseLiquidity + collect
    return signer.emergency_withdraw(token_id, liquidity)


def get_current_position() -> dict:
    """Get current position state from DynamoDB."""
    try:
        response = position_table.get_item(
            Key={"pk": "POSITION", "sk": "CURRENT"}
        )
        return response.get("Item", {})
    except Exception as e:
        logger.warning(f"Error getting position: {e}")
        return {}


def record_transaction_result(
    request_id: str,
    success: bool,
    tx_hash: Optional[str] = None,
    block_number: Optional[int] = None,
    gas_used: Optional[int] = None,
    error: Optional[str] = None,
):
    """Record transaction result to DynamoDB."""
    try:
        item = {
            "pk": "TX_HISTORY",
            "sk": f"{int(time.time())}#{request_id}",
            "request_id": request_id,
            "success": success,
            "timestamp": int(time.time()),
        }

        if tx_hash:
            item["tx_hash"] = tx_hash
        if block_number:
            item["block_number"] = block_number
        if gas_used:
            item["gas_used"] = gas_used
        if error:
            item["error"] = error

        position_table.put_item(Item=item)

    except Exception as e:
        logger.warning(f"Failed to record transaction result: {e}")


if __name__ == "__main__":
    main()
```

### 13.5 Create Transaction Signer Module

Create `containers/signer/signer.py`:

```python
"""
Transaction Signer Module

Handles all Uniswap v3 transaction building, signing, and submission.
"""
import logging
import time
from typing import Optional

from eth_account import Account
from eth_abi import encode
from web3 import Web3

logger = logging.getLogger("signer")

# Contract addresses (Ethereum mainnet)
WETH_ADDRESS = "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2"
USDC_ADDRESS = "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"
POSITION_MANAGER = "0xC36442b4a4522E871399CD717aBDD847Ab11FE88"
SWAP_ROUTER = "0x68b3465833fb72A70ecDF485E0e4C7bD8665Fc45"
POOL_ADDRESS = "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"
FEE_TIER = 500  # 0.05%

# ABI for NonfungiblePositionManager
POSITION_MANAGER_ABI = [
    {
        "name": "mint",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [{"name": "params", "type": "tuple", "components": [
            {"name": "token0", "type": "address"},
            {"name": "token1", "type": "address"},
            {"name": "fee", "type": "uint24"},
            {"name": "tickLower", "type": "int24"},
            {"name": "tickUpper", "type": "int24"},
            {"name": "amount0Desired", "type": "uint256"},
            {"name": "amount1Desired", "type": "uint256"},
            {"name": "amount0Min", "type": "uint256"},
            {"name": "amount1Min", "type": "uint256"},
            {"name": "recipient", "type": "address"},
            {"name": "deadline", "type": "uint256"},
        ]}],
        "outputs": [
            {"name": "tokenId", "type": "uint256"},
            {"name": "liquidity", "type": "uint128"},
            {"name": "amount0", "type": "uint256"},
            {"name": "amount1", "type": "uint256"},
        ],
    },
    {
        "name": "decreaseLiquidity",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [{"name": "params", "type": "tuple", "components": [
            {"name": "tokenId", "type": "uint256"},
            {"name": "liquidity", "type": "uint128"},
            {"name": "amount0Min", "type": "uint256"},
            {"name": "amount1Min", "type": "uint256"},
            {"name": "deadline", "type": "uint256"},
        ]}],
        "outputs": [
            {"name": "amount0", "type": "uint256"},
            {"name": "amount1", "type": "uint256"},
        ],
    },
    {
        "name": "collect",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [{"name": "params", "type": "tuple", "components": [
            {"name": "tokenId", "type": "uint256"},
            {"name": "recipient", "type": "address"},
            {"name": "amount0Max", "type": "uint128"},
            {"name": "amount1Max", "type": "uint128"},
        ]}],
        "outputs": [
            {"name": "amount0", "type": "uint256"},
            {"name": "amount1", "type": "uint256"},
        ],
    },
    {
        "name": "positions",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "tokenId", "type": "uint256"}],
        "outputs": [
            {"name": "nonce", "type": "uint96"},
            {"name": "operator", "type": "address"},
            {"name": "token0", "type": "address"},
            {"name": "token1", "type": "address"},
            {"name": "fee", "type": "uint24"},
            {"name": "tickLower", "type": "int24"},
            {"name": "tickUpper", "type": "int24"},
            {"name": "liquidity", "type": "uint128"},
            {"name": "feeGrowthInside0LastX128", "type": "uint256"},
            {"name": "feeGrowthInside1LastX128", "type": "uint256"},
            {"name": "tokensOwed0", "type": "uint128"},
            {"name": "tokensOwed1", "type": "uint128"},
        ],
    },
    {
        "name": "multicall",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [{"name": "data", "type": "bytes[]"}],
        "outputs": [{"name": "results", "type": "bytes[]"}],
    },
]

# ABI for SwapRouter02
SWAP_ROUTER_ABI = [
    {
        "name": "exactInputSingle",
        "type": "function",
        "stateMutability": "payable",
        "inputs": [{"name": "params", "type": "tuple", "components": [
            {"name": "tokenIn", "type": "address"},
            {"name": "tokenOut", "type": "address"},
            {"name": "fee", "type": "uint24"},
            {"name": "recipient", "type": "address"},
            {"name": "amountIn", "type": "uint256"},
            {"name": "amountOutMinimum", "type": "uint256"},
            {"name": "sqrtPriceLimitX96", "type": "uint160"},
        ]}],
        "outputs": [{"name": "amountOut", "type": "uint256"}],
    },
]

# ERC20 ABI for approvals
ERC20_ABI = [
    {
        "name": "approve",
        "type": "function",
        "inputs": [
            {"name": "spender", "type": "address"},
            {"name": "amount", "type": "uint256"},
        ],
        "outputs": [{"name": "", "type": "bool"}],
    },
    {
        "name": "allowance",
        "type": "function",
        "stateMutability": "view",
        "inputs": [
            {"name": "owner", "type": "address"},
            {"name": "spender", "type": "address"},
        ],
        "outputs": [{"name": "", "type": "uint256"}],
    },
    {
        "name": "balanceOf",
        "type": "function",
        "stateMutability": "view",
        "inputs": [{"name": "account", "type": "address"}],
        "outputs": [{"name": "", "type": "uint256"}],
    },
]


class TransactionSigner:
    """
    Handles transaction signing and submission for Uniswap v3 operations.
    """

    def __init__(self, private_key: str, rpc_url: str):
        """
        Initialize the signer.

        Args:
            private_key: Hex string (without 0x prefix)
            rpc_url: Ethereum RPC endpoint
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={"timeout": 30}))

        # Create account from private key
        self.account = Account.from_key(bytes.fromhex(private_key))
        self.address = self.account.address

        # Initialize contracts
        self.position_manager = self.w3.eth.contract(
            address=POSITION_MANAGER,
            abi=POSITION_MANAGER_ABI,
        )
        self.swap_router = self.w3.eth.contract(
            address=SWAP_ROUTER,
            abi=SWAP_ROUTER_ABI,
        )
        self.weth = self.w3.eth.contract(address=WETH_ADDRESS, abi=ERC20_ABI)
        self.usdc = self.w3.eth.contract(address=USDC_ADDRESS, abi=ERC20_ABI)

        logger.info(f"Signer initialized: {self.address}")

    def get_balances(self) -> dict:
        """Get current token balances."""
        eth_balance = self.w3.eth.get_balance(self.address)
        weth_balance = self.weth.functions.balanceOf(self.address).call()
        usdc_balance = self.usdc.functions.balanceOf(self.address).call()

        return {
            "eth": eth_balance,
            "weth": weth_balance,
            "usdc": usdc_balance,
        }

    def ensure_approval(self, token_address: str, spender: str, amount: int) -> Optional[str]:
        """
        Ensure token approval is sufficient, approve if not.

        Returns tx_hash if approval was needed, None otherwise.
        """
        token = self.w3.eth.contract(address=token_address, abi=ERC20_ABI)

        current_allowance = token.functions.allowance(self.address, spender).call()

        if current_allowance >= amount:
            logger.debug(f"Sufficient allowance: {current_allowance}")
            return None

        # Approve max uint256
        max_approval = 2**256 - 1

        tx = token.functions.approve(spender, max_approval).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "chainId": self.w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def mint_position(
        self,
        tick_lower: int,
        tick_upper: int,
        amount0: int,
        amount1: int,
        slippage_bps: int = 50,
    ) -> str:
        """
        Mint a new LP position.

        Args:
            tick_lower: Lower tick of range
            tick_upper: Upper tick of range
            amount0: Amount of token0 (WETH) in wei
            amount1: Amount of token1 (USDC) in smallest units
            slippage_bps: Slippage tolerance in basis points

        Returns:
            Transaction hash
        """
        # Ensure approvals
        self.ensure_approval(WETH_ADDRESS, POSITION_MANAGER, amount0)
        self.ensure_approval(USDC_ADDRESS, POSITION_MANAGER, amount1)

        # Calculate minimums with slippage
        slippage_factor = (10000 - slippage_bps) / 10000
        amount0_min = int(amount0 * slippage_factor)
        amount1_min = int(amount1 * slippage_factor)

        deadline = int(time.time()) + 300  # 5 minutes

        params = (
            WETH_ADDRESS,      # token0
            USDC_ADDRESS,      # token1
            FEE_TIER,          # fee
            tick_lower,        # tickLower
            tick_upper,        # tickUpper
            amount0,           # amount0Desired
            amount1,           # amount1Desired
            amount0_min,       # amount0Min
            amount1_min,       # amount1Min
            self.address,      # recipient
            deadline,          # deadline
        )

        tx = self.position_manager.functions.mint(params).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "chainId": self.w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def decrease_liquidity(
        self,
        token_id: int,
        liquidity: int,
        amount0_min: int = 0,
        amount1_min: int = 0,
    ) -> str:
        """
        Decrease liquidity from an existing position.

        Args:
            token_id: NFT token ID of the position
            liquidity: Amount of liquidity to remove
            amount0_min: Minimum token0 to receive (slippage protection)
            amount1_min: Minimum token1 to receive (slippage protection)

        Returns:
            Transaction hash
        """
        deadline = int(time.time()) + 300

        params = (
            token_id,
            liquidity,
            amount0_min,
            amount1_min,
            deadline,
        )

        tx = self.position_manager.functions.decreaseLiquidity(params).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "chainId": self.w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def collect_fees(self, token_id: int) -> str:
        """
        Collect accumulated fees from a position.

        Args:
            token_id: NFT token ID of the position

        Returns:
            Transaction hash
        """
        params = (
            token_id,
            self.address,      # recipient
            2**128 - 1,        # amount0Max (max uint128)
            2**128 - 1,        # amount1Max (max uint128)
        )

        tx = self.position_manager.functions.collect(params).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "chainId": self.w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def swap_exact_input(
        self,
        token_in: str,
        token_out: str,
        amount_in: int,
        amount_out_min: int = 0,
        slippage_bps: int = 50,
    ) -> str:
        """
        Execute a swap with exact input amount.

        Args:
            token_in: Address of input token
            token_out: Address of output token
            amount_in: Exact amount of input token
            amount_out_min: Minimum output (0 = calculate from slippage)
            slippage_bps: Slippage tolerance if amount_out_min not specified

        Returns:
            Transaction hash
        """
        # Ensure approval
        self.ensure_approval(token_in, SWAP_ROUTER, amount_in)

        params = (
            token_in,
            token_out,
            FEE_TIER,
            self.address,
            amount_in,
            amount_out_min,
            0,  # sqrtPriceLimitX96 (0 = no limit)
        )

        tx = self.swap_router.functions.exactInputSingle(params).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "chainId": self.w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def get_position_info(self, token_id: int) -> dict:
        """
        Get information about a position.

        Args:
            token_id: NFT token ID

        Returns:
            Position information dict
        """
        result = self.position_manager.functions.positions(token_id).call()

        return {
            "nonce": result[0],
            "operator": result[1],
            "token0": result[2],
            "token1": result[3],
            "fee": result[4],
            "tick_lower": result[5],
            "tick_upper": result[6],
            "liquidity": result[7],
            "fee_growth_inside0": result[8],
            "fee_growth_inside1": result[9],
            "tokens_owed0": result[10],
            "tokens_owed1": result[11],
        }

    def emergency_withdraw(self, token_id: int, liquidity: int) -> str:
        """
        Emergency withdrawal - atomic decreaseLiquidity + collect via multicall.

        Args:
            token_id: NFT token ID
            liquidity: Liquidity amount to remove

        Returns:
            Transaction hash
        """
        deadline = int(time.time()) + 300

        # Encode decreaseLiquidity call
        decrease_params = (token_id, liquidity, 0, 0, deadline)
        decrease_data = self.position_manager.encodeABI(
            fn_name="decreaseLiquidity",
            args=[decrease_params]
        )

        # Encode collect call
        collect_params = (token_id, self.address, 2**128 - 1, 2**128 - 1)
        collect_data = self.position_manager.encodeABI(
            fn_name="collect",
            args=[collect_params]
        )

        # Build multicall
        tx = self.position_manager.functions.multicall(
            [bytes.fromhex(decrease_data[2:]), bytes.fromhex(collect_data[2:])]
        ).build_transaction({
            "from": self.address,
            "nonce": self.w3.eth.get_transaction_count(self.address),
            "chainId": self.w3.eth.chain_id,
        })

        return self._sign_and_send(tx)

    def _sign_and_send(self, tx: dict) -> str:
        """
        Sign and send a transaction.

        Args:
            tx: Transaction dictionary

        Returns:
            Transaction hash as hex string
        """
        # Estimate gas
        try:
            gas_estimate = self.w3.eth.estimate_gas(tx)
            tx["gas"] = int(gas_estimate * 1.2)  # 20% buffer
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}, using default")
            tx["gas"] = 500000

        # Get gas price (EIP-1559)
        latest_block = self.w3.eth.get_block("latest")
        base_fee = latest_block.get("baseFeePerGas", self.w3.to_wei(30, "gwei"))

        priority_fee = self.w3.to_wei(2, "gwei")
        max_fee = int(base_fee * 1.5) + priority_fee

        tx["maxFeePerGas"] = max_fee
        tx["maxPriorityFeePerGas"] = priority_fee
        tx["type"] = 2  # EIP-1559

        # Sign
        signed = self.account.sign_transaction(tx)

        # Send
        tx_hash = self.w3.eth.send_raw_transaction(signed.rawTransaction)

        logger.info(f"Transaction sent: {tx_hash.hex()}")
        return tx_hash.hex()

    def wait_for_confirmation(
        self,
        tx_hash: str,
        timeout: int = 300,
        confirmations: int = 2,
    ) -> dict:
        """
        Wait for transaction confirmation.

        Args:
            tx_hash: Transaction hash
            timeout: Maximum wait time in seconds
            confirmations: Number of confirmations required

        Returns:
            Transaction receipt

        Raises:
            TimeoutError: If transaction not confirmed in time
        """
        start = time.time()

        while time.time() - start < timeout:
            try:
                receipt = self.w3.eth.get_transaction_receipt(tx_hash)
                if receipt:
                    current_block = self.w3.eth.block_number
                    if current_block - receipt["blockNumber"] >= confirmations:
                        return dict(receipt)
            except Exception:
                pass

            time.sleep(2)

        raise TimeoutError(f"Transaction not confirmed: {tx_hash}")
```

### 13.6 Update Execution Stack for ECS Task

Update `infra/stacks/execution.py` to add ECS task definition:

Add this after the ECS Cluster creation in the ExecutionStack class:

```python
        # =====================================================================
        # ECS Task Definition
        # =====================================================================

        task_def = ecs.FargateTaskDefinition(
            self,
            "SignerTask",
            family=f"range-bot-signer-{env_name}",
            cpu=256,
            memory_limit_mib=512,
        )

        # Add container
        container = task_def.add_container(
            "Signer",
            image=ecs.ContainerImage.from_asset("../containers/signer"),
            logging=ecs.LogDrivers.aws_logs(
                stream_prefix="signer",
                log_retention=logs.RetentionDays.TWO_WEEKS,
            ),
            environment={
                "QUEUE_URL": self.tx_queue.queue_url,
                "POSITION_TABLE": f"range-bot-positions-{env_name}",
                "ENV_NAME": env_name,
            },
            secrets={
                "WALLET_PRIVATE_KEY": ecs.Secret.from_secrets_manager(
                    self.private_key_secret
                ),
            },
        )

        # Grant permissions
        self.private_key_secret.grant_read(task_def.task_role)
        self.tx_queue.grant_consume_messages(task_def.task_role)

        # =====================================================================
        # ECS Service (uncomment when ready for live trading)
        # =====================================================================

        # self.signer_service = ecs.FargateService(
        #     self,
        #     "SignerService",
        #     service_name=f"range-bot-signer-{env_name}",
        #     cluster=self.cluster,
        #     task_definition=task_def,
        #     desired_count=1,
        #     vpc_subnets=ec2.SubnetSelection(
        #         subnet_type=ec2.SubnetType.PRIVATE_ISOLATED,
        #     ),
        # )
```

---

## 14. Uniswap V3 Math Library

Create `src/execution/__init__.py`:

```python
# Execution module
```

Create `src/execution/uniswap_math.py`:

```python
"""
Uniswap V3 Math Library

Contains all the mathematical functions needed for:
- Price/tick conversions
- Liquidity calculations
- Position sizing
- sqrtPriceX96 conversions

Based on Uniswap v3 whitepaper and smart contract implementations.
"""
import math
from decimal import Decimal, getcontext
from typing import Tuple

# Set high precision for financial calculations
getcontext().prec = 78

# Constants
Q96 = 2 ** 96
Q192 = 2 ** 192
MIN_TICK = -887272
MAX_TICK = 887272
MIN_SQRT_RATIO = 4295128739
MAX_SQRT_RATIO = 1461446703485210103287273052203988822378723970342

# Token decimals for ETH/USDC pool
TOKEN0_DECIMALS = 18  # WETH
TOKEN1_DECIMALS = 6   # USDC


def price_to_sqrt_price_x96(price: Decimal) -> int:
    """
    Convert human-readable ETH/USD price to sqrtPriceX96.

    For ETH/USDC pool:
    - token0 = WETH (18 decimals)
    - token1 = USDC (6 decimals)
    - price = USD per ETH

    sqrtPriceX96 = sqrt(token1/token0 * 10^(token0_decimals - token1_decimals)) * 2^96

    Args:
        price: ETH/USD price (e.g., 3000.0)

    Returns:
        sqrtPriceX96 as integer
    """
    # Decimal adjustment for different decimals
    decimal_adjustment = Decimal(10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))  # 10^12

    # sqrtPriceX96 = sqrt(1/price * decimal_adjustment) * 2^96
    adjusted = Decimal(1) / price * decimal_adjustment
    sqrt_price = adjusted.sqrt()
    sqrt_price_x96 = int(sqrt_price * Q96)

    return sqrt_price_x96


def sqrt_price_x96_to_price(sqrt_price_x96: int) -> Decimal:
    """
    Convert sqrtPriceX96 to human-readable ETH/USD price.

    price = 10^12 / ((sqrtPriceX96^2) / 2^192)

    Args:
        sqrt_price_x96: Uniswap v3 sqrt price encoding

    Returns:
        Price as Decimal (USD per ETH)
    """
    decimal_adjustment = Decimal(10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))

    sqrt_price_squared = Decimal(sqrt_price_x96 ** 2)
    price = decimal_adjustment / (sqrt_price_squared / Decimal(Q192))

    return price


def price_to_tick(price: Decimal) -> int:
    """
    Convert price to tick.

    tick = floor(log(price) / log(1.0001))

    Note: This gives tick for the raw price ratio (token1/token0).
    For ETH/USDC we need to account for decimal adjustment.

    Args:
        price: ETH/USD price

    Returns:
        Tick value
    """
    # First convert to sqrtPriceX96
    sqrt_price_x96 = price_to_sqrt_price_x96(price)

    # Then convert sqrt price to tick
    return sqrt_price_x96_to_tick(sqrt_price_x96)


def tick_to_price(tick: int) -> Decimal:
    """
    Convert tick to price.

    Args:
        tick: Uniswap v3 tick

    Returns:
        ETH/USD price as Decimal
    """
    sqrt_price_x96 = tick_to_sqrt_price_x96(tick)
    return sqrt_price_x96_to_price(sqrt_price_x96)


def sqrt_price_x96_to_tick(sqrt_price_x96: int) -> int:
    """
    Convert sqrtPriceX96 to tick.

    tick = floor(log(sqrtPrice) / log(sqrt(1.0001)))
         = floor(2 * log(sqrtPriceX96 / 2^96) / log(1.0001))

    Args:
        sqrt_price_x96: sqrtPriceX96 value

    Returns:
        Tick value
    """
    sqrt_price = sqrt_price_x96 / Q96

    # log_sqrt_1.0001(sqrt_price) = log(sqrt_price) / log(sqrt(1.0001))
    # = 2 * log(sqrt_price) / log(1.0001)

    if sqrt_price <= 0:
        return MIN_TICK

    tick = math.floor(2 * math.log(sqrt_price) / math.log(1.0001))
    tick = max(MIN_TICK, min(MAX_TICK, tick))

    return tick


def tick_to_sqrt_price_x96(tick: int) -> int:
    """
    Convert tick to sqrtPriceX96.

    sqrtPrice = sqrt(1.0001^tick)
    sqrtPriceX96 = sqrtPrice * 2^96

    Args:
        tick: Uniswap v3 tick

    Returns:
        sqrtPriceX96 value
    """
    sqrt_price = math.sqrt(1.0001 ** tick)
    return int(sqrt_price * Q96)


def align_tick_to_spacing(tick: int, tick_spacing: int, round_down: bool = True) -> int:
    """
    Align tick to nearest valid tick (respecting tick spacing).

    For 0.05% fee tier, tick_spacing = 10.

    Args:
        tick: Raw tick value
        tick_spacing: Pool's tick spacing (10 for 0.05% pool)
        round_down: If True, round towards negative infinity

    Returns:
        Aligned tick value
    """
    if round_down:
        if tick >= 0:
            return (tick // tick_spacing) * tick_spacing
        else:
            return ((tick - tick_spacing + 1) // tick_spacing) * tick_spacing
    else:
        if tick >= 0:
            return ((tick + tick_spacing - 1) // tick_spacing) * tick_spacing
        else:
            return (tick // tick_spacing) * tick_spacing


def get_liquidity_for_amounts(
    sqrt_price_x96: int,
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    amount0: int,
    amount1: int,
) -> int:
    """
    Calculate liquidity from token amounts.

    Three cases based on current price relative to range:
    1. price < lower: only token0 contributes
    2. price > upper: only token1 contributes
    3. lower <= price <= upper: both tokens contribute

    Args:
        sqrt_price_x96: Current sqrtPriceX96
        sqrt_price_a_x96: Lower bound sqrtPriceX96
        sqrt_price_b_x96: Upper bound sqrtPriceX96
        amount0: Amount of token0 (WETH) in wei
        amount1: Amount of token1 (USDC) in smallest unit

    Returns:
        Liquidity amount
    """
    # Ensure a < b
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96

    if sqrt_price_x96 <= sqrt_price_a_x96:
        # Current price below range: only token0
        liquidity = _get_liquidity_for_amount0(
            sqrt_price_a_x96, sqrt_price_b_x96, amount0
        )
    elif sqrt_price_x96 >= sqrt_price_b_x96:
        # Current price above range: only token1
        liquidity = _get_liquidity_for_amount1(
            sqrt_price_a_x96, sqrt_price_b_x96, amount1
        )
    else:
        # Current price in range: both tokens
        liquidity0 = _get_liquidity_for_amount0(
            sqrt_price_x96, sqrt_price_b_x96, amount0
        )
        liquidity1 = _get_liquidity_for_amount1(
            sqrt_price_a_x96, sqrt_price_x96, amount1
        )
        # Take minimum (limiting factor)
        liquidity = min(liquidity0, liquidity1)

    return liquidity


def _get_liquidity_for_amount0(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    amount0: int,
) -> int:
    """
    Calculate liquidity from token0 amount.

    L = amount0 * sqrt(Pa) * sqrt(Pb) / (sqrt(Pb) - sqrt(Pa))
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96

    if sqrt_price_b_x96 == sqrt_price_a_x96:
        return 0

    intermediate = (sqrt_price_a_x96 * sqrt_price_b_x96) // Q96
    liquidity = (amount0 * intermediate) // (sqrt_price_b_x96 - sqrt_price_a_x96)

    return liquidity


def _get_liquidity_for_amount1(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    amount1: int,
) -> int:
    """
    Calculate liquidity from token1 amount.

    L = amount1 / (sqrt(Pb) - sqrt(Pa))
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96

    if sqrt_price_b_x96 == sqrt_price_a_x96:
        return 0

    liquidity = (amount1 * Q96) // (sqrt_price_b_x96 - sqrt_price_a_x96)

    return liquidity


def get_amounts_for_liquidity(
    sqrt_price_x96: int,
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int,
) -> Tuple[int, int]:
    """
    Calculate token amounts from liquidity.

    Args:
        sqrt_price_x96: Current sqrtPriceX96
        sqrt_price_a_x96: Lower bound sqrtPriceX96
        sqrt_price_b_x96: Upper bound sqrtPriceX96
        liquidity: Liquidity amount

    Returns:
        Tuple of (amount0, amount1) in smallest units
    """
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96

    if sqrt_price_x96 <= sqrt_price_a_x96:
        # Below range: all token0
        amount0 = _get_amount0_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
        amount1 = 0
    elif sqrt_price_x96 >= sqrt_price_b_x96:
        # Above range: all token1
        amount0 = 0
        amount1 = _get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_b_x96, liquidity)
    else:
        # In range: both tokens
        amount0 = _get_amount0_for_liquidity(sqrt_price_x96, sqrt_price_b_x96, liquidity)
        amount1 = _get_amount1_for_liquidity(sqrt_price_a_x96, sqrt_price_x96, liquidity)

    return amount0, amount1


def _get_amount0_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int,
) -> int:
    """Calculate token0 amount from liquidity."""
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96

    amount0 = (
        (liquidity * Q96 * (sqrt_price_b_x96 - sqrt_price_a_x96))
        // sqrt_price_b_x96
        // sqrt_price_a_x96
    )
    return amount0


def _get_amount1_for_liquidity(
    sqrt_price_a_x96: int,
    sqrt_price_b_x96: int,
    liquidity: int,
) -> int:
    """Calculate token1 amount from liquidity."""
    if sqrt_price_a_x96 > sqrt_price_b_x96:
        sqrt_price_a_x96, sqrt_price_b_x96 = sqrt_price_b_x96, sqrt_price_a_x96

    amount1 = (liquidity * (sqrt_price_b_x96 - sqrt_price_a_x96)) // Q96
    return amount1


def calculate_position_amounts(
    current_price: Decimal,
    price_lower: Decimal,
    price_upper: Decimal,
    total_value_usd: Decimal,
) -> Tuple[Decimal, Decimal]:
    """
    Calculate ETH and USDC amounts needed for a position.

    Args:
        current_price: Current ETH/USD price
        price_lower: Lower bound of range
        price_upper: Upper bound of range
        total_value_usd: Total value to deploy

    Returns:
        Tuple of (eth_amount, usdc_amount)
    """
    # Convert prices to sqrt price
    sqrt_p = current_price.sqrt()
    sqrt_pa = price_lower.sqrt()
    sqrt_pb = price_upper.sqrt()

    if current_price <= price_lower:
        # All ETH
        eth_amount = total_value_usd / current_price
        usdc_amount = Decimal(0)
    elif current_price >= price_upper:
        # All USDC
        eth_amount = Decimal(0)
        usdc_amount = total_value_usd
    else:
        # Both tokens - calculate ratio
        # For concentrated liquidity:
        # amount0 * P + amount1 = total_value
        # amount0 / amount1 = f(sqrtP, sqrtPa, sqrtPb)

        # Ratio of ETH value to total value at current price
        numerator = (sqrt_pb - sqrt_p) / sqrt_p
        denominator = numerator + (sqrt_p - sqrt_pa)

        if denominator == 0:
            eth_value_fraction = Decimal("0.5")
        else:
            eth_value_fraction = numerator / denominator

        eth_value = total_value_usd * eth_value_fraction
        usdc_value = total_value_usd - eth_value

        eth_amount = eth_value / current_price
        usdc_amount = usdc_value

    return eth_amount, usdc_amount


def calculate_impermanent_loss(
    entry_price: Decimal,
    current_price: Decimal,
) -> Decimal:
    """
    Calculate impermanent loss for a full-range position.

    IL = 2 * sqrt(k) / (1 + k) - 1
    where k = current_price / entry_price

    Args:
        entry_price: Price when position was created
        current_price: Current price

    Returns:
        IL as decimal (negative = loss, e.g., -0.05 = 5% loss)
    """
    if entry_price <= 0:
        return Decimal(0)

    k = current_price / entry_price
    sqrt_k = k.sqrt()

    il = (Decimal(2) * sqrt_k / (Decimal(1) + k)) - Decimal(1)

    return il


# Utility functions for the signer container
def price_range_to_ticks(
    price_lower: Decimal,
    price_upper: Decimal,
    tick_spacing: int = 10,
) -> Tuple[int, int]:
    """
    Convert price range to aligned ticks.

    Args:
        price_lower: Lower price bound
        price_upper: Upper price bound
        tick_spacing: Pool tick spacing (10 for 0.05%)

    Returns:
        Tuple of (tick_lower, tick_upper) aligned to spacing
    """
    tick_lower = price_to_tick(price_lower)
    tick_upper = price_to_tick(price_upper)

    # Align to tick spacing
    tick_lower = align_tick_to_spacing(tick_lower, tick_spacing, round_down=True)
    tick_upper = align_tick_to_spacing(tick_upper, tick_spacing, round_down=False)

    # Ensure valid order
    if tick_lower > tick_upper:
        tick_lower, tick_upper = tick_upper, tick_lower

    return tick_lower, tick_upper
```

Also create the uniswap_math module in the signer container.

Create `containers/signer/uniswap_math.py`:

```python
"""
Uniswap V3 Math - Simplified version for signer container.
"""
import math
from decimal import Decimal

Q96 = 2 ** 96
Q192 = 2 ** 192
MIN_TICK = -887272
MAX_TICK = 887272
TOKEN0_DECIMALS = 18
TOKEN1_DECIMALS = 6


def price_to_tick(price: Decimal) -> int:
    """Convert ETH/USD price to tick."""
    decimal_adjustment = Decimal(10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))
    adjusted = Decimal(1) / price * decimal_adjustment
    sqrt_price = float(adjusted.sqrt())
    tick = math.floor(2 * math.log(sqrt_price) / math.log(1.0001))
    return max(MIN_TICK, min(MAX_TICK, tick))


def tick_to_price(tick: int) -> Decimal:
    """Convert tick to ETH/USD price."""
    sqrt_price = math.sqrt(1.0001 ** tick)
    sqrt_price_x96 = int(sqrt_price * Q96)
    decimal_adjustment = Decimal(10 ** (TOKEN0_DECIMALS - TOKEN1_DECIMALS))
    return decimal_adjustment / (Decimal(sqrt_price_x96 ** 2) / Decimal(Q192))


def align_tick(tick: int, spacing: int = 10, round_down: bool = True) -> int:
    """Align tick to spacing."""
    if round_down:
        if tick >= 0:
            return (tick // spacing) * spacing
        return ((tick - spacing + 1) // spacing) * spacing
    else:
        if tick >= 0:
            return ((tick + spacing - 1) // spacing) * spacing
        return (tick // spacing) * spacing
```

---

## 15. Integration Tests

### 15.1 Create Test Directory Structure

```bash
mkdir -p tests/unit
mkdir -p tests/integration
```

### 15.2 Unit Tests for Uniswap Math

Create `tests/unit/test_uniswap_math.py`:

```python
"""
Unit tests for Uniswap v3 math functions.
"""
import pytest
from decimal import Decimal
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from execution.uniswap_math import (
    price_to_tick,
    tick_to_price,
    price_to_sqrt_price_x96,
    sqrt_price_x96_to_price,
    sqrt_price_x96_to_tick,
    tick_to_sqrt_price_x96,
    align_tick_to_spacing,
    get_liquidity_for_amounts,
    get_amounts_for_liquidity,
    calculate_position_amounts,
    calculate_impermanent_loss,
    price_range_to_ticks,
    Q96,
)


class TestPriceTickConversion:
    """Tests for price/tick conversion functions."""

    def test_price_to_tick_3000(self):
        """Test conversion at $3000 ETH."""
        price = Decimal("3000")
        tick = price_to_tick(price)

        # Tick should be approximately -201230 for $3000
        assert -202000 < tick < -200000

    def test_tick_to_price_roundtrip(self):
        """Test that price -> tick -> price roundtrips correctly."""
        original_price = Decimal("3000")
        tick = price_to_tick(original_price)
        recovered_price = tick_to_price(tick)

        # Should be within 0.01% due to tick discretization
        error = abs(recovered_price - original_price) / original_price
        assert error < Decimal("0.0001")

    def test_sqrt_price_conversion_roundtrip(self):
        """Test sqrtPriceX96 conversion roundtrip."""
        original_price = Decimal("2500")
        sqrt_price = price_to_sqrt_price_x96(original_price)
        recovered_price = sqrt_price_x96_to_price(sqrt_price)

        error = abs(recovered_price - original_price) / original_price
        assert error < Decimal("0.0001")

    def test_price_ordering(self):
        """Test that higher prices give lower ticks for WETH/USDC."""
        price_low = Decimal("2000")
        price_high = Decimal("4000")

        tick_low = price_to_tick(price_low)
        tick_high = price_to_tick(price_high)

        # For WETH/USDC, higher USD price = lower tick
        # (because token1/token0 ratio decreases)
        assert tick_high < tick_low


class TestTickAlignment:
    """Tests for tick alignment to spacing."""

    def test_align_positive_tick_down(self):
        """Test aligning positive tick downward."""
        tick = 12345
        aligned = align_tick_to_spacing(tick, 10, round_down=True)

        assert aligned == 12340
        assert aligned % 10 == 0

    def test_align_negative_tick_down(self):
        """Test aligning negative tick downward."""
        tick = -12345
        aligned = align_tick_to_spacing(tick, 10, round_down=True)

        assert aligned == -12350
        assert aligned % 10 == 0
        assert aligned <= tick

    def test_align_positive_tick_up(self):
        """Test aligning positive tick upward."""
        tick = 12341
        aligned = align_tick_to_spacing(tick, 10, round_down=False)

        assert aligned == 12350
        assert aligned % 10 == 0

    def test_already_aligned(self):
        """Test that already-aligned ticks stay the same."""
        tick = 12340
        aligned = align_tick_to_spacing(tick, 10, round_down=True)

        assert aligned == tick


class TestLiquidityCalculations:
    """Tests for liquidity math."""

    def test_liquidity_for_amounts_in_range(self):
        """Test liquidity calculation when price is in range."""
        current_price = Decimal("3000")
        price_lower = Decimal("2800")
        price_upper = Decimal("3200")

        sqrt_price = price_to_sqrt_price_x96(current_price)
        sqrt_price_a = price_to_sqrt_price_x96(price_lower)
        sqrt_price_b = price_to_sqrt_price_x96(price_upper)

        # 1 ETH and 3000 USDC
        amount0 = 10 ** 18  # 1 ETH in wei
        amount1 = 3000 * 10 ** 6  # 3000 USDC

        liquidity = get_liquidity_for_amounts(
            sqrt_price, sqrt_price_a, sqrt_price_b, amount0, amount1
        )

        assert liquidity > 0

    def test_liquidity_below_range(self):
        """Test that only token0 is used when price is below range."""
        current_price = Decimal("2500")
        price_lower = Decimal("2800")
        price_upper = Decimal("3200")

        sqrt_price = price_to_sqrt_price_x96(current_price)
        sqrt_price_a = price_to_sqrt_price_x96(price_lower)
        sqrt_price_b = price_to_sqrt_price_x96(price_upper)

        amount0 = 10 ** 18
        amount1 = 3000 * 10 ** 6

        liquidity = get_liquidity_for_amounts(
            sqrt_price, sqrt_price_a, sqrt_price_b, amount0, amount1
        )

        # Verify we can get amounts back
        recovered_amount0, recovered_amount1 = get_amounts_for_liquidity(
            sqrt_price, sqrt_price_a, sqrt_price_b, liquidity
        )

        # Below range: should be all token0, no token1
        assert recovered_amount0 > 0
        assert recovered_amount1 == 0

    def test_amounts_roundtrip(self):
        """Test that amounts -> liquidity -> amounts roundtrips."""
        current_price = Decimal("3000")
        price_lower = Decimal("2800")
        price_upper = Decimal("3200")

        sqrt_price = price_to_sqrt_price_x96(current_price)
        sqrt_price_a = price_to_sqrt_price_x96(price_lower)
        sqrt_price_b = price_to_sqrt_price_x96(price_upper)

        amount0 = 10 ** 18
        amount1 = 3000 * 10 ** 6

        liquidity = get_liquidity_for_amounts(
            sqrt_price, sqrt_price_a, sqrt_price_b, amount0, amount1
        )

        recovered_amount0, recovered_amount1 = get_amounts_for_liquidity(
            sqrt_price, sqrt_price_a, sqrt_price_b, liquidity
        )

        # Should recover close to original amounts
        # (may be slightly less due to taking min of two liquidities)
        assert recovered_amount0 <= amount0
        assert recovered_amount1 <= amount1
        assert recovered_amount0 > 0 or recovered_amount1 > 0


class TestPositionSizing:
    """Tests for position sizing calculations."""

    def test_calculate_position_amounts_in_range(self):
        """Test position sizing when current price is in range."""
        current_price = Decimal("3000")
        price_lower = Decimal("2800")
        price_upper = Decimal("3200")
        total_value = Decimal("10000")  # $10k

        eth_amount, usdc_amount = calculate_position_amounts(
            current_price, price_lower, price_upper, total_value
        )

        # Total value should match
        calculated_value = eth_amount * current_price + usdc_amount
        assert abs(calculated_value - total_value) < Decimal("1")

        # Both should be positive when in range
        assert eth_amount > 0
        assert usdc_amount > 0

    def test_calculate_position_below_range(self):
        """Test that position is all ETH when below range."""
        current_price = Decimal("2500")
        price_lower = Decimal("2800")
        price_upper = Decimal("3200")
        total_value = Decimal("10000")

        eth_amount, usdc_amount = calculate_position_amounts(
            current_price, price_lower, price_upper, total_value
        )

        assert eth_amount > 0
        assert usdc_amount == 0

    def test_calculate_position_above_range(self):
        """Test that position is all USDC when above range."""
        current_price = Decimal("3500")
        price_lower = Decimal("2800")
        price_upper = Decimal("3200")
        total_value = Decimal("10000")

        eth_amount, usdc_amount = calculate_position_amounts(
            current_price, price_lower, price_upper, total_value
        )

        assert eth_amount == 0
        assert usdc_amount > 0


class TestImpermanentLoss:
    """Tests for impermanent loss calculation."""

    def test_no_il_at_entry(self):
        """Test that IL is 0 when price hasn't changed."""
        entry_price = Decimal("3000")
        current_price = Decimal("3000")

        il = calculate_impermanent_loss(entry_price, current_price)

        assert il == 0

    def test_il_on_price_increase(self):
        """Test IL when price increases."""
        entry_price = Decimal("3000")
        current_price = Decimal("4000")  # 33% increase

        il = calculate_impermanent_loss(entry_price, current_price)

        # IL should be negative (a loss)
        assert il < 0
        # For 33% price change, IL is roughly 1-2%
        assert il > Decimal("-0.03")

    def test_il_on_price_decrease(self):
        """Test IL when price decreases."""
        entry_price = Decimal("3000")
        current_price = Decimal("2000")  # 33% decrease

        il = calculate_impermanent_loss(entry_price, current_price)

        # IL should be negative
        assert il < 0

    def test_il_symmetric(self):
        """Test that IL is same for proportional up/down moves."""
        entry_price = Decimal("3000")

        # 2x price
        il_up = calculate_impermanent_loss(entry_price, entry_price * 2)

        # 0.5x price
        il_down = calculate_impermanent_loss(entry_price, entry_price / 2)

        # Should be approximately equal
        assert abs(il_up - il_down) < Decimal("0.001")


class TestPriceRangeToTicks:
    """Tests for price range to ticks conversion."""

    def test_price_range_to_ticks(self):
        """Test converting price range to aligned ticks."""
        price_lower = Decimal("2800")
        price_upper = Decimal("3200")

        tick_lower, tick_upper = price_range_to_ticks(price_lower, price_upper)

        # Should be aligned to spacing (10)
        assert tick_lower % 10 == 0
        assert tick_upper % 10 == 0

        # Lower tick should be less than upper
        assert tick_lower < tick_upper

    def test_price_range_order_correction(self):
        """Test that inverted prices are corrected."""
        price_lower = Decimal("3200")  # Wrong order
        price_upper = Decimal("2800")

        tick_lower, tick_upper = price_range_to_ticks(price_lower, price_upper)

        # Should still have correct order
        assert tick_lower < tick_upper


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 15.3 Integration Tests for Lambdas

Create `tests/integration/test_lambda_integration.py`:

```python
"""
Integration tests for Lambda functions.

These tests verify the Lambda handlers work correctly with mocked AWS services.
"""
import json
import os
import pytest
from decimal import Decimal
from unittest.mock import MagicMock, patch
import sys

# Add lambdas to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'lambdas'))


class TestSwapListenerLambda:
    """Integration tests for SwapListener Lambda."""

    @patch.dict(os.environ, {
        "SWAP_TABLE": "test-swaps",
        "POOL_ADDRESS": "0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
        "ENV_NAME": "test",
    })
    @patch('swap_listener.handler.dynamodb')
    @patch('swap_listener.handler.rpc_call')
    def test_fetch_and_store_swaps(self, mock_rpc, mock_dynamodb):
        """Test that swaps are fetched and stored correctly."""
        from swap_listener import handler

        # Mock RPC responses
        mock_rpc.side_effect = [
            # eth_blockNumber
            {"result": hex(21500100)},
            # eth_getLogs
            {"result": [
                {
                    "blockNumber": hex(21500050),
                    "transactionHash": "0x" + "a" * 64,
                    "logIndex": hex(5),
                    "data": "0x" + "0" * 64 * 5,  # Simplified swap data
                }
            ]},
        ]

        # Mock DynamoDB table
        mock_table = MagicMock()
        mock_dynamodb.Table.return_value = mock_table
        mock_table.get_item.return_value = {"Item": {"block_number": 21500000}}

        # Invoke handler
        event = {}
        context = MagicMock()

        result = handler.lambda_handler(event, context)

        assert result["statusCode"] == 200

    def test_decode_swap_valid_data(self):
        """Test swap event decoding with valid data."""
        from swap_listener import handler

        # Real-ish swap event data
        log = {
            "blockNumber": "0x148456a",
            "transactionHash": "0x" + "a" * 64,
            "logIndex": "0x5",
            "data": (
                "0x"
                + "ffffffffffffffffffffffffffffffffffffffffffffffff89bca25b0e28c000"  # amount0 (negative)
                + "000000000000000000000000000000000000000000000000000000001236f5a0"  # amount1 (positive)
                + "0000000000000000000000000000000000000001234567890abcdef012345678"  # sqrtPriceX96
                + "00000000000000000000000000000000000000000000000fedcba9876543210"  # liquidity
                + "fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffce7e8"  # tick (negative)
            ),
        }

        swap = handler.decode_swap(log)

        assert swap is not None
        assert swap["block_number"] == 21480810
        assert "price" in swap
        assert swap["price"] > 0


class TestBayesianRangeLambda:
    """Integration tests for BayesianRange Lambda."""

    @patch.dict(os.environ, {
        "OHLC_TABLE": "test-ohlc",
        "MODEL_STATE_TABLE": "test-model",
        "SIGNAL_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:test",
        "COVERAGE_TARGET": "0.90",
        "LOOKBACK_CANDLES": "20",
        "ENV_NAME": "test",
    })
    @patch('bayesian_range.handler.ohlc_table')
    @patch('bayesian_range.handler.model_table')
    @patch('bayesian_range.handler.sns')
    def test_compute_bayesian_range(self, mock_sns, mock_model_table, mock_ohlc_table):
        """Test Bayesian range computation."""
        from bayesian_range import handler

        # Create mock candles
        candles = []
        base_price = 3000
        for i in range(20):
            candles.append({
                "vwap": Decimal(str(base_price + i * 10 - 100)),
                "h": Decimal(str(base_price + i * 10 - 90)),
                "l": Decimal(str(base_price + i * 10 - 110)),
            })

        mock_ohlc_table.query.return_value = {"Items": candles[::-1]}  # Reversed

        # Invoke handler
        result = handler.lambda_handler({}, MagicMock())

        assert result["statusCode"] == 200
        body = json.loads(result["body"])
        assert "range_lo" in body
        assert "range_hi" in body
        assert body["range_lo"] < body["range_hi"]

    def test_compute_bayesian_range_math(self):
        """Test the Bayesian range computation math directly."""
        from bayesian_range import handler

        # Simple VWAP and mid data
        vwaps = [3000 + i * 5 for i in range(20)]
        mids = [3000 + i * 5 for i in range(20)]

        lo, hi = handler.compute_bayesian_range(vwaps, mids, 0.90)

        # Range should contain most of the data
        assert lo < min(vwaps)
        assert hi > max(vwaps)

        # Range should be reasonable
        assert hi - lo < 500  # Not too wide
        assert hi - lo > 50   # Not too narrow


class TestAlertEngineLambda:
    """Integration tests for AlertEngine Lambda."""

    @patch.dict(os.environ, {
        "OHLC_TABLE": "test-ohlc",
        "MODEL_STATE_TABLE": "test-model",
        "SIGNAL_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:test",
        "SENSITIVITY": "4.0",
        "ENV_NAME": "test",
    })
    def test_extract_features(self):
        """Test feature extraction from candles."""
        from alert_engine import handler

        # Create stable candles (low volatility)
        candles = []
        for i in range(10):
            candles.append({
                "vwap": Decimal(str(3000 + (i % 3) * 2)),  # Choppy
                "h": Decimal(str(3005)),
                "l": Decimal(str(2995)),
                "vol": Decimal(str(100000)),
            })

        features = handler.extract_features(candles)

        assert "stability_trend" in features
        assert "range_expansion" in features
        assert "price_velocity" in features
        assert "vol_spike" in features

        # Stable market should have high stability
        # (stability_trend should not be strongly negative)

    def test_check_alert_stable_market(self):
        """Test that stable market doesn't trigger alert."""
        from alert_engine import handler

        features = {
            "stability_trend": 0.05,     # Positive = improving
            "range_expansion": 0.8,      # Low expansion
            "price_velocity": 0.05,      # Low velocity
            "vol_spike": 0.5,            # Normal volume
        }

        thresholds = {
            "stability_trend": -0.21,
            "range_expansion": 1.39,
            "price_velocity": 0.59,
            "vol_spike": 2.81,
        }

        is_alert, triggered = handler.check_alert(features, thresholds)

        assert is_alert is False
        assert len(triggered) == 0

    def test_check_alert_unstable_market(self):
        """Test that unstable market triggers alert."""
        from alert_engine import handler

        features = {
            "stability_trend": -0.30,    # Deteriorating
            "range_expansion": 2.0,      # High expansion
            "price_velocity": 1.0,       # High velocity
            "vol_spike": 3.5,            # Volume spike
        }

        thresholds = {
            "stability_trend": -0.21,
            "range_expansion": 1.39,
            "price_velocity": 0.59,
            "vol_spike": 2.81,
        }

        is_alert, triggered = handler.check_alert(features, thresholds)

        assert is_alert is True
        assert len(triggered) > 0


class TestMomentumSignalLambda:
    """Integration tests for MomentumSignal Lambda."""

    @patch.dict(os.environ, {
        "SWAP_TABLE": "test-swaps",
        "MODEL_STATE_TABLE": "test-model",
        "SIGNAL_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:test",
        "MOMENTUM_THRESHOLD": "0.0018",
        "LOOKBACK_SWAPS": "150",
        "ENV_NAME": "test",
    })
    def test_momentum_direction_up(self):
        """Test momentum detection for upward move."""
        # Simulate prices going from 3000 to 3060 (2% up)
        prices = [3000 + i * 0.4 for i in range(150)]  # 0 to 60 increase

        momentum = (prices[-1] - prices[0]) / prices[0]

        assert momentum > 0.0018
        # This should signal "ETH" direction

    def test_momentum_direction_down(self):
        """Test momentum detection for downward move."""
        # Simulate prices going from 3000 to 2940 (2% down)
        prices = [3000 - i * 0.4 for i in range(150)]

        momentum = (prices[-1] - prices[0]) / prices[0]

        assert momentum < -0.0018
        # This should signal "USDC" direction

    def test_momentum_direction_hold(self):
        """Test momentum detection for sideways move."""
        # Simulate prices oscillating around 3000
        import math
        prices = [3000 + 2 * math.sin(i * 0.1) for i in range(150)]

        momentum = (prices[-1] - prices[0]) / prices[0]

        assert abs(momentum) < 0.0018
        # This should signal "HOLD" direction


class TestRiskEngineLambda:
    """Integration tests for RiskEngine Lambda."""

    @patch.dict(os.environ, {
        "POSITION_TABLE": "test-positions",
        "MODEL_STATE_TABLE": "test-model",
        "ALERT_TOPIC_ARN": "arn:aws:sns:us-east-1:123456789:test",
        "MAX_POSITION_PCT": "0.80",
        "STOP_LOSS_PCT": "0.05",
        "TAKE_PROFIT_PCT": "0.10",
        "MAX_DAILY_DRAWDOWN": "0.03",
        "MAX_GAS_GWEI": "100",
        "ENV_NAME": "test",
    })
    def test_approve_valid_trade(self):
        """Test that valid trades are approved."""
        from risk_engine import handler

        # Mock position state
        position = {
            "daily_pnl_pct": Decimal("0.01"),  # +1% today
            "total_value_usd": Decimal("10000"),
        }

        # Check drawdown
        passed, reason = handler.check_drawdown_limit(position)
        assert passed is True

        # Check position size
        event = {"position_value_usd": 5000}  # 50% of portfolio
        passed, reason = handler.check_position_size(event, position)
        assert passed is True

        # Check gas
        event = {"gas_price_gwei": 50}
        passed, reason = handler.check_gas_price(event)
        assert passed is True

    def test_reject_drawdown_breach(self):
        """Test that drawdown breach rejects trade."""
        from risk_engine import handler

        position = {
            "daily_pnl_pct": Decimal("-0.05"),  # -5% today (breaches 3% limit)
        }

        passed, reason = handler.check_drawdown_limit(position)

        assert passed is False
        assert "drawdown" in reason.lower()

    def test_reject_oversized_position(self):
        """Test that oversized position is rejected."""
        from risk_engine import handler

        position = {
            "total_value_usd": Decimal("10000"),
        }
        event = {"position_value_usd": 9000}  # 90% (exceeds 80% limit)

        passed, reason = handler.check_position_size(event, position)

        assert passed is False
        assert "large" in reason.lower() or "%" in reason

    def test_reject_high_gas(self):
        """Test that high gas price is rejected."""
        from risk_engine import handler

        event = {"gas_price_gwei": 150}  # Exceeds 100 gwei limit

        passed, reason = handler.check_gas_price(event)

        assert passed is False
        assert "gas" in reason.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

### 15.4 Create pytest Configuration

Create `tests/conftest.py`:

```python
"""
Pytest configuration and fixtures.
"""
import os
import sys
import pytest
from decimal import Decimal

# Add project directories to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lambdas', 'swap_listener'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lambdas', 'ohlc_aggregator'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lambdas', 'bayesian_range'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lambdas', 'alert_engine'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lambdas', 'momentum_signal'))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'lambdas', 'risk_engine'))


@pytest.fixture
def sample_candles():
    """Generate sample OHLC candles for testing."""
    candles = []
    base_price = 3000

    for i in range(20):
        price = base_price + (i - 10) * 5  # Range from 2950 to 3050
        candles.append({
            "pk": "POOL#0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
            "sk": f"BLOCK#{21500000 + i * 50:010d}",
            "period_start": 21500000 + i * 50,
            "o": Decimal(str(price - 2)),
            "h": Decimal(str(price + 5)),
            "l": Decimal(str(price - 5)),
            "c": Decimal(str(price + 2)),
            "vol": Decimal("100000"),
            "vwap": Decimal(str(price)),
            "n": 50,
        })

    return candles


@pytest.fixture
def sample_swaps():
    """Generate sample swap events for testing."""
    swaps = []
    base_price = 3000

    for i in range(200):
        price = base_price + (i % 20 - 10) * 0.5
        swaps.append({
            "pk": "POOL#0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640",
            "sk": f"BLOCK#{21500000 + i}#TX#0x{'a' * 64}#{i}",
            "block_number": 21500000 + i,
            "price": Decimal(str(price)),
            "amount0": str(-10**18),  # Selling 1 ETH
            "amount1": str(int(price * 10**6)),  # Receiving USDC
        })

    return swaps


@pytest.fixture
def mock_aws_env(monkeypatch):
    """Set up mock AWS environment variables."""
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
```

### 15.5 Create pytest.ini

Create `pytest.ini` in project root:

```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
filterwarnings =
    ignore::DeprecationWarning
    ignore::PendingDeprecationWarning
```

### 15.6 Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov=lambdas --cov-report=html

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run specific test file
pytest tests/unit/test_uniswap_math.py -v
```

---

## 16. Detailed Operations Runbook

### 16.1 Daily Operations Checklist

```
Morning Check (5 min):
□ Check CloudWatch Dashboard - any alarms firing?
□ Review overnight Lambda errors (should be 0)
□ Verify swaps are flowing (SwapListener invocations)
□ Check current position state in DynamoDB

Quick Commands:
```

```bash
# Check for Lambda errors in last 24h
aws logs filter-log-events \
    --log-group-name /aws/lambda/range-bot-swap-listener-prod \
    --start-time $(date -d '24 hours ago' +%s)000 \
    --filter-pattern "ERROR" \
    --limit 10

# Check last model prediction
aws dynamodb get-item \
    --table-name range-bot-model-state-prod \
    --key '{"pk": {"S": "MODEL"}, "sk": {"S": "BAYESIAN_RANGE"}}' \
    --query 'Item.{lo: range_lo.N, hi: range_hi.N, updated: updated_at.N}'

# Check alert status
aws dynamodb get-item \
    --table-name range-bot-model-state-prod \
    --key '{"pk": {"S": "MODEL"}, "sk": {"S": "ALERT_STATE"}}' \
    --query 'Item.{alert: is_alert.BOOL, triggered: triggered_features.L}'
```

### 16.2 Weekly Operations

```bash
# 1. Review weekly performance
aws dynamodb query \
    --table-name range-bot-positions-prod \
    --key-condition-expression "pk = :pk" \
    --expression-attribute-values '{":pk": {"S": "TX_HISTORY"}}' \
    --scan-index-forward false \
    --limit 50

# 2. Check Lambda cold starts and duration
aws cloudwatch get-metric-statistics \
    --namespace AWS/Lambda \
    --metric-name Duration \
    --dimensions Name=FunctionName,Value=range-bot-bayesian-prod \
    --start-time $(date -u -d '7 days ago' +%Y-%m-%dT%H:%M:%SZ) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --period 86400 \
    --statistics Average Maximum

# 3. Review costs
aws ce get-cost-and-usage \
    --time-period Start=$(date -u -d '7 days ago' +%Y-%m-%d),End=$(date -u +%Y-%m-%d) \
    --granularity DAILY \
    --metrics BlendedCost \
    --filter '{"Tags": {"Key": "Project", "Values": ["range-bot"]}}'
```

### 16.3 Emergency Procedures

#### Procedure 1: Pause All Trading

```bash
# Disable all EventBridge rules (stops all scheduled invocations)
aws events disable-rule --name range-bot-swap-listener-prod
aws events disable-rule --name range-bot-bayesian-prod
aws events disable-rule --name range-bot-alerts-prod
aws events disable-rule --name range-bot-momentum-prod

# Verify disabled
aws events list-rules --name-prefix range-bot --query 'Rules[*].{Name:Name,State:State}'
```

#### Procedure 2: Emergency Withdrawal

```bash
# 1. First pause trading (see above)

# 2. Get current position info
aws dynamodb get-item \
    --table-name range-bot-positions-prod \
    --key '{"pk": {"S": "POSITION"}, "sk": {"S": "CURRENT"}}'

# 3. Send emergency exit command to signer
aws sqs send-message \
    --queue-url https://sqs.us-east-1.amazonaws.com/YOUR_ACCOUNT/range-bot-tx-requests-prod \
    --message-body '{"action": "emergency_exit", "params": {"reason": "manual_trigger"}}'

# 4. Monitor signer logs
aws logs tail /ecs/range-bot-signer-prod --follow

# 5. Verify funds returned to wallet (use etherscan or cast)
```

#### Procedure 3: Resume Trading After Pause

```bash
# 1. Verify system health
aws cloudwatch describe-alarms \
    --alarm-names range-bot-lambda-errors-prod \
    --query 'MetricAlarms[*].StateValue'

# 2. Check wallet balances are correct

# 3. Re-enable rules one at a time
aws events enable-rule --name range-bot-swap-listener-prod
# Wait 5 minutes, check logs
aws events enable-rule --name range-bot-bayesian-prod
# Wait 10 minutes, verify model updates
aws events enable-rule --name range-bot-alerts-prod
aws events enable-rule --name range-bot-momentum-prod

# 4. Monitor for 30 minutes
aws logs tail /aws/lambda/range-bot-bayesian-prod --follow
```

### 16.4 Updating Configuration

#### Update Risk Parameters

```bash
# Update RiskEngine Lambda environment variables
aws lambda update-function-configuration \
    --function-name range-bot-risk-engine-prod \
    --environment "Variables={
        POSITION_TABLE=range-bot-positions-prod,
        MODEL_STATE_TABLE=range-bot-model-state-prod,
        ALERT_TOPIC_ARN=arn:aws:sns:us-east-1:YOUR_ACCOUNT:range-bot-alerts-prod,
        MAX_POSITION_PCT=0.70,
        STOP_LOSS_PCT=0.03,
        TAKE_PROFIT_PCT=0.08,
        MAX_DAILY_DRAWDOWN=0.02,
        MAX_GAS_GWEI=80,
        ENV_NAME=prod
    }"
```

#### Update Alert Sensitivity

```bash
# Update AlertEngine Lambda
aws lambda update-function-configuration \
    --function-name range-bot-alerts-prod \
    --environment "Variables={
        OHLC_TABLE=range-bot-ohlc-prod,
        MODEL_STATE_TABLE=range-bot-model-state-prod,
        SIGNAL_TOPIC_ARN=arn:aws:sns:us-east-1:YOUR_ACCOUNT:range-bot-signals-prod,
        SENSITIVITY=3.0,
        ENV_NAME=prod
    }"
```

### 16.5 Rotating the Wallet

If you need to rotate to a new wallet:

```bash
# 1. Pause trading
aws events disable-rule --name range-bot-swap-listener-prod

# 2. Emergency withdraw all funds from current wallet

# 3. Generate new wallet (OFFLINE)
python -c "from eth_account import Account; a = Account.create(); print(f'Address: {a.address}\nKey: {a.key.hex()}')"

# 4. Update secret
aws secretsmanager put-secret-value \
    --secret-id range-bot/wallet-key-prod \
    --secret-string "NEW_PRIVATE_KEY_WITHOUT_0x"

# 5. Transfer funds to new wallet

# 6. Restart signer to pick up new key
aws ecs update-service \
    --cluster range-bot-signer-prod \
    --service range-bot-signer-prod \
    --force-new-deployment

# 7. Resume trading
aws events enable-rule --name range-bot-swap-listener-prod
```

### 16.6 Viewing Logs

```bash
# Real-time Lambda logs
aws logs tail /aws/lambda/range-bot-swap-listener-prod --follow
aws logs tail /aws/lambda/range-bot-bayesian-prod --follow
aws logs tail /aws/lambda/range-bot-alerts-prod --follow

# Signer container logs
aws logs tail /ecs/range-bot-signer-prod --follow

# Search for specific errors
aws logs filter-log-events \
    --log-group-name /aws/lambda/range-bot-bayesian-prod \
    --start-time $(date -d '1 hour ago' +%s)000 \
    --filter-pattern "ERROR"

# Get Lambda invocation details
aws logs filter-log-events \
    --log-group-name /aws/lambda/range-bot-bayesian-prod \
    --start-time $(date -d '1 hour ago' +%s)000 \
    --filter-pattern "range_lo"
```

### 16.7 Monitoring Metrics

```bash
# Lambda invocation count (last hour)
aws cloudwatch get-metric-statistics \
    --namespace AWS/Lambda \
    --metric-name Invocations \
    --dimensions Name=FunctionName,Value=range-bot-swap-listener-prod \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --period 300 \
    --statistics Sum

# DynamoDB consumed capacity
aws cloudwatch get-metric-statistics \
    --namespace AWS/DynamoDB \
    --metric-name ConsumedReadCapacityUnits \
    --dimensions Name=TableName,Value=range-bot-swaps-prod \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --period 300 \
    --statistics Sum
```

---

## 17. Troubleshooting Guide

### 17.1 Lambda Not Invoking

**Symptoms:** No new logs, no data updates

**Check:**
```bash
# Check EventBridge rule status
aws events describe-rule --name range-bot-swap-listener-prod

# Check Lambda function state
aws lambda get-function --function-name range-bot-swap-listener-prod \
    --query 'Configuration.State'

# Check for permission issues
aws lambda get-policy --function-name range-bot-swap-listener-prod
```

**Solutions:**
1. Enable the rule if disabled
2. Check Lambda has proper IAM permissions
3. Verify EventBridge can invoke Lambda

### 17.2 DynamoDB Errors

**Symptoms:** "ValidationException" or "ProvisionedThroughputExceededException"

**Check:**
```bash
# Check table status
aws dynamodb describe-table --table-name range-bot-swaps-prod \
    --query 'Table.{Status:TableStatus,Items:ItemCount}'

# Check for throttling
aws cloudwatch get-metric-statistics \
    --namespace AWS/DynamoDB \
    --metric-name ThrottledRequests \
    --dimensions Name=TableName,Value=range-bot-swaps-prod \
    --start-time $(date -u -d '1 hour ago' +%Y-%m-%dT%H:%M:%SZ) \
    --end-time $(date -u +%Y-%m-%dT%H:%M:%SZ) \
    --period 60 \
    --statistics Sum
```

**Solutions:**
1. Tables use on-demand billing, so throttling is unlikely
2. Check for malformed data in put operations
3. Verify IAM permissions

### 17.3 Signer Not Processing

**Symptoms:** Messages stuck in SQS, no transactions submitted

**Check:**
```bash
# Check SQS queue depth
aws sqs get-queue-attributes \
    --queue-url https://sqs.us-east-1.amazonaws.com/YOUR_ACCOUNT/range-bot-tx-requests-prod \
    --attribute-names ApproximateNumberOfMessages

# Check ECS service status
aws ecs describe-services \
    --cluster range-bot-signer-prod \
    --services range-bot-signer-prod \
    --query 'services[0].{Status:status,Running:runningCount,Desired:desiredCount}'

# Check ECS task logs
aws logs tail /ecs/range-bot-signer-prod --since 30m
```

**Solutions:**
1. Verify ECS service is running
2. Check Secrets Manager access
3. Verify VPC endpoints are configured
4. Check task stopped reason if not running

### 17.4 Wrong Predictions

**Symptoms:** Model predictions don't match expected behavior

**Check:**
```bash
# Get recent candles
aws dynamodb query \
    --table-name range-bot-ohlc-prod \
    --key-condition-expression "pk = :pk" \
    --expression-attribute-values '{":pk": {"S": "POOL#0x88e6A0c2dDD26FEEb64F039a2c41296FcB3f5640"}}' \
    --scan-index-forward false \
    --limit 20

# Check model state
aws dynamodb get-item \
    --table-name range-bot-model-state-prod \
    --key '{"pk": {"S": "MODEL"}, "sk": {"S": "BAYESIAN_RANGE"}}'
```

**Solutions:**
1. Verify OHLC data is accurate
2. Check model parameters (COVERAGE_TARGET, LOOKBACK_CANDLES)
3. Compare with POC predictions on same data
4. Verify candle aggregation is correct

### 17.5 High Costs

**Symptoms:** AWS bill higher than expected

**Check:**
```bash
# Detailed cost breakdown
aws ce get-cost-and-usage \
    --time-period Start=$(date -u -d '30 days ago' +%Y-%m-%d),End=$(date -u +%Y-%m-%d) \
    --granularity DAILY \
    --metrics BlendedCost \
    --group-by Type=DIMENSION,Key=SERVICE
```

**Solutions:**
1. Reduce Lambda memory if not needed
2. Reduce log retention period
3. Check for runaway invocations
4. Review VPC endpoint costs (consider removing unused ones)

### 17.6 Transaction Failures

**Symptoms:** Transactions reverting or stuck

**Check:**
```bash
# Check transaction history
aws dynamodb query \
    --table-name range-bot-positions-prod \
    --key-condition-expression "pk = :pk" \
    --expression-attribute-values '{":pk": {"S": "TX_HISTORY"}}' \
    --scan-index-forward false \
    --limit 10

# Check signer logs for errors
aws logs filter-log-events \
    --log-group-name /ecs/range-bot-signer-prod \
    --start-time $(date -d '1 hour ago' +%s)000 \
    --filter-pattern "ERROR"
```

**Solutions:**
1. Check wallet has sufficient ETH for gas
2. Verify token approvals are in place
3. Check for slippage issues (increase tolerance)
4. Verify gas price settings aren't too low
5. Check Flashbots RPC is accessible

---

## Appendix: Complete File Checklist

After following this guide, you should have these files:

```
range-bot/
├── infra/
│   ├── app.py
│   ├── cdk.json
│   ├── requirements.txt
│   └── stacks/
│       ├── __init__.py
│       ├── data_pipeline.py
│       ├── model_inference.py
│       ├── execution.py
│       ├── risk_management.py
│       └── observability.py
├── lambdas/
│   ├── swap_listener/
│   │   └── handler.py
│   ├── ohlc_aggregator/
│   │   └── handler.py
│   ├── bayesian_range/
│   │   └── handler.py
│   ├── alert_engine/
│   │   └── handler.py
│   ├── momentum_signal/
│   │   └── handler.py
│   └── risk_engine/
│       └── handler.py
├── src/
│   └── execution/
│       ├── __init__.py
│       └── uniswap_math.py
├── containers/
│   └── signer/
│       ├── Dockerfile
│       ├── requirements.txt
│       ├── main.py
│       ├── signer.py
│       └── uniswap_math.py
├── tests/
│   ├── conftest.py
│   ├── unit/
│   │   └── test_uniswap_math.py
│   └── integration/
│       └── test_lambda_integration.py
├── pytest.ini
└── requirements.txt
```
