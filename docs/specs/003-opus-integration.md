# 003: Opus Integration - AI Chart Analysis

## Overview

Module for integrating Claude (Opus/Sonnet) as a "visual technical analyst" that contributes probabilistic inputs to the Bayesian prior. The model analyzes chart images or structured OHLC data and returns expectations about future price behavior.

## Core Concept

```
Traditional quant: Hard-coded pattern recognition (head & shoulders, wedges, etc.)
This approach: Let a multimodal LLM identify patterns and express uncertainty

The LLM doesn't need to be highly accurate - it just needs to:
1. Have some signal (better than random)
2. Be well-calibrated (know when it's uncertain)
3. Output structured, composable predictions

Even a 55% directional accuracy with good calibration improves the ensemble.
```

## Module Structure

```
src/
  opus/
    __init__.py
    types.py           # Prediction types, prompt types
    client.py          # Anthropic API client wrapper
    prompts.py         # Prompt templates (versioned)
    chart.py           # Chart image generation
    parser.py          # Response parsing and validation
    calibration.py     # Track and adjust for model calibration
```

## Data Types

```python
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Optional

class Direction(Enum):
    UP = "up"
    DOWN = "down"
    SIDEWAYS = "sideways"

class Confidence(Enum):
    """Discretized confidence levels for stability."""
    VERY_LOW = 0.2
    LOW = 0.35
    MEDIUM = 0.5
    HIGH = 0.65
    VERY_HIGH = 0.8

@dataclass(frozen=True)
class ChartAnalysisRequest:
    """Input to Opus for analysis."""
    image_base64: Optional[str]           # Chart image (preferred)
    ohlc_data: Optional[tuple[OHLC, ...]] # Fallback: structured data
    current_price: Decimal
    timeframe: str                        # "1h", "4h", "1d"
    context: str                          # "ETH/USDC Uniswap v3"

@dataclass(frozen=True)
class ChartAnalysisPrediction:
    """Opus prediction output."""
    direction: Direction
    confidence: Confidence
    expected_range_percent: Decimal       # Expected price range as % of current
    support_level: Optional[Decimal]      # Identified support
    resistance_level: Optional[Decimal]   # Identified resistance
    pattern_identified: Optional[str]     # "ascending triangle", "double bottom", etc.
    reasoning: str                        # Brief explanation
    raw_response: str                     # Full model response for debugging

@dataclass(frozen=True)
class OpusPriorContribution:
    """
    Transformed prediction suitable for Bayesian combination.
    """
    center_adjustment: Decimal   # Shift from VWAP center (+ = up, - = down)
    scale_multiplier: Decimal    # Widen/narrow prior (>1 = more uncertain)
    weight: float                # How much to weight this input (0-1)

@dataclass(frozen=True)
class CalibrationRecord:
    """Track prediction accuracy over time."""
    timestamp: int
    prediction: ChartAnalysisPrediction
    actual_direction: Direction
    actual_range_percent: Decimal
    was_correct: bool
```

## Core Functions

### Chart Generation

```python
def generate_chart_image(
    ohlc_data: Sequence[OHLC],
    indicators: Sequence[str] = ("vwap", "volume"),
    width: int = 800,
    height: int = 600
) -> bytes:
    """
    Generate a clean chart image for Opus analysis.

    Design principles:
    - Minimal decoration (no grid noise)
    - Clear candlesticks
    - VWAP overlay
    - Volume bars below
    - No text labels (let model infer from visual)

    Returns PNG bytes.
    """
    ...

def encode_chart_for_api(image_bytes: bytes) -> str:
    """Base64 encode image for API submission."""
    ...
```

### Prompting

```python
ANALYSIS_PROMPT_V1 = """
You are analyzing a {timeframe} candlestick chart for {context}.

Current price: {current_price}

Analyze this chart and provide your prediction for the next {timeframe} period.

You must respond with ONLY a JSON object in this exact format:
{
  "direction": "up" | "down" | "sideways",
  "confidence": "very_low" | "low" | "medium" | "high" | "very_high",
  "expected_range_percent": <number between 0.1 and 10.0>,
  "support_level": <number or null>,
  "resistance_level": <number or null>,
  "pattern_identified": "<pattern name or null>",
  "reasoning": "<1-2 sentence explanation>"
}

Be conservative with confidence. Only use "high" or "very_high" when patterns are very clear.
If uncertain, prefer "sideways" with "medium" confidence.
"""

def build_prompt(request: ChartAnalysisRequest, version: str = "v1") -> str:
    """Build prompt from request. Versioned for A/B testing."""
    ...
```

### API Client

```python
from typing import Union

@dataclass(frozen=True)
class APIError:
    message: str
    retryable: bool

Result = Union[ChartAnalysisPrediction, APIError]

async def analyze_chart(
    request: ChartAnalysisRequest,
    model: str = "claude-sonnet-4-20250514",
    temperature: float = 0.3,
    timeout_seconds: int = 30
) -> Result:
    """
    Submit chart to Opus/Sonnet for analysis.

    Default to Sonnet for cost efficiency - upgrade to Opus for
    complex/ambiguous charts if needed.

    Low temperature for consistency across runs.
    """
    ...

async def analyze_chart_with_retry(
    request: ChartAnalysisRequest,
    max_retries: int = 3,
    backoff_base: float = 2.0
) -> Result:
    """Retry with exponential backoff on retryable errors."""
    ...
```

### Response Parsing

```python
def parse_prediction_response(raw: str) -> Union[ChartAnalysisPrediction, ParseError]:
    """
    Parse JSON response from model.

    Strict validation:
    - All required fields present
    - Values in expected ranges
    - Confidence and direction are valid enums

    Returns ParseError if response is malformed.
    """
    ...

def validate_prediction_sanity(
    prediction: ChartAnalysisPrediction,
    current_price: Decimal
) -> list[str]:
    """
    Sanity check predictions.

    Warnings (not errors):
    - Support above current price
    - Resistance below current price
    - Expected range > 10% for hourly
    - Very high confidence with "sideways"

    Returns list of warning messages.
    """
    ...
```

### Prior Contribution

```python
def prediction_to_prior_contribution(
    prediction: ChartAnalysisPrediction,
    calibration_adjustment: float = 1.0
) -> OpusPriorContribution:
    """
    Transform raw prediction into Bayesian prior contribution.

    Mapping:
    - direction + confidence -> center_adjustment
    - expected_range_percent -> scale_multiplier
    - historical calibration -> weight

    Conservative by default: low weight until calibration is established.
    """
    direction_multiplier = {
        Direction.UP: 1.0,
        Direction.DOWN: -1.0,
        Direction.SIDEWAYS: 0.0
    }

    confidence_magnitude = {
        Confidence.VERY_LOW: 0.002,   # 0.2% shift
        Confidence.LOW: 0.005,        # 0.5% shift
        Confidence.MEDIUM: 0.01,      # 1% shift
        Confidence.HIGH: 0.015,       # 1.5% shift
        Confidence.VERY_HIGH: 0.02    # 2% shift
    }

    # Scale multiplier: higher expected range = more uncertainty
    scale = Decimal("1.0") + (prediction.expected_range_percent / Decimal("100"))

    # Weight: start conservative, adjust based on calibration
    base_weight = 0.1  # Only 10% influence initially
    weight = base_weight * calibration_adjustment

    return OpusPriorContribution(
        center_adjustment=Decimal(str(
            direction_multiplier[prediction.direction] *
            confidence_magnitude[prediction.confidence]
        )),
        scale_multiplier=scale,
        weight=weight
    )
```

### Calibration Tracking

```python
def record_prediction_outcome(
    prediction: ChartAnalysisPrediction,
    actual_ohlc: OHLC
) -> CalibrationRecord:
    """Record whether a prediction was accurate."""
    ...

def compute_calibration_adjustment(
    records: Sequence[CalibrationRecord],
    min_samples: int = 50
) -> float:
    """
    Compute adjustment factor based on historical accuracy.

    If model is 60% accurate on direction:
      adjustment = 1.2 (trust it more)
    If model is 45% accurate:
      adjustment = 0.5 (trust it less)
    If < min_samples:
      adjustment = 0.5 (conservative until proven)
    """
    ...

def compute_confidence_calibration(
    records: Sequence[CalibrationRecord]
) -> dict[Confidence, float]:
    """
    Per-confidence-level accuracy.

    Ideally: high confidence predictions are more accurate.
    If not: model confidence is not well-calibrated.

    Returns {confidence_level: actual_accuracy} mapping.
    """
    ...
```

## Model Selection Strategy

| Scenario | Model | Rationale |
|----------|-------|-----------|
| Standard hourly analysis | Sonnet | Cost-efficient, fast |
| High volatility / unusual patterns | Opus | Better reasoning |
| Backtesting (high volume) | Haiku | Cheapest, speed |

```python
def select_model(
    volatility_percentile: float,
    pattern_complexity: Optional[str] = None
) -> str:
    """Select appropriate model based on market conditions."""
    if volatility_percentile > 0.9:
        return "claude-opus-4-20250514"
    elif pattern_complexity in ("complex", "unusual"):
        return "claude-opus-4-20250514"
    else:
        return "claude-sonnet-4-20250514"
```

## Prompt Engineering Notes

### Stability Techniques
1. **Structured output**: JSON-only response reduces variance
2. **Discretized confidence**: Enum values, not free-form percentages
3. **Temperature 0.3**: Low enough for consistency, high enough for nuance
4. **Explicit uncertainty option**: "sideways" gives model an out when unsure

### Avoiding Hallucination
1. Don't ask for specific price targets (just direction + range)
2. Don't ask for time predictions (just "next period")
3. Provide current price as anchor
4. Request reasoning after prediction (not before)

## Testing Requirements

### Unit Tests
- [ ] Chart generation produces valid PNG
- [ ] Prompt building with all parameters
- [ ] Response parsing handles valid JSON
- [ ] Response parsing rejects malformed JSON
- [ ] Sanity checks catch obvious errors
- [ ] Prior contribution calculation is correct

### Integration Tests
- [ ] End-to-end API call with test image
- [ ] Retry logic works on simulated failures
- [ ] Rate limiting is respected

### Evaluation Tests
- [ ] Run on 100 historical chart images
- [ ] Compute accuracy vs random baseline
- [ ] Check confidence calibration
- [ ] Measure response time distribution

## Cost Estimation

| Model | Input (1 image + prompt) | Output (~200 tokens) | Per Call |
|-------|--------------------------|----------------------|----------|
| Haiku | ~$0.001 | ~$0.0001 | ~$0.001 |
| Sonnet | ~$0.01 | ~$0.001 | ~$0.011 |
| Opus | ~$0.05 | ~$0.01 | ~$0.06 |

At hourly frequency:
- Sonnet: ~$8/month
- Opus: ~$44/month

## Dependencies

```python
anthropic = "^0.30"      # Anthropic API client
matplotlib = "^3.8"      # Chart generation
mplfinance = "^0.12"     # Candlestick charts
pillow = "^10.0"         # Image processing
```

## Acceptance Criteria

- [ ] Chart image generation working
- [ ] API client with retry logic
- [ ] Prompt versioning system
- [ ] Response parser with validation
- [ ] Prior contribution transformer
- [ ] Calibration tracking storage
- [ ] Evaluation script for backtesting
- [ ] Documentation of prompt design rationale
- [ ] Cost monitoring/alerting hooks

## Future Enhancements

1. **A/B test prompts**: Track which prompt versions perform better
2. **Fine-tuned embeddings**: Use chart embeddings for similarity search
3. **Ensemble models**: Query multiple models, aggregate predictions
4. **Active learning**: Flag uncertain predictions for human review

## References

- [Anthropic API Docs](https://docs.anthropic.com/)
- [Vision capabilities](https://docs.anthropic.com/en/docs/build-with-claude/vision)
