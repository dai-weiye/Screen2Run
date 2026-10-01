"""Token, latency, and cost accounting without provider dependencies."""

from dataclasses import asdict, dataclass
from typing import Any, Dict


@dataclass(frozen=True)
class UsageRecord:
    stage: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def calculate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_million: float,
    output_usd_per_million: float,
) -> float:
    values = (
        input_tokens,
        output_tokens,
        input_usd_per_million,
        output_usd_per_million,
    )
    if any(value < 0 for value in values):
        raise ValueError("usage and prices must be non-negative")
    return (
        input_tokens * input_usd_per_million
        + output_tokens * output_usd_per_million
    ) / 1_000_000


def make_usage_record(
    stage: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    input_usd_per_million: float = 0.0,
    output_usd_per_million: float = 0.0,
) -> UsageRecord:
    return UsageRecord(
        stage=stage,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        cost_usd=calculate_cost_usd(
            input_tokens,
            output_tokens,
            input_usd_per_million,
            output_usd_per_million,
        ),
    )
