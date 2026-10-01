"""Serializable provenance trace for an Android-adapted DCGenGrid run."""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List

from .config import UPSTREAM_COMMIT
from .prompts_android import prompt_hashes
from .usage_logger import UsageRecord
from .xml_assembler import count_leaves, expected_call_count


@dataclass
class AdapterTrace:
    bbox_tree: Dict[str, Any]
    model_metadata: Dict[str, Any] = field(default_factory=dict)
    usage: List[UsageRecord] = field(default_factory=list)
    upstream_commit: str = UPSTREAM_COMMIT
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    leaf_count: int = field(init=False)
    expected_calls: int = field(init=False)
    prompt_hashes: Dict[str, str] = field(default_factory=prompt_hashes)

    def __post_init__(self) -> None:
        self.leaf_count = count_leaves(self.bbox_tree)
        self.expected_calls = expected_call_count(self.leaf_count)
        forbidden = ("api_key", "secret", "token", "password", "credential")
        bad_keys = {
            key
            for key in self.model_metadata
            if any(word in key.lower() for word in forbidden)
        }
        if bad_keys:
            raise ValueError(f"model_metadata must not contain credentials: {bad_keys}")

    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result["total_input_tokens"] = sum(item.input_tokens for item in self.usage)
        result["total_output_tokens"] = sum(item.output_tokens for item in self.usage)
        result["total_latency_ms"] = sum(item.latency_ms for item in self.usage)
        result["total_cost_usd"] = sum(item.cost_usd for item in self.usage)
        return result
