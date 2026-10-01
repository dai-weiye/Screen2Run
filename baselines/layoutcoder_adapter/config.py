"""Secret-free configuration for the Android-adapted LayoutCoder runner."""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple


@dataclass(frozen=True)
class AndroidXMLConfig:
    """Runtime settings. Credentials are deliberately not represented here."""

    allow_online: bool = False
    model: str = "gpt-4.1-mini"
    base_url: Optional[str] = None
    api_key_env: str = "OPENAI_API_KEY"
    seed: int = 42
    temperature: float = 0.0
    max_tokens: int = 4096
    max_retries: int = 2
    max_calls: Optional[int] = None
    max_cost_usd: Optional[float] = None
    target: str = "android_xml"
    upstream_commit: str = "bf5b0032923ea68a0aff9f98fa9cd544d8cd9ee8"
    drawable_placeholder: str = "@drawable/img"
    input_cost_per_million: Mapping[str, float] = field(default_factory=dict)
    output_cost_per_million: Mapping[str, float] = field(default_factory=dict)

    def prices_for(self, model: str) -> Tuple[float, float]:
        return (
            float(self.input_cost_per_million.get(model, 0.0)),
            float(self.output_cost_per_million.get(model, 0.0)),
        )

    def reserved_call_cost(self) -> float:
        """Conservative output-only reservation used by the pre-call hard cap."""

        _, output_rate = self.prices_for(self.model)
        return self.max_tokens * output_rate / 1_000_000


DEFAULT_CONFIG = AndroidXMLConfig()
