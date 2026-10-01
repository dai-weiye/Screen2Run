"""Configuration for the dependency-free Android adapter."""

from dataclasses import dataclass, field
from typing import FrozenSet

UPSTREAM_COMMIT = "22af69b238fb5aa967cf3f05985a25b52481105f"
ANDROID_NS = "http://schemas.android.com/apk/res/android"
SEGMENT_ID_PREFIX = "dcgen_seg_"


@dataclass(frozen=True)
class AdapterConfig:
    """Deterministic conversion settings.

    ``px_per_dp`` is intentionally fixed per run: this adapter does not infer
    Android display density from screenshots.
    """

    px_per_dp: float = 1.0
    declared_resources: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"@drawable/img"})
    )

    def __post_init__(self) -> None:
        if self.px_per_dp <= 0:
            raise ValueError("px_per_dp must be greater than zero")
