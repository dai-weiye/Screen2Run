"""Secret-free per-call JSONL usage accounting."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config import AndroidXMLConfig, DEFAULT_CONFIG


class UsageLogger:
    """Append call metadata and maintain token/cost totals."""

    def __init__(
        self, path: str, config: AndroidXMLConfig = DEFAULT_CONFIG
    ) -> None:
        self.path = Path(path)
        self.config = config
        self._records = []

    def log_call(
        self,
        *,
        call_id: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        status: str,
        latency_ms: Optional[float] = None,
        error: Optional[str] = None,
        mode: str = "online",
        prompt_hash: Optional[str] = None,
        image_hash: Optional[str] = None,
        cache_key: Optional[str] = None,
        retry_count: int = 0,
    ) -> Dict[str, Any]:
        prompt_tokens = int(prompt_tokens)
        completion_tokens = int(completion_tokens)
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts cannot be negative.")
        input_rate, output_rate = self.config.prices_for(model)
        cost_usd = (
            prompt_tokens * input_rate + completion_tokens * output_rate
        ) / 1_000_000
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "call_id": str(call_id),
            "model": str(model),
            "mode": str(mode),
            "prompt_hash": prompt_hash,
            "image_hash": image_hash,
            "cache_key": cache_key,
            "retry_count": int(retry_count),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": round(cost_usd, 12),
            "status": str(status),
            "latency_ms": None if latency_ms is None else float(latency_ms),
            "error": None if error is None else str(error),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        self._records.append(record)
        return record

    @staticmethod
    def summarize(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        records = list(records)
        return {
            "calls": len(records),
            "prompt_tokens": sum(int(row["prompt_tokens"]) for row in records),
            "completion_tokens": sum(
                int(row["completion_tokens"]) for row in records
            ),
            "total_tokens": sum(int(row["total_tokens"]) for row in records),
            "cost_usd": round(sum(float(row["cost_usd"]) for row in records), 12),
            "retries": sum(int(row.get("retry_count", 0)) for row in records),
        }

    def summary(self) -> Dict[str, Any]:
        return self.summarize(self._records)


def summarize_jsonl(path: str) -> Dict[str, Any]:
    """Summarize a previously written usage log."""

    with Path(path).open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    return UsageLogger.summarize(records)
