"""Audited OpenAI-compatible chat-completions client."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import httpx

from .usage_logger import calculate_cost_usd


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


@dataclass(frozen=True)
class Price:
    input_usd_per_million: float
    output_usd_per_million: float


@dataclass(frozen=True)
class HardCaps:
    max_calls_global: int
    max_calls_screen: int
    max_cost_global_usd: float
    max_cost_screen_usd: float
    max_cost_call_usd: float
    max_input_tokens_call: int
    max_output_tokens_call: int


class BudgetExceeded(RuntimeError):
    pass


class TransientProviderError(RuntimeError):
    """Gateway flake: empty body, SSL EOF, or connect reset. Not a DCGen outcome."""


class Budget:
    def __init__(self, caps: HardCaps, price: Price) -> None:
        self.caps = caps
        self.price = price
        self.calls_global = 0
        self.cost_global = 0.0
        self.calls_by_screen: dict[str, int] = {}
        self.cost_by_screen: dict[str, float] = {}

    def worst_case_call_cost(self) -> float:
        return calculate_cost_usd(
            self.caps.max_input_tokens_call,
            self.caps.max_output_tokens_call,
            self.price.input_usd_per_million,
            self.price.output_usd_per_million,
        )

    def reserve(self, screen_id: str) -> None:
        worst = self.worst_case_call_cost()
        screen_calls = self.calls_by_screen.get(screen_id, 0)
        screen_cost = self.cost_by_screen.get(screen_id, 0.0)
        checks = (
            (self.calls_global + 1 <= self.caps.max_calls_global, "global call cap"),
            (screen_calls + 1 <= self.caps.max_calls_screen, "screen call cap"),
            (worst <= self.caps.max_cost_call_usd, "per-call cost cap"),
            (
                self.cost_global + worst <= self.caps.max_cost_global_usd,
                "global cost cap",
            ),
            (
                screen_cost + worst <= self.caps.max_cost_screen_usd,
                "screen cost cap",
            ),
        )
        for allowed, label in checks:
            if not allowed:
                raise BudgetExceeded(f"{label} would be exceeded")

    def commit(self, screen_id: str, cost: float) -> None:
        if cost > self.caps.max_cost_call_usd:
            raise BudgetExceeded("provider usage exceeded per-call hard cap")
        self.calls_global += 1
        self.cost_global += cost
        self.calls_by_screen[screen_id] = self.calls_by_screen.get(screen_id, 0) + 1
        self.cost_by_screen[screen_id] = self.cost_by_screen.get(screen_id, 0.0) + cost

    def restore(self, screen_id: str, audits: list[dict[str, Any]]) -> None:
        """Account for already-paid calls before a resumed request."""

        self.restore_many({screen_id: audits})

    def restore_many(
        self, audits_by_screen: dict[str, list[dict[str, Any]]]
    ) -> None:
        """Restore shared global and per-screen usage for a batch resume."""

        self.calls_by_screen = {
            screen_id: len(audits)
            for screen_id, audits in audits_by_screen.items()
        }
        self.cost_by_screen = {
            screen_id: sum(float(item.get("cost_usd", 0.0)) for item in audits)
            for screen_id, audits in audits_by_screen.items()
        }
        self.calls_global = sum(self.calls_by_screen.values())
        self.cost_global = sum(self.cost_by_screen.values())
        if self.calls_global > self.caps.max_calls_global:
            raise BudgetExceeded("resumed usage already exceeds global call cap")
        if self.cost_global > self.caps.max_cost_global_usd:
            raise BudgetExceeded("resumed usage already exceeds global cost cap")
        for screen_id, calls in self.calls_by_screen.items():
            if calls > self.caps.max_calls_screen:
                raise BudgetExceeded(
                    f"resumed usage for {screen_id} exceeds screen call cap"
                )
        for screen_id, cost in self.cost_by_screen.items():
            if cost > self.caps.max_cost_screen_usd:
                raise BudgetExceeded(
                    f"resumed usage for {screen_id} exceeds screen cost cap"
                )


@dataclass(frozen=True)
class Completion:
    content: str
    audit: dict[str, Any]


class OpenAICompatibleClient:
    """Reads credentials only from an environment variable."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key_env: str,
        model_snapshot: str,
        price: Price,
        budget: Budget,
        timeout_seconds: float = 180,
        retries: int = 8,
        retry_backoff_seconds: float = 2,
        temperature: float = 0.0,
        seed: int | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.api_key_env = api_key_env
        self.model_snapshot = model_snapshot
        self.price = price
        self.budget = budget
        self.retries = retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.temperature = temperature
        self.seed = seed
        self.client = httpx.Client(
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_seconds,
            transport=transport,
        )

    def complete(
        self,
        *,
        screen_id: str,
        stage: str,
        prompt: str,
        image_path: Path,
    ) -> Completion:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            raise ValueError(
                f"API key environment variable is unset: {self.api_key_env}"
            )
        self.budget.reserve(screen_id)
        image = image_path.read_bytes()
        input_hash = sha256_bytes(image)
        prompt_hash = sha256_text(prompt)
        payload = {
            "model": self.model_snapshot,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/png;base64,"
                                + base64.b64encode(image).decode("ascii")
                            },
                        },
                    ],
                }
            ],
            "temperature": self.temperature,
            "max_tokens": self.budget.caps.max_output_tokens_call,
        }
        if self.seed is not None:
            payload["seed"] = self.seed
        request_hash = sha256_text(json.dumps(payload, sort_keys=True))
        started = time.monotonic()
        response: httpx.Response | None = None
        body: dict[str, Any] | None = None
        content = ""
        retry_count = 0
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            retry_count = attempt
            try:
                response = self.client.post(
                    "chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                choices = body.get("choices") or []
                if not choices:
                    raise TransientProviderError("provider returned no choices")
                content = str((choices[0].get("message") or {}).get("content") or "")
                if not content.strip():
                    raise TransientProviderError("provider returned empty content")
                last_error = None
                break
            except (httpx.HTTPError, TransientProviderError, ValueError, KeyError) as exc:
                last_error = exc
                if attempt == self.retries:
                    break
                time.sleep(min(30.0, self.retry_backoff_seconds * (2**attempt)))
        if last_error is not None and (response is None or body is None or not content.strip()):
            raise last_error
        assert response is not None and body is not None
        latency_ms = (time.monotonic() - started) * 1000
        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        if output_tokens > self.budget.caps.max_output_tokens_call:
            raise BudgetExceeded("provider output tokens exceeded per-call hard cap")
        cost = calculate_cost_usd(
            input_tokens,
            output_tokens,
            self.price.input_usd_per_million,
            self.price.output_usd_per_million,
        )
        self.budget.commit(screen_id, cost)
        audit = {
            "screen_id": screen_id,
            "stage": stage,
            "request_hash": request_hash,
            "response_hash": sha256_bytes(response.content),
            "prompt_hash": prompt_hash,
            "input_hash": input_hash,
            "requested_model": self.model_snapshot,
            "resolved_model": body.get("model", self.model_snapshot),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "retry_count": retry_count,
            "cost_usd": cost,
            "api_key_env": self.api_key_env,
        }
        return Completion(content=content, audit=audit)


def caps_dict(caps: HardCaps) -> dict[str, Any]:
    return asdict(caps)
