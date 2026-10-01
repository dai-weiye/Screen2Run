"""OpenAI-compatible vision client with caching, retries, and hard budgets."""

import base64
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from PIL import Image

from .config import AndroidXMLConfig
from .usage_logger import UsageLogger, summarize_jsonl


class BudgetExceeded(RuntimeError):
    """Raised before a request that would exceed a configured hard cap."""


class MissingAPIKey(RuntimeError):
    """Raised only when an online request is attempted without a key."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def relay_body_unusable(content: str, usage: Optional[Dict[str, Any]] = None) -> bool:
    """True for empty/truncated relay bodies that must not freeze as answers."""
    if usage and usage.get("error"):
        return True
    text = (content or "").strip()
    if not text:
        return True
    if text.startswith("```") and text.count("```") < 2:
        return True
    return "<" in text and ">" not in text


def cache_key(
    *, image_hash: str, prompt_hash: str, model: str, commit: str, target: str
) -> str:
    payload = {
        "image": image_hash,
        "prompt": prompt_hash,
        "model": model,
        "commit": commit,
        "target": target,
    }
    return sha256_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))


class OpenAICompatibleClient:
    """One request per non-white atomic crop; import/network occur lazily."""

    def __init__(
        self,
        config: AndroidXMLConfig,
        ledger: UsageLogger,
        cache_dir: Path,
        transport: Optional[Callable[..., Any]] = None,
        sleep: Callable[[float], None] = time.sleep,
        screen_cost_cap_usd: Optional[float] = None,
    ) -> None:
        self.config = config
        self.ledger = ledger
        self.cache_dir = Path(cache_dir)
        self.transport = transport
        self.sleep = sleep
        if self.ledger.path.exists():
            prior = summarize_jsonl(str(self.ledger.path))
            self.calls = int(prior["calls"])
            self.cost_usd = float(prior["cost_usd"])
        else:
            self.calls = 0
            self.cost_usd = 0.0
        self.screen_cost_cap_usd = screen_cost_cap_usd
        self.screen_cost_start_usd = self.cost_usd

    def _check_budget(self, prompt: str, image_path: Path) -> None:
        if self.config.max_calls is not None and self.calls >= self.config.max_calls:
            raise BudgetExceeded(f"hard call cap reached ({self.config.max_calls})")
        input_rate, output_rate = self.config.prices_for(self.config.model)
        # Conservative multimodal reservation. Claude-family vision charging is
        # pixel-based; pixels/500 is deliberately stricter than the documented
        # approximately pixels/750 estimate, while prompt bytes upper-bound
        # ordinary text tokens.
        with Image.open(image_path) as opened:
            vision_tokens = (opened.width * opened.height + 499) // 500
        input_upper_bound = len(prompt.encode("utf-8")) + vision_tokens
        reservation = (
            input_upper_bound * input_rate + self.config.max_tokens * output_rate
        ) / 1_000_000
        projected = self.cost_usd + reservation
        if self.config.max_cost_usd is not None and projected > self.config.max_cost_usd:
            raise BudgetExceeded(
                f"hard cost cap would be exceeded ({projected:.8f} > "
                f"{self.config.max_cost_usd:.8f} USD)"
            )
        if (
            self.screen_cost_cap_usd is not None
            and projected - self.screen_cost_start_usd > self.screen_cost_cap_usd
        ):
            raise BudgetExceeded(
                f"screen cost cap would be exceeded "
                f"({projected - self.screen_cost_start_usd:.8f} > "
                f"{self.screen_cost_cap_usd:.8f} USD)"
            )

    def _request(self, *, prompt: str, image_data_url: str) -> Any:
        if self.transport is not None:
            return self.transport(
                model=self.config.model,
                prompt=prompt,
                image_data_url=image_data_url,
                seed=self.config.seed,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
            )
        key = os.environ.get(self.config.api_key_env)
        if not key:
            raise MissingAPIKey(
                f"{self.config.api_key_env} is required for generate/all/resume"
            )
        from openai import OpenAI  # optional dependency, never imported in dry-run

        client = OpenAI(api_key=key, base_url=self.config.base_url)
        return client.chat.completions.create(
            model=self.config.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            seed=self.config.seed,
        )

    @staticmethod
    def _unpack(response: Any) -> tuple[str, int, int]:
        if isinstance(response, dict):
            message = response["choices"][0]["message"]
            content = message.get("content")
            usage = response.get("usage", {})
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
            completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        else:
            message = response.choices[0].message
            content = getattr(message, "content", None)
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        if isinstance(content, list):
            parts = []
            for part in content:
                if isinstance(part, dict):
                    parts.append(str(part.get("text") or ""))
                else:
                    parts.append(str(getattr(part, "text", None) or ""))
            content = "".join(parts)
        return content or "", prompt_tokens, completion_tokens

    def generate(self, image_path: Path, prompt: str, call_id: str) -> Dict[str, Any]:
        image_path = Path(image_path)
        image = image_path.read_bytes()
        image_hash = sha256_bytes(image)
        prompt_hash = sha256_bytes(prompt.encode("utf-8"))
        key = cache_key(
            image_hash=image_hash,
            prompt_hash=prompt_hash,
            model=self.config.model,
            commit=self.config.upstream_commit,
            target=self.config.target,
        )
        cached = self.cache_dir / f"{key}.json"
        if cached.exists():
            cached_result = json.loads(cached.read_text(encoding="utf-8"))
            if not relay_body_unusable(
                cached_result.get("content") or "",
                cached_result.get("usage"),
            ):
                return cached_result
            cached.unlink()

        self._check_budget(prompt, image_path)
        last_error: Optional[Exception] = None
        image_data_url = (
            "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        )
        for attempt in range(self.config.max_retries + 1):
            started = time.perf_counter()
            response = None
            try:
                response = self._request(
                    prompt=prompt, image_data_url=image_data_url
                )
            except MissingAPIKey:
                raise
            except Exception as exc:  # provider errors are intentionally generic
                last_error = exc
                latency_ms = (time.perf_counter() - started) * 1000
                self.calls += 1
                self.ledger.log_call(
                    call_id=call_id, model=self.config.model, prompt_tokens=0,
                    completion_tokens=0, status="error", latency_ms=latency_ms,
                    error=type(exc).__name__, prompt_hash=prompt_hash,
                    image_hash=image_hash, cache_key=key, retry_count=attempt,
                )
                if attempt >= self.config.max_retries:
                    raise RuntimeError(
                        f"provider request failed after {attempt} retries"
                    ) from last_error
                self.sleep(2 ** attempt)
                continue

            latency_ms = (time.perf_counter() - started) * 1000
            self.calls += 1
            content, prompt_tokens, completion_tokens = self._unpack(response)
            if relay_body_unusable(content):
                # Empty 1-token stop or a cut-off fence/tag is a relay miss,
                # not a LayoutCoder atomic answer. Never cache it.
                error_name = (
                    "EmptyRelayReply" if not content.strip() else "TruncatedRelayReply"
                )
                last_error = RuntimeError(error_name)
                record = self.ledger.log_call(
                    call_id=call_id, model=self.config.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    status="error", latency_ms=latency_ms,
                    error=error_name, prompt_hash=prompt_hash,
                    image_hash=image_hash, cache_key=key, retry_count=attempt,
                )
                self.cost_usd += float(record["cost_usd"])
                if attempt >= self.config.max_retries:
                    return {
                        "content": content,
                        "cache_key": key,
                        "usage": record,
                    }
                self.sleep(2 ** attempt)
                continue

            record = self.ledger.log_call(
                call_id=call_id, model=self.config.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                status="ok", latency_ms=latency_ms, prompt_hash=prompt_hash,
                image_hash=image_hash, cache_key=key, retry_count=attempt,
            )
            self.cost_usd += float(record["cost_usd"])
            if (
                self.config.max_cost_usd is not None
                and self.cost_usd > self.config.max_cost_usd
            ):
                raise BudgetExceeded(
                    "provider-reported usage exceeded the hard cost cap"
                )
            result = {"content": content, "cache_key": key, "usage": record}
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(
                json.dumps(result, sort_keys=True) + "\n", encoding="utf-8"
            )
            return result

        raise RuntimeError("provider request failed after retries") from last_error
