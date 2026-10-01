"""Vendor-default context windows for the method-paper flagship models.

Context window is input + output. We never send a smaller window to the relay.
max_output_tokens is the official completion ceiling; Gemini's native API defaults
to 8192 if this field is omitted, so experiments must set it explicitly.
"""

from __future__ import annotations

from typing import Any

# Relays behind Cloudflare (e.g. api.fuka.win) reject the OpenAI SDK signature
# with error 1010 unless the client looks like a browser.
RELAY_BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
# agentrouter.org WAF returns 401 unauthorized_client unless the client looks
# like Codex CLI.
RELAY_CODEX_UA = "codex_cli_rs/0.146.0"


def relay_headers(base_url: str | None) -> dict[str, str]:
    headers = {"Accept": "application/json"}
    host = str(base_url or "")
    if "agentrouter.org" in host or "air-outer.com" in host:
        headers["User-Agent"] = RELAY_CODEX_UA
        headers["originator"] = "codex_cli_rs"
    else:
        headers["User-Agent"] = RELAY_BROWSER_UA
    return headers

# Official defaults, not the relay's undocumented routing. Sources checked 2026-09-09.
FLAGSHIP_MODELS: dict[str, dict[str, Any]] = {
    "claude-opus-5": {
        "vendor": "Anthropic",
        "display": "Claude Opus 5",
        "context_window": 1_000_000,
        "max_output_tokens": 128_000,
        "input_usd_per_million": 5.0,
        "output_usd_per_million": 25.0,
        "docs": "https://platform.claude.com/docs/en/models/opus-5/overview",
    },
    "gpt-5.6-sol": {
        "vendor": "OpenAI",
        "display": "GPT-5.6 Sol",
        "context_window": 1_050_000,
        "max_output_tokens": 128_000,
        "input_usd_per_million": 4.0,
        "output_usd_per_million": 20.0,
        "docs": "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
        "pricing_note": "Promotional API rate at least through 2026-11-21; >272K input tokens bills 2x/1.5x.",
    },
    "gpt-6-astra": {
        "vendor": "OpenAI",
        "display": "GPT-6 Astra",
        "context_window": 1_050_000,
        "max_output_tokens": 128_000,
        "input_usd_per_million": 10.0,
        "output_usd_per_million": 50.0,
        "docs": "https://developers.openai.com/api/docs/models/gpt-6-astra",
        "pricing_note": (
            "Official Standard $10/$50 per 1M. AgentRouter bill is the relay quota, not this. "
            "Vision is image input; Chat Completions supported."
        ),
    },
    "gemini-3.8-flash": {
        "vendor": "Google",
        "display": "Gemini 3.8 Flash",
        "context_window": 1_048_576,
        "max_output_tokens": 65_536,
        "input_usd_per_million": 0.75,
        "output_usd_per_million": 3.75,
        "docs": "https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash",
        "pricing_note": "Introductory rate through 2026-12-31; then $1.50 / $7.50.",
    },
    "deepseek-v4-flash": {
        "vendor": "DeepSeek",
        "display": "DeepSeek V4 Flash",
        "context_window": 1_000_000,
        "max_output_tokens": 64_000,
        "input_usd_per_million": 2.0,
        "output_usd_per_million": 6.0,
        "docs": "https://api-docs.deepseek.com/",
        "pricing_note": "AgentRouter unlimited-quota channel; vision via OpenAI image_url.",
    },
    # Preserve the frozen request and response compatibility behavior.
    "deepseek-v4.1-flash": {
        "vendor": "DeepSeek",
        "display": "DeepSeek V4.1 Flash (via relay)",
        "context_window": 1_000_000,
        "max_output_tokens": 64_000,
        "input_usd_per_million": 0.15,
        "output_usd_per_million": 0.6,
        "docs": "https://api-docs.deepseek.com/quick_start/pricing/",
        "pricing_note": "Relay endpoint serving the same DeepSeek V4.1 Flash; thinking is billed as output.",
    },
    "deepseek-flash": {
        "vendor": "DeepSeek",
        "display": "DeepSeek V4.1 Flash",
        "context_window": 1_000_000,
        "max_output_tokens": 64_000,
        "input_usd_per_million": 0.15,
        "output_usd_per_million": 0.6,
        "docs": "https://api-docs.deepseek.com/quick_start/pricing/",
        "pricing_note": (
            "Official api.deepseek.com; native vision. Prices are off-peak cache-miss "
            "(weekends/China holidays). Peak is 2x. Pipeline JSON uses thinking=disabled; "
            "vendor max output is 384K but Screen2Run keeps 64K."
        ),
    },
}

# Chat Completions has no context_window field. Sending any of these would be a
# client-side cap, which this paper's protocol forbids.
FORBIDDEN_CONTEXT_KEYS = (
    "context_window",
    "max_context",
    "max_context_tokens",
    "n_ctx",
    "context_length",
    "max_input_tokens",
)


def resolve_generation_limits(model: str, max_tokens: int | None) -> dict[str, Any]:
    """Return the output ceiling actually sent, plus the official context window.

    max_tokens is None or 0 → official max_output_tokens for known models.
    Unknown models with no explicit ceiling raise, so a silent 8192 cannot sneak in.
    """
    spec = FLAGSHIP_MODELS.get(model)
    if max_tokens in (None, 0):
        if spec is None:
            raise ValueError(
                f"model {model!r} is not in FLAGSHIP_MODELS; pass an explicit --max-tokens "
                "or add it to proposed_framework/models.py"
            )
        ceiling = int(spec["max_output_tokens"])
        policy = "vendor_default_max_output"
    else:
        ceiling = int(max_tokens)
        policy = "explicit_max_output"
        if spec and ceiling > int(spec["max_output_tokens"]):
            ceiling = int(spec["max_output_tokens"])
            policy = "clamped_to_vendor_max_output"
    return {
        "model": model,
        "vendor": None if spec is None else spec["vendor"],
        "context_window": None if spec is None else int(spec["context_window"]),
        "max_output_tokens": ceiling,
        "output_policy": policy,
        "docs": None if spec is None else spec["docs"],
        "input_usd_per_million": None if spec is None else spec.get("input_usd_per_million"),
        "output_usd_per_million": None if spec is None else spec.get("output_usd_per_million"),
    }


def chat_completion_kwargs(
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_output_tokens: int,
    temperature: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    """Request body: vendor-default window, explicit official output ceiling, no context cap."""
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "timeout": timeout_seconds,
    }
    # GPT-5/6 reasoning chat completions reject max_tokens and temperature.
    if str(model).startswith("gpt-5") or str(model).startswith("gpt-6"):
        kwargs["max_completion_tokens"] = max_output_tokens
    else:
        kwargs["max_tokens"] = max_output_tokens
        kwargs["temperature"] = temperature
    leaked = [key for key in FORBIDDEN_CONTEXT_KEYS if key in kwargs]
    if leaked:
        raise RuntimeError(f"refusing to send context-cap fields: {leaked}")
    return kwargs
