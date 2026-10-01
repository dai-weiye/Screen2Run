#!/usr/bin/env python3
"""Direct, chain-of-thought, and self-refinement Android prompt baselines.

The call, retry, extraction, and refinement routines retain the experimental
implementation. Only repository paths and the command-line wrapper are changed.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
import fcntl
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths
from baseline_model_options import chat_completion_kwargs

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

def load_prompt(name):
    return (Path(__file__).resolve().parent / "prompts" / name).read_text(encoding="utf-8")

def prompt_too_long(exc: BaseException) -> bool:
    return is_prompt_too_long(exc)

def _stop_file() -> Path | None:
    raw = os.environ.get("ESE_SHARED_STOP_FILE")
    return Path(raw) if raw else None

def _request_stop(reason: str) -> None:
    path = _stop_file()
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reason.strip() + "\n", encoding="utf-8")

def _stop_requested() -> str | None:
    path = _stop_file()
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return text or "stop file"

def _read_shared_cost() -> float | None:
    path = os.environ.get("ESE_SHARED_COST_LEDGER")
    if not path:
        return None
    ledger = Path(path)
    if not ledger.is_file():
        return 0.0
    try:
        return float(ledger.read_text(encoding="utf-8").strip() or 0.0)
    except ValueError:
        return 0.0

def _guard_shared_budget() -> None:
    reason = _stop_requested()
    if reason:
        raise SystemExit(f"shared stop: {reason}")
    cap_raw = os.environ.get("ESE_SHARED_COST_CAP_USD")
    if not cap_raw:
        return
    total = _read_shared_cost()
    if total is not None and total > float(cap_raw):
        msg = (
            f"shared cost guard: ${total:.4f} exceeds ESE_SHARED_COST_CAP_USD "
            f"{float(cap_raw):.4f}"
        )
        _request_stop(msg)
        raise SystemExit(msg)

def _bump_shared_cost(delta: float) -> float | None:
    """Process-wide spend file so parallel workers share one dollar cap."""
    path = os.environ.get("ESE_SHARED_COST_LEDGER")
    if not path:
        return None
    ledger = Path(path)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    with ledger.open("a+", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        handle.seek(0)
        raw = handle.read().strip()
        try:
            total = float(raw) if raw else 0.0
        except ValueError:
            total = 0.0
        total += float(delta)
        handle.seek(0)
        handle.truncate()
        handle.write(f"{total:.6f}\n")
        handle.flush()
    cap_raw = os.environ.get("ESE_SHARED_COST_CAP_USD")
    if cap_raw:
        cap = float(cap_raw)
        if total > cap:
            msg = (
                f"shared cost guard: ${total:.4f} exceeds ESE_SHARED_COST_CAP_USD {cap:.4f}"
            )
            _request_stop(msg)
            raise SystemExit(msg)
    return total

def quota_exhausted(exc: BaseException) -> bool:
    return is_quota_exhausted(exc)

class OpenAIClient:
    def __init__(
        self,
        client: OpenAI,
        model: str,
        max_tokens: int,
        temperature: float,
        timeout_seconds: float = 180.0,
        max_retries: int = 2,
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
        max_total_cost_usd: float | None = None,
        api_model: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.api_model = api_model or model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.max_total_cost_usd = max_total_cost_usd
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.estimated_cost_usd = 0.0
        self.empty_replies = 0
        self.call_log: list[dict] = []
        # Provider finish_reason of the last reply; "length" lets the Reviewer report
        # truncation instead of malformed XML.
        self.last_finish_reason: str | None = None

    def complete(self, prompt: str, image_path: Path, expect: str = "any") -> str:
        import base64

        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 6):
            try:
                _guard_shared_budget()
                kwargs = chat_completion_kwargs(
                    model=self.api_model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {"type": "image_url", "image_url": {"url": url}},
                            ],
                        }
                    ],
                    max_output_tokens=self.max_tokens,
                    temperature=self.temperature,
                    timeout_seconds=self.timeout_seconds,
                )
                # Official V4.1 Flash thinks by default; CoT is billed as output and
                # breaks Layout/Visual JSON. AgentRouter extra_body is left alone.
                base = str(getattr(self.client, "base_url", "") or "")
                # S2R_DEEPSEEK_THINKING=default keeps the vendor default (thinking on) so every
                # arm of a same-backbone comparison runs the model in the same mode
                if "deepseek.com" in base and os.environ.get("S2R_DEEPSEEK_THINKING", "disabled") != "default":
                    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
                response = self.client.chat.completions.create(**kwargs)
                self.calls += 1
                usage = response.usage
                prompt_tokens = completion_tokens = reasoning = 0
                if usage is not None:
                    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                    details = getattr(usage, "completion_tokens_details", None)
                    reasoning = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
                self.input_tokens += prompt_tokens
                self.output_tokens += completion_tokens
                self.reasoning_tokens += reasoning
                cost = (
                    prompt_tokens * self.input_price_per_million
                    + completion_tokens * self.output_price_per_million
                ) / 1_000_000
                self.estimated_cost_usd += cost
                _bump_shared_cost(cost)
                choice = response.choices[0]
                self.last_finish_reason = getattr(choice, "finish_reason", None)
                self.call_log.append(
                    {
                        "screen_id": getattr(self, "current_screen", None),
                        "model": getattr(response, "model", self.model),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "reasoning_tokens": reasoning,
                        "finish_reason": self.last_finish_reason,
                        "estimated_cost_usd": round(cost, 6),
                    }
                )
                if self.max_total_cost_usd is not None and self.estimated_cost_usd > self.max_total_cost_usd:
                    raise SystemExit(
                        f"cost guard: ${self.estimated_cost_usd:.4f} exceeds --max-total-cost-usd "
                        f"{self.max_total_cost_usd:.4f}; stopping after this call"
                    )
                content = choice.message.content or ""
                # Preserve the frozen request and response compatibility behavior.
                #
                # Preserve the frozen request and response compatibility behavior.
                #
                # Preserve the frozen request and response compatibility behavior.
                stub = len(content.strip()) < 16
                wrong_shape = expect == "xml" and not looks_like_android_xml(content)
                if (stub or wrong_shape) and self.last_finish_reason != "length":
                    self.call_log[-1]["empty_reply"] = True
                    self.empty_replies += 1
                    if attempt < self.max_retries:
                        time.sleep(2 ** attempt)
                        continue
                return content
            except Exception as exc:  # noqa: BLE001 - retries are for the pilot, not hidden.
                last_error = exc
                if (
                    prompt_too_long(exc)
                    or quota_exhausted(exc)
                    or is_request_timeout(exc)
                    or is_content_blocked(exc)
                ):
                    break
                conn = "Connection error" in str(exc) or "UNEXPECTED_EOF" in str(exc)
                limit = self.max_retries + (5 if conn else 0)
                if attempt >= limit:
                    break
                time.sleep(min(45.0, (4 if conn else 1) * (2 ** attempt)))
        if last_error is None:
            # every attempt came back empty
            return ""
        if quota_exhausted(last_error):
            _request_stop(f"quota exhausted: {last_error}")
            raise SystemExit(f"quota exhausted: {last_error}") from last_error
        raise RuntimeError(f"model call failed after retries: {last_error}") from last_error

def run_direct_arm(
    client: OpenAIClient,
    screenshot: Path,
    output_dir: Path,
    *,
    self_refine: bool,
    cot: bool = False,
    on_call=None,
) -> dict:

    output_dir.mkdir(parents=True, exist_ok=True)
    board: dict = {
        "screen_id": screenshot.stem,
        "variant": "self_refine" if self_refine else ("cot" if cot else "direct"),
        "stages": [],
        "layout_errors": [],
        "visual_topology_reverted": False,
        "early_stop": False,
        "review_rounds": 1 if self_refine else 0,
        "continuations": 0,
        "asset_binding": "none",
        "missing_crop": True,
        "crop_count": 0,
    }

    def call(stage: str, prompt: str) -> str:
        # Preserve the frozen request and response compatibility behavior.
        raw = client.complete(prompt, screenshot, expect="xml")
        if on_call:
            on_call(stage, prompt, raw)
        board["stages"].append(stage)
        return raw

    first_prompt = load_prompt("cot_agent.txt" if cot else "direct_agent.txt")
    xml = extract_xml(call("cot" if cot else "direct", first_prompt))
    (output_dir / "draft.xml").write_text(xml + "\n", encoding="utf-8")
    if self_refine:
        xml = extract_xml(
            call(
                "self_refine",
                f"{load_prompt('self_refine_agent.txt')}\n\nCURRENT XML:\n{xml}",
            )
        )
    if not looks_like_android_xml(xml):
        board["failed_stage"] = "direct"
    (output_dir / "repaired.xml").write_text(xml + "\n", encoding="utf-8")
    (output_dir / "final.xml").write_text(xml + "\n", encoding="utf-8")
    board["final_xml"] = "final.xml"
    board["hierarchy"] = xml_hierarchy_stats(xml)
    write_json(output_dir / "board.json", board)
    return board

_REASONING_BLOCK = re.compile(
    r"<(think|thinking|reasoning)\b[^>]*>.*?</\1\s*>", re.I | re.S
)

def strip_reasoning(value: str) -> str:
    return _REASONING_BLOCK.sub("", value)

def extract_xml(value: str) -> str:
    text = strip_reasoning(value)
    fenced = re.search(r"```(?:xml)?\s*(.*?)```", text, re.I | re.S)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("<"), text.rfind(">")
    if 0 <= start < end:
        text = text[start : end + 1]
    return text.strip()

def looks_like_android_xml(value: str) -> bool:
    text = value.strip()
    return text.startswith("<") and ">" in text

_TAG_RE = re.compile(r"<(/?)([A-Za-z][\w.:-]*)(?:\s[^<>]*?)?(/?)>")

CONTAINER_VIEWS = {
    "LinearLayout",
    "RelativeLayout",
    "FrameLayout",
    "ScrollView",
    "HorizontalScrollView",
    "ListView",
    "GridView",
}

def xml_hierarchy_stats(xml_text: str) -> dict[str, int]:
    """Depth and container counts of a layout, for the structure ledger.

    ``max_depth`` counts nesting below the root (a flat layout scores 1);
    ``containers`` counts ViewGroup elements other than the root; ``elements``
    counts every element. Works on truncated text too, so it never raises.
    """
    depth = 0
    max_depth = 0
    elements = 0
    containers = 0
    seen_root = False
    for match in _TAG_RE.finditer(xml_text):
        closing, tag, self_closing = match.group(1), match.group(2), match.group(3)
        if tag.startswith("?") or tag.startswith("!"):
            continue
        if closing:
            depth = max(0, depth - 1)
            continue
        elements += 1
        if seen_root and tag.split(".")[-1] in CONTAINER_VIEWS:
            containers += 1
        seen_root = True
        if self_closing:
            max_depth = max(max_depth, depth)
            continue
        depth += 1
        max_depth = max(max_depth, depth)
    return {"max_depth": max_depth, "containers": containers, "elements": elements}

TOO_LONG_MARKERS = (
    "Prompt is too long",
    "Request too large",
    "32MB request limit",
)

def is_prompt_too_long(exc: BaseException | str) -> bool:
    text = str(exc)
    return any(marker in text for marker in TOO_LONG_MARKERS)

def is_request_timeout(exc: BaseException | str) -> bool:
    lowered = str(exc).lower()
    return "timed out" in lowered or "timeout" in lowered

QUOTA_MARKERS = (
    "insufficient_user_quota",
    "insufficient_quota",
    "insufficient credits",
    "credit is insufficient",
    "\u9884\u6263\u8d39",
    "balance is insufficient",
    "\u4f59\u989d\u4e0d\u8db3",
    "\u989d\u5ea6\u4e0d\u8db3",
    "\u989d\u5ea6\u4e0d\u591f",
    "Error code: 402",
    "You exceeded your current quota",
    "quota is not enough",
    "quota exceeded",
    "no remaining quota",
)

def is_quota_exhausted(exc: BaseException | str) -> bool:
    text = str(exc)
    return any(marker in text for marker in QUOTA_MARKERS)

CONTENT_BLOCK_MARKERS = (
    "sensitive_words_detected",
    "sensitive words detected",
    "content-blocked",
    "content_blocked",
)

def is_content_blocked(exc: BaseException | str) -> bool:
    lowered = str(exc).lower()
    return any(marker in lowered for marker in CONTENT_BLOCK_MARKERS)

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=["direct", "cot", "self_refine"], required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--input-price", type=float, required=True, help="USD per million input tokens")
    parser.add_argument("--output-price", type=float, required=True, help="USD per million output tokens")
    parser.add_argument("--max-total-cost-usd", type=float, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY; credentials are not stored in the release.")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty; keep each run independent.")
    from openai import OpenAI
    client = OpenAIClient(OpenAI(base_url=args.base_url), args.model, args.max_tokens,
        args.temperature, max_retries=args.max_retries,
        input_price_per_million=args.input_price, output_price_per_million=args.output_price,
        max_total_cost_usd=args.max_total_cost_usd)
    run_direct_arm(client, args.image, args.out,
        self_refine=args.arm == "self_refine", cot=args.arm == "cot",
        on_call=lambda stage, prompt, raw: write_json(args.out / (stage + "_response.json"),
            {"prompt": prompt, "response": raw}))
    write_json(args.out / "usage.json", client.call_log)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
