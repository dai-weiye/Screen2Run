#!/usr/bin/env python3
"""CLI for the Android-adapted LayoutCoder preprocessing/generation pipeline."""

import argparse
import json
from pathlib import Path
from typing import Any

from .client import OpenAICompatibleClient
from .config import AndroidXMLConfig
from .manifest import write_run_manifest
from .pipeline import environment_check, generate, preprocess
from .structure_renderer import render_structure
from .usage_logger import UsageLogger


def _read_json(path: str) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_offline(structure_path: str, responses_path: str, output_path: str) -> str:
    """Fuse saved structure and atomic responses without any network call."""

    structure = _read_json(structure_path)
    responses = _read_json(responses_path)
    if isinstance(responses, dict) and "responses" in responses:
        responses = responses["responses"]
    xml = render_structure(structure, responses)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(xml, encoding="utf-8")
    return xml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Android-adapted LayoutCoder (upstream bf5b003; no fine-tuning)."
    )
    parser.add_argument(
        "command", choices=("preprocess", "generate", "all", "dry-run", "resume"),
        help="dry-run preprocesses and plans calls but never accesses an API",
    )
    parser.add_argument("--image", type=Path, help="Input screenshot")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--preprocessed", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--screen-id", default="1")
    parser.add_argument("--run-id", default="layoutcoder_android_pilot")
    parser.add_argument("--model", default="gpt-4.1-mini")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--max-calls", type=int)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument("--env-check", action="store_true")
    return parser


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "datasets" / "Reference").exists():
            return parent
    return current.parent


def _preprocessed_path(args: argparse.Namespace) -> Path:
    if args.preprocessed:
        return args.preprocessed
    if not args.image:
        raise SystemExit("--image or --preprocessed is required")
    return args.work_dir / "struct" / f"{args.image.stem}_android_preprocess.json"


def main() -> int:
    args = build_parser().parse_args()
    if args.env_check:
        print(json.dumps(environment_check(), indent=2, sort_keys=True))
        if args.command == "preprocess" and not args.image:
            return 0

    if args.command in ("preprocess", "all", "dry-run"):
        if not args.image:
            raise SystemExit(f"{args.command} requires --image")
        data = preprocess(args.image, args.work_dir, args.seed)
    else:
        data = _read_json(str(_preprocessed_path(args)))

    if args.command == "preprocess":
        print(json.dumps(data["report"], indent=2, sort_keys=True))
        return 0

    config = AndroidXMLConfig(
        allow_online=args.command != "dry-run",
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        seed=args.seed,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_calls=args.max_calls,
        max_cost_usd=args.max_cost_usd,
        input_cost_per_million={args.model: args.input_cost_per_million},
        output_cost_per_million={args.model: args.output_cost_per_million},
    )
    ledger_path = args.work_dir / "ledger.jsonl"
    client = OpenAICompatibleClient(
        config, UsageLogger(str(ledger_path), config), args.work_dir / "cache"
    )
    if args.command == "dry-run":
        report = generate(data, client, Path("/dev/null"), dry_run=True)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0

    if not args.output or not args.run_dir:
        raise SystemExit("generate/all/resume require --output and --run-dir")
    report = generate(data, client, args.output)
    write_run_manifest(
        run_dir=args.run_dir,
        repository_root=_repo_root(),
        run_id=args.run_id,
        screen_id=args.screen_id,
        image_path=args.image or Path(data.get("image_path", "")),
        xml_path=args.output,
        model=args.model,
        seed=args.seed,
        report=report,
        ledger_path=ledger_path,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
