#!/usr/bin/env python3
"""Explicit model-generation plan; no provider switching or stored credentials.

The default is a dry run. --execute launches the selected generation CLI with
the caller's OPENAI_API_KEY environment variable. No credentials are read from
files, printed, written to logs, or added to command arguments. Configure a
provider-side spending cap before running a paid generation experiment.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("ours", "no_planning", "no_visual", "direct", "cot", "self_refine", "dcgen", "layoutcoder"), required=True)
    parser.add_argument("--model", required=True, help="Exact provider model identifier")
    parser.add_argument("--base-url", required=True, help="Explicit OpenAI-compatible provider endpoint")
    parser.add_argument("--cohort", type=Path, required=True, help="One image path per line, relative to SCREEN2RUN_SCREENSHOTS or absolute")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-screens", type=int, required=True, help="Explicit limit; use a small smoke test before the full cohort")
    parser.add_argument("--input-price", type=float, help="Baseline input-token price in USD per million")
    parser.add_argument("--output-price", type=float, help="Baseline output-token price in USD per million")
    parser.add_argument("--max-estimated-cost-usd", type=float,
                        help="Baseline accounting cap divided across screens; post-call estimates are not a provider-enforced hard cap")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.max_screens < 1:
        parser.error("max-screens must be positive")
    images = [Path(line.strip()) for line in args.cohort.read_text(encoding="utf-8").splitlines() if line.strip()]
    images = [(p if p.is_absolute() else release_paths.SCREENSHOTS / p).resolve() for p in images][:args.max_screens]
    if not images or len({p.stem for p in images}) != len(images):
        parser.error("Cohort is empty or contains duplicate screen IDs")
    is_framework = args.method in ("ours", "no_planning", "no_visual")
    if not is_framework and (args.input_price is None or args.output_price is None or args.max_estimated_cost_usd is None):
        parser.error("Baseline runs require --input-price, --output-price, and --max-estimated-cost-usd")
    if any(value is not None and value < 0 for value in (args.input_price, args.output_price, args.max_estimated_cost_usd)):
        parser.error("Prices and estimated spending cap cannot be negative")
    out = args.out.resolve()
    env = dict(os.environ)
    env["OPENAI_BASE_URL"] = args.base_url
    env["S2R_MAX_TOKENS"] = "64000"
    env.pop("S2R_REPLAY_ROOT", None)
    env.pop("S2R_ABLATION", None)
    if args.method in ("no_planning", "no_visual"):
        env["S2R_ABLATION"] = "no_s1" if args.method == "no_planning" else "no_s2"
    commands = []
    for image in images:
        if is_framework:
            command = [sys.executable, str(release_paths.module_file("run_model_sessions.py")), "--screens", image.stem + ":custom",
                       "--shot-list", str(args.cohort.resolve()), "--candidate", str(out), "--model", args.model,
                       "--base-url", args.base_url, "--workers", "1"]
        else:
            command = [sys.executable, str(release_paths.module_file("run_baseline.py")), "--arm", args.method,
                       "--image", str(image), "--out", str(out / "full" / image.stem), "--model", args.model,
                       "--base-url", args.base_url, "--input-price", str(args.input_price),
                       "--output-price", str(args.output_price), "--max-total-cost-usd", str(args.max_estimated_cost_usd / len(images))]
        commands.append((image, command))
        print(shlex.join(command))
    if not args.execute:
        print(f"Plan only: {len(commands)} screens; no model calls.")
        return 0
    if not env.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY in the process environment")
    if out.exists():
        parser.error("Use a fresh output directory to preserve earlier generations")
    if any(not image.is_file() for image, _ in commands):
        parser.error("One or more reference images are missing")
    out.mkdir(parents=True)
    records = []
    for image, command in commands:
        # Child scripts record request/response evidence and usage. Avoid copying
        # arbitrary provider errors into shared logs, where URLs may be sensitive.
        result = subprocess.run(command, env=env, cwd=release_paths.RELEASE)
        records.append({"screen_id": image.stem, "returncode": result.returncode, "model": args.model, "method": args.method})
        (out / "generation_progress.json").write_text(json.dumps(records, indent=2) + "\n")
        if result.returncode:
            print(f"Stopped after {image.stem}; no later calls were launched.", file=sys.stderr)
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
