#!/usr/bin/env python3
"""Run one of the five comparison baselines with explicit model settings.

No Screen2Run generation, grounding, or asset binding is added to a baseline.
Use the shared postprocessing and capture tools after generation.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths

def materialize_dcgen_output(output: Path, screen_id: str) -> None:
    """Expose the unchanged upstream XML under the shared candidate contract."""
    import hashlib
    import json
    import shutil
    source = output / screen_id / f"{screen_id}.xml"
    if not source.is_file():
        raise FileNotFoundError(f"DCGen did not produce its expected XML: {source}")
    destination = output / "final.xml"
    if destination.exists() and destination.read_bytes() != source.read_bytes():
        raise FileExistsError(f"Refusing to replace a different candidate: {destination}")
    shutil.copy2(source, destination)
    resources = []
    placeholder = source.parent / "img.xml"
    if placeholder.is_file():
        resources_dir = output / "drawables"
        resources_dir.mkdir(exist_ok=True)
        shutil.copy2(placeholder, resources_dir / placeholder.name)
        resources.append(placeholder.name)
    (output / "candidate_manifest.json").write_text(json.dumps({
        "source_xml": str(source.relative_to(output)),
        "final_xml": "final.xml", "xml_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "copied_resources": resources, "transformation": "byte-identical candidate contract copy"
    }, indent=2) + "\n", encoding="utf-8")

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True,
        choices=["direct", "cot", "self_refine", "dcgen", "layoutcoder"])
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    parser.add_argument("--max-tokens", type=int, default=64000)
    parser.add_argument("--input-price", type=float, required=True)
    parser.add_argument("--output-price", type=float, required=True)
    parser.add_argument("--max-total-cost-usd", type=float, required=True)
    parser.add_argument("--preprocessed", type=Path,
                        help="Existing LayoutCoder preprocessing JSON; otherwise preprocess the screenshot")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the exact command without calling models or preprocessing")
    args = parser.parse_args(argv)
    image = release_paths.resolve_screenshot(args.image).resolve()
    output = args.out.resolve()
    if args.max_total_cost_usd <= 0:
        parser.error("Budget must be positive")
    if not image.is_file():
        parser.error("Screenshot does not exist: " + str(image))
    if args.arm in {"direct", "cot", "self_refine"}:
        command = [sys.executable, str(Path(__file__).with_name("prompt_baselines_run.py")),
            "--arm", args.arm, "--image", str(image), "--out", str(output),
            "--model", args.model, "--max-tokens", str(args.max_tokens),
            "--input-price", str(args.input_price), "--output-price", str(args.output_price),
            "--max-total-cost-usd", str(args.max_total_cost_usd)]
    elif args.arm == "dcgen":
        from PIL import Image
        with Image.open(image) as reference:
            px_per_dp = reference.width / (1080 / 2.625)
        command = [sys.executable, "-m", "baselines.dcgen_adapter.online_cli",
            "--pipeline", "paper", "--screen-id", image.stem,
            "--input", str(image), "--output-dir", str(output),
            "--model-snapshot", args.model, "--px-per-dp", f"{px_per_dp:.4f}",
            "--max-output-tokens-call", str(args.max_tokens),
            "--max-input-tokens-call", "128000", "--retries", "8",
            "--input-price", str(args.input_price), "--output-price", str(args.output_price),
            "--max-cost-global", str(args.max_total_cost_usd),
            "--max-cost-screen", str(args.max_total_cost_usd),
            "--max-cost-call", str(min(15, args.max_total_cost_usd))]
    else:
        command = [sys.executable, "-m", "baselines.layoutcoder_adapter.cli",
            "generate" if args.preprocessed else "all", "--image", str(image),
            "--work-dir", str(output / "work"), "--run-dir", str(output / "run"),
            "--output", str(output / "final.xml"), "--screen-id", image.stem,
            "--model", args.model, "--max-tokens", str(args.max_tokens),
            "--input-cost-per-million", str(args.input_price),
            "--output-cost-per-million", str(args.output_price),
            "--max-cost-usd", str(args.max_total_cost_usd)]
        if args.preprocessed:
            command.extend(["--preprocessed", str(args.preprocessed.resolve())])
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.dry_run:
        import shlex
        print(shlex.join(command))
        return 0
    if not os.environ.get("OPENAI_API_KEY"):
        parser.error("Set OPENAI_API_KEY before running a paid request")
    env = dict(os.environ, LAYOUTCODER_ANDROID_PROMPT="v2",
               ANDROID_EXTRACTION_CONTRACT="charitable")
    result = subprocess.run(command, cwd=release_paths.RELEASE, env=env)
    if result.returncode == 0:
        if args.arm == "dcgen":
            materialize_dcgen_output(output, image.stem)
        if not (output / "final.xml").is_file():
            raise FileNotFoundError(f"Baseline returned without its final XML: {output}")
    return result.returncode

if __name__ == "__main__":
    raise SystemExit(main())
