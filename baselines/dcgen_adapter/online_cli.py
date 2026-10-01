"""CLI for dry-run planning and online Android-adapted DCGen generation."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from .openai_compatible import (
    Budget,
    HardCaps,
    OpenAICompatibleClient,
    Price,
    caps_dict,
)
from .paper_agent import PaperDCGenAgent, plan_dry_run, tree_from_bbox_tree
from .pipeline import PIPELINE_PHASES, AndroidDCGenPipeline, manifest_for_run
from .renderer import LEGACY_UINT8, METRIC_MODES, FrozenHarnessRenderer
from .segmentation import segment_screenshot
from .xml_assembler import count_leaves, expected_call_count

import release_paths
REPO_ROOT = release_paths.RELEASE


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screen-id", required=True)
    parser.add_argument(
        "--pipeline",
        choices=("grid", "paper"),
        default="grid",
        help=(
            "grid: upstream DCGenGrid, 2 candidates per leaf + min-MAE selection "
            "(2L+1 calls). paper: published Algorithm 2 (DCGen-Agent), one call "
            "per segmentation node with no candidates or MAE. --phase and "
            "--avd-snapshot are grid-only."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Screenshot; defaults to datasets/Reference/Reference/<screen-id>.png",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--metric-mode", choices=METRIC_MODES, default=LEGACY_UINT8)
    parser.add_argument("--input-price", type=float, required=True)
    parser.add_argument("--output-price", type=float, required=True)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--var-thresh", type=float, default=50)
    parser.add_argument("--diff-thresh", type=float, default=45)
    parser.add_argument("--diff-portion", type=float, default=0.9)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--px-per-dp", type=float, default=1.0)
    parser.add_argument("--max-input-tokens-call", type=int, default=16000)
    parser.add_argument("--max-output-tokens-call", type=int, default=4096)
    parser.add_argument("--max-calls-global", type=int, default=1000)
    parser.add_argument("--max-calls-screen", type=int, default=1000)
    parser.add_argument("--max-cost-global", type=float, default=100.0)
    parser.add_argument("--max-cost-screen", type=float, default=100.0)
    parser.add_argument("--max-cost-call", type=float, default=10.0)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--harness-script", type=Path)
    parser.add_argument("--android-project", type=Path)
    parser.add_argument("--avd-snapshot")
    parser.add_argument("--device-serial", default="")
    parser.add_argument("--dataset-label", default="Redraw_D")
    parser.add_argument(
        "--phase",
        choices=PIPELINE_PHASES,
        default="full",
        help=(
            "full: leaf LLM then emulator select then root. "
            "llm_leaves: all 2L leaf calls, no emulator. "
            "select_and_finish: reuse persisted leaves, then select + root."
        ),
    )
    return parser


def _write_new(path: Path, content: str, *, resume: bool = False) -> None:
    if path.exists():
        if resume and path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_path = (
        args.input
        or REPO_ROOT / "datasets" / "Reference" / "Reference" / f"{args.screen_id}.png"
    ).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"input screenshot does not exist: {image_path}")
    run_dir = args.output_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise SystemExit(f"output directory is non-empty; pass --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.resume and (run_dir / "manifest.json").exists() and (run_dir / "items.jsonl").exists():
        print(json.dumps({"status": "already_complete", "manifest": str(run_dir / "manifest.json")}))
        return 0
    tree_path = run_dir / args.screen_id / "segmentation.json"
    if args.resume and tree_path.exists():
        tree = json.loads(tree_path.read_text(encoding="utf-8"))
    else:
        tree = segment_screenshot(
            image_path,
            max_depth=args.max_depth,
            var_thresh=args.var_thresh,
            diff_thresh=args.diff_thresh,
            diff_portion=args.diff_portion,
            window_size=args.window_size,
        )
        _write_new(
            tree_path,
            json.dumps(tree, indent=2, ensure_ascii=False) + "\n",
            resume=args.resume,
        )
    leaves = count_leaves(tree)
    calls = expected_call_count(leaves)
    call_formula = "2L+1"
    # Algorithm 2 pays one call per segmentation node (leaf or assembly), which
    # is strictly cheaper than 2L+1; the persisted bbox_tree is the same
    # init_tree subdivision, so it is consumed directly without re-segmenting.
    paper_plan: dict | None = None
    paper_tree = None
    if args.pipeline == "paper":
        paper_tree = tree_from_bbox_tree(tree)
        paper_plan = plan_dry_run(paper_tree)
        leaves = paper_plan["leaf_count"]
        calls = paper_plan["expected_calls"]
        call_formula = paper_plan["call_formula"]
    price = Price(args.input_price, args.output_price)
    caps = HardCaps(
        max_calls_global=args.max_calls_global,
        max_calls_screen=args.max_calls_screen,
        max_cost_global_usd=args.max_cost_global,
        max_cost_screen_usd=args.max_cost_screen,
        max_cost_call_usd=args.max_cost_call,
        max_input_tokens_call=args.max_input_tokens_call,
        max_output_tokens_call=args.max_output_tokens_call,
    )
    budget = Budget(caps, price)
    upper_cost = calls * budget.worst_case_call_cost()
    plan = {
        "screen_id": args.screen_id,
        "input": str(image_path),
        "model_snapshot": args.model_snapshot,
        "dry_run": args.dry_run,
        "leaf_count": leaves,
        "expected_calls": calls,
        "call_formula": call_formula,
        "estimated_cost_upper_usd": upper_cost,
        "price": {
            "input_usd_per_million": args.input_price,
            "output_usd_per_million": args.output_price,
        },
        "hard_caps": caps_dict(caps),
        "segmentation": {
            "max_depth": args.max_depth,
            "var_thresh": args.var_thresh,
            "diff_thresh": args.diff_thresh,
            "diff_portion": args.diff_portion,
            "window_size": args.window_size,
        },
    }
    if paper_plan is not None:
        # Paper-only diagnostics; grid plan output is left byte-identical.
        plan["pipeline"] = "paper"
        plan["node_count"] = paper_plan["node_count"]
        plan["depth_levels"] = paper_plan["depth_levels"]
        plan["nodes_per_depth"] = paper_plan["nodes_per_depth"]
        plan["leaves_per_depth"] = paper_plan["leaves_per_depth"]
    plan_path = run_dir / args.screen_id / "dry_run_plan.json"
    if not plan_path.exists():
        plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if calls > min(caps.max_calls_global, caps.max_calls_screen):
        raise SystemExit(f"expected {call_formula} calls exceed configured call cap")
    # Do not abort on calls * worst-case tokens. That bound is logged in
    # dry_run_plan.json; actual spend is a few cents and is enforced by
    # Budget.reserve/commit on each call. A $100 screen cap previously
    # dropped 40 FSE screens before any request.
    if budget.worst_case_call_cost() > caps.max_cost_call_usd:
        raise SystemExit("worst-case call cost exceeds per-call cap")
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    renderer = None
    # Algorithm 2 has no candidate render or MAE, so the paper pipeline needs no
    # emulator at all; the harness stays a grid-only dependency.
    if args.pipeline == "grid" and args.phase != "llm_leaves":
        if not args.avd_snapshot:
            raise SystemExit("--avd-snapshot is required for online candidate rendering")
        harness = (
            args.harness_script
            or release_paths.module_file("device_session.py")
        )
        renderer = FrozenHarnessRenderer(
            repo_root=REPO_ROOT,
            harness_script=harness,
            avd_snapshot=args.avd_snapshot,
            evidence_root=run_dir / "candidate_evidence",
            input_png=image_path,
            model={
                "identifier": args.model_snapshot,
                "provider": "openai_compatible",
                "revision": args.model_snapshot,
            },
            project=args.android_project,
            device_serial=args.device_serial,
            dataset=args.dataset_label,
        )
    trace_path = run_dir / args.screen_id / "trace.json"
    if args.pipeline == "grid" and args.resume and not trace_path.exists():
        deviation_path = run_dir / "deviations.json"
        if not deviation_path.exists():
            _atomic_json(
                deviation_path,
                [
                    {
                        "code": "legacy_unpersisted_candidate_responses",
                        "recorded_at": datetime.now(timezone.utc).astimezone().isoformat(),
                        "screen_id": args.screen_id,
                        "known_lost_call_count": 2,
                        "cost_usd": None,
                        "usage_recoverable": False,
                        "detail": (
                            "A pre-fix pilot failed during first-leaf selection before "
                            "trace/ledger/raw responses were persisted. The two paid "
                            "responses and exact usage cannot be recovered; resume must "
                            "regenerate only those unavailable responses."
                        ),
                    }
                ],
            )
    resume_trace = (
        json.loads(trace_path.read_text(encoding="utf-8"))
        if args.resume and trace_path.exists()
        else None
    )
    if resume_trace:
        budget.restore(args.screen_id, resume_trace.get("usage", []))
    client = OpenAICompatibleClient(
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        model_snapshot=args.model_snapshot,
        price=price,
        budget=budget,
        retries=args.retries,
        temperature=args.temperature,
        seed=args.seed,
    )
    if args.pipeline == "paper":
        agent = PaperDCGenAgent(
            client=client,
            output_dir=run_dir,
            px_per_dp=args.px_per_dp,
        )
        result = agent.run_screen(
            screen_id=args.screen_id,
            image_path=image_path,
            tree=paper_tree,
            resume_trace=resume_trace,
        )
    else:
        pipeline = AndroidDCGenPipeline(
            client=client,
            renderer=renderer,
            output_dir=run_dir,
            metric_mode=args.metric_mode,
            px_per_dp=args.px_per_dp,
        )
        result = pipeline.run_screen(
            screen_id=args.screen_id,
            image_path=image_path,
            tree=tree,
            resume_trace=resume_trace,
            phase=args.phase,
        )
    if args.pipeline == "grid" and args.phase == "llm_leaves":
        print(
            json.dumps(
                {
                    "status": "llm_leaves_complete",
                    "screen_id": args.screen_id,
                    "trace": str(trace_path),
                    **plan,
                },
                indent=2,
            )
        )
        return 0
    if result is None:
        raise SystemExit("pipeline returned no screen result")
    manifest = manifest_for_run(
        run_id=run_dir.name,
        model_snapshot=args.model_snapshot,
        result=result,
        price=plan["price"],
        caps=plan["hard_caps"],
        dataset=args.dataset_label,
    )
    if args.pipeline == "paper":
        # manifest_for_run's default label describes the grid variant; the
        # paper run must be distinguishable in the frozen records.
        manifest["method"] = "android_adapted_dcgen_paper_algorithm2"
    _write_new(
        run_dir / "manifest.json",
        json.dumps(manifest, indent=2) + "\n",
        resume=args.resume,
    )
    item = {"run_id": run_dir.name, **result.__dict__}
    _write_new(
        run_dir / "items.jsonl",
        json.dumps(item, sort_keys=True) + "\n",
        resume=args.resume,
    )
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        events_path.write_text("", encoding="utf-8")
    print(json.dumps({"manifest": str(run_dir / "manifest.json"), **plan}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
