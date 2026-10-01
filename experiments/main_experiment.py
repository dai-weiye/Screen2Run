#!/usr/bin/env python3
"""Plan or run one local Screen2Run realization/RQ2 pipeline.

This driver never calls a model or starts an emulator. It prints the complete
plan by default. --execute runs local preparation, rendering, grounding, and
the same content guard in a fresh output directory on already-running devices.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths
from ablation_variants import ARMS


def endpoints(out):
    prefix = out.name + "_" + hashlib.sha256(str(out.resolve()).encode()).hexdigest()[:8]
    return {name: out / f"{prefix}_{name}" for name in ("pre", "grounded", "fallback", "final")}


def plan(variant, source, cohort, out, devices):
    python = sys.executable
    script = lambda name: str(release_paths.module_file(name))
    def render(candidate):
        return [python, script("batch_render.py"), "--cand", str(candidate), "--hierarchy", "--devices", str(devices)]
    paths = endpoints(out)
    if variant in ("no_realization", "no_raster_output"):
        candidate = paths["final"]
        return [("terminal_intervention", [python, script("ablation_variants.py"), "terminal", "--arm", variant,
            "--source", str(source), "--sids", str(cohort), "--out", str(candidate)]), ("final_capture", render(candidate))]
    pre, grounded, fallback = (paths[name] for name in ("pre", "grounded", "fallback"))
    if variant == "ours":
        prepare = [python, script("run_image_filling.py"), "prepare", "--sids", str(cohort), "--src", str(source), "--out", str(pre)]
    else:
        prepare = [python, script("ablation_variants.py"), "prepare", "--arm", variant, "--gen", str(source),
                   "--sids", str(cohort), "--out", str(pre)]
    return [("prepare", prepare), ("pre_capture", render(pre)),
        ("ground", [python, script("run_image_filling.py"), "ground", "--pre", str(pre), "--out", str(grounded)]),
        ("grounded_capture", render(grounded)),
        ("fallback", [python, script("ablation_variants.py"), "bindonly", "--pre", str(pre), "--out", str(fallback)]),
        ("content_guard", [python, script("content_guard.py"), "--pre", str(pre), "--grounded", str(grounded),
            "--fallback", str(fallback), "--cohort", str(cohort), "--apply"]),
        ("final_capture", render(grounded))]


def audit_step(name, variant, cohort, out):
    """A zero process return code alone is not evidence of complete captures."""
    from run_image_filling import read_sids, shot_map
    from collect_renders import collect, render_dir, receipt_rows
    from screenshot_metrics import split_of
    ids, refs, paths = read_sids(cohort), shot_map(), endpoints(out)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Invalid fixed cohort")
    target = paths["final"] if variant in ("no_realization", "no_raster_output") else (
        paths["pre"] if name in ("prepare", "pre_capture") else paths["fallback"] if name == "fallback" else paths["grounded"])
    for sid in ids:
        screen = target / "full" / sid
        if not (screen / "final.xml").is_file() or (screen / "FAILED.json").exists():
            raise ValueError(f"Incomplete candidate after {name}: {sid}")
    if name.endswith("capture"):
        manifest = {"schema": "screen2run-experiment/1",
            "cohort": [{"screen_id": sid, "split": split_of(sid), "reference": str(refs[sid])} for sid in ids],
            "methods": {variant: {"candidate": str(target), "render_dir": str(render_dir(target))}}}
        pairs, missing = collect(manifest, out)
        if missing or len(pairs) != len(ids):
            raise ValueError(f"Incomplete capture after {name}: {len(pairs)}/{len(ids)}")
        if name == "pre_capture":
            receipts = list(receipt_rows(render_dir(target)))
            for pair in pairs:
                sid = pair["screen_id"]
                hierarchy = render_dir(target) / "raw/hierarchy" / f"{sid}.xml"
                if not hierarchy.is_file() or not any(r.get("screen_id") == sid and r.get("status") == "render_success"
                        and r.get("xml_sha256") == pair["xml_sha256"] and r.get("png_sha256") == pair["render_path_sha256"]
                        and r.get("hierarchy") is True for r in receipts):
                    raise ValueError(f"Current runtime hierarchy is missing: {sid}")
    return {"candidate": str(target), "screens": len(ids), "verified": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("ours", *ARMS), required=True)
    parser.add_argument("--source", type=Path, required=True,
                        help="Generation root for ours/planning/visual/review; original pre for no_realization; Full for no_raster_output")
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--devices", type=int, choices=(1, 2, 3), default=1)
    parser.add_argument("--execute", action="store_true", help="Run local commands; default only prints the plan")
    args = parser.parse_args()
    source, cohort, out = args.source.resolve(), args.cohort.resolve(), args.out.resolve()
    if out == source or out.is_relative_to(source) or source.is_relative_to(out):
        parser.error("Source and output must be disjoint")
    steps = plan(args.variant, source, cohort, out, args.devices)
    for index, (name, command) in enumerate(steps, 1):
        print(f"{index}. {name}: {shlex.join(command)}")
    if not args.execute:
        print("Plan only: no model, renderer, or device was invoked.")
        return 0
    if not source.is_dir() or not cohort.is_file():
        parser.error("Source directory and cohort file must exist")
    if out.exists():
        parser.error("Use a fresh output directory; existing experiment outputs are never overwritten")
    os.environ["SCREEN2RUN_SCREENSHOT_LIST"] = str(cohort)
    out.mkdir(parents=True)
    state = {"schema": "screen2run-local-pipeline/1", "variant": args.variant, "model_calls": 0,
             "cohort_sha256": hashlib.sha256(cohort.read_bytes()).hexdigest(), "steps": []}
    for name, command in steps:
        state["active_step"] = name
        (out / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
        with (out / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=release_paths.RELEASE)
        state["steps"].append({"name": name, "returncode": result.returncode})
        if result.returncode:
            state["status"] = "failed"
            (out / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
            print(f"Stopped at {name}; inspect its log. Completed stages are retained.", file=sys.stderr)
            return result.returncode
        try:
            state["steps"][-1]["audit"] = audit_step(name, args.variant, cohort, out)
        except (ValueError, OSError, KeyError) as exc:
            state.update(status="failed", failure={"step": name, "type": type(exc).__name__, "message": str(exc)})
            (out / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
            print(f"Stopped after {name}: incomplete artifact evidence. See progress.json.", file=sys.stderr)
            return 2
    state.update(status="complete", active_step=None,
                 final_candidate=str(endpoints(out)["final" if args.variant in ("no_realization", "no_raster_output") else "grounded"]))
    (out / "progress.json").write_text(json.dumps(state, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
