#!/usr/bin/env python3
"""Recompute paper statistics from frozen numeric data, without model calls."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import frozen_statistics as statistics

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def markdown_table(summary):
    lines = ["# Fixed-cohort comparison", ""]
    for scope, table in summary["table"].items():
        lines += [f"## {scope}", "", "| Method | n | " + " | ".join(statistics.METRICS) + " |",
                  "|---|---:|" + "---:|" * len(statistics.METRICS)]
        for arm, row in table.items():
            lines.append(f"| {arm} | {row['n']} | " + " | ".join(f"{row[m]:.6f}" for m in statistics.METRICS) + " |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=("all", "rq1", "rq2", "rq3", "rq4", "backbones"), default="all")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--plots", action="store_true", help="Generate the released paper-style figures after statistics")
    parser.add_argument("--expected-count", type=int, default=600)
    args = parser.parse_args(argv)
    data, out = args.data_dir.resolve(), args.out.resolve()
    if out == data or out.is_relative_to(data) or data.is_relative_to(out):
        parser.error("Output and released source data must be disjoint")
    out.mkdir(parents=True, exist_ok=True)
    inputs, completed = {}, []

    def read(relative):
        path = data / relative
        inputs[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        return load(path)

    def wanted(name):
        return args.experiment in ("all", name)

    rows = None
    if wanted("rq1") or wanted("rq3") or args.plots:
        rows = read("results/rq1_rows.json")
    if wanted("rq1"):
        result = statistics.rq1(rows, expected_count=args.expected_count)
        save(out / "rq1_summary.json", result)
        (out / "rq1.md").write_text(markdown_table(result), encoding="utf-8")
        completed.append("rq1")
    if wanted("backbones"):
        files = sorted((data / "results").glob("backbone_*_rows.json"))
        if not files:
            raise ValueError("No released backbone score rows")
        for path in files:
            records = read(path.relative_to(data).as_posix())
            focal = "ours" if any(r["arm"] == "ours" for r in records) else path.name.removeprefix("backbone_").removesuffix("_rows.json")
            result = statistics.rq1(records, focal=focal, expected_count=120)
            save(out / path.name.replace("_rows", "_summary"), result)
        completed.append("backbones")
    if wanted("rq2"):
        records = read("results/rq2_rows.json")
        result = statistics.rq2(records, expected_count=args.expected_count)
        save(out / "rq2_data.json", result)
        save(out / "rq2_paper.json", statistics.rq2_paper(result))
        save(out / "rq2_acceptance.json", statistics.rq2_acceptance(result))
        if (data / "results/rq2_data.json").is_file():
            expected = read("results/rq2_data.json")
            compare = {(r["scope"], r["arm"], r["metric"]): r for r in expected}
            for cell in result:
                key = cell["scope"], cell["arm"], cell["metric"]
                if key not in compare:
                    raise ValueError(f"Missing frozen RQ2 comparison: {key}")
                for name, value in cell.items():
                    reference = compare[key][name]
                    if isinstance(value, float):
                        if abs(value - reference) > 1e-12:
                            raise ValueError(f"RQ2 reproduction differs: {key}/{name}")
                    elif value != reference:
                        raise ValueError(f"RQ2 reproduction differs: {key}/{name}")
            if len(compare) != len(result):
                raise ValueError("Frozen RQ2 cell count differs")
            save(out / "rq2_verification.json", {"cells": len(result), "matches_frozen_results": True})
        completed.append("rq2")
    if wanted("rq3"):
        bins = read("results/rq3_bins.json")
        save(out / "rq3_primary.json", statistics.rq3(rows, bins, expected_count=args.expected_count))
        save(out / "rq3_seven_metrics.json", statistics.rq3(rows, bins, expected_count=args.expected_count, metrics=statistics.METRICS))
        completed.append("rq3")
    if wanted("rq4"):
        import human_statistics
        ratings = read("human_study/ratings.json")
        save(out / "rq4_summary.json", human_statistics.analyze(ratings))
        completed.append("rq4")
    if args.plots:
        import make_plots
        make_plots.generate(data, out / "figures")
    for relative, expected in inputs.items():
        if hashlib.sha256((data / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Source changed while analyzing: {relative}")
    import numpy, scipy
    save(out / "reproduction_manifest.json", {"schema": "screen2run-reproduction/1", "completed": completed,
         "sources_sha256": inputs, "source_files_unchanged": True,
         "numpy_version": numpy.__version__, "scipy_version": scipy.__version__, "model_calls": 0})
    print(f"Completed {', '.join(completed)}; results: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
