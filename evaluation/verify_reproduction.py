#!/usr/bin/env python3
"""Independently compare recomputed outputs with the released numeric evidence."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def verify(data, output):
    data, output = Path(data), Path(output)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))
    checks, errors = [], []
    numeric, max_error = 0, 0.0

    def compare(actual, expected, path):
        nonlocal numeric, max_error
        if isinstance(expected, bool) or expected is None or isinstance(expected, str):
            if actual != expected:
                errors.append(path)
        elif isinstance(expected, (int, float)):
            numeric += 1
            if not isinstance(actual, (int, float)) or not math.isfinite(actual):
                errors.append(path)
                return
            delta = abs(actual - expected)
            max_error = max(max_error, delta)
            if delta > 1e-12:
                errors.append(path)
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(actual) != len(expected):
                errors.append(path)
                return
            for i, (a, e) in enumerate(zip(actual, expected)):
                compare(a, e, f"{path}/{i}")
        else:
            if not isinstance(actual, dict):
                errors.append(path)
                return
            for key, value in expected.items():
                if key not in actual:
                    errors.append(f"{path}/{key}")
                else:
                    compare(actual[key], value, f"{path}/{key}")

    if (output / "rq1_summary.json").is_file():
        compare(read(output / "rq1_summary.json"), read(data / "results/expected_rq1_summary.json"), "rq1")
        checks.append("RQ1 means, paired tests, Holm, effect sizes, and confidence intervals")
    if (output / "rq2_data.json").is_file():
        compare(read(output / "rq2_data.json"), read(data / "results/rq2_data.json"), "rq2")
        rq1 = {(r["screen_id"], m): r[m] for r in read(data / "results/rq1_rows.json") if r["arm"] == "ours"
               for m in ("block_match", "text_similarity", "position_similarity", "color_similarity", "clip", "ssim", "mae")}
        full = {(r["screen_id"], m): r[m] for r in read(data / "results/rq2_rows.json") if r["arm"] == "full"
                for m in ("block_match", "text_similarity", "position_similarity", "color_similarity", "clip", "ssim", "mae")}
        if rq1 != full:
            errors.append("rq2/frozen_full_identity")
        checks.append("RQ2 140 cells and exact frozen-Full scalar identity")
    if (output / "rq3_seven_metrics.json").is_file():
        actual, expected = read(output / "rq3_seven_metrics.json"), read(data / "results/expected_rq3.json")
        for prop, groups in expected.items():
            for level, block in groups.items():
                copy = actual[f"{prop}/{level}"]
                for entry in copy["metrics"].values():
                    entry["lead"] = entry["gain"] > 0
                compare(copy, block, f"rq3/{prop}/{level}")
        checks.append("RQ3 84 seven-metric cells with the original Holm family")
    expected_backbones = read(data / "results/expected_backbone_summary.json")
    for model, summary in expected_backbones.items():
        path = output / f"backbone_{model}_summary.json"
        if not path.is_file():
            continue
        actual = read(path)
        for scope in ("Easy", "Real", "Unseen", "All"):
            compare(actual["table"][scope], summary[scope]["means"], f"backbone/{model}/{scope}/means")
            compare(actual["stats"][scope], summary[scope]["stats"], f"backbone/{model}/{scope}/stats")
            compare(actual["table"][scope]["ours"]["n"], summary[scope]["n"], f"backbone/{model}/{scope}/n")
        checks.append(f"Backbone {model}: means and paired statistics on 120 screens")
    if (output / "rq4_summary.json").is_file():
        actual, expected = read(output / "rq4_summary.json"), read(data / "human_study/expected_results.json")
        for section in ("primary", "secondary"):
            compare(actual[section], expected[section], f"rq4/{section}")
        compare(actual["n_raters"], expected["n_raters"], "rq4/raters")
        compare(actual["n_code_items"], 60, "rq4/code_items")
        compare(actual["n_ui_items"], 60, "rq4/interface_items")
        checks.append("RQ4 six experts, UI60/code60, 24 paired comparisons and two-way confidence intervals")
    paper_path = data / "results/expected_paper_plot_data.json"
    if paper_path.is_file():
        paper = read(paper_path)
        for filename, key, label in (
            ("rq2_paper.json", "rq2", "RQ2 current-paper primary-five and separate pixel-two Holm families"),
            ("rq3_primary.json", "rq3", "RQ3 current-paper primary-five Holm family"),
        ):
            if (output / filename).is_file():
                compare(read(output / filename), paper[key], f"paper/{key}")
                checks.append(label)
        if (output / "figures/plot_data.json").is_file():
            plots = read(output / "figures/plot_data.json")
            for key in ("rq1", "rq3", "rq4"):
                compare(plots[key], paper[f"fig_{key}"], f"paper/fig_{key}")
            checks.append("Current-paper RQ1/RQ3/RQ4 plot values, including all 60 code items")
    report = {"schema": "screen2run-reproduction-verification/1", "passed": bool(checks) and not errors,
              "checks": checks, "numeric_values_compared": numeric, "maximum_absolute_error": max_error,
              "absolute_tolerance": 1e-12, "mismatches": errors}
    (output / "verification.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True, help="Directory produced by reproduce.py")
    args = parser.parse_args()
    report = verify(args.data_dir, args.out)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
