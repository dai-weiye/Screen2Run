#!/usr/bin/env python3
"""Generate the paper's RQ1, RQ3, and UI60/code60 RQ4 figures from released data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import plot_style as style
import plot_results as figures
from human_statistics import validate


def generate(data, out):
    data, out = Path(data), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    style.apply()
    read = lambda relative: json.loads((data / relative).read_text(encoding="utf-8"))
    rows, bins = read("results/rq1_rows.json"), read("results/rq3_bins.json")
    bins = bins.get("bins", bins)
    human = read("human_study/ratings.json")
    val = validate(human)
    visual = [("overall", "Overall"), ("layout", "Layout"), ("text", "Text"), ("image", "Images"), ("style", "Style")]
    code = [("readability", "Readability"), ("maintainability", "Maintainability"),
            ("practice", "Android practice"), ("usability", "Usability")]
    raw = {"a": {}, "b": {}}
    for part, tag, dimensions in (("v", "a", visual), ("c", "b", code)):
        for dim, _ in dimensions:
            raw[tag][dim] = {arm: [val[part, rater, item, arm, dim] for rater in human["raters"]
                                  for item in human["items"][part]] for arm in human["methods"]}
    summary = {"rq1": figures.rq1_box(rows, out / "rq1_boxplots"),
               "rq3": figures.rq3_lines(rows, bins, out / "rq3_robustness"),
               "rq4": figures.rq4_likert(raw, [("a", "Interface quality (60 screens)", visual),
                                               ("b", "Code quality (60 screens)", code)], out / "rq4_expert_ratings")}
    (out / "plot_data.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    generate(args.data_dir, args.out)
    print(f"Figures and plot data: {args.out}")


if __name__ == "__main__":
    main()
