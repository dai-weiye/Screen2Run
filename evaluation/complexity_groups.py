#!/usr/bin/env python3
"""Reference-only complexity strata and fixed-cohort robustness statistics.

Properties: UIED components plus OCR lines; OCR whitespace tokens; elements per
100,000 pixels on the 1080-wide reference-aspect canvas; and height/width.
Quantiles are 1/3 and 2/3; boundary ties remain in the lower stratum.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths

PROPERTIES = ("elements", "text_tokens", "density", "aspect")
LEVELS = ("low", "mid", "high")


def features(sid, reference):
    from PIL import Image
    from image_filling import load_components, load_ocr
    with Image.open(reference) as image:
        width, height = image.size
    components, text = load_components(sid, width, height), load_ocr(reference, width, height)
    count = len(components) + len(text)
    area = 1080.0 * (1080.0 * height / width)
    return {"elements": count, "text_tokens": sum(len(line["text"].split()) for line in text),
            "density": count / (area / 1e5), "aspect": height / width}


def assign_bins(features_by_screen):
    cuts, bins = {}, {sid: {} for sid in features_by_screen}
    if not bins:
        raise ValueError("Cannot stratify an empty cohort")
    for prop in PROPERTIES:
        values = np.array([features_by_screen[sid][prop] for sid in bins], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite complexity property")
        q1, q2 = (float(v) for v in np.quantile(values, [1 / 3, 2 / 3]))
        cuts[prop] = [q1, q2]
        for sid in bins:
            value = features_by_screen[sid][prop]
            bins[sid][prop] = "low" if value <= q1 else "mid" if value <= q2 else "high"
    return {"source": "reference screenshots only (UIED + OCR)", "properties": list(PROPERTIES),
            "cuts_tertile": cuts, "features": features_by_screen, "bins": bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("bins")
    p.add_argument("--sids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("eval")
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--bins", type=Path, required=True)
    p.add_argument("--metric-set", choices=("primary", "seven"), default="primary")
    p.add_argument("--expected-count", type=int, default=600)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("Output already exists; preserve frozen strata and earlier results")
    if args.command == "bins":
        from run_image_filling import read_sids, shot_map
        refs, ids = shot_map(), read_sids(args.sids)
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate cohort IDs")
        result = assign_bins({sid: features(sid, refs[sid]) for sid in ids})
    else:
        from frozen_statistics import PRIMARY, METRICS, rq3
        rows = json.loads(args.rows.read_text(encoding="utf-8"))
        bins = json.loads(args.bins.read_text(encoding="utf-8"))
        result = rq3(rows, bins, metrics=PRIMARY if args.metric_set == "primary" else METRICS, expected_count=args.expected_count)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
