"""Published UI2Code block metrics, separated from Android block extraction.

Reuse the inspected DCGen/Design2Code matching functions verbatim from the
pinned local source, without importing its HTML mutation, browser or CLIP code.
Inputs are ordered text blocks with top-left normalized xywh boxes and optional
foreground RGB colors. No XML/VH, screen IDs, arm names or generation tools enter
this module. Android OCR is an explicit input adaptation, not OCR-free replay.

The published names are intentionally distinct from legacy component_f1_geo,
word-level text_f1, position_iou, and tree-edit metrics. Missing colors remain
pending instead of silently measuring the background or substituting a score.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from copy import deepcopy
from difflib import SequenceMatcher
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO = release_paths.RELEASE
UPSTREAM = release_paths.DCGEN / "scripts/metric/Design2Code/metrics/visual_score.py"
UPSTREAM_SHA256 = "b2ca62e4bf6c657bd2c2b194a6f223369d6c62cee49528c16f00f454eb30aaeb"
MATCH_FUNCTIONS = frozenset({
    "calculate_similarity", "adjust_cost_for_context", "create_cost_matrix",
    "calculate_current_cost", "merge_blocks_wo_check", "find_maximum_matching",
    "remove_indices", "merge_blocks_by_list", "print_matching",
    "difference_of_means", "find_possible_merge", "merge_blocks_by_bbox",
})
SCORES = ("block_match", "text_similarity", "position_similarity", "color_similarity")


class MetricInputError(ValueError):
    pass


def _number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_blocks(blocks: Any) -> list[dict]:
    if not isinstance(blocks, list):
        raise MetricInputError("Blocks must be an ordered list, not a missing extraction")
    result = []
    for block in blocks:
        if not isinstance(block, dict) or not isinstance(block.get("text"), str):
            raise MetricInputError("Every block requires literal text")
        box = block.get("bbox")
        if (not isinstance(box, (list, tuple)) or len(box) != 4 or
                not all(_number(v) for v in box)):
            raise MetricInputError("Box must contain four finite xywh numbers")
        x, y, w, h = box
        if x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > 1.000000001 or y + h > 1.000000001:
            raise MetricInputError("Box must lie inside the normalized image with positive area")
        color = block.get("color")
        if color is not None and (not isinstance(color, (list, tuple)) or len(color) != 3 or
                                  not all(_number(c) and 0 <= c <= 255 for c in color)):
            raise MetricInputError("Foreground color must be RGB in 0..255 or null")
        result.append({"text": block["text"], "bbox": list(box),
                       "color": list(color) if color is not None else None})
    return result


@lru_cache(maxsize=1)
def _matching() -> dict:
    raw = UPSTREAM.read_bytes()
    if hashlib.sha256(raw).hexdigest() != UPSTREAM_SHA256:
        raise MetricInputError("Published matching source changed; explicit protocol review required")
    tree = ast.parse(raw, filename=str(UPSTREAM))
    definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in MATCH_FUNCTIONS]
    if {node.name for node in definitions} != MATCH_FUNCTIONS:
        raise MetricInputError("Published matching functions are incomplete")
    # Only top-level definitions are executed; imports/global initialization,
    # HTML rewriting, browser invocation and network/model loading are excluded.
    namespace = {"np": np, "Counter": Counter, "deepcopy": deepcopy,
                 "SequenceMatcher": SequenceMatcher, "linear_sum_assignment": linear_sum_assignment}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(UPSTREAM), "exec"), namespace)
    return namespace


def color_similarity(rgb_a: list[float], rgb_b: list[float]) -> float:
    """Published max(0, 1-DeltaE00/100), using standard sRGB/D65/2° Lab."""
    from skimage.color import deltaE_ciede2000, rgb2lab
    rgb = np.asarray([rgb_a, rgb_b], dtype=np.float64) / 255.0
    lab = rgb2lab(rgb, illuminant="D65", observer="2")
    return max(0.0, 1.0 - float(deltaE_ciede2000(lab[0], lab[1])) / 100.0)


def score_blocks(reference: list[dict], generated: list[dict] | None, *,
                 deployed: bool = True) -> dict:
    ref = validate_blocks(reference)
    if type(deployed) is not bool:
        raise MetricInputError("Deployment must be an explicit boolean")
    if deployed and generated is None:
        raise MetricInputError("Successful render with missing extraction is pending, not an empty screen")
    pred = validate_blocks(generated) if generated is not None else []
    colors_available = all(b["color"] is not None for b in ref + pred)
    report = {
        "protocol": "dcgen_design2code_blocks_android_input_adapter_v1",
        "matching_source_sha256": UPSTREAM_SHA256,
        "matching_parameters": {"consecutive_bonus": 0.1, "window_size": 1, "text_threshold": 0.5},
        "color_backend": "skimage sRGB/D65/2deg CIEDE2000",
        "reference_blocks": len(ref), "generated_blocks": len(pred),
        "color_evaluable": colors_available, "matches": [],
        "scores": {key: 0.0 for key in SCORES},
    }
    if not deployed:
        report["status"] = "deployment_failure"
        return report
    if not colors_available:
        report["scores"]["color_similarity"] = None
    if not ref or not pred:
        report["status"] = "empty_reference_blocks" if not ref else "empty_generated_blocks"
        return report
    # The upstream merger carries color alongside text; matching does not read
    # color. Placeholders are never scored when any real color is unavailable.
    for block in ref + pred:
        if block["color"] is None:
            block["color"] = [0, 0, 0]
    functions = _matching()
    ref = functions["merge_blocks_by_bbox"](ref)
    pred = functions["merge_blocks_by_bbox"](pred)
    pred, ref, matches = functions["find_possible_merge"](pred, deepcopy(ref), 0.1, 1, debug=False)
    matched_area, text_scores, position_scores, color_scores = [], [], [], []
    for i, j in matches:
        p, r = pred[i], ref[j]
        text = SequenceMatcher(None, p["text"], r["text"]).ratio()
        if text < 0.5:
            continue
        px, py, pw, ph = p["bbox"]
        rx, ry, rw, rh = r["bbox"]
        position = 1.0 - max(abs(px + pw / 2 - rx - rw / 2), abs(py + ph / 2 - ry - rh / 2))
        matched_area.append(pw * ph + rw * rh)
        text_scores.append(text)
        position_scores.append(position)
        if colors_available:
            color_scores.append(color_similarity(p["color"], r["color"]))
        report["matches"].append({"generated_index": int(i), "reference_index": int(j),
                                  "text_similarity": text, "position_similarity": position})
    total_area = sum(b["bbox"][2] * b["bbox"][3] for b in pred + ref)
    report.update({"merged_reference_blocks": len(ref), "merged_generated_blocks": len(pred),
                   "status": "scored" if colors_available else "pending_foreground_color_extraction"})
    if matched_area:
        report["scores"].update({"block_match": float(sum(matched_area) / total_area),
                                "text_similarity": float(np.mean(text_scores)),
                                "position_similarity": float(np.mean(position_scores)),
                                "color_similarity": float(np.mean(color_scores)) if colors_available else None})
    return report


def dcgen_code_similarity(reference: str, generated: str, *,
                          reference_representation: str, generated_representation: str) -> float:
    """DCGen released implementation uses RapidFuzz ratio (normalized Indel).

    The caller must establish code/reference provenance; XML vs VH or a mapped
    component list is not a comparable code pair. This does not infer reference
    availability from a dataset name or synthesize unavailable author code.
    """
    from rapidfuzz.fuzz import ratio
    if (reference_representation != generated_representation or
            reference_representation not in {"android-layout-xml", "html-css", "pix2code-dsl"}):
        raise MetricInputError("Code similarity requires the same supported code representation")
    if not isinstance(reference, str) or not isinstance(generated, str):
        raise MetricInputError("Code inputs must be strings")
    return float(ratio(reference, generated)) / 100.0


def vision_lines_to_blocks(record: dict) -> list[dict]:
    """Convert raw OCR; lowercase as DCGen get_blocks_from_image_diff_pixels does.

    The raw OCR record stays untouched; no punctuation/whitespace stripping,
    tokenization, ground-truth text replacement or confidence filtering occurs.
    """
    if not isinstance(record, dict) or record.get("error") != "" or not isinstance(record.get("lines"), list):
        raise MetricInputError("OCR failure/incomplete record cannot be treated as empty text")
    blocks = []
    for line in record["lines"]:
        if not isinstance(line, dict) or not isinstance(line.get("text"), str):
            raise MetricInputError("Invalid OCR line")
        bounds = line.get("bbox")
        if not isinstance(bounds, list) or len(bounds) != 4 or not all(_number(v) for v in bounds):
            raise MetricInputError("Invalid Vision bottom-left xywh box")
        x, bottom, w, h = bounds
        if w <= 0 or h <= 0:
            raise MetricInputError("Vision box must have positive area")
        # Vision can return a box extending a fraction of a pixel beyond the
        # screenshot (including negative x). Measure only its visible area,
        # symmetrically on both sides; retain the original box in raw records.
        # An entirely off-canvas detection is an extraction error, not a text
        # segment to silently drop from the denominator.
        left, top = max(0.0, x), max(0.0, 1.0 - bottom - h)
        right, lower = min(1.0, x + w), min(1.0, 1.0 - bottom)
        if right <= left or lower <= top:
            raise MetricInputError("Vision box has no visible image intersection")
        blocks.append({"text": line["text"].lower(),
                       "bbox": [left, top, right - left, lower - top], "color": None})
    return validate_blocks(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-blocks", required=True, type=Path)
    parser.add_argument("--generated-blocks", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a metric report")
    inputs = [args.reference_blocks, args.generated_blocks]
    raw = [p.read_bytes() for p in inputs]
    report = score_blocks(*(json.loads(data) for data in raw))
    report["inputs"] = [{"path": str(p.resolve()), "sha256": hashlib.sha256(data).hexdigest()}
                        for p, data in zip(inputs, raw)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
