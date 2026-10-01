#!/usr/bin/env python3
"""Conservative S3 -> S5 structural / vertical-boundary regression audit.

Offline only: consumes generated XML and an explicitly supplied harness viewport,
not reference XML, view hierarchies, images, models or Android. This is deliberately
not a general Android layout engine. Unsupported measurement stays unknown. It
never rewrites XML or pairs anonymous icons by order, geometry or drawable name.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

ANDROID = "{http://schemas.android.com/apk/res/android}"
# Measurement equations audited against AOSP android15-release:
# https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/android15-release/core/java/android/widget/FrameLayout.java
# https://raw.githubusercontent.com/aosp-mirror/platform_frameworks_base/android15-release/core/java/android/view/ViewGroup.java
# This module omits device-pixel rounding and all runtime mutations. Its gate
# advice is therefore hold-for-verification, never a final visual failure.


def attr(node: ET.Element, key: str, default: str = "") -> str:
    return node.get(ANDROID + key, default)


def kind(node: ET.Element) -> str:
    return str(node.tag).rsplit("}", 1)[-1].rsplit(".", 1)[-1]


def dp(value: str) -> float | None:
    """No density guesses for px/sp or dimension-resource indirection."""
    if value in {"0", "0.0"}:
        return 0.0
    if not re.fullmatch(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:dp|dip)", value):
        return None
    result = float(re.sub(r"(?:dp|dip)$", "", value))
    return result if math.isfinite(result) else None


def node_id(node: ET.Element) -> str:
    raw = attr(node, "id")
    match = re.fullmatch(r"@\+?id/([A-Za-z_][A-Za-z0-9_]*)", raw)
    return match.group(1) if match else ""


def index_tree(root: ET.Element) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    def walk(node: ET.Element, path: str, parent: str | None) -> None:
        indexed[path] = {"element": node, "parent": parent, "kind": kind(node),
                         "id": node_id(node), "text": attr(node, "text"),
                         "description": attr(node, "contentDescription")}
        for i, child in enumerate(node):
            walk(child, f"{path}/{i}", path)
    walk(root, "0", None)
    return indexed


def correlate(before: dict, after: dict) -> tuple[dict, list]:
    """Shared unique IDs, then exact unique semantic labels for ID-less nodes.

    A unique label is legacy evidence, not proof of original model identity. No
    fallback for changed/duplicate IDs. Type changes are independently reported.
    """
    matches, uncertainty = {}, []
    id_a, id_b = defaultdict(list), defaultdict(list)
    for path, row in before.items():
        if row["id"]:
            id_a[row["id"]].append(path)
    for path, row in after.items():
        if row["id"]:
            id_b[row["id"]].append(path)
    for identity in sorted(id_a.keys() | id_b.keys()):
        aa, bb = id_a[identity], id_b[identity]
        if len(aa) == len(bb) == 1:
            matches[aa[0]] = {"after_path": bb[0], "basis": "shared_unique_android_id",
                              "identity": identity, "semantic_verified": False}
        else:
            uncertainty.append({"reason": "missing_or_duplicate_id", "identity": identity,
                                "before_paths": aa, "after_paths": bb})
    semantic_a, semantic_b = defaultdict(list), defaultdict(list)
    for nodes, bucket in ((before, semantic_a), (after, semantic_b)):
        for path, row in nodes.items():
            if row["id"] or not (row["text"].strip() or row["description"].strip()):
                continue
            # Preserve spelling/case; source-coordinate or sibling order is not evidence.
            key = (row["kind"], row["text"].strip(), row["description"].strip())
            bucket[key].append(path)
    for key in sorted(semantic_a.keys() | semantic_b.keys()):
        aa, bb = semantic_a[key], semantic_b[key]
        if len(aa) == len(bb) == 1:
            matches[aa[0]] = {"after_path": bb[0], "basis": "unique_exact_legacy_semantics",
                              "identity": list(key), "semantic_verified": False}
        else:
            uncertainty.append({"reason": "unmatched_or_ambiguous_legacy_semantics",
                                "semantic_key": list(key), "before_paths": aa, "after_paths": bb})
    for stage, nodes, paired in (("s3", before, set(matches)),
                                 ("s5", after, {m["after_path"] for m in matches.values()})):
        for path, row in nodes.items():
            if path not in paired and not list(row["element"]) and not row["id"] and not (
                    row["text"].strip() or row["description"].strip()):
                uncertainty.append({"reason": "anonymous_leaf_not_correlated", "stage": stage,
                                    "path": path, "kind": row["kind"]})
    return matches, uncertainty


def vertical_intervals(root: ET.Element, viewport_height_dp: float) -> dict:
    """Resolve only explicit fixed/parent heights and simple Android containers.

    Vertical LinearLayout: exact fixed children or positive 0dp weight children;
    no wrap-content estimation. FrameLayout: top/bottom/center vertical gravity.
    RelativeLayout supports independent top-positioned children only; sibling
    rules and non-top parent gravity remain unknown. Horizontal/custom layouts
    remain unknown. A fixed
    top below viewport is sufficient even if wrap-content height is unknown.
    """
    results: dict = {}

    def number(node: ET.Element, name: str, fallback: float = 0.0) -> float | None:
        raw = attr(node, name)
        return dp(raw) if raw else fallback

    def edge(node: ET.Element, name: str, umbrella: str) -> float | None:
        if umbrella == "layout_margin":
            # MarginLayoutParams: all-margin, then vertical, then side-specific.
            for key in (umbrella, "layout_marginVertical", name):
                if attr(node, key):
                    value = number(node, key)
                    if value is None:
                        return None
                    if key == name or value >= 0:
                        return value
            return 0.0
        # Deliberately do not guess precedence for conflicting padding variants.
        values = [number(node, key) for key in (umbrella, "paddingVertical", name) if attr(node, key)]
        if any(value is None or value < 0 for value in values) or len(set(values)) > 1:
            return None
        return values[0] if values else 0.0

    def gravity(node: ET.Element, name: str) -> str | None:
        # Android ORs flags. top|bottom is FILL_VERTICAL, not BOTTOM.
        flags = {"top": 48, "bottom": 80, "center_vertical": 16, "center": 16,
                 "fill_vertical": 112, "fill": 112, "left": 0, "right": 0,
                 "start": 0, "end": 0, "center_horizontal": 0, "fill_horizontal": 0,
                 "clip_horizontal": 0, "clip_vertical": 0, "no_gravity": 0, "": 0}
        mask = 0
        for token in attr(node, name).split("|"):
            if token not in flags:
                return None
            mask |= flags[token]
        return {16: "center", 80: "bottom"}.get(mask, "top")

    def intersection(a: tuple[float, float] | None, b: tuple[float, float] | None):
        return (max(a[0], b[0]), min(a[1], b[1])) if a is not None and b is not None else None

    def visit(node: ET.Element, path: str, top: float | None,
              height: float | None, reason: str = "", *,
              clip: tuple[float, float] | None = None, ancestors_visible: bool = True) -> None:
        translation = number(node, "translationY")
        visibility = attr(node, "visibility", "visible")
        if node.get("style") or attr(node, "style"):
            top, height, reason = None, None, "style_not_resolved"
        if height is not None and height < 0:
            top, height, reason = None, None, "invalid_negative_layout_height"
        if attr(node, "rotation") or attr(node, "rotationX") or attr(node, "rotationY") or (
                attr(node, "scaleY", "1") not in {"1", "1.0"}):
            top, reason = None, "transform_not_supported"
        if translation is None:
            top, reason = None, "translation_not_resolved"
        if attr(node, "fitsSystemWindows", "false") != "false" or attr(node, "layoutMode"):
            top, height, reason = None, None, "runtime_insets_or_optical_layout_unknown"
        if top is not None:
            top += translation or 0
        bottom = top + height if top is not None and height is not None else None
        visible = ancestors_visible and visibility == "visible" and attr(node, "alpha", "1") in {"1", "1.0"}
        positive = height is not None and height > 0
        bounds_inside = positive and top is not None and bottom is not None and 0 <= top < bottom <= viewport_height_dp
        bounds_outside = positive and top is not None and bottom is not None and (top >= viewport_height_dp or bottom <= 0)
        unclipped = clip is not None and positive and top is not None and bottom is not None and clip[0] <= top < bottom <= clip[1]
        fully_clipped = clip is not None and positive and top is not None and bottom is not None and (
            bottom <= clip[0] or top >= clip[1] or clip[0] >= clip[1])
        results[path] = {"top_dp": top, "bottom_dp": bottom, "height_dp": height,
                         "visibility": visibility, "reason": reason,
                         "ancestor_clip_dp": list(clip) if clip is not None else None,
                         "bounds_fully_inside_viewport": bounds_inside,
                         "bounds_fully_outside_viewport": bounds_outside,
                         "proven_fully_clipped_by_ancestors": fully_clipped,
                         "proven_fully_outside": visible and bounds_outside,
                         "proven_fully_inside": visible and bounds_inside and unclipped,
                         "runtime_visibility_verified": False}
        children = list(node)
        if not children:
            return
        pt, pb = edge(node, "paddingTop", "padding"), edge(node, "paddingBottom", "padding")
        background = attr(node, "background")
        if background and not (background.startswith("#") or background == "@null") and not (
                attr(node, "padding") or (attr(node, "paddingTop") and attr(node, "paddingBottom"))):
            pt, pb = None, None  # A resource drawable may supply implicit padding.
        if attr(node, "foreground") not in {"", "@null"}:
            pt, pb = None, None  # FrameLayout foreground can alter layout padding.
        child_clip = clip
        clip_children, clip_padding = attr(node, "clipChildren", "true"), attr(node, "clipToPadding", "true")
        if clip_children not in {"true", "false"} or clip_padding not in {"true", "false"} or attr(node, "clipToOutline", "false") != "false":
            child_clip = None
        else:
            if clip_children == "true":
                child_clip = intersection(child_clip, (top, bottom) if top is not None and bottom is not None else None)
            horizontal_padding_declared = any(attr(node, key) not in {"", "0", "0dp", "0dip"} for key in (
                "paddingLeft", "paddingRight", "paddingStart", "paddingEnd", "paddingHorizontal"))
            if clip_padding == "true" and (pt != 0 or pb != 0 or horizontal_padding_declared):
                inner_clip = (top + pt, bottom - pb) if top is not None and bottom is not None and pt is not None and pb is not None else None
                child_clip = intersection(child_clip, inner_clip)

        def descend(child: ET.Element, child_path: str, child_top: float | None,
                    child_height: float | None, child_reason: str = "") -> None:
            visit(child, child_path, child_top, child_height, child_reason,
                  clip=child_clip, ancestors_visible=visible)

        if attr(node, "scrollY") not in {"", "0", "0dp", "0dip"} or attr(node, "scrollX") not in {"", "0", "0dp", "0dip"}:
            for i, child in enumerate(children):
                descend(child, f"{path}/{i}", None, None, "scroll_offset_not_supported")
            return
        if visibility == "gone" or top is None or height is None or pt is None or pb is None or attr(node, "style") or node.get("style"):
            for i, child in enumerate(children):
                descend(child, f"{path}/{i}", None, None, "parent_measurement_unknown")
            return
        inner_top, inner_height = top + pt, height - pt - pb
        tag = kind(node)
        raw_tag = str(node.tag).rsplit("}", 1)[-1]
        if "." in raw_tag and raw_tag != f"android.widget.{tag}":
            tag = "custom_layout"
        if tag in {"FrameLayout", "RelativeLayout"}:
            for i, child in enumerate(children):
                if tag == "RelativeLayout":
                    rules = ("layout_above", "layout_below", "layout_alignTop", "layout_alignBottom",
                             "layout_alignBaseline", "layout_alignParentBottom", "layout_centerVertical",
                             "layout_centerInParent")
                    if any(attr(child, rule) not in {"", "false"} for rule in rules) or gravity(node, "gravity") != "top":
                        descend(child, f"{path}/{i}", None, None, "relative_vertical_rules_not_supported")
                        continue
                mt = edge(child, "layout_marginTop", "layout_margin")
                mb = edge(child, "layout_marginBottom", "layout_margin")
                h = dp(attr(child, "layout_height"))
                if mt is None or mb is None:
                    descend(child, f"{path}/{i}", None, h, "margin_not_resolved")
                    continue
                if attr(child, "layout_height") in {"match_parent", "fill_parent"}:
                    h = max(0, inner_height - mt - mb)
                child_gravity = gravity(child, "layout_gravity") if tag == "FrameLayout" else "top"
                if child_gravity is None:
                    descend(child, f"{path}/{i}", None, h, "gravity_not_resolved")
                    continue
                if child_gravity == "bottom":
                    y = inner_top + inner_height - mb - h if h is not None else None
                elif child_gravity == "center":
                    y = inner_top + (inner_height - h) / 2 + mt - mb if h is not None else None
                else:
                    y = inner_top + mt
                descend(child, f"{path}/{i}", y, h, "" if h is not None else "height_not_resolved")
        elif tag == "LinearLayout" and attr(node, "orientation", "horizontal") == "vertical":
            specs = []
            for child in children:
                h = dp(attr(child, "layout_height"))
                mt, mb = edge(child, "layout_marginTop", "layout_margin"), edge(child, "layout_marginBottom", "layout_margin")
                try:
                    weight = float(attr(child, "layout_weight", "0"))
                except ValueError:
                    weight = math.nan
                if attr(child, "visibility") == "gone":
                    h, mt, mb, weight = 0.0, 0.0, 0.0, 0.0
                elif child.get("style") or attr(child, "style") or attr(child, "visibility", "visible") not in {"visible", "invisible"}:
                    h = None
                specs.append((h, mt, mb, weight))
            supported = not attr(node, "weightSum") and not attr(node, "showDividers") and not attr(node, "divider") and not attr(node, "baselineAlignedChildIndex") and attr(node, "measureWithLargestChild", "false") == "false" and gravity(node, "gravity") is not None and all(
                h is not None and mt is not None and mb is not None and math.isfinite(w)
                and h >= 0 and mt >= 0 and mb >= 0 and w >= 0 and (w == 0 or h == 0) for h, mt, mb, w in specs)
            fixed = sum(h + mt + mb for h, mt, mb, _ in specs) if supported else 0
            weights = sum(s[3] for s in specs) if supported else 0
            if not supported or (weights and fixed > inner_height):
                for i, child in enumerate(children):
                    descend(child, f"{path}/{i}", None, None, "linear_measurement_not_supported")
                return
            parent_gravity = gravity(node, "gravity")
            remaining = inner_height - fixed
            offset = 0 if weights else (remaining if parent_gravity == "bottom" else
                                         remaining / 2 if parent_gravity == "center" else 0)
            cursor = inner_top + offset
            for i, (child, spec) in enumerate(zip(children, specs)):
                h, mt, mb, weight = spec
                h += remaining * weight / weights if weights else 0
                descend(child, f"{path}/{i}", cursor + mt, h)
                cursor += mt + h + mb
        else:
            for i, child in enumerate(children):
                descend(child, f"{path}/{i}", None, None, "container_measurement_not_supported")

    root_height = dp(attr(root, "layout_height"))
    if attr(root, "layout_height") in {"match_parent", "fill_parent"}:
        root_height = viewport_height_dp
    if root_height is None or root_height <= 0 or attr(root, "layout_weight"):
        visit(root, "0", None, None, "root_height_or_parent_measurement_unknown", clip=(0, viewport_height_dp))
    elif any(attr(root, name) for name in ("layout_margin", "layout_marginTop", "layout_marginBottom", "layout_marginVertical", "layout_gravity")):
        visit(root, "0", None, root_height, "root_parent_layout_params_not_resolved", clip=(0, viewport_height_dp))
    else:
        visit(root, "0", 0, root_height, clip=(0, viewport_height_dp))
    return results


def audit_xml(before_xml: str, after_xml: str, viewport_height_dp: float | None = None) -> dict:
    if viewport_height_dp is not None and (not math.isfinite(viewport_height_dp) or viewport_height_dp <= 0):
        raise ValueError("viewport height must be finite and positive")
    roots = ET.fromstring(before_xml), ET.fromstring(after_xml)
    before, after = map(index_tree, roots)
    matches, uncertainty = correlate(before, after)
    geometry = [vertical_intervals(root, viewport_height_dp) for root in roots] if viewport_height_dp else [{}, {}]
    findings = []
    for path, match in matches.items():
        dest = match["after_path"]
        a, b = before[path], after[dest]
        common = {"before_path": path, "after_path": dest, "match": match}
        if a["kind"] != b["kind"]:
            findings.append({**common, "kind": "widget_type_changed", "before": a["kind"], "after": b["kind"],
                             "status": "requires_review"})
        if (a["text"], a["description"]) != (b["text"], b["description"]):
            findings.append({**common, "kind": "text_or_description_changed", "before": [a["text"], a["description"]],
                             "after": [b["text"], b["description"]], "status": "requires_review"})
        old_parent = a["parent"]
        expected_parent = matches.get(old_parent, {}).get("after_path")
        if expected_parent is not None and expected_parent != b["parent"]:
            findings.append({**common, "kind": "matched_parent_anchor_changed", "before_parent": old_parent,
                             "expected_after_parent": expected_parent, "actual_after_parent": b["parent"],
                             "status": "requires_review"})
        elif path.count("/") > dest.count("/"):
            findings.append({**common, "kind": "ancestor_depth_reduced", "before_parent": old_parent,
                             "actual_after_parent": b["parent"], "status": "requires_review"})
        ga, gb = geometry[0].get(path, {}), geometry[1].get(dest, {})
        if ga.get("proven_fully_inside") and gb.get("proven_fully_outside"):
            findings.append({**common, "kind": "new_fully_outside_viewport", "before_interval": ga,
                             "after_interval": gb, "status": "confirmed_static_regression"})
    matched_after = {m["after_path"]: p for p, m in matches.items()}
    for path, interval in geometry[1].items():
        if interval["proven_fully_outside"]:
            findings.append({"kind": "s5_fully_outside_viewport", "after_path": path,
                             "before_path": matched_after.get(path), "widget_kind": after[path]["kind"],
                             "interval": interval, "status": "confirmed_static_bounds_only",
                             "source_identity_known": path in matched_after})
    return {
        "schema_version": 1, "diagnostic_only": True, "xml_modified": False,
        "viewport_height_dp": viewport_height_dp,
        "measurement_scope": "static_vertical_supported_layouts_only_not_render_validation",
        "pixel_rounding": "not_modeled_requires_runtime_verification",
        "identity_scope": "shared_unique_ids_or_unique_exact_legacy_labels_no_positional_pairing",
        "before_sha256": hashlib.sha256(before_xml.encode()).hexdigest(),
        "after_sha256": hashlib.sha256(after_xml.encode()).hexdigest(),
        "node_counts": {"s3": len(before), "s5": len(after), "correlated": len(matches)},
        "matches": matches, "findings": findings, "uncertainties": uncertainty,
        "geometry": {"s3": geometry[0], "s5": geometry[1]},
        "summary": dict(Counter(f["kind"] for f in findings)),
        "gate_advice": {
            "hold_for_render_verification": any(f["kind"] in {"new_fully_outside_viewport", "s5_fully_outside_viewport"} for f in findings),
            "hierarchy_review_required": any(f["status"] == "requires_review" for f in findings),
            "hard_visual_failure_proven": False,
            "zero_findings_is_not_pass": True,
        },
        "restoration": {"performed": False, "reason": "source_subtree_correctness_and_target_parent_semantics_not_proven"},
        "acceptance": "pending_render_and_semantic_review",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s3", type=Path, required=True)
    parser.add_argument("--s5", type=Path, required=True)
    parser.add_argument("--viewport-height-dp", type=float)
    parser.add_argument("--viewport-source", default="explicit_caller_value_not_independently_verified")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error("refusing to overwrite an existing report")
    report = audit_xml(args.s3.read_bytes().decode("utf-8"), args.s5.read_bytes().decode("utf-8"), args.viewport_height_dp)
    report["inputs"] = {"s3": str(args.s3), "s5": str(args.s5)}
    report["viewport_source"] = args.viewport_source
    report["auditor_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": str(args.out), "summary": report["summary"],
                      "acceptance": report["acceptance"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
