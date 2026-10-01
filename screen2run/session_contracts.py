"""Local S1--S5 syntax/transfer contracts; passing is not semantic completeness.

This module has no model, OCR, emulator, or network dependencies. Legacy bounds
arrays are preserved as data, never silently interpreted as xyxy or xywh.
"""
from __future__ import annotations

import json
import copy
import math
import re
import xml.etree.ElementTree as ET
from typing import Any


class StageContractError(ValueError):
    """A stage output cannot be safely passed to the next stage."""


def _outer_fence(text: str, language: str) -> str:
    text = text.strip()
    match = re.fullmatch(r"```(?:" + language + r")?\s*\n(.*?)\n```", text,
                         flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text


def parse_stage_json(text: str) -> dict | list:
    """Parse a nonempty JSON object/array, accepting only a whole-document fence."""
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise StageContractError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise StageContractError(f"Non-finite JSON value: {value}")

    try:
        result = json.loads(_outer_fence(text, "json"), object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
    except (ValueError, TypeError, AttributeError) as exc:
        # a model may narrate before/after the document ("I'm mapping the screenshot..."):
        # accept exactly one outermost object if everything around it is prose
        body = text if isinstance(text, str) else ""
        start, end = body.find("{"), body.rfind("}")
        try:
            if start < 0 or end <= start:
                raise ValueError("no enclosed JSON object")
            result = json.loads(body[start:end + 1], object_pairs_hook=unique_object,
                                parse_constant=reject_constant)
        except (ValueError, TypeError) as inner:
            raise StageContractError(f"Stage JSON is invalid: {exc}") from inner
    if not isinstance(result, (dict, list)) or not result:
        raise StageContractError("Stage JSON must be a nonempty object or array")
    return result


# These fields may contain nested visual descriptions, but are not child nodes.
_VISUAL_FIELDS = {
    "text", "color", "colour", "style", "styles", "font", "font_size",
    "font_family", "text_color", "background_color", "background", "description",
    "visual_style",
}
_GEOMETRY_FIELDS = ("bounds", "bbox", "rect", "position")
_ID_FIELDS = ("id", "node_id", "element_id")
_ANDROID = "{http://schemas.android.com/apk/res/android}"
_COORDINATE_KEY = "_coordinate_contract"
REFERENCE_BOUNDS_EPSILON_PX = 1e-6


_STYLE_KEYWORDS = (
    ("corner_radius", ("shape", "corner_radius")), ("progress_fraction", ("widget_state", "progress_fraction")),
    ("font_weight", ("typography", "font_weight")), ("font_style", ("typography", "font_style")),
    ("font_family", ("typography", "font_family")), ("alignment", ("typography", "alignment")),
    ("max_lines", ("typography", "max_lines")), ("include_font_padding", ("typography", "include_font_padding")),
    ("gradient angle", ("background", "angle_degrees")), ("background", ("background",)),
    ("shape", ("shape",)), ("checked", ("widget_state",)), ("enabled", ("widget_state",)),
    ("selected", ("widget_state",)),
)


def coerce_visual_styles(tree: dict, fill_missing: bool = True) -> list[dict]:
    """Turn malformed observations into "unknown" instead of failing the screen.

    A field that violates the typed contract is removed (the contract's own meaning of
    "not observed"); nothing is filled in.  An app node without ``visual_style`` gets the
    explicit empty style ``{}`` the contract asks for.  Every coercion is returned so the
    manifest can report it.
    """
    coerced = []

    def valid(style):
        try:
            validate_visual_styles({"node_id": "_probe", "visual_style": style, "children": []})
            return None
        except StageContractError as exc:
            return str(exc)

    def drop(style, keys):
        target = style
        for key in keys[:-1]:
            target = target.get(key) if isinstance(target, dict) else None
            if not isinstance(target, dict):
                return False
        if keys[-1] in target:
            del target[keys[-1]]
            return True
        return False

    def visit(node, chrome=False):
        chrome = chrome or node.get("system_chrome") in ("status_bar", "navigation_bar")
        identity = node.get("node_id", "unassigned_source_node")
        style = node.get("visual_style")
        if style is None and not chrome and "visual_style" not in node and fill_missing:
            node["visual_style"] = {}
            coerced.append({"node_id": identity, "field": "visual_style", "action": "missing->{}"})
        elif style is not None and not isinstance(style, dict):
            node["visual_style"] = {}
            coerced.append({"node_id": identity, "field": "visual_style", "action": "non-object->{}"})
        elif isinstance(style, dict):
            for _ in range(24):
                error = valid(style)
                if error is None:
                    break
                path = error.split(" at ", 1)[-1].split(":", 1)[0]
                keys = [k for k in path.split(".")[1:] if k]
                done = bool(keys) and drop(style, keys)
                if not done:
                    for word, ks in _STYLE_KEYWORDS:
                        if word in error and drop(style, list(ks)):
                            keys, done = list(ks), True
                            break
                if not done:
                    style.clear()
                    keys = ["*"]
                coerced.append({"node_id": identity, "field": ".".join(keys), "action": "invalid->unknown",
                                "error": error[:160]})
        for child in node.get("children", []) or []:
            if isinstance(child, dict):
                visit(child, chrome)

    visit(tree)
    return coerced


def validate_visual_styles(tree: dict, *, require_declared: bool = False) -> dict:
    """Validate typed observations, not their visual truth; never fill defaults.

    Unknown fields are omitted/null. An empty style is retained as incomplete,
    not replaced with guessed font metrics, radius, stroke or widget state.
    """
    report = {"schema_version": 1, "app_nodes": 0, "declared_nodes": 0,
              "empty_or_missing_node_ids": [], "semantic_validation": "pending_reference_and_render_review"}
    color_pattern = re.compile(r"#[0-9a-fA-F]{6}(?:[0-9a-fA-F]{2})?$")

    def fail(path, message):
        raise StageContractError(f"Invalid visual_style at {path}: {message}")

    def object_fields(value, allowed, path):
        if not isinstance(value, dict):
            fail(path, "expected object or null")
        if set(value) - set(allowed):
            fail(path, f"unknown fields {sorted(set(value) - set(allowed))}")

    def number(value, path, minimum=0, strictly_positive=False):
        if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                not math.isfinite(value) or value < minimum or (strictly_positive and value == 0)):
            fail(path, "expected finite value in the declared range")

    def color(value, path):
        if value is not None and (not isinstance(value, str) or not color_pattern.fullmatch(value)):
            fail(path, "expected #RRGGBB or #AARRGGBB")

    def quantity(value, path, units, positive=False, minimum=0):
        if value is None:
            return
        object_fields(value, {"value", "unit"}, path)
        if (set(value) != {"value", "unit"} or not isinstance(value["unit"], str)
                or value["unit"] not in units):
            fail(path, f"quantity requires value and explicit unit in {sorted(units)}")
        number(value["value"], path, minimum=minimum, strictly_positive=positive)

    def check(style, path):
        object_fields(style, {"background", "shape", "stroke", "typography", "content_padding", "widget_state"}, path)
        background = style.get("background")
        if background is not None:
            object_fields(background, {"kind", "color", "start_color", "end_color", "angle_degrees"}, path + ".background")
            kind = background.get("kind")
            if kind not in (None, "solid", "gradient"):
                fail(path, "background.kind must be solid, gradient or null")
            for key in ("color", "start_color", "end_color"):
                color(background.get(key), path + ".background." + key)
            angle = background.get("angle_degrees")
            if angle is not None and (isinstance(angle, bool) or angle not in (0, 45, 90, 135, 180, 225, 270, 315)):
                fail(path, "gradient angle must be a supported explicit degree value")
            if kind == "solid" and any(background.get(k) is not None for k in ("start_color", "end_color", "angle_degrees")):
                fail(path, "solid background cannot also declare gradient fields")
            if kind == "gradient" and background.get("color") is not None:
                fail(path, "gradient background cannot also declare solid color")
        shape = style.get("shape")
        if shape is not None:
            object_fields(shape, {"kind", "corner_radius"}, path + ".shape")
            if shape.get("kind") not in (None, "rectangle", "oval", "circle"):
                fail(path, "shape.kind must be rectangle, oval, circle or null")
            quantity(shape.get("corner_radius"), path + ".shape.corner_radius", {"reference_px", "dp"})
            if shape.get("corner_radius") is not None and shape.get("kind") != "rectangle":
                fail(path, "corner_radius requires an explicit rectangle shape")
        stroke = style.get("stroke")
        if stroke is not None:
            object_fields(stroke, {"color", "width"}, path + ".stroke")
            color(stroke.get("color"), path + ".stroke.color")
            quantity(stroke.get("width"), path + ".stroke.width", {"reference_px", "dp"}, positive=True)
        typography = style.get("typography")
        if typography is not None:
            object_fields(typography, {"color", "font_size", "font_weight", "font_family", "font_style",
                                      "line_height", "letter_spacing", "max_lines", "alignment", "include_font_padding"}, path + ".typography")
            color(typography.get("color"), path + ".typography.color")
            quantity(typography.get("font_size"), path + ".typography.font_size", {"reference_px", "sp"}, positive=True)
            quantity(typography.get("line_height"), path + ".typography.line_height", {"reference_px", "dp"}, positive=True)
            quantity(typography.get("letter_spacing"), path + ".typography.letter_spacing", {"em"}, minimum=-math.inf)
            if typography.get("font_weight") not in (None, "normal", "bold", "100", "200", "300", "400", "500", "600", "700", "800", "900"):
                fail(path, "font_weight must be a named/explicit string weight or null")
            if typography.get("font_style") not in (None, "normal", "italic"):
                fail(path, "font_style must be normal, italic or null")
            family = typography.get("font_family")
            if family is not None and (not isinstance(family, str) or not family.strip()):
                fail(path, "font_family must be a nonempty name or null")
            if typography.get("alignment") not in (None, "start", "center", "end"):
                fail(path, "alignment must be start, center, end or null")
            maximum = typography.get("max_lines")
            if maximum is not None and (isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0):
                fail(path, "max_lines must be a positive integer or null")
            padding = typography.get("include_font_padding")
            if padding is not None and not isinstance(padding, bool):
                fail(path, "include_font_padding must be boolean or null")
        padding = style.get("content_padding")
        if padding is not None:
            object_fields(padding, {"left", "top", "right", "bottom"}, path + ".content_padding")
            for key, value in padding.items():
                quantity(value, path + ".content_padding." + key, {"reference_px", "dp"})
        state = style.get("widget_state")
        if state is not None:
            object_fields(state, {"checked", "enabled", "selected", "progress_fraction"}, path + ".widget_state")
            for key in ("checked", "enabled", "selected"):
                if state.get(key) is not None and not isinstance(state[key], bool):
                    fail(path, key + " must be boolean or null")
            if state.get("progress_fraction") is not None:
                number(state["progress_fraction"], path + ".widget_state.progress_fraction")
                if state["progress_fraction"] > 1:
                    fail(path, "progress_fraction must be in [0,1]")

    def visit(node, inherited_chrome=False):
        chrome = inherited_chrome or node.get("system_chrome") in ("status_bar", "navigation_bar")
        identity = node.get("node_id", "unassigned_source_node")
        style = node.get("visual_style")
        if style is not None:
            check(style, identity)
        if not chrome:
            report["app_nodes"] += 1
            if "visual_style" not in node or style is None:
                if require_declared:
                    fail(identity, "new S2 must explicitly provide visual_style ({} if observations are unknown)")
            else:
                report["declared_nodes"] += 1
            def has_observation(value):
                if isinstance(value, dict):
                    return any(has_observation(v) for v in value.values())
                return value is not None
            if not style or not has_observation(style):
                report["empty_or_missing_node_ids"].append(identity)
        for child in node.get("children", []):
            visit(child, chrome)

    visit(tree)
    return report


def prepare_reference_tree(tree: dict | list, reference_size: tuple[int, int]) -> dict:
    """Canonicalize *new* S1 output; never infer the format of historical bounds.

    IDs belong to this host-defined tree contract, not the screen or model. The
    raw model response must be retained separately. Bounds are visible reference
    pixels; overlapping children and bounds outside their parent are permitted.
    """
    result = copy.deepcopy(tree)
    if not isinstance(result, dict):
        raise StageContractError("New S1 must be one UI root object")
    declaration = {
        "version": 2, "space": "reference_pixels", "format": "xyxy",
        "reference_size_px": list(reference_size), "identity_owner": "host_tree_path",
    }
    if _COORDINATE_KEY in result and result[_COORDINATE_KEY] != declaration:
        raise StageContractError("S1 declared a conflicting coordinate convention")
    result[_COORDINATE_KEY] = declaration

    def assign(node, parts):
        if not isinstance(node, dict):
            raise StageContractError("Every UI child must be an object")
        node["node_id"] = "s2r_n_" + "_".join(map(str, parts))
        node.setdefault("children", [])
        if not isinstance(node["children"], list):
            raise StageContractError("UI children must be a list")
        for index, child in enumerate(node["children"]):
            assign(child, (*parts, index))

    assign(result, (0,))
    validate_reference_tree(result, reference_size)
    validate_visual_styles(result)
    return result


def validate_reference_tree(tree: dict, reference_size: tuple[int, int]) -> None:
    """Require the declared new schema, finite in-frame xyxy and stable identity."""
    width, height = reference_size
    expected = {"version": 2, "space": "reference_pixels", "format": "xyxy",
                "reference_size_px": [width, height], "identity_owner": "host_tree_path"}
    if tree.get(_COORDINATE_KEY) != expected:
        raise StageContractError("Missing or changed reference coordinate declaration")
    if width <= 0 or height <= 0:
        raise StageContractError("Reference size must be positive")

    def visit(node, parts):
        expected_id = "s2r_n_" + "_".join(map(str, parts))
        if not isinstance(node, dict) or node.get("node_id") != expected_id:
            raise StageContractError(f"Missing or changed stable node identity: {expected_id}")
        if not isinstance(node.get("type"), str) or not node["type"].strip():
            raise StageContractError(f"Missing UI type: {expected_id}")
        if ("system_chrome" in node and
            node["system_chrome"] not in ("status_bar", "navigation_bar")):
            raise StageContractError(f"Unknown explicit system chrome role: {expected_id}")
        if parts == (0,) and "system_chrome" in node:
            raise StageContractError("The entire UI root cannot be excluded as system chrome")
        box = node.get("bounds")
        if (not isinstance(box, list) or len(box) != 4 or
            any(isinstance(v, bool) or not isinstance(v, (int, float)) or
                not math.isfinite(v) for v in box)):
            raise StageContractError(f"Expected finite reference xyxy bounds: {expected_id}")
        x0, y0, x1, y1 = box
        epsilon = REFERENCE_BOUNDS_EPSILON_PX
        if not (-epsilon <= x0 < x1 <= width + epsilon and
                -epsilon <= y0 < y1 <= height + epsilon):
            raise StageContractError(f"Reference bounds outside image or degenerate: {expected_id}")
        if any(key in node for key in ("bbox", "rect", "position", "elements", "items")):
            raise StageContractError(f"Ambiguous alternate geometry/children schema: {expected_id}")
        if not isinstance(node.get("children"), list):
            raise StageContractError(f"Missing children list: {expected_id}")
        for index, child in enumerate(node["children"]):
            visit(child, (*parts, index))

    visit(tree, (0,))


def xml_identity_report(tree: dict, xml: str) -> dict:
    """Account for declared mappings, not visual/semantic correctness.

    Merged containers/compound widgets are allowed; repeated source assignments
    remain pending because they can duplicate visible content.
    android:tag='s2r_nodes=id1,id2' declares a many-to-one mapping explicitly;
    nothing is guessed from labels, positions, or list order. Explicit model
    system-chrome declarations remain unverified exclusions in the report.
    """
    root = ET.fromstring(validate_xml(xml))
    sources = {}
    exclusions = []

    def collect(node, inherited_chrome=None):
        chrome = node.get("system_chrome", inherited_chrome)
        if chrome not in ("status_bar", "navigation_bar"):
            chrome = inherited_chrome
        if chrome:
            exclusions.append({"node_id": node["node_id"], "reason": chrome,
                               "evidence": "S1 declaration; semantic verification pending"})
        else:
            sources[node["node_id"]] = node
        for child in node["children"]:
            collect(child, chrome)

    collect(tree)
    targets, unknown, malformed, xml_ids, composites = {}, [], [], {}, []
    for index, node in enumerate(root.iter()):
        identity = node.get(_ANDROID + "id", "")
        match = re.fullmatch(r"@\+?id/([a-zA-Z_][a-zA-Z0-9_]*)", identity)
        target_id = match.group(1) if match else None
        if target_id:
            xml_ids.setdefault(target_id, []).append(index)
        declarations = set()
        if target_id and target_id.startswith("s2r_n_"):
            declarations.add(target_id)
        tag = node.get(_ANDROID + "tag", "")
        if tag.startswith("s2r_nodes="):
            declared = tag.removeprefix("s2r_nodes=").split(",")
            if not target_id or any(not re.fullmatch(r"s2r_n_\d+(?:_\d+)*", x.strip())
                                    for x in declared):
                malformed.append({"xml_index": index, "reason": "Invalid explicit node mapping"})
            else:
                declarations.update(x.strip() for x in declared)
        if len(declarations) > 1:
            composites.append({"android_id": target_id, "source_node_ids": sorted(declarations),
                               "verification": "pending; must verify actual representation, not just tags"})
        for source_id in sorted(declarations):
            if source_id not in sources:
                unknown.append({"node_id": source_id, "xml_index": index})
            else:
                targets.setdefault(source_id, []).append({"android_id": target_id,
                                                         "xml_index": index})
    duplicate_ids = sorted(key for key, locations in xml_ids.items() if len(locations) > 1)
    repeated_sources = sorted(key for key, locations in targets.items() if len(locations) > 1)
    missing = sorted(set(sources) - set(targets))
    return {
        "status": "pending_mapping" if missing or unknown or malformed or duplicate_ids or repeated_sources else "declared_mapping_complete",
        "semantic_validation": "pending_render_and_review",
        "acceptance_claim": "none; declarations do not prove that source nodes were rendered",
        "expected_app_node_count": len(sources), "mapped_node_count": len(targets),
        "mappings": targets, "missing_node_ids": missing,
        "system_chrome_exclusions": exclusions, "unknown_or_excluded_claims": unknown,
        "malformed_mapping_declarations": malformed, "duplicate_android_ids": duplicate_ids,
        "repeated_source_mappings": repeated_sources, "composite_mapping_claims": composites,
    }


def _frozen_geometry(value):
    if isinstance(value, dict):
        return ("object", tuple((k, _frozen_geometry(v)) for k, v in sorted(value.items())))
    if isinstance(value, list):
        return ("array", tuple(_frozen_geometry(v) for v in value))
    if isinstance(value, bool):
        return ("boolean", value)
    return value


def _structure_signature(value: Any, path: str = "$") -> list[tuple]:
    """Retain UI tree topology and geometry without interpreting legacy bounds."""
    result = []
    if isinstance(value, dict):
        is_node = "type" in value or any(k in value for k in _GEOMETRY_FIELDS)
        if is_node:
            geometry = tuple((k, _frozen_geometry(value[k]))
                             for k in _GEOMETRY_FIELDS if k in value)
            identity = tuple((k, str(value[k])) for k in _ID_FIELDS if k in value)
            result.append((path, value.get("type"), geometry, identity))
        for key, child in value.items():
            if key in _VISUAL_FIELDS or key in _GEOMETRY_FIELDS:
                continue
            if isinstance(child, (dict, list)):
                if key in ("children", "elements", "items"):
                    # Include even empty lists: silently deleting or changing the
                    # child-container type is a contract change, not enrichment.
                    result.append((f"{path}.{key}", "container", type(child).__name__))
                result.extend(_structure_signature(child, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.extend(_structure_signature(child, f"{path}[{index}]"))
    return result


def assert_structure_preserved(s1: dict | list, s2: dict | list) -> None:
    """S2 may enrich text/style, but cannot silently move, reorder, or drop nodes."""
    if type(s1) is not type(s2):
        raise StageContractError("S1/S2 root container types differ")
    if isinstance(s1, dict) and _COORDINATE_KEY in s1:
        contract = s1[_COORDINATE_KEY]
        validate_reference_tree(s2, tuple(contract["reference_size_px"]))
        def chrome_signature(node):
            return (node.get("system_chrome"), tuple(chrome_signature(c) for c in node["children"]))
        if chrome_signature(s1) != chrome_signature(s2):
            raise StageContractError("S2 changed system-chrome declarations")
        validate_visual_styles(s2)

        def preserve_observations(before, after, path):
            if before is None:
                return
            if isinstance(before, dict):
                if not isinstance(after, dict):
                    raise StageContractError(f"S2 dropped visual observations at {path}")
                for key, value in before.items():
                    preserve_observations(value, after.get(key), f"{path}.{key}")
            elif before != after:
                raise StageContractError(f"S2 changed known visual observation at {path}")

        def preserve_node_visuals(before, after):
            if before.get("visual_style") is not None:
                preserve_observations(before["visual_style"], after.get("visual_style"), before["node_id"])
            for a, b in zip(before["children"], after["children"]):
                preserve_node_visuals(a, b)

        preserve_node_visuals(s1, s2)
    before, after = _structure_signature(s1), _structure_signature(s2)
    if not before or not after:
        raise StageContractError("S1/S2 contain no recognizable UI nodes")
    if before != after:
        first = next((i for i, pair in enumerate(zip(before, after))
                      if pair[0] != pair[1]), min(len(before), len(after)))
        left = before[first] if first < len(before) else "<missing>"
        right = after[first] if first < len(after) else "<missing>"
        raise StageContractError(
            f"S2 changed structure/geometry at entry {first}: S1={left!r}, S2={right!r}")


def validate_xml(text: str) -> str:
    """Return XML text if well formed; this does not guarantee Android compilation."""
    clean = _outer_fence(text, "xml")
    try:
        ET.fromstring(clean)
    except (ET.ParseError, TypeError, ValueError) as exc:
        raise StageContractError(f"Stage XML is invalid: {exc}") from exc
    return clean


def xml_native_widget_report(tree: dict, xml: str) -> dict:
    """Detect explicit native-control -> image substitutions using source IDs.

    No label/order/geometry guessing and no automatic XML changes. Correct type
    declarations alone cannot prove appearance, state, text fit or interaction.
    """
    expected_types = {
        "switch": {"Switch"}, "checkbox": {"CheckBox"}, "check_box": {"CheckBox"},
        "radio_button": {"RadioButton"}, "radiobutton": {"RadioButton"},
        "slider": {"SeekBar"}, "seekbar": {"SeekBar"}, "seek_bar": {"SeekBar"},
        "progress_bar": {"ProgressBar"}, "progressbar": {"ProgressBar"},
        "ratingbar": {"RatingBar"}, "rating_bar": {"RatingBar"},
        "togglebutton": {"ToggleButton"}, "toggle_button": {"ToggleButton"},
    }
    mapping = xml_identity_report(tree, xml)
    elements = list(ET.fromstring(validate_xml(xml)).iter())
    issues, checked = [], []

    def visit(node, inherited_chrome=False):
        chrome = inherited_chrome or node.get("system_chrome") in ("status_bar", "navigation_bar")
        identity = node["node_id"]
        source_type = str(node.get("type", "")).strip().lower().replace("-", "_")
        expected = expected_types.get(source_type)
        if expected and not chrome:
            targets = mapping["mappings"].get(identity, [])
            if len(targets) != 1:
                issues.append({"node_id": identity, "reason": "native_control_mapping_unresolved",
                               "source_type": source_type})
            else:
                element = elements[targets[0]["xml_index"]]
                actual = element.tag.rsplit("}", 1)[-1]
                row = {"node_id": identity, "source_type": source_type,
                       "expected_xml_types": sorted(expected), "actual_xml_type": actual}
                checked.append(row)
                native_class = actual.removeprefix("android.widget.") if actual.startswith("android.widget.") else actual
                if native_class not in expected:
                    issues.append({**row, "reason": "native_control_replaced_by_non_native_type"})
                for key in ("thumb", "track", "button", "tickMark", "progressDrawable"):
                    if re.fullmatch(r"@drawable/(?:img|guigpt\w*|s2r_(?:iv|slot)\w*)", element.get(_ANDROID + key, "")):
                        issues.append({**row, "reason": "native_control_part_uses_image_placeholder_or_crop",
                                       "attribute": key})
        for child in node["children"]:
            visit(child, chrome)

    visit(tree)
    return {"status": "pending_native_widget_contract" if issues else "declared_native_types_consistent",
            "checked_nodes": checked, "issues": issues,
            "semantic_validation": "pending_state_text_fit_render_and_interaction_review",
            "acceptance_claim": "none; native classes do not prove visual fidelity or text fit"}
