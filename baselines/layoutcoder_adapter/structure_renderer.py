"""Deterministic fusion from LayoutCoder's structure tree to Android XML."""

import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional

from .xml_extract import ANDROID_NS, extract_android_xml


A = f"{{{ANDROID_NS}}}"
ET.register_namespace("android", ANDROID_NS)


class StructureRenderError(ValueError):
    """Raised for an invalid tree or missing atomic response."""


class _ResponseResolver:
    def __init__(self, responses: Any):
        self.responses = responses
        self.index = 0

    def get(self, node: Dict[str, Any], atomic_id: int) -> str:
        if isinstance(self.responses, Mapping):
            value = self.responses.get(node.get("id", atomic_id))
            if value is None:
                value = self.responses.get(str(node.get("id", atomic_id)))
        elif isinstance(self.responses, Sequence) and not isinstance(
            self.responses, (str, bytes)
        ):
            if self.index >= len(self.responses):
                value = None
            else:
                value = self.responses[self.index]
            self.index += 1
        else:
            raise StructureRenderError("Atomic responses must be a JSON object or array.")

        if isinstance(value, Mapping):
            value = value.get("response", value.get("content"))
        if not isinstance(value, str):
            raise StructureRenderError(
                f"Missing saved XML response for atomic component {atomic_id}."
            )
        return value


def _set_size(element: ET.Element, width: str, height: str) -> None:
    element.set(A + "layout_width", width)
    element.set(A + "layout_height", height)


def _weight(node: Dict[str, Any]) -> str:
    try:
        value = float(node.get("portion", 1))
    except (TypeError, ValueError) as exc:
        raise StructureRenderError("Node portion must be numeric.") from exc
    if value <= 0:
        raise StructureRenderError("Node portion must be greater than zero.")
    return f"{value:g}"


def _indent(element: ET.Element, level: int = 0) -> None:
    whitespace = "\n" + "    " * level
    child_whitespace = "\n" + "    " * (level + 1)
    if len(element):
        if not element.text or not element.text.strip():
            element.text = child_whitespace
        for child in element:
            _indent(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = child_whitespace
        element[-1].tail = whitespace


def _render_node(
    node: Dict[str, Any], resolver: _ResponseResolver, counter: list
) -> ET.Element:
    if not isinstance(node, dict):
        raise StructureRenderError("Every structure node must be a JSON object.")
    node_type = node.get("type")

    if node_type == "atomic":
        counter[0] += 1
        response = resolver.get(node, counter[0])
        generated = ET.fromstring(extract_android_xml(response))
        _set_size(generated, "match_parent", "match_parent")
        import os
        if os.environ.get("LAYOUTCODER_ANDROID_PROMPT", "v2") != "v1":
            # upstream: outermost fills the container with margin and padding 0 (w-full h-full)
            for key in list(generated.attrib):
                local = key.rsplit("}", 1)[-1]
                if local.startswith("layout_margin") or local.startswith("padding"):
                    del generated.attrib[key]
        wrapper = ET.Element("FrameLayout")
        _set_size(wrapper, "match_parent", "match_parent")
        wrapper.append(generated)
        return wrapper

    orientation = {"row": "horizontal", "column": "vertical", "col": "vertical"}.get(
        node_type
    )
    if orientation is None:
        raise StructureRenderError(
            f"Unsupported structure type {node_type!r}; expected row, column, or atomic."
        )
    children = node.get("value")
    if not isinstance(children, list) or not children:
        raise StructureRenderError(f"{node_type} node must contain a non-empty value list.")

    layout = ET.Element("LinearLayout")
    _set_size(layout, "match_parent", "match_parent")
    layout.set(A + "orientation", orientation)
    for child_node in children:
        child = _render_node(child_node, resolver, counter)
        if orientation == "horizontal":
            _set_size(child, "0dp", "match_parent")
        else:
            _set_size(child, "match_parent", "0dp")
        child.set(A + "layout_weight", _weight(child_node))
        layout.append(child)
    return layout


def render_structure(structure: Dict[str, Any], atomic_responses: Any) -> str:
    """Render a LayoutCoder row/column/atomic tree as one Android XML document."""

    if "structure" in structure and "type" not in structure:
        structure = structure["structure"]
    resolver = _ResponseResolver(atomic_responses)
    root = _render_node(structure, resolver, [0])
    _set_size(root, "match_parent", "match_parent")
    _indent(root)
    xml = ET.tostring(root, encoding="unicode", short_empty_elements=True)
    # Parsing here guards namespace and fusion regressions before writing output.
    ET.fromstring(xml)
    return xml + "\n"
