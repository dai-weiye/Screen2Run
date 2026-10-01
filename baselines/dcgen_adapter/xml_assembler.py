"""Deterministic Android FrameLayout assembly over an upstream bbox tree."""

import copy
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterator, Mapping, Sequence, Tuple, Union

from .config import ANDROID_NS, AdapterConfig, SEGMENT_ID_PREFIX
from .xml_parser import parse_android_fragment

ET.register_namespace("android", ANDROID_NS)
Tree = Dict[str, Any]
Fragment = Union[str, ET.Element]


def _attr(name: str) -> str:
    return f"{{{ANDROID_NS}}}{name}"


def _bbox(node: Mapping[str, Any]) -> Tuple[float, float, float, float]:
    value = node.get("bbox")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("every tree node must have bbox [left, top, right, bottom]")
    left, top, right, bottom = (float(item) for item in value)
    if right <= left or bottom <= top:
        raise ValueError(f"invalid bbox: {value}")
    return left, top, right, bottom


def _children(node: Mapping[str, Any]) -> list[Tree]:
    value = node.get("children", [])
    if not isinstance(value, list):
        raise ValueError("children must be a list")
    return value


def iter_preorder(tree: Tree) -> Iterator[Tuple[int, Tree]]:
    counter = 0

    def visit(node: Tree) -> Iterator[Tuple[int, Tree]]:
        nonlocal counter
        current = counter
        counter += 1
        yield current, node
        for child in _children(node):
            yield from visit(child)

    yield from visit(tree)


def count_leaves(tree: Tree) -> int:
    children = _children(tree)
    return 1 if not children else sum(count_leaves(child) for child in children)


def expected_call_count(tree_or_leaf_count: Union[Tree, int]) -> int:
    leaves = (
        tree_or_leaf_count
        if isinstance(tree_or_leaf_count, int)
        else count_leaves(tree_or_leaf_count)
    )
    if leaves < 1:
        raise ValueError("leaf count must be positive")
    return 2 * leaves + 1


def leaf_segment_ids(tree: Tree) -> list[str]:
    return [
        f"{SEGMENT_ID_PREFIX}{index}"
        for index, node in iter_preorder(tree)
        if not _children(node)
    ]


def _dp(px: float, px_per_dp: float) -> str:
    value = px / px_per_dp
    rounded = round(value)
    text = str(int(rounded)) if abs(value - rounded) < 1e-9 else f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{text}dp"


def _layout_attrs(
    element: ET.Element,
    bbox: Tuple[float, float, float, float],
    parent_bbox: Tuple[float, float, float, float] | None,
    segment_id: int,
    config: AdapterConfig,
) -> None:
    left, top, right, bottom = bbox
    element.set(_attr("id"), f"@+id/{SEGMENT_ID_PREFIX}{segment_id}")
    element.set(_attr("layout_width"), _dp(right - left, config.px_per_dp))
    element.set(_attr("layout_height"), _dp(bottom - top, config.px_per_dp))
    if parent_bbox is not None:
        element.set(_attr("layout_marginStart"), _dp(left - parent_bbox[0], config.px_per_dp))
        element.set(_attr("layout_marginTop"), _dp(top - parent_bbox[1], config.px_per_dp))


def assemble_android_xml(
    bbox_tree: Tree,
    leaf_fragments: Mapping[str, Fragment],
    config: AdapterConfig | None = None,
) -> str:
    """Build one namespace-owning FrameLayout root from selected leaf fragments."""

    config = config or AdapterConfig()
    available_ids = set(leaf_segment_ids(bbox_tree))
    if set(leaf_fragments) != available_ids:
        missing = sorted(available_ids - set(leaf_fragments))
        extra = sorted(set(leaf_fragments) - available_ids)
        raise ValueError(f"leaf fragment ids mismatch; missing={missing}, extra={extra}")
    ids = {id(node): index for index, node in iter_preorder(bbox_tree)}

    def build(node: Tree, parent_bbox: Tuple[float, float, float, float] | None) -> ET.Element:
        bbox = _bbox(node)
        segment_id = ids[id(node)]
        container = ET.Element("FrameLayout")
        _layout_attrs(container, bbox, parent_bbox, segment_id, config)
        children = _children(node)
        if children:
            for child in children:
                container.append(build(child, bbox))
        else:
            key = f"{SEGMENT_ID_PREFIX}{segment_id}"
            raw_fragment = leaf_fragments[key]
            fragment = (
                parse_android_fragment(raw_fragment, config.declared_resources)
                if isinstance(raw_fragment, str)
                else copy.deepcopy(raw_fragment)
            )
            fragment.set(_attr("layout_width"), fragment.get(_attr("layout_width"), "match_parent"))
            fragment.set(_attr("layout_height"), fragment.get(_attr("layout_height"), "match_parent"))
            container.append(fragment)
        return container

    root = build(bbox_tree, None)
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)
