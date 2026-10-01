"""Small Android-only compatibility repairs, applied identically to all arms.

No screenshots, ground truth, model calls, or screen identifiers are inputs.  These
repairs preserve native text and existing child geometry; they do not infer absent
string resource values or replace a failed screen by another screen.
"""
from __future__ import annotations

import html
import re
import xml.etree.ElementTree as ET

ANDROID = "http://schemas.android.com/apk/res/android"
_TEXT_ATTR = re.compile(
    r'''(\bandroid:(?:text|hint|contentDescription|textOn|textOff)\s*=\s*)(["'])(.*?)(\2)''',
    re.DOTALL,
)
_RESOURCE = re.compile(r"@(?:(?:[A-Za-z_][\w.]*):)?[A-Za-z_]\w*/[A-Za-z_]\w*\Z")
_THEME = re.compile(r"\?(?:(?:[A-Za-z_][\w.]*):)?(?:attr/)?[A-Za-z_]\w*\Z")


def escape_literal_text_prefixes(xml: str) -> tuple[str, int]:
    """Escape text such as ``?123`` that AAPT otherwise treats as a resource.

Explicit resource and theme references stay references. Existing escapes remain
byte-identical. Missing ``@string/name`` is deliberately NOT guessed from its name.
"""
    changed = 0

    def replace(m: re.Match) -> str:
        nonlocal changed
        raw = m.group(3)
        value = html.unescape(raw)
        if not value.startswith(("?", "@")):
            return m.group(0)
        if value in {"@null", "@empty"} or _RESOURCE.fullmatch(value) or _THEME.fullmatch(value):
            return m.group(0)
        changed += 1
        return m.group(1) + m.group(2) + "\\" + raw + m.group(4)

    # Avoid interpreting text inside comments as an attribute.
    parts = re.split(r"(<!--[\s\S]*?-->)", xml)
    for i in range(0, len(parts), 2):
        parts[i] = _TEXT_ATTR.sub(replace, parts[i])
    return "".join(parts), changed


def wrap_scroll_children(xml: str) -> tuple[str, int]:
    """Give an invalid multi-child ScrollView its required one FrameLayout child.

    ScrollView uses FrameLayout.LayoutParams already, so child sizes, margins,
    gravity, draw order, text, and IDs are retained. No vertical reflow is invented.
    Existing valid single-child scroll containers are byte-for-byte unchanged.
    """
    try:
        root = ET.fromstring(xml, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))
    except ET.ParseError:
        return xml, 0
    changed = 0
    for node in list(root.iter()):
        if not isinstance(node.tag, str):
            continue
        tag = node.tag.rsplit("}", 1)[-1].rsplit(".", 1)[-1]
        if tag not in {"ScrollView", "HorizontalScrollView", "NestedScrollView"}:
            continue
        views = [c for c in node if isinstance(c.tag, str)
                 and c.tag.rsplit("}", 1)[-1] not in {"requestFocus", "tag"}]
        if len(views) <= 1:
            continue
        horizontal = tag == "HorizontalScrollView"
        wrapper = ET.Element("FrameLayout", {
            f"{{{ANDROID}}}layout_width": "wrap_content" if horizontal else "match_parent",
            f"{{{ANDROID}}}layout_height": "match_parent" if horizontal else "wrap_content",
            f"{{{ANDROID}}}clipChildren": "false",
            f"{{{ANDROID}}}clipToPadding": "false",
        })
        position = list(node).index(views[0])
        for child in views:
            node.remove(child)
            wrapper.append(child)
        node.insert(position, wrapper)
        changed += 1
    if not changed:
        return xml, 0
    ET.register_namespace("android", ANDROID)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n", changed


def apply_android_compat(xml: str) -> tuple[str, dict[str, int]]:
    xml, literals = escape_literal_text_prefixes(xml)
    xml, scrolls = wrap_scroll_children(xml)
    return xml, {"escaped_text_literals": literals, "wrapped_scroll_containers": scrolls}


def restore_missing_text_resources(
    xml: str, tree: dict | list, existing_strings: set[str] | frozenset[str] = frozenset(),
) -> tuple[str, dict]:
    """Resolve only absent text/hint strings from same-arm, same-screen upstream S2.

    The caller supplies that screen's tree; there is no filesystem lookup or S5
    fallback. Both the source node_id and target Android id must be unique. Each
    field is restored from its exact same named field (hint is not inferred from
    text). Report every applied mapping and every unresolved reference.
    """
    from collections import Counter, defaultdict

    nodes = defaultdict(list)

    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get("node_id"), str):
                nodes[value["node_id"]].append(value)
            for child in value.values():
                if isinstance(child, (dict, list)):
                    walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(tree)
    root = ET.fromstring(xml, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=True)))

    def ident(node):
        return node.get(f"{{{ANDROID}}}id", "").rsplit("/", 1)[-1]

    ids = Counter(ident(n) for n in root.iter() if isinstance(n.tag, str))
    report = {"restored": [], "unresolved": []}
    for node in root.iter():
        if not isinstance(node.tag, str):
            continue
        node_id = ident(node)
        for field in ("text", "hint"):
            value = node.get(f"{{{ANDROID}}}{field}", "")
            m = re.fullmatch(r"@string/([A-Za-z_]\w*)", value)
            if not m or m.group(1) in existing_strings:
                continue
            evidence = {"node_id": node_id, "field": field, "resource": value}
            source = nodes.get(node_id, [])
            if not node_id or ids[node_id] != 1 or len(source) != 1:
                report["unresolved"].append({**evidence, "reason": "non_unique_node_id"})
                continue
            literal = source[0].get(field)
            if not isinstance(literal, str) or not literal or literal.startswith(("@", "?")):
                report["unresolved"].append({**evidence, "reason": "missing_upstream_literal"})
                continue
            node.set(f"{{{ANDROID}}}{field}", literal)
            report["restored"].append({**evidence, "source": "upstream_s2", "value": literal})
    if not report["restored"]:
        return xml, report
    ET.register_namespace("android", ANDROID)
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode") + "\n", report
