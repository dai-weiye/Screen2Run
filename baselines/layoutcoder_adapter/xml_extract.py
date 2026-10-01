"""Strict extraction of one Android XML subtree from model text.

Setting ``ANDROID_EXTRACTION_CONTRACT=charitable`` switches to the frozen
salvage contract of
``ESE_replication/protocol/amendment_2026-07-29_matched_sota_and_pipeline_stability.md``,
which reads a rejected response as favourably as its text allows before the
checks below are applied. The default is unchanged.
"""

import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from charitable_contract.salvage import salvage_fragment  # noqa: E402


ANDROID_NS = "http://schemas.android.com/apk/res/android"
ET.register_namespace("android", ANDROID_NS)


class XMLExtractionError(ValueError):
    """Raised when a response violates the local Android XML contract."""


_TAG_RE = re.compile(
    r"<\s*(/?)\s*([A-Za-z_][\w.:-]*)\b[^>]*?(/?)\s*>", re.DOTALL
)
_IGNORED_RE = re.compile(r"<!--.*?-->|<!\[CDATA\[.*?\]\]>", re.DOTALL)
_HTML_TAGS = {
    "html", "head", "body", "div", "span", "script", "style", "img", "p",
    "a", "section", "main", "header", "footer", "button", "input",
}


def _visible_for_scanning(text: str) -> str:
    return _IGNORED_RE.sub(lambda match: " " * len(match.group(0)), text)


def _candidate_subtrees(text: str) -> List[str]:
    visible = _visible_for_scanning(text)
    stack: List[str] = []
    start = None
    candidates: List[str] = []

    for match in _TAG_RE.finditer(visible):
        closing, tag, self_closing = match.groups()
        if not stack:
            if closing:
                continue
            start = match.start()
            if self_closing:
                candidates.append(text[start:match.end()])
                start = None
            else:
                stack.append(tag)
            continue

        if closing:
            if tag != stack[-1]:
                raise XMLExtractionError(
                    f"Malformed XML: closing tag </{tag}> does not match <{stack[-1]}>."
                )
            stack.pop()
            if not stack and start is not None:
                candidates.append(text[start:match.end()])
                start = None
        elif not self_closing:
            stack.append(tag)

    if stack:
        raise XMLExtractionError(f"Malformed XML: unclosed <{stack[-1]}> element.")
    return candidates


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _validate_android_tree(root: ET.Element) -> None:
    # HTML tags from the original LayoutCoder are lowercase. Android view class
    # names are PascalCase, including Button / Input collisions with HTML.
    for element in root.iter():
        raw = _local_name(element.tag)
        if raw.lower() in _HTML_TAGS and raw == raw.lower():
            raise XMLExtractionError(
                f"HTML element <{raw}> is not Android XML."
            )
    root_name = _local_name(root.tag)
    class_name = root_name.rsplit(".", 1)[-1]
    if not class_name or not class_name[0].isupper():
        raise XMLExtractionError(
            f"Root <{root_name}> is not an Android View element."
        )


def charitable_contract_enabled() -> bool:
    return os.environ.get("ANDROID_EXTRACTION_CONTRACT", "strict").lower() == "charitable"


def extract_android_xml(response: str) -> str:
    """Return exactly one parseable Android XML subtree from a response."""

    if charitable_contract_enabled():
        response = salvage_fragment(response)
    if not isinstance(response, str) or not response.strip():
        raise XMLExtractionError("Response is empty; expected one Android XML subtree.")
    if re.search(r"<\?xml\b", response, re.IGNORECASE):
        raise XMLExtractionError("XML declarations are forbidden in atomic responses.")
    if re.search(r"@Composable\b|\bModifier\.|\bsetContent\s*\(", response):
        raise XMLExtractionError("Jetpack Compose code is forbidden; use Android Views XML.")

    candidates = _candidate_subtrees(response)
    parsed: List[Tuple[str, ET.Element]] = []
    parse_errors: List[str] = []
    for candidate in candidates:
        try:
            parsed.append((candidate.strip(), ET.fromstring(candidate)))
        except ET.ParseError as exc:
            parse_errors.append(str(exc))

    if not parsed:
        detail = f" ({parse_errors[0]})" if parse_errors else ""
        raise XMLExtractionError(
            f"No complete, parseable Android XML subtree found{detail}."
        )
    if len(parsed) != 1:
        raise XMLExtractionError(
            f"Expected exactly one Android XML root, found {len(parsed)}."
        )

    candidate, root = parsed[0]
    _validate_android_tree(root)
    return candidate
