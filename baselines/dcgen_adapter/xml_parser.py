"""Strict extraction and validation of model-produced Android XML fragments.

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
from typing import Iterable, Optional

from .config import ANDROID_NS

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from charitable_contract.salvage import salvage_fragment  # noqa: E402


def charitable_contract_enabled() -> bool:
    return os.environ.get("ANDROID_EXTRACTION_CONTRACT", "strict").lower() == "charitable"


_ATTR_NAME_RE = re.compile(r'<attr\s+name="([\w.]+)"')
_PLATFORM_ATTRS: set[str] | None = None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def platform_android_attributes() -> set[str]:
    """Attribute names the compile SDK resource linker knows. Empty if SDK is absent."""

    global _PLATFORM_ATTRS
    if _PLATFORM_ATTRS is not None:
        return _PLATFORM_ATTRS
    import release_paths
    sdk = release_paths.ANDROID_SDK
    platforms = sdk / "platforms"
    roots: list[Path] = []
    if platforms.is_dir():
        roots = sorted(platforms.glob("android-*"), reverse=True)
    known: set[str] = set()
    for platform in roots:
        values = platform / "data" / "res" / "values"
        if not values.is_dir():
            continue
        for path in values.glob("*.xml"):
            known |= set(_ATTR_NAME_RE.findall(path.read_text(encoding="utf-8", errors="replace")))
        if len(known) >= 1000:
            break
    _PLATFORM_ATTRS = known
    return known


def drop_unknown_android_attributes(root: ET.Element) -> list[str]:
    """Remove android: names AAPT cannot link. Does not add or rename attributes."""

    known = platform_android_attributes()
    if len(known) < 1000:
        return []
    ns = f"{{{ANDROID_NS}}}"
    removed: list[str] = []
    for element in root.iter():
        for name in list(element.attrib):
            if name.startswith(ns):
                local = name[len(ns) :]
            elif name.startswith("android:"):
                local = name.split(":", 1)[1]
            else:
                continue
            if local not in known:
                del element.attrib[name]
                removed.append(local)
    return removed

_FENCE_RE = re.compile(r"```(?:xml|android|android-xml)?\s*(.*?)```", re.I | re.S)
_RESOURCE_RE = re.compile(r"^@(?:\+)?([A-Za-z_][\w]*)/([A-Za-z_][\w.]*)$")
_FORBIDDEN_TEXT_RE = re.compile(
    r"<!DOCTYPE|<html\b|<body\b|<head\b|<script\b|<style\b|"
    r"\b(?:@Composable|setContent\s*\(|Modifier\.|androidx\.compose)\b",
    re.I,
)


class AndroidXMLParseError(ValueError):
    """Raised when a response is not a safe, single Android View fragment."""


def _extract(response: str) -> str:
    if not isinstance(response, str) or not response.strip():
        raise AndroidXMLParseError("response is empty")
    fences = _FENCE_RE.findall(response)
    if fences:
        if len(fences) != 1:
            raise AndroidXMLParseError("response must contain exactly one fenced fragment")
        outside = _FENCE_RE.sub("", response).strip()
        if "```" in outside or re.search(r"<\s*/?[A-Za-z]", outside):
            raise AndroidXMLParseError("response contains XML outside the single fence")
        return fences[0].strip()
    value = response.strip()
    if not value.startswith("<") or not value.endswith(">"):
        raise AndroidXMLParseError("bare response must contain XML only")
    return value


def _validate_resources(root: ET.Element, declared_resources: set[str]) -> None:
    declared_ids = {
        value.replace("@+id/", "@id/", 1)
        for element in root.iter()
        for value in element.attrib.values()
        if value.startswith("@+id/")
    }
    for element in root.iter():
        for value in element.attrib.values():
            if re.search(r"(?:https?:)?//", value, re.I):
                raise AndroidXMLParseError("network URLs are forbidden")
            if value.startswith("?") and not value.startswith("?android:"):
                if value not in declared_resources:
                    raise AndroidXMLParseError(f"undeclared custom attribute: {value}")
            if not value.startswith("@"):
                continue
            if value.startswith("@android:"):
                continue
            match = _RESOURCE_RE.match(value)
            if not match:
                raise AndroidXMLParseError(f"invalid resource reference: {value}")
            resource_type = match.group(1)
            normalized = value.replace("@+id/", "@id/", 1)
            if value.startswith("@+id/") or normalized in declared_ids:
                continue
            if resource_type == "drawable" and value != "@drawable/img":
                raise AndroidXMLParseError("images must use @drawable/img")
            if value not in declared_resources:
                raise AndroidXMLParseError(f"undeclared custom resource: {value}")


def parse_android_fragment(
    response: str, declared_resources: Optional[Iterable[str]] = None
) -> ET.Element:
    """Extract one leaf fragment; namespace declarations belong to the assembler."""

    if charitable_contract_enabled():
        response = salvage_fragment(
            response, declared_resources=set(declared_resources or ()), strip_namespaces=True
        )
    fragment = _extract(response)
    if re.search(r"<\?xml\b", fragment, re.I):
        raise AndroidXMLParseError("XML declarations are forbidden in leaf fragments")
    if re.search(r"\sxmlns(?::\w+)?\s*=", fragment, re.I):
        raise AndroidXMLParseError("leaf returned a complete document/namespace root")
    if _FORBIDDEN_TEXT_RE.search(fragment):
        raise AndroidXMLParseError("HTML, scripts, doctypes, and Compose are forbidden")
    if re.search(r"(?:https?:)?//", fragment, re.I):
        raise AndroidXMLParseError("network URLs are forbidden")
    wrapped = (
        f'<dcgen-wrapper xmlns:android="{ANDROID_NS}">'
        f"{fragment}</dcgen-wrapper>"
    )
    try:
        wrapper = ET.fromstring(wrapped)
    except ET.ParseError as exc:
        raise AndroidXMLParseError(f"invalid Android XML: {exc}") from exc
    if len(wrapper) != 1:
        raise AndroidXMLParseError("fragment must have exactly one root View")
    root = wrapper[0]
    if root.tag.lower().split("}")[-1] in {
        "html",
        "body",
        "head",
        "layout",
        "manifest",
        "resources",
    }:
        raise AndroidXMLParseError("complete documents and HTML are forbidden")
    allowed = set(declared_resources or ())
    allowed.add("@drawable/img")
    _validate_resources(root, allowed)
    if charitable_contract_enabled():
        drop_unknown_android_attributes(root)
    return root


def parse_android_document(
    response: str, declared_resources: Optional[Iterable[str]] = None
) -> ET.Element:
    """Validate root-refine output as one complete Android layout document."""

    if charitable_contract_enabled():
        response = salvage_fragment(response, declared_resources=set(declared_resources or ()))
    document = _extract(response)
    if _FORBIDDEN_TEXT_RE.search(document):
        raise AndroidXMLParseError("HTML, scripts, doctypes, and Compose are forbidden")
    if re.search(r"(?:https?:)?//", re.sub(r'xmlns:android="[^"]+"', "", document), re.I):
        raise AndroidXMLParseError("network URLs are forbidden")
    try:
        root = ET.fromstring(document)
    except ET.ParseError as exc:
        raise AndroidXMLParseError(f"invalid Android XML document: {exc}") from exc
    if root.tag.split("}")[-1] != "FrameLayout":
        raise AndroidXMLParseError("root-refine document must have one FrameLayout root")
    allowed = set(declared_resources or ())
    allowed.add("@drawable/img")
    _validate_resources(root, allowed)
    if charitable_contract_enabled():
        drop_unknown_android_attributes(root)
    return root
