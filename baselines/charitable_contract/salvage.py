"""Charitable Android XML extraction contract for the external baselines.

Both Android adaptations reject any response that violates the strict Android
output contract. That rejection is a property of the adaptation rather than of
the published method, so this module implements a deterministic salvage pass
that reads a rejected response as favourably as the recorded text allows.

The ruleset is frozen in
``ESE_replication/protocol/amendment_2026-07-29_matched_sota_and_pipeline_stability.md``.
It never adds content the model did not emit, never reads the reference
screenshot, makes no model call, and is idempotent. A salvaged layout still has
to pass the same strict Android validation as every other arm.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional


ANDROID_NS = "http://schemas.android.com/apk/res/android"
DRAWABLE_PLACEHOLDER = "@drawable/img"

_FENCE_RE = re.compile(r"```(?:[A-Za-z0-9_-]+)?\s*(.*?)```", re.S)
_TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w.:-]*)((?:[^<>\"']|\"[^\"]*\"|'[^']*')*?)(/?)\s*>", re.S)
_ATTR_RE = re.compile(r"([A-Za-z_:][-\w:.]*)\s*=\s*(\"[^\"]*\"|'[^']*')")
_RESOURCE_RE = re.compile(r"^@(?:\+)?([A-Za-z_][\w]*)/([A-Za-z_][\w.]*)$")
_URL_RE = re.compile(r"(?:https?:)?//", re.I)
_LENGTH_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(px|dp|dip|sp|pt|rem|em|%)?\s*$", re.I)
_COLOR_RE = re.compile(r"^#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_BARE_AMPERSAND_RE = re.compile(r"&(?!(?:amp|lt|gt|quot|apos|#\d+|#x[0-9a-fA-F]+);)")
_REDUNDANT_DELIMITER_RE = re.compile(
    r"""(=\s*(["'])[^"']*\2)\s*\2(\s*(?:/?>|[A-Za-z_][-\w:.]*\s*=))"""
)
# A backslash-escaped quote has no single reading, so it is resolved only behind the
# guard in ``salvage_fragment``: where the escaped quotes open and close a phrase
# inside a value, resolving them extends the value past its real end.
_BACKSLASH_DELIMITER_RE = re.compile(r"""\\(["'])(?=[^\s/>])""")
_XML_ENTITY_FOR = {'"': "&quot;", "'": "&apos;"}

# HTML elements with an unambiguous Android counterpart.
TAG_MAP: Dict[str, str] = {
    "button": "Button",
    "div": "LinearLayout",
    "section": "LinearLayout",
    "main": "LinearLayout",
    "header": "LinearLayout",
    "footer": "LinearLayout",
    "nav": "LinearLayout",
    "aside": "LinearLayout",
    "article": "LinearLayout",
    "form": "LinearLayout",
    "ul": "LinearLayout",
    "ol": "LinearLayout",
    "table": "LinearLayout",
    "tr": "LinearLayout",
    "tbody": "LinearLayout",
    "thead": "LinearLayout",
    "span": "TextView",
    "p": "TextView",
    "label": "TextView",
    "li": "TextView",
    "td": "TextView",
    "th": "TextView",
    "a": "TextView",
    "strong": "TextView",
    "em": "TextView",
    "small": "TextView",
    "h1": "TextView",
    "h2": "TextView",
    "h3": "TextView",
    "h4": "TextView",
    "h5": "TextView",
    "h6": "TextView",
    "img": "ImageView",
    "image": "ImageView",
    "picture": "ImageView",
    "input": "EditText",
    "textarea": "EditText",
    "select": "Spinner",
    "hr": "View",
    "br": "View",
}

_VERTICAL_CONTAINERS = {"LinearLayout"}

# Chat/gateway chrome around an already-emitted Android tree. These tags are not
# Views; dropping them does not add markup the model did not emit.
_GATEWAY_BLOCK_RE = re.compile(
    r"<(thinking_mode|reasoning_effort|think|thinking)\b[^>]*>.*?</\1\s*>",
    re.I | re.S,
)
_CHAT_WRAPPER_RE = re.compile(
    r"\A<(answer|response|result|output)\b[^>]*>(.*)</\1\s*>\Z",
    re.I | re.S,
)

# HTML/CSS attributes with an Android counterpart.
_ATTR_MAP = {
    "width": "layout_width",
    "height": "layout_height",
    "background-color": "background",
    "background": "background",
    "color": "textColor",
    "padding": "padding",
    "font-size": "textSize",
    "text-align": "gravity",
    "alt": "contentDescription",
    "placeholder": "hint",
    "value": "text",
    "id": None,
    "class": None,
    "style": None,
    "href": None,
    "src": "src",
    "type": None,
    "onclick": None,
}
_GRAVITY_MAP = {"left": "start", "right": "end", "center": "center", "justify": "start"}


def _largest_fenced_block(response: str) -> str:
    blocks = [block.strip() for block in _FENCE_RE.findall(response) if "<" in block]
    if blocks:
        return max(blocks, key=len)
    return response.strip()


def _strip_prose(text: str) -> str:
    start = text.find("<")
    end = text.rfind(">")
    return text[start : end + 1] if 0 <= start < end else text


def _drop_chat_chrome(text: str) -> str:
    """Remove gateway metadata and unwrap a chat envelope around a View tree.

    Relay endpoints wrap the model XML in ``<answer>`` or emit
    ``<reasoning_effort>`` / ``<thinking_mode>`` as sibling roots. The View
    subtree is already in the text; keeping the envelope makes the strict
    Android validator reject a layout the baseline actually produced.
    """

    text = _GATEWAY_BLOCK_RE.sub("", text).strip()
    match = _CHAT_WRAPPER_RE.match(text)
    if match:
        inner = match.group(2).strip()
        if "<" in inner:
            text = inner
    return text.strip()


_CHAT_WRAPPER_TAGS = {"answer", "response", "result", "output"}


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _looks_like_android_view(element: ET.Element) -> bool:
    name = _local_tag(element.tag)
    class_name = name.rsplit(".", 1)[-1]
    return bool(class_name) and class_name[0].isupper()


def _unwrap_chat_root(root: ET.Element) -> ET.Element:
    """Prefer the Android View inside a chat/gateway wrapper root."""

    for _ in range(4):
        local = _local_tag(root.tag).lower()
        kids = list(root)
        if local not in _CHAT_WRAPPER_TAGS:
            return root
        views = [child for child in kids if _looks_like_android_view(child)]
        if len(views) == 1:
            root = views[0]
            continue
        if views:
            root = max(views, key=lambda element: len(ET.tostring(element)))
            continue
        return root
    return root


def _to_dp(value: str) -> Optional[str]:
    match = _LENGTH_RE.match(value)
    if not match:
        lowered = value.strip().lower()
        if lowered in {"match_parent", "wrap_content", "fill_parent"}:
            return lowered
        if lowered == "auto":
            return "wrap_content"
        return None
    number, unit = match.group(1), (match.group(2) or "px").lower()
    if unit == "%":
        return "match_parent" if float(number) >= 90 else "wrap_content"
    if unit in {"dp", "dip"}:
        return f"{number}dp"
    if unit == "sp":
        return f"{number}sp"
    if unit in {"rem", "em"}:
        return f"{float(number) * 16:.0f}dp"
    return f"{number}dp"


def _rewrite_attribute(name: str, value: str) -> Optional[tuple[str, str]]:
    """Return an Android attribute for an HTML/CSS attribute, or None to drop it."""

    plain = name.split(":", 1)[-1].lower()
    if name.lower().startswith("android:") or ":" in name:
        return (name, value)
    target = _ATTR_MAP.get(plain, "__unmapped__")
    if target is None:
        return None
    if target == "__unmapped__":
        return None
    if target in {"layout_width", "layout_height"}:
        converted = _to_dp(value)
        return (f"android:{target}", converted) if converted else None
    if target in {"background", "textColor"}:
        return (f"android:{target}", value) if _COLOR_RE.match(value.strip()) else None
    if target in {"padding", "textSize"}:
        converted = _to_dp(value)
        return (f"android:{target}", converted) if converted else None
    if target == "gravity":
        mapped = _GRAVITY_MAP.get(value.strip().lower())
        return (f"android:{target}", mapped) if mapped else None
    if target == "src":
        return (f"android:{target}", DRAWABLE_PLACEHOLDER)
    return (f"android:{target}", value)


def _rewrite_tags(text: str) -> str:
    """Map HTML element names to Android view names and normalise attributes."""

    def replace(match: re.Match[str]) -> str:
        closing, tag, attrs, self_closing = match.groups()
        local = tag.rsplit(":", 1)[-1].lower()
        mapped = TAG_MAP.get(local)
        name = mapped or tag
        if closing:
            return f"</{name}>"

        seen: Dict[str, str] = {}
        for attr_match in _ATTR_RE.finditer(attrs or ""):
            raw_name = attr_match.group(1)
            raw_value = attr_match.group(2)[1:-1]
            if mapped is None:
                if raw_name in seen:
                    continue
                if _URL_RE.search(raw_value) and raw_name.lower().endswith(("src", "srccompat")):
                    seen[raw_name] = DRAWABLE_PLACEHOLDER
                else:
                    seen[raw_name] = raw_value
                continue
            rewritten = _rewrite_attribute(raw_name, raw_value)
            if rewritten is None:
                continue
            key, value = rewritten
            if key not in seen:
                seen[key] = value

        if mapped is not None:
            seen.setdefault("android:layout_width", "wrap_content")
            seen.setdefault("android:layout_height", "wrap_content")
            if mapped in _VERTICAL_CONTAINERS:
                seen.setdefault("android:orientation", "vertical")
        rendered = "".join(f' {key}="{_escape_attribute_value(value)}"' for key, value in seen.items())
        suffix = "/" if self_closing else ""
        return f"<{name}{rendered}{suffix}>"

    return _TAG_RE.sub(replace, text)


def _drop_redundant_delimiter(text: str) -> str:
    """Remove a duplicated attribute delimiter immediately before a tag close.

    A model that writes ``android:gravity="center"">`` has emitted one delimiter too
    many. The stray quote makes the tag pattern's quoted-string alternation lose
    phase, so the element is never matched and every later rule skips it, exactly as
    with the escaping defect above. Our own arms recover from this because the
    compiler names it and a repair round follows.

    The pattern requires a complete ``name="value"`` before the stray delimiter and
    either a tag close or the next attribute name after it, which is what keeps a
    legitimately empty value safe: in ``x="" y="z"`` the delimiter that follows the
    empty value is the one that opens it, and no second delimiter follows. Nothing is
    added and no character of any value is changed.

    Only the tag-close position occurs anywhere in the recorded corpus; the
    next-attribute position is covered so that the rule is stated by defect class
    rather than by the position we happened to observe, and it changes no output.
    """

    return _REDUNDANT_DELIMITER_RE.sub(r"\1\3", text)


def _escape_attribute_value(value: str) -> str:
    """Make ``value`` safe to serialise inside double quotes.

    Screen text legitimately contains quotation marks and inch marks, and a model
    that writes ``android:text='Lenovo N22 11.6" HD Chromebook'`` has produced
    perfectly valid XML by choosing the other delimiter. Tag rewriting below
    re-emits every attribute with double quotes, so without this escape our own
    contract turns that valid response into a document no parser accepts, and the
    baseline is charged for a defect we introduced. Found by
    ``scripts/ese/audit_extraction_asymmetry.py``.

    Escaping is idempotent: an ampersand that already begins an entity reference is
    left alone, so re-running the contract on its own output is a no-op.
    """

    value = _BARE_AMPERSAND_RE.sub("&amp;", value)
    return value.replace('"', "&quot;").replace("<", "&lt;")


def _close_truncated(text: str) -> str:
    """Cut an incomplete tail and close the elements the model left open."""

    stack: List[str] = []
    cut_at = None
    for match in _TAG_RE.finditer(text):
        closing, tag, _attrs, self_closing = match.groups()
        if closing:
            while stack and stack[-1] != tag:
                stack.pop()
            if stack:
                stack.pop()
        elif not self_closing:
            stack.append(tag)
        cut_at = match.end()
    if not stack:
        return text
    body = text[:cut_at] if cut_at else text
    return body + "".join(f"</{tag}>" for tag in reversed(stack))


def _drop_namespace_declarations(text: str) -> str:
    return re.sub(r"\s+xmlns(?::[A-Za-z_][\w.-]*)?\s*=\s*(\"[^\"]*\"|'[^']*')", "", text)


def _bind_android_namespace(text: str) -> str:
    match = _TAG_RE.search(text)
    if not match or match.group(1):
        return text
    if re.search(r"xmlns:android\s*=", match.group(0)):
        return text
    if not re.search(r"\bandroid:", text):
        return text
    insert_at = text.index(match.group(2), match.start()) + len(match.group(2))
    return f'{text[:insert_at]} xmlns:android="{ANDROID_NS}"{text[insert_at:]}'


def _resolve_resources(root: ET.Element, declared: set[str]) -> None:
    for element in root.iter():
        for name in list(element.attrib):
            value = element.attrib[name]
            if _URL_RE.search(value):
                if name.rsplit("}", 1)[-1] in {"src", "srcCompat", "background"}:
                    element.attrib[name] = DRAWABLE_PLACEHOLDER
                else:
                    del element.attrib[name]
                continue
            if value.startswith("?") and not value.startswith("?android:"):
                if value not in declared:
                    del element.attrib[name]
                continue
            if not value.startswith("@") or value.startswith("@android:"):
                continue
            match = _RESOURCE_RE.match(value)
            if not match:
                del element.attrib[name]
                continue
            resource_type = match.group(1)
            if value.startswith("@+id/") or value in declared:
                continue
            if resource_type in {"drawable", "mipmap"}:
                element.attrib[name] = DRAWABLE_PLACEHOLDER
            elif resource_type == "id":
                continue
            else:
                del element.attrib[name]


def salvage_fragment(
    response: str,
    *,
    declared_resources: Optional[set[str]] = None,
    strip_namespaces: bool = False,
) -> str:
    """Return the most favourable Android XML reading of ``response``.

    ``strip_namespaces`` is used for a leaf fragment whose assembler owns the
    namespace declarations. The return value is serialised XML text; the caller
    still applies its own strict validation.

    Where the plain reading yields nothing a parser accepts, one alternative reading
    is tried under a guard. It resolves backslash-escaped quotes, which a model
    borrows from C-family string literals and XML does not have, and it is returned
    only when it parses and the plain reading did not. The guard is what makes the
    alternative safe to attempt at all: applied unconditionally it shortens values
    whose escaped quotes open and close a phrase inside the value, and it then costs
    more than it recovers.
    """

    if not isinstance(response, str) or not response.strip():
        return response if isinstance(response, str) else ""

    plain = _salvage_once(
        response, declared_resources=declared_resources, strip_namespaces=strip_namespaces
    )
    if _accepted(plain) or not _BACKSLASH_DELIMITER_RE.search(response):
        return plain
    alternative = _salvage_once(
        _BACKSLASH_DELIMITER_RE.sub(lambda m: _XML_ENTITY_FOR[m.group(1)], response),
        declared_resources=declared_resources,
        strip_namespaces=strip_namespaces,
    )
    return alternative if _accepted(alternative) else plain


def _accepted(text: str) -> bool:
    """Whether a parser accepts this text under the convention the callers use.

    A fragment returned without namespace declarations is wrapped in a bound root
    before it is parsed downstream, so both forms are tried here.
    """

    if not text.strip():
        return False
    for candidate in (text, f'<merge xmlns:android="{ANDROID_NS}">{text}</merge>'):
        try:
            root = ET.fromstring(candidate)
        except ET.ParseError:
            continue
        if any(True for _ in root.iter()):
            return True
    return False


def _salvage_once(
    response: str,
    *,
    declared_resources: Optional[set[str]] = None,
    strip_namespaces: bool = False,
) -> str:
    text = _strip_prose(_largest_fenced_block(response))
    text = re.sub(r"<\?xml[^>]*\?>", "", text, flags=re.I).strip()
    text = re.sub(r"<!DOCTYPE[^>]*>", "", text, flags=re.I).strip()
    text = _drop_chat_chrome(text)
    # Must precede tag rewriting: a stray delimiter throws the tag pattern's
    # quoted-string alternation out of phase, so the element is never matched.
    text = _drop_redundant_delimiter(text)
    text = _rewrite_tags(text)
    text = _close_truncated(text)
    # The prefix must be bound before parsing even when the caller owns the
    # declarations. Dropping them first leaves ``android:`` unbound, the parse
    # then fails, and the resource rule below is skipped for every leaf
    # fragment; declarations are removed after serialisation instead.
    text = _bind_android_namespace(text)

    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return _drop_namespace_declarations(text) if strip_namespaces else text

    root = _unwrap_chat_root(root)
    _resolve_resources(root, set(declared_resources or ()))
    ET.register_namespace("android", ANDROID_NS)
    salvaged = ET.tostring(root, encoding="unicode")
    if strip_namespaces:
        salvaged = _drop_namespace_declarations(salvaged)
    return salvaged
