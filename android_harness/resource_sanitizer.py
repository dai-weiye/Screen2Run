#!/usr/bin/env python3
"Shared Android resource compatibility rules. Preserve public framework drawables; resolve inaccessible resources to a placeholder. Run before occlusion repair."
from __future__ import annotations
import argparse, re, shutil, sys
from pathlib import Path
import sys
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)

REPO = release_paths.RELEASE
SRC = release_paths.CANDIDATES / "fse_baselines"
DST = release_paths.CANDIDATES / "fse_baselines_sanitized"
PLACEHOLDER = "img"

# See docs/RUNTIME.md for the shared compatibility and capture contracts.
KEEP_LOCAL = re.compile(r"^(img|guigpt[A-Za-z0-9_]*)$")
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
PRIVATE_ANDROID = re.compile(r"@android:drawable/([A-Za-z0-9_]+)")
_PUBLIC_XML = release_paths.ANDROID_SDK / "platforms" / "android-35" / "data" / "res" / "values" / "public-final.xml"


def _public_drawables() -> frozenset:
    if not _PUBLIC_XML.is_file():
        raise FileNotFoundError(f"public resource list missing: {_PUBLIC_XML}")
    text = _PUBLIC_XML.read_text(encoding="utf-8", errors="ignore")
    return frozenset(re.findall(r'<public\s+type="drawable"\s+name="([A-Za-z0-9_]+)"', text))


PUBLIC_DRAWABLES = None  # Loaded only when compatibility processing is requested.
DANGLING_LOCAL = re.compile(r"@drawable/([A-Za-z0-9_.]+)")

# ---------------------------------------------------------------------------
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
#
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
#
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
_DIMEN = re.compile(r"-?\d+(?:\.\d+)?(?:dp|dip|sp|px|pt|mm|in)\Z")
_COLOR = re.compile(r"#(?:[0-9A-Fa-f]{3}|[0-9A-Fa-f]{4}|[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})\Z")
_REF = re.compile(r"[@?][A-Za-z0-9_./:+]+\Z")
_DIMEN_ENUM = ("match_parent", "fill_parent", "wrap_content")
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
DIMEN_ATTRS = ("layout_width", "layout_height", "layout_margin", "layout_marginStart",
               "layout_marginEnd", "layout_marginTop", "layout_marginBottom", "layout_marginLeft",
               "layout_marginRight", "layout_padding", "padding", "paddingStart", "paddingEnd",
               "paddingTop", "paddingBottom", "paddingLeft", "paddingRight", "textSize",
               "minWidth", "minHeight", "maxWidth", "maxHeight", "lineSpacingExtra")
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
COLOR_ATTRS = ("background", "textColor", "textColorHint", "tint", "backgroundTint", "shadowColor")
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
REF_ATTRS = ("thumb", "track", "prompt", "src", "srcCompat", "drawableStart", "drawableEnd",
             "drawableLeft", "drawableRight", "drawableTop", "drawableBottom", "button")
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
ATTR_RENAME = {"allCaps": "textAllCaps"}


def _attr_ok(name: str, value: str) -> bool:
    v = value.strip()
    if name in ATTR_RENAME:
        return v.lower() in ("true", "false")
    if name in DIMEN_ATTRS:
        return v in _DIMEN_ENUM or bool(_REF.match(v)) or bool(_DIMEN.match(v))
    if name in COLOR_ATTRS:
        return bool(_COLOR.match(v)) or bool(_REF.match(v))
    if name in REF_ATTRS:
        return bool(_REF.match(v))
    return True


ATTR_RE = re.compile(r'\s+android:([A-Za-z][A-Za-z0-9_]*)\s*=\s*"([^"]*)"')


def neutralize_invalid_attrs(xml: str) -> tuple[str, int]:
    "Drop invalid Android attribute values; preserve all remaining XML."
    n = 0

    def fix_element(m):
        nonlocal n
        head, attrs = m.group(1), m.group(2)

        def one(am):
            nonlocal n
            name, val = am.group(1), am.group(2)
            if name in ATTR_RENAME and _attr_ok(name, val):
                return f' android:{ATTR_RENAME[name]}="{val}"'
            if _attr_ok(name, val):
                return am.group(0)
            n += 1
            return ""
        new = ATTR_RE.sub(one, attrs)
        return head + new + ("/>" if m.group(3) else ">")

    out = re.sub(r"(<[A-Za-z][A-Za-z0-9_.]*)((?:\s+[A-Za-z:][^>]*?)?)(/?)>", fix_element, xml)
    from xml_compat import apply_android_compat
    out, compat = apply_android_compat(out)
    return out, n + sum(compat.values())


# See docs/RUNTIME.md for the shared compatibility and capture contracts.
_START_TAG = re.compile(r'<([A-Za-z_][A-Za-z0-9_.:-]*)'
                        r'((?:\s+[A-Za-z_:][-A-Za-z0-9_:.]*\s*=\s*(?:"[^"]*"|\'[^\']*\'))*)(\s*)(/?)>')
_NOT_A_VIEW = {"merge", "include", "requestFocus", "tag", "shape", "solid", "stroke", "corners",
               "gradient", "padding", "size", "item", "selector", "layer-list", "vector", "path",
               "group", "clip-path", "inset", "ripple", "bitmap", "nine-patch", "set"}


def fill_missing_layout_dims(xml: str) -> tuple[str, int]:
    "Add wrap_content only for absent layout dimensions."
    if "xmlns:android=" not in xml:
        return xml, 0
    n = 0

    def fix(m):
        nonlocal n
        tag, attrs = m.group(1), m.group(2)
        # a local @style/ may carry the dimensions; framework widget styles (?android:attr/..., @android:style/...) never do
        if tag.split(":")[-1] in _NOT_A_VIEW or re.search(r'\sstyle\s*=\s*"@style/', attrs):
            return m.group(0)
        add = ""
        for dim in ("layout_width", "layout_height"):
            if not re.search(rf"\sandroid:{dim}\s*=", attrs):
                add += f' android:{dim}="wrap_content"'
        if not add:
            return m.group(0)
        n += 1
        return f"<{tag}{attrs}{add}{m.group(3)}{m.group(4)}>"

    return _START_TAG.sub(fix, xml), n


def sanitize(xml: str) -> tuple[str, int, int, int]:
    global PUBLIC_DRAWABLES
    if PUBLIC_DRAWABLES is None:
        PUBLIC_DRAWABLES = _public_drawables()
    n_priv = 0
    n_local = 0

    def repl_priv(m):
        nonlocal n_priv
        if m.group(1) in PUBLIC_DRAWABLES:
            return m.group(0)
        n_priv += 1
        return f"@drawable/{PLACEHOLDER}"

    def repl_local(m):
        nonlocal n_local
        if KEEP_LOCAL.match(m.group(1)):
            return m.group(0)
        n_local += 1
        return f"@drawable/{PLACEHOLDER}"

    out = PRIVATE_ANDROID.sub(repl_priv, xml)
    out = DANGLING_LOCAL.sub(repl_local, out)
    out, n_attr = neutralize_invalid_attrs(out)
    return out, n_priv, n_local, n_attr


def ensure_placeholder(drawables: Path) -> None:
    drawables.mkdir(parents=True, exist_ok=True)
    p = drawables / f"{PLACEHOLDER}.png"
    if not p.is_file():
        from PIL import Image
        Image.new("RGB", (64, 64), (200, 200, 200)).save(p)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for arm in args.arm:
        src = SRC / arm
        if not src.is_dir():
            print(f"!! missing {src}"); continue
        dst = DST / arm
        screens = sorted(d for d in src.iterdir() if (d / "final.xml").is_file())
        if args.limit:
            screens = screens[:args.limit]
        tot_p = tot_l = tot_a = changed = 0
        for d in screens:
            text = (d / "final.xml").read_text(encoding="utf-8", errors="ignore")
            out, np_, nl, na = sanitize(text)
            tot_p += np_; tot_l += nl; tot_a += na
            if np_ or nl or na:
                changed += 1
            if args.dry_run:
                continue
            target = dst / d.name
            target.mkdir(parents=True, exist_ok=True)
            (target / "final.xml").write_text(out, encoding="utf-8")
            for sub in ("drawables", "res/drawable", "resources"):
                s = d / sub
                if s.is_dir():
                    t = target / sub
                    if not t.exists():
                        shutil.copytree(s, t)
            ensure_placeholder(target / "drawables")
        print(f"{arm:<20} screens {len(screens):>4}; changed {changed:>4} screens "
              f"({changed/max(1,len(screens)):>4.0%})   private resources {tot_p:>4}   "
              f"unresolved resources {tot_l:>4}; invalid attributes {tot_a:>4}")
        if not args.dry_run:
            print(f"{'':<20} -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
