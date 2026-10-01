#!/usr/bin/env python3
"Shared compatibility rule for opaque, parent-filling leaves. Text, colors, and alignment attributes are preserved."
from __future__ import annotations
import argparse, re
from pathlib import Path
import sys
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)

SANITIZED = release_paths.CANDIDATES / "baselines_sanitized"
LEAF = ("TextView", "ImageView", "Button", "EditText", "ImageButton",
        "ProgressBar", "CheckBox", "Switch", "RadioButton", "View", "SeekBar", "RatingBar")
ELEM = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9_.]*)\b([^>]*?)(/?)>", re.S)


def repair(xml: str) -> tuple[str, int]:
    "Repair non-root opaque leaves that fill their parent in both dimensions."
    out = []
    pos = 0
    stack: list[list] = []          # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    fixed = 0
    for m in ELEM.finditer(xml):
        closing, tag, attrs, selfclose = m.group(1), m.group(2), m.group(3), m.group(4)
        if closing:
            if stack:
                stack.pop()
            continue
        if stack:
            stack[-1][1] += 1        # See docs/RUNTIME.md for the shared compatibility and capture contracts.
        stack.append([tag, 0])
        if selfclose:
            stack.pop()              # See docs/RUNTIME.md for the shared compatibility and capture contracts.
        if tag not in LEAF:
            continue
        has_bg = re.search(r'android:background="(#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?|@(?:color|drawable)/[^"]+)"', attrs)
        opaque = bool(has_bg) and not (has_bg.group(1).startswith("#") and len(has_bg.group(1)) == 9
                                       and not has_bg.group(1)[1:3].lower() == "ff")
        if not opaque:
            continue
        both = 'android:layout_width="match_parent"' in attrs and 'android:layout_height="match_parent"' in attrs
        if not both:
            continue
        # See docs/RUNTIME.md for the shared compatibility and capture contracts.
        if len(stack) <= 1:
            continue
        new = attrs.replace('android:layout_width="match_parent"', 'android:layout_width="wrap_content"')
        new = new.replace('android:layout_height="match_parent"', 'android:layout_height="wrap_content"')
        out.append(xml[pos:m.start(3)]); out.append(new); pos = m.end(3)
        fixed += 1
    out.append(xml[pos:])
    return "".join(out), fixed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=SANITIZED)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    grand = 0
    for arm in sorted(args.root.iterdir()):
        if not arm.is_dir():
            continue
        n = hit = 0
        for f in sorted(arm.glob("*/final.xml")):
            t = f.read_text(encoding="utf-8", errors="ignore")
            new, k = repair(t)
            n += 1
            if k:
                hit += 1
                grand += k
                if not args.dry_run:
                    f.write_text(new, encoding="utf-8")
        print(f"{arm.name:<22} screens {n:>4}; repaired {hit:>4} screens ({hit/max(1,n):>4.0%})")
    print(f"\nRepaired {grand} occluding nodes -> {args.root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
