"""Framework attribute whitelist shared by every arm's compile-level cleanup.

AAPT rejects the whole layout for one ``android:`` attribute the platform does not define
(``android:selected``, ``android:allCaps``), so a single invented attribute turns a screen
into a deployment failure.  The same string-level drop is applied to our system and to
every baseline; values and every other attribute are untouched.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO = release_paths.RELEASE
PUBLIC = release_paths.ANDROID_SDK / "platforms" / "android-35" / "data" / "res" / "values"
_ATTR = re.compile(r'(\s+)android:([A-Za-z_][A-Za-z0-9_]*)\s*=\s*("[^"]*"|\'[^\']*\')')


@lru_cache(maxsize=1)
def public_attrs() -> frozenset:
    names = set()
    for name in ("public-final.xml", "public-staging.xml"):
        path = PUBLIC / name
        if path.is_file():
            names.update(re.findall(r'<public\s+type="attr"\s+name="([A-Za-z0-9_]+)"',
                                    path.read_text(encoding="utf-8", errors="ignore")))
    if len(names) < 1000:
        raise FileNotFoundError(f"framework attribute list incomplete under {PUBLIC}")
    return frozenset(names)


def drop_unknown_android_attrs(xml: str) -> tuple[str, list[str]]:
    known = public_attrs()
    dropped: list[str] = []

    def repl(m):
        if m.group(2) in known:
            return m.group(0)
        dropped.append(m.group(2))
        return ""

    return _ATTR.sub(repl, xml), dropped
