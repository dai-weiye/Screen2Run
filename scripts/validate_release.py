#!/usr/bin/env python3
"""Check release integrity, portable paths, English source text, and data shape."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED = {".git", ".venv", "__pycache__", ".pytest_cache", "work"}
TEXT_EXTENSIONS = {".py", ".swift", ".java", ".kt", ".md", ".txt", ".json", ".xml", ".html", ".sh", ".gradle", ".properties"}
CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
# Assemble patterns to avoid embedding a personal path or token-like literal.
PRIVATE_PATH = re.compile(r"(?<![A-Za-z0-9])/(?:" + "Users|home" + r")/(?!path(?:/|$))[^\s'\"<>]+")
SECRET = re.compile(r"\b" + "sk" + r"-[A-Za-z0-9]{24,}")


def files(root):
    return sorted(p for p in root.rglob("*") if p.is_file()
                  and not (set(p.relative_to(root).parts) & EXCLUDED)
                  and p.name != ".DS_Store" and p.suffix not in {".pyc", ".pyo"})


def check(root):
    errors = []
    entries = files(root)
    for path in entries:
        relative = path.relative_to(root).as_posix()
        if not relative.isascii():
            errors.append(f"Non-ASCII path: {relative}")
        if path.is_symlink():
            errors.append(f"Symlink must be materialized: {relative}")
        if path.suffix not in TEXT_EXTENSIONS and path.name not in {"LICENSE", "NOTICE", "gradlew", ".gitignore"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            errors.append(f"Invalid UTF-8: {relative}")
            continue
        if SECRET.search(text):
            errors.append(f"Possible credential: {relative}")
        if PRIVATE_PATH.search(text):
            errors.append(f"Machine-specific path: {relative}")
        research_text = relative.startswith(("data/element_measurements/", "data/human_study/code/"))
        if CJK.search(text) and not research_text:
            errors.append(f"Non-English authored text: {relative}")
        if path.suffix == ".py":
            try:
                tree = ast.parse(text, filename=relative)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.Name, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        name = getattr(node, "id", getattr(node, "name", ""))
                        if not name.isascii():
                            errors.append(f"Non-ASCII identifier: {relative}")
            except SyntaxError as exc:
                errors.append(f"Python syntax: {relative}:{exc.lineno}")
    manifest = root / "MANIFEST.json"
    if manifest.is_file():
        expected = json.loads(manifest.read_text())["files"]
        actual = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in entries if p != manifest}
        if set(expected) != set(actual):
            errors.append("Manifest member set differs")
        for name, checksum in expected.items():
            if actual.get(name) != checksum:
                errors.append(f"Checksum mismatch: {name}")
    else:
        errors.append("MANIFEST.json is missing")
    load = lambda path: json.loads((root / path).read_text())
    rows = load("data/results/rq1_rows.json")
    ablations = load("data/results/rq2_rows.json")
    for name, data in (("RQ1", rows), ("RQ2", ablations)):
        if len(data) != 3600 or len({(r["screen_id"], r["arm"]) for r in data}) != 3600:
            errors.append(f"{name}: expected 3600 unique paired rows")
    ratings = load("data/human_study/ratings.json")
    if len(ratings["records"]) != 2160:
        errors.append("RQ4: expected 2160 screen-method-rater records")
    return {"files_checked": len(entries), "manifest_verified": manifest.is_file() and not errors,
            "model_calls": 0, "errors": errors, "passed": not errors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = check(args.root.resolve())
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
