#!/usr/bin/env python3
"""Assemble the frozen screenshot cohort from independently obtained datasets."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def beneath(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest paths must be relative and contain no parent traversal")
    root = root.resolve()
    target = (root / relative).resolve()
    if not target.is_relative_to(root):
        raise ValueError("Manifest path escapes the selected root")
    return target


def locate(root, member):
    if root is None:
        raise ValueError("Missing input dataset directory")
    member = Path(member)
    candidates = [beneath(root, member)]
    if member.parts and root.name == member.parts[0]:
        candidates.append(beneath(root, Path(*member.parts[1:])))
    existing = [p for p in candidates if p.is_file()]
    if len(existing) != 1:
        raise FileNotFoundError(f"Expected exactly one source for {member} beneath {root}")
    return existing[0]


def prepare(args):
    from PIL import Image
    manifest = json.loads(args.manifest.read_text())
    completed, failures = [], []
    for item in manifest["screens"]:
        try:
            target = beneath(args.out, item["file"])
            if target.is_file():
                raw = target.read_bytes()
            elif args.verify_only:
                raise FileNotFoundError(item["file"])
            else:
                source = item["upstream"]
                if item["split"] in ("Easy", "Real"):
                    source_root = args.pix2code_root if item["split"] == "Easy" else args.redraw_root
                    raw = locate(source_root, source["member"]).read_bytes()
                else:
                    archive = locate(args.mobileviews_root, source["shard"])
                    with zipfile.ZipFile(archive) as zf:
                        jpeg = zf.read(source["member"])
                    if sha(jpeg) != source["jpeg_sha256"]:
                        raise ValueError("Upstream JPEG checksum mismatch")
                    stream = io.BytesIO()
                    with Image.open(io.BytesIO(jpeg)) as image:
                        image.convert("RGB").save(stream, format="PNG")
                    raw = stream.getvalue()
            if sha(raw) != item["sha256"]:
                raise ValueError("Prepared PNG checksum mismatch; existing files are never replaced")
            with Image.open(io.BytesIO(raw)) as image:
                if image.size != (item["width"], item["height"]):
                    raise ValueError("Screenshot dimensions do not match")
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as fh:
                    fh.write(raw)
            completed.append(item["screen_id"])
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            failures.append({"screen_id": item["screen_id"], "error": str(exc)})
    print(json.dumps({"verified": len(completed), "expected": len(manifest["screens"]), "failures": failures}, indent=2))
    return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/screen_lists/screens.json")
    parser.add_argument("--pix2code-root", type=Path)
    parser.add_argument("--redraw-root", type=Path)
    parser.add_argument("--mobileviews-root", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "data/screenshots")
    parser.add_argument("--verify-only", action="store_true")
    return prepare(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
