#!/usr/bin/env python3
"""Apply the shared deterministic Android compatibility rules to any baseline.

Processing preserves the original XML as generated.xml. A fresh prepared
candidate uses the same full/<screen_id>/final.xml contract as the renderer.
No model calls, grounding, or image-asset inference are performed here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths
from attribute_filter import drop_unknown_android_attrs
from occlusion_repair import repair
from resource_sanitizer import ensure_placeholder, fill_missing_layout_dims, sanitize
from run_image_filling import finish_candidate, read_sids

ANDROID_ROOT = re.compile(r"<[A-Za-z][\w.]*\s[^<>]*xmlns:android=")


def parses(xml: str) -> bool:
    text = xml.strip()
    if not text:
        return False
    try:
        ET.fromstring(text.split("?>", 1)[-1] if text.startswith("<?xml") else text)
        return True
    except ET.ParseError:
        return False


def strip_leading_prose(xml: str) -> tuple[str, int]:
    if parses(xml):
        return xml, 0
    root = ANDROID_ROOT.search(xml)
    for start in (xml.find("<?xml"), root.start() if root else -1):
        if start > 0 and parses(xml[start:]):
            return xml[start:], 1
    return xml, 0


def post(xml: str) -> tuple[str, dict]:
    xml, n_prose = strip_leading_prose(xml)
    out, dropped = drop_unknown_android_attrs(xml)
    out, n_priv, n_local, n_attr = sanitize(out)
    out, n_occ = repair(out)
    out, n_dims = fill_missing_layout_dims(out)
    return out, {"unknown_attrs": len(dropped), "private_drawables": n_priv, "dangling_local": n_local,
                 "invalid_attrs": n_attr, "occluding_leaves": n_occ, "missing_layout_dims": n_dims,
                 "leading_prose": n_prose}


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def copy_resources(src_dir: Path, dst_dir: Path) -> None:
    for sub in ("drawables", "res/drawable", "resources"):
        source = src_dir / sub
        if source.is_dir() and not (dst_dir / sub).exists():
            shutil.copytree(source, dst_dir / sub)
    ensure_placeholder(dst_dir / "drawables")


def cmd_prepare(args) -> int:
    source, out = args.source.resolve(), args.out.resolve()
    if out.exists() or out == source or out.is_relative_to(source) or source.is_relative_to(out):
        raise FileExistsError("Use a fresh output directory disjoint from the generation root")
    ids = read_sids(args.cohort)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Cohort must contain unique screenshot IDs")
    missing = [sid for sid in ids if not (source / "full" / sid / "final.xml").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing baseline XML for {len(missing)} screens: {missing}")
    os.environ["SCREEN2RUN_SCREENSHOT_LIST"] = str(args.cohort.resolve())
    report = []
    for sid in ids:
        original = source / "full" / sid
        destination = out / "full" / sid
        destination.mkdir(parents=True)
        raw = original / "final.xml"
        shutil.copy2(raw, destination / "generated.xml")
        xml, counts = post(raw.read_text(encoding="utf-8", errors="ignore"))
        (destination / "final.xml").write_text(xml, encoding="utf-8")
        copy_resources(original, destination)
        report.append({"screen_id": sid, "source_sha256": sha(raw.read_bytes()),
                       "final_sha256": sha(xml.encode("utf-8")), **counts})
    finish_candidate(out, ids, "shared baseline compatibility")
    (out / "compatibility_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Prepared {len(ids)} baseline screens in {out}")
    return 0


def cmd_post(args) -> int:
    total = {}
    for directory in sorted((args.cand / "full").iterdir()):
        final = directory / "final.xml"
        if not final.is_file():
            continue
        raw = directory / "generated.xml"
        if not raw.is_file():
            shutil.copy2(final, raw)
        xml, counts = post(raw.read_text(encoding="utf-8", errors="ignore"))
        final.write_text(xml, encoding="utf-8")
        ensure_placeholder(directory / "drawables")
        for key, value in counts.items():
            total[key] = total.get(key, 0) + value
    print(args.cand, total)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    prepare = sub.add_parser("prepare", help="Copy a fixed cohort into a fresh render candidate")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--cohort", type=Path, required=True)
    prepare.add_argument("--out", type=Path, required=True)
    post_parser = sub.add_parser("post", help="Process an existing candidate, retaining generated.xml")
    post_parser.add_argument("--cand", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": cmd_prepare, "post": cmd_post}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
