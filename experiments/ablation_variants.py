#!/usr/bin/env python3
"""Prepare the five published RQ2 variants without altering the frozen Full.

Planning/review/visual variants reuse their own recorded model-stage outputs.
The no-realization variant copies the original pre-execution endpoint. The
no-raster variant changes only actual raster drawable references in Full.
The bindonly command is solely the original content-guard fallback, NOT a
standalone execution-loop ablation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths

ARMS = ("no_planning", "no_review", "no_realization", "no_raster_output", "no_visual")
ANDROID = "{http://schemas.android.com/apk/res/android}"
DRAWABLE = re.compile(r"@drawable/([A-Za-z0-9_.]+)")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root, ids):
    result = {}
    for sid in ids:
        directory = root / "full" / sid
        files = [directory / "final.xml", *sorted((directory / "drawables").glob("*"))]
        result[sid] = {p.relative_to(directory).as_posix(): digest(p) for p in files if p.is_file()}
        if "final.xml" not in result[sid]:
            raise ValueError(f"Missing source layout: {sid}")
    return result


def fresh(out, sources):
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"Refusing existing output: {out}")
    for source in sources:
        source = source.resolve()
        if out == source or out.is_relative_to(source) or source.is_relative_to(out):
            raise ValueError("Candidate output and sources must be disjoint")
    out.mkdir(parents=True)
    return out


def native_projection(xml):
    root = ET.fromstring(xml)
    return [(node.tag, dict(node.attrib), node.text, node.tail, len(node)) for node in root.iter()]


def remove_raster_references(xml, rasters):
    changed = []
    def replace(match):
        if match[1] in rasters:
            changed.append(match[1])
            return "@drawable/img"
        return match[0]
    result = DRAWABLE.sub(replace, xml)
    before, after = native_projection(xml), native_projection(result)
    if len(before) != len(after):
        raise ValueError("Raster intervention changed hierarchy")
    for left, right in zip(before, after):
        if left[0] != right[0] or left[2:] != right[2:] or set(left[1]) != set(right[1]):
            raise ValueError("Raster intervention changed element content")
        for key, value in left[1].items():
            if value != right[1][key] and (key in {ANDROID + "text", ANDROID + "hint", ANDROID + "contentDescription"}
                    or value not in {f"@drawable/{name}" for name in rasters} or right[1][key] != "@drawable/img"):
                raise ValueError("Raster intervention changed native text or geometry")
    return result, changed


def terminal(args):
    from PIL import Image
    from run_image_filling import read_sids, finish_candidate
    from resource_sanitizer import ensure_placeholder
    ids = read_sids(args.sids)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Require a unique nonempty cohort")
    source = args.source.resolve()
    before = snapshot(source, ids)
    out = fresh(args.out, [source])
    for sid in ids:
        src, dst = source / "full" / sid, out / "full" / sid
        dst.mkdir(parents=True)
        original = (src / "final.xml").read_text(encoding="utf-8")
        changed = []
        if args.arm == "no_realization":
            shutil.copy2(src / "final.xml", dst / "final.xml")
            if (src / "drawables").is_dir():
                shutil.copytree(src / "drawables", dst / "drawables")
        else:
            rasters = set()
            for path in (src / "drawables").glob("*"):
                if path.suffix.lower() == ".xml":
                    ET.fromstring(path.read_bytes())
                    continue
                with Image.open(path) as image:
                    image.verify()
                name = path.name[:-6] if path.name.lower().endswith(".9.png") else path.stem
                if name != "img":
                    rasters.add(name)
            xml, changed = remove_raster_references(original, rasters)
            (dst / "final.xml").write_text(xml, encoding="utf-8")
            (dst / "drawables").mkdir()
            for path in (src / "drawables").glob("*.xml"):
                shutil.copy2(path, dst / "drawables" / path.name)
            ensure_placeholder(dst / "drawables")
        (dst / "board.json").write_text(json.dumps({"screen_id": sid, "ablation": args.arm,
            "source_files_sha256": before[sid], "replaced_rasters": changed,
            "intervention_scope": "joint_realization_module" if args.arm == "no_realization" else "nested_terminal_presentation"}, indent=2) + "\n")
    if snapshot(source, ids) != before:
        raise ValueError("Source changed during ablation preparation")
    finish_candidate(out, ids, f"RQ2: {args.arm}", model="recorded_source")
    return 0


def prepare(args):
    from prepare_ablation import prepare_candidate
    from run_image_filling import read_sids
    expected = {"no_planning": "no_s1", "no_visual": "no_s2", "no_review": None}[args.arm]
    for sid in read_sids(args.sids):
        path = args.gen / "full" / sid / "generation_evidence/generation_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("ablation") != expected:
            raise ValueError(f"Generation source is not the intended intervention: {args.arm}/{sid}")
    prepare_candidate(args.gen, args.sids, "s3" if args.arm == "no_review" else "s5", args.out)
    return 0


def cmd_bindonly(args):
    """Original S5-geometry crop fallback, used only by the content guard."""
    from PIL import Image
    from image_filling import aget, aset, id_name, local, parse_vh
    from run_image_filling import read_sids, shot_map, finish_candidate
    refs = shot_map()
    pre = args.pre.resolve()
    run = release_paths.RENDERS / f"tsc_{pre.name}_full"
    ids = read_sids(pre / "screenshots.txt")
    before = snapshot(pre, ids)
    out = fresh(args.out, [pre, run])
    for sid in ids:
        src, dst = pre / "full" / sid, out / "full" / sid
        hierarchy = run / "raw/hierarchy" / f"{sid}.xml"
        boxes = parse_vh(hierarchy)
        with Image.open(refs[sid]) as image:
            reference = image.convert("RGB")
        scale = reference.width / 1080.0
        root = ET.fromstring((src / "final.xml").read_text(encoding="utf-8"))
        (dst / "drawables").mkdir(parents=True)
        tag, count = hashlib.sha1(sid.encode()).hexdigest()[:6], 0
        for element in root.iter():
            name = id_name(element)
            attribute = "src" if local(element.tag) in ("ImageView", "ImageButton") else "background"
            if aget(element, attribute) != "@drawable/img" or name is None or name not in boxes:
                continue
            box = boxes[name]
            crop = [max(0, int(round(box[0] * scale))), max(0, int(round(box[1] * scale))),
                    min(reference.width, int(round(box[2] * scale))), min(reference.height, int(round(box[3] * scale)))]
            if crop[2] - crop[0] < 2 or crop[3] - crop[1] < 2:
                continue
            resource = f"abl{tag}_{count}"
            reference.crop(crop).save(dst / "drawables" / f"{resource}.png")
            aset(element, attribute, f"@drawable/{resource}")
            if attribute == "src":
                aset(element, "scaleType", "fitXY")
            count += 1
        xml = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")
        (dst / "final.xml").write_text(xml + "\n", encoding="utf-8")
        for path in (src / "drawables").glob("*"):
            if not (dst / "drawables" / path.name).exists():
                shutil.copy2(path, dst / "drawables" / path.name)
        (dst / "board.json").write_text(json.dumps({"screen_id": sid, "operation": "content_guard_fallback",
            "is_independent_ablation": False, "crops_at_model_geometry": count, "source_files_sha256": before[sid]}) + "\n")
    if snapshot(pre, ids) != before:
        raise ValueError("Pre-execution source changed during fallback construction")
    finish_candidate(out, ids, "Content-guard fallback", model="recorded_source")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Same-run S3/S5 preparation with source-hash checks")
    p.add_argument("--arm", choices=("no_planning", "no_review", "no_visual"), required=True)
    p.add_argument("--gen", type=Path, required=True)
    p.add_argument("--sids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("terminal", help="Copy the pre endpoint or remove terminal raster references")
    p.add_argument("--arm", choices=("no_realization", "no_raster_output"), required=True)
    p.add_argument("--source", type=Path, required=True, help="Original pre endpoint for no_realization; frozen Full for no_raster_output")
    p.add_argument("--sids", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("bindonly", help="Content-guard fallback ONLY, not an RQ2 intervention")
    p.add_argument("--pre", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"prepare": prepare, "terminal": terminal, "bindonly": cmd_bindonly}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
