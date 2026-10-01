#!/usr/bin/env python3
"""Collect captured image pairs from an explicit experiment manifest.

No score is assigned to an absent file or an environment failure. Current XML,
drawable resources, and PNG must agree with a successful capture receipt.
The output is input to evaluation/screenshot_metrics.py --pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths


def candidate_path(candidate):
    path = Path(candidate)
    return path if path.is_absolute() else release_paths.CANDIDATES / path


def render_dir(candidate):
    return release_paths.RENDERS / f"tsc_{candidate_path(candidate).name}_full"


def receipt_rows(run):
    # The serial collector and batch workers use two receipt naming schemes.
    derived = Path(run) / "derived"
    paths = set(derived.glob("loading_status_*.jsonl"))
    if (derived / "loading_status.jsonl").is_file():
        paths.add(derived / "loading_status.jsonl")
    seen = set()
    for path in sorted(paths):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            identity = json.dumps(row, sort_keys=True, separators=(",", ":"))
            if identity in seen:
                continue
            seen.add(identity)
            yield row


def successes(candidate):
    """Legacy content-guard compatibility: successful current-layout hashes."""
    result = {}
    for row in receipt_rows(render_dir(candidate)):
        if row.get("status") == "render_success" and row.get("xml_sha256"):
            result.setdefault(row["screen_id"], set()).add(row["xml_sha256"])
    return result


def current_capture(candidate, sid, ok):
    """Read-only lookup retained for the unchanged content-guard algorithm."""
    xml = candidate_path(candidate) / "full" / sid / "final.xml"
    png = render_dir(candidate) / "raw/screenshots" / f"{sid}.png"
    if xml.is_file() and png.is_file() and hashlib.sha256(xml.read_bytes()).hexdigest() in ok.get(sid, set()):
        return png
    return None


def resolve(base, name):
    path = Path(name)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def collect(manifest, base):
    if manifest.get("schema") != "screen2run-experiment/1":
        raise ValueError("Unknown experiment manifest schema")
    cohort, methods = manifest["cohort"], manifest["methods"]
    ids = [row["screen_id"] for row in cohort]
    if not ids or len(ids) != len(set(ids)) or not methods:
        raise ValueError("Empty or duplicate experiment cohort")
    from batch_render import screen_resource_provenance
    pairs, missing = [], []
    for arm, definition in methods.items():
        candidate = resolve(base, definition["candidate"])
        run = resolve(base, definition["render_dir"])
        receipts = {}
        for row in receipt_rows(run):
            if row.get("status") == "render_success":
                receipts.setdefault(row["screen_id"], []).append(row)
        for item in cohort:
            sid = item["screen_id"]
            ref = resolve(base, item["reference"])
            xml = candidate / "full" / sid / "final.xml"
            png = run / "raw/screenshots" / f"{sid}.png"
            if not all(path.is_file() for path in (ref, xml, png)):
                missing.append({"arm": arm, "screen_id": sid, "reason": "missing_input_or_capture"})
                continue
            xhash, phash = (hashlib.sha256(path.read_bytes()).hexdigest() for path in (xml, png))
            resources = screen_resource_provenance(xml.parent)
            valid = [r for r in receipts.get(sid, []) if r.get("xml_sha256") == xhash and r.get("png_sha256") == phash
                     and r.get("resources_sha256") == resources["resources_sha256"]
                     and r.get("resource_file_sha256") == resources["resource_file_sha256"]]
            if not valid:
                missing.append({"arm": arm, "screen_id": sid, "reason": "no_matching_XML_resource_PNG_receipt"})
                continue
            pairs.append({"arm": arm, "screen_id": sid, "split": item["split"], "status": "render_success",
                          "reference": str(ref), "render_path": str(png),
                          "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(), "render_path_sha256": phash,
                          "xml_sha256": xhash, "resources_sha256": resources["resources_sha256"]})
    return pairs, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    pairs, missing = collect(manifest, args.manifest.resolve().parent)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "capture_audit.json").write_text(json.dumps({"complete": not missing, "pairs": len(pairs), "missing": missing}, indent=2) + "\n")
    target = args.out / ("partial_pairs.json" if missing else "pairs.json")
    target.write_text(json.dumps(pairs, indent=2) + "\n")
    print(f"Verified {len(pairs)} captured pairs; pending {len(missing)}")
    return 2 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
