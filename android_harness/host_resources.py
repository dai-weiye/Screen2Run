#!/usr/bin/env python3
"""Turn TSC arm outputs into final-harness runs (build → install → launch → foreground).

Every arm under <output-root>/<arm>/<screen>/final.xml becomes a manifest-driven run
under work/renders/tsc_<matrix>_<arm>/ that scripts/device_session.py can
validate. Drawable resources are collected from the arm's per-screen ``drawables/``
directory (and any adapter resource directory); ``@drawable/img`` is satisfied by the
shared placeholder ``ESE_replication/resources/img.xml`` exactly as in the frozen
ESE harness, so every arm meets the same contract.

  python3 host_resources.py --output-root work/candidates/smoke20 \
      --screenshot-list data/screen_lists/smoke20.txt --run-harness
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
EMPIRICAL = release_paths.RELEASE
REPO = release_paths.RELEASE
RUNS = release_paths.RENDERS
HARNESS = release_paths.module_file("device_session.py")
PLACEHOLDER = release_paths.PLACEHOLDER_DRAWABLE
DRAWABLE_REF_RE = re.compile(r"@drawable/([A-Za-z0-9_]+)")
SKIP_DIRS = {"work", "run", "cache"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO))
    except ValueError:
        return str(path.resolve())


def screenshot_map(list_path: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for line in list_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            path = release_paths.resolve_screenshot(line)
            mapping[path.stem] = path
    return mapping


def find_resource(screen_dir: Path, name: str) -> Path | None:
    preferred = screen_dir / "drawables"
    for ext in ("png", "xml", "jpg", "webp"):
        candidate = preferred / f"{name}.{ext}"
        if candidate.is_file():
            return candidate
    for candidate in screen_dir.rglob(f"{name}.*"):
        if candidate.suffix.lower() in {".png", ".xml", ".jpg", ".webp"} and "drawable" in str(candidate.parent).lower():
            return candidate
    return None


def materialize_arm(arm_dir: Path, matrix: str, screenshots: dict[str, Path], model: str) -> Path:
    run_id = f"tsc_{matrix}_{arm_dir.name}"
    run_dir = RUNS / run_id
    resources_root = run_dir / "resources"
    resources_root.mkdir(parents=True, exist_ok=True)
    placeholder_dst = resources_root / "img.xml"
    if not placeholder_dst.is_file():
        shutil.copyfile(PLACEHOLDER, placeholder_dst)
    items: list[dict[str, Any]] = []
    summary_path = arm_dir / "run_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    screen_ids = [str(s.get("screen_id")) for s in summary.get("screens") or [] if s.get("screen_id")]
    seen = set(screen_ids)
    for board_file in sorted(arm_dir.glob("*/board.json")):
        screen_id = board_file.parent.name
        if screen_id in SKIP_DIRS or screen_id in seen:
            continue
        screen_ids.append(screen_id)
        seen.add(screen_id)
    if not screen_ids:
        screen_ids = sorted(p.name for p in arm_dir.iterdir() if p.is_dir() and p.name not in SKIP_DIRS)
    if screenshots:
        screen_ids = [s for s in screen_ids if s in screenshots]
    for screen_id in screen_ids:
        screen_dir = arm_dir / screen_id
        final = screen_dir / "final.xml"
        input_png = screenshots.get(screen_id)
        item: dict[str, Any] = {
            "run_id": run_id,
            "screen_id": screen_id,
            "arm": arm_dir.name,
            "xml_path": rel(final) if final.is_file() else "",
            "xml_sha256": sha256_file(final) if final.is_file() else "",
            "input_png": rel(input_png) if input_png is not None else "",
            "input_png_sha256": sha256_file(input_png) if input_png is not None and input_png.is_file() else "",
            "resource_files": [],
        }
        if final.is_file():
            xml_text = final.read_text(encoding="utf-8", errors="replace")
            names = sorted(set(DRAWABLE_REF_RE.findall(xml_text)))
            for name in names:
                if name == "img":
                    item["resource_files"].append({"path": rel(placeholder_dst), "sha256": sha256_file(placeholder_dst)})
                    continue
                found = find_resource(screen_dir, name)
                if found is not None:
                    item["resource_files"].append({"path": rel(found), "sha256": sha256_file(found)})
        item["resources_sha256"] = hashlib.sha256(
            "\n".join(sorted(f["sha256"] for f in item["resource_files"])).encode("utf-8")
        ).hexdigest()
        items.append(item)
    (run_dir / "items.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in items), encoding="utf-8"
    )
    manifest = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "dataset": matrix,
        "method": arm_dir.name,
        "model": summary.get("model", model),
        "label": summary.get("label", ""),
        "derived_from": summary.get("derived_from"),
        "item_file": "items.jsonl",
        "resources_root": "resources",
        "screen_count": len(items),
        "source_output_root": rel(arm_dir),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return run_dir


def run_harness(run_dir: Path, args: argparse.Namespace) -> int:
    cmd = [
        sys.executable,
        "-u",
        str(HARNESS),
        "--run-dir",
        str(run_dir),
        "--project",
        str(args.project),
        "--android-home",
        str(args.android_home),
        "--device-serial",
        args.device_serial,
        "--no-snapshot-restore",
    ]
    if getattr(args, "overwrite_harness", False):
        cmd.append("--overwrite")
    else:
        cmd.append("--resume-status")
    env = dict(os.environ)
    env.setdefault("GRADLE_OPTS", "-Xmx1024m -XX:MaxMetaspaceSize=512m")
    return subprocess.run(cmd, check=False, env=env).returncode


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--screenshot-list", type=Path, required=True)
    parser.add_argument("--arms", nargs="*", help="Default: every arm directory with a run_summary.json")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--run-harness", action="store_true")
    parser.add_argument("--project", type=Path, default=release_paths.HOST_APP)
    parser.add_argument("--android-home", type=Path, default=release_paths.ANDROID_SDK)
    parser.add_argument("--device-serial", default="emulator-5554")
    parser.add_argument(
        "--resume-harness",
        action="store_true",
        help="Default. Keep recorded outcomes whose xml_sha256 still matches; recapture the rest.",
    )
    parser.add_argument(
        "--overwrite-harness",
        action="store_true",
        help="Replace loading_status.jsonl and recapture every screen.",
    )
    args = parser.parse_args()

    screenshots = screenshot_map(args.screenshot_list)
    matrix = args.output_root.name
    arm_dirs = [args.output_root / arm for arm in args.arms] if args.arms else sorted(
        p for p in args.output_root.iterdir() if p.is_dir() and (p / "run_summary.json").is_file()
    )
    ledger = []
    for arm_dir in arm_dirs:
        run_dir = materialize_arm(arm_dir, matrix, screenshots, args.model)
        entry: dict[str, Any] = {"arm": arm_dir.name, "run_dir": rel(run_dir)}
        print(f"materialized {arm_dir.name} -> {run_dir}", flush=True)
        if args.run_harness:
            clip = run_dir / "derived" / "clip_i.json"
            if clip.is_file():
                clip.unlink()
            entry["harness_exit"] = run_harness(run_dir, args)
            status = run_dir / "derived" / "loading_status.jsonl"
            entry["loading_status"] = rel(status) if status.is_file() else None
        ledger.append(entry)
    (args.output_root / "harness_ledger.json").write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
