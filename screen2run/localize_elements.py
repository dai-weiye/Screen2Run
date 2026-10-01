#!/usr/bin/env python3
"""Measure screenshot elements with UIED text/non-text detection and merging.

The separately installed LayoutCoder checkout supplies UIED. Its relation,
layout, division, and generation stages are disabled. Each output records image
dimensions, text/component classes, and pixel bounds for downstream grounding.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths
REPO_ROOT = release_paths.RELEASE
DETECTOR_ROOT = release_paths.LAYOUTCODER
DETECTOR_PYTHON = Path(os.environ.get("SCREEN2RUN_UIED_PYTHON", sys.executable))

# The detector runs under its own interpreter, because its OCR stack pins versions this
# repository does not otherwise carry. Driving it as a subprocess keeps that dependency out
# of the analysis environment entirely.
_DRIVER = r"""
import json, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
sys.path.insert(0, str(root / "UIED"))
from run_single import ui2code_pipeline

image, work = sys.argv[2], sys.argv[3]
ui2code_pipeline(
    input_path_img=image,
    output_root=work,
    is_uied=True,
    is_lines=False,
    is_layout=False,
    is_divide=False,
    is_global_gen=False,
)
merged = Path(work) / "merge" / (Path(image).stem + ".json")
print("MERGED:" + str(merged))
"""


def detect(image: Path, work_dir: Path) -> dict:
    """Return UIED's merged detections for one screenshot.

    Paths are resolved before the call, because the detector runs with its own working
    directory and a relative path handed to it silently reads as nothing: OpenCV returns
    None rather than raising, and the failure surfaces several frames later as a missing
    attribute on the image it never loaded.
    """
    image, work_dir = image.resolve(), work_dir.resolve()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(_DRIVER)
        driver = handle.name
    try:
        env = dict(os.environ, PYTHONWARNINGS="ignore")
        proc = subprocess.run(
            [str(DETECTOR_PYTHON), driver, str(DETECTOR_ROOT), str(image), str(work_dir)],
            capture_output=True, text=True, env=env, cwd=str(DETECTOR_ROOT),
        )
    finally:
        os.unlink(driver)
    line = next((l for l in proc.stdout.splitlines() if l.startswith("MERGED:")), None)
    if line is None:
        raise RuntimeError(f"detector produced no merge file for {image.name}:\n{proc.stdout[-800:]}\n{proc.stderr[-800:]}")
    payload = json.loads(Path(line[len("MERGED:"):]).read_text(encoding="utf-8"))
    shape = payload.get("img_shape") or []
    elements = []
    for compo in payload.get("compos", []):
        pos = compo.get("position") or {}
        if not pos:
            continue
        elements.append({
            "id": compo.get("id"),
            # UIED reports "Text" for OCR-detected runs and "Compo" for everything else. The
            # distinction is kept because it is a measurement, and dropped nowhere downstream.
            "kind": "text" if compo.get("class") == "Text" else "component",
            "bounds": [pos["column_min"], pos["row_min"], pos["column_max"], pos["row_max"]],
        })
    # Reading order. A layout is written top to bottom and left to right, so the elements are
    # handed over in that order rather than in the detector's internal one.
    elements.sort(key=lambda e: (e["bounds"][1], e["bounds"][0]))
    for index, element in enumerate(elements):
        element["id"] = index
    return {
        "image": str(image),
        "width": int(shape[1]) if len(shape) > 1 else None,
        "height": int(shape[0]) if shape else None,
        "detector": "UIED (Xie et al. 2020), detection stage only",
        "elements": elements,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screens", nargs="*", default=[], help="Screen ids (Reference/{id}.png).")
    parser.add_argument("--image-list", type=Path, help="Text file: one PNG path per line.")
    parser.add_argument("--reference-dir", type=Path,
                        default=release_paths.SCREENSHOTS)
    parser.add_argument("--out-dir", type=Path,
                        default=release_paths.MEASUREMENTS)
    parser.add_argument("--work-dir", type=Path, default=release_paths.WORK / "localization")
    args = parser.parse_args()

    if not DETECTOR_PYTHON.is_file():
        print(f"detector interpreter missing at {DETECTOR_PYTHON}", file=sys.stderr)
        return 1
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[str, Path]] = []
    if args.image_list:
        for line in args.image_list.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            image = release_paths.resolve_screenshot(line).resolve()
            jobs.append((image.stem, image))
    for screen in args.screens:
        jobs.append((screen, args.reference_dir / f"{screen}.png"))
    if not jobs:
        print("provide --screens and/or --image-list", file=sys.stderr)
        return 2

    counts = []
    for stem, image in jobs:
        target = args.out_dir / stem / "measured_elements.json"
        if target.is_file():
            counts.append((stem, len(json.loads(target.read_text())["elements"]), "cached"))
            continue
        if not image.is_file():
            print(f"  {stem}: image not found at {image}", file=sys.stderr)
            continue
        record = detect(image, args.work_dir / stem)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        counts.append((stem, len(record["elements"]), "measured"))
        print(f"  {stem}: {len(record['elements'])} elements", flush=True)

    if counts:
        totals = sorted(n for _s, n, _w in counts)
        print(f"\n{len(counts)} screens; median {totals[len(totals) // 2]} elements, "
              f"range {totals[0]}-{totals[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
