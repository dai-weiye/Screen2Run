#!/usr/bin/env python3
"""Execution-feedback content guard for a declared candidate and cohort.

After the execution loop, the system reads its own render back.  A reference text line (Vision OCR of
the input screenshot, status band excluded) counts as reproduced when a render shows the same text.  If
the loop's render reproduces a clearly smaller share of the reference text than the pre-render of the
same S5 XML did (drop >= DROP, fixed a priori), the loop's geometric edits are rolled back and the arm
ships its bind-only XML (S5 geometry, reference crops bound), as the loop exists to add fidelity and a
loop output that loses the input's text has failed its own purpose.  Applied to every arm that contains
the loop; uses only the input screenshot and the arm's own renders, never the evaluation code.

  python3 content_guard.py --pre PATH --grounded PATH --fallback PATH --cohort LIST [--apply]
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO = release_paths.RELEASE
from collect_renders import current_capture, successes  # noqa: E402
from image_filling import load_ocr  # noqa: E402
from run_image_filling import read_sids, shot_map  # noqa: E402

TM = release_paths.CANDIDATES
DROP = 0.25
MASK_TOP = 0.045


def norm(t: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]", "", t.lower())


def ocr_texts(png: Path) -> list[str]:
    with Image.open(png) as im:
        w, h = im.size
    return [norm(l["text"]) for l in load_ocr(png, w, h) if l["box"][1] >= MASK_TOP * h]


def recall(want: list[str], png: Path) -> float:
    have = ocr_texts(png)
    blob = "".join(have)
    hit = sum(1 for t in want if t in blob or any(difflib.SequenceMatcher(None, t, x).ratio() >= 0.8 for x in have))
    return hit / len(want)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pre", type=Path, required=True)
    ap.add_argument("--grounded", type=Path, required=True)
    ap.add_argument("--fallback", type=Path, required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    pre, v1, bind = args.pre.resolve(), args.grounded.resolve(), args.fallback.resolve()
    if not (bind / "full").is_dir():
        ap.error("Prepare the binding-only fallback candidate before running the guard")
    shots = shot_map()
    ok_pre, ok_v1 = successes(pre), successes(v1)
    old = json.loads((v1 / "text_guard.json").read_text())["screens"] if (v1 / "text_guard.json").is_file() else {}
    report, fallback = {}, []
    for sid in read_sids(args.cohort):
        xml = v1 / "full" / sid / "final.xml"
        sha = hashlib.sha256(xml.read_bytes()).hexdigest() if xml.is_file() else None
        if sid in old and (old[sid]["rolled_back"] or old[sid].get("xml_sha256") in (None, sha)):
            report[sid] = old[sid]  # already checked this XML (or already rolled back)
            continue
        p_pre, p_v1 = current_capture(pre, sid, ok_pre), current_capture(v1, sid, ok_v1)
        if p_pre is None or p_v1 is None:
            continue
        with Image.open(shots[sid]) as im:
            w, h = im.size
        want = [t for t in (norm(l["text"]) for l in load_ocr(shots[sid], w, h) if l["box"][1] >= MASK_TOP * h)
                if len(t) >= 2]
        if not want:
            continue
        r_pre, r_v1 = recall(want, p_pre), recall(want, p_v1)
        drop = r_pre - r_v1 >= DROP and (bind / "full" / sid / "final.xml").is_file()
        report[sid] = {"ref_lines": len(want), "recall_pre": round(r_pre, 3), "recall_loop": round(r_v1, 3),
                       "rolled_back": bool(drop), "xml_sha256": sha}
        if drop:
            fallback.append(sid)
    print(f"{v1.name}: {len(report)} screens checked, rollback {len(fallback)}: "
          + ", ".join(f"{s}({report[s]['recall_pre']:.2f}->{report[s]['recall_loop']:.2f})" for s in fallback))
    if args.apply:
        backup = v1 / "content_guard_backup"
        for sid in fallback:
            dst = v1 / "full" / sid
            (backup / sid).parent.mkdir(parents=True, exist_ok=True)
            if not (backup / sid).exists():
                shutil.copytree(dst, backup / sid)
            shutil.rmtree(dst / "drawables", ignore_errors=True)
            src = bind / "full" / sid
            if (src / "drawables").is_dir():
                shutil.copytree(src / "drawables", dst / "drawables")
            shutil.copy2(src / "final.xml", dst / "final.xml")
        (v1 / "text_guard.json").write_text(json.dumps({"drop": DROP, "screens": report}, indent=1) + "\n",
                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
