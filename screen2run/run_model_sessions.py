#!/usr/bin/env python3
"Batch generation through sessions S1-S5. Image Filling runs separately."
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO = release_paths.RELEASE

from openai import OpenAI
from model_sessions import generate

DATASET = release_paths.DATASETS


def run_one(client, model, sid, shot, candidate, measured_root=None) -> Path:
    "Generate one unbound screen from an explicitly resolved screenshot path."
    if not Path(shot).is_file():
        raise SystemExit(f"no screenshot: {shot}")
    import json as _j
    measured = None
    if measured_root:
        mp = Path(measured_root) / sid / "measured_elements.json"
        if mp.is_file():
            measured = _j.loads(mp.read_text())
    out = candidate / "full" / sid
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing screen artifacts: {out}")
    claim = _claim(candidate, sid)
    try:
        return _run_claimed(client, model, sid, shot, candidate, measured, out)
    except BaseException:
        claim.unlink(missing_ok=True)
        raise


class ClaimedElsewhere(Exception):
    """Another batch process sharing this candidate is already generating the screen."""


def _claim(candidate: Path, sid: str) -> Path:
    # Several batch processes may share one candidate and one screen list; the
    # exclusive create makes sure each screen is paid for exactly once.
    claims = candidate / "claims"
    claims.mkdir(parents=True, exist_ok=True)
    path = claims / sid
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise ClaimedElsewhere(sid) from None
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    if (candidate / "full" / sid / "final.xml").is_file():
        raise ClaimedElsewhere(sid)
    return path


def _run_claimed(client, model, sid, shot, candidate, measured, out) -> Path:
    import tempfile
    work_root = candidate / "generation_work"
    work_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=f"{sid}_", dir=work_root))
    generate(client, model, Path(shot), work, measured)
    xml = work / "activity_main.xml"
    (work / "drawables").mkdir(parents=True, exist_ok=True)
    # Assets are bound later by Image Filling (image_filling.py), from each view's measured box.
    (work / "imaged.xml").write_text(xml.read_text(encoding="utf-8"), encoding="utf-8")
    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    (out / "drawables").mkdir(parents=True, exist_ok=True)
    (out / "final.xml").write_text((work / "imaged.xml").read_text(encoding="utf-8"), encoding="utf-8")
    import shutil
    shutil.copy2(xml, out / "final_unbound.xml")
    shutil.copy2(xml, out / f"{sid}_final.xml")
    evidence = out / "generation_evidence"
    evidence.mkdir()
    for p in work.iterdir():
        if p.is_file() and (p.name.startswith(("stage", "s1_", "s2_", "s3_", "s4_", "s5_", "missing_text"))
                            or p.name.startswith("native_") or p.name == "generation_manifest.json"):
            shutil.copy2(p, evidence / p.name)
    copy_generated_drawables(work / "drawables", out / "drawables")
    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    for src, dst in (("stage1.json", f"{sid}_s1.json"),
                     ("stage2.json", f"{sid}_s2.json"),
                     ("stage3_original.xml", f"{sid}_original.xml")):
        p = work / src
        if p.is_file():
            import shutil
            shutil.copy2(p, out / dst)
    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    critique = work / "stage4_critique.txt"
    if critique.is_file():
        text = critique.read_text(encoding="utf-8")
        (out / f"{sid}_review.json").write_text(
            json.dumps(_parse_critique(text), ensure_ascii=False, indent=1) + "\n",
            encoding="utf-8")
    return out / "final.xml"


def copy_generated_drawables(source: Path, target: Path) -> None:
    """Keep native shape XML alongside bitmaps; never silently lose the surface."""
    import shutil
    from native_drawables import PREFIX, is_native_shape_resource
    target.mkdir(parents=True, exist_ok=True)
    for path in source.iterdir():
        if path.stem.startswith(PREFIX) and path.suffix.lower() != '.xml':
            raise ValueError('Native shape name cannot identify a bitmap: ' + path.name)
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp'}:
            shutil.copy2(path, target / path.name)
        elif path.suffix.lower() == '.xml' and path.stem.startswith(PREFIX):
            if not is_native_shape_resource(path):
                raise ValueError('Invalid native shape resource: ' + path.name)
            shutil.copy2(path, target / path.name)


def _parse_critique(text: str) -> dict:
    "Archive S4 review issues with the unchanged raw report."
    issues, kind = [], "note"
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("- ") or s.startswith("* "):
            issues.append({"type": kind, "msg": s[2:].strip()})
        elif s.endswith(":"):
            kind = s.rstrip(":").strip().lower().replace(" ", "_") or "note"
        else:
            issues.append({"type": kind, "msg": s})
    return {"issues": issues, "raw": text}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screens", nargs="+", required=True, help="screen_id:dataset_name specifications")
    ap.add_argument("--shot-list", type=Path,
                    help="Screenshot path list indexed by filename stem. "
                         "Use this when screenshots span multiple directories.")
    ap.add_argument("--candidate", default=str(release_paths.CANDIDATES / "screen2run"))
    ap.add_argument("--measured-root", default=str(release_paths.MEASUREMENTS))
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--workers", type=int, default=1,
                    help="Concurrent screenshot generation jobs")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip screens whose final.xml already exists")
    args = ap.parse_args()

    shot_map: dict[str, str] = {}
    if args.shot_list:
        for line in args.shot_list.read_text().splitlines():
            line = line.strip()
            if line:
                shot_map[Path(line).stem] = str(release_paths.resolve_screenshot(line))

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=600.0)
    jobs = []
    for spec in args.screens:
        sid = spec.split(":")[0]
        if args.shot_list:
            if sid not in shot_map:
                print(f"  !! {sid} not in screenshot list; skipped", flush=True)
                continue
            shot = shot_map[sid]
        else:
            split = spec.split(":")[1]
            shot = str(release_paths.SCREENSHOTS / split / f"{sid}.png")
        if args.skip_existing and (Path(args.candidate) / "full" / sid / "final.xml").is_file():
            print(f"  {sid}: already exists; skipped", flush=True)
            continue
        jobs.append((sid, shot))

    print(f"Queued {len(jobs)} screens; workers {args.workers}", flush=True)
    done = failed = 0
    if args.workers <= 1:
        for sid, shot in jobs:
            print(f"=== {sid} ===", flush=True)
            try:
                out = run_one(client, args.model, sid, shot, Path(args.candidate), args.measured_root)
                print(f"  -> {out}", flush=True)
                done += 1
            except ClaimedElsewhere:
                print(f"  -- {sid} claimed by another process", flush=True)
            except Exception as exc:
                failed += 1
                print(f"  !! {sid} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    else:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, client, args.model, sid, shot,
                                   Path(args.candidate), args.measured_root): sid
                       for sid, shot in jobs}
            for fut in as_completed(futures):
                sid = futures[fut]
                try:
                    fut.result()
                    done += 1
                    print(f"  ok {sid}  ({done}/{len(jobs)}, failed {failed})", flush=True)
                except ClaimedElsewhere:
                    print(f"  -- {sid} claimed by another process", flush=True)
                except Exception as exc:
                    failed += 1
                    print(f"  !! {sid} failed: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
    print(f"\nCompleted {done} screens; failed {failed} screens", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
