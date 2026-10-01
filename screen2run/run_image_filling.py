#!/usr/bin/env python3
"""Batch driver for the Screen2Run grounding loop (image_filling.py).

  prepare  S5 XML -> id-injected pre-render candidate (new directory, sources untouched)
  render   render a candidate on emulator-5554 only (parallel_render --devices 1)
  ground   pre-render hierarchy + reference measurements -> grounded, asset-bound candidate
  sheets   reference | before | after comparison sheets for visual review

No model calls anywhere in this file.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO = release_paths.RELEASE
RUNS = release_paths.RENDERS
SHOT_LIST = release_paths.DATASETS / "backbone_120.txt"

from image_filling import Grounder, inject_ids  # noqa: E402


def shot_map() -> dict[str, Path]:
    out = {}
    lists = [SHOT_LIST.parent / "main_600.txt", SHOT_LIST]
    extra = os.environ.get("SCREEN2RUN_SCREENSHOT_LIST")
    if extra:
        lists.append(Path(extra).expanduser())
    for lst in lists:
        if not lst.is_file():
            continue
        for line in lst.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and Path(line).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                out[Path(line).stem] = release_paths.resolve_screenshot(line)
    return out


def read_sids(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        data = json.loads(text)
        items = data.get("screens") if isinstance(data, dict) else data
        return [s["screen_id"] if isinstance(s, dict) else str(s) for s in items]
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name = line.rsplit("/", 1)[-1]
        out.append(name[:-4] if name.lower().endswith(".png") else name)
    return out


def settling(screen_dir: Path, quiet: float = 30.0) -> bool:
    """True while a generation process may still be writing this screen's files."""
    import time
    newest = max((p.stat().st_mtime for p in screen_dir.rglob("*") if p.is_file()), default=0.0)
    return time.time() - newest < quiet


def find_source(sid: str, roots: list[Path]) -> Path | None:
    for root in roots:
        for name in (f"{sid}_final.xml", "final_unbound.xml"):
            p = root / "full" / sid / name
            if p.is_file():
                return p
    return None


def write_screen(dst: Path, sid: str, xml: str, src_dir: Path | None, extra: dict | None = None) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "final.xml").write_text(xml if xml.endswith("\n") else xml + "\n", encoding="utf-8")
    if src_dir is not None:
        for name in set(re.findall(r"@drawable/(\w+)", xml)) - {"img"}:
            for ext in ("png", "xml", "jpg", "webp"):
                p = src_dir / "drawables" / f"{name}.{ext}"
                if p.is_file():
                    (dst / "drawables").mkdir(exist_ok=True)
                    shutil.copy2(p, dst / "drawables" / p.name)
    board = {"screen_id": sid, "variant": "full", "diagnostic_only": True,
             "review_status": "grounding_loop_candidate"}
    board.update(extra or {})
    (dst / "board.json").write_text(json.dumps(board, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def finish_candidate(cand: Path, sids: list[str], label: str, model: str | None = None) -> None:
    shots = shot_map()
    arm = cand / "full"
    arm.mkdir(parents=True, exist_ok=True)
    have = [s for s in sids if (arm / s / "final.xml").is_file()]
    # Model identity must come from a generation manifest, never from a folder label.
    (arm / "run_summary.json").write_text(json.dumps(
        {"label": label, "model": model, "diagnostic_only": True,
         "screens": [{"screen_id": s} for s in have], "calls": 0}, indent=1) + "\n", encoding="utf-8")
    (cand / "screenshots.txt").write_text("".join(f"{shots[s]}\n" for s in have), encoding="utf-8")


def compile_cleanup(xml: str, existing: set[str] | frozenset = frozenset()) -> tuple[str, dict]:
    """The deterministic engineering-layer cleanup every arm receives (see postprocess.post).

    ``existing`` are the drawables shipped with the screen (our native shapes); only
    references to resources that exist nowhere become the placeholder.
    """
    from attribute_filter import drop_unknown_android_attrs
    from resource_sanitizer import PLACEHOLDER, PRIVATE_ANDROID, _public_drawables, \
        fill_missing_layout_dims, neutralize_invalid_attrs
    public_drawables = _public_drawables()
    xml, dropped = drop_unknown_android_attrs(xml)
    counts = {"private": 0, "dangling": 0}

    def priv(m):
        if m.group(1) in public_drawables:
            return m.group(0)
        counts["private"] += 1
        return f"@drawable/{PLACEHOLDER}"

    def local_ref(m):
        if m.group(1) == PLACEHOLDER or m.group(1) in existing:
            return m.group(0)
        counts["dangling"] += 1
        return f"@drawable/{PLACEHOLDER}"

    xml = PRIVATE_ANDROID.sub(priv, xml)
    xml = re.sub(r"@drawable/([A-Za-z0-9_.]+)", local_ref, xml)
    xml, n_attr = neutralize_invalid_attrs(xml)
    xml, n_dims = fill_missing_layout_dims(xml)
    return xml, {"unknown_attrs": dropped, "private_drawables": counts["private"],
                 "dangling_local": counts["dangling"], "invalid_attr_values": n_attr,
                 "missing_layout_dims": n_dims}


def cmd_prepare(args) -> int:
    sids = read_sids(args.sids)
    cand = args.out
    if (cand / "full").exists() and not args.incremental:
        raise SystemExit(f"refusing to overwrite {cand}")
    missing = []
    for sid in sids:
        if args.incremental and (cand / "full" / sid / "final.xml").is_file():
            continue
        src = find_source(sid, args.src)
        if src is None:
            missing.append(sid)
            continue
        if args.incremental and settling(src.parent):
            continue
        shipped = {f.stem for f in (src.parent / "drawables").glob("*")} if (src.parent / "drawables").is_dir() else set()
        xml, cleanup = compile_cleanup(inject_ids(src.read_text(encoding="utf-8")), shipped)
        write_screen(cand / "full" / sid, sid, xml, src.parent,
                     {"source_xml": release_paths.record_path(src), "compile_cleanup": cleanup})
    finish_candidate(cand, sids, "pre-render (S5 XML with stable ids)")
    print(f"prepared {len(sids) - len(missing)} screens in {cand}; missing S5: {missing}")
    return 2 if missing else 0


def cmd_render(args) -> int:
    cand = args.cand.resolve()
    cmd = [sys.executable, str(release_paths.module_file("parallel_render.py")), "--candidate-root", str(cand),
           "--matrix", cand.name, "--arm", "full", "--screenshot-list", str(cand / "screenshots.txt"),
           "--devices", os.environ.get("S2R_DEVICES", "1")]
    if args.resume:
        cmd.append("--resume")
    if args.no_hierarchy:
        cmd.append("--no-hierarchy")
    return subprocess.run(cmd, check=False).returncode


def cmd_ground(args) -> int:
    pre = args.pre.resolve()
    run = RUNS / f"tsc_{pre.name}_full"
    shots = shot_map()
    sids = [s for s in read_sids(pre / "screenshots.txt")]
    out = args.out
    if (out / "full").exists() and not (args.force or args.incremental):
        raise SystemExit(f"refusing to overwrite {out}")
    reports = []
    if args.incremental and (out / "ground_reports.json").is_file():
        reports = json.loads((out / "ground_reports.json").read_text(encoding="utf-8"))
        reports = [r for r in reports if "error" not in r]
    for sid in sids:
        if args.only and sid not in args.only:
            continue
        xml = (pre / "full" / sid / "final.xml").read_text(encoding="utf-8")
        vh = run / "raw" / "hierarchy" / f"{sid}.xml"
        png = run / "raw" / "screenshots" / f"{sid}.png"
        dst = out / "full" / sid
        if args.incremental and ((dst / "final.xml").is_file() or not vh.is_file()):
            continue
        if dst.exists():
            shutil.rmtree(dst)
        try:
            if not vh.is_file():
                raise FileNotFoundError(f"no pre-render hierarchy for {sid}")
            rep = Grounder(sid, shots[sid], xml, vh, png).run(dst)
            shipped = pre / "full" / sid / "drawables"
            out_xml = (dst / "final.xml").read_text(encoding="utf-8")
            for name in set(re.findall(r"@drawable/([A-Za-z0-9_.]+)", out_xml)):
                if shipped.is_dir() and not any((dst / "drawables").glob(f"{name}.*")):
                    for f in shipped.glob(f"{name}.*"):
                        (dst / "drawables").mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dst / "drawables" / f.name)
            write_screen(dst, sid, (dst / "final.xml").read_text(encoding="utf-8"), None,
                         {"grounding": "image_filling", "pre_render_run": release_paths.record_path(run)})
            reports.append(rep)
            print(f"  {sid[:36]:36s} anchors={rep['calibration']['anchors']:2d} "
                  f"text={rep['text_matched']}/{rep['text_nodes']} +text={rep['added_text']} "
                  f"+img={rep['added_images']} crops={rep['crops_bound']} inpaint={rep['crops_inpainted']} "
                  f"removed={rep['removed']} ay={rep['calibration']['ay']:.3f}", flush=True)
        except Exception as exc:  # keep going; the screen stays visible as a failure
            print(f"  !! {sid}: {type(exc).__name__}: {exc}", flush=True)
            traceback.print_exc(limit=2)
            reports.append({"screen_id": sid, "error": f"{type(exc).__name__}: {exc}"})
    finish_candidate(out, sids, "grounded + asset-bound (image_filling)")
    (out / "ground_reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
    return 2 if any("error" in report for report in reports) else 0


def cmd_sheets(args) -> int:
    from PIL import Image, ImageDraw
    shots = shot_map()
    runs = [(spec.split("=", 1)[0], Path(spec.split("=", 1)[1])) for spec in args.run]
    sids = read_sids(args.sids)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    tiles = []
    for sid in sids:
        panes = [("ref", shots[sid])] + [(name, r / "raw" / "screenshots" / f"{sid}.png") for name, r in runs]
        ims = []
        for name, p in panes:
            if p.is_file():
                im = Image.open(p).convert("RGB")
            else:
                im = Image.new("RGB", (540, 960), (255, 0, 255))
            ims.append((name, im.resize((max(1, round(im.width * args.height / im.height)), args.height))))
        w = sum(im.width for _, im in ims) + 8 * (len(ims) - 1)
        tile = Image.new("RGB", (w, args.height + 22), (30, 30, 30))
        x = 0
        for name, im in ims:
            tile.paste(im, (x, 22))
            ImageDraw.Draw(tile).text((x + 4, 4), name, fill=(255, 255, 0))
            x += im.width + 8
        ImageDraw.Draw(tile).text((w - 260, 4), sid[:38], fill=(255, 255, 255))
        tile.save(out / f"{sid}.png")
        tiles.append(tile)
    per = args.per_sheet
    for k in range(0, len(tiles), per):
        chunk = tiles[k:k + per]
        cols = 2 if len(chunk) > 1 else 1
        rows = (len(chunk) + cols - 1) // cols
        cw = max(t.width for t in chunk)
        ch = max(t.height for t in chunk)
        sheet = Image.new("RGB", (cols * (cw + 12), rows * (ch + 12)), (15, 15, 15))
        for i, t in enumerate(chunk):
            sheet.paste(t, ((i % cols) * (cw + 12), (i // cols) * (ch + 12)))
        sheet.save(out / f"sheet_{k // per:02d}.png")
    print(f"wrote {len(tiles)} tiles and {(len(tiles) + per - 1) // per} sheets to {out}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--sids", type=Path, required=True)
    p.add_argument("--src", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--incremental", action="store_true",
                   help="add screens not yet prepared to an existing candidate")
    p = sub.add_parser("render")
    p.add_argument("--cand", type=Path, required=True)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-hierarchy", action="store_true")
    p = sub.add_parser("ground")
    p.add_argument("--pre", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--only", nargs="*")
    p.add_argument("--force", action="store_true")
    p.add_argument("--incremental", action="store_true",
                   help="ground only screens that have a pre-render capture and no grounded output yet")
    p = sub.add_parser("sheets")
    p.add_argument("--sids", type=Path, required=True)
    p.add_argument("--run", nargs="+", required=True, help="name=run_dir")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--per-sheet", type=int, default=4)
    args = ap.parse_args()
    return {"prepare": cmd_prepare, "render": cmd_render, "ground": cmd_ground, "sheets": cmd_sheets}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
