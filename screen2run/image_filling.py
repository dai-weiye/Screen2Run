#!/usr/bin/env python3
"""Screen2Run execution loop: ground the model's XML on the reference's measured geometry.

The MLLM stages (S1-S5) decide *what* the screen contains: widget types, text,
colours, hierarchy. They are unreliable about *where*: DeepSeek reads a 688x1070
screenshot as if it were 656x1134, so every margin it converts is off, the page
drifts down and grows, and crops cut from the drifted slot land on half an icon.
Geometry is therefore owned by measurement, not by the model:

  prebind   every view gets a stable android:id; placeholders stay.
  render    (emulator) where each view actually lands -> uiautomator hierarchy.
  ground    match rendered views to reference measurements (Vision OCR lines by
            text, UIED components by position), fit the model's drift (global
            scale/shift + local residual) and give every view a target box in
            reference pixels.  Nothing from GT XML/VH is used.
  rewrite   explicit dp size/position per view inside its parent; text size fitted
            to the measured line; colours sampled where the model's disagree.
  assets    every image slot is cropped from its own reference box (refined to the
            object's boundary) and bound by id.  Large photos get the UI drawn on top
            of them inpainted out, so no crop ever carries native text or controls.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unicodedata
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO = release_paths.RELEASE
OCR_CACHE = release_paths.CACHE / "ocr"
OCR_SWIFT = release_paths.OCR_SWIFT
MEASURED_ROOT = release_paths.MEASUREMENTS
FONT_PATH = release_paths.FONT

ANDROID = "http://schemas.android.com/apk/res/android"
A = "{" + ANDROID + "}"
for _prefix, _uri in (("android", ANDROID), ("tools", "http://schemas.android.com/tools"),
                      ("app", "http://schemas.android.com/apk/res-auto")):
    ET.register_namespace(_prefix, _uri)

DENSITY = 2.625
CANVAS_W = 1080
PKG = "com.example.myapplication"
ROBOTO_LINE = 1.172          # (ascent + descent) / em for Roboto, includeFontPadding=false
BIG_IMAGE_FRAC = 0.25        # image slots at least this large are page photos, not glyphs

TEXT_TAGS = {"TextView", "Button", "EditText", "CheckBox", "RadioButton", "Switch", "ToggleButton",
             "CheckedTextView", "AutoCompleteTextView", "MultiAutoCompleteTextView"}
FIELD_TAGS = {"EditText", "AutoCompleteTextView", "MultiAutoCompleteTextView"}
BUTTON_TAGS = {"Button", "ToggleButton"}
COMPOUND_TAGS = {"CheckBox", "RadioButton", "Switch", "CheckedTextView"}
IMAGE_TAGS = {"ImageView", "ImageButton"}
WIDGET_TAGS = {"SeekBar", "ProgressBar", "RatingBar", "Spinner"}
SCROLL_TAGS = {"ScrollView", "HorizontalScrollView", "NestedScrollView"}
FLATTEN_TAGS = {"LinearLayout", "RelativeLayout", "RadioGroup", "TableLayout", "TableRow",
                "GridLayout", "ConstraintLayout", "CoordinatorLayout", "CardView"}
NON_VIEW_TAGS = {"requestFocus", "tag"}
CHILD_LAYOUT_ATTRS = (
    "layout_weight", "layout_gravity", "layout_margin", "layout_marginTop", "layout_marginBottom",
    "layout_marginLeft", "layout_marginRight", "layout_marginStart", "layout_marginEnd",
    "layout_marginHorizontal", "layout_marginVertical", "layout_above", "layout_below",
    "layout_toLeftOf", "layout_toRightOf", "layout_toStartOf", "layout_toEndOf", "layout_alignLeft",
    "layout_alignRight", "layout_alignStart", "layout_alignEnd", "layout_alignTop", "layout_alignBottom",
    "layout_alignBaseline", "layout_alignParentLeft", "layout_alignParentRight", "layout_alignParentStart",
    "layout_alignParentEnd", "layout_alignParentTop", "layout_alignParentBottom", "layout_centerInParent",
    "layout_centerHorizontal", "layout_centerVertical", "layout_alignWithParentIfMissing", "layout_column",
    "layout_row", "layout_span", "layout_columnWeight", "layout_rowWeight", "layout_x", "layout_y",
    "layout_columnSpan", "layout_rowSpan")
PADDING_ATTRS = ("padding", "paddingTop", "paddingBottom", "paddingLeft", "paddingRight", "paddingStart",
                 "paddingEnd", "paddingHorizontal", "paddingVertical")
CONTAINER_ONLY_ATTRS = ("orientation", "gravity", "weightSum", "baselineAligned", "divider", "showDividers",
                        "dividerPadding", "measureWithLargestChild", "fitsSystemWindows", "stretchColumns",
                        "shrinkColumns", "columnCount", "rowCount", "useDefaultMargins", "checkedButton")
TEXT_SIZE_ATTRS = ("lines", "minLines", "maxLines", "singleLine", "ellipsize", "maxEms", "ems", "minEms",
                   "maxWidth", "maxHeight", "autoSizeTextType", "autoSizeMinTextSize", "autoSizeMaxTextSize",
                   "autoSizeStepGranularity", "autoSizePresetSizes", "lineSpacingExtra",
                   "lineSpacingMultiplier", "lineHeight", "firstBaselineToTopHeight",
                   "lastBaselineToBottomHeight")
COMPOUND_DRAWABLE_ATTRS = ("drawableLeft", "drawableRight", "drawableStart", "drawableEnd", "drawableTop",
                           "drawableBottom")
PLACEHOLDER_ONLY_ATTRS = ("button", "thumb", "track", "progressDrawable", "foreground", "tickMark",
                          "indeterminateDrawable", "progressDrawableTiled")


# --------------------------------------------------------------------------- geometry

Box = list  # [x0, y0, x1, y1] in reference pixels


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a: Box, b: Box) -> float:
    i = inter(a, b)
    u = area(a) + area(b) - i
    return i / u if u > 0 else 0.0


def frac_inside(a: Box, b: Box) -> float:
    """Fraction of a's area inside b."""
    return inter(a, b) / area(a) if area(a) > 0 else 0.0


def union(boxes) -> Box | None:
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def center(b: Box) -> tuple[float, float]:
    return (0.5 * (b[0] + b[2]), 0.5 * (b[1] + b[3]))


def clip(b: Box, w: float, h: float) -> Box:
    return [max(0.0, min(b[0], w)), max(0.0, min(b[1], h)), max(0.0, min(b[2], w)), max(0.0, min(b[3], h))]


def expand(b: Box, fx: float, fy: float | None = None, minpx: float = 0.0) -> Box:
    fy = fx if fy is None else fy
    dx = max(minpx, (b[2] - b[0]) * fx)
    dy = max(minpx, (b[3] - b[1]) * fy)
    return [b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy]


def contains_point(b: Box, p) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


@dataclass
class Frame:
    ref_w: int
    ref_h: int

    @property
    def k(self) -> float:
        return CANVAS_W / self.ref_w

    @property
    def screen_h(self) -> int:
        return max(1, round(self.ref_h * CANVAS_W / self.ref_w))

    def dp(self, ref_px: float) -> float:
        return ref_px * self.k / DENSITY


# --------------------------------------------------------------------------- perception

def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ocr(ref_png: Path, w: int, h: int) -> list[dict]:
    cache = OCR_CACHE / f"{sha256_file(ref_png)}.json"
    rec = None
    if cache.is_file():
        try:
            rec = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rec = None
        if rec is not None and str(rec.get("error") or ""):
            rec = None
    if rec is None:
        fd, tmp = tempfile.mkstemp(prefix="s2g_ocr_", suffix=".jsonl")
        os.close(fd)
        env = dict(os.environ, CLANG_MODULE_CACHE_PATH=str(OCR_CACHE / "swift_module_cache"))
        binary = release_paths.CACHE / "ocr" / "vision_ocr"  # swiftc -O build of OCR_SWIFT
        cmd = [str(binary)] if binary.is_file() else ["swift", str(OCR_SWIFT)]
        subprocess.run(cmd + ["--output", tmp, str(ref_png)], capture_output=True,
                       timeout=300, env=env, check=False)
        text = Path(tmp).read_text(encoding="utf-8") if Path(tmp).is_file() else ""
        Path(tmp).unlink(missing_ok=True)
        first = next((ln for ln in text.splitlines() if ln.strip()), "")
        rec = json.loads(first) if first else {"error": "no_output", "lines": []}
        if not rec.get("error"):
            OCR_CACHE.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    lines = []
    for i, line in enumerate(rec.get("lines") or []):
        t = str(line.get("text") or "").strip()
        if not t or float(line.get("confidence") or 0) < 0.3:
            continue
        x, y, bw, bh = (float(v) for v in line["bbox"][:4])
        lines.append({"id": i, "text": t, "box": [x * w, (1.0 - y - bh) * h, (x + bw) * w, (1.0 - y) * h]})
    return lines


def load_components(sid: str, w: int, h: int) -> list[dict]:
    path = MEASURED_ROOT / sid / "measured_elements.json"
    comps = []
    if path.is_file():
        measured = json.loads(path.read_text(encoding="utf-8"))
        if measured.get("width") == w and measured.get("height") == h:
            for e in measured.get("elements") or []:
                b = e.get("bounds")
                if e.get("kind") == "component" and isinstance(b, list) and len(b) >= 4:
                    box = [float(v) for v in b[:4]]
                    if box[2] > box[0] + 1 and box[3] > box[1] + 1:
                        comps.append({"id": e.get("id"), "box": box})
    return comps


def _row_fraction(arr: np.ndarray, y: int, color: np.ndarray, tol: float) -> float:
    return float(np.mean(np.sqrt(((arr[y] - color) ** 2).sum(axis=1)) < tol))


def detect_bands(arr: np.ndarray, ocr: list[dict], comps: list[dict]) -> dict:
    """System status bar (top) and 3-button/gesture navigation bar (bottom), in ref px."""
    h, w = arr.shape[:2]
    # the platform status bar is 24dp; a toolbar of the same colour must not be swallowed by it,
    # so only items that start in the upper half of that height and end within it count
    E = 24.0 * w / 411.43

    def status_like(b):
        return b[1] < 0.5 * E and b[3] <= 1.15 * E and (b[3] - b[1]) <= 0.8 * E

    top_items = [l["box"] for l in ocr if status_like(l["box"])] + \
                [c["box"] for c in comps if status_like(c["box"]) and (c["box"][2] - c["box"][0]) < 0.3 * w]
    icon_bottom = max((b[3] for b in top_items), default=0.0)
    c0 = np.median(arr[min(1, h - 1)], axis=0)
    y = 0
    while y < int(0.07 * h) and _row_fraction(arr, y, c0, 26.0) >= 0.7:
        y += 1
    row_bottom = float(y)
    status = 0.0
    if 0.012 * h <= row_bottom <= min(0.05 * h, 1.3 * E) and icon_bottom <= row_bottom + 6:
        status = row_bottom
    elif icon_bottom > 0:
        status = icon_bottom + 0.2 * max(8.0, icon_bottom - min(b[1] for b in top_items))
    status = min(status, 0.045 * h, 1.3 * E)
    status_color = [int(v) for v in np.median(arr[: max(1, int(status))].reshape(-1, 3), axis=0)] if status else None

    cb = np.median(arr[h - 2], axis=0)
    y = h - 1
    while y > h - int(0.11 * h) and _row_fraction(arr, y, cb, 26.0) >= 0.75:
        y -= 1
    nav_rows = h - 1 - y
    nav_top = None
    if nav_rows > 0.10 * h:
        # the app's own dark footer continues the bar; locate the bar by its 3 system buttons
        low = [c["box"] for c in comps if c["box"][1] >= 0.88 * h and (c["box"][2] - c["box"][0]) < 0.2 * w]
        trio = [b for b in low if any(abs(center(b)[0] / w - t) < 0.08 for t in (0.25, 0.5, 0.75, 0.2, 0.8))]
        if len(trio) >= 2:
            cy = float(np.median([center(b)[1] for b in trio]))
            nav_rows = int(max(0.03 * h, min(0.1 * h, 2 * (h - cy))))
        else:
            nav_rows = int(0.054 * h)
    if 0.03 * h <= nav_rows <= 0.10 * h:
        band = [0, h - nav_rows, w, h]
        icons = [c["box"] for c in comps if frac_inside(c["box"], band) > 0.8 and (c["box"][2] - c["box"][0]) < 0.2 * w]
        centers = sorted(center(b)[0] / w for b in icons)
        three = any(all(any(abs(cx - t) < 0.08 for cx in centers) for t in trio)
                    for trio in ((0.25, 0.5, 0.75), (0.2, 0.5, 0.8), (0.22, 0.5, 0.78)))
        texty = any(frac_inside(l["box"], band) > 0.6 for l in ocr)
        if (float(np.max(cb)) < 45 or three) and not texty:
            nav_top = float(h - nav_rows)
    nav_color = [int(v) for v in cb] if nav_top is not None else None
    return {"status_bottom": status, "status_color": status_color, "nav_top": nav_top, "nav_color": nav_color}


# --------------------------------------------------------------------------- colour sampling

def _pixels(arr: np.ndarray, b: Box) -> np.ndarray:
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in clip(b, w, h))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 3), dtype=np.float32)
    return arr[y0:y1, x0:x1].reshape(-1, 3)


def mode_color(arr: np.ndarray, b: Box) -> tuple[list[int] | None, float]:
    px = _pixels(arr, b)
    if len(px) == 0:
        return None, 0.0
    q = (px // 12).astype(np.int32)
    codes = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(codes, return_counts=True)
    top = vals[np.argmax(counts)]
    sel = px[codes == top]
    return [int(v) for v in sel.mean(axis=0)], float(counts.max() / len(px))


def ink_color(arr: np.ndarray, b: Box) -> list[int] | None:
    h, w = arr.shape[:2]
    ring = _pixels(arr, expand(b, 0, 0, 3))
    inner = _pixels(arr, b)
    if len(inner) < 6 or len(ring) == 0:
        return None
    bg, _ = mode_color(arr, expand(b, 0, 0, 3))
    if bg is None:
        return None
    d = np.sqrt(((inner - np.array(bg, dtype=np.float32)) ** 2).sum(axis=1))
    if d.max() < 40:
        return None
    # anti-aliased strokes are lighter than the colour that drew them: the stroke core is
    # the best estimate of the declared text colour
    core = inner[d >= max(40.0, 0.8 * float(d.max()))]
    ink = core if len(core) >= 3 else inner[d >= max(40.0, 0.6 * float(d.max()))]
    if len(ink) < 3:
        return None
    return [int(v) for v in np.median(ink, axis=0)]


def paints_surface(bg: str | None) -> bool:
    """A flat colour or a shape drawable (not a reference crop, not the placeholder)."""
    if not bg or "transparent" in bg:
        return False
    if parse_color(bg) is not None:
        return True
    return bg.startswith("@drawable/") and not bg.startswith("@drawable/s2g") and bg != "@drawable/img"


def appearance_color(arr: np.ndarray, b: Box) -> list[int] | None:
    """Text colour as it appears at this resolution: mean of the ink cluster of a two-means split."""
    px = _pixels(arr, b)
    if len(px) < 8:
        return None
    px = px.astype(np.float32)
    lum = px.mean(axis=1)
    c0, c1 = px[int(np.argmin(lum))].copy(), px[int(np.argmax(lum))].copy()
    for _ in range(12):
        near0 = ((px - c0) ** 2).sum(axis=1) <= ((px - c1) ** 2).sum(axis=1)
        if near0.all() or not near0.any():
            return None
        c0, c1 = px[near0].mean(axis=0), px[~near0].mean(axis=0)
    ink = c0 if near0.sum() <= (~near0).sum() else c1
    return [int(round(v)) for v in ink]


def hex_color(c) -> str:
    return "#%02X%02X%02X" % tuple(int(max(0, min(255, v))) for v in c[:3])


def parse_color(s: str | None) -> list[int] | None:
    if not s:
        return None
    m = re.fullmatch(r"#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{8}|[0-9A-Fa-f]{3})", s.strip())
    if not m:
        named = {"@android:color/white": [255, 255, 255], "@android:color/black": [0, 0, 0]}
        return named.get(s.strip())
    v = m.group(1)
    if len(v) == 3:
        return [int(ch * 2, 16) for ch in v]
    if len(v) == 8:
        if int(v[:2], 16) < 128:
            return None
        v = v[2:]
    return [int(v[i:i + 2], 16) for i in (0, 2, 4)]


def cdist(a, b) -> float:
    return float(math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a[:3], b[:3]))))


def snap_color_region(arr: np.ndarray, pred: Box, color, tol: float = 20.0, accept=None,
                      grow: float = 0.35) -> Box | None:
    from scipy import ndimage
    h, w = arr.shape[:2]
    win = [int(round(v)) for v in clip(expand(pred, grow, grow, 6), w, h)]
    x0, y0, x1, y1 = win
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    sub = arr[y0:y1, x0:x1]
    mask = np.sqrt(((sub - np.array(color, dtype=np.float32)) ** 2).sum(axis=2)) <= tol
    if mask.mean() < 0.02:
        return None
    labels, n = ndimage.label(mask)
    if n == 0:
        return None
    px0, py0, px1, py1 = (int(round(v)) for v in (pred[0] - x0, pred[1] - y0, pred[2] - x0, pred[3] - y0))
    px0, py0 = max(0, px0), max(0, py0)
    region = labels[py0:max(py0 + 1, py1), px0:max(px0 + 1, px1)]
    if region.size == 0:
        return None
    ids, counts = np.unique(region[region > 0], return_counts=True)
    if len(ids) == 0:
        return None
    best = ids[np.argmax(counts)]
    ys, xs = np.nonzero(labels == best)
    box = [float(xs.min() + x0), float(ys.min() + y0), float(xs.max() + 1 + x0), float(ys.max() + 1 + y0)]
    if accept is not None:
        return box if accept(box) else None
    ratio = area(box) / max(1.0, area(pred))
    if iou(box, pred) >= 0.35 and 0.4 <= ratio <= 2.5:
        return box
    return None


def surface_style(arr: np.ndarray, box: Box) -> dict | None:
    """Fill, stroke and corner radius of a code-drawn surface, measured from its pixels."""
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in clip(box, w, h))
    if x1 - x0 < 10 or y1 - y0 < 10:
        return None
    inset = max(3, int(0.12 * min(x1 - x0, y1 - y0)))
    fill, frac = mode_color(arr, [x0 + inset, y0 + inset, x1 - inset, y1 - inset])
    if fill is None:
        return None
    outside, _ = mode_color(arr, [max(0, x0 - 3), max(0, y0 - 3), min(w, x1 + 3), min(h, y1 + 3)])
    sides = [arr[y0 + 1, x0 + inset:x1 - inset], arr[y1 - 2, x0 + inset:x1 - inset],
             arr[y0 + inset:y1 - inset, x0 + 1], arr[y0 + inset:y1 - inset, x1 - 2]]
    meds = [np.median(sd, axis=0) for sd in sides if len(sd)]
    edge = [int(v) for v in np.median(np.stack(meds), axis=0)] if meds else fill
    consistent = len(meds) == 4 and max(cdist(a, b) for a in meds for b in meds) < 30
    stroke = None
    stroke_px = 0
    if consistent and cdist(edge, fill) > 25:
        stroke = edge
        mid = (y0 + y1) // 2
        for k in range(0, min(8, (x1 - x0) // 3)):
            if cdist(arr[mid, x0 + k], edge) < 30:
                stroke_px = k + 1
            elif stroke_px:
                break
        stroke_px = max(1, stroke_px)
    bgc = outside if outside is not None else fill
    radii = []
    for cx, cy, sx, sy in ((x0, y0, 1, 1), (x1 - 1, y0, -1, 1), (x0, y1 - 1, 1, -1), (x1 - 1, y1 - 1, -1, -1)):
        k = 0
        lim = min(x1 - x0, y1 - y0) // 2
        while k < lim and cdist(arr[cy + sy * k, cx + sx * k], bgc) < 22 and cdist(bgc, fill) > 18:
            k += 1
        radii.append(k * 3.41)
    radius = float(np.median(radii)) if radii else 0.0
    return {"fill": fill, "fill_frac": frac, "stroke": stroke, "stroke_px": stroke_px, "radius_px": radius,
            "outside": outside}


def refine_image_box(arr: np.ndarray, box: Box) -> Box:
    """Grow/shrink a predicted icon/photo slot to the object's own boundary."""
    from scipy import ndimage
    h, w = arr.shape[:2]
    win = [int(round(v)) for v in clip(expand(box, 0.3, 0.3, 6), w, h)]
    x0, y0, x1, y1 = win
    if x1 - x0 < 4 or y1 - y0 < 4:
        return box
    sub = arr[y0:y1, x0:x1]
    border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]], axis=0)
    if float(border.std(axis=0).mean()) > 18:
        return box  # textured surroundings: a foreground mask would swallow the backdrop
    bg = np.median(border, axis=0)
    fg = np.sqrt(((sub - bg) ** 2).sum(axis=2)) > 28
    if fg.mean() < 0.01:
        return box
    fg = ndimage.binary_closing(fg, structure=np.ones((3, 3)), iterations=2)
    labels, n = ndimage.label(fg)
    if n == 0:
        return box
    rel = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
    keep = []
    for sl_id, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        cb = [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
        if area(cb) < 4:
            continue
        if frac_inside(cb, rel) >= 0.35 or contains_point(cb, center(rel)):
            keep.append(cb)
    got = union(keep)
    if got is None:
        return box
    got = [got[0] + x0, got[1] + y0, got[2] + x0, got[3] + y0]
    if area(got) > 1.8 * area(box) or area(got) < 0.25 * area(box) or iou(got, box) < 0.3:
        return box
    return got


def continues_past(arr: np.ndarray, b: Box) -> list[str]:
    """Sides of the box that the object runs through (its pixels continue just outside)."""
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    if x1 - x0 < 6 or y1 - y0 < 6:
        return []
    pad = max(4, (x1 - x0 + y1 - y0) // 16)
    X0, Y0, X1, Y1 = max(0, x0 - pad), max(0, y0 - pad), min(W, x1 + pad), min(H, y1 + pad)
    ring = np.concatenate([arr[Y0:y0, X0:X1].reshape(-1, 3), arr[y1:Y1, X0:X1].reshape(-1, 3),
                           arr[y0:y1, X0:x0].reshape(-1, 3), arr[y0:y1, x1:X1].reshape(-1, 3)])
    if not len(ring):
        return []
    bg = np.median(ring, axis=0).astype(np.float32)
    f = arr.astype(np.float32)
    sides = []
    for name, outside, inside in (
            ("left", f[y0:y1, max(0, x0 - 3):max(0, x0 - 1)], f[y0:y1, x0:x0 + 2]),
            ("right", f[y0:y1, min(W, x1 + 1):min(W, x1 + 3)], f[y0:y1, x1 - 2:x1]),
            ("top", f[max(0, y0 - 3):max(0, y0 - 1), x0:x1], f[y0:y0 + 2, x0:x1]),
            ("bottom", f[min(H, y1 + 1):min(H, y1 + 3), x0:x1], f[y1 - 2:y1, x0:x1])):
        if outside.size == 0 or inside.size == 0:
            continue
        if (np.abs(outside - bg).max(axis=-1) > 40).mean() > 0.35 and (np.abs(inside - bg).max(axis=-1) > 40).mean() > 0.35:
            sides.append(name)
    return sides


def grow_to_object(arr: np.ndarray, box: Box, texts: list, max_growth: float = 4.0) -> Box:
    """A slot that holds only part of an icon/picture (the model's box was small or offset)
    grows to the whole object when the object stands on a flat backdrop."""
    from scipy import ndimage
    if not continues_past(arr, box):
        return box
    h, w = arr.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    mx, my = 1.6 * bw, 1.6 * bh
    win = [int(round(v)) for v in clip([box[0] - mx, box[1] - my, box[2] + mx, box[3] + my], w, h)]
    x0, y0, x1, y1 = win
    sub = arr[y0:y1, x0:x1].astype(np.float32)
    if sub.shape[0] < 6 or sub.shape[1] < 6:
        return box
    border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]], axis=0)
    if float(border.std(axis=0).mean()) > 18:
        return box
    bg = np.median(border, axis=0)
    fg = np.sqrt(((sub - bg) ** 2).sum(axis=2)) > 28
    fg = ndimage.binary_closing(fg, structure=np.ones((3, 3)), iterations=1)
    labels, n = ndimage.label(fg)
    if n == 0:
        return box
    rel = [box[0] - x0, box[1] - y0, box[2] - x0, box[3] - y0]
    keep = []
    for sl in ndimage.find_objects(labels):
        if sl is None:
            continue
        cb = [sl[1].start, sl[0].start, sl[1].stop, sl[0].stop]
        if inter(cb, rel) >= 0.2 * area(rel) or contains_point(cb, center(rel)):
            keep.append(cb)
    got = union(keep)
    if got is None:
        return box
    if (got[0] <= 0 and x0 > 0) or (got[1] <= 0 and y0 > 0) or (got[2] >= x1 - x0 and x1 < w) or \
            (got[3] >= y1 - y0 and y1 < h):
        return box                      # still cut by the search window: not an isolated object
    got = [got[0] + x0, got[1] + y0, got[2] + x0, got[3] + y0]
    if frac_inside(box, expand(got, 0, 0, 2)) < 0.6 or area(got) > max_growth * area(box):
        return box
    # never swallow a caption next to the picture
    if any(frac_inside(t, got) >= 0.6 and frac_inside(t, box) < 0.3 for t in texts):
        return box
    return union([got, box]) if frac_inside(box, got) < 0.95 else got


_LEAD_GLYPH = re.compile(r"^\s*[•●○◯◉⊙□■☐☑✓✔▪▫◦·*oO0]\s+(?=\S)")


def tight_text(arr: np.ndarray, box: Box, text: str) -> tuple[Box, str]:
    """Ink-tight box of the text itself (drops a leading checkbox/radio glyph Vision reads as '•')."""
    h, w = arr.shape[:2]
    stripped = text
    m = _LEAD_GLYPH.match(text)
    if m and len(text) - m.end() >= 2:
        stripped = text[m.end():]
    x0, y0, x1, y1 = (int(round(v)) for v in clip(box, w, h))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return list(box), stripped
    sub = arr[y0:y1, x0:x1]
    border = np.concatenate([sub[0], sub[-1], sub[:, 0], sub[:, -1]], axis=0)
    q = (border // 12).astype(np.int32)
    codes = q[:, 0] * 10000 + q[:, 1] * 100 + q[:, 2]
    vals, counts = np.unique(codes, return_counts=True)
    bg = border[codes == vals[np.argmax(counts)]].mean(axis=0)
    ink = np.sqrt(((sub - bg) ** 2).sum(axis=2)) > 45
    cols = np.nonzero(ink.any(axis=0))[0]
    rows_all = ink.any(axis=1)
    if len(cols) < 2 or rows_all.sum() < 2:
        return list(box), stripped
    if stripped is not text:
        # skip the first ink run (the glyph) when it is separated by a gap
        gaps = np.nonzero(np.diff(cols) > max(2, int(0.18 * (y1 - y0))))[0]
        if len(gaps):
            cols = cols[gaps[0] + 1:]
    cx0, cx1 = int(cols[0]), int(cols[-1]) + 1
    rows = np.nonzero(ink[:, cx0:cx1].any(axis=1))[0]
    if len(rows) < 2:
        return list(box), stripped
    tb = [float(x0 + cx0), float(y0 + rows[0]), float(x0 + cx1), float(y0 + rows[-1] + 1)]
    if (tb[2] - tb[0]) < 0.3 * (x1 - x0) or (tb[3] - tb[1]) < 0.3 * (y1 - y0):
        return list(box), stripped
    return tb, stripped


# --------------------------------------------------------------------------- text metrics

_FONT = None


def _font():
    global _FONT
    if _FONT is None:
        _FONT = ImageFont.truetype(str(FONT_PATH), 100)
    return _FONT


def text_width_em(text: str) -> float:
    return max(0.3, _font().getlength(text) / 100.0)


_VFONTS: dict = {}
VAR_FONT_PATH = FONT_PATH.with_name("Roboto-Regular.ttf")    # variable: wght 100..900


def _vfont(weight: int, size: int):
    key = (weight, size)
    if key not in _VFONTS:
        f = ImageFont.truetype(str(VAR_FONT_PATH), size)
        f.set_variation_by_axes([weight, 100, 0])
        _VFONTS[key] = f
    return _VFONTS[key]


def _coverage(mask: np.ndarray) -> float:
    return float(mask.mean()) if mask.size else 0.0


def text_weight(arr: np.ndarray, tbox: Box, text: str, ink, bg) -> int | None:
    """400 / 500 / 700: the Roboto weight whose rasterisation covers the tight box like the ink does.

    Same string, same box, same binarisation on both sides, so letter shapes cancel out.
    None when the evidence is weak (small or low-contrast text, or no clear nearest weight).
    """
    if ink is None or bg is None or len(text.strip()) < 2:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in tbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(arr.shape[1], x1), min(arr.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 7:
        return None
    ink_v, bg_v = np.asarray(ink, np.float64), np.asarray(bg, np.float64)
    axis = ink_v - bg_v
    den = float((axis ** 2).sum())
    if den < 60.0 ** 2:
        return None
    patch = np.asarray(arr[y0:y1, x0:x1], dtype=np.float64)
    ref_cov = float(((((patch - bg_v) * axis).sum(axis=2) / den) > 0.5).mean())
    covs = {}
    # rasterise at the size on screen: small text hints heavier than a downsampled large render
    size = int(max(8, round((y1 - y0) / max(0.3, ink_ratio(text) - 0.07))))
    for weight in (400, 500, 700):
        f = _vfont(weight, size)
        l, tp, r, bt = f.getbbox(text)
        if r - l < 4 or bt - tp < 4:
            return None
        im = Image.new("L", (r - l + 4, bt - tp + 4), 0)
        ImageDraw.Draw(im).text((2 - l, 2 - tp), text, font=f, fill=255)
        g = np.asarray(im)
        ys, xs = np.nonzero(g > 127)
        if len(xs) == 0:
            return None
        g = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).resize((x1 - x0, y1 - y0), Image.BILINEAR)
        covs[weight] = float((np.asarray(g) > 127).mean())
    if covs[700] - covs[400] < 0.04:
        return None
    ranked = sorted(covs, key=lambda w: abs(covs[w] - ref_cov))
    step = (covs[700] - covs[400]) / 2.0
    if abs(covs[ranked[1]] - ref_cov) - abs(covs[ranked[0]] - ref_cov) < 0.25 * step:
        return None
    return ranked[0]


def is_bold(arr: np.ndarray, tbox: Box, text: str, ink, bg) -> bool | None:
    """Stroke weight of a read line, compared with Roboto 400 and 700 drawn into the same box.

    Coverage of the ink mask inside the tight text box is letter-dependent, so the same
    string is rasterised at both weights, resized to the box and binarised identically.
    None when the evidence is too weak (tiny text, low contrast, unrenderable glyphs).
    """
    if ink is None or bg is None or not text.strip() or len(text.strip()) < 2:
        return None
    x0, y0, x1, y1 = (int(round(v)) for v in tbox)
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(arr.shape[1], x1), min(arr.shape[0], y1)
    if x1 - x0 < 6 or y1 - y0 < 7:
        return None
    ink_v, bg_v = np.asarray(ink, float), np.asarray(bg, float)
    axis = ink_v - bg_v
    den = float((axis ** 2).sum())
    if den < 60.0 ** 2:
        return None
    patch = arr[y0:y1, x0:x1].astype(float)
    t = ((patch - bg_v) @ axis) / den
    ref_cov = _coverage(t > 0.5)
    covs = {}
    for weight in (400, 700):
        f = _vfont(weight, 64)
        l, tp, r, bt = f.getbbox(text)
        if r - l < 4 or bt - tp < 4:
            return None
        im = Image.new("L", (r - l + 4, bt - tp + 4), 0)
        ImageDraw.Draw(im).text((2 - l, 2 - tp), text, font=f, fill=255)
        g = np.asarray(im)
        ys, xs = np.nonzero(g > 127)
        if len(xs) == 0:
            return None
        g = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)).resize((x1 - x0, y1 - y0), Image.BILINEAR)
        covs[weight] = _coverage(np.asarray(g) > 127)
    lo, hi = covs[400], covs[700]
    if hi - lo < 0.03:
        return None
    mid = 0.5 * (lo + hi)
    if abs(ref_cov - mid) < 0.15 * (hi - lo):
        return None
    return ref_cov > mid


def ring_is_textured(arr: np.ndarray, b: Box, pad: float) -> bool:
    """The surface around a text box is artwork (photo, illustration, gradient), not a flat fill."""
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (int(round(v)) for v in b)
    p = max(3, int(round(pad)))
    X0, Y0, X1, Y1 = max(0, x0 - p), max(0, y0 - p), min(W, x1 + p), min(H, y1 + p)
    parts = [arr[Y0:max(Y0, y0), X0:X1], arr[min(Y1, y1):Y1, X0:X1],
             arr[max(0, y0):min(H, y1), X0:max(X0, x0)], arr[max(0, y0):min(H, y1), min(X1, x1):X1]]
    ring = np.concatenate([q.reshape(-1, 3) for q in parts if q.size]) if any(q.size for q in parts) else None
    if ring is None or len(ring) < 30:
        return False
    q = np.round(ring.astype(np.float32) / 8.0)
    vals, counts = np.unique(q, axis=0, return_counts=True)
    mode = vals[int(np.argmax(counts))] * 8.0
    d = np.sqrt(((ring.astype(np.float32) - mode) ** 2).sum(axis=1))
    return float(np.percentile(d, 60)) > 18.0


def stylized_ink(arr: np.ndarray, b: Box) -> bool:
    """Lettering drawn in two or more distinct inks (multi-coloured logo letters, outlined game
    captions): a single-colour native font cannot show it."""
    H, W = arr.shape[:2]
    x0, y0, x1, y1 = (max(0, int(round(b[0]))), max(0, int(round(b[1]))),
                      min(W, int(round(b[2]))), min(H, int(round(b[3]))))
    if x1 - x0 < 6 or y1 - y0 < 6:
        return False
    px = arr[y0:y1, x0:x1].reshape(-1, 3).astype(np.float32)
    q = np.round(px / 8.0)
    vals, counts = np.unique(q, axis=0, return_counts=True)
    bg = vals[int(np.argmax(counts))] * 8.0
    fg = px[np.sqrt(((px - bg) ** 2).sum(axis=1)) > 70]
    if len(fg) < 40:
        return False
    fq = np.round(fg / 32.0)
    fv, fc = np.unique(fq, axis=0, return_counts=True)
    order = np.argsort(-fc)
    major = [fv[i] * 32.0 for i in order if fc[i] >= 0.2 * len(fg)]
    return len(major) >= 2 and max(np.sqrt(((a - c) ** 2).sum()) for a in major for c in major) >= 100


def ink_ratio(text: str) -> float:
    """Expected Vision box height / font size for this string (Roboto, calibrated)."""
    has_desc = bool(re.search(r"[gjpqy,;_()\[\]{}|/@$]", text))
    has_tall = bool(re.search(r"[A-Z0-9bdfhklt!?&%#'\"()\[\]{}|/\\@$]", text)) or bool(
        re.search(r"[^\x00-\x7f]", text))
    top = 0.75 if has_tall else 0.53
    return top + (0.21 if has_desc else 0.0) + 0.07


def fit_font_px(text: str, box: Box, lines: int = 1, tight: bool = False, weight: int = 400) -> float:
    """Font size (reference px) whose Roboto rendering spans this measured box."""
    bw = box[2] - box[0]
    bh = (box[3] - box[1]) / max(1, lines)
    pad = 0.0 if tight else 0.07
    by_h = bh / max(0.3, ink_ratio(text) - 0.07 + pad)
    if lines == 1 and len(text.strip()) >= 3:
        em = text_width_em(text) if weight == 400 else max(0.3, _vfont(weight, 100).getlength(text) / 100.0)
        by_w = bw / ((1.0 if tight else 1.05) * em)
        if 0.72 <= by_w / max(1e-6, by_h) <= 1.38:
            return by_w
        return min(by_w, by_h)
    return by_h


def is_vertical_line(l: dict) -> bool:
    """Text set at 90 degrees (rotated captures, side tabs): Vision returns a tall box."""
    b = l["box"]
    return len(tkey(l.get("ttext") or l["text"])) >= 3 and (b[2] - b[0]) < 0.5 * (b[3] - b[1])


def has_descender(text: str) -> bool:
    return bool(re.search(r"[gjpqy,;_()\[\]{}|@$]", text))


# --------------------------------------------------------------------------- XML model

def local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].rsplit(".", 1)[-1]


def aget(el, name):
    return el.get(A + name)


def aset(el, name, value):
    el.set(A + name, str(value))


def adel(el, *names):
    for name in names:
        el.attrib.pop(A + name, None)


def id_name(el) -> str | None:
    v = aget(el, "id")
    m = re.match(r"@\+?id/(\w+)$", (v or "").strip())
    return m.group(1) if m else None


@dataclass(eq=False)
class Node:
    el: ET.Element
    name: str
    tag: str
    parent: "Node | None"
    children: list = field(default_factory=list)
    text: str = ""
    role: str = "view"
    actual: Box | None = None
    pred: Box | None = None
    target: Box | None = None
    lines: list = field(default_factory=list)
    comp: dict | None = None
    font_px: float | None = None
    pitch: float | None = None
    remove: bool = False
    added: bool = False
    baked: bool = False
    measured: bool = False
    order: int = 0
    snapped_from: Box | None = None


def build_tree(root_el) -> tuple[Node, list[Node]]:
    nodes: list[Node] = []

    def rec(el, parent):
        n = Node(el=el, name=id_name(el) or "", tag=local(el.tag), parent=parent)
        n.text = (aget(el, "text") or "").strip()
        if not n.text and n.tag in FIELD_TAGS:
            n.text = (aget(el, "hint") or "").strip()
        n.order = len(nodes)
        nodes.append(n)
        for child in list(el):
            if local(child.tag) in NON_VIEW_TAGS:
                continue
            n.children.append(rec(child, n))
        if n.children or n.tag in SCROLL_TAGS or n.tag.endswith("Layout") or n.tag in FLATTEN_TAGS:
            n.role = "container"
        elif n.tag in IMAGE_TAGS:
            n.role = "image"
        elif n.tag in TEXT_TAGS:
            n.role = "text" if n.text else "widget"
        elif n.tag in WIDGET_TAGS:
            n.role = "widget"
        else:
            n.role = "view"
        return n

    return rec(root_el, None), nodes


def inject_ids(xml_text: str) -> str:
    root = ET.fromstring(xml_text)
    used: set[str] = set()
    k = 0
    for el in root.iter():
        if local(el.tag) in NON_VIEW_TAGS:
            continue
        name = id_name(el)
        if name and name not in used:
            used.add(name)
            continue
        while f"s2r_v{k}" in used:
            k += 1
        aset(el, "id", f"@+id/s2r_v{k}")
        used.add(f"s2r_v{k}")
    return '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding="unicode")


# --------------------------------------------------------------------------- matching

def tkey(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "").casefold()
    return re.sub(r"[\W_]+", "", s)


def tsim(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def parse_vh(path: Path) -> dict[str, Box]:
    boxes: dict[str, Box] = {}
    if not path.is_file():
        return boxes
    s = path.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"<node\b([^>]*)/?>", s):
        attrs = dict(re.findall(r'([\w:-]+)="([^"]*)"', m.group(1)))
        rid = attrs.get("resource-id", "")
        if not rid.startswith(PKG + ":id/"):
            continue
        b = re.match(r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]", attrs.get("bounds", ""))
        if not b:
            continue
        x0, y0, x1, y1 = (int(v) for v in b.groups())
        if x1 > x0 and y1 > y0:
            boxes.setdefault(rid.split("/", 1)[1], [x0, y0, x1, y1])
    return boxes


def robust_fit(src, dst, spread_min: float) -> tuple[float, float]:
    src = np.asarray(src, dtype=float)
    dst = np.asarray(dst, dtype=float)
    if len(src) == 0:
        return 1.0, 0.0
    keep = np.ones(len(src), dtype=bool)
    a, b = 1.0, float(np.median(dst - src))
    for _ in range(5):
        s, d = src[keep], dst[keep]
        if len(s) >= 3 and np.ptp(s) >= spread_min:
            a, b = np.polyfit(s, d, 1)
            if not 0.55 <= a <= 1.8:
                a = float(np.clip(a, 0.55, 1.8))
                b = float(np.median(d - a * s))
        else:
            a, b = 1.0, float(np.median(d - s))
        res = np.abs(dst - (a * src + b))
        thr = max(3.0 * float(np.median(res[keep])), 8.0)
        new = res <= thr
        if new.sum() < max(2, int(0.4 * len(src))) or (new == keep).all():
            break
        keep = new
    return float(a), float(b)


class Grounder:
    def __init__(self, sid: str, ref_png: Path, xml_text: str, vh_path: Path, screen_png: Path | None):
        self.sid = sid
        self.ref_png = ref_png
        with Image.open(ref_png) as im:
            self.img = im.convert("RGB")
        self.arr = np.asarray(self.img, dtype=np.float32)
        self.H, self.W = self.arr.shape[:2]
        self.frame = Frame(self.W, self.H)
        self.screen_w = CANVAS_W
        if screen_png is not None and Path(screen_png).is_file():
            with Image.open(screen_png) as im:
                self.screen_w = im.width
        self.root_el = ET.fromstring(xml_text)
        self.root, self.nodes = build_tree(self.root_el)
        self.ocr = load_ocr(ref_png, self.W, self.H)
        for line in self.ocr:
            line["tbox"], line["ttext"] = tight_text(self.arr, line["box"], line["text"])
        self.comps = load_components(sid, self.W, self.H)
        self.bands = detect_bands(self.arr, self.ocr, self.comps)
        vh = parse_vh(vh_path)
        s = self.W / float(self.screen_w)
        self.render_lines = []
        self.render_h_ref = float(self.H)
        self.render_bottom_color = None
        if screen_png is not None and Path(screen_png).is_file():
            with Image.open(screen_png) as im:
                sw, sh = im.size
                rarr = np.asarray(im.convert("RGB"), dtype=np.float32)
            self.render_h_ref = sh * s
            lo, hi = int(sh * 0.935), max(int(sh * 0.935) + 1, sh - 40)
            band = np.median(rarr[lo:hi].reshape(-1, 3), axis=0)
            self.render_bottom_color = [float(v) for v in band]
            for line in load_ocr(Path(screen_png), sw, sh):
                if line["box"][3] <= 66:  # the emulator's own status bar
                    continue
                tb, tt = tight_text(rarr, line["box"], line["text"])
                self.render_lines.append({"text": line["text"], "ttext": tt, "tbox": [v * s for v in tb]})
        for n in self.nodes:
            b = vh.get(n.name)
            if b is not None:
                n.actual = [b[0] * s, b[1] * s, b[2] * s, b[3] * s]
        self.report: dict = {"screen_id": sid, "ref_size": [self.W, self.H], "vh_boxes": len(vh),
                             "nodes": len(self.nodes), "ocr_lines": len(self.ocr), "components": len(self.comps),
                             "bands": self.bands}
        self.used_lines: set[int] = set()
        self.used_comps: set[int] = set()
        self.baked_lines: set[int] = set()
        self.shapes: dict[str, str] = {}
        self.cal = (1.0, 0.0, 1.0, 0.0)
        self.ymap = (np.array([0.0]), np.array([0.0]))
        self.anchor_y: list[tuple[float, float]] = []
        self.anchor_pts: list[tuple[float, float, float, float]] = []

    # -- helpers
    def in_status(self, b: Box) -> bool:
        return self.bands["status_bottom"] > 0 and b[3] <= self.bands["status_bottom"] + 2

    def in_nav(self, b: Box) -> bool:
        nav = self.bands["nav_top"]
        return nav is not None and b[1] >= nav - 2

    def content_part(self, b: Box) -> Box | None:
        """b without the system bands; None when nothing app-owned is left."""
        top = self.bands["status_bottom"] if self.bands["status_bottom"] > 0 else 0.0
        bottom = self.bands["nav_top"] if self.bands["nav_top"] is not None else float(self.H)
        out = [b[0], max(b[1], top), b[2], min(b[3], bottom)]
        return out if out[3] - out[1] >= 4 and out[2] - out[0] >= 4 else None

    def content_lines(self):
        return [l for l in self.ocr if not self.in_status(l["box"]) and not self.in_nav(l["box"])]

    def fy(self, y: float) -> float:
        src, dst = self.ymap
        if len(src) < 2:
            ax, bx, ay, by = self.cal
            return ay * y + by
        if y <= src[0]:
            return float(dst[0] + (y - src[0]) * ((dst[1] - dst[0]) / max(1e-6, src[1] - src[0])))
        if y >= src[-1]:
            return float(dst[-1] + (y - src[-1]) * ((dst[-1] - dst[-2]) / max(1e-6, src[-1] - src[-2])))
        return float(np.interp(y, src, dst))

    def build_ymap(self, pts: list[tuple[float, float]]):
        """Monotone piecewise-linear rendered-y -> reference-y through the matched anchors.

        One global scale cannot describe a layout whose header is right but whose body
        drifts, or whose footer is pinned to the bottom while the middle is stretched.
        """
        H = float(self.H)
        pts = [(float(a), float(b)) for a, b in pts]
        if len(pts) >= 3:
            a, b = robust_fit([p for p, _ in pts], [q for _, q in pts], 0.2 * H)
            res = [abs(q - (a * p + b)) for p, q in pts]
            thr = max(0.06 * H, 3.0 * float(np.median(res)))
            pts = [pq for pq, r in zip(pts, res) if r <= thr]
        pts.sort()
        merged: list[list[float]] = []
        for p, q in pts:
            if merged and p - merged[-1][0] < 6:
                merged[-1][1] = 0.5 * (merged[-1][1] + q)
            else:
                merged.append([p, q])
        # longest non-decreasing subsequence in reference order
        n = len(merged)
        best = [1] * n
        prev = [-1] * n
        for i in range(n):
            for j in range(i):
                if merged[j][1] <= merged[i][1] + 2 and best[j] + 1 > best[i]:
                    best[i], prev[i] = best[j] + 1, j
        chain = []
        if n:
            k = int(np.argmax(best))
            while k >= 0:
                chain.append(merged[k])
                k = prev[k]
            chain.reverse()
        nav_top = self.bands.get("nav_top")
        end = H
        if nav_top is not None and self.render_bottom_color is not None and \
                cdist(self.render_bottom_color, self.bands.get("nav_color") or [0, 0, 0]) > 60:
            end = float(nav_top)
        anchors = [[0.0, 0.0]] + [c for c in chain if 0 < c[0] < self.render_h_ref and 0 < c[1] < H] + \
                  [[self.render_h_ref, end]]
        src, dst = [], []
        for p, q in anchors:
            if src and (p <= src[-1] + 1 or q < dst[-1]):
                continue
            src.append(p)
            dst.append(q)
        self.ymap = (np.array(src), np.array(dst))
        self.report["ymap"] = [[round(a, 1), round(b, 1)] for a, b in zip(src, dst)]

    def tf(self, b: Box, role: str | None = None) -> Box:
        ax, bx, ay, by = self.cal
        box = [ax * b[0] + bx, self.fy(b[1]), ax * b[2] + bx, self.fy(b[3])]
        if self.anchor_pts and (box[2] - box[0]) < 0.6 * self.W:
            cx, cy = center(box)
            ws, dxs, dys = [], [], []
            radius = 0.25 * self.H
            for (px, py, rx, ry) in self.anchor_pts:
                d = math.hypot(px - cx, py - cy)
                if d <= radius:
                    wgt = 1.0 / (d * d + (0.04 * self.H) ** 2)
                    ws.append(wgt)
                    dxs.append(rx)
                    dys.append(ry)
            if ws:
                sw = sum(ws)
                dx = sum(wi * v for wi, v in zip(ws, dxs)) / sw
                # the drift is measured on text; an icon or picture only borrows a small share of it
                if role in ("image", "widget") and abs(dx) > 0.05 * self.W:
                    dx = 0.0
                if box[0] >= 0 and box[0] + dx < 0:
                    dx = -box[0]
                if box[2] <= self.W and box[2] + dx > self.W:
                    dx = self.W - box[2]
                box = [box[0] + dx, box[1], box[2] + dx, box[3]]
        return box

    # -- stage 1: anchors and drift
    def _render_pairs(self, lines) -> list[tuple[Box, Box]]:
        """(rendered text box, reference text box) for strings that occur on both screens."""
        by_ref, by_ren = defaultdict(list), defaultdict(list)
        for l in lines:
            by_ref[tkey(l["ttext"])].append(l)
        for l in self.render_lines:
            by_ren[tkey(l["ttext"])].append(l)
        pairs = []
        for key, rs in by_ren.items():
            fs = by_ref.get(key, [])
            if len(key) < 2 or not fs or len(rs) != len(fs):
                continue
            order = lambda l: (round(l["tbox"][1] / 8), l["tbox"][0])
            pairs += [(r["tbox"], f["tbox"]) for r, f in zip(sorted(rs, key=order), sorted(fs, key=order))]
        if len(pairs) < 3:
            used_r = {id(r) for r in self.render_lines if any(r["tbox"] is p for p, _ in pairs)}
            for r in self.render_lines:
                kr = tkey(r["ttext"])
                if id(r) in used_r or len(kr) < 3 or by_ref.get(kr):
                    continue
                best = max(((tsim(kr, tkey(l["ttext"])), l) for l in lines), key=lambda t: t[0], default=(0, None))
                if best[1] is not None and best[0] >= 0.85:
                    pairs.append((r["tbox"], best[1]["tbox"]))
        return pairs

    def calibrate(self):
        lines = self.content_lines()
        pairs = self._render_pairs(lines) if self.render_lines else []
        if len(pairs) >= 2:
            ax, bx = robust_fit([center(a)[0] for a, _ in pairs], [center(b)[0] for _, b in pairs], 0.2 * self.W)
            ay, by = robust_fit([center(a)[1] for a, _ in pairs], [center(b)[1] for _, b in pairs], 0.2 * self.H)
            self.cal = (ax, bx, ay, by)
            self.anchor_y = [(center(a)[1], center(b)[1]) for a, b in pairs]
            self.build_ymap(self.anchor_y)
            self.anchor_pts = []
            for a, b in pairs:
                c = center([ax * a[0] + bx, self.fy(a[1]), ax * a[2] + bx, self.fy(a[3])])
                lc = center(b)
                if abs(lc[0] - c[0]) < 0.2 * self.W and abs(lc[1] - c[1]) < 0.15 * self.H:
                    self.anchor_pts.append((c[0], c[1], lc[0] - c[0], lc[1] - c[1]))
            self.report["calibration"] = {"ax": ax, "bx": bx, "ay": ay, "by": by, "anchors": len(pairs),
                                          "local_anchors": len(self.anchor_pts), "source": "render_ocr"}
            for n in self.nodes:
                if n.actual is not None:
                    n.pred = self.tf(n.actual, n.role)
            return
        text_nodes = [n for n in self.nodes if n.role == "text" and tkey(n.text) and n.actual is not None]
        by_node = defaultdict(list)
        for n in text_nodes:
            by_node[tkey(n.text)].append(n)
        by_line = defaultdict(list)
        for l in lines:
            by_line[tkey(l["ttext"])].append(l)
        anchors = []
        for key, ns in by_node.items():
            ls = by_line.get(key, [])
            if len(key) >= 2 and len(ns) == 1 and len(ls) == 1:
                anchors.append((ns[0], ls[0]))
        if len(anchors) < 3:
            taken = {id(n) for n, _ in anchors} | {id(l) for _, l in anchors}
            for n in text_nodes:
                if id(n) in taken:
                    continue
                kn = tkey(n.text)
                best = max(((tsim(kn, tkey(l["ttext"])), l) for l in lines if id(l) not in taken),
                           key=lambda t: t[0], default=(0.0, None))
                if best[1] is not None and best[0] >= 0.88 and len(kn) >= 3:
                    anchors.append((n, best[1]))
                    taken |= {id(n), id(best[1])}
        xs = [(center(n.actual)[0], center(l["tbox"])[0]) for n, l in anchors]
        ys = [(center(n.actual)[1], center(l["tbox"])[1]) for n, l in anchors]
        ax, bx = robust_fit([p for p, _ in xs], [q for _, q in xs], 0.2 * self.W)
        ay, by = robust_fit([p for p, _ in ys], [q for _, q in ys], 0.2 * self.H)
        self.cal = (ax, bx, ay, by)
        self.anchor_y = list(ys)
        self.build_ymap(self.anchor_y)
        self.anchor_pts = []
        for n, l in anchors:
            c = center([ax * n.actual[0] + bx, self.fy(n.actual[1]), ax * n.actual[2] + bx, self.fy(n.actual[3])])
            lc = center(l["tbox"])
            if abs(lc[0] - c[0]) < 0.25 * self.W and abs(lc[1] - c[1]) < 0.2 * self.H:
                self.anchor_pts.append((c[0], c[1], lc[0] - c[0], lc[1] - c[1]))
        self.report["calibration"] = {"ax": ax, "bx": bx, "ay": ay, "by": by, "anchors": len(anchors),
                                      "local_anchors": len(self.anchor_pts)}
        for n in self.nodes:
            if n.actual is not None:
                n.pred = self.tf(n.actual, n.role)

    # -- stage 2: text
    def _ink_cols(self, b: Box) -> np.ndarray:
        x0, y0, x1, y1 = (int(round(v)) for v in clip(b, self.W, self.H))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return np.zeros(0, dtype=bool)
        sub = self.arr[y0:y1, x0:x1]
        bg, _ = mode_color(self.arr, b)
        return (np.sqrt(((sub - np.array(bg, dtype=np.float32)) ** 2).sum(axis=2)) > 45).any(axis=0)

    def _sub_line(self, line: dict, text: str) -> dict:
        """Part of an OCR line that belongs to one view (keypads read '2 ABC' as one line)."""
        full = line["ttext"]
        tb = line["tbox"]
        low = full.lower()
        i = low.find(text.lower())
        if i < 0:
            kf, kt = tkey(full), tkey(text)
            j = max(0, kf.find(kt))
            frac0, frac1 = j / max(1, len(kf)), (j + len(kt)) / max(1, len(kf))
        else:
            wt = text_width_em(full)
            frac0 = text_width_em(full[:i]) / wt if i else 0.0
            frac1 = min(1.0, frac0 + text_width_em(full[i:i + len(text)]) / wt)
        bw = tb[2] - tb[0]
        x0, x1 = tb[0] + frac0 * bw, tb[0] + frac1 * bw
        cols = self._ink_cols(tb)
        if cols.size:
            gaps = [k for k in range(1, len(cols)) if not cols[k] and cols[k - 1]]
            starts = [k for k in range(1, len(cols)) if cols[k] and not cols[k - 1]]
            if frac0 > 0 and starts:
                cand = min(starts, key=lambda k: abs(tb[0] + k - x0))
                if abs(tb[0] + cand - x0) <= 0.3 * bw:
                    x0 = tb[0] + cand
            if frac1 < 1 and gaps:
                cand = min(gaps, key=lambda k: abs(tb[0] + k - x1))
                if abs(tb[0] + cand - x1) <= 0.3 * bw and tb[0] + cand > x0 + 2:
                    x1 = tb[0] + cand
        sub, _ = tight_text(self.arr, [x0, tb[1], x1, tb[3]], text)
        return {"id": line["id"], "text": text, "ttext": text, "box": sub, "tbox": sub, "partial": True}

    def match_text(self):
        lines = self.content_lines()
        pending = [n for n in self.nodes if n.role == "text" and tkey(n.text)]
        pending.sort(key=lambda n: -len(tkey(n.text)))
        claim = defaultdict(int)
        for l in lines:
            kl = tkey(l["ttext"])
            claim[l["id"]] = sum(1 for n in pending if tkey(n.text) and tkey(n.text) in kl)
        for n in pending:
            kn = tkey(n.text)
            best = None
            for l in lines:
                kl = tkey(l["ttext"])
                if not kl:
                    continue
                inside = len(kn) >= 1 and kn in kl and len(kl) > 1.3 * len(kn)
                if l["id"] in self.used_lines and not (inside and claim[l["id"]] >= 2):
                    continue
                s = tsim(kn, kl)
                contain = (len(kl) >= 3 and kl in kn) or (len(kn) >= 2 and kn in kl)
                if s < 0.55 and not contain:
                    continue
                score = max(s, 0.8 if contain else 0.0)
                if n.pred is not None:
                    d = math.hypot(*(np.subtract(center(n.pred), center(l["tbox"])))) / self.H
                    score -= 1.6 * d
                if best is None or score > best[0]:
                    best = (score, l)
            if best is None or best[0] < 0.3:
                continue
            first = best[1]
            kl = tkey(first["ttext"])
            if len(kn) >= 1 and kn in kl and len(kl) > 1.3 * len(kn):
                if claim[first["id"]] <= 1 and first["id"] not in self.used_lines and \
                        kl.startswith(kn) and len(kn) >= 0.25 * len(kl):
                    # the model truncated a caption the screenshot shows in full
                    self.report.setdefault("text_extended", []).append([n.text, first["ttext"]])
                    n.text = first["ttext"]
                    aset(n.el, "text", first["ttext"])
                    kn = tkey(n.text)
                else:
                    n.lines = [self._sub_line(first, n.text)]
                    continue
            n.lines = [first]
            self.used_lines.add(first["id"])
            covered = len(kl)
            ub = list(first["tbox"])
            lh = first["tbox"][3] - first["tbox"][1]
            while covered < len(kn) - 1:
                nxt = None
                for l in lines:
                    if l["id"] in self.used_lines:
                        continue
                    kl2 = tkey(l["ttext"])
                    if not kl2 or kl2 not in kn:
                        continue
                    below = l["tbox"][1] - ub[3]
                    above = ub[1] - l["tbox"][3]
                    aligned = (abs(l["tbox"][0] - ub[0]) < 0.12 * self.W or
                               abs(center(l["tbox"])[0] - center(ub)[0]) < 0.12 * self.W)
                    gap = below if below > -0.3 * lh else above
                    if aligned and -0.3 * lh <= gap <= 1.6 * lh:
                        if nxt is None or gap < nxt[0]:
                            nxt = (gap, l)
                if nxt is None:
                    break
                l = nxt[1]
                n.lines.append(l)
                self.used_lines.add(l["id"])
                covered += len(tkey(l["ttext"]))
                ub = union([ub, l["tbox"]])
            n.lines.sort(key=lambda l: l["tbox"][1])
        self.report["text_matched"] = sum(1 for n in pending if n.lines)
        self.report["text_nodes"] = len(pending)

    # -- stage 3: targets
    def _line_union(self, n: Node) -> Box | None:
        return union([l["tbox"] for l in n.lines])

    def _font_from_lines(self, n: Node) -> float | None:
        if not n.lines:
            return None
        if len(n.lines) == 1:
            l = n.lines[0]
            kn, kl = tkey(n.text), tkey(l["ttext"])
            # the node's own words only when the line shows all of them; a longer caption wraps
            text = n.text if tsim(kn, kl) >= 0.75 and len(kn) <= 1.05 * len(kl) and "\n" not in n.text \
                else l["ttext"]
            return fit_font_px(text, l["tbox"], tight=True)
        per = [fit_font_px(l["ttext"], l["tbox"], tight=True) for l in n.lines]
        tops = sorted(l["tbox"][1] for l in n.lines)
        pitches = [b - a for a, b in zip(tops, tops[1:]) if b - a > 0]
        if pitches:
            n.pitch = float(np.median(pitches))
        return float(np.median(per))

    def _text_box(self, tb: Box, text: str, fpx: float, lines: int = 1, pitch: float | None = None) -> Box:
        width = max(tb[2] - tb[0], text_width_em(text) * fpx if lines == 1 else 0.0) * 1.03 + 0.1 * fpx + 2
        x0 = tb[0] - 0.05 * fpx - 1
        if lines == 1:
            base = tb[3] - (0.21 * fpx if has_descender(text) else 0.0)
            top = base - 0.928 * fpx - 1
            return [x0, top, x0 + width, top + ROBOTO_LINE * fpx + 2]
        top = tb[1] - (0.928 - 0.75) * fpx - 1
        return [x0, top, x0 + width * 1.02, tb[3] + 0.3 * fpx + 2]

    def _enclosing_comp(self, lb: Box, lh: float, max_h_mult: float, own: set | None = None) -> dict | None:
        best = None
        for c in self.comps:
            b = c["box"]
            if frac_inside(lb, expand(b, 0, 0, 3)) < 0.9:
                continue
            if (b[3] - b[1]) > max(max_h_mult * lh, 0.02 * self.H) or area(b) > 0.35 * self.W * self.H:
                continue
            if (b[2] - b[0]) < (lb[2] - lb[0]):
                continue
            # a box holding captions of other views is the row / toolbar, not this control
            if own is not None and any(l["id"] not in own and frac_inside(l["tbox"], b) >= 0.6
                                       for l in self.content_lines()):
                continue
            if best is None or area(b) < area(best["box"]):
                best = c
        return best

    def glyph_components(self) -> list[dict]:
        """OCR 'words' of one or two letters that no detector boxed are usually icons (a
        magnifier read as 'Q'); offered to image slots as component evidence."""
        out = []
        for l in self.content_lines():
            key = tkey(l["ttext"])
            if not (1 <= len(key) <= 2) or key.isdigit() or l["id"] in self.used_lines:
                continue
            b = l["box"]
            w, h = b[2] - b[0], b[3] - b[1]
            if h < 6 or not (0.5 <= w / max(h, 1.0) <= 2.0):
                continue
            if any(iou(b, c["box"]) >= 0.3 or frac_inside(b, c["box"]) >= 0.8 for c in self.comps):
                continue
            out.append({"box": list(b), "glyph": True, "line": l["id"]})
        return out

    def match_components(self):
        """Image/widget slots <-> UIED components; matched pairs become extra drift anchors."""
        extra = []
        self.glyph_comps = self.glyph_components()
        cand = [c for c in self.comps if not self.in_status(c["box"]) and not self.in_nav(c["box"])]
        texty = {id(c) for c in cand if any(frac_inside(l["tbox"], c["box"]) >= 0.6 for l in self.ocr)}
        cand = cand + self.glyph_comps
        slots = [n for n in self.nodes if n.role in ("image", "widget") and n.actual is not None]
        pairs = []
        for n in slots:
            for c in cand:
                if c.get("glyph") and n.role != "image":
                    continue
                if n.role == "widget" and id(c) in texty:
                    continue  # a box around a caption is a row/button, not the switch next to it
                if n.role == "image" and sum(1 for l in self.ocr if frac_inside(l["tbox"], c["box"]) >= 0.6) >= 2:
                    continue
                cb = c["box"]
                best_d = None
                for src in (n.pred, n.actual):
                    if src is None:
                        continue
                    ov = iou(src, cb)
                    dx = abs(center(src)[0] - center(cb)[0]) / max(src[2] - src[0], cb[2] - cb[0], 1.0)
                    dy = abs(center(src)[1] - center(cb)[1]) / max(src[3] - src[1], cb[3] - cb[1], 1.0)
                    ratio = area(cb) / max(1.0, area(src))
                    # a page-sized picture must not shrink onto a texture patch UIED boxed inside it
                    frac = area(src) / (self.W * self.H)
                    lo = 0.7 if frac >= 0.2 else (0.5 if frac >= 0.02 and n.role == "image" else 0.3)
                    if (ov >= 0.3 or (dx < 0.8 and dy < 0.8)) and lo <= ratio <= 3.0:
                        score = ov - 0.3 * (dx + dy)
                        best_d = score if best_d is None else max(best_d, score)
                if best_d is not None:
                    pairs.append((best_d, id(n), id(c), n, c))
        pairs.sort(key=lambda t: -t[0])
        taken_n, taken_c = set(), set()
        for score, nid, cid, n, c in pairs:
            if nid in taken_n or cid in taken_c:
                continue
            taken_n.add(nid)
            taken_c.add(cid)
            n.comp = c
            self.used_comps.add(cid)
            p = n.pred if n.pred is not None else n.actual
            pc = center(p)
            cc = center(c["box"])
            extra.append((pc[0], pc[1], cc[0] - pc[0], cc[1] - pc[1]))
        if extra:
            ys = []
            for (px, py, rx, ry), (score, nid, cid, n, c) in zip(extra, [t for t in pairs if t[1] in taken_n and t[2] in taken_c]):
                pass
            for n in slots:
                if n.comp is not None and n.actual is not None:
                    ys.append((center(n.actual)[1], center(n.comp["box"])[1]))
            self.build_ymap(self.anchor_y + ys)
            self.anchor_pts.extend((x, y, rx, 0.0) for x, y, rx, _ in extra)
            for n in self.nodes:
                if n.actual is not None:
                    n.pred = self.tf(n.actual, n.role)
        self.report["component_anchors"] = len(extra)

    def assign_targets(self):
        W, H = self.W, self.H
        self.match_components()
        for n in self.nodes:
            if n is self.root:
                n.target = [0.0, 0.0, float(W), float(H)]
                continue
            if n.role == "text" and n.lines:
                lb = self._line_union(n)
                n.font_px = self._font_from_lines(n)
                fpx = n.font_px or (lb[3] - lb[1])
                lh = max(lb[3] - lb[1], 0.7 * fpx) if len(n.lines) == 1 else fpx
                if n.tag in BUTTON_TAGS:
                    c = self._enclosing_comp(lb, lh, 4.5, {l["id"] for l in n.lines})
                    box = c["box"] if c else None
                    if box is None:
                        ring = expand(lb, 0, 0, 0.6 * lh)
                        fill, frac = mode_color(self.arr, ring)
                        page, _ = mode_color(self.arr, expand(lb, 0, 0, 2.5 * lh))
                        if fill is not None and frac > 0.3 and (page is None or cdist(fill, page) > 12):
                            seed = n.pred if n.pred is not None and contains_point(n.pred, center(lb)) else ring

                            def ok(b, lb=lb, lh=lh):
                                return (frac_inside(lb, expand(b, 0, 0, 2)) >= 0.9 and (b[3] - b[1]) <= 5 * lh
                                        and (b[2] - b[0]) <= 0.98 * W and area(b) >= 1.2 * area(lb))
                            box = snap_color_region(self.arr, seed, fill, accept=ok, grow=1.0)
                    if box is None and n.pred is not None and contains_point(n.pred, center(lb)) and \
                            (n.pred[3] - n.pred[1]) <= 5 * lh:
                        # the model's button, centred on the caption the reference shows
                        (pcx, pcy), (lcx, lcy) = center(n.pred), center(lb)
                        box = [n.pred[0] + lcx - pcx, n.pred[1] + lcy - pcy, n.pred[2] + lcx - pcx, n.pred[3] + lcy - pcy]
                    n.target = box if box is not None else expand(lb, 0, 0, 0.6 * fpx)
                    if c:
                        self.used_comps.add(id(c))
                        n.comp = c
                elif n.tag in FIELD_TAGS:
                    c = self._enclosing_comp(lb, lh, 4.0, {l["id"] for l in n.lines})
                    box = c["box"] if c else None
                    if box is None and n.pred is not None and contains_point(n.pred, center(lb)):
                        half = max(0.5 * (n.pred[3] - n.pred[1]), 0.9 * fpx)
                        box = [min(n.pred[0], lb[0]), center(lb)[1] - half, max(n.pred[2], lb[2]), center(lb)[1] + half]
                    n.target = box if box is not None else [lb[0] - 0.3 * fpx, lb[1] - 0.7 * fpx, lb[2] + 2 * fpx,
                                                            lb[3] + 0.7 * fpx]
                    if c:
                        self.used_comps.add(id(c))
                        n.comp = c
                elif n.tag in COMPOUND_TAGS:
                    cy = center(lb)[1]
                    half = max(0.5 * ROBOTO_LINE * fpx + 2, 0.5 * (lb[3] - lb[1]) + 2)
                    glyph = None
                    for c in self.comps:
                        b = c["box"]
                        if id(c) in self.used_comps:
                            continue
                        if abs(center(b)[1] - cy) < 0.8 * fpx and b[2] <= lb[0] + 4 and lb[0] - b[2] < 3 * fpx \
                                and (b[2] - b[0]) < 3.5 * fpx:
                            if glyph is None or b[2] > glyph["box"][2]:
                                glyph = c
                    if n.tag == "Switch":
                        x1 = n.pred[2] if n.pred is not None and n.pred[2] > lb[2] else lb[2] + 3.2 * fpx
                        n.target = [lb[0] - 0.1 * fpx, cy - half, x1, cy + half]
                    else:
                        if glyph:
                            self.used_comps.add(id(glyph))
                            half = max(half, 0.5 * (glyph["box"][3] - glyph["box"][1]))
                            x0 = glyph["box"][0]
                        else:
                            x0 = lb[0] - 2.0 * fpx
                        n.target = [x0, cy - half, lb[2] + 1.2 * fpx, cy + half]
                else:
                    n.target = self._text_box(lb, n.text if len(n.lines) == 1 else "", fpx, len(n.lines), n.pitch)
                continue
            if n.role == "image" and n.actual is not None and n.comp is None and \
                    (n.actual[2] - n.actual[0]) >= 0.95 * W and (n.actual[3] - n.actual[1]) >= 0.85 * H:
                n.target = [0.0, 0.0, float(W), float(H)]
                continue
            if n.role in ("image", "widget") and n.pred is not None:
                n.target = list(n.comp["box"]) if n.comp is not None else list(n.pred)
                continue
            if n.role == "container" and n.actual is not None and \
                    (n.actual[2] - n.actual[0]) >= 0.95 * W and (n.actual[3] - n.actual[1]) >= 0.9 * H:
                n.target = [0.0, 0.0, float(W), float(H)]
                continue
            if n.pred is not None:
                n.target = list(n.pred)
                bg_attr = aget(n.el, "background")
                bg = parse_color(bg_attr)
                if n.role in ("container", "view") and bg_attr and area(n.pred) < 0.9 * W * H:
                    snapped = None
                    # a bar drawn from y=0 by the model also covers the status band; the band is
                    # repainted at the end, so only the content part may seed or receive the snap
                    seed = self.content_part(n.pred)
                    if bg is not None and seed is not None:
                        fill, frac = mode_color(self.arr, seed)
                        if fill is not None and frac >= 0.3 and cdist(fill, bg) < 70:
                            snapped = snap_color_region(self.arr, seed, fill)
                            if snapped is not None and self.content_part(snapped) is None:
                                snapped = None
                    if snapped is None:
                        best = max(((iou(n.pred, c["box"]), c) for c in self.comps if id(c) not in self.used_comps),
                                   key=lambda t: t[0], default=(0.0, None))
                        if best[1] is not None and best[0] >= 0.6:
                            snapped = list(best[1]["box"])
                            self.used_comps.add(id(best[1]))
                    if snapped is not None:
                        n.target = snapped
                        n.snapped_from = list(seed) if seed is not None else list(n.pred)
        for n in self.nodes:
            if n is self.root or (n.role == "text" and n.lines) or n.comp is not None:
                n.measured = True
            elif n.target is not None and n.pred is not None and \
                    max(abs(a - b) for a, b in zip(n.target, n.pred)) > 1.0:
                n.measured = True
            elif n.target is not None and n.pred is None:
                n.measured = True
        self._guard_container_snaps()
        self._propagate_snaps(self.root, 0.0, 0.0)
        self._contain_read_captions()
        for n in self.nodes:
            if n.target is None and n.pred is not None:
                n.target = list(n.pred)
        for n in reversed(self.nodes):
            if n.target is None and n.role == "container":
                n.target = union([c.target for c in n.children if c.target is not None])
        for n in self.nodes:
            if n.target is not None:
                n.target = [float(v) for v in n.target]

    def _contain_read_captions(self):
        """A painted button/panel must lie under the caption it owns (the caption is measured).

        When the caption sits mostly outside its painted container, the container was placed
        on the wrong region: move it over the caption (keeping its size), or grow it to it.
        """
        fixes = self.report.setdefault("container_recentred", [])
        for n in self.nodes:
            if n is self.root or n.role != "container" or n.target is None or \
                    not paints_surface(aget(n.el, "background")):
                continue
            own = [self._line_union(m) for m in self.nodes if m.lines and m.role == "text" and
                   self._is_descendant(m, n) and m.target is not None]
            own = [b for b in own if b is not None]
            if not own:
                continue
            u = union(own)
            if frac_inside(u, n.target) >= 0.5:
                continue
            t = n.target
            w, h = t[2] - t[0], t[3] - t[1]
            if (u[2] - u[0]) <= w and (u[3] - u[1]) <= h:
                cx, cy = center(u)
                new = [t[0], cy - h / 2, t[2], cy + h / 2]
                if not (t[0] <= u[0] and u[2] <= t[2]):
                    new[0], new[2] = cx - w / 2, cx + w / 2
            else:
                new = expand(u, 0.1, 0.3, 4)
            fixes.append({"id": n.name, "from": [round(v, 1) for v in t], "to": [round(v, 1) for v in new]})
            dx, dy = new[0] - t[0], new[1] - t[1]
            n.target = [float(v) for v in new]
            for m in self.nodes:          # unmeasured decorations inside ride along
                if self._is_descendant(m, n) and not m.measured and m.target is not None:
                    m.target = [m.target[0] + dx, m.target[1] + dy, m.target[2] + dx, m.target[3] + dy]

    def _guard_container_snaps(self):
        """A colour snap must keep what the container owns and must not swallow its neighbours.

        Opaque absolute frames are drawn in sibling order, so a bar that grew over the row
        above it paints that row out.  The grown edge is pulled back to the foreign content
        it covered; content the container owns but the region missed is re-included.
        """
        fixes = self.report.setdefault("snap_guard", [])

        def box_of(m):
            return self._line_union(m) if m.lines else m.target

        content = [m for m in self.nodes if m.target is not None and m.role != "container"
                   and (m.lines or m.comp is not None)]
        for n in self.nodes:
            if n.snapped_from is None or n.target is None or n is self.root:
                continue
            t, seed = list(n.target), n.snapped_from
            own = [box_of(m) for m in content if self._is_descendant(m, n)]
            missed = [b for b in own if b is not None and frac_inside(b, expand(t, 0, 0, 3)) < 0.8]
            if missed:
                grown = union([t] + missed)
                t = grown if area(grown) <= 1.6 * area(t) else union([seed] + own)
            for m in content:
                if self._is_descendant(m, n) or self._is_descendant(n, m):
                    continue
                b = box_of(m)
                if b is None or frac_inside(b, t) < 0.5 or frac_inside(b, seed) >= 0.2:
                    continue
                if b[3] <= seed[1] + 2 and t[1] < seed[1]:
                    t[1] = max(t[1], b[3] + 1)
                elif b[1] >= seed[3] - 2 and t[3] > seed[3]:
                    t[3] = min(t[3], b[1] - 1)
                elif b[2] <= seed[0] + 2 and t[0] < seed[0]:
                    t[0] = max(t[0], b[2] + 1)
                elif b[0] >= seed[2] - 2 and t[2] > seed[2]:
                    t[2] = min(t[2], b[0] - 1)
            if t[2] - t[0] < 4 or t[3] - t[1] < 4:
                t = list(seed)
            if [round(v, 1) for v in t] != [round(v, 1) for v in n.target]:
                fixes.append({"id": n.name, "from": [round(v, 1) for v in n.target],
                              "to": [round(v, 1) for v in t]})
                n.target = [float(v) for v in t]

    DISPLAY_SP = 36.0

    def bake_display_text(self):
        """Logo / splash lettering is artwork: keep its pixels instead of a native font guess."""
        baked = 0
        for n in self.nodes:
            if n.remove or n.role != "text" or not n.lines or n.tag not in {"TextView", "CheckedTextView"}:
                continue
            vertical = all(is_vertical_line(l) for l in n.lines)
            styled = all(stylized_ink(self.arr, l["tbox"]) for l in n.lines)
            if not vertical and not styled and (n.font_px is None or self.frame.dp(n.font_px) <= self.DISPLAY_SP):
                continue
            lb = union([l["box"] for l in n.lines]) if vertical else self._line_union(n)
            el = n.el
            for key in list(el.attrib):
                if key.startswith(A) and key[len(A):] not in ("id", "layout_width", "layout_height"):
                    del el.attrib[key]
            el.tag = "ImageView"
            aset(el, "src", "@drawable/img")
            aset(el, "contentDescription", n.text)
            n.tag, n.role, n.baked, n.comp = "ImageView", "image", True, None
            n.target = clip(expand(lb, 0.04, 0.12, 4), self.W, self.H)
            for l in n.lines:
                self.baked_lines.add(l["id"])
            baked += 1
        self.report["display_text_baked"] = baked

    def _row_shift(self, n: Node):
        """Shift of the measured views sharing n's row: a better guess than the parent's,
        which moves with whatever colour region the container snapped to.

        Peers are siblings, or anywhere in the tree the same kind of view with the same height
        (a tab bar or badge row whose items are each wrapped in their own frame).  With peers on
        both sides the shift is interpolated by x, so a row the model spaced differently from
        the reference still lands item by item."""
        if n.pred is None:
            return None
        h = max(1.0, n.pred[3] - n.pred[1])
        cx = center(n.pred)[0]
        peers = []
        for s in self.nodes:
            if s is n or s.remove or not s.measured or s.pred is None or s.target is None or s.role == "container":
                continue
            sibling = s.parent is n.parent
            if not sibling and not (s.tag == n.tag and 0.7 <= (s.pred[3] - s.pred[1]) / h <= 1.4):
                continue
            ov = min(s.pred[3], n.pred[3]) - max(s.pred[1], n.pred[1])
            if ov >= 0.5 * min(h, max(1.0, s.pred[3] - s.pred[1])):
                cs, ct = center(s.pred), center(s.target)
                peers.append((cs[0], ct[0] - cs[0], ct[1] - cs[1], sibling))
        if not peers:
            return None
        left = [p for p in peers if p[0] <= cx]
        right = [p for p in peers if p[0] > cx]
        if left and right and not all(p[3] for p in peers):
            a, b = max(left, key=lambda p: p[0]), min(right, key=lambda p: p[0])
            t = (cx - a[0]) / max(1.0, b[0] - a[0])
            return a[1] + t * (b[1] - a[1]), a[2] + t * (b[2] - a[2])
        if not all(p[3] for p in peers):
            q = min(peers, key=lambda p: abs(p[0] - cx))
            return q[1], q[2]
        return float(np.median([d[1] for d in peers])), float(np.median([d[2] for d in peers]))

    def _propagate_snaps(self, n: Node, dx: float, dy: float):
        """Unmeasured views ride along with their nearest measured ancestor (keeps insets/borders)."""
        if n.measured and n.pred is not None and n.target is not None and n is not self.root:
            dx, dy = n.target[0] - n.pred[0], n.target[1] - n.pred[1]
        elif not n.measured and n.target is not None:
            row = self._row_shift(n)
            if row is not None:
                dx, dy = row
            if dx or dy:
                n.target = [n.target[0] + dx, n.target[1] + dy, n.target[2] + dx, n.target[3] + dy]
        for c in n.children:
            self._propagate_snaps(c, dx, dy)

    # -- stage 4: drop chrome, duplicates, unresolved
    def prune(self):
        removed = defaultdict(int)
        big = BIG_IMAGE_FRAC * self.W * self.H
        band_lines = [l for l in self.ocr if self.in_status(l["box"]) or self.in_nav(l["box"])]

        details = self.report.setdefault("removed_nodes", [])

        def kill(n, why):
            if not n.remove:
                n.remove = True
                removed[why] += 1
                details.append({"id": n.name, "tag": n.tag, "text": n.text, "reason": why})
            for c in n.children:
                kill(c, why)

        for n in self.nodes:
            if n is self.root or n.remove:
                continue
            if n.target is None:
                kill(n, "unresolved")
                continue
            chrome_text = n.role == "text" and not n.lines and (
                any(tsim(tkey(n.text), tkey(l["ttext"])) >= 0.7 for l in band_lines) or
                (re.fullmatch(r"\s*\d{1,2}[:.]\d{2}\s*(am|pm)?\s*", n.text, re.I) is not None and
                 n.target[1] < 0.08 * self.H))
            if n.role == "container" and self.in_status(n.target):
                owned = [m.target for m in self.nodes if not m.remove and m.target is not None and
                         m.role != "container" and (m.lines or m.comp is not None) and
                         self._is_descendant(m, n) and not self.in_status(m.target)]
                if owned:
                    n.target = union(owned)
                    self.report.setdefault("kept_out_of_status_band", []).append(n.name)
                    continue
            if self.in_status(n.target) or chrome_text or \
                    (n.lines and all(self.in_status(l["box"]) for l in n.lines)):
                kill(n, "status_bar")
            elif n.role != "container" and self.bands["nav_top"] is not None and \
                    center(n.target)[1] >= self.bands["nav_top"] - 2:
                kill(n, "nav_bar")
            elif area(clip(n.target, self.W, self.H)) < 1:
                kill(n, "offscreen")
        crops = [m for m in self.nodes if not m.remove and m.target is not None and
                 (m.role == "image" or (aget(m.el, "background") or "").startswith("@drawable/img"))]
        for n in self.nodes:
            if n.remove or n.role != "text" or n.lines or n.target is None:
                continue
            if any(frac_inside(n.target, m.target) >= 0.9 for m in crops if m is not n):
                kill(n, "unread_text_over_image")
        # a view whose text is the concatenation of finer views on the same line is a duplicate
        texts = [n for n in self.nodes if n.role == "text" and n.lines and not n.remove]
        for a in texts:
            if a.remove:
                continue
            ka = tkey(a.text)
            inner = [b for b in texts if b is not a and not b.remove and tkey(b.text) and tkey(b.text) in ka
                     and len(tkey(b.text)) < len(ka)
                     and frac_inside(self._line_union(b), expand(self._line_union(a), 0, 0, 3)) >= 0.8]
            if inner and sum(len(tkey(b.text)) for b in inner) >= 0.5 * len(ka):
                kill(a, "merged_duplicate")
        images = [n for n in self.nodes if n.role == "image" and not n.remove and n.target is not None
                  and area(n.target) < big]
        for n in self.nodes:
            if n.remove or n.role != "text" or not n.lines:
                continue
            lb = self._line_union(n)
            for im in images:
                if frac_inside(lb, expand(im.target, 0, 0, 2)) >= 0.85 and im is not n:
                    lines_in = sum(1 for l in self.ocr if frac_inside(l["tbox"], im.target) >= 0.6)
                    if area(im.target) > 4.0 * area(lb) or lines_in > 2:
                        continue  # a card/banner/dialog image: the caption stays native, inpainted from the crop
                    kill(n, "text_inside_image")
                    im.baked = True
                    grown = union([im.target, lb])
                    if area(grown) <= 1.6 * area(im.target):
                        im.target = grown
                    for l in n.lines:
                        if not l.get("partial"):
                            self.used_lines.discard(l["id"])
                        self.baked_lines.add(l["id"])
                    break
        self.report["removed"] = dict(removed)

    # -- stage 5: add what the model missed
    def complete(self):
        added_text, added_img = 0, 0
        big = BIG_IMAGE_FRAC * self.W * self.H
        live = [n for n in self.nodes if not n.remove and n.target is not None]
        small_images = [n for n in live if n.role == "image" and area(n.target) < big]
        big_images = [n for n in live if n.role == "image" and area(n.target) >= big]
        partial_cover = defaultdict(float)
        for n in live:
            for l in n.lines:
                if l.get("partial"):
                    partial_cover[l["id"]] += (l["tbox"][2] - l["tbox"][0])
        for l in self.content_lines():
            if l["id"] in self.used_lines or l["id"] in self.baked_lines:
                continue
            tw = l["tbox"][2] - l["tbox"][0]
            if partial_cover.get(l["id"], 0.0) >= 0.4 * tw:
                continue
            if any(frac_inside(l["tbox"], expand(im.target, 0, 0, 2)) >= 0.7 for im in small_images):
                continue
            # a view the model placed on this line but captioned differently: the screenshot's words win
            owners = [n for n in live if n.role == "text" and not n.lines and not n.added and
                      n.tag not in FIELD_TAGS and n.target is not None and frac_inside(l["tbox"], n.target) > 0.6]
            n = max(owners, key=lambda m: iou(m.target, l["tbox"])) if owners else None
            # one- or two-glyph reads inside badges are the least reliable OCR output: they only
            # replace a caption the model left empty
            short = len(tkey(l["ttext"])) < 3 and n is not None and bool(tkey(n.text))
            if n is not None and len(tkey(l["ttext"])) >= 2 and not short and not is_vertical_line(l):
                self.report.setdefault("text_adopted", []).append([n.name, n.text, l["ttext"]])
                n.text = l["ttext"]
                aset(n.el, "text", l["ttext"])
                n.lines = [l]
                n.font_px = fit_font_px(l["ttext"], l["tbox"], tight=True)
                if n.tag not in BUTTON_TAGS and n.tag not in COMPOUND_TAGS:
                    n.target = self._text_box(l["tbox"], l["ttext"], n.font_px)
                n.measured = True
                self.used_lines.add(l["id"])
                continue
            if any(frac_inside(l["tbox"], n.target) > 0.6 and n.role == "text" for n in live):
                continue
            if len(tkey(l["ttext"])) < 1:
                continue
            if len(tkey(l["ttext"])) <= 2 and any(iou(l["tbox"], c["box"]) >= 0.35 or
                                                  frac_inside(c["box"], expand(l["tbox"], 0.3, 0.3, 4)) >= 0.8
                                                  for c in self.comps):
                continue  # e.g. a magnifier read as 'Q': the component completion adds it as an image
            fpx = fit_font_px(l["ttext"], l["tbox"], tight=True)
            glyph = any(g.get("line") == l["id"] for g in getattr(self, "glyph_comps", []))
            vertical = is_vertical_line(l)
            if vertical:
                glyph = True     # the crop is the read box itself, like a glyph
            # words printed on artwork (banner, illustration, logo): a native caption would
            # have to guess font, outline and colour and inherits every OCR misread
            on_art = ring_is_textured(self.arr, l["tbox"], 0.6 * (l["tbox"][3] - l["tbox"][1])) or \
                stylized_ink(self.arr, l["tbox"])
            if self.frame.dp(fpx) > self.DISPLAY_SP or glyph or on_art:
                if any(frac_inside(l["tbox"], im.target) > 0.8 for im in big_images):
                    continue
                el = ET.Element("ImageView")
                name = f"s2r_d{added_text}"
                aset(el, "id", f"@+id/{name}")
                aset(el, "src", "@drawable/img")
                aset(el, "contentDescription", l["ttext"])
                parent = self._deepest_container(center(l["tbox"]))
                src_box = union([l["tbox"], l["box"]]) if glyph else l["tbox"]
                n = Node(el=el, name=name, tag="ImageView", parent=parent, role="image", baked=True,
                         target=clip(expand(src_box, 0.04, 0.12, 4 if not glyph else 3), self.W, self.H),
                         added=True, order=len(self.nodes))
                parent.el.append(el)
                parent.children.append(n)
                self.nodes.append(n)
                self.baked_lines.add(l["id"])
                self.report.setdefault("added_nodes", []).append(
                    {"id": name, "kind": "icon_glyph" if glyph else ("art_text" if on_art else "display_text"),
                     "text": l["ttext"], "source": "reference_ocr"})
                added_text += 1
                continue
            box = self._text_box(l["tbox"], l["ttext"], fpx)
            el = ET.Element("TextView")
            aset(el, "id", f"@+id/s2r_c{added_text}")
            aset(el, "text", l["ttext"])
            color = ink_color(self.arr, l["tbox"]) or [0, 0, 0]
            aset(el, "textColor", hex_color(color))
            parent = self._deepest_container(center(l["tbox"]))
            n = Node(el=el, name=f"s2r_c{added_text}", tag="TextView", parent=parent, text=l["ttext"], role="text",
                     target=box, lines=[l], font_px=fpx, added=True, order=len(self.nodes))
            parent.el.append(el)
            parent.children.append(n)
            self.nodes.append(n)
            self.used_lines.add(l["id"])
            self.report.setdefault("added_nodes", []).append(
                {"id": n.name, "kind": "text", "text": l["ttext"], "source": "reference_ocr"})
            added_text += 1
        live = [n for n in self.nodes if not n.remove and n.target is not None]
        leaves = [n for n in live if n.role != "container"]
        for c in self.comps:
            if id(c) in self.used_comps:
                continue
            b = c["box"]
            a = area(b)
            if self.in_status(b) or self.in_nav(b):
                continue
            if not (0.0004 * self.W * self.H <= a <= 0.2 * self.W * self.H) or min(b[2] - b[0], b[3] - b[1]) < 8:
                continue
            if any(frac_inside(l["box"], b) > 0.5 for l in self.ocr):
                continue
            def occupied(n):
                # a captioned view owns its caption, not the whole painted button around it
                if n.role == "text" and n.lines:
                    return [expand(l["tbox"], 0.02, 0.2, 2) for l in n.lines]
                return [expand(n.target, 0, 0, 2)]
            if any(frac_inside(b, ob) > 0.5 or frac_inside(ob, b) > 0.5 for n in leaves for ob in occupied(n)):
                continue
            if any(frac_inside(b, im.target) > 0.9 for im in big_images):
                continue
            px = _pixels(self.arr, b)
            if len(px) == 0 or float(px.std(axis=0).mean()) < 12:
                continue
            el = ET.Element("ImageView")
            name = f"s2r_i{added_img}"
            aset(el, "id", f"@+id/{name}")
            aset(el, "src", "@drawable/img")
            parent = self._deepest_container(center(b))
            n = Node(el=el, name=name, tag="ImageView", parent=parent, role="image", target=list(b), comp=c,
                     added=True, order=len(self.nodes))
            parent.el.append(el)
            parent.children.append(n)
            self.nodes.append(n)
            self.used_comps.add(id(c))
            leaves.append(n)
            self.report.setdefault("added_nodes", []).append(
                {"id": name, "kind": "image", "box": [round(v, 1) for v in b], "source": "reference_uied"})
            added_img += 1
        self.report["added_text"] = added_text
        self.report["added_images"] = added_img

    def _deepest_container(self, p) -> Node:
        best = self.root
        best_depth = 0
        for n in self.nodes:
            if n.remove or n.role != "container" or n.target is None or n.tag in SCROLL_TAGS:
                continue
            if contains_point(n.target, p):
                depth = 0
                q = n
                while q.parent is not None:
                    depth += 1
                    q = q.parent
                if depth >= best_depth and area(n.target) <= area(best.target or [0, 0, self.W, self.H]):
                    best, best_depth = n, depth
        return best

    def _lca_children(self, a: Node, b: Node):
        pa, pb = [], []
        q = a
        while q is not None:
            pa.append(q)
            q = q.parent
        q = b
        while q is not None:
            pb.append(q)
            q = q.parent
        common = next((x for x in pa if x in pb), None)
        if common is None:
            return None, None, None
        ca = pa[pa.index(common) - 1] if pa.index(common) > 0 else None
        cb = pb[pb.index(common) - 1] if pb.index(common) > 0 else None
        return common, ca, cb

    def text_above_images(self):
        """Absolute FrameLayouts make sibling order pure z-order: draw crops below the captions on them."""
        moved = 0
        live = [n for n in self.nodes if not n.remove and n.target is not None]
        images = [n for n in live if n.role == "image" or (aget(n.el, "background") or "").startswith("@drawable/img")]
        # an opaque box that clips even part of a read caption hides real text: text goes on top
        opaque = {id(n) for n in live if n.role in ("container", "view") and n not in images and
                  paints_surface(aget(n.el, "background"))}
        # a native switch/slider on a newer platform draws its thumb past its own bounds
        # (min size 48dp), so a read caption next to it must be drawn after it
        widgets = {id(n) for n in live if n.role == "widget"}
        boxes = images + [n for n in live if id(n) in opaque or id(n) in widgets]
        texts = [n for n in live if n.role in ("text", "widget")]
        for t in texts:
            tb = self._line_union(t) if t.lines else t.target
            for im in boxes:
                if id(im) in widgets:
                    if not t.lines or t.role == "widget":
                        continue
                    h = im.target[3] - im.target[1]
                    area_box = expand(im.target, 0, 0, max(4.0, 0.6 * h))
                    need = 0.02
                else:
                    area_box = im.target
                    need = 0.1 if (id(im) in opaque and t.lines) else 0.5
                if im is t or self._is_descendant(t, im) or frac_inside(tb, area_box) < need:
                    continue
                common, ci, ct = self._lca_children(im, t)
                if common is None or ci is None or ct is None or ci is ct:
                    continue
                kids = common.children
                if kids.index(ci) > kids.index(ct):
                    kids.remove(ci)
                    kids.insert(kids.index(ct), ci)
                    els = list(common.el)
                    if ci.el in els and ct.el in els:
                        common.el.remove(ci.el)
                        common.el.insert(list(common.el).index(ct.el), ci.el)
                    moved += 1
        self.report["z_reordered"] = self.report.get("z_reordered", 0) + moved

    def images_above_foreign_fills(self):
        """A reference crop the screenshot shows must not sit under a flat box it does not belong to.

        Runs after binding, when image boxes are final.  The overlap is judged on the reference
        itself: if the pixels there are not the box's fill colour, the image was on top.
        """
        moved = 0
        live = [n for n in self.nodes if not n.remove and n.target is not None]
        crops = [n for n in live if n.role == "image" and (aget(n.el, "src") or aget(n.el, "background") or "")
                 .startswith("@drawable/s2g")]
        fills = []
        for n in live:
            bg = aget(n.el, "background") or ""
            if n.role not in ("container", "view") or not bg or bg.startswith("@drawable/s2g") \
                    or bg == "@drawable/img" or "transparent" in bg:
                continue
            color = parse_color(bg)
            if color is None:          # a shape drawable paints what the box shows in the reference
                color, frac = mode_color(self.arr, n.target)
                if color is None or frac < 0.3:
                    continue
            fills.append((n, color))
        own_fill = {id(n): c for n, c in fills}
        # a painted panel (toolbar, card) the reference shows is just as visible as a crop
        items = crops + [n for n, _ in fills if n is not self.root]
        for im in items:
            ia = area(im.target)
            if ia <= 0:
                continue
            for box, fill in fills:
                if box is im or box is self.root or self._is_descendant(im, box) or self._is_descendant(box, im):
                    continue
                ov = [max(im.target[0], box.target[0]), max(im.target[1], box.target[1]),
                      min(im.target[2], box.target[2]), min(im.target[3], box.target[3])]
                if ov[2] - ov[0] < 2 or ov[3] - ov[1] < 2 or area(ov) < 0.05 * ia:
                    continue
                # the box's own captions are painted on it: judge its surface, not its text
                x0, y0, x1, y1 = (int(round(v)) for v in ov)
                x0, y0 = max(0, x0), max(0, y0)
                x1, y1 = min(self.arr.shape[1], x1), min(self.arr.shape[0], y1)
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                keep = np.ones((y1 - y0, x1 - x0), dtype=bool)
                for l in self.ocr:
                    b = l["box"]
                    lx0, ly0 = int(max(x0, b[0])) - x0, int(max(y0, b[1])) - y0
                    lx1, ly1 = int(min(x1, b[2])) - x0, int(min(y1, b[3])) - y0
                    if lx1 > lx0 and ly1 > ly0:
                        keep[ly0:ly1, lx0:lx1] = False
                if keep.mean() < 0.5:
                    continue
                seen = np.median(self.arr[y0:y1, x0:x1][keep].reshape(-1, 3), axis=0)
                if cdist(seen, fill) <= 35:
                    continue
                if id(im) in own_fill and cdist(seen, own_fill[id(im)]) > 25:
                    continue
                common, cb, ci = self._lca_children(box, im)
                if common is None or cb is None or ci is None or cb is ci:
                    continue
                kids = common.children
                if kids.index(cb) > kids.index(ci):
                    kids.remove(cb)
                    kids.insert(kids.index(ci), cb)
                    els = list(common.el)
                    if cb.el in els and ci.el in els:
                        common.el.remove(cb.el)
                        common.el.insert(list(common.el).index(ci.el), cb.el)
                    moved += 1
        self.report["images_raised"] = moved

    def grow_containers(self):
        """A container without paint is only a coordinate frame: make it enclose its children."""
        for n in reversed(self.nodes):
            if n.remove or n is self.root or n.role != "container" or n.target is None:
                continue
            if aget(n.el, "background") or n.tag in SCROLL_TAGS:
                continue
            kids = [c.target for c in n.children if not c.remove and c.target is not None]
            if kids:
                n.target = union([n.target] + kids)

    # -- stage 6: write geometry
    def rewrite(self):
        fr = self.frame
        for n in self.nodes:
            if n.remove:
                continue
            el = n.el
            adel(el, "fitsSystemWindows", "rotation", "rotationX", "rotationY", "scaleX", "scaleY",
                 "translationX", "translationY", "translationZ", "transformPivotX", "transformPivotY")
            if n is self.root:
                if n.tag not in SCROLL_TAGS:
                    el.tag = "FrameLayout"
                    adel(el, *CONTAINER_ONLY_ATTRS)
                aset(el, "layout_width", "match_parent")
                aset(el, "layout_height", "match_parent")
                adel(el, *PADDING_ATTRS)
                aset(el, "clipChildren", "false")
                aset(el, "clipToPadding", "false")
                continue
            p = n.parent
            while p is not None and p.remove:
                p = p.parent
            pt = p.target if p is not None and p.target is not None else [0.0, 0.0, float(self.W), float(self.H)]
            t = n.target
            adel(el, *CHILD_LAYOUT_ATTRS)
            aset(el, "layout_width", f"{max(0.5, fr.dp(t[2] - t[0])):.1f}dp")
            aset(el, "layout_height", f"{max(0.5, fr.dp(t[3] - t[1])):.1f}dp")
            aset(el, "layout_gravity", "top|start")
            aset(el, "layout_marginStart", f"{fr.dp(t[0] - pt[0]):.1f}dp")
            aset(el, "layout_marginLeft", f"{fr.dp(t[0] - pt[0]):.1f}dp")
            aset(el, "layout_marginTop", f"{fr.dp(t[1] - pt[1]):.1f}dp")
            if n.role == "container":
                if n.tag in FLATTEN_TAGS or (n.tag not in SCROLL_TAGS and n.tag != "FrameLayout"):
                    el.tag = "FrameLayout"
                adel(el, *CONTAINER_ONLY_ATTRS)
                adel(el, *PADDING_ATTRS)
                aset(el, "clipChildren", "false")
                aset(el, "clipToPadding", "false")
            elif n.role == "text":
                self._style_text(n)
            elif n.role == "image":
                adel(el, "tint", "tintMode", "adjustViewBounds", "cropToPadding", "maxWidth", "maxHeight",
                     "backgroundTint", *PADDING_ATTRS)
                aset(el, "scaleType", "fitXY")
                if n.tag == "ImageButton" and aget(el, "background") in (None, "@drawable/img"):
                    aset(el, "background", "@android:color/transparent")
            elif n.role == "widget" and n.tag in COMPOUND_TAGS | {"SeekBar", "ProgressBar", "RatingBar"}:
                aset(el, "minWidth", "0dp")
                aset(el, "minHeight", "0dp")
                if parse_color(aget(el, "background")) is not None:
                    adel(el, "background")
        self._reorder_tree()
        self._system_bands()

    def _style_text(self, n: Node):
        el = n.el
        fr = self.frame
        if n.font_px is None:
            m = re.fullmatch(r"\s*([\d.]+)\s*(sp|dp|px)?\s*", aget(el, "textSize") or "")
            if m:
                sp = float(m.group(1)) * (self.cal[2] if 0.55 <= self.cal[2] <= 1.8 else 1.0)
                aset(el, "textSize", f"{sp:.1f}sp")
            return
        weight = None
        if n.lines and len(n.lines) == 1 and n.tag not in FIELD_TAGS:
            lb0 = self._line_union(n)
            bg0, _ = mode_color(self.arr, expand(lb0, 0, 0, 3))
            weight = text_weight(self.arr, lb0, n.lines[0]["ttext"], ink_color(self.arr, lb0), bg0)
            if weight is not None and weight != 400:
                n.font_px = fit_font_px(n.lines[0]["ttext"], lb0, tight=True, weight=weight)
        sp = fr.dp(n.font_px)
        adel(el, *TEXT_SIZE_ATTRS)
        aset(el, "textSize", f"{sp:.1f}sp")
        if weight is not None:
            family = (aget(el, "fontFamily") or "").strip()
            if weight == 500:
                aset(el, "fontFamily", "sans-serif-medium")
                adel(el, "textStyle")
            else:
                if family in ("", "sans-serif", "sans-serif-medium", "sans"):
                    aset(el, "fontFamily", "sans-serif")
                if weight == 700:
                    aset(el, "textStyle", "bold")
                else:
                    adel(el, "textStyle")
            self.report.setdefault("text_weight", {})[n.name] = weight
        aset(el, "includeFontPadding", "false")
        aset(el, "minWidth", "0dp")
        aset(el, "minHeight", "0dp")
        grav = (aget(el, "gravity") or "").lower()
        if len(n.lines) <= 1:
            aset(el, "maxLines", "1")
            lo = min(max(4.0, 0.55 * sp), sp - 0.6)
            if lo >= 2.0 and n.tag not in COMPOUND_TAGS:
                aset(el, "autoSizeTextType", "uniform")
                aset(el, "autoSizeMinTextSize", f"{lo:.1f}sp")
                aset(el, "autoSizeMaxTextSize", f"{sp:.1f}sp")
                aset(el, "autoSizeStepGranularity", "1px")
        elif n.pitch:
            extra = n.pitch - ROBOTO_LINE * n.font_px
            aset(el, "lineSpacingExtra", f"{fr.dp(extra):.1f}dp")
        if n.tag in BUTTON_TAGS:
            adel(el, *PADDING_ATTRS)
            aset(el, "padding", "0dp")
            aset(el, "gravity", "center")
            aset(el, "stateListAnimator", "@null")
            if n.text != n.text.upper():
                aset(el, "textAllCaps", "false")
            self._fill_background(n)
        elif n.tag in FIELD_TAGS:
            adel(el, *PADDING_ATTRS)
            lb = self._line_union(n)
            aset(el, "paddingStart", f"{max(0.0, fr.dp(lb[0] - n.target[0])):.1f}dp")
            aset(el, "paddingLeft", f"{max(0.0, fr.dp(lb[0] - n.target[0])):.1f}dp")
            aset(el, "paddingTop", "0dp")
            aset(el, "paddingBottom", "0dp")
            aset(el, "paddingEnd", "0dp")
            aset(el, "gravity", "center_vertical|start")
            par = n.parent
            while par is not None and par.remove:
                par = par.parent
            if par is not None and par is not self.root and parse_color(aget(par.el, "background")) is not None \
                    and par.target is not None and frac_inside(n.target, expand(par.target, 0, 0, 4)) >= 0.9:
                aset(el, "background", "@android:color/transparent")
            elif n.measured and n.comp is not None:
                self._fill_background(n)
        elif n.tag in COMPOUND_TAGS:
            adel(el, *PADDING_ATTRS)
        else:
            adel(el, *PADDING_ATTRS)
            if "center" in grav and "vertical" not in grav.replace("center_vertical", ""):
                horiz = "center_horizontal" if ("center_horizontal" in grav or grav.strip() == "center") else "start"
            else:
                horiz = "end" if ("end" in grav or "right" in grav) else "start"
            vert = "center_vertical" if len(n.lines) <= 1 else "top"
            aset(el, "gravity", f"{vert}|{horiz}")
        if os.environ.get("S2R_TEXT_COLOR", "core") == "appearance":
            color = appearance_color(self.arr, self._line_union(n)) or ink_color(self.arr, self._line_union(n))
        else:
            color = ink_color(self.arr, self._line_union(n))
        model = parse_color(aget(el, "textColor"))
        if color is not None and (model is None or cdist(model, color) > 12):
            aset(el, "textColor", hex_color(color))

    def _fill_background(self, n: Node):
        """Measured surface: flat colour, or a native shape when the reference has corners/stroke."""
        bg = aget(n.el, "background")
        if bg and bg.startswith("@drawable/") and bg != "@drawable/img":
            return
        st = surface_style(self.arr, n.target)
        if st is None or st["fill_frac"] < 0.35:
            return
        fill = st["fill"]
        lb = self._line_union(n) if n.lines else None
        if lb is not None and cdist(fill, ink_color(self.arr, lb) or [-999, -999, -999]) < 30:
            return
        model = parse_color(bg)
        rounded = st["radius_px"] >= 4
        if not rounded and st["stroke"] is None:
            if model is None or cdist(model, fill) > 6:
                aset(n.el, "background", hex_color(fill))
            return
        fr = self.frame
        name = f"s2gshape{hashlib.sha1((self.sid + n.name).encode()).hexdigest()[:8]}"
        parts = [f'<shape xmlns:android="{ANDROID}" android:shape="rectangle">',
                 f'  <solid android:color="{hex_color(fill)}"/>']
        if st["stroke"] is not None:
            parts.append(f'  <stroke android:width="{max(0.5, fr.dp(st["stroke_px"])):.1f}dp" '
                         f'android:color="{hex_color(st["stroke"])}"/>')
        if rounded:
            parts.append(f'  <corners android:radius="{fr.dp(st["radius_px"]):.1f}dp"/>')
        parts.append("</shape>")
        self.shapes[name] = "\n".join(parts) + "\n"
        aset(n.el, "background", f"@drawable/{name}")

    def _art_score(self, n: Node) -> float | None:
        """How far the panel's exposed surface is from one flat colour (60th pct RGB distance).

        Exposed = inside the panel, outside everything its own descendants draw.  A flat card
        or bar scores ~0-5; a glossy button, parchment, gradient or icon art scores >18.
        """
        t = clip(n.target, self.W, self.H)
        a = area(t)
        if a < 0.002 * self.W * self.H or a > 0.45 * self.W * self.H:
            return None
        if (t[2] - t[0]) >= 0.97 * self.W and (t[3] - t[1]) >= 0.5 * self.H:
            return None
        x0, y0, x1, y1 = (int(round(v)) for v in t)
        if x1 - x0 < 10 or y1 - y0 < 10:
            return None
        keep = np.ones((y1 - y0, x1 - x0), dtype=bool)
        own_caption = [self._line_union(n)] if n.lines else []
        for m in [None] + self.nodes:
            if m is None:
                if not own_caption:
                    continue
                b = own_caption[0]
            else:
                if m.remove or m is n or m.target is None or not self._is_descendant(m, n):
                    continue
                if m.role == "container" and not paints_surface(aget(m.el, "background")):
                    continue
                b = self._line_union(m) if m.lines else m.target
            bx0, by0 = int(max(x0, b[0])) - x0, int(max(y0, b[1])) - y0
            bx1, by1 = int(min(x1, b[2])) - x0, int(min(y1, b[3])) - y0
            if bx1 > bx0 and by1 > by0:
                keep[by0:by1, bx0:bx1] = False
        if keep.mean() < 0.4:
            return None
        px = self.arr[y0:y1, x0:x1][keep].reshape(-1, 3).astype(np.float32)
        q = np.round(px / 8.0)
        vals, counts = np.unique(q, axis=0, return_counts=True)
        mode = vals[int(np.argmax(counts))] * 8.0
        d = np.sqrt(((px - mode) ** 2).sum(axis=1))
        return float(np.percentile(d, 60))

    ART_SCORE = 18.0
    ART_BUDGET = 0.5            # at most half of the screen may be carried by panel crops

    def ground_surfaces(self):
        """Cards, dialogs and panels the model painted with a colour get the measured shape too.

        A panel whose reference surface is artwork (glossy button, parchment, gradient, icon)
        is not flattened into one colour: it carries its reference crop, with the captions it
        owns inpainted out at binding time and drawn natively on top.
        """
        art = []
        native_drawn = {"Switch", "SeekBar", "CheckBox", "RadioButton", "ProgressBar", "RatingBar", "EditText",
                        "ToggleButton", "Spinner"}
        for n in self.nodes:
            if n.remove or n is self.root or n.target is None or n.tag in native_drawn:
                continue
            if n.role not in ("container", "view", "text", "widget"):
                continue
            if not paints_surface(aget(n.el, "background")):
                continue
            score = self._art_score(n)
            if score is not None and score > self.ART_SCORE:
                art.append((score, n))
        used = 0.0
        crop_backed = self.report.setdefault("art_panels", [])
        for score, n in sorted(art, key=lambda t: -t[0]):
            a = area(clip(n.target, self.W, self.H)) / (self.W * self.H)
            if used + a > self.ART_BUDGET:
                continue
            if any(self._is_descendant(n, m) for _, m in art if m.name in crop_backed):
                continue            # already inside a crop-backed panel
            used += a
            aset(n.el, "background", "@drawable/img")
            crop_backed.append(n.name)
        for n in self.nodes:
            if n.remove or n is self.root or n.target is None or n.role not in ("container", "view"):
                continue
            if parse_color(aget(n.el, "background")) is None:
                continue
            if area(n.target) >= 0.6 * self.W * self.H or (n.target[2] - n.target[0]) >= 0.97 * self.W:
                continue
            painted = [c.target for c in n.children if not c.remove and c.target is not None and
                       (parse_color(aget(c.el, "background")) is not None or c.role == "image")]
            if painted and inter(union(painted), n.target) >= 0.85 * area(n.target):
                continue  # only a border frame of this view is visible; the model's colour is the border
            self._fill_background(n)

    def _reorder_tree(self):
        for n in self.nodes:
            if n.remove:
                continue
            kids = [c for c in n.children if not c.remove]
            for c in n.children:
                if c.remove and c.el in list(n.el):
                    n.el.remove(c.el)
            n.children = kids

    def _system_bands(self):
        root = self.root_el
        inserts = []
        sb = self.bands["status_bottom"]
        if sb and self.bands["status_color"] is not None:
            inserts.append(("s2r_status", [0, 0, self.W, sb], self.bands["status_color"]))
        if self.bands["nav_top"] is not None and self.bands["nav_color"] is not None:
            inserts.append(("s2r_nav", [0, self.bands["nav_top"], self.W, self.H], self.bands["nav_color"]))
        for i, (name, box, color) in enumerate(inserts):
            el = ET.Element("View")
            aset(el, "id", f"@+id/{name}")
            aset(el, "layout_width", "match_parent")
            aset(el, "layout_height", f"{self.frame.dp(box[3] - box[1]):.1f}dp")
            aset(el, "layout_gravity", "top|start")
            aset(el, "layout_marginTop", f"{self.frame.dp(box[1]):.1f}dp")
            aset(el, "background", hex_color(color))
            root.append(el)
        page, frac = mode_color(self.arr, [0, self.bands["status_bottom"], self.W,
                                           self.bands["nav_top"] or self.H])
        model = parse_color(aget(root, "background"))
        if page is not None and frac >= 0.3 and (model is None or cdist(model, page) > 3):
            aset(root, "background", hex_color(page))

    # -- stage 7: assets
    def bind_assets(self, drawables: Path) -> int:
        from scipy import ndimage  # noqa: F401  (cv2 fallback below)
        import cv2
        drawables.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha1(self.sid.encode()).hexdigest()[:6]
        big = BIG_IMAGE_FRAC * self.W * self.H
        live = [n for n in self.nodes if not n.remove and n.target is not None]
        bound = 0
        inpainted = 0
        cleared = self.report.setdefault("captions_left_to_crop", [])
        rank = {el: i for i, el in enumerate(self.root_el.iter())}

        def above(m: Node, n: Node) -> bool:
            """m is drawn over n (pre-order of absolute frames is paint order)."""
            if self._is_descendant(m, n):
                return True
            return rank.get(m.el, -1) > rank.get(n.el, -1) and not self._is_descendant(n, m)

        for n in live:
            el = n.el
            for attr in COMPOUND_DRAWABLE_ATTRS + PLACEHOLDER_ONLY_ATTRS:
                v = aget(el, attr)
                if v and v.startswith("@drawable/img"):
                    adel(el, attr)
            slot = None
            if n.role == "image" and (aget(el, "src") or "@drawable/img").startswith("@drawable/img"):
                slot = "src"
            elif (aget(el, "background") or "").startswith("@drawable/img"):
                slot = "background"
            elif n.role == "image" and aget(el, "src") is None:
                slot = "src"
            if slot is None:
                continue
            box = clip(n.target, self.W, self.H)
            if slot == "src" and n.comp is None and not n.added and not n.baked and area(box) < big:
                box = clip(refine_image_box(self.arr, box), self.W, self.H)
                n.target = box
            if slot == "src" and not n.baked and area(box) < big:
                grown = clip(grow_to_object(self.arr, box, [l["tbox"] for l in self.ocr]), self.W, self.H)
                if grown != box:
                    self.report.setdefault("crops_grown", []).append(
                        {"id": n.name, "from": [round(v, 1) for v in box], "to": [round(v, 1) for v in grown]})
                    box = n.target = grown
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            if x1 - x0 < 2 or y1 - y0 < 2:
                adel(el, slot)
                continue
            crop = np.asarray(self.img)[y0:y1, x0:x1].copy()
            overlays = []
            # text is removed from a crop only where a native caption is painted over it;
            # a caption drawn under the crop would otherwise vanish from both layers
            native_lines = {l["id"]: l for m in live if m.role == "text" and m is not n and above(m, n)
                            for l in m.lines if not l.get("partial")}
            if slot == "background":
                # the crop already carries whatever the reference shows under an unread caption:
                # a native guess drawn on top of it would double or invent the text
                for m in [n] + [m for m in live if self._is_descendant(m, n)]:
                    if m.role == "text" and not m.lines and aget(m.el, "text") and \
                            frac_inside(m.target, box) >= 0.6:
                        aset(m.el, "text", "")
                        cleared.append(m.name)
            for lid, l in native_lines.items():
                if lid not in self.baked_lines and frac_inside(l["tbox"], box) >= 0.6:
                    overlays.append(expand(l["tbox"], 0.03, 0.25, 3))
            if area(box) >= big or slot == "background":
                for m in live:
                    if m is n or m.target is None or (m.role == "container" and not self._opaque_panel(m)):
                        continue
                    if not above(m, n):
                        continue
                    if inter(m.target, box) <= 0:
                        continue
                    if m.role == "text" and m.lines:
                        for l in m.lines:
                            overlays.append(expand(l["tbox"], 0.03, 0.25, 3))
                    elif m.role == "text":
                        continue
                    else:
                        overlays.append(expand(m.target, 0.03, 0.03, 2))
                if slot == "background" and n.lines:
                    overlays += [expand(l["box"], 0.02, 0.15, 3) for l in n.lines]
                native = {l["id"] for m in live if m.role == "text" and above(m, n) for l in m.lines}
                for l in self.ocr:
                    if frac_inside(l["box"], box) > 0.6 and l["id"] in native and l["id"] not in self.baked_lines:
                        overlays.append(expand(l["box"], 0.02, 0.15, 3))
            if overlays:
                mask = np.zeros(crop.shape[:2], dtype=np.uint8)
                for ob in overlays:
                    ob = clip(ob, self.W, self.H)
                    ax0, ay0 = int(max(0, math.floor(ob[0]) - x0)), int(max(0, math.floor(ob[1]) - y0))
                    ax1, ay1 = int(min(x1 - x0, math.ceil(ob[2]) - x0)), int(min(y1 - y0, math.ceil(ob[3]) - y0))
                    if ax1 > ax0 and ay1 > ay0:
                        mask[ay0:ay1, ax0:ax1] = 255
                if mask.any():
                    bgr = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
                    bgr = cv2.inpaint(bgr, mask, 7, cv2.INPAINT_TELEA)
                    crop = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    inpainted += 1
            name = f"s2g{tag}_{bound}"
            Image.fromarray(crop).save(drawables / f"{name}.png")
            aset(el, slot, f"@drawable/{name}")
            native_ids = {l["id"] for m in live if m.role == "text" and m is not n for l in m.lines}
            inside = [l["id"] for l in self.ocr if frac_inside(l["box"], box) > 0.6]
            self.report.setdefault("crops", []).append({
                "id": n.name, "tag": n.tag, "drawable": name, "slot": slot,
                "source_box": [round(v, 1) for v in box], "area_frac": round(area(box) / (self.W * self.H), 4),
                "inpainted": bool(overlays),
                "ocr_lines_inside": inside,
                "native_lines_inside": [i for i in inside if i in native_ids],
                "baked_lines_inside": [i for i in inside if i in self.baked_lines]})
            bound += 1
        for el in self.root_el.iter():
            for key in list(el.attrib):
                if el.attrib[key].startswith("@drawable/img"):
                    del el.attrib[key]
        self.report["crops_bound"] = bound
        self.report["crops_inpainted"] = inpainted
        return bound

    def _opaque_panel(self, m: Node) -> bool:
        """The view paints an opaque surface, so whatever the photo shows under it is never seen."""
        bg = (aget(m.el, "background") or "").strip()
        if bg in ("@android:color/white", "@android:color/black"):
            return True
        if bg.startswith("@drawable/"):
            body = self.shapes.get(bg.split("/", 1)[1])
            hit = re.search(r'<solid android:color="(#[0-9A-Fa-f]+)"', body or "")
            bg = hit.group(1) if hit else ""
        v = bg[1:] if bg.startswith("#") else ""
        return len(v) in (3, 6) or (len(v) == 8 and int(v[:2], 16) >= 0xF0)

    @staticmethod
    def _is_descendant(m: Node, n: Node) -> bool:
        q = m.parent
        while q is not None:
            if q is n:
                return True
            q = q.parent
        return False

    # -- driver
    def run(self, out_dir: Path) -> dict:
        self.calibrate()
        self.match_text()
        self.assign_targets()
        self.bake_display_text()
        self.prune()
        self.complete()
        self.grow_containers()
        self.text_above_images()
        self.rewrite()
        self.ground_surfaces()
        self.text_above_images()     # surfaces grounded above may have become opaque
        self.bind_assets(out_dir / "drawables")
        self.images_above_foreign_fills()
        for name, body in self.shapes.items():
            (out_dir / "drawables" / f"{name}.xml").write_text(body, encoding="utf-8")
        self.report["native_shapes"] = len(self.shapes)
        # rewrite() already ran; geometry of the image slots refined during binding is re-applied
        for n in self.nodes:
            if not n.remove and n.role == "image" and n is not self.root and n.target is not None:
                p = n.parent
                while p is not None and p.remove:
                    p = p.parent
                pt = p.target if p is not None and p.target is not None else [0, 0, self.W, self.H]
                t = n.target
                aset(n.el, "layout_width", f"{max(0.5, self.frame.dp(t[2] - t[0])):.1f}dp")
                aset(n.el, "layout_height", f"{max(0.5, self.frame.dp(t[3] - t[1])):.1f}dp")
                aset(n.el, "layout_marginStart", f"{self.frame.dp(t[0] - pt[0]):.1f}dp")
                aset(n.el, "layout_marginLeft", f"{self.frame.dp(t[0] - pt[0]):.1f}dp")
                aset(n.el, "layout_marginTop", f"{self.frame.dp(t[1] - pt[1]):.1f}dp")
        xml = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(self.root_el, encoding="unicode")
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "final.xml").write_text(xml + "\n", encoding="utf-8")
        self.report["live_nodes"] = sum(1 for n in self.nodes if not n.remove)
        (out_dir / "ground_report.json").write_text(json.dumps(self.report, ensure_ascii=False, indent=1),
                                                    encoding="utf-8")
        return self.report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sid", required=True)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--xml", type=Path, required=True, help="id-injected pre-render XML")
    ap.add_argument("--vh", type=Path, required=True)
    ap.add_argument("--screen", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    g = Grounder(args.sid, args.ref, args.xml.read_text(encoding="utf-8"), args.vh, args.screen)
    print(json.dumps(g.run(args.out), ensure_ascii=False)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
