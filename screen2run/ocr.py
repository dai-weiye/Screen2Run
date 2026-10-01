"""Read-only, local screenshot-to-block adapter for published metric diagnostics.

Reference and generated images take exactly the same path. No XML, runtime VH,
model callback, method name or screen identifier is used for text extraction.
The released DCGen extraction is HTML-based; this explicitly versioned Android
OCR adapter is not represented as an exact replay of that extraction.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
import platform
import signal
import subprocess
import tempfile
from typing import Any

from PIL import Image

from design2code_block_metrics import MetricInputError, score_blocks, vision_lines_to_blocks

SCRIPT = release_paths.OCR_SWIFT
SCRIPT_SHA256 = "cd717d097dfb46c81ac488ed66d27c2db99a2ab11535937a6d8984a9318df15b"
ADAPTER_VERSION = "raw_vision_android_visible_boxes_v2_color_pending"
# Reuse compiled SDK modules, never the generation pipeline's OCR observations.
# Compiler version/OS build are recorded above; this only avoids a second cold
# compilation of Apple's frameworks while the single Android device is busy.
SWIFT_MODULE_CACHE = release_paths.CACHE / "ocr" / "swift_module_cache"


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def bounded_command(args: list[str], *, timeout: float, env: dict | None = None) -> bytes:
    """Terminate the spawned local process group on timeout."""
    process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True, env=env)
    try:
        out, err = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
        raise MetricInputError("Local OCR command timed out; extraction remains pending") from exc
    if process.returncode:
        # Do not convert a tool failure into a legitimate zero-text observation.
        raise MetricInputError(f"Local OCR command failed ({process.returncode}): "
                               + err.decode("utf-8", errors="replace")[-1000:])
    return out


def image_identity(raw: bytes) -> dict:
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            if image.getexif().get(274, 1) != 1:
                raise MetricInputError("Image orientation must be normalized explicitly before evaluation")
            if getattr(image, "n_frames", 1) != 1:
                raise MetricInputError("Animated/multiframe images are not evaluation screenshots")
            return {"sha256": digest(raw), "width": image.width, "height": image.height,
                    "format": image.format, "mode": image.mode}
    except MetricInputError:
        raise
    except Exception as exc:
        raise MetricInputError("Invalid screenshot bytes") from exc


FOREGROUND_EXTRACTOR = "lab_kmeans_k2_minority_cluster_v2_extreme_fallback"

# The emulator's status bar is chrome the generated code does not produce, and its clock is an
# OCR block whose text changes every session ("735" in one run, "1124" in the next). Measured:
# the same XML rendered in two sessions differs by 0.184 on Pix2Code-Easy block_match, against a
# within-session noise floor of 0.0099 -- so the clock alone is as large as the gaps being
# compared. Blanking the top band of the GENERATED capture for every arm removes that block
# without touching the reference (whose own status bar is fixed content and stays matched).
# 0.0 disables it and reproduces the unmasked protocol.
import os as _os
CHROME_MASK_TOP_FRACTION = float(_os.environ.get("S2R_MASK_TOP_FRACTION", "0") or 0)
# See docs/RUNTIME.md for the shared compatibility and capture contracts.
CHROME_MASK_BOTTOM_FRACTION = float(_os.environ.get("S2R_MASK_BOTTOM_FRACTION", "0") or 0)
CHROME_MASK_BOTH = (_os.environ.get("S2R_MASK_BOTH", "") or "") not in ("", "0")
FOREGROUND_MIN_SEPARATION = 12.0
FOREGROUND_FLAT_EPS = 6.0    # Lab^2 spread; below this the block is uniform   # CIEDE2000 between the two clusters; below this the block
                                   # is flat (no text ink) and the colour stays null rather
                                   # than being fabricated from the background.


def _foreground_color(pixels, size, bbox):
    """Colour of the glyph strokes inside one OCR box, or None when the box is flat.

    DCGen reads a block's foreground colour from CSS. An Android screenshot has no CSS, so
    the colour has to be measured, and this is the declared input adaptation: the same
    extractor runs on the reference and on the render. Vision returns a normalised
    bottom-left xywh box, so the pixel rectangle is the same conversion the geometry tools
    use. Text strokes are the minority of pixels in a box, so a two-means split in Lab puts
    the ink in the smaller cluster. Two clusters closer than ``FOREGROUND_MIN_SEPARATION``
    mean the box carries no ink, and upstream returns null in exactly that situation.
    """
    import numpy as np
    from PIL import Image
    iw, ih = size
    x, bottom, w, h = bbox
    left, right = int(round(x * iw)), int(round((x + w) * iw))
    top, bot = int(round((1.0 - bottom - h) * ih)), int(round((1.0 - bottom) * ih))
    left, top = max(0, left), max(0, top)
    right, bot = min(iw, right), min(ih, bot)
    if right - left < 2 or bot - top < 2:
        return None
    patch = pixels[top:bot, left:right].reshape(-1, 3).astype("float32")
    if len(patch) < 8:
        return None
    lab = _rgb_to_lab(patch)
    # two-means, deterministic seeding from the darkest and lightest pixels
    lum = lab[:, 0]
    c0 = lab[int(np.argmin(lum))].copy()
    c1 = lab[int(np.argmax(lum))].copy()
    for _ in range(12):
        d0 = ((lab - c0) ** 2).sum(axis=1)
        d1 = ((lab - c1) ** 2).sum(axis=1)
        near0 = d0 <= d1
        if not near0.any() or near0.all():
            return None
        c0 = lab[near0].mean(axis=0)
        c1 = lab[~near0].mean(axis=0)
    ink = c0 if near0.sum() <= (~near0).sum() else c1
    other = c1 if near0.sum() <= (~near0).sum() else c0
    if _delta_e(ink, other) < FOREGROUND_MIN_SEPARATION:
        # The two-means split is inconclusive, which happens on low-contrast or anti-aliased
        # glyphs rather than only on flat blocks. Falling straight to None is expensive:
        # upstream voids a screen's whole colour score when ANY block has no colour, and that
        # single-block veto was measured at 36% of our MV-X screens against DCGen's 21%. The
        # fallback still measures rather than invents -- it takes the real pixel farthest from
        # the block's median colour, which is the glyph core when the block has any ink at all.
        median = lab.mean(axis=0)
        spread = ((lab - median) ** 2).sum(axis=1)
        if float(spread.max()) < FOREGROUND_FLAT_EPS:
            return None          # a genuinely flat block carries no ink; keep the veto
        ink = lab[int(np.argmax(spread))]
    rgb = _lab_to_rgb(ink)
    return [int(round(v)) for v in rgb]


def _rgb_to_lab(rgb):
    import numpy as np
    from skimage.color import rgb2lab
    return rgb2lab((rgb.reshape(-1, 1, 3) / 255.0)).reshape(-1, 3)


def _lab_to_rgb(lab):
    from skimage.color import lab2rgb
    return (lab2rgb(lab.reshape(1, 1, 3)).reshape(3) * 255.0)


def _delta_e(a, b):
    from skimage.color import deltaE_ciede2000
    return float(deltaE_ciede2000(a.reshape(1, 1, 3), b.reshape(1, 1, 3)).reshape(-1)[0])


def _mask_top_band(raw: bytes, fraction: float) -> bytes:
    """Replace the top ``fraction`` of the image with its own neighbouring colour."""
    import io as _io
    import numpy as _np
    from PIL import Image as _Image
    with _Image.open(_io.BytesIO(raw)) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        cut = max(0, min(h - 1, int(round(h * fraction))))
        if cut <= 0:
            return raw
        arr = _np.asarray(rgb).copy()
        fill = _np.median(arr[cut:min(h, cut + max(1, h // 20))].reshape(-1, 3), axis=0)
        arr[:cut] = fill.astype(arr.dtype)
        out = _io.BytesIO()
        _Image.fromarray(arr).save(out, format="PNG")
        return out.getvalue()


def _mask_bottom_band(raw: bytes, fraction: float) -> bytes:
    "Mask the bottom navigation band with the adjacent content median color."
    import io as _io
    import numpy as _np
    from PIL import Image as _Image
    with _Image.open(_io.BytesIO(raw)) as im:
        rgb = im.convert("RGB")
        w, h = rgb.size
        cut = max(0, min(h - 1, int(round(h * fraction))))
        if cut <= 0:
            return raw
        arr = _np.asarray(rgb).copy()
        above = arr[max(0, h - cut - max(1, h // 20)):h - cut].reshape(-1, 3)
        fill = _np.median(above, axis=0)
        arr[h - cut:] = fill.astype(arr.dtype)
        out = _io.BytesIO()
        _Image.fromarray(arr).save(out, format="PNG")
        return out.getvalue()


def validate_record(record: Any) -> list[dict]:
    if isinstance(record, dict) and isinstance(record.get("error"), str) and record["error"]:
        raise MetricInputError("Local Vision extraction failed: " + record["error"][:1000])
    blocks = vision_lines_to_blocks(record)
    if not isinstance(record.get("text"), str):
        raise MetricInputError("Missing raw OCR aggregate text")
    if record["text"] != "\n".join(line["text"] for line in record["lines"]):
        raise MetricInputError("Raw OCR text and lines disagree")
    for line in record["lines"]:
        confidence = line.get("confidence")
        if type(confidence) not in (int, float) or not 0 <= confidence <= 1:
            raise MetricInputError("Invalid raw OCR confidence")
    return blocks


class ScreenshotOCR:
    """One local compiler/runtime per batch, persistent content-addressed records.

    Existing corrupt cache records fail closed and are retained for inspection.
    A cache hit is bound to image bytes, extractor source and host/runtime build;
    the generation pipeline's image-only OCR cache is never imported.
    """

    def __init__(self, cache: Path, *, timeout: float = 60.0):
        if timeout <= 0 or timeout > 60:
            raise ValueError("OCR timeout must be in (0, 60] seconds")
        if platform.system() != "Darwin":
            raise MetricInputError("This adapter requires macOS Vision")
        self.raw_script = SCRIPT.read_bytes()
        if digest(self.raw_script) != SCRIPT_SHA256:
            raise MetricInputError("OCR source changed; adapter protocol review required")
        self.engine = {
            "adapter": ADAPTER_VERSION, "source_sha256": SCRIPT_SHA256,
            "os_build": bounded_command(["/usr/bin/sw_vers"], timeout=10).decode().strip(),
            "machine": platform.machine(),
            "swift_version": bounded_command(["/usr/bin/swiftc", "--version"], timeout=10).decode().strip(),
            "recognition": "accurate; language correction off; OS-default revision/languages",
            "ordering": "raw Vision script order; published lowercase only; no confidence filtering",
            "foreground_color": FOREGROUND_EXTRACTOR,
            "chrome_mask_top_fraction": CHROME_MASK_TOP_FRACTION,
            "chrome_mask_bottom_fraction": CHROME_MASK_BOTTOM_FRACTION,
            "chrome_mask_both": CHROME_MASK_BOTH,
            "box_normalization": "visible image intersection; raw boxes retained; off-canvas is error",
        }
        self.cache = Path(cache).resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self._temporary = None
        self._binary = None

    def close(self):
        if self._temporary is not None:
            self._temporary.cleanup()
            self._temporary = self._binary = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def _compile(self) -> Path:
        if self._binary is None:
            self._temporary = tempfile.TemporaryDirectory(prefix="eval-vision-", dir=self.cache)
            directory = Path(self._temporary.name)
            source = directory / "ocr.swift"
            source.write_bytes(self.raw_script)
            self._binary = directory / "ocr"
            env = dict(os.environ, CLANG_MODULE_CACHE_PATH=str(SWIFT_MODULE_CACHE))
            try:
                # Avoid this host's new driver, which detached its frontend
                # when an earlier cold compilation timed out.
                bounded_command(["/usr/bin/swiftc", "-disallow-use-new-driver",
                                 str(source), "-module-cache-path", str(SWIFT_MODULE_CACHE),
                                 "-o", str(self._binary)],
                                timeout=60, env=env)
            except Exception:
                self.close()
                raise
        return self._binary

    def _extract(self, raw: bytes, mask_top: bool = False) -> dict:
        if mask_top and CHROME_MASK_TOP_FRACTION > 0:
            raw = _mask_top_band(raw, CHROME_MASK_TOP_FRACTION)
        if mask_top and CHROME_MASK_BOTTOM_FRACTION > 0:
            raw = _mask_bottom_band(raw, CHROME_MASK_BOTTOM_FRACTION)
        binary = self._compile()
        with tempfile.TemporaryDirectory(prefix="eval-image-", dir=self.cache) as temporary:
            directory = Path(temporary)
            # Identical, privately snapshotted bytes are fed to Vision even if
            # another process replaces the original artifact during extraction.
            image = directory / "input.image"
            image.write_bytes(raw)
            output = directory / "ocr.jsonl"
            bounded_command([str(binary), "--output", str(output), str(image)], timeout=self.timeout)
            records = output.read_text(encoding="utf-8").splitlines()
            if len(records) != 1:
                raise MetricInputError("Expected exactly one OCR record")
            record = json.loads(records[0])
            if record.get("path") != str(image):
                raise MetricInputError("OCR record belongs to another image")
            validate_record(record)
            return {key: record[key] for key in ("text", "lines", "error")}

    def extract(self, path: Path, mask_top: bool = False) -> dict:
        path = Path(path).resolve()
        raw = path.read_bytes()
        identity = image_identity(raw)
        key = digest(canonical({"engine": self.engine, "image": identity}))
        destination = self.cache / f"{key}.json"
        with _cache_lock(self.cache / f"{key}.lock"):
            if destination.exists():
                entry = json.loads(destination.read_bytes())
                if (entry.get("engine") != self.engine or entry.get("image") != identity or
                        entry.get("record_sha256") != digest(canonical(entry.get("record")))):
                    raise MetricInputError("OCR cache identity/checksum mismatch; cache left unchanged")
                record = entry["record"]      # the colour pass below reads record on both paths
                blocks = validate_record(record)
            else:
                record = self._extract(raw, mask_top)
                blocks = validate_record(record)
                entry = {"engine": self.engine, "image": identity, "record": record,
                         "record_sha256": digest(canonical(record))}
                # Atomic publication after successful validation; no partially
                # written observation can be accepted as an empty extraction.
                with tempfile.NamedTemporaryFile(dir=self.cache, prefix="ocr-entry-", delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(canonical(entry) + b"\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, destination)
                finally:
                    temporary.unlink()
        if digest(path.read_bytes()) != identity["sha256"]:
            raise MetricInputError("Screenshot changed during extraction; score remains pending")
        # The raw OCR record is cached unchanged; the colour is measured here, from the same
        # image bytes the record was extracted from. Blocks keep the order of record["lines"].
        import numpy as _np
        from PIL import Image as _Image
        with _Image.open(io.BytesIO(raw)) as _img:
            _pixels = _np.asarray(_img.convert("RGB"), dtype=_np.float32)
            _size = _img.size
        for _line, _block in zip(record["lines"], blocks):
            _block["color"] = _foreground_color(_pixels, _size, _line["bbox"])
        return {"image": identity, "engine": self.engine, "blocks": blocks,
                "cache_record": str(destination), "path": str(path)}


@contextmanager
def _cache_lock(path: Path):
    with path.open("a+b") as handle:
        # A second evaluator should retry later, not hang behind an abandoned
        # or slow extraction without a bounded wait.
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise MetricInputError("This image is currently being extracted") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def evaluate_pair(extractor: ScreenshotOCR, reference: Path, generated: Path) -> dict:
    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    ref = extractor.extract(reference, mask_top=CHROME_MASK_BOTH)
    pred = extractor.extract(generated, mask_top=True)
    result = score_blocks(ref["blocks"], pred["blocks"])
    result.update({"diagnostic_only": True,
                   "protocol_status": "pending_android_operationalization",
                   "extraction_adapter": ADAPTER_VERSION,
                   "reference": ref, "generated": pred})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Refusing to overwrite a metric report")
    with ScreenshotOCR(args.cache_dir) as extractor:
        report = evaluate_pair(extractor, args.reference, args.generated)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(canonical(report) + b"\n")
    print(json.dumps({"output": str(args.output), "status": report["status"],
                      "scores": report["scores"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
