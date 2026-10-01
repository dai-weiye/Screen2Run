#!/usr/bin/env python3
"""Paper screenshot metric engines and an explicit pair-manifest CLI.

Status-band masking, OCR matching, CLIP preprocessing, SSIM, and RGB MAE retain
the experiment definitions. Missing captures must be declared, not inferred
from a missing path. Scoring failures remain pending instead of becoming zeros.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths
import numpy as np
from PIL import Image
from frozen_statistics import METRICS as QUALITY_METRICS, bootstrap_ci, cliffs_delta, holm

METRICS = QUALITY_METRICS + ("render",)
LOWER_BETTER = {"mae"}
WORST = {"mae": 255.0}


def split_of(sid):
    if len(sid) == 36 and sid.count("-") == 4:
        return "Easy"
    return "Real" if sid.startswith("rr") and sid[2:].isdigit() else "Unseen"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


class Evaluator:
    def __init__(self, cache_dir, mask_top=0.045):
        from ocr import ScreenshotOCR
        from clip_similarity import load_local_model
        import torch
        if not 0 <= mask_top < 0.5:
            raise ValueError("mask_top must be in [0, 0.5)")
        torch.set_num_threads(4)
        self.cache_dir, self.mask_top = Path(cache_dir), float(mask_top)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ocr = ScreenshotOCR(self.cache_dir / "ocr").__enter__()
        self.clip_model, self.clip_proc, self.clip_provenance = load_local_model()

    def close(self):
        self.ocr.__exit__(None, None, None)

    def _masked(self, path, fill):
        out = self.cache_dir / "masked" / f"{sha(path)}_{self.mask_top:.3f}_{'%02x%02x%02x' % tuple(fill)}.png"
        if not out.is_file():
            out.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGB")).copy()
            array[:int(round(self.mask_top * array.shape[0]))] = fill
            Image.fromarray(array).save(out)
        return out

    def score(self, ref, gen):
        ref, gen = Path(ref), Path(gen)
        key = f"{sha(ref)}_{sha(gen)}" + (f"_m{self.mask_top:.3f}" if self.mask_top > 0 else "")
        path = self.cache_dir / "pairs" / f"{key}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        from ocr import evaluate_pair
        from clip_similarity import score_pair
        from skimage.metrics import structural_similarity
        if self.mask_top > 0:
            with Image.open(ref) as image:
                array = np.asarray(image.convert("RGB"))
            cut = int(round(self.mask_top * array.shape[0]))
            below = array[cut:cut + max(2, array.shape[0] // 40)].reshape(-1, 3)
            fill = [int(v) for v in np.median(below, axis=0)]
            ref, gen = self._masked(ref, fill), self._masked(gen, fill)
        for attempt in range(60):
            try:
                blocks = evaluate_pair(self.ocr, ref, gen)
                break
            except Exception as exc:
                if "currently being extracted" not in str(exc) or attempt == 59:
                    raise
                time.sleep(2.0)
        row = dict(blocks.get("scores") or {})
        with Image.open(ref) as source, Image.open(gen) as rendered:
            source = source.convert("RGB")
            rendered = rendered.convert("RGB").resize(source.size, Image.BILINEAR)
            row["clip"] = float(score_pair(self.clip_model, self.clip_proc, source, rendered))
            a, b = np.asarray(source.convert("L"), dtype=np.float64), np.asarray(rendered.convert("L"), dtype=np.float64)
            row["ssim"] = float(structural_similarity(a, b, data_range=255.0))
            row["mae"] = float(np.abs(np.asarray(source, dtype=np.float64) - np.asarray(rendered, dtype=np.float64)).mean())
        row["render"] = 1.0
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(row, allow_nan=False) + "\n", encoding="utf-8")
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True, help="JSON list: screen_id, split, arm, reference, render_path, status")
    parser.add_argument("--cache", type=Path, default=release_paths.CACHE / "metrics")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mask-top", type=float, default=0.045)
    parser.add_argument("--legacy-color-null-zero", action="store_true", help="Record and apply the historical Color aggregation policy")
    args = parser.parse_args()
    pairs = json.loads(args.pairs.read_text(encoding="utf-8"))
    if not isinstance(pairs, list) or not pairs:
        parser.error("Pair manifest must be a nonempty list")
    seen, rows, pending, conversions = set(), [], [], []
    evaluator = None
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        for item in pairs:
            key = item["arm"], item["screen_id"]
            if key in seen or item["split"] not in ("Easy", "Real", "Unseen"):
                raise ValueError("Duplicate pair or unknown split")
            seen.add(key)
            row = {name: item[name] for name in ("arm", "screen_id", "split")}
            status = item.get("status")
            if status == "method_failure":
                if not item.get("failure_reason"):
                    raise ValueError("Method failures require an explicit failure reason")
                row.update({metric: WORST.get(metric, 0.0) for metric in METRICS})
                rows.append(row)
                continue
            if status != "render_success":
                raise ValueError("Only explicit render_success or method_failure is scoreable")
            paths = []
            for name in ("reference", "render_path"):
                path = Path(item[name])
                path = path if path.is_absolute() else args.pairs.resolve().parent / path
                if not path.is_file():
                    raise FileNotFoundError(path)
                expected = item.get(name + "_sha256")
                if expected and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError(f"Image source hash mismatch: {name}/{key}")
                paths.append(path)
            try:
                if evaluator is None:
                    evaluator = Evaluator(args.cache, args.mask_top)
                values = evaluator.score(*paths)
                for metric in METRICS:
                    value = values.get(metric)
                    if value is None and metric == "color_similarity" and args.legacy_color_null_zero:
                        value = 0.0
                        conversions.append({**row, "metric": metric, "source_value": None, "aggregation_value": 0.0})
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise ValueError(f"Unresolved metric: {metric}")
                    row[metric] = float(value)
                rows.append(row)
            except Exception as exc:
                pending.append({**row, "error_type": type(exc).__name__, "error": str(exc)})
                print(f"Pending: {key[0]}/{key[1]} ({type(exc).__name__})", file=sys.stderr)
    finally:
        if evaluator is not None:
            evaluator.close()
    report = {"status": "pending" if pending else "complete", "expected_pairs": len(pairs),
              "scored_pairs": len(rows), "pending": pending, "mask_top": args.mask_top,
              "legacy_color_conversions": conversions,
              "clip_provenance": evaluator.clip_provenance if evaluator else None}
    (args.out / "score_report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    filename = "partial_rows.json" if pending else "rows.json"
    (args.out / filename).write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return 2 if pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
