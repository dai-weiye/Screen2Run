#!/usr/bin/env python3
"""Offline, provenance-checked S3/S5 preparation for the redesigned RQ2.

Both stages receive exactly the same sanitizer, native-surface materializer,
stable-ID injection and compile cleanup. S5 is replayed from its last *recorded
model output*, never from an already-materialized layout. The replay must match
the generation's final unbound XML byte for byte. No model or renderer is called.

This creates independent candidates; existing output directories are rejected.
Only ``summary.json`` with status ``complete`` authorizes downstream use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
import tempfile
import threading
import xml.etree.ElementTree as ET
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths
REPO = release_paths.RELEASE

import native_drawables as native_shapes
from model_sessions import _sanitize_drawables, _strip_fence
from session_contracts import validate_xml
from image_filling import inject_ids
from run_image_filling import compile_cleanup, read_sids, shot_map

# Historical experiment aliases; request model identifiers remain unchanged.
APPROVED_MODEL_ALIASES = {"deepseek-flash": "deepseek-v4.1-flash",
                          "deepseek-v4.1-flash": "deepseek-v4.1-flash"}
_NAMESPACE_LOCK = threading.RLock()


class EvidenceError(ValueError):
    """A candidate cannot be linked to its own recorded generation."""


class CohortEvidenceError(EvidenceError):
    """All invalid cohort members are retained rather than silently omitted."""

    def __init__(self, failures: list[dict]):
        self.failures = failures
        super().__init__("source validation failed: " + json.dumps(failures, ensure_ascii=False))


@contextmanager
def generation_serializer_context():
    """Undo only downstream namespace registration during generation replay.

    The model-session materializer registers android, but not tools/app.
    Image filling registers those extra aliases at import. They must not alter a
    saved generation's XML bytes. Restore downstream aliases before inject_ids.
    This offline builder is sequential, not thread-safe by design.
    """
    with _NAMESPACE_LOCK:
        saved = dict(ET._namespace_map)
        try:
            for uri in ("http://schemas.android.com/tools", "http://schemas.android.com/apk/res-auto"):
                ET._namespace_map.pop(uri, None)
            yield
        finally:
            ET._namespace_map.clear()
            ET._namespace_map.update(saved)


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256(path.read_bytes())}


def _safe_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or Path(name).name != name or name in {"", ".", ".."}:
        raise EvidenceError(f"unsafe evidence filename: {name!r}")
    path = root / name
    if not path.is_file() or path.is_symlink() or path.resolve().parent != root.resolve():
        raise EvidenceError(f"missing or non-local evidence file: {path}")
    return path


def _checked_file(root: Path, name: str, expected: str) -> Path:
    path = _safe_file(root, name)
    if not isinstance(expected, str) or sha256(path.read_bytes()) != expected:
        raise EvidenceError(f"evidence hash mismatch: {path}")
    return path


def _stage_text(evidence: Path, stages: dict, key: str, reference_sha: str) -> tuple[str, dict]:
    entry = stages.get(key)
    if not isinstance(entry, dict) or entry.get("reference_sha256") != reference_sha:
        raise EvidenceError(f"missing or mismatched stage: {key}")
    expected = {"s1": "stage1.json", "s2": "stage2.json", "s3": "stage3_original.xml",
                "s4": "stage4_critique.txt", "s5": "stage5_initial.xml"}.get(key)
    if key.startswith("s5_repair"):
        expected = f"stage5_repair{key.removeprefix('s5_repair')}.xml"
    if entry.get("output_file") != expected:
        raise EvidenceError(f"stage output filename mismatch: {key}")
    output = _checked_file(evidence, entry.get("output_file"), entry.get("output_file_sha256"))
    text = output.read_text(encoding="utf-8")
    canonical = entry.get("canonical_output_sha256")
    # XML stage files append exactly one newline after recording the canonical XML.
    if sha256(text.encode()) != canonical:
        if not text.endswith("\n") or sha256(text[:-1].encode()) != canonical:
            raise EvidenceError(f"canonical output mismatch: {key}")
        text = text[:-1]
    prompt = _checked_file(evidence, f"{key}_input.txt", entry.get("input_sha256"))
    return text, {"stage": key, "output": file_record(output), "initial_prompt": file_record(prompt),
                  "dependencies": entry.get("dependencies")}


def _verify_stage_requests(evidence: Path, key: str, entry: dict, manifest: dict) -> list[dict]:
    """Prove model/reference identity of every recorded successful dependency."""
    requests = []
    for attempt in entry.get("attempts", []):
        if attempt.get("status") != "response_received":
            continue
        if attempt.get("model") != manifest["model"] or attempt.get("reference_sha256") != manifest["reference_sha256"]:
            raise EvidenceError(f"dependency request model/reference mismatch: {key}")
        prompt = _checked_file(evidence, attempt.get("prompt_file"), attempt.get("prompt_file_sha256"))
        response = _checked_file(evidence, attempt.get("output_file"), attempt.get("output_file_sha256"))
        requests.append({"attempt": attempt.get("attempt"), "prompt": file_record(prompt),
                         "raw_response": file_record(response)})
    intentionally_skipped = (key == "s1" and manifest.get("ablation") == "no_s1") or (
        key == "s2" and manifest.get("ablation") in {"no_s1", "no_s2"})
    if not requests and not intentionally_skipped:
        raise EvidenceError(f"actual model response missing for dependency: {key}")
    return requests


def _prove_xml_attempt(evidence: Path, entry: dict, canonical_xml: str, manifest: dict) -> dict:
    for attempt in reversed(entry.get("attempts", [])):
        if attempt.get("status") != "response_received":
            continue
        if attempt.get("model") != manifest["model"] or attempt.get("reference_sha256") != manifest["reference_sha256"]:
            raise EvidenceError("attempt model/reference differs from generation manifest")
        prompt = _checked_file(evidence, attempt.get("prompt_file"), attempt.get("prompt_file_sha256"))
        raw = _checked_file(evidence, attempt.get("output_file"), attempt.get("output_file_sha256"))
        raw_text = raw.read_text(encoding="utf-8")
        extraction = "current_xml_parser"
        try:
            parsed = validate_xml(_strip_fence(raw_text))
        except ValueError:
            # Older generations accepted an orphan trailing fence / leading
            # language label. Do not repair or reparse that reply differently:
            # prove the already-recorded canonical XML is an exact substring
            # and that absolutely nothing but wrapper tokens surrounds it.
            if raw_text.count(canonical_xml) != 1:
                continue
            before, after = raw_text.split(canonical_xml)
            if before.strip().lower() not in {"", "xml", "```", "```xml"} or after.strip() not in {"", "```"}:
                continue
            parsed = canonical_xml
            extraction = "historical_wrapper_only_exact_canonical_substring"
        if parsed == canonical_xml:
            return {"attempt": attempt.get("attempt"), "model": attempt["model"],
                    "prompt": file_record(prompt), "raw_response": file_record(raw),
                    "canonical_extraction": extraction,
                    "image_base64_sha256": attempt.get("image_base64_sha256"),
                    "replayed_from_previous_attempt": attempt.get("replayed_from_previous_attempt", False)}
    raise EvidenceError("no hash-verified actual model response matches the selected XML stage")


@dataclass
class Source:
    sid: str
    screen: Path
    reference: Path
    manifest: dict
    xml: str
    tree: dict
    scale: float
    provenance: dict


def load_source(generation_root: Path, sid: str, stage: str, reference: Path) -> Source:
    """Validate one same-run source without writing or using evaluation GT."""
    if stage not in {"s3", "s5"}:
        raise ValueError("stage must be s3 or s5")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,254}", sid):
        raise EvidenceError(f"unsafe screen ID: {sid!r}")
    screen = generation_root.resolve() / "full" / sid
    evidence = screen / "generation_evidence"
    if screen.is_symlink() or evidence.is_symlink():
        raise EvidenceError(f"symlinked generation source: {screen}")
    manifest_file = _safe_file(evidence, "generation_manifest.json")
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not manifest.get("run_id") or not manifest.get("model"):
        raise EvidenceError("generation identity missing")
    if not reference.is_file() or sha256(reference.read_bytes()) != manifest.get("reference_sha256"):
        raise EvidenceError(f"reference image identity mismatch: {sid}")
    stages = manifest.get("stages", {})
    selected = stage
    repairs = sorted((int(key.removeprefix("s5_repair")), key) for key in stages
                     if re.fullmatch(r"s5_repair[1-9][0-9]*", key))
    if stage == "s5" and repairs:
        if [n for n, _ in repairs] != list(range(1, repairs[-1][0] + 1)):
            raise EvidenceError("non-contiguous S5 repair chain")
        selected = repairs[-1][1]
    required = ["s1", "s2", "s3"]
    if stage == "s5":
        required += ["s4", "s5"] + [key for _, key in repairs]
    records = {}
    texts = {}
    for key in required:
        texts[key], records[key] = _stage_text(evidence, stages, key, manifest["reference_sha256"])
        records[key]["successful_requests"] = _verify_stage_requests(evidence, key, stages[key], manifest)
        dependencies = stages[key].get("dependencies")
        if not isinstance(dependencies, list) or any(dep not in texts for dep in dependencies):
            raise EvidenceError(f"broken stage dependency chain: {key}")
    tree = json.loads(texts["s2"])
    if not isinstance(tree, dict):
        raise EvidenceError("S2 must be a JSON object")
    xml = validate_xml(texts[selected])
    proof = _prove_xml_attempt(evidence, stages[selected], xml, manifest)
    try:
        scale = float(manifest["target_frame"]["source_to_screen_dp_scale_xy"][0])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise EvidenceError("missing target-frame x scale") from exc
    if not math.isfinite(scale) or scale <= 0:
        raise EvidenceError("invalid target-frame x scale")
    final = None
    surface = None
    if stage == "s5":
        final = _checked_file(screen, "final_unbound.xml", manifest.get("final_xml_sha256"))
        surface_meta = manifest.get("native_surface_materialization", {})
        surface_file = _checked_file(evidence, surface_meta.get("file"), surface_meta.get("file_sha256"))
        surface = json.loads(surface_file.read_text(encoding="utf-8"))
        for item in surface.get("applied", []):
            _checked_file(screen / "drawables", item["resource"], item["resource_sha256"])
    provenance = {"generation_manifest": file_record(manifest_file), "run_id": manifest["run_id"],
                  "model": manifest["model"], "ablation": manifest.get("ablation"),
                  "reference": file_record(reference), "selected_stage": selected,
                  "stages": records, "selected_xml_attempt": proof,
                  "target_frame": manifest["target_frame"],
                  "original_unbound": file_record(final) if final else None,
                  "original_native_surface_report": surface}
    return Source(sid, screen, reference, manifest, xml, tree, scale, provenance)


def materialize_source(source: Source, destination: Path, stage: str) -> dict:
    """Write one new screen using the identical engineering layer for both stages."""
    destination.mkdir(parents=True, exist_ok=False)
    # Never copy the S5 drawable directory into an S3 arm. Every native surface
    # is regenerated from this arm's own S2 and identities in the selected XML.
    with tempfile.TemporaryDirectory(prefix="s2r-rq2-native-") as temporary:
        native_dir = Path(temporary)
        with generation_serializer_context():
            materialized, report = native_shapes.bind_declared_native_shapes(
                _sanitize_drawables(source.xml), source.tree, native_dir,
                reference_px_per_dp=1.0 / source.scale)
        materialized_bytes = (materialized + "\n").encode()
        replay_equal = None
        if stage == "s5":
            replay_equal = sha256(materialized_bytes) == source.manifest["final_xml_sha256"]
            if not replay_equal:
                raise EvidenceError(f"S5 replay differs from recorded Full generation: {source.sid}")
            historical_resources = {item["resource"]: item["resource_sha256"]
                                    for item in source.provenance["original_native_surface_report"].get("applied", [])}
            replay_resources = {item["resource"]: item["resource_sha256"] for item in report["applied"]}
            if replay_resources != historical_resources:
                raise EvidenceError(f"S5 resource replay differs from recorded generation: {source.sid}")
        shipped = {p.stem for p in native_dir.glob("*.xml")}
        unknown = set(re.findall(r"@drawable/([A-Za-z0-9_.]+)", materialized)) - shipped - {"img"}
        # A guigpt-prefixed asset can survive the historical sanitizer, but the
        # historical manifest has no per-stage creator ledger for these assets.
        # It must not silently borrow a file produced later during S5/imgfill.
        ambiguous = [name for name in unknown if any((source.screen / "drawables").glob(f"{name}.*"))]
        if ambiguous:
            raise EvidenceError(f"resource stage ownership unproven: {ambiguous}")
        xml, cleanup = compile_cleanup(inject_ids(materialized), shipped)
        validate_xml(xml)
        resources = {}
        for name in sorted(set(re.findall(r"@drawable/([A-Za-z0-9_.]+)", xml)) - {"img"}):
            resource = native_dir / f"{name}.xml"
            if not native_shapes.is_native_shape_resource(resource, name):
                raise EvidenceError(f"not a freshly materialized native resource: {name}")
            (destination / "drawables").mkdir(exist_ok=True)
            shutil.copy2(resource, destination / "drawables" / resource.name)
            resources[resource.name] = sha256(resource.read_bytes())
        (destination / "replayed_unbound.xml").write_bytes(materialized_bytes)
        final_bytes = (xml if xml.endswith("\n") else xml + "\n").encode()
        (destination / "final.xml").write_bytes(final_bytes)
    provenance = {**source.provenance, "schema_version": 1,
                  "preparation_order": ["sanitize_drawables", "bind_declared_native_shapes", "inject_ids", "compile_cleanup"],
                  "implementation": {name: file_record(release_paths.module_file(name)) for name in (
                      "prepare_ablation.py", "model_sessions.py", "native_drawables.py",
                      "image_filling.py", "run_image_filling.py", "attribute_filter.py", "resource_sanitizer.py")},
                  "s5_replay_equal": replay_equal, "native_surface_materialization": report,
                  "compile_cleanup": cleanup, "drawables": resources,
                  "replayed_unbound_sha256": sha256(materialized_bytes), "final_xml_sha256": sha256(final_bytes),
                  "calls": 0, "rendered": False, "quality_claim": "none"}
    (destination / "preparation.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "board.json").write_text(json.dumps({"screen_id": source.sid, "variant": "full",
        "diagnostic_only": True, "review_status": "rq2_shared_engineering_layer_not_rendered",
        "source_xml": source.provenance["stages"][source.provenance["selected_stage"]]["output"]["path"],
        "compile_cleanup": cleanup, "preparation": "preparation.json"}, indent=2) + "\n", encoding="utf-8")
    return {"screen_id": source.sid, "selected_stage": source.provenance["selected_stage"],
            "s5_replay_equal": replay_equal, "native_resource_count": len(resources),
            "native_surfaces_applied": len(report["applied"]), "final_xml_sha256": sha256(final_bytes)}


def prepare_candidate(generation_root: Path, sids_file: Path, stage: str, out: Path,
                      *, reference_paths: dict[str, Path] | None = None) -> dict:
    """Build a complete fixed-cohort candidate; fail closed on missing evidence.

    ``reference_paths`` is a dependency-injection seam for tests. Normal CLI
    resolves the reference images from the project's fixed screenshot lists.
    The frozen development manifest supplies screen IDs only; its old model
    stage hashes are intentionally not used for a different generation run.
    """
    generation_root, out, sids_file = Path(generation_root).resolve(), Path(out).absolute(), Path(sids_file)
    if out.exists() or out.is_symlink():
        raise FileExistsError(f"refusing to overwrite {out}")
    resolved_out = out.resolve()
    if resolved_out == generation_root or resolved_out.is_relative_to(generation_root) or generation_root.is_relative_to(resolved_out):
        raise EvidenceError("output and generation root must be disjoint")
    sids = read_sids(sids_file)
    if not sids or len(sids) != len(set(sids)):
        raise EvidenceError("empty or duplicate screen cohort")
    if sids_file.suffix == ".json":
        cohort = json.loads(sids_file.read_text(encoding="utf-8"))
        if isinstance(cohort, dict) and cohort.get("expected_count", len(sids)) != len(sids):
            raise EvidenceError("cohort expected_count mismatch")
    references = reference_paths if reference_paths is not None else shot_map()
    missing = [sid for sid in sids if sid not in references]
    if missing:
        raise EvidenceError(f"reference paths missing for {missing}")
    sources, failures = [], []
    for sid in sids:
        try:
            sources.append(load_source(generation_root, sid, stage, Path(references[sid])))
        except (EvidenceError, OSError, ValueError, KeyError, TypeError) as exc:
            failures.append({"screen_id": sid, "error_type": type(exc).__name__, "error": str(exc)})
    if failures:
        raise CohortEvidenceError(failures)
    models = {source.manifest["model"] for source in sources}
    model_families = {APPROVED_MODEL_ALIASES.get(model, model) for model in models}
    ablations = {source.manifest.get("ablation") for source in sources}
    if len(model_families) != 1 or len(ablations) != 1:
        raise EvidenceError("mixed model or generation ablation in one candidate")
    out.mkdir(parents=True, exist_ok=False)
    records = []
    try:
        for source in sources:
            records.append(materialize_source(source, out / "full" / source.sid, stage))
        summary = {"schema_version": 1, "status": "complete", "stage": stage,
                   "generation_root": str(generation_root), "cohort": file_record(sids_file),
                   "expected_count": len(sids), "prepared_count": len(records),
                   "model": next(iter(model_families)),
                   "exact_model_ids": dict(Counter(source.manifest["model"] for source in sources)),
                   "model_alias_policy": "historical DeepSeek provider aliases; exact request identities remain recorded",
                   "generation_ablation": next(iter(ablations)),
                   "screens": records, "calls": 0, "rendered": False, "quality_claim": "none",
                   "s5_replay_equal_count": sum(item["s5_replay_equal"] is True for item in records),
                   "native_surfaces_applied": sum(item["native_surfaces_applied"] for item in records)}
        (out / "screenshots.txt").write_text("".join(f"{source.reference.resolve()}\n" for source in sources), encoding="utf-8")
        (out / "full" / "run_summary.json").write_text(json.dumps({"label": f"RQ2 shared preparation {stage}",
            "model": summary["model"], "diagnostic_only": True, "screens": [{"screen_id": sid} for sid in sids],
            "calls": 0}, indent=2) + "\n", encoding="utf-8")
        (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary
    except Exception as exc:
        (out / "FAILED.json").write_text(json.dumps({"status": "failed", "prepared_count": len(records),
            "error_type": type(exc).__name__, "error": str(exc), "downstream_allowed": False}, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--sids", type=Path, required=True)
    parser.add_argument("--stage", choices=("s3", "s5"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = prepare_candidate(args.generation_root, args.sids, args.stage, args.out)
    print(json.dumps({key: summary[key] for key in ("status", "stage", "expected_count", "prepared_count",
          "s5_replay_equal_count", "native_surfaces_applied", "calls", "rendered")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
