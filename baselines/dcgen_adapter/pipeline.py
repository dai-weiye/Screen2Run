"""Online Android-adapted DCGen orchestration."""

from __future__ import annotations

import hashlib
import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from .config import AdapterConfig, UPSTREAM_COMMIT
from .openai_compatible import OpenAICompatibleClient
from .prompts_android import LEAF_PROMPT, root_refine_prompt
from .renderer import CORRECTED_INT16, LEGACY_UINT8, Renderer, score_render

PIPELINE_PHASES = ("full", "llm_leaves", "select_and_finish")
from .xml_assembler import assemble_android_xml, leaf_segment_ids
from .xml_parser import (
    AndroidXMLParseError,
    parse_android_document,
    parse_android_fragment,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AllCandidatesFailed(RuntimeError):
    def __init__(
        self, labels: list[str], scores: list[dict[str, object]]
    ) -> None:
        self.scores = scores
        failures = [
            f"{label}={score.get('failure') or 'unknown_failure'}"
            for label, score in zip(labels, scores)
        ]
        super().__init__(
            "all Android candidates failed compilation or rendering: "
            + ", ".join(failures)
        )


def bbox_by_leaf_id(tree: dict[str, Any]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    counter = 0

    def visit(node: dict[str, Any]) -> None:
        nonlocal counter
        current = counter
        counter += 1
        children = node.get("children", [])
        if children:
            for child in children:
                visit(child)
        else:
            result[f"dcgen_seg_{current}"] = list(node["bbox"])

    visit(tree)
    return result


@dataclass
class ScreenResult:
    screen_id: str
    xml_path: str
    xml_sha256: str
    input_png: str
    input_png_sha256: str
    trace_path: str
    expected_screenshot_path: str
    resource_dir: str
    resource_files: list[dict[str, str]]


class AndroidDCGenPipeline:
    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        renderer: Renderer | None = None,
        output_dir: Path,
        metric_mode: str = LEGACY_UINT8,
        px_per_dp: float = 1.0,
    ) -> None:
        self.client = client
        self.renderer = renderer
        self.output_dir = output_dir
        self.metric_mode = metric_mode
        self.config = AdapterConfig(px_per_dp=px_per_dp)

    def _select(
        self,
        *,
        xml_candidates: list[str],
        reference: Image.Image,
        screen_id: str,
        labels: list[str],
    ) -> tuple[int, list[dict[str, object]]]:
        if self.renderer is None:
            raise RuntimeError("candidate selection requires a renderer")
        scores = [
            score_render(
                self.renderer,
                xml,
                reference,
                screen_id=screen_id,
                label=label,
            )
            for xml, label in zip(xml_candidates, labels)
        ]
        if all(math.isinf(float(score[self.metric_mode])) for score in scores):
            raise AllCandidatesFailed(labels, scores)
        selected = min(range(len(scores)), key=lambda index: float(scores[index][self.metric_mode]))
        return selected, scores

    def _generate_leaf_candidates(
        self,
        *,
        leaf_id: str,
        leaf_boxes: dict[str, list[int]],
        reference: Image.Image,
        screen_id: str,
        screen_dir: Path,
        tree: dict[str, Any],
        neutral: dict[str, str],
        trace: dict[str, Any],
    ) -> None:
        left, top, right, bottom = leaf_boxes[leaf_id]
        crop_path = screen_dir / f"{leaf_id}.png"
        if not crop_path.exists():
            reference.crop((left, top, right, bottom)).save(crop_path)
        leaf_trace = trace["leaves"].setdefault(
            leaf_id, {"bbox": leaf_boxes[leaf_id], "candidates": []}
        )
        candidates = leaf_trace["candidates"]
        for candidate_index in range(2):
            occupied = candidate_index < len(candidates) and bool(
                candidates[candidate_index].get("raw_response")
            )
            if occupied:
                candidate = candidates[candidate_index]
            else:
                completion = self.client.complete(
                    screen_id=screen_id,
                    stage=f"leaf:{leaf_id}:candidate:{candidate_index}",
                    prompt=LEAF_PROMPT,
                    image_path=crop_path,
                )
                raw_path = (
                    screen_dir
                    / "raw_responses"
                    / leaf_id
                    / f"candidate_{candidate_index}.txt"
                )
                candidate = {
                    "raw_response": completion.content,
                    "raw_response_path": str(raw_path),
                    "raw_fragment": completion.content,
                    "audit": completion.audit,
                }
                if candidate_index > len(candidates):
                    raise ValueError("resume trace has a non-contiguous candidate list")
                if candidate_index < len(candidates):
                    candidates[candidate_index] = candidate
                else:
                    candidates.append(candidate)
                trace["usage"].append(completion.audit)
                self._atomic_write(raw_path, completion.content)
                self._write_trace(screen_dir, trace)
            try:
                parsed = parse_android_fragment(
                    candidate["raw_response"], self.config.declared_resources
                )
                normalized = ET.tostring(parsed, encoding="unicode")
                candidate["raw_fragment"] = candidate["raw_response"]
                candidate["normalized_fragment"] = normalized
                candidate.pop("parse_error", None)
            except AndroidXMLParseError as exc:
                # One empty/unparseable candidate must not abort the leaf; the
                # other candidate (and MAE selection) is the published method.
                candidate["parse_error"] = str(exc)
                candidate["raw_fragment"] = candidate.get("raw_response") or ""
            self._write_trace(screen_dir, trace)

    def _leaf_candidate_xmls(
        self,
        *,
        leaf_id: str,
        candidates: list[dict[str, Any]],
        tree: dict[str, Any],
        neutral: dict[str, str],
    ) -> list[str]:
        xmls: list[str] = []
        for candidate in candidates:
            if candidate.get("parse_error"):
                xmls.append(assemble_android_xml(tree, dict(neutral), self.config))
                continue
            fragments = dict(neutral)
            fragments[leaf_id] = candidate["raw_response"]
            xmls.append(assemble_android_xml(tree, fragments, self.config))
        return xmls

    @staticmethod
    def _assembly_fragment(candidate: dict[str, Any]) -> str:
        if candidate.get("parse_error"):
            return "<View />"
        return candidate.get("raw_fragment") or "<View />"

    def _select_leaf(
        self,
        *,
        leaf_id: str,
        reference: Image.Image,
        screen_id: str,
        screen_dir: Path,
        tree: dict[str, Any],
        neutral: dict[str, str],
        trace: dict[str, Any],
    ) -> str:
        leaf_trace = trace["leaves"][leaf_id]
        candidates = leaf_trace["candidates"]
        if "selected" in leaf_trace:
            winner = candidates[leaf_trace["selected"]]
            if winner.get("raw_response"):
                return self._assembly_fragment(winner)
            leaf_trace.pop("selected", None)
        candidate_xmls = self._leaf_candidate_xmls(
            leaf_id=leaf_id,
            candidates=candidates,
            tree=tree,
            neutral=neutral,
        )
        labels = [f"{leaf_id}/candidate_0", f"{leaf_id}/candidate_1"]
        try:
            selected, scores = self._select(
                xml_candidates=candidate_xmls,
                reference=reference,
                screen_id=screen_id,
                labels=labels,
            )
        except AllCandidatesFailed as exc:
            for candidate, score in zip(candidates, exc.scores):
                candidate["scores"] = score
            leaf_trace["selection_failure"] = str(exc)
            self._write_trace(screen_dir, trace)
            raise
        for candidate, score in zip(candidates, scores):
            candidate["scores"] = score
        leaf_trace["selected"] = selected
        leaf_trace.pop("selection_failure", None)
        self._write_trace(screen_dir, trace)
        return self._assembly_fragment(candidates[selected])

    def run_screen(
        self,
        *,
        screen_id: str,
        image_path: Path,
        tree: dict[str, Any],
        resume_trace: dict[str, Any] | None = None,
        phase: str = "full",
    ) -> ScreenResult | None:
        if phase not in PIPELINE_PHASES:
            raise ValueError(f"unknown pipeline phase: {phase}")
        screen_dir = self.output_dir / screen_id
        screen_dir.mkdir(parents=True, exist_ok=True)
        reference = Image.open(image_path).convert("RGB")
        leaf_boxes = bbox_by_leaf_id(tree)
        leaf_ids = leaf_segment_ids(tree)
        neutral = {leaf_id: "<View />" for leaf_id in leaf_ids}
        deviation_path = self.output_dir / "deviations.json"
        deviations = (
            json.loads(deviation_path.read_text(encoding="utf-8"))
            if deviation_path.exists()
            else []
        )
        trace: dict[str, Any] = resume_trace or {
            "screen_id": screen_id,
            "upstream_commit": UPSTREAM_COMMIT,
            "created_at": utc_now(),
            "bbox_tree": tree,
            "leaf_count": len(leaf_ids),
            "expected_calls": 2 * len(leaf_ids) + 1,
            "metric_default": LEGACY_UINT8,
            "metric_selected": self.metric_mode,
            "corrected_sensitivity_recorded": True,
            "deviations": deviations,
            "leaves": {},
            "usage": [],
        }
        self._write_trace(screen_dir, trace)

        # Phase A: every leaf LLM call first. Leaf prompts only need the crop;
        # they do not depend on prior MAE winners. Holding the emulator here
        # just serializes paid API time behind Gradle.
        for leaf_id in leaf_ids:
            self._generate_leaf_candidates(
                leaf_id=leaf_id,
                leaf_boxes=leaf_boxes,
                reference=reference,
                screen_id=screen_id,
                screen_dir=screen_dir,
                tree=tree,
                neutral=neutral,
                trace=trace,
            )
        if phase == "llm_leaves":
            return None

        selected_fragments = {
            leaf_id: self._select_leaf(
                leaf_id=leaf_id,
                reference=reference,
                screen_id=screen_id,
                screen_dir=screen_dir,
                tree=tree,
                neutral=neutral,
                trace=trace,
            )
            for leaf_id in leaf_ids
        }

        assembled = assemble_android_xml(tree, selected_fragments, self.config)
        raw_path = screen_dir / "assembled_raw.xml"
        self._atomic_write(raw_path, assembled + "\n")
        root_trace = trace.get("root_refine")
        if not root_trace or not root_trace.get("raw_response"):
            completion = self.client.complete(
                screen_id=screen_id,
                stage="root_refine",
                prompt=root_refine_prompt(assembled),
                image_path=image_path,
            )
            response_path = screen_dir / "raw_responses" / "root_refine.txt"
            root_trace = {
                "raw_response": completion.content,
                "raw_response_path": str(response_path),
                "audit": completion.audit,
            }
            trace["root_refine"] = root_trace
            trace["usage"].append(completion.audit)
            self._atomic_write(response_path, completion.content)
            self._write_trace(screen_dir, trace)
        if "selected" not in root_trace:
            labels = ["root_refine/refined", "root_refine/assembled_raw"]
            try:
                refined_root = parse_android_document(
                    root_trace["raw_response"], self.config.declared_resources
                )
                refined = ET.tostring(refined_root, encoding="unicode")
                root_trace["normalized_document"] = refined
                refined_score = score_render(
                    self.renderer,
                    refined,
                    reference,
                    screen_id=screen_id,
                    label=labels[0],
                )
                root_trace.pop("parse_failure", None)
            except AndroidXMLParseError as exc:
                refined_score = {
                    LEGACY_UINT8: float("inf"),
                    CORRECTED_INT16: float("inf"),
                    "failure": "root_refined_parse_error",
                    "details": {
                        "phase": "parse",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_response_path": root_trace.get("raw_response_path", ""),
                    },
                }
                root_trace["parse_failure"] = {
                    "failure_code": "root_refined_parse_error",
                    "error_type": type(exc).__name__,
                    "detail": str(exc),
                    "raw_response_path": root_trace.get("raw_response_path", ""),
                }
            self._write_trace(screen_dir, trace)
            raw_score = score_render(
                self.renderer,
                assembled,
                reference,
                screen_id=screen_id,
                label=labels[1],
            )
            scores = [refined_score, raw_score]
            root_trace["candidates"] = ["refined", "assembled_raw"]
            root_trace["scores"] = scores
            root_trace["assembled_raw_validation"] = {
                "failure_code": raw_score.get("failure", ""),
                "valid": not math.isinf(float(raw_score[self.metric_mode])),
                "evidence": raw_score.get("details", {}),
            }
            if all(
                math.isinf(float(score[self.metric_mode])) for score in scores
            ):
                exc = AllCandidatesFailed(labels, scores)
                root_trace["selection_failure"] = str(exc)
                root_trace["selection_reason"] = (
                    "both root candidates invalid; no fallback candidate passed "
                    "Android validation"
                )
                self._write_trace(screen_dir, trace)
                raise exc
            refined_invalid = math.isinf(float(refined_score[self.metric_mode]))
            raw_invalid = math.isinf(float(raw_score[self.metric_mode]))
            if refined_invalid:
                selected = 1
                selection_reason = (
                    f"refined invalid ({refined_score.get('failure')}); "
                    "assembled raw passed Android validation"
                )
            elif raw_invalid:
                selected = 0
                selection_reason = (
                    f"assembled raw invalid ({raw_score.get('failure')}); "
                    "refined passed Android validation"
                )
            else:
                selected = min(
                    range(2),
                    key=lambda index: float(scores[index][self.metric_mode]),
                )
                selection_reason = (
                    f"both candidates valid; selected minimum {self.metric_mode} MAE"
                )
            root_trace.update(
                {
                    "selected": selected,
                    "selection_reason": selection_reason,
                }
            )
            root_trace.pop("selection_failure", None)
            self._write_trace(screen_dir, trace)
        final_xml = (
            trace["root_refine"]["normalized_document"]
            if trace["root_refine"]["selected"] == 0
            else assembled
        )
        final_path = screen_dir / f"{screen_id}.xml"
        if final_path.exists() and final_path.read_text(encoding="utf-8").strip() != final_xml.strip():
            raise FileExistsError(f"refusing to overwrite differing output: {final_path}")
        final_path.write_text(final_xml + "\n", encoding="utf-8")
        drawable = screen_dir / "img.xml"
        if not drawable.exists():
            drawable.write_text(
                '<shape xmlns:android="http://schemas.android.com/apk/res/android" '
                'android:shape="rectangle"><solid android:color="#00000000"/></shape>\n',
                encoding="utf-8",
            )
        trace_path = screen_dir / "trace.json"
        self._write_trace(screen_dir, trace)
        return ScreenResult(
            screen_id=screen_id,
            xml_path=str(final_path.resolve()),
            xml_sha256=sha256_file(final_path),
            input_png=str(image_path.resolve()),
            input_png_sha256=sha256_file(image_path),
            trace_path=str(trace_path.resolve()),
            expected_screenshot_path=f"raw/screenshots/{screen_id}.png",
            resource_dir=str(screen_dir.resolve()),
            resource_files=[{"path": str(drawable.resolve()), "sha256": sha256_file(drawable)}],
        )

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

    @classmethod
    def _write_trace(
        cls, screen_dir: Path, trace: dict[str, Any]
    ) -> None:
        path = screen_dir / "trace.json"
        cls._atomic_write(
            path, json.dumps(trace, indent=2, ensure_ascii=False) + "\n"
        )
        ledger = "".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
            for item in trace.get("usage", [])
        )
        cls._atomic_write(screen_dir / "ledger.jsonl", ledger)


def manifest_for_run(
    *,
    run_id: str,
    model_snapshot: str,
    result: ScreenResult,
    price: dict[str, float],
    caps: dict[str, Any],
    dataset: str = "Redraw_D",
) -> dict[str, Any]:
    trace = json.loads(Path(result.trace_path).read_text(encoding="utf-8"))
    usage = trace.get("usage", [])
    resolved_models = sorted(
        {
            item["resolved_model"]
            for item in usage
            if isinstance(item.get("resolved_model"), str)
        }
    )
    return {
        "schema_version": "1.0",
        "run_id": run_id,
        "dataset": dataset,
        "method": "android_adapted_dcgen",
        "model": {
            "identifier": model_snapshot,
            "provider": "openai_compatible",
            "revision": model_snapshot,
        },
        "screen_count": 1,
        "expected_screen_count": 1,
        "items_file": "items.jsonl",
        "events_file": "events.jsonl",
        "selection": {"screen_ids": [result.screen_id]},
        "deviations": trace.get("deviations", []),
        "upstream_commit": UPSTREAM_COMMIT,
        "price": price,
        "hard_caps": caps,
        "call_budget": {
            "planned": trace.get("expected_calls", 2 * len(trace.get("leaves", {})) + 1),
            "actual": len(usage),
            "formula": "2L+1",
        },
        "cost": {
            "amount": sum(float(item.get("cost_usd", 0.0)) for item in usage),
            "currency": "USD",
            "basis": "provider tokens and configured prices",
            "input_tokens": sum(int(item.get("input_tokens", 0)) for item in usage),
            "output_tokens": sum(int(item.get("output_tokens", 0)) for item in usage),
        },
        "resolved_models": resolved_models,
        "created_at": utc_now(),
    }
