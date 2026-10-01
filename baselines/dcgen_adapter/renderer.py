"""Renderer contract and frozen Android-harness implementation."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

LEGACY_UINT8 = "legacy_uint8"
CORRECTED_INT16 = "corrected_int16"
METRIC_MODES = (LEGACY_UINT8, CORRECTED_INT16)
INFRA_RENDER_FAILURES = frozenset({"environment_missing", "screenshot_missing"})


def is_infra_render_failure(failure: str) -> bool:
    """Host/guest flakes. Do not cache these as a candidate outcome."""

    if failure in INFRA_RENDER_FAILURES:
        return True
    return failure.startswith("harness_execution_failed")


def image_mae(reference: Image.Image, candidate: Image.Image, mode: str) -> float:
    """Compute explicit upstream-compatible or corrected MAE."""

    if mode not in METRIC_MODES:
        raise ValueError(f"unknown MAE mode: {mode}")
    candidate = candidate.convert("RGB").resize(reference.size)
    left = np.asarray(reference.convert("RGB"))
    right = np.asarray(candidate)
    if mode == LEGACY_UINT8:
        return float(np.mean(np.abs(left - right)))
    return float(np.mean(np.abs(left.astype(np.int16) - right.astype(np.int16))))


@dataclass(frozen=True)
class RenderResult:
    image: Image.Image | None
    failure: str = ""
    details: dict[str, object] | None = None

    @property
    def succeeded(self) -> bool:
        return self.image is not None and not self.failure


class Renderer(ABC):
    """Test seam for candidate compilation and screenshot capture."""

    @abstractmethod
    def render(self, xml: str, *, screen_id: str, label: str) -> RenderResult:
        raise NotImplementedError


class FrozenHarnessRenderer(Renderer):
    """Invoke the repository's frozen Gradle/emulator runner via a minimal run."""

    def __init__(
        self,
        *,
        repo_root: Path,
        harness_script: Path,
        avd_snapshot: str,
        evidence_root: Path,
        input_png: Path,
        model: dict[str, str],
        project: Path | None = None,
        device_serial: str = "",
        timeout_seconds: int = 900,
        dataset: str = "Redraw_D",
        fast_candidate_render: bool = True,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.harness_script = harness_script.resolve()
        self.avd_snapshot = avd_snapshot
        self.evidence_root = evidence_root.resolve()
        self.input_png = input_png.resolve()
        self.model = model
        self.project = (project or self.repo_root / "artifacts/android_test_project").resolve()
        self.device_serial = device_serial
        self.timeout_seconds = timeout_seconds
        self.dataset = dataset
        self.fast_candidate_render = fast_candidate_render

    @staticmethod
    def _sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _write(path: Path, content: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)

    def _attempt_dir(self, label: str) -> Path:
        candidate_dir = self.evidence_root / Path(label)
        candidate_dir.mkdir(parents=True, exist_ok=True)
        attempts = sorted(candidate_dir.glob("attempt_*"))
        attempt = candidate_dir / f"attempt_{len(attempts) + 1:03d}"
        attempt.mkdir()
        return attempt

    def _cached_result(
        self, label: str, xml: str, screen_id: str
    ) -> RenderResult | None:
        candidate_dir = self.evidence_root / Path(label)
        expected_hash = hashlib.sha256((xml + "\n").encode("utf-8")).hexdigest()
        for run_dir in reversed(sorted(candidate_dir.glob("attempt_*"))):
            summary_path = run_dir / "evidence_summary.json"
            status_path = run_dir / "status.json"
            if not summary_path.is_file() or not status_path.is_file():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("xml_sha256") != expected_hash:
                continue
            status = json.loads(status_path.read_text(encoding="utf-8"))
            details = {
                **status,
                "evidence_dir": str(run_dir),
                "evidence_summary": summary,
                "reused": True,
            }
            failure = str(summary.get("failure_code") or "")
            if failure:
                if is_infra_render_failure(failure):
                    continue
                return RenderResult(None, failure, details)
            screenshot = run_dir / "raw" / "screenshots" / f"{screen_id}.png"
            if screenshot.is_file():
                with Image.open(screenshot) as image:
                    return RenderResult(image.convert("RGB").copy(), details=details)
        return None

    def _harness_command(self, run_dir: Path) -> list[str]:
        command = [
            os.environ.get("PYTHON", "python"),
            str(self.harness_script),
            "--run-dir",
            str(run_dir),
            "--project",
            str(self.project),
            "--overwrite",
        ]
        # "none" selects the resident-guest device protocol of the shared
        # harness, used where per-screen snapshot restore leaves the guest
        # without a serving package manager. The adapter must pass the same
        # protocol as every other block validated in the same session.
        if self.avd_snapshot == "none":
            command.append("--no-snapshot-restore")
        else:
            command.extend(["--avd-snapshot", self.avd_snapshot])
        if self.device_serial:
            command.extend(["--device-serial", self.device_serial])
        if self.fast_candidate_render:
            command.append("--fast-candidate-render")
        return command

    def _finish_evidence(
        self,
        run_dir: Path,
        *,
        status: dict[str, object],
        completed: subprocess.CompletedProcess[str] | None,
        failure: str,
    ) -> dict[str, object]:
        stdout = completed.stdout if completed else ""
        stderr = completed.stderr if completed else failure
        self._write(run_dir / "harness_stdout.log", stdout)
        self._write(run_dir / "harness_stderr.log", stderr)
        self._write(
            run_dir / "status.json",
            json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        )
        gradle_parts = []
        for path in sorted((run_dir / "raw" / "gradle_logs").glob("*.log")):
            gradle_parts.append(path.read_text(encoding="utf-8", errors="replace"))
        self._write(run_dir / "gradle.log", "\n".join(gradle_parts))
        summary: dict[str, object] = {
            "created_at": datetime.now(timezone.utc).astimezone().isoformat(),
            "failure_code": failure,
            "xml_sha256": self._sha(run_dir / "candidate.xml"),
            "manifest_sha256": self._sha(run_dir / "manifest.json"),
            "items_sha256": self._sha(run_dir / "items.jsonl"),
            "input_png_sha256": self._sha(self.input_png),
            "status_sha256": self._sha(run_dir / "status.json"),
            "gradle_log_sha256": self._sha(run_dir / "gradle.log"),
            "harness_stdout_sha256": self._sha(run_dir / "harness_stdout.log"),
            "harness_stderr_sha256": self._sha(run_dir / "harness_stderr.log"),
        }
        screenshot = run_dir / "raw" / "screenshots" / status.get("screen_id", "")
        screenshot = screenshot.with_suffix(".png")
        if screenshot.is_file():
            summary["screenshot_sha256"] = self._sha(screenshot)
        self._write(
            run_dir / "evidence_summary.json",
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        )
        return {**status, "evidence_dir": str(run_dir), "evidence_summary": summary}

    def render(self, xml: str, *, screen_id: str, label: str) -> RenderResult:
        if not self.avd_snapshot:
            return RenderResult(None, "missing_avd_snapshot")
        cached = self._cached_result(label, xml, screen_id)
        if cached is not None:
            return cached
        last: RenderResult | None = None
        for infra_attempt in range(3):
            last = self._render_once(xml, screen_id=screen_id, label=label)
            if last.succeeded or not is_infra_render_failure(last.failure):
                return last
            if infra_attempt < 2:
                time.sleep(2.0 * (infra_attempt + 1))
        assert last is not None
        return last

    def _render_once(self, xml: str, *, screen_id: str, label: str) -> RenderResult:
        run_dir = self._attempt_dir(label)
        artifact = run_dir / "candidate.xml"
        self._write(artifact, xml + "\n")
        drawable = run_dir / "img.xml"
        self._write(
            drawable,
            '<shape xmlns:android="http://schemas.android.com/apk/res/android" '
            'android:shape="rectangle"><solid android:color="#00000000"/></shape>\n',
        )
        run_id = f"dcgen-render-{screen_id}-{label.replace('/', '-')}-{run_dir.name}"
        manifest = {
            "schema_version": "1.0",
            "run_id": run_id,
            "dataset": self.dataset,
            "dataset_version": None,
            "method": "android_adapted_dcgen_candidate_render",
            "model": self.model,
            "resolved_models": [self.model["revision"]],
            "screen_count": 1,
            "expected_screen_count": 1,
            "items_file": "items.jsonl",
            "events_file": "events.jsonl",
            "reference_dir": str(self.input_png.parent),
            "selection": {"screen_ids": [screen_id], "selection_rule": "single candidate"},
            "deviations": [],
            "call_budget": {"planned": 0, "actual": 0, "by_stage": {}},
            "cost": {
                "amount": 0.0,
                "basis": "renderer-only",
                "currency": "USD",
                "input_tokens": 0,
                "output_tokens": 0,
            },
        }
        item = {
            "run_id": run_id,
            "screen_id": screen_id,
            "input_png": str(self.input_png),
            "input_png_sha256": self._sha(self.input_png),
            "expected_screenshot_path": f"raw/screenshots/{screen_id}.png",
            "xml_path": str(artifact),
            "xml_sha256": self._sha(artifact),
            "resource_dir": str(run_dir),
            "resource_files": [{"path": str(drawable), "sha256": self._sha(drawable)}],
        }
        self._write(
            run_dir / "manifest.json",
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        )
        self._write(run_dir / "items.jsonl", json.dumps(item, sort_keys=True) + "\n")
        self._write(run_dir / "events.jsonl", "")
        command = self._harness_command(run_dir)
        completed: subprocess.CompletedProcess[str] | None = None
        try:
            completed = subprocess.run(
                command,
                cwd=self.repo_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                errors="replace",
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            reason = f"harness_execution_failed:{exc}"
            details = self._finish_evidence(
                run_dir, status={"screen_id": screen_id}, completed=None, failure=reason
            )
            return RenderResult(None, reason, details)
        status_path = run_dir / "derived" / "loading_status.jsonl"
        status: dict[str, object] = {"screen_id": screen_id}
        if status_path.exists():
            lines = status_path.read_text(encoding="utf-8").strip().splitlines()
            status = json.loads(lines[-1]) if lines else status
        screenshot = run_dir / "raw" / "screenshots" / f"{screen_id}.png"
        reason = str(
            status.get("failure_code")
            or (f"harness_exit_{completed.returncode}" if completed.returncode else "")
            or ("" if screenshot.exists() else "screenshot_missing")
        )
        details = self._finish_evidence(
            run_dir, status=status, completed=completed, failure=reason
        )
        if reason:
            return RenderResult(None, reason, details)
        with Image.open(screenshot) as image:
            return RenderResult(image.convert("RGB").copy(), details=details)


def score_render(
    renderer: Renderer,
    xml: str,
    reference: Image.Image,
    *,
    screen_id: str,
    label: str,
) -> dict[str, object]:
    result = renderer.render(xml, screen_id=screen_id, label=label)
    if not result.succeeded:
        return {
            LEGACY_UINT8: float("inf"),
            CORRECTED_INT16: float("inf"),
            "failure": result.failure or "render_failed",
            "details": result.details or {},
        }
    assert result.image is not None
    return {
        LEGACY_UINT8: image_mae(reference, result.image, LEGACY_UINT8),
        CORRECTED_INT16: image_mae(reference, result.image, CORRECTED_INT16),
        "failure": "",
        "details": result.details or {},
    }
