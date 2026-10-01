#!/usr/bin/env python3
"""Run Android loading experiments from an artifact run manifest.

This runner is manifest-driven and writes one structured status record per
screen. It supports a static-only mode for auditing XML/resource readiness
without touching an Android project, and a full mode that replaces the test
project layout, builds, installs, launches, and captures screenshots.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import threading
import time
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Queue
from typing import Any


import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO_ROOT = release_paths.RELEASE

# See docs/RUNTIME.md for the shared compatibility and capture contracts.
CANVAS_WIDTH = 1080

ANDROID_XML_MARKERS = (
    "LinearLayout",
    "RelativeLayout",
    "ConstraintLayout",
    "ScrollView",
    "FrameLayout",
    "CoordinatorLayout",
    "TextView",
    "ImageView",
    "Button",
    "EditText",
    "RecyclerView",
    "CheckBox",
    "Switch",
    "RadioButton",
    "Spinner",
    "SeekBar",
    "ProgressBar",
    "CardView",
    "FloatingActionButton",
    "TabLayout",
    "BottomNavigationView",
)

DRAWABLE_REF_RE = re.compile(r"@drawable/([A-Za-z0-9_]+)")
VALID_RESOURCE_NAME_RE = re.compile(r"^[a-z][a-z0-9_]*$")
# Pix2Code-Easy ids. RQ1 Easy scores XML F1/BLEU, not the uiautomator dump.
PIX2CODE_ID_RE = re.compile(
    r"^[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
)


def is_pix2code_easy_id(screen_id: str) -> bool:
    return bool(PIX2CODE_ID_RE.match(screen_id))

AVD_LOCK_DIR = release_paths.CANDIDATES / "i48"

FAILURE_CODES = {
    "artifact_hash_mismatch",
    "missing_output",
    "non_xml_output",
    "xml_parse_error",
    "missing_drawable",
    "invalid_resource_name",
    "gradle_compile_error",
    "apk_missing",
    "apk_install_failed",
    "activity_launch_failed",
    "activity_crash",
    "launch_timeout",
    "screenshot_failed",
    "blank_or_wrong_activity",
    "metric_input_missing",
    "environment_missing",
    "unexpected_exception",
}


@dataclass(frozen=True)
class RunnerConfig:
    run_dir: Path
    project: Path
    android_home: Path
    adb: Path
    app_package: str
    app_activity: str
    device_serial: str
    avd_snapshot: str
    screenshot_delay: float
    screenshot_settle_seconds: float
    screenshot_settle_interval: float
    foreground_poll_seconds: float
    static_only: bool
    limit: int | None
    screen_ids: set[str] | None
    overwrite: bool
    fast_candidate_render: bool = False
    resume_status: bool = False
    no_gradle_clean: bool = False
    force_hierarchy_dump: bool = False
    skip_hierarchy_dump: bool = False
    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    match_reference_canvas: bool = True

    @property
    def device_lock_path(self) -> Path:
        """One lock per emulator, shared by every harness process on this host.

        Concurrent runs install the same ``app_package`` on the same device; without
        serialising install → launch → capture per item, one process photographs the
        other's layout. The lock is held only for the device steps, so Gradle builds
        from different projects still overlap.
        """
        import tempfile

        serial = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.device_serial or "default")
        return Path(tempfile.gettempdir()) / f"ese-device-{serial}.lock"

    @property
    def target_xml_path(self) -> Path:
        return self.project / "app" / "src" / "main" / "res" / "layout" / "activity_main.xml"

    @property
    def drawable_path(self) -> Path:
        return self.project / "app" / "src" / "main" / "res" / "drawable"

    @property
    def apk_dir_path(self) -> Path:
        return self.project / "app" / "build" / "outputs" / "apk" / "debug"

    @property
    def gradlew(self) -> Path:
        return self.project / "gradlew"


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def repo_path(rel_path: str) -> Path:
    return REPO_ROOT / rel_path if rel_path else Path("")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def captured_text(value: Any) -> str:
    """subprocess.TimeoutExpired can leave stdout/stderr as bytes even with text=True."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value if isinstance(value, str) else str(value)


def json_default(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default) + "\n")


def strip_markdown_fence(content: str) -> str:
    fenced = re.search(r"```(?:xml)?\s*(.*?)```", content, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return content.strip()


def xml_candidate(content: str) -> str:
    stripped = strip_markdown_fence(content)
    start = stripped.find("<")
    end = stripped.rfind(">")
    if start == -1 or end == -1 or end <= start:
        return ""
    return stripped[start : end + 1].strip()


def is_xml_like(candidate: str) -> bool:
    if not candidate or not candidate.startswith("<"):
        return False
    return any(re.search(rf"<(?:[\w.]+[.:])?{re.escape(marker)}\b", candidate) for marker in ANDROID_XML_MARKERS)


def parse_xml(candidate: str) -> tuple[ET.Element | None, str]:
    try:
        return ET.fromstring(candidate), ""
    except ET.ParseError as error:
        return None, str(error)


def command_record(
    cmd: list[str],
    cwd: Path | None = None,
    timeout: int = 120,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    printable_cmd = [str(part) for part in cmd]
    try:
        proc = subprocess.Popen(
            printable_cmd,
            cwd=str(cwd) if cwd else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            env=env,
            start_new_session=True,
        )
    except Exception as error:
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "cmd": printable_cmd,
            "cwd": str(cwd) if cwd else "",
            "returncode": -1,
            "duration_ms": duration_ms,
            "stdout": "",
            "stderr": str(error),
            "timed_out": False,
        }
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "cmd": printable_cmd,
            "cwd": str(cwd) if cwd else "",
            "returncode": proc.returncode,
            "duration_ms": duration_ms,
            "stdout": stdout,
            "stderr": stderr,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = proc.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, stderr = proc.communicate()
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "cmd": printable_cmd,
            "cwd": str(cwd) if cwd else "",
            "returncode": -1,
            "duration_ms": duration_ms,
            "stdout": captured_text(stdout if stdout is not None else error.stdout),
            "stderr": captured_text(stderr if stderr is not None else error.stderr) or "Timeout",
            "timed_out": True,
        }
    except Exception as error:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
        duration_ms = int((time.monotonic() - started) * 1000)
        return {
            "cmd": printable_cmd,
            "cwd": str(cwd) if cwd else "",
            "returncode": -1,
            "duration_ms": duration_ms,
            "stdout": "",
            "stderr": str(error),
            "timed_out": False,
        }


def runner_env(config: RunnerConfig) -> dict[str, str]:
    env = dict(os.environ)
    java_home = release_paths.JAVA_HOME
    gradle_home = release_paths.GRADLE_HOMES / "default"
    tmp_dir = release_paths.WORK / "tmp"
    gradle_home.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    env.setdefault("JAVA_HOME", str(java_home))
    env.setdefault("ANDROID_HOME", str(config.android_home))
    env.setdefault("GRADLE_USER_HOME", str(gradle_home))
    env.setdefault("TMPDIR", str(tmp_dir))
    existing_java_opts = env.get("JAVA_OPTS", "")
    tmp_opt = f"-Djava.io.tmpdir={tmp_dir}"
    if tmp_opt not in existing_java_opts:
        env["JAVA_OPTS"] = f"{existing_java_opts} {tmp_opt}".strip()
    return env


def adb_cmd(config: RunnerConfig, *args: str) -> list[str]:
    cmd = [str(config.adb)]
    if config.device_serial:
        cmd.extend(["-s", config.device_serial])
    cmd.extend(args)
    return cmd


def reference_canvas(item: dict[str, Any]) -> tuple[int, int] | None:
    "Return a 1080-pixel-wide canvas with the reference aspect ratio and unchanged density."
    path = repo_path(item.get("input_png", ""))
    if not path.is_file():
        return None
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
    except Exception:
        return None
    if w <= 0 or h <= 0:
        return None
    return (CANVAS_WIDTH, max(1, round(h * CANVAS_WIDTH / w)))


def match_canvas_to_reference(
    config: RunnerConfig, item: dict[str, Any], status: dict[str, Any]
) -> None:
    "Apply the reference-aspect canvas without changing density; log failures."
    target = reference_canvas(item)
    if target is None:
        return
    commands = status["commands"]
    want = f"{target[0]}x{target[1]}"
    try:
        cur = subprocess.run(adb_cmd(config, "shell", "wm", "size"),
                             capture_output=True, text=True, timeout=20)
        if re.search(r"Override size:\s*" + re.escape(want) + r"\b", cur.stdout):
            commands["wm_size"] = {"requested": want, "skipped": "already_applied"}
            return
        res = command_record(adb_cmd(config, "shell", "wm", "size", want),
                             timeout=20, env=runner_env(config))
        commands["wm_size"] = compact_command_result(res)
        # See docs/RUNTIME.md for the shared compatibility and capture contracts.
        time.sleep(1.5)
    except Exception as exc:  # See docs/RUNTIME.md for the shared compatibility and capture contracts.
        commands["wm_size"] = {"requested": want, "error": str(exc)[:200]}


def canvas_key(item: dict[str, Any], enabled: bool = True) -> str:
    "Return the expected canvas fingerprint for capture-cache validation."
    if not enabled:
        return ""
    size = reference_canvas(item)
    return f"{size[0]}x{size[1]}" if size else ""


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(captured_text(content), encoding="utf-8", errors="replace")


def compact_command_result(result: dict[str, Any]) -> dict[str, Any]:
    stdout = captured_text(result.get("stdout"))
    stderr = captured_text(result.get("stderr"))
    return {
        "returncode": result["returncode"],
        "duration_ms": result["duration_ms"],
        "timed_out": result["timed_out"],
        "stdout_tail": stdout[-1000:],
        "stderr_tail": stderr[-1000:],
    }


def referenced_drawables(candidate: str) -> set[str]:
    return set(DRAWABLE_REF_RE.findall(candidate))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_static(item: dict[str, Any]) -> tuple[dict[str, Any], str, str]:
    stages: dict[str, Any] = {
        "xml_exists": False,
        "xml_like": False,
        "xml_well_formed": False,
        "android_namespace": False,
        "resources_resolved": False,
        "artifact_hashes": False,
    }
    xml_path = repo_path(item.get("xml_path", ""))
    if not item.get("xml_path") or not xml_path.exists():
        return stages, "missing_output", "XML output path is missing or does not exist."
    if item.get("xml_sha256") and file_sha256(xml_path) != item["xml_sha256"]:
        return stages, "artifact_hash_mismatch", f"XML hash mismatch: {item['xml_path']}"
    input_path = repo_path(item.get("input_png", ""))
    if item.get("input_png_sha256") and (
        not input_path.is_file() or file_sha256(input_path) != item["input_png_sha256"]
    ):
        return stages, "artifact_hash_mismatch", f"Input image hash mismatch: {item.get('input_png', '')}"
    for file_info in item.get("resource_files", []):
        resource_path = repo_path(file_info.get("path", ""))
        if not resource_path.is_file():
            return stages, "artifact_hash_mismatch", f"Declared resource is missing: {file_info.get('path', '')}"
        if file_info.get("sha256") and file_sha256(resource_path) != file_info["sha256"]:
            return stages, "artifact_hash_mismatch", f"Resource hash mismatch: {file_info['path']}"
    stages["artifact_hashes"] = True

    content = xml_path.read_text(encoding="utf-8", errors="ignore")
    candidate = xml_candidate(content)
    stages["xml_exists"] = True
    stages["xml_like"] = is_xml_like(candidate)
    if not stages["xml_like"]:
        return stages, "non_xml_output", "Output does not contain an Android XML-like layout."

    _, parse_error = parse_xml(candidate)
    if parse_error:
        return stages, "xml_parse_error", parse_error
    stages["xml_well_formed"] = True
    stages["android_namespace"] = "xmlns:android" in candidate

    resource_names = {
        Path(file_info["path"]).stem
        for file_info in item.get("resource_files", [])
        if file_info.get("path")
    }
    drawable_refs = {name for name in referenced_drawables(candidate) if not name.startswith("@")}
    invalid_names = sorted(name for name in drawable_refs if not VALID_RESOURCE_NAME_RE.match(name))
    if invalid_names:
        return stages, "invalid_resource_name", f"Invalid drawable resource names: {', '.join(invalid_names[:10])}"

    missing = sorted(name for name in drawable_refs if name not in resource_names)
    if missing:
        return stages, "missing_drawable", f"Missing drawable resources: {', '.join(missing[:10])}"

    stages["resources_resolved"] = True
    return stages, "", ""


def environment_status(config: RunnerConfig) -> tuple[dict[str, Any], bool]:
    checks = {
        "project": config.project.exists(),
        "target_xml_parent": config.target_xml_path.parent.exists(),
        "drawable_dir": config.drawable_path.exists(),
        "gradlew": config.gradlew.exists(),
        "adb": config.adb.exists(),
    }
    if config.static_only:
        return checks, True

    ok = all(checks.values())
    if ok:
        devices = command_record(adb_cmd(config, "devices"), timeout=30, env=runner_env(config))
        connected = devices["returncode"] == 0 and any(
            "\tdevice" in line for line in devices["stdout"].splitlines()[1:]
        )
        checks["adb_device_connected"] = connected
        ok = ok and connected
    return checks, ok


def copy_resources(item: dict[str, Any], config: RunnerConfig) -> list[str]:
    config.drawable_path.mkdir(parents=True, exist_ok=True)
    copied = []
    for old in config.drawable_path.iterdir():
        if old.is_file():
            old.unlink()
    for file_info in item.get("resource_files", []):
        source = repo_path(file_info["path"])
        if source.exists():
            target = config.drawable_path / source.name
            shutil.copy2(source, target)
            copied.append(str(target))
    return copied


def prepare_project(item: dict[str, Any], config: RunnerConfig) -> None:
    xml_path = repo_path(item["xml_path"])
    config.target_xml_path.parent.mkdir(parents=True, exist_ok=True)
    copy_resources(item, config)
    shutil.copy2(xml_path, config.target_xml_path)
    if config.gradlew.exists() and not os.access(config.gradlew, os.X_OK):
        os.chmod(config.gradlew, 0o755)


def gradle_assemble_args(config: RunnerConfig) -> list[str]:
    """Formal scoring cleans; DCGen candidate MAE only needs an incremental assemble.

    ``prepare_project`` already replaces ``activity_main.xml`` and wipes
    ``res/drawable``, so ``assembleDebug`` without ``clean`` yields the same APK.
    Recapture passes ``--no-gradle-clean`` to skip the redundant clean.
    Lint/unit-test tasks are never the layout under test.
    """
    extra = ["-x", "lint", "-x", "test"]
    if config.fast_candidate_render or config.no_gradle_clean:
        return ["assembleDebug", *extra]
    return ["clean", "assembleDebug", *extra]


PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
HIERARCHY_DUMP_TIMEOUT = 12


def screencap_to_file(config: RunnerConfig, dest: Path) -> dict[str, Any]:
    """One adb round-trip: ``exec-out screencap -p`` writes the PNG locally.

    Falls back to ``screencap`` + ``pull`` if the emulator prefixes junk or
    CRLF-corrupts stdout, which some host adb builds still do.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    cmd = adb_cmd(config, "exec-out", "screencap", "-p")
    printable = [str(part) for part in cmd]
    try:
        result = subprocess.run(
            printable,
            capture_output=True,
            timeout=30,
            env=runner_env(config),
        )
        duration_ms = int((time.monotonic() - started) * 1000)
        data = result.stdout or b""
        idx = data.find(PNG_MAGIC)
        if result.returncode == 0 and idx >= 0:
            dest.write_bytes(data[idx:])
            return {
                "cmd": printable,
                "cwd": "",
                "returncode": 0,
                "duration_ms": duration_ms,
                "stdout": "",
                "stderr": captured_text(result.stderr),
                "timed_out": False,
            }
    except subprocess.TimeoutExpired as error:
        return {
            "cmd": printable,
            "cwd": "",
            "returncode": -1,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "stdout": "",
            "stderr": captured_text(error.stderr) or "Timeout",
            "timed_out": True,
        }
    except Exception as error:
        return {
            "cmd": printable,
            "cwd": "",
            "returncode": -1,
            "duration_ms": int((time.monotonic() - started) * 1000),
            "stdout": "",
            "stderr": str(error),
            "timed_out": False,
        }

    remote = f"/sdcard/gui_codegen_screencap_{os.getpid()}.png"
    capture = command_record(adb_cmd(config, "shell", "screencap", "-p", remote), timeout=30, env=runner_env(config))
    pull = command_record(adb_cmd(config, "pull", remote, str(dest)), timeout=60, env=runner_env(config))
    command_record(adb_cmd(config, "shell", "rm", "-f", remote), timeout=30, env=runner_env(config))
    pull["cmd"] = printable + ["#fallback-pull"]
    pull["duration_ms"] = int(capture.get("duration_ms") or 0) + int(pull.get("duration_ms") or 0)
    return pull


def extract_hierarchy_xml(blob: str) -> str:
    """Keep the uiautomator document; ``dump /dev/tty`` prefixes a status line."""
    text = captured_text(blob)
    start = text.find("<?xml")
    if start < 0:
        start = text.find("<hierarchy")
    if start < 0:
        return ""
    text = text[start:]
    end = text.rfind("</hierarchy>")
    if end >= 0:
        text = text[: end + len("</hierarchy>")]
    return text.strip()


def dumpsys_focus(config: RunnerConfig) -> dict[str, Any]:
    """Foreground probe. ``dumpsys activity activities`` is the resumed task, not the 10 MB WM dump.

    Piping ``dumpsys window | grep`` through ``sh -c`` hung the guest; the full window
    dump is also too slow on a loaded AVD. Activity manager already prints
    ``topResumedActivity`` / ``mResumedActivity`` with the package name.
    """
    return command_record(
        adb_cmd(config, "shell", "dumpsys", "activity", "activities"),
        timeout=10,
        env=runner_env(config),
    )


def newest_apk(config: RunnerConfig) -> Path | None:
    apk_files = sorted(config.apk_dir_path.glob("*.apk"), key=lambda path: path.stat().st_mtime, reverse=True)
    return apk_files[0] if apk_files else None


def non_empty_png(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return handle.read(8) == b"\x89PNG\r\n\x1a\n"


def frame_is_blank(path: Path) -> bool:
    """True when the app area below the status bar is one flat near-black field.

    swiftshader emulators under load sometimes hand back the surface before the window's
    first frame: the navigation handle is drawn, everything above it is black. A layout that
    really is black still carries glyphs or edges, so it has spread; this frame has none.
    """
    from PIL import Image
    import numpy as np

    with Image.open(path) as im:
        a = np.asarray(im.convert("L"), dtype=np.float32)
    h = a.shape[0]
    app = a[int(0.06 * h): int(0.92 * h)]
    return app.size > 0 and float((app < 12).mean()) > 0.995 and float(app.std()) < 3.0


def wait_for_drawn_frame(config: RunnerConfig, screenshot_path: Path, attempts: int = 8,
                         interval: float = 1.5) -> dict[str, Any]:
    captures = 0
    blank = non_empty_png(screenshot_path) and frame_is_blank(screenshot_path)
    started = time.monotonic()
    while blank and captures < attempts:
        time.sleep(interval)
        candidate = screenshot_path.with_name(f"{screenshot_path.stem}_blank_tmp.png")
        capture = screencap_to_file(config, candidate)
        captures += 1
        if capture["returncode"] != 0 or not non_empty_png(candidate):
            candidate.unlink(missing_ok=True)
            continue
        candidate.replace(screenshot_path)
        blank = frame_is_blank(screenshot_path)
    return {"recaptures": captures, "still_blank": blank, "waited_ms": int((time.monotonic() - started) * 1000)}


def settle_screenshot(
    config: RunnerConfig,
    run_dir: Path,
    run_id: str,
    screen_id: str,
    attempt_suffix: str,
    screenshot_path: Path,
) -> dict[str, Any]:
    """Recapture until two consecutive frames match, and leave the settled frame in place.

    Returns what happened rather than only the image, because a screen that never settles is a
    finding about that screen and has to be visible in the record instead of silently becoming
    whatever the last capture was. ``settled`` is false in that case and the final frame is
    still written, so the run continues and the reader can see which screens were measured
    without the guarantee.
    """
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    deadline = started + config.screenshot_settle_seconds
    previous = hashlib.sha256(screenshot_path.read_bytes()).hexdigest() if screenshot_path.is_file() else ""
    captures, settled = 1, False

    while time.monotonic() < deadline:
        time.sleep(config.screenshot_settle_interval)
        candidate = screenshot_path.with_name(f"{screenshot_path.stem}_settle_tmp.png")
        capture = screencap_to_file(config, candidate)
        captures += 1
        records.append(
            {
                "capture_returncode": capture["returncode"],
                "pull_returncode": capture["returncode"],
                "elapsed_ms": int((time.monotonic() - started) * 1000),
            }
        )
        if capture["returncode"] != 0 or not non_empty_png(candidate):
            candidate.unlink(missing_ok=True)
            continue
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        candidate.replace(screenshot_path)
        if digest == previous:
            settled = True
            break
        previous = digest

    return {
        "commands": records,
        "waited_ms": int((time.monotonic() - started) * 1000),
        "settled": settled,
        "captures": captures,
    }


def splash_screen_covering(dumpsys_output: str, app_package: str) -> bool:
    """True while the system splash window still owns the pixels we would photograph.

    Android 12+ keeps a ``Splash Screen <package>`` window above the activity for a
    while after ``am start``. The activity can already be resumed, so a focus-only
    gate photographs the launcher icon on a white field and records render_success.
    ``EXITING`` means the splash is leaving; treat that as clear enough to capture.
    """
    splash_window = re.compile(
        r"Window\{[^\s{}]+\s+(?:u\d+\s+)?Splash Screen "
        + re.escape(app_package) + r"(?=[\s}])[^{}\n]*\}"
    )
    for line in dumpsys_output.splitlines():
        stripped = line.strip()
        windows = splash_window.findall(stripped)
        if not any("EXITING" not in window for window in windows):
            continue
        if (
            stripped.startswith("Window{")
            or "mTopFullscreenOpaqueWindowState=" in stripped
            or "mCurrentFocus=" in stripped
            or "mFocusedWindow=" in stripped
            or "startingWindow=" in stripped
            or "mNavBarColorWindowCandidate=" in stripped
            or "mNavBarBackgroundWindowCandidate=" in stripped
        ):
            return True
    return False


def foreground_activity_ok(dumpsys_output: str, app_package: str) -> bool:
    if splash_screen_covering(dumpsys_output, app_package):
        return False
    # A resumed ActivityRecord can coexist with a launcher window while an app
    # starts. A concrete focused-window report (including null) takes priority;
    # an activity elsewhere in the dump must never override it.
    package = re.compile(r"(?<![\w.])" + re.escape(app_package) + r"(?=[/\s}])")
    for markers in (("mCurrentFocus=", "mFocusedWindow="), ("mFocusedApp=",)):
        entries = [line.strip() for line in dumpsys_output.splitlines()
                   if any(marker in line for marker in markers)]
        if entries:
            return all(package.search(line) is not None for line in entries)
    focus_markers = (
        "topResumedActivity=",
        "mResumedActivity=",
        "mFocusedActivity=",
        "ResumedActivity:",
        "topDisplayFocusedRootTask=",
    )
    for line in dumpsys_output.splitlines():
        stripped = line.strip()
        if any(marker in stripped for marker in focus_markers) and package.search(stripped):
            return True
    return False


# A system ANR dialog sits on top of the activity under test. dumpsys still
# reports that activity as focused, so the harness would otherwise photograph
# "Pixel Launcher isn't responding" and record render_success.
ANR_RESOURCE_IDS = (
    "android:id/aerr_close",
    "android:id/aerr_wait",
    "android:id/aerr_report",
    "android:id/aerr_restart",
)
_AERR_WAIT_BOUNDS = re.compile(
    r'resource-id="android:id/aerr_wait"[^>]*bounds="\[(\d+),(\d+)\]\[(\d+),(\d+)\]"'
)


def hierarchy_has_system_anr(xml_text: str) -> bool:
    """Require system-owned ANR evidence, not app content quoting an error.

    Reference screens can legitimately contain "isn't responding" in a native
    TextView. BACK must not close those successfully generated app screens.
    """
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return False
    for node in root.iter():
        if node.get("resource-id") in ANR_RESOURCE_IDS:
            return True
        if node.get("package") in {"android", "com.android.systemui"}:
            message = node.get("text", "").replace("’", "'").lower()
            if "isn't responding" in message:
                return True
    return False


def hierarchy_foreground_evidence(xml_text: str, app_package: str) -> dict[str, Any]:
    """A valid dump of exclusively other packages contradicts a focus success.

    Empty, malformed, or package-free dumps remain unavailable evidence, not
    layout failures: uiautomator collection is still best effort.
    """
    try:
        root = ET.fromstring(xml_text or "")
    except ET.ParseError:
        return {"status": "unavailable", "packages": []}
    if root.tag != "hierarchy":
        return {"status": "unavailable", "packages": []}
    packages = sorted({node.get("package", "").strip() for node in root.iter()
                       if node.get("package", "").strip()})
    if not packages:
        return {"status": "unavailable", "packages": []}
    return {"status": "pass" if app_package in packages else "fail", "packages": packages}


def anr_wait_tap_point(xml_text: str) -> tuple[int, int] | None:
    match = _AERR_WAIT_BOUNDS.search(xml_text or "")
    if not match:
        return None
    left, top, right, bottom = (int(v) for v in match.groups())
    return (left + right) // 2, (top + bottom) // 2


def dismiss_system_anr(config: RunnerConfig, dump_xml: str = "") -> dict[str, Any]:
    """Act only on proven system ANR: tap Wait, or BACK if Wait is unavailable."""
    if not hierarchy_has_system_anr(dump_xml):
        return {"detected": False, "tapped_wait": False, "commands": []}
    records: list[dict[str, Any]] = []
    point = anr_wait_tap_point(dump_xml)
    if point is not None:
        records.append(
            compact_command_result(
                command_record(
                    adb_cmd(config, "shell", "input", "tap", str(point[0]), str(point[1])),
                    timeout=8,
                    env=runner_env(config),
                )
            )
        )
    else:
        records.append(
            compact_command_result(
                command_record(
                    adb_cmd(config, "shell", "input", "keyevent", "4"),
                    timeout=8,
                    env=runner_env(config),
                )
            )
        )
    return {"detected": True, "tapped_wait": point is not None, "commands": records}


# Install failures whose text names the guest rather than the package under test.
# Each of these fails identically for any APK, so attributing it to the layout
# being validated would record an infrastructure event as a screen outcome.
GUEST_INSTALL_ENVIRONMENT_MARKERS = (
    "INSTALL_FAILED_INSUFFICIENT_STORAGE",
    "could not be assigned a valid UID",
    "INSTALL_FAILED_MEDIA_UNAVAILABLE",
    "Can't find service: package",
    "Cannot access system provider",
    "device offline",
    "device not found",
    "closed",
)

GUEST_REBOOT_RETRY_MARKERS = (
    "INSTALL_FAILED_INSUFFICIENT_STORAGE",
    "could not be assigned a valid UID",
    "INSTALL_FAILED_MEDIA_UNAVAILABLE",
    "Can't find service: package",
    "Cannot access system provider",
    "device offline",
    "device not found",
)


def package_service_ready(config: RunnerConfig) -> dict[str, Any]:
    """Check that the guest's package manager is serving, not merely booted.

    ``sys.boot_completed`` is set before the package manager accepts requests,
    and a guest whose system server has been killed still reports it, so every
    installation would fail for an infrastructure reason that must not be
    recorded as a screen outcome.
    """
    probe = command_record(
        adb_cmd(config, "shell", "pm", "list", "packages"),
        timeout=15,
        env=runner_env(config),
    )
    serving = probe["returncode"] == 0 and "package:" in probe["stdout"]
    return {"probe": compact_command_result(probe), "ok": serving}


def guest_package_boot_ready(config: RunnerConfig, timeout_s: int = 180) -> dict[str, Any]:
    """Wait until the guest package manager is serving after a reboot or snapshot load."""
    wait = command_record(
        adb_cmd(config, "wait-for-device"),
        timeout=120,
        env=runner_env(config),
    )
    deadline = time.monotonic() + timeout_s
    last_boot: dict[str, Any] | None = None
    last_ready: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last_boot = command_record(
            adb_cmd(config, "shell", "getprop", "sys.boot_completed"),
            timeout=30,
            env=runner_env(config),
        )
        last_ready = package_service_ready(config)
        boot_done = "1" in (last_boot.get("stdout") or "")
        if wait["returncode"] == 0 and boot_done and last_ready["ok"]:
            return {
                "ok": True,
                "wait_for_device": compact_command_result(wait),
                "boot_completed": compact_command_result(last_boot),
                "package_service": last_ready["probe"],
            }
        time.sleep(2)
    return {
        "ok": False,
        "wait_for_device": compact_command_result(wait),
        "boot_completed": compact_command_result(last_boot or {}),
        "package_service": (last_ready or {}).get("probe") or {},
    }


def reboot_guest(config: RunnerConfig) -> dict[str, Any]:
    """Reboot the resident guest to recover package-manager UID exhaustion.

    Repeated install/uninstall on a no-snapshot AVD eventually fails with
    INSTALL_FAILED_INSUFFICIENT_STORAGE / 'could not be assigned a valid UID'
    even when /data still has gigabytes free. An in-process adb reboot keeps
    the host qemu pid and restores a serving package manager.
    """
    reboot = command_record(
        adb_cmd(config, "reboot"),
        timeout=60,
        env=runner_env(config),
    )
    ready = guest_package_boot_ready(config)
    return {
        "mode": "guest_reboot_uid_recovery",
        "reboot": compact_command_result(reboot),
        "ok": ready["ok"],
        "wait_for_device": ready.get("wait_for_device"),
        "boot_completed": ready.get("boot_completed"),
        "package_service": ready.get("package_service"),
    }


def environment_failure_needs_guest_reboot(status: dict[str, Any]) -> bool:
    """Reboot when the guest, not the layout, stopped accepting packages."""
    detail = str(status.get("failure_detail") or "")
    code = str(status.get("failure_code") or "")
    if code == "blank_or_wrong_activity" and (
        "system ANR dialog" in detail or "Timeout" in detail or "timed out" in detail.lower()
    ):
        return True
    if code == "activity_launch_failed" and (
        "does not exist" in detail or "Error type 3" in detail or "Timeout" in detail
    ):
        return True
    if code == "environment_missing":
        return True
    if code == "apk_install_failed" and "Timeout" in detail:
        return True
    return any(marker in detail for marker in GUEST_REBOOT_RETRY_MARKERS)


def reset_emulator(config: RunnerConfig) -> dict[str, Any]:
    """Return the guest to the frozen device state before validating a screen.

    With ``avd_snapshot`` set, the frozen snapshot is restored, which is the
    protocol used by every block validated before 2026-07-29. When snapshot
    restore is unavailable on the host, the caller may disable it explicitly;
    the guest is then left resident across screens and only the readiness of the
    package manager is asserted. The mode is recorded per screen so that no
    reader has to infer which device protocol produced a status row.
    """
    if not config.avd_snapshot:
        ready = package_service_ready(config)
        return {
            "mode": "resident_guest_no_snapshot_restore",
            "package_service": ready["probe"],
            "ok": ready["ok"],
        }
    load = command_record(
        adb_cmd(config, "emu", "avd", "snapshot", "load", config.avd_snapshot),
        timeout=120,
        env=runner_env(config),
    )
    wait = command_record(
        adb_cmd(config, "wait-for-device"),
        timeout=120,
        env=runner_env(config),
    )
    ready = package_service_ready(config)
    return {
        "mode": "snapshot_restore",
        "load": compact_command_result(load),
        "wait_for_device": compact_command_result(wait),
        "package_service": ready["probe"],
        "ok": load["returncode"] == 0 and wait["returncode"] == 0 and ready["ok"],
    }


def acquire_device_lock(config: RunnerConfig, status: dict[str, Any]):
    """Blocking exclusive lock on the device; records how long the item waited."""
    handle = config.device_lock_path.open("w", encoding="utf-8")
    started = time.monotonic()
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    status["device_lock_wait_ms"] = int((time.monotonic() - started) * 1000)
    return handle


def release_device_lock(handle) -> None:
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def assemble_apk(item: dict[str, Any], config: RunnerConfig, status: dict[str, Any]) -> Path | None:
    """Host-only: rewrite the test project and assembleDebug. Does not touch the emulator.

    The APK is copied out of ``outputs/apk`` so a pipelined next-screen assemble
    can overwrite that directory while this screen is still installing.
    """
    run_dir = config.run_dir
    artifacts = status["artifacts"]
    stages = status["stages"]
    commands = status["commands"]
    screen_id = status["screen_id"]
    attempt_suffix = "_retry1" if status.get("validation_attempt") == 2 else ""
    status["fast_candidate_render"] = bool(config.fast_candidate_render)

    prepare_project(item, config)
    build = command_record(
        [str(config.gradlew), *gradle_assemble_args(config)],
        cwd=config.project,
        timeout=900,
        env=runner_env(config),
    )
    commands["gradle_build"] = compact_command_result(build)
    gradle_log = run_dir / "raw" / "gradle_logs" / f"{screen_id}{attempt_suffix}.log"
    write_text(gradle_log, build["stdout"] + "\n" + build["stderr"])
    artifacts["gradle_log"] = str(gradle_log.relative_to(run_dir))
    stages["gradle_build"] = "pass" if build["returncode"] == 0 else "fail"
    if build["returncode"] != 0:
        status["status"] = "loading_failure"
        status["failure_code"] = "gradle_compile_error"
        status["failure_detail"] = build["stderr"][-2000:] or build["stdout"][-2000:]
        return None

    apk = newest_apk(config)
    if apk is None:
        stages["apk_found"] = False
        status["status"] = "loading_failure"
        status["failure_code"] = "apk_missing"
        status["failure_detail"] = f"No APK found under {config.apk_dir_path}"
        return None
    stages["apk_found"] = True
    artifacts["apk_path"] = str(apk)
    staging = run_dir / "raw" / "apk_staging" / f"{screen_id}{attempt_suffix}.apk"
    staging.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(apk, staging)
    artifacts["apk_staging"] = str(staging.relative_to(run_dir))
    return staging


def run_full_loading(item: dict[str, Any], config: RunnerConfig, status: dict[str, Any]) -> dict[str, Any]:
    apk = assemble_apk(item, config, status)
    if apk is None:
        return status
    try:
        return run_device_loading(item, config, status, apk)
    finally:
        apk.unlink(missing_ok=True)


def run_device_loading(
    item: dict[str, Any],
    config: RunnerConfig,
    status: dict[str, Any],
    apk: Path,
) -> dict[str, Any]:
    run_id = status["run_id"]
    screen_id = status["screen_id"]
    run_dir = config.run_dir
    artifacts = status["artifacts"]
    stages = status["stages"]
    commands = status["commands"]
    attempt_suffix = "_retry1" if status.get("validation_attempt") == 2 else ""
    status["fast_candidate_render"] = bool(config.fast_candidate_render)
    device_lock = None

    try:
        device_lock = acquire_device_lock(config, status)
        commands["anr_pre_install"] = dismiss_system_anr(config)

        # See docs/RUNTIME.md for the shared compatibility and capture contracts.
        if config.match_reference_canvas:
            match_canvas_to_reference(config, item, status)

        # install -r replaces the previous package; uninstall is redundant on recapture.
        if not config.fast_candidate_render and not config.no_gradle_clean:
            uninstall = command_record(
                adb_cmd(config, "uninstall", config.app_package),
                timeout=60,
                env=runner_env(config),
            )
            commands["apk_uninstall_before_install"] = compact_command_result(uninstall)

        install = command_record(adb_cmd(config, "install", "-r", str(apk)), timeout=45, env=runner_env(config))
        commands["apk_install"] = compact_command_result(install)
        adb_install_log = run_dir / "raw" / "adb_logs" / f"{screen_id}{attempt_suffix}_install.log"
        write_text(adb_install_log, install["stdout"] + "\n" + install["stderr"])
        artifacts["adb_install_log"] = str(adb_install_log.relative_to(run_dir))
        stages["apk_install"] = "pass" if install["returncode"] == 0 else "fail"
        if install["returncode"] != 0:
            # The guest is probed at reset, but it can degrade later, during this
            # screen: host memory pressure kills its system server, or its data
            # partition stops accepting packages. Every install then fails for an
            # infrastructure reason, and an install failure is otherwise a
            # legitimate screen outcome, so both conditions are checked before the
            # failure is attributed to the layout under test.
            install_output = install["stderr"] + install["stdout"]
            environment_cause = next(
                (marker for marker in GUEST_INSTALL_ENVIRONMENT_MARKERS if marker in install_output),
                None,
            )
            still_serving = package_service_ready(config)
            commands["package_service_after_install_failure"] = still_serving["probe"]
            if environment_cause or not still_serving["ok"] or install.get("timed_out"):
                status["status"] = "environment_failure"
                status["failure_code"] = "environment_missing"
                status["failure_detail"] = (
                    f"the guest could not accept any package during this screen ({environment_cause or 'install timeout'}); "
                    "the installation failure is an environment failure and not a screen outcome"
                    if environment_cause or install.get("timed_out")
                    else "the guest package manager stopped serving during this screen; "
                    "the installation failure is an environment failure and not a screen outcome"
                )
                return status
            status["status"] = "loading_failure"
            status["failure_code"] = "apk_install_failed"
            status["failure_detail"] = install["stderr"][-2000:] or install["stdout"][-2000:]
            return status

        launch = command_record(
            adb_cmd(config, "shell", "am", "start", "-n", f"{config.app_package}/{config.app_activity}"),
            timeout=15,
            env=runner_env(config),
        )
        launch_text = (launch.get("stderr") or "") + (launch.get("stdout") or "")
        guest_not_ready = launch["returncode"] != 0 and (
            "does not exist" in launch_text
            or "Error type 3" in launch_text
            or bool(launch.get("timed_out"))
        )
        if guest_not_ready:
            for _ in range(3):
                time.sleep(1.0)
                launch = command_record(
                    adb_cmd(config, "shell", "am", "start", "-n", f"{config.app_package}/{config.app_activity}"),
                    timeout=15,
                    env=runner_env(config),
                )
                launch_text = (launch.get("stderr") or "") + (launch.get("stdout") or "")
                guest_not_ready = launch["returncode"] != 0 and (
                    "does not exist" in launch_text
                    or "Error type 3" in launch_text
                    or bool(launch.get("timed_out"))
                )
                if launch["returncode"] == 0 or not guest_not_ready:
                    break
        commands["activity_launch"] = compact_command_result(launch)
        adb_launch_log = run_dir / "raw" / "adb_logs" / f"{screen_id}{attempt_suffix}_launch.log"
        write_text(adb_launch_log, launch["stdout"] + "\n" + launch["stderr"])
        artifacts["adb_launch_log"] = str(adb_launch_log.relative_to(run_dir))
        stages["activity_launch"] = "pass" if launch["returncode"] == 0 else "fail"
        if launch["returncode"] != 0:
            status["failure_detail"] = launch["stderr"][-2000:] or launch["stdout"][-2000:]
            if guest_not_ready:
                status["status"] = "environment_failure"
                status["failure_code"] = "environment_missing"
            else:
                status["status"] = "loading_failure"
                status["failure_code"] = "activity_launch_failed"
            return status

        time.sleep(config.screenshot_delay)
        skip_easy_dump = is_pix2code_easy_id(screen_id)
        if not config.fast_candidate_render and not skip_easy_dump and not config.no_gradle_clean:
            logcat = command_record(adb_cmd(config, "logcat", "-d", "-t", "300"), timeout=30, env=runner_env(config))
            logcat_path = run_dir / "raw" / "logcat" / f"{screen_id}{attempt_suffix}.log"
            write_text(logcat_path, logcat["stdout"] + "\n" + logcat["stderr"])
            artifacts["logcat"] = str(logcat_path.relative_to(run_dir))

        focus = dumpsys_focus(config)
        commands["foreground_activity"] = compact_command_result(focus)
        focus_log = run_dir / "raw" / "adb_logs" / f"{screen_id}{attempt_suffix}_foreground.log"
        write_text(focus_log, focus["stdout"] + "\n" + focus["stderr"])
        artifacts["adb_foreground_log"] = str(focus_log.relative_to(run_dir))
        foreground_detail = focus["stdout"][-2000:] or focus["stderr"][-2000:]
        foreground_ok = focus["returncode"] == 0 and foreground_activity_ok(focus["stdout"], config.app_package)
        stages["foreground_activity"] = "pass" if foreground_ok else "fail"

        # First dumpsys after the fixed delay can still see the splash or a null
        # focus. Poll until the activity holds the foreground; that is the gate.
        if not foreground_ok:
            deadline = time.monotonic() + config.foreground_poll_seconds
            polled_ok, waited_ms = False, 0
            while time.monotonic() < deadline:
                time.sleep(0.25)
                repeat = dumpsys_focus(config)
                waited_ms = int((time.monotonic() - (deadline - config.foreground_poll_seconds)) * 1000)
                if repeat["returncode"] == 0 and foreground_activity_ok(repeat["stdout"], config.app_package):
                    polled_ok = True
                    break
            stages["foreground_activity_polled"] = "pass" if polled_ok else "fail"
            status["foreground_poll_ms"] = waited_ms
            if polled_ok:
                # The first dumpsys is often too early (splash / null focus). Poll already
                # proved the activity holds the foreground; that is the deployability gate.
                foreground_ok = True
                stages["foreground_activity"] = "pass"
                stages["foreground_recovered"] = "poll"
                commands["foreground_activity_verified"] = compact_command_result(repeat)
                write_text(focus_log, repeat["stdout"] + "\n" + repeat["stderr"])

        screenshot_path = run_dir / "raw" / "screenshots" / f"{screen_id}{attempt_suffix}.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)
        screencap = screencap_to_file(config, screenshot_path)
        commands["screencap"] = compact_command_result(screencap)
        pull = screencap
        commands["screenshot_pull"] = compact_command_result(screencap)

        # A fixed delay decides when to look, not whether there is anything to see, so on a
        # slow screen it photographs the platform's startup splash: a white field with the
        # app icon centred on it. Two screens recorded that way scored 0.03 and 0.01 global
        # structural similarity against references they otherwise match at 0.88 and 0.84, which
        # is a measurement destroyed rather than a layout at fault. Raising the constant would
        # only move the boundary, and choosing the constant by looking at which screens improve
        # would let the outcome pick the protocol.
        #
        # Waiting for the frame to stop changing removes the constant from the decision. The
        # capture is repeated until two consecutive frames are byte-identical, which is what
        # "the activity has finished drawing" means operationally, and the deadline exists only
        # so an animation that never settles cannot hang the run. The rule refers to nothing
        # about the screen's content or which arm produced it.
        if config.screenshot_settle_seconds > 0:
            settle = settle_screenshot(config, run_dir, run_id, screen_id, attempt_suffix, screenshot_path)
            commands["screenshot_settle"] = settle["commands"]
            status["screenshot_settle_ms"] = settle["waited_ms"]
            status["screenshot_settled"] = settle["settled"]
            status["screenshot_settle_captures"] = settle["captures"]
        if non_empty_png(screenshot_path):
            drawn = wait_for_drawn_frame(config, screenshot_path)
            if drawn["recaptures"]:
                status["blank_frame_recaptures"] = drawn["recaptures"]
                status["blank_frame_wait_ms"] = drawn["waited_ms"]
            if drawn["still_blank"]:
                status["blank_frame"] = True
        # Bracket the captured frame with foreground checks. The app can still
        # be starting (or leave the foreground) between the first check and adb
        # screencap; a later hierarchy dump alone does not validate that image.
        capture_focus = dumpsys_focus(config)
        commands["foreground_after_capture"] = compact_command_result(capture_focus)
        capture_focus_log = run_dir / "raw" / "adb_logs" / f"{screen_id}{attempt_suffix}_capture_foreground.log"
        write_text(capture_focus_log, capture_focus["stdout"] + "\n" + capture_focus["stderr"])
        artifacts["adb_capture_foreground_log"] = str(capture_focus_log.relative_to(run_dir))
        capture_ok = capture_focus["returncode"] == 0 and foreground_activity_ok(
            capture_focus["stdout"], config.app_package
        )
        if capture_ok and not foreground_ok:
            # A late launch can reach the target only after the first screenshot.
            # That later focus cannot certify the old pixels. Take a fresh frame
            # between two target-focus checks instead of either accepting the
            # unchecked frame or retaining an already-obsolete launch failure.
            late_focus_log = run_dir / "raw" / "adb_logs" / f"{screen_id}{attempt_suffix}_late_foreground.log"
            write_text(late_focus_log, capture_focus["stdout"] + "\n" + capture_focus["stderr"])
            artifacts["adb_initial_foreground_log"] = artifacts["adb_foreground_log"]
            artifacts["adb_foreground_log"] = str(late_focus_log.relative_to(run_dir))
            commands["foreground_before_late_capture"] = compact_command_result(capture_focus)
            if screenshot_path.is_file():
                previous_capture = screenshot_path.with_name(screenshot_path.stem + "_before_late.png")
                shutil.copy2(screenshot_path, previous_capture)
                artifacts["screenshot_before_late_capture"] = str(previous_capture.relative_to(run_dir))
            pull = screencap_to_file(config, screenshot_path)
            commands["late_screencap"] = compact_command_result(pull)
            if pull["returncode"] == 0 and config.screenshot_settle_seconds > 0:
                settle = settle_screenshot(config, run_dir, run_id, screen_id, attempt_suffix, screenshot_path)
                commands["late_screenshot_settle"] = settle["commands"]
                status["screenshot_settle_ms"] = settle["waited_ms"]
                status["screenshot_settled"] = settle["settled"]
                status["screenshot_settle_captures"] = settle["captures"]
            capture_focus = dumpsys_focus(config)
            commands["foreground_after_capture"] = compact_command_result(capture_focus)
            write_text(capture_focus_log, capture_focus["stdout"] + "\n" + capture_focus["stderr"])
            capture_ok = capture_focus["returncode"] == 0 and foreground_activity_ok(
                capture_focus["stdout"], config.app_package
            )
            foreground_ok = capture_ok
            if capture_ok:
                stages["foreground_activity"] = "pass"
                stages["foreground_recovered"] = "late_recapture"
        capture_splash_covering = capture_focus["returncode"] == 0 and splash_screen_covering(
            capture_focus["stdout"], config.app_package
        )
        stages["foreground_after_capture"] = "pass" if capture_ok else "fail"
        if not capture_ok:
            foreground_ok = False
            stages["foreground_activity"] = "fail"
            foreground_detail = capture_focus["stdout"][-2000:] or capture_focus["stderr"][-2000:]
        # Rendered view hierarchy of the same frame, for the structure metric. Best effort:
        # a dump failure never changes the deployability verdict. 12s cap; a hung dump
        # used to sit on the 45s timeout and stall the whole recapture.
        hierarchy_remote = f"/sdcard/gui_codegen_{run_id}_{screen_id}{attempt_suffix}_hierarchy.xml"
        hierarchy_path = run_dir / "raw" / "hierarchy" / f"{screen_id}{attempt_suffix}.xml"
        if not config.fast_candidate_render and not config.skip_hierarchy_dump and \
                (not skip_easy_dump or config.force_hierarchy_dump):
            hierarchy_path.parent.mkdir(parents=True, exist_ok=True)
            dump = command_record(
                adb_cmd(config, "shell", "uiautomator", "dump", hierarchy_remote),
                timeout=HIERARCHY_DUMP_TIMEOUT,
                env=runner_env(config),
            )
            commands["hierarchy_dump"] = compact_command_result(dump)
            if dump.get("timed_out"):
                command_record(
                    adb_cmd(config, "shell", "pkill", "uiautomator"),
                    timeout=5,
                    env=runner_env(config),
                )
            if dump["returncode"] == 0:
                pull_hierarchy = command_record(
                    adb_cmd(config, "pull", hierarchy_remote, str(hierarchy_path)),
                    timeout=15,
                    env=runner_env(config),
                )
                commands["hierarchy_pull"] = compact_command_result(pull_hierarchy)
                command_record(
                    adb_cmd(config, "shell", "rm", "-f", hierarchy_remote), timeout=8, env=runner_env(config)
                )
                if pull_hierarchy["returncode"] == 0 and hierarchy_path.is_file() and hierarchy_path.stat().st_size > 0:
                    artifacts["hierarchy"] = str(hierarchy_path.relative_to(run_dir))
        dumped_xml = ""
        if artifacts.get("hierarchy"):
            dumped_xml = (run_dir / artifacts["hierarchy"]).read_text(encoding="utf-8", errors="replace")
        if hierarchy_has_system_anr(dumped_xml):
            commands["anr_dismiss"] = dismiss_system_anr(config, dumped_xml)
            time.sleep(0.4)
            recap = screencap_to_file(config, screenshot_path)
            commands["anr_screencap"] = compact_command_result(recap)
            pull = recap
            # ANR recovery replaced the image checked above. Validate this final
            # capture too; the earlier foreground verdict cannot certify it.
            recap_focus = dumpsys_focus(config)
            commands["foreground_after_anr_capture"] = compact_command_result(recap_focus)
            recap_log = run_dir / "raw" / "adb_logs" / f"{screen_id}{attempt_suffix}_anr_capture_foreground.log"
            write_text(recap_log, recap_focus["stdout"] + "\n" + recap_focus["stderr"])
            artifacts["adb_anr_capture_foreground_log"] = str(recap_log.relative_to(run_dir))
            recap_ok = recap_focus["returncode"] == 0 and foreground_activity_ok(
                recap_focus["stdout"], config.app_package
            )
            capture_splash_covering = recap_focus["returncode"] == 0 and splash_screen_covering(
                recap_focus["stdout"], config.app_package
            )
            stages["foreground_after_anr_capture"] = "pass" if recap_ok else "fail"
            if not recap_ok:
                foreground_ok = False
                stages["foreground_activity"] = "fail"
                foreground_detail = recap_focus["stdout"][-2000:] or recap_focus["stderr"][-2000:]
            if artifacts.get("hierarchy"):
                dump = command_record(
                    adb_cmd(config, "shell", "uiautomator", "dump", hierarchy_remote),
                    timeout=HIERARCHY_DUMP_TIMEOUT,
                    env=runner_env(config),
                )
                commands["anr_hierarchy_dump"] = compact_command_result(dump)
                if dump.get("timed_out"):
                    command_record(
                        adb_cmd(config, "shell", "pkill", "uiautomator"),
                        timeout=5,
                        env=runner_env(config),
                    )
                if dump["returncode"] == 0:
                    pull_hierarchy = command_record(
                        adb_cmd(config, "pull", hierarchy_remote, str(hierarchy_path)),
                        timeout=15,
                        env=runner_env(config),
                    )
                    commands["anr_hierarchy_pull"] = compact_command_result(pull_hierarchy)
                    command_record(
                        adb_cmd(config, "shell", "rm", "-f", hierarchy_remote),
                        timeout=8,
                        env=runner_env(config),
                    )
                    if pull_hierarchy["returncode"] == 0 and hierarchy_path.is_file():
                        artifacts["hierarchy"] = str(hierarchy_path.relative_to(run_dir))
                        dumped_xml = hierarchy_path.read_text(encoding="utf-8", errors="replace")
            if hierarchy_has_system_anr(dumped_xml):
                command_record(
                    adb_cmd(config, "shell", "am", "force-stop", config.app_package),
                    timeout=8,
                    env=runner_env(config),
                )
                artifacts["screenshot"] = str(screenshot_path.relative_to(run_dir))
                status["status"] = "loading_failure"
                status["failure_code"] = "blank_or_wrong_activity"
                status["failure_detail"] = "system ANR dialog still on screen after dismiss"
                return status
        hierarchy_evidence = hierarchy_foreground_evidence(dumped_xml, config.app_package)
        stages["hierarchy_foreground"] = hierarchy_evidence["status"]
        status["hierarchy_packages"] = hierarchy_evidence["packages"]
        if hierarchy_evidence["status"] == "fail":
            foreground_ok = False
            stages["foreground_activity"] = "fail"
            foreground_detail = (
                "final hierarchy belongs exclusively to non-target packages: "
                + ", ".join(hierarchy_evidence["packages"])
            )
        command_record(
            adb_cmd(config, "shell", "am", "force-stop", config.app_package),
            timeout=8,
            env=runner_env(config),
        )

        artifacts["screenshot"] = str(screenshot_path.relative_to(run_dir))
        stages["screenshot_capture"] = "pass" if pull["returncode"] == 0 and non_empty_png(screenshot_path) else "fail"
        if stages["screenshot_capture"] != "pass":
            status["status"] = "loading_failure"
            status["failure_code"] = "screenshot_failed"
            status["failure_detail"] = pull["stderr"][-2000:] or pull["stdout"][-2000:]
            return status

        if not foreground_ok:
            focus_timed_out = bool((commands.get("foreground_activity") or {}).get("timed_out")) or bool(
                (commands.get("foreground_after_capture") or {}).get("timed_out")
            ) or bool(
                (commands.get("foreground_after_anr_capture") or {}).get("timed_out")
            ) or (
                "Timeout" in (foreground_detail or "")
            )
            status["failure_detail"] = foreground_detail
            # A successful dumpsys command can still prove the platform splash
            # has not released the captured pixels. This is an exhausted capture
            # wait, not a layout outcome. Use the final capture's full window
            # evidence, never a word in app text or the truncated detail string.
            status["splash_screen_covering"] = capture_splash_covering
            if capture_splash_covering:
                status["failure_detail"] = "active system splash still covers the target at final capture; " + foreground_detail
            if focus_timed_out or capture_splash_covering:
                status["status"] = "environment_failure"
                status["failure_code"] = "environment_missing"
            else:
                status["status"] = "loading_failure"
                status["failure_code"] = "blank_or_wrong_activity"
            return status

        if attempt_suffix:
            artifacts["retry_screenshot"] = artifacts["screenshot"]
            canonical_screenshot = run_dir / "raw" / "screenshots" / f"{screen_id}.png"
            shutil.copy2(screenshot_path, canonical_screenshot)
            artifacts["screenshot"] = str(canonical_screenshot.relative_to(run_dir))
            if artifacts.get("hierarchy"):
                hierarchy_path = run_dir / artifacts["hierarchy"]
                if hierarchy_path.is_file():
                    canonical_hierarchy = run_dir / "raw" / "hierarchy" / f"{screen_id}.xml"
                    shutil.copy2(hierarchy_path, canonical_hierarchy)
                    artifacts["hierarchy"] = str(canonical_hierarchy.relative_to(run_dir))
        status["status"] = "render_success"
        return status
    except Exception as error:
        status["status"] = "loading_failure"
        status["failure_code"] = "unexpected_exception"
        status["failure_detail"] = f"{error}\n{traceback.format_exc()}"
        return status
    finally:
        release_device_lock(device_lock)


def base_status(item: dict[str, Any], manifest: dict[str, Any],
                config: RunnerConfig) -> dict[str, Any]:
    return {
        "run_id": item["run_id"],
        "screen_id": item["screen_id"],
        "dataset": manifest.get("dataset", ""),
        "method": manifest.get("method", ""),
        "model": manifest.get("model", ""),
        "timestamp": utc_now(),
        "xml_path": item.get("xml_path", ""),
        "xml_sha256": item.get("xml_sha256", ""),
        "resources_sha256": item.get("resources_sha256", ""),
        # See docs/RUNTIME.md for the shared compatibility and capture contracts.
        "canvas": canvas_key(item, config.match_reference_canvas),
        "input_png": item.get("input_png", ""),
        "input_png_sha256": item.get("input_png_sha256", ""),
        "status": "not_run",
        "failure_code": "",
        "failure_detail": "",
        "stages": {},
        "commands": {},
        "artifacts": {},
    }


def finish_built_item(
    item: dict[str, Any],
    manifest: dict[str, Any],
    config: RunnerConfig,
    status: dict[str, Any],
    apk: Path,
    static_stages: dict[str, Any],
) -> dict[str, Any]:
    """Install/capture a already-built APK; infrastructure retry reuses that APK."""
    result = run_device_loading(item, config, status, apk)
    needs_snapshot_retry = result.get("failure_code") in {"screenshot_failed", "unexpected_exception"}
    needs_reboot_retry = environment_failure_needs_guest_reboot(result)
    if not needs_snapshot_retry and not needs_reboot_retry:
        return result

    retry_reset = reboot_guest(config) if needs_reboot_retry else reset_emulator(config)
    retry_status = base_status(item, manifest, config)
    retry_status["stages"].update(static_stages)
    for key in ("gradle_build", "apk_found"):
        if key in (result.get("stages") or {}):
            retry_status["stages"][key] = result["stages"][key]
    retry_status["commands"]["gradle_build"] = (status.get("commands") or {}).get("gradle_build") or (
        (result.get("commands") or {}).get("gradle_build")
    )
    retry_status["commands"]["emulator_reset"] = retry_reset
    retry_status["artifacts"]["apk_path"] = (status.get("artifacts") or {}).get("apk_path") or str(apk)
    retry_status["infrastructure_retry"] = {
        "performed": True,
        "first_failure_code": result.get("failure_code"),
        "first_attempt": result,
        "guest_reboot": bool(needs_reboot_retry),
        "reused_apk": True,
    }
    retry_status["validation_attempt"] = 2
    if not retry_reset["ok"]:
        retry_status["status"] = "environment_failure"
        retry_status["failure_code"] = "environment_missing"
        retry_status["failure_detail"] = (
            "Guest reboot failed before the sole UID-exhaustion retry."
            if needs_reboot_retry
            else "Frozen AVD snapshot reset failed before the sole infrastructure retry."
        )
        return retry_status
    return run_device_loading(item, config, retry_status, apk)


def build_item_job(
    item: dict[str, Any],
    manifest: dict[str, Any],
    config: RunnerConfig,
    env_ok: bool,
) -> tuple[dict[str, Any], Path | None]:
    """Static check + assemble. No emulator I/O, so it can overlap the previous capture."""
    status = base_status(item, manifest, config)
    static_stages, static_failure, static_detail = validate_static(item)
    status["stages"].update(static_stages)
    if static_failure:
        status["status"] = "static_failure"
        status["failure_code"] = static_failure
        status["failure_detail"] = static_detail
        return status, None
    status["status"] = "static_pass"
    if config.static_only:
        return status, None
    if not env_ok:
        status["status"] = "environment_failure"
        status["failure_code"] = "environment_missing"
        status["failure_detail"] = "Android project, ADB, Gradle, or connected device is missing."
        return status, None
    try:
        return status, assemble_apk(item, config, status)
    except Exception as error:
        status["status"] = "loading_failure"
        status["failure_code"] = "unexpected_exception"
        status["failure_detail"] = f"{error}\n{traceback.format_exc()}"
        return status, None


def capture_item_job(
    item: dict[str, Any],
    manifest: dict[str, Any],
    config: RunnerConfig,
    status: dict[str, Any],
    apk: Path | None,
) -> dict[str, Any]:
    if apk is None:
        return status
    static_stages = dict(status.get("stages") or {})
    if config.avd_snapshot:
        reset = reset_emulator(config)
        status["commands"]["emulator_reset"] = reset
        if not reset["ok"]:
            status["status"] = "environment_failure"
            status["failure_code"] = "environment_missing"
            status["failure_detail"] = "Frozen AVD snapshot reset failed before validation."
            return status
    else:
        status["commands"]["emulator_reset"] = {
            "mode": "resident_guest_no_snapshot_restore",
            "ok": True,
            "package_service": "deferred_to_install",
        }
    return finish_built_item(item, manifest, config, status, apk, static_stages)


def process_item(item: dict[str, Any], manifest: dict[str, Any], config: RunnerConfig, env_ok: bool) -> dict[str, Any]:
    status = base_status(item, manifest, config)
    static_stages, static_failure, static_detail = validate_static(item)
    status["stages"].update(static_stages)

    if static_failure:
        status["status"] = "static_failure"
        status["failure_code"] = static_failure
        status["failure_detail"] = static_detail
        return status

    status["status"] = "static_pass"
    if config.static_only:
        return status

    if not env_ok:
        status["status"] = "environment_failure"
        status["failure_code"] = "environment_missing"
        status["failure_detail"] = "Android project, ADB, Gradle, or connected device is missing."
        return status

    reset = reset_emulator(config)
    status["commands"]["emulator_reset"] = reset
    if not reset["ok"]:
        status["status"] = "environment_failure"
        status["failure_code"] = "environment_missing"
        status["failure_detail"] = "Frozen AVD snapshot reset failed before validation."
        return status

    result = run_full_loading(item, config, status)
    needs_snapshot_retry = result.get("failure_code") in {"screenshot_failed", "unexpected_exception"}
    needs_reboot_retry = environment_failure_needs_guest_reboot(result)
    if not needs_snapshot_retry and not needs_reboot_retry:
        return result

    retry_reset = reboot_guest(config) if needs_reboot_retry else reset_emulator(config)
    retry_status = base_status(item, manifest, config)
    retry_status["stages"].update(static_stages)
    retry_status["commands"]["emulator_reset"] = retry_reset
    retry_status["infrastructure_retry"] = {
        "performed": True,
        "first_failure_code": result.get("failure_code"),
        "first_attempt": result,
        "guest_reboot": bool(needs_reboot_retry),
    }
    retry_status["validation_attempt"] = 2
    if not retry_reset["ok"]:
        retry_status["status"] = "environment_failure"
        retry_status["failure_code"] = "environment_missing"
        retry_status["failure_detail"] = (
            "Guest reboot failed before the sole UID-exhaustion retry."
            if needs_reboot_retry
            else "Frozen AVD snapshot reset failed before the sole infrastructure retry."
        )
        return retry_status
    return run_full_loading(item, config, retry_status)


def flatten_for_csv(row: dict[str, Any]) -> dict[str, Any]:
    stages = row.get("stages", {})
    return {
        "run_id": row.get("run_id", ""),
        "screen_id": row["screen_id"],
        "dataset": row.get("dataset", ""),
        "method": row.get("method", ""),
        "model": row.get("model", ""),
        "status": row.get("status", ""),
        "failure_code": row.get("failure_code", ""),
        "xml_path": row.get("xml_path", ""),
        "xml_exists": stages.get("xml_exists", ""),
        "xml_like": stages.get("xml_like", ""),
        "xml_well_formed": stages.get("xml_well_formed", ""),
        "android_namespace": stages.get("android_namespace", ""),
        "resources_resolved": stages.get("resources_resolved", ""),
        "gradle_build": stages.get("gradle_build", ""),
        "apk_install": stages.get("apk_install", ""),
        "activity_launch": stages.get("activity_launch", ""),
        "foreground_activity": stages.get("foreground_activity", ""),
        "screenshot_capture": stages.get("screenshot_capture", ""),
    }


def write_csv_status(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "run_id",
        "screen_id",
        "dataset",
        "method",
        "model",
        "status",
        "failure_code",
        "xml_path",
        "xml_exists",
        "xml_like",
        "xml_well_formed",
        "android_namespace",
        "resources_resolved",
        "gradle_build",
        "apk_install",
        "activity_launch",
        "foreground_activity",
        "screenshot_capture",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(flatten_for_csv(row))


def update_env(config: RunnerConfig, manifest: dict[str, Any], env_checks: dict[str, Any]) -> None:
    env_path = config.run_dir / "env.json"
    env = read_json(env_path) if env_path.exists() else {}
    frozen_device: dict[str, Any] = {}
    if (
        not config.static_only
        and env_checks.get("adb_device_connected")
        and not config.fast_candidate_render
    ):
        probes = {
            "api_level": ["shell", "getprop", "ro.build.version.sdk"],
            "system_image": ["shell", "getprop", "ro.build.fingerprint"],
            "locale": ["shell", "getprop", "persist.sys.locale"],
            "screen_size": ["shell", "wm", "size"],
            "density": ["shell", "wm", "density"],
            "font_scale": ["shell", "settings", "get", "system", "font_scale"],
            "window_animation_scale": ["shell", "settings", "get", "global", "window_animation_scale"],
            "transition_animation_scale": ["shell", "settings", "get", "global", "transition_animation_scale"],
            "animator_duration_scale": ["shell", "settings", "get", "global", "animator_duration_scale"],
        }
        for name, command in probes.items():
            result = command_record(adb_cmd(config, *command), timeout=30, env=runner_env(config))
            frozen_device[name] = compact_command_result(result)
        gradle = command_record([str(config.gradlew), "--version"], cwd=config.project, timeout=60, env=runner_env(config))
        frozen_device["gradle_and_jvm"] = compact_command_result(gradle)
    env.update(
        {
            "updated_at": utc_now(),
            "runner": "android_harness/device_session.py",
            "static_only": config.static_only,
            "android_project": str(config.project),
            "android_home": str(config.android_home),
            "adb_path": str(config.adb),
            "app_package": config.app_package,
            "app_activity": config.app_activity,
            "device_serial": config.device_serial,
            "avd_snapshot": config.avd_snapshot,
            "fast_candidate_render": config.fast_candidate_render,
            "frozen_device": frozen_device,
            "env_checks": env_checks,
            "manifest_dataset": manifest.get("dataset", ""),
            "manifest_method": manifest.get("method", ""),
            "manifest_model": manifest.get("model", ""),
        }
    )
    env_path.write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def selected_items(items: list[dict[str, Any]], config: RunnerConfig) -> list[dict[str, Any]]:
    if config.screen_ids is not None:
        items = [item for item in items if item["screen_id"] in config.screen_ids]
    if config.limit is not None:
        items = items[: config.limit]
    return items


def run(config: RunnerConfig) -> list[dict[str, Any]]:
    """Only one writer may inspect, resume, or rewrite a run's status ledger."""
    derived_dir = config.run_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    name = ".static_status.lock" if config.static_only else ".loading_status.lock"
    with (derived_dir / name).open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Run already active: {config.run_dir}; refusing concurrent ledger rewrite")
        return _run_locked(config)


def _run_locked(config: RunnerConfig) -> list[dict[str, Any]]:
    manifest = read_json(config.run_dir / "manifest.json")
    items = selected_items(read_jsonl(config.run_dir / "items.jsonl"), config)
    env_checks, env_ok = environment_status(config)
    update_env(config, manifest, env_checks)

    derived_dir = config.run_dir / "derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    status_path = derived_dir / ("static_status.jsonl" if config.static_only else "loading_status.jsonl")
    csv_path = derived_dir / ("static_status.csv" if config.static_only else "loading_status.csv")

    # Resume: keep recorded outcomes except environment failures and rows whose XML
    # or drawable hashes no longer match items.jsonl, so a derived-arm or asset-stage
    # rematerialize re-captures changed screens instead of scoring stale screenshots.
    previous: list[dict[str, Any]] = []
    if config.resume_status and status_path.exists():
        current_hash = {item["screen_id"]: item.get("xml_sha256") or "" for item in items}
        current_resources = {item["screen_id"]: item.get("resources_sha256") or "" for item in items}
        current_canvas = {item["screen_id"]: canvas_key(item, config.match_reference_canvas)
                          for item in items}
        kept: list[dict[str, Any]] = []
        for row in read_jsonl(status_path):
            if row.get("status") == "environment_failure":
                continue
            sid = str(row.get("screen_id") or "")
            want = current_hash.get(sid, "")
            got = str(row.get("xml_sha256") or "")
            if want and want != got:
                continue
            # See docs/RUNTIME.md for the shared compatibility and capture contracts.
            if "canvas" not in row:
                continue
            want_canvas = current_canvas.get(sid)
            if want_canvas is None or str(row.get("canvas") or "") != want_canvas:
                continue
            want_res = current_resources.get(sid, "")
            got_res = str(row.get("resources_sha256") or "")
            # Missing legacy resource hashes cannot establish screenshot freshness.
            if want_res and want_res != got_res:
                continue
            kept.append(row)
        previous = kept
        done = {row["screen_id"] for row in previous}
        items = [item for item in items if item["screen_id"] not in done]
        write_jsonl(status_path, previous)
    elif status_path.exists() and not config.overwrite:
        raise SystemExit(f"{status_path} already exists; pass --overwrite to replace it")

    lock_handles: list[Any] = []
    statuses: list[dict[str, Any]] = []
    try:
        if not config.static_only and items:
            AVD_LOCK_DIR.mkdir(parents=True, exist_ok=True)
            avd_lock_path = AVD_LOCK_DIR / f"avd_{config.device_serial or 'default'}.lock"
            avd_handle = avd_lock_path.open("w", encoding="utf-8")
            fcntl.flock(avd_handle.fileno(), fcntl.LOCK_EX)
            lock_handles.append(avd_handle)
            if config.project.exists():
                lock_path = config.project / ".ese_validation.lock"
                lock_handle = lock_path.open("w", encoding="utf-8")
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                lock_handles.append(lock_handle)
        if config.static_only or not items:
            for item in items:
                outcome = process_item(item, manifest, config, env_ok)
                statuses.append(outcome)
                if config.resume_status:
                    append_jsonl(status_path, [outcome])
        else:
            # Overlap assemble of screen N+1 with install/capture of screen N.
            # maxsize=2 keeps one extra APK staged without rewriting the project
            # while that APK is still the one being installed.
            jobs: Queue = Queue(maxsize=2)

            def builder() -> None:
                try:
                    for item in items:
                        status, apk = build_item_job(item, manifest, config, env_ok)
                        jobs.put((item, status, apk))
                except Exception as error:
                    jobs.put(
                        (
                            {"screen_id": "_builder"},
                            {
                                "status": "loading_failure",
                                "failure_code": "unexpected_exception",
                                "failure_detail": f"{error}\n{traceback.format_exc()}",
                                "screen_id": "_builder",
                            },
                            None,
                        )
                    )
                finally:
                    jobs.put(None)

            worker = threading.Thread(target=builder, name="apk-builder", daemon=True)
            worker.start()
            while True:
                job = jobs.get()
                if job is None:
                    break
                item, status, apk = job
                try:
                    outcome = capture_item_job(item, manifest, config, status, apk)
                except Exception as error:
                    status["status"] = "loading_failure"
                    status["failure_code"] = "unexpected_exception"
                    status["failure_detail"] = f"{error}\n{traceback.format_exc()}"
                    outcome = status
                finally:
                    if apk is not None:
                        Path(apk).unlink(missing_ok=True)
                statuses.append(outcome)
                if config.resume_status:
                    append_jsonl(status_path, [outcome])
            worker.join(timeout=60)
    finally:
        for handle in reversed(lock_handles):
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()
    if config.resume_status:
        statuses = previous + statuses
    else:
        write_jsonl(status_path, statuses)
    write_csv_status(csv_path, statuses)
    append_jsonl(
        config.run_dir / "events.jsonl",
        [
            {
                "run_id": manifest["run_id"],
                "event": "android_loading_runner_completed",
                "timestamp": utc_now(),
                "status": "static_only" if config.static_only else "full",
                "items_processed": len(statuses),
                "status_file": str(status_path.relative_to(config.run_dir)),
                "csv_file": str(csv_path.relative_to(config.run_dir)),
            }
        ],
    )
    return statuses


def default_adb(android_home: Path) -> Path:
    return android_home / "platform-tools" / "adb"


def main() -> None:
    default_android_home = release_paths.ANDROID_SDK
    parser = argparse.ArgumentParser(description="Run Android loading from an artifact run manifest.")
    parser.add_argument("--run-dir", required=True, help="Path under work/renders/<run_id>.")
    parser.add_argument("--project", default=str(release_paths.HOST_APP))
    parser.add_argument("--android-home", default=str(default_android_home))
    parser.add_argument("--adb", help="ADB path. Defaults to <android-home>/platform-tools/adb.")
    parser.add_argument("--app-package", default="com.example.myapplication")
    parser.add_argument("--app-activity", default=".MainActivity")
    parser.add_argument("--device-serial", default="", help="Optional adb -s device serial.")
    parser.add_argument("--avd-snapshot", help="Frozen AVD snapshot loaded before each full validation item.")
    parser.add_argument("--screenshot-delay", type=float, default=3.0)
    parser.add_argument(
        "--screenshot-settle-seconds",
        type=float,
        default=0.0,
        help=(
            "Recapture until two consecutive frames are identical, up to this many seconds. "
            "Zero keeps the fixed-delay rule the earlier runs were frozen under."
        ),
    )
    parser.add_argument("--screenshot-settle-interval", type=float, default=1.5)
    parser.add_argument(
        "--foreground-poll-seconds",
        type=float,
        default=20.0,
        help=(
            "Deadline for the diagnostic re-query of the foreground state, used only after "
            "the frozen single-query verdict has already been recorded as a failure. Set to 0 "
            "to reproduce runs made before this field existed."
        ),
    )
    parser.add_argument("--static-only", action="store_true", help="Only validate XML/resource readiness.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--screen-id", action="append", help="Only process selected screen id. Can be repeated.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing status JSONL.")
    parser.add_argument(
        "--resume-status",
        action="store_true",
        help=(
            "Keep recorded screen outcomes (except environment failures and rows whose "
            "xml_sha256 no longer matches items.jsonl), process only the missing or "
            "changed screens, and append each outcome as soon as it exists."
        ),
    )
    parser.add_argument(
        "--no-snapshot-restore",
        action="store_true",
        help=(
            "Validate against a resident guest instead of restoring the frozen AVD "
            "snapshot before each screen. Required where snapshot restore leaves the "
            "guest without a serving package manager; the mode is recorded per screen."
        ),
    )
    parser.add_argument(
        "--fast-candidate-render",
        action="store_true",
        help=(
            "DCGen leaf/root candidate MAE only: incremental assembleDebug, install -r, "
            "screenshot. Skips gradle clean, uninstall, logcat, uiautomator dump, and "
            "per-attempt device probes. Formal scoring must not pass this flag."
        ),
    )
    parser.add_argument(
        "--skip-hierarchy-dump",
        action="store_true",
        help=(
            "Never run the uiautomator dump. The dump happens after the capture, so the "
            "screenshot and every scored field are unchanged; used for renders whose view "
            "tree nobody reads (baselines, final renders scored from pixels)."
        ),
    )
    parser.add_argument(
        "--force-hierarchy-dump",
        action="store_true",
        help=(
            "Also dump the uiautomator hierarchy for Pix2Code-Easy ids, which normally skip "
            "it because Easy's metrics read the .gui DSL rather than the view tree. Our asset "
            "binding needs each ImageView's real pixel box on every split, so the pipeline "
            "passes this. It only adds a best-effort dump; the screenshot is unchanged."
        ),
    )
    parser.add_argument(
        "--no-match-reference-canvas",
        action="store_true",
        help=(
            "Disable reference-aspect canvas configuration. "
            "The default preserves reference aspect ratio at fixed width. "
            "Use this only for an explicitly different capture protocol."
        ),
    )
    parser.add_argument(
        "--no-gradle-clean",
        action="store_true",
        help=(
            "assembleDebug without gradle clean. Drawables and activity_main.xml are "
            "replaced per screen before the build, so the APK matches a clean assemble. "
            "Formal first-pass scoring can still clean; recapture uses this flag."
        ),
    )
    args = parser.parse_args()
    if not args.static_only and not args.avd_snapshot and not args.no_snapshot_restore:
        parser.error(
            "--avd-snapshot is required for full validation, or pass "
            "--no-snapshot-restore to validate against a resident guest"
        )
    if args.no_snapshot_restore and args.avd_snapshot:
        parser.error("--no-snapshot-restore and --avd-snapshot are mutually exclusive")

    android_home = Path(args.android_home).expanduser().resolve()
    adb = Path(args.adb).expanduser().resolve() if args.adb else default_adb(android_home)
    config = RunnerConfig(
        run_dir=Path(args.run_dir).expanduser().resolve(),
        project=Path(args.project).expanduser().resolve(),
        android_home=android_home,
        adb=adb,
        app_package=args.app_package,
        app_activity=args.app_activity,
        device_serial=args.device_serial,
        avd_snapshot=args.avd_snapshot or "",
        screenshot_delay=args.screenshot_delay,
        screenshot_settle_seconds=args.screenshot_settle_seconds,
        screenshot_settle_interval=args.screenshot_settle_interval,
        foreground_poll_seconds=args.foreground_poll_seconds,
        static_only=args.static_only,
        limit=args.limit,
        screen_ids=set(args.screen_id) if args.screen_id else None,
        overwrite=args.overwrite,
        fast_candidate_render=args.fast_candidate_render,
        resume_status=args.resume_status,
        no_gradle_clean=args.no_gradle_clean,
        force_hierarchy_dump=args.force_hierarchy_dump,
        skip_hierarchy_dump=args.skip_hierarchy_dump,
        match_reference_canvas=not args.no_match_reference_canvas,
    )
    statuses = run(config)
    failures = sum(1 for status in statuses if status.get("failure_code"))
    success = sum(1 for status in statuses if not status.get("failure_code"))
    print(f"Processed {len(statuses)} items: success={success}, failures={failures}")
    if failures:
        counts: dict[str, int] = {}
        for status in statuses:
            code = status.get("failure_code")
            if code:
                counts[code] = counts.get(code, 0) + 1
        for code, count in sorted(counts.items()):
            if code not in FAILURE_CODES:
                print(f"  {code}: {count} (non-standard failure code)")
            else:
                print(f"  {code}: {count}")


if __name__ == "__main__":
    main()
