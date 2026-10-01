#!/usr/bin/env python3
"""Batch renderer: many screens per APK build, same capture protocol for every arm.

The per-screen harness (device_session.py) spends ~5 of its ~11 s per screen on a Gradle
build and an install. Here one build carries up to ``--batch`` layouts (each screen's drawables
renamed with a per-screen prefix, so no resource of one screen can resolve for another) and the
host activity picks the layout from an intent extra. Per screen the device then only resizes the
canvas, launches, waits for a drawn and settled frame, captures, and optionally dumps the view
hierarchy.

Protocol (identical for every arm rendered with it):
  canvas 1080 x round(1080 * ref_h / ref_w), density untouched (411 dp wide), light status bar;
  ``am start -W``; after 1.0 s, wait for the target window with no covering Splash Screen;
  take fresh frames every 0.5 s until two are identical (cap 3 s); a flat black app area
  is re-taken up to 8 times and never recorded as success if still blank; verify the foreground
  window again after capture. An app that crashes (device crash buffer,
  cleared before each launch) is a loading failure of that screen, not an environment failure.
A layout the resource compiler rejects is a compile failure of that screen alone: the batch is
rebuilt without it and the failure is recorded, exactly as a per-screen build would record it.

  python3 batch_render.py --cand <candidate root> [--hierarchy] [--devices 3] [--batch 24]
Outputs go to work/renders/tsc_<cand>_full/{raw/screenshots,raw/hierarchy,derived} as before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from multiprocessing import Process
from pathlib import Path

from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
REPO = release_paths.RELEASE
RUNS = release_paths.RENDERS
from device_session import foreground_activity_ok, frame_is_blank, hierarchy_has_system_anr  # noqa: E402
from run_image_filling import read_sids, shot_map  # noqa: E402
from host_resources import PLACEHOLDER, find_resource  # noqa: E402

DEVICES = [
    ("emulator-5554", "proposed_pilot_w00", "gradle-home-proposed-pilot-w00"),
    ("emulator-5558", "proposed_pilot_w02", "gradle-home-proposed-pilot-w02"),
    ("emulator-5560", "proposed_pilot_w03", "gradle-home-proposed-pilot-w03"),
]
PKG = "com.example.myapplication"
# v2 stopped recording all remaining screens as failures when a device goes offline.
# v3 adds window/Splash gating and fresh-frame, PNG/resource provenance verification.
RENDERER = "s2r_batch_render_v4_canvas_gate"
# resources every new Android Studio project ships with (Empty Views Activity template); a layout may
# reference them like any other project resource
TEMPLATE_STRINGS = ('<?xml version="1.0" encoding="utf-8"?>\n<resources>\n'
                    '    <string name="app_name">My Application</string>\n</resources>\n')
TEMPLATE_MIPMAPS = ("ic_launcher", "ic_launcher_round")
# failures of the harness, not of the arm's code: the screen is rendered again
ENV_LOADING = frozenset({"activity_launch_failed", "screenshot_failed", "capture_exception",
                         "apk_install_failed", "environment_missing", "blank_frame", "system_anr",
                         "canvas_size_mismatch", "canvas_not_declared"})
# build errors caused by template resources the project lacked before TEMPLATE_* were added
HARNESS_FIXED = ("string/app_name", "mipmap/ic_launcher")
SDK = release_paths.ANDROID_SDK
ADB = SDK / "platform-tools" / "adb"
JAVA_HOME = release_paths.JAVA_HOME
DRAWABLE_REF = re.compile(r"@drawable/([A-Za-z0-9_.]+)")
RES_EXT = (".png", ".jpg", ".jpeg", ".webp", ".xml", ".9.png")

MAIN_ACTIVITY = """package com.example.myapplication;

import android.app.Activity;
import android.os.Bundle;

public class MainActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        String name = getIntent() == null ? null : getIntent().getStringExtra("layout");
        int id = name == null ? 0 : getResources().getIdentifier(name, "layout", getPackageName());
        setContentView(id != 0 ? id : R.layout.activity_main);
    }
}
"""


def run(cmd: list, timeout: float = 120, env: dict | None = None) -> subprocess.CompletedProcess:
    try:
        return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, timeout=timeout,
                              env={**os.environ, **(env or {})})
    except subprocess.TimeoutExpired as exc:
        text = lambda v: v.decode("utf-8", "replace") if isinstance(v, bytes) else (v or "")
        return subprocess.CompletedProcess(exc.cmd, 124, text(exc.stdout), text(exc.stderr) + "\ntimeout")


def adb(serial: str, *args, timeout: float = 30) -> subprocess.CompletedProcess:
    return run([ADB, "-s", serial, *args], timeout=timeout)


def online(serial: str) -> bool:
    r = run([ADB, "-s", serial, "get-state"], timeout=10)
    return r.returncode == 0 and r.stdout.strip() == "device"


def wait_online(serial: str, limit: float = 120.0) -> bool:
    t0 = time.monotonic()
    while not online(serial):
        if time.monotonic() - t0 > limit:
            return False
        time.sleep(5)
    return True


def final_outcome(rec: dict) -> bool:
    """True when a record settles its screen for that XML: a capture, or a failure of the arm's own code."""
    st = rec.get("status")
    if st == "render_success":
        # a pre-render is complete only with the view hierarchy the execution loop reads
        return rec.get("hierarchy") is not False
    if st != "loading_failure" or (rec.get("failure_code") or "") in ENV_LOADING:
        return False
    return not (rec.get("failure_code") == "gradle_build_failed"
                and any(h in (rec.get("failure_detail") or "") for h in HARNESS_FIXED))


def canvas_of(ref: Path) -> tuple[int, int]:
    with Image.open(ref) as im:
        w, h = im.size
    return 1080, max(1, round(h * 1080 / w))


def screen_resources(screen_dir: Path) -> dict[str, Path]:
    res = {}
    d = screen_dir / "drawables"
    if d.is_dir():
        for f in d.iterdir():
            if f.is_file() and f.name.lower().endswith(RES_EXT):
                stem = f.name[:-6] if f.name.lower().endswith(".9.png") else f.stem
                res.setdefault(stem, f)
    return res


def screen_resource_provenance(screen_dir: Path) -> dict:
    """Hash the screen's packaged resources, including transitive drawables.

    The shared placeholder is included because @drawable/img resolves to it rather
    than to a candidate's own similarly named file. No credentials or logs are read.
    """
    own = screen_resources(screen_dir)
    files = {name: hashlib.sha256(path.read_bytes()).hexdigest()
             for name, path in sorted(own.items())}
    text = (screen_dir / "final.xml").read_text(encoding="utf-8")
    pending = list(set(DRAWABLE_REF.findall(text)))
    visited = set()
    while pending:
        name = pending.pop()
        if name in visited:
            continue
        visited.add(name)
        path = PLACEHOLDER if name == "img" else own.get(name) or find_resource(screen_dir, name)
        if path is None:
            files[name] = "missing"
            continue
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix.lower() == ".xml":
            pending.extend(DRAWABLE_REF.findall(path.read_text(encoding="utf-8")))
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"resources_sha256": hashlib.sha256(encoded).hexdigest(), "resource_file_sha256": files}


def stage_screen(k: int, sid: str, screen_dir: Path, layout_dir: Path, drawable_dir: Path) -> str | None:
    """Write layout s2r_<k>.xml with drawables renamed b<k>_<name>. Returns a failure code or None."""
    xml_path = screen_dir / "final.xml"
    if not xml_path.is_file():
        return "missing_output"
    text = xml_path.read_text(encoding="utf-8", errors="ignore")
    import xml.etree.ElementTree as ET
    try:
        ET.fromstring(text.split("?>", 1)[-1] if text.lstrip().startswith("<?xml") else text)
    except ET.ParseError:
        return "xml_parse_error"
    # the same resolution as host_resources: @drawable/img is the shared placeholder,
    # every other name is looked up among the screen's own drawables
    res = {}
    own = screen_resources(screen_dir)
    pending = list(sorted(set(DRAWABLE_REF.findall(text))))
    while pending:
        name = pending.pop()
        if name in res:
            continue
        f = PLACEHOLDER if name == "img" else own.get(name) or find_resource(screen_dir, name)
        if f is None:
            return "missing_drawable"
        res[name] = f
        if f.suffix.lower() == ".xml":
            resource_xml = f.read_text(encoding="utf-8")
            try:
                ET.fromstring(resource_xml)
            except ET.ParseError:
                return "drawable_xml_parse_error"
            pending.extend(set(DRAWABLE_REF.findall(resource_xml)) - set(res))
    if len({name.lower() for name in res}) != len(res):
        return "drawable_name_collision"
    prefix = f"b{k}_"
    for name, f in res.items():
        ext = ".9.png" if f.name.lower().endswith(".9.png") else f.suffix.lower()
        dest = drawable_dir / f"{prefix}{name.lower()}{ext}"
        if ext == ".xml":
            resource_xml = DRAWABLE_REF.sub(lambda m: f"@drawable/{prefix}{m.group(1).lower()}",
                                           f.read_text(encoding="utf-8"))
            dest.write_text(resource_xml, encoding="utf-8")
        else:
            shutil.copy2(f, dest)
    text = DRAWABLE_REF.sub(lambda m: f"@drawable/{prefix}{m.group(1).lower()}", text)
    (layout_dir / f"s2r_{k}.xml").write_text(text, encoding="utf-8")
    return None


def configure_natural_canvas_manifest(project: Path) -> dict:
    """Remove the harness's portrait lock while preserving its exact old source.

    ``nosensor`` follows the display's natural canvas instead of sensor rotation;
    the existing ``wm size WIDTHxHEIGHT`` remains the sole sizing operation. No
    screenshot rotation/resampling or generated layout modification is performed.
    """
    import xml.etree.ElementTree as ET
    manifest = project / "app" / "src" / "main" / "AndroidManifest.xml"
    original = manifest.read_bytes()
    root = ET.fromstring(original)
    ns = "{http://schemas.android.com/apk/res/android}"
    activities = [node for node in root.findall("./application/activity")
                  if node.get(ns + "name") in {".MainActivity", "MainActivity", PKG + ".MainActivity"}]
    if len(activities) != 1:
        raise ValueError("harness manifest must identify exactly one MainActivity")
    activity = activities[0]
    old = activity.get(ns + "screenOrientation")
    provenance_path = project / ".s2r_canvas_orientation.json"
    if old == "nosensor":
        return json.loads(provenance_path.read_text()) if provenance_path.is_file() else {
            "policy": "natural_canvas_nosensor", "previous_screenOrientation": old,
            "manifest_sha256": hashlib.sha256(original).hexdigest(), "changed": False,
        }
    before_sha = hashlib.sha256(original).hexdigest()
    backup = project / f".s2r_manifest_before_canvas_{before_sha[:16]}.xml"
    if backup.exists() and backup.read_bytes() != original:
        raise ValueError("orientation backup identity conflict")
    if not backup.exists():
        backup.write_bytes(original)
    activity.set(ns + "screenOrientation", "nosensor")
    ET.register_namespace("android", ns[1:-1])
    updated = ET.tostring(root, encoding="utf-8", xml_declaration=True) + b"\n"
    manifest.write_bytes(updated)
    report = {
        "policy": "natural_canvas_nosensor", "activity": activity.get(ns + "name"),
        "previous_screenOrientation": old, "screenOrientation": "nosensor", "changed": True,
        "source_manifest": str(manifest), "source_manifest_sha256": before_sha,
        "source_backup": str(backup), "manifest_sha256": hashlib.sha256(updated).hexdigest(),
        "renderer": RENDERER,
    }
    provenance_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


class Device:
    def __init__(self, serial: str, project: str, home: str):
        self.serial = serial
        self.project = release_paths.ensure_host_project(project)
        self.env = {"GRADLE_USER_HOME": str(release_paths.GRADLE_HOMES / home),
                    "GRADLE_OPTS": "-Xmx512m -Dorg.gradle.daemon=true -Dorg.gradle.caching=true",
                    "JAVA_HOME": str(JAVA_HOME), "ANDROID_HOME": str(SDK),
                    "PATH": f"{JAVA_HOME / 'bin'}:{SDK / 'platform-tools'}:{os.environ.get('PATH', '')}"}
        self.size = None
        self.orientation_provenance = configure_natural_canvas_manifest(self.project)
        act = self.project / "app" / "src" / "main" / "java" / "com" / "example" / "myapplication" / "MainActivity.java"
        if act.read_text(encoding="utf-8") != MAIN_ACTIVITY:
            act.write_text(MAIN_ACTIVITY, encoding="utf-8")

    @property
    def res(self) -> Path:
        return self.project / "app" / "src" / "main" / "res"

    def ensure_template_resources(self) -> None:
        strings = self.res / "values" / "strings.xml"
        if not strings.is_file() or strings.read_text(encoding="utf-8") != TEMPLATE_STRINGS:
            strings.write_text(TEMPLATE_STRINGS, encoding="utf-8")
        mipmap = self.res / "mipmap"
        mipmap.mkdir(exist_ok=True)
        for name in TEMPLATE_MIPMAPS:
            png = mipmap / f"{name}.png"
            # v2 mistakenly copied the XML placeholder with a PNG suffix. Only
            # remove that exact harness-created file, never a genuine icon.
            if png.is_file() and png.read_bytes() == PLACEHOLDER.read_bytes():
                png.unlink()
            target = mipmap / f"{name}{PLACEHOLDER.suffix.lower()}"
            if not target.is_file() and not png.is_file():
                shutil.copy2(PLACEHOLDER, target)

    def build(self) -> subprocess.CompletedProcess:
        return run([self.project / "gradlew", "-p", self.project, "assembleDebug", "-x", "lint", "-x", "test", "-q"],
                   timeout=900, env=self.env)

    def apk(self) -> Path:
        return self.project / "app" / "build" / "outputs" / "apk" / "debug" / "app-debug.apk"

    def set_size(self, size: tuple[int, int]) -> None:
        want = f"{size[0]}x{size[1]}"
        if self.size == want:
            return
        adb(self.serial, "shell", "wm", "size", want)
        time.sleep(1.5)
        self.size = want

    def frame(self, dest: Path) -> bool:
        r = subprocess.run([str(ADB), "-s", self.serial, "exec-out", "screencap", "-p"], capture_output=True, timeout=30)
        if r.returncode != 0 or not r.stdout.startswith(b"\x89PNG"):
            return False
        dest.write_bytes(r.stdout)
        return True


def app_crash(serial: str) -> str | None:
    """The layout's own inflate/runtime crash, read from the device crash buffer."""
    r = adb(serial, "logcat", "-d", "-b", "crash", timeout=15)
    if "FATAL EXCEPTION" not in r.stdout or PKG not in r.stdout:
        return None
    lines = [l.split("AndroidRuntime:", 1)[-1].strip() for l in r.stdout.splitlines()
             if "Caused by:" in l or "Exception:" in l]
    return (lines[-1] if lines else "FATAL EXCEPTION")[:500]


def capture(dev: Device, k: int, dest_png: Path, dest_vh: Path | None,
            expected_size: tuple[int, int] | None = None) -> dict:
    rec: dict = {}
    if expected_size is None:
        declared = re.fullmatch(r"([1-9]\d*)x([1-9]\d*)", str(getattr(dev, "size", "")))
        if declared:
            expected_size = (int(declared.group(1)), int(declared.group(2)))
    if expected_size is None:
        return {"status": "environment_failure", "failure_code": "canvas_not_declared"}
    rec["expected_png_size_px"] = list(expected_size)
    adb(dev.serial, "shell", "am", "force-stop", PKG)
    adb(dev.serial, "logcat", "-b", "crash", "-c")
    t0 = time.monotonic()
    launch = adb(dev.serial, "shell", "am", "start", "-W", "-n", f"{PKG}/.MainActivity", "--es", "layout", f"s2r_{k}",
                 timeout=40)
    rec["launch_ms"] = int((time.monotonic() - t0) * 1000)
    if launch.returncode != 0 or "Error" in launch.stdout:
        return {**rec, "status": "loading_failure", "failure_code": "activity_launch_failed",
                "failure_detail": (launch.stdout + launch.stderr)[-500:]}
    time.sleep(1.0)
    crash = app_crash(dev.serial)
    if crash:
        return {**rec, "status": "loading_failure", "failure_code": "runtime_crash", "failure_detail": crash}
    tmp = dest_png.with_suffix(".tmp.png")
    # ResumedActivity is not sufficient: Android's Splash Screen can still cover
    # it. Window state is checked both before and after freshly acquiring pixels.
    deadline = time.monotonic() + 15.0
    while True:
        # Android 16's `window windows` omits the focused-window fields; the
        # complete window dump contains both focus and Splash window evidence.
        focus = adb(dev.serial, "shell", "dumpsys", "window", timeout=15)
        if focus.returncode == 0 and foreground_activity_ok(focus.stdout, PKG):
            if not dev.frame(tmp):
                return {**rec, "status": "environment_failure", "failure_code": "screenshot_failed"}
            prev = hashlib.sha256(tmp.read_bytes()).hexdigest()
            settled, waited = False, 0.0
            while waited < 3.0:
                time.sleep(0.5)
                waited += 0.5
                if not dev.frame(tmp):
                    continue
                cur = hashlib.sha256(tmp.read_bytes()).hexdigest()
                if cur == prev and not frame_is_blank(tmp):
                    settled = True
                    break
                prev = cur
            blank_tries = 0
            while frame_is_blank(tmp) and blank_tries < 8:
                time.sleep(1.5)
                blank_tries += 1
                dev.frame(tmp)
            rec.update({"settled": settled, "settle_wait_ms": int(waited * 1000),
                        "blank_frame_recaptures": blank_tries, "blank_frame": frame_is_blank(tmp)})
            focus = adb(dev.serial, "shell", "dumpsys", "window", timeout=15)
            if focus.returncode == 0 and foreground_activity_ok(focus.stdout, PKG):
                if rec["blank_frame"]:
                    tmp.unlink(missing_ok=True)
                    return {**rec, "status": "environment_failure", "failure_code": "blank_frame"}
                rec["foreground_window_verified"] = True
                break
        if time.monotonic() > deadline:
            tmp.unlink(missing_ok=True)
            crash = app_crash(dev.serial)
            if crash:
                return {**rec, "status": "loading_failure", "failure_code": "runtime_crash",
                        "failure_detail": crash}
            seen = [l.strip() for l in focus.stdout.splitlines()
                    if "mCurrentFocus=" in l or "mFocusedApp=" in l or "ResumedActivity" in l][:4]
            return {**rec, "status": "environment_failure", "failure_code": "environment_missing",
                    "failure_detail": "target activity not in the foreground at capture: " + " | ".join(seen)}
        time.sleep(0.25)
        # Do not acquire/certify a screenshot while Splash or another app covers it.
    try:
        with Image.open(tmp) as captured:
            actual_size = captured.size
            captured.verify()
    except (OSError, ValueError):
        tmp.unlink(missing_ok=True)
        return {**rec, "status": "environment_failure", "failure_code": "screenshot_failed",
                "failure_detail": "captured PNG cannot be decoded"}
    rec["png_size_px"] = list(actual_size)
    if actual_size != tuple(expected_size):
        tmp.unlink(missing_ok=True)
        return {**rec, "status": "environment_failure", "failure_code": "canvas_size_mismatch",
                "failure_detail": f"requested {expected_size[0]}x{expected_size[1]}, captured "
                                  f"{actual_size[0]}x{actual_size[1]}; no rotation/rescaling applied"}
    rec["canvas_size_verified"] = True
    if dest_vh is not None:
        remote = f"/sdcard/s2r_vh_{k}.xml"
        ok = False
        for _ in range(3):
            d = adb(dev.serial, "shell", "uiautomator", "dump", remote, timeout=12)
            if d.returncode == 0:
                p = adb(dev.serial, "pull", remote, str(dest_vh), timeout=20)
                if p.returncode == 0 and dest_vh.is_file():
                    text = dest_vh.read_text(encoding="utf-8", errors="replace")
                    if hierarchy_has_system_anr(text):
                        adb(dev.serial, "shell", "input", "keyevent", "KEYCODE_BACK")
                        adb(dev.serial, "shell", "pkill", "uiautomator")
                        adb(dev.serial, "shell", "rm", "-f", remote)
                        dest_vh.unlink(missing_ok=True)
                        tmp.unlink(missing_ok=True)
                        return {**rec, "status": "environment_failure", "failure_code": "system_anr"}
                    ok = True
                    break
            adb(dev.serial, "shell", "pkill", "uiautomator")
        adb(dev.serial, "shell", "rm", "-f", remote)
        rec["hierarchy"] = ok
        if not ok:
            dest_vh.unlink(missing_ok=True)
    tmp.replace(dest_png)
    rec["png_sha256"] = hashlib.sha256(dest_png.read_bytes()).hexdigest()
    adb(dev.serial, "shell", "am", "force-stop", PKG)
    return {**rec, "status": "render_success", "failure_code": ""}


def worker(dev_spec: tuple, cand: Path, sids: list[str], out: Path, hierarchy: bool, batch: int) -> None:
    import fcntl
    project = release_paths.ensure_host_project(dev_spec[1])
    lock = open(project / ".s2r_render.lock", "w")
    # one renderer per device/project: two would rewrite each other's layouts between build and capture
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        dev = Device(*dev_spec)
    except BaseException:
        lock.close()
        raise
    shots = shot_map()
    status_f = out / "derived" / f"loading_status_{dev.serial}.jsonl"
    log_f = out / f"batch_render_{dev.serial}.log"
    (out / "raw" / "screenshots").mkdir(parents=True, exist_ok=True)
    (out / "raw" / "hierarchy").mkdir(parents=True, exist_ok=True)
    status_f.parent.mkdir(parents=True, exist_ok=True)

    def log(msg):
        with log_f.open("a", encoding="utf-8") as fh:
            fh.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")

    source_provenance = {}

    def emit(sid, rec):
        xml = cand / "full" / sid / "final.xml"
        rec = {"screen_id": sid, "renderer": RENDERER, "device": dev.serial,
               "harness_orientation": dev.orientation_provenance,
               "xml_sha256": hashlib.sha256(xml.read_bytes()).hexdigest() if xml.is_file() else None,
               **source_provenance.get(sid, {}),
               "canvas": "x".join(map(str, canvas_of(shots[sid]))), "timestamp": datetime.now().astimezone().isoformat(),
               **rec}
        with status_f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

    if not wait_online(dev.serial, 60):
        log("device offline at start; nothing rendered")
        return
    dev.ensure_template_resources()
    queue = sorted(sids, key=lambda s: canvas_of(shots[s]))
    layout_dir, drawable_dir = dev.res / "layout", dev.res / "drawable"
    # the per-screen harness leaves its last layout (and its drawable references) behind
    (layout_dir / "activity_main.xml").write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android"'
        ' android:layout_width="match_parent" android:layout_height="match_parent"/>\n', encoding="utf-8")
    nb = 0
    while queue:
        chunk, queue = queue[:batch], queue[batch:]
        nb += 1
        for f in list(layout_dir.glob("s2r_*.xml")):
            f.unlink()
        shutil.rmtree(drawable_dir, ignore_errors=True)
        drawable_dir.mkdir(parents=True)
        staged = {}
        for k, sid in enumerate(chunk):
            screen_dir = cand / "full" / sid
            source_provenance[sid] = {
                "xml_sha256": hashlib.sha256((screen_dir / "final.xml").read_bytes()).hexdigest(),
                **screen_resource_provenance(screen_dir),
            }
            code = stage_screen(k, sid, cand / "full" / sid, layout_dir, drawable_dir)
            if code:
                emit(sid, {"status": "loading_failure", "failure_code": code})
            else:
                staged[k] = sid
        t0 = time.monotonic()
        for attempt in range(len(staged) + 1):
            if not staged:
                break
            b = dev.build()
            if b.returncode == 0:
                break
            err = b.stdout + b.stderr
            bad = {int(m) for m in re.findall(r"layout/s2r_(\d+)\.xml", err)} | \
                {int(m) for m in re.findall(r"drawable/b(\d+)_", err)}
            bad &= set(staged)
            if not bad:
                if len(staged) == 1:
                    bad = set(staged)
                else:
                    # unattributable error: split the batch by building the halves in turn
                    keep = sorted(staged)[: len(staged) // 2]
                    bad = set(staged) - set(keep)
                    for k in bad:
                        (layout_dir / f"s2r_{k}.xml").unlink(missing_ok=True)
                        for f in drawable_dir.glob(f"b{k}_*"):
                            f.unlink()
                    pend = {k: staged.pop(k) for k in bad}
                    queue = [pend[k] for k in sorted(pend)] + queue
                    log(f"unattributed build error; deferring {len(pend)} screens to their own batch")
                    continue
            for k in bad:
                emit(staged[k], {"status": "loading_failure", "failure_code": "gradle_build_failed",
                                 "failure_detail": err[-800:]})
                (layout_dir / f"s2r_{k}.xml").unlink(missing_ok=True)
                for f in drawable_dir.glob(f"b{k}_*"):
                    f.unlink()
                staged.pop(k)
        if not staged:
            continue
        build_s = time.monotonic() - t0
        if not wait_online(dev.serial):
            log("device offline before install; leaving the rest for the next run")
            return
        inst = adb(dev.serial, "install", "-r", "-t", str(dev.apk()), timeout=300)
        if inst.returncode != 0:
            if not online(dev.serial):
                log("device went offline during install; leaving the rest for the next run")
                return
            for k, sid in staged.items():
                emit(sid, {"status": "environment_failure", "failure_code": "apk_install_failed",
                           "failure_detail": (inst.stdout + inst.stderr)[-500:]})
            log(f"install failed: {(inst.stdout + inst.stderr)[-300:]}")
            continue
        t1 = time.monotonic()
        for k, sid in sorted(staged.items()):
            if not wait_online(dev.serial):
                log("device offline; leaving the rest for the next run")
                return
            dev.set_size(canvas_of(shots[sid]))
            png = out / "raw" / "screenshots" / f"{sid}.png"
            vh = out / "raw" / "hierarchy" / f"{sid}.xml" if hierarchy else None
            rec = None
            for _ in range(2):
                try:
                    got = capture(dev, k, png, vh, expected_size=canvas_of(shots[sid]))
                except Exception as exc:  # a device hiccup is an environment failure of this screen only
                    got = {"status": "environment_failure", "failure_code": "capture_exception",
                           "failure_detail": f"{type(exc).__name__}: {str(exc)[:300]}"}
                rec = got if rec is None or got["status"] == "render_success" else rec
                if rec["status"] == "render_success":
                    break
            if rec["status"] != "render_success" and not online(dev.serial):
                log(f"{sid}: device went offline during capture; not recorded")
                return
            emit(sid, rec)
        log(f"batch {nb}: {len(staged)} screens, build {build_s:.0f}s, capture {time.monotonic() - t1:.0f}s")
    for f in list(layout_dir.glob("s2r_*.xml")):
        f.unlink()
    for f in list(drawable_dir.glob("b*_*")):
        f.unlink()
    adb(dev.serial, "shell", "wm", "size", "reset")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cand", type=Path, required=True)
    ap.add_argument("--hierarchy", action="store_true")
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--rerender", action="store_true", help="capture again even if a capture exists")
    args = ap.parse_args()
    cand = args.cand.resolve()
    out = RUNS / f"tsc_{cand.name}_full"
    sids = [s for s in read_sids(cand / "screenshots.txt") if (cand / "full" / s / "final.xml").is_file()]
    if args.only:
        sids = [s for s in sids if s in set(args.only)]
    done = set()
    if not args.rerender:
        for f in (out / "derived").glob("loading_status*.jsonl") if (out / "derived").is_dir() else []:
            for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                xml = cand / "full" / r.get("screen_id", "") / "final.xml"
                if final_outcome(r) and xml.is_file() and \
                        r.get("xml_sha256") == hashlib.sha256(xml.read_bytes()).hexdigest():
                    done.add(r["screen_id"])
    todo = [s for s in sids if s not in done]
    print(f"{cand.name}: {len(todo)} of {len(sids)} screens to render on {args.devices} devices", flush=True)
    if not todo:
        return 0
    groups = [todo[i::args.devices] for i in range(args.devices)]
    procs = [Process(target=worker, args=(DEVICES[i], cand, g, out, args.hierarchy, args.batch))
             for i, g in enumerate(groups) if g]
    import signal

    def stop(signum, frame):
        for p in procs:
            if p.is_alive():
                p.terminate()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    (out / "derived").mkdir(parents=True, exist_ok=True)
    rows = {}
    every = []
    for f in sorted((out / "derived").glob("loading_status_*.jsonl")):
        every += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    for r in sorted(every, key=lambda r: r.get("timestamp", "")):
        rows[r["screen_id"]] = r
    (out / "derived" / "loading_status.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows.values()),
                                                         encoding="utf-8")
    ok = sum(1 for r in rows.values() if r["status"] == "render_success")
    print(f"{cand.name}: {ok}/{len(rows)} rendered", flush=True)
    requested_successes = {sid for sid in sids if rows.get(sid, {}).get("status") == "render_success"}
    return 0 if len(requested_successes) == len(sids) and all(p.exitcode == 0 for p in procs) else 2


if __name__ == "__main__":
    raise SystemExit(main())
