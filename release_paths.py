"""Paths used by every Screen2Run script.

Code lives in this repository. Everything a run produces (model outputs, candidates, renders,
caches) goes to a work directory, by default ./work, which can be moved with SCREEN2RUN_WORK.
The Android SDK and JDK are found through the usual environment variables.

Importing this module also puts every code folder on sys.path, so the scripts can import one
another by module name from any folder.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

RELEASE = Path(__file__).resolve().parent
CODE_DIRS = ("screen2run", "android_harness", "baselines", "evaluation", "experiments", "paper")
for _d in CODE_DIRS:
    if str(RELEASE / _d) not in sys.path:
        sys.path.insert(0, str(RELEASE / _d))

WORK = Path(os.environ.get("SCREEN2RUN_WORK", RELEASE / "work")).resolve()
CANDIDATES = WORK / "candidates"          # generated XML and drawables, one folder per run
RENDERS = WORK / "renders"                # screenshots, view hierarchies and render logs
CACHE = WORK / "cache"                    # OCR and CLIP caches
HOST_PROJECTS = WORK / "host_projects"    # one copy of the host app per emulator
GRADLE_HOMES = WORK / "gradle"            # one Gradle home per emulator

DATA = RELEASE / "data"
DATASETS = DATA / "screen_lists"          # main_600.txt, backbone_120.txt
MEASUREMENTS = Path(os.environ.get("SCREEN2RUN_MEASUREMENTS", DATA / "element_measurements")).expanduser()
SCREENSHOTS = Path(os.environ.get("SCREEN2RUN_SCREENSHOTS", DATA / "screenshots"))
FONTS = RELEASE / "android_harness" / "fonts"
FONT = Path(os.environ.get("SCREEN2RUN_FONT", FONTS / "RobotoStatic-Regular.ttf")).expanduser()
HOST_APP = RELEASE / "android_harness" / "host_app"
PLACEHOLDER_DRAWABLE = RELEASE / "android_harness" / "placeholder" / "img.xml"
OCR_SWIFT = RELEASE / "screen2run" / "ocr_vision.swift"

ANDROID_SDK = Path(os.environ.get("ANDROID_SDK_ROOT", os.environ.get("ANDROID_HOME", "~/Library/Android/sdk"))).expanduser()
JAVA_HOME = Path(os.environ.get("JAVA_HOME", "/usr/lib/jvm/java-17")).expanduser()

# Upstream baseline repositories, cloned separately (see baselines/README.md).
DCGEN = Path(os.environ.get("DCGEN_HOME", RELEASE / "third_party" / "DCGen"))
LAYOUTCODER = Path(os.environ.get("LAYOUTCODER_HOME", RELEASE / "third_party" / "LayoutCoder"))


def module_file(name: str) -> Path:
    """Path of a release script given its file name, wherever it lives among the code folders."""
    for d in CODE_DIRS:
        p = RELEASE / d / name
        if p.is_file():
            return p
    raise FileNotFoundError(name)


def resolve_screenshot(value: str | Path) -> Path:
    """Resolve portable manifest paths against SCREEN2RUN_SCREENSHOTS."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else SCREENSHOTS / path


def ensure_host_project(name: str) -> Path:
    """Create an isolated source-only host project for one rendering worker."""
    import shutil
    destination = HOST_PROJECTS / name
    if not destination.exists():
        shutil.copytree(HOST_APP, destination)
        (destination / "gradlew").chmod(0o755)
    return destination


def record_path(path: str | Path) -> str:
    """Record a release-relative path when possible, otherwise the explicit path."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(RELEASE))
    except ValueError:
        return str(resolved)
