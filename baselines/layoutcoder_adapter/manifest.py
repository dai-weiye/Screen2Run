"""Artifact manifest writer compatible with scripts/run_android_loading.py."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _display_path(path: Path, repository_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository_root.resolve()))
    except ValueError:
        return str(path.resolve())


def write_run_manifest(
    *,
    run_dir: Path,
    repository_root: Path,
    run_id: str,
    screen_id: str,
    image_path: Path,
    xml_path: Path,
    model: str,
    seed: int,
    report: Dict[str, Any],
    ledger_path: Path,
) -> Dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    resource_dir = run_dir / "resources" / screen_id
    resource_dir.mkdir(parents=True, exist_ok=True)
    drawable = resource_dir / "img.xml"
    drawable.write_text(
        '<shape xmlns:android="http://schemas.android.com/apk/res/android" '
        'android:shape="rectangle"><solid android:color="#FFBDBDBD"/></shape>\n',
        encoding="utf-8",
    )
    item = {
        "run_id": run_id,
        "screen_id": screen_id,
        "input_png": _display_path(image_path, repository_root),
        "input_png_sha256": _hash(image_path),
        "xml_path": _display_path(xml_path, repository_root),
        "xml_sha256": _hash(xml_path),
        "resource_dir": _display_path(resource_dir, repository_root),
        "resource_files": [{
            "path": _display_path(drawable, repository_root),
            "sha256": _hash(drawable),
        }],
        "expected_screenshot_path": f"raw/screenshots/{screen_id}.png",
        "stage_files": [],
    }
    (run_dir / "items.jsonl").write_text(
        json.dumps(item, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "1.0",
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "Redraw_D",
        "method": "layoutcoder_android_adapted",
        "model": {"provider": "openai-compatible", "identifier": model, "revision": model},
        "randomness": {"seed": seed, "temperature": 0.0, "top_p": None},
        "screen_count": 1,
        "expected_screen_count": 1,
        "selection": {"screen_ids": [screen_id], "selection_rule": "explicit pilot"},
        "ledger_path": _display_path(ledger_path, repository_root),
        "call_budget": {
            "planned": report["planned_calls"],
            "actual": report["actual_calls"],
        },
        "atomic_count": report["atomic_count"],
        "white_skip_count": report["white_skip_count"],
        "target": "android_xml",
        "adaptation_status": "adapted_not_reproduced",
        "upstream_commit": "bf5b0032923ea68a0aff9f98fa9cd544d8cd9ee8",
        "items_file": "items.jsonl",
        "events_file": "events.jsonl",
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    return manifest
