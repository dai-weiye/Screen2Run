"""Preserved LayoutCoder preprocessing plus Android atomic generation."""

import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from PIL import Image, ImageStat

from .client import OpenAICompatibleClient, sha256_bytes
from .prompts import build_atomic_prompt
from .structure_renderer import render_structure
from .xml_extract import extract_android_xml


SPACE_XML = (
    '<Space xmlns:android="http://schemas.android.com/apk/res/android" '
    'android:layout_width="match_parent" android:layout_height="match_parent" />'
)


def environment_check() -> Dict[str, Any]:
    """Report, rather than conceal, the legacy environment constraints."""

    modules = ("cv2", "numpy", "PIL", "paddleocr", "paddle")
    found = {name: importlib.util.find_spec(name) is not None for name in modules}
    return {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "modules": found,
        "ready": all(found.values()),
        "required_python": "3.8 (upstream); macOS uses CPU paddlepaddle",
    }


def normalize_detection_payload(
    payload: Any, *, image_width: int, image_height: int, key: str
) -> Dict[str, Any]:
    """Make empty OCR/component outputs explicit and downstream-safe."""

    if not isinstance(payload, dict):
        payload = {}
    value = payload.get(key)
    return {
        **payload,
        "img_shape": payload.get("img_shape", [image_height, image_width, 3]),
        key: value if isinstance(value, list) else [],
    }


def _walk_atomic(node: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    if node.get("type") == "atomic":
        yield node
        return
    for child in node.get("value", []):
        yield from _walk_atomic(child)


def _fallback_structure(width: int, height: int) -> Dict[str, Any]:
    return {
        "type": "atomic",
        "portion": 1,
        "position": {
            "column_min": 0, "row_min": 0,
            "column_max": width, "row_max": height,
        },
    }


def crop_atomics(
    image_path: Path, structure: Dict[str, Any], crop_dir: Path
) -> List[Dict[str, Any]]:
    """Assign stable IDs, crop every atomic, and mark white skips."""

    image = Image.open(image_path).convert("RGB")
    crop_dir.mkdir(parents=True, exist_ok=True)
    result = []
    for atomic_id, node in enumerate(_walk_atomic(structure), 1):
        node["id"] = atomic_id
        position = node.get("position") or {
            "column_min": 0, "row_min": 0,
            "column_max": image.width, "row_max": image.height,
        }
        box = (
            max(0, int(position["column_min"])),
            max(0, int(position["row_min"])),
            min(image.width, int(position["column_max"])),
            min(image.height, int(position["row_max"])),
        )
        crop = image.crop(box)
        crop_path = crop_dir / f"atomic_{atomic_id:03d}.png"
        crop.save(crop_path)
        mean = sum(ImageStat.Stat(crop.convert("L")).mean) / 1
        is_white = mean >= 250.0
        result.append({"id": atomic_id, "path": str(crop_path), "white": is_white})
    return result


def preprocess(image_path: Path, work_dir: Path, seed: int = 42) -> Dict[str, Any]:
    """Run UIED → relation construction → layout parsing → mask tree."""

    random.seed(seed)
    image_path = Path(image_path).resolve()
    work_dir = Path(work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    env = environment_check()
    if not env["ready"]:
        missing = [name for name, ok in env["modules"].items() if not ok]
        raise RuntimeError("preprocess dependencies unavailable: " + ", ".join(missing))

    # UIED's legacy imports expect its directory on sys.path.
    import release_paths
    project_root = release_paths.LAYOUTCODER
    uied_root = project_root / "UIED"
    for path in (str(project_root), str(uied_root)):
        if path not in sys.path:
            sys.path.insert(0, path)
    from run_single import ui2code_pipeline
    from utils.code_gen.layout_extract import mask2json

    ui2code_pipeline(
        input_path_img=str(image_path),
        output_root=str(work_dir),
        is_uied=True,
        is_lines=True,
        is_layout=True,
        is_divide=True,
        is_global_gen=False,
    )
    name = image_path.stem
    data = mask2json(str(work_dir / "sep"), str(work_dir / "struct"), name)
    width, height = Image.open(image_path).size
    structure = data.get("structure")
    if not isinstance(structure, dict) or not structure.get("type"):
        structure = _fallback_structure(width, height)
        data["structure"] = structure
    atomics = crop_atomics(image_path, structure, work_dir / "struct" / "partial")
    data["atomics"] = atomics
    data["image_path"] = str(image_path)
    data["report"] = {
        "atomic_count": len(atomics),
        "white_skip_count": sum(int(item["white"]) for item in atomics),
        "planned_calls": sum(int(not item["white"]) for item in atomics),
        "seed": seed,
    }
    path = work_dir / "struct" / f"{name}_android_preprocess.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return data


def generate(
    data: Dict[str, Any],
    client: OpenAICompatibleClient,
    output_xml: Path,
    *,
    dry_run: bool = False,
    call_prefix: str = "",
) -> Dict[str, Any]:
    """Generate exactly once per non-white atomic and fuse via XML renderer."""

    responses: Dict[str, str] = {}
    calls_before = client.calls
    for atomic in data["atomics"]:
        atomic_id = str(atomic["id"])
        if atomic["white"]:
            responses[atomic_id] = SPACE_XML
        elif not dry_run:
            result = client.generate(
                Path(atomic["path"]),
                build_atomic_prompt(),
                f"{call_prefix}atomic-{atomic_id}",
            )
            responses[atomic_id] = extract_android_xml(result["content"])
    if dry_run:
        return {
            **data["report"],
            "actual_calls": 0,
            "status": "dry-run",
        }
    xml = render_structure(data["structure"], responses)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    output_xml.write_text(xml, encoding="utf-8")
    return {
        **data["report"],
        "actual_calls": client.calls - calls_before,
        "xml_sha256": sha256_bytes(xml.encode("utf-8")),
        "status": "generated",
    }
