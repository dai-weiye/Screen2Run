"""Published DCGen Algorithm 2 (DCGen-Agent), adapted to Android View XML.

Why this module exists next to :mod:`pipeline`: the upstream repository that the
sibling modules reproduce runs a different mechanism -- two candidates per leaf
plus a min-MAE render selection -- and the FSE 2025 paper describes no such
step. To keep both comparable, the paper's algorithm lives here as a purely
additive implementation and ``pipeline.py`` is left untouched.

Algorithm 2 recursion (paper lines 489-497 and 523-548)::

    dcgen(img, depth):
        if depth >= max_depth:      return generate(leaf_prompt, img)
        cuts = segment_image(img)
        if not cuts:                return generate(leaf_prompt, img)
        parts = [dcgen(cut, depth+1) for cut in cuts]
        return generate(node_prompt + parts, img)

Two properties matter and are preserved exactly:

* **One call per node, no candidates, no selection, no MAE.** A leaf is a single
  Leaf-solver call; an internal node is a single Assembly call. The expected
  budget is therefore the number of nodes in the segmentation tree, not ``2L+1``.
* **The Assembly call sees the node's own image plus the code of all its
  children.** The root is just the depth-0 Assembly call over the whole
  screenshot.

Android adaptation, mirroring the sibling adapter's already-validated choices:
leaf and assembly output is Android View XML (never HTML/CSS, never
androidx/ConstraintLayout), images are always ``@drawable/img``. The paper's
leaf prompt does not manage position -- assembly does -- so this module injects
the bbox geometry deterministically, converting pixel coordinates to dp with the
shared ``_dp`` logic and a per-screen ``px_per_dp`` supplied by the caller.
"""

from __future__ import annotations

import copy
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping

from PIL import Image

from .config import SEGMENT_ID_PREFIX, AdapterConfig
from .openai_compatible import OpenAICompatibleClient
from .pipeline import ScreenResult, sha256_file, utc_now, AndroidDCGenPipeline
from .prompts_android import PAPER_LEAF_PROMPT, paper_node_prompt
from .segmentation import UpstreamImgSegmentation
from .xml_assembler import _layout_attrs
from .xml_parser import AndroidXMLParseError, parse_android_fragment

# Attributes the model may legitimately return on an assembly ViewGroup; the
# deterministic geometry pass overwrites the first two on every node.
_XMLNS_DECL_RE = re.compile(r'\s+xmlns(?::[\w.-]+)?\s*=\s*"[^"]*"', re.I)
_BLANK_DRAWABLE = (
    '<shape xmlns:android="http://schemas.android.com/apk/res/android" '
    'android:shape="rectangle"><solid android:color="#00000000"/></shape>\n'
)


@dataclass
class SegmentNode:
    """One node of the paper's segmentation tree, in reference-image pixels."""

    bbox: tuple[int, int, int, int]
    depth: int
    index: int
    children: list["SegmentNode"] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return f"{SEGMENT_ID_PREFIX}{self.index}"

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_json_tree(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),
            "children": [child.to_json_tree() for child in self.children],
        }


# --------------------------------------------------------------------------- #
# Segmentation: the paper's ``segment_image`` over the pinned upstream logic.
# --------------------------------------------------------------------------- #


def segment_image(
    image: Image.Image,
    bbox: tuple[int, int, int, int],
    *,
    var_thresh: float = 50,
    diff_thresh: float = 45,
    diff_portion: float = 0.9,
    window_size: int = 50,
) -> list[tuple[int, int, int, int]]:
    """Return the immediate cuts of one segment, in reference-image pixels.

    The paper's ``segment_image(img)`` cuts a single segment one level deep and
    recurses explicitly, whereas upstream ``init_tree`` is depth-first. This
    mirrors the two lines ``init_tree`` runs per node -- x first, y only when x
    yields nothing -- calling the pinned ``cut_img_bbox`` on the *full* image
    with the node bbox. Segmenting the full image rather than a crop is what
    makes one level here provably identical to one level there: ``cut_img_bbox``
    rotates around ``image.size`` for vertical cuts, so a crop would shift the
    y-axis cuts.
    """

    left, top, right, bottom = bbox
    if right <= left or bottom <= top:
        return []
    segmenter = UpstreamImgSegmentation(
        image,
        var_thresh=var_thresh,
        diff_thresh=diff_thresh,
        diff_portion=diff_portion,
        window_size=window_size,
    )
    cuts = segmenter.cut_img_bbox(image, bbox, line_direct="x")
    if not cuts:
        cuts = segmenter.cut_img_bbox(image, bbox, line_direct="y")
    return [(int(l), int(t), int(r), int(b)) for (l, t, r, b) in cuts]


def build_tree(
    image: Image.Image,
    *,
    max_depth: int = 2,
    var_thresh: float = 50,
    diff_thresh: float = 45,
    diff_portion: float = 0.9,
    window_size: int = 50,
) -> SegmentNode:
    """Run the paper's recursive subdivision to a real segmentation tree."""

    counter = 0

    def visit(bbox: tuple[int, int, int, int], depth: int) -> SegmentNode:
        nonlocal counter
        node = SegmentNode(bbox=bbox, depth=depth, index=counter)
        counter += 1
        if depth >= max_depth:
            return node
        cuts = segment_image(
            image,
            bbox,
            var_thresh=var_thresh,
            diff_thresh=diff_thresh,
            diff_portion=diff_portion,
            window_size=window_size,
        )
        node.children = [visit(cut, depth + 1) for cut in cuts]
        return node

    width, height = image.size
    return visit((0, 0, width, height), 0)


def tree_from_bbox_tree(tree: Mapping[str, Any]) -> SegmentNode:
    """Adapt the pinned ``to_json_tree()`` output into paper recursion nodes.

    ``segment_screenshot`` and :func:`build_tree` implement the identical
    ``init_tree`` subdivision, so a persisted bbox_tree is the paper's tree and
    can be consumed directly -- no second segmentation pass, no API.
    """

    counter = 0

    def visit(raw: Mapping[str, Any], depth: int) -> SegmentNode:
        nonlocal counter
        bbox = raw.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("every tree node must have bbox [left, top, right, bottom]")
        node = SegmentNode(
            bbox=(int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])),
            depth=depth,
            index=counter,
        )
        counter += 1
        node.children = [visit(child, depth + 1) for child in raw.get("children", [])]
        return node

    return visit(tree, 0)


def iter_nodes(root: SegmentNode) -> Iterator[SegmentNode]:
    yield root
    for child in root.children:
        yield from iter_nodes(child)


def node_count(root: SegmentNode) -> int:
    return sum(1 for _ in iter_nodes(root))


def leaf_count(root: SegmentNode) -> int:
    return sum(1 for node in iter_nodes(root) if node.is_leaf)


def depth_levels(root: SegmentNode) -> int:
    return max(node.depth for node in iter_nodes(root)) + 1


def expected_call_count(root: SegmentNode) -> int:
    """Algorithm 2 pays exactly one call per node -- leaf or assembly."""

    return node_count(root)


def plan_dry_run(root: SegmentNode) -> dict[str, Any]:
    """Summarise the recursion without touching the API, for dry-run checks."""

    per_depth: dict[int, int] = {}
    leaves_per_depth: dict[int, int] = {}
    kinds: dict[str, str] = {}
    for node in iter_nodes(root):
        per_depth[node.depth] = per_depth.get(node.depth, 0) + 1
        if node.is_leaf:
            leaves_per_depth[node.depth] = leaves_per_depth.get(node.depth, 0) + 1
        kinds[node.node_id] = (
            "leaf" if node.is_leaf else f"assembly({len(node.children)} children)"
        )
    return {
        "depth_levels": depth_levels(root),
        "nodes_per_depth": {str(k): v for k, v in sorted(per_depth.items())},
        "leaves_per_depth": {str(k): v for k, v in sorted(leaves_per_depth.items())},
        "leaf_count": leaf_count(root),
        "node_count": node_count(root),
        "expected_calls": expected_call_count(root),
        "call_formula": "nodes",
        "nodes": kinds,
    }


# --------------------------------------------------------------------------- #
# Algorithm 2 itself: one call per node, bottom-up assembly.
# --------------------------------------------------------------------------- #


class PaperDCGenAgent:
    """Paper-faithful DCGen-Agent; one completion per segmentation node."""

    def __init__(
        self,
        *,
        client: OpenAICompatibleClient,
        output_dir: Path,
        px_per_dp: float = 1.0,
    ) -> None:
        self.client = client
        self.output_dir = output_dir
        # ``px_per_dp`` must come from the caller per screen; inferring density
        # from a screenshot was the earlier failure this adapter already fixed.
        self.config = AdapterConfig(px_per_dp=px_per_dp)

    # -- geometry ---------------------------------------------------------- #

    def _apply_geometry(
        self,
        element: Any,
        node: SegmentNode,
        parent_bbox: tuple[int, int, int, int] | None,
    ) -> str:
        """Write the node's bbox into the layout as dp, reusing ``_dp``/ids.

        This is the validated way the sibling adapter keeps bbox fidelity while
        the leaf prompt stays position-free.
        """

        _layout_attrs(element, node.bbox, parent_bbox, node.index, self.config)
        return ET.tostring(element, encoding="unicode", short_empty_elements=True)

    @staticmethod
    def _parse_assembly(response: str, declared: frozenset[str]) -> Any:
        """Validate an assembly fragment, tolerating its own xmlns declaration.

        The model legitimately declares ``xmlns:android`` because the child
        fragments we hand it already carry that prefix. The namespace is
        re-owned by the serializer, so the declaration is stripped before the
        strict leaf validator runs.
        """

        sanitized = _XMLNS_DECL_RE.sub("", response)
        return parse_android_fragment(sanitized, declared)

    def _crop(self, reference: Image.Image, node: SegmentNode, screen_dir: Path) -> Path:
        path = screen_dir / "crops" / f"{node.node_id}.png"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            reference.crop(node.bbox).save(path)
        return path

    def _generate(
        self,
        node: SegmentNode,
        *,
        reference: Image.Image,
        screen_id: str,
        screen_dir: Path,
        parent_bbox: tuple[int, int, int, int] | None,
        trace: dict[str, Any],
        original_image: Path | None = None,
    ) -> str:
        """One recursive step of Algorithm 2, returning positioned XML."""

        entry = trace["nodes"].setdefault(
            node.node_id,
            {
                "bbox": list(node.bbox),
                "depth": node.depth,
                "kind": "leaf" if node.is_leaf else "assembly",
                "children": [child.node_id for child in node.children],
            },
        )
        # The paper's root assembly call takes the whole screenshot; every other
        # node takes the crop of its own bbox.
        image_path = (
            original_image
            if original_image is not None
            else self._crop(reference, node, screen_dir)
        )
        if node.is_leaf:
            fragment = self._leaf_call(node, entry, image_path, screen_id, screen_dir, trace)
        else:
            # Children first (the paper parallelises same-depth siblings; the
            # result is identical, only wall-clock differs).
            child_codes = [
                self._generate(
                    child,
                    reference=reference,
                    screen_id=screen_id,
                    screen_dir=screen_dir,
                    parent_bbox=node.bbox,
                    trace=trace,
                )
                for child in node.children
            ]
            fragment = self._assembly_call(
                node, entry, image_path, child_codes, screen_id, screen_dir, trace
            )
        return self._apply_geometry(fragment, node, parent_bbox)

    def _complete(
        self,
        *,
        node: SegmentNode,
        stage: str,
        prompt: str,
        image_path: Path,
        screen_id: str,
        screen_dir: Path,
        trace: dict[str, Any],
    ) -> str:
        completion = self.client.complete(
            screen_id=screen_id, stage=stage, prompt=prompt, image_path=image_path
        )
        raw_path = screen_dir / "raw_responses" / f"{node.node_id}.txt"
        AndroidDCGenPipeline._atomic_write(raw_path, completion.content)
        trace["usage"].append(completion.audit)
        return completion.content

    def _leaf_call(
        self,
        node: SegmentNode,
        entry: dict[str, Any],
        image_path: Path,
        screen_id: str,
        screen_dir: Path,
        trace: dict[str, Any],
    ) -> Any:
        response = entry.get("raw_response")
        if not response:
            response = self._complete(
                node=node,
                stage=f"paper:leaf:{node.node_id}",
                prompt=PAPER_LEAF_PROMPT,
                image_path=image_path,
                screen_id=screen_id,
                screen_dir=screen_dir,
                trace=trace,
            )
            entry["raw_response"] = response
            entry["raw_response_path"] = str(
                screen_dir / "raw_responses" / f"{node.node_id}.txt"
            )
            self._write_trace(screen_dir, trace)
        try:
            return parse_android_fragment(response, self.config.declared_resources)
        except AndroidXMLParseError as exc:
            # A leaf that fails validation still must not abort the screen; a
            # neutral View keeps the tree shape for its parent's assembly call.
            entry["parse_error"] = str(exc)
            return parse_android_fragment("<View />", self.config.declared_resources)

    def _assembly_call(
        self,
        node: SegmentNode,
        entry: dict[str, Any],
        image_path: Path,
        child_codes: list[str],
        screen_id: str,
        screen_dir: Path,
        trace: dict[str, Any],
    ) -> Any:
        response = entry.get("raw_response")
        if not response:
            response = self._complete(
                node=node,
                stage=f"paper:assembly:{node.node_id}",
                prompt=paper_node_prompt(child_codes),
                image_path=image_path,
                screen_id=screen_id,
                screen_dir=screen_dir,
                trace=trace,
            )
            entry["raw_response"] = response
            entry["raw_response_path"] = str(
                screen_dir / "raw_responses" / f"{node.node_id}.txt"
            )
            self._write_trace(screen_dir, trace)
        try:
            return self._parse_assembly(response, self.config.declared_resources)
        except AndroidXMLParseError as exc:
            # Deterministic fallback: an unparseable container must not lose the
            # children that were already generated at this node.
            entry["parse_error"] = str(exc)
            container = parse_android_fragment(
                "<FrameLayout />", self.config.declared_resources
            )
            for code in child_codes:
                # Children were validated when generated; re-validating deep
                # copies here keeps the fallback free of raw XML concat.
                container.append(
                    copy.deepcopy(self._parse_assembly(code, self.config.declared_resources))
                )
            entry["fallback"] = "frame_layout_of_children"
            return container

    def run_screen(
        self,
        *,
        screen_id: str,
        image_path: Path,
        tree: SegmentNode,
        resume_trace: dict[str, Any] | None = None,
    ) -> ScreenResult:
        screen_dir = self.output_dir / screen_id
        screen_dir.mkdir(parents=True, exist_ok=True)
        reference = Image.open(image_path).convert("RGB")
        plan = plan_dry_run(tree)
        trace: dict[str, Any] = resume_trace or {
            "screen_id": screen_id,
            "pipeline": "paper",
            "algorithm": "dcgen_agent_alg2",
            "created_at": utc_now(),
            "bbox_tree": tree.to_json_tree(),
            "leaf_count": plan["leaf_count"],
            "node_count": plan["node_count"],
            "expected_calls": plan["expected_calls"],
            "call_formula": "nodes",
            "usage": [],
            "leaves": {},
            "nodes": {},
        }
        self._write_trace(screen_dir, trace)

        # Algorithm 2's single entry point: the depth-0 root assembly call.
        final_xml = self._generate(
            tree,
            reference=reference,
            screen_id=screen_id,
            screen_dir=screen_dir,
            parent_bbox=None,
            trace=trace,
            original_image=image_path,
        )
        trace["final_root"] = tree.node_id
        self._write_trace(screen_dir, trace)

        final_path = screen_dir / f"{screen_id}.xml"
        if final_path.exists() and final_path.read_text(encoding="utf-8").strip() != final_xml.strip():
            raise FileExistsError(f"refusing to overwrite differing output: {final_path}")
        final_path.write_text(final_xml + "\n", encoding="utf-8")
        drawable = screen_dir / "img.xml"
        if not drawable.exists():
            drawable.write_text(_BLANK_DRAWABLE, encoding="utf-8")
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
    def _write_trace(screen_dir: Path, trace: dict[str, Any]) -> None:
        AndroidDCGenPipeline._write_trace(screen_dir, trace)
