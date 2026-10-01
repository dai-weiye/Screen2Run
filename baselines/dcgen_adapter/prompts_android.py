"""Android equivalents of the two DCGenGrid prompt stages."""

import hashlib
from typing import Dict

LEAF_PROMPT = """You are implementing one cropped GUI segment as Android View XML.
Return exactly one Android View XML fragment (for example, a TextView, ImageView,
or ViewGroup). Return the fragment only, either bare or in one ```xml fence.
Use @drawable/img for every image. Do not use any other undeclared custom
resource. Do not include an XML declaration, namespace declaration, full layout
document, network URL, HTML/CSS/JavaScript, or Jetpack Compose/Kotlin.
Use only attributes supported by Android resource XML. In particular,
android:colorFilter is not a valid layout XML attribute; use android:tint on
ImageView when a tint is required.
Preserve the segment's text, color, hierarchy, and visual styling. The caller
will provide position and size when assembling the bbox tree."""

ROOT_REFINE_PROMPT = """Compare the prototype screenshot with the deterministically
assembled Android View XML below. Produce one refined Android View XML document.
Keep a single FrameLayout root and the stable dcgen_seg_N ids and bbox geometry.
Use @drawable/img for every image. Do not use network URLs or undeclared custom
resources. Return Android View XML only. HTML/CSS/JavaScript and Jetpack
Compose/Kotlin are forbidden.

[CODE]"""


# The two prompts below belong to the published Algorithm 2 (DCGen-Agent),
# which calls the model exactly once per node: once as the Leaf-solver for a
# leaf segment, and once as the Assembly MLLM for a parent segment. They are
# kept separate from LEAF_PROMPT/ROOT_REFINE_PROMPT, which describe the
# upstream repository's different 2-candidate + MAE mechanism, so that neither
# behaviour is disturbed.

PAPER_LEAF_PROMPT = """You are the Leaf-solver MLLM of DCGen. The image is one
cropped segment of an Android app screenshot. Return exactly one Android View
XML fragment that reproduces that segment (for example a TextView, ImageView,
Button, or a ViewGroup of framework views). Return the fragment only, either
bare or inside one ```xml fence.
Use @drawable/img for every image and no other undeclared resource. Do not emit
an XML declaration, an xmlns:android declaration, a full layout document,
HTML/CSS/JavaScript, Jetpack Compose/Kotlin, or any network URL.
Do not set android:layout_width, android:layout_height, or any margin: the
caller supplies this segment's position and size from its bounding box in dp.
Preserve visible text, colors, hierarchy, and styling."""

PAPER_NODE_PROMPT = """You are the Assembly MLLM of DCGen. The image is one
node of an Android app screenshot. The text below lists the Android View XML
fragments already generated for this node's child segments, in order. Return
exactly one Android View XML fragment: a ViewGroup (FrameLayout, LinearLayout,
RelativeLayout, or similar) that contains every child fragment and arranges
them as the image shows. Return the fragment only, either bare or inside one
```xml fence.
Keep every child fragment unchanged: preserve its android:id, text, children,
and @drawable/img references. Never delete, rename, or paraphrase a child. Use
only framework views. Do not emit an XML declaration, an xmlns:android
declaration, a full layout document, HTML/CSS/JavaScript, Jetpack
Compose/Kotlin, or any network URL. The caller supplies this node's own position
and size.

[PARTS]"""


def paper_node_prompt(child_codes: list[str]) -> str:
    """Render the Assembly prompt with this node's child fragments inlined."""

    parts = "\n\n".join(
        f"Child fragment {index}:\n{code}"
        for index, code in enumerate(child_codes)
    )
    return PAPER_NODE_PROMPT.replace("[PARTS]", parts)


def leaf_prompts() -> tuple[str, str]:
    """Return the two candidate prompts required for every leaf."""

    return (LEAF_PROMPT, LEAF_PROMPT)


def root_refine_prompt(assembled_xml: str) -> str:
    return ROOT_REFINE_PROMPT.replace("[CODE]", assembled_xml)


def prompt_hashes() -> Dict[str, str]:
    return {
        "leaf_candidate_1": hashlib.sha256(LEAF_PROMPT.encode("utf-8")).hexdigest(),
        "leaf_candidate_2": hashlib.sha256(LEAF_PROMPT.encode("utf-8")).hexdigest(),
        "root_refine": hashlib.sha256(ROOT_REFINE_PROMPT.encode("utf-8")).hexdigest(),
    }
