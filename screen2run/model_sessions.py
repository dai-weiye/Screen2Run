#!/usr/bin/env python3
"Screen2Run S1-S5: layout extraction, visual enrichment, XML translation, code review, and review-guided repair. Image slots remain placeholders for Image Filling."
from __future__ import annotations
import argparse, base64, hashlib, json, math, os, re, sys, time, uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from openai import OpenAI

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)
_REPO = release_paths.RELEASE
from session_contracts import (
    parse_stage_json, prepare_reference_tree, assert_structure_preserved, validate_xml,
    xml_identity_report, REFERENCE_BOUNDS_EPSILON_PX,
    validate_visual_styles, xml_native_widget_report, coerce_visual_styles,
)
import s5_regression_check as s5_regression
import native_drawables as native_shapes
from typography_check import audit_declared_typography

PROMPTS = {
    "analyze_structure": """You are part of an elite automated software generating team.
Your job is to take a screenshot of a reference app and describe its UI structure as a JSON tree.
- Exactly identify the type of each element (text, button, image, input, container, list, icon, switch, checkbox, radio_button, slider, progress_bar, rating_bar, toggle_button).
- A switch, slider or checkbox is a native control, not an image. A round button remains a button with an observed shape, not a crop. Keep text regions as native text, except text intrinsically inside a logo or photograph.
- Exactly identify the corresponding relationship between components (which element is nested inside which).
- Exactly identify the position of each element.
- Use bounds=[left, top, right, bottom] in ORIGINAL screenshot pixels, not dp or normalized coordinates. Preserve these bounds in later JSON stages.
- Bounds describe the visible extent inside the screenshot: 0 <= left < right <= image width and 0 <= top < bottom <= image height. Use finite numbers, never xywh. Every node, including containers, must have bounds and children (use [] for leaves).
- Mark only the phone's actual OS bars with system_chrome="status_bar" or system_chrome="navigation_bar". Never mark app tabs, toolbars, menus, or bottom app navigation as system chrome. The host will assign stable node_id values after extraction.
- Make sure the description matches the screenshot exactly.
- Repeat elements as needed to match the screenshot. For example, if there are 15 items, the description should have 15 items.
- Output a SINGLE JSON object representing the UI as a tree. Each element is a JSON object with keys: "type" (the element type), "text" (its text, if any), "bounds" (its position), and "children" (a list of nested elements).
- Do NOT output HTML, XML, or any markup language. Do NOT output any code. Output ONLY the JSON tree.
- Do not add comments. WRITE THE FULL JSON TREE.
Finally, only output the JSON tree describing the type and relationship of the elements.""",

    "fill_details": """You are part of an elite automated software generating team.
Your job is to recover observed visual styling and text without changing the extracted UI hierarchy.
- Recover text, foreground/background colours, native surface shape/stroke/corners, typography and visible widget state according to the typed visual_style contract below.
- Make sure the result matches the screenshot exactly.
- Repeat elements as needed to match the screenshot.
- Keep the SAME JSON tree structure (same types, bounds, identities and hierarchy); enrich "text" and "visual_style". Existing "color" is a legacy ambiguous field, not a replacement for separate background and text colours.
- Preserve any already recorded non-null visual_style observations; do not silently erase known stroke, radius or typography. If an observation appears wrong, leave it traceable for S4 review rather than changing node identities or geometry.
- Preserve every node_id, bounds, children list, system_chrome declaration and the root _coordinate_contract EXACTLY. Do not add, drop, reparent or reorder nodes; report design disagreements later during review rather than rewriting identities here.
- Do NOT output HTML, XML, or any markup language. Output ONLY the JSON tree.
- Do not add comments. WRITE THE FULL JSON TREE.
Finally, only output the filled JSON tree.""",

    "generate_xml": """You are part of an elite automated software generating team.
Your job is to read the JSON code you received, and convert the JSON code into XML code like the activity_main.xml code in Android Studio.
- Exactly identify the type and corresponding relationship of each elements in the code
- Exactly identify the color and text of each elements in the code
- Replace all the images with '@drawable/img' you meet
- Make sure the app looks exactly like the screenshot.
- Repeat elements as needed to match the screenshot.
- Do not add comments in the code. WRITE THE FULL CODE.
- Using the activity_main.xml code in Android Studio as an example
- Pay close attention to background color, text color, font size, font family, padding, margin, border, etc. Match the colors and sizes exactly.
- Use the exact text from the screenshot.
- If you meet any icons, use @drawable/img to replace.
- Use ONLY framework view classes: LinearLayout, FrameLayout, RelativeLayout, View, Space, TextView, ImageView, Button, EditText, ScrollView, HorizontalScrollView, ListView, GridView, ProgressBar, SeekBar, Switch, CheckBox, RadioButton, RadioGroup, ToggleButton, RatingBar. Do NOT use androidx.* or ConstraintLayout or library views.
- Preserve native widget semantics. A slider with a draggable thumb is a SeekBar, NOT a ProgressBar. A switch/checkbox/radio button must use the corresponding native widget, NOT a cropped screenshot, an ImageView, or stacked rectangular TextViews. Use native checked/progress/tint attributes for the observed state; do not put @drawable/img in thumb, track, button, tickMark, or progressDrawable.
- Text present in android:text is not enough: it must fit and remain fully visible. Check fixed bounds against textSize, padding and expected lines; do not truncate or enlarge text beyond its reference box.
- **Do NOT draw the phone's own system chrome.** The status bar (clock, wifi, signal, battery) and the
  navigation bar (back / home / recent, or the gesture pill) are drawn by Android itself, at the top
  and bottom of every screen. Never emit views for them, and never crop them as images -- reproducing
  them puts a second copy on top of the real one. Layout only the app's own content.
- Do NOT invent drawable resource names (no @drawable/fab_*, @drawable/shape_*, @drawable/white_* etc.). Use @drawable/img only for true image/logo/icon assets, and literal colors or framework native styling for code-drawn backgrounds. A <shape> resource cannot be embedded inside layout XML.
- A genuine background photograph can have a background asset slot with UI controls overlaid. Never use a crop of the whole reference UI or a composite containing labels/controls as that asset. Unavailable photograph pixels are unresolved; do not fill them by copying the UI screenshot.
Finally, Return only the full code like the activity_main.xml code in Android Studio""",

    "critique_xml": """You are part of an elite automated software fixing team.
Your job is to **compare the extracted design JSON against the generated XML** and report every
difference. You are given both. The JSON is a model prediction, NOT guaranteed complete or correct.
Check it against the screenshot too. Distinguish native text/widgets from text embedded in a logo
or photograph, which belongs to the image asset rather than a duplicate TextView.
Do NOT request Android system status/navigation bar views: Android draws them itself.

Report in this order, and be exhaustive:

1. MISSING IN XML -- list EVERY element, text string and colour that appears **either in the JSON
   or in the "TEXT VISIBLE IN THE SCREENSHOT" list below**, but has no counterpart in the XML.
   That OCR list is an **independent** reading of the reference image: the design JSON was extracted
   by a model and silently misses strings, so an item can be absent from the JSON and still be on
   screen -- those count as missing too. Quote the exact text string. A single omitted digit, label,
   button caption or digit-key letter is a defect. Check each OCR line off one by one. Do not
   summarise this list; enumerate it. If nothing is missing, say exactly "MISSING IN XML: none".
2. BUILD ERRORS -- anything that would fail on Android Studio.
3. IMAGES -- every image must be '@drawable/img'; no invented resource names.
4. FIDELITY -- colours, text content, font size, padding, margin, borders, and the positional
   relationship between components, compared against the screenshot.
   Check native widget semantics: SeekBar for a draggable slider; Switch, CheckBox and RadioButton
   for their respective controls. ProgressBar or cropped widgets are not substitutes. Check whether
   text actually fits its fixed box and padding, not only whether android:text contains the words.
5. ESCAPING -- use "&amp;" for every "&".

Return the line of the error code with your suggestions.""",

    "fix_xml": """You are part of an elite automated software fixing team.
Your job is to figure out what went wrong and suggest changes to the code so that you can run the code perfectly in Android Studio

**Restore every item listed under "MISSING IN XML"** in the review above: add those elements and
text strings back into the layout, using the EXACT strings quoted there, at the position they
occupy in the screenshot. Cross-check the review against the supplied design JSON and screenshot;
neither model prediction is guaranteed complete. Do not duplicate text that belongs inside an image.
Do not drop, shorten or
paraphrase any text while restoring.
- Replace all the images with '@drawable/img' you meet
- Make sure the app looks as same as the screenshot.
- Pay close attention to background color, text color, font size, font family, padding, margin, border, etc. Match the colors and sizes exactly.
- Exactly identify the corresponding relationship between components
- Use the exact text from the screenshot.
- Do not add comments in the code.
- Repeat elements as needed to match the screenshot.
- If you meet any icons,use @drawable/img to replace.
- Use ONLY framework view classes (LinearLayout, FrameLayout, RelativeLayout, View, Space, TextView, ImageView, Button, EditText, ScrollView, HorizontalScrollView, ListView, GridView, ProgressBar, SeekBar, Switch, CheckBox, RadioButton, RadioGroup, ToggleButton, RatingBar). Do NOT use androidx.* or ConstraintLayout.
- Preserve native widget semantics: a draggable slider is a SeekBar, not a ProgressBar. Use native Switch/CheckBox/RadioButton rather than image crops or stacked rectangles; use native state/tint attributes, never @drawable/img in thumb, track, button, tickMark or progressDrawable. Preserve complete visible text within its bounds, not merely the text attribute.
- **Do NOT draw the phone's system chrome** (status bar: clock/wifi/signal/battery; navigation bar: back/home/recent or the gesture pill) -- Android draws those itself, so emitting views for them puts a second copy on top of the real one. If the current code contains such views, DELETE them.
- Do NOT invent drawable resource names. Use @drawable/img for every image; use color values (#RRGGBB) for backgrounds.
- A genuine background photograph may stay behind UI controls. A whole reference UI screenshot or composite containing labels/controls must never be used as a background asset. Unavailable photograph content remains unresolved, not a license to copy the UI screenshot. Do not embed <shape> resources inside layout XML.
Return only the full code like the activity_main.xml code in Android Studio""",

    "replace_images": """You are updating an Android activity_main.xml layout.

Task:
- Replace placeholder drawable references such as @drawable/img with concrete drawable resource names from the provided JSON image-cut metadata when a clear match exists.
- Preserve the original Android XML structure and attributes unless a drawable reference must be changed.
- If no JSON metadata is provided, keep the XML structurally valid and replace generic @drawable/img placeholders with stable local drawable-style resource names.
- If there is no @drawable/img placeholder, return the full XML unchanged.
- The JSON fields "left_top" and "right_bottom" describe image positions; use them only to infer which XML image element corresponds to which resource. Do not copy coordinates into the XML.

Output rules:
- Return only the complete activity_main.xml content.
- Do not use Markdown fences.
- Do not include explanations or comments.

Image-cut JSON metadata:
{cut}

Current activity_main.xml:
{code}""",
}


# Missing visible text is restored deterministically at its measured position by the
# execution loop.  The former S5 "add these OCR strings" rounds pasted OCR fragments
# ('zabc', '/ pors', status-bar clocks) as new views, so they are disabled.
OCR_REPAIR_ROUNDS = 0

GEOMETRY_NOTE = """
LAYOUT GEOMETRY:
- Write a natural, maintainable activity_main.xml (LinearLayout / FrameLayout / RelativeLayout,
  match_parent / wrap_content / dp) whose structure, order and approximate sizes follow the screenshot.
- Exact positions and sizes are measured from the screenshot and corrected afterwards, so do not
  spend effort on pixel arithmetic; never drop, merge away or invent a view to make numbers fit.
- The screenshot is 411dp wide on the target device; the root is match_parent in both directions.
- Every view must keep android:id="@+id/<node_id>" of the JSON node it represents.
"""

IDENTITY_MAPPING_INSTRUCTIONS = """
SOURCE-TO-XML IDENTITY CONTRACT (traceability, not proof of fidelity):
- Every JSON node has a stable node_id. Normally assign the corresponding Android
  view android:id="@+id/<node_id>". Preserve these IDs through all fixes.
- It is legal to merge a purely structural container into another view or represent
  several source parts with a native compound widget. Declare ALL represented
  source IDs using android:tag="s2r_nodes=id1,id2" on that view, which must have its
  own unique android:id. Do not insert fake/invisible views merely to satisfy IDs.
- Tags are only explicit mapping claims, never evidence that the appearance or text
  is correct. Do not dump unrelated source IDs into the root: each declared source
  must actually be represented by that view or native compound widget.
- Account for each source node once. Do not declare the same source on multiple views.
- Nodes explicitly marked system_chrome=status_bar/navigation_bar, and their children,
  are excluded from app XML. Ordinary app bottom navigation, headers and toolbars are
  NOT system chrome and must remain mapped. Do not change source node identities.
- Reference bounds are SOURCE xyxy pixels, not Android dp and not local child coordinates.
  Convert to the declared canvas and preserve Android parent-child positioning;
  do not set every child position relative to the screen or flatten real containers.
"""


VISUAL_STYLE_INSTRUCTIONS = """
TYPED VISUAL STYLE CONTRACT (observations, not verified ground truth):
- For every app node output visual_style as an object. An unknown property is
  omitted or null; {} means not yet observed, NOT a default rectangular surface.
  System chrome need not receive app styles. Never invent a radius, font size,
  stroke or hidden pixel to make a declaration look complete.
- Allowed visual_style keys and fields:
  background: {kind: "solid", color: "#RRGGBB" or "#AARRGGBB"}, OR
              {kind: "gradient", start_color, end_color, angle_degrees}.
              Gradient angles are explicitly observed 0/45/90/135/180/225/270/315.
  shape: {kind: "rectangle"|"oval"|"circle", corner_radius: quantity or null}.
         corner_radius is only for a rectangle. A circle must actually be circular;
         do not relabel a rounded rectangle as a circle or change its bounds.
  stroke: {color, width: quantity}. A border-only surface needs an explicitly
          observed transparent solid background (#00000000), not an invented fill.
  typography: {color, font_size: quantity, font_weight, font_family, font_style,
               line_height: quantity, letter_spacing: {value,unit:"em"},
               max_lines, alignment, include_font_padding}.
         font_weight is "normal"/"bold" or a string weight "100"..."900";
         font_style is "normal"/"italic"; alignment is "start"/"center"/"end";
         max_lines is a positive integer only when actually supported by evidence;
         include_font_padding is boolean only when justified, otherwise null.
  content_padding: {left: quantity, top: quantity, right: quantity, bottom: quantity}.
  widget_state: {checked: boolean, enabled: boolean, selected: boolean,
                 progress_fraction: number in [0,1]} (unknown fields null/omitted).
- Newly extracted dimensional quantities use {value: finite observed number,
  unit:"reference_px"}, i.e. original screenshot pixels. Do NOT confuse glyph ink
  height with font size or a TextView's box. Font family, font metrics and hidden
  padding often cannot be known exactly from a screenshot; retain that uncertainty.
  Letter spacing alone uses em and may be negative. No bare unlabelled dimensions.
- S3/S5 must consume these observations while preserving node IDs. Convert surface
  dimensions through the declared frame once. Font size in source pixels is not
  directly an Android sp value: account for target density AND verified font scale;
  do not copy pixel values as sp or pretend unknown font metrics are exact.
- Native rectangle/oval/circle/gradient/stroke surfaces are code-generated drawables,
  NOT screenshot assets. Preserve shape observations in JSON for the host native
  shape materializer and retain unique mapped XML IDs. Do not invent resource names
  or embed <shape> inside layout XML. A literal fill alone does not reproduce known
  corners or stroke; the surface remains pending until its declared resource exists.
- In S3/S4/S5 verify every full text string fits its native view: width, height,
  line count, glyph ascenders/descenders, line spacing, content padding, widget
  minWidth/minHeight and platform default insets matter. No arbitrary singleLine,
  ellipsize, clipping, font enlargement, fixed tiny text box or duplicate image text.
  Do not force all text into OCR ink boxes or set every padding/minimum to zero.
  Report conflicts; preserve real layout and text instead of hiding missing parts.
"""


def generation_target_frame(reference_size: tuple[int, int]) -> dict:
    """Declare the current harness protocol, not a claim of device measurement.

    Keep source screenshot coordinates distinct from the Android root's padded
    content origin. Existing/legacy XML is never reinterpreted by this function.
    Runtime capture must verify these declared dimensions before using its VH.
    """
    width, height = reference_size
    if (isinstance(width, bool) or isinstance(height, bool) or
            not isinstance(width, int) or not isinstance(height, int) or
            width <= 0 or height <= 0):
        raise ValueError("Reference dimensions must be positive integer pixels")
    screen_width = 1080
    screen_height = max(1, round(height * screen_width / width))
    density_dpi = 420
    density = density_dpi / 160
    if screen_height <= 24 * density:
        raise ValueError("Declared target has no content area below its explicit top padding")
    return {
        "version": 1,
        "protocol": "fixed_width_reference_aspect_explicit_top_padding_v1",
        "evidence": "declared_generation_target; not an observed device configuration",
        "runtime_verification": "pending_capture_screen_density_and_content_bounds",
        "source_frame": {"size_px": [width, height], "origin": "screenshot_top_left",
                         "unit": "px", "bounds_format": "xyxy"},
        "screen_size_px": [screen_width, screen_height],
        "density_dpi": density_dpi, "px_per_dp": density,
        "screen_size_dp": [screen_width / density, screen_height / density],
        "android_content_view_origin_px": [0, 0],
        "root_padding_dp": [0, 24, 0, 0],
        "root_child_content_origin_screen_dp": [0, 24],
        "root_child_content_size_dp": [screen_width / density, screen_height / density - 24],
        "padding_owner": "generated_XML_root; not automatic harness/system padding",
        "source_to_screen_dp_scale_xy": [screen_width / width / density,
                                         screen_height / height / density],
        "source_to_screen_dp_translation_xy": [0, 0],
        "source_content_bounds": "not inferred; system_chrome roles remain fallible S1 predictions",
        "legacy_fixed_height_allowed": False,
    }


def target_frame_instructions(frame: dict) -> str:
    """One origin policy shared by S3, S4, S5 and every existing repair call."""
    return """
ANDROID TARGET FRAME AND SINGLE-CONVERSION CONTRACT:
JSON bounds are reference xyxy pixels, not dp. The following is a declared target,
NOT evidence of observed runtime geometry:
""" + json.dumps(frame, ensure_ascii=False, sort_keys=True) + """
- The Android content view starts at screen (0,0). The XML root is match_parent
  in both dimensions and MUST explicitly set android:paddingTop="24dp". Keep
  its other paddings, margins and translations zero; put design spacing in inner
  containers. Do not use fitsSystemWindows=true or add another inset wrapper.
  This is EXPLICIT XML padding, NOT an additional automatic system inset.
- Convert source coordinates exactly once: screen_x_dp = source_x_px * scale_x;
  screen_y_dp = source_y_px * scale_y, using source_to_screen_dp_scale_xy above.
  Those are SCREEN positions, not XML margins. For a root child, the content
  origin is (0,24)dp, so a top/start offset is (screen_x_dp, screen_y_dp - 24).
  For a nested child, subtract its actual PARENT CONTENT origin instead. Do not
  subtract 24 again at every depth, nor copy full-screen y into local marginTop.
- A parent's content origin includes its position and its own padding. With
  LinearLayout flow, weights, gravity or relative anchors, derive local layout
  constraints that give the intended position; do not assign global margins to
  every child. Preserve real containers and native widget semantics.
- The supplied screen_size_dp height is reference-specific. Never reuse a fixed
  731dp canvas for all images, stretch all nodes into a smaller content rectangle,
  or add status-bar space twice. Root padding changes the origin/available area,
  NOT the source-to-screen scale. The pixel-height rounding is already in scale_y.
- Source system chrome is not an app view. If source geometry, parentage or the
  available app area conflicts with this frame, report that conflict in review;
  do not silently clamp, shift every element, infer a new source canvas, or hide
  missing components. Declarations alone do not prove visual correspondence.
"""


def xml_target_frame_report(xml: str, frame: dict) -> dict:
    """Check only observable root declarations; never infer child positions.

    IDs identify nodes but do not make arbitrary Android layout constraints
    solvable without rendering. This report therefore never certifies fidelity.
    """
    root = ET.fromstring(validate_xml(xml))
    android = "{http://schemas.android.com/apk/res/android}"
    issues = []

    def dp_value(name):
        value = root.get(android + name)
        if value is None:
            return None
        match = re.fullmatch(r"([-+]?(?:\d+(?:\.\d*)?|\.\d+))dp", value.strip())
        return float(match.group(1)) if match else None

    for name in ("layout_width", "layout_height"):
        if root.get(android + name) not in ("match_parent", "fill_parent"):
            issues.append(f"root_{name}_must_match_parent")
    top = dp_value("paddingTop")
    if top is None or not math.isclose(top, frame["root_padding_dp"][1], abs_tol=1e-6):
        issues.append("root_paddingTop_must_explicitly_equal_24dp")
    if root.get(android + "fitsSystemWindows", "false").lower() != "false":
        issues.append("root_automatic_insets_conflict_with_explicit_padding")
    for name in ("padding", "paddingHorizontal", "paddingVertical", "paddingLeft",
                 "paddingStart", "paddingRight", "paddingEnd", "paddingBottom",
                 "layout_margin", "layout_marginHorizontal", "layout_marginVertical",
                 "layout_marginTop", "layout_marginBottom", "layout_marginLeft",
                 "layout_marginStart", "layout_marginRight", "layout_marginEnd",
                 "translationX", "translationY"):
        if root.get(android + name) is not None and dp_value(name) != 0:
            issues.append(f"root_{name}_conflicts_with_declared_origin")
    return {
        "status": "pending_coordinate_contract" if issues else "root_declaration_consistent",
        "issues": issues,
        "target_protocol": frame["protocol"],
        "scope": "root attributes only; no child-coordinate inference or rewriting",
        "runtime_verification": "pending_capture_screen_density_and_content_bounds",
        "child_geometry_validation": "pending_render_and_source_identity_review",
        "acceptance_claim": "none; root declarations do not prove correct positions",
    }


def _img_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()



_MODEL_MAX_TOKENS = {"claude-opus-5": 128000, "deepseek-flash": 384000, "deepseek-v4.1-flash": 64000,
                       "deepseek-v4-pro": 384000, "deepseek-v4.1-flash": 64000}


def _max_tokens(model: str) -> int:
    # one output ceiling across the endpoints that serve the same model
    if os.environ.get("S2R_MAX_TOKENS"):
        return int(os.environ["S2R_MAX_TOKENS"])
    return _MODEL_MAX_TOKENS.get(model, 128000)


def _measured_hint(measured, target_frame: dict | None = None) -> str:
    "Convert measured UIED bounds to device-independent coordinates for prompt context."
    if not measured:
        return ""
    W, H = measured.get("width"), measured.get("height")
    if target_frame is None:
        try:
            target_frame = generation_target_frame((W, H))
        except ValueError:
            return "MEASURED ELEMENT POSITIONS WITHHELD: unknown source dimensions; do not infer coordinates."
    if [W, H] != target_frame["source_frame"]["size_px"]:
        return "MEASURED ELEMENT POSITIONS WITHHELD: source dimensions conflict with the reference; do not infer coordinates."
    sx, sy = target_frame["source_to_screen_dp_scale_xy"]
    canvas_w, canvas_h = target_frame["screen_size_dp"]
    lines = [f"MEASURED ELEMENT POSITIONS (SCREEN dp, canvas {canvas_w:g} x {canvas_h:g} dp; NOT local margins)."]
    lines.append("Root child top offset = screen_y_dp - 24dp; nested children use their own parent content origin. Do not apply the root inset twice.")
    lines.append("These are image-derived candidate regions, not verified widget bounds. OCR ink bounds are not TextView layout bounds. Use the screenshot to identify semantic types and parent-child relationships; do not turn a region into an image merely because it is large.")
    for e in measured.get("elements") or []:
        k = e.get("kind"); t = str(e.get("text") or "").strip(); b = e.get("bounds")
        if not isinstance(b, list) or len(b) < 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in b[:4]]
        left = round(x0 * sx, 1); top = round(y0 * sy, 1)
        w = round((x1 - x0) * sx, 1); h = round((y1 - y0) * sy, 1)
        if k == "text" and t:
            lines.append(f'- TEXT "{t}": left={left}dp top={top}dp w={w}dp h={h}dp')
        elif k == "component":
            lines.append(f'- unclassified region: left={left}dp top={top}dp w={w}dp h={h}dp')
    return "\n".join(lines)


def _sanitize_drawables(xml: str) -> str:
    "Replace unresolved drawable references with the shared image placeholder."
    import re as _re
    def repl(m):
        name = m.group(1)
        if name in ("img",) or name.startswith("guigpt"):
            return m.group(0)
        return "@drawable/img"
    return _re.sub(r"@drawable/([A-Za-z0-9_.]+)", repl, xml)

# See docs/RUNTIME.md for the shared compatibility and capture contracts.
TEMPERATURE = 0.0


def _log_usage(model: str, resp) -> None:
    ledger = os.environ.get("S2R_USAGE_LEDGER")
    usage = getattr(resp, "usage", None)
    if not ledger or usage is None:
        return
    row = {"t": time.time(), "model": model, "prompt_tokens": getattr(usage, "prompt_tokens", None),
           "completion_tokens": getattr(usage, "completion_tokens", None)}
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None and getattr(details, "cached_tokens", None) is not None:
        row["cached_tokens"] = details.cached_tokens
    with open(ledger, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")


def _call(client: OpenAI, model: str, prompt: str, image_b64: str, max_tokens: int) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        max_tokens=max_tokens,
        temperature=TEMPERATURE,
    )
    _log_usage(model, resp)
    # Preserve the exact textual model response for attempt-level evidence.
    return resp.choices[0].message.content or ""


def _strip_fence(s: str) -> str:
    """Extract one unambiguous XML code block, preserving raw evidence elsewhere.

    Explanations after a complete fenced XML document are not part of Android
    XML. Previously they made otherwise complete S5 responses fail three times.
    With no fence, require the whole reply to be XML; with multiple/incomplete
    fences, never guess which alternative the model intended.
    """
    text = s.strip()
    markers = list(re.finditer(r"(?m)^[ \t]*```([^\r\n]*)[ \t]*\r?$", text))
    if not markers:
        return text
    if len(markers) != 2:
        return text
    opening, closing = markers
    if opening.group(1).strip().lower() not in {"", "xml"} or closing.group(1).strip():
        return text
    body = text[opening.end():closing.start()].strip()
    return body if _looks_complete_xml(body) else text


def _looks_complete_xml(s: str) -> bool:
    "Check for a nonempty, parseable XML root."
    s = s.strip()
    if not s or not s.startswith("<"):
        return False
    try:
        import xml.etree.ElementTree as ET
        ET.fromstring(s)
        return True
    except Exception:
        return False


def _call_xml(client, model, prompt, b64, retries=3, request=None):
    "Request complete XML and retry truncated responses."
    last = ""
    for _ in range(retries):
        raw = (request(prompt, _max_tokens(model)) if request else
               _call(client, model, prompt, b64, _max_tokens(model)))
        last = _strip_fence(raw)
        if _looks_complete_xml(last):
            return last
        prompt = prompt + "\n\nYour previous output was truncated or incomplete. Return the COMPLETE XML, do not truncate, close every tag."
    raise ValueError("Model did not produce parseable complete XML; refusing downstream stages")


def _visible_text_evidence(xml: str, screenshot: Path) -> dict[str, list[str]]:
    """Attribute evidence is not a rendering check; asset text remains pending.

    Accessibility descriptions do not paint text. A declared image label can
    explain why OCR text is absent from native widgets, but must be verified
    after binding instead of forcibly drawing a second copy of a logo label.
    """
    import xml.etree.ElementTree as ET
    normal = lambda value: ''.join(c for c in value.casefold() if c.isalnum())
    want = []
    for line in _visible_texts(screenshot).splitlines():
        s = line[2:].strip() if line.startswith("- ") else line.strip()
        if s:
            want.append(s)
    root = ET.fromstring(xml)
    ns = '{http://schemas.android.com/apk/res/android}'
    native, image_labels = [], []
    for node in root.iter():
        native.extend(normal(node.get(ns + attr, '')) for attr in ('text', 'hint'))
        if node.tag.rsplit('}', 1)[-1].rsplit('.', 1)[-1] == 'ImageView':
            image_labels.append(normal(node.get(ns+'contentDescription', '')))
    out = {'native_attribute_evidence': [], 'asset_text_pending_render': [], 'missing_candidates': []}
    for s in want:
        key = normal(s)
        if not key:
            continue
        if any(key in t for t in native):
            out['native_attribute_evidence'].append(s)
        elif key in image_labels:
            out['asset_text_pending_render'].append(s)
        else:
            out['missing_candidates'].append(s)
    return out


def _missing_visible(xml: str, screenshot: Path) -> list[str]:
    return _visible_text_evidence(xml, screenshot)['missing_candidates']


def _visible_texts(screenshot: Path, min_chars: int = 2) -> str:
    "Read visible reference text for independent S4 review context. Return an empty string if OCR is unavailable."
    try:
        import re as _re
        import sys as _sys
        scripts = release_paths.RELEASE
        if str(scripts) not in _sys.path:
            _sys.path.insert(0, str(scripts))
        from ocr import ScreenshotOCR
        cache = release_paths.CACHE / "ocr"
        with ScreenshotOCR(cache) as ex:
            rec = ex.extract(screenshot)
        out, seen = [], set()
        for b in rec.get("blocks") or []:
            tx = (b.get("text") or "").strip()
            if sum(c.isalnum() for c in tx) < min_chars:
                continue
            if tx.lower() in seen:
                continue
            seen.add(tx.lower())
            out.append(f"- {tx}")
        return "\n".join(out)
    except Exception:
        return ""


def _replayed_response(screenshot: Path, stage: str, prompt: str, model: str | None = None) -> str | None:
    """Reuse a previous attempt's response for the identical request (S2R_REPLAY_ROOT).

    Only a response whose recorded input is byte-identical to this prompt is reused, so a
    re-run after a local (non-model) failure spends no new call and cannot change inputs.
    """
    root = os.environ.get("S2R_REPLAY_ROOT")
    if not root:
        return None
    only = os.environ.get("S2R_REPLAY_STAGES")
    if only and stage not in only.split(","):
        return None
    sid = screenshot.name[:-4] if screenshot.name.lower().endswith(".png") else screenshot.name
    dirs = sorted(Path(root).glob(f"{sid}_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    for d in dirs:
        # a reply is only ever reused for the model that produced it (prompts are model-independent)
        try:
            if json.loads((d / "generation_manifest.json").read_text(encoding="utf-8")).get("model") != model:
                continue
        except (OSError, json.JSONDecodeError):
            continue
        for inp in sorted(d.glob(f"{stage}_attempt*_input.txt")):
            out = inp.with_name(inp.name.replace("_input.txt", "_raw.txt"))
            if out.is_file() and inp.read_text(encoding="utf-8") == prompt:
                return out.read_text(encoding="utf-8")
    return None


def generate(client: OpenAI, model: str, screenshot: Path, out_dir: Path, measured: dict | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    b64 = _img_b64(screenshot)
    from PIL import Image
    with Image.open(screenshot) as image:
        width, height = image.size
    target_frame = generation_target_frame((width, height))
    reference_sha = hashlib.sha256(screenshot.read_bytes()).hexdigest()
    manifest = {"schema_version": 2, "run_id": uuid.uuid4().hex, "model": model,
                "reference_sha256": reference_sha, "reference_size_px": [width, height],
                "coordinate_space": "reference_pixels", "bbox_format": "xyxy",
                "bounds_tolerance_px": REFERENCE_BOUNDS_EPSILON_PX,
                "bounds_tolerance_action": "preserve original finite values; never silently clamp",
                "contract_scope": "new_generation_only; legacy artifacts are not reinterpreted",
                "target_frame": target_frame,
                "semantic_validation": "pending", "stages": {}}

    def persist():
        (out_dir / "generation_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def request(stage, actual_prompt, max_tokens):
        """Record every actual request/response, including retries and failed calls.

        Credentials and endpoints are not stored. The image is identified by its
        source hash and the exact transmitted base64 hash, rather than duplicated.
        """
        entry = manifest["stages"].setdefault(stage, {"attempts": []})
        attempt = len(entry["attempts"]) + 1
        stem = f"{stage}_attempt{attempt:02d}"
        prompt_file = out_dir / f"{stem}_input.txt"
        prompt_file.write_text(actual_prompt, encoding="utf-8")
        call = {
            "attempt": attempt, "prompt_file": prompt_file.name,
            "prompt_file_sha256": hashlib.sha256(prompt_file.read_bytes()).hexdigest(),
            "reference_sha256": reference_sha,
            "image_base64_sha256": hashlib.sha256(b64.encode()).hexdigest(),
            "image_transport": "data:image/png;base64", "model": model,
            "max_tokens": max_tokens, "temperature": TEMPERATURE,
            "message_roles": ["user"], "status": "requested",
        }
        entry["attempts"].append(call)
        persist()
        replay = _replayed_response(screenshot, stage, actual_prompt, model)
        try:
            raw = replay if replay is not None else _call(client, model, actual_prompt, b64, max_tokens)
            if replay is not None:
                call["replayed_from_previous_attempt"] = True
        except Exception as exc:
            # Exception messages may contain gateway details; retain type only.
            call.update({"status": "request_failed", "error_type": type(exc).__name__})
            persist()
            raise
        output_file = out_dir / f"{stem}_raw.txt"
        output_file.write_text(raw, encoding="utf-8")
        call.update({"status": "response_received", "output_file": output_file.name,
                     "output_file_sha256": hashlib.sha256(output_file.read_bytes()).hexdigest()})
        persist()
        return raw

    def request_xml(stage, prompt):
        return validate_xml(_call_xml(client, model, prompt, b64,
                            request=lambda p, tokens: request(stage, p, tokens)))

    def record(stage, prompt, output, dependencies):
        # Store the actual stage text, not retrospective file times, without API credentials.
        (out_dir / f"{stage}_input.txt").write_text(prompt, encoding="utf-8")
        manifest["stages"].setdefault(stage, {"attempts": []}).update({
            "input_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "input_scope": "initial_prompt; every actual request is recorded in attempts",
            "canonical_output_sha256": hashlib.sha256(output.encode()).hexdigest(),
            "dependencies": dependencies, "reference_sha256": reference_sha})
        output_file = {"s1": "stage1.json", "s2": "stage2.json",
                       "s3": "stage3_original.xml", "s4": "stage4_critique.txt",
                       "s5": "stage5_initial.xml"}.get(stage)
        if stage.startswith("s5_repair"):
            output_file = f"stage5_repair{stage.removeprefix('s5_repair')}.xml"
        if output_file and (out_dir / output_file).is_file():
            manifest["stages"][stage].update({"output_file": output_file,
                "output_file_sha256": hashlib.sha256((out_dir / output_file).read_bytes()).hexdigest()})
        persist()

    def record_mapping(stage, xml, tree):
        report = xml_identity_report(tree, xml)
        filename = f"{stage}_identity_mapping.json"
        path = out_dir / filename
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["stages"][stage]["identity_mapping"] = {
            "status": report["status"], "file": filename,
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "semantic_validation": report["semantic_validation"],
        }
        coordinate_report = xml_target_frame_report(xml, target_frame)
        coordinate_path = out_dir / f"{stage}_target_frame.json"
        coordinate_path.write_text(json.dumps(coordinate_report, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["stages"][stage]["target_frame_check"] = {
            "status": coordinate_report["status"], "file": coordinate_path.name,
            "file_sha256": hashlib.sha256(coordinate_path.read_bytes()).hexdigest(),
        }
        native_report = xml_native_widget_report(tree, xml)
        native_path = out_dir / f"{stage}_native_widget_contract.json"
        native_path.write_text(json.dumps(native_report, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["stages"][stage]["native_widget_check"] = {
            "status": native_report["status"], "file": native_path.name,
            "file_sha256": hashlib.sha256(native_path.read_bytes()).hexdigest(),
        }
        typography = audit_declared_typography(xml, tree, target_frame, font_scale=1)
        typography_path = out_dir / f"{stage}_typography_contract.json"
        typography_path.write_text(json.dumps(typography, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest["stages"][stage]["typography_check"] = {
            "status": typography["status"], "file": typography_path.name,
            "file_sha256": hashlib.sha256(typography_path.read_bytes()).hexdigest(),
            "font_scale_source": "declared generation scale=1; requires runtime verification",
        }
        persist()
        return report

    # RQ2 ablations (unset for the full system): no_s1 drops layout analysis (S3 receives a
    # root-only tree and must lay out from the screenshot), no_s2 drops visual enrichment
    # (S3-S5 receive the S1 layout tree as the design JSON).
    ablation = os.environ.get("S2R_ABLATION", "")
    manifest["ablation"] = ablation or None
    prompt = PROMPTS["analyze_structure"] + f"\nOriginal screenshot size: {width} x {height} pixels."
    if ablation == "no_s1":
        s1 = json.dumps({"type": "container", "bounds": [0, 0, width, height], "children": []})
    else:
        s1 = request("s1", prompt, _max_tokens(model))
    (out_dir / "stage1_raw.txt").write_text(s1, encoding="utf-8")
    s1_parsed = parse_stage_json(s1)
    s1_coercions = coerce_visual_styles(s1_parsed, fill_missing=False) if isinstance(s1_parsed, dict) else []
    s1_tree = prepare_reference_tree(s1_parsed, (width, height))
    s1 = json.dumps(s1_tree, ensure_ascii=False)
    (out_dir / "stage1.json").write_text(s1, encoding="utf-8")
    record("s1", prompt, s1, [])

    prompt = PROMPTS["fill_details"] + VISUAL_STYLE_INSTRUCTIONS + f"\n\nHere is the JSON code you received:\n{s1}"
    if ablation in ("no_s1", "no_s2"):
        s2 = s1
    else:
        s2 = request("s2", prompt, _max_tokens(model))
    (out_dir / "stage2_raw.txt").write_text(s2, encoding="utf-8")
    s2_tree = parse_stage_json(s2)
    coercions = coerce_visual_styles(s2_tree) if isinstance(s2_tree, dict) else []
    assert_structure_preserved(s1_tree, s2_tree)
    visual_report = validate_visual_styles(s2_tree, require_declared=not ablation)
    visual_report["coerced_to_unknown"] = {"s1": s1_coercions, "s2": coercions}
    visual_path = out_dir / "s2_visual_style_contract.json"
    visual_path.write_text(json.dumps(visual_report, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["visual_style_contract"] = {"file": visual_path.name,
        "file_sha256": hashlib.sha256(visual_path.read_bytes()).hexdigest(),
        "semantic_validation": visual_report["semantic_validation"],
        "empty_or_missing_node_ids": visual_report["empty_or_missing_node_ids"]}
    s2 = json.dumps(s2_tree, ensure_ascii=False)
    (out_dir / "stage2.json").write_text(s2, encoding="utf-8")
    record("s2", prompt, s2, ["s1"])

    hint = _measured_hint(measured, target_frame)
    # Geometry is owned by measurement: the execution loop (image_filling.py) renders this XML,
    # measures every view and re-grounds it on the reference.  Asking the model for exact
    # pixel->dp arithmetic only produced drift (DeepSeek reads 688x1070 as 656x1134).
    canvas = GEOMETRY_NOTE
    prompt = PROMPTS["generate_xml"] + canvas + IDENTITY_MAPPING_INSTRUCTIONS + VISUAL_STYLE_INSTRUCTIONS + f"\n\nHere is the JSON code you received:\n{s2}\n\n{hint}"
    xml = request_xml("s3", prompt)
    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    (out_dir / "stage3_original.xml").write_text(xml + "\n", encoding="utf-8")
    record("s3", prompt, xml, ["s2"])
    mapping = record_mapping("s3", xml, s2_tree)

    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    visible = _visible_texts(screenshot)
    visible_block = ("\n\nTEXT VISIBLE IN THE SCREENSHOT (fallible OCR candidates): "
                     "check native captions as android:text/hint, and keep text intrinsic to logos "
                     "inside their image assets. contentDescription is accessibility metadata, "
                     "NOT visible rendered text. Exclude system chrome and OCR mistakes:\n"
                     + visible) if visible else ""
    typography = audit_declared_typography(xml, s2_tree, target_frame, font_scale=1)
    typography_issues = {"conversion_issues": typography["issues"],
        "checks": [{"source_node_id": node["source_node_id"], **check}
                   for node in typography["nodes"] for check in node["checks"] if check["status"] != "matched"]}
    prompt = (PROMPTS["critique_xml"] + canvas + IDENTITY_MAPPING_INSTRUCTIONS + VISUAL_STYLE_INSTRUCTIONS
                     + f"\n\nHere is the extracted design JSON (unverified prediction):\n{s2}"
                     + f"\n\nDeterministic source identity accounting (NOT semantic verification):\n{json.dumps(mapping, ensure_ascii=False)}"
                     + f"\n\nRoot coordinate declaration checks (child geometry still unverified):\n{json.dumps(xml_target_frame_report(xml, target_frame), ensure_ascii=False)}"
                     + f"\n\nNative widget type checks (state and text fit still unverified):\n{json.dumps(xml_native_widget_report(s2_tree, xml), ensure_ascii=False)}"
                     + f"\n\nTypography declaration discrepancies (NOT runtime fit; inspect reference before resolving):\n{json.dumps(typography_issues, ensure_ascii=False)}"
                     + visible_block
                     + f"\n\nHere is the code you generated:\n{xml}")
    critique = request("s4", prompt, _max_tokens(model))
    if not critique.strip():
        raise ValueError("S4 review is empty; refusing to silently skip review")
    (out_dir / "stage4_critique.txt").write_text(critique, encoding="utf-8")
    record("s4", prompt, critique, ["s2", "s3"])

    prompt = (PROMPTS["fix_xml"] + canvas + IDENTITY_MAPPING_INSTRUCTIONS + VISUAL_STYLE_INSTRUCTIONS + f"\n\nComplete design JSON:\n{s2}"
              + f"\n\nSource identity accounting to resolve or preserve explicitly:\n{json.dumps(mapping, ensure_ascii=False)}"
              + visible_block + f"\n\nHere is the code you generated:\n{xml}\n\nHere are the suggestions:\n{critique}\n\n{hint}")
    xml = request_xml("s5", prompt)
    (out_dir / "stage5_initial.xml").write_text(xml + "\n", encoding="utf-8")
    record("s5", prompt, xml, ["s2", "s3", "s4"])
    record_mapping("s5", xml, s2_tree)

    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    for round_no in range(1, OCR_REPAIR_ROUNDS + 1):
        missing = _missing_visible(xml, screenshot)
        if not missing:
            break
        (out_dir / f"missing_text_round{round_no}.txt").write_text("\n".join(missing), encoding="utf-8")
        checklist = "\n".join(f"- {m}" for m in missing)
        prompt = (PROMPTS["fix_xml"] + canvas + IDENTITY_MAPPING_INSTRUCTIONS + VISUAL_STYLE_INSTRUCTIONS
            + "\n\n**The following strings are visible in the screenshot but are MISSING from the"
              " XML. Add every one of them now, using the EXACT characters shown, as android:text"
              " (or android:hint for an input field's placeholder). Do not merge, shorten, translate"
              " or paraphrase them. Do not remove anything else.**\n" + checklist
            + f"\n\nComplete design JSON:\n{s2}\n\nHere is the code you generated:\n{xml}\n\n{hint}")
        xml = request_xml(f"s5_repair{round_no}", prompt)
        (out_dir / f"stage5_repair{round_no}.xml").write_text(xml + "\n", encoding="utf-8")
        record(f"s5_repair{round_no}", prompt, xml, ["s2", "s5" if round_no == 1 else "s5_repair1"])
        record_mapping(f"s5_repair{round_no}", xml, s2_tree)
    still = _missing_visible(xml, screenshot)
    if still:
        (out_dir / "missing_text_unresolved.txt").write_text("\n".join(still), encoding="utf-8")

    xml = _sanitize_drawables(xml)
    # Materialize only explicit S2 surfaces, after placeholder sanitization so
    # trusted content-addressed shape resources are not changed back to img.
    # Surface lengths use x-scale; y-scale differs only by target-height rounding.
    reference_px_per_dp = 1.0 / target_frame["source_to_screen_dp_scale_xy"][0]
    xml, native_surfaces = native_shapes.bind_declared_native_shapes(
        xml, s2_tree, out_dir / "drawables", reference_px_per_dp=reference_px_per_dp)
    native_surfaces.update({
        "conversion_source": "declared target_frame x-scale; not runtime-verified density",
        "dimension_policy": "surface lengths use source x-scale; y-scale only differs by canvas pixel-height rounding",
        "materializer_sha256": hashlib.sha256(Path(native_shapes.__file__).read_bytes()).hexdigest(),
    })
    native_surfaces_path = out_dir / "s5_native_surface_materialization.json"
    native_surfaces_path.write_text(json.dumps(native_surfaces, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["native_surface_materialization"] = {
        "file": native_surfaces_path.name,
        "file_sha256": hashlib.sha256(native_surfaces_path.read_bytes()).hexdigest(),
        "status": native_surfaces["status"],
        "applied_resources": [item["resource"] for item in native_surfaces["applied"]],
        "conversion_source": native_surfaces["conversion_source"],
        "acceptance_claim": "none; resource creation is not render verification",
    }
    validate_xml(xml)
    out = out_dir / "activity_main.xml"
    out.write_text(xml + "\n", encoding="utf-8")
    manifest["final_xml_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    manifest["missing_text_candidates"] = still
    manifest["visible_text_evidence"] = _visible_text_evidence(xml, screenshot)
    manifest["review_resolution"] = "not_independently_verified"
    manifest["final_identity_mapping"] = xml_identity_report(s2_tree, xml)
    manifest["final_target_frame_check"] = xml_target_frame_report(xml, target_frame)
    manifest["final_native_widget_check"] = xml_native_widget_report(s2_tree, xml)
    final_typography = audit_declared_typography(out.read_text(encoding="utf-8"), s2_tree, target_frame, font_scale=1)
    typography_path = out_dir / "s5_final_typography_contract.json"
    typography_path.write_text(json.dumps(final_typography, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["final_typography_check"] = {
        "status": final_typography["status"], "file": typography_path.name,
        "file_sha256": hashlib.sha256(typography_path.read_bytes()).hexdigest(),
        "counts": final_typography["counts"],
        "scope": "declared font_scale=1; no runtime fit or fidelity claim",
    }
    manifest["generation_issues"] = []
    manifest["generation_acceptance"] = "pending_visual_semantic_review"
    if final_typography["status"] == "pending_typography_contract":
        manifest["generation_acceptance"] = "pending_typography_contract"
        manifest["generation_issues"].append({"kind": "pending_typography_contract",
                                               "report": typography_path.name})
    if native_surfaces["pending"]:
        manifest["generation_acceptance"] = "pending_native_surface_materialization"
        manifest["generation_issues"].append({
            "kind": "pending_native_surface_materialization",
            "issues": native_surfaces["pending"],
        })
    if manifest["final_target_frame_check"]["status"] == "pending_coordinate_contract":
        manifest["generation_acceptance"] = "pending_coordinate_contract"
        manifest["generation_issues"].append({
            "kind": "pending_coordinate_contract",
            "issues": manifest["final_target_frame_check"]["issues"],
        })
    if manifest["final_identity_mapping"]["status"] == "pending_mapping":
        manifest["generation_acceptance"] = "pending_identity_mapping"
        manifest["generation_issues"].append({
            "kind": "pending_identity_mapping",
            "report": "final_identity_mapping",
        })
    if manifest["final_native_widget_check"]["status"] == "pending_native_widget_contract":
        manifest["generation_acceptance"] = "pending_native_widget_contract"
        manifest["generation_issues"].append({
            "kind": "pending_native_widget_contract",
            "issues": manifest["final_native_widget_check"]["issues"],
        })
    # This is a local supported-layout hold, not an extra model/fix round and
    # not a general Android renderer. Never automatically roll back S5.
    s3_path = out_dir / "stage3_original.xml"
    regression = s5_regression.audit_xml(s3_path.read_text(encoding="utf-8"),
                                         out.read_text(encoding="utf-8"),
                                         target_frame["screen_size_dp"][1])
    regression.update({
        "viewport_source": "declared target_frame.screen_size_dp; NOT a runtime measurement",
        "target_protocol": target_frame["protocol"],
        "inputs": {"s3": s3_path.name, "final": out.name},
        "auditor_sha256": hashlib.sha256(Path(s5_regression.__file__).read_bytes()).hexdigest(),
    })
    outside_kinds = ("s5_fully_outside_viewport", "new_fully_outside_viewport")
    outside_counts = {kind: regression["summary"][kind] for kind in outside_kinds
                      if regression["summary"].get(kind, 0)}
    regression_path = out_dir / "s3_to_final_geometry_regression.json"
    regression_path.write_text(json.dumps(regression, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest["final_geometry_regression"] = {
        "file": regression_path.name,
        "file_sha256": hashlib.sha256(regression_path.read_bytes()).hexdigest(),
        "viewport_source": regression["viewport_source"],
        "summary": regression["summary"],
        "status": "pending_geometry_regression" if outside_counts else "no_supported_static_outside_bounds_found",
        "acceptance_claim": "none; zero alerts is not visual or runtime validation",
    }
    if outside_counts:
        manifest["generation_acceptance"] = "pending_geometry_regression"
        manifest["generation_issues"].append({"kind": "pending_geometry_regression",
                                               "findings": outside_counts})
    persist()
    return out


def main() -> int:
    global TEMPERATURE
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    ap.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"))
    ap.add_argument("--temperature", type=float, default=TEMPERATURE,
                    help="Sampling temperature; 0.0 by default, matching the baselines.")
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit("need OPENAI_API_KEY")
    TEMPERATURE = args.temperature
    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    t0 = time.time()
    out = generate(client, args.model, args.image, args.out)
    print(f"done in {time.time() - t0:.1f}s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
