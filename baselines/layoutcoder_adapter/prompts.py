"""Prompt contracts for generating one local Android View subtree."""

ANDROID_XML_LOCAL_PROMPT = """
You generate one local Android XML layout subtree from a cropped GUI screenshot.

Contract:
- Return exactly one XML element subtree and no explanation or XML declaration.
- Use standard Android Views (for example FrameLayout, LinearLayout, TextView,
  ImageView, Button, or Space), not HTML, CSS, JavaScript, Jetpack Compose, or
  custom code.
- Every element must have android:layout_width and android:layout_height.
- Declare xmlns:android="http://schemas.android.com/apk/res/android" on the
  returned root so the fragment is independently parseable.
- Use only literal values and built-in Android attributes. Do not reference
  undeclared colors, strings, styles, fonts, dimensions, IDs, or other
  resources.
- Do not use URLs. Every image or icon must use android:src="@drawable/img".
- Use only attributes accepted by Android resource XML. In particular,
  android:colorFilter is not a valid layout XML attribute; use android:tint on
  ImageView when a tint is required.
- Do not emit data binding expressions, tools attributes, comments, or
  placeholders for omitted views.
- Match visible text, colors, spacing, alignment, and hierarchy as closely as
  possible while keeping the subtree nestable.
""".strip()


# v2 (2026-09-29): carries over every contract line of upstream LayoutCoder's local
# prompt (utils/code_gen/partial_code.py, HTML_TAILWINDCSS_LOCAL_PROMPT) into Android
# terms.  v1 dropped "exactly like the screenshot / exact colors and sizes", "repeat
# elements", "fill and adapt to the container", "relative units", "outermost fills the
# container" and "margin and padding 0", so atomics overflowed or clipped their slots.
ANDROID_XML_LOCAL_PROMPT_V2 = ANDROID_XML_LOCAL_PROMPT + """

Upstream LayoutCoder local contract, in Android terms:
- Make sure the layout looks exactly like the screenshot. Pay close attention to
  background color, text color, font size, font family, padding, margin and border.
  Match the colors and sizes exactly. Use the exact text from the screenshot.
- Write the full code. Repeat elements as needed to match the screenshot; for example,
  if there are 15 items, the code must contain 15 items.
- Keep the aspect ratio of images identical to the screenshot.
- The subtree is nested inside a parent container: it must extend to fill the entire
  container and adapt to a varying container size. Use relative sizing (match_parent,
  wrap_content, LinearLayout weights) rather than fixed absolute sizes.
- The outermost view uses android:layout_width="match_parent" and
  android:layout_height="match_parent", with margin and padding set to 0.
- Do not use maxWidth or maxHeight.
""".rstrip()


def _prompt_version() -> str:
    import os
    return os.environ.get("LAYOUTCODER_ANDROID_PROMPT", "v2")


def build_atomic_prompt(context: str = "") -> str:
    """Build the deterministic local-generation prompt."""

    base = ANDROID_XML_LOCAL_PROMPT if _prompt_version() == "v1" else ANDROID_XML_LOCAL_PROMPT_V2
    if not context.strip():
        return base
    return f"{base}\n\nAdditional screenshot context:\n{context.strip()}"
