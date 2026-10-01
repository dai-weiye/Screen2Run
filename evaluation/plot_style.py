"""Shared figure style for every data figure in the paper.

All figures are drawn at the exact text width of the ACM acmsmall layout
(395.8 pt = 5.48 in) and included without scaling, so the font sizes below are
the printed sizes.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "screen2run-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TEXT_WIDTH = 395.8225 / 72.27  # inches

# One palette for the whole paper.
OURS = "#3C5488"         # Screen2Run
DCGEN = "#D9644A"        # DCGen
LAYOUTCODER = "#2A9D8F"  # LayoutCoder
NEUTRAL = "#9AA3AD"      # other baselines
INK = "#222831"
MUTED = "#5F6873"
RULE = "#C8CDD3"
GRID = "#ECEEF1"
BAND = "#F6F7F9"

METHOD_COLORS = {"ours": OURS, "dcgen": DCGEN, "layoutcoder": LAYOUTCODER}


def apply() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.titleweight": "regular", "axes.titlepad": 4,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.edgecolor": RULE, "axes.linewidth": 0.6,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.2, "ytick.major.size": 2.2, "xtick.major.pad": 2, "ytick.major.pad": 2,
        "text.color": INK, "axes.labelcolor": INK,
        "legend.frameon": False, "legend.handlelength": 1.2, "legend.columnspacing": 1.2,
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.sf": "Arial",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 600, "savefig.facecolor": "white",
    })


def panel_label(fig, ax, letter: str, title: str, x_offset: float = 0.0) -> None:
    """Nature-style panel heading: bold letter followed by a plain title, left-aligned to the axes."""
    bbox = ax.get_position()
    fig.text(bbox.x0 + x_offset, bbox.y1 + 0.035, letter, fontsize=8, fontweight="bold", color=INK,
             va="bottom", ha="left")
    fig.text(bbox.x0 + x_offset + 0.022, bbox.y1 + 0.035, title, fontsize=7, color=INK, va="bottom", ha="left")


def layout_problems(fig) -> list[str]:
    """Text that overlaps other text, runs into another panel or a bar, leaves the canvas, or breaks 5-8 pt."""
    from matplotlib.patches import Rectangle
    from matplotlib.text import Text
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    drawn, hidden = set(), set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                hidden.update((id(tick.label1), id(tick.label2)))
            for tick in axis._update_ticks():
                drawn.update((id(tick.label1), id(tick.label2)))
    texts = [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()
             and (id(t) in drawn or id(t) not in hidden) and t.get_window_extent(r).width > 0]
    boxes = [(t, t.get_window_extent(r).expanded(1.0, 1.0)) for t in texts]
    W, H = fig.bbox.width, fig.bbox.height
    pt = fig.dpi / 72
    out = []
    name = lambda t: repr(t.get_text()[:24])
    for t, b in boxes:
        if not 5 <= t.get_fontsize() <= 8:
            out.append(f"size {t.get_fontsize():.1f}pt {name(t)}")
        if b.x0 < -.5 or b.y0 < -.5 or b.x1 > W + .5 or b.y1 > H + .5:
            out.append(f"outside canvas {name(t)}")
    pad = .4 * pt
    for i, (t1, b1) in enumerate(boxes):
        for t2, b2 in boxes[i + 1:]:
            if (min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > pad and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > pad):
                out.append(f"text overlap {name(t1)} / {name(t2)}")
    for t, b in boxes:
        for ax in fig.axes:
            if ax is t.axes or not ax.get_visible() or not ax.axison:
                continue
            a = ax.bbox
            if min(b.x1, a.x1) - max(b.x0, a.x0) > pad and min(b.y1, a.y1) - max(b.y0, a.y0) > pad:
                out.append(f"text {name(t)} enters another panel")
        if t.axes is not None and t.axes.axison:
            for patch in t.axes.patches:
                if isinstance(patch, Rectangle) and patch.get_visible():
                    pb = patch.get_window_extent(r)
                    if min(b.x1, pb.x1) - max(b.x0, pb.x0) > pad and min(b.y1, pb.y1) - max(b.y0, pb.y0) > pad:
                        out.append(f"text {name(t)} covers a bar")
                        break
    return sorted(set(out))


def save(fig, path_without_suffix) -> None:
    problems = layout_problems(fig)
    if problems:
        raise RuntimeError(f"{path_without_suffix}: " + "; ".join(problems))
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_without_suffix}.{ext}")
    plt.close(fig)
