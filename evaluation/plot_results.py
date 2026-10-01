"""Result figures, drawn in the formats used by DCGen (FSE'25) and LayoutCoder (ISSTA'25).

RQ1: box plots of per-screen scores (LayoutCoder Fig. 7).
RQ3: mean score per complexity level, one line per method (LayoutCoder Fig. 8).
RQ4: distribution of expert ratings as diverging stacked bars (DCGen Fig. 9).
"""
from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt

import plot_style as S

PRIMARY = ("block_match", "text_similarity", "position_similarity", "color_similarity", "clip")
LABELS = {"block_match": "Block-Match", "text_similarity": "Text", "position_similarity": "Position",
          "color_similarity": "Color", "clip": "CLIP"}
METHODS = ("direct", "cot", "self_refine", "dcgen", "layoutcoder", "ours")
NAMES = {"direct": "Direct", "cot": "CoT", "self_refine": "Self-Refine", "dcgen": "DCGen",
         "layoutcoder": "LayoutCoder", "ours": "Screen2Run"}
COLORS = {"direct": "#C9CED4", "cot": "#AEB5BD", "self_refine": "#8F98A3",
          "dcgen": S.DCGEN, "layoutcoder": S.LAYOUTCODER, "ours": S.OURS}


# --------------------------------------------------------------------------- RQ1

def rq1_box(rows: list[dict], path) -> dict:
    by = {(r["arm"], r["screen_id"]): r for r in rows}
    ids = sorted({r["screen_id"] for r in rows})
    fig = plt.figure(figsize=(S.TEXT_WIDTH, 2.0))
    left, right, bottom, top, gap = .058, .995, .045, .80, .062
    w = (right - left - 4 * gap) / 5
    summary = {}
    for k, m in enumerate(PRIMARY):
        ax = fig.add_axes([left + k * (w + gap), bottom, w, top - bottom])
        data = [np.array([by[(a, s)][m] for s in ids]) for a in METHODS]
        lows = []
        for x, (a, v) in enumerate(zip(METHODS, data)):
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            iqr = q3 - q1
            lo = v[v >= q1 - 1.5 * iqr].min()
            hi = v[v <= q3 + 1.5 * iqr].max()
            lows.append(lo)
            c = COLORS[a]
            ax.plot([x, x], [lo, q1], color=c, linewidth=.7, solid_capstyle="butt")
            ax.plot([x, x], [q3, hi], color=c, linewidth=.7, solid_capstyle="butt")
            ax.add_patch(plt.Rectangle((x - .32, q1), .64, max(q3 - q1, 1e-3), facecolor=c,
                                       alpha=.95 if a == "ours" else .85, edgecolor="none", zorder=2))
            ax.plot([x - .32, x + .32], [med, med], color="white", linewidth=1.0, zorder=3,
                    solid_capstyle="butt")
            ax.plot(x, v.mean(), "o", markersize=2.2, markerfacecolor="white", markeredgecolor=S.INK,
                    markeredgewidth=.5, zorder=4)
            summary[f"{m}/{a}"] = {"q1": float(q1), "median": float(med), "q3": float(q3), "mean": float(v.mean())}
        floor = max(0.0, np.floor((min(lows) - .03) * 10) / 10)
        ax.set_ylim(floor, 1.015)
        ax.set_xlim(-.6, len(METHODS) - .4)
        ax.set_xticks([])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax.spines["bottom"].set_visible(False)
        ax.yaxis.grid(True, color=S.GRID, linewidth=.5)
        ax.set_axisbelow(True)
        ax.set_title(LABELS[m], loc="left", fontsize=7, fontweight="bold", pad=3)
    handles = [Patch(facecolor=COLORS[a], edgecolor="none", label=NAMES[a]) for a in METHODS]
    handles.append(Line2D([], [], marker="o", linestyle="none", markersize=2.6, markerfacecolor="white",
                          markeredgecolor=S.INK, markeredgewidth=.5, label="Mean"))
    fig.legend(handles=handles, loc="upper center", ncol=7, bbox_to_anchor=(.5, 1.0), handlelength=.9,
               handletextpad=.4, columnspacing=1.0)
    S.save(fig, path)
    return summary


# --------------------------------------------------------------------------- RQ3

PROPS = (("elements", "Element count"), ("text_tokens", "Text tokens"), ("density", "Element density"),
         ("aspect", "Aspect ratio"))
LEVELS = (("low", "Low"), ("mid", "Mid"), ("high", "High"))
RQ3_METRICS = ("block_match", "clip")
RQ3_METHODS = (("dcgen", "o"), ("layoutcoder", "s"), ("ours", "D"))


def rq3_lines(rows: list[dict], bins: dict, path) -> dict:
    by = {(r["arm"], r["screen_id"]): r for r in rows}
    fig = plt.figure(figsize=(S.TEXT_WIDTH, 2.55))
    left, right, bottom, top, hgap, vgap = .085, .995, .085, .83, .025, .075
    w = (right - left - 3 * hgap) / 4
    h = (top - bottom - vgap) / 2
    out = {}
    for r, m in enumerate(RQ3_METRICS):
        axes = []
        for c, (prop, ptitle) in enumerate(PROPS):
            ax = fig.add_axes([left + c * (w + hgap), top - (r + 1) * h - r * vgap, w, h])
            axes.append(ax)
            for arm, marker in RQ3_METHODS:
                means, half = [], []
                for lv, _ in LEVELS:
                    v = np.array([by[(arm, s)][m] for s in bins if bins[s][prop] == lv])
                    means.append(v.mean())
                    half.append(1.96 * v.std(ddof=1) / np.sqrt(len(v)))
                    out[f"{m}/{prop}/{lv}/{arm}"] = {"mean": float(v.mean()), "n": len(v)}
                xs = np.arange(3)
                means, half = np.array(means), np.array(half)
                color = S.METHOD_COLORS[arm]
                ax.fill_between(xs, means - half, means + half, color=color, alpha=.13, linewidth=0)
                ax.plot(xs, means, color=color, linewidth=1.1 if arm == "ours" else .9, marker=marker,
                        markersize=3.0 if marker != "D" else 2.7, markeredgewidth=0, zorder=3)
            ax.set_xlim(-.25, 2.25)
            ax.set_xticks(range(3), [l for _, l in LEVELS] if r == 1 else [])
            ax.yaxis.grid(True, color=S.GRID, linewidth=.5)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(ptitle, loc="left", fontsize=7, fontweight="bold", pad=3)
            if c == 0:
                ax.set_ylabel(LABELS[m])
            else:
                ax.set_yticklabels([])
        lo = min(a.get_ylim()[0] for a in axes)
        hi = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(lo, hi)
    handles = [Line2D([], [], color=S.METHOD_COLORS[a], marker=mk, markersize=3.0, markeredgewidth=0,
                      linewidth=1.0, label=NAMES[a]) for a, mk in RQ3_METHODS]
    fig.legend(handles=handles, loc="upper right", ncol=3, bbox_to_anchor=(right, 1.0))
    S.save(fig, path)
    return out


# --------------------------------------------------------------------------- RQ4

LIKERT = ("#B5533E", "#E2A592", "#DADDE1", "#9AABD0", S.OURS)  # 1 .. 5
LIKERT_NAMES = ("1  Very poor", "2", "3", "4", "5  Very good")
RQ4_ARMS = ("dcgen", "layoutcoder", "ours")


def rq4_likert(raw: dict, groups: list[tuple[str, str, list[tuple[str, str]]]], path) -> dict:
    """raw[part][dim][arm] -> list of integer ratings over all screens and raters.

    Diverging stacked bars centred on the neutral rating, one group of three bars per dimension.
    """
    fig = plt.figure(figsize=(S.TEXT_WIDTH, 3.15))
    bottom, top = .135, .905
    panels = [(.125, .345), (.635, .34)]
    gap, head = 1.55, 1.05  # vertical units between groups / header above the first bar
    rows = max(len(dims) for _, _, dims in groups)
    span = rows * (len(RQ4_ARMS) + gap)
    out = {}
    for (x0, pw), (part, title, dims) in zip(panels, groups):
        ax = fig.add_axes([x0, bottom, pw, top - bottom])
        y, yt, yl = head, [], []
        for dim, dlabel in dims:
            ax.text(-1.0, y - head, dlabel, ha="left", va="center", fontsize=6.5, fontweight="bold", color=S.INK)
            for arm in RQ4_ARMS:
                v = np.rint(np.array(raw[part][dim][arm])).astype(int)
                share = np.array([(v == k).mean() for k in range(1, 6)])
                out[f"{part}:{dim}/{arm}"] = share.round(4).tolist()
                start = -(share[0] + share[1] + share[2] / 2)
                for k in range(5):
                    if share[k] > 0:
                        ax.barh(y, share[k], left=start, height=.68, color=LIKERT[k], edgecolor="white",
                                linewidth=.3)
                    start += share[k]
                yt.append(y)
                yl.append(NAMES[arm])
                y += 1
            y += gap
        ax.axvline(0, color=S.MUTED, linewidth=.5, zorder=3)
        ax.set_ylim(span - gap + .6, -.1)
        ax.set_yticks(yt, yl, fontsize=6.3)
        for lab in ax.get_yticklabels():
            if lab.get_text() == NAMES["ours"]:
                lab.set_fontweight("bold")
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -.5, 0, .5, 1], ["100%", "50%", "0", "50%", "100%"])
        ax.spines["left"].set_visible(False)
        ax.xaxis.grid(True, color=S.GRID, linewidth=.5)
        ax.set_axisbelow(True)
        S.panel_label(fig, ax, part, title, x_offset=-.11)
    handles = [Patch(facecolor=LIKERT[k], edgecolor="none", label=LIKERT_NAMES[k]) for k in range(5)]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(.5, .0), handlelength=1.4,
               columnspacing=1.6)
    S.save(fig, path)
    return out
