"""Statistics for the released, fixed-cohort Screen2Run score rows.

This module has no renderer, OCR, network, model, or private-path dependency.
It preserves the paper's distinct RQ1, RQ2, and RQ3 multiplicity families.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np
from scipy.stats import wilcoxon

PRIMARY = ("block_match", "text_similarity", "position_similarity", "color_similarity", "clip")
METRICS = PRIMARY + ("ssim", "mae")
SCOPES = ("Easy", "Real", "Unseen", "All")
BASELINES = ("direct", "cot", "self_refine", "dcgen", "layoutcoder")
ABLATIONS = ("no_planning", "no_review", "no_realization", "no_raster_output", "no_visual")


def validate_rows(rows, *, arms=None, expected_count=None, require_render=False):
    """Require the same complete cohort for every method; never intersect rows."""
    if not isinstance(rows, list) or not rows:
        raise ValueError("Score rows must be a nonempty JSON list")
    indexed, splits = {}, {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each score row must be an object")
        sid, arm, split = row.get("screen_id"), row.get("arm"), row.get("split")
        if not isinstance(sid, str) or not sid or not isinstance(arm, str) or not arm:
            raise ValueError("Every row requires a screen_id and arm")
        if split not in SCOPES[:-1] or (sid in splits and splits[sid] != split):
            raise ValueError("Missing, unknown, or inconsistent dataset split")
        splits[sid] = split
        if (arm, sid) in indexed:
            raise ValueError(f"Duplicate method/screen row: {arm}/{sid}")
        for metric in METRICS + ("render",):
            value = row.get(metric)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Missing/nonfinite score: {arm}/{sid}/{metric}")
            lo, hi = ((0, 255) if metric == "mae" else ((-1, 1) if metric in ("clip", "ssim") else (0, 1)))
            if not lo - 1e-12 <= value <= hi + 1e-12:
                raise ValueError(f"Out-of-range score: {arm}/{sid}/{metric}")
        if row["render"] not in (0, 1) or (require_render and (row["render"] != 1 or row.get("error"))):
            raise ValueError(f"Unresolved capture/scoring row: {arm}/{sid}")
        indexed[arm, sid] = row
    methods = tuple(arms) if arms is not None else tuple(dict.fromkeys(r["arm"] for r in rows))
    if len(set(methods)) != len(methods) or len(methods) < 2:
        raise ValueError("Require distinct focal and comparator methods")
    ids = sorted(splits)
    if expected_count is not None and len(ids) != expected_count:
        raise ValueError(f"Expected {expected_count} screens, received {len(ids)}")
    if set(indexed) != {(arm, sid) for arm in methods for sid in ids}:
        raise ValueError("Incomplete or extra fixed-cohort method/screen rows")
    return indexed, ids, methods


def holm(pvalues):
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in pvalues):
        raise ValueError("Invalid p-value in Holm family")
    result = [0.0] * len(pvalues)
    running = 0.0
    for rank, index in enumerate(sorted(range(len(pvalues)), key=pvalues.__getitem__)):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        result[index] = running
    return result


def cliffs_delta(a, b):
    """Distributional all-cross-pair Cliff's delta, not a paired effect size."""
    ordered = np.sort(b)
    greater = sum(int(v) for v in np.searchsorted(ordered, a, side="left"))
    less = sum(len(b) - int(v) for v in np.searchsorted(ordered, a, side="right"))
    return (greater - less) / (len(a) * len(b))


def bootstrap_ci(difference, reps=2000, seed=7):
    if reps < 100 or not len(difference):
        raise ValueError("Bootstrap requires observations and at least 100 replicates")
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(difference, size=len(difference), replace=True).mean()) for _ in range(reps)]
    return [float(v) for v in np.percentile(means, [2.5, 97.5])]


def paired_stats(focal, baseline, metric, *, with_ci=True, reps=2000, seed=7):
    a, b = np.asarray(focal, dtype=float), np.asarray(baseline, dtype=float)
    difference = b - a if metric == "mae" else a - b
    p = float(wilcoxon(a, b, alternative="two-sided", method="auto").pvalue) if np.any(a != b) else 1.0
    if not math.isfinite(p):
        raise ValueError("Wilcoxon returned a nonfinite p-value")
    result = {"mean_focal": float(a.mean()), "mean_baseline": float(b.mean()),
              "mean_gain": float(difference.mean()), "p": p,
              "cliffs_delta": cliffs_delta(-a if metric == "mae" else a, -b if metric == "mae" else b)}
    if with_ci:
        result["ci95"] = bootstrap_ci(difference, reps, seed)
    return result


def rq1(rows, *, focal="ours", expected_count=600, reps=2000):
    by, ids, arms = validate_rows(rows, expected_count=expected_count)
    if focal not in arms:
        raise ValueError("Focal arm absent")
    table, statistics = {}, {}
    for scope in SCOPES:
        selected = [sid for sid in ids if scope == "All" or by[focal, sid]["split"] == scope]
        if not selected:
            continue
        # Table means retain the released row order, as in the paper builder.
        table[scope] = {}
        for arm in arms:
            values = [r for r in rows if r["arm"] == arm and (scope == "All" or r["split"] == scope)]
            table[scope][arm] = {**{m: float(np.mean([r[m] for r in values])) for m in METRICS + ("render",)},
                                 "n": len(values)}
        entries = []
        for baseline in arms:
            if baseline == focal:
                continue
            for metric in METRICS:
                entries.append({"baseline": baseline, "metric": metric,
                    **paired_stats([by[focal, s][metric] for s in selected],
                                   [by[baseline, s][metric] for s in selected], metric, reps=reps)})
        for metric in METRICS:
            family = [entry for entry in entries if entry["metric"] == metric]
            for entry, p in zip(family, holm([entry["p"] for entry in family])):
                entry["p_holm"] = p
        statistics[scope] = entries
    return {"table": table, "stats": statistics, "focal": focal}


def rq2(rows, *, full="full", expected_count=600, reps=2000, seed=20260930, ids=None):
    """Return 140 seven-metric ablation cells, preserving actual direction."""
    by, sorted_ids, arms = validate_rows(rows, arms=(full, *ABLATIONS), expected_count=expected_count,
                                        require_render=True)
    ids = [row["screen_id"] for row in rows if row["arm"] == full] if ids is None else list(ids)
    if len(ids) != len(set(ids)) or set(ids) != set(sorted_ids):
        raise ValueError("RQ2 ordered cohort differs from scored rows")
    cells = []
    for scope in ("All", *SCOPES[:-1]):
        selected = [sid for sid in ids if scope == "All" or by[full, sid]["split"] == scope]
        if not selected:
            continue
        for arm in ABLATIONS:
            family = []
            for index, metric in enumerate(METRICS):
                f = np.array([by[full, s][metric] for s in selected], dtype=float)
                a = np.array([by[arm, s][metric] for s in selected], dtype=float)
                difference = a - f if metric == "mae" else f - a
                mean = float(math.fsum(float(d) for d in difference) / len(difference))
                # RQ2's historical test accepts the oriented differences directly.
                p = float(wilcoxon(difference, alternative="two-sided", method="auto").pvalue) if np.any(difference != 0) else 1.0
                delta = float(np.mean(f[:, None] > a) - np.mean(f[:, None] < a))
                if metric == "mae":
                    delta = -delta
                family.append({"scope": scope, "arm": arm, "metric": metric,
                    "full": float(f.mean()), "ablated": float(a.mean()), "drop": mean,
                    "direction": "down" if mean > 0 else "up" if mean < 0 else "equal_mean",
                    "within_legacy_display_band": abs(mean) < (0.2 if metric == "mae" else 0.002),
                    "identical_pairs": bool(np.all(difference == 0)),
                    "screen_down_equal_up": [int(np.sum(difference > 0)), int(np.sum(difference == 0)), int(np.sum(difference < 0))],
                    "p": p, "paired_mean_ci95": bootstrap_ci(difference, reps, seed + index),
                    "cliffs_delta_distributional": delta})
            for entry, p in zip(family, holm([entry["p"] for entry in family])):
                entry["p_holm"] = p
            cells.extend(family)
    return cells


def rq2_acceptance(cells):
    report = {}
    for scope in dict.fromkeys(row["scope"] for row in cells):
        for arm in ABLATIONS:
            family = [row for row in cells if row["scope"] == scope and row["arm"] == arm]
            if {row["metric"] for row in family} != set(METRICS) or len(family) != 7:
                raise ValueError("RQ2 acceptance requires all seven metrics per scope/variant")
            count = Counter(row["direction"] for row in family)
            report[f"{scope}/{arm}"] = {"down": count["down"], "equal_mean": count["equal_mean"],
                "up": count["up"], "strict_seven_down": count["down"] == 7,
                "at_most_two_exact_ties_no_rises": count["down"] >= 5 and count["equal_mean"] <= 2 and count["up"] == 0}
    return report


def rq2_paper(cells):
    """Current paper presentation: separate primary-five and pixel-two families."""
    indexed = {(r["arm"], r["scope"], r["metric"]): r for r in cells}
    output = {}
    for arm in ABLATIONS:
        for scope in SCOPES:
            for family in (PRIMARY, ("ssim", "mae")):
                values = [indexed[arm, scope, metric] for metric in family]
                for metric, row, p in zip(family, values, holm([row["p"] for row in values])):
                    output[f"{arm}/{scope}/{metric}"] = {"full": row["full"], "ablated": row["ablated"],
                        "change": row["ablated"] - row["full"], "p_holm5": p, "ci95_drop": row["paired_mean_ci95"]}
    return output


def rq3(rows, bins, *, expected_count=600, metrics=PRIMARY):
    """Paper RQ3: mean-best comparator; Holm across five metrics per bin."""
    by, ids, _ = validate_rows(rows, arms=("ours", *BASELINES), expected_count=expected_count)
    bins = bins.get("bins", bins)
    if set(bins) != set(ids):
        raise ValueError("Robustness bins differ from the fixed scored cohort")
    cells = {}
    for prop in ("elements", "text_tokens", "density", "aspect"):
        if any(bins[s].get(prop) not in ("low", "mid", "high") for s in ids):
            raise ValueError(f"Invalid robustness level for {prop}")
        for level in ("low", "mid", "high"):
            selected = [s for s in ids if bins[s][prop] == level]
            if not selected:
                raise ValueError(f"Empty robustness bin: {prop}/{level}")
            entries = {}
            for metric in metrics:
                means = {a: float(np.mean([by[a, s][metric] for s in selected])) for a in sorted(BASELINES)}
                best = (min if metric == "mae" else max)(means, key=means.get)
                stats = paired_stats([by["ours", s][metric] for s in selected],
                                     [by[best, s][metric] for s in selected], metric, with_ci=False)
                entries[metric] = {"best_baseline": best, "focal": stats["mean_focal"], "baseline": means[best],
                    "gain": stats["mean_gain"], "p": stats["p"], "cliffs_delta": stats["cliffs_delta"]}
            for metric, p in zip(metrics, holm([entries[m]["p"] for m in metrics])):
                entries[metric]["p_holm"] = p
            cells[f"{prop}/{level}"] = {"n": len(selected), "metrics": entries}
    return cells
