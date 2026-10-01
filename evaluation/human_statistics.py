"""Reproduce the six-rater UI60/code60 study from anonymized numeric records."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from frozen_statistics import cliffs_delta, holm

VISUAL = ("overall", "layout", "text", "image", "style")
CODE = ("readability", "maintainability", "practice", "usability")
OUTCOMES = [("v", d) for d in VISUAL] + [("v", "replace"), ("v", "rank")] + [("c", d) for d in CODE] + [("c", "rank")]


def validate(data, *, expected_items=60, expected_raters=6):
    if data.get("schema") != "screen2run-human-ratings/1":
        raise ValueError("Unknown human ratings schema")
    raters, arms, items = data["raters"], data["methods"], data["items"]
    if len(raters) != expected_raters or len(set(raters)) != expected_raters:
        raise ValueError("Rater cohort is incomplete or duplicated")
    if set(arms) != {"ours", "dcgen", "layoutcoder"} or len(arms) != 3:
        raise ValueError("Human study requires exactly the three declared methods")
    for part in ("v", "c"):
        if len(items[part]) != expected_items or len(set(items[part])) != expected_items:
            raise ValueError(f"Incomplete or duplicate {part} item cohort")
    val = {}
    for row in data["records"]:
        part, rater, item, arm = (row.get(k) for k in ("part", "rater", "item_id", "method"))
        if part not in items or rater not in raters or item not in items[part] or arm not in arms:
            raise ValueError("Unexpected human-study record")
        required = set(VISUAL + ("replace", "rank")) if part == "v" else set(CODE + ("rank",))
        if set(row["values"]) != required:
            raise ValueError(f"Missing or extra rating dimension: {part}/{rater}/{item}/{arm}")
        for dim, value in row["values"].items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("Rating must be a finite number")
            allowed = (0, 1) if dim == "replace" else ((1, 2, 3) if dim == "rank" else (1, 2, 3, 4, 5))
            if value not in allowed:
                raise ValueError("Rating outside its declared scale")
            key = (part, rater, item, arm, dim)
            if key in val:
                raise ValueError("Duplicate human-study rating")
            val[key] = float(value)
    expected = {(part, r, item, arm, dim) for part, dim in OUTCOMES for r in raters for item in items[part] for arm in arms}
    if set(val) != expected:
        raise ValueError("Human-study rating matrix is incomplete")
    for part in ("v", "c"):
        for r in raters:
            for item in items[part]:
                if sorted(val[part, r, item, a, "rank"] for a in arms) != [1, 2, 3]:
                    raise ValueError("Method rankings must be a permutation of 1, 2, 3")
    return val


def grid(val, part, dim, items, raters, arm):
    return np.array([[val[part, r, item, arm, dim] for item in items] for r in raters], dtype=float)


def rank_biserial(difference):
    difference = difference[difference != 0]
    if not len(difference):
        return 0.0
    ranks = rankdata(np.abs(difference))
    return float((ranks[difference > 0].sum() - ranks[difference < 0].sum()) / ranks.sum())


def compare(val, part, dim, items, raters, arms, focal, rng, bootstraps):
    sign = -1.0 if dim == "rank" else 1.0
    matrices = {a: grid(val, part, dim, items, raters, a) for a in arms}
    means = {a: matrices[a].mean(axis=0) for a in arms}
    result = {"n_items": len(items), "n_raters": len(raters),
              "mean": {a: float(means[a].mean()) for a in arms}, "vs": {}}
    if len(items) >= 5:
        statistic, p = friedmanchisquare(*[means[a] for a in arms])
        # A completely identical matrix has no rank dispersion.
        result["friedman"] = {"chi2": float(statistic) if math.isfinite(statistic) else 0.0,
                              "p": float(p) if math.isfinite(p) else 1.0}
    for baseline in (a for a in arms if a != focal):
        difference = sign * (means[focal] - means[baseline])
        comparison = {"diff": float(difference.mean()),
            "p": float(wilcoxon(means[focal], means[baseline]).pvalue) if np.any(difference != 0) else 1.0,
            "rank_biserial": rank_biserial(difference),
            "cliffs_delta": cliffs_delta(sign * means[focal], sign * means[baseline])}
        delta = sign * (matrices[focal] - matrices[baseline])
        nr, ni = delta.shape
        draws = [np.mean(delta[np.ix_(rng.integers(0, nr, nr), rng.integers(0, ni, ni))]) for _ in range(bootstraps)]
        comparison["ci95_two_way"] = [float(v) for v in np.percentile(draws, [2.5, 97.5])]
        result["vs"][baseline] = comparison
    return result


def analyze(data, *, expected_items=60, expected_raters=6, bootstraps=5000):
    if bootstraps < 100:
        raise ValueError("At least 100 two-way bootstrap replicates are required")
    val = validate(data, expected_items=expected_items, expected_raters=expected_raters)
    rng = np.random.default_rng(20260930)
    results, family = {"primary": {}, "secondary": {}}, []
    for part, dim in OUTCOMES:
        slot = "primary" if (part, dim) in (("v", "overall"), ("v", "rank")) else "secondary"
        outcome = f"{part}:{dim}"
        results[slot][outcome] = compare(val, part, dim, data["items"][part], data["raters"], data["methods"],
                                        "ours", rng, bootstraps)
        family += [(slot, outcome, a) for a in data["methods"] if a != "ours"]
    for (slot, outcome, arm), p in zip(family, holm([results[s][o]["vs"][a]["p"] for s, o, a in family])):
        results[slot][outcome]["vs"][arm]["p_holm"] = p
    return {"schema": "screen2run-human-statistics/1", "n_raters": len(data["raters"]),
            "n_ui_items": len(data["items"]["v"]), "n_code_items": len(data["items"]["c"]),
            "bootstrap_reps": bootstraps, "bootstrap_seed": 20260930,
            "holm_family": "all 24 prespecified comparisons", **results}
