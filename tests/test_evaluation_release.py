"""Offline contract tests; no paid models, screenshots, emulators, or keys."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import release_paths
import frozen_statistics as stats
import human_statistics as human
from ablation_variants import remove_raster_references
from complexity_groups import assign_bins
from main_experiment import endpoints, plan
from collect_renders import receipt_rows


def rows_for(arms, count=9):
    rows = []
    for index in range(count):
        for arm in arms:
            full = arm in ("ours", "full")
            value = 0.9 - index / 100 if full else 0.6 - index / 100
            rows.append({"screen_id": f"screen_{index}", "split": ("Easy", "Real", "Unseen")[index % 3],
                "arm": arm, "render": 1, **{m: value for m in stats.METRICS if m != "mae"}, "mae": 5 if full else 30})
    return rows


def test_complete_fixed_cohort():
    rows = rows_for(("ours", *stats.BASELINES))
    _, ids, arms = stats.validate_rows(rows, expected_count=9)
    assert len(ids) == 9 and len(arms) == 6


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "wrong_split", "unexpected", "boolean"])
def test_invalid_rows_are_not_silently_scored(change):
    rows = rows_for(("ours", *stats.BASELINES))
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    elif change == "nan":
        rows[0]["clip"] = float("nan")
    elif change == "wrong_split":
        rows[0]["split"] = "Unknown"
    elif change == "unexpected":
        rows[0]["arm"] = "unexpected"
    else:
        rows[0]["clip"] = True
    with pytest.raises(ValueError):
        stats.validate_rows(rows, arms=("ours", *stats.BASELINES), expected_count=9)


def test_lower_is_better_and_bootstrap():
    result = stats.paired_stats(np.array([1, 2, 3]), np.array([5, 6, 7]), "mae", reps=100)
    assert result["mean_gain"] == 4
    assert result["cliffs_delta"] == 1
    assert result["ci95"] == [4, 4]


def test_holm_preserves_input_order():
    assert stats.holm([0.04, 0.001, 0.03]) == [0.06, 0.003, 0.06]


def test_rq2_seven_metrics_direction_and_exact_ties():
    rows = rows_for(("full", *stats.ABLATIONS))
    for row in rows:
        if row["arm"] == "no_planning":
            full = next(r for r in rows if r["arm"] == "full" and r["screen_id"] == row["screen_id"])
            row["clip"] = full["clip"] + 1e-8
            row["ssim"] = full["ssim"]
    cells = stats.rq2(rows, expected_count=9, reps=100)
    assert len(cells) == 140
    clip = next(r for r in cells if r["scope"] == "All" and r["arm"] == "no_planning" and r["metric"] == "clip")
    assert clip["direction"] == "up" and clip["drop"] < 0
    gate = stats.rq2_acceptance(cells)
    assert not gate["All/no_planning"]["at_most_two_exact_ties_no_rises"]
    assert gate["All/no_planning"]["equal_mean"] == 1
    assert gate["All/no_review"]["strict_seven_down"]


def test_rq2_preserves_input_cohort_order_for_bootstrap():
    rows = rows_for(("full", *stats.ABLATIONS))
    default = stats.rq2(rows, expected_count=9, reps=100)
    explicit = stats.rq2(rows, expected_count=9, reps=100,
                         ids=[r["screen_id"] for r in rows if r["arm"] == "full"])
    assert default == explicit


def test_rq2_requires_successful_capture():
    rows = rows_for(("full", *stats.ABLATIONS))
    rows[0]["render"] = 0
    with pytest.raises(ValueError):
        stats.rq2(rows, expected_count=9, reps=100)


def test_rq3_rejects_dropped_bin_members():
    rows = rows_for(("ours", *stats.BASELINES))
    bins = {f"screen_{i}": {p: ("low", "mid", "high")[i % 3] for p in ("elements", "text_tokens", "density", "aspect")} for i in range(8)}
    with pytest.raises(ValueError):
        stats.rq3(rows, bins, expected_count=9)


def test_quantile_ties_use_lower_stratum():
    result = assign_bins({f"s{i}": {p: 1 for p in ("elements", "text_tokens", "density", "aspect")} for i in range(9)})
    assert all(all(level == "low" for level in values.values()) for values in result["bins"].values())
    assert "frozen_before_scoring" not in result  # A new execution cannot assert historical timing.


def test_human_data_is_ui60_code60():
    data = json.loads((ROOT / "data/human_study/ratings.json").read_text())
    values = human.validate(data)
    assert len(data["records"]) == 2160
    assert len(data["items"]["v"]) == len(data["items"]["c"]) == 60
    assert len(values) == 6 * 60 * 3 * 12


@pytest.mark.parametrize("change", ["missing", "duplicate", "range", "ranking"])
def test_human_incomplete_or_modified_contracts_fail(change):
    data = json.loads((ROOT / "data/human_study/ratings.json").read_text())
    if change == "missing":
        data["records"].pop()
    elif change == "duplicate":
        data["records"].append(copy.deepcopy(data["records"][0]))
    elif change == "range":
        data["records"][0]["values"]["rank"] = 4
    else:
        first = data["records"][0]
        for row in data["records"]:
            if all(row[k] == first[k] for k in ("part", "rater", "item_id")):
                row["values"]["rank"] = 1
    with pytest.raises(ValueError):
        human.validate(data)


def test_raster_ablation_changes_no_native_geometry_or_text():
    xml = '<FrameLayout xmlns:android="http://schemas.android.com/apk/res/android" android:layout_width="match_parent"><ImageView android:src="@drawable/photo"/><TextView android:text="Keep caption" android:background="@drawable/rounded_native"/></FrameLayout>'
    result, names = remove_raster_references(xml, {"photo"})
    assert names == ["photo"]
    assert "@drawable/img" in result and "@drawable/rounded_native" in result and "Keep caption" in result


def test_raster_ablation_rejects_text_replacement():
    xml = '<TextView xmlns:android="http://schemas.android.com/apk/res/android" android:text="@drawable/photo"/>'
    with pytest.raises(ValueError):
        remove_raster_references(xml, {"photo"})


def test_pipeline_endpoint_names_cannot_collide_between_runs(tmp_path):
    left, right = endpoints(tmp_path / "a/run"), endpoints(tmp_path / "b/run")
    assert all(left[name].name != right[name].name for name in left)


def test_receipts_accept_serial_and_batch_without_duplicate_rows(tmp_path):
    derived = tmp_path / "derived"
    derived.mkdir()
    first = {"screen_id": "one", "status": "render_success"}
    second = {"screen_id": "two", "status": "render_success"}
    (derived / "loading_status.jsonl").write_text(json.dumps(first) + "\n")
    (derived / "loading_status_worker.jsonl").write_text(json.dumps(first) + "\n" + json.dumps(second) + "\n")
    assert list(receipt_rows(tmp_path)) == [first, second]


def test_pipeline_plan_has_guard_and_final_capture(tmp_path):
    steps = plan("no_review", tmp_path / "source", tmp_path / "cohort", tmp_path / "result", 1)
    assert len(steps) == 7
    assert steps[0][0] == "prepare" and steps[-1][0] == "final_capture"
    assert any("--apply" in command and "--fallback" in command for _, command in steps)
    assert all("model_call_scheduler.py" not in " ".join(command) for _, command in steps)


@pytest.mark.parametrize("script", ["evaluation/reproduce.py", "evaluation/verify_reproduction.py",
    "evaluation/screenshot_metrics.py", "evaluation/precompute_metrics.py", "evaluation/complexity_groups.py",
    "evaluation/clip_similarity.py", "evaluation/design2code_block_metrics.py", "evaluation/make_plots.py",
    "experiments/ablation_variants.py", "experiments/prepare_ablation.py", "experiments/main_experiment.py",
    "experiments/model_call_scheduler.py", "experiments/collect_renders.py"])
def test_cli_help_runs_without_external_actions(script):
    result = subprocess.run([sys.executable, str(ROOT / script), "--help"], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()
