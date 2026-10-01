# Evaluation and experiment reproduction

## 1. Recompute all published numeric results without models

Run from the release root:

```bash
python evaluation/reproduce.py all --data-dir data --out work/reproduced --plots
python evaluation/verify_reproduction.py --data-dir data --out work/reproduced
python -m pytest tests/test_evaluation_release.py -q
```

This route reads numeric observations, never calls a model, never launches an
emulator, never reads a credential, and does not need the reference images. It
produces RQ1 and backbone summaries, both RQ2 reporting families, RQ3 summaries,
RQ4 statistics, and the RQ1/RQ3/RQ4 figures in PDF and PNG. Verification compares
the outputs with the released expected evidence and checks that RQ2's Full
control is exactly the RQ1 Full scalar data. It fails on a changed result rather
than rewriting the expected values.

The released data contain:

- `data/results/rq1_rows.json`: 600 screens, six methods, 3,600 rows.
- `data/results/backbone_*_rows.json`: three backbone comparisons, each with
  the same 120-screen subset and six methods, 720 rows per comparison.
- `data/results/rq2_rows.json`: Full and five redesigned ablations, each with
  600 screens, 3,600 rows. These are the completed redesigned variants, not the
  earlier exploratory `no_s1`/`no_s45` result tables.
- `data/results/rq3_bins.json`: the original reference-only complexity groups.
- `data/human_study/ratings.json`: six raters, 60 interface items and 60 code
  items, three methods, 2,160 method-level records. Raw contact information,
  browser metadata, timestamps, and private source paths are not included.

The human record schema preserves item, rater, and method ordering because the
two-way bootstrap uses that ordering. `values` contains the five interface
ratings plus `replace` and `rank`, or the four code ratings plus `rank`. Code
quality is scored from all 60 items; the preserved initial 12 and the additional
48 are not counted as different raters. Collection-round labels remain available
for provenance, while analysis uses the one complete 60-item cohort.

### Tested lightweight environment

The release was verified using CPython 3.13.5, NumPy 2.2.6, SciPy 1.16.2,
Matplotlib 3.11.0, Pillow 12.0.0, and pytest 9.1.1. Only NumPy/SciPy are needed
for numeric statistics. Plotting additionally needs Matplotlib; Pillow is used
by the broader evaluation/runtime modules. Python 3.11 or newer is required by
the pinned dependency set.
`--help` does not load OCR or CLIP weights and performs no external action.

## 2. Metrics and statistical definitions

| Quantity | Definition and direction |
|---|---|
| Block-Match | The pinned DCGen/Design2Code block matching computation, using screenshot-extracted Android text blocks; larger is better. |
| Text | Character-sequence similarity of matched blocks; larger is better. |
| Position | Matched block-center agreement in normalized screenshot coordinates; larger is better. |
| Color | Foreground-color similarity based on CIEDE2000; larger is better. |
| CLIP | OpenAI ViT-B/32 image-image cosine under the actual experiment preprocessing; larger is better. |
| SSIM | Grayscale structural similarity with data range 255; larger is better. |
| MAE | Mean absolute RGB difference on the 0--255 scale; smaller is better. |
| Render | Capture availability (0 or 1), a separate deployment observation. |

Raw image scoring blanks the top 4.5% status band identically on both images
using the reference-derived median color below that band. The generated image
is resized to the reference size using bilinear interpolation for CLIP, SSIM,
and MAE. CLIP's canonical processor subsequently resizes/crops for ViT-B/32.
The active experiment scorer calls `score_pair`, not the optional
`score_pair_text_masked` helper. Do not silently substitute text-inpainted CLIP
for the released scores. DOM/CSS block extraction is adapted to screenshot OCR;
matching remains the pinned upstream computation.

The frozen numeric records retain the historical Color-null aggregation values.
Fresh image scoring reports unresolved Color as pending by default. The explicit
`--legacy-color-null-zero` flag reproduces the historical aggregation rule and
records every conversion; it does not turn arbitrary extraction errors into zero.

### Statistical families

- **RQ1 and backbones:** two-sided paired Wilcoxon signed-rank tests; Holm over
  the five baselines within each metric and dataset scope; distributional
  Cliff's delta; 2,000 paired bootstrap replicates, seed 7. Paired calculations
  use sorted screen IDs, while displayed means retain input row order.
- **RQ2 original complete analysis:** seven metrics within each of five
  variant contrasts and four scopes (All/Easy/Real/Unseen), 140 cells. The paired
  difference is Full minus ablated for similarities, and ablated minus Full for
  MAE. Tests use that oriented difference. Bootstrap: 2,000 replicates, seed
  20260930 plus the metric index, in the frozen cohort order. Output:
  `rq2_data.json`.
- **RQ2 current paper table:** the same observations, differences, raw p-values,
  and CIs; Holm is applied separately within the five primary metrics and the
  two pixel-level metrics. Output: `rq2_paper.json`. Its historical field name
  `p_holm5` is retained for table compatibility; on SSIM/MAE it denotes the
  two-metric family. It is not the seven-metric adjustment above.
- **RQ3 current paper:** four reference properties, three strata each, five
  primary metrics. In every cell the highest-mean baseline is selected before
  comparison (lowest mean for MAE in the seven-metric output). Holm is within
  each property/stratum's five primary metrics. `rq3_primary.json` has 60 cells;
  `rq3_seven_metrics.json` preserves the seven-metric, 84-cell analysis with its
  own seven-metric Holm family. Strata and their membership are not recomputed
  when replaying the paper.
- **RQ4:** average the six ratings per item, then compare matched item means.
  Report two-sided Wilcoxon, Holm over all 24 prespecified pairwise outcomes,
  paired rank-biserial correlation, distributional Cliff's delta, Friedman
  tests, and 5,000 two-way rater-by-item bootstrap replicates with seed 20260930.
  Rank is the only lower-is-better human outcome. All other human outcomes are
  oriented higher-is-better.

Missing methods or screens, duplicate rows, nonfinite values, and inconsistent
dataset labels are errors. Neither an insignificant difference nor a small
rounded difference is treated as an exact tie. `rq2_acceptance.json` reports
actual decreases, exact equal means, and increases, including both engineering
criteria (all seven worse; or at most two exact ties and no improvements).

## 3. Score fresh reference/render pairs

Fresh screenshot extraction uses Apple Vision through the included Swift bridge.
This path requires **macOS**, Xcode command-line tools (`swiftc`), and the pinned
DCGen metric source. Numeric-only reproduction above works without Apple Vision.
The tested image environment additionally used scikit-image 0.25.2,
PyTorch 2.9.1, Transformers 4.57.1, Hugging Face Hub 0.36.0, and
opencv-python-headless 4.12.0.88. OpenCV is required by optional text inpainting
and image-localization code, not by the frozen numeric replay.

CLIP uses `openai/clip-vit-base-patch32`, revision
`3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`, from the local Hugging Face cache.
`load_local_model()` sets `local_files_only=True`; it does not silently fetch or
upgrade weights. Obtain the declared public model separately if it is not
cached. The block metric loader requires the upstream `visual_score.py` file
whose SHA256 is
`b2ca62e4bf6c657bd2c2b194a6f223369d6c62cee49528c16f00f454eb30aaeb`;
set `DCGEN_HOME` as described in the baseline instructions. A different upstream
file is rejected rather than substituted.

Create a pair manifest, with paths relative to that JSON file or absolute:

```json
[
  {
    "screen_id": "example",
    "split": "Real",
    "arm": "ours",
    "status": "render_success",
    "reference": "reference.png",
    "render_path": "rendered.png"
  }
]
```

For a documented method failure, use `status: "method_failure"` and a nonempty
`failure_reason`; the fixed-denominator worst scores are 0 (MAE 255). A missing
PNG, unknown status, or scoring exception is not a method-failure declaration.
Optional `reference_sha256` and `render_path_sha256` lock image identity.

```bash
python evaluation/screenshot_metrics.py --pairs pairs.json --out work/scored --cache work/cache/metrics
python evaluation/precompute_metrics.py --pairs pairs.json --cache work/cache/metrics --workers 1
```

The scorer produces `rows.json` only when complete. Otherwise it writes
`partial_rows.json` plus `score_report.json` and exits nonzero. Successful pair
caches remain reusable; failures are never replaced by synthetic successful rows.

`experiments/collect_renders.py --manifest experiment.json --out work/pairs`
constructs such a manifest from successful captures after verifying XML, PNG,
and drawable hashes. Its experiment schema is:

```json
{
  "schema": "screen2run-experiment/1",
  "cohort": [{"screen_id": "example", "split": "Real", "reference": "reference.png"}],
  "methods": {
    "ours": {"candidate": "candidate", "render_dir": "renders/tsc_candidate_full"}
  }
}
```

Paths are relative to the experiment manifest. Receipts lacking the required
capture-time hashes are not retroactively promoted to new verified captures.
The released historical scalar data remain reproducible independently of this
stricter fresh-capture collection interface.

## 4. The five formal RQ2 interventions

| Published variant | Input and intervention | Subsequent stages |
|---|---|---|
| `no_planning` | Generation with S1 and S2 omitted; internal model-session flag `no_s1` means both stages, not S1 alone. | Same S3--S5, realization, and guard. |
| `no_visual` | Generation with only S2 omitted; S1 tree supplies the later design input. | Same S3--S5, realization, and guard. |
| `no_review` | The Full generation's recorded S3 output, with the same deterministic native-surface materialization and compile preparation. | Same realization and guard; no model S4/S5. |
| `no_realization` | The original pre-execution S5 endpoint, including native resources. | Render directly; jointly omit grounding, binding, and associated geometry rewriting. |
| `no_raster_output` | Frozen final Full XML; replace references to actual raster drawable files with the shared placeholder, retaining native XML resources. | Render the nested terminal intervention. |

`prepare_ablation.py` retains the same-run request, output, reference, and
native-resource checks of the finalized shared S3/S5 preparation. It never
borrows another screen/model's output. `ablation_variants.py bindonly` preserves
the original coarse-binding fallback used by the content guard; it is **not**
the standalone `no_realization` experiment.

For each variant, prepare a new output directory and inspect a local plan:

```bash
python experiments/main_experiment.py --variant no_review --source work/generation --cohort data/screen_lists/main_600.txt --out work/rq2/no_review
python experiments/main_experiment.py --variant no_realization --source work/frozen_pre --cohort data/screen_lists/main_600.txt --out work/rq2/no_realization
python experiments/main_experiment.py --variant no_raster_output --source work/frozen_full --cohort data/screen_lists/main_600.txt --out work/rq2/no_raster_output
```

Add `--execute` only after the images, saved stage outputs, Android host, and
already-running emulator are available. The pipeline performs preparation,
pre-capture, original grounding, grounded capture, fallback construction, the
unchanged 0.25 text-recall guard, and final capture. Terminal variants need only
their intervention and capture. It never starts an AVD or calls a model. Every
step requires complete artifacts; zero process exit status alone is insufficient.
Capture collection validates hashes, and pre-capture requires a current runtime
view hierarchy. `progress.json` and per-step logs remain in the new output root.
Candidate names are unique across experiment roots to prevent renderer collisions.
The generation-root argument contains `full/<screen_id>/`; do not pass its
`full/` subdirectory as the root.

To obtain new `no_planning`/`no_visual` generation, use the explicit scheduler.
It is a new experiment, not a byte-identical replay of historic model responses:

```bash
python experiments/model_call_scheduler.py --method no_planning --model MODEL_ID --base-url PROVIDER_URL --cohort data/screen_lists/main_600.txt --out work/generation/no_planning --max-screens 1
```

The default prints commands without calling the provider. `--execute` requires
`OPENAI_API_KEY` in the environment. Run a small smoke test before a full cohort.
No keychain account, private endpoint, automatic route switching, or credentials
are embedded in this artifact. Provider-side spending caps are necessary for a
hard billing limit. Baseline adapters additionally require explicit input/output
token prices and `--max-estimated-cost-usd`; post-call token accounting is not a
provider-enforced hard cap. A screen-count limit is not a dollar-budget guarantee.

The shipped frozen score rows reproduce all published analyses without any of
these paid or Android steps. Regenerating models and rerendering code requires
the upstream datasets and dependencies and may vary with provider responses,
runtime versions, fonts, and OCR versions; that does not authorize changing the
frozen numeric observations.
