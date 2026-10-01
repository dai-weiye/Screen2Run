# Runtime and platform setup

## Scope

`screen2run/model_sessions.py` implements the five model sessions in the paper:
S1 layout extraction, S2 visual enrichment, S3 XML translation, S4 code review,
and S5 review-guided repair. S4 produces a text review. S5 consumes that review
and returns a complete XML document; it does not call the model once per issue.
Native shape materialization preserves observed colors, corners, and borders as
Android drawable XML. `image_filling.py` implements execution-guided grounding
and image-asset binding. It is not a replacement model or another S1-S5 session.

The reviewed XML is an intermediate artifact. The final runnable layout consists
of the XML after Image Filling and its drawable resources. Reference screenshots,
measured regions, runtime view identities, and viewport metadata must correspond
to the same screen. UIED measurements are observations of input screenshots,
not ground-truth source layouts.

## Host environment

The recorded OCR implementation uses Apple Vision through
`screen2run/ocr_vision.swift`; generation and exact screenshot-metric replay require
macOS with Swift and the Vision framework. Use Python 3.11 or newer for the
complete package. Install the requirements before invoking a model.
The Android host uses JDK 17, Gradle 8.9, Android Gradle Plugin 8.7.3,
compile SDK 35, minimum SDK 23, and the API 35 capture environment.

Configure these environment variables with paths on your own machine:

```sh
export JAVA_HOME=/path/to/jdk-17
export ANDROID_HOME=/path/to/android-sdk
export SCREEN2RUN_GRADLE=/path/to/gradle-8.9/bin/gradle
export SCREEN2RUN_SCREENSHOTS=/path/to/benchmark/screenshots
export SCREEN2RUN_WORK=/path/to/screen2run-work
```

All distributed manifest paths are relative to `SCREEN2RUN_SCREENSHOTS`.
Generated artifacts are written under `SCREEN2RUN_WORK`, not into source data.
`release_paths.py` centralizes paths. No personal directories, credentials,
Android SDK binaries, Gradle caches, APKs, or emulator images are bundled.
The first render copies the source-only host project for each worker. The supplied
`gradlew` is a small launcher for an independently installed Gradle distribution;
it does not download Gradle or contain a wrapper JAR.

The geometry code uses `android_harness/fonts/RobotoStatic-Regular.ttf` and
`Roboto-Regular.ttf` for text width and variable-weight measurements, respectively.
`SCREEN2RUN_FONT` may point to the exact static font at another location, with
the supplied variable font alongside it under its original filename;
substituting a different font changes measured text widths.

## Input localization

Obtain the pinned LayoutCoder checkout separately and install its UIED/Paddle OCR
dependencies in an isolated Python environment. Set `LAYOUTCODER_HOME` to that
checkout and `SCREEN2RUN_UIED_PYTHON` to its Python executable. The localization
entry point invokes only UIED detection and merging, with LayoutCoder relation,
layout, division, and code-generation stages disabled:

```sh
python screen2run/localize_elements.py \
  --image-list data/screen_lists/main_600.txt \
  --out-dir work/element_measurements
```

The output contract is `<screen_id>/measured_elements.json`, with image dimensions,
text/component type, and pixel bounds. Use the supplied measurement snapshot for
replaying the paper cohort. Recomputing measurements with another UIED/OCR version
is a new execution, not a bitwise replay of the snapshot.

## Generation and Image Filling

Supply credentials through environment variables; do not place keys in commands,
configuration files, or committed logs. Set `OPENAI_API_KEY` and `OPENAI_BASE_URL`
for a compatible provider, then pass the exact model request identifier explicitly.
The scripts do not obtain keys from a keychain and do not change providers.

```sh
python screen2run/run_model_sessions.py --help
python screen2run/run_image_filling.py --help
python screen2run/content_guard.py --help
python android_harness/batch_render.py --help
```

For one input named `sample.png`, place the screenshot under
`$SCREEN2RUN_SCREENSHOTS/custom/`, localize it, and explicitly select a provider
model. Replace `MODEL_IDENTIFIER` and the provider URL with the intended values.
Create `work/sample.txt` containing the single line `custom/sample.png`.

```sh
export SCREEN2RUN_SCREENSHOT_LIST="$PWD/work/sample.txt"
export SCREEN2RUN_MEASUREMENTS="$PWD/work/sample_measurements"
python screen2run/localize_elements.py --image-list work/sample.txt \
  --out-dir work/sample_measurements
python screen2run/run_model_sessions.py --screens sample:custom \
  --shot-list work/sample.txt --candidate work/sample_generation \
  --measured-root work/sample_measurements --model MODEL_IDENTIFIER \
  --base-url https://PROVIDER.example/v1 --workers 1
python experiments/main_experiment.py --variant ours \
  --source work/sample_generation --cohort work/sample.txt \
  --out work/sample_realization --devices 1
```

The last command prints the local seven-step plan. Add `--execute` after starting
the required existing Android device to run it. `progress.json` identifies the
final candidate directory. The local realization stage makes no model requests.
The generation command does make paid requests; set a provider-side spending cap
before invoking it. For the paper cohort, use the supplied screenshot lists and
measurement snapshot instead of regenerating localization.

The batch generator preserves S1/S2 JSON, S3 XML, raw S4 review, the structured
review archive, S5 XML, native drawables, and contract reports. The grounding driver
prepares stable view IDs, captures the reviewed XML, reads runtime view bounds,
then produces a separate grounded and asset-bound candidate. The content guard
uses the retained pre-grounding candidate if its frozen OCR-loss rule is triggered.
Its binding-only fallback is a recovery branch, not the no-realization ablation.

Rendering does not start an emulator. Start the required API 35 device yourself.
The default parallel renderer uses one existing device. Each worker uses its own
host project and Gradle directory. Capture uses a fixed 1080-pixel width and the
reference aspect ratio at density 2.625. The shared harness checks foreground
activity, splash windows, capture dimensions, and artifact identity. Baselines
and Screen2Run use the same capture and compatibility policy.

## Five baselines

`baselines/run_baseline.py` exposes only Direct, CoT, Self-refine, DCGen, and
LayoutCoder. It has no legacy `full` branch. Direct and CoT each request one XML
answer; Self-refine makes one additional refinement request. The original prompt
templates, request construction, token accounting, response extraction, and retry
conditions are retained. Budget flags report estimated provider cost; actual
provider billing remains authoritative.

```sh
python baselines/run_baseline.py --help
python -m baselines.dcgen_adapter.online_cli --help
python -m baselines.layoutcoder_adapter.cli --help
```

The DCGen adapter uses the published recursive, one-call-per-node procedure
(`--pipeline paper`), not the earlier candidate-selection mode. It converts
reference coordinates with `reference_width / (1080 / 2.625)` pixels per dp.
The upstream DCGen source is not redistributed because its local checkout did
not include a license granting redistribution. Obtain the fixed upstream revision
and set `DCGEN_HOME` to the checkout:

```sh
git clone https://github.com/WebPAI/DCGen.git /path/to/DCGen
git -C /path/to/DCGen checkout 22af69b238fb5aa967cf3f05985a25b52481105f
export DCGEN_HOME=/path/to/DCGen
```

The adapter loads only the original `ImgNode` and `ImgSegmentation` classes from
`utils.py` and verifies their source-fragment hashes before use. A missing or different
source fails explicitly; no alternate segmentation is substituted.

The LayoutCoder adapter preserves UIED preprocessing, relations, layout parsing,
atomic generation, and structural fusion. Obtain the pinned checkout from
`https://github.com/ay7u1009/LayoutCoder`, set `LAYOUTCODER_HOME`, and install the
upstream requirements. Apply the two recorded compatibility files to that separate
checkout before preprocessing:

```sh
git clone https://github.com/ay7u1009/LayoutCoder.git /path/to/LayoutCoder
git -C /path/to/LayoutCoder checkout bf5b0032923ea68a0aff9f98fa9cd544d8cd9ee8
python baselines/upstream_compat/apply.py --checkout /path/to/LayoutCoder --apply
export LAYOUTCODER_HOME=/path/to/LayoutCoder
```

The compatibility files preserve the experiment's PaddleOCR API translation and
saturated edge-array arithmetic. Original checkout files are retained as backups.
The observed macOS preprocessing environment used Python 3.12.13,
PaddlePaddle 3.3.0, PaddleOCR 3.7.0, NumPy 2.3.5, Pillow 12.1.0,
and OpenCV contrib 4.10.0.84. The Paddle models are `PP-OCRv6_medium_det`
and `PP-OCRv6_medium_rec`; obtaining their weights is an explicit setup step.
The Android adaptation uses the retained v2 atomic prompt
and the shared extraction compatibility setting. Existing preprocessing JSON can
be supplied through `--preprocessed`. `LAYOUTCODER_ANDROID_PROMPT=v2` is set by
the five-baseline runner. The upstream Apache-2.0 license applies to its derived
material; see the release's third-party notices.

Shared Android postprocessing preserves valid public framework resources, replaces
unresolved resources with the same placeholder, removes invalid attributes, and
applies the recorded occluding-leaf compatibility rule. Resource sanitization
requires the compile SDK's public-resource list; it is loaded on actual processing,
not on module import or `--help`.

To inspect the exact one-screen baseline request without calling a model:

```sh
python baselines/run_baseline.py --arm direct --image custom/sample.png \
  --out work/direct_generation/full/sample --model MODEL_IDENTIFIER \
  --base-url https://PROVIDER.example/v1 --input-price 1 --output-price 1 \
  --max-total-cost-usd 1 --dry-run
```

Replace the illustrative prices and estimated budget with the actual provider
rates before removing `--dry-run`. Change `--arm` to `cot`, `self_refine`, `dcgen`,
or `layoutcoder` for the other baselines. The upstream dependencies are required
for the last two. All five runners expose `final.xml` and any drawables in the
same per-screen candidate directory; DCGen's original nested outputs are retained
and copied byte-for-byte to this shared contract. Apply the shared baseline
postprocessing and rendering before scoring:

```sh
python baselines/postprocess.py prepare --source work/direct_generation \
  --cohort work/sample.txt --out work/direct_ready
python android_harness/batch_render.py --cand work/direct_ready --devices 1 --hierarchy
```

Use a distinct candidate name for each method/model run. Scoring instructions are
in `docs/EVALUATION.md`; baseline candidates do not pass through Screen2Run grounding.

## Source-to-release mapping and permitted changes

| Experimental component | Release component | Changes |
| --- | --- | --- |
| `guigpt3_generate.py` | `screen2run/model_sessions.py` | English documentation; imports and resource paths |
| `guigpt3_batch.py` | `screen2run/run_model_sessions.py` | English CLI/logs; portable screenshot paths |
| `guigpt3_stage_contract.py` | `screen2run/session_contracts.py` | Module name only |
| `guigpt3_native_shapes.py` | `screen2run/native_drawables.py` | Module name only |
| `guigpt3_s5_regression.py` | `screen2run/s5_regression_check.py` | Module name only |
| `guigpt3_typography_contract.py` | `screen2run/typography_check.py` | Module name only |
| `s2r_ground.py` | `screen2run/image_filling.py` | Resource and cache locations |
| `s2r_ground_batch.py` | `screen2run/run_image_filling.py` | Imports and portable input/output roots |
| Shared Android harness | `android_harness/` | Paths, English documentation, lazy SDK lookup, source-template initialization |
| Prompt-baseline branch | `baselines/prompt_baselines_run.py` | Extracted unchanged call/extraction/refinement routines into a dedicated CLI |
| DCGen Android adapter | `baselines/dcgen_adapter/` | Portable paths; separately supplied, source-hash-checked upstream segmentation |
| LayoutCoder Android adapter | `baselines/layoutcoder_adapter/` | Portable imports and external preprocessing checkout |

No inference prompts, geometry thresholds, score definitions, or recorded results
are changed by this packaging step. Historical provider error messages retain
their exact Unicode values through escaped source literals. They are compatibility
markers, not untranslated user-facing prose. The release does not include the
earlier deterministic translator as an alternative implementation of S1-S5.

## Offline checks

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -p test_runtime_release.py -v
```

These checks import runtime modules, parse source files, exercise CLI help, verify
frozen prompt hashes, check coordinate conversion and the source-only host template,
mock all three prompt baselines, and preserve the raw S4 review. They make no API
requests and perform no Android build or rendering. Online reruns and device-level
capture need the dependencies and explicit execution steps above.
