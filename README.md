# Screen2Run

**Screen2Run: From Screenshots to Runnable Android GUIs**

Screen2Run reconstructs an Android interface from a screenshot. Five model
sessions extract a layout, enrich its appearance, generate XML, review the draft,
and repair it. Image Filling uses runtime view bounds and screenshot measurements
to correct geometry and bind image assets. The output is a layout and drawable
resources for the supplied Android host app.

## Start here

The numeric replication requires no API key, emulator, screenshot download, or
model invocation. Run commands from this directory using Python 3.11 or newer:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-analysis.txt
python scripts/validate_release.py
python evaluation/reproduce.py all --out work/reproduced --plots
python evaluation/verify_reproduction.py --out work/reproduced
```

The last command recomputes comparisons, ablations, complexity groups, backbone
comparisons, human-study statistics, and paper-style figures from frozen data.
It writes only to the selected output directory. See
[evaluation instructions](docs/EVALUATION.md) for statistical families and outputs.

For the complete offline test suite, install `requirements-test.txt` and run
`python -m pytest tests -q`. Runtime contract tests use synthetic inputs and
mock model responses; they do not call paid APIs or start an emulator.

To generate new layouts, install `requirements-runtime.txt`, obtain the input
images, and configure a model endpoint and Android environment as described in
[runtime setup](docs/RUNTIME.md). Generation is an explicit, potentially paid
action; verification and statistical replication do not invoke models.

## Contents

| Directory | Contents |
| --- | --- |
| `screen2run/` | S1-S5, contracts, OCR, native drawables, grounding, and asset binding |
| `prompts/` | Exact English prompts, context fragments, and composition recipes |
| `baselines/` | Direct, CoT, Self-refine, and Android adapters for DCGen and LayoutCoder |
| `android_harness/` | Host app sources, compatibility rules, capture, fonts, and placeholder |
| `experiments/` | Explicit generation and rendering plans, including five ablation variants |
| `evaluation/` | Screenshot scoring, paired statistics, human-study analysis, and plotting |
| `data/results/` | Frozen per-screen scores and expected paper statistics |
| `data/element_measurements/` | Input-derived element measurements for the 600-screen cohort |
| `data/screen_lists/` | Main 600-screen and backbone 120-screen lists, source mappings, and hashes |
| `data/human_study/` | Anonymous 60-interface + 60-code ratings, questionnaire, and 180 displayed XML files |
| `docs/` | Reproduction, dataset acquisition, protocol, and paper-to-code mapping |
| `tests/` | Offline contract, runtime, statistics, and human-data tests |

The main score file contains 600 screens for each of six methods. RQ2 contains
the same frozen Full control and five 600-screen variants. Each backbone study
uses the same 120-screen subset. Human ratings retain six participants and both
60-item tracks. `data/results/data_contract.json` records the released data schema.

## Method and paper alignment

The three modules are **Initial Code Generation**, **Code Review and Fix**, and
**Image Filling**. S5 produces the **Reviewed XML**; Image Filling produces the
**Final XML + assets**. Image Filling uses UIED and OCR measurements.
[The correspondence table](docs/METHOD_MAPPING.md) maps the figure and paper to
the actual entry points and artifacts.

All authored documentation, prompts, code comments, and paths are English.
Multilingual text inside measured screenshots or generated interface XML is
research input/output and remains unchanged.

## Data and licenses

Input screenshots are obtained from the original dataset providers using the
exact source mappings and checksums in the package; see [Datasets](docs/DATASETS.md).
No reference source layout is needed by generation. Participant identities,
private raw response metadata, credentials, provider billing logs, and local
machine paths are excluded.

Original Screen2Run code is MIT-licensed. Third-party material retains its own
license; see [NOTICE](NOTICE) and `LICENSES/`. Upstream baseline repositories,
Android SDK, Gradle, and pretrained model weights are external dependencies.

AI assistants supported implementation, experimental and analysis code,
documentation, and manuscript editing. Released numerical results are derived
from the recorded experimental outputs and participant ratings.
