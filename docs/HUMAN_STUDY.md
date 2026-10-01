# Human study: anonymized ratings and replication materials

## What is included

`data/human_study/ratings.json` contains every quantitative response for the 60 main visual tasks and 60 code tasks: six pseudonymous raters, three methods, two tracks, and **2,160 candidate-level records**. The same six raters participated in both tracks; the two collection rounds do not create twelve independent participants.

The initial collection covered 60 visual tasks and 12 code tasks. A separate, code-only collection covered the remaining 48 screens. The public `collection_round` field preserves that distinction. Initial responses have not been replaced with later responses. Screen IDs are benchmark identifiers, not participant identities.

The release also includes:

- `expected_results.json`: sanitized numerical targets for the joint 60+60 analysis, including round-specific code means.
- `ratings.schema.json`: the public score schema.
- `stimuli_manifest.json`: the fixed 60-screen roster, dataset and complexity strata, code-collection round, and stimulus hashes.
- `code/<item_id>/<method>.xml`: all **180 exact de-identified XML texts shown for code assessment**.
- `questionnaire_template.html`: an English, blank replication template. It is not a historical questionnaire, an original expert return, or evidence that another collection occurred.

All descriptions, field names, and questionnaire interface text in this release are English. Original interface strings and comments inside the displayed XML are research data and may use other languages; they have not been translated or rewritten.

## Privacy and scope

Only numeric judgments, benchmark IDs, method names, pseudonymous rater IDs, and coarse collection-round labels are published. Raw browser answer files, identity keys, free-text comments, demographic answers, exact collection/export times, device/browser fingerprints, and private filesystem paths are excluded. The original free-text responses were not translated into invented English responses.

The organizer reported that the supplementary answers were independently completed by the same six experts. File and hash checks establish data consistency, not independent authentication of the person operating a browser. The archived source contains clock-order warnings; original timestamps were neither repaired nor published.

The initial protocol included one practice task, three repeated tasks, and two attention-check tasks in addition to the 60 main visual tasks. They are not included in the main-rating denominators. One rater failed one attention-check rule, and one participant did not satisfy the stated experience criterion. The primary analysis retains all six raters; do not silently exclude them to improve results. This numerical release does not establish ethics approval or permission to redistribute third-party images.

## Experimental design and rubric

The main roster comprises 25 Easy, 25 Real, and 10 Unseen screens, sampled across the dataset and interface-complexity strata. The evaluated methods are Screen2Run (`ours`), LayoutCoder (`layoutcoder`), and DCGen (`dcgen`). Visual candidates and code candidates were assigned independent anonymous display labels. The public numerical records are already mapped to method names, so no private organizer key is required to reproduce their statistics.

Each visual task presented a reference screenshot and three rendered candidates. Raters scored every candidate independently from 1 to 5:

| Field | Assessment question |
|---|---|
| `overall` | How similar is the rendered interface to the reference overall? |
| `layout` | How closely do position, size, alignment, spacing, and hierarchy match? |
| `text` | Are text content, font size, weight, and wrapping reproduced? |
| `image` | Are images, icons, and logos present, correct, and correctly positioned? |
| `style` | How closely do colors, backgrounds, borders, corners, and control appearance match? |
| `replace` | Is the appearance and content close enough to replace the reference interface? 0 = no; 1 = yes. |
| `rank` | Relative visual similarity: 1 = most similar; 3 = least similar. No ties. |

Visual anchors: **1** almost entirely different or substantially missing; **2** clearly different, with few matching details; **3** broadly similar but with visible errors; **4** minor differences; **5** almost indistinguishable. The replacement judgment is a subjective visual judgment, not a measured production-readiness guarantee.

Each code task presented the reference screenshot and three XML texts. Raters did not execute the code. All code underwent the same formatting and semantics-preserving removal/renaming of method-specific bookkeeping identifiers before display.

| Field | Assessment question |
|---|---|
| `readability` | Is the structure, hierarchy, and naming easy to understand? |
| `maintainability` | Is the code easy to modify and reuse, without excessive hard-coding, repetition, or nesting? |
| `practice` | Are Android layout containers and attributes used appropriately, with reasonable adaptability? |
| `usability` | Would you use this code as a starting point for further development? |
| `rank` | Relative preference for continuing development: 1 = most preferred; 3 = least preferred. No ties. |

Code anchors: **1** very poor; **2** poor; **3** fair; **4** good; **5** very good. Code judgments must not be inferred from visual rankings or adjusted to agree with them.

## Public data contract

The top-level `schema` is `screen2run-human-ratings/1`. `raters`, `methods`, and `items` are explicit analysis orders, not unordered sets. Both `items.v` and `items.c` contain the same frozen 60-item order. The code analysis order is **not** the concatenation of the original 12 and supplementary 48 presentation orders.

Each record has exactly these fields:

```json
{
  "part": "c",
  "rater": "R1",
  "item_id": "PUBLIC_ITEM_ID",
  "screen_id": "BENCHMARK_SCREEN_ID",
  "dataset": "Real",
  "method": "ours",
  "values": {
    "readability": 3,
    "maintainability": 3,
    "practice": 3,
    "usability": 3,
    "rank": 1
  },
  "collection_round": "code_supplement"
}
```

This is a schema illustration, not an additional observation. `part=v` denotes visual assessment and `part=c` denotes code assessment. Scores are integers; missing values must not be replaced by zero. For each rater/item/track, the three ranks must be a permutation of 1, 2, and 3.

## Numerical reproduction

For each outcome and method, first average the six rater scores within each screen, then average the 60 screen means. Compare paired screen means with two-sided Wilcoxon tests. Apply Holm adjustment jointly to the 24 method comparisons (12 outcomes, two baselines). The focal method is `ours`; rank is lower-is-better, while all other outcomes are higher-is-better.

The saved targets also include Friedman tests, rank-biserial correlation, Cliff's delta, and two-way bootstrap confidence intervals. To reproduce the stored bootstrap draws, preserve the public rater, method, item, and outcome orders; use NumPy's `default_rng(20260930)` with 5,000 draws. In each draw, sample six rater indices and sixty screen indices independently with replacement, using the paired difference matrix. Do not reset the generator between comparisons. See `expected_results.json.statistical_protocol.outcome_order` for the exact traversal order.

The visual means, raw tests, effects, and intervals preserve the initial visual analysis. Its jointly adjusted Holm values change when the code outcomes change from 12 to 60 items. Report the 12+48 code rounds explicitly; round-specific means are descriptive and are not additional independent experiments.

Run the release checks from the repository root:

```bash
python -m unittest discover -s tests -p 'test_human_release.py' -v
```

The tests verify complete coverage, valid ratings/ranks, exact displayed-XML hashes, absence of prohibited metadata fields, collection-round membership, and all 36 outcome/method means against the stored targets. Numerical agreement does not replace the limitations above.

## Obtaining or reconstructing image stimuli

The release does **not** include third-party reference screenshots, rendered screenshots, cropped image assets, or drawable bundles. Redistribution rights for those images have not been established. Obtain the benchmark materials from their authorized dataset sources under the applicable terms, using the dataset/screen IDs in `stimuli_manifest.json`. Do not scrape private applications or substitute different screens silently.

The manifest includes reference-display image hashes, candidate capture hashes, display-image hashes, original XML hashes, and supplied display-XML hashes. Original model outputs are stochastic: rerunning a model or rendering different assets is not guaranteed to reproduce the historic bytes. Label such outputs as a new replication, not the original stimuli. Exact original image assets require an authorized artifact distribution; a hash is an identity check, not a download location or a license grant.

The historical display preprocessing converted images to RGB, masked the top 4.5% with the median color in the band immediately below it, resized to height 1,280 pixels with Pillow LANCZOS, and encoded lossless WebP with method 4. For a source height `h`, `cut=round(0.045*h)`; the median band is rows `cut:cut+max(2,h//40)`. Width is scaled proportionally. Library/encoder versions can affect byte hashes; preserve processing provenance. Raters were instructed not to score system status or navigation bars.

The supplied XML is the exact assessment text, including references to drawable resources that are not redistributed. It is therefore **not** a standalone Android project. Do not remove those references, rasterize whole reference screens, or replace code with screenshots to reproduce the reported ratings.

## Using the English replication template

Open `data/human_study/questionnaire_template.html` locally. The organizer prepares a **new**, blind, self-contained task JSON containing an opaque study ID, one rater ID, and a `tasks` array. Each task has `item_id`, `track` (`visual` or `code`), `reference_image` (an authorized image data URI), and exactly three candidate strings under `A/B/C` for visual tasks or `X/Y/Z` for code tasks. Visual candidate strings are image data URIs; code strings are the displayed XML. Do not include method names, source paths, or the answers from this release in the participant file.

Randomize positions independently across tracks and preserve the new study's assignment privately. The template shows the rubric, allows rankings to be edited, validates all required ratings, and can download incomplete progress. It performs no network requests and includes no existing participant scores. It intentionally does not auto-load historic answers. Its export schema is `screen2run-replication-return/1`, not an original-study return. Save a progress copy before closing; reloading the blank template does not restore a session automatically.

Recruit eligible raters who have not developed the evaluated methods, obtain appropriate consent and institutional review where required, keep method identities hidden, and let participants assess candidates independently. Complete code and visual judgments separately, allow breaks, and preserve unfavorable as well as favorable responses.
