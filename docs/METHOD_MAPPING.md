# Paper, figure, and implementation correspondence

The system is **Screen2Run**. The paper title is **Screen2Run: From Screenshots to
Runnable Android GUIs**. Session labels denote model calls; Image Filling is a
deterministic, execution-guided module after those calls.

| Figure / paper label | Implementation | Input and output |
| --- | --- | --- |
| Initial Code Generation | `screen2run/model_sessions.py` | S1-S3 |
| S1: Layout extraction | `generate`, layout prompt and session contracts | Screenshot and dimensions to identified layout JSON |
| S2: Visual enrichment | `generate`, visual prompt and session contracts | Screenshot and S1 JSON to visually enriched JSON |
| S3: XML translation | `generate`, XML prompt | Screenshot, S2 JSON, and measured-region hints to draft XML |
| Code Review and Fix | `screen2run/model_sessions.py` | S4-S5 |
| S4: Code review | `generate`, review prompt | Screenshot, draft, OCR, and contract diagnostics to `stage4_critique.txt` |
| S5: Review-guided repair | `generate`, repair prompt and `native_drawables.py` | Review and draft to **Reviewed XML**, plus native drawable resources |
| UIED + OCR | `localize_elements.py`, `ocr.py`, `ocr_vision.swift` | Reference screenshot to candidate regions, text, and coordinates |
| Image regions | `data/element_measurements/`, `image_filling.py` | Measured candidates refined for crop selection |
| Execution-guided grounding | `image_filling.py`, `run_image_filling.py` | Runtime hierarchy and target regions to corrected view geometry |
| Asset binding | `image_filling.py` | Candidate crops and stable view identities to drawable references |
| Content preservation | `content_guard.py` | Captured candidate and OCR evidence to retained output or recorded fallback |
| Final XML + assets | Candidate directory's `full/<screen_id>/final.xml` and `drawables/` | Runnable output used by the common capture harness |

S4 returns a textual review, stored with a host-side JSON wrapper. S5 consumes
the complete review in one repair request using the prompt recipe
`s5_issue_by_issue_fix`.

The gear in Figure 1 combines grounding and asset binding. Grounding internally
uses execution and view-hierarchy measurements. The common final build/install/
capture procedure evaluates all methods and is described in the experimental
setup rather than as an additional model session.

The image branch contains candidate crops and region coordinates.
Text and native controls are represented in XML; detailed imagery is
recovered as resources. Whole-screen reference-image backgrounds are not an
alternative rendering mode.

## RQ2 names

| Paper variant | Manifest arm | Intervention |
| --- | --- | --- |
| w/o S1-S2 | `no_planning` | Omit explicit planning sessions |
| w/o S4-S5 | `no_review` | Omit model review and repair |
| w/o Image Filling | `no_realization` | Omit execution-guided grounding and asset realization |
| w/o assets binding | `no_raster_output` | Retain grounded geometry; replace final bound raster resources with placeholders |
| w/o S2 | `no_visual` | Omit visual enrichment only |

The `full` arm in the ablation table is the frozen `ours` arm of RQ1. Its seven
quality values are identical screen by screen. All measured differences,
including increases, remain in the released rows and statistical outputs.
