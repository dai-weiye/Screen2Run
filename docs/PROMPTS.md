# Prompt artifacts

The `prompts/` directory contains the exact English instructions used by the
released implementation, including Screen2Run's five sessions and the five
baseline methods. `recipes.json` records the order of literal and dynamic text.
`.base.txt` files preserve runtime instruction strings; `.template.txt` files
show the initial composition with named placeholders.

| Session | Dynamic text context | Output |
| --- | --- | --- |
| S1: Layout extraction | Reference width and height | Layout JSON |
| S2: Visual enrichment | Canonical S1 JSON | Enriched design JSON |
| S3: XML translation | Canonical S2 JSON and measured-region hints | Draft XML |
| S4: Code review | S2 JSON, draft XML, contract diagnostics, and visible OCR text | Text review |
| S5: Review-guided repair | S2 JSON, draft XML, review, identity mapping, OCR, and measurements | Reviewed XML |

All five sessions also receive the original reference screenshot as an image
item. The text composer does not read an image or contact a model. Dynamic JSON
uses the generator's canonical serialization and original insertion order.
OCR context includes its prefix only when OCR text is nonempty.

Compose a prompt locally with:

```sh
python scripts/compose_prompt.py direct
python scripts/compose_prompt.py s1_layout_extraction --context context.json
```

For S1, `context.json` contains `{"width": 1080, "height": 1920}`. The recipe
lists the fields for every other stage. For S3 and S5, `--xml-retry 1` or `2`
appends the original retry suffix once or twice, matching the bounded parsing
retry. The base prompt and successfully parsed XML are not rewritten by this
utility. Session contracts, response extraction, and native drawable generation
are implemented in `screen2run/` rather than in the text composer.

Direct and CoT use the original screenshot. Self-refine receives the Direct XML
and the same screenshot. LayoutCoder receives each atomic crop. DCGen receives
the current segment and its ordered child XML fragments. The adapters retain
their own decomposition and assembly procedures; they do not use Screen2Run's
post-generation grounding.

The name `s5_issue_by_issue_fix` is a stable recipe identifier. The method and
figure call this session **Review-guided repair**: the model receives the whole
review and returns a complete revised layout in one call.
