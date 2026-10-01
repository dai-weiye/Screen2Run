#!/usr/bin/env python3
"""Compose prompt text offline. This utility never contacts a model provider."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def compose(name: str, context: dict, *, xml_retry: int = 0) -> str:
    recipes = json.loads((ROOT / "prompts/recipes.json").read_text(encoding="utf-8"))
    if name not in recipes:
        raise ValueError("Unknown prompt name")
    if isinstance(xml_retry, bool) or not isinstance(xml_retry, int) or xml_retry not in (0, 1, 2):
        raise ValueError("XML retry must be 0, 1 or 2")
    if xml_retry and name not in ("s3_xml_translation", "s5_issue_by_issue_fix"):
        raise ValueError("XML retry suffix applies only to S3 and S5")
    result = []
    for item in recipes[name]["segments"]:
        if "literal" in item:
            result.append(item["literal"])
            continue
        key, mode = item["context"], item["mode"]
        if key not in context:
            raise ValueError("Missing dynamic context: " + key)
        value = context[key]
        if mode == "text":
            if key in ("width", "height"):
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError("Screenshot dimensions must be positive integers")
            elif not isinstance(value, str):
                raise ValueError("Text context must already contain the canonical stage text: " + key)
            result.append(str(value))
        elif mode == "json_ensure_ascii_false":
            if not isinstance(value, dict):
                raise ValueError("Diagnostic context must be a report object: " + key)
            result.append(json.dumps(value, ensure_ascii=False))
        elif mode == "optional_prefix":
            if not isinstance(value, str):
                raise ValueError("OCR context must be text")
            result.append(item["prefix"] + value if value else "")
        elif mode == "dcgen_ordered_children":
            if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
                raise ValueError("Child fragments must be an ordered list of XML strings")
            result.append("\n\n".join(f"Child fragment {i}:\n{code}" for i, code in enumerate(value)))
        else:
            raise ValueError("Unknown dynamic-context mode")
    text = "".join(result)
    if xml_retry:
        text += (ROOT / "prompts/shared/xml_retry_suffix.txt").read_text(encoding="utf-8") * xml_retry
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("--context", type=Path, help="Local JSON containing only the required dynamic context")
    parser.add_argument("--xml-retry", type=int, default=0)
    args = parser.parse_args()
    context = json.loads(args.context.read_text(encoding="utf-8")) if args.context else {}
    print(compose(args.name, context, xml_retry=args.xml_retry), end="")


if __name__ == "__main__":
    main()
