#!/usr/bin/env python3
"""Install the recorded LayoutCoder compatibility files into a separate checkout."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import shutil

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Otherwise only list the two replacements")
    args = parser.parse_args()
    root = args.checkout.resolve()
    if not (root / "run_single.py").is_file() or not (root / "LICENSE").is_file():
        parser.error("Expected a separately obtained LayoutCoder checkout")
    pairs = [("ocr.py", "UIED/detect_text/ocr.py"),
             ("edge_detection.py", "utils/edge_detection.py")]
    for source_name, relative in pairs:
        source = Path(__file__).parent / source_name
        target = root / relative
        if not target.is_file():
            parser.error("Missing upstream file: " + relative)
        print(relative + " <- " + source_name)
        if not args.apply or target.read_bytes() == source.read_bytes():
            continue
        backup = target.with_name(target.name + ".screen2run-original")
        if backup.exists() and target.read_bytes() != backup.read_bytes():
            parser.error("Checkout has additional changes; use a clean independent copy: " + relative)
        if not backup.exists():
            shutil.copy2(target, backup)
        shutil.copy2(source, target)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
