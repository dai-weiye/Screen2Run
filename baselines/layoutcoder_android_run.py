#!/usr/bin/env python3
"""Convenience entry point for the Android-adapted LayoutCoder baseline."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths
from run_baseline import main

if __name__ == "__main__":
    raise SystemExit(main(["--arm", "layoutcoder", *sys.argv[1:]]))
