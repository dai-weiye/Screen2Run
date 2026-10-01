#!/usr/bin/env python3
"""Warm the unchanged pair cache from an explicit captured-pair manifest."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path


def worker(task):
    pairs, cache, mask = task
    from screenshot_metrics import Evaluator
    evaluator = Evaluator(Path(cache), mask)
    completed, failed = 0, []
    try:
        for reference, rendered, arm, sid in pairs:
            try:
                evaluator.score(Path(reference), Path(rendered))
                completed += 1
            except Exception as exc:
                failed.append({"arm": arm, "screen_id": sid, "error_type": type(exc).__name__})
    finally:
        evaluator.close()
    return {"completed": completed, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--mask-top", type=float, default=0.045)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    rows = json.loads(args.pairs.read_text(encoding="utf-8"))
    partitions = [[] for _ in range(args.workers)]
    screen_index = {}
    for row in rows:
        if row.get("status") == "method_failure":
            continue
        if row.get("status") != "render_success":
            raise ValueError("Unresolved capture in pair manifest")
        paths = []
        for name in ("reference", "render_path"):
            path = Path(row[name])
            paths.append(str(path if path.is_absolute() else args.pairs.resolve().parent / path))
        index = screen_index.setdefault(row["screen_id"], len(screen_index))
        partitions[index % args.workers].append((*paths, row["arm"], row["screen_id"]))
    tasks = [(pairs, str(args.cache), args.mask_top) for pairs in partitions if pairs]
    if not tasks:
        print("No successful captures require metric precomputation")
        return 0
    with mp.get_context("spawn").Pool(min(args.workers, len(tasks))) as pool:
        results = pool.map(worker, tasks)
    print(json.dumps(results, indent=2))
    return 2 if any(row["failed"] for row in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
