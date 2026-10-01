#!/usr/bin/env python3
"""Render one arm across several emulators in parallel.

`device_session.py` takes a per-device AVD lock and a shared status lock, so
several harness processes may share one run dir as long as each drives a different
device and a disjoint `--screen-id` shard. Each worker also needs its own Gradle
project copy and Gradle home, or they contend on the same build directory.

  parallel_render.py --candidate-root work/candidates/<matrix> --arm full \
      --screenshot-list <list> --matrix <name> [--limit N]

Devices default to emulator-5554/5556/5558/5560 paired with android_test_project_cf1..4.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
import sys
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import release_paths  # noqa: E402  (release paths; puts every code folder on sys.path)

REPO = release_paths.RELEASE
RUNS = release_paths.RENDERS

# 5556 is excluded: it keeps failing the foreground check even with a known-good
# project, while 5554/5558/5560 pass. Order matters -- --devices takes the prefix.
DEVICES = [
    ("emulator-5554", "proposed_pilot_w00", "gradle-home-proposed-pilot-w00"),
    ("emulator-5558", "proposed_pilot_w02", "gradle-home-proposed-pilot-w02"),
    ("emulator-5560", "proposed_pilot_w03", "gradle-home-proposed-pilot-w03"),
    ("emulator-5556", "proposed_pilot_w01", "gradle-home-proposed-pilot-w01"),
]


def shard(items: list[str], n: int) -> list[list[str]]:
    return [items[i::n] for i in range(n)]


def _stop_stale_build_daemons(limit: int = 8) -> None:
    """Stop accumulated Gradle daemons before a render batch.

    Every parallel batch leaves its own daemons resident. Across a session they pile
    up (21 were alive at once here, on a 16 GB machine) and starve the emulators:
    CPU climbs, swap thrashes, and captures start failing as
    ``blank_or_wrong_activity`` -- a 33% render failure rate traced to exactly this.
    Emulators are the CPU-critical resource, so the daemons go first and rebuild
    cold on demand.
    """
    import subprocess as _sp
    try:
        out = _sp.run(["pgrep", "-f", "GradleDaemon"], capture_output=True, text=True)
    except Exception:
        return
    pids = [x for x in out.stdout.split() if x.strip()]
    if len(pids) <= limit:
        return
    print(f"stopping {len(pids)} resident Gradle daemons (emulators need the CPU)", flush=True)
    _sp.run(["pkill", "-f", "GradleDaemon"], check=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--matrix", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--screenshot-list", type=Path, required=True)
    ap.add_argument("--devices", type=int, default=1)
    ap.add_argument("--limit", type=int, help="Render only the first N screens (pipeline check).")
    ap.add_argument("--device-offset", type=int, default=0,
                    help="Start at this index in DEVICES, so two arms can share the pool.")
    ap.add_argument("--no-hierarchy", action="store_true",
                    help="skip the post-capture uiautomator dump (screenshot unchanged)")
    ap.add_argument("--resume", action="store_true",
                    help="Keep what this arm has already rendered and capture only the rest. "
                         "The harness already implements this per screen via --resume-status "
                         "(it keeps outcomes whose xml_sha256 still matches and recaptures the "
                         "rest), but the driver used to wipe the run and shard dirs first, which "
                         "threw that away and redid the whole arm. Never re-render finished work.")
    args = ap.parse_args()

    root = args.candidate_root.resolve()
    arm_dir = root / args.arm
    if not arm_dir.is_dir():
        raise SystemExit(f"missing arm dir: {arm_dir}")

    shots = [l.strip() for l in args.screenshot_list.read_text().splitlines() if l.strip()]
    have = [s for s in shots if (arm_dir / Path(s).stem / "final.xml").is_file()]
    if args.limit:
        have = have[: args.limit]
    if not have:
        raise SystemExit("no screens with final.xml")

    # materialize names the run dir after the candidate root, not --matrix.
    # The harness refuses a second process on one run dir ("refusing concurrent ledger
    # rewrite"), so each device renders into its own copy of the run and we merge after.
    base = RUNS / f"tsc_{root.name}_{args.arm}"
    if base.exists() and not args.resume:
        shutil.rmtree(base)
    print(f"materializing {len(have)} screens", flush=True)
    subprocess.run(
        [sys.executable, str(release_paths.module_file("host_resources.py")),
         "--output-root", str(root), "--arms", args.arm,
         "--screenshot-list", str(args.screenshot_list)],
        check=False,
    )
    if not (base / "items.jsonl").is_file():
        raise SystemExit(f"materialize produced no items.jsonl in {base}")
    if args.limit:
        keep = {Path(s).stem for s in have}
        rows = [l for l in (base / "items.jsonl").read_text().splitlines()
                if l.strip() and json.loads(l)["screen_id"] in keep]
        (base / "items.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")

    # materialize recognises a screen by its board.json and silently drops any screen dir
    # that lacks one, as long as at least one sibling has it. A candidate root built by a
    # repair pass that writes only final.xml therefore renders *fewer screens than asked*
    # with no error, and the arm looks complete. Refuse to render a short run: a silently
    # partial arm is worse than no arm, because it still produces a full-looking table.
    import json as _json
    materialized = {_json.loads(l)["screen_id"] for l in (base / "items.jsonl").read_text().splitlines() if l.strip()}
    missing = sorted(Path(s).stem for s in have if Path(s).stem not in materialized)
    if missing:
        raise SystemExit(
            f"REFUSING to render: materialize produced {len(materialized)} of {len(have)} "
            f"requested screens; {len(missing)} are absent from items.jsonl. "
            f"First few: {missing[:5]}. A screen dir without board.json is invisible "
            f"to host_resources.")

    todo = sorted(Path(s).stem for s in have)
    if args.resume:
        # Shards are re-dealt from the sorted list on every call, so a growing candidate
        # would move captured screens to another device's ledger and capture them again.
        shots_dir = base / "raw" / "screenshots"

        def fresh(sid: str) -> bool:
            png = shots_dir / f"{sid}.png"
            return png.is_file() and png.stat().st_mtime >= (arm_dir / sid / "final.xml").stat().st_mtime

        kept = [s for s in todo if fresh(s)]
        todo = [s for s in todo if s not in set(kept)]
        print(f"resume: {len(kept)} screens already captured, {len(todo)} to render", flush=True)
    groups = shard(todo, args.devices)
    procs = []
    pool = DEVICES[args.device_offset:args.device_offset + args.devices]
    if len(pool) < args.devices:
        raise SystemExit(f"only {len(pool)} devices available from offset {args.device_offset}")
    for k, ((serial, project, home), screens) in enumerate(zip(pool, groups)):
        if not screens:
            continue
        release_paths.ensure_host_project(project)
        run_dir = RUNS / f"tsc_{root.name}_{args.arm}_d{args.device_offset + k}"
        if run_dir.exists() and not args.resume:
            shutil.rmtree(run_dir)
        # each device owns its own ledger; on --resume the existing one is kept so the
        # harness's --resume-status can skip the screens this shard already captured
        shutil.copytree(base, run_dir, dirs_exist_ok=args.resume)
        cmd = [sys.executable, "-u", str(release_paths.module_file("device_session.py")),
               "--run-dir", str(run_dir),
               "--project", str(release_paths.HOST_PROJECTS / f"{project}"),
               "--android-home", str(release_paths.ANDROID_SDK),
               "--device-serial", serial,
               "--no-snapshot-restore", "--no-gradle-clean",
               # Harness defaults, which is what the baselines were captured under: their
               # ledgers carry no settle rows and no poll, i.e. a fixed 3.0s delay. Our
               # earlier 0.4s+settle was both a different protocol and too short -- the
               # system splash was still up, so cold starts failed as environment_missing.
               "--screenshot-delay", "3.0",
               "--screenshot-settle-seconds", "0",
               "--foreground-poll-seconds", "6",
               # See docs/RUNTIME.md for the shared compatibility and capture contracts.
               "--skip-hierarchy-dump" if args.no_hierarchy else "--force-hierarchy-dump",
               "--resume-status"]
        for s in screens:
            cmd += ["--screen-id", s]
        env = {"GRADLE_USER_HOME": str(release_paths.GRADLE_HOMES / home),
               "GRADLE_OPTS": "-Xmx512m -Dorg.gradle.daemon=true -Dorg.gradle.caching=true",
               "JAVA_HOME": str(release_paths.JAVA_HOME),
               "ANDROID_HOME": str(release_paths.ANDROID_SDK),
               "PATH": f"{release_paths.JAVA_HOME / 'bin'}:{release_paths.ANDROID_SDK / 'platform-tools'}:{__import__('os').environ.get('PATH','')}"}
        import os
        full_env = {**os.environ, **env}
        log = base / f"render_{serial}.log"
        procs.append((serial, len(screens), run_dir,
                      subprocess.Popen(cmd, stdout=log.open("w"), stderr=subprocess.STDOUT, env=full_env)))
        print(f"  {serial}: {len(screens)} screens -> {run_dir.name}", flush=True)

    failed = []
    shards_done = []
    for serial, n, run_dir, p in procs:
        rc = p.wait()
        print(f"  {serial} done rc={rc} ({n} screens)", flush=True)
        shards_done.append(run_dir)
        if rc != 0:
            failed.append(serial)

    # Merge every shard's screenshots, hierarchies and status rows into the base run.
    merged = 0
    for run_dir in [base] + shards_done:
        for sub, pat in (("screenshots", "*.png"), ("hierarchy", "*.xml")):
            src = run_dir / "raw" / sub
            if src.is_dir():
                dst_dir = base / "raw" / sub
                dst_dir.mkdir(parents=True, exist_ok=True)
                # See docs/RUNTIME.md for the shared compatibility and capture contracts.
                if src.resolve() == dst_dir.resolve():
                    continue
                for f in src.glob(pat):
                    shutil.copy2(f, dst_dir / f.name)
                    if sub == "screenshots":
                        merged += 1

    # See docs/RUNTIME.md for the shared compatibility and capture contracts.
    (base / "derived").mkdir(parents=True, exist_ok=True)
    latest: dict[str, str] = {}
    for run_dir in [base] + shards_done:
        status = run_dir / "derived" / "loading_status.jsonl"
        if not status.is_file():
            continue
        with status.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    sid = json.loads(line).get("screen_id")
                except json.JSONDecodeError:
                    continue
                if sid:
                    latest[sid] = line
    if latest:
        (base / "derived" / "loading_status.jsonl").write_text(
            "\n".join(latest.values()) + "\n", encoding="utf-8")
    print(f"merged {merged} screenshots into {base}  (status rows: {len(latest)})")
    print(f"run dir: {base}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
