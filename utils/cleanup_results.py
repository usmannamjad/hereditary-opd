#!/usr/bin/env python3
"""Delete run directories where generation did not complete.
A run is incomplete if meta.json is missing or has no 'finished_at' field."""

import argparse
import json
import shutil
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="print what would be deleted without deleting")
    args = parser.parse_args()

    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return

    incomplete = []
    complete = []

    for run_dir in sorted(RESULTS_DIR.iterdir()):
        if not run_dir.is_dir():
            continue
        meta_path = run_dir / "meta.json"
        if not meta_path.exists():
            incomplete.append((run_dir, "no meta.json"))
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if "finished_at" not in meta or meta["finished_at"] is None:
            incomplete.append((run_dir, "generation not finished"))
        else:
            complete.append(run_dir)

    print(f"Complete runs: {len(complete)}")
    print(f"Incomplete runs: {len(incomplete)}")

    if not incomplete:
        print("Nothing to clean up.")
        return

    for run_dir, reason in incomplete:
        if args.dry_run:
            print(f"  [dry-run] would delete {run_dir.name} ({reason})")
        else:
            shutil.rmtree(run_dir)
            print(f"  deleted {run_dir.name} ({reason})")


if __name__ == "__main__":
    main()