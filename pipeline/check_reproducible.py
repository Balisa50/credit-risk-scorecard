#!/usr/bin/env python3
"""Assert that a fresh pipeline run reproduces the committed results.

The loan book is generated from a fixed seed, so every figure in
public/data/pipeline_results.json must come back identical. Only the
generated_at timestamp is allowed to move.

This is the check that keeps the README honest. Its results table used to
quote a Gini of 0.29 against an actual 0.268 and a PSI of 0.008 against an
actual 0.002, because the prose was written once and the pipeline moved on.
Nothing but a check like this catches that.

Usage: python check_reproducible.py   (run from pipeline/, after run_pipeline.py)
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

RESULTS = "public/data/pipeline_results.json"
VOLATILE = ("generated_at",)


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in VOLATILE}


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    current = json.loads((root / RESULTS).read_text(encoding="utf-8"))

    show = subprocess.run(
        ["git", "show", f"HEAD:{RESULTS}"],
        cwd=root, capture_output=True, text=True,
    )
    if show.returncode != 0:
        print(f"could not read {RESULTS} from HEAD:\n{show.stderr}", file=sys.stderr)
        return 2

    committed = json.loads(show.stdout)

    if strip(current) == strip(committed):
        print(f"OK: a fresh seeded run reproduces {RESULTS} exactly.")
        return 0

    print(f"DRIFT: a fresh run does not match the committed {RESULTS}.", file=sys.stderr)
    for key in sorted(set(strip(current)) | set(strip(committed))):
        if current.get(key) != committed.get(key):
            print(f"  section differs: {key}", file=sys.stderr)
    print("\nIf the change is intended, commit the regenerated file.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
