#!/usr/bin/env python3
"""Assert that a fresh pipeline run reproduces the committed results.

The loan book is seeded, so every figure in public/data/pipeline_results.json
comes back the same on a given machine. It does NOT come back bit-identical
across machines: the logistic regression is fitted by lbfgs, an iterative
optimiser, and scikit-learn and NumPy sit on whatever BLAS the platform ships.
Those differ in the last few digits between Windows and Linux, which is
ordinary floating point, not a broken seed.

So the check is a tolerance, not an equality. RTOL is set far tighter than any
real change in an assumption would produce (retuning a lapse rate or a select
factor moves figures by percent, not by a thousandth of a percent) and far
looser than platform noise. Structure is still compared exactly: a new key, a
dropped key, or a changed string fails immediately.

This is what keeps the README honest. Its results table used to quote a Gini
of 0.29 against an actual 0.268 and a PSI of 0.008 against an actual 0.002,
because the prose was written once and the pipeline moved on. Nothing but a
check like this catches that.

Usage: python check_reproducible.py   (run from pipeline/, after run_pipeline.py)
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

RESULTS = "public/data/pipeline_results.json"
VOLATILE = ("generated_at",)
RTOL = 1e-3
ATOL = 1e-9


def compare(a, b, path="", out=None):
    """Collect every difference between two JSON structures."""
    out = [] if out is None else out

    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k in VOLATILE and not path:
                continue
            if k not in a:
                out.append(f"{path}/{k}: missing from the fresh run")
            elif k not in b:
                out.append(f"{path}/{k}: not in the committed file")
            else:
                compare(a[k], b[k], f"{path}/{k}", out)
        return out

    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} against {len(b)}")
            return out
        for i, (x, y) in enumerate(zip(a, b)):
            compare(x, y, f"{path}[{i}]", out)
        return out

    if isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            out.append(f"{path}: {a!r} against {b!r}")
        return out

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if not math.isclose(a, b, rel_tol=RTOL, abs_tol=ATOL):
            rel = abs(a - b) / max(abs(b), 1e-30)
            out.append(f"{path}: {a!r} against {b!r}  (relative {rel:.2e})")
        return out

    if a != b:
        out.append(f"{path}: {a!r} against {b!r}")
    return out


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

    diffs = compare(current, json.loads(show.stdout))

    if not diffs:
        print(f"OK: a fresh seeded run reproduces {RESULTS} to within {RTOL:g} relative.")
        return 0

    print(f"DRIFT: a fresh run does not match the committed {RESULTS}.", file=sys.stderr)
    for line in diffs[:25]:
        print(f"  {line}", file=sys.stderr)
    if len(diffs) > 25:
        print(f"  ... and {len(diffs) - 25} more", file=sys.stderr)
    print("\nIf the change is intended, commit the regenerated file.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
