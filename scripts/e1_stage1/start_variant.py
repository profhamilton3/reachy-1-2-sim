"""Stage 2 policy A verification (decision note
outputs/e1-stage2-decision-2026-09-15.md §4): classify a settled pose's
`start_variant` from a recorder log -- the same JSON
`measure_route_clearance.py` (via `leg.sh`/`parked.sh`) writes -- and exit
non-zero if it matches neither known variant. "Failure stops progression":
`parked.sh` treats a non-zero exit here exactly like a failed linker or
recorder step, writing `control/stop`.

The classification itself is pure (`plan.classify_start_variant`); this
module is only the file-reading/CLI wrapper the decision note asks for
("the start-variant check as a small script over the settle output", §6).
No SDK, no server, no motion -- reads one JSON file.

`--cycle <CYCLE>` (PR #124 review, M3) embeds the cycle id this evidence
is being recorded for into the output JSON's `"cycle"` field, so
`plan.start_variant_gate` can refuse evidence that is missing, stale, or
recorded for a different cycle -- without it, nothing bound a parked
recording's classification to the specific cycle it is meant to authorize.

Usage:
  python -m scripts.e1_stage1.start_variant <recording.json> [--cycle <CYCLE>]
"""
from __future__ import annotations

import json
import pathlib
import sys
from typing import List, Optional

from . import plan


def last_pose(recording: dict) -> dict:
    samples = recording.get("samples") or []
    if not samples:
        raise ValueError("recording has no samples to classify")
    joints = samples[-1].get("joints")
    if not isinstance(joints, dict):
        raise ValueError("recording's last sample has no 'joints' dict")
    return dict(joints)


def classify_recording(recording: dict) -> Optional[str]:
    return plan.classify_start_variant(last_pose(recording))


def _parse_argv(argv: List[str]) -> Optional[tuple]:
    """Returns `(path, cycle)` or `None` on a bad argv (caller prints usage
    and exits 2). `--cycle` is optional so existing callers that only ever
    cared about "does this match a known variant at all" keep working."""
    positional: List[str] = []
    cycle: Optional[str] = None
    it = iter(argv)
    for tok in it:
        if tok == "--cycle":
            cycle = next(it, None)
            if cycle is None:
                return None
        else:
            positional.append(tok)
    if len(positional) != 1:
        return None
    return positional[0], cycle


def main(argv: List[str] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parsed = _parse_argv(argv)
    if parsed is None:
        print("usage: start_variant.py <recording.json> [--cycle <CYCLE>]",
              file=sys.stderr)
        return 2
    arg_path, cycle = parsed
    path = pathlib.Path(arg_path)
    try:
        doc = json.loads(path.read_text())
        pose = last_pose(doc)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"start_variant": None, "cycle": cycle, "error": str(exc)}, indent=2))
        return 1
    variant = plan.classify_start_variant(pose)
    print(json.dumps({"start_variant": variant, "pose": pose, "cycle": cycle}, indent=2))
    return 0 if variant is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
