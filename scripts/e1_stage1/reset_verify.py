"""Robust reset verification for `reset.sh` (E1 Stage 2 B1 `reset_18`,
`docs/reviews/probes-2026-09-19-e1-stage2-b1/control/reset_18.txt`,
`stop_analysis.txt`): the script's old `tail -1 states.jsonl | python3 -c
'...json.loads(...)'` read a growing (~20 Hz, ~5.9 KB/record) file and
could catch two records in one read (`Extra data`) or a torn last line,
turning a correctly-applied reset into a false STOP. This module makes the
*read* robust without making the *verification* more permissive: every
failure mode below still STOPs, just with a specific reason instead of a
bash arithmetic error.

`snapshot <run_dir>` records the state immediately before a reset is
requested. `verify <run_dir> <gen> <ack> <snapshot_json>` re-reads the
record after the ack and exits 0 only for fresh, live, unambiguous
evidence that exactly this reset happened -- see each check's comment
below for what it rules out.

Reuses `e1_identity._read_last_state` (tail-anchored, tolerates a torn
last line from a concurrent writer, fails closed to `None`) for every
states.jsonl read -- do not re-implement that reader here.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from typing import List, Optional, Tuple

_HERE = pathlib.Path(__file__).resolve().parent
_SCRIPTS = _HERE.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import e1_identity  # noqa: E402

_read_last_state = e1_identity._read_last_state

DEFAULT_TIMEOUT_S = 10.0
DEFAULT_POLL_S = 0.25
DEFAULT_MAX_STATE_AGE_S = 1.0

#: Literal substring `record_reset`/`record_command` write via
#: `json.dumps(..., separators=(",", ":"))` -- no space after the colon.
#: Matches the original `reset.sh`'s `grep -c '"type":"reset"'` exactly,
#: so a torn reset line (cut after `"type":"reset"` but before `sim_step`)
#: still counts as "a reset happened" for the count check, distinct from
#: it being unparseable for the sim_step check.
_RESET_MARKER = b'"type":"reset"'

#: Reasons that cannot resolve with more polling -- the count can only
#: grow, so ack mismatch and a second reset are permanent, checked once.
_TERMINAL_REASONS = frozenset({"ack mismatch", "more than one reset recorded"})


class VerifyFailed(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


def _is_strict_int(value: object) -> bool:
    """True int only -- `bool` is a subtype of `int` in Python but must be
    rejected explicitly (issue #130), and this also rejects `float` (so a
    JSON `NaN`, which `json.loads` parses to `float('nan')`, is rejected
    the same way without special-casing it)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _require_state_fields(state: dict) -> Tuple[int, int, int]:
    for name in ("seq", "sim_step", "wall_time_ns"):
        if not _is_strict_int(state.get(name)):
            raise VerifyFailed(f"state malformed: {name}")
    return state["seq"], state["sim_step"], state["wall_time_ns"]


def _reset_lines(commands_path: pathlib.Path) -> List[str]:
    """Every line of `commands.jsonl` containing the reset marker, in file
    order. A full-file read: `commands.jsonl` is not the file this
    assignment's growing-file defect is about (states.jsonl, appended at
    ~20 Hz; commands.jsonl gains a line per reset and per joint command
    only while a leg is flying, and reset.sh runs between legs)."""
    try:
        data = commands_path.read_bytes()
    except OSError:
        return []
    return [raw.decode("utf-8", "replace")
            for raw in data.split(b"\n") if _RESET_MARKER in raw]


def snapshot(run_dir: pathlib.Path) -> dict:
    """The state to compare a later reset's evidence against."""
    state = _read_last_state(run_dir / "states.jsonl")
    if state is None:
        raise VerifyFailed("no complete state")
    seq, sim_step, wall_time_ns = _require_state_fields(state)
    reset_count = len(_reset_lines(run_dir / "commands.jsonl"))
    return {"seq": seq, "sim_step": sim_step, "wall_time_ns": wall_time_ns,
            "reset_count": reset_count}


def _evaluate(
    run_dir: pathlib.Path, snap: dict, max_state_age_s: float,
) -> Tuple[bool, Optional[str], Optional[dict]]:
    """One (non-polling) attempt. Returns `(ok, reason, result)`; `result`
    is set only when `ok` is True."""
    before = snap["reset_count"]
    reset_lines = _reset_lines(run_dir / "commands.jsonl")
    after = len(reset_lines)
    if after < before + 1:
        return False, "reset not recorded", None
    if after > before + 1:
        return False, "more than one reset recorded", None

    try:
        reset_obj = json.loads(reset_lines[-1])
    except json.JSONDecodeError:
        return False, "reset line malformed", None
    reset_sim_step = reset_obj.get("sim_step")
    if not _is_strict_int(reset_sim_step) or reset_sim_step < snap["sim_step"]:
        return False, "reset line malformed", None

    state = _read_last_state(run_dir / "states.jsonl")
    if state is None:
        return False, "no complete state", None
    try:
        seq, sim_step, wall_time_ns = _require_state_fields(state)
    except VerifyFailed as exc:
        return False, exc.reason, None

    # Fresh evidence: a sample recorded strictly after the snapshot, by
    # both the server's monotonic sequence and its own wall clock.
    if not (seq > snap["seq"] and wall_time_ns > snap["wall_time_ns"]):
        return False, "state not fresh", None
    # ...whose step counter actually restarted -- below the snapshot AND
    # below the reset line's own sim_step, so a sample from an earlier,
    # still-in-flight reset can't pass as evidence of this one.
    if not (sim_step < snap["sim_step"] and sim_step < reset_sim_step):
        return False, "sim_step not restarted", None
    age_s = abs(time.time_ns() - wall_time_ns) / 1e9
    if age_s > max_state_age_s:
        return False, f"state stale (age {age_s:.3f}s)", None

    return True, None, {
        "reset_count_before": before, "reset_count_after": after,
        "sim_step_before": snap["sim_step"], "sim_step_after": sim_step,
    }


def verify(
    run_dir: pathlib.Path, gen: str, ack: str, snap: dict, *,
    timeout_s: float = DEFAULT_TIMEOUT_S, poll_s: float = DEFAULT_POLL_S,
    max_state_age_s: float = DEFAULT_MAX_STATE_AGE_S,
) -> dict:
    if ack != gen:
        raise VerifyFailed("ack mismatch")
    deadline = time.monotonic() + timeout_s
    while True:
        ok, reason, result = _evaluate(run_dir, snap, max_state_age_s)
        if ok:
            return result
        if reason in _TERMINAL_REASONS or time.monotonic() >= deadline:
            raise VerifyFailed(reason)
        time.sleep(poll_s)


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print("usage: reset_verify.py snapshot|verify ...", file=sys.stderr)
        return 2

    if argv[0] == "snapshot":
        parser = argparse.ArgumentParser(prog="reset_verify.py snapshot")
        parser.add_argument("run_dir")
        args = parser.parse_args(argv[1:])
        try:
            snap = snapshot(pathlib.Path(args.run_dir))
        except VerifyFailed as exc:
            print(exc.reason)
            return 9
        print(json.dumps(snap, separators=(",", ":")))
        return 0

    if argv[0] == "verify":
        parser = argparse.ArgumentParser(prog="reset_verify.py verify")
        parser.add_argument("run_dir")
        parser.add_argument("gen")
        parser.add_argument("ack")
        parser.add_argument("snapshot_json")
        parser.add_argument("--timeout-s", type=float, default=DEFAULT_TIMEOUT_S)
        parser.add_argument("--poll-s", type=float, default=DEFAULT_POLL_S)
        parser.add_argument("--max-state-age-s", type=float,
                             default=DEFAULT_MAX_STATE_AGE_S)
        args = parser.parse_args(argv[1:])
        snap = json.loads(args.snapshot_json)
        try:
            result = verify(
                pathlib.Path(args.run_dir), args.gen, args.ack, snap,
                timeout_s=args.timeout_s, poll_s=args.poll_s,
                max_state_age_s=args.max_state_age_s)
        except VerifyFailed as exc:
            print(exc.reason)
            return 9
        print(f"reset gen={args.gen} ack={args.ack} resets_recorded "
              f"{result['reset_count_before']}->{result['reset_count_after']} "
              f"sim_step {result['sim_step_before']}->{result['sim_step_after']}")
        return 0

    print(f"unknown subcommand: {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
