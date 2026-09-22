#!/usr/bin/env python3
"""Fake `docker` for `tests/unit/test_e1_reset_verify.py`'s
`TestResetShEndToEnd` (§4.F): a copy of this file, named `docker` and put
first on `PATH`, stands in for `docker compose exec -T reachy-sim sh -c
'...'` -- both calls `scripts/e1_stage1/reset.sh` makes (the sentinel
write and the ack poll) -- without a container, so the test runs the real
`reset.sh` end to end (`bash scripts/e1_stage1/reset.sh ...`) with no
Docker, server, or SDK.

Recognizes only the two invocations `reset.sh` makes, by substring match
over the full argv (not just the last element): the request-publish call
now reads (E1 Stage 2 B2 s1 reset 1 fix, 2026-09-22) `sh -c 'SCRIPT' sh
"$GEN"` -- `$GEN` is a trailing positional argument, not interpolated
into the `-c` script text, so it is `sys.argv[-1]` directly rather than
regex-extracted from the script string as before. The ack-read call is
unchanged (`sh -c 'cat ...'`, a single string containing
`reachy_reset_ack`). Anything else exits 1, so an unexpected caller shows
up as a Docker failure rather than a silent no-op.

Controlled by environment variables (all required except the mode):
  FAKE_DOCKER_RUN_DIR   the native-server run dir reset.sh operates on.
  FAKE_DOCKER_ACK_FILE  a scratch file this script uses to remember the
                         requested generation between the two calls (the
                         real ack file lives inside the container).
  FAKE_DOCKER_MODE      "ok"     (default) -- the sentinel write appends
                                  a reset line (sim_step SNAPSHOT_STEP+10)
                                  and one complete post-reset state
                                  (sim_step 10) to the run dir, then acks
                                  immediately.
                         "no_append" -- acks immediately but appends
                                  nothing (`reset not recorded`).
                         "torn_growing" -- appends the reset line, spawns
                                  `e1_reset_growth_writer.py` (detached --
                                  it outlives this process) to grow
                                  states.jsonl with a torn tail while
                                  `verify` polls, and acks immediately.
                         "publish_fail" -- the request-publish call itself
                                  exits non-zero (simulating a dropped
                                  `docker compose exec`, not a sentinel
                                  race): nothing is appended, no ack is
                                  written, and reset.sh must STOP without
                                  ever polling for an ack.
  FAKE_DOCKER_SNAPSHOT_STEP  the sim_step the test's fixture snapshot was
                         written with (default 5000) -- the reset line
                         and post-reset states are built relative to it
                         so they satisfy reset_verify.py's ordering
                         checks regardless of the fixture's exact values.
"""
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_GROWTH_WRITER = os.path.join(_HERE, "e1_reset_growth_writer.py")


def _joints(n=21):
    return [{"name": f"j{i}", "position_rad": 0.0, "compliant": False,
              "effort": 0.0} for i in range(n)]


def _state_line(seq, sim_step, wall_time_ns):
    doc = {"type": "state", "seq": seq, "sim_step": sim_step,
           "sim_time_s": sim_step * 0.05, "wall_time_ns": wall_time_ns,
           "cmd_seq": 0, "scene_revision": "r1", "paused": False,
           "joints": _joints(), "objects": [], "grippers": [],
           "force_sensors": [], "interactive": [], "warnings": [],
           "contacts": []}
    return json.dumps(doc, separators=(",", ":")) + "\n"


def main() -> int:
    run_dir = os.environ["FAKE_DOCKER_RUN_DIR"]
    ack_file = os.environ["FAKE_DOCKER_ACK_FILE"]
    mode = os.environ.get("FAKE_DOCKER_MODE", "ok")
    snapshot_step = int(os.environ.get("FAKE_DOCKER_SNAPSHOT_STEP", "5000"))

    argv = sys.argv[1:]
    joined = " ".join(argv)

    if "reachy_reset_request" in joined:
        gen = argv[-1] if argv else ""
        if mode == "publish_fail":
            return 3
        if mode in ("ok", "torn_growing"):
            with open(os.path.join(run_dir, "commands.jsonl"), "a") as f:
                f.write(json.dumps(
                    {"type": "reset", "seed": None,
                     "sim_step": snapshot_step + 10, "wall_time_s": 1.0},
                    separators=(",", ":")) + "\n")
        if mode == "ok":
            with open(os.path.join(run_dir, "states.jsonl"), "a") as f:
                f.write(_state_line(1_000_001, 10, time.monotonic_ns()))
        elif mode == "torn_growing":
            subprocess.Popen(
                [sys.executable, _GROWTH_WRITER,
                 os.path.join(run_dir, "states.jsonl"), "1000001", "10", "20", "0.02"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        with open(ack_file, "w") as f:
            f.write(gen)
        return 0

    if "reachy_reset_ack" in joined:
        try:
            with open(ack_file) as f:
                sys.stdout.write(f.read())
        except OSError:
            pass
        return 0

    sys.stderr.write(f"e1_reset_fake_docker: unrecognized invocation: {sys.argv!r}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
