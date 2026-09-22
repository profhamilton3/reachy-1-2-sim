"""The container-side half of the acknowledged-reset sentinel protocol
(see scripts/e1_stage1/reset.sh, the client half, for the full contract
and its own header for the fix this module implements).

`reset_watcher` runs as a background thread inside the simulator
container (`fake_reachy_server.py`, mujoco-remote backend only) and polls
for a generation token dropped at `REACHY_SIM_RESET_REQUEST` (default
/tmp/reachy_reset_request). On a valid token it triggers an acknowledged
physics reset and publishes the same token to `REACHY_SIM_RESET_ACK`
(default /tmp/reachy_reset_ack) so `reset.sh` can confirm completion
without a fixed sleep.

Kept dependency-light (stdlib only) and separate from fake_reachy_server.py
(which pulls in grpc/reachy_sdk_api/scipy/numpy and the rest of the fake
gRPC servicer) so this half of the protocol -- and the race it used to
have -- is unit testable offline, the same way reset_verify.py is kept
separate from native_mujoco/server.py so it's testable without a server
or SDK. See tests/unit/test_reset_sentinel_protocol.py.

E1 Stage 2 B2 s1 reset 1 (2026-09-22,
docs/reviews/probes-2026-09-22-e1-stage2-b2/control/stop_analysis.txt):
reset.sh used to publish the request with a plain `echo $GEN > file`,
which truncates the target before the write completes. A poll landing in
that window read gen="", and the OLD watcher consumed it unconditionally
-- triggering an unrequested reset and a 0-byte ack, which reset.sh read
back as `ack mismatch` and STOPped. Two independent fixes close this:
reset.sh now publishes atomically (temp file + same-directory rename --
see its own header), and this module never acts on anything that doesn't
read back as a complete, valid generation (`VALID_GEN_RE`) -- so it is
safe even against a non-atomic publisher (scripts/demo_control_panel.py,
notebooks/*_training.ipynb still write the old way, an empty/partial
read from either is simply left for a later poll instead of being
treated as "no request, discard it").

`VALID_GEN_RE` is deliberately not digit-only: reset.sh and
demo_control_panel.py publish decimal counters (`str(N)` /
`str(time.monotonic_ns())`), but both notebooks/*_training.ipynb publish
`uuid.uuid4().hex` (32 lowercase hex chars) instead -- a real producer,
not a hypothetical one, and PR #135 as first merged (`88c31d2`)
restricted the regex to `^[0-9]+$`, which silently broke the notebooks'
`reset_scene()` handshake (uuid tokens were read back, never matched,
and left on disk forever -- no reset, no ack, nothing above `debug`).
The regex now accepts any bounded alphanumeric token so it validates
*shape* (non-empty, no embedded whitespace/control bytes, no
torn-together concatenation of two generations, capped length) rather
than a specific producer's format -- see the B1 finding in the PR #135
review for the "notebooks unaffected" claim this corrects.
"""
from __future__ import annotations

import logging
import os
import pathlib
import re
import time
from typing import Optional

log = logging.getLogger(__name__)

# Distinct non-empty malformed values already warned about, so a stuck or
# broken publisher (e.g. writing something outside VALID_GEN_RE on every
# poll) logs once per distinct bad value instead of flooding at 10 Hz.
_WARNED_MALFORMED: set = set()

REQUEST_PATH = os.environ.get(
    "REACHY_SIM_RESET_REQUEST", "/tmp/reachy_reset_request"
)
ACK_PATH = os.environ.get("REACHY_SIM_RESET_ACK", "/tmp/reachy_reset_ack")

# A published generation is a bounded, opaque token: either the decimal
# counter string reset.sh's $GEN argument (scripts/e1_stage1/plan.py's
# Stage 2 gens, 1..N) and demo_control_panel.py's
# str(time.monotonic_ns()) both are, or the uuid.uuid4().hex (32
# lowercase hex chars) both notebooks/*_training.ipynb publish. This
# module never interprets the token, only matches it back on the ack, so
# it doesn't need to know which producer sent it -- just that a read is a
# complete, plausible token and not an empty/partial/torn one. Anything
# outside the charset (whitespace, control bytes, a second generation
# concatenated onto this one) or the length cap must never trigger a
# physics reset or an acknowledgement a caller's generation match could
# be satisfied by.
VALID_GEN_RE = re.compile(r"^[0-9A-Za-z]{1,64}$")


def atomic_write_text(path: pathlib.Path, text: str) -> None:
    """Write `text` to `path` via a same-directory temp file + `os.replace`,
    so a concurrent reader (reset.sh's ack poll) only ever observes the
    previous complete content or the new complete content, never a
    truncated or partially written file."""
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def reset_watcher_step(
    remote, req_path: pathlib.Path, ack_path: pathlib.Path
) -> Optional[str]:
    """One poll of the sentinel-file reset protocol. Returns the
    generation that was processed, or `None` if nothing was (no request
    file present, or one that read empty/malformed and was left
    untouched for a later poll).

    A request is consumed (unlinked, reset triggered, ack published) only
    when it reads back as a complete, valid generation (`VALID_GEN_RE`).
    An empty or malformed read is left in place, untouched: a genuine
    request either completes on a later poll (a writer that creates then
    writes the file in two steps -- reset.sh used to, see its header) or
    was never meant for this watcher; either way it is never treated as
    "no reset requested, discard it" the way an unconditional unlink
    would. See the module docstring for the bug this closes.
    """
    if not req_path.exists():
        return None
    try:
        raw = req_path.read_text()
    except OSError:
        return None
    gen = raw.strip()
    if not VALID_GEN_RE.match(gen):
        if gen and gen not in _WARNED_MALFORMED:
            # Non-empty and invalid: either a broken/misconfigured
            # publisher or a read caught mid-write by a non-atomic
            # writer. An empty read (the ordinary truncate-then-write
            # window) stays at debug -- it's expected, not a signal.
            _WARNED_MALFORMED.add(gen)
            log.warning("reset_watcher: ignoring malformed request %r", raw)
        else:
            log.debug("reset_watcher: ignoring empty/malformed request %r", raw)
        return None
    req_path.unlink(missing_ok=True)
    event = remote.request_reset()
    ok = event.wait(timeout=6.0)
    if not ok:
        log.warning("reset_watcher: reset ack timed out")
    # Write the ack generation (even on timeout so the demo doesn't hang).
    try:
        atomic_write_text(ack_path, gen)
    except OSError:
        pass
    return gen


def reset_watcher(remote, path: str = REQUEST_PATH, hz: float = 10.0) -> None:
    """Trigger an acknowledged physics reset when a client publishes a
    sentinel file. Polls at `hz`, delegating each poll to
    `reset_watcher_step` (the unit tested half -- see its docstring and
    the module docstring for the race/validation contract).
    """
    period = 1.0 / hz
    req_path = pathlib.Path(path)
    ack_path = pathlib.Path(ACK_PATH)
    while True:
        try:
            reset_watcher_step(remote, req_path, ack_path)
        except Exception:
            log.debug("reset_watcher error", exc_info=True)
        time.sleep(period)
