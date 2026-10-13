"""The readiness-wait every generated notebook cell polls on, pulled out of
the cell-source template (previously an inline closure only exercisable by
`exec`-ing a whole cell) so its stop-takes-priority-over-any-marker behavior
is directly unit-testable (decision note
outputs/e1-stage2-decision-2026-09-15.md §6: "initialization failure
preventing progression"). No behaviour change from the inline version this
replaces: `wait_for` still checks `control/stop` first on every poll, before
the caller's own predicate.
"""
from __future__ import annotations

import pathlib
import time
from typing import Callable


def wait_for(
    ctrl_dir: pathlib.Path,
    pred: Callable[[], bool],
    timeout_s: float,
    period: float = 0.25,
    *,
    now: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    """`"stop"` if `ctrl_dir/stop` exists (checked before `pred`, every
    poll -- a stop written by an earlier leg, cycle, or Stage 0's policy-A
    setup action outranks any later marker, including one written after
    the stop); `"ready"` once `pred()` is true; `"timeout"` after
    `timeout_s`."""
    t0 = now()
    while now() - t0 < timeout_s:
        if (ctrl_dir / "stop").exists():
            return "stop"
        if pred():
            return "ready"
        sleep(period)
    return "timeout"
