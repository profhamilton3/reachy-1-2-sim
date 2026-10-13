"""Latest-wins delivery of live observations across a thread or process hop.

Three pieces, each usable alone:

* ``LatestSlot`` -- one value, overwritten by every newer one; a reader never
  walks a backlog.  Each value carries ``ingest_mono``: when the newest data
  entered THIS HOST (the moment the source's message was parsed), so the age
  a reader computes is end to end -- pipe, queue and scheduling delay
  included -- instead of "since it reached me".
* ``stale_reason`` -- an explicit status instead of a bare number: ``None``
  (fresh), ``"no observation"``, ``"delivery"`` (it was already old when it
  arrived: the hop is slow) or ``"no update"`` (it arrived promptly and
  nothing newer has come: the source or the sender is quiet).
* ``LineDrain`` + ``newest_of`` -- for a newline-delimited pipe: take
  everything already available in one read, keep every control message in
  order, and only the newest observation (drain-to-latest).

The module never reads a clock.  Callers pass ``time.monotonic()`` readings,
which keeps it testable with a fake clock and keeps the clock rule visible
at every call site.  Standard library only; no simulator, panel or robot
assumptions.

PORTING NOTE (reachy-tabletop-ai stereo-viewer panel, workstation-hosted,
robot and simulator sources):
  1. ``ingest_mono`` must be stamped by a process on the SAME machine as the
     reader: CLOCK_MONOTONIC is per host and per boot.  Stamp when the
     workstation parses the robot's (or the simulator's) message; never
     subtract a robot-side or simulator-side timestamp from a workstation
     one.  Hop 0 (source -> workstation transit) is therefore not in the
     age; measure it separately if it matters.
  2. One ``LatestSlot`` per source (robot, simulator).  Do not share one
     slot between sources: "newest" is only meaningful within one stream.
  3. ``LineDrain`` needs a pipe-like object with ``fileno()`` (os.read +
     select); on a platform without select on pipes (Windows), fall back to
     a reader thread feeding a ``LatestSlot`` directly.
  4. Keep the staleness limit in the consumer (the motion guard owns it);
     this module only reports age and the reason.
"""
from __future__ import annotations

import io
import os
import select
import threading
from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Tuple

NO_OBSERVATION = "no observation"
DELIVERY = "delivery"
NO_UPDATE = "no update"


@dataclass(frozen=True)
class Reading:
    value: Any
    ingest_mono: float      # when the data entered this host
    received_mono: float    # when it reached the slot

    def age_s(self, now_mono: float) -> float:
        return max(0.0, now_mono - self.ingest_mono)


def stale_reason(reading: Optional[Reading], now_mono: float,
                 limit_s: float) -> Optional[str]:
    """Why ``reading`` is not live, or None if it is.  When it is stale, the
    larger of the two delays is named: before receipt (``delivery``) or
    since receipt (``no update``)."""
    if reading is None:
        return NO_OBSERVATION
    if reading.age_s(now_mono) <= limit_s:
        return None
    transit = reading.received_mono - reading.ingest_mono
    waiting = now_mono - reading.received_mono
    return DELIVERY if transit >= waiting else NO_UPDATE


class LatestSlot:
    """A single latest-wins value, safe across threads."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reading: Optional[Reading] = None
        self._unread = False
        self.overwritten = 0        # values replaced before anyone read them

    def put(self, value: Any, ingest_mono: float, received_mono: float) -> None:
        with self._lock:
            if self._unread:
                self.overwritten += 1
            self._reading = Reading(value, ingest_mono, received_mono)
            self._unread = True

    def latest(self) -> Optional[Reading]:
        with self._lock:
            self._unread = False
            return self._reading


class LineDrain:
    """Everything already written to a pipe, as complete lines, per call.

    ``next_batch`` blocks until something arrives, then also takes whatever
    else is already there without waiting.  A partial last line is kept for
    the next call.  Returns ``(lines, eof)``.  Streams without a file
    descriptor (tests) are read whole.
    """

    def __init__(self, stream) -> None:
        self._stream = stream
        try:
            self._fd: Optional[int] = stream.fileno()
        except (AttributeError, OSError, io.UnsupportedOperation):
            self._fd = None
        self._carry = b""

    def next_batch(self) -> Tuple[List[str], bool]:
        if self._fd is None:
            return [l for l in self._stream.read().splitlines() if l.strip()], True
        chunks, eof = [], False
        data = os.read(self._fd, 65536)            # blocks until data or EOF
        while data:
            chunks.append(data)
            if not select.select([self._fd], [], [], 0)[0]:
                break
            data = os.read(self._fd, 65536)
        else:
            eof = True
        buf = self._carry + b"".join(chunks)
        *complete, self._carry = buf.split(b"\n")
        if eof and self._carry:
            complete.append(self._carry)
            self._carry = b""
        lines = [l.decode("utf-8", "replace") for l in complete if l.strip()]
        return lines, eof


def newest_of(messages: Iterable[Any], is_observation: Callable[[Any], bool]
              ) -> Tuple[List[Any], Optional[Any], int]:
    """Split a drained batch: control messages in order, the newest
    observation, and how many older observations were skipped."""
    controls, newest, skipped = [], None, 0
    for m in messages:
        if is_observation(m):
            skipped += newest is not None
            newest = m
        else:
            controls.append(m)
    return controls, newest, skipped
