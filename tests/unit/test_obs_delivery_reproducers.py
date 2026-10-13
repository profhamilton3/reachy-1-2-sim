"""Phase 1.2 reproducers for the observation-delivery delay (attempt 5 of
the 2026-10-08 crane measurement halted on "feedback stale (0.67 s)").

They were written first, against main, where all but the control failed
(Phase 1.2); Phase 1.3 makes them pass, and they stay as its regression
tests.  Each drives the real code path -- SimLink ingest, the executor's
feed, MotionWorker.run / _send_observation, and the worker's _reader and
_observer -- with every module's `time` replaced by one fake clock.  Nothing
sleeps.  (T1 waits on a threading.Event for a delivery that main never
makes; that wait is bounded and is the only real-time step.)

  T1  forwarding is independent of the job loop: while the loop is blocked
      in should_cancel (the task lock held), fresh snapshots still reach the
      worker.
  T2  the worker's age is honest: pipe/reader delay is counted, not hidden.
  T3  staleness is reported explicitly, naming which stamp is old.
  T4  after a worker-wide pause (gen-2 GC), the reader drains to the latest:
      one publication (the newest), control messages kept, end-to-end age.
"""
import io
import json
import os
import queue
import sys
import threading
import types

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../../web"))
sys.path.insert(0, os.path.join(_HERE, "../../native_mujoco"))
sys.path.insert(0, os.path.join(_HERE, "../../src"))

import motion_worker as W  # noqa: E402
import panel_executor as PE  # noqa: E402
import panel_sim_link as SL  # noqa: E402

OID = "red_cube"
STALE_S = 0.5            # crane_pick_live.FEEDBACK_STALE_S (unchanged)


class _Clock:
    def __init__(self):
        self.now = 1000.0

    def advance(self, s):
        self.now += s


@pytest.fixture
def clock(monkeypatch):
    c = _Clock()

    def fake_time():
        def _sleep(_s):
            raise AssertionError("reproducers never sleep")
        return types.SimpleNamespace(
            monotonic=lambda: c.now, monotonic_ns=lambda: int(c.now * 1e9),
            time=lambda: c.now, sleep=_sleep)

    for mod in (W, PE, SL):
        monkeypatch.setattr(mod, "time", fake_time())
    monkeypatch.setattr(W, "OBSERVATIONS", W.Observations())
    return c


class _Link(SL.SimLink):
    """The real SimLink, fed state messages directly (no socket)."""

    def __init__(self):
        super().__init__("ws://unused")
        self._seq = 0

    def publish(self):
        self._seq += 1
        self._ingest_state({
            "type": "state", "seq": self._seq, "sim_step": 10 * self._seq,
            "sim_time_s": 0.02 * self._seq, "scene_revision": "r",
            "objects": [{"object_id": OID, "pos_xyz": [0.43, 0.0, 0.77],
                         "quat_wxyz": [1.0, 0.0, 0.0, 0.0]}],
            "grippers": [{"side": "right", "grip_force_n": 0.0, "grasping": False}],
        })


def _feed(link):
    return PE.SimulatorExecutor._observation_feed(types.SimpleNamespace(_link=link), OID)


class _Stdin:
    """The child's stdin.  `deliver` decides when a written line reaches the
    worker's real reader; by default immediately (a perfectly scheduled
    reader)."""

    def __init__(self, deliver=True):
        self.lines, self.deliver = [], deliver
        self.arrived = threading.Event()

    def write(self, s):
        self.lines.append(s)
        if self.deliver and '"observe"' in s:
            W._reader(io.StringIO(s), queue.Queue(), threading.Event())
            self.arrived.set()

    def flush(self):
        pass


def _worker(stdin):
    mw = PE.MotionWorker(deadline_s=60.0)
    mw._proc = types.SimpleNamespace(stdin=stdin, poll=lambda: None)
    mw._lines = queue.Queue()
    return mw


def test_t1_forwarding_continues_while_the_job_loop_is_blocked(clock):
    link, stdin = _Link(), _Stdin()
    mw = _worker(stdin)
    observe = W._observer(OID)
    link.publish()
    seen = {"calls": 0, "age_after_block": None}

    def should_cancel():
        seen["calls"] += 1
        if seen["calls"] == 2:
            # The task lock is held for 0.6 s (another thread), while the
            # simulator keeps publishing at 10 Hz.
            stdin.arrived.clear()
            for _ in range(6):
                clock.advance(0.1)
                link.publish()
            stdin.arrived.wait(timeout=2.0)   # main: nothing forwards here
            seen["age_after_block"] = observe().age_s
        if seen["calls"] == 3:
            mw._lines.put(json.dumps({"result": {"status": "moved"}}) + "\n")
        return False

    out = mw.run({"kind": "ability"}, should_cancel=should_cancel, feed=_feed(link))
    assert out == {"status": "moved"}
    assert seen["age_after_block"] is not None
    assert seen["age_after_block"] < 0.1, (
        f"worker saw a {seen['age_after_block']:.2f} s old observation although "
        "the simulator published every 0.1 s: forwarding stalled with the job loop")


def test_t2_pipe_and_reader_delay_is_counted_in_the_age(clock):
    link, stdin = _Link(), _Stdin(deliver=False)
    mw = _worker(stdin)
    link.publish()                           # ingested at t0
    obs = _feed(link)()
    assert mw._send_observation(obs)         # written at t0
    line = stdin.lines[-1]
    clock.advance(0.6)                       # the line waits in the pipe / for the reader
    W._reader(io.StringIO(line), queue.Queue(), threading.Event())
    age = W._observer(OID)().age_s
    assert age >= 0.6 - 1e-6, (
        f"reported age {age:.3f} s for an observation ingested 0.6 s ago: "
        "the delay between send and receipt is not counted")


def test_t3a_no_observation_is_an_explicit_stale_status(clock):
    o = W._observer(OID)()
    assert getattr(o, "stale_reason", None) == "no observation"


def test_t3b_late_delivery_names_the_delivery_stamp(clock):
    link, stdin = _Link(), _Stdin(deliver=False)
    mw = _worker(stdin)
    link.publish()
    assert mw._send_observation(_feed(link)())
    clock.advance(0.6)
    W._reader(io.StringIO(stdin.lines[-1]), queue.Queue(), threading.Event())
    o = W._observer(OID)()
    assert o.age_s > STALE_S
    assert getattr(o, "stale_reason", None) == "delivery"


def test_t3c_no_update_since_receipt_names_the_receipt_stamp(clock):
    link, stdin = _Link(), _Stdin()
    mw = _worker(stdin)
    link.publish()
    assert mw._send_observation(_feed(link)())   # delivered at once
    clock.advance(0.6)                           # nothing newer arrives
    o = W._observer(OID)()
    assert o.age_s > STALE_S
    assert getattr(o, "stale_reason", None) == "no update"


def test_t3_fresh_observation_has_no_stale_reason(clock):
    """Control: must pass on main and after the fix."""
    link, stdin = _Link(), _Stdin()
    mw = _worker(stdin)
    link.publish()
    assert mw._send_observation(_feed(link)())
    o = W._observer(OID)()
    assert o.age_s < 0.05
    assert getattr(o, "stale_reason", None) in (None, "")


def test_t4_after_a_worker_wide_pause_the_reader_drains_to_the_latest(clock, monkeypatch):
    """T4 (the H3 mechanism, drain-to-latest semantics).

    A gen-2 GC pause stops every thread in the worker -- reader and main --
    for ~0.6 s, while SimLink keeps ingesting and the panel keeps writing.
    When the worker resumes, the backlog is already in the pipe.  The reader
    must consume everything available in one drain, publish ONLY the newest
    observation (no walk through stale intermediates), keep every control
    message (a cancel in the backlog is not dropped), and the age the first
    `observe()` reports must be end-to-end: since SimLink ingested the newest
    line, not since the reader got to it.
    """
    link, stdin = _Link(), _Stdin(deliver=False)
    mw = _worker(stdin)
    published = []
    real_put = W.OBSERVATIONS.put
    monkeypatch.setattr(W.OBSERVATIONS, "put",
                        lambda obs, *a, **k: (published.append(obs.get("seq")),
                                              real_put(obs, *a, **k))[1])
    feed = _feed(link)

    # Before the pause: one observation delivered normally.
    link.publish()
    assert mw._send_observation(feed())
    W._reader(io.StringIO(stdin.lines.pop()), queue.Queue(), threading.Event())
    published.clear()

    # The pause: 6 publishes at 0.1 s, each forwarded by the panel at once,
    # with a cancel queued in the middle; nothing in the worker runs.
    for i in range(6):
        clock.advance(0.1)
        link.publish()
        assert mw._send_observation(feed())
        if i == 2:
            assert mw._send_cancel()
    newest_seq = link._seq
    clock.advance(0.05)                 # the worker resumes 50 ms after the last ingest

    r, w = os.pipe()
    with os.fdopen(w, "w") as pipe_in:
        pipe_in.write("".join(stdin.lines))
    jobs, cancel = queue.Queue(), threading.Event()
    with os.fdopen(r, "r") as pipe_out:   # what main() hands the reader (text, like sys.stdin)
        W._reader(pipe_out, jobs, cancel)

    assert cancel.is_set(), "drain-to-latest dropped a control message"
    assert published == [newest_seq], (
        f"the reader published {published} for one backlog; drain-to-latest "
        f"publishes only the newest (seq {newest_seq})")
    o = W._observer(OID)()
    assert 0.05 - 1e-6 <= o.age_s < 0.1, (
        f"first observe() after the pause reports {o.age_s:.3f} s; the newest line "
        "was ingested 0.05 s before the worker resumed (age must be end-to-end)")
    assert getattr(o, "stale_reason", None) in (None, "")
