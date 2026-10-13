"""`reachy_ai.feedback.latest`: latest-wins slot, stale reasons, pipe drain,
and the panel side's non-blocking forwarding that uses them."""
import io
import json
import os
import queue
import sys
import threading
import time
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../web"))

from reachy_ai.feedback import latest as L  # noqa: E402

import panel_executor as PE  # noqa: E402


# -- LatestSlot / Reading ------------------------------------------------------

def test_slot_keeps_only_the_newest_and_counts_unread_overwrites():
    s = L.LatestSlot()
    assert s.latest() is None
    s.put("a", ingest_mono=1.0, received_mono=1.1)
    s.put("b", ingest_mono=2.0, received_mono=2.1)      # "a" never read
    r = s.latest()
    assert (r.value, r.ingest_mono, r.received_mono) == ("b", 2.0, 2.1)
    s.put("c", ingest_mono=3.0, received_mono=3.1)      # "b" was read
    assert s.overwritten == 1


def test_age_is_from_ingest_not_receipt():
    r = L.Reading("x", ingest_mono=10.0, received_mono=10.4)
    assert r.age_s(10.5) == 0.5
    assert r.age_s(9.0) == 0.0                           # never negative


# -- stale_reason ----------------------------------------------------------------

def test_stale_reasons():
    assert L.stale_reason(None, 5.0, 0.5) == L.NO_OBSERVATION
    fresh = L.Reading("x", 10.0, 10.01)
    assert L.stale_reason(fresh, 10.2, 0.5) is None
    late = L.Reading("x", 10.0, 10.6)                    # arrived already old
    assert L.stale_reason(late, 10.6, 0.5) == L.DELIVERY
    quiet = L.Reading("x", 10.0, 10.01)                  # arrived fresh, nothing since
    assert L.stale_reason(quiet, 10.7, 0.5) == L.NO_UPDATE


def test_exactly_at_the_limit_is_fresh():
    assert L.stale_reason(L.Reading("x", 10.0, 10.0), 10.5, 0.5) is None


# -- LineDrain / newest_of ---------------------------------------------------------

def test_drain_takes_everything_available_and_carries_a_partial_line():
    r, w = os.pipe()
    with os.fdopen(r, "r") as rs:
        drain = L.LineDrain(rs)
        os.write(w, b'{"a": 1}\n{"b": 2}\n{"c"')
        lines, eof = drain.next_batch()
        assert lines == ['{"a": 1}', '{"b": 2}'] and not eof
        os.write(w, b': 3}\n')
        lines, eof = drain.next_batch()
        assert lines == ['{"c": 3}'] and not eof
        os.close(w)
        lines, eof = drain.next_batch()
        assert lines == [] and eof


def test_drain_returns_an_unterminated_last_line_at_eof():
    r, w = os.pipe()
    os.write(w, b'{"a": 1}\n{"b": 2}')
    os.close(w)
    with os.fdopen(r, "r") as rs:
        lines, eof = L.LineDrain(rs).next_batch()
    assert lines == ['{"a": 1}', '{"b": 2}'] and eof


def test_drain_reads_a_stream_without_a_descriptor_whole():
    lines, eof = L.LineDrain(io.StringIO('{"a": 1}\n\n{"b": 2}\n')).next_batch()
    assert lines == ['{"a": 1}', '{"b": 2}'] and eof


def test_newest_of_keeps_controls_in_order_and_one_observation():
    msgs = [{"observe": 1}, {"cancel": True}, {"observe": 2}, {"job": "j"}, {"observe": 3}]
    controls, newest, skipped = L.newest_of(msgs, lambda m: "observe" in m)
    assert controls == [{"cancel": True}, {"job": "j"}]
    assert newest == {"observe": 3} and skipped == 2


# -- the panel side ------------------------------------------------------------------

def _worker_on_pipe():
    r, w = os.pipe()
    mw = PE.MotionWorker(deadline_s=5.0)
    mw._proc = types.SimpleNamespace(stdin=os.fdopen(w, "w"), poll=lambda: None)
    return mw, r, w


def test_a_full_pipe_drops_the_observation_instead_of_blocking():
    mw, r, w = _worker_on_pipe()
    try:
        os.set_blocking(w, False)
        try:
            while True:
                os.write(w, b"x" * 4096)
        except BlockingIOError:
            pass
        os.set_blocking(w, True)
        t0 = time.monotonic()
        assert mw._send_observation({"seq": 1}) is True
        assert time.monotonic() - t0 < 0.5
        assert mw.observations_dropped == 1
        assert os.get_blocking(w)                       # restored for job/cancel lines
    finally:
        mw._proc.stdin.close()
        os.close(r)


def test_observation_lines_are_written_whole():
    mw, r, w = _worker_on_pipe()
    try:
        assert mw._send_observation({"seq": 7, "ingest_mono": 1.5})
        got = os.read(r, 65536).decode()
        assert json.loads(got) == {"observe": {"seq": 7, "ingest_mono": 1.5}}
    finally:
        mw._proc.stdin.close()
        os.close(r)


def test_a_closed_worker_is_a_lost_pipe_not_a_crash():
    mw = PE.MotionWorker(deadline_s=5.0)
    assert mw._send_observation({"seq": 1}) is False


def test_the_forwarder_stops_when_the_job_returns():
    mw = PE.MotionWorker(deadline_s=5.0)
    sent = []
    mw._proc = types.SimpleNamespace(
        stdin=types.SimpleNamespace(write=sent.append, flush=lambda: None),
        poll=lambda: None)
    mw._lines = queue.Queue()
    feed_calls = {"n": 0}

    def feed():
        feed_calls["n"] += 1
        if feed_calls["n"] == 3:
            mw._lines.put(json.dumps({"result": {"status": "moved"}}) + "\n")
        return {"seq": feed_calls["n"]}

    before = {t.ident for t in threading.enumerate()}
    assert mw.run({"kind": "ability"}, feed=feed) == {"status": "moved"}
    leftover = [t for t in threading.enumerate()
                if t.ident not in before and t.name == "observe-forwarder"]
    assert leftover == []
    assert any('"observe"' in line for line in sent)


def test_a_dead_pipe_while_forwarding_fails_the_job():
    mw = PE.MotionWorker(deadline_s=5.0)

    def write(_s):
        if '"observe"' in _s:
            raise BrokenPipeError
    mw._proc = types.SimpleNamespace(
        stdin=types.SimpleNamespace(write=write, flush=lambda: None),
        poll=lambda: None, wait=lambda timeout=None: 0, kill=lambda: None)
    mw._lines = queue.Queue()
    out = mw.run({"kind": "ability"}, feed=lambda: {"seq": 1})
    assert out["status"] == "failed"
    assert out["evidence"].get("worker_died") is True
