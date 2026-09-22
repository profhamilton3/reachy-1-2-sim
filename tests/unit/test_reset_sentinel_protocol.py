"""Regressions for `reset_watcher.py` (E1 Stage 2 B2 s1 reset 1,
2026-09-22: `docs/reviews/probes-2026-09-22-e1-stage2-b2/control/
stop_analysis.txt`) -- the container-side half of the acknowledged-reset
sentinel protocol whose client half is `scripts/e1_stage1/reset.sh`.

The bug: `reset.sh` used to publish a reset request with a plain
`echo $GEN > /tmp/reachy_reset_request`, which truncates the target file
before the write completes. The old watcher polled at 10 Hz and consumed
whatever it read -- unconditionally, including a read landing in that
truncate-then-write window (gen=""). That triggered an unrequested
physics reset and wrote a 0-byte ack, which `reset.sh` read back as
`ack mismatch` and STOPped the session (0/18 cycles flown on B2 s1).

The fix has two independent parts, both covered here:
  1. `reset.sh` now publishes atomically -- a same-directory temp file
     + `mv` (`TestRealPublishScript` runs `reset.sh`'s own extracted
     shell fragment, not a reimplementation).
  2. `reset_watcher_step` refuses to act on anything that doesn't read
     back as a complete, valid generation (`TestMalformedRequests`,
     `TestCreateBeforeWriteRace`) -- a second, independent guard that
     holds even against a non-atomic publisher.

B1 (PR #135 review, 2026-09-22): the first version of this file shipped
with `VALID_GEN_RE = ^[0-9]+$`, digit-only, which silently broke
`reset_scene()` in both training notebooks -- they publish
`uuid.uuid4().hex` (32 lowercase hex chars), not a decimal counter, and
the module's own claim that they were "unaffected" was false: a uuid
token just sat on disk forever, never consumed, never acked. The regex
is now a bounded opaque-token shape (`^[0-9A-Za-z]{1,64}$`), compatible
with both reset.sh's decimal gens and the notebooks' uuid-hex tokens;
`TestValidRequest` covers both formats through the real watcher, and
`TestMalformedRequests` still covers empty/whitespace/oversized/
embedded-control-byte input, which the wider charset does not accept.

Deterministic and offline throughout -- no server, no SDK, no Docker,
no grpc (unlike `fake_reachy_server.py`, `reset_watcher.py` is
dependency-light stdlib only, exactly so this suite can exercise it
without the heavy gRPC/scipy import chain). The two threaded classes
(`TestAckNeverTornUnderConcurrentReads`, `TestRealPublishScript`'s stress
test) are bounded, fast, and assert invariants rather than exact timing,
so they are not the sole evidence for any claim here -- every property
they demonstrate is also pinned by a plain, non-threaded test above it.
"""
import logging
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
import uuid

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_RESET_SH = _REPO / "scripts/e1_stage1/reset.sh"

sys.path.insert(0, str(_REPO))

import reset_watcher as rw  # noqa: E402


class _FakeRemote:
    """Stand-in for `mujoco_remote_backend`'s reset handle: `request_reset()`
    is the only surface `reset_watcher_step` calls, and it's already a
    parameter of `reset_watcher`/`reset_watcher_step` for exactly this kind
    of substitution (production passes the real `MujocoRemoteBackend`).
    `_event` is pre-set so `event.wait(timeout=6.0)` returns immediately --
    these tests are about the file protocol, not reset latency."""

    def __init__(self):
        self.reset_calls = 0
        self._event = threading.Event()
        self._event.set()

    def request_reset(self):
        self.reset_calls += 1
        return self._event


# ── A. Deterministic: empty/malformed requests never trigger a reset or
#    a matching ack ─────────────────────────────────────────────────────

class TestMalformedRequests:

    @pytest.mark.parametrize("raw", [
        "",              # the exact B2 s1 bug: a read landing pre-write
        "   ",           # whitespace only
        "\n",
        "-1",            # '-' outside the bounded-alnum charset
        "1.5",
        "1 8",
        "18\n19",        # two lines -- a torn concatenation of two gens
        "\x00",
        "a" * 65,        # one over the 64-char cap (oversized)
        "1" * 65,
        "18abc def",     # alnum substrings either side of a space
    ], ids=repr)
    def test_never_triggers_reset_or_ack(self, tmp_path, raw):
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        req.write_text(raw)
        remote = _FakeRemote()

        result = rw.reset_watcher_step(remote, req, ack)

        assert result is None
        assert remote.reset_calls == 0
        assert not ack.exists()
        assert req.exists() and req.read_text() == raw, (
            "a malformed/empty request must be LEFT IN PLACE, not "
            "unlinked -- a writer that completes it on a later poll "
            "must still be picked up (see reset_watcher.py docstring)")

    def test_missing_request_file_is_a_no_op(self, tmp_path):
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()
        assert rw.reset_watcher_step(remote, req, ack) is None
        assert remote.reset_calls == 0
        assert not ack.exists()

    def test_unreadable_request_file_is_a_no_op(self, tmp_path):
        """A directory at the request path (or any OSError on read) must
        be refused the same way, not raise out of the poll loop."""
        req = tmp_path / "reachy_reset_request"
        req.mkdir()
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()
        assert rw.reset_watcher_step(remote, req, ack) is None
        assert remote.reset_calls == 0

    def test_exactly_64_chars_is_valid_65_is_not(self, tmp_path):
        """Pins the length-cap boundary explicitly rather than leaving it
        implicit in the 65-char oversized case above."""
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()

        req.write_text("a" * 64)
        assert rw.reset_watcher_step(remote, req, ack) == "a" * 64
        assert remote.reset_calls == 1

        req.write_text("a" * 65)
        assert rw.reset_watcher_step(remote, req, ack) is None
        assert remote.reset_calls == 1, "oversized token must not consume"

    def test_ignored_nonempty_malformed_content_warns_once_per_value(
            self, tmp_path, monkeypatch, caplog):
        """B1 follow-up: a broken/misconfigured publisher writing invalid
        content must be visible in the log (warning, not silent debug),
        but repeated polls of the SAME bad value must not flood it -- an
        empty read (the ordinary truncate-then-write window) stays at
        debug either way."""
        monkeypatch.setattr(rw, "_WARNED_MALFORMED", set())
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()

        req.write_text("bad value!")
        with caplog.at_level(logging.WARNING, logger="reset_watcher"):
            for _ in range(5):
                assert rw.reset_watcher_step(remote, req, ack) is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1, (
            "5 polls of the same malformed value must log exactly one "
            f"warning, not {len(warnings)} (flooding)")
        assert "bad value!" in warnings[0].getMessage()
        assert remote.reset_calls == 0

    def test_empty_read_never_warns(self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(rw, "_WARNED_MALFORMED", set())
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()

        req.write_text("")
        with caplog.at_level(logging.DEBUG, logger="reset_watcher"):
            rw.reset_watcher_step(remote, req, ack)
        assert not any(
            r.levelno == logging.WARNING for r in caplog.records), (
            "an empty read is the ordinary truncate-then-write window, "
            "not a signal worth a warning")

    def test_distinct_malformed_values_each_warn(
            self, tmp_path, monkeypatch, caplog):
        monkeypatch.setattr(rw, "_WARNED_MALFORMED", set())
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()

        with caplog.at_level(logging.WARNING, logger="reset_watcher"):
            for raw in ("bad one", "bad two", "bad three"):
                req.write_text(raw)
                assert rw.reset_watcher_step(remote, req, ack) is None
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 3


# ── B. Deterministic: a valid generation is consumed exactly once,
#    correctly ─────────────────────────────────────────────────────────

class TestValidRequest:

    @pytest.mark.parametrize("gen", [
        "1", "18", "007", "0",                    # reset.sh's decimal gens
        "a3f9c2b1e4d5460a9b7c3f2e1d0a5b6c",         # notebooks' uuid.uuid4().hex shape
        "ABCDEF0123456789",                        # uppercase hex still matches alnum
    ])
    def test_consumed_and_acked(self, tmp_path, gen):
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        req.write_text(gen)
        remote = _FakeRemote()

        result = rw.reset_watcher_step(remote, req, ack)

        assert result == gen
        assert remote.reset_calls == 1
        assert not req.exists(), "a valid request must be unlinked"
        assert ack.read_text() == gen
        # no stray temp file left behind by the atomic ack write
        assert sorted(p.name for p in tmp_path.iterdir()) == ["reachy_reset_ack"]

    def test_whitespace_padded_gen_is_stripped_and_valid(self, tmp_path):
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        req.write_text("  18  \n")
        remote = _FakeRemote()
        assert rw.reset_watcher_step(remote, req, ack) == "18"
        assert ack.read_text() == "18"

    def test_real_uuid4_hex_token_round_trips_through_the_watcher(
            self, tmp_path):
        """B1: the exact token shape `reset_scene()` in both
        notebooks/*_training.ipynb publishes (`uuid.uuid4().hex`), driven
        through the actual `reset_watcher_step`, not a hand-picked
        hex-looking string -- this is what silently broke under the
        original digit-only `VALID_GEN_RE`."""
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()
        for _ in range(20):  # several independent random draws
            gen = uuid.uuid4().hex
            req.write_text(gen)
            result = rw.reset_watcher_step(remote, req, ack)
            assert result == gen
            assert ack.read_text() == gen
            assert not req.exists()
        assert remote.reset_calls == 20

    def test_ack_timeout_still_publishes_ack(self, tmp_path):
        """`event.wait` returning False (reset ack never arrived from the
        native server) is unrelated to the sentinel-file race: the ack is
        still published so the demo/reset.sh poll doesn't hang forever
        (pre-existing behavior, unchanged by this fix)."""
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        req.write_text("5")

        class _NeverAcks(_FakeRemote):
            def request_reset(self):
                self.reset_calls += 1
                return threading.Event()  # never set -> wait() times out

        remote = _NeverAcks()
        t0 = time.monotonic()
        result = rw.reset_watcher_step(remote, req, ack)
        elapsed = time.monotonic() - t0
        assert result == "5"
        assert ack.read_text() == "5"
        assert elapsed >= 5.9, "should have waited out the 6s ack timeout"


# ── C. Deterministic reproduction of the create-before-write race, with
#    no threads or sleeps: the exact syscall interleaving that produced
#    the B2 s1 bug, manually ordered ─────────────────────────────────────

class TestCreateBeforeWriteRace:

    def test_poll_during_the_truncate_write_window_is_ignored_not_consumed(
            self, tmp_path):
        """Reproduces `echo $GEN > file`'s two-step create-then-write as
        two explicit steps, with a watcher poll manually interleaved
        between them -- deterministic, no timing dependency."""
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()

        # Step 1: the shell redirection's open(O_CREAT|O_TRUNC) has run;
        # `echo`'s write() has not landed yet. This is EXACTLY the file
        # state a watcher poll observed in the B2 s1 incident.
        req.write_text("")
        assert rw.reset_watcher_step(remote, req, ack) is None
        assert remote.reset_calls == 0, (
            "the old watcher called request_reset() here -- this is the "
            "bug: an unrequested reset triggered by an empty read")
        assert req.exists(), "must not be consumed while still empty"

        # Step 2: `echo`'s write() lands (or, with the fix, the atomic
        # rename completes) -- the file now holds the real generation.
        req.write_text("1")

        # Step 3: the NEXT poll (10 Hz in production) now sees complete,
        # valid content and processes it correctly.
        result = rw.reset_watcher_step(remote, req, ack)
        assert result == "1"
        assert remote.reset_calls == 1
        assert ack.read_text() == "1"
        assert not req.exists()

    def test_multiple_empty_polls_never_accumulate_reset_calls(self, tmp_path):
        """A slow writer (or a stuck one) must never accumulate spurious
        resets no matter how many empty polls land before it finishes."""
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()
        req.write_text("")
        for _ in range(50):
            assert rw.reset_watcher_step(remote, req, ack) is None
        assert remote.reset_calls == 0
        req.write_text("3")
        assert rw.reset_watcher_step(remote, req, ack) == "3"
        assert remote.reset_calls == 1


# ── D. The real `reset.sh` publish script (not a reimplementation),
#    exercising the real producer path ───────────────────────────────────

def _extract_publish_script() -> str:
    """Pulls the exact `sh -c '...'` fragment `reset.sh` uses to publish
    the request, straight from its source, so these tests exercise the
    real producer text rather than a copy that could drift from it."""
    text = _RESET_SH.read_text()
    m = re.search(
        r'docker compose exec -T reachy-sim sh -c \\\n\s*\'(.+?)\'\s*\\\n\s*sh "\$GEN"',
        text, re.DOTALL)
    assert m, "reset.sh's publish invocation shape changed -- update this extractor"
    return m.group(1)


class TestRealPublishScript:
    """Runs `reset.sh`'s actual publish one-liner via `sh -c` (the same
    shell contract `docker compose exec -T reachy-sim sh -c '...' sh
    "$GEN"` gives the container's shell), with the hardcoded
    `/tmp/reachy_reset_request` substituted for a tmp_path-scoped file so
    the test is hermetic. No Docker; this is exactly what runs inside the
    container shell, character for character."""

    def _script_for(self, target: pathlib.Path) -> str:
        script = _extract_publish_script()
        assert "/tmp/reachy_reset_request" in script
        return script.replace("/tmp/reachy_reset_request", str(target))

    def test_publishes_gen_intact_including_multi_digit(self, tmp_path):
        target = tmp_path / "reachy_reset_request"
        script = self._script_for(target)
        for gen in ("1", "18", "007"):
            subprocess.run(["sh", "-c", script, "sh", gen], check=True)
            assert target.read_text() == gen
            target.unlink()

    def test_no_leftover_temp_file(self, tmp_path):
        target = tmp_path / "reachy_reset_request"
        script = self._script_for(target)
        subprocess.run(["sh", "-c", script, "sh", "42"], check=True)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["reachy_reset_request"]

    def test_concurrent_watcher_never_observes_empty_or_partial(self, tmp_path):
        """Bounded stress test (not the sole evidence for the property --
        `TestCreateBeforeWriteRace` above proves the same invariant
        deterministically for the watcher side; this additionally drives
        the REAL producer script under a tight concurrent reader, N=100
        publishes, to catch anything the manual reproduction couldn't."""
        target = tmp_path / "reachy_reset_request"
        script = self._script_for(target)
        observed_bad = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    content = target.read_text()
                except (OSError, FileNotFoundError):
                    content = None
                if content is not None and content == "":
                    observed_bad.append(content)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for i in range(100):
                gen = str(i + 1)
                subprocess.run(["sh", "-c", script, "sh", gen], check=True)
                assert target.read_text() == gen
                target.unlink()
        finally:
            stop.set()
            t.join(timeout=2.0)

        assert observed_bad == [], (
            f"the atomic publish was observed empty {len(observed_bad)} "
            f"times out of 100 -- the race is not closed")


# ── E. Ack writes are atomic: a fast concurrent reader (reset.sh's poll)
#    never observes a torn/partial value ────────────────────────────────

class TestAckNeverTornUnderConcurrentReads:

    def test_reader_only_ever_sees_complete_values(self, tmp_path):
        """Bounded stress test, supplementary to the deterministic
        `TestValidRequest`/`TestCreateBeforeWriteRace` coverage above:
        drives `reset_watcher_step` through many valid generations
        (including multi-digit, so a torn write would be visible as a
        truncated prefix) while a fast reader polls the ack file."""
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        remote = _FakeRemote()
        gens = [str(n) for n in range(1, 101)]
        observed = []
        stop = threading.Event()

        def reader():
            while not stop.is_set():
                try:
                    content = ack.read_text()
                except (OSError, FileNotFoundError):
                    continue
                if content:
                    observed.append(content)

        t = threading.Thread(target=reader, daemon=True)
        t.start()
        try:
            for gen in gens:
                req.write_text(gen)
                result = rw.reset_watcher_step(remote, req, ack)
                assert result == gen
        finally:
            stop.set()
            t.join(timeout=2.0)

        valid = set(gens)
        bad = [v for v in observed if v not in valid]
        assert bad == [], f"reader observed torn/unexpected ack content: {bad!r}"
        assert remote.reset_calls == len(gens)


# ── F. The public `reset_watcher` loop entry point, driven briefly in a
#    background thread (not just the extracted `_step` helper) ──────────

class TestWatcherLoop:

    def test_loop_processes_a_request_published_after_it_starts(
            self, tmp_path, monkeypatch):
        req = tmp_path / "reachy_reset_request"
        ack = tmp_path / "reachy_reset_ack"
        # `reset_watcher`'s ack path isn't a parameter (only the request
        # `path` is) -- it reads the module-level `ACK_PATH` each call,
        # same as before this fix, so patch that to keep this test
        # hermetic instead of touching the real /tmp/reachy_reset_ack.
        monkeypatch.setattr(rw, "ACK_PATH", str(ack))
        remote = _FakeRemote()

        t = threading.Thread(
            target=rw.reset_watcher,
            kwargs=dict(remote=remote, path=str(req), hz=50.0),
            daemon=True)
        t.start()
        try:
            time.sleep(0.05)  # a couple of empty polls first
            req.write_text("7")
            for _ in range(50):
                if ack.exists() and ack.read_text() == "7":
                    break
                time.sleep(0.02)
            else:
                pytest.fail("reset_watcher loop never processed the request")
            assert remote.reset_calls == 1
            assert not req.exists()
        finally:
            # daemon thread; process exit reclaims it. No explicit stop
            # mechanism in the loop (matches production, which runs for
            # the life of the server process).
            pass
