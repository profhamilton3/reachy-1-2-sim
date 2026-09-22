"""Regressions for `scripts/e1_stage1/reset_verify.py` (E1 Stage 2 B1
session s1, `reset_18`: `docs/reviews/probes-2026-09-19-e1-stage2-b1/
control/reset_18.txt`, `stop_analysis.txt`). The old `reset.sh` read
`states.jsonl`'s last line with a bare `tail -1 | python3 -c
'...json.loads(...)'`: against a file the native server keeps appending
to at ~20 Hz (~5.9 KB/record), that read could return two records in one
`tail -1` (`Extra data`) or a torn last line, and either turned a
correctly-applied reset into an unverified STOP -- one failure in 36
scheduled resets across the B4/B1 sessions.

`reset_verify.py` fixes the *read* (reusing `e1_identity._read_last_state`,
the repo's tail-anchored, torn-line-tolerant, fail-closed reader) without
loosening the *verification*: every case below still STOPs, with a
specific reason. Offline throughout -- no server, no SDK, no Docker,
except `TestResetShEndToEnd`, which runs the real `reset.sh` against a
fake `docker` on `PATH` (pattern of `test_e1_leg_chain.py`'s fake
`E1_PYTHON`).

Fixtures write `states.jsonl`/`commands.jsonl` the way the recorder does
(`json.dumps(..., separators=(",", ":")) + "\\n"`, one JSON object per
line) with a `joints` list wide enough that a record is a realistic
multi-KB size relative to `_read_last_state`'s tail-read window, per
`native_mujoco/recorder.py` and `native_mujoco/protocol.py`'s `State`.
"""
import json
import os
import pathlib
import re
import stat
import subprocess
import sys
import threading
import time

import pytest

_HERE = pathlib.Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
_RESET_SH = _REPO / "scripts/e1_stage1/reset.sh"
_FAKE_DOCKER_PY = _HERE / "e1_reset_fake_docker.py"

sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO / "native_mujoco"))

from e1_stage1 import reset_verify as rv  # noqa: E402
import e1_identity as ei  # noqa: E402


# ── Fixture builders -- realistic record shapes, not conftest.py (project
# note: no conftest.py in this repo; every test module is self-contained
# so a passing suite can't depend on sys.path leakage from another file
# run earlier in the same session) ──────────────────────────────────────

def _joints(n=21):
    return [{"name": f"j{i}", "position_rad": 0.001 * i,
              "velocity_rad_s": 0.0, "effort": 0.01 * i,
              "compliant": False, "target_rad": 0.001 * i}
            for i in range(n)]


def _objects(n=6):
    return [{"object_id": f"obj_{i}", "pos_xyz": [0.1 * i, 0.2 * i, 0.3 * i],
              "quat_wxyz": [1.0, 0.0, 0.0, 0.0]} for i in range(n)]


def _state_line(seq, sim_step, wall_time_ns):
    return _state_line_raw(seq, sim_step, wall_time_ns)


def _state_line_raw(seq_val, sim_step_val, wall_time_ns_val):
    """Like `_state_line` but takes the three checked fields verbatim
    (no type coercion) -- lets a test pass `True` for `seq_val` or
    `float('nan')` for `wall_time_ns_val` to build a malformed record."""
    doc = {"type": "state", "seq": seq_val, "sim_step": sim_step_val,
           "sim_time_s": 0.5, "wall_time_ns": wall_time_ns_val,
           "cmd_seq": 0, "scene_revision": "r1", "paused": False,
           "joints": _joints(), "objects": _objects(),
           "grippers": [{"side": "right", "grasp": False, "force_n": 0.0}],
           "force_sensors": [{"name": "r_wrist", "force_n": [0.0, 0.0, 0.0]}],
           "interactive": [], "warnings": [], "contacts": []}
    return json.dumps(doc, separators=(",", ":")) + "\n"


def _reset_line(sim_step, seed=None, wall_time_s=1.0):
    return json.dumps(
        {"type": "reset", "seed": seed, "sim_step": sim_step,
         "wall_time_s": wall_time_s}, separators=(",", ":")) + "\n"


def _make_run_dir(tmp_path, *, name="run_001", snapshot_seq=1000,
                   snapshot_step=50000, snapshot_wall_ns=None):
    """One run dir with only the pre-reset snapshot state on disk."""
    run_dir = tmp_path / name
    run_dir.mkdir()
    wall_ns = snapshot_wall_ns if snapshot_wall_ns is not None else time.monotonic_ns()
    (run_dir / "commands.jsonl").write_text("")
    (run_dir / "states.jsonl").write_text(
        _state_line(snapshot_seq, snapshot_step, wall_ns))
    return run_dir


# ── A. Normal reset verification ────────────────────────────────────────

class TestNormalVerification:

    def test_verify_passes_and_prints_expected_summary(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_seq=1000, snapshot_step=50000)
        snap = rv.snapshot(run_dir)
        assert snap["seq"] == 1000
        assert snap["sim_step"] == 50000
        assert snap["reset_count"] == 0

        (run_dir / "commands.jsonl").write_text(_reset_line(50010))
        with (run_dir / "states.jsonl").open("a") as f:
            f.write(_state_line(1001, 10, time.monotonic_ns()))

        result = rv.verify(run_dir, "3", "3", snap, timeout_s=2.0, poll_s=0.05)
        assert result == {
            "reset_count_before": 0, "reset_count_after": 1,
            "sim_step_before": 50000, "sim_step_after": 10,
        }

    def test_cli_prints_exact_summary_line(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_seq=1000, snapshot_step=50000)
        snap_out = subprocess.run(
            [sys.executable, str(_REPO / "scripts/e1_stage1/reset_verify.py"),
             "snapshot", str(run_dir)], capture_output=True, text=True)
        assert snap_out.returncode == 0, snap_out.stderr
        snap_json = snap_out.stdout.strip()

        (run_dir / "commands.jsonl").write_text(_reset_line(50010))
        with (run_dir / "states.jsonl").open("a") as f:
            f.write(_state_line(1001, 10, time.monotonic_ns()))

        out = subprocess.run(
            [sys.executable, str(_REPO / "scripts/e1_stage1/reset_verify.py"),
             "verify", str(run_dir), "3", "3", snap_json,
             "--timeout-s", "2", "--poll-s", "0.05"],
            capture_output=True, text=True)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == (
            "reset gen=3 ack=3 resets_recorded 0->1 sim_step 50000->10")


# ── B. Partial write (torn last line) ───────────────────────────────────

class TestTornTrailingLine:

    def test_torn_last_line_is_dropped_in_favor_of_last_complete(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=50000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(50010))
        with (run_dir / "states.jsonl").open("a") as f:
            f.write(_state_line(1001, 10, time.monotonic_ns()))
            torn = _state_line(1002, 20, time.monotonic_ns())
            f.write(torn[: len(torn) // 2])  # no trailing newline, cut mid-joints

        result = rv.verify(run_dir, "4", "4", snap, timeout_s=2.0, poll_s=0.05)
        assert result["sim_step_after"] == 10  # the last COMPLETE record, not the torn one

    def test_only_post_reset_record_torn_stops_state_not_fresh(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=50000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(50010))
        with (run_dir / "states.jsonl").open("a") as f:
            torn = _state_line(1001, 10, time.monotonic_ns())
            f.write(torn[: len(torn) // 2])  # the ONLY post-reset write, and it's torn

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "4", "4", snap, timeout_s=0.5, poll_s=0.05)
        assert exc.value.reason == "state not fresh"


# ── C. Multiple records arriving during a read ──────────────────────────

class TestConcurrentWriter:

    def test_fast_concurrent_writer_with_torn_tail_still_verifies(self, tmp_path):
        """Reproduces the B1 shape directly: a real background thread
        appends complete records at >=100 Hz while `verify` polls, ending
        with one deliberately torn line."""
        run_dir = _make_run_dir(tmp_path, snapshot_step=50000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(50010))

        def writer():
            states_path = run_dir / "states.jsonl"
            seq, sim_step = 1001, 10
            with open(states_path, "a") as f:
                for _ in range(40):
                    f.write(_state_line(seq, sim_step, time.monotonic_ns()))
                    f.flush()
                    seq += 1
                    sim_step += 1
                    time.sleep(0.008)  # >100 Hz
                torn = _state_line(seq, sim_step, time.monotonic_ns())
                f.write(torn[: len(torn) // 2])
                f.flush()

        t = threading.Thread(target=writer)
        t.start()
        try:
            result = rv.verify(run_dir, "5", "5", snap, timeout_s=3.0, poll_s=0.02)
        finally:
            t.join(timeout=5)
        assert result["sim_step_before"] == 50000
        assert 10 <= result["sim_step_after"] < 50000

    def test_deterministic_two_complete_records_and_fragment_in_one_window(self, tmp_path):
        """Deterministic version of the same shape: a states.jsonl small
        enough that `_read_last_state`'s default tail window covers the
        WHOLE file in one read (no monkeypatch needed -- the window is
        `min(tail_bytes, file_size)`, and this file is far under the 64
        KiB default), so the window provably contains two complete
        records and a trailing fragment. Asserts the last complete one is
        returned, not the fragment."""
        states_path = tmp_path / "states.jsonl"
        rec1 = _state_line(1, 100, 1_000)
        rec2 = _state_line(2, 200, 2_000)
        fragment_full = _state_line(3, 300, 3_000)
        fragment = fragment_full[: len(fragment_full) // 2]
        content = rec1 + rec2 + fragment
        states_path.write_text(content)
        assert len(content.encode()) < 64 * 1024  # confirms the "whole file in one window" premise

        result = ei._read_last_state(states_path)
        assert result is not None
        assert result["seq"] == 2
        assert result["sim_step"] == 200


# ── D. Malformed evidence still stops ───────────────────────────────────

class TestMalformedEvidence:

    def test_unparseable_states_file_is_no_complete_state(self, tmp_path):
        run_dir = tmp_path / "run_d1"
        run_dir.mkdir()
        (run_dir / "commands.jsonl").write_text("")
        (run_dir / "states.jsonl").write_text(
            "not json at all\nalso not json\nstill not json\n")

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.snapshot(run_dir)
        assert exc.value.reason == "no complete state"

    def test_bool_seq_rejected_not_coerced(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(5010))
        with (run_dir / "states.jsonl").open("a") as f:
            f.write(_state_line_raw(True, 10, time.monotonic_ns()))  # "seq": true

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "6", "6", snap, timeout_s=0.3, poll_s=0.05)
        assert exc.value.reason == "state malformed: seq"

    def test_nan_wall_time_ns_rejected(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(5010))
        with (run_dir / "states.jsonl").open("a") as f:
            f.write(_state_line_raw(1001, 10, float("nan")))  # "wall_time_ns": NaN

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "6", "6", snap, timeout_s=0.3, poll_s=0.05)
        assert exc.value.reason == "state malformed: wall_time_ns"

    def test_malformed_reset_line_stops(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        # contains the reset marker (counts as +1) but is not valid JSON
        (run_dir / "commands.jsonl").write_text('{"type":"reset","seed":null,"sim_step":\n')
        with (run_dir / "states.jsonl").open("a") as f:
            f.write(_state_line(1001, 10, time.monotonic_ns()))

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "6", "6", snap, timeout_s=0.3, poll_s=0.05)
        assert exc.value.reason == "reset line malformed"


# ── E. Stale / insufficient evidence still stops ────────────────────────

class TestStaleOrInsufficientEvidence:

    def test_ack_mismatch(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=5000)
        snap = rv.snapshot(run_dir)

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "7", "8", snap, timeout_s=5.0, poll_s=0.1)
        assert exc.value.reason == "ack mismatch"

    def test_reset_not_recorded_after_timeout(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        # commands.jsonl stays empty -- no reset ever appears

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "7", "7", snap, timeout_s=0.3, poll_s=0.05)
        assert exc.value.reason == "reset not recorded"

    def test_more_than_one_reset_recorded_fails_fast(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        with (run_dir / "commands.jsonl").open("w") as f:
            f.write(_reset_line(5010))
            f.write(_reset_line(5020))

        start = time.monotonic()
        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "7", "7", snap, timeout_s=5.0, poll_s=0.1)
        elapsed = time.monotonic() - start
        assert exc.value.reason == "more than one reset recorded"
        assert elapsed < 1.0  # ambiguity is permanent -- must not wait out the timeout

    def test_state_not_fresh_when_states_file_unchanged(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(5010))
        # states.jsonl untouched -- last complete state is still the snapshot's own

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "7", "7", snap, timeout_s=0.3, poll_s=0.05)
        assert exc.value.reason == "state not fresh"

    def test_sim_step_not_restarted(self, tmp_path):
        run_dir = _make_run_dir(tmp_path, snapshot_seq=1000, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(5010))
        with (run_dir / "states.jsonl").open("a") as f:
            # fresh seq/wall_time_ns, but sim_step did NOT restart (>= snapshot)
            f.write(_state_line(1001, 5005, time.monotonic_ns()))

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "7", "7", snap, timeout_s=0.3, poll_s=0.05)
        assert exc.value.reason == "sim_step not restarted"

    def test_state_stale(self, tmp_path):
        now = time.monotonic_ns()
        run_dir = _make_run_dir(
            tmp_path, snapshot_seq=1000, snapshot_step=5000,
            snapshot_wall_ns=now - 20_000_000_000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(5010))
        with (run_dir / "states.jsonl").open("a") as f:
            # fresh + restarted relative to the snapshot, but 10s old NOW
            f.write(_state_line(1001, 10, now - 10_000_000_000))

        with pytest.raises(rv.VerifyFailed) as exc:
            rv.verify(run_dir, "7", "7", snap, timeout_s=0.3, poll_s=0.05,
                      max_state_age_s=1.0)
        assert exc.value.reason.startswith("state stale (age ")

    def test_wall_time_ns_clock_contract(self, tmp_path):
        """Pins the clock `wall_time_ns` is actually measured on --
        `time.monotonic_ns()`, matching `native_mujoco/protocol.py`'s
        `_now_ns()` and `e1_identity.verify_identity`'s `now_ns` default
        (issue #130 B1: comparing it against `time.time_ns()` made every
        state look ~decades stale). A future regression back to wall-clock
        in either `reset_verify.verify`'s default `now_ns` or a fixture
        here must fail this test."""
        run_dir = _make_run_dir(tmp_path, snapshot_seq=1000, snapshot_step=5000)
        snap = rv.snapshot(run_dir)
        (run_dir / "commands.jsonl").write_text(_reset_line(5010))

        # A fresh, correctly monotonic-stamped state verifies.
        with (run_dir / "states.jsonl").open("a") as f:
            f.write(_state_line(1001, 10, time.monotonic_ns()))
        result = rv.verify(run_dir, "7", "7", snap, timeout_s=0.3, poll_s=0.05)
        assert result["sim_step_after"] == 10

        # A stale monotonic-stamped state fails "state stale", not a crash.
        now = time.monotonic_ns()
        run_dir2 = _make_run_dir(tmp_path, name="run_002",
                                  snapshot_seq=1000, snapshot_step=5000,
                                  snapshot_wall_ns=now - 20_000_000_000)
        snap2 = rv.snapshot(run_dir2)
        (run_dir2 / "commands.jsonl").write_text(_reset_line(5010))
        with (run_dir2 / "states.jsonl").open("a") as f:
            f.write(_state_line(1001, 10, now - 10_000_000_000))
        with pytest.raises(rv.VerifyFailed) as exc2:
            rv.verify(run_dir2, "7", "7", snap2, timeout_s=0.3, poll_s=0.05,
                      max_state_age_s=1.0)
        assert exc2.value.reason.startswith("state stale (age ")

        # An epoch (time.time_ns()) stamped state -- what a "fix" back to
        # wall-clock would write -- must also fail "state stale": on this
        # host the two clocks differ by decades, so it can never look fresh.
        run_dir3 = _make_run_dir(tmp_path, name="run_003",
                                  snapshot_seq=1000, snapshot_step=5000)
        snap3 = rv.snapshot(run_dir3)
        (run_dir3 / "commands.jsonl").write_text(_reset_line(5010))
        with (run_dir3 / "states.jsonl").open("a") as f:
            f.write(_state_line(1001, 10, time.time_ns()))
        with pytest.raises(rv.VerifyFailed) as exc3:
            rv.verify(run_dir3, "7", "7", snap3, timeout_s=0.3, poll_s=0.05,
                      max_state_age_s=1.0)
        assert exc3.value.reason.startswith("state stale (age ")


# ── F. Real reset.sh end to end ─────────────────────────────────────────

class TestResetShEndToEnd:

    @pytest.fixture
    def fake_docker(self, tmp_path):
        path = tmp_path / "fakebin" / "docker"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"#!/usr/bin/env bash\nexec {sys.executable} {_FAKE_DOCKER_PY} \"$@\"\n")
        path.chmod(path.stat().st_mode | stat.S_IEXEC)
        return path

    def _env(self, fake_docker, run_dir, ack_file, mode, snapshot_step,
              timeout_s=1.0, poll_s=0.05):
        env = dict(os.environ)
        env["PATH"] = f"{fake_docker.parent}{os.pathsep}{env['PATH']}"
        env["FAKE_DOCKER_RUN_DIR"] = str(run_dir)
        env["FAKE_DOCKER_ACK_FILE"] = str(ack_file)
        env["FAKE_DOCKER_MODE"] = mode
        env["FAKE_DOCKER_SNAPSHOT_STEP"] = str(snapshot_step)
        env["RESET_VERIFY_TIMEOUT_S"] = str(timeout_s)
        env["RESET_VERIFY_POLL_S"] = str(poll_s)
        return env

    def _run(self, tmp_path, fake_docker, *, mode, gen="7", snapshot_step=5000,
              timeout_s=1.0, poll_s=0.05):
        evidence_dir = tmp_path / "evidence"
        (evidence_dir / "control").mkdir(parents=True, exist_ok=True)
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        (run_dir / "commands.jsonl").write_text("")
        (run_dir / "states.jsonl").write_text(
            _state_line(1, snapshot_step, time.monotonic_ns()))
        ack_file = tmp_path / "ack.txt"
        ack_file.write_text("")

        env = self._env(fake_docker, run_dir, ack_file, mode, snapshot_step,
                         timeout_s=timeout_s, poll_s=poll_s)
        result = subprocess.run(
            ["bash", str(_RESET_SH), str(_REPO), str(evidence_dir), gen, str(run_dir)],
            env=env, capture_output=True, text=True)
        return result, evidence_dir, run_dir

    def test_clean_reset_verifies_and_no_stop(self, tmp_path, fake_docker):
        result, evidence_dir, _ = self._run(tmp_path, fake_docker, mode="ok", gen="9")
        assert result.returncode == 0, result.stdout + result.stderr
        assert re.match(r"^reset gen=9 ack=9 resets_recorded 0->1 sim_step 5000->\d+$",
                         result.stdout.strip())
        assert not (evidence_dir / "control/stop").exists()

    def test_ack_but_no_append_stops(self, tmp_path, fake_docker):
        result, evidence_dir, _ = self._run(
            tmp_path, fake_docker, mode="no_append", gen="10", timeout_s=0.3)
        assert result.returncode == 9
        stop_text = (evidence_dir / "control/stop").read_text()
        assert "STOP: reset 10 not verified: reset not recorded" in stop_text

    def test_concurrent_writer_growing_torn_tail_still_reaches_verified(
            self, tmp_path, fake_docker):
        """§4.F(c): acks and appends while a concurrent writer keeps
        growing the file with a torn tail -- the fake docker's own
        `torn_growing` mode spawns `e1_reset_growth_writer.py` (detached)
        the instant the sentinel write happens, which is strictly after
        `reset.sh`'s own snapshot read, so the writer can't contaminate
        the pre-reset snapshot."""
        result, evidence_dir, run_dir = self._run(
            tmp_path, fake_docker, mode="torn_growing", gen="11",
            snapshot_step=5000, timeout_s=3.0, poll_s=0.05)
        assert result.returncode == 0, result.stdout + result.stderr
        assert re.match(r"^reset gen=11 ack=11 resets_recorded 0->1 sim_step 5000->\d+$",
                         result.stdout.strip())
        assert not (evidence_dir / "control/stop").exists()

    def test_multi_digit_gen_published_whole(self, tmp_path, fake_docker):
        """Regression for the atomic-publish fix (E1 Stage 2 B2 s1 reset 1,
        2026-09-22): `$GEN` now reaches the container as a trailing
        positional argument to `sh`, not interpolated into the `-c`
        script text -- a multi-digit generation (the real B1/B2 sessions
        run gens up to 18) must arrive whole, not truncated or split by
        the new argv-based passing."""
        result, evidence_dir, _ = self._run(tmp_path, fake_docker, mode="ok", gen="18")
        assert result.returncode == 0, result.stdout + result.stderr
        assert re.match(r"^reset gen=18 ack=18 resets_recorded 0->1 sim_step 5000->\d+$",
                         result.stdout.strip())
        assert not (evidence_dir / "control/stop").exists()

    def test_publish_failure_stops_before_ack_poll(self, tmp_path, fake_docker):
        """The new `pub_rc` check (reset.sh, atomic-publish fix): if the
        request-publish call itself fails (container exec dropped, not a
        sentinel race), reset.sh STOPs immediately with a specific reason
        and never enters the 30x0.5s ack-poll loop -- distinct from
        `ack mismatch`, which only happens after polling. Bounded
        wall-clock assertion proves it didn't just poll-then-timeout by
        coincidence."""
        t0 = time.monotonic()
        result, evidence_dir, _ = self._run(
            tmp_path, fake_docker, mode="publish_fail", gen="13", timeout_s=0.3)
        elapsed = time.monotonic() - t0
        assert result.returncode == 9
        stop_text = (evidence_dir / "control/stop").read_text()
        assert "STOP: reset 13 not verified: request publish failed (rc=3)" in stop_text
        assert elapsed < 5.0, (
            f"took {elapsed:.1f}s -- looks like it fell through to the "
            f"15s ack-poll loop instead of stopping immediately")

    def test_existing_stop_marker_refuses_at_top_unchanged(self, tmp_path, fake_docker):
        evidence_dir = tmp_path / "evidence"
        (evidence_dir / "control").mkdir(parents=True, exist_ok=True)
        (evidence_dir / "control/stop").write_text("STOP: pre-existing\n")
        run_dir = tmp_path / "run_001"
        run_dir.mkdir()
        (run_dir / "commands.jsonl").write_text("")
        (run_dir / "states.jsonl").write_text(_state_line(1, 5000, time.monotonic_ns()))
        ack_file = tmp_path / "ack.txt"
        ack_file.write_text("")

        env = self._env(fake_docker, run_dir, ack_file, "ok", 5000)
        result = subprocess.run(
            ["bash", str(_RESET_SH), str(_REPO), str(evidence_dir), "12", str(run_dir)],
            env=env, capture_output=True, text=True)
        assert result.returncode == 9
        assert "STOP marker present; reset refused" in result.stdout
        assert (evidence_dir / "control/stop").read_text() == "STOP: pre-existing\n"
