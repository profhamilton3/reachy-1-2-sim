"""Synthetic tests of the capture shim (tests/fixtures/goalfix_cmp/capture.py).

Deterministic: no SDK client, no gRPC server, no websocket, no network.  The
REAL ``FakeJointService._submit_batch`` is driven against a REAL
``MujocoRemoteBackend`` that is constructed but never started (its __init__
does no I/O), seeded so ``_build_command`` can run.  The "drain" mirrors
``_send``'s own (``cmds = list(pending); pending.clear(); build``), performed
here at deterministic points, including BETWEEN two joints of one batch.

Importing the production modules needs the gRPC protos; gating matches the
sibling goalfix_cmp tests (skip on the system interpreter, fail under
REQUIRE_REACHY_SDK=1).
"""
from __future__ import annotations

import copy
import os
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "native_mujoco"),
           os.path.join(_ROOT, "tests", "fixtures", "goalfix_cmp")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if os.environ.get("REQUIRE_REACHY_SDK") == "1":
    import grpc  # noqa: F401
    import reachy_sdk  # noqa: F401
else:
    pytest.importorskip("grpc")
    pytest.importorskip("reachy_sdk")

import capture as cap  # noqa: E402
import fake_reachy_server as frs  # noqa: E402
import mujoco_remote_backend as mrb  # noqa: E402
from mujoco_remote_backend import KinematicBridge, MujocoRemoteBackend  # noqa: E402

UIDS = [10, 11, 12, 13]   # r_shoulder_pitch .. r_elbow_pitch -> protocol indices 0..3


# ---------------------------------------------------------------------------
# Duck-typed SDK batch (only what _submit_batch touches)
# ---------------------------------------------------------------------------

class _Id:
    def __init__(self, uid):
        self.uid = uid

    def WhichOneof(self, _name):
        return "uid"


class _Val:
    def __init__(self, v):
        self.value = v


class _JCmd:
    def __init__(self, uid, goal=None, compliant=None):
        self.id = _Id(uid)
        self._goal, self._compliant = goal, compliant
        self.goal_position = _Val(goal)
        self.compliant = _Val(compliant)

    def HasField(self, name):
        if name == "goal_position":
            return self._goal is not None
        if name == "compliant":
            return self._compliant is not None
        return False


class _Commands:
    """Iterates the batch's commands, calling ``before(k)`` just before
    yielding entry ``k`` -- the deterministic hook that lets a drain land
    between two joints of ONE batch."""

    def __init__(self, cmds, before=None):
        self._cmds, self._before = cmds, before

    def __len__(self):
        return len(self._cmds)

    def __iter__(self):
        for k, c in enumerate(self._cmds):
            if self._before is not None:
                self._before(k)
            yield c


class _Batch:
    def __init__(self, cmds, before=None):
        self.commands = _Commands(cmds, before)


def _batch(goals, before=None):
    return _Batch([_JCmd(u, g) for u, g in zip(UIDS, goals)], before)


def _make_rig():
    backend = MujocoRemoteBackend(url="ws://127.0.0.1:1")        # never started
    with backend._lock:
        backend._seed_locked({i: (0.0 if i == 10 else 0.01 * i) for i in range(21)})
    svc = frs.FakeJointService(KinematicBridge(backend))
    return backend, svc


def _drain(backend):
    """Mirrors ``_send``'s own drain: list, clear, build (under the lock),
    then the seq bump ``_send`` performs after building."""
    with backend._lock:
        cmds = list(backend._pending_cmds)
        backend._pending_cmds.clear()
        built = backend._build_command(cmds) if cmds else None
    if built is not None:
        backend._cmd_seq += 1
    return built


B1 = [1.0, 1.1, 1.2, 1.3]
B2 = [2.0, 2.1, 2.2, 2.3]


def _run_split(svc, backend):
    """B1 and B2, each drained mid-batch: after joint 1 of B1 and after joint 1 of B2, then a
    final drain after B2's tail."""
    svc._submit_batch(_batch(B1, before=lambda k: _drain(backend) if k == 2 else None))
    svc._submit_batch(_batch(B2, before=lambda k: _drain(backend) if k == 2 else None))
    _drain(backend)


def _run_intact(svc, backend):
    """Same batches; drains only BETWEEN batches."""
    svc._submit_batch(_batch(B1))
    _drain(backend)
    svc._submit_batch(_batch(B2))
    _drain(backend)


def _captured(run):
    with cap.Capture() as c:
        backend, svc = _make_rig()
        run(svc, backend)
    return c, backend


# ---------------------------------------------------------------------------
# Split / intact
# ---------------------------------------------------------------------------

def test_deliberately_split_batch_is_split_and_mixed_command_found():
    c, backend = _captured(_run_split)
    recs = c.records()
    builds = [r for r in recs if r["type"] == "build"]
    assert len(builds) == 3
    assert [b["seq"] for b in builds] == [1, 2, 3]
    b1id, b2id = [r["batch_id"] for r in recs if r["type"] == "batch"]

    # build 1: B1 head only (indices 0, 1); everything else carried.
    assert builds[0]["source"][:4] == [b1id, b1id, None, None]
    assert all(s is None for s in builds[0]["source"][4:])
    # build 2: B1 tail (idx 2, 3) + B2 head (idx 0, 1): last writer per index, two batches.
    assert builds[1]["source"][:4] == [b2id, b2id, b1id, b1id]
    assert builds[1]["target"][:4] == [2.0, 2.1, 1.2, 1.3]
    # build 3: B2 tail; idx 0, 1 carried.
    assert builds[2]["source"][:4] == [None, None, b2id, b2id]
    assert builds[2]["target"][:4] == [2.0, 2.1, 2.2, 2.3]
    assert all(s is None for s in builds[1]["source"][4:] + builds[2]["source"][4:])

    cl = cap.classify(recs)
    assert cl["split_batches"] == sorted([b1id, b2id])
    assert [b["class"] for b in cl["builds"]] == [
        "split_involved_not_mixed", "mixed_batch", "split_involved_not_mixed"]
    assert cl["builds"][1]["mixed"] is True
    assert cl["builds"][1]["source_batches"] == sorted([b1id, b2id])
    assert cl["builds"][1]["source_span_ns"] is not None and cl["builds"][1]["source_span_ns"] >= 0
    assert cl["n_mixed"] == 1 and cl["n_split_batches"] == 2
    assert cl["n_unattributed_submits"] == 0
    assert len(cl["batch_gaps_ns"]) == 1 and cl["batch_gaps_ns"][0]["gap_ns"] >= 0


def test_intact_batches_control_has_no_split_and_no_mixed():
    c, backend = _captured(_run_intact)
    recs = c.records()
    cl = cap.classify(recs)
    assert cl["n_builds"] == 2
    assert cl["split_batches"] == []
    assert cl["n_mixed"] == 0 and cl["n_split_involved_not_mixed"] == 0
    assert [b["class"] for b in cl["builds"]] == ["neither", "neither"]
    builds = [r for r in recs if r["type"] == "build"]
    b1id, b2id = [r["batch_id"] for r in recs if r["type"] == "batch"]
    assert builds[0]["source"][:4] == [b1id] * 4
    assert builds[1]["source"][:4] == [b2id] * 4


def test_last_writer_wins_when_one_build_holds_two_batches_for_the_same_index():
    """No drain between B1 and B2: both drain into ONE build; every index is
    written twice and the LAST writer (B2) is the source (as _build_command
    itself does), so the build is NOT mixed and no batch is split."""
    def run(svc, backend):
        svc._submit_batch(_batch(B1))
        svc._submit_batch(_batch(B2))
        _drain(backend)
    c, backend = _captured(run)
    recs = c.records()
    build = [r for r in recs if r["type"] == "build"][0]
    b1id, b2id = [r["batch_id"] for r in recs if r["type"] == "batch"]
    assert build["source"][:4] == [b2id] * 4
    assert build["target"][:4] == B2
    cl = cap.classify(recs)
    assert cl["builds"][0]["mixed"] is False
    assert cl["split_batches"] == []


def test_submits_are_tagged_with_batch_entry_uid_and_index():
    c, _ = _captured(_run_intact)
    subs = [r for r in c.records() if r["type"] == "submit"]
    assert [(s["batch_id"], s["entry"], s["uid"], s["index"]) for s in subs[:4]] == [
        (subs[0]["batch_id"], k, UIDS[k], k) for k in range(4)]
    assert all(s["batch_id"] is not None for s in subs)
    batches = [r for r in c.records() if r["type"] == "batch"]
    assert all(b["n_entries"] == 4 for b in batches)


def test_submit_outside_a_batch_is_unattributed_and_fails_cross_check():
    def run(svc, backend):
        backend.submit_command(frs.JointCommand(uid=10, goal_position=0.5))
        _drain(backend)
    c, _ = _captured(run)
    cl = cap.classify(c.records())
    assert cl["n_unattributed_submits"] == 1
    res = cap.cross_check(c.records(), _rows_from(c.records()))
    assert not res.ok and any("unattributed" in f for f in res.failures)


# ---------------------------------------------------------------------------
# Transparency, no batch-wide lock, restoration
# ---------------------------------------------------------------------------

def _targets(run):
    got = []
    orig = MujocoRemoteBackend._build_command

    def spy(self, cmds):
        r = orig(self, cmds)
        got.append(list(r[0]))
        return r
    MujocoRemoteBackend._build_command = spy
    try:
        backend, svc = _make_rig()
        run(svc, backend)
    finally:
        MujocoRemoteBackend._build_command = orig
    return got


def test_capture_is_transparent_to_built_targets():
    off = _targets(_run_split)
    with cap.Capture():
        on = _targets(_run_split)
    assert on == off and len(on) == 3


def test_originals_restored_after_exit_and_on_exception():
    names = [(frs.FakeJointService, "_submit_batch"), (MujocoRemoteBackend, "submit_command"),
             (MujocoRemoteBackend, "_build_command"), (MujocoRemoteBackend, "__init__")]
    before = [cls.__dict__[n] for cls, n in names]
    with cap.Capture():
        assert [cls.__dict__[n] for cls, n in names] != before
    assert [cls.__dict__[n] for cls, n in names] == before
    with pytest.raises(RuntimeError):
        with cap.Capture():
            raise RuntimeError("boom")
    assert [cls.__dict__[n] for cls, n in names] == before
    # a second entry on the same object while active is refused, and does not corrupt restoration
    c = cap.Capture()
    with c:
        with pytest.raises(RuntimeError):
            c.__enter__()
    assert [cls.__dict__[n] for cls, n in names] == before


def test_bookkeeping_lock_is_never_held_across_an_original_call():
    """The 'originals' here are spies installed BEFORE capture; each tries the
    shim's own lock non-blockingly while running inside the wrapper."""
    held = []
    c = cap.Capture()

    def probe():
        ok = c._lock.acquire(blocking=False)
        if ok:
            c._lock.release()
        held.append(not ok)

    orig_batch = frs.FakeJointService._submit_batch
    orig_submit = MujocoRemoteBackend.submit_command
    orig_build = MujocoRemoteBackend._build_command

    def spy_batch(self, batch):
        probe()
        return orig_batch(self, batch)

    def spy_submit(self, cmd):
        probe()
        return orig_submit(self, cmd)

    def spy_build(self, cmds):
        probe()
        return orig_build(self, cmds)

    frs.FakeJointService._submit_batch = spy_batch
    MujocoRemoteBackend.submit_command = spy_submit
    MujocoRemoteBackend._build_command = spy_build
    try:
        with c:
            backend, svc = _make_rig()
            _run_split(svc, backend)
    finally:
        frs.FakeJointService._submit_batch = orig_batch
        MujocoRemoteBackend.submit_command = orig_submit
        MujocoRemoteBackend._build_command = orig_build
    assert len(held) == 2 + 8 + 3   # 2 batches + 8 submits + 3 builds all probed
    assert not any(held)


def test_generations_are_ordinals_of_constructed_backends():
    with cap.Capture() as c:
        b0, s0 = _make_rig()
        b1, s1 = _make_rig()
        s0._submit_batch(_batch(B1))
        _drain(b0)
        s1._submit_batch(_batch(B2))
        _drain(b1)
    recs = c.records()
    assert [r["gen"] for r in recs if r["type"] == "gen"] == [0, 1]
    assert [r["gen"] for r in recs if r["type"] == "batch"] == [0, 1]
    assert [r["gen"] for r in recs if r["type"] == "build"] == [0, 1]
    assert all(r["late"] is False for r in recs if r["type"] == "gen")


# ---------------------------------------------------------------------------
# Cross-check against a synthetic commands.jsonl
# ---------------------------------------------------------------------------

def _rows_from(recs, reset_between_gens=True):
    """A commands.jsonl that matches the capture: one joint_command row per build,
    in build order, with a reset row where the generation changes."""
    rows = []
    prev_gen = None
    for r in recs:
        if r["type"] != "build":
            continue
        if prev_gen is not None and r["gen"] != prev_gen and reset_between_gens:
            rows.append({"type": "reset", "seed": 1, "sim_step": 100, "wall_time_s": 0.0})
        prev_gen = r["gen"]
        rows.append({"type": "joint_command", "seq": r["seq"], "target_rad": list(r["target"]),
                     "mask": None, "compliant": None, "speed_limit_rad_s": None,
                     "torque_limit_percent": None})
    return rows


def _two_gen_capture():
    with cap.Capture() as c:
        b0, s0 = _make_rig()
        s0._submit_batch(_batch(B1))
        _drain(b0)
        s0._submit_batch(_batch(B2))
        _drain(b0)
        b1, s1 = _make_rig()
        s1._submit_batch(_batch(B2))
        _drain(b1)
    return c.records()


def test_cross_check_passes_on_a_matching_commands_jsonl():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    res = cap.cross_check(recs, rows)
    assert res.ok, res.failures
    assert res.n_rows == 3 and res.n_builds == 3 and res.n_generations == 2
    assert res.joined == {(0, 1): 0, (0, 2): 1, (1, 1): 2}


def test_cross_check_fails_on_a_dropped_row():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    rows = [r for i, r in enumerate(rows) if not (r["type"] == "joint_command" and r["seq"] == 2)]
    res = cap.cross_check(recs, rows)
    assert not res.ok and any("builds without a row [2]" in f for f in res.failures)


def test_cross_check_fails_on_an_extra_row():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    rows.append(copy.deepcopy(rows[-1]))
    rows[-1]["seq"] = 2
    res = cap.cross_check(recs, rows)
    assert not res.ok


def test_cross_check_fails_on_a_changed_target_even_by_one_bit():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    first = next(r for r in rows if r["type"] == "joint_command")
    import struct
    bits = struct.unpack("<q", struct.pack("<d", first["target_rad"][2]))[0] + 1
    first["target_rad"][2] = struct.unpack("<d", struct.pack("<q", bits))[0]
    res = cap.cross_check(recs, rows)
    assert not res.ok and any("idx 2" in f for f in res.failures)


def test_cross_check_distinguishes_negative_zero():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    for r in rows:
        if r["type"] == "joint_command":
            r["target_rad"][10] = -0.0 if r["target_rad"][10] == 0.0 else r["target_rad"][10]
    res = cap.cross_check(recs, rows)
    assert not res.ok


def test_cross_check_fails_closed_on_an_ambiguous_generation_boundary():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    # The second generation's row is renumbered to continue the first's seq, so the
    # strictly-increasing-seq delimiter finds ONE run where the capture has two.
    for r in rows:
        if r["type"] == "joint_command" and r["seq"] == 1 and r is not rows[0]:
            r["seq"] = 3
    res = cap.cross_check(recs, rows)
    assert not res.ok and any("generation count mismatch" in f for f in res.failures)


def test_cross_check_fails_closed_on_a_duplicated_seq():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    dup = copy.deepcopy(next(r for r in rows if r["type"] == "joint_command" and r["seq"] == 2))
    idx = max(i for i, r in enumerate(rows) if r.get("seq") == 2)
    rows.insert(idx + 1, dup)
    res = cap.cross_check(recs, rows)
    assert not res.ok


def test_cross_check_allowed_overrides_are_explicit_and_exact():
    recs = _two_gen_capture()
    rows = _rows_from(recs)
    jc = [r for r in rows if r["type"] == "joint_command"]
    jc[1]["target_rad"][3] = 0.987          # the harness's synthetic echo injection
    assert not cap.cross_check(recs, rows).ok          # never silently accepted
    res = cap.cross_check(recs, rows, allowed_overrides={1: {3: 0.987}})
    assert res.ok, res.failures
    assert res.overridden and res.overridden[0]["index"] == 3 and res.overridden[0]["row_ordinal"] == 1
    # a declared override that does not match the row's value fails
    assert not cap.cross_check(recs, rows, allowed_overrides={1: {3: 0.5}}).ok
    # an override does not excuse any OTHER index
    jc[1]["target_rad"][4] = 9.0
    assert not cap.cross_check(recs, rows, allowed_overrides={1: {3: 0.987}}).ok


def test_jsonl_round_trip(tmp_path):
    with cap.Capture() as c:
        backend, svc = _make_rig()
        _run_split(svc, backend)
    p = tmp_path / "capture.jsonl"
    c.write_jsonl(p)
    back = cap.load_jsonl(p)
    assert back == c.records()
    assert cap.classify(back)["n_mixed"] == 1
