"""What goal and compliance MujocoRemoteBackend reports, rule by rule.

Offline and SDK-free: states go in through `_ingest_state`, commands through
`submit_command` / `_build_command`, and the report comes out of
`to_kinematic_snapshot` (what FakeJointService serves).  The end-to-end path
through the real reachy_sdk is tests/integration/test_goal_reporting_sdk_path.py.

The bug these pin: the bridge reported goal_position = present position, which
reachy_sdk caches, starts every goto from, and could echo back as a setpoint.
"""
import math
import os
import random
import sys
import threading

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from kinematic_backend import JOINT_DEFS, JointCommand  # noqa: E402
from mujoco_remote_backend import MujocoRemoteBackend  # noqa: E402

UID = dict(JOINT_DEFS)
SP = UID["r_shoulder_pitch"]
EL = UID["r_elbow_pitch"]


def _backend():
    return MujocoRemoteBackend(url="ws://localhost:9999")


def _state(b, step, pos=None, compliant=None):
    pos = pos or {}
    compliant = compliant or {}
    b._ingest_state({
        "seq": step, "sim_step": step, "sim_time_s": step * 0.002,
        "joints": [{"name": n, "uid": u, "position_rad": pos.get(u, 0.0),
                    "velocity_rad_s": 0.0, "effort": 0.0,
                    "compliant": compliant.get(u, False)}
                   for n, u in JOINT_DEFS],
    })


def _goal(b, uid):
    name = next(n for n, u in JOINT_DEFS if u == uid)
    return b.to_kinematic_snapshot().joints[name].goal_position


def _build(b):
    with b._lock:
        cmds = list(b._pending_cmds)
        b._pending_cmds.clear()
        return b._build_command(cmds)


# ── startup ───────────────────────────────────────────────────────────────────

class TestStartup:
    def test_before_any_state_reports_present(self):
        b = _backend()
        assert _goal(b, SP) == 0.0

    def test_first_state_seeds_and_a_later_sag_is_not_the_goal(self):
        b = _backend()
        _state(b, 10, {SP: -0.5})
        assert _goal(b, SP) == pytest.approx(-0.5)
        _state(b, 20, {SP: -0.45})               # the arm sags; nothing was commanded
        assert _goal(b, SP) == pytest.approx(-0.5)
        assert b.to_kinematic_snapshot().joints["r_shoulder_pitch"].present_position == pytest.approx(-0.45)

    def test_accepted_goal_is_reported_before_it_is_sent(self):
        b = _backend()
        _state(b, 10, {SP: -0.5})
        b.submit_command(JointCommand(uid=SP, goal_position=-0.7))
        assert _goal(b, SP) == pytest.approx(-0.7)
        _state(b, 20, {SP: -0.52})
        assert _goal(b, SP) == pytest.approx(-0.7), "a state must never overwrite a commanded goal"

    def test_newest_accepted_goal_wins(self):
        b = _backend()
        _state(b, 10)
        for g in (-0.1, -0.2, -0.3):
            b.submit_command(JointCommand(uid=SP, goal_position=g))
        assert _goal(b, SP) == pytest.approx(-0.3)
        t, *_ = _build(b)
        assert t[0] == pytest.approx(-0.3)


# ── joints not commanded ─────────────────────────────────────────────────────

def test_uncommanded_joint_keeps_its_baseline_goal():
    b = _backend()
    _state(b, 10, {EL: -1.0, SP: -0.5})
    b.submit_command(JointCommand(uid=SP, goal_position=-0.6))
    t, *_ = _build(b)
    _state(b, 20, {EL: -0.9, SP: -0.6})
    assert t[3] == pytest.approx(-1.0)
    assert _goal(b, EL) == pytest.approx(-1.0)


# ── compliance ────────────────────────────────────────────────────────────────

class TestCompliance:
    def test_compliant_flag_is_reported_from_the_native_state(self):
        b = _backend()
        _state(b, 10, compliant={SP: True})
        snap = b.to_kinematic_snapshot()
        assert snap.joints["r_shoulder_pitch"].compliant is True
        assert snap.joints["r_elbow_pitch"].compliant is False

    def test_a_compliant_joint_goal_follows_its_pose(self):
        b = _backend()
        _state(b, 10, {EL: -1.0}, {EL: True})
        _state(b, 20, {EL: -0.3}, {EL: True})    # moved by hand
        assert _goal(b, EL) == pytest.approx(-0.3)

    def test_stiffening_without_a_goal_holds_where_it_is(self):
        b = _backend()
        _state(b, 10, {EL: -1.0})
        b.submit_command(JointCommand(uid=EL, compliant=True))
        _build(b)
        _state(b, 20, {EL: -0.3}, {EL: True})
        b.submit_command(JointCommand(uid=EL, compliant=False))
        t, c, *_ = _build(b)
        assert c[3] is False
        assert t[3] == pytest.approx(-0.3), "re-stiffening must not snap back to the old target"
        assert _goal(b, EL) == pytest.approx(-0.3)

    def test_stale_compliant_state_after_stiffening_does_not_move_the_goal(self):
        b = _backend()
        _state(b, 10, {EL: -0.3}, {EL: True})
        b.submit_command(JointCommand(uid=EL, compliant=False))
        _build(b)
        b.submit_command(JointCommand(uid=EL, goal_position=-0.8))
        _state(b, 20, {EL: -0.31}, {EL: True})   # sent before the native side applied it
        assert _goal(b, EL) == pytest.approx(-0.8)
        t, *_ = _build(b)
        assert t[3] == pytest.approx(-0.8)

    def test_stiffening_an_already_stiff_joint_keeps_its_target(self):
        b = _backend()
        _state(b, 10, {EL: -1.0})
        b.submit_command(JointCommand(uid=EL, goal_position=-1.2))
        _build(b)
        _state(b, 20, {EL: -1.15})               # stiff, sagging a little
        b.submit_command(JointCommand(uid=EL, compliant=False))
        t, *_ = _build(b)
        assert t[3] == pytest.approx(-1.2)

    def test_goal_in_the_same_batch_as_stiffening_wins(self):
        b = _backend()
        _state(b, 10, {EL: -0.3}, {EL: True})
        b.submit_command(JointCommand(uid=EL, compliant=False))
        b.submit_command(JointCommand(uid=EL, goal_position=-0.9))
        t, *_ = _build(b)
        assert t[3] == pytest.approx(-0.9)
        assert _goal(b, EL) == pytest.approx(-0.9)


# ── reset ─────────────────────────────────────────────────────────────────────

class TestReset:
    def _commanded(self):
        b = _backend()
        _state(b, 1000, {SP: -0.7})
        b.submit_command(JointCommand(uid=SP, goal_position=-0.7))
        _build(b)
        return b

    def test_request_reset_forgets_goals(self):
        b = self._commanded()
        b.request_reset()
        assert b._last_target is None
        assert _goal(b, SP) == pytest.approx(-0.7)   # present, until a baseline arrives

    def test_pre_reset_states_after_the_ack_do_not_seed(self):
        """The native server queues states and acks separately, so pre-reset
        states can still arrive after the ack."""
        b = self._commanded()
        b.request_reset()
        b._reset_ack_step = 0                    # ack received (sim_step 0)
        _state(b, 1010, {SP: -0.68})             # still pre-reset
        assert b._last_target is None
        _state(b, 0, {SP: 0.0})                  # the reset pose
        assert b._last_target is not None
        assert _goal(b, SP) == pytest.approx(0.0)

    def test_step_drop_seeds_even_before_the_ack(self):
        b = self._commanded()
        b.request_reset()
        _state(b, 10, {SP: 0.02})
        assert _goal(b, SP) == pytest.approx(0.02)

    def test_goals_waiting_to_be_sent_survive_the_reseed(self):
        b = self._commanded()
        b.request_reset()
        b.submit_command(JointCommand(uid=SP, goal_position=-0.2))   # held during the reset
        _state(b, 0, {SP: 0.0})
        assert _goal(b, SP) == pytest.approx(-0.2)
        t, *_ = _build(b)
        assert t[0] == pytest.approx(-0.2)
        assert t[3] == pytest.approx(0.0)

    def test_ack_timeout_takes_the_next_state_as_the_baseline(self):
        b = self._commanded()
        b.request_reset()
        b._on_reset_timeout()
        _state(b, 1010, {SP: -0.69})
        assert b._last_target is not None
        assert _goal(b, SP) == pytest.approx(-0.69)

    def test_a_reset_nobody_here_asked_for_still_reseeds(self):
        b = self._commanded()
        _state(b, 0, {SP: 0.0})                  # native reset by another client
        assert _goal(b, SP) == pytest.approx(0.0)
        t, *_ = _build(b)
        assert t[0] == pytest.approx(0.0)


# ── concurrent updates ────────────────────────────────────────────────────────

def test_concurrent_submits_states_and_builds_never_report_a_present_pose():
    """gRPC workers submit and read while the bridge thread ingests and builds.
    Every reported goal must be the baseline or a value someone submitted --
    never a present position, never torn -- and the newest submission wins."""
    b = _backend()
    _state(b, 10, {SP: -0.5})
    submitted = {-0.5}
    lock = threading.Lock()
    stop = threading.Event()
    bad = []

    def writer(k):
        rnd = random.Random(k)
        while not stop.is_set():
            g = round(rnd.uniform(-1.0, 0.0), 6)
            with lock:
                submitted.add(g)
            b.submit_command(JointCommand(uid=SP, goal_position=g))

    def bridge():
        step = 20
        rnd = random.Random(99)
        while not stop.is_set():
            _state(b, step, {SP: 5.0 + rnd.random()})   # present poses are all > 5
            step += 10
            _build(b)

    def reader():
        while not stop.is_set():
            g = _goal(b, SP)
            with lock:
                ok = any(math.isclose(g, s, abs_tol=1e-9) for s in submitted)
            if not ok:
                bad.append(g)

    ts = [threading.Thread(target=writer, args=(k,)) for k in range(3)]
    ts += [threading.Thread(target=bridge), threading.Thread(target=reader), threading.Thread(target=reader)]
    for t in ts:
        t.start()
    threading.Event().wait(1.0)
    stop.set()
    for t in ts:
        t.join()
    assert not bad, f"reported goals that nobody submitted: {bad[:5]}"
    last = -0.123456
    b.submit_command(JointCommand(uid=SP, goal_position=last))
    assert _goal(b, SP) == pytest.approx(last)
    t, *_ = _build(b)
    assert t[0] == pytest.approx(last)
