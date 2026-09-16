"""Issue #116: merged pending joint commands, reset-boundary drop, and the
`State.cmd_seq` diagnostic.

Offline throughout: a real `MjModel`/`SimState` (no server, no socket, no
SDK, no motion) -- built exactly as `tests/unit/test_placement.py` builds
its `scene_xml`/`pool_doc` fixtures, so this pins the actual joint indexing
(`JOINT_TABLE`) and actuator model (`ActuatorController`) rather than a
hand-rolled stand-in.
"""

import logging
import os
import random
import sys
import threading
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../native_mujoco"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from scene_io import load_scene  # noqa: E402

mujoco = pytest.importorskip("mujoco")

from joint_map import JOINT_TABLE, NUM_JOINTS  # noqa: E402
from objects import build_scene_model_xml  # noqa: E402

_SCENES = os.path.join(os.path.dirname(__file__), "../../scenes")
_POOL_SCENE = os.path.join(_SCENES, "FWDCenterLabSivaPool.yaml")

#: Right-arm mjcf indices, in JOINT_TABLE order (0..7): r_shoulder_pitch,
#: r_shoulder_roll, r_arm_yaw, r_elbow_pitch, r_forearm_yaw, r_wrist_pitch,
#: r_wrist_roll, r_gripper.
R_IDX = [e.mjcf_index for e in JOINT_TABLE if e.mjcf_index < 8]


@pytest.fixture(scope="module")
def pool_doc():
    return load_scene(_POOL_SCENE)


@pytest.fixture(scope="module")
def scene_xml(pool_doc):
    return build_scene_model_xml(pool_doc)


@pytest.fixture
def sim(scene_xml, pool_doc):
    """A fresh SimState per test -- apply_pending mutates actuator/goal
    state, so tests must not share one."""
    from server import SimState
    model = mujoco.MjModel.from_xml_string(scene_xml)
    return SimState(model, scene_doc=pool_doc)


def _server_for(sim_state):
    """A `ReachyMujocoServer` with only what `_build_state` touches set --
    same pattern as test_placement.py's TestBroadcastFanOut (__new__, no
    __init__, no socket)."""
    from server import ReachyMujocoServer
    srv = ReachyMujocoServer.__new__(ReachyMujocoServer)
    srv._sim = sim_state
    srv._record_contacts = False
    srv._seq = 0
    return srv


def cmd(seq, base_targets, *, targets=None, compliant=None, speed=None,
        torque=None, mask=None):
    """Build a 21-joint joint_command dict shaped like the real bridge's
    (native_mujoco/protocol.py JointCommand): target_rad always a full
    21-vector (defaulting to `base_targets`, i.e. "the current goal
    vector" -- mujoco_remote_backend.py always seeds from `_last_target`),
    the three flag lists `None` unless overrides are given for this
    message.

    `targets`/`compliant`/`speed`/`torque` are `{mjcf_index: value}`
    overrides; `mask` is the raw `List[bool]` or `None` (apply all).
    """
    tgt = list(base_targets)
    for i, v in (targets or {}).items():
        tgt[i] = v

    def _flags(overrides):
        if overrides is None:
            return None
        out = [None] * NUM_JOINTS
        for i, v in overrides.items():
            out[i] = v
        return out

    return {
        "type": "joint_command",
        "seq": seq,
        "target_rad": tgt,
        "mask": mask,
        "compliant": _flags(compliant),
        "speed_limit_rad_s": _flags(speed),
        "torque_limit_percent": _flags(torque),
    }


def _base(sim_state):
    return [st.goal_position for st in sim_state.controller.state]


class TestSplitTurnOn:
    """T1: the reported shape -- turn_on sent as seq 1 (shoulder_pitch
    compliant=False), seq 2 (the other seven), seq 3 (targets only,
    BACK-like). The old single-slot _pending_cmd lost seq 2 when seq 3
    replaced it before one apply_pending() tick; the merge must not."""

    def test_all_eight_stiff_after_one_apply(self, sim):
        base = _base(sim)
        back = {i: 0.3 + 0.01 * i for i in R_IDX}

        sim.submit_command(cmd(1, base, compliant={R_IDX[0]: False}))
        sim.submit_command(
            cmd(2, base, compliant={i: False for i in R_IDX[1:]}))
        sim.submit_command(cmd(3, base, targets=back))
        sim.apply_pending()

        for i in R_IDX:
            assert sim.controller.state[i].compliant is False
            assert sim.controller.state[i].goal_position == pytest.approx(back[i])
        assert sim._cmd_seq == 3

    def test_build_state_reports_the_applied_seq(self, sim):
        base = _base(sim)
        sim.submit_command(cmd(1, base, compliant={R_IDX[0]: False}))
        sim.submit_command(
            cmd(2, base, compliant={i: False for i in R_IDX[1:]}))
        sim.submit_command(cmd(3, base, targets={i: 0.2 for i in R_IDX}))
        sim.apply_pending()

        state = _server_for(sim)._build_state()
        assert state.cmd_seq == 3


class TestOneShotFieldMerge:
    def test_newer_explicit_compliance_supersedes(self, sim):
        base = _base(sim)
        j = R_IDX[3]
        sim.submit_command(cmd(1, base, compliant={j: False}))
        sim.submit_command(cmd(2, base, compliant={j: True}))
        sim.apply_pending()
        assert sim.controller.state[j].compliant is True

    def test_one_shot_fields_survive_position_only_commands(self, sim):
        base = _base(sim)
        j = R_IDX[3]
        sim.submit_command(
            cmd(1, base, compliant={j: False}, speed={j: 0.5}, torque={j: 40.0}))
        sim.submit_command(cmd(2, base, targets={j: 0.1}))
        sim.submit_command(cmd(3, base, targets={j: 0.2}))
        sim.apply_pending()

        st = sim.controller.state[j]
        assert st.compliant is False
        assert st.speed_limit == pytest.approx(0.5)
        assert st.torque_limit == pytest.approx(40.0)
        assert st.goal_position == pytest.approx(0.2)

    def test_partial_supersede_only_the_resent_field_changes(self, sim):
        base = _base(sim)
        j = R_IDX[3]
        sim.submit_command(cmd(1, base, compliant={j: False}, torque={j: 40.0}))
        sim.submit_command(cmd(2, base, torque={j: 70.0}))
        sim.apply_pending()

        st = sim.controller.state[j]
        assert st.compliant is False       # survived (not resent)
        assert st.torque_limit == pytest.approx(70.0)  # superseded

    def test_mask_true_only_for_one_joint(self, sim):
        base = _base(sim)
        a = {i: 0.4 + 0.001 * i for i in range(NUM_JOINTS)}
        sim.submit_command(cmd(1, base, targets=a))  # mask=None: full A
        b5 = 0.9
        mask = [False] * NUM_JOINTS
        mask[R_IDX[5]] = True
        sim.submit_command(cmd(2, base, targets={R_IDX[5]: b5}, mask=mask))
        sim.apply_pending()

        for i in range(NUM_JOINTS):
            expected = b5 if i == R_IDX[5] else a[i]
            assert sim.controller.state[i].goal_position == pytest.approx(expected)


class TestResetBoundary:
    def test_reset_drops_a_stale_command_no_replay(self, sim, caplog):
        base = _base(sim)
        # Compliance change #1: applied BEFORE the reset, in its own tick --
        # today's behaviour (compliance persists across reset) is pinned
        # here, separately from the dropped command below.
        already = R_IDX[0]
        sim.submit_command(cmd(1, base, compliant={already: False}))
        sim.apply_pending()
        assert sim.controller.state[already].compliant is False

        # Far-from-keyframe targets + a compliance change on a DIFFERENT
        # joint, still pending when the reset is asked for.
        dropped_joint = R_IDX[1]
        far = {i: 1.0 for i in R_IDX}
        sim.submit_command(
            cmd(2, base, targets=far, compliant={dropped_joint: False}))
        sim.submit_reset({"request_id": "r1"})

        with caplog.at_level(logging.WARNING):
            sim.apply_pending()

        assert sim.step == 0
        # No replay: every right-arm goal is the post-reset qpos, not `far`.
        for i in R_IDX:
            assert sim.controller.state[i].goal_position == pytest.approx(
                sim.data.qpos[i])
            assert sim.controller.state[i].goal_position != pytest.approx(1.0)
        assert sim._pending_update is None
        assert any("seq 2" in r.message and "1 merged" in r.message
                  for r in caplog.records)

        # The dropped command's compliance flag was never applied...
        assert sim.controller.state[dropped_joint].compliant is True
        # ...but the compliance change from BEFORE the reset persisted
        # through it (sync_targets_to_current only rewrites goal_position).
        assert sim.controller.state[already].compliant is False

        state = _server_for(sim)._build_state()
        assert any("seq 2 (1 merged) dropped at reset" in w
                  for w in state.warnings)

    def test_reset_then_command_in_the_same_tick_is_dropped(self, sim, caplog):
        base = _base(sim)
        sim.submit_reset({"request_id": "r1"})
        sim.submit_command(cmd(1, base, targets={R_IDX[0]: 1.0}))

        with caplog.at_level(logging.WARNING):
            sim.apply_pending()

        assert sim.step == 0
        assert sim._pending_update is None
        assert any("seq 1" in r.message and "1 merged" in r.message
                  for r in caplog.records)
        assert sim.controller.state[R_IDX[0]].goal_position == pytest.approx(
            sim.data.qpos[R_IDX[0]])

        state = _server_for(sim)._build_state()
        assert any("seq 1 (1 merged) dropped at reset" in w
                  for w in state.warnings)


class TestConsumedOnce:
    def test_second_apply_with_nothing_new_is_a_no_op(self, sim):
        base = _base(sim)
        j = R_IDX[2]
        sim.submit_command(cmd(1, base, targets={j: 0.33}, compliant={j: False}))
        sim.apply_pending()
        assert sim._cmd_seq == 1

        sim.apply_pending()  # nothing submitted since
        assert sim._cmd_seq == 1
        assert sim.controller.state[j].goal_position == pytest.approx(0.33)
        assert sim.controller.state[j].compliant is False


class TestOnePerTickUnchanged:
    """No behaviour change for the common case: one command per tick,
    applied alone each time, must match applying each command in isolation
    (the pre-#116 semantics for the non-coalescing case)."""

    def test_alternating_submit_apply_100_times(self, sim):
        base = _base(sim)
        rng = random.Random(7)
        for seq in range(1, 101):
            j = rng.choice(R_IDX)
            targets = {i: rng.uniform(-0.3, 0.3) for i in R_IDX}
            compliant = {j: bool(seq % 2)}
            sim.submit_command(cmd(seq, base, targets=targets, compliant=compliant))
            sim.apply_pending()
            for i in R_IDX:
                assert sim.controller.state[i].goal_position == pytest.approx(
                    targets[i])
            assert sim.controller.state[j].compliant is bool(seq % 2)
            assert sim._cmd_seq == seq


class TestThreadStress:
    """T8: 2000 commands from a producer thread (alternating a
    compliance/limit-flag-bearing message and a targets-only message) while
    the test thread calls apply_pending() in a loop with random short
    sleeps. No exception, and once the producer is done and one final
    apply_pending() has run, every joint's state equals the LAST explicit
    value submitted for it -- proved by tracking expected state in the
    producer as it submits, independent of how apply_pending interleaved.
    """

    def test_merge_converges_to_last_explicit_values_under_concurrency(self, sim):
        base = _base(sim)
        n = 2000
        rng = random.Random(20260915)
        expected_targets = list(base)
        expected_compliant: dict = {}
        expected_speed: dict = {}
        expected_torque: dict = {}
        errors = []

        def producer():
            nonlocal expected_targets
            try:
                for i in range(n):
                    seq = i + 1
                    tgt = {j: rng.uniform(-0.3, 0.3) for j in range(NUM_JOINTS)}
                    compliant = speed = torque = None
                    if i % 2 == 0:
                        compliant, speed, torque = {}, {}, {}
                        for j in range(NUM_JOINTS):
                            if rng.random() < 0.3:
                                compliant[j] = rng.random() < 0.5
                            if rng.random() < 0.3:
                                speed[j] = rng.uniform(0.0, 2.0)
                            if rng.random() < 0.3:
                                torque[j] = rng.uniform(0.0, 100.0)
                        expected_compliant.update(compliant)
                        expected_speed.update(speed)
                        expected_torque.update(torque)
                    msg = cmd(seq, base, targets=tgt, compliant=compliant or None,
                              speed=speed or None, torque=torque or None)
                    expected_targets = [tgt[j] for j in range(NUM_JOINTS)]
                    sim.submit_command(msg)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        t = threading.Thread(target=producer)
        t.start()
        jitter = random.Random(1)
        while t.is_alive():
            sim.apply_pending()
            time.sleep(jitter.uniform(0.0, 0.005))
        t.join()
        assert not errors
        sim.apply_pending()  # final tick: guarantee the last batch is applied

        for j in range(NUM_JOINTS):
            assert sim.controller.state[j].goal_position == pytest.approx(
                expected_targets[j])
            if j in expected_compliant:
                assert sim.controller.state[j].compliant == expected_compliant[j]
            if j in expected_speed:
                assert sim.controller.state[j].speed_limit == pytest.approx(
                    expected_speed[j])
            if j in expected_torque:
                assert sim.controller.state[j].torque_limit == pytest.approx(
                    expected_torque[j])


class TestBridgeShapedSplitTurnOn:
    """Pins the REAL bridge message shapes (mujoco_remote_backend.py's
    _build_command / kinematic_backend.JointCommand), not hand-typed dicts
    -- the same construction test_mujoco_remote_backend.py already uses."""

    def test_real_build_command_output_all_eight_stiff(self, sim):
        from kinematic_backend import JointCommand as SDKJointCommand
        from mujoco_remote_backend import MujocoRemoteBackend

        right_arm_uids = [e.uid for e in JOINT_TABLE if e.mjcf_index < 8]
        backend = MujocoRemoteBackend(url="ws://localhost:9999")

        t1, c1, s1, tq1 = backend._build_command(
            [SDKJointCommand(uid=right_arm_uids[0], compliant=False)])
        t2, c2, s2, tq2 = backend._build_command(
            [SDKJointCommand(uid=u, compliant=False) for u in right_arm_uids[1:]])
        back_cmds = [SDKJointCommand(uid=u, goal_position=0.3 + 0.01 * i)
                     for i, u in enumerate(right_arm_uids)]
        t3, c3, s3, tq3 = backend._build_command(back_cmds)

        sim.submit_command({"type": "joint_command", "seq": 1, "target_rad": t1,
                            "mask": None, "compliant": c1,
                            "speed_limit_rad_s": s1, "torque_limit_percent": tq1})
        sim.submit_command({"type": "joint_command", "seq": 2, "target_rad": t2,
                            "mask": None, "compliant": c2,
                            "speed_limit_rad_s": s2, "torque_limit_percent": tq2})
        sim.submit_command({"type": "joint_command", "seq": 3, "target_rad": t3,
                            "mask": None, "compliant": c3,
                            "speed_limit_rad_s": s3, "torque_limit_percent": tq3})
        sim.apply_pending()

        for i in R_IDX:
            assert sim.controller.state[i].compliant is False
            assert sim.controller.state[i].goal_position == pytest.approx(t3[i])
