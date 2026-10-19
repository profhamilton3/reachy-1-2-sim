"""Contract: the SDK's arm FK/IK speak in the gripper tip frame (#187).

The real `reachy_sdk` RightArm / LeftArm talk over gRPC to the fake server's
`FakeArmKinematicsService`, served in-process on a loopback port; nothing else
of the simulator runs.  Skipped where grpc, reachy_sdk or the server module
cannot be imported (the bare host); they run in the container, or in any env
that has reachy-sdk installed.

  * B2_SAFE_JOINTS through `r_arm.forward_kinematics()` give the pose the real
    robot gave, within 0.1 mm / 0.01 deg (tests/fixtures/kinematics/
    b2_safe_real_fk.json, with its provenance);
  * `inverse_kinematics(forward_kinematics(q))` reproduces the tip;
  * the planner's pads through the SDK are where they were before #187;
  * the left arm's FK is the local chain's left tip;
  * an `inverse_kinematics(pose)` with no q0 is solved from the arm's present
    joint positions, not from zero (#199); a seeded request is not changed.
"""

from __future__ import annotations

import json
import math
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

grpc = pytest.importorskip("grpc")
pytest.importorskip("reachy_sdk")
pytest.importorskip("scipy")

_HERE = os.path.dirname(__file__)
for _p in ("../..", "../../src"):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

try:
    import fake_reachy_server as F
    _SERVER_AVAILABLE = True
except ImportError:
    _SERVER_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not _SERVER_AVAILABLE, reason="fake_reachy_server not importable"
)

from reachy_ai.motion import kinematics as K  # noqa: E402

_FIXTURES = os.path.join(_HERE, "..", "fixtures", "kinematics")


def _load(name):
    with open(os.path.join(_FIXTURES, name), encoding="utf-8") as f:
        return json.load(f)


def _rot_err_deg(a, b):
    c = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


class _Backend:
    """Just the part of a joint backend the arm service reads: a snapshot."""

    def __init__(self):
        self.q = {}

    def set_present_deg(self, side, q_deg):
        for n, d in zip(F._ARM_CHAIN, q_deg):
            self.q[f"{side[0]}_{n}"] = math.radians(d)

    def latest_snapshot(self):
        return types.SimpleNamespace(joints={
            n: types.SimpleNamespace(present_position=v) for n, v in self.q.items()})


@pytest.fixture(scope="module")
def backend():
    return _Backend()


@pytest.fixture(scope="module")
def arms(backend):
    from reachy_sdk.arm import LeftArm, RightArm
    from reachy_sdk_api import arm_kinematics_pb2_grpc

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    arm_kinematics_pb2_grpc.add_ArmKinematicsServicer_to_server(
        F.FakeArmKinematicsService(backend), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")

    def make(cls):
        # The arm needs only each joint's name and uid to build its requests.
        joints = [types.SimpleNamespace(name=n, uid=i)
                  for i, n in enumerate(cls._required_joints)]
        return cls(joints, channel)

    yield make(RightArm), make(LeftArm)
    channel.close()
    server.stop(0)


def test_b2_safe_joints_give_the_real_robots_tip_pose(arms):
    r_arm, _ = arms
    b2 = _load("b2_safe_real_fk.json")
    T = r_arm.forward_kinematics(b2["joints_deg"])
    ref = np.asarray(b2["right_tip_pose"])
    assert np.linalg.norm(T[:3, 3] - ref[:3, 3]) <= 1e-4
    assert _rot_err_deg(T[:3, :3], ref[:3, :3]) <= 0.01


@pytest.mark.parametrize("name", ["present", "b2_safe", "b2_safe_roll_-45",
                                  "b2_safe_roll_+45", "crane_pick_rung_00",
                                  "crane_dest_rung_09"])
def test_inverse_kinematics_round_trips_the_tip(arms, name):
    r_arm, _ = arms
    q = _load("pad_invariance_187.json")["poses"][name]["joints_deg"]
    T = r_arm.forward_kinematics(q)
    # Unseeded only off the crane's ladders (see test_tip_frame_187._UNSEEDED).
    for q0 in ((q,) if name.startswith("crane_") else (q, None)):
        back = r_arm.forward_kinematics(r_arm.inverse_kinematics(T, q0=q0))
        assert np.linalg.norm(back[:3, 3] - T[:3, 3]) <= 1e-4, q0
        assert _rot_err_deg(back[:3, :3], T[:3, :3]) <= 0.01, q0


def test_the_planners_pads_through_the_sdk_are_where_they_were(arms):
    r_arm, _ = arms
    planner = K.CartesianPlanner(r_arm)
    for name, pose in _load("pad_invariance_187.json")["poses"].items():
        pad = np.asarray(planner.fk_world(pose["joints_deg"]))
        assert np.linalg.norm(pad - pose["pad_world_m"]) <= 1e-4, name


def test_the_left_arms_fk_is_the_left_tip(arms):
    _, l_arm = arms
    rng = np.random.default_rng(187)
    for _ in range(10):
        q = list(rng.uniform(-40.0, 40.0, 7))
        q[3] = -abs(q[3])
        T = l_arm.forward_kinematics(q)
        assert np.allclose(T, K.tip_pose(q, side="left"), atol=1e-9)


# ── No q0: seeded from the arm's present joints (#199) ────────────────────────

#: Crane-ladder poses the tip IK stalls on from zero though the pre-#187 wrist
#: IK solved them (#187 PR, "Known limitations").
_STALLED_FROM_ZERO = ["crane_pick_rung_09", "crane_pick_rung_10", "crane_pick_rung_18",
                      "crane_dest_rung_15", "crane_dest_rung_18"]


def _solves(arm, T, **kw):
    try:
        back = arm.forward_kinematics(arm.inverse_kinematics(T, **kw))
    except ValueError:
        return False
    return (np.linalg.norm(back[:3, 3] - T[:3, 3]) <= 1e-4
            and _rot_err_deg(back[:3, :3], T[:3, :3]) <= 0.01)


@pytest.mark.parametrize("name", _STALLED_FROM_ZERO)
def test_unseeded_ik_starts_from_the_present_pose(arms, backend, name):
    r_arm, _ = arms
    poses = _load("pad_invariance_187.json")["poses"]
    T = r_arm.forward_kinematics(poses[name]["joints_deg"])
    backend.q.clear()
    assert not _solves(r_arm, T), "stalls from zero (nothing present)"
    backend.set_present_deg("right", poses["crane_pick_rung_12"]["joints_deg"])
    assert _solves(r_arm, T)


def test_a_seeded_request_ignores_the_present_pose(arms, backend):
    r_arm, _ = arms
    poses = _load("pad_invariance_187.json")["poses"]
    T = r_arm.forward_kinematics(poses["crane_pick_rung_09"]["joints_deg"])
    q0 = poses["b2_safe"]["joints_deg"]
    backend.q.clear()
    alone = r_arm.inverse_kinematics(T, q0=q0)
    backend.set_present_deg("right", poses["crane_pick_rung_12"]["joints_deg"])
    assert r_arm.inverse_kinematics(T, q0=q0) == alone


def test_the_left_arm_is_seeded_from_the_left_joints(arms, backend):
    _, l_arm = arms
    q = [10.0, 5.0, 0.0, -50.0, 0.0, 0.0, 0.0]
    T = l_arm.forward_kinematics(q)
    backend.q.clear()
    backend.set_present_deg("left", q)
    # The right arm's joints are absent, so only the left ones can have seeded it.
    assert _solves(l_arm, T)
    assert l_arm.inverse_kinematics(T) == l_arm.inverse_kinematics(T, q0=q)


def test_no_backend_is_the_old_zero_start():
    svc = F.FakeArmKinematicsService()
    assert svc._present_chain("right") is None
