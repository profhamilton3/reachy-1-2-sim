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
  * the left arm's FK is the local chain's left tip.
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


@pytest.fixture(scope="module")
def arms():
    from reachy_sdk.arm import LeftArm, RightArm
    from reachy_sdk_api import arm_kinematics_pb2_grpc

    server = grpc.server(ThreadPoolExecutor(max_workers=2))
    arm_kinematics_pb2_grpc.add_ArmKinematicsServicer_to_server(
        F.FakeArmKinematicsService(), server)
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
