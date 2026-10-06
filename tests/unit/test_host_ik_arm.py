"""The host IK mirror is the simulator's IK, or these tests say it is not.

`host_ik_arm.py` re-states `fake_reachy_server._arm_fk/_arm_ik` so the planner
can be exercised offline with real IK.  Two checks keep that honest:

  * a DRIFT PIN -- the server's kinematics source, extracted with `ast` and
    whitespace-stripped, hashes to the value recorded in the mirror;
  * an FK CHECK -- the mirror's FK agrees with `link_frames` (the planner's own
    pure-Python chain) at random poses.
"""

import os
import sys

import numpy as np
import pytest

pytest.importorskip("scipy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import host_ik_arm  # noqa: E402
from reachy_ai.motion.kinematics import link_frames  # noqa: E402


def test_the_mirror_still_matches_the_fake_servers_kinematics_source():
    assert host_ik_arm.source_digest() == host_ik_arm.SOURCE_SHA256, (
        "fake_reachy_server's arm kinematics changed: re-derive "
        "tests/unit/host_ik_arm.py and update SOURCE_SHA256")


def test_mirror_fk_equals_link_frames_at_random_poses():
    rng = np.random.default_rng(56)
    lim = np.degrees(np.array(host_ik_arm._R_ARM_LIMITS))
    for _ in range(50):
        q = rng.uniform(lim[:, 0], lim[:, 1])
        T = host_ik_arm.HostIKArm().forward_kinematics(q)
        _s, _e, wrist, R = link_frames(list(q))
        assert np.allclose(T[:3, 3] + np.array([0.0, 0.0, 1.0]), wrist, atol=1e-9)
        assert np.allclose(T[:3, :3], R, atol=1e-9)


def test_mirror_ik_round_trips_a_reachable_pose():
    arm = host_ik_arm.HostIKArm()
    q = [-40.0, -20.0, 5.0, -70.0, 0.0, 10.0, 0.0]
    T = arm.forward_kinematics(q)
    back = arm.inverse_kinematics(T, q0=q)
    assert np.allclose(arm.forward_kinematics(back)[:3, 3], T[:3, 3], atol=1e-3)
