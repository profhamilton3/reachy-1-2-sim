"""#55: planning honours the actual joint travel.

The IK service minimises without the joints' stops, so near a stop it answers
with a pose slightly past it.  The planner used to accept up to 0.5 deg past a
rounded stop (r_arm_yaw 90.1-90.5 against 89.954, r_wrist_pitch 45.3 against
44.977 in the 2026-10-07 crane plan), and the simulator clipped those
commands.  These tests pin that an out-of-travel answer is RE-SOLVED inside the
travel -- not clipped -- and that the re-solved pose reaches the same point.
Offline: the local kinematic chain only.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../src"))

from reachy_ai.motion import kinematics as K  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    JOINT_LIMITS_DEG, R_ARM_JOINTS, joint_limits, link_frames, pad_point,
    pad_residual, resolve_within_limits, within_limits)

#: Commanded rungs of the recorded 2026-10-07 r2c2 crane plan (hover rung +9
#: and rung +8): both past a stop.
HOVER_RUNG = [-82.732, -19.523, 90.495, -84.137, -49.905, 44.11, 35.163]
RUNG_8 = [-81.55, -18.86, 90.479, -83.35, -47.78, 45.26, 36.38]
#: Only r_arm_yaw past its stop: the arm's redundancy can hold the same point
#: AND the same hand attitude inside the travel.
YAW_ONLY = [-75.0, -17.5, 90.4, -80.0, -40.0, 40.0, 38.0]


def _overshoot(q):
    return max(max(lo - v, v - hi, 0.0) for v, (lo, hi) in zip(q, joint_limits("right")))


@pytest.mark.parametrize("q_service", [HOVER_RUNG, RUNG_8], ids=["hover", "rung8"])
def test_an_out_of_travel_answer_is_resolved_inside_and_reaches_the_point(q_service):
    assert not within_limits(q_service)                    # the defect
    target = pad_point(q_service)
    R = link_frames(q_service)[3]
    q, err = resolve_within_limits(pad_residual(target, R), q_service, tol_m=5e-4)
    assert within_limits(q)                                # inside, stops included
    assert _overshoot(q) == 0.0
    assert err < 5e-4                                      # reaches the same point
    assert float(np.linalg.norm(pad_point(q) - target)) < 5e-4


def test_clipping_instead_would_miss_the_point():
    """What the simulator did with the old command: the hand lands elsewhere."""
    clipped = [min(max(v, lo), hi) for v, (lo, hi) in zip(RUNG_8, joint_limits("right"))]
    miss = float(np.linalg.norm(pad_point(clipped) - pad_point(RUNG_8)))
    assert miss > 5e-4


def test_an_unreachable_target_is_not_forced_inside():
    """Far outside the workspace: the re-solve stays in the travel and reports
    the error instead of pretending."""
    q, err = resolve_within_limits(pad_residual([3.0, 0.0, 0.8], np.eye(3)), HOVER_RUNG, tol_m=5e-4)
    assert within_limits(q)
    assert err > 0.5


class _ServiceArm:
    """An IK service that answers past a stop, like the real one near it."""

    def __init__(self, answer):
        self.answer = list(answer)

    def inverse_kinematics(self, M, q0=None):
        return list(self.answer)

    def forward_kinematics(self, q):
        _s, _e, wrist, R = link_frames(q)
        F = np.eye(4)
        F[:3, :3] = R
        F[:3, 3] = wrist - K._BASE
        return F


def test_the_planner_resolves_rather_than_skips_or_clips(monkeypatch):
    target = pad_point(YAW_ONLY)
    planner = K.CartesianPlanner(_ServiceArm(YAW_ONLY), tol=1e-3)
    # The swept orientations are the planner's own; make the only one the
    # attitude the service answered with, so the test is about the stops.
    R = link_frames(YAW_ONLY)[3]
    monkeypatch.setattr(planner, "_R0", R)
    monkeypatch.setattr(planner, "_orientations", lambda prefer: [(0.0, 0.0)])
    q = planner.solve(tuple(target))
    assert within_limits(q)
    assert float(np.linalg.norm(pad_point(q) - target)) <= 1e-3
    assert K.attitude_error_deg(q, R) <= K.RESOLVE_MAX_ATTITUDE_DEG
    assert q != [min(max(v, lo), hi) for v, (lo, hi) in zip(YAW_ONLY, joint_limits("right"))]


def test_the_planner_rejects_a_resolve_that_would_turn_the_hand():
    """Position alone is not acceptance.  RUNG_8 has the wrist pitch past its
    stop too: inside the travel its point is reachable only by changing the
    hand's attitude, which is a different solution -- refused here, not
    reported as the requested orientation (the sweep tries others itself)."""
    planner = K.CartesianPlanner(_ServiceArm(RUNG_8), tol=1e-3)
    R8 = link_frames(RUNG_8)[3]
    assert planner._resolve_inside(R8, pad_point(RUNG_8), RUNG_8) is None


def test_an_accepted_resolve_holds_the_attitude_and_the_point():
    planner = K.CartesianPlanner(_ServiceArm(YAW_ONLY), tol=1e-3)
    R = link_frames(YAW_ONLY)[3]
    q = planner._resolve_inside(R, pad_point(YAW_ONLY), YAW_ONLY)
    assert q is not None and within_limits(q)
    assert K.attitude_error_deg(q, R) <= K.RESOLVE_MAX_ATTITUDE_DEG
    assert float(np.linalg.norm(pad_point(q) - pad_point(YAW_ONLY))) <= 1e-3
