"""Unit tests for tools/goalfix_cmp/pathcheck.py (C0-C8)."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402


def full_pose(**kw) -> dict:
    base = {j: 0.0 for j in mf.R_JOINTS}
    base.update(kw)
    return base


START = full_pose()
SWING_1_POSE = full_pose(r_shoulder_pitch=-0.3)
HOVER_POSE = full_pose(r_shoulder_pitch=-0.70, r_shoulder_roll=-0.17,
                        r_elbow_pitch=-1.05, r_wrist_pitch=-0.26)
REST_SHUT_POSE = full_pose(r_shoulder_pitch=-0.70, r_shoulder_roll=-0.17,
                            r_elbow_pitch=-0.79, r_wrist_pitch=-0.17, r_wrist_roll=0.52)
REST_POSE = dict(REST_SHUT_POSE, r_gripper=-0.7)
ROUTE = [
    mf.Waypoint("SWING_1", SWING_1_POSE, 1.0),
    mf.Waypoint("HOVER", HOVER_POSE, 1.2),
    mf.Waypoint("REST_SHUT", REST_SHUT_POSE, 1.0),
    mf.Waypoint("REST", REST_POSE, 1.0),
]


def _fly(**kwargs):
    sim = mf.FlightSim(START, **kwargs)
    sim.fly(ROUTE)
    return sim.result()


def _targets21(command_rows):
    return np.array([row["target_rad"] for row in command_rows])


def _t_hi(result):
    # Lockstep sim: command i's own tick reports it (see FlightSim docs).
    return [row_i * mf.TICK_S for row_i in range(len(result.command_rows))]


def _run(result):
    targets21 = _targets21(result.command_rows)
    t_hi_s = _t_hi(result)
    return pc.run_all(targets21, t_hi_s, START, ROUTE, guard=())


class TestCleanRoute:
    def test_all_checks_pass(self):
        result = _fly()
        results = _run(result)
        for cid in pc.ALL_CHECKS:
            assert results[cid].passed, (cid, results[cid].detail)


class TestC1:
    def test_start_discontinuity_fails_c1(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        targets21[0][0] += 0.01  # perturb the very first command's shoulder_pitch
        t_hi_s = _t_hi(result)
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=())
        assert not results["C1"].passed
        assert results["C1"].first_violation_index == 0

    def test_boundary_at_1e_minus_6(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        targets21[0][0] += 9e-7  # inside tolerance
        t_hi_s = _t_hi(result)
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=())
        assert results["C1"].passed

        targets21b = _targets21(result.command_rows)
        targets21b[0][0] += 2e-6  # outside tolerance
        results_b = pc.run_all(targets21b, t_hi_s, START, ROUTE, guard=())
        assert not results_b["C1"].passed


class TestC2Skew:
    def test_30ms_skew_passes(self):
        result = _fly(skew_ms={"r_wrist_pitch": 30.0})
        results = _run(result)
        assert results["C2"].passed, results["C2"].detail

    def test_31ms_skew_fails(self):
        result = _fly(skew_ms={"r_wrist_pitch": 90.0})
        results = _run(result)
        assert not results["C2"].passed


class TestC3:
    def test_tau_decrease_fails(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        assignment = seg.assign_goals(ROUTE, START, [
            {name: float(row[i]) for i, name in enumerate(mf.R_JOINTS)} for row in targets21])
        hover_idx = 1
        idxs = [i for i, g in enumerate(assignment.goal_index) if g == hover_idx]
        mid = idxs[len(idxs) // 2]
        # Swap two interior samples so shoulder_pitch's implied tau goes
        # backwards for one tick.
        targets21[[mid, mid + 1]] = targets21[[mid + 1, mid]]
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=())
        assert not results["C3"].passed


class TestC4:
    def test_end_off_goal_fails(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        assignment = seg.assign_goals(ROUTE, START, [
            {name: float(row[i]) for i, name in enumerate(mf.R_JOINTS)} for row in targets21])
        hover_idx = 1
        last = max(i for i, g in enumerate(assignment.goal_index) if g == hover_idx)
        targets21[last][0] += 0.05  # shoulder_pitch ends well short of HOVER's goal
        # R-tie (coordinator ruling, 2026-09-25 stage-1 rulings, §1):
        # HOVER's own last two samples in this fixture are bit-identical
        # duplicates of its exact goal, and R-tie's boundary tie-break
        # compares an exact-goal sample against the IMMEDIATELY PRECEDING
        # command to decide whether it is a carry. Mutating `last` in
        # place (the line above) breaks that adjacency for the OTHER
        # duplicate sample, which changes assign_goals' OWN
        # re-segmentation of the following (unrelated) boundary if it is
        # recomputed from the mutated array -- exactly the case
        # `run_all`'s own docstring calls out ("a large enough injected
        # violation can itself push the sample out of tolerance ...
        # silently hiding the very violation being tested"). Passing the
        # CLEAN assignment isolates C4's own logic, per that documented
        # contract.
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=(), assignment=assignment)
        assert not results["C4"].passed
        assert results["C4"].first_violation_index == last


class TestC5:
    def test_restream_deviation_fails(self):
        result = _fly(restream_passes=2, settle_s=0.05)
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        assignment = seg.assign_goals(ROUTE, START, [
            {name: float(row[i]) for i, name in enumerate(mf.R_JOINTS)} for row in targets21])
        rest_shut_idx = 2
        idxs = [i for i, g in enumerate(assignment.goal_index) if g == rest_shut_idx]
        last = idxs[-1]
        targets21[last][0] += 0.02
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=(), assignment=assignment)
        assert not results["C5"].passed


class TestC6:
    def test_non_carry_during_hold_fails(self):
        result = _fly(settle_s=0.3)
        targets21 = _targets21(result.command_rows).tolist()
        # Insert a spurious fresh command between HOVER's last setpoint and
        # REST_SHUT's first -- a real command during what should be a hold.
        assignment = seg.assign_goals(ROUTE, START, [
            {name: float(v) for name, v in zip(mf.R_JOINTS, row)} for row in targets21])
        hover_idx, rest_shut_idx = 1, 2
        hover_last = max(i for i, g in enumerate(assignment.goal_index) if g == hover_idx)
        spurious = list(targets21[hover_last])
        spurious[0] += 0.001  # a value that is neither HOVER nor REST_SHUT nor a carry
        targets21.insert(hover_last + 1, spurious)
        new_goal_index = (assignment.goal_index[:hover_last + 1] + [None]
                           + assignment.goal_index[hover_last + 1:])
        spliced_assignment = seg.GoalAssignment(new_goal_index, True)
        t_hi_s = [i * mf.TICK_S for i in range(len(targets21))]
        results = pc.run_all(np.array(targets21), t_hi_s, START, ROUTE, guard=(),
                              assignment=spliced_assignment)
        assert not results["C6"].passed


class TestC7:
    def test_gripper_out_of_range_fails(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        targets21[10][7] = 5.0  # way outside the commanded gripper range
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=())
        assert not results["C7"].passed

    def test_gripper_moves_outside_gripper_changing_waypoint_fails(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        assignment = seg.assign_goals(ROUTE, START, [
            {name: float(row[i]) for i, name in enumerate(mf.R_JOINTS)} for row in targets21])
        hover_idx = 1  # SWING_1 -> HOVER does not change the gripper
        i = next(i for i, g in enumerate(assignment.goal_index) if g == hover_idx)
        targets21[i][7] += 0.05
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=(), assignment=assignment)
        assert not results["C7"].passed


class TestC8:
    def test_other_joint_changes_fails(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        targets21[15][10] = 0.3  # a left-arm joint, should always carry at 0
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=())
        assert not results["C8"].passed
        assert results["C8"].first_violation_index == 15

    def test_first_command_seed_reported_not_a_failure(self):
        result = _fly()
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        keyframe = np.zeros(21)
        keyframe[12] = 0.2  # first command's non-arm joints differ from "keyframe"
        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=(), seed_keyframe21=keyframe)
        assert results["C8"].passed
        assert results["_seed_deviation_rad"] == pytest.approx(0.2)


class TestALikeRoute:
    def test_c1_c2_c5_expected_to_fail(self):
        """An A-like (echo-corrupted) leg is expected to fail C1 (an echo
        breaks start continuity), C2 (an echo is off the planned line) and
        C5 (an echo during what should be a held re-stream). Rather than
        rely on the probabilistic echo_rate injection to happen to land in
        all three spots (it deliberately avoids each goto's arrival tail --
        see make_fixtures.py), inject one clean, deterministic example of
        each directly, exactly as a real echo would: a bit-exact copy of an
        earlier realised position substituted for one commanded setpoint."""
        result = _fly(restream_passes=1, settle_s=0.05)
        targets21 = _targets21(result.command_rows)
        t_hi_s = _t_hi(result)
        assignment = seg.assign_goals(ROUTE, START, [
            {name: float(row[i]) for i, name in enumerate(mf.R_JOINTS)} for row in targets21])

        # C1: the first command of a goto (HOVER) echoes an unrelated
        # earlier position instead of continuing from SWING_1's own goal.
        hover_first = next(i for i, g in enumerate(assignment.goal_index) if g == 1)
        targets21[hover_first][0] += 0.01

        # C2: a mid-goto sample (REST_SHUT) echoes a stale position, off
        # the planned line by more than the C2 tolerance allows.
        rest_shut_idx = 2
        rs_idxs = [i for i, g in enumerate(assignment.goal_index) if g == rest_shut_idx]
        mid = rs_idxs[len(rs_idxs) // 2]
        targets21[mid][3] += 0.05  # elbow_pitch, a moving joint on this leg

        # C5: the re-stream pass for REST_SHUT holds everywhere except one
        # echoed setpoint.
        targets21[rs_idxs[-1]][0] += 0.02

        results = pc.run_all(targets21, t_hi_s, START, ROUTE, guard=(), assignment=assignment)
        assert not results["C1"].passed
        assert not results["C2"].passed
        assert not results["C5"].passed
