"""Unit tests for tools/goalfix_cmp/segments.py."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402

def full_pose(**kw) -> dict:
    """A complete 8-joint pose, matching rig_routes.pose()'s own contract:
    every named waypoint is a full dict, never a sparse override that
    relies on "carrying forward" a joint from whatever came before."""
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


def _targets8(command_rows):
    out = []
    for row in command_rows:
        t = row["target_rad"]
        out.append({name: t[i] for i, name in enumerate(mf.R_JOINTS)})
    return out


class TestCleanRoute:
    def test_c0_sequence_matches_route(self):
        sim = mf.FlightSim(START, skew_ms={"r_wrist_pitch": 12.0})
        sim.fly(ROUTE)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        assignment = seg.assign_goals(ROUTE, START, targets8)
        assert assignment.ok
        assert seg.c0_goal_sequence(assignment, ROUTE) == [wp.name for wp in ROUTE]
        assert all(g is not None for g in assignment.goal_index)

    def test_affected_segment_found(self):
        sim = mf.FlightSim(START)
        sim.fly(ROUTE)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        assignment = seg.assign_goals(ROUTE, START, targets8)
        affected = seg.find_affected_segment(assignment, ROUTE)
        assert affected is not None
        assert not affected.indeterminate
        assert assignment.goal_index[affected.start_command_index] == affected.hover_goal_index
        assert assignment.goal_index[affected.end_command_index] == affected.rest_shut_goal_index
        assert affected.end_command_index > affected.start_command_index


class TestRestreamPasses:
    @pytest.mark.parametrize("passes", [0, 1, 3])
    def test_restream_passes_stay_on_same_goal(self, passes):
        sim = mf.FlightSim(START, restream_passes=passes, settle_s=0.1)
        sim.fly(ROUTE)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        assignment = seg.assign_goals(ROUTE, START, targets8)
        assert assignment.ok
        assert seg.c0_goal_sequence(assignment, ROUTE) == [wp.name for wp in ROUTE]


class TestEchoCorrupted:
    def test_a_like_still_finds_boundaries(self):
        sim = mf.FlightSim(START, seed=5)
        sim.fly(ROUTE, echo_rate=0.44)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        assignment = seg.assign_goals(ROUTE, START, targets8)
        # Some setpoints may be indeterminate (echoed to an off-segment
        # value), but the goal sequence's *determinate* entries must still
        # trace the real route -- no C0 failure from corruption alone.
        assert assignment.ok
        determinate_names = seg.c0_goal_sequence(assignment, ROUTE)
        assert determinate_names == [wp.name for wp in ROUTE]
        affected = seg.find_affected_segment(assignment, ROUTE)
        assert affected is not None
        assert not affected.indeterminate


class TestC0Failures:
    def test_skipped_waypoint(self):
        sim = mf.FlightSim(START)
        sim.fly(ROUTE)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        # Drop every command assigned to REST_SHUT, plus a small buffer on
        # both sides, so the visible sequence jumps from HOVER's real
        # interior (not its arrival, which coincides exactly with
        # REST_SHUT's own start) straight to REST's real interior (not its
        # anchor, which coincides exactly with REST_SHUT's own goal) --
        # a genuine skip, with no boundary-value coincidence to mask it.
        clean_assignment = seg.assign_goals(ROUTE, START, targets8)
        rest_shut_idx = next(i for i, wp in enumerate(ROUTE) if wp.name == "REST_SHUT")
        rest_shut_positions = [i for i, g in enumerate(clean_assignment.goal_index)
                                if g == rest_shut_idx]
        lo, hi = min(rest_shut_positions) - 2, max(rest_shut_positions) + 2
        filtered = [t for i, t in enumerate(targets8) if not (lo <= i <= hi)]
        assignment = seg.assign_goals(ROUTE, START, filtered)
        assert not assignment.ok
        assert assignment.violation_kind == seg.SKIPPED_WAYPOINT

    def test_extra_waypoint_regression(self):
        sim = mf.FlightSim(START)
        sim.fly(ROUTE)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        # Splice a handful of SWING_1-goal setpoints back in after HOVER
        # has already been reached -- an unexpected regression to an
        # earlier goal.
        clean_assignment = seg.assign_goals(ROUTE, START, targets8)
        swing_idx = 0
        hover_idx = 1
        # A middle slice of the SWING_1 goto -- clearly inside SWING_1's own
        # range and clearly outside HOVER's, unlike the tail (which sits
        # right at the shared boundary between the two segments).
        swing_samples = [t for t, g in zip(targets8, clean_assignment.goal_index)
                          if g == swing_idx][10:13]
        hover_first = next(i for i, g in enumerate(clean_assignment.goal_index) if g == hover_idx)
        spliced = targets8[:hover_first + 2] + swing_samples + targets8[hover_first + 2:]
        assignment = seg.assign_goals(ROUTE, START, spliced)
        assert not assignment.ok
        assert assignment.violation_kind == seg.EXTRA_WAYPOINT


class TestROver:
    """R-over (owner rulings, 2026-09-25 owner-comparison-definitions):
    a setpoint that lies more than 1 float32 ULP beyond ``wp_k``'s own
    pose, in the direction of goto k+1, never belongs to goto k -- the
    "F-e" shape (proposal §1 F-e; assignment B11: "the F-e sequence
    (commands 740-747 shape)"): goto k+1's own first sample, arriving
    after the settle-hold gap, has moved a small amount past ``wp_k`` in
    the CONTINUING direction, and the old, far more permissive
    ``ON_SEGMENT_TOL_DEG`` box admitted it to goto k. Built here as a
    synthetic fixture matching that shape (never reading recorded
    evidence): a clean two-waypoint flight, then HOVER's own last write
    is perturbed to leak past its own goal, by the same 6.5e-5deg scale
    the proposal's F-e finding measured."""

    def test_next_gotos_first_sample_is_not_absorbed_into_the_previous_goto(self):
        sim = mf.FlightSim(START)
        sim.fly(ROUTE)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        assignment = seg.assign_goals(ROUTE, START, targets8)

        # HOVER's own last sample, by the CURRENT (clean) assignment.
        hover_last = max(i for i, g in enumerate(assignment.goal_index) if g == 1)

        # The F-e shape: overshoot HOVER's own goal by a small amount in
        # the direction REST_SHUT continues (a few 1e-4deg-scale, well
        # inside the OLD ON_SEGMENT_TOL_DEG box) at HOVER's OWN last
        # command -- i.e. HOVER's last write "leaks" toward REST_SHUT.
        hover_goal = HOVER_POSE["r_shoulder_pitch"]
        restshut_goal = REST_SHUT_POSE["r_shoulder_pitch"]
        direction = 1.0 if restshut_goal > hover_goal else -1.0
        leak = hover_goal + direction * np.radians(6.5e-5)  # proposal F-a/F-e's own scale
        targets8[hover_last] = dict(targets8[hover_last], r_shoulder_pitch=leak)

        reassigned = seg.assign_goals(ROUTE, START, targets8)
        # R-over: HOVER (goto 1) never admits this leaked sample -- it is
        # excluded from goto 1 (either reassigned forward or
        # indeterminate), never silently kept as HOVER's own last write.
        assert reassigned.goal_index[hover_last] != 1, (
            "the leaked, over-goal sample must not stay assigned to HOVER (R-over)")

    def test_mutation_revert_r_over_would_admit_the_overshoot(self):
        """Mutation guard: the OLD (pre-R-over) admission box -- the full
        ``ON_SEGMENT_TOL_DEG`` on BOTH sides, no 1-ULP goal-side
        tightening -- admits the exact same leaked sample to HOVER.
        Reproduced directly against a local re-implementation of the OLD
        rule (verified separately, in the handoff, against a scratch
        export with R-over's own tightening reverted)."""
        def on_segment_old(start8, goal8, value8, tol_deg=seg.ON_SEGMENT_TOL_DEG):
            tol = np.radians(tol_deg)
            for j in mf.R_JOINTS:
                a, b, v = start8.get(j, 0.0), goal8.get(j, 0.0), value8.get(j, 0.0)
                lo, hi = (a, b) if a <= b else (b, a)
                if not (lo - tol <= v <= hi + tol):
                    return False
            return True

        hover_goal = HOVER_POSE["r_shoulder_pitch"]
        restshut_goal = REST_SHUT_POSE["r_shoulder_pitch"]
        direction = 1.0 if restshut_goal > hover_goal else -1.0
        leak = hover_goal + direction * np.radians(6.5e-5)
        leaked_pose = full_pose(r_shoulder_pitch=leak, r_shoulder_roll=HOVER_POSE["r_shoulder_roll"],
                                 r_elbow_pitch=HOVER_POSE["r_elbow_pitch"],
                                 r_wrist_pitch=HOVER_POSE["r_wrist_pitch"])
        mutant_admits = on_segment_old(SWING_1_POSE, HOVER_POSE, leaked_pose)
        assert mutant_admits, "the reported bug: the old box admits the overshoot to HOVER"


class TestTruncatedLeg:
    def test_truncated_mid_segment_is_indeterminate(self):
        sim = mf.FlightSim(START)
        sim.fly(ROUTE)
        result = sim.result()
        targets8 = _targets8(result.command_rows)
        assignment_full = seg.assign_goals(ROUTE, START, targets8)
        hover_idx = 1
        cutoff = next(i for i, g in enumerate(assignment_full.goal_index) if g == hover_idx) + 2
        truncated = targets8[:cutoff]
        assignment = seg.assign_goals(ROUTE, START, truncated)
        affected = seg.find_affected_segment(assignment, ROUTE)
        assert affected is not None
        assert affected.indeterminate
