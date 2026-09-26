"""R-tie and R-const (coordinator ruling, 2026-09-25 stage-1 rulings, §1):
segment membership at goto boundaries, frozen from the coordinator's own
V3 diagnostic run so these tests do not need the venv.

Both fix `segments.assign_goals`'s boundary tie-break, which previously
mis-assigned a goto's own final approach sample (or a genuinely-moved
next-waypoint sample admitted by the tol box) to the WRONG side of a
GRIP_SHUT -> BACK -style boundary -- producing false C1/C2/C4/C7 failures
on clean, uninjected B data (Stage 1's F6 finding, since corrected)."""
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402


def full_pose(**kw):
    base = {j: 0.0 for j in mf.R_JOINTS}
    base.update(kw)
    return base


class TestRTie:
    """Frozen from the ruling's §1 "Observed, cycle 2" sequence
    (GRIP_SHUT -> BACK, commands 1656-1659, gripper/shoulder-pitch in
    rad): a settle hold (no commands, F3) separates GRIP_SHUT's last
    write from BACK's first visible setpoint. GRIP_SHUT's last write is
    exact; BACK's own first two samples are the ambiguous carry pair."""

    _GRIP_SHUT_GRIP = 0.349065840  # float32(deg2rad(20)) -- GRIP_SHUT's own goal
    START = full_pose(r_gripper=0.0)  # HOME-ish OPEN-direction value, below GRIP_SHUT's goal (approach increases)
    ROUTE = [
        mf.Waypoint("GRIP_SHUT", full_pose(r_gripper=_GRIP_SHUT_GRIP), 3.5),
        mf.Waypoint("BACK", full_pose(r_gripper=_GRIP_SHUT_GRIP, r_shoulder_pitch=0.698), 3.0),
    ]

    def _targets(self):
        return [
            full_pose(r_gripper=0.349045068),                          # approach (differs from goal)
            full_pose(r_gripper=0.349058032),                          # approach (differs from goal)
            full_pose(r_gripper=self._GRIP_SHUT_GRIP),                 # exact GRIP_SHUT write, DIFFERS from prev
            full_pose(r_gripper=self._GRIP_SHUT_GRIP, r_shoulder_pitch=4.57e-07),  # BACK's own first sample
        ]

    def test_exact_write_that_differs_from_prev_stays_with_grip_shut(self):
        """R-tie: index 2 bit-equals GRIP_SHUT's own goal and DIFFERS
        from index 1 (the preceding command) -- it continues GRIP_SHUT's
        approach, and must NOT be reassigned to BACK even though index 3
        looks like real BACK progress (the exact bug: the OLD tie-break
        reassigned it purely from index 3's shape, ignoring index 1)."""
        targets8 = self._targets()
        assignment = seg.assign_goals(self.ROUTE, self.START, targets8)
        assert assignment.goal_index == [0, 0, 0, 1], assignment.goal_index

    def test_mutation_revert_tie_break_would_misassign(self):
        """Mutation (revert R-tie to the old "check index i+1 alone"
        rule): reproduces the ruling's exact false assignment. Verified
        directly by reverting `assign_goals`'s tie-break block in a
        scratch export of this repo and re-running this file -- the
        FIRST test above fails (goal_index[2] becomes 1, not 0)."""
        # A minimal re-implementation of the OLD (da3a81c) tie-break,
        # applied only to this frozen 4-sample sequence, to pin exactly
        # what the mutation would produce without needing a live revert.
        targets8 = self._targets()
        route = self.ROUTE

        def old_assign(route, start_pose8, targets8):
            cur, seg_start = 0, dict(start_pose8)
            result = [None] * len(targets8)
            for i, tgt in enumerate(targets8):
                goal8 = route[cur].pose
                on_seg = all(
                    (min(seg_start.get(j, 0.0), goal8.get(j, 0.0)) - 1e-6
                     <= tgt.get(j, 0.0) <=
                     max(seg_start.get(j, 0.0), goal8.get(j, 0.0)) + 1e-6)
                    for j in mf.R_JOINTS)
                if cur < len(route) and on_seg:
                    exact = all(tgt.get(j, 0.0) == goal8.get(j, 0.0) for j in mf.R_JOINTS)
                    if cur + 1 < len(route) and i + 1 < len(targets8) and exact:
                        nxt_start, nxt_goal = goal8, route[cur + 1].pose
                        nxt = targets8[i + 1]
                        nxt_on_seg = all(
                            (min(nxt_start.get(j, 0.0), nxt_goal.get(j, 0.0)) - 1e-6
                             <= nxt.get(j, 0.0) <=
                             max(nxt_start.get(j, 0.0), nxt_goal.get(j, 0.0)) + 1e-6)
                            for j in mf.R_JOINTS)
                        nxt_exact_start = all(nxt.get(j, 0.0) == nxt_start.get(j, 0.0) for j in mf.R_JOINTS)
                        nxt_exact_goal = all(nxt.get(j, 0.0) == nxt_goal.get(j, 0.0) for j in mf.R_JOINTS)
                        if nxt_on_seg and not nxt_exact_start and not nxt_exact_goal:
                            cur += 1
                            seg_start = goal8
                            result[i] = cur
                            continue
                    result[i] = cur
                    continue
                result[i] = cur  # not exercised by this frozen sequence
            return result

        mutated = old_assign(route, self.START, targets8)
        assert mutated == [0, 0, 1, 1], mutated  # the reported bug, reproduced
        # ... and the shipped code (with the fix) disagrees with it:
        fixed = seg.assign_goals(route, self.START, targets8).goal_index
        assert fixed != mutated


class TestRConst:
    """Frozen from the ruling's §1 "Observed, cycle 1" note: a genuinely-
    moved next-waypoint sample (BACK's real first setpoint, shoulder_pitch
    a few 1e-5deg off zero) is admitted into the CONSTANT-joint tol box of
    the PREVIOUS waypoint (GRIP_SHUT, whose shoulder_pitch is nominally
    constant at 0) purely because the drift is smaller than
    ON_SEGMENT_TOL_DEG."""

    ROUTE = [
        mf.Waypoint("GRIP_SHUT", full_pose(r_gripper=0.35), 3.5),
        mf.Waypoint("BACK", full_pose(r_gripper=0.35, r_shoulder_pitch=0.698), 3.0),
    ]
    START = full_pose(r_gripper=0.0)

    def test_small_shoulder_pitch_drift_is_rejected_from_grip_shut(self):
        import numpy as np
        # Establish GRIP_SHUT's own const_ref for shoulder_pitch at
        # exactly 0.0 with a few clean, still-approaching (gripper not
        # yet at its own goal) samples first.
        targets8 = [full_pose(r_gripper=0.30) for _ in range(3)]
        # BACK's real first sample: gripper has JUST arrived at GRIP_SHUT's
        # own goal (0.35 -- also BACK's own constant nominal value, so it
        # is admissible there too), and shoulder_pitch is a few 1e-5 deg
        # off zero -- inside the tol box (1e-4 deg) around GRIP_SHUT's own
        # constant 0.0, but NOT equal to the established reference. The
        # WHOLE 8-joint tuple does not exactly equal GRIP_SHUT's own pose
        # (shoulder_pitch differs), so R-tie's own tie-break never
        # engages here -- this sample tests R-const in isolation.
        targets8.append(full_pose(r_gripper=0.35, r_shoulder_pitch=np.radians(4e-5)))
        assignment = seg.assign_goals(self.ROUTE, self.START, targets8)
        assert assignment.goal_index == [0, 0, 0, 1], assignment.goal_index

    def test_mutation_revert_const_check_would_admit_it_to_grip_shut(self):
        """Mutation (revert R-const to the plain tol-box test for a
        constant joint too, i.e. the pre-ruling `_on_segment`): the same
        4th sample above is admitted to GRIP_SHUT (its shoulder_pitch is
        within the tol box of GRIP_SHUT's own nominal 0.0), never
        advancing to BACK at all."""
        import numpy as np

        def on_segment_old(start8, goal8, value8, tol_deg=1e-4):
            tol = np.radians(tol_deg)
            for j in mf.R_JOINTS:
                a, b, v = start8.get(j, 0.0), goal8.get(j, 0.0), value8.get(j, 0.0)
                lo, hi = (a, b) if a <= b else (b, a)
                if not (lo - tol <= v <= hi + tol):
                    return False
            return True

        sample = full_pose(r_gripper=0.35, r_shoulder_pitch=np.radians(4e-5))
        mutated_accepts_grip_shut = on_segment_old(self.START, self.ROUTE[0].pose, sample)
        assert mutated_accepts_grip_shut is True  # the reported bug, reproduced
        # ... and the shipped code (with the fix) disagrees:
        targets8 = [full_pose(r_gripper=0.30) for _ in range(3)] + [sample]
        fixed = seg.assign_goals(self.ROUTE, self.START, targets8).goal_index
        assert fixed[-1] == 1  # correctly BACK, not GRIP_SHUT
