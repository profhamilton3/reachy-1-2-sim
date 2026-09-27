"""R-carry (coordinator ruling, 2026-09-25 Stage-A slice review, §3;
"B16" in the Stage B assignment): a per-joint carry -- a value bit-exact
to the IMMEDIATELY PRECEDING command's own value for that joint -- is
never a setpoint of any goto. It is exempt from R-const's exact-match
requirement (N1) and from C2's tau-agreement (owner rulings, 2026-09-25).

Frozen from the review's own G-a finding (real V3-harness r3, clean B,
uninjected): TUCK's last setpoint leaves ``r_wrist_pitch`` non-exact
(``-0.78539789`` rad; TUCK's own goal, and SWING_1's own nominal constant
value, is ``-0.78539819``). SWING_1's own first server command carries
that TUCK value forward on ``r_wrist_pitch`` (SWING_1 does not move that
joint) while ``r_shoulder_roll`` is SWING_1's own genuine first tick.

Under the OLD (bootstrapped) R-const, that leading carry became SWING_1's
own constant-joint REFERENCE, so its true, later exact-S samples
(``-0.78539819``, 130 of them) were rejected as ``None`` -- the G-a defect.
Under R-carry, the leading carry is exempt, and S itself (never a
bootstrapped value) is the one true reference.
"""
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


TUCK_LAST_WRIST_PITCH = -0.78539789   # non-exact: real asyncio-jitter residual
S_WRIST_PITCH = -0.78539819           # TUCK's real goal == SWING_1's own nominal (constant) value

ROUTE = [
    mf.Waypoint("TUCK", full_pose(r_wrist_pitch=S_WRIST_PITCH), 2.0),
    mf.Waypoint("SWING_1", full_pose(r_wrist_pitch=S_WRIST_PITCH, r_shoulder_roll=0.5), 3.0),
]
START = full_pose(r_wrist_pitch=0.0)


def _targets(n_const=130):
    rows = [full_pose(r_wrist_pitch=TUCK_LAST_WRIST_PITCH)]  # TUCK's own last (index 0)
    # SWING_1's own first server command: wrist_pitch CARRIED from TUCK
    # (bit-exact to index 0), shoulder_roll SWING_1's own genuine first tick.
    rows.append(full_pose(r_wrist_pitch=TUCK_LAST_WRIST_PITCH, r_shoulder_roll=0.01))
    for k in range(n_const):
        rows.append(full_pose(r_wrist_pitch=S_WRIST_PITCH, r_shoulder_roll=0.01 + 0.003 * (k + 1)))
    return rows


class TestRCarryAssignment:
    def test_leading_carry_then_true_constant_all_assigned_to_swing1(self):
        targets8 = _targets()
        assignment = seg.assign_goals(ROUTE, START, targets8)
        assert assignment.ok, assignment
        assert assignment.goal_index[0] == 0                    # TUCK
        assert all(g == 1 for g in assignment.goal_index[1:]), assignment.goal_index  # all of SWING_1, incl. the leading carry

    def test_carry_length_is_reportable(self):
        """R-carry point 6: "each joint's leading-carry length, as a
        command count ... report-only". The LEADING carry run for
        r_wrist_pitch in SWING_1's own span is exactly 1 command (index
        1 carries TUCK's own last write forward; index 2 is the first
        TRUE, non-carry constant value S -- after that point, S simply
        repeats itself, which is carry-shaped too, but is no longer part
        of the LEADING run this counts)."""
        targets8 = _targets()
        carries = seg.carry_mask(targets8, START)
        swing1_wrist_pitch_carries = [carries[i]["r_wrist_pitch"] for i in range(1, len(targets8))]
        leading_carry_len = 0
        for c in swing1_wrist_pitch_carries:
            if not c:
                break
            leading_carry_len += 1
        assert leading_carry_len == 1

    def test_mutation_revert_r_const_reference_to_first_command_would_reject(self):
        """Mutation (revert R-const's reference to "the goto's own first
        accepted setpoint", i.e. the pre-ruling bootstrap): the leading
        carry (-0.78539789) becomes the reference, so the later, TRUE
        exact-S samples (-0.78539819) are rejected. Reproduced directly
        against the shipped rule's own inputs by re-implementing the OLD
        bootstrap rule inline (verified separately, in the handoff, by
        reverting `_on_segment_strict` itself in a scratch export)."""
        targets8 = _targets()

        def on_segment_old_bootstrap(seg_start, goal8, value8, tol_deg, const_ref):
            tol = np.radians(tol_deg)
            for j in mf.R_JOINTS:
                a, b, v = seg_start.get(j, 0.0), goal8.get(j, 0.0), value8.get(j, 0.0)
                if a == b:
                    if j in const_ref:
                        if v != const_ref[j]:
                            return False
                    elif not (a - tol <= v <= a + tol):
                        return False
                    continue
                lo, hi = (a, b) if a <= b else (b, a)
                if not (lo - tol <= v <= hi + tol):
                    return False
            return True

        seg_start = ROUTE[0].pose  # TUCK's own pose == SWING_1's own reported start
        goal8 = ROUTE[1].pose
        const_ref: dict = {}
        # index1 (the leading carry) bootstraps the (wrong) reference:
        assert on_segment_old_bootstrap(seg_start, goal8, targets8[1], seg.ON_SEGMENT_TOL_DEG, const_ref)
        for j in mf.R_JOINTS:
            if seg_start.get(j, 0.0) == goal8.get(j, 0.0):
                const_ref.setdefault(j, targets8[1].get(j, 0.0))
        # index2 (a TRUE exact-S sample) is now rejected under the mutant:
        mutant_rejects = not on_segment_old_bootstrap(
            seg_start, goal8, targets8[2], seg.ON_SEGMENT_TOL_DEG, const_ref)
        assert mutant_rejects, "the reported bug: the old bootstrap rejects the TRUE constant value"
        # ... and the shipped code (R-carry) disagrees -- it accepts it:
        fixed = seg.assign_goals(ROUTE, START, targets8).goal_index
        assert fixed[2] == 1


class TestN1ConstantJointCarryExemption:
    """N1 (owner rulings, 2026-09-25): after a joint's first non-carry
    value in a goto, every later value must equal S exactly -- a leading
    carry (bit-exact to the immediately preceding command) is exempt.
    Tested directly against ``pathcheck.check_c2_c3``'s own N1 fold (its
    "a constant joint moved" C2 reason), reusing this file's own
    TUCK->SWING_1 fixture with a HAND-BUILT assignment (isolating
    check_c2_c3 from assign_goals's own admission). ``seg_start`` for
    SWING_1 (``check_c2_c3``'s own convention: ``_goal8(route[g-1])``) is
    TUCK's NOMINAL pose (``S_WRIST_PITCH`` exactly) -- distinct from the
    real recorded TUCK command's own slightly-off value
    (``TUCK_LAST_WRIST_PITCH``), exactly as the real defect requires."""

    def test_leading_carry_then_true_constant_passes(self):
        targets8 = _targets(n_const=3)
        assignment = seg.GoalAssignment(goal_index=[0] + [1] * (len(targets8) - 1), ok=True)
        t_hi_s = [0.02 * (i + 1) for i in range(len(targets8))]
        c2, _c3 = pc.check_c2_c3(targets8, t_hi_s, START, ROUTE, assignment)
        assert c2.passed, c2

    def test_a_later_value_that_is_neither_carry_nor_s_fails_n1(self):
        """The leading carry (index 1) stays exempt; a LATER sample
        (index 2) that is neither a carry of its own preceding command
        nor equal to S is what N1 catches."""
        targets8 = _targets(n_const=1)
        targets8[2] = dict(targets8[2], r_wrist_pitch=-0.5)  # neither a carry of index 1 nor == S
        assignment = seg.GoalAssignment(goal_index=[0, 1, 1], ok=True)
        c2, _c3 = pc.check_c2_c3(targets8, [0.0, 0.02, 0.04], START, ROUTE, assignment)
        assert not c2.passed
        assert "constant joint moved" in c2.detail
        assert c2.first_violation_index == 2  # not index 1, the exempt leading carry

    def test_mutation_carry_exemption_removed_would_reject_the_leading_carry(self):
        """Mutation guard: dropping the `is_carry` exemption in
        check_c2_c3's N1 fold would flag the LEADING CARRY itself
        (index 1 in the first test above) as "a constant joint moved",
        since TUCK_LAST_WRIST_PITCH != S_WRIST_PITCH -- pinned here as
        the precondition that makes the exemption load-bearing."""
        assert TUCK_LAST_WRIST_PITCH != S_WRIST_PITCH
