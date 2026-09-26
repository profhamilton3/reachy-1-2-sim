"""B5 (H5; V1-a; owner-approved plan §7.1; coordinator Stage B
authorization): ``echo._subclassify``'s non-near-end path-coincidence
branch must check plan §7.1's own rule -- "at an implied tau that agrees
with the command's other moving joints within 30 ms (C2)" -- not just
"the value round-trips through the minimum-jerk formula for SOME tau",
which holds for every value in [start, goal] and made path coincidence
mask real echoes (V1 found 36 on real A evidence).
"""
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp._minjerk import pose_at  # noqa: E402

START_A, GOAL_A = 0.0, 1.0     # r_shoulder_pitch's own goto
SECONDS = 2.0
TAU_A = 0.5                    # mid-goto, well clear of either end


def _ctx(seconds=SECONDS):
    return echo.GotoContext(
        start8={"r_shoulder_pitch": START_A}, goal8={"r_shoulder_pitch": GOAL_A}, seconds=seconds)


class TestTauAgreement:
    def test_mid_goto_value_with_tau_disagreement_is_genuine_echo(self):
        """V1's own shape: the value IS exactly on r_shoulder_pitch's own
        minimum-jerk curve at tau=0.5 (so the OLD "round trips for some
        tau" test alone would call this path coincidence), but the
        command's OTHER moving joint is nowhere near tau=0.5 -- a real
        race-echo disagreement, at 400ms >> 30ms."""
        value = pose_at(START_A, GOAL_A, TAU_A)
        moving_taus = {"r_shoulder_pitch": TAU_A, "r_shoulder_roll": 0.9}  # 0.4 * 2.0s = 800ms skew
        label, vacuous = echo._subclassify(
            name="r_shoulder_pitch", value=value, epoch=0, command_index=0,
            turn_on_state_index=None, ctx=_ctx(), src_global_index=0, moving_taus=moving_taus)
        assert label == echo.GENUINE_ECHO
        assert vacuous is False

    def test_mid_goto_value_with_tau_agreement_is_path_coincidence(self):
        """The SAME on-path value, but the other moving joint's own tau
        is within 30 ms (at this goto's own 2.0s duration, a tau
        difference of <= 0.015)."""
        value = pose_at(START_A, GOAL_A, TAU_A)
        moving_taus = {"r_shoulder_pitch": TAU_A, "r_shoulder_roll": TAU_A + 0.005}  # 10ms skew
        label, vacuous = echo._subclassify(
            name="r_shoulder_pitch", value=value, epoch=0, command_index=0,
            turn_on_state_index=None, ctx=_ctx(), src_global_index=0, moving_taus=moving_taus)
        assert label == echo.PATH_COINCIDENCE
        assert vacuous is False

    def test_no_other_moving_joint_is_vacuous_path_coincidence(self):
        """No other joint to compare against -- the rule is vacuous
        (plan/assignment's own instruction): still path coincidence
        (on-path alone), but flagged and counted as vacuous, never
        silently indistinguishable from a real tau-agreement pass."""
        value = pose_at(START_A, GOAL_A, TAU_A)
        moving_taus = {"r_shoulder_pitch": TAU_A}  # only itself
        label, vacuous = echo._subclassify(
            name="r_shoulder_pitch", value=value, epoch=0, command_index=0,
            turn_on_state_index=None, ctx=_ctx(), src_global_index=0, moving_taus=moving_taus)
        assert label == echo.PATH_COINCIDENCE
        assert vacuous is True

    def test_counts_report_vacuous_separately(self):
        """echo.count_labels surfaces path_coincidence_vacuous, never
        folding it silently into the plain path_coincidence total."""
        cr = echo.CommandResult(0)
        cr.joints["r_shoulder_pitch"] = echo.JointResult(
            echo.PATH_COINCIDENCE, age_s=0.1, source_state_index=0, tau_agreement_vacuous=True)
        cr.joints["r_shoulder_roll"] = echo.JointResult(
            echo.PATH_COINCIDENCE, age_s=0.1, source_state_index=0, tau_agreement_vacuous=False)
        counts = echo.count_labels([cr])
        assert counts.path_coincidence == 2
        assert counts.path_coincidence_vacuous == 1

    def test_mutation_tau_agreement_check_removed_would_mask_the_echo(self):
        """Mutation guard: reverting to the OLD rule (accept any value
        that round-trips through implied_tau/pose_at for some tau, with
        no cross-joint comparison at all) reclassifies the SAME
        disagreeing-tau echo above as path_coincidence -- reproduced
        directly via a local re-implementation of the OLD branch, then
        compared against the shipped function's own, disagreeing
        output."""
        from tools.goalfix_cmp._minjerk import implied_tau

        def old_subclassify_on_path_branch(value, a, b):
            tau = implied_tau(a, b, value)
            predicted = pose_at(a, b, tau)
            return echo._within_float32_ulps(predicted, value, echo.ULP_FLOAT32_FACTOR)

        value = pose_at(START_A, GOAL_A, TAU_A)
        mutant_on_path = old_subclassify_on_path_branch(value, START_A, GOAL_A)
        assert mutant_on_path is True, "the mutant's own (old) rule accepts this value unconditionally"

        moving_taus = {"r_shoulder_pitch": TAU_A, "r_shoulder_roll": 0.9}
        fixed_label, _vacuous = echo._subclassify(
            name="r_shoulder_pitch", value=value, epoch=0, command_index=0,
            turn_on_state_index=None, ctx=_ctx(), src_global_index=0, moving_taus=moving_taus)
        assert fixed_label == echo.GENUINE_ECHO
        assert fixed_label != echo.PATH_COINCIDENCE
