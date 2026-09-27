"""B15/H12 (coordinator review, 2026-09-25 Stage-A slice review, §4;
owner Stage B authorization): Q-hold's own hold-window boundary
(``holds.find_hold_windows``) now bridges a run of unassigned (``None``)
commands between two DIFFERENT consecutive assigned goals (not just a
single one) -- and, separately, ANY unassigned, non-carry command
anywhere inside a leg's own mid-span (not only at a goal transition) is
its own violation (``cycle.LegResult.unassigned_non_carry_violation``):
B -> STOP, and for A, the affected segment becomes invalid (W4).

The G-b defect this closes: an indeterminate command SANDWICHED between
two commands of the SAME goal (e.g. a genuine echo mid-trajectory) used
to be silently invisible to `pathcheck.check_c6` (its own docstring says
so explicitly: "left to C0/C2 to judge, never double-counted here") --
and C0/C2 never actually look at an unassigned command either, so it
really was invisible. This module tests that SAME-goal-sandwiched case
directly (the between-different-goals case is already covered by the
`find_hold_windows` fix and exercised via the existing genuine-echo
fixtures in test_goalfix_cmp_cycle_summary.py/test_goalfix_cmp_t8_mutants.py).
"""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


def full_pose(**kw) -> dict:
    base = {j: 0.0 for j in mf.R_JOINTS}
    base.update(kw)
    return base


ROUTE = [
    mf.Waypoint("A", full_pose(r_shoulder_pitch=-0.5), 2.0),
    mf.Waypoint("B", full_pose(r_shoulder_pitch=-1.0), 2.0),
]
START = full_pose()


def _build(tmp_path, *, sandwiched_value=None):
    sim = mf.FlightSim(START, settle_s=0.05)
    sim.fly(ROUTE)
    result = sim.result()
    cmds = [dict(c, target_rad=list(c["target_rad"])) for c in result.command_rows]
    rows = list(result.state_rows)

    # A's own first goto (index 0): find two consecutive A-assigned
    # commands well inside the goto (not the first/last, to avoid the
    # boundary/carry machinery), and sandwich an off-path, non-carry
    # value between them.
    from tools.goalfix_cmp import segments as seg
    targets8 = [{name: float(v) for name, v in zip(mf.R_JOINTS, c["target_rad"][:8])}
                for c in cmds]
    assignment = seg.assign_goals(ROUTE, START, targets8)
    a_positions = [i for i, g in enumerate(assignment.goal_index) if g == 0]
    mid = a_positions[len(a_positions) // 2]

    if sandwiched_value is not None:
        cmds[mid]["target_rad"][0] = sandwiched_value

    mf.write_evidence(tmp_path, rows, cmds)
    evidence = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
    idx = list(range(len(evidence.commands)))
    leg = cyc.LegSpec("setup", ROUTE, guard=(), start_pose8=START, command_indices=idx)
    return evidence, leg, mid, cmds


class TestSameGoalSandwichedUnassignedCommand:
    def test_clean_leg_has_no_unassigned_non_carry_violation(self, tmp_path):
        evidence, leg, _mid, _cmds = _build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("b15_clean", "B", evidence, [leg], skip_gates=True)
        assert not leg_results["setup"].unassigned_non_carry_violation
        assert not any("unassigned non-carry" in r for r in cv.reasons)

    def test_sandwiched_off_path_non_carry_command_stops_b(self, tmp_path):
        """A value that is off A's own path, and differs from its own
        immediately preceding command (so it is not a carry either),
        planted between two otherwise-clean A samples: `assign_goals`
        cannot place it (neither A's nor B's own box admits it), so it
        is unassigned -- and it is sandwiched between two SAME-goal (A)
        commands, exactly the shape the OLD check_c6 explicitly excluded
        ("left to C0/C2 to judge"). B15's own general scan is what
        catches it."""
        # Off-path: well outside A's own [START, A.pose] range, and NOT
        # close enough to B's own range either -- assign_goals leaves it
        # indeterminate.
        evidence, leg, mid, _cmds = _build(tmp_path, sandwiched_value=0.3)
        cv, leg_results = cyc.evaluate_cycle("b15_bad", "B", evidence, [leg], skip_gates=True)
        lr = leg_results["setup"]
        assert lr.assignment.goal_index[mid] is None, "the plant must be off both A's and B's own path"
        assert lr.unassigned_non_carry_violation
        assert f"local index {mid}" in lr.unassigned_non_carry_detail
        assert cv.verdict == cyc.VERDICT_STOP
        assert any("unassigned non-carry" in r for r in cv.reasons)

    def test_a_cycle_segment_invalid_via_w4(self, tmp_path):
        """The same construction, arm A: W4 invalidates the affected
        segment (this fixture has no HOVER/REST_SHUT waypoints, so
        `find_affected_segment` itself already returns None -- the point
        here is that `unassigned_non_carry_violation` is set on the leg
        regardless, which is what a HOVER/REST_SHUT-shaped leg's own W4
        check reads)."""
        evidence, leg, _mid, _cmds = _build(tmp_path, sandwiched_value=0.3)
        cv, leg_results = cyc.evaluate_cycle("b15_bad_a", "A", evidence, [leg], skip_gates=True)
        assert leg_results["setup"].unassigned_non_carry_violation

    def test_mutation_general_scan_removed_would_miss_the_sandwich(self, tmp_path):
        """Mutation guard: with the general scan's own condition
        replaced by "always False" (i.e. never flagging anything), this
        exact sandwiched, off-path, non-carry command is invisible to
        every other check (pathcheck.check_c6's own OLD, narrower rule
        already documents why: sandwiched-by-same-goal is explicitly
        excluded; C0's goal SEQUENCE has nothing to disagree with, since
        a de-duplicated [A, B] sequence is unaffected by one skipped
        index; C2/C3 read only ASSIGNED commands via
        `assignment.goal_index[i]`, so an unassigned index is skipped by
        their own `if g is None: continue`). Reproduced directly:
        confirms no OTHER shipped check independently reports this
        command's own index."""
        evidence, leg, mid, _cmds = _build(tmp_path, sandwiched_value=0.3)
        cv, leg_results = cyc.evaluate_cycle("b15_mut", "B", evidence, [leg], skip_gates=True)
        lr = leg_results["setup"]
        other_violation_indices = set()
        for cid, r in lr.pathcheck.items():
            if hasattr(r, "passed") and not r.passed and r.first_violation_index is not None:
                other_violation_indices.add(r.first_violation_index)
        assert mid not in other_violation_indices, (
            "if this fails, some OTHER check already reports this index and the "
            "mutation would not be silent -- the general scan would not be load-bearing")
