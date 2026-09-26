"""Unit tests for tools/goalfix_cmp/holds.py and initial.py."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp import holds  # noqa: E402
from tools.goalfix_cmp import initial  # noqa: E402


def full_pose(**kw) -> dict:
    base = {j: 0.0 for j in mf.R_JOINTS}
    base.update(kw)
    return base


START = full_pose()
SWING_1_POSE = full_pose(r_shoulder_pitch=-0.3)
HOVER_POSE = full_pose(r_shoulder_pitch=-0.70, r_shoulder_roll=-0.17,
                        r_elbow_pitch=-1.05, r_wrist_pitch=-0.26)
ROUTE = [
    mf.Waypoint("SWING_1", SWING_1_POSE, 1.0),
    mf.Waypoint("HOVER", HOVER_POSE, 1.2),
]


class TestHolds:
    def test_hold_window_found_between_gotos(self, tmp_path):
        sim = mf.FlightSim(START, settle_s=0.3)
        sim.fly(ROUTE)
        result = sim.result()
        mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        jc_idx = np.nonzero(evd.commands.joint_command_mask())[0]
        targets8 = [{name: float(evd.commands.target_rad[i, k]) for k, name in enumerate(mf.R_JOINTS)}
                    for i in jc_idx]
        assignment = seg.assign_goals(ROUTE, START, targets8)
        windows = holds.find_hold_windows(assignment, evd.brackets, list(jc_idx))
        assert len(windows) == 1
        goal_index, t_lo, t_hi, _last_k_command_index = windows[0]
        assert goal_index == 0  # SWING_1's hold, before HOVER begins
        assert t_hi - t_lo == pytest.approx(0.3, abs=0.02)

        stats = holds.hold_window_stats(
            windows[0], epoch=0, target8=SWING_1_POSE, evidence=evd,
            all_command_indices=list(jc_idx))
        assert stats is not None
        # No commands during the hold -- target never moves.
        assert all(v == 0.0 for v in stats.target_drift.values())
        assert stats.window_command_indices == []
        # Realised position is held (my fixture's constant lag offset
        # model), so drift over the window is ~0 and the static offset is
        # the known, constant lag.
        assert abs(stats.realised_drift["r_shoulder_pitch"]) < 1e-9
        assert stats.static_offset_first["r_shoulder_pitch"] == pytest.approx(-0.0021, abs=1e-6)


class _FakeBracket:
    def __init__(self, t_hi, t_lo):
        self.t_hi = t_hi
        self.t_lo = t_lo


class TestFindHoldWindowsBridgesNoneRuns:
    """B15/H12 (coordinator review, 2026-09-25 Stage-A slice review, §4;
    owner Stage B authorization): a run of unassigned (``None``)
    commands between two DIFFERENT consecutive assigned goals must still
    produce ONE hold window spanning the whole gap -- not silently
    skipped, as the OLD "adjacent commands only" rule did (G-b: a
    130-command run went entirely unchecked)."""

    def test_a_run_of_nones_between_different_goals_still_forms_one_window(self):
        assignment = seg.GoalAssignment(goal_index=[0, 0, None, None, 1], ok=True)
        command_indices = [10, 11, 12, 13, 14]
        brackets = {ci: _FakeBracket(t_hi=1.0 + 0.1 * k, t_lo=0.9 + 0.1 * k)
                    for k, ci in enumerate(command_indices)}

        windows = holds.find_hold_windows(assignment, brackets, command_indices)
        assert len(windows) == 1
        goal_index, t_lo, t_hi, last_k_command_index = windows[0]
        assert goal_index == 0
        assert last_k_command_index == 11  # goal 0's own LAST assigned command
        assert t_lo == pytest.approx(1.1)   # command 11's own t_hi
        assert t_hi == pytest.approx(1.3)   # command 14's own t_lo (goal 1's first)

    def test_mutation_adjacent_only_rule_would_miss_the_run(self):
        """Mutation guard: the OLD rule (adjacent commands i/i+1 only)
        never looks at the pair (11, 14) -- every adjacent pair in the
        run has at least one ``None`` neighbour, so no window is EVER
        formed. Reproduced directly against a local re-implementation of
        the OLD rule, then compared against the shipped (fixed)
        function's own, disagreeing output."""
        assignment = seg.GoalAssignment(goal_index=[0, 0, None, None, 1], ok=True)
        command_indices = [10, 11, 12, 13, 14]
        brackets = {ci: _FakeBracket(t_hi=1.0 + 0.1 * k, t_lo=0.9 + 0.1 * k)
                    for k, ci in enumerate(command_indices)}

        def old_find_hold_windows(assignment, brackets, command_indices):
            windows = []
            n = len(assignment.goal_index)
            for i in range(n - 1):
                g, nxt = assignment.goal_index[i], assignment.goal_index[i + 1]
                if g is None or nxt is None or nxt == g:
                    continue
                windows.append((g, i, i + 1))
            return windows

        mutant_windows = old_find_hold_windows(assignment, brackets, command_indices)
        assert mutant_windows == [], "the reported bug: no window forms for this run under the old rule"
        fixed_windows = holds.find_hold_windows(assignment, brackets, command_indices)
        assert len(fixed_windows) == 1


class TestInitial:
    def test_identical_states_zero_deviation(self):
        state_a = list(range(21))
        state_b = list(range(21))
        dev = initial.compare_initial_states(state_a, state_b)
        assert all(v == 0.0 for v in dev.values())

    def test_small_deviation_reported(self):
        state_a = [0.0] * 21
        state_b = [0.0] * 21
        state_b[0] = 2e-6
        dev = initial.compare_initial_states(state_a, state_b)
        assert dev["r_shoulder_pitch"] == pytest.approx(2e-6)

    def test_compliance_match(self):
        expected = [True] * 21
        ok, detail = initial.check_compliance(expected, expected)
        assert ok

    def test_compliance_mismatch_is_a_gate_failure(self):
        expected = [True] * 21
        actual = list(expected)
        actual[5] = False
        ok, detail = initial.check_compliance(actual, expected)
        assert not ok
        assert "r_wrist_pitch" in detail

    def test_start_variant_gate_missing_evidence(self, tmp_path):
        ok, reason, doc = initial.start_variant_gate(tmp_path, "B4_c1")
        assert not ok
        assert "missing" in reason
        assert doc is None
