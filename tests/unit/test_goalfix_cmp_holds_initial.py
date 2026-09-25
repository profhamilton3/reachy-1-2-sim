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
        goal_index, t_lo, t_hi = windows[0]
        assert goal_index == 0  # SWING_1's hold, before HOVER begins
        assert t_hi - t_lo == pytest.approx(0.3, abs=0.02)

        stats = holds.hold_window_stats(windows[0], epoch=0, target8=SWING_1_POSE, states=evd.states)
        assert stats is not None
        # No commands during the hold -- target never moves.
        assert all(v == 0.0 for v in stats.target_drift.values())
        # Realised position is held (my fixture's constant lag offset
        # model), so drift over the window is ~0 and the static offset is
        # the known, constant lag.
        assert abs(stats.realised_drift["r_shoulder_pitch"]) < 1e-9
        assert stats.static_offset_first["r_shoulder_pitch"] == pytest.approx(-0.0021, abs=1e-6)


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
