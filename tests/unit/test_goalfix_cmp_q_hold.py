"""Q-hold (coordinator ruling, 2026-09-25 stage-1 rulings, §2): hold
windows located BY ASSIGNMENT (under R-tie/R-const), with a per-joint
MAXIMUM |target - last-k target| drift formula, replacing the previous
first-vs-last-of-window formula that could never see a drift (MB4)."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import holds  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402

ROUTE = [
    mf.Waypoint("A", {"r_shoulder_pitch": -0.3}, 1.0),
    mf.Waypoint("B", {"r_shoulder_pitch": -0.6, "r_gripper": 0.3}, 1.0),
]
START = {j: 0.0 for j in mf.R_JOINTS}


def _load(tmp_path, settle_s=0.3, inject=None):
    """``inject`` is (tick_after_a_ends, delta_rad) -- a stray
    joint_command spliced into the hold via `_hold_ticks` patching,
    mirroring probe P-hold's own construction."""
    sim = mf.FlightSim(START, settle_s=settle_s)
    if inject is not None:
        tick, delta = inject
        orig = sim._hold_ticks

        def patched(n, held_pose):
            if n > tick:
                t = dict(held_pose)
                t["r_shoulder_pitch"] = held_pose["r_shoulder_pitch"] + delta
                sim._tick_command_then_state(t)
                return orig(n - 1, held_pose)
            return orig(n, held_pose)
        sim._hold_ticks = patched
    sim.fly(ROUTE)
    result = sim.result()
    mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
    return ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")


def _hold_stats_for_a(evd):
    jc_idx = np.nonzero(evd.commands.joint_command_mask())[0]
    targets8 = [{name: float(evd.commands.target_rad[i, k]) for k, name in enumerate(mf.R_JOINTS)}
                for i in jc_idx]
    assignment = seg.assign_goals(ROUTE, START, targets8)
    windows = holds.find_hold_windows(assignment, evd.brackets, list(jc_idx))
    assert len(windows) == 1, windows
    stats = holds.hold_window_stats(windows[0], epoch=0, target8=ROUTE[0].pose,
                                     evidence=evd, all_command_indices=list(jc_idx))
    assert stats is not None
    return stats


class TestSettleHoldDrift:
    def test_clean_hold_drifts_zero(self, tmp_path):
        evd = _load(tmp_path)
        stats = _hold_stats_for_a(evd)
        assert stats.window_command_indices == []
        assert all(v == 0.0 for v in stats.target_drift.values())

    def test_mutation_first_vs_last_formula_misses_a_middle_spike(self, tmp_path):
        """Mutation (revert `target_drift_over_window` to the pre-ruling
        first-vs-last-of-window formula): two stray commands that return
        to the SAME value (a spike, not a ramp) have first==last==0
        drift under the old formula, even though a real, non-carry
        command sat in the middle of the window -- the old formula
        could never see this shape at all. The new (max-vs-last-k)
        formula correctly reports it."""
        evd = _load(tmp_path, settle_s=0.5)
        jc_idx = np.nonzero(evd.commands.joint_command_mask())[0]
        # Build the window's command list by hand and splice a spike:
        # [last_k_target, +delta, last_k_target] -- old formula's
        # first/last are both last_k_target, so diff = 0.
        targets8 = [{name: float(evd.commands.target_rad[i, k]) for k, name in enumerate(mf.R_JOINTS)}
                    for i in jc_idx]
        assignment = seg.assign_goals(ROUTE, START, targets8)
        windows = holds.find_hold_windows(assignment, evd.brackets, list(jc_idx))
        goal_index, t_lo_s, t_hi_s, last_k_command_index = windows[0]
        last_k_target8 = {name: float(evd.commands.target_rad[last_k_command_index, k])
                           for k, name in enumerate(mf.R_JOINTS)}

        # Simulate a window containing [last_k, spike, last_k] under the
        # OLD (first-vs-last) formula:
        spiked = dict(last_k_target8)
        spiked["r_shoulder_pitch"] += np.radians(0.05)
        old_style_window = [last_k_target8, spiked, last_k_target8]

        def old_formula(window_vals):
            if not window_vals:
                return {j: 0.0 for j in mf.R_JOINTS}
            first_t, last_t = window_vals[0], window_vals[-1]
            return {j: first_t.get(j, 0.0) - first_t.get(j, 0.0) if False else
                    (last_t.get(j, 0.0) - first_t.get(j, 0.0)) for j in mf.R_JOINTS}

        old_drift = old_formula(old_style_window)
        assert old_drift["r_shoulder_pitch"] == 0.0  # the old formula's own blind spot

        new_drift = {j: max(abs(v.get(j, 0.0) - last_k_target8.get(j, 0.0)) for v in old_style_window)
                     for j in mf.R_JOINTS}
        assert new_drift["r_shoulder_pitch"] == pytest.approx(np.radians(0.05), abs=1e-9)
