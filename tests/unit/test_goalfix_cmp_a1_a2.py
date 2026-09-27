"""A1 (edge-window boundary-by-property) and A2 (C3 raw-value
monotonicity, no near-end exemption) -- coordinator ruling, 2026-09-25
stage-2a rulings addendum."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import holds  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp._io import RC_STOP, read_result  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    # A1's turn_on present-match property (F4) requires target ==
    # present exactly at turn_on -- make_fixtures' deliberate constant
    # lag would otherwise make even a genuinely-clean synthetic leg's
    # OWN first command fail the match. See
    # test_goalfix_cmp_t4_end_to_end.py's own identical note.
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


ROUTE = [
    mf.Waypoint("A", {"r_shoulder_pitch": -0.3}, 1.0),
    mf.Waypoint("B", {"r_shoulder_pitch": -0.6, "r_gripper": 0.3}, 1.0),
]
START = {j: 0.0 for j in mf.R_JOINTS}


# ---------------------------------------------------------------------------
# A2: C3 raw-value monotonicity, no near-end exemption
# ---------------------------------------------------------------------------

class TestA2C3NoNearEndExemption:
    def _clean_targets(self):
        sim = mf.FlightSim(START)
        sim.fly(ROUTE)
        result = sim.result()
        return [{name: float(row[i]) for i, name in enumerate(mf.R_JOINTS)}
                for row in [r["target_rad"] for r in result.command_rows]]

    def test_frozen_p_hold_residual_stops_via_c3(self):
        """The P-hold residual (Q-hold's handoff): a stray landing on the
        exact-carry anchor tie right after goto A's own end, moving AWAY
        from A's goal by a tiny (<0.05deg) amount, followed by the real
        anchor at A's own exact goal -- a raw-value decrease (moving back
        toward the START, away from the goal) within 0.05deg of the end,
        which C2's near-end exemption used to hide from C3 too."""
        targets8 = self._clean_targets()
        assignment = seg.assign_goals(ROUTE, START, targets8)
        a_positions = [i for i, g in enumerate(assignment.goal_index) if g == 0]
        last_a = a_positions[-1]
        # last_a's own value is A's exact goal (-0.3 rad); insert two
        # samples that move AWAY from it (back toward START, i.e. less
        # negative) by <0.05deg, then restore the exact anchor -- exactly
        # the frozen P-hold shape.
        a_goal = ROUTE[0].pose["r_shoulder_pitch"]
        stray1 = dict(targets8[last_a]); stray1["r_shoulder_pitch"] = a_goal + np.radians(0.02)
        stray2 = dict(targets8[last_a]); stray2["r_shoulder_pitch"] = a_goal + np.radians(0.04)
        spliced = targets8[:last_a + 1] + [stray1, stray2] + targets8[last_a + 1:]
        t_hi_s = list(range(len(spliced)))
        results = pc.run_all(np.array([[row.get(n, 0.0) for n in mf.R_JOINTS] + [0.0] * 13
                                        for row in spliced]),
                              t_hi_s, START, ROUTE, guard=())
        assert not results["C3"].passed, results["C3"]

    def test_mutation_near_end_exemption_restored_would_miss_it(self):
        """Mutation (restore C2's 0.05deg exemption inside C3's own
        check, i.e. `continue` before the raw-value comparison when
        `_near_end(v, a, b)`): every value in the spliced sequence above
        is within 0.05deg of A's own goal, so the exempted mutant would
        `continue` past all of them and never reach the comparison at
        all -- reproduced directly against `_near_end` itself, which the
        mutation would gate on."""
        b = ROUTE[0].pose["r_shoulder_pitch"]
        vals = [b, b + np.radians(0.02), b + np.radians(0.04), b]
        assert all(pc._near_end(v, 0.0, b) for v in vals), (
            "every spliced value must be near-end -- otherwise this "
            "isn't proof the mutation's exemption would swallow all of them")
        # The shipped code (verified in the test above, and against a
        # da3a81c copy of pathcheck.py in the session's manual mutation
        # pass) does NOT skip these -- it compares raw values directly.


# ---------------------------------------------------------------------------
# A1: edge windows located by property, not by "first/last command"
# ---------------------------------------------------------------------------

CYCLE = "S2-B4-c-r1"


def _write_sidecar(control_dir, cycle, kind, first_seq, last_seq):
    import json
    path = control_dir / f"{cycle}-{kind}.link.json"
    path.write_text(json.dumps({"alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}]}))
    return path


class TestA1EdgeWindows:
    """Builds a realistic single-leg PLACE_ROUTE cycle (mf.FlightSim over
    the real route, `_LAG_RAD` zeroed) and splices a stray command
    directly into the states/commands stream just inside the sidecar's
    aligned span, before the leg's own real turn_on -- the exact defect
    A1 describes (the stray would otherwise BECOME `command_indices[0]`
    and the window would vanish)."""

    def _build(self, tmp_path, lead_in_stray=False, parked_tail_stray=False):
        home = dict(R.HOME)
        sim = mf.FlightSim(home, pose_units="deg", restream_passes=1, settle_s=0.05)
        sim.fly(R.PLACE_ROUTE)
        result = sim.result()
        cmds, rows = list(result.command_rows), list(result.state_rows)

        # The REAL turn_on command is commands[0] / states[0] in this
        # single-leg fixture (no prior epoch). Sidecar span covers every
        # row (first_seq=0, last_seq=last).
        if lead_in_stray:
            stray_target = list(cmds[0]["target_rad"])
            stray_target[0] += 0.05  # shoulder_pitch, well off the present-position match
            cmds = [dict(cmds[0], target_rad=stray_target)] + cmds
            # Duplicate the first state row so the stray has its own
            # reporting tick before the real turn_on's state.
            rows = [dict(rows[0])] + rows
        if parked_tail_stray:
            last_target = list(cmds[-1]["target_rad"])
            stray_target = list(last_target)
            stray_target[0] += 0.05
            new_seq = cmds[-1]["seq"] + 1
            cmds = cmds + [dict(cmds[-1], target_rad=stray_target, seq=new_seq)]
            # The appended row's own cmd_seq must report the STRAY's new
            # seq (not the original last command's), or its bracket is
            # unplaceable and invisible to the parked-tail window.
            rows = rows + [dict(rows[-1], cmd_seq=new_seq)]

        # Renumber every state's seq/sim_step (and every command's own
        # seq) sequentially after any insertion -- a raw duplicate row
        # would otherwise collide with an existing seq, making the
        # sidecar's own alignment ambiguous. Only the ORDERING and
        # COUNT matter for this fixture, not the absolute values.
        rows = [dict(r, seq=i, sim_step=i) for i, r in enumerate(rows)]
        new_cmds = []
        for c in cmds:
            if c.get("type") == "joint_command":
                c = dict(c, seq=len(new_cmds))
            new_cmds.append(c)
        cmds = new_cmds

        mf.write_evidence(tmp_path, rows, cmds)
        control_dir = tmp_path.parent / (tmp_path.name + "-control")
        control_dir.mkdir(exist_ok=True)
        _write_sidecar(control_dir, CYCLE, "setup", 0, len(rows) - 1)
        return tmp_path, control_dir

    def test_clean_leg_has_no_edge_violation(self, tmp_path):
        ev_dir, control_dir = self._build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                               R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        lr = cyc.evaluate_leg(evd, leg)
        assert lr.lead_in_violation is False, lr.lead_in_detail

    def test_stray_before_turn_on_is_a_lead_in_violation(self, tmp_path):
        ev_dir, control_dir = self._build(tmp_path, lead_in_stray=True)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                               R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        lr = cyc.evaluate_leg(evd, leg)
        assert lr.lead_in_violation is True, "the stray should not be absorbed as the real turn_on"

    def test_mutation_bounding_by_first_command_would_miss_the_stray(self, tmp_path):
        """Mutation (revert to bounding the lead-in window by
        `command_indices[0]`, i.e. `ev.leg_turn_on_state_index`): the
        stray IS `command_indices[0]` in this fixture, so the old code's
        window is `[sidecar_t_lo, state-before-the-stray]` -- empty by
        construction, and the stray itself is never inside any window at
        all. Verified directly: `ev.leg_turn_on_state_index(evd, leg.
        command_indices)` (the pre-A1 boundary source) resolves to the
        state just before the STRAY, not the real turn_on, on this exact
        fixture."""
        ev_dir, control_dir = self._build(tmp_path, lead_in_stray=True)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                               R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        old_boundary_state = ev.leg_turn_on_state_index(evd, leg.command_indices)
        turn_on_match = holds.find_turn_on_command(evd, leg.command_indices)
        assert turn_on_match is not None
        assert turn_on_match.state_index != old_boundary_state, (
            "the fix must locate a DIFFERENT (later) state than the pre-A1 boundary")

    def test_non_carry_after_final_waypoint_is_parked_tail_violation(self, tmp_path):
        ev_dir, control_dir = self._build(tmp_path, parked_tail_stray=True)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                               R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        lr = cyc.evaluate_leg(evd, leg)
        edge_stats = [hs for hs in lr.hold_stats if hs.window.goal_index == holds.PARKED_TAIL_GOAL_INDEX]
        assert edge_stats, "expected a parked-tail window to be found"
        assert any(v != 0.0 for v in edge_stats[0].target_drift.values()), (
            "the appended stray must show up as non-zero parked-tail drift")
