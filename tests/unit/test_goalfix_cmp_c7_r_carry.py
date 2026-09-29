"""C7 constant-gripper check under R-const / R-carry / R-carry' (2026-09-28).

Defect (2026-09-28 seven-run harness batch, run-04 cycle S2-B4-c-r3, setup
leg): ``check_c7``'s constant-gripper branch used the goto's FIRST assigned
command as its reference (``g0``), so a legitimate bit-exact leading carry
(BACK's first command carried GRIP_SHUT's last gripper value, 1 float32 ULP
below the nominal S) followed by the exact S was flagged. N1 in
``check_c2_c3`` iterates ARM7 only, so C7 is the only check of gripper
constancy. The fix applies N1's own rule to ``r_gripper`` through the same
``seg.carry_mask`` (nominal S reference, carries exempt, R-carry' withdraw).

Frozen real evidence: ``tests/fixtures/goalfix_cmp/c7_run04_r3_setup.json``
(extracted once, never re-extracted by a test; the extraction script sits
beside it and is never run here).

Note on invalid controls: ``assign_goals`` itself already refuses a NON-carry
constant-joint sample that is not S (R-const/N1), so such a command is
unassigned and never reaches C7 when the assignment is recomputed from the
corrupted trajectory. As ``run_all``'s own docstring prescribes for
single-violation fixtures, the controls therefore pass the assignment
computed from the CLEAN trajectory (or an explicit one).
"""
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import pathcheck as pc  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp._minjerk import pose_at  # noqa: E402
from tools.goalfix_cmp._units import route_rad  # noqa: E402

FIXTURE = os.path.join(_HERE, "../fixtures/goalfix_cmp/c7_run04_r3_setup.json")

S_REAL = 0.3490658402442932   # float32(20 deg): BACK's constant S = G
O_REAL = 0.3490658104419708   # GRIP_SHUT's last write: S - 1 float32 ULP


def _ulp32_up(x: float) -> float:
    return float(np.nextafter(np.float32(x), np.float32(np.inf)))


def _ulp32_down(x: float) -> float:
    return float(np.nextafter(np.float32(x), np.float32(-np.inf)))


# ---------------------------------------------------------------------------
# Frozen real leg
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frozen():
    with open(FIXTURE) as f:
        fx = json.load(f)
    rows = fx["rows"]
    other = fx["other13_constant"]
    targets21 = np.array([r["arm8"] + other for r in rows], dtype=np.float64)
    t_hi = [r["t_hi"] for r in rows]
    start = dict(fx["start_pose8"])
    route = route_rad(R.PLACE_ROUTE)
    targets8 = [{n: float(v) for n, v in zip(pc.R_JOINTS, row[:8])} for row in targets21]
    assignment = seg.assign_goals(route, start, targets8)
    return dict(fx=fx, targets21=targets21, t_hi=t_hi, start=start, route=route,
                targets8=targets8, assignment=assignment)


def _run_all(fz, targets21=None, assignment=None):
    return pc.run_all(fz["targets21"] if targets21 is None else targets21, fz["t_hi"], fz["start"],
                      fz["route"], R.CRITICAL_JOINTS,
                      assignment=fz["assignment"] if assignment is None else assignment)


def _goal_of(fz, name):
    return next(k for k, wp in enumerate(fz["route"]) if wp.name == name)


class TestFrozenRealRegression:
    def test_fixture_provenance_and_chain(self, frozen):
        fx = frozen["fx"]
        assert fx["source"]["M"] == "155fc1549812146a8579080c72560c2c1875262c"
        assert fx["source"]["cycle"] == "S2-B4-c-r3" and fx["source"]["leg"] == "setup"
        assert fx["indices"] == {"chain_origin_local": 154, "carry_local": 155,
                                 "first_exact_S_local": 156,
                                 "c7_recorded_first_violation_index": 156}
        g = [r["arm8"][pc.R_JOINTS.index("r_gripper")] for r in fx["rows"]]
        assert g[154] == O_REAL and g[155] == O_REAL and g[156] == S_REAL
        # nominal S of BACK, bit-exact
        back = frozen["route"][_goal_of(frozen, "BACK")]
        assert back.pose["r_gripper"] == S_REAL
        assert frozen["route"][_goal_of(frozen, "GRIP_SHUT")].pose["r_gripper"] == S_REAL

    def test_gripper_chain_origin_is_fresh_write_and_carry_has_no_message(self, frozen):
        fx = frozen["fx"]
        # recorded shipped-classifier label of the gripper chain's origin: not genuine_echo
        runs = fx["echo_labels_runs"]["r_gripper"]

        def label(i):
            return next(lab for lo, hi, lab in runs if lo <= i <= hi)
        assert label(154) != echo.GENUINE_ECHO
        assert label(154) == echo.FRESH
        assert label(155) == echo.CARRY
        assert all(lab != echo.GENUINE_ECHO and lab != echo.ECHO_CARRY
                   for _lo, _hi, lab in runs)
        chain = {c["local"]: c for c in fx["gripper_chain"]}
        j = pc.R_JOINTS.index("r_gripper")
        # (a) origin: a fresh GRIP_SHUT SDK write carrying the gripper
        assert chain[154]["gripper_source_batch"] == 5522
        assert j in chain[154]["messages"][0]["joints_carried"]
        assert chain[154]["messages"][0]["gripper_value"] == O_REAL
        # (b) BACK's first command: no message supplied the gripper (carry)
        assert chain[155]["gripper_source_batch"] is None
        assert j not in chain[155]["messages"][0]["joints_carried"]
        assert chain[155]["messages"][0]["joints_carried"] == [0]
        # (c) exact S came from a later message
        assert chain[156]["gripper_source_batch"] == 5524
        assert chain[156]["messages"][0]["gripper_value"] == S_REAL

    def test_c7_passes_and_other_checks_identical_to_recorded_at_M(self, frozen):
        res = _run_all(frozen)
        assert res["C7"].passed, (res["C7"].first_violation_index, res["C7"].detail)
        rec = frozen["fx"]["recorded_setup_checks"]
        for cid in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C8"):
            assert res[cid].passed == rec[cid]["passed"], cid
            assert res[cid].first_violation_index == rec[cid]["index"], cid
        # C2 still fails on this leg, at the recorded index (untouched by this repair)
        assert not res["C2"].passed and res["C2"].first_violation_index == 476

    def test_leading_carry_is_assigned_to_back(self, frozen):
        back = _goal_of(frozen, "BACK")
        gi = frozen["assignment"].goal_index
        assert gi[155] == back and gi[156] == back


class TestRealDerivedInvalidControl:
    def test_non_carry_within_one_ulp_of_S_fails_at_that_index(self, frozen):
        """BACK's first command's gripper replaced by S + 1 float32 ULP: neither
        a carry (previous is O) nor S. Clean assignment (see module note)."""
        j = pc.R_JOINTS.index("r_gripper")
        t21 = frozen["targets21"].copy()
        bad = _ulp32_up(S_REAL)
        assert bad != S_REAL and bad != O_REAL
        t21[155, j] = bad
        res = _run_all(frozen, targets21=t21)
        assert not res["C7"].passed
        assert res["C7"].first_violation_index == 155
        assert "BACK" in res["C7"].detail


# ---------------------------------------------------------------------------
# Synthetic controls (run_all with an explicit / clean assignment)
# ---------------------------------------------------------------------------

def _pose(**kw):
    base = {j: 0.0 for j in mf.R_JOINTS}
    base.update(kw)
    return base


def _t21(rows):
    return np.array([mf.full21(r) for r in rows], dtype=np.float64)


def _asg(goal_index):
    return seg.GoalAssignment(list(goal_index), True)


def _const_goto_targets(n=6):
    sp = [pose_at(0.0, -1.0, i / (n - 1)) for i in range(n)]
    return [_pose(r_shoulder_pitch=s, r_gripper=S_REAL) for s in sp]


def _run_c7_synth(route, start, targets, goal_index):
    t21 = _t21(targets)
    t_hi = [0.1 * i for i in range(len(targets))]
    return pc.run_all(t21, t_hi, start, route, (), assignment=_asg(goal_index))["C7"]


class TestSyntheticInvalidControls:
    ROUTE_CONST = [mf.Waypoint("HOLDG", _pose(r_shoulder_pitch=-1.0, r_gripper=S_REAL), 3.0)]
    START_CONST = _pose(r_gripper=S_REAL)

    def test_clean_constant_goto_passes(self):
        c7 = _run_c7_synth(self.ROUTE_CONST, self.START_CONST, _const_goto_targets(), [0] * 6)
        assert c7.passed

    def test_a_mid_goto_fresh_value_not_S_fails(self):
        t = _const_goto_targets()
        t[3]["r_gripper"] = _ulp32_up(S_REAL)
        c7 = _run_c7_synth(self.ROUTE_CONST, self.START_CONST, t, [0] * 6)
        assert not c7.passed and c7.first_violation_index == 3

    def test_b_first_non_carry_value_not_S_fails(self):
        """Two-goto route (gripper-changing goto, then constant): a leading bit-exact
        carry of O is allowed, but the goto's first NON-carry value must equal S."""
        route = [mf.Waypoint("SHUT", _pose(r_gripper=S_REAL), 2.0),
                 mf.Waypoint("BACKISH", _pose(r_shoulder_pitch=-1.0, r_gripper=S_REAL), 3.0)]
        start = _pose(r_gripper=-0.6)
        t = [_pose(r_gripper=-0.6), _pose(r_gripper=O_REAL),                   # goto 0
             _pose(r_shoulder_pitch=pose_at(0.0, -1.0, 0.1), r_gripper=O_REAL),  # carry: allowed
             _pose(r_shoulder_pitch=pose_at(0.0, -1.0, 0.3),
                   r_gripper=_ulp32_down(O_REAL)),                             # first non-carry != S
             _pose(r_shoulder_pitch=-1.0, r_gripper=S_REAL)]
        c7 = _run_c7_synth(route, start, t, [0, 0, 1, 1, 1])
        assert not c7.passed and c7.first_violation_index == 3
        # same input, first non-carry value exactly S: passes
        t[3]["r_gripper"] = S_REAL
        assert _run_c7_synth(route, start, t, [0, 0, 1, 1, 1]).passed

    def test_c_gripper_leaves_S_after_reaching_it_fails(self):
        t = _const_goto_targets()
        for i in (3, 4, 5):
            t[i]["r_gripper"] = O_REAL
        c7 = _run_c7_synth(self.ROUTE_CONST, self.START_CONST, t, [0] * 6)
        assert not c7.passed and c7.first_violation_index == 3

    def test_d_value_outside_gripper_range_fails_existing_branch(self):
        t = _const_goto_targets()
        t[2]["r_gripper"] = pc.GRIPPER_HI_RAD + 0.2
        c7 = _run_c7_synth(self.ROUTE_CONST, self.START_CONST, t, [0] * 6)
        assert not c7.passed and c7.first_violation_index == 2
        assert "outside commanded range" in c7.detail


# ---------------------------------------------------------------------------
# R-carry' through cycle.evaluate_cycle
# ---------------------------------------------------------------------------

G0 = -0.6
X_ECHO = float(np.float32(-0.9))   # off [G0, S]: cannot be on-path in goto 0


def _build_carry_prime(tmp_path, *, echo_origin: bool):
    """Goto 0 SHUT (gripper G0 -> S, arm constant); goto 1 BACK (arm moves, gripper
    constant S). Goto 1's first command carries X, written at t2 (goto 0's tail).
    echo_origin: state 0 already reports X (t2's value is then a genuine echo);
    otherwise no state ever reports X (t2 is a fresh write)."""
    route = [mf.Waypoint("SHUT", _pose(r_gripper=S_REAL), 2.0),
             mf.Waypoint("BACK", _pose(r_shoulder_pitch=-1.0, r_gripper=S_REAL), 3.0)]
    start = _pose(r_gripper=G0)
    targets = [
        _pose(r_gripper=G0),                                                        # t0
        _pose(r_gripper=G0 + 0.5 * (S_REAL - G0)),                                  # t1
        _pose(r_gripper=X_ECHO),                                                    # t2: origin of X
        _pose(r_shoulder_pitch=pose_at(0.0, -1.0, 0.10), r_gripper=X_ECHO),         # t3: leading repeat
        _pose(r_shoulder_pitch=pose_at(0.0, -1.0, 0.20), r_gripper=S_REAL),         # t4: exact S
        _pose(r_shoulder_pitch=pose_at(0.0, -1.0, 0.30), r_gripper=S_REAL),         # t5
        _pose(r_shoulder_pitch=-1.0, r_gripper=S_REAL),                             # t6
    ]
    dt = 0.1
    state0 = _pose(r_gripper=X_ECHO if echo_origin else G0)
    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1, wall_time_ns=1,
                          position_rad21=mf.full21(state0))]
    cmds = []
    for i, tgt in enumerate(targets):
        cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
        reported = dict(tgt)
        if reported["r_gripper"] == X_ECHO:
            reported["r_gripper"] = S_REAL   # the plant never reports X after state 0
        rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * dt, cmd_seq=i,
                                  wall_time_ns=2 + i, position_rad21=mf.full21(reported)))
    mf.write_evidence(tmp_path, rows, cmds)
    evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
    leg = cyc.LegSpec("setup", route_rad=route, guard=(), start_pose8=start,
                       command_indices=list(range(len(cmds))))
    return evd, leg


class TestRCarryPrimeThroughEvaluateCycle:
    def test_echo_origin_repeat_gets_no_carry_exemption(self, tmp_path):
        evd, leg = _build_carry_prime(tmp_path, echo_origin=True)
        cv, lrs = cyc.evaluate_cycle("c7ec", "B", evd, [leg], skip_gates=True)
        lr = lrs["setup"]
        j = "r_gripper"
        # the repeat really is a leading command of goto 1 (scenario validity)
        assert lr.assignment.goal_index[3] == 1 and lr.assignment.goal_index[4] == 1
        results = echo.classify_commands(
            evd, [lr.goto_context[i] for i in range(len(evd.commands))],
            leg_turn_on_state_index={0: ev.leg_turn_on_state_index(evd, leg.command_indices)})
        assert results[2].joints[j].label == echo.GENUINE_ECHO
        assert results[3].joints[j].label == echo.ECHO_CARRY
        c7 = lr.pathcheck["C7"]
        assert not c7.passed
        assert c7.detail.startswith("R-carry′:"), c7.detail
        assert c7.first_violation_index == 3
        assert "setup:C7" in cv.reasons and cv.verdict == cyc.VERDICT_STOP

    def test_fresh_origin_repeat_is_a_clean_carry_c7_passes(self, tmp_path):
        evd, leg = _build_carry_prime(tmp_path, echo_origin=False)
        cv, lrs = cyc.evaluate_cycle("c7fr", "B", evd, [leg], skip_gates=True)
        lr = lrs["setup"]
        assert lr.assignment.goal_index[3] == 1 and lr.assignment.goal_index[4] == 1
        results = echo.classify_commands(
            evd, [lr.goto_context[i] for i in range(len(evd.commands))],
            leg_turn_on_state_index={0: ev.leg_turn_on_state_index(evd, leg.command_indices)})
        assert results[2].joints["r_gripper"].label == echo.FRESH
        assert results[3].joints["r_gripper"].label == echo.CARRY
        assert lr.pathcheck["C7"].passed, lr.pathcheck["C7"].detail
        assert "setup:C7" not in cv.reasons

    def test_direct_check_c7_withdraw_flag(self, tmp_path):
        """Direct call, in addition: carry_withdraw threads to seg.carry_mask."""
        evd, leg = _build_carry_prime(tmp_path, echo_origin=True)
        t8 = [{n: float(v) for n, v in zip(pc.R_JOINTS, evd.commands.target_rad[i, :8])}
              for i in leg.command_indices]
        asg = seg.assign_goals(leg.route_rad, leg.start_pose8, t8)
        assert pc.check_c7(t8, leg.start_pose8, leg.route_rad, asg).passed
        wd = [{n: False for n in pc.R_JOINTS} for _ in t8]
        wd[3]["r_gripper"] = True
        c7 = pc.check_c7(t8, leg.start_pose8, leg.route_rad, asg, carry_withdraw=wd)
        assert not c7.passed and c7.first_violation_index == 3


# ---------------------------------------------------------------------------
# Unchanged behaviour
# ---------------------------------------------------------------------------

class TestUnchangedBranches:
    def test_gripper_changing_goto_is_range_checked_only(self):
        route = [mf.Waypoint("SHUT", _pose(r_gripper=S_REAL), 2.0)]
        start = _pose(r_gripper=G0)
        wander = [-0.6, -0.2, -0.5, 0.3, 0.1, S_REAL]      # non-monotonic, all in range
        t = [_pose(r_gripper=g) for g in wander]
        assert _run_c7_synth(route, start, t, [0] * 6).passed
        t[2]["r_gripper"] = -5.0                             # out of range
        c7 = _run_c7_synth(route, start, t, [0] * 6)
        assert not c7.passed and c7.first_violation_index == 2

    def test_gripper_only_waypoint_arm_branch_as_at_M(self):
        route = [mf.Waypoint("SHUT", _pose(r_gripper=S_REAL), 2.0)]
        start = _pose(r_gripper=G0)
        t = [_pose(r_gripper=g) for g in (-0.6, -0.4, -0.2, 0.0, 0.2, S_REAL)]
        assert _run_c7_synth(route, start, t, [0] * 6).passed
        t[4]["r_shoulder_pitch"] = 0.01                      # arm moves mid gripper-only goto
        c7 = _run_c7_synth(route, start, t, [0] * 6)
        assert not c7.passed and c7.first_violation_index == 4
        assert c7.detail == "SHUT: arm joint r_shoulder_pitch moved during a gripper-only waypoint"
