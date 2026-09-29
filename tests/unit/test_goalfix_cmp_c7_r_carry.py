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


# ---------------------------------------------------------------------------
# Gripper-only-waypoint ARM branch: R-const / R-carry / R-carry' (2026-09-28
# arm-branch repair, PR #147 extension)
#
# Defect: the branch used the goto's FIRST assigned arm command (``a0``) as
# every arm joint's reference, so a legitimate bit-exact leading arm carry
# (the previous goto's last non-exact write) followed by the exact nominal S
# was flagged. Repair: reference = nominal ``seg_start[j]``, carries exempt
# through ``seg.carry_mask(..., withdraw=carry_withdraw)``, any other value
# not bit-equal to S fails (N1's own rule, no tolerance).
# ---------------------------------------------------------------------------

S_ARM = float(np.float32(-0.5))                                     # nominal S = G of GRIPONLY's arm
SP_ARM = float(np.nextafter(np.float32(S_ARM), np.float32(-np.inf)))  # S - 1 float32 ULP (R-over: assignable)
SU_ARM = float(np.nextafter(np.float32(S_ARM), np.float32(np.inf)))   # S + 1 float32 ULP
GA0 = -0.6                                                          # gripper before GRIPONLY
ARM_J = "r_shoulder_pitch"

ROUTE_ARM = [mf.Waypoint("ARM", _pose(r_shoulder_pitch=S_ARM, r_gripper=GA0), 2.0),
             mf.Waypoint("GRIPONLY", _pose(r_shoulder_pitch=S_ARM, r_gripper=0.3), 2.0)]
START_ARM = _pose(r_gripper=GA0)


def _arm_targets(lead=SP_ARM, *, bad_at=None, bad=SU_ARM):
    """6 commands. Goto 0 (ARM): c0, c1 (its last write ``lead``). Goto 1 (GRIPONLY,
    arm constant at S, gripper moving): c2 carries ``lead`` bit-exactly, c3..c5 exact S.
    ``bad_at`` replaces that command's arm value with ``bad`` (fresh, non-carry, != S)."""
    t = [_pose(r_shoulder_pitch=pose_at(0.0, S_ARM, 0.001), r_gripper=GA0),
         _pose(r_shoulder_pitch=lead, r_gripper=GA0),
         _pose(r_shoulder_pitch=lead, r_gripper=-0.4),
         _pose(r_shoulder_pitch=S_ARM, r_gripper=-0.2),
         _pose(r_shoulder_pitch=S_ARM, r_gripper=0.0),
         _pose(r_shoulder_pitch=S_ARM, r_gripper=0.3)]
    if bad_at is not None:
        t[bad_at]["r_shoulder_pitch"] = bad
    return t


def _run_all_arm(targets, assignment=None):
    return pc.run_all(_t21(targets), [0.1 * i for i in range(len(targets))], START_ARM,
                      ROUTE_ARM, (), assignment=assignment)


def _build_arm_cycle(tmp_path, targets, *, state0_arm=None, never_reported=None):
    """Evidence for ``targets``: the plant reports each command's own value, except that
    ``never_reported`` (an arm value) is reported as S instead, and state 0 (before any
    command) may report ``state0_arm`` (the origin of a genuine echo of that value)."""
    s0 = _pose(r_gripper=GA0)
    if state0_arm is not None:
        s0[ARM_J] = state0_arm
    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1, wall_time_ns=1,
                          position_rad21=mf.full21(s0))]
    cmds = []
    for i, tgt in enumerate(targets):
        cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
        reported = dict(tgt)
        if never_reported is not None and reported[ARM_J] == never_reported:
            reported[ARM_J] = S_ARM
        rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * 0.1, cmd_seq=i,
                                  wall_time_ns=2 + i, position_rad21=mf.full21(reported)))
    mf.write_evidence(tmp_path, rows, cmds)
    evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
    leg = cyc.LegSpec("setup", route_rad=ROUTE_ARM, guard=(), start_pose8=START_ARM,
                       command_indices=list(range(len(cmds))))
    return evd, leg


class TestGripperOnlyArmBranchLeadingCarry:
    def test_leading_arm_carry_then_exact_S_passes_via_run_all(self):
        """The section-6 reproduction: route ARM then GRIPONLY; goto 0 ends at S - 1 ULP;
        goto 1's first command carries it bit-exactly, its second is exact S."""
        t = [_pose(r_shoulder_pitch=pose_at(0.0, S_ARM, 0.001), r_gripper=GA0),
             _pose(r_shoulder_pitch=SP_ARM, r_gripper=GA0),
             _pose(r_shoulder_pitch=SP_ARM, r_gripper=-0.4),
             _pose(r_shoulder_pitch=S_ARM, r_gripper=-0.1),
             _pose(r_shoulder_pitch=S_ARM, r_gripper=0.3)]
        asg = seg.assign_goals(ROUTE_ARM, START_ARM, t)
        assert asg.ok and asg.goal_index == [0, 0, 1, 1, 1]   # the carry IS goto 1's leading command
        assert t[2][ARM_J] == t[1][ARM_J] != S_ARM
        # run_all computes its own assignment (assignment=None)
        res = _run_all_arm(t)
        assert res["C7"].passed, (res["C7"].first_violation_index, res["C7"].detail)

    def test_cycle_path_leading_arm_carry_passes(self, tmp_path):
        evd, leg = _build_arm_cycle(tmp_path, _arm_targets(), never_reported=SP_ARM)
        cv, lrs = cyc.evaluate_cycle("c7arm-ok", "B", evd, [leg], skip_gates=True)
        lr = lrs["setup"]
        assert lr.assignment.goal_index == [0, 0, 1, 1, 1, 1]
        assert lr.pathcheck["C7"].passed, lr.pathcheck["C7"].detail
        assert "setup:C7" not in cv.reasons


class TestGripperOnlyArmBranchFreshInvalid:
    """A fresh (non-carry) arm value != S inside a gripper-only goto. On the shipped path
    ``assign_goals`` rejects the sample first (R-const/N1), so it is an unassigned
    non-carry command and the cycle STOPs on that rule; C7 is the second line of
    defence and is exercised through ``run_all`` with the clean assignment."""

    CLEAN = [0, 0, 1, 1, 1, 1]

    @pytest.mark.parametrize("bad_at", [4, 3], ids=["a_mid_goto", "b_first_non_carry"])
    def test_shipped_path_stops_on_unassigned_non_carry(self, tmp_path, bad_at):
        t = _arm_targets(bad_at=bad_at)
        assert seg.assign_goals(ROUTE_ARM, START_ARM, t).goal_index[bad_at] is None
        evd, leg = _build_arm_cycle(tmp_path, t, never_reported=SP_ARM)
        cv, lrs = cyc.evaluate_cycle(f"c7arm-bad{bad_at}", "B", evd, [leg], skip_gates=True)
        lr = lrs["setup"]
        assert cv.verdict == cyc.VERDICT_STOP
        assert lr.assignment.goal_index[bad_at] is None
        unassigned = [r for r in cv.reasons
                      if "unassigned non-carry command" in r and f"local index {bad_at}" in r]
        assert unassigned, cv.reasons
        # the unassigned index never reaches C7, so C7 itself does not fire on this path
        assert "setup:C7" not in cv.reasons

    @pytest.mark.parametrize("bad_at", [4, 3], ids=["a_mid_goto", "b_first_non_carry"])
    def test_c7_fails_at_that_index_with_clean_assignment(self, bad_at):
        c7 = _run_all_arm(_arm_targets(bad_at=bad_at), assignment=_asg(self.CLEAN))["C7"]
        assert not c7.passed and c7.first_violation_index == bad_at
        assert c7.detail == f"GRIPONLY: arm joint {ARM_J} moved during a gripper-only waypoint"
        # same inputs, the bad value restored to exact S: passes
        assert _run_all_arm(_arm_targets(), assignment=_asg(self.CLEAN))["C7"].passed

    def test_first_non_carry_below_S_also_fails(self):
        """Not only S + 1 ULP: S - 1 ULP as a fresh non-carry value (previous is exact S)."""
        t = _arm_targets(lead=SP_ARM)
        t[4][ARM_J] = SP_ARM       # previous command is exact S, so this is NOT a carry
        c7 = _run_all_arm(t, assignment=_asg(self.CLEAN))["C7"]
        assert not c7.passed and c7.first_violation_index == 4


class TestGripperOnlyArmBranchRCarryPrime:
    """A leading arm repeat whose chain origin the shipped classifier labels
    ``genuine_echo`` is an ``echo_carry`` and gets no exemption; a fresh-origin
    repeat is a clean carry."""

    def _classify(self, evd, lr, leg):
        return echo.classify_commands(
            evd, [lr.goto_context[i] for i in range(len(evd.commands))],
            leg_turn_on_state_index={0: ev.leg_turn_on_state_index(evd, leg.command_indices)})

    def test_echo_origin_arm_repeat_gets_no_carry_exemption(self, tmp_path):
        # state 0 already reports S - 1 ULP, so command 1's write of that value is a genuine echo
        evd, leg = _build_arm_cycle(tmp_path, _arm_targets(), state0_arm=SP_ARM,
                                    never_reported=None)
        cv, lrs = cyc.evaluate_cycle("c7arm-echo", "B", evd, [leg], skip_gates=True)
        lr = lrs["setup"]
        assert lr.assignment.goal_index[2] == 1 and lr.assignment.goal_index[3] == 1
        res = self._classify(evd, lr, leg)
        assert res[1].joints[ARM_J].label == echo.GENUINE_ECHO
        assert res[2].joints[ARM_J].label == echo.ECHO_CARRY
        c7 = lr.pathcheck["C7"]
        assert not c7.passed
        assert c7.detail.startswith("R-carry′:"), c7.detail
        assert c7.first_violation_index == 2
        assert "arm joint r_shoulder_pitch moved during a gripper-only waypoint" in c7.detail
        assert "setup:C7" in cv.reasons and cv.verdict == cyc.VERDICT_STOP

    def test_fresh_origin_arm_repeat_is_a_clean_carry_c7_passes(self, tmp_path):
        evd, leg = _build_arm_cycle(tmp_path, _arm_targets(), never_reported=SP_ARM)
        cv, lrs = cyc.evaluate_cycle("c7arm-fresh", "B", evd, [leg], skip_gates=True)
        lr = lrs["setup"]
        assert lr.assignment.goal_index[2] == 1 and lr.assignment.goal_index[3] == 1
        res = self._classify(evd, lr, leg)
        assert res[1].joints[ARM_J].label == echo.FRESH
        assert res[2].joints[ARM_J].label == echo.CARRY
        assert lr.pathcheck["C7"].passed, lr.pathcheck["C7"].detail
        assert "setup:C7" not in cv.reasons

    def test_direct_check_c7_withdraw_flag_arm_branch(self):
        t = _arm_targets()
        asg = _asg(self.CLEAN)
        assert pc.check_c7(t, START_ARM, ROUTE_ARM, asg).passed
        wd = [{n: False for n in pc.R_JOINTS} for _ in t]
        wd[2][ARM_J] = True
        c7 = pc.check_c7(t, START_ARM, ROUTE_ARM, asg, carry_withdraw=wd)
        assert not c7.passed and c7.first_violation_index == 2

    CLEAN = [0, 0, 1, 1, 1, 1]
