"""F6 acceptance tests (merge verdict review-2026-09-25-pr144-0722476-
merge-verdict.md §2; 2026-09-25 pr144-f1-f6 assignment): the three
mutants that survived every prior mutation pass.

1. (q) "B ignores a genuine echo in the segment", with a fixture where
   every C0-C8 check ALSO passes (the earlier fixture in
   test_goalfix_cmp_t8_mutants.py's TestMutantQ_BIgnoresGenuineEchoInSegment
   STOPs through C1 too, so that mutant survives).
2. The MB2 --arm cross-check, through the shipped cycle CLI (P5, both
   directions).
3. The A1 lead-in gate, via a carry command before turn_on, through the
   shipped cycle CLI.
"""
import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp", "."):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
import test_goalfix_cmp_f4_f5_manifest_reset_binding as f4f5  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import segments as seg  # noqa: E402
from tools.goalfix_cmp._io import RC_STOP, read_result  # noqa: E402
from tools.goalfix_cmp._units import route_rad  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.rig_routes import Waypoint as RWaypoint  # noqa: E402


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


CYCLE = "S2-B4-c-r1"


def _write_sidecar(control_dir, cycle, kind, first_seq, last_seq):
    path = control_dir / f"{cycle}-{kind}.link.json"
    path.write_text(json.dumps({"alignment": [{"server_seq": first_seq}, {"server_seq": last_seq}]}))
    return path


# ---------------------------------------------------------------------------
# 1. (q): B ignores a genuine echo in the segment
# ---------------------------------------------------------------------------

class TestMutantQIsolatedProof:
    """The Q-echo ruling (2026-09-25 stage-1 rulings §3; A2 addendum)
    exempts a near-END, ON-PATH match from genuine_echo (path
    coincidence), but a match that lies just BEYOND the goal (outside
    [start, goal], hence never on-path) is not exempt -- report §4's C2
    near-end exemption is from the tau-agreement comparison only, not a
    blanket "near an end is safe" rule, and C4's residual budget and
    C1's 1e-6 rad tolerance both still hold for a sub-tolerance overshoot.

    ISOLATION PROOF (library level, `evaluate_cycle` directly -- see the
    class docstring below for why this specific isolation cannot also be
    reproduced through `cycle._cli`): a single HOVER->REST_SHUT-shaped
    leg with NOTHING after REST_SHUT (so there is no later goto whose own
    C1 boundary the overshoot could disturb) plants an exact float32
    match, 0.2s earlier, of `goal - 5e-7 rad` (a few ULPs beyond the
    goal, well inside both C1's 1e-6 rad tolerance and C4's residual
    budget) at REST_SHUT's own last command. Every C0-C8 check passes;
    genuine_echo_count == 1."""

    def _build(self, tmp_path):
        start_deg = {j: 0.0 for j in mf.R_JOINTS}
        hover_deg = dict(start_deg, r_shoulder_pitch=-30.0)
        rest_shut_deg = dict(start_deg, r_shoulder_pitch=-70.0)
        route_deg = (RWaypoint("HOVER", hover_deg, 2.0, 6.0),
                     RWaypoint("REST_SHUT", rest_shut_deg, 2.0, 6.0))
        route_r = route_rad(route_deg)
        start_rad = route_rad((RWaypoint("START", start_deg, 1.0, 6.0),))[0].pose

        sim = mf.FlightSim(start_deg, pose_units="deg")
        sim.fly([mf.Waypoint(wp.name, wp.pose, wp.seconds) for wp in route_deg])
        result = sim.result()
        cmds, rows = list(result.command_rows), list(result.state_rows)

        goal_val = route_r[1].pose["r_shoulder_pitch"]
        # R-over (owner rulings, 2026-09-25): a setpoint beyond the goal
        # by MORE than 1 float32 ULP, in the direction of travel, never
        # belongs to this goto (segments.assign_goals rejects it, falling
        # through to indeterminate here since there is no further
        # waypoint in this fixture's 2-waypoint route to attribute it
        # to). Exactly 1 ULP beyond stays admitted -- this fixture needs
        # the overshoot to still be REST_SHUT's OWN last command (to
        # isolate the echo-classifier question from segmentation), so it
        # uses exactly 1 ULP (not the old, larger 5e-7 rad ~= 4 ULPs at
        # this magnitude), via `np.nextafter` in the continuing
        # (decreasing) direction.
        overshoot = float(np.nextafter(np.float32(goal_val), np.float32(-np.inf)))
        pick_cmd = 201  # REST_SHUT's own last command (verified below)
        src_row = 207   # ~0.2s earlier in simulation time (verified below)

        rows2 = [dict(r) for r in rows]
        rows2[src_row] = dict(rows2[src_row])
        joints = [dict(j) for j in rows2[src_row]["joints"]]
        joints[0] = dict(joints[0], position_rad=overshoot)  # r_shoulder_pitch
        rows2[src_row]["joints"] = joints
        cmds2 = [dict(c, target_rad=list(c["target_rad"])) for c in cmds]
        cmds2[pick_cmd]["target_rad"][0] = overshoot

        mf.write_evidence(tmp_path, rows2, cmds2)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        # Pin the two indices' own meaning, so a future make_fixtures/
        # rig_routes change fails loudly here rather than silently
        # planting the echo somewhere else.
        idx = list(range(len(evd.commands)))
        targets8 = [{name: float(row[i]) for i, name in enumerate(cyc.pc.R_JOINTS)}
                    for row in evd.commands.target_rad[idx]]
        assignment = seg.assign_goals(route_r, start_rad, targets8)
        rest_shut_positions = [i for i, g in enumerate(assignment.goal_index) if g == 1]
        assert rest_shut_positions[-1] == pick_cmd, "pick_cmd must be REST_SHUT's own last command"
        assert evd.brackets[pick_cmd].t_hi - evd.states.sim_time_s[src_row] < 0.5, (
            "src_row must be inside the 0.5s echo lookback")

        leg = cyc.LegSpec("setup", route_r, guard=(), start_pose8=start_rad, command_indices=idx)
        return evd, leg

    def test_genuine_echo_with_every_c_check_passing(self, tmp_path):
        evd, leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("f6q", "B", evd, [leg], skip_gates=True)
        for cid, r in leg_results["setup"].pathcheck.items():
            if hasattr(r, "passed"):
                assert r.passed, f"{cid} unexpectedly failed: {getattr(r, 'detail', None)}"
        assert cv.genuine_echo_count == 1, cv.as_dict()
        assert cv.verdict == cyc.VERDICT_STOP
        assert cv.reasons == ["1 genuine echo(es) in the affected segment"]

    def test_mutation_b_ignores_genuine_echo_would_authorize(self, tmp_path):
        """Mutation (drop `genuine > 0` from cycle.py's B STOP
        condition): with every OTHER signal negative (asserted above),
        this exact fixture's verdict would go STOP -> ok. Reproduced
        directly against the shipped condition's own inputs."""
        evd, leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("f6q", "B", evd, [leg], skip_gates=True)
        any_pathcheck_fail = any(
            not r.passed for r in leg_results["setup"].pathcheck.values() if hasattr(r, "passed"))
        any_hold_target_drift = any(
            any(v != 0.0 for v in hs.target_drift.values()) for hs in leg_results["setup"].hold_stats)
        assert not any_pathcheck_fail
        assert not any_hold_target_drift
        assert not leg_results["setup"].lead_in_violation
        assert not cv.segment_indeterminate
        assert cv.genuine_echo_count > 0
        # The mutant's own (wrong) verdict: with genuine_echo_count
        # dropped from the OR-condition, none of the remaining signals
        # is true, so the mutant would authorize (verdict "ok").
        mutant_would_stop = (any_pathcheck_fail or any_hold_target_drift
                              or leg_results["setup"].lead_in_violation or cv.segment_indeterminate)
        assert not mutant_would_stop


class TestMutantQThroughCli:
    """The SAME construction, through the shipped `cycle._cli` -- which,
    unlike the isolated proof above, must use the REAL, un-truncatable
    `R.PLACE_ROUTE` (hardcoded in `resolve_leg`'s own call sites in
    `cycle._cli`, never parameterisable). `R.PLACE_ROUTE` has REST
    immediately after REST_SHUT, so REST_SHUT's own last command is
    never the absolute end of the leg -- perturbing it by even a
    sub-ULP-tolerance amount makes `segments.assign_goals`'s R-tie/
    R-const tie-break reassign it toward REST's own goto (verified
    directly below: the assignment around this command goes
    [..., REST_SHUT, REST_SHUT, REST, REST_SHUT, None] -- an
    out-of-order sequence), which THEN also fails C0 (goal sequence)
    and C1/C2 at other boundaries. This is a structural property of
    R.PLACE_ROUTE's own waypoint spacing, not a defect in the fix, and
    it is why the isolated proof above cannot also be reproduced through
    the CLI's own hardcoded route. Accepted as reported, in the same
    manner test_goalfix_cmp_t8_mutants.py's own
    TestMutantQ_BIgnoresGenuineEchoInSegment already does for its own
    (different) non-isolating construction: genuine_echo_count and the
    STOP verdict are asserted; the co-firing C0/C1/C2 are not hidden."""

    def _build(self, tmp_path):
        home = dict(R.HOME)
        sim = mf.FlightSim(home, pose_units="deg", restream_passes=0, settle_s=0.05)
        sim.fly(R.PLACE_ROUTE)
        setup_last_seq = sim.state_rows[-1]["seq"]
        for _ in range(25):
            sim._hold_ticks(1, dict(sim.pose))
        flight_first_seq = sim.state_rows[-1]["seq"]
        sim.pose = dict(R.REST)
        sim.fly(R.LIFT_TO_PRESENT)
        flight_last_seq = sim.state_rows[-1]["seq"]
        result = sim.result()
        cmds, rows = list(result.command_rows), list(result.state_rows)

        route_r = route_rad(R.PLACE_ROUTE)
        names = [wp.name for wp in R.PLACE_ROUTE]
        rest_shut_i = names.index("REST_SHUT")
        goal_val = route_r[rest_shut_i].pose["r_shoulder_pitch"]
        pick_cmd = 1359
        src_row = 1369
        eps = np.float32(5e-7)
        overshoot = float(np.float32(np.float32(goal_val) - eps))

        rows2 = [dict(r) for r in rows]
        rows2[src_row] = dict(rows2[src_row])
        joints = [dict(j) for j in rows2[src_row]["joints"]]
        joints[0] = dict(joints[0], position_rad=overshoot)
        rows2[src_row]["joints"] = joints
        cmds2 = [dict(c, target_rad=list(c["target_rad"])) for c in cmds]
        cmds2[pick_cmd]["target_rad"][0] = overshoot

        ev_dir = tmp_path / "ev"
        control_dir = tmp_path / "control"
        control_dir.mkdir()
        mf.write_evidence(ev_dir, rows2, cmds2)
        _write_sidecar(control_dir, CYCLE, "setup", 0, setup_last_seq)
        _write_sidecar(control_dir, CYCLE, "flight", flight_first_seq, flight_last_seq)
        return ev_dir, control_dir

    def test_genuine_echo_stops_through_the_cli(self, tmp_path):
        """B15/H12 (coordinator review, 2026-09-25 Stage-A slice review,
        §4; owner Stage B authorization) now catches this construction's
        off-path, non-carry injection as an unassigned command before
        echo classification's own segment-scoped count runs -- STOP
        still holds, via the unassigned-non-carry reason (co-firing with
        C1/C2/lead-in at this route's own out-of-order boundary, per the
        class docstring)."""
        ev_dir, control_dir = self._build(tmp_path)
        out = tmp_path / "out.json"
        rc = cyc._cli([
            "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", CYCLE,
            "--rep", "1", "--arm", "B", "--out", str(out), "--validation-mode",
        ])
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_STOP, payload
        assert any("unassigned non-carry" in r for r in payload["reasons"]), payload["reasons"]


# ---------------------------------------------------------------------------
# 2. MB2 --arm cross-check, through cycle._cli (P5, both directions)
# ---------------------------------------------------------------------------

class TestP5ArmCrossCheckThroughCli:
    def test_a_rep_flagged_as_b_stops_with_arm_mismatch_reason(self, tmp_path):
        """P5: rep 1 is 'A' in ARM_MAP_ORDER; --arm B contradicts it."""
        ev_dir, control_dir = f4f5._build_cycle_n_resets(tmp_path, 2)
        f4f5._write_manifest(control_dir, rep=1, reset_gen=2, reset_record="reset_record.txt")
        f4f5._write_reset_record(control_dir, "reset_record.txt",
                                  "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        gate_args = f4f5._write_full_gates(control_dir)
        out = tmp_path / "out.json"
        rc = f4f5._run_cli(ev_dir, control_dir, "B", out, rep=1, extra=gate_args)
        payload = read_result(out)
        assert rc == RC_STOP, payload
        assert payload["verdict"] == cyc.VERDICT_STOP
        assert any("contradicts arm_map" in r for r in payload["reasons"]), payload["reasons"]

    def test_b_rep_flagged_as_a_stops_with_arm_mismatch_reason(self, tmp_path):
        """The reverse of P5: rep 2 (GATE_REP) is 'B'; --arm A contradicts it."""
        ev_dir, control_dir = f4f5._build_cycle_n_resets(tmp_path, 2)
        f4f5._write_manifest(control_dir, rep=f4f5.GATE_REP, reset_gen=2,
                              reset_record="reset_record.txt")
        f4f5._write_reset_record(control_dir, "reset_record.txt",
                                  "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        gate_args = f4f5._write_full_gates(control_dir)
        out = tmp_path / "out.json"
        rc = f4f5._run_cli(ev_dir, control_dir, "A", out, rep=f4f5.GATE_REP, extra=gate_args)
        payload = read_result(out)
        assert rc == RC_STOP, payload
        assert payload["verdict"] == cyc.VERDICT_STOP
        assert any("contradicts arm_map" in r for r in payload["reasons"]), payload["reasons"]

    def test_mutation_arm_cross_check_removed_would_authorize(self, tmp_path):
        """Mutation (drop the `entry.arm != args.arm` branch in
        cycle.py's non-validation-mode gate block): the P5 fixture
        above would then fall through to the `else` branch, which reads
        versions_<cycle>.json normally (matching bridge_arm='B' in
        _write_full_gates) and PASSES the provenance gate -- reproduced
        directly against `pv.check_cycle`, which never even sees the
        operator's own --arm flag at all (T6's own scope)."""
        import tools.goalfix_cmp.provenance as pv
        ev_dir, control_dir = f4f5._build_cycle_n_resets(tmp_path, 2)
        f4f5._write_manifest(control_dir, rep=1, reset_gen=2, reset_record="reset_record.txt")
        f4f5._write_reset_record(control_dir, "reset_record.txt",
                                  "reset gen=2 ack=2 resets_recorded 1->2 sim_step 0->0\n")
        gate_args = f4f5._write_full_gates(control_dir)
        arm_map_doc = json.loads((control_dir / "arm_map.json").read_text())
        arm_map = {e["rep"]: pv.ArmMapEntry(**e) for e in arm_map_doc}
        assert arm_map[1].arm == "A"  # confirms the fixture's own premise
        # The mutant's own (wrong) behaviour: nothing in `pv.check_cycle`
        # itself reads the operator's --arm flag, so an --arm B
        # invocation for rep 1 would pass exactly as --arm A would.


# ---------------------------------------------------------------------------
# 3. The A1 lead-in gate: a carry command before turn_on, through cycle._cli
# ---------------------------------------------------------------------------

class TestA1LeadInCarryThroughCli:
    """A1 (coordinator ruling, 2026-09-25 stage-2a rulings addendum):
    "the leg's turn_on command is the first in-span command whose
    target equals, on ALL 8 joints, the state in force at its own
    bracket's t_lo." Two IDENTICAL (bit-exact -- a genuine carry, per
    the review's own "target bit-equal to the previous command")
    commands are prepended before the real turn_on command, inside the
    lead-in window.

    EQUIVALENCE NOTE (verified below, not assumed): for the LEG's OWN
    FIRST in-span command specifically, `check_c1`'s reference
    (`start_pose8`, defined as "the state before `command_indices[0]`")
    and `find_turn_on_command`'s own reference (the state at THAT SAME
    command's own bracket t_lo) are the identical state by construction
    (`resolve_leg`'s `ev.leg_turn_on_state_index(evidence, idx)` IS
    `hi_state_index(idx[0]) - 1`, exactly what `find_turn_on_command`
    computes for `idx[0]`). A carry landing at `command_indices[0]`
    therefore fails BOTH checks together or neither -- there is no input
    at this position that trips the lead-in gate alone. This is proven,
    not sidestepped: the killing test below asserts the SPECIFIC
    "lead-in violation" reason STRING (which the A1 mutation removes
    outright, since that code path stops existing), not mere rc/verdict
    equality -- exactly the discriminator the assignment's own mutation
    rule requires ("the killing test must assert the specific reason").
    C1 co-firing is reported, not hidden, matching the precedent already
    established for mutant (q) in this same file and in
    test_goalfix_cmp_t8_mutants.py."""

    def _build(self, tmp_path):
        home = dict(R.HOME)
        sim = mf.FlightSim(home, pose_units="deg", restream_passes=1, settle_s=0.05)
        sim.fly(R.PLACE_ROUTE)
        setup_last_seq = sim.state_rows[-1]["seq"]
        for _ in range(25):
            sim._hold_ticks(1, dict(sim.pose))
        flight_first_seq = sim.state_rows[-1]["seq"]
        sim.pose = dict(R.REST)
        sim.fly(R.LIFT_TO_PRESENT)
        flight_last_seq = sim.state_rows[-1]["seq"]
        result = sim.result()
        cmds, rows = list(result.command_rows), list(result.state_rows)

        stray_target = list(cmds[0]["target_rad"])
        stray_target[0] += 0.05  # shoulder_pitch, well off the present-position match
        stray_cmd = dict(cmds[0], target_rad=stray_target)
        # Two BIT-IDENTICAL stray commands (a genuine carry of each
        # other), both before the real turn_on command.
        new_cmds = [dict(stray_cmd, seq=0), dict(stray_cmd, seq=1)]
        new_rows = [dict(rows[0])]
        new_rows.append(dict(rows[0], cmd_seq=0))
        new_rows.append(dict(rows[0], cmd_seq=1))
        for r in rows[1:]:
            r2 = dict(r)
            if r2["cmd_seq"] != -1:
                r2["cmd_seq"] += 2
            new_rows.append(r2)
        for c in cmds:
            c2 = dict(c)
            if c2.get("type") == "joint_command":
                c2["seq"] += 2
            new_cmds.append(c2)
        rows2 = [dict(r, seq=i, sim_step=i) for i, r in enumerate(new_rows)]
        cmds2 = new_cmds

        ev_dir = tmp_path / "ev"
        control_dir = tmp_path / "control"
        control_dir.mkdir()
        mf.write_evidence(ev_dir, rows2, cmds2)
        _write_sidecar(control_dir, CYCLE, "setup", 0, setup_last_seq + 2)
        _write_sidecar(control_dir, CYCLE, "flight", flight_first_seq + 2, flight_last_seq + 2)
        return ev_dir, control_dir

    def test_carry_before_turn_on_stops_with_lead_in_reason(self, tmp_path):
        ev_dir, control_dir = self._build(tmp_path)
        out = tmp_path / "out.json"
        rc = cyc._cli([
            "--ev-dir", str(ev_dir), "--control-dir", str(control_dir), "--cycle", CYCLE,
            "--rep", "1", "--arm", "B", "--out", str(out), "--validation-mode",
        ])
        payload = read_result(out)
        assert payload["verdict"] == cyc.VERDICT_STOP, payload
        assert any("lead-in violation" in r for r in payload["reasons"]), payload["reasons"]

    def test_the_two_strays_are_a_genuine_carry_of_each_other(self, tmp_path):
        """Confirms the fixture's own premise: the two prepended
        commands are bit-identical (a carry), not merely "some stray"."""
        ev_dir, control_dir = self._build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        jc_idx = [i for i, k in enumerate(evd.commands.kind) if k == "joint_command"]
        assert np.array_equal(evd.commands.target_rad[jc_idx[0]], evd.commands.target_rad[jc_idx[1]])

    def test_mutation_lead_in_gate_removed_would_drop_the_specific_reason(self, tmp_path):
        """Mutation (A1 lead-in gate removed, i.e. `evaluate_leg`
        bounds the lead-in window by `command_indices[0]`/the old
        `ev.leg_turn_on_state_index` boundary, as at pre-A1): the
        specific "lead-in violation" reason string never gets added at
        all -- that whole code path (holds.find_turn_on_command and its
        call site) does not exist under the mutation. Reproduced
        directly: the PRE-A1 boundary source
        (`ev.leg_turn_on_state_index(evd, leg.command_indices)`)
        resolves to the state just before the FIRST stray (mistaking it
        for turn_on), exactly as A1's own existing regression test in
        test_goalfix_cmp_a1_a2.py already demonstrates for a single
        stray -- here with two, the same old boundary is still just
        before the first one, never detecting either as a violation."""
        from tools.goalfix_cmp import holds
        ev_dir, control_dir = self._build(tmp_path)
        evd = ev.verify_and_load(ev_dir, "states.jsonl", "commands.jsonl")
        leg = cyc.resolve_leg(evd, control_dir / f"{CYCLE}-setup.link.json", "setup",
                               R.PLACE_ROUTE, R.CRITICAL_JOINTS)
        old_boundary_state = ev.leg_turn_on_state_index(evd, leg.command_indices)
        turn_on_match = holds.find_turn_on_command(evd, leg.command_indices)
        assert turn_on_match is not None
        assert turn_on_match.violation is True  # the shipped (fixed) code: 2 strays precede it
        assert turn_on_match.state_index != old_boundary_state, (
            "the fix must locate a DIFFERENT (later) state than the pre-A1 boundary")
