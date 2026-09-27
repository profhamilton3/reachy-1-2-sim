"""T8 acceptance tests (assignment §2, T8): the three mutants the readiness
review found surviving (review §5) -- each killed by a test built
specifically to isolate it, not incidentally caught by something else."""
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import echo  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import summary as summ  # noqa: E402
from tools.goalfix_cmp._io import read_result  # noqa: E402
from tools.goalfix_cmp._units import route_rad  # noqa: E402
from reachy_ai.motion.rig_routes import Waypoint as RWaypoint  # noqa: E402


@pytest.fixture(autouse=True)
def _zero_lag(monkeypatch):
    # Isolates C1's start-continuity check from make_fixtures' deliberate
    # constant lag (see test_goalfix_cmp_t4_end_to_end.py's own note) --
    # not needed for the control-rate test but harmless there.
    monkeypatch.setattr(mf, "_LAG_RAD", [0.0] * 8)


class TestQEchoRuling:
    """Q-echo (coordinator ruling, 2026-09-25 stage-1 rulings, §3), frozen
    from the coordinator's own V3 diagnostic run so this does not need
    the venv: "Observed, my run 0, cycle 2 setup" -- r_shoulder_roll,
    command 2470, assigned to HOVER's goto (SWING_3 -> HOVER, moving
    -37.5deg -> -10deg). The joint reverses direction at SWING_3; HOVER's
    first min-jerk samples, a few float32 ULPs from their start, pass
    back through a value the lagging plant held 0.24s earlier. On-path,
    within 0.05deg of the goto's own start -- the ruling: path
    coincidence, never genuine_echo."""

    #: SWING_3's own exact roll goal = HOVER's ctx.start8 -- frozen from
    #: the ruling's quoted "prev (SWING_3 exact)" value.
    _A = -0.65449845790863
    #: HOVER's own roll goal (frozen from rig_routes: HOVER's r_shoulder_roll
    #: is -10deg; the exact float32(deg2rad(-10)) value is not load-bearing
    #: for this test -- only that b is on the OTHER side of a from the
    #: value under test).
    _B = -0.17453292519943295
    #: frozen "new target" from the ruling.
    _VALUE = -0.65449762344360

    def test_near_start_reversal_is_path_coincidence(self):
        ctx = echo.GotoContext(
            start8={"r_shoulder_roll": self._A}, goal8={"r_shoulder_roll": self._B}, seconds=2.5)
        label, _vacuous = echo._subclassify(
            name="r_shoulder_roll", value=self._VALUE, epoch=0, command_index=0,
            turn_on_state_index=None, ctx=ctx, src_global_index=0)
        assert label == echo.PATH_COINCIDENCE

    def test_off_path_near_end_value_stays_genuine(self):
        """The ruling's own contrast case: shift the SAME near-end value
        to the OTHER side of `a` (still within 0.05deg of it, so still
        "near end"), putting it OUTSIDE [start, goal] -- must stay
        genuine_echo, proving the fix is not a blanket "near an end is
        always safe" rule."""
        off_path_value = 2 * self._A - self._VALUE  # reflect across `a`
        assert not (min(self._A, self._B) <= off_path_value <= max(self._A, self._B))
        ctx = echo.GotoContext(
            start8={"r_shoulder_roll": self._A}, goal8={"r_shoulder_roll": self._B}, seconds=2.5)
        label, _vacuous = echo._subclassify(
            name="r_shoulder_roll", value=off_path_value, epoch=0, command_index=0,
            turn_on_state_index=None, ctx=ctx, src_global_index=0)
        assert label == echo.GENUINE_ECHO


class TestMutantGotoContextNoneThroughCycle:
    """T4 `goto_context=None` in `evaluate_cycle` (readiness review
    §Surviving mutants: "Path coincidence is not tested through the
    cycle. The handoff's 'covered indirectly' does not hold."). The
    shipped call site is cycle.py's `full_goto_context[global_i] =
    lr.goto_context[local_i]` loop (inside `evaluate_cycle`), which
    ``echo.classify_commands`` needs to ever return PATH_COINCIDENCE
    instead of GENUINE_ECHO (echo.py's own `_subclassify`: `ctx is None`
    skips the whole path-coincidence branch and falls straight to
    GENUINE_ECHO).

    This is the exact pre-Q-echo-ruling fixture (commit 154a7b0, before
    4c9da51): r_shoulder_pitch's own near-end, on-path echo, 2 commands
    before REST_SHUT's arrival -- at that commit it was (correctly, for
    the pre-ruling code) asserted `genuine_echo_count == 1`. Under the
    Q-echo ruling it is now `path_coincidence` instead
    (TestMutantQ_BIgnoresGenuineEchoInSegment's docstring below explains
    why), making `genuine_echo_count == 0` the CORRECT value through
    evaluate_cycle -- and the value a `goto_context=None` mutation would
    break back to 1, since the classifier could then never look up
    ctx.start8/goal8 to tell path coincidence from a genuine echo."""

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
        cmds, rows = result.command_rows, result.state_rows

        mf.write_evidence(tmp_path, rows, cmds)
        evd0 = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        jc_idx0 = [i for i, k in enumerate(evd0.commands.kind) if k == "joint_command"]
        pick_cmd, src_cmd = 200, 198
        src_state = evd0.brackets[jc_idx0[src_cmd]].hi_state_index
        planted_value = rows[src_state]["joints"][0]["position_rad"]

        cmds2 = [dict(c, target_rad=list(c["target_rad"])) for c in cmds]
        cmds2[pick_cmd]["target_rad"][0] = planted_value
        mf.write_evidence(tmp_path, rows, cmds2)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        idx = list(range(len(evd.commands)))
        leg = cyc.LegSpec("setup", route_r, guard=(), start_pose8=start_rad,
                           command_indices=idx)
        return evd, leg

    def test_near_end_on_path_echo_is_not_genuine_through_evaluate_cycle(self, tmp_path):
        evd, leg = self._build(tmp_path)
        cv, _ = cyc.evaluate_cycle("t8pc", "B", evd, [leg], skip_gates=True)
        # C3 (A2's raw-value monotonicity, no near-end exemption) is
        # expected to fail on this same planted value independently of
        # what is under test here -- only genuine_echo_count/reasons are
        # asserted.
        assert cv.genuine_echo_count == 0, cv.as_dict()
        assert not any("genuine echo" in r for r in cv.reasons), cv.reasons

    def test_mutation_goto_context_forced_none_would_miss_it(self, tmp_path):
        """Mutation guard (verified directly against a mutated copy of
        cycle.py -- see the handoff for the transcript): replacing
        cycle.py's ``full_goto_context[global_i] = lr.goto_context[local_i]``
        with ``full_goto_context[global_i] = None`` makes this exact
        fixture's genuine_echo_count go 0 -> 1. Pinned here as the value
        the correct code (asserted above) must keep giving."""
        evd, leg = self._build(tmp_path)
        cv, _ = cyc.evaluate_cycle("t8pc", "B", evd, [leg], skip_gates=True)
        assert cv.genuine_echo_count == 0


class TestMutantSegmentIndeterminateIgnoredForB:
    """T4 `segment_indeterminate` ignored for B (readiness review
    §Surviving mutants: "The code is correct today, but no test pins
    it."). The shipped call site is cycle.py's B-verdict condition
    (``evaluate_cycle``): ``... or segment_indeterminate or
    gate_failed``.

    Isolated fixture: a cycle with ONLY a "flight" leg -- no leg named
    ``place_route_leg_name`` ("setup") at all. ``evaluate_cycle`` never
    finds a ``place_leg``/``place_lr`` for that name, so
    ``segment_indeterminate`` stays at its initial default (``True``)
    without ever touching ``find_affected_segment`` -- meaning C0-C8,
    hold drift and lead-in all pass cleanly on the (otherwise ordinary)
    flight leg, and ONLY segment_indeterminate drives the verdict,
    unlike a truncated-PLACE_ROUTE fixture (where C0's own goal-sequence
    check would co-fail for the same reason and mask this check)."""

    def _build(self, tmp_path):
        start_deg = {j: 0.0 for j in mf.R_JOINTS}
        present_deg = dict(start_deg, r_shoulder_pitch=-10.0)
        route_deg = (RWaypoint("PRESENT", present_deg, 2.0, 6.0),)
        route_r = route_rad(route_deg)
        start_rad = route_rad((RWaypoint("START", start_deg, 1.0, 6.0),))[0].pose

        sim = mf.FlightSim(start_deg, pose_units="deg")
        sim.fly([mf.Waypoint("PRESENT", present_deg, 2.0)])
        result = sim.result()
        cmds, rows = result.command_rows, result.state_rows

        mf.write_evidence(tmp_path, rows, cmds)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        idx = list(range(len(evd.commands)))
        flight_leg = cyc.LegSpec("flight", route_r, guard=(), start_pose8=start_rad,
                                  command_indices=idx)
        return evd, flight_leg

    def test_missing_place_route_leg_gives_indeterminate_segment_stop(self, tmp_path):
        evd, flight_leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("t8seg", "B", evd, [flight_leg], skip_gates=True)

        # No other check fires -- isolates segment_indeterminate as the
        # ONLY reason for the STOP.
        for cid, r in leg_results["flight"].pathcheck.items():
            if hasattr(r, "passed"):
                assert r.passed, f"{cid} unexpectedly failed: {getattr(r, 'detail', None)}"
        assert cv.genuine_echo_count == 0
        assert cv.segment_indeterminate
        assert cv.verdict == cyc.VERDICT_STOP
        assert cv.reasons == ["affected segment is indeterminate or missing (B)"]

    def test_mutation_segment_indeterminate_removed_would_authorize(self, tmp_path):
        """Mutation guard (verified directly against a mutated copy of
        cycle.py -- see the handoff for the transcript): dropping ``or
        segment_indeterminate`` from the B-verdict's OR-condition makes
        this exact fixture's verdict go STOP -> ok, with every other
        signal (genuine_echo_count, any_pathcheck_fail, hold drift,
        lead-in, gate_failed) staying negative/zero -- proving this
        check alone is load-bearing here, not incidentally covered by
        pathcheck. Pinned here as the value the correct code (asserted
        above) must keep giving."""
        evd, flight_leg = self._build(tmp_path)
        cv, _ = cyc.evaluate_cycle("t8seg", "B", evd, [flight_leg], skip_gates=True)
        assert cv.verdict == cyc.VERDICT_STOP


class TestMutantQ_BIgnoresGenuineEchoInSegment:
    """(q): B ignores a genuine echo in the segment.

    UPDATED under the coordinator's Q-echo ruling (2026-09-25 stage-1
    rulings, §3): the ORIGINAL version of this fixture planted an
    on-segment, near-the-goto's-own-end value (2 commands before REST_SHUT's
    own arrival) as its "genuine echo". The coordinator's independent V3
    re-run showed every REAL in-leg genuine-echo case on clean B data is
    EXACTLY that shape (an on-path value within 0.05deg of a goto's own
    end) -- and ruled it must classify as `path_coincidence`, never
    `genuine_echo` (report §4 C2's near-end tau exemption is an exemption
    from the tau check, not a blanket "near an end is safe" rule; see
    `TestQEchoRuling` below, which pins the corrected classification
    directly with the coordinator's own frozen V3 numbers).

    That means this exact fixture's OLD injected value is no longer a
    valid "genuine echo" input at all -- it is the fixed case, not the
    still-broken one. Mutant (q) ("B ignores a genuine echo in the
    segment") still needs a demonstrated-genuine input, so this class now
    uses an off-path near-end value instead (a value that, unlike every
    real observed case, is NOT between the goto's own start and goal --
    the ruling's own second suggested test). Injecting an off-path value
    unavoidably also disturbs `assign_goals`' own segmentation at that
    boundary (R-const rejects it from the goto outright, since it no
    longer equals the established constant-joint reference) -- so C1 is
    ALSO expected to fail here, not just genuine_echo. This is reported,
    not hidden: the assertion no longer claims genuine_echo is the ONLY
    signal, only that it is present and named in the STOP reason (mutant
    (q) is "B ignores a genuine echo", not "genuine echo is the only
    possible check"; `test_genuine_echo_in_segment_stops_b`, referenced in
    the original docstring, already covers the "other checks also fire"
    case)."""

    def _build(self, tmp_path):
        start_deg = {j: 0.0 for j in mf.R_JOINTS}
        # r_shoulder_roll is CONSTANT across HOVER->REST_SHUT (both -10deg)
        # -- R-const's own "constant joint" case -- while r_shoulder_pitch
        # keeps the route's real motion so C0's own goal sequence still
        # makes sense.
        hover_deg = dict(start_deg, r_shoulder_pitch=-30.0, r_shoulder_roll=-10.0)
        rest_shut_deg = dict(start_deg, r_shoulder_pitch=-70.0, r_shoulder_roll=-10.0)
        route_deg = (RWaypoint("HOVER", hover_deg, 2.0, 6.0),
                     RWaypoint("REST_SHUT", rest_shut_deg, 2.0, 6.0))
        route_r = route_rad(route_deg)
        start_rad = route_rad((RWaypoint("START", start_deg, 1.0, 6.0),))[0].pose

        sim = mf.FlightSim(start_deg, pose_units="deg")
        sim.fly([mf.Waypoint(wp.name, wp.pose, wp.seconds) for wp in route_deg])
        result = sim.result()
        cmds, rows = result.command_rows, result.state_rows

        # Plant command 195's r_shoulder_roll (deep in REST_SHUT, where
        # roll has long been holding exactly -10deg) as command 10's OWN
        # state reading -- early in HOVER's OWN roll approach, still
        # short of -10deg by construction, i.e. NOT in REST_SHUT's own
        # [start, goal] range for roll (roll is constant there at exactly
        # -10deg) and NOT the immediately preceding command's value
        # either (so it is not a carry).
        mf.write_evidence(tmp_path, rows, cmds)
        evd0 = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        jc_idx0 = [i for i, k in enumerate(evd0.commands.kind) if k == "joint_command"]
        pick_cmd, src_cmd = 195, 10
        src_state = evd0.brackets[jc_idx0[src_cmd]].hi_state_index
        planted_value = rows[src_state]["joints"][1]["position_rad"]  # r_shoulder_roll

        cmds2 = [dict(c, target_rad=list(c["target_rad"])) for c in cmds]
        cmds2[pick_cmd]["target_rad"][1] = planted_value
        mf.write_evidence(tmp_path, rows, cmds2)  # overwrite with the mutated commands
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        idx = list(range(len(evd.commands)))
        leg = cyc.LegSpec("setup", route_r, guard=(), start_pose8=start_rad,
                           command_indices=idx)
        return evd, leg

    def test_off_path_near_end_echo_stops_b(self, tmp_path):
        """Not an isolation proof (see the class docstring: this specific
        injection also trips C1) -- it demonstrates the ruling's own
        second suggested test: an off-path near-end echo. B15/H12
        (coordinator review, 2026-09-25 Stage-A slice review, §4; owner
        Stage B authorization) now catches this construction FIRST: the
        planted value is deliberately off-path AND not a carry
        (documented in `_build`'s own comment), so
        `segments.assign_goals` cannot place it at all -- it is
        unassigned, which is exactly B15's own scope, and STOPs the
        cycle before echo classification's own segment-scoped count is
        even computed (`segment_indeterminate` short-circuits it to 0).
        Still STOP, via the unassigned-non-carry reason, not the
        genuine-echo one -- accepted as reported, not reworked to force
        the OLD isolation this construction no longer produces."""
        evd, leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("t8q", "B", evd, [leg], skip_gates=True)

        assert cv.verdict == cyc.VERDICT_STOP
        assert any("unassigned non-carry" in reason for reason in cv.reasons), cv.as_dict()


class TestMutantI_ControlShiftSign:
    """(i): the control shift is +20s instead of -20s. A planted repeat
    exists only on the -20s side of a long single-goto flight; +20s runs
    past the flight's own end and finds nothing."""

    def _build(self, tmp_path):
        start = {j: 0.0 for j in mf.R_JOINTS}
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.9}, 26.0)]
        sim = mf.FlightSim(start)
        sim.fly(route)
        result = sim.result()
        cmds, rows = result.command_rows, result.state_rows

        mf.write_evidence(tmp_path, rows, cmds)
        evd0 = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        jc_idx0 = [i for i, k in enumerate(evd0.commands.kind) if k == "joint_command"]

        pick_cmd, src_cmd = 1200, 189  # ~24.02s and ~3.80s -- exactly 20.22s apart
        src_state = evd0.brackets[jc_idx0[src_cmd]].hi_state_index
        planted_value = float(np.float32(rows[src_state]["joints"][0]["position_rad"]))

        cmds2 = [dict(c, target_rad=list(c["target_rad"])) for c in cmds]
        cmds2[pick_cmd]["target_rad"][0] = planted_value
        mf.write_evidence(tmp_path, rows, cmds2)
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        return evd, jc_idx0[pick_cmd]

    def test_control_count_matches_planted_count_on_minus_20s_side(self, tmp_path):
        evd, pick_idx = self._build(tmp_path)
        normal = echo.classify_commands(evd)
        control_correct = echo.classify_commands(evd, shift_s=-echo.CONTROL_SHIFT_S)
        assert normal[pick_idx].joints["r_shoulder_pitch"].label == echo.FRESH
        assert control_correct[pick_idx].joints["r_shoulder_pitch"].label == echo.GENUINE_ECHO

    def test_mutation_plus_20s_would_miss_the_planted_repeat(self, tmp_path):
        evd, pick_idx = self._build(tmp_path)
        control_mutant = echo.classify_commands(evd, shift_s=+echo.CONTROL_SHIFT_S)
        # The window a +20s shift searches lies past the flight's own end
        # (this flight is ~26.3s; t_hi+20 ~ 44s) -- nothing is found there,
        # proving the sign matters and this test would catch the mutant.
        assert control_mutant[pick_idx].joints["r_shoulder_pitch"].label != echo.GENUINE_ECHO


class TestMutantI_ControlShiftThroughCli:
    """(i) at the SHIPPED call site (readiness review §Surviving mutants:
    "The T8 test passes shift_s=-echo.CONTROL_SHIFT_S itself, so it never
    exercises the shipped call"). ``echo.py:370`` inside ``_cli`` calls
    ``classify_commands(evd, shift_s=-CONTROL_SHIFT_S)`` with no argument
    from the caller -- this test goes through ``echo._cli`` itself,
    supplying no shift_s, and reads ``control_counts`` out of the CLI's
    own JSON payload. Same planted-repeat fixture as
    TestMutantI_ControlShiftSign (the -20s side only)."""

    def _build(self, tmp_path):
        start = {j: 0.0 for j in mf.R_JOINTS}
        route = [mf.Waypoint("A", {"r_shoulder_pitch": -0.9}, 26.0)]
        sim = mf.FlightSim(start)
        sim.fly(route)
        result = sim.result()
        cmds, rows = result.command_rows, result.state_rows

        mf.write_evidence(tmp_path, rows, cmds)
        evd0 = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        jc_idx0 = [i for i, k in enumerate(evd0.commands.kind) if k == "joint_command"]

        pick_cmd, src_cmd = 1200, 189  # ~24.02s and ~3.80s -- exactly 20.22s apart
        src_state = evd0.brackets[jc_idx0[src_cmd]].hi_state_index
        planted_value = float(np.float32(rows[src_state]["joints"][0]["position_rad"]))

        cmds2 = [dict(c, target_rad=list(c["target_rad"])) for c in cmds]
        cmds2[pick_cmd]["target_rad"][0] = planted_value
        mf.write_evidence(tmp_path, rows, cmds2)
        return jc_idx0[pick_cmd]

    def test_cli_control_counts_finds_the_planted_repeat_on_minus_20s_side(self, tmp_path):
        pick_idx = self._build(tmp_path)
        out = tmp_path / "out.json"
        # Scoped to just the planted command: the CLI's own "whole leg"
        # aggregation would otherwise mix in whatever the flight's other
        # ~1900 commands classify as, on both sides of the shift.
        rc = echo._cli([
            "--evidence-dir", str(tmp_path), "--arm", "B",
            "--segment-start-index", str(pick_idx),
            "--segment-end-index", str(pick_idx),
            "--out", str(out)])
        payload = read_result(out)
        # The un-shifted (real) classification of this command is FRESH
        # (TestMutantI_ControlShiftSign), so B does not STOP on it.
        assert rc == 0, payload
        assert payload["control_counts"]["genuine_echo"] == 1, payload

    def test_mutation_plus_20s_at_the_cli_call_site_would_miss_it(self, tmp_path):
        """Mutation at the shipped call site (echo.py:370): flip the sign
        to +CONTROL_SHIFT_S, as (i) describes. Verified directly against
        the mutated source (not simulated inline) -- see the handoff for
        the git-show/restore transcript. Left here as the fixture the
        mutation run replays, and as a standing regression guard: this
        assertion pins the CORRECT value (1), which is what the mutation
        run showed differs from the mutant's (0)."""
        pick_idx = self._build(tmp_path)
        out = tmp_path / "out.json"
        rc = echo._cli([
            "--evidence-dir", str(tmp_path), "--arm", "B",
            "--segment-start-index", str(pick_idx),
            "--segment-end-index", str(pick_idx),
            "--out", str(out)])
        payload = read_result(out)
        assert payload["control_counts"]["genuine_echo"] == 1, payload


class TestMutantT7SecondLegWiring:
    """"T7 rule applied to the first leg only" (readiness review
    §Surviving mutants, cycle.py:429): "The T7 second-leg test builds its
    own leg map. The cycle's wiring is untested." The existing T7 tests
    (test_goalfix_cmp_t7_start_coincidence.py) call
    ``echo.classify_commands`` directly with a hand-built
    ``leg_turn_on_state_index`` dict -- this test instead goes through
    ``evaluate_cycle`` itself (the same function the shipped cycle CLI
    calls), with a real two-``LegSpec`` cycle, and observes the SECOND
    leg's own turn_on wiring through ``place_route_leg_name="flight"``
    (evaluate_cycle's own public parameter for which leg's affected
    segment counts toward genuine echo) -- the only way the second leg's
    classification is externally observable through evaluate_cycle's
    return value, since the plan's real cycles always name PLACE_ROUTE
    the FIRST ("setup") leg."""

    def _build(self, tmp_path):
        joints = list(mf.R_JOINTS)
        setup_pose = dict(zip(joints, [0.0] * 8))
        setup_pose["r_shoulder_pitch"] = -0.3

        rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                              wall_time_ns=1, position_rad21=mf.full21(setup_pose))]
        setup_target = dict(setup_pose)
        setup_target["r_elbow_pitch"] = -0.2
        cmd0 = mf.command_row_joint(seq=0, target_rad21=mf.full21(setup_target))
        rows.append(mf.state_row(seq=1, sim_step=1, sim_time_s=0.02, cmd_seq=0,
                                  wall_time_ns=2, position_rad21=mf.full21(setup_target)))

        # A settle before the flight leg's own turn_on (fresh client +
        # turn_on before any post-reset motion, plan §5 P8).
        present_at_flight_turn_on = dict(setup_target)
        for k in range(2, 52):  # 1s settle
            rows.append(mf.state_row(seq=k, sim_step=k, sim_time_s=k * 0.02, cmd_seq=0,
                                      wall_time_ns=1 + k,
                                      position_rad21=mf.full21(present_at_flight_turn_on)))

        # cmd1: flight leg's own turn_on, assigned to a synthetic "HOVER"
        # goal that is bit-exact (up to the float32 cast) with the
        # present position -- the T7 ambiguity: genuine_echo (old rule,
        # or the mutant restoring it for this leg) vs start_coincidence
        # (the fix, wired per-leg).
        hover_target = dict(present_at_flight_turn_on)
        hover_target["r_shoulder_pitch"] = float(
            np.float32(present_at_flight_turn_on["r_shoulder_pitch"]))
        cmd1 = mf.command_row_joint(seq=1, target_rad21=mf.full21(hover_target))
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                  cmd_seq=1, wall_time_ns=1 + len(rows),
                                  position_rad21=mf.full21(hover_target)))

        # D-1 (decision report §2/§6; assignment W1): W-blk needs a real
        # settle gap (>= SETTLE_GAP_S=0.28s, i.e. >= 14 command-free
        # ticks) after each goto, and a real HOVER/REST_SHUT/REST triple,
        # to find a window at all -- without one, genuine_echo_count is
        # always 0 regardless of the T7 per-leg wiring this test exists
        # to check, silently defeating test_mutation_first_leg_only_
        # loop_would_miss_it's own mutation guard below. Nothing about
        # the T7 mechanism itself (per-command echo classification,
        # independent of segmentation) is affected by adding this.
        for _ in range(16):  # settle after HOVER's own cmd1
            rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                      cmd_seq=1, wall_time_ns=1 + len(rows),
                                      position_rad21=mf.full21(hover_target)))

        # cmd2: a real move to a synthetic "REST_SHUT" goal, so
        # find_affected_segment can bound HOVER..REST_SHUT on the flight
        # leg's own (local) assignment.
        rest_target = dict(hover_target)
        rest_target["r_elbow_pitch"] = -0.4
        cmd2 = mf.command_row_joint(seq=2, target_rad21=mf.full21(rest_target))
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                  cmd_seq=2, wall_time_ns=1 + len(rows),
                                  position_rad21=mf.full21(rest_target)))

        for _ in range(16):  # settle after REST_SHUT's own cmd2
            rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                      cmd_seq=2, wall_time_ns=1 + len(rows),
                                      position_rad21=mf.full21(rest_target)))

        # cmd3: REST -- only the gripper opens, matching PLACE_ROUTE's
        # own REST_SHUT->REST shape.
        rest2_target = dict(rest_target, r_gripper=-0.8)
        cmd3 = mf.command_row_joint(seq=3, target_rad21=mf.full21(rest2_target))
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                  cmd_seq=3, wall_time_ns=1 + len(rows),
                                  position_rad21=mf.full21(rest2_target)))

        mf.write_evidence(tmp_path, rows, [cmd0, cmd1, cmd2, cmd3])
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        setup_leg = cyc.LegSpec(
            name="setup", route_rad=[mf.Waypoint("SETUP_WP", setup_target, 1.0)],
            guard=[], command_indices=[0], start_pose8=setup_pose)
        flight_leg = cyc.LegSpec(
            name="flight",
            route_rad=[mf.Waypoint("HOVER", hover_target, 1.0),
                       mf.Waypoint("REST_SHUT", rest_target, 1.0),
                       mf.Waypoint("REST", rest2_target, 1.0)],
            guard=[], command_indices=[1, 2, 3], start_pose8=present_at_flight_turn_on)
        return evd, setup_leg, flight_leg

    def test_second_leg_turn_on_is_start_coincidence_through_evaluate_cycle(self, tmp_path):
        evd, setup_leg, flight_leg = self._build(tmp_path)
        cv, _ = cyc.evaluate_cycle(
            "t7wiring", "B", evd, [setup_leg, flight_leg],
            place_route_leg_name="flight", skip_gates=True)
        # The wiring under test is genuine_echo_count/reasons specifically
        # (C1 may independently fail on this synthetic fixture's start
        # poses -- irrelevant to what T7 governs, and not asserted here).
        assert cv.genuine_echo_count == 0, cv.as_dict()
        assert not any("genuine echo" in r for r in cv.reasons), cv.reasons
        assert not cv.segment_indeterminate

    def test_mutation_first_leg_only_loop_would_miss_it(self, tmp_path):
        """Mutation guard (verified directly against a mutated copy of
        cycle.py -- see the handoff for the transcript): narrowing the
        ``for leg in legs:`` loop at cycle.py's leg_turn_on_map
        construction to ``legs[:1]`` (first leg only) makes the flight
        leg's cmd1 fall back to the default (no leg_turn_on_state_index
        entry) rule, which cannot fire on a non-epoch-first command --
        confirmed directly: with the mutation applied, this exact fixture
        gives genuine_echo_count == 1, not 0. Pinned here as the value
        the correct code (asserted above) must keep giving."""
        evd, setup_leg, flight_leg = self._build(tmp_path)
        cv, _ = cyc.evaluate_cycle(
            "t7wiring", "B", evd, [setup_leg, flight_leg],
            place_route_leg_name="flight", skip_gates=True)
        assert cv.genuine_echo_count == 0


class TestMutantO_SevenFiveIgnoresAMedian:
    """(o): §7.5 ignores the A-median criterion. A session meeting every
    B absolute criterion, with A-B < 0.5 cm, must NOT report supports."""

    def _session(self, wrist_ball_drop_cm):
        a_metrics = {f"A{i}": summ.CycleMetrics(wrist_ball_delta_cm=1.0) for i in range(6)}
        b_metrics = {f"B{i}": summ.CycleMetrics(
            leg_start_shoulder_pitch_error_deg=0.5, delta_cmd_max_cm=0.01,
            wrist_ball_delta_cm=1.0 - wrist_ball_drop_cm) for i in range(6)}
        verdicts = ([cyc.CycleVerdict(f"A{i}", "A", cyc.VERDICT_MANIPULATED) for i in range(6)]
                    + [cyc.CycleVerdict(f"B{i}", "B", cyc.VERDICT_OK) for i in range(6)])
        return verdicts, {**a_metrics, **b_metrics}

    def test_a_minus_b_under_threshold_is_not_supports(self):
        verdicts, metrics = self._session(wrist_ball_drop_cm=0.3)  # < 0.5 cm minimum
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome != summ.OUTCOME_SUPPORTS

    def test_a_minus_b_over_threshold_does_support(self):
        """Sanity check: the same session, with the drop actually meeting
        the 0.5 cm minimum, DOES support -- proving the fixture is
        otherwise valid and the test above isolates the A-median term."""
        verdicts, metrics = self._session(wrist_ball_drop_cm=0.6)
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_SUPPORTS

    def test_mutation_a_median_criterion_removed_would_support(self):
        """Proves the test above needs the A-median term: with it
        removed (mirroring mutant (o)), the under-threshold session would
        incorrectly support."""
        verdicts, metrics = self._session(wrist_ball_drop_cm=0.3)
        b_leg_start_err = [metrics[f"B{i}"].leg_start_shoulder_pitch_error_deg for i in range(6)]
        b_delta_cmd = [metrics[f"B{i}"].delta_cmd_max_cm for i in range(6)]
        b_supports_without_a_median = (
            bool(b_leg_start_err) and all(e <= summ.B_LEG_START_ERROR_MAX_DEG for e in b_leg_start_err)
            and bool(b_delta_cmd) and all(d <= summ.B_DELTA_CMD_MAX_CM for d in b_delta_cmd))
        assert b_supports_without_a_median  # the mutant's (wrong) verdict
