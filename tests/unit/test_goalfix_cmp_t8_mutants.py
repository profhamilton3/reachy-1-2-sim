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
        label = echo._subclassify(
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
        label = echo._subclassify(
            name="r_shoulder_roll", value=off_path_value, epoch=0, command_index=0,
            turn_on_state_index=None, ctx=ctx, src_global_index=0)
        assert label == echo.GENUINE_ECHO


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
        second suggested test: an off-path near-end echo is still
        genuine_echo, named in the STOP reason. Mutant (q)'s "removing
        genuine>0 would still STOP via pathcheck" caveat applies here the
        same way it already does to `test_genuine_echo_in_segment_stops_b`
        -- both are accepted as reported, not reworked to force isolation
        that a correct R-const/Q-echo no longer allows for an off-path
        injection at a goto boundary."""
        evd, leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("t8q", "B", evd, [leg], skip_gates=True)

        assert cv.genuine_echo_count >= 1, cv.as_dict()
        assert cv.verdict == cyc.VERDICT_STOP
        assert any("genuine echo" in reason for reason in cv.reasons)


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
