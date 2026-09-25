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


class TestMutantQ_BIgnoresPathConsistentSegmentEcho:
    """(q): B ignores a genuine echo in the segment. The existing
    test_genuine_echo_in_segment_stops_b only passes because its echo
    ALSO trips C1/C3 -- removing the genuine>0 check there would still
    STOP via pathcheck. This fixture's single injected echo, chosen 2
    commands before the very end of the REST_SHUT goto (where minimum-
    jerk's own near-zero velocity makes the "backward" step small enough
    that C0-C8 all pass), is path-consistent: tau monotone, on segment,
    every check green -- ONLY the genuine-echo count can catch it."""

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

        # Locate the two commands' own reporting states from a clean load,
        # then plant command 200's target as command 198's OWN state
        # reading (2 ticks earlier, both deep in REST_SHUT's final,
        # near-zero-velocity tail).
        mf.write_evidence(tmp_path, rows, cmds)
        evd0 = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
        jc_idx0 = [i for i, k in enumerate(evd0.commands.kind) if k == "joint_command"]
        pick_cmd, src_cmd = 200, 198
        src_state = evd0.brackets[jc_idx0[src_cmd]].hi_state_index
        planted_value = rows[src_state]["joints"][0]["position_rad"]

        cmds2 = [dict(c, target_rad=list(c["target_rad"])) for c in cmds]
        cmds2[pick_cmd]["target_rad"][0] = planted_value
        mf.write_evidence(tmp_path, rows, cmds2)  # overwrite with the mutated commands
        evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")

        idx = list(range(len(evd.commands)))
        leg = cyc.LegSpec("setup", route_r, guard=(), start_pose8=start_rad,
                           command_indices=idx)
        return evd, leg

    def test_path_consistent_segment_echo_stops_b(self, tmp_path):
        evd, leg = self._build(tmp_path)
        cv, leg_results = cyc.evaluate_cycle("t8q", "B", evd, [leg], skip_gates=True)

        assert cv.genuine_echo_count == 1, cv.as_dict()
        for cid, r in leg_results["setup"].pathcheck.items():
            if hasattr(r, "passed"):
                assert r.passed, f"{cid} unexpectedly failed: {r.detail}"
        assert cv.verdict == cyc.VERDICT_STOP
        assert any("genuine echo" in reason for reason in cv.reasons)

    def test_mutation_genuine_check_removed_would_miss_it(self, tmp_path):
        """Proves the test above actually needs the genuine>0 check:
        with it patched out (mirroring mutant (q)), the same fixture's
        pathcheck-only verdict would be OK, not STOP."""
        evd, leg = self._build(tmp_path)
        leg_results = {"setup": cyc.evaluate_leg(evd, leg)}
        any_pathcheck_fail = any(
            not r.passed for r in leg_results["setup"].pathcheck.values()
            if hasattr(r, "passed"))
        assert not any_pathcheck_fail  # pathcheck alone sees nothing wrong


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
