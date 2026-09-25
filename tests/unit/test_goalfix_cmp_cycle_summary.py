"""Unit tests for tools/goalfix_cmp/cycle.py and summary.py."""
import os
import sys
import time

import numpy as np
import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../scripts", "../../native_mujoco", "../..", "../fixtures/goalfix_cmp"):
    sys.path.insert(0, os.path.join(_HERE, _p))

import make_fixtures as mf  # noqa: E402
from tools.goalfix_cmp import evidence as ev  # noqa: E402
from tools.goalfix_cmp import cycle as cyc  # noqa: E402
from tools.goalfix_cmp import summary as summ  # noqa: E402


def full_pose(**kw) -> dict:
    base = {j: 0.0 for j in mf.R_JOINTS}
    base.update(kw)
    return base


START = full_pose()
SWING_1_POSE = full_pose(r_shoulder_pitch=-0.3)
HOVER_POSE = full_pose(r_shoulder_pitch=-0.70, r_shoulder_roll=-0.17,
                        r_elbow_pitch=-1.05, r_wrist_pitch=-0.26)
REST_SHUT_POSE = full_pose(r_shoulder_pitch=-0.70, r_shoulder_roll=-0.17,
                            r_elbow_pitch=-0.79, r_wrist_pitch=-0.17, r_wrist_roll=0.52)
REST_POSE = dict(REST_SHUT_POSE, r_gripper=-0.7)
PLACE_ROUTE = [
    mf.Waypoint("SWING_1", SWING_1_POSE, 1.0),
    mf.Waypoint("HOVER", HOVER_POSE, 1.2),
    mf.Waypoint("REST_SHUT", REST_SHUT_POSE, 1.0),
    mf.Waypoint("REST", REST_POSE, 1.0),
]


def _write_and_load(tmp_path, state_rows, command_rows):
    mf.write_evidence(tmp_path, state_rows, command_rows)
    return ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")


def _place_route_legspec(evidence):
    idx = list(range(len(evidence.commands)))
    idx = [i for i in idx if evidence.commands.kind[i] == "joint_command"]
    return cyc.LegSpec("setup", PLACE_ROUTE, guard=(), start_pose8=START,
                        command_indices=idx)


class TestEvaluateCycle:
    def test_clean_b_cycle_is_ok(self, tmp_path):
        sim = mf.FlightSim(START, restream_passes=1, settle_s=0.05)
        sim.fly(PLACE_ROUTE)
        result = sim.result()
        evidence = _write_and_load(tmp_path, result.state_rows, result.command_rows)
        legs = [_place_route_legspec(evidence)]
        cv, _ = cyc.evaluate_cycle("B_c1", "B", evidence, legs, skip_gates=True)
        assert cv.verdict == cyc.VERDICT_OK
        assert cv.rc() == 0

    def test_genuine_echo_in_segment_stops_b(self, tmp_path):
        sim = mf.FlightSim(START, seed=4)
        sim.fly(PLACE_ROUTE, echo_rate=0.44)
        result = sim.result()
        evidence = _write_and_load(tmp_path, result.state_rows, result.command_rows)
        legs = [_place_route_legspec(evidence)]
        cv, _ = cyc.evaluate_cycle("B_c2", "B", evidence, legs)
        assert cv.verdict == cyc.VERDICT_STOP
        assert cv.rc() == 2

    def test_a_cycle_with_no_echoes_is_inconclusive_baseline(self, tmp_path):
        sim = mf.FlightSim(START, restream_passes=1, settle_s=0.05)
        sim.fly(PLACE_ROUTE)
        result = sim.result()
        evidence = _write_and_load(tmp_path, result.state_rows, result.command_rows)
        legs = [_place_route_legspec(evidence)]
        cv, _ = cyc.evaluate_cycle("A_c1", "A", evidence, legs)
        assert cv.verdict == cyc.VERDICT_INCONCLUSIVE_BASELINE
        assert cv.rc() == 3

    def test_a_cycle_with_echoes_is_manipulated(self, tmp_path):
        sim = mf.FlightSim(START, seed=4)
        sim.fly(PLACE_ROUTE, echo_rate=0.44)
        result = sim.result()
        evidence = _write_and_load(tmp_path, result.state_rows, result.command_rows)
        legs = [_place_route_legspec(evidence)]
        cv, _ = cyc.evaluate_cycle("A_c2", "A", evidence, legs)
        assert cv.verdict == cyc.VERDICT_MANIPULATED
        assert cv.rc() == 0
        assert cv.genuine_echo_count > 0

    def test_evidence_incomplete_on_corrupt_integrity(self, tmp_path):
        sim = mf.FlightSim(START)
        sim.fly(PLACE_ROUTE)
        result = sim.result()
        _, _, sha_path = mf.write_evidence(tmp_path, result.state_rows, result.command_rows)
        mf.corrupt_sha256sums(sha_path, "commands.jsonl")

        def legs_factory(evidence):
            return [_place_route_legspec(evidence)]

        cv = cyc.evaluate_cycle_safe("B_c3", "B", tmp_path, "states.jsonl", "commands.jsonl",
                                      legs_factory)
        assert cv.verdict == cyc.VERDICT_EVIDENCE_INCOMPLETE
        assert cv.rc() == 3

    def test_pathcheck_failure_stops_b(self, tmp_path):
        sim = mf.FlightSim(START)
        sim.fly(PLACE_ROUTE)
        result = sim.result()
        # Corrupt a non-right-arm joint mid-stream: a clean C8 failure.
        rows = result.command_rows
        rows[20]["target_rad"][10] = 0.5
        evidence = _write_and_load(tmp_path, result.state_rows, rows)
        legs = [_place_route_legspec(evidence)]
        cv, _ = cyc.evaluate_cycle("B_c4", "B", evidence, legs)
        assert cv.verdict == cyc.VERDICT_STOP
        assert any("C8" in r for r in cv.reasons)

    def test_twelve_minute_cycle_runs_within_60s(self, tmp_path):
        long_route = [
            mf.Waypoint("SWING_1", SWING_1_POSE, 120.0),
            mf.Waypoint("HOVER", HOVER_POSE, 180.0),
            mf.Waypoint("REST_SHUT", REST_SHUT_POSE, 120.0),
            mf.Waypoint("REST", REST_POSE, 120.0),
        ]
        sim = mf.FlightSim(START, restream_passes=1, settle_s=0.3)
        sim.fly(long_route)
        result = sim.result()
        assert result.state_rows[-1]["sim_time_s"] >= 540.0  # >= 9 minutes, budget-representative
        evidence = _write_and_load(tmp_path, result.state_rows, result.command_rows)
        idx = [i for i in range(len(evidence.commands)) if evidence.commands.kind[i] == "joint_command"]
        legs = [cyc.LegSpec("setup", long_route, guard=(), start_pose8=START,
                             command_indices=idx)]
        t0 = time.monotonic()
        cv, _ = cyc.evaluate_cycle("B_long", "B", evidence, legs, skip_gates=True)
        elapsed = time.monotonic() - t0
        assert elapsed < 60.0
        assert cv.verdict == cyc.VERDICT_OK


class TestCheckpoint:
    def _cv(self, cycle, arm, verdict):
        return cyc.CycleVerdict(cycle, arm, verdict)

    def test_all_valid_no_hold(self):
        verdicts = [self._cv("A1", "A", cyc.VERDICT_MANIPULATED),
                    self._cv("B1", "B", cyc.VERDICT_OK),
                    self._cv("A2", "A", cyc.VERDICT_MANIPULATED),
                    self._cv("B2", "B", cyc.VERDICT_OK)]
        r = summ.checkpoint_after(verdicts)
        assert not r.any_stop and not r.hold_idle

    def test_both_a_inconclusive_holds_idle(self):
        verdicts = [self._cv("A1", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE),
                    self._cv("B1", "B", cyc.VERDICT_OK),
                    self._cv("A2", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE),
                    self._cv("B2", "B", cyc.VERDICT_OK)]
        r = summ.checkpoint_after(verdicts)
        assert r.hold_idle

    def test_b_stop_flagged(self):
        verdicts = [self._cv("A1", "A", cyc.VERDICT_MANIPULATED),
                    self._cv("B1", "B", cyc.VERDICT_STOP)]
        r = summ.checkpoint_after(verdicts)
        assert r.any_stop


class TestAggregate:
    def _cv(self, cycle, arm, verdict, reasons=()):
        return cyc.CycleVerdict(cycle, arm, verdict, list(reasons))

    def test_b_stop_gives_incomplete_no_verdict(self):
        verdicts = [self._cv("A1", "A", cyc.VERDICT_MANIPULATED),
                    self._cv("B1", "B", cyc.VERDICT_OK),
                    self._cv("B2", "B", cyc.VERDICT_STOP, ["C1 fail"])]
        r = summ.aggregate(verdicts, {})
        assert r.outcome == summ.OUTCOME_STOPPED
        assert r.rc() == 2

    def test_three_of_six_a_inconclusive_gives_inconclusive_comparison(self):
        verdicts = ([self._cv(f"A{i}", "A", cyc.VERDICT_MANIPULATED) for i in range(3)]
                    + [self._cv(f"A{i}", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE) for i in range(3, 6)]
                    + [self._cv(f"B{i}", "B", cyc.VERDICT_OK) for i in range(6)])
        metrics = {f"B{i}": summ.CycleMetrics(
            leg_start_shoulder_pitch_error_deg=0.5, delta_cmd_max_cm=0.01,
            wrist_ball_delta_cm=0.9) for i in range(6)}
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_INCONCLUSIVE_COMPARISON
        assert r.rc() == 3

    def test_supports_outcome_when_criteria_met(self):
        a_cycles = [self._cv(f"A{i}", "A", cyc.VERDICT_MANIPULATED) for i in range(4)]
        b_cycles = [self._cv(f"B{i}", "B", cyc.VERDICT_OK) for i in range(4)]
        metrics = {}
        for v in a_cycles:
            metrics[v.cycle] = summ.CycleMetrics(wrist_ball_delta_cm=2.0)
        for v in b_cycles:
            metrics[v.cycle] = summ.CycleMetrics(
                leg_start_shoulder_pitch_error_deg=0.5, delta_cmd_max_cm=0.01,
                wrist_ball_delta_cm=0.9)
        r = summ.aggregate(a_cycles + b_cycles, metrics, a_target_count=4, a_manipulated_min=4)
        assert r.outcome == summ.OUTCOME_SUPPORTS
        assert r.rc() == 0


"""Tripwire tests (T9) now live in test_goalfix_cmp_t9_tripwires.py -- the
old count_tripwires(single_blob) API this class exercised no longer
exists (T9 grounds each pattern in its OWN real log source, since
"lease acquisition"/"pause" matched nothing real and merging every
source into one blob can't tell a missing log apart from a clean one)."""
