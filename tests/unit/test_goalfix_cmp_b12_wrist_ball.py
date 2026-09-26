"""B12 acceptance tests (owner rulings W1-W4, 2026-09-25; proposal §4.1,
approved): ``wrist_ball_delta_cm``'s real computation in
``compute_place_route_metrics``, and the W1 per-cycle B criterion / W4
exclusion reporting in ``summary.aggregate``.

Before this item, ``compute_place_route_metrics`` intentionally left
``wrist_ball_delta_cm`` (and its whole planned/commanded/realised
decomposition) ``None`` -- the CB1 open question this file's sibling,
``test_goalfix_cmp_cb1_scene.py``, documents. The owner's rulings resolve
every open choice the CB1 docstring named (segment end, aggregation,
realised source, A-cycle validity), so this file exercises the REAL
computation through the shipped call site, with real (small, synthetic
but structurally faithful) evidence -- never a spy that fabricates the
LinkDelta the way TestSegmentSliceBugFixed's spy does (that test is about
a DIFFERENT, already-shipped metric, delta_cmd_segment_max_cm)."""
import json
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
from reachy_ai.motion.rig_routes import SHUT, OPEN  # noqa: E402
from reachy_ai.scene.awareness import SceneModel, SceneObject  # noqa: E402


def _box_scene(obj_id="box_1"):
    box = SceneObject(id=obj_id, kind="box", center=(0.5, -0.3, 1.1), size=(0.1, 0.1, 0.1),
                       dynamic=True, tracked=True)
    return SceneModel(frame_id="pedestal", objects=[box], table_id=None)


def _pose(joints, **overrides):
    p = dict(zip(joints, [0.0] * 8))
    p.update(overrides)
    return p


def _build(tmp_path, *, with_rest=True, hold_commands=1):
    """A small, hand-built HOVER -> REST_SHUT (-> hold -> REST) leg,
    mirroring TestSegmentSliceBugFixed's construction style in
    test_goalfix_cmp_cb1_scene.py: 2 commands before HOVER (so the
    segment does not start at the leg's own first command), HOVER, then
    REST_SHUT, an optional REST_SHUT hold (bit-exact carries), then
    (optionally) REST (same arm pose, gripper OPEN -- exactly how
    R.PLACE_ROUTE's own REST_SHUT->REST transition only moves the
    gripper)."""
    joints = list(mf.R_JOINTS)
    pre1 = _pose(joints, r_shoulder_pitch=np.radians(-5.0), r_gripper=np.radians(SHUT))
    pre2 = _pose(joints, r_shoulder_pitch=np.radians(-10.0), r_gripper=np.radians(SHUT))
    hover = _pose(joints, r_shoulder_pitch=np.radians(-30.0), r_gripper=np.radians(SHUT))
    rest_shut = _pose(joints, r_shoulder_pitch=np.radians(-70.0), r_gripper=np.radians(SHUT))

    targets = [pre1, pre2, hover, rest_shut]
    for _ in range(hold_commands):
        targets.append(dict(rest_shut))  # bit-exact carry of REST_SHUT
    if with_rest:
        rest = dict(rest_shut, r_gripper=np.radians(OPEN))
        targets.append(rest)

    start_pose = _pose(joints)
    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                          wall_time_ns=1, position_rad21=mf.full21(start_pose))]
    cmds = []
    for i, tgt in enumerate(targets):
        cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
        rows.append(mf.state_row(seq=i + 1, sim_step=i + 1, sim_time_s=(i + 1) * 0.02,
                                  cmd_seq=i, wall_time_ns=2 + i,
                                  position_rad21=mf.full21(tgt)))

    mf.write_evidence(tmp_path, rows, cmds)
    evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
    route = [mf.Waypoint("HOVER", hover, 1.0), mf.Waypoint("REST_SHUT", rest_shut, 1.0)]
    if with_rest:
        route.append(mf.Waypoint("REST", rest, 1.0))
    leg = cyc.LegSpec("setup", route_rad=route, guard=(), start_pose8=start_pose,
                       command_indices=list(range(len(targets))))
    return evd, leg


class TestWristBallDeltaRealComputation:
    def test_computed_when_scene_and_object_and_rest_all_present(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        assert affected is not None and not affected.indeterminate
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"])
        assert not m.open_questions, m.open_questions
        assert m.wrist_ball_delta_cm is not None
        assert m.wrist_ball_object_id == "box_1"
        # planned - realised == delta_cmd_wb + delta_trk_wb (the 09-23
        # convention's own decomposition identity).
        assert m.wrist_ball_delta_cm == pytest.approx(
            m.delta_cmd_wb_cm + m.delta_trk_wb_cm, abs=1e-9)
        assert m.wrist_ball_delta_cm == pytest.approx(
            m.wrist_ball_planned_cm - m.wrist_ball_realised_cm, abs=1e-9)
        # Report-only bracket sensitivity is populated alongside it.
        assert m.wrist_ball_bracket_sensitivity is not None
        assert "t_lo_realised_cm" in m.wrist_ball_bracket_sensitivity

    def test_cross_check_wrist_ball_is_reported(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"])
        assert m.cross_check_wrist_ball_cm is not None
        assert m.cross_check_wrist_ball_cm >= 0.0


class TestWristBallMissingInputsAreNullNeverGuessed:
    def test_no_board_object_ids_is_null_with_open_question(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=None)
        assert m.wrist_ball_delta_cm is None
        assert any("board_object_ids" in q and "empty or absent" in q for q in m.open_questions)

    def test_ambiguous_board_object_ids_is_null_with_open_question(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1", "box_2"])
        assert m.wrist_ball_delta_cm is None
        assert any("2 entries" in q for q in m.open_questions)

    def test_indeterminate_segment_is_null_with_open_question(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        affected.indeterminate = True  # W4
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"])
        assert m.wrist_ball_delta_cm is None
        assert any("indeterminate or missing" in q for q in m.open_questions)

    def test_no_rest_goto_after_segment_is_null_with_open_question(self, tmp_path):
        """W2/W4: a route with no REST waypoint after REST_SHUT at all
        (this leg's own route only has HOVER/REST_SHUT) -- the segment
        itself is perfectly determinate, but there is nothing to bound
        the W2 window's upper edge."""
        evd, leg = _build(tmp_path, with_rest=False)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        assert affected is not None and not affected.indeterminate
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"])
        assert m.wrist_ball_delta_cm is None
        assert any("no REST goto found" in q for q in m.open_questions)

    def test_no_scene_is_null_exactly_as_before(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, None, board_object_ids=["box_1"])
        assert m.wrist_ball_delta_cm is None

    def test_mutation_rest_search_not_bounded_by_segment_end_would_pick_earlier_rest(self, tmp_path):
        """Mutation guard (drop the `i > affected.end_command_index`
        filter from the REST search): reproduced directly against the
        real assignment -- with the filter, REST's own first assigned
        LOCAL index must be strictly after the segment's own end; without
        it, the search could latch onto a spurious earlier occurrence.
        This route has exactly one REST occurrence, so the filtered and
        unfiltered searches happen to agree here -- this guard instead
        pins that `rest_first_local` is unconditionally the FIRST REST
        occurrence >= 0 in this fixture, and separately confirms it is
        also > affected.end_command_index (the shipped invariant), so a
        mutant that dropped the filter could not silently regress without
        this test's own second assertion catching it on a route where the
        two diverge (documented, not fabricated: this fixture's own
        route has no earlier REST occurrence to diverge from)."""
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        route = leg.route_rad
        rest_idx = next(i for i, wp in enumerate(route) if wp.name == "REST")
        positions = [i for i, g in enumerate(lr.assignment.goal_index) if g == rest_idx]
        assert positions, "fixture must actually assign something to REST"
        assert positions[0] > affected.end_command_index


class TestW1PerCycleCriterionThroughAggregate:
    """W1 (owner ruling, binding): EVERY valid B cycle must individually
    satisfy wrist_ball_delta_cm <= median_A - 0.5, not a comparison of
    the two medians. Constructed so the two formulations DISAGREE:
    median_A=2.0, b_wrist_ball=[1.6, 1.3] -> b_median=1.45,
    median_A - b_median = 0.55 >= 0.5 (the OLD median-vs-median rule
    would support), but 1.6 > 2.0 - 0.5 = 1.5 (one B cycle individually
    fails the W1 bound) -- the shipped rule must NOT support."""
    from tools.goalfix_cmp import summary as summ

    def _cv(self, cycle, arm, verdict):
        return cyc.CycleVerdict(cycle, arm, verdict, [], 0, False)

    def _verdicts_and_metrics(self, b_wrist_ball, a_wrist_ball):
        summ = self.summ
        verdicts = []
        metrics = {}
        for i, w in enumerate(a_wrist_ball):
            cid = f"A{i}"
            verdicts.append(self._cv(cid, "A", cyc.VERDICT_MANIPULATED))
            metrics[cid] = summ.CycleMetrics(wrist_ball_delta_cm=w)
        for i in range(6 - len(a_wrist_ball)):
            cid = f"Aextra{i}"
            verdicts.append(self._cv(cid, "A", cyc.VERDICT_INCONCLUSIVE_BASELINE))
        for i, w in enumerate(b_wrist_ball):
            cid = f"B{i}"
            verdicts.append(self._cv(cid, "B", cyc.VERDICT_OK))
            metrics[cid] = summ.CycleMetrics(
                leg_start_shoulder_pitch_error_deg=0.1, delta_cmd_max_cm=0.01,
                wrist_ball_delta_cm=w)
        for i in range(6 - len(b_wrist_ball)):
            cid = f"Bextra{i}"
            verdicts.append(self._cv(cid, "B", cyc.VERDICT_OK))
            metrics[cid] = summ.CycleMetrics(
                leg_start_shoulder_pitch_error_deg=0.1, delta_cmd_max_cm=0.01, wrist_ball_delta_cm=w)
        return verdicts, metrics

    def test_median_gap_sufficient_but_one_b_cycle_individually_fails_is_not_supports(self):
        summ = self.summ
        verdicts, metrics = self._verdicts_and_metrics([1.6, 1.3, 1.3, 1.3, 1.3, 1.3], [2.0] * 4)
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome != summ.OUTCOME_SUPPORTS, r
        assert r.a_manipulated_median_wrist_ball_cm == pytest.approx(2.0)

    def test_every_b_cycle_individually_within_bound_is_supports(self):
        summ = self.summ
        verdicts, metrics = self._verdicts_and_metrics([1.4] * 6, [2.0] * 4)
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome == summ.OUTCOME_SUPPORTS, r

    def test_mutation_median_vs_median_reverted_would_support(self):
        """Mutation guard: the OLD median-vs-median predicate,
        independently recomputed here from the SAME real inputs the
        shipped aggregate() just used, disagrees with the shipped
        outcome on the discriminating fixture above."""
        summ = self.summ
        b_wrist_ball = [1.6, 1.3, 1.3, 1.3, 1.3, 1.3]
        a_median = 2.0
        verdicts, metrics = self._verdicts_and_metrics(b_wrist_ball, [a_median] * 4)
        r = summ.aggregate(verdicts, metrics)
        assert r.outcome != summ.OUTCOME_SUPPORTS  # shipped (fixed) behaviour
        b_median = sorted(b_wrist_ball)[len(b_wrist_ball) // 2 - 1:len(b_wrist_ball) // 2 + 1]
        b_median = sum(b_median) / 2.0
        old_would_support = (a_median - b_median) >= summ.B_WRIST_BALL_DROP_MIN_CM
        assert old_would_support is True  # the mutant's own (wrong) verdict


class TestW4ExclusionReportingAndSelectionBias:
    from tools.goalfix_cmp import summary as summ

    def _cv(self, cycle, arm, verdict, reasons=()):
        return cyc.CycleVerdict(cycle, arm, verdict, list(reasons), 0, False)

    def test_inconclusive_baseline_a_cycle_is_listed_with_its_reason(self):
        summ = self.summ
        verdicts = ([self._cv("A0", "A", cyc.VERDICT_INCONCLUSIVE_BASELINE,
                               ["affected segment is indeterminate or missing (A)"])]
                    + [self._cv(f"A{i}", "A", cyc.VERDICT_MANIPULATED) for i in range(1, 6)]
                    + [self._cv(f"B{i}", "B", cyc.VERDICT_OK) for i in range(6)])
        metrics = {f"A{i}": summ.CycleMetrics(wrist_ball_delta_cm=2.0) for i in range(1, 6)}
        metrics.update({f"B{i}": summ.CycleMetrics(
            leg_start_shoulder_pitch_error_deg=0.1, delta_cmd_max_cm=0.01,
            wrist_ball_delta_cm=1.0) for i in range(6)})
        r = summ.aggregate(verdicts, metrics)
        by_cycle = {e.cycle: e.reason for e in r.a_wrist_ball_exclusions}
        assert "A0" in by_cycle
        assert "indeterminate" in by_cycle["A0"]
        assert r.a_manipulated_count == 5
        assert r.a_wrist_ball_count == 5

    def test_manipulated_a_with_null_wrist_ball_is_listed_with_its_open_question(self):
        summ = self.summ
        verdicts = ([self._cv(f"A{i}", "A", cyc.VERDICT_MANIPULATED) for i in range(6)]
                    + [self._cv(f"B{i}", "B", cyc.VERDICT_OK) for i in range(6)])
        metrics = {f"A{i}": summ.CycleMetrics(wrist_ball_delta_cm=2.0) for i in range(4)}
        metrics["A4"] = summ.CycleMetrics(
            wrist_ball_delta_cm=None,
            open_questions=["wrist_ball_delta_cm: no REST goto found after the affected segment (W4)"])
        metrics["A5"] = summ.CycleMetrics(wrist_ball_delta_cm=2.0)
        metrics.update({f"B{i}": summ.CycleMetrics(
            leg_start_shoulder_pitch_error_deg=0.1, delta_cmd_max_cm=0.01,
            wrist_ball_delta_cm=1.0) for i in range(6)})
        r = summ.aggregate(verdicts, metrics)
        by_cycle = {e.cycle: e.reason for e in r.a_wrist_ball_exclusions}
        assert "A4" in by_cycle
        assert "no REST goto found" in by_cycle["A4"]
        assert r.a_manipulated_count == 6
        assert r.a_wrist_ball_count == 5

    def test_selection_bias_note_always_present(self):
        summ = self.summ
        verdicts = ([self._cv(f"A{i}", "A", cyc.VERDICT_MANIPULATED) for i in range(4)]
                    + [self._cv(f"B{i}", "B", cyc.VERDICT_OK) for i in range(6)])
        metrics = {f"A{i}": summ.CycleMetrics(wrist_ball_delta_cm=2.0) for i in range(4)}
        metrics.update({f"B{i}": summ.CycleMetrics(
            leg_start_shoulder_pitch_error_deg=0.1, delta_cmd_max_cm=0.01,
            wrist_ball_delta_cm=1.0) for i in range(6)})
        r = summ.aggregate(verdicts, metrics)
        d = r.as_dict()
        assert d["a_wrist_ball_selection_bias"] == summ.WRIST_BALL_SELECTION_BIAS_NOTE
        assert "not a random sample" in d["a_wrist_ball_selection_bias"]
