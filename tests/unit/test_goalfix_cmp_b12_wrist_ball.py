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
from tools.goalfix_cmp import window  # noqa: E402
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
    gripper).

    D-1 (decision report §2/§6; assignment W2 item 5): the wrist_ball
    segment/window now come from W-blk, which needs a real command-free
    settle gap (>= window.SETTLE_GAP_S) at each goto boundary to find its
    own blocks at all -- inserted here (state-only rows, no command)
    between HOVER's own arrival and REST_SHUT's approach, and again
    before REST. pre1/pre2 stay inside HOVER's own block (no settle gap
    before them), so the segment still starts at HOVER's own LAST
    command, not the leg's own first -- the same property the original,
    settle-gap-free fixture tested under the old affected-based method.
    """
    joints = list(mf.R_JOINTS)
    pre1 = _pose(joints, r_shoulder_pitch=np.radians(-5.0), r_gripper=np.radians(SHUT))
    pre2 = _pose(joints, r_shoulder_pitch=np.radians(-10.0), r_gripper=np.radians(SHUT))
    hover = _pose(joints, r_shoulder_pitch=np.radians(-30.0), r_gripper=np.radians(SHUT))
    rest_shut = _pose(joints, r_shoulder_pitch=np.radians(-70.0), r_gripper=np.radians(SHUT))
    rest = dict(rest_shut, r_gripper=np.radians(OPEN))

    start_pose = _pose(joints)
    rows = [mf.state_row(seq=0, sim_step=0, sim_time_s=0.0, cmd_seq=-1,
                          wall_time_ns=1, position_rad21=mf.full21(start_pose))]
    cmds = []

    def _emit(tgt):
        i = len(cmds)
        cmds.append(mf.command_row_joint(seq=i, target_rad21=mf.full21(tgt)))
        rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                  cmd_seq=i, wall_time_ns=2 + len(rows), position_rad21=mf.full21(tgt)))

    def _settle(pose, n=16):  # >= window.SETTLE_GAP_S (0.28s) of command-free ticks
        for _ in range(n):
            rows.append(mf.state_row(seq=len(rows), sim_step=len(rows), sim_time_s=len(rows) * 0.02,
                                      cmd_seq=len(cmds) - 1, wall_time_ns=2 + len(rows),
                                      position_rad21=mf.full21(pose)))

    _emit(pre1)
    _emit(pre2)
    _emit(hover)
    _settle(hover)
    _emit(rest_shut)
    for _ in range(hold_commands):
        _emit(dict(rest_shut))  # bit-exact carry of REST_SHUT, same block

    route = [mf.Waypoint("HOVER", hover, 1.0), mf.Waypoint("REST_SHUT", rest_shut, 1.0)]
    if with_rest:
        _settle(rest_shut)
        _emit(rest)
        route.append(mf.Waypoint("REST", rest, 1.0))

    mf.write_evidence(tmp_path, rows, cmds)
    evd = ev.verify_and_load(tmp_path, "states.jsonl", "commands.jsonl")
    leg = cyc.LegSpec("setup", route_rad=route, guard=(), start_pose8=start_pose,
                       command_indices=list(range(len(cmds))))
    return evd, leg


class TestWristBallDeltaRealComputation:
    def test_computed_when_scene_and_object_and_rest_all_present(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        assert affected is not None and not affected.indeterminate
        win = window.identify_window(evd, leg)
        assert win.valid, win.reasons
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"], win=win)
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
        win = window.identify_window(evd, leg)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"], win=win)
        assert m.cross_check_wrist_ball_cm is not None
        assert m.cross_check_wrist_ball_cm >= 0.0


class TestWristBallMissingInputsAreNullNeverGuessed:
    def test_no_board_object_ids_is_null_with_open_question(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        win = window.identify_window(evd, leg)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=None, win=win)
        assert m.wrist_ball_delta_cm is None
        assert any("board_object_ids" in q and "empty or absent" in q for q in m.open_questions)

    def test_ambiguous_board_object_ids_is_null_with_open_question(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        win = window.identify_window(evd, leg)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1", "box_2"], win=win)
        assert m.wrist_ball_delta_cm is None
        assert any("2 entries" in q for q in m.open_questions)

    def test_indeterminate_segment_is_null_with_open_question(self, tmp_path):
        """D-1 (decision report §2/§6; assignment W2 item 5) supersedes
        this test's original mechanism: the wrist_ball block's nullity is
        no longer driven by `affected.indeterminate` at all (forcing it
        True here would now have zero effect on `m`) -- it comes
        entirely from `win.valid`. Constructs an explicitly invalid
        window directly, the same contract (null + open_question) now
        exercised through its real trigger."""
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        bad_win = window.WindowResult(valid=False, reasons=["wblk:block_count=1"])
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"], win=bad_win)
        assert m.wrist_ball_delta_cm is None
        assert any("indeterminate or missing" in q for q in m.open_questions)

    def test_no_rest_goto_after_segment_is_null_with_open_question(self, tmp_path):
        """D-1 (assignment W1 "missing_waypoint"): a route with no REST
        waypoint after REST_SHUT at all (this leg's own route only has
        HOVER/REST_SHUT) -- W-blk itself cannot even locate a window
        (`wblk:missing_waypoint=REST`), superseding the old affected-
        based "no REST goto found after the segment" reason text (the
        segment concept itself no longer exists independently of the
        window)."""
        evd, leg = _build(tmp_path, with_rest=False)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        assert affected is not None and not affected.indeterminate
        win = window.identify_window(evd, leg)
        assert not win.valid
        assert any("missing_waypoint=REST" in r for r in win.reasons), win.reasons
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, _box_scene(), board_object_ids=["box_1"], win=win)
        assert m.wrist_ball_delta_cm is None
        assert any("missing_waypoint=REST" in q for q in m.open_questions)

    def test_no_scene_is_null_exactly_as_before(self, tmp_path):
        evd, leg = _build(tmp_path)
        lr = cyc.evaluate_leg(evd, leg)
        affected = cyc.seg.find_affected_segment(lr.assignment, leg.route_rad)
        win = window.identify_window(evd, leg)
        m = cyc.compute_place_route_metrics(
            evd, leg, lr, affected, None, board_object_ids=["box_1"], win=win)
        assert m.wrist_ball_delta_cm is None

    # D-1 (decision report §2/§6; assignment W2 item 5): the mutation
    # guard formerly here (`test_mutation_rest_search_not_bounded_by_
    # segment_end_would_pick_earlier_rest`) pinned the OLD REST-assigned
    # search's own `i > affected.end_command_index` filter in
    # `compute_place_route_metrics`. That search no longer exists at all
    # -- the wrist_ball segment/window come from `window.identify_window`
    # unconditionally now, with no goal-assignment-based REST search to
    # guard. Deleted rather than reworked: there is no D-1 equivalent
    # mechanism for it to pin (window.py's own `first_rest`/block
    # partitioning is exercised directly by the W-blk acceptance tests,
    # not by a REST-search filter).


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
