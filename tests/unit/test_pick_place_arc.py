"""Issue #56: the pick/place arc is planned whole, checked against the whole
arm, and ends by a checked return -- not by `raise_to_side`.

Offline.  Inverse kinematics is the host mirror of the simulator's fake-server
IK (`host_ik_arm.py`, drift-pinned); the "robot" records every command it is
given, and the primitives that would drive hardware are replaced by recorders.
Nothing here starts a simulator or commands an arm.

NOT PROVEN HERE: grasp reliability, physical safety, or how the physics arm
tracks the return (residual R2) -- only what the PLANNER and the sequence of
commands do.
"""

import os
import sys

import pytest

pytest.importorskip("scipy")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arc_test_fakes import (  # noqa: E402
    DEMO_SCENE, RecordingRobot, Spies, board)
from reachy_ai.motion import primitives as P  # noqa: E402
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion import transit  # noqa: E402
from reachy_ai.motion.kinematics import (  # noqa: E402
    ARM_PATH_MAX_SAMPLE_STEP_DEG,
    WHOLE_ARM_MARGIN_M,
    ArmClearanceError,
    CartesianPlanner,
    R_ARM_JOINTS,
    link_frames,
)
from reachy_ai.scene.awareness import SceneModel  # noqa: E402
from reachy_ai.tasks import pick_place_live as L  # noqa: E402

SEGMENT_NAMES = [
    "swing in over table", "descend to hover", "descend to grasp",
    "lift to clear height", "carry over table", "descend to place",
    "retract", "return to the raised pose",
]


@pytest.fixture(scope="module")
def planned():
    """One real plan (about 4 s), shared: red_cube r2c2 -> r2c3 alone on the
    lab board, with every check call recorded."""
    scene, cells = board({"red_cube": "cell_r2c2"})
    robot = RecordingRobot()
    planner = CartesianPlanner(robot.r_arm, scene=scene)
    recorded = []
    mp = pytest.MonkeyPatch()
    original = CartesianPlanner.check_arm_path

    def spy(self, joints_seq, *, segment, gripper_deg, target=None):
        recorded.append((segment, gripper_deg, target))
        return original(self, joints_seq, segment=segment,
                        gripper_deg=gripper_deg, target=target)

    mp.setattr(CartesianPlanner, "check_arm_path", spy)
    try:
        plan = L.plan_pick_and_place(planner, scene, "red_cube",
                                     cells["cell_r2c3"][:2])
    finally:
        mp.undo()
    return {"plan": plan, "scene": scene, "cells": cells, "planner": planner,
            "robot": robot, "recorded": recorded}


def _fresh_robot_planner(plan_scene):
    robot = RecordingRobot()
    return robot, CartesianPlanner(robot.r_arm, scene=plan_scene)


# ── planning ──────────────────────────────────────────────────────────────────

class TestArcPlanning:
    def test_a_single_object_move_passes_with_eight_segments_ending_at_present(
            self, planned):
        plan = planned["plan"]
        assert [s.name for s in plan.segments] == SEGMENT_NAMES
        assert plan.segments[-1].name == "return to the raised pose"
        assert plan.end == L.present_joints()          # exactly, not approximately
        assert plan.segments[-1].traj[-1] == L.present_joints()
        assert plan.start == L.present_joints()
        assert plan.segments[-1].cart is None          # a joint-space line

    def test_segment_rates_and_actions_match_the_old_arc(self, planned):
        segs = {s.name: s for s in planned["plan"].segments}
        assert [segs[n].rate_hz for n in SEGMENT_NAMES] == [
            transit.CARRY_HZ, transit.STEP_HZ, transit.STEP_HZ,
            transit.CARRY_HZ, transit.CARRY_HZ, transit.STEP_HZ,
            transit.CARRY_HZ, transit.CARRY_HZ]
        assert [len(segs[n].traj) for n in SEGMENT_NAMES[:7]] == [
            40, 25, 15, 30, 40, 30, 25]
        assert segs["lift to clear height"].before == "close"
        assert segs["retract"].before == "open"
        assert [segs[n].attach for n in SEGMENT_NAMES] == [
            False, False, False, True, True, True, False, False]

    def test_the_planning_snapshot_is_a_copy_not_the_shared_scene(self, planned):
        plan, scene = planned["plan"], planned["scene"]
        assert plan.scene is not scene
        scene.update_poses({"red_cube": (0.0, 0.0, 0.0)})
        try:
            assert plan.scene.get("red_cube").center != (0.0, 0.0, 0.0)
        finally:
            c = planned["cells"]["cell_r2c2"]
            obj = scene.get("red_cube")
            scene.update_poses({"red_cube": (c[0], c[1], scene.table_surface_z
                                             + obj.half_height)})

    def test_hand_contact_is_true_exactly_for_segments_two_to_seven(self, planned):
        calls = [c for c in planned["recorded"] if c[0] in SEGMENT_NAMES]
        assert [c[0] for c in calls] == SEGMENT_NAMES
        assert [c[2].hand_contact for c in calls] == [
            False, True, True, True, True, True, True, False]
        assert [c[2].carried for c in calls] == [
            False, False, False, True, True, True, False, False]
        place = planned["plan"].place
        assert calls[6][2].center == place and calls[7][2].center == place
        for c in calls:
            assert c[1] == R.OPEN            # every check uses the OPEN aperture
            assert c[2].object_id == "red_cube"

    def test_planning_sends_no_command_of_any_kind(self, planned, monkeypatch):
        # (the fixture already planned against a recording robot)
        assert planned["robot"].r_arm.writes == []
        assert planned["robot"].turn_on_calls == []

    def test_the_demo_boards_red_cube_is_refused_naming_the_cause_with_zero_motion(
            self, monkeypatch):
        spies = Spies(monkeypatch)
        scene = SceneModel.from_yaml(DEMO_SCENE)
        robot = RecordingRobot()
        planner = CartesianPlanner(robot.r_arm, scene=scene)
        with pytest.raises(ArmClearanceError) as info:
            L.plan_pick_and_place(planner, scene, "red_cube", (0.42, -0.02))
        e = info.value
        assert e.obstacle == "blue_cylinder"
        assert e.link in ("upper_arm", "forearm", "hand")
        assert e.segment in SEGMENT_NAMES
        assert e.model == "tube" and e.margin_m == 0.0 and e.clearance_m < 0
        for part in ("blue_cylinder", e.link, e.segment, "tube"):
            assert part in str(e)
        # ZERO of each: goal writes, turn_on, gripper commands, head commands.
        assert robot.r_arm.writes == []
        assert robot.turn_on_calls == []
        assert spies.calls == []

    def test_an_obstructed_return_refuses_the_whole_arc_before_motion(
            self, planned, monkeypatch):
        """An object that exists only on the return path.

        On this arc no real placement can do that: the swing-in's first joint
        step sweeps the same wedge as the return's last, so any object placed
        on the sampled return is also hit by an EARLIER segment.  So the
        object is injected for the return segment alone, through the same
        `check_arm_path` the planner calls, and put where the planned return
        passes (the forearm mid-point of its middle pose).
        """
        spies = Spies(monkeypatch)
        scene, cells = board({"red_cube": "cell_r2c2"})
        robot = RecordingRobot()
        planner = CartesianPlanner(robot.r_arm, scene=scene)
        original = CartesianPlanner.check_arm_path

        def obstruct(self, joints_seq, *, segment, gripper_deg, target=None):
            if segment == L.SEG_RETURN:
                _s, elbow, wrist, _R = link_frames(joints_seq[len(joints_seq) // 2])
                where = tuple(float(x) for x in (elbow + wrist) / 2)
                blocked = self.scene.copy()
                blocked.update_poses({"soda_can": where})
                self = self.with_scene(blocked)
            return original(self, joints_seq, segment=segment,
                            gripper_deg=gripper_deg, target=target)

        monkeypatch.setattr(CartesianPlanner, "check_arm_path", obstruct)
        with pytest.raises(ArmClearanceError) as info:
            L.plan_pick_and_place(planner, scene, "red_cube",
                                  cells["cell_r2c3"][:2])
        assert info.value.segment == "return to the raised pose"
        assert info.value.obstacle == "soda_can"
        assert robot.r_arm.writes == [] and robot.turn_on_calls == []
        assert spies.calls == []

    def test_present_not_side_high_is_the_start_and_end(self, planned, monkeypatch):
        scene = planned["scene"]
        monkeypatch.setattr(P, "SIDE_HIGH", dict(P.SIDE_HIGH, r_shoulder_roll=-60.0,
                                                 r_elbow_pitch=-30.0))
        robot, planner = _fresh_robot_planner(scene)
        assert list(L.side_hub(planner)[0]) == L.present_joints()
        plan = L.plan_pick_and_place(planner, scene, "red_cube",
                                     planned["cells"]["cell_r2c3"][:2])
        assert plan.start == plan.end == L.present_joints()
        assert plan == planned["plan"]       # the plan did not change


# ── executing, and the return ────────────────────────────────────────────────

def _execute(planned, monkeypatch, *, lag_streams=(), obstruct_correction=False,
             should_abort=None, close_hold_s=0.0):
    spies = Spies(monkeypatch)
    plan = planned["plan"]
    robot = RecordingRobot()
    arm = robot.r_arm
    arm.place_at(plan.start)
    spies.stream_hook = lambda n: setattr(arm, "lagging", n in lag_streams)
    # `plan.scene` is the plan's own snapshot, so adjusting it here never
    # touches the shared board.
    plan = type(plan)(**{**plan.__dict__, "scene": plan.scene.copy()})
    if obstruct_correction:
        q_m = list(plan.end)
        q_m[0] += arm.lag_offset["r_shoulder_pitch"]
        _s, elbow, wrist, _R = link_frames(q_m)
        plan.scene.update_poses(
            {"soda_can": tuple(float(x) for x in (elbow + wrist) / 2)})
    planner = CartesianPlanner(arm, scene=plan.scene)
    events = []
    original = CartesianPlanner.check_arm_path

    def spy(self, joints_seq, *, segment, gripper_deg, target=None):
        events.append(("check", segment))
        return original(self, joints_seq, segment=segment,
                        gripper_deg=gripper_deg, target=target)

    monkeypatch.setattr(CartesianPlanner, "check_arm_path", spy)
    phases = []
    return plan, robot, planner, spies, events, phases, dict(
        should_abort=should_abort, close_hold_s=close_hold_s,
        on_phase=phases.append)


class TestOldFinalStepRegression:
    def test_the_old_raise_to_side_rejects_the_post_place_posture(
            self, planned, monkeypatch):
        """Since eb64b83, `raise_to_side` requires HOME.  The arm is above the
        place site when the retract ends, so the old last step could never
        have worked."""
        plan = planned["plan"]
        robot = RecordingRobot()
        robot.r_arm.place_at(plan.segments[6].traj[-1])
        with pytest.raises(RuntimeError, match="HOME"):
            P.raise_to_side(robot.r_arm)

    def test_the_replacement_is_in_the_plan_and_arrives_where_the_stow_needs(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch)
        # the stow's precondition holds at plan.end ...
        probe = RecordingRobot()
        probe.r_arm.place_at(plan.end)
        assert P.stow_entry_ok(probe.r_arm)[0] is True
        # ... and execute_arc never reaches for raise_to_side.
        final = L.execute_arc(robot, plan, planner, **kw)
        assert final == L.present_joints()
        assert spies.count("raise_to_side") == 0
        assert P.stow_entry_ok(robot.r_arm)[0] is True

    def test_the_old_posture_would_fail_the_stow_entry_check(self, planned):
        """PRESENT and the post-retract pose are not interchangeable."""
        robot = RecordingRobot()
        robot.r_arm.place_at(planned["plan"].segments[6].traj[-1])
        ok, joint, off = P.stow_entry_ok(robot.r_arm)
        assert ok is False and off > 12.0


class TestExecutionOrder:
    def test_phases_gripper_and_streams_run_in_the_old_order(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch, close_hold_s=0.5)
        L.execute_arc(robot, plan, planner, **kw)
        assert phases == [
            "swing in over table", "descend to hover", "descend to grasp",
            "close gripper", "lift to clear height", "carry over table",
            "descend to place", "release", "retract",
            "return to the raised pose"]
        names = spies.names()
        assert names[0] == "look_at"
        assert names.index("close_gripper") < [
            i for i, c in enumerate(spies.calls) if c[:2] == ("stream", 4)][0]
        assert names.index("open_gripper") < [
            i for i, c in enumerate(spies.calls) if c[:2] == ("stream", 7)][0]
        assert spies.streams == 8
        assert 0.5 in spies.sleeps
        for banned in ("raise_to_side", "converge", "smooth_move", "go_home"):
            assert spies.count(banned) == 0
        assert events == []          # no corrections were needed

    def test_a_cancel_before_the_close_stops_at_the_boundary(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch, should_abort=lambda: True)
        final = L.execute_arc(robot, plan, planner, **kw)
        assert final == L.present_joints()      # never left PRESENT
        assert spies.streams == 0 and spies.count("close_gripper") == 0

    def test_a_cancel_after_the_close_is_ignored_and_the_arc_finishes(
            self, planned, monkeypatch):
        seen = []

        def abort():
            seen.append(1)
            return len(seen) > 3          # True only from the close onward

        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch, should_abort=abort)
        L.execute_arc(robot, plan, planner, **kw)
        assert spies.streams == 8 and spies.count("close_gripper") == 1


class TestReturnGuarantee:
    def test_a_arrives_after_the_stream_with_no_corrective_move(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch)
        L.execute_arc(robot, plan, planner, **kw)
        assert spies.streams == 8
        assert [e for e in events if "correction" in e[1]] == []
        assert spies.count("converge") == spies.count("smooth_move") == 0

    def test_b_a_lagging_arm_gets_one_checked_corrective_sequence(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch, lag_streams=(8,))
        before = len(robot.r_arm.writes)
        L.execute_arc(robot, plan, planner, **kw)
        corrections = [e for e in events if "correction" in e[1]]
        assert len(corrections) == 1
        assert spies.streams == 9
        assert P.stow_entry_ok(robot.r_arm)[0] is True
        assert len(robot.r_arm.writes) > before
        assert spies.count("converge") == spies.count("smooth_move") == 0

    def test_b_the_correction_check_precedes_the_correction_write(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch, lag_streams=(8,))
        order = []
        original = CartesianPlanner.check_arm_path

        def spy(self, joints_seq, *, segment, gripper_deg, target=None):
            order.append(("check", segment, spies.streams))
            return original(self, joints_seq, segment=segment,
                            gripper_deg=gripper_deg, target=target)

        monkeypatch.setattr(CartesianPlanner, "check_arm_path", spy)
        L.execute_arc(robot, plan, planner, **kw)
        # the correction check was made when only the 8 planned streams existed
        assert order == [("check", "return to the raised pose (correction 1)", 8)]

    def test_c_an_obstructed_correction_is_never_commanded(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch, lag_streams=range(8, 100),
            obstruct_correction=True)
        with pytest.raises(L.ReturnArrivalError) as info:
            L.execute_arc(robot, plan, planner, **kw)
        e = info.value
        assert isinstance(e.refusal, ArmClearanceError)
        assert e.refusal.obstacle == "soda_can"
        assert e.attempts == 1 and e.worst_joint == "r_shoulder_pitch"
        assert e.off_deg > 12.0
        assert spies.streams == 8                # no corrective stream at all
        assert spies.count("go_home") == 0

    def test_d_an_arm_that_never_arrives_gets_exactly_the_maximum_checked_attempts(
            self, planned, monkeypatch):
        plan, robot, planner, spies, events, phases, kw = _execute(
            planned, monkeypatch, lag_streams=range(8, 100))
        with pytest.raises(L.ReturnArrivalError) as info:
            L.execute_arc(robot, plan, planner, **kw)
        e = info.value
        assert e.attempts == L.RETURN_CORRECTIONS_MAX == 2
        assert e.refusal is None
        corrections = [x for x in events if "correction" in x[1]]
        assert len(corrections) == L.RETURN_CORRECTIONS_MAX
        assert spies.streams == 8 + L.RETURN_CORRECTIONS_MAX
        assert spies.count("go_home") == 0


# ── pins ──────────────────────────────────────────────────────────────────────

class TestPins:
    def test_the_whole_arm_margin_is_the_footprint_margin(self):
        assert WHOLE_ARM_MARGIN_M == R.FOOTPRINT_MARGIN == 0.0

    def test_open_is_the_primitives_open(self):
        assert R.OPEN == P._GRIPPER_OPEN_DEG == -45.0

    def test_transit_values_are_unchanged_and_re_exported(self):
        assert transit.CLEAR_Z == 1.00
        assert transit.CARRY_HZ == 8
        assert transit.STEP_HZ == 25
        assert L.CLEAR_Z is transit.CLEAR_Z

    def test_sampling_step_is_two_degrees(self):
        assert ARM_PATH_MAX_SAMPLE_STEP_DEG == 2.0

    def test_nothing_in_production_selects_the_max_clearance_policy(self):
        import ast
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.join(here, "..", "..")
        for rel in ("src/reachy_ai/tasks/pick_place_live.py",
                    "web/motion_worker.py", "scripts/demo_pick_place.py"):
            with open(os.path.join(root, rel), encoding="utf-8") as f:
                tree = ast.parse(f.read())
            used = [n for n in ast.walk(tree)
                    if (isinstance(n, ast.Attribute) and n.attr == "MAX_CLEARANCE")
                    or (isinstance(n, ast.Name) and n.id == "MAX_CLEARANCE")
                    or (isinstance(n, ast.Constant) and n.value == "max_clearance")
                    or (isinstance(n, ast.keyword) and n.arg == "maximise_clearance")]
            assert used == [], rel

    def test_raise_to_side_is_not_called_by_the_arc_module(self):
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "..", "..", "src", "reachy_ai", "tasks",
                               "pick_place_live.py"), encoding="utf-8") as f:
            text = f.read()
        assert "P.raise_to_side(" not in text
        assert "P.converge(" not in text and "P.smooth_move(" not in text
