"""Issue #64: the rig pose set and routes; #192: the October layout.

The poses still taken from the notebook (PRESENT, the wave) are compared
against the notebook's own cell source rather than trusted.  REST is not the
notebook's any more: it is the October layout's settled pose (#192), and the
rail-pocket routes the notebook flew went with the August mount.

Nothing here connects to anything.  The notebook is parsed, never executed:
running its cells would connect to a robot.
"""

import ast
import json
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, "../../src"))
sys.path.insert(0, os.path.join(_HERE, "../../web"))

from reachy_ai.motion import rig_routes as R  # noqa: E402

_NOTEBOOK = os.path.join(_HERE, "../../notebooks/tlh_motion-routine.ipynb")


def _notebook_namespace():
    """Evaluate ONLY the pose definitions from the notebook's cell source.

    Not `exec` of the cell: it imports the SDK and builds a planner against a
    live arm.  The pose set is a run of plain assignments, so they are lifted
    out of the parsed tree and evaluated against a namespace holding just the
    handful of names they use.
    """
    with open(_NOTEBOOK) as fh:
        nb = json.load(fh)

    names = {"OPEN": None, "SHUT": None, "TRACK_TOL": None}
    env = {"dict": dict, "pose": None}

    # OPEN/SHUT come from the joint-map cell, TRACK_TOL from the helpers cell.
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in names:
                continue
            names[target.id] = ast.literal_eval(node.value)
    assert all(v is not None for v in names.values()), names
    env.update(names)

    def pose(**kw):
        base = dict.fromkeys(env["ARM7"], 0.0)
        base["r_gripper"] = env["OPEN"]
        base.update(kw)
        return base

    env["ARM7"] = [
        "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw", "r_elbow_pitch",
        "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll",
    ]
    env["pose"] = pose

    # Now the pose-set cell and the wave cell, assignment by assignment.
    wanted = {"HOME", "GRIP_SHUT", "BACK", "CURL", "CURL_HIGH", "TUCK",
              "SWING_1", "SWING_2", "SWING_3", "HOVER", "REST_SHUT", "REST",
              "PRESENT", "PLACE_ROUTE", "STOW_ROUTE", "WAVE_A", "WAVE_B"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if not isinstance(node, ast.Assign):
                continue
            target = node.targets[0]
            if not isinstance(target, ast.Name) or target.id not in wanted:
                continue
            env[target.id] = eval(  # noqa: S307 - notebook source, parsed above
                compile(ast.Expression(node.value), "<notebook>", "eval"), env)
    missing = wanted - set(env)
    assert not missing, f"not found in the notebook: {sorted(missing)}"
    return env


@pytest.fixture(scope="module")
def notebook():
    return _notebook_namespace()


# ---------------------------------------------------------------------------
# What is still the notebook's
# ---------------------------------------------------------------------------

def test_present_matches_the_notebook(notebook):
    assert R.PRESENT == notebook["PRESENT"]


def test_the_gripper_constants_match(notebook):
    assert R.OPEN == notebook["OPEN"]
    assert R.SHUT == notebook["SHUT"]
    assert R.TRACK_TOL == notebook["TRACK_TOL"]


def test_the_gripper_sign_is_the_inverted_one():
    """Negative OPENS.  Verified on the robot, not inferred from the name —
    and a tidy-up that "fixed" the sign would close the hand on approach."""
    assert R.OPEN < 0 < R.SHUT


def test_the_wave_poses_match_the_notebook(notebook):
    assert R.WAVE_A == notebook["WAVE_A"]
    assert R.WAVE_B == notebook["WAVE_B"]


def test_the_wave_is_bounded_and_relative_to_present():
    assert R.WAVE_CYCLES == 3
    assert R.WAVE_SECONDS == 1.8
    assert R.WAVE_END == "present"
    for pose in (R.WAVE_A, R.WAVE_B):
        for joint in ("r_shoulder_pitch", "r_shoulder_roll", "r_elbow_pitch"):
            assert pose[joint] == R.PRESENT[joint]


# ---------------------------------------------------------------------------
# REST and its routes, October layout (#192)
# ---------------------------------------------------------------------------

def test_rest_is_shut_and_rest_shut_is_rest():
    assert R.REST["r_gripper"] == R.SHUT
    assert R.REST_SHUT == R.REST


def test_rest_out_is_rests_arm_swung_out_at_presents_roll():
    moved = {j for j in R.REST if R.REST_OUT[j] != R.REST[j]}
    assert moved == {"r_shoulder_pitch", "r_shoulder_roll"}
    assert R.REST_OUT["r_shoulder_roll"] == R.PRESENT["r_shoulder_roll"]


def test_the_lowering_is_the_lift_reversed():
    lift = [w.name for w in R.LIFT_TO_PRESENT]
    lower = [w.name for w in R.LOWER_TO_REST]
    assert lift == ["REST_OUT", "PRESENT_SHUT", "PRESENT"]
    assert lower == ["PRESENT_SHUT", "REST_OUT", "REST"]


def test_the_hand_is_shut_whenever_it_is_over_the_board():
    """The footprint guard sizes the hand to its aperture: the hand opens only
    at PRESENT, raised."""
    for route in (R.LIFT_TO_PRESENT, R.LOWER_TO_REST):
        for wp in route:
            if wp.name != "PRESENT":
                assert wp.pose["r_gripper"] == R.SHUT, wp.name


def test_the_august_mount_is_retired():
    """Owner decision 2026-10-18: no August routes, poses or replays."""
    for name in ("HOME", "GRIP_SHUT", "BACK", "CURL", "CURL_HIGH", "TUCK",
                 "SWING_1", "SWING_1_NOTEBOOK", "SWING_2", "SWING_3", "HOVER",
                 "REST_AUGUST", "REST_SHUT_AUGUST", "PLACE_ROUTE", "STOW_ROUTE",
                 "RAISE_TO_SIDE", "STOW_FROM_SIDE", "LIFT_TO_PRESENT_AUGUST",
                 "LOWER_TO_REST_AUGUST", "MOUNT_AUGUST", "CORRIDOR_ROUTES",
                 "corridor_entry_refusal", "with_notebook_swing_1"):
        assert not hasattr(R, name), name
    for name in ("PLACE_ROUTE", "STOW_ROUTE", "RAISE_TO_SIDE", "STOW_FROM_SIDE"):
        with pytest.raises(KeyError):
            R.route_named(name)


# ---------------------------------------------------------------------------
# Scene compatibility
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("route", ["LIFT_TO_PRESENT", "LOWER_TO_REST", "WAVE", "POINT"])
@pytest.mark.parametrize("scene", ["FWDCenterLabSivaPool", "FWDCenterLabMCC"])
def test_the_october_routes_are_validated_in_both_lab_scenes(route, scene):
    assert R.check_route(route, scene) == (True, "")
    assert R.validation_for(route, scene).mount == R.CURRENT_MOUNT


def test_every_row_is_at_the_current_mount():
    for row in R.ROUTE_COMPATIBILITY:
        assert row.mount == R.CURRENT_MOUNT, (row.route, row.scene)


def test_a_row_must_say_where_the_robot_stood():
    with pytest.raises(ValueError, match="names no mount"):
        R.RouteValidation("WAVE", "SomeRig", "2026-10-18: test row")


def test_a_row_flown_at_another_mount_is_refused_and_says_so(monkeypatch):
    row = R.RouteValidation("WAVE", "SomeRig", "2026-10-18: test row",
                            mount="somewhere else")
    monkeypatch.setattr(R, "ROUTE_COMPATIBILITY", (row,))
    ok, why = R.check_route("WAVE", "SomeRig")
    assert not ok
    assert "robot mounted elsewhere" in why and R.CURRENT_MOUNT in why


def test_pointing_is_validated_up_to_one_object():
    """Scoped on purpose: what has not been flown is a reach threading past a
    second object, and `_ability_available` enforces that."""
    row = R.validation_for("POINT", "FWDCenterLabSivaPool")
    assert "AN EMPTY BOARD" in row.evidence
    assert "ONE OBJECT AT A TIME" in row.evidence
    assert "TWO OR MORE OBJECTS IS NOT COVERED" in row.evidence


def test_a_route_nobody_flew_here_is_a_different_answer_from_one_that_failed():
    _, why = R.check_route("LOWER_TO_REST", "SomeSceneNobodyTried")
    assert "has not been flown in SomeSceneNobodyTried" in why
    assert "rejected" not in why


def test_a_rejected_attempt_travels(monkeypatch):
    monkeypatch.setattr(R, "VALIDATION_ATTEMPTS", (R.ValidationAttempt(
        "WAVE", "SomeRig", "2026-10-18", "rejected", "moved a can 0.189 m"),))
    monkeypatch.setattr(R, "ROUTE_COMPATIBILITY", ())
    _, why = R.check_route("WAVE", "SomeRig")
    assert "rejected" in why and "0.189 m" in why


def test_a_route_never_validated_anywhere_is_refused():
    assert R.check_route("NO_SUCH_ROUTE", "FWDCenterLabSivaPool")[0] is False


def test_a_refusal_names_the_scene_not_the_arm():
    _, why = R.check_route("LOWER_TO_REST", "SomeOtherScene")
    assert "SomeOtherScene" in why
    assert "arm is unavailable" not in why


def test_every_validation_row_carries_dated_evidence():
    """A row without a flown run recorded would undo this module's whole
    safety argument in one line."""
    for row in R.ROUTE_COMPATIBILITY:
        assert row.evidence.startswith("2026-10-1"), (row.route, row.scene)
        assert "#192" in row.evidence


def test_every_attempt_records_an_outcome_and_the_numbers():
    for attempt in R.VALIDATION_ATTEMPTS:
        assert attempt.outcome in ("rejected", "not attempted")
        assert attempt.when
        assert len(attempt.detail) > 40


# ---------------------------------------------------------------------------
# Recovery reports; it does not authorise
# ---------------------------------------------------------------------------

def test_nearest_waypoint_finds_the_one_the_arm_is_actually_at():
    name, distance = R.nearest_waypoint(dict(R.REST_OUT))
    assert name == "REST_OUT"
    assert distance == pytest.approx(0.0)


def test_nearest_waypoint_ignores_the_gripper():
    """On a freshly reset sim the gripper falls open and wrist_roll drifts;
    judging where the arm is on those reports a clean reset as a fault."""
    name, _ = R.nearest_waypoint(dict(R.REST_OUT, r_gripper=R.OPEN))
    assert name == "REST_OUT"


def test_at_pose_checks_only_the_gross_joints_when_asked():
    drifted = dict(R.REST, r_wrist_roll=10.0, r_gripper=R.OPEN)
    assert R.at_pose(drifted, R.REST, joints=list(R.GROSS_JOINTS))
    assert not R.at_pose(drifted, R.REST)


# ---------------------------------------------------------------------------
# The posture graph (issue #65)
# ---------------------------------------------------------------------------

def test_the_named_postures_are_recognised_from_a_pose():
    assert R.posture_of(dict(R.REST)) == R.POSTURE_REST
    assert R.posture_of(dict(R.PRESENT)) == R.POSTURE_PRESENT


def test_home_is_rest():
    assert R.POSTURE_HOME == R.POSTURE_REST
    assert set(R.POSTURES) == {R.POSTURE_REST, R.POSTURE_PRESENT}


def test_a_pose_between_postures_is_not_any_of_them():
    assert R.posture_of(dict(R.REST_OUT)) is None


def test_posture_is_judged_on_the_gross_joints():
    drifted = dict(R.REST, r_wrist_roll=10.0, r_gripper=R.OPEN)
    assert R.posture_of(drifted) == R.POSTURE_REST


def test_only_measured_transitions_exist():
    """There is no "just move there" edge, and the absence is the point."""
    assert R.POSTURE_TRANSITIONS == {
        (R.POSTURE_REST, R.POSTURE_PRESENT): "LIFT_TO_PRESENT",
        (R.POSTURE_PRESENT, R.POSTURE_REST): "LOWER_TO_REST",
        (R.POSTURE_PRESENT, R.POSTURE_PRESENT): "WAVE",
    }
    assert not hasattr(R, "POSTURE_SIDE_HUB")


def test_the_wave_is_a_transition_from_present_to_itself():
    assert R.transition(R.POSTURE_PRESENT, R.POSTURE_PRESENT) == "WAVE"
    assert R.POSTURE_PRESENT in R.reachable_from(R.POSTURE_PRESENT)


def test_a_stored_arm_is_raised_before_it_waves():
    """"Wave" asked of an arm at rest answers with the lift, not a refusal."""
    assert R.path(R.POSTURE_REST, R.POSTURE_PRESENT) == ["LIFT_TO_PRESENT"]
    assert R.path(R.POSTURE_PRESENT, R.POSTURE_HOME) == ["LOWER_TO_REST"]


def test_already_there_is_no_moves_and_the_wave_is_the_exception():
    assert R.path(R.POSTURE_REST, R.POSTURE_REST) == []
    assert R.path(R.POSTURE_PRESENT, R.POSTURE_PRESENT) == ["WAVE"]


def test_a_path_that_does_not_exist_is_still_a_real_answer():
    """It is what stops a plan inventing a way through."""
    assert R.path("nowhere", R.POSTURE_REST) is None
    assert R.path(R.POSTURE_REST, "nowhere") is None


def test_every_edge_has_something_that_can_fly_it():
    from reachy_ai.tasks import rig_motion as M

    for route in set(R.POSTURE_TRANSITIONS.values()):
        assert route in M.ROUTE_RUNNERS, route


# ---------------------------------------------------------------------------
# #82: FOOTPRINT_LEGS, the legs swept against live object poses
# ---------------------------------------------------------------------------

class TestFootprintLegs:
    def test_every_named_route_is_a_real_posture_edge(self):
        edges = set(R.POSTURE_TRANSITIONS.values())
        for route in R.FOOTPRINT_LEGS:
            assert route in edges, route

    def test_the_legs_are_the_whole_of_each_route(self):
        """October layout: every waypoint is over or beside the board."""
        assert R.FOOTPRINT_LEGS["LOWER_TO_REST"] == (R.PRESENT,) + tuple(
            w.pose for w in R.LOWER_TO_REST)
        assert R.FOOTPRINT_LEGS["LIFT_TO_PRESENT"] == (R.REST,) + tuple(
            w.pose for w in R.LIFT_TO_PRESENT)

    def test_the_lift_is_the_lowering_reversed(self):
        assert R.FOOTPRINT_LEGS["LIFT_TO_PRESENT"] == tuple(
            reversed(R.FOOTPRINT_LEGS["LOWER_TO_REST"]))

    def test_wave_is_absent_on_purpose(self):
        """WAVE never lands the arm on REST or departs from it."""
        assert "WAVE" not in R.FOOTPRINT_LEGS
