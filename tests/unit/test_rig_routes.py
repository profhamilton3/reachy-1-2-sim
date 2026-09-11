"""Issue #64: the rig pose set and routes, extracted from the notebook.

The first test is the one that matters.  Every other test here could pass
against a transcription that quietly dropped a decimal, so the values are
compared against the notebook's own cell source rather than trusted — the
notebook stays the source of truth, and this file is what notices when the two
drift apart.

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
# The extraction is faithful
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "HOME", "GRIP_SHUT", "BACK", "CURL", "CURL_HIGH", "TUCK",
    "SWING_1", "SWING_2", "SWING_3", "HOVER", "REST_SHUT", "REST", "PRESENT",
])
def test_every_pose_matches_the_notebook(notebook, name):
    assert getattr(R, name) == notebook[name], name


def test_the_gripper_constants_match(notebook):
    assert R.OPEN == notebook["OPEN"]
    assert R.SHUT == notebook["SHUT"]
    assert R.TRACK_TOL == notebook["TRACK_TOL"]


def test_the_gripper_sign_is_the_inverted_one():
    """Negative OPENS.  Verified on the robot, not inferred from the name —
    and a tidy-up that "fixed" the sign would close the hand on approach."""
    assert R.OPEN < 0 < R.SHUT


@pytest.mark.parametrize("route_name", ["PLACE_ROUTE", "STOW_ROUTE"])
def test_route_waypoints_match_the_notebook(notebook, route_name):
    mine = getattr(R, route_name)
    theirs = notebook[route_name]
    assert len(mine) == len(theirs)
    for got, (name, target, secs, tol) in zip(mine, theirs):
        assert got.name == name
        assert got.pose == target, name
        assert got.seconds == secs, name
        assert got.tol == tol, name


def test_the_wave_poses_match_the_notebook(notebook):
    assert R.WAVE_A == notebook["WAVE_A"]
    assert R.WAVE_B == notebook["WAVE_B"]


def test_the_tolerances_are_per_waypoint_not_uniform():
    """Settling into GRIP_SHUT has 79 mm of clearance all round; SWING_3 to
    HOVER passes the elbow 4.8 mm from the board's edge.  One tolerance for
    both would be a rewrite, not an extraction."""
    tols = {w.name: w.tol for w in R.STOW_ROUTE}
    assert tols["GRIP_SHUT"] != tols["SWING_3"]
    assert len({w.tol for w in R.STOW_ROUTE}) > 1


def test_the_stow_route_is_the_placement_route_reversed():
    place = [w.name for w in R.PLACE_ROUTE]
    stow = [w.name for w in R.STOW_ROUTE]
    # Exactly reversed, once the two differing endpoints are set aside:
    # placement finishes by opening the gripper at REST, and stow finishes by
    # settling at HOME.  Everything between is the same corridor backwards,
    # which is the property that makes it safe — nothing may cut across it.
    assert stow[:-1] == list(reversed(place[:-1]))
    assert place[-1] == "REST" and stow[-1] == "HOME"


def test_the_wave_is_bounded_and_relative_to_present():
    assert R.WAVE_CYCLES == 3
    assert R.WAVE_SECONDS == 1.8
    assert R.WAVE_END == "present"
    for pose in (R.WAVE_A, R.WAVE_B):
        for joint in ("r_shoulder_pitch", "r_shoulder_roll", "r_elbow_pitch"):
            assert pose[joint] == R.PRESENT[joint]


# ---------------------------------------------------------------------------
# Scene compatibility
# ---------------------------------------------------------------------------

def test_the_routes_are_validated_where_they_were_measured():
    for route in ("PLACE_ROUTE", "STOW_ROUTE", "WAVE"):
        ok, why = R.check_route(route, "FWDCenterLabMCC")
        assert ok, why


@pytest.mark.parametrize("route", ["PLACE_ROUTE", "STOW_ROUTE"])
def test_the_panel_scene_is_validated_on_flights_through_the_flight(route):
    """FWDCenterLabSivaPool is where the panel runs, and it is listed now.

    It was not, and the verdict changed on evidence rather than on wishing.
    The first round sampled the WAYPOINTS of an empty board and found SWING_1
    2 mm inside `rig_rail_outer_right`.  The second sampled at 20 Hz THROUGH
    eighteen legs with objects on the board and found the same few millimetres
    of graze and nothing moved, at the one rail #73 already documents as
    disagreeing with the model in both directions.  See docs/adr/0002.
    """
    ok, why = R.check_route(route, "FWDCenterLabSivaPool")
    assert ok, why
    row = R.validation_for(route, "FWDCenterLabSivaPool")
    assert "2026-09-10" in row.evidence
    assert "undisturbed" in row.evidence or "RAISE_TO_SIDE" in row.evidence


def test_a_rejected_route_says_it_was_flown_and_why_it_failed():
    """"Not listed" reads as "nobody has got to it yet", which invites a row
    on the strength of one clean run.  Pointing in the scene it was DESIGNED
    for is the standing example: flown, rejected, and the number travels."""
    _, why = R.check_route("POINT", "FWDCenterLabMCC")
    assert "rejected" in why
    assert "0.189 m" in why


def test_a_route_nobody_flew_here_is_a_different_answer_from_one_that_failed():
    _, why = R.check_route("PLACE_ROUTE", "SomeSceneNobodyTried")
    assert "has not been flown in SomeSceneNobodyTried" in why
    assert "rejected" not in why


def test_the_hub_routes_were_reflown_before_they_were_relisted():
    """The rows that were here certified four hand-built steps through a roll
    -88 hub.  They were deleted, not edited, and what replaced them names the
    flights that happened AFTER the rebuild — not the six that happened before
    it against an empty board."""
    for route in ("WAVE", "RAISE_TO_SIDE", "STOW_FROM_SIDE", "LIFT_TO_PRESENT"):
        row = R.validation_for(route, "FWDCenterLabSivaPool")
        assert row is not None
        assert "hub" not in row.evidence or "deleted" in row.evidence
    assert "REBUILT" in R.validation_for("WAVE", "FWDCenterLabSivaPool").evidence


def test_the_hub_routes_are_validated_where_the_notebook_flew_them():
    for route in ("RAISE_TO_SIDE", "STOW_FROM_SIDE", "LIFT_TO_PRESENT",
                  "LOWER_TO_REST"):
        ok, why = R.check_route(route, "FWDCenterLabMCC")
        assert ok, "{}: {}".format(route, why)


def test_every_attempt_records_an_outcome_and_the_numbers():
    for attempt in R.VALIDATION_ATTEMPTS:
        assert attempt.outcome in ("rejected", "not attempted")
        assert attempt.when
        assert len(attempt.detail) > 40


def test_pointing_is_validated_over_an_empty_board_and_nowhere_else():
    """Scoped on purpose.  The empty-board run had nothing to hit, so it says
    nothing about section 4.7's catalogue of what an occupied board does — a
    can moved 0.189 m by an arm reporting +5.5 cm of clearance.  The row says
    "empty board only" and `_ability_available` is what enforces it."""
    ok, why = R.check_route("POINT", "FWDCenterLabSivaPool")
    assert ok, why
    row = R.validation_for("POINT", "FWDCenterLabSivaPool")
    assert "EMPTY BOARD ONLY" in row.evidence
    assert "cell_r3c1" in row.evidence          # the arm's own limit, named

    # And it is still refused where nobody has flown it.
    ok, why = R.check_route("POINT", "FWDCenterLabMCC")
    assert not ok
    assert "0.189 m" in why


def test_a_refusal_names_the_scene_not_the_arm():
    _, why = R.check_route("STOW_ROUTE", "SomeOtherScene")
    assert "SomeOtherScene" in why
    assert "arm is unavailable" not in why


def test_every_validation_row_carries_its_evidence():
    """A row without a flown run recorded would undo this module's whole
    safety argument in one line."""
    for row in R.ROUTE_COMPATIBILITY:
        assert row.evidence.strip()
        # Either the notebook that measured it, or a dated flight of it.
        assert ("tlh_motion-routine" in row.evidence
                or "2026-" in row.evidence), row.route


# ---------------------------------------------------------------------------
# Recovery reports; it does not authorise
# ---------------------------------------------------------------------------

def test_nearest_waypoint_finds_the_one_the_arm_is_actually_at():
    name, distance = R.nearest_waypoint(dict(R.SWING_2))
    assert name == "SWING_2"
    assert distance == pytest.approx(0.0)


def test_nearest_waypoint_ignores_the_gripper():
    """On a freshly reset sim the gripper falls open and wrist_roll drifts;
    judging where the arm is on those reports a clean reset as a fault."""
    drifted = dict(R.TUCK, r_gripper=R.OPEN)
    name, _ = R.nearest_waypoint(drifted)
    assert name == "TUCK"


def test_at_pose_checks_only_the_gross_joints_when_asked():
    drifted = dict(R.HOME, r_wrist_roll=40.0, r_gripper=R.OPEN)
    assert R.at_pose(drifted, R.HOME, joints=list(R.GROSS_JOINTS))
    assert not R.at_pose(drifted, R.HOME)


# ---------------------------------------------------------------------------
# The posture graph (issue #65)
# ---------------------------------------------------------------------------

def test_the_named_postures_are_recognised_from_a_pose():
    assert R.posture_of(dict(R.HOME)) == R.POSTURE_HOME
    assert R.posture_of(dict(R.REST)) == R.POSTURE_REST
    assert R.posture_of(dict(R.PRESENT)) == R.POSTURE_PRESENT


def test_a_pose_between_postures_is_not_any_of_them():
    """"Somewhere in the corridor" is an answer, and it is not "close enough
    to HOME"."""
    assert R.posture_of(dict(R.SWING_2)) is None


def test_posture_is_judged_on_the_gross_joints():
    drifted = dict(R.HOME, r_wrist_roll=40.0, r_gripper=R.OPEN)
    assert R.posture_of(drifted) == R.POSTURE_HOME


def test_only_measured_transitions_exist():
    """There is no "just move there" edge, and the absence is the point."""
    assert R.transition(R.POSTURE_HOME, R.POSTURE_REST) == "PLACE_ROUTE"
    assert R.transition(R.POSTURE_REST, R.POSTURE_HOME) == "STOW_ROUTE"
    # Measured, all four, and every one of them is in the notebook.
    assert R.transition(R.POSTURE_REST, R.POSTURE_PRESENT) == "LIFT_TO_PRESENT"
    assert R.transition(R.POSTURE_PRESENT, R.POSTURE_REST) == "LOWER_TO_REST"
    assert R.transition(R.POSTURE_HOME, R.POSTURE_PRESENT) == "RAISE_TO_SIDE"
    assert R.transition(R.POSTURE_PRESENT, R.POSTURE_HOME) == "STOW_FROM_SIDE"
    # Still not a posture, and not an edge: the invented side hub.
    assert not hasattr(R, "POSTURE_SIDE_HUB")
    assert "side_hub" not in R.POSTURES


def test_the_wave_is_a_transition_from_present_to_itself():
    """It ends where it started.  Getting to rest or the pocket afterwards is
    a separate transition, and there is not one."""
    assert R.transition(R.POSTURE_PRESENT, R.POSTURE_PRESENT) == "WAVE"
    assert R.POSTURE_PRESENT in R.reachable_from(R.POSTURE_PRESENT)


def test_the_way_out_of_the_pocket_is_the_measured_route():
    """It used to abduct to roll -88, which the notebook names as its FAILURE
    exercise.  The route crosses the rail band in two joints at once and never
    rolls past -37.5."""
    assert [w.name for w in R.RAISE_TO_SIDE] == \
        [w.name for w in R.PLACE_ROUTE] + ["PRESENT"]
    assert [w.name for w in R.STOW_FROM_SIDE] == [w.name for w in R.STOW_ROUTE]
    deepest = min(w.pose["r_shoulder_roll"] for w in R.RAISE_TO_SIDE)
    assert deepest == -37.5


def test_an_arm_in_the_pocket_is_routed_out_of_it():
    """"Wave" asked of a stored arm answers with the route, not a refusal."""
    assert R.path(R.POSTURE_HOME, R.POSTURE_PRESENT) == ["RAISE_TO_SIDE"]
    assert R.path(R.POSTURE_HOME, R.POSTURE_HOME) == []


def test_a_path_from_rest_is_the_notebooks_single_move():
    """It used to be STOW_ROUTE, RAISE_TO_SIDE, PRESENT_ROUTE — all the way
    home and back out through the hub.  The notebook lifts straight off the
    board (cell 18)."""
    assert R.path(R.POSTURE_REST, R.POSTURE_PRESENT) == ["LIFT_TO_PRESENT"]


def test_already_there_is_no_moves_and_the_wave_is_the_exception():
    assert R.path(R.POSTURE_HOME, R.POSTURE_HOME) == []
    assert R.path(R.POSTURE_PRESENT, R.POSTURE_PRESENT) == ["WAVE"]


def test_a_path_that_does_not_exist_is_still_a_real_answer():
    """It is what stops a plan inventing a way through."""
    assert R.path("nowhere", R.POSTURE_HOME) is None
    assert R.path(R.POSTURE_HOME, "nowhere") is None


def test_every_edge_has_something_that_can_fly_it():
    from reachy_ai.tasks import rig_motion as M

    for route in set(R.POSTURE_TRANSITIONS.values()):
        assert route in M.ROUTE_RUNNERS, route


# ---------------------------------------------------------------------------
# #82: FOOTPRINT_LEGS names the tabletop-adjacent tail/head of each route the
# forearm-footprint check sweeps against live object poses.
# ---------------------------------------------------------------------------

class TestFootprintLegs:
    def test_every_named_route_is_a_real_posture_edge(self):
        """A route this points at that does not exist in the posture graph
        would be checking a footprint nothing ever flies."""
        edges = set(R.POSTURE_TRANSITIONS.values())
        for route in R.FOOTPRINT_LEGS:
            assert route in edges, route

    def test_place_route_and_raise_to_side_share_the_same_tail(self):
        """RAISE_TO_SIDE is PLACE_ROUTE + LIFT_TO_PRESENT, so it crosses the
        identical tabletop corridor on its way through — checked once, by
        route name, rather than duplicated per ability."""
        assert R.FOOTPRINT_LEGS["RAISE_TO_SIDE"] == R.FOOTPRINT_LEGS["PLACE_ROUTE"]

    def test_stow_route_and_stow_from_side_share_the_same_tail(self):
        assert (R.FOOTPRINT_LEGS["STOW_FROM_SIDE"]
                == R.FOOTPRINT_LEGS["STOW_ROUTE"])

    def test_place_routes_tail_matches_its_own_last_three_waypoints(self):
        """The checked legs are not an approximation of PLACE_ROUTE's real
        tail — they are it, in the same order the route actually flies."""
        last_three = tuple(w.pose for w in R.PLACE_ROUTE[-3:])
        assert last_three == (R.HOVER, R.REST_SHUT, R.REST)
        assert R.FOOTPRINT_LEGS["PLACE_ROUTE"] == last_three

    def test_stow_routes_head_is_place_routes_tail_reversed(self):
        """STOW_ROUTE departs the board the way PLACE_ROUTE arrives at it,
        flown backwards — the same corridor, not a second one."""
        assert (R.FOOTPRINT_LEGS["STOW_ROUTE"]
                == tuple(reversed(R.FOOTPRINT_LEGS["PLACE_ROUTE"])))

    def test_lower_to_rest_starts_from_present_not_hover(self):
        """LOWER_TO_REST is flown FROM the presentation pose, not from the
        pocket-exit corridor — a different entry, the same landing."""
        assert R.FOOTPRINT_LEGS["LOWER_TO_REST"][0] == R.PRESENT
        assert R.FOOTPRINT_LEGS["LOWER_TO_REST"][-2:] == (R.REST_SHUT, R.REST)

    def test_wave_and_lift_to_present_are_absent_on_purpose(self):
        """Neither route lands the arm on REST or departs from it, so neither
        can catch an object its own named route did not already put there."""
        assert "WAVE" not in R.FOOTPRINT_LEGS
        assert "LIFT_TO_PRESENT" not in R.FOOTPRINT_LEGS
