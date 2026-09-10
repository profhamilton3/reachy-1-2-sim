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


@pytest.mark.parametrize("route", ["PLACE_ROUTE", "STOW_ROUTE", "WAVE"])
def test_the_panel_scene_is_not_validated_yet(route):
    """FWDCenterLabSivaPool is where the panel runs and is deliberately absent.

    Endpoint equality does not prove the corridor is clear, and the two scenes
    differ in object population.
    """
    ok, why = R.check_route(route, "FWDCenterLabSivaPool")
    assert not ok
    assert "FWDCenterLabSivaPool" in why
    assert "FWDCenterLabMCC" in why


def test_pointing_is_not_a_validated_route_anywhere():
    """Section 4.7 records runs that moved objects — a can shifted 0.189 m by
    an arm reporting positive clearance."""
    for scene in ("FWDCenterLabMCC", "FWDCenterLabSivaPool"):
        ok, why = R.check_route("POINT", scene)
        assert not ok
        assert "not been validated in any scene" in why


def test_a_refusal_names_the_scene_not_the_arm():
    _, why = R.check_route("STOW_ROUTE", "SomeOtherScene")
    assert "SomeOtherScene" in why
    assert "arm is unavailable" not in why


def test_every_validation_row_carries_its_evidence():
    """A row without a flown run recorded would undo this module's whole
    safety argument in one line."""
    for row in R.ROUTE_COMPATIBILITY:
        assert row.evidence.strip()
        assert "tlh_motion-routine" in row.evidence


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
