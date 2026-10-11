"""#172 option C: one contact rule table, by role and phase, for every ability.

Offline and fast: contacts here are constructed records.  The flights that
produce real ones are in `tests/integration/test_panel_ability_recipes.py`.
"""

import os
import re
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco"):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

from reachy_ai.evaluation import contact_rules as CR  # noqa: E402
from reachy_ai.evaluation.contact_rules import (  # noqa: E402
    ContactSample, Phase, Role,
)

_REPO = os.path.abspath(os.path.join(_HERE, "../.."))
P = Phase


def judged(b1, b2, phase, *, pose=None, route="", target="red_cube",
           sample_route=""):
    return CR.judge(ContactSample(b1, b2, phase, pose, route=sample_route),
                    route=route, target_id=target)


# ---------------------------------------------------------------------------
# Roles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("body,role", [
    ("r_gripper_thumb", Role.PAD), ("r_gripper_finger", Role.PAD),
    ("r_forearm", Role.ARM_LINK), ("r_upper_arm", Role.ARM_LINK),
    ("l_forearm", Role.ARM_LINK), ("torso", Role.ARM_LINK),
    ("head", Role.ARM_LINK),
    ("red_cube", Role.TARGET), ("table_top", Role.SUPPORT),
    ("rig_rail_inner_right", Role.RIG), ("rig_rail_back", Role.RIG),
    ("pedestal", Role.RIG), ("world", Role.RIG),
    ("soda_can", Role.OTHER_OBJECT), ("pool_box_1", Role.OTHER_OBJECT),
    ("something_new", Role.OTHER_OBJECT),
])
def test_each_body_plays_one_role(body, role):
    assert CR.role_of(body, target_id="red_cube") is role


def test_the_target_is_whichever_object_the_ability_acts_on():
    assert CR.role_of("soda_can", target_id="soda_can") is Role.TARGET
    assert CR.role_of("red_cube", target_id="soda_can") is Role.OTHER_OBJECT
    assert CR.role_of("red_cube") is Role.OTHER_OBJECT


# ---------------------------------------------------------------------------
# The table
# ---------------------------------------------------------------------------

GRASP_TO_RELEASE = (P.GRASP, P.HOLD, P.CARRY, P.SET_DOWN, P.RELEASE)


@pytest.mark.parametrize("phase", list(Phase))
def test_pads_on_the_target_only_from_grasp_to_release(phase):
    for pad in ("r_gripper_thumb", "r_gripper_finger"):
        j = judged(pad, "red_cube", phase)
        assert j.allowed is (phase in GRASP_TO_RELEASE), j.why


@pytest.mark.parametrize("phase", list(Phase))
def test_the_target_on_its_support_except_while_held(phase):
    """Standing there, being lifted off, being set down, standing again: yes.
    Dragged along the table while held (hold, carry): no."""
    j = judged("red_cube", "table_top", phase)
    assert j.allowed is (phase not in (P.HOLD, P.CARRY)), j.why


@pytest.mark.parametrize("phase", list(Phase))
def test_an_arm_link_on_the_board_is_unintended_without_an_exception(phase):
    j = judged("r_forearm", "table_top", phase, pose="rest", route="WAVE")
    assert not j.allowed
    assert "declared exception" in j.why


@pytest.mark.parametrize("pair", [
    ("r_forearm", "rig_rail_inner_right"), ("r_gripper_thumb", "rig_rail_back"),
    ("r_gripper_finger", "table_top"), ("r_upper_arm", "red_cube"),
    ("r_gripper_thumb", "soda_can"), ("red_cube", "soda_can"),
    ("red_cube", "rig_rail_back"), ("l_forearm", "table_top"),
])
@pytest.mark.parametrize("phase", list(Phase))
def test_everything_else_is_unintended(pair, phase):
    j = judged(*pair, phase)
    assert j is not None and not j.allowed, (pair, phase)


@pytest.mark.parametrize("pair", [
    ("soda_can", "world"), ("pool_box_1", "table_top"), ("table_top", "world"),
    ("r_forearm", "torso"),
])
def test_the_world_at_rest_and_self_contact_are_not_judged(pair):
    """Neither side is the robot or the target (or both are the robot): not
    the ability's doing; drift and self-collision ask about those."""
    assert judged(*pair, P.HOLD) is None


def test_the_table_is_shared_and_unchanged_by_route():
    """One table for every ability: the same contact in the same phase gets
    the same answer whatever route is named, unless an exception applies."""
    for route in ("CRANE_LIFT", "WAVE", "POINT", ""):
        assert judged("r_gripper_thumb", "red_cube", P.HOLD, route=route).allowed
        assert not judged("red_cube", "table_top", P.CARRY, route=route).allowed


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

def test_the_exception_list_is_short_and_every_entry_says_why():
    assert len(CR.ROUTE_EXCEPTIONS) <= 3
    for e in CR.ROUTE_EXCEPTIONS:
        assert e.reason and len(e.reason) > 20, e


@pytest.mark.parametrize("route", ["PLACE_ROUTE", "STOW_ROUTE"])
def test_forearm_on_the_board_at_rest_for_rest_and_stow(route):
    j = judged("r_forearm", "table_top", CR.MEASURED_ROUTE_PHASES[route],
               pose="rest", route=route)
    assert j.allowed
    assert j.why.startswith("route exception:")


@pytest.mark.parametrize("route", ["PLACE_ROUTE", "STOW_ROUTE"])
def test_the_exception_is_that_link_at_that_pose_only(route):
    phase = CR.MEASURED_ROUTE_PHASES[route]
    assert not judged("r_forearm", "table_top", phase, pose=None, route=route).allowed
    assert not judged("r_forearm", "table_top", phase, pose="home", route=route).allowed
    assert not judged("r_upper_arm", "table_top", phase, pose="rest", route=route).allowed
    assert not judged("r_forearm", "rig_rail_back", phase, pose="rest", route=route).allowed


def test_no_exception_was_given_to_the_lift_or_its_approach():
    """Owner ruling (2026-10-10): the lift's approach is judged under its own
    route, and the lift route gets no exception for the forearm at REST."""
    for route in ("CRANE_LIFT", "RAISE_TO_SIDE"):
        assert not CR.exceptions_for(route)
        assert not judged("r_forearm", "table_top", P.APPROACH, pose="rest",
                          route=route).allowed


def test_a_sample_is_judged_under_the_route_it_was_flown_on():
    """A leg flown inside another ability carries its own route, and that
    route's exceptions, not the ability's."""
    on_place = judged("r_forearm", "table_top", P.APPROACH, pose="rest",
                      route="CRANE_LIFT", sample_route="PLACE_ROUTE")
    assert on_place.allowed
    on_raise = judged("r_forearm", "table_top", P.APPROACH, pose="rest",
                      route="CRANE_LIFT", sample_route="RAISE_TO_SIDE")
    assert not on_raise.allowed


# ---------------------------------------------------------------------------
# Phases the abilities tag
# ---------------------------------------------------------------------------

def _crane_segments():
    src = open(os.path.join(_REPO, "src/reachy_ai/tasks/crane_pick_live.py")).read()
    return re.findall(r'ev\("phase", name=f?"([^"]+)"', src)


def test_every_crane_segment_is_a_phase():
    """The crane tags its phases with the segments it already names; a new
    segment nobody mapped would be judged as nothing."""
    names = _crane_segments()
    assert len(names) >= 10
    for name in names:
        assert CR.crane_phase(name) is not None, name


@pytest.mark.parametrize("segment,phase", [
    ("transit PRESENT -> via", P.APPROACH), ("descent", P.APPROACH),
    ("HOVER pause (10 s)", P.APPROACH), ("close", P.GRASP), ("lift", P.GRASP),
    ("hold (2 s)", P.HOLD), ("replace", P.SET_DOWN), ("withdrawal", P.WITHDRAW),
    ("back at PRESENT", P.WITHDRAW),
])
def test_the_crane_segments_map_onto_the_shared_phases(segment, phase):
    assert CR.crane_phase(segment) is phase


def test_every_measured_route_has_a_phase():
    for route in ("PLACE_ROUTE", "STOW_ROUTE", "WAVE", "POINT"):
        assert route in CR.MEASURED_ROUTE_PHASES


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def test_aggregate_folds_per_pair_phase_pose_and_route_keeping_the_deepest():
    raw = [("table_top", "r_forearm", P.APPROACH, "rest", -0.0002),
           ("r_forearm", "table_top", P.APPROACH, "rest", -0.0005),
           ("r_forearm", "table_top", P.APPROACH, None, 0.0),
           ("r_forearm", "table_top", P.APPROACH, "rest", -0.0001, "RAISE_TO_SIDE")]
    out = CR.aggregate(raw)
    assert len(out) == 3
    rest = next(s for s in out if s.pose == "rest" and not s.route)
    assert (rest.body1, rest.body2, rest.samples) == ("r_forearm", "table_top", 2)
    assert rest.deepest_m == pytest.approx(0.0005)
    assert any(s.route == "RAISE_TO_SIDE" for s in out)


def test_a_sample_round_trips():
    s = ContactSample("r_gripper_finger", "red_cube", P.WITHDRAW, None, 30,
                      0.000115, "")
    assert ContactSample.from_dict(s.to_dict()) == s
    old = {"body1": "a", "body2": "b", "phase": "hold"}   # before `route`
    assert ContactSample.from_dict(old).route == ""


def test_describe_names_the_roles_phase_route_pose_and_depth():
    j = judged("r_forearm", "table_top", P.APPROACH, pose="rest",
               sample_route="RAISE_TO_SIDE")
    text = j.describe()
    for word in ("r_forearm (arm_link)", "table_top (support)", "approach",
                 "RAISE_TO_SIDE", "REST", "mm deep"):
        assert word in text
