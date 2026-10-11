"""#172 D5/D7: what "the lift worked" means, and the planned route type.

Offline and fast: every EpisodeResult here is constructed.  The verdicts over
the ten recorded Phase 1.5 lifts come from `scripts/replay_lift_evaluation.py`.
"""

import copy
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco"):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

from reachy_ai.evaluation import contact_rules as CR  # noqa: E402
from reachy_ai.evaluation.base import EpisodeVerdict, ViolationKind  # noqa: E402
from reachy_ai.evaluation.panel_routes import (  # noqa: E402
    ABILITY_ROUTES, CONTACT_SAMPLES_KEY, LIFT_PUT_BACK_KEY, LIFT_RISE_KEY,
    LIFT_SLIP_KEY, WITHDRAWAL_KEY, PanelRoutePolicy, RouteType,
    check_plan_integrity, check_route_integrity, evaluate, route_type,
)
from reachy_ai.experience.models import (  # noqa: E402
    EpisodeResult, EpisodeStatus, PanelRouteTaskSpec,
)
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.recipe import PrimitiveStep, TrajectoryRecipe  # noqa: E402

P = CR.Phase
CUBE = [0.4318, 0.0, 0.7699]
CAN = [0.55, 0.95, 0.0609]
PLAN = {"preflight_passed": True, "plan_sha256": "ab" * 32, "rungs": 13}

#: The contacts a clean lift makes: pads on the cube from grasp to release,
#: the cube on the table while it is not held.
CLEAN = [
    ("r_gripper_thumb", "red_cube", P.GRASP), ("r_gripper_finger", "red_cube", P.GRASP),
    ("r_gripper_thumb", "red_cube", P.HOLD), ("r_gripper_finger", "red_cube", P.HOLD),
    ("r_gripper_thumb", "red_cube", P.SET_DOWN), ("r_gripper_finger", "red_cube", P.RELEASE),
    ("red_cube", "table_top", P.APPROACH), ("red_cube", "table_top", P.GRASP),
    ("red_cube", "table_top", P.SET_DOWN), ("red_cube", "table_top", P.WITHDRAW),
]


def samples(rows=CLEAN, extra=()):
    out = []
    for row in list(rows) + list(extra):
        b1, b2, phase = row[:3]
        pose = row[3] if len(row) > 3 else None
        route = row[4] if len(row) > 4 else ""
        depth = row[5] if len(row) > 5 else 0.0001
        out.append(CR.ContactSample(b1, b2, phase, pose, 1, depth, route).to_dict())
    return out


def a_lift(*, rise=0.050, slip=0.00002, put_back=0.004, withdrawal=1.0,
           posture="present", contacts=None, cube_end=None, can_end=CAN,
           drop=()):
    metrics = {LIFT_RISE_KEY: rise, LIFT_SLIP_KEY: slip,
               LIFT_PUT_BACK_KEY: put_back, WITHDRAWAL_KEY: withdrawal}
    for key in drop:
        metrics.pop(key, None)
    metrics = {k: v for k, v in metrics.items() if v is not None}
    summary = {} if contacts is False else {
        CONTACT_SAMPLES_KEY: samples() if contacts is None else contacts}
    return EpisodeResult(
        episode_id="ep", trial_id="tr", status=EpisodeStatus.SUCCEEDED,
        metrics=metrics, contact_summary=summary,
        final_object_states={
            "red_cube": {"pos_xyz": cube_end or [CUBE[0] + 0.004, CUBE[1], CUBE[2]]},
            "soda_can": {"pos_xyz": list(can_end)}},
        final_joint_positions_deg=dict(R.POSTURES[posture]) if posture else {})


def a_spec():
    return PanelRouteTaskSpec(
        task_id="panel:lift_object", task_type="lift_object",
        ability="lift_object", route="CRANE_LIFT", route_version=1,
        expected_end_posture="present", target_object_id="red_cube",
        initial_object_positions={"red_cube": list(CUBE), "soda_can": list(CAN)})


def lift(result, plan=PLAN, **kw):
    return evaluate("lift_object", result, a_spec(), plan=plan, **kw)


# ---------------------------------------------------------------------------
# D7: a planned route
# ---------------------------------------------------------------------------

def test_the_lift_flies_a_planned_route():
    assert ABILITY_ROUTES["lift_object"] == "CRANE_LIFT"
    assert route_type("CRANE_LIFT") is RouteType.PLANNED


def test_a_route_with_no_type_raises():
    with pytest.raises(KeyError, match="no type"):
        route_type("SOMERSAULT")


def test_a_recipe_may_not_vary_a_planned_route():
    recipe = TrajectoryRecipe(
        recipe_id="lift-attempt", task_type="lift_object", route="CRANE_LIFT",
        arm="right",
        primitive_sequence=[PrimitiveStep("waypoint", {"name": "HOVER"})])
    out = check_route_integrity(recipe, "CRANE_LIFT")
    assert len(out) == 1 and "planned route" in out[0].description
    v = lift(a_lift(), recipe=recipe)
    assert not v.is_valid and not v.is_successful
    assert "NOT THIS ROUTE" in v.explanation


def test_plan_integrity_is_preflight_passed_and_hash_recorded():
    assert check_plan_integrity("CRANE_LIFT", PLAN) == []
    assert check_plan_integrity("CRANE_LIFT", None) == []    # "not checked"
    no_preflight = dict(PLAN, preflight_passed=False)
    no_hash = dict(PLAN, plan_sha256="")
    short = dict(PLAN, plan_sha256="abc")
    for plan, word in ((no_preflight, "preflight"), (no_hash, "hash"),
                       (short, "hash")):
        out = check_plan_integrity("CRANE_LIFT", plan)
        assert len(out) == 1 and word in out[0].description
        assert out[0].kind is ViolationKind.INVALID_EPISODE


def test_a_lift_without_a_plan_record_says_it_was_not_checked():
    v = lift(a_lift(), plan=None)
    assert v.is_successful
    assert "plan's integrity" in v.explanation and "not checked" in v.explanation


def test_a_lift_with_a_bad_plan_record_is_invalid():
    v = lift(a_lift(), plan=dict(PLAN, plan_sha256=""))
    assert not v.is_valid and not v.is_successful


# ---------------------------------------------------------------------------
# D5: the five criteria
# ---------------------------------------------------------------------------

def test_a_full_cycle():
    v = lift(a_lift())
    assert v.is_valid and v.is_safe and v.is_successful
    assert v.reported["withdrawal"] == "completed"
    assert v.reported["full_cycle"] is True
    assert v.metrics["full_cycle"] == 1.0
    for line in ("RISE: 5.00 cm", "SLIP: 0.020 mm", "PUT BACK: 4.00 mm",
                 "ARRIVED: at present", "FULL CYCLE: yes"):
        assert line in v.explanation, v.explanation


@pytest.mark.parametrize("kw,word", [
    ({"rise": 0.0399}, "rose"),
    ({"slip": 0.00201}, "slipped"),
    ({"put_back": 0.00501}, "put back"),
])
def test_each_threshold_fails_the_lift_just_past_it(kw, word):
    v = lift(a_lift(**kw))
    assert v.is_valid and not v.is_successful
    assert any(word in x.description for x in v.violations
               if x.kind is ViolationKind.TASK_FAILURE)
    assert v.reported["full_cycle"] is False


@pytest.mark.parametrize("kw", [{"rise": 0.040}, {"slip": 0.002},
                                {"put_back": 0.005}])
def test_each_threshold_is_inclusive(kw):
    assert lift(a_lift(**kw)).is_successful


def test_the_thresholds_are_the_issues():
    p = PanelRoutePolicy()
    assert (p.lift_rise_required_m, p.lift_slip_max_m,
            p.lift_put_back_xy_tol_m) == (0.040, 0.002, 0.005)


@pytest.mark.parametrize("key", [LIFT_RISE_KEY, LIFT_SLIP_KEY, LIFT_PUT_BACK_KEY])
def test_an_unrecorded_measurement_is_invalid_not_zero(key):
    v = lift(a_lift(drop=(key,)))
    assert not v.is_valid and not v.is_successful
    assert key not in v.metrics


def test_a_completed_withdrawal_that_does_not_end_at_present_fails_the_lift():
    v = lift(a_lift(posture="rest"))
    assert not v.is_successful
    assert "DID NOT ARRIVE" in v.explanation


def test_another_object_moved_fails_and_the_target_is_excused():
    v = lift(a_lift(can_end=[CAN[0] + 0.05, CAN[1], CAN[2]]))
    assert not v.is_safe and not v.is_successful
    assert "soda_can moved" in v.explanation
    # The cube itself moved 4 mm (and could move more): judged by put-back.
    ok = lift(a_lift(cube_end=[CUBE[0] + 0.03, CUBE[1], CUBE[2]]))
    assert ok.is_successful


# ---------------------------------------------------------------------------
# Contacts: approach legs, the lift, the withdrawal (owner ruling 2026-10-10)
# ---------------------------------------------------------------------------

def test_an_unintended_contact_during_the_lift_fails_it():
    v = lift(a_lift(contacts=samples(extra=[("red_cube", "table_top", P.HOLD)])))
    assert not v.is_safe and not v.is_successful
    assert "UNINTENDED CONTACT" in v.explanation
    assert any(x.kind is ViolationKind.FORBIDDEN_CONTACT for x in v.violations)


def test_an_unrecorded_contact_record_is_invalid():
    v = lift(a_lift(contacts=False))
    assert not v.is_valid and not v.is_successful
    assert "cannot be ruled out" in v.explanation


def test_the_approach_leg_is_judged_under_its_own_route_and_reported():
    """RAISE_TO_SIDE from HOME lays the forearm on the board at REST.  That
    is not the lift: it is reported under its own route, outside the verdict."""
    leg = ("r_forearm", "table_top", P.APPROACH, "rest", "RAISE_TO_SIDE", 0.0004)
    v = lift(a_lift(contacts=samples(extra=[leg])))
    assert v.is_successful and v.reported["full_cycle"]
    found = v.reported["approach"]["RAISE_TO_SIDE"]
    assert len(found) == 1 and "r_forearm" in found[0]
    assert "APPROACH (RAISE_TO_SIDE" in v.explanation

    clean_leg = ("r_forearm", "table_top", P.APPROACH, "rest", "PLACE_ROUTE")
    v = lift(a_lift(contacts=samples(extra=[clean_leg])))
    assert v.reported["approach"] == {"PLACE_ROUTE": []}


def test_a_refused_withdrawal_is_a_lift_pass_and_not_a_full_cycle():
    v = lift(a_lift(withdrawal=0.0, posture=None))
    assert v.is_successful
    assert v.reported["withdrawal"] == "refused"
    assert v.reported["full_cycle"] is False
    assert v.reported["ends_at_present"] is False
    assert "ENDS AT PRESENT: not judged" in v.explanation
    assert not any(x.kind is ViolationKind.INVALID_EPISODE for x in v.violations)


def test_residual_pad_contact_after_release_is_the_withdrawals():
    """A1/B4: the finger stays on the replaced cube after the release.  The
    lift verdict ends at release; the withdrawal reports the residue."""
    residue = ("r_gripper_finger", "red_cube", P.WITHDRAW, None, "", 0.000202)
    v = lift(a_lift(withdrawal=0.0, posture=None,
                    contacts=samples(extra=[residue])))
    assert v.is_successful and v.is_safe
    assert v.reported["withdrawal_detail"] == "refused, residual pad contact 0.20 mm"
    assert len(v.reported["withdrawal_contacts"]) == 1


def test_residual_contact_after_a_completed_withdrawal_is_not_a_full_cycle():
    residue = ("r_gripper_thumb", "red_cube", P.WITHDRAW)
    v = lift(a_lift(contacts=samples(extra=[residue])))
    assert v.is_successful
    assert v.reported["withdrawal"] == "completed"
    assert v.reported["full_cycle"] is False


def test_a_lift_that_never_reached_the_withdrawal_judges_present():
    v = lift(a_lift(withdrawal=None, posture=None))
    assert v.reported["withdrawal"] == "not reached"
    assert not v.is_successful


def test_the_verdict_round_trips_with_what_it_reported():
    v = lift(a_lift(withdrawal=0.0, posture=None))
    back = EpisodeVerdict.from_dict(copy.deepcopy(v.to_dict()))
    assert back.reported == v.reported
    old = v.to_dict()
    old.pop("reported")
    assert EpisodeVerdict.from_dict(old).reported == {}
