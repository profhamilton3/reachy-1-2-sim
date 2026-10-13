"""#174 D6: what "the object was placed" means, and the placement's phases.

Offline and fast: every EpisodeResult here is constructed.  The verdicts of
flown placements come from `tests/integration/test_place_offline.py` and the
live check.
"""

import os
import sys

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco"):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

from reachy_ai.evaluation import contact_rules as CR  # noqa: E402
from reachy_ai.evaluation.panel_routes import (  # noqa: E402
    ABILITY_ROUTES, CONTACT_SAMPLES_KEY, PLACE_ABORTED_KEY, PLACE_ERROR_KEY,
    PLACE_TILT_KEY, RELEASE_DRAG_KEY, WITHDRAWAL_KEY, RouteType, evaluate,
    route_type,
)
from reachy_ai.experience.models import (  # noqa: E402
    EpisodeResult, EpisodeStatus, PanelRouteTaskSpec,
)
from reachy_ai.motion import rig_routes as R  # noqa: E402

P = CR.Phase
CUBE_START = [0.4318, 0.0, 0.77]
CUBE_END = [0.4318, 0.1524, 0.77]
CAN = [0.55, 0.95, 0.0609]
PLAN = {"preflight_passed": True, "plan_sha256": "cd" * 32, "rungs": 38}

#: A clean placement's contacts: pads on the cube from the grasp to the
#: release (the carry included), the cube on the table only when it is not
#: held -- standing before the grasp, set down, and after the release.
CLEAN = [
    ("r_gripper_thumb", "red_cube", P.GRASP), ("r_gripper_finger", "red_cube", P.GRASP),
    ("r_gripper_thumb", "red_cube", P.HOLD), ("r_gripper_finger", "red_cube", P.CARRY),
    ("r_gripper_thumb", "red_cube", P.SET_DOWN), ("r_gripper_finger", "red_cube", P.RELEASE),
    ("red_cube", "table_top", P.APPROACH), ("red_cube", "table_top", P.SET_DOWN),
    ("red_cube", "table_top", P.WITHDRAW),
]


def samples(rows=CLEAN, extra=()):
    return [CR.ContactSample(b1, b2, ph, None, 1, 0.0001).to_dict()
            for b1, b2, ph in list(rows) + list(extra)]


def a_place(*, error=0.004, tilt=0.2, withdrawal=1.0, posture="present",
            contacts=None, can_end=CAN, aborted=False, drag=0.003, drop=()):
    metrics = {PLACE_ERROR_KEY: error, PLACE_TILT_KEY: tilt,
               WITHDRAWAL_KEY: withdrawal, RELEASE_DRAG_KEY: drag}
    if aborted:
        metrics[PLACE_ABORTED_KEY] = 1.0
    for key in drop:
        metrics.pop(key, None)
    metrics = {k: v for k, v in metrics.items() if v is not None}
    summary = {} if contacts is False else {
        CONTACT_SAMPLES_KEY: samples() if contacts is None else contacts}
    return EpisodeResult(
        episode_id="ep", trial_id="tr", status=EpisodeStatus.SUCCEEDED,
        metrics=metrics, contact_summary=summary,
        final_object_states={"red_cube": {"pos_xyz": list(CUBE_END)},
                             "soda_can": {"pos_xyz": list(can_end)}},
        final_joint_positions_deg=dict(R.POSTURES[posture]) if posture else {})


def a_spec():
    return PanelRouteTaskSpec(
        task_id="panel:place_on_cell", task_type="place_on_cell",
        ability="place_on_cell", route="CRANE_PLACE", route_version=1,
        expected_end_posture="present", target_object_id="red_cube",
        initial_object_positions={"red_cube": list(CUBE_START), "soda_can": list(CAN)})


def place(result, plan=PLAN):
    return evaluate("place_on_cell", result, a_spec(), plan=plan)


def test_the_placement_flies_a_planned_route():
    assert ABILITY_ROUTES["place_on_cell"] == "CRANE_PLACE"
    assert route_type("CRANE_PLACE") is RouteType.PLANNED


def test_a_clean_placement_passes_and_completes_the_cycle():
    v = place(a_place())
    assert v.is_successful, v.explanation
    assert v.reported["full_cycle"] is True
    assert v.reported["placement_error_mm"] == 4.0
    assert v.reported["withdrawal"] == "completed"
    assert v.reported["others_moved_max_mm"] == 0.0


def test_d6_position_is_one_centimetre_from_the_cell_centre():
    assert place(a_place(error=0.0099)).is_successful
    v = place(a_place(error=0.0101))
    assert not v.is_successful
    assert any("10.10 mm from the target" in x.description for x in v.violations)


def test_d6_tilt_is_five_degrees():
    assert place(a_place(tilt=4.9)).is_successful
    assert not place(a_place(tilt=5.1)).is_successful


def test_d6_no_other_object_may_move_more_than_a_millimetre():
    moved = [CAN[0] + 0.0009, CAN[1], CAN[2]]
    assert place(a_place(can_end=moved)).is_successful
    v = place(a_place(can_end=[CAN[0] + 0.0012, CAN[1], CAN[2]]))
    assert not v.is_successful and not v.is_safe
    assert any("soda_can moved 1.20 mm" in x.description for x in v.violations)


def test_d6_no_pad_on_the_object_after_the_release():
    v = place(a_place(contacts=samples(extra=[("r_gripper_finger", "red_cube", P.WITHDRAW)])))
    assert not v.is_successful
    assert any("a pad touched the object after the release" in x.description
               for x in v.violations)
    assert v.reported["full_cycle"] is False


def test_the_held_object_on_the_table_during_the_carry_is_unintended():
    """The rule table (#172) forbids the target on its support while held;
    the carry is a held phase."""
    v = place(a_place(contacts=samples(extra=[("red_cube", "table_top", P.CARRY)])))
    assert not v.is_successful
    assert any("during carry" in x.description for x in v.violations)


def test_an_aborted_placement_did_not_place():
    v = place(a_place(aborted=True, error=0.152))
    assert not v.is_successful
    assert v.reported["aborted"] is True


def test_the_withdrawal_is_reported_not_judged():
    v = place(a_place(withdrawal=0.0, posture=None))
    assert v.is_successful, v.explanation
    assert v.reported["withdrawal"] == "refused"
    assert v.reported["full_cycle"] is False


def test_what_was_not_measured_cannot_pass():
    assert not place(a_place(drop=(PLACE_ERROR_KEY,))).is_valid
    assert not place(a_place(contacts=False)).is_valid
    v = place(a_place(), plan={"preflight_passed": False, "plan_sha256": "x"})
    assert not v.is_valid


def test_the_crane_names_its_placement_phases():
    assert CR.crane_phase("carry to r2c1") is P.CARRY
    assert CR.crane_phase("set-down at r2c1") is P.SET_DOWN
    assert CR.crane_phase("abort: carry back to where it was picked up") is P.CARRY
    assert CR.crane_phase("replace") is P.SET_DOWN          # the abort's set-down
