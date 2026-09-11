"""Issue #91: success defined for the five abilities, and a wall around the route.

Offline and fast: every EpisodeResult here is constructed, not simulated, so
these run without MuJoCo.  The end-to-end flights live in
`tests/integration/test_panel_ability_recipes.py`.

The headline tests are the ones that fail the build if a recipe could reorder
a corridor, drop a waypoint, loosen a tolerance, stop guarding a joint, swing
wider than the measurement or wave for longer than it — and still be scored.
"""

import copy
import math
import os
import sys

import pytest

_HERE = os.path.dirname(__file__)
for _p in ("../../src", "../../native_mujoco", "../../web"):
    _abs = os.path.join(_HERE, _p)
    if _abs not in sys.path:
        sys.path.insert(0, _abs)

from reachy_ai.evaluation.base import ViolationKind  # noqa: E402
from reachy_ai.evaluation.panel_routes import (  # noqa: E402
    ABILITY_ROUTES, CONTACT_BODIES_KEY, GRID_CELL_M, EVALUATORS,
    PanelRoutePolicy, align_to_parameters, canonical_steps,
    check_route_integrity, evaluate,
)
from reachy_ai.experience.models import (  # noqa: E402
    EpisodeResult, EpisodeStatus, PanelRouteTaskSpec,
)
from reachy_ai.motion import rig_routes as R  # noqa: E402
from reachy_ai.motion.recipe import TrajectoryRecipe  # noqa: E402

_REPO = os.path.abspath(os.path.join(_HERE, "../.."))
_RECIPES = os.path.join(_REPO, "recipes", "panel")


def load(ability):
    return TrajectoryRecipe.load(
        os.path.join(_RECIPES, f"{ability}_baseline_v1.yaml"))


def mutated(ability, mutate):
    r = TrajectoryRecipe.from_dict(copy.deepcopy(load(ability).to_dict()))
    mutate(r)
    return r


def at(posture):
    """Final joint positions standing at a named posture."""
    return dict(R.POSTURES[posture])


def a_result(posture="home", *, objects=None, touched=None, metrics=None,
             status=EpisodeStatus.SUCCEEDED):
    return EpisodeResult(
        episode_id="ep", trial_id="tr", status=status,
        success=(status == EpisodeStatus.SUCCEEDED),
        metrics=dict(metrics or {}),
        contact_summary={"forbidden_total": 0,
                         CONTACT_BODIES_KEY: dict(touched or {})},
        final_object_states={
            oid: {"object_id": oid, "pos_xyz": list(pos)}
            for oid, pos in (objects or {}).items()},
        final_joint_positions_deg=at(posture) if posture else {},
    )


def a_spec(ability, *, initial=None, intended=(), cycles=0):
    return PanelRouteTaskSpec(
        task_id=f"panel:{ability}", task_type=ability, ability=ability,
        route=ABILITY_ROUTES[ability], route_version=1,
        initial_object_positions=dict(initial or {}),
        intended_contact_bodies=list(intended),
        expected_wave_cycles=cycles)


# ---------------------------------------------------------------------------
# Every ability has a recipe and an evaluator
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ability", sorted(ABILITY_ROUTES))
def test_every_ability_has_a_recipe(ability):
    r = load(ability)
    assert r.task_type == ability
    assert r.route == ABILITY_ROUTES[ability]
    assert r.arm == "right"
    assert r.expected_start_posture


@pytest.mark.parametrize("ability", sorted(ABILITY_ROUTES))
def test_every_ability_has_an_evaluator(ability):
    assert ability in EVALUATORS


def test_the_registry_and_the_evaluators_agree_on_which_abilities_exist():
    """A route the panel flies with no evaluator is an ability nobody has
    defined success for, and a search pointed at it would optimise a default
    verdict — a number with no meaning attached."""
    import panel_abilities  # noqa: E402  (web/ is on the path via the recipes test)

    flown = {name for name, a in panel_abilities.REGISTRY.items() if a.route}
    assert flown == set(ABILITY_ROUTES) == set(EVALUATORS)


def test_an_unknown_ability_raises_rather_than_scoring():
    with pytest.raises(KeyError) as e:
        evaluate("somersault", a_result(), a_spec("wave"))
    assert "success has not been defined" in str(e.value)


# ---------------------------------------------------------------------------
# Baselines are variations of the measured route, and are pinned to it
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ability", sorted(ABILITY_ROUTES))
def test_the_baseline_recipe_is_the_measured_route(ability):
    assert check_route_integrity(load(ability), ABILITY_ROUTES[ability]) == []


@pytest.mark.parametrize("ability,route", sorted(ABILITY_ROUTES.items()))
def test_the_recipe_file_still_matches_rig_routes(ability, route):
    """The files are GENERATED from `rig_routes` and this is what keeps them
    that way: edit a waypoint in the module the arm actually flies and the
    recipe that claims to be a variation of it stops being one."""
    steps = load(ability).primitive_sequence
    canonical = canonical_steps(route)
    assert len(steps) == len(canonical)
    for got, want in zip(steps, canonical):
        assert got.primitive == want.primitive
        assert got.parameters["name"] == want.name
        assert math.isclose(float(got.parameters["seconds"]), want.seconds)
        assert math.isclose(float(got.parameters["tol"]), want.tol)


# ---------------------------------------------------------------------------
# The wall: what a search may not reach a better score by doing
# ---------------------------------------------------------------------------

def _reorder(r):
    r.primitive_sequence.insert(0, r.primitive_sequence.pop(3))


def _drop(r):
    r.primitive_sequence.pop(4)


def _loosen(r):
    r.primitive_sequence[2].parameters["tol"] = 45.0


def _ungurad(r):
    r.primitive_sequence[2].parameters["guard"] = ["r_shoulder_pitch"]


def _no_guard_at_all(r):
    r.primitive_sequence[2].parameters["guard"] = []


@pytest.mark.parametrize("mutate,expect", [
    (_reorder, "ORDER"),
    (_drop, "dropped from a measured"),
    (_loosen, "loosens the tracking tolerance"),
    (_ungurad, "stops guarding"),
    (_no_guard_at_all, "stops guarding"),
])
def test_a_recipe_that_mutates_the_corridor_is_rejected(mutate, expect):
    violations = check_route_integrity(mutated("stow_arm", mutate), "STOW_ROUTE")
    assert violations, "the corridor geometry is the safety argument, not a knob"
    assert any(expect in v.description for v in violations)
    assert all(v.severity == "hard" for v in violations)


def test_a_mutated_recipe_makes_the_episode_invalid_not_merely_unsuccessful():
    """A result from a recipe that reordered the corridor is not a measurement
    of that corridor, so reporting its accuracy would report some other
    motion's."""
    verdict = evaluate(
        "stow_arm",
        a_result("home", objects={}),
        a_spec("stow_arm", initial={}),
        mutated("stow_arm", _reorder))
    assert not verdict.is_valid
    assert not verdict.is_successful
    assert any(v.kind is ViolationKind.INVALID_EPISODE for v in verdict.violations)
    assert "NOT THIS ROUTE" in verdict.explanation


def test_a_tighter_tolerance_is_allowed():
    """The rule is one-way: a search may make a check stricter."""
    def tighten(r):
        r.primitive_sequence[2].parameters["tol"] = 3.0
    assert check_route_integrity(mutated("stow_arm", tighten), "STOW_ROUTE") == []


def test_an_extra_guarded_joint_is_allowed():
    def add(r):
        canonical = canonical_steps("STOW_ROUTE")[2].guard
        r.primitive_sequence[2].parameters["guard"] = list(canonical) + ["r_gripper"]
    assert check_route_integrity(mutated("stow_arm", add), "STOW_ROUTE") == []


def test_a_guard_the_recipe_does_not_mention_inherits_the_route_s():
    """None and [] are different: None is "the recipe did not say", which takes
    the measured guard; [] is "guard nothing", which is a dropped check."""
    def silent(r):
        r.primitive_sequence[2].parameters.pop("guard", None)
    assert check_route_integrity(mutated("stow_arm", silent), "STOW_ROUTE") == []


# ---------------------------------------------------------------------------
# The measured envelope
# ---------------------------------------------------------------------------

def test_the_wave_may_not_run_more_cycles_than_were_measured():
    def more(r):
        r.bounded_parameters["wave_cycles"]["value"] = R.WAVE_CYCLES + 2
    v = check_route_integrity(align_to_parameters(mutated("wave", more)), "WAVE")
    assert any("were measured" in x.description for x in v)


def test_the_wave_may_not_swing_wider_than_were_measured():
    def wider(r):
        r.bounded_parameters["wave_amplitude_deg"]["value"] = 80.0
    v = check_route_integrity(mutated("wave", wider), "WAVE")
    assert any("clearance argument" in x.description for x in v)


def test_the_wave_may_not_swing_faster_than_were_measured():
    """At 0.9 s the three weak joints reversing together finished tens of
    degrees short and tripped the tracking guard."""
    def faster(r):
        r.bounded_parameters["wave_seconds"]["value"] = 0.9
    v = check_route_integrity(mutated("wave", faster), "WAVE")
    assert any("tripped the tracking guard" in x.description for x in v)


def test_the_wave_may_run_fewer_cycles_and_swing_slower():
    def gentler(r):
        r.bounded_parameters["wave_cycles"]["value"] = 1
        r.bounded_parameters["wave_seconds"]["value"] = 2.6
        r.bounded_parameters["wave_amplitude_deg"]["value"] = 40.0
    assert check_route_integrity(
        align_to_parameters(mutated("wave", gentler)), "WAVE") == []


@pytest.mark.parametrize("key,floor", [
    ("point_clearance_m", R.POINT_CLEARANCE),
    ("point_hover_floor_m", R.POINT_HOVER_FLOOR),
    ("point_margin_m", R.POINT_MARGIN),
])
def test_pointing_clearances_may_be_tightened_and_never_loosened(key, floor):
    def looser(r):
        r.bounded_parameters[key]["value"] = floor / 2.0
    v = check_route_integrity(mutated("point_cell", looser), "POINT")
    assert any("stricter and never looser" in x.description for x in v)

    def stricter(r):
        r.bounded_parameters[key]["value"] = floor * 1.5
    assert check_route_integrity(mutated("point_cell", stricter), "POINT") == []


def test_fewer_approach_legs_measures_the_arm_in_fewer_places():
    """Refused twice over, and both refusals are the point: the step list no
    longer matches the route, AND the count is under the measured minimum.

    The legs are the path-clearance check — the guard models the arm between
    two poses as a joint-space straight line, which is the assumption that
    failed when cell_r2c1 reported +5.5 cm and still moved the can 0.189 m,
    because the guard measured the ENDPOINTS and the can was hit in the
    middle."""
    def fewer(r):
        r.bounded_parameters["point_legs"]["value"] = 2

    # The step list still carries six approaches, so the shapes disagree.
    v = check_route_integrity(mutated("point_cell", fewer), "POINT")
    assert any("step(s) and the recipe has" in x.description for x in v)

    # And with the list shortened to match, the floor still refuses it.
    def fewer_and_shorter(r):
        fewer(r)
        del r.primitive_sequence[:4]
    v = check_route_integrity(mutated("point_cell", fewer_and_shorter), "POINT")
    assert any("measured minimum" in x.description for x in v)


def test_aligning_to_parameters_cannot_smuggle_a_mutation_through():
    """`align_to_parameters` regenerates the step list FROM the measured route,
    so a recipe that reordered the corridor does not survive alignment."""
    aligned = align_to_parameters(mutated("wave", _reorder))
    assert check_route_integrity(aligned, "WAVE") == []
    assert [s.parameters["name"] for s in aligned.primitive_sequence] == [
        s.name for s in canonical_steps("WAVE")]


# ---------------------------------------------------------------------------
# Success, ability by ability
# ---------------------------------------------------------------------------

def test_rest_succeeds_when_it_arrives_touching_only_the_table():
    v = evaluate("rest_forearm",
                 a_result("rest", objects={"can": (0.3, 0.0, 0.8)},
                          touched={"table_top": 400}),
                 a_spec("rest_forearm", initial={"can": (0.3, 0.0, 0.8)},
                        intended=("table_top",)),
                 load("rest_forearm"))
    assert v.is_successful
    assert "CONTACT: table_top, all intended." in v.explanation
    # The precaution that does not exist is named in every verdict it writes.
    assert "#82" in v.explanation


def test_a_verdict_never_reports_a_contact_that_did_not_happen():
    """The line used to print the ALLOWED set, so a route that touched nothing
    reported "CONTACT: only table_top, which this route intends" — a sentence
    about a contact that never occurred, in the summary written to be read
    instead of the raw episode."""
    v = evaluate("rest_forearm",
                 a_result("rest", objects={}, touched={}),
                 a_spec("rest_forearm", initial={}, intended=("table_top",)),
                 load("rest_forearm"))
    assert "CONTACT: none." in v.explanation
    assert "table_top" not in v.explanation


def test_rest_fails_when_it_touches_something_it_did_not_intend():
    v = evaluate("rest_forearm",
                 a_result("rest", objects={"can": (0.3, 0.0, 0.8)},
                          touched={"table_top": 400, "outer_rail": 12}),
                 a_spec("rest_forearm", initial={"can": (0.3, 0.0, 0.8)},
                        intended=("table_top",)),
                 load("rest_forearm"))
    assert not v.is_safe
    assert "outer_rail" in v.explanation


def test_an_unrecorded_contact_is_not_an_intended_one():
    """The runner counts fixture contacts and reports a total.  For rest that
    total includes the tabletop, which is the task — so a result with a count
    and no per-body record cannot be told apart, and says so."""
    result = a_result("rest", objects={}, touched=None)
    result.contact_summary = {"forbidden_total": 5}
    v = evaluate("rest_forearm", result,
                 a_spec("rest_forearm", initial={}, intended=("table_top",)),
                 load("rest_forearm"))
    assert not v.is_safe
    assert "cannot be told from" in v.explanation


def test_stow_fails_when_it_ends_anywhere_but_the_pocket():
    v = evaluate("stow_arm", a_result("rest", objects={}),
                 a_spec("stow_arm", initial={}), load("stow_arm"))
    assert not v.is_successful
    assert "DID NOT ARRIVE" in v.explanation


def test_a_disturbed_board_fails_however_well_the_route_flew():
    v = evaluate("stow_arm",
                 a_result("home", objects={"can": (0.30, 0.19, 0.8)}),
                 a_spec("stow_arm", initial={"can": (0.30, 0.0, 0.8)}),
                 load("stow_arm"))
    assert not v.is_safe
    assert "moved 19.0 cm" in v.explanation


def test_the_drift_threshold_is_the_rig_s_one_number():
    """The live executor's post-move check, the episode recorder and these
    evaluators all answer "did this move anything?", and three literals is how
    they come to disagree with what the operator was told."""
    import panel_episodes

    assert PanelRoutePolicy().object_drift_tolerance_m == R.OBJECT_DRIFT_TOL
    assert panel_episodes.DRIFT_TOLERANCE_M == R.OBJECT_DRIFT_TOL


def test_an_episode_with_no_starting_positions_cannot_say_the_board_was_clean():
    """"Nothing moved" is a comparison, and the result records only where
    things ended."""
    v = evaluate("stow_arm", a_result("home", objects={"can": (0.3, 0, 0.8)}),
                 a_spec("stow_arm", initial=None), load("stow_arm"))
    assert not v.is_valid
    assert "cannot be told" in v.explanation


def test_an_episode_with_no_final_pose_cannot_say_the_arm_arrived():
    v = evaluate("stow_arm", a_result(posture=None, objects={}),
                 a_spec("stow_arm", initial={}), load("stow_arm"))
    assert not v.is_valid
    assert "final pose was not recorded" in v.explanation


def test_the_wave_is_scored_on_cycles_actually_completed():
    v = evaluate("wave",
                 a_result("present", objects={},
                          metrics={"wave_cycles_completed": 1}),
                 a_spec("wave", initial={}, cycles=3), load("wave"))
    assert not v.is_successful
    assert "CYCLES: 1 of 3" in v.explanation
    assert v.ranking_scores["accuracy_score"] == pytest.approx(1 / 3)


def test_a_wave_that_finishes_off_the_presentation_pose_fails():
    """One run completed its cycles and left the shoulder far enough out that
    `posture_of` no longer recognised PRESENT, and the next request refused for
    want of a posture to start from."""
    result = a_result("present", objects={},
                      metrics={"wave_cycles_completed": 3})
    result.final_joint_positions_deg["r_shoulder_pitch"] += 25.0
    v = evaluate("wave", result, a_spec("wave", initial={}, cycles=3),
                 load("wave"))
    assert not v.is_successful
    assert "DID NOT ARRIVE" in v.explanation


def test_pointing_is_scored_against_half_a_grid_cell():
    """It is a hover, not a calibrated ray — the executor says so, and the
    current miss over an empty board is 2 to 9 cm."""
    policy = PanelRoutePolicy()
    assert policy.hover_miss_tolerance_m == pytest.approx(GRID_CELL_M / 2)

    near = evaluate("point_cell",
                    a_result("present", objects={},
                             metrics={"pad_miss_m": 0.03,
                                      "worst_clearance_m": 0.08}),
                    a_spec("point_cell", initial={}), load("point_cell"))
    assert near.is_successful

    far = evaluate("point_cell",
                   a_result("present", objects={},
                            metrics={"pad_miss_m": 0.09,
                                     "worst_clearance_m": 0.08}),
                   a_spec("point_cell", initial={}), load("point_cell"))
    assert not far.is_successful
    assert any("over the next cell as readily as this one" in x.description
               for x in far.violations)


def test_an_unmeasured_clearance_is_not_a_clearance():
    """cell_r2c1 reported +5.5 cm and still moved the can 0.189 m; "nobody
    measured it" and "it was clear" are the two things this ability's history
    most needs kept apart."""
    v = evaluate("point_cell",
                 a_result("present", objects={}, metrics={"pad_miss_m": 0.01}),
                 a_spec("point_cell", initial={}), load("point_cell"))
    assert not v.is_safe
    assert "CLEARANCE: not recorded" in v.explanation
    assert any("is not 'it was clear'" in x.description for x in v.violations)


def test_a_verdict_names_the_ability_it_judged():
    v = evaluate("wave", a_result("present", objects={},
                                  metrics={"wave_cycles_completed": 3}),
                 a_spec("wave", initial={}, cycles=3), load("wave"))
    assert v.task_type == "wave"
    assert v.policy_version == PanelRoutePolicy().policy_version


def test_the_policy_says_which_policy_it_is():
    """`policy_version` is an integer counted per policy, so "version 1" alone
    is ambiguous between this policy and the base one."""
    d = PanelRoutePolicy().to_dict()
    assert d["policy_scope"] == "panel_routes"
    assert "object_drift_tolerance_m" in d


def test_an_integer_parameter_is_recorded_as_the_integer_that_was_flown():
    """A continuous sampler hands back 1.6066 for a count of cycles.  The arm
    waves once; a recipe left declaring 1.6066 tells a later reader the trial
    ran one-and-a-bit cycles, which is not a thing that happened."""
    r = TrajectoryRecipe.from_dict(copy.deepcopy(load("wave").to_dict()))
    r.bounded_parameters["wave_cycles"]["value"] = 1.6066

    aligned = align_to_parameters(r)
    assert aligned.bounded_parameters["wave_cycles"]["value"] == 1
    # One cycle is two swings and a return.
    assert len(aligned.primitive_sequence) == 3
    assert check_route_integrity(aligned, "WAVE") == []


def test_aligning_does_not_mutate_the_recipe_it_was_given():
    r = load("wave")
    before = copy.deepcopy(r.to_dict())
    align_to_parameters(r)
    assert r.to_dict() == before


# ---------------------------------------------------------------------------
# Review of #99: the placement, the legs, and measurements nobody made
# ---------------------------------------------------------------------------

def test_more_approach_legs_is_a_longer_route_not_a_declaration():
    """The legs ARE the path-clearance check.  `canonical_steps` used to take
    `point_legs` and return the same three steps whatever it was, so a recipe
    could declare fourteen legs, carry one approach step, and pass."""
    six = canonical_steps("POINT", point_legs=6)
    ten = canonical_steps("POINT", point_legs=10)
    assert len(six) == 6 + 2       # approaches, then hover and return
    assert len(ten) == 10 + 2
    assert [s.primitive for s in six[:6]] == ["point_approach"] * 6


def test_a_recipe_claiming_more_legs_than_it_carries_is_refused():
    def claim(r):
        r.bounded_parameters["point_legs"]["value"] = 12
    v = check_route_integrity(mutated("point_cell", claim), "POINT")
    assert v, "a leg count that shapes nothing is not a clearance check"
    assert any("step(s) and the recipe has" in x.description for x in v)


def test_aligning_keeps_the_duration_the_search_chose():
    """Rebuilding every step from the canonical route hard-coded 1.8 s onto
    each swing, so a winner searched to 3.0 s exported a file declaring 1.8 —
    and anyone reading the export as the flown motion got the wrong number."""
    r = TrajectoryRecipe.from_dict(copy.deepcopy(load("wave").to_dict()))
    r.bounded_parameters["wave_seconds"]["value"] = 3.0

    aligned = align_to_parameters(r)
    for step in aligned.primitive_sequence:
        assert float(step.parameters["seconds"]) == pytest.approx(3.0)
    assert check_route_integrity(aligned, "WAVE") == []


def test_an_unmeasured_point_records_no_measurement():
    """Writing 0.0 for an unrecorded miss records the BEST POSSIBLE one, and
    0.0 for an unrecorded clearance records contact — both a measurement, in
    the same verdict that just said no measurement exists."""
    v = evaluate("point_cell", a_result("present", objects={}),
                 a_spec("point_cell", initial={}), load("point_cell"))
    assert "pad_miss_m" not in v.metrics
    assert "worst_clearance_m" not in v.metrics
    assert not v.is_valid


def test_the_policy_promises_nothing_it_does_not_check():
    """The offline executor streams open-loop — it reads no joint positions,
    so it cannot tell whether a waypoint converged.  A tracking tolerance on
    the policy was a knob nothing read, beside a success sentence claiming it
    was applied."""
    assert not hasattr(PanelRoutePolicy(), "tracking_tolerance_deg")
    import reachy_ai.evaluation.panel_routes as mod
    assert "inside the tracking tolerance" not in mod.__doc__
