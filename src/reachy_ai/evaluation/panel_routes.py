"""What "it worked" means for the five abilities the panel flies (#91).

Epic 8's evaluators cover two task types — the control panel and pick-and-place
— and neither describes a measured rig route.  `rest_forearm`, `stow_arm`,
`wave`, `point_cell` and `point_object` had no evaluator at all, so none of the
existing search could be pointed at them.  This module is that evaluator.

SUCCESS IS DEFINED HERE, NOT IN A COMMENT
-----------------------------------------
Each `evaluate_*` states the rule it applies in the verdict's explanation, in
the terms it applied it in, so a reader of a stored trial can see what was
actually asked of the motion:

  * **Rest** reaches the supported posture, touches only what it meant to
    touch, and leaves the board undisturbed.
  * **Stow** reaches the pocket by walking the corridor, and touches nothing.
  * **Wave** completes the cycles it was asked for and finishes recognisably
    at the presentation pose.

    NOT "within the tracking tolerance", which is what this said until a
    reviewer asked which code checked it.  The offline executor streams a
    pose sequence open-loop: it reads no joint positions, so it cannot tell
    whether a waypoint converged, and there is no per-waypoint tracking check
    in this path at all.  The live route runner does re-stream and re-check;
    these evaluators judge the ENDPOINT and the board.  A criterion nothing
    enforces is worse than an absent one, because the verdict reads as though
    it was checked.
  * **Point** reaches the hover region it selected, keeps its clearance, and
    leaves the board undisturbed.

THE POINT EVALUATOR MEASURES WHAT THE ABILITY CLAIMS
----------------------------------------------------
The ability hovers over a cell.  It is not a calibrated ray and the executor
already says so out loud — the miss over an empty board is 2 to 9 cm.  So the
tolerance here is HALF A GRID CELL (0.0635 m of a measured 0.127 m cell):
pointing at a cell means being over that cell rather than over its neighbour.
Scoring it against a millimetre would fail every run the ability has ever made
and measure a promise nobody made; scoring it against 20 cm would pass a hover
over the wrong square.

WHAT THE SEARCH MAY VARY, AND WHAT IT MAY NOT
---------------------------------------------
`check_route_integrity` is the wall.  A recipe may vary segment durations, the
wave's amplitude and cycle count, and the pointing clearances — and it may only
ever make a clearance STRICTER, never looser.  It may not reorder a route,
insert or drop a waypoint, loosen a waypoint tolerance, drop a guarded joint,
swing wider than the measured amplitude, or wave for more cycles than were
measured.  The corridor geometry is the safety argument, not a hyperparameter,
and a search that can reach a better score by removing a check will find that
before it finds a better motion.

A recipe that breaks the wall is INVALID, not merely unsuccessful: it did not
fly the route it claims to be a variation of, so its result says nothing about
that route.  Failures are still recorded — a failed trial is evidence about the
bound, which is why nothing here prunes them.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from reachy_ai.evaluation.base import (
    EpisodeVerdict, EvaluationPolicy, Violation, ViolationKind,
)
from reachy_ai.evaluation.contacts import (
    check_invalid_episode, check_joint_limits, check_saturation,
    compute_contact_metrics, compute_effort_metrics,
)
from reachy_ai.experience.models import EpisodeResult, PanelRouteTaskSpec
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.recipe import PrimitiveStep, TrajectoryRecipe

#: The abilities this module evaluates, and the route each one flies.  Taken
#: from `web/panel_abilities.py`, which is where an ability's route is decided;
#: a copy that disagreed with it would evaluate a motion nobody flies.
ABILITY_ROUTES: Dict[str, str] = {
    "rest_forearm": "PLACE_ROUTE",
    "stow_arm": "STOW_ROUTE",
    "wave": "WAVE",
    "point_cell": "POINT",
    "point_object": "POINT",
}

#: A measured grid cell, in metres — `scenes/FWDCenterLabMCC.yaml` marker geoms.
GRID_CELL_M = 0.127


@dataclasses.dataclass(frozen=True)
class PanelRoutePolicy(EvaluationPolicy):
    """Thresholds for the ability routes.  Frozen and versioned like the base.

    `policy_scope` exists because `policy_version` is an integer counted per
    policy, and a verdict carrying "policy_version 1" is otherwise ambiguous
    between this policy and the base one.  The pair is what identifies a
    policy; the scope is recorded in `to_dict()` and so in the stored trial.
    """

    policy_scope: str = "panel_routes"

    #: Read from the rig rather than repeated.  The live executor's post-move
    #: check and the episode recorder judge a disturbed board by this same
    #: number — see `rig_routes.OBJECT_DRIFT_TOL`.
    object_drift_tolerance_m: float = R.OBJECT_DRIFT_TOL

    #: How near a named posture counts as standing at it.  8 degrees, matching
    #: `posture_of` and `LIFT_TO_PRESENT`'s waypoint tolerance — a route that
    #: says it arrived and a posture check that says it did not are the same
    #: question asked twice with different answers.
    posture_tolerance_deg: float = 8.0

    #: Half a grid cell.  See the module docstring: this measures the hover the
    #: ability claims, not the calibrated ray it does not.
    hover_miss_tolerance_m: float = GRID_CELL_M / 2.0



# ---------------------------------------------------------------------------
# The canonical route — what a recipe is allowed to be a variation OF
# ---------------------------------------------------------------------------

@dataclasses.dataclass(frozen=True)
class RouteStep:
    """One step of the canonical route, as the integrity check compares it.

    `guard` is the set of joints that must arrive.  A recipe step may name more
    joints than this (a stricter check is always allowed); naming fewer drops a
    check the corridor was measured with.
    """

    primitive: str
    name: str
    seconds: float
    tol: float
    guard: Tuple[str, ...]


def canonical_steps(route: str, *, wave_cycles: int = R.WAVE_CYCLES,
                    point_legs: int = R.POINT_LEGS) -> Tuple[RouteStep, ...]:
    """The route as `rig_routes` defines it, in the recipe's own vocabulary.

    Generated from the module the arm actually flies, never transcribed.  The
    two corridor routes are their waypoint tuples; the wave and the point are
    built the way `rig_motion.wave` and `rig_motion.point_at` build them.
    """
    if route in ("PLACE_ROUTE", "STOW_ROUTE", "LIFT_TO_PRESENT",
                 "LOWER_TO_REST", "RAISE_TO_SIDE", "STOW_FROM_SIDE"):
        return tuple(
            RouteStep("waypoint", wp.name, wp.seconds, wp.tol,
                      tuple(wp.guard or R.CRITICAL_JOINTS))
            for wp in R.route_named(route)
        )

    if route == "WAVE":
        steps: List[RouteStep] = []
        for i in range(max(0, int(wave_cycles))):
            for side in ("a", "b"):
                # Unguarded on purpose, and `rig_motion.wave` says why: the
                # swing is judged at LESSON_TOL because three weak joints
                # reversing together do not arrive in step.
                steps.append(RouteStep("wave_swing", f"WAVE_{i + 1}{side.upper()}",
                                       R.WAVE_SECONDS, R.LESSON_TOL, ()))
        # The return DOES have to arrive, on the joints that say where the arm
        # is standing — one run finished the cycles with the shoulder far
        # enough out that `posture_of` no longer recognised PRESENT.
        steps.append(RouteStep("wave_return", "PRESENT", R.WAVE_SECONDS, 8.0,
                               tuple(R.GROSS_JOINTS)))
        return tuple(steps)

    if route == "POINT":
        # ONE STEP PER LEG.  The legs ARE the path-clearance check — the guard
        # models the arm between two poses as a joint-space straight line, and
        # six legs puts a measurement every ~4 cm of pad travel.  An earlier
        # version took `point_legs` as an argument and returned a fixed
        # three-step route regardless, so the one parameter this module calls
        # load-bearing shaped nothing the integrity check compared: a recipe
        # could declare fourteen legs, carry one approach step, and pass.
        legs = max(1, int(point_legs))
        return tuple(
            RouteStep("point_approach", f"APPROACH_{i + 1}", R.WAVE_SECONDS,
                      R.LESSON_TOL, ())
            for i in range(legs)
        ) + (
            RouteStep("point_hover", "HOVER", R.WAVE_SECONDS, R.LESSON_TOL, ()),
            RouteStep("point_return", "PRESENT", R.WAVE_SECONDS, 8.0,
                      tuple(R.GROSS_JOINTS)),
        )

    raise KeyError(f"no canonical route called {route!r}")


def align_to_parameters(recipe: TrajectoryRecipe) -> TrajectoryRecipe:
    """A copy of the recipe whose step list matches its bounded parameters.

    `wave_cycles` is a searchable parameter AND a fact about the step list —
    a two-cycle wave has four swings in it — so a sampler that varies the
    parameter leaves the two disagreeing, and `check_route_integrity` then
    quite correctly refuses a recipe the search itself produced.

    Regenerating the steps from `canonical_steps` is what reconciles them, and
    it does not weaken the wall: the sequence is rebuilt FROM the measured
    route, so an aligned recipe cannot be one that reordered the corridor,
    dropped a waypoint or loosened a tolerance.  What the integrity check
    exists to catch is a recipe that came out of a store or off disk, and
    those are not aligned by anybody.

    THE DURATIONS THE SEARCH CHOSE ARE KEPT.  An earlier version rebuilt every
    step from the canonical route, which hard-coded 1.8 s onto each swing — so
    a winner searched to 3.0 s exported a YAML declaring 1.8, and anyone
    reading the exported file as the flown motion got the wrong number.  The
    swings and the return take their seconds from `wave_seconds`, which is what
    the executor actually flies them at.
    """
    route = str(recipe.route or "")
    if route != "WAVE":
        return recipe

    cycles = _int(recipe.bounded_parameters.get("wave_cycles"), R.WAVE_CYCLES)

    # THE STORED PARAMETER IS THE ONE THAT WAS FLOWN.  A continuous sampler
    # hands back 1.6066 for a count of cycles; `_int` truncates it to 1 and the
    # arm waves once, but a recipe left declaring 1.6066 tells a later reader
    # the trial ran one-and-a-bit cycles, which is not a thing that happened.
    # Written back as the integer, so the record and the motion agree.
    bounded = dict(recipe.bounded_parameters or {})
    spec = bounded.get("wave_cycles")
    if isinstance(spec, dict):
        bounded["wave_cycles"] = dict(spec, value=cycles)
    elif spec is not None:
        bounded["wave_cycles"] = cycles

    seconds = _bp_float(bounded, "wave_seconds", R.WAVE_SECONDS)
    steps = [
        PrimitiveStep(primitive=s.primitive,
                      parameters={"name": s.name, "seconds": seconds,
                                  "tol": s.tol,
                                  "guard": list(s.guard) if s.guard else None})
        for s in canonical_steps(route, wave_cycles=cycles)
    ]
    return dataclasses.replace(recipe, primitive_sequence=steps,
                               bounded_parameters=bounded)


def _steps_of(recipe: TrajectoryRecipe) -> List[PrimitiveStep]:
    return list(recipe.primitive_sequence or [])


def _guard_of(step: PrimitiveStep) -> Optional[Tuple[str, ...]]:
    """The joints a recipe step guards, or None when it names none.

    None and the empty list are DIFFERENT.  None means "the recipe did not say",
    which inherits the canonical guard; an explicit empty list means "guard
    nothing", which is a dropped check and is refused as one.
    """
    raw = step.parameters.get("guard", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        return (raw,)
    return tuple(str(j) for j in raw)


def _float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def check_route_integrity(
    recipe: TrajectoryRecipe,
    route: str,
    *,
    policy: Optional[PanelRoutePolicy] = None,
) -> List[Violation]:
    """Every way this recipe is not a variation of the route it names.

    Returns an empty list when the recipe varies only what it is allowed to
    vary.  Each violation is hard: a recipe that reordered the corridor did not
    fly the corridor, and its score is about some other motion.
    """
    policy = policy or PanelRoutePolicy()
    out: List[Violation] = []

    def no(description: str, kind: ViolationKind = ViolationKind.INVALID_EPISODE,
           step: Optional[int] = None) -> None:
        out.append(Violation(kind=kind, description=description, step=step,
                             severity="hard"))

    steps = _steps_of(recipe)
    cycles = _int(recipe.bounded_parameters.get("wave_cycles"), R.WAVE_CYCLES) \
        if route == "WAVE" else R.WAVE_CYCLES
    legs = _int(recipe.bounded_parameters.get("point_legs"), R.POINT_LEGS) \
        if route == "POINT" else R.POINT_LEGS

    try:
        canonical = canonical_steps(route, wave_cycles=cycles, point_legs=legs)
    except KeyError:
        no(f"'{route}' is not a route this evaluator knows, so there is "
           "nothing to compare the recipe against")
        return out

    if len(steps) != len(canonical):
        no(f"the route has {len(canonical)} step(s) and the recipe has "
           f"{len(steps)}: a step inserted into or dropped from a measured "
           "corridor is a different motion, not a variation of this one")
        return out

    for i, (got, want) in enumerate(zip(steps, canonical)):
        if got.primitive != want.primitive:
            no(f"step {i} is '{got.primitive}' where the route has "
               f"'{want.primitive}'", step=i)
            continue
        got_name = str(got.parameters.get("name", ""))
        if got_name != want.name:
            no(f"step {i} is '{got_name or 'unnamed'}' where the route has "
               f"'{want.name}': the waypoints are an ORDER, and each one's "
               "clearance was measured from the one before it", step=i)
            continue

        # A tolerance may be tightened, never loosened.  Loosening one is how a
        # recipe scores better by being allowed to arrive less accurately.
        got_tol = _float(got.parameters.get("tol", want.tol), want.tol)
        if got_tol > want.tol:
            no(f"step {i} ({want.name}) loosens the tracking tolerance from "
               f"{want.tol} to {got_tol} degrees, which drops the check rather "
               "than passing it", step=i)

        guard = _guard_of(got)
        if guard is not None:
            missing = [j for j in want.guard if j not in guard]
            if missing:
                no(f"step {i} ({want.name}) stops guarding "
                   f"{', '.join(missing)}, and those joints are part of the "
                   "shape that fits the corridor", step=i)

        seconds = _float(got.parameters.get("seconds", want.seconds), want.seconds)
        if seconds <= 0.0:
            no(f"step {i} ({want.name}) asks for {seconds} seconds, which is "
               "not a duration", step=i)

    out.extend(_check_envelope(recipe, route))
    return out


def _check_envelope(recipe: TrajectoryRecipe, route: str) -> List[Violation]:
    """The measured envelope: what the rig was measured doing, and no more."""
    out: List[Violation] = []
    bp = recipe.bounded_parameters or {}

    def no(description: str) -> None:
        out.append(Violation(kind=ViolationKind.INVALID_EPISODE,
                             description=description, severity="hard"))

    if route == "WAVE":
        cycles = _int(bp.get("wave_cycles"), R.WAVE_CYCLES)
        if cycles > R.WAVE_CYCLES:
            no(f"{cycles} wave cycles were asked for and {R.WAVE_CYCLES} were "
               "measured; `rig_motion.wave` caps the count rather than trusting "
               "its caller, and so does this")
        if cycles < 0:
            no(f"{cycles} is not a number of cycles")
        amplitude = _bp_float(bp, "wave_amplitude_deg", abs(R.WAVE_A["r_forearm_yaw"]))
        measured = abs(R.WAVE_A["r_forearm_yaw"])
        if amplitude > measured + 1e-9:
            no(f"the wave swings to {amplitude} degrees of forearm yaw and "
               f"{measured} were measured: WAVE_A and WAVE_B are defined "
               "relative to PRESENT so the wave inherits PRESENT's clearance "
               "argument, and a wider swing leaves the envelope that argument "
               "covers")
        seconds = _bp_float(bp, "wave_seconds", R.WAVE_SECONDS)
        if seconds < R.WAVE_SECONDS - 1e-9:
            no(f"a {seconds} s swing is faster than the measured {R.WAVE_SECONDS} "
               "s: at 0.9 s the three weak joints reversing together finished "
               "tens of degrees short and tripped the tracking guard")

    if route == "POINT":
        legs = _int(bp.get("point_legs"), R.POINT_LEGS)
        if legs < R.POINT_LEGS:
            no(f"{legs} approach legs were asked for and {R.POINT_LEGS} is the "
               "measured minimum: the legs ARE the path-clearance check, and "
               "fewer of them measures the arm in fewer places — which is how "
               "cell_r2c1 reported +5.5 cm and still moved the can 0.189 m")
        for key, floor, why in (
            ("point_clearance_m", R.POINT_CLEARANCE,
             "air wanted under the pad"),
            ("point_hover_floor_m", R.POINT_HOVER_FLOOR,
             "the floor under the hover"),
            ("point_margin_m", R.POINT_MARGIN,
             "the clearance margin the whole rig uses"),
        ):
            value = _bp_float(bp, key, floor)
            if value < floor - 1e-9:
                no(f"{key} is {value} m and the measured floor is {floor} m "
                   f"({why}); this search may make a clearance stricter and "
                   "never looser")
    return out


def _bp_float(bp: Dict[str, Any], key: str, default: float) -> float:
    spec = bp.get(key)
    if isinstance(spec, dict):
        return _float(spec.get("value", default), default)
    return _float(spec, default) if spec is not None else default


def _int(spec: Any, default: int) -> int:
    if isinstance(spec, dict):
        spec = spec.get("value", default)
    try:
        return int(spec)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# What the evaluators read out of an EpisodeResult
# ---------------------------------------------------------------------------

#: Where the per-body fixture contact tally lives in `EpisodeResult`.
#:
#: The runner counts robot-to-fixture contacts and reports a TOTAL.  That is
#: enough for every task where touching a fixture is a failure, and not enough
#: for rest — where laying the forearm on the table IS the task, and the same
#: counter goes up.  Telling the intended contact from the unintended one needs
#: to know WHAT was touched, so the driver folds a per-body tally in here and
#: an evaluator that does not find one says it cannot tell rather than assuming
#: the contact was the good kind.
CONTACT_BODIES_KEY = "fixture_bodies"

#: Where the list of bodies that COULD have been hit lives.
#:
#: An empty contact tally means two very different things depending on this.
#: If the scene had rails and a board in it and the arm touched none of them,
#: the route kept clear.  If the scene's only collidable bodies are the floor,
#: the pedestal and some objects off in a corner — which is the case in
#: `FWDCenterLabSivaPool`, where the rig fixtures the corridor's geometry was
#: measured against are not collidable bodies at all — then the arm touching
#: nothing is not evidence about the corridor.  A verdict says which it had.
COLLIDABLE_KEY = "collidable_bodies"


def contact_bodies(result: EpisodeResult) -> Optional[Dict[str, int]]:
    """Which fixture bodies the arm touched and for how many steps, or None.

    None means nobody recorded it.  That is a real answer and the conservative
    one: an empty dict would claim the arm touched nothing.
    """
    raw = (result.contact_summary or {}).get(CONTACT_BODIES_KEY)
    if not isinstance(raw, dict):
        return None
    out: Dict[str, int] = {}
    for body, steps in raw.items():
        try:
            out[str(body)] = int(steps)
        except (TypeError, ValueError):
            out[str(body)] = 0
    return out


def object_drift(result: EpisodeResult,
                 spec: PanelRouteTaskSpec) -> Optional[Dict[str, float]]:
    """How far each object moved, in metres, or None if that cannot be told.

    "Nothing moved" is a COMPARISON, and the result records only where things
    ended.  A spec with no initial positions cannot supply the other half, so
    this answers None and the caller fails the clause rather than passing it on
    an assumption.
    """
    if not spec.initial_object_positions:
        # AN EMPTY BOARD IS NOT AN UNREAD BOARD.  A scene with nothing on it
        # records no starting positions and moves nothing, which is a real and
        # trivially satisfied answer.  A scene that HAS objects and recorded no
        # starting positions is the case this cannot decide, and the two are
        # told apart by whether the result found any objects at the end.
        if not (result.final_object_states or {}):
            return {}
        return None
    drift: Dict[str, float] = {}
    for oid, was in spec.initial_object_positions.items():
        now = (result.final_object_states or {}).get(oid)
        pos = (now or {}).get("pos_xyz") if isinstance(now, dict) else None
        if pos is None or len(pos) < 3 or len(was) < 3:
            # An object that ended nowhere is not an object that did not move.
            return None
        drift[oid] = math.sqrt(sum((float(a) - float(b)) ** 2
                                   for a, b in zip(pos[:3], was[:3])))
    return drift


def final_posture(result: EpisodeResult,
                  policy: PanelRoutePolicy) -> Optional[str]:
    """Which named posture the arm finished at, or None.

    None covers both "it finished somewhere that is not a posture" and "nobody
    recorded where it finished".  The callers distinguish them, because the
    operator-facing sentences are different and so is the fix.
    """
    joints = result.final_joint_positions_deg or {}
    if not joints:
        return None
    return R.posture_of(joints, tol=policy.posture_tolerance_deg)


# ---------------------------------------------------------------------------
# The evaluators
# ---------------------------------------------------------------------------

def _common(result: EpisodeResult, policy: PanelRoutePolicy,
            recipe: Optional[TrajectoryRecipe], route: str,
            lines: List[str]) -> List[Violation]:
    """Validity, route integrity, joint limits and saturation.

    Route integrity comes FIRST among the things that can invalidate an
    episode: a result from a recipe that reordered the corridor is not a
    measurement of this route at all, and reporting its accuracy would be
    reporting the accuracy of some other motion.
    """
    violations: List[Violation] = []

    if recipe is not None:
        integrity = check_route_integrity(recipe, route, policy=policy)
        violations.extend(integrity)
        for v in integrity:
            lines.append(f"NOT THIS ROUTE: {v.description}")

    invalid = check_invalid_episode(result, policy)
    violations.extend(invalid)
    if invalid:
        lines.append(f"INVALID: {invalid[0].description}")

    violations.extend(check_joint_limits(result, policy))
    violations.extend(check_saturation(result, policy))
    return violations


def _undisturbed(result: EpisodeResult, spec: PanelRouteTaskSpec,
                 policy: PanelRoutePolicy, violations: List[Violation],
                 lines: List[str]) -> bool:
    """Did the board come through untouched?  False also for "cannot tell"."""
    drift = object_drift(result, spec)
    if drift is None:
        lines.append(
            "UNDISTURBED: cannot be told — the episode records where the "
            "objects ended and the task spec carries no record of where they "
            "began.")
        violations.append(Violation(
            kind=ViolationKind.INVALID_EPISODE,
            description="no initial object positions were recorded, so "
                        "whether the board was disturbed cannot be decided",
            severity="hard"))
        return False

    moved = {oid: d for oid, d in drift.items()
             if d > policy.object_drift_tolerance_m}
    if moved:
        lines.append("DISTURBED: " + ", ".join(
            f"{oid} moved {d * 100:.1f} cm" for oid, d in sorted(moved.items())))
        for oid, d in sorted(moved.items()):
            violations.append(Violation(
                kind=ViolationKind.FORBIDDEN_CONTACT,
                description=f"{oid} moved {d:.4f} m, and anything over "
                            f"{policy.object_drift_tolerance_m} m is the arm "
                            "rather than the physics engine's jitter",
                severity="hard"))
        return False

    lines.append(f"UNDISTURBED: nothing moved more than "
                 f"{policy.object_drift_tolerance_m * 100:.0f} mm.")
    return True


def _arrived(result: EpisodeResult, wanted: str, policy: PanelRoutePolicy,
             violations: List[Violation], lines: List[str]) -> bool:
    """Did the arm finish at the posture the ability promised?"""
    if not (result.final_joint_positions_deg or {}):
        lines.append("ARRIVED: cannot be told — the arm's final pose was not "
                     "recorded.")
        violations.append(Violation(
            kind=ViolationKind.INVALID_EPISODE,
            description="the episode recorded no final joint positions, so "
                        "whether the arm arrived cannot be decided",
            severity="hard"))
        return False

    got = final_posture(result, policy)
    if got != wanted:
        lines.append(f"DID NOT ARRIVE: wanted {wanted}, finished at "
                     f"{got or 'no named posture'}.")
        violations.append(Violation(
            kind=ViolationKind.TASK_FAILURE,
            description=f"the route promised to leave the arm at {wanted} and "
                        f"it finished at {got or 'no named posture'}",
            severity="soft"))
        return False

    lines.append(f"ARRIVED: at {wanted}, within "
                 f"{policy.posture_tolerance_deg:g} degrees.")
    return True


def _only_intended_contact(result: EpisodeResult, spec: PanelRouteTaskSpec,
                           policy: PanelRoutePolicy,
                           violations: List[Violation],
                           lines: List[str]) -> bool:
    """Was everything the arm touched something it meant to touch?

    Rest is the reason this is not simply "no contact": laying the forearm on
    the table is the task, and the runner's forbidden-contact counter goes up
    for it exactly as it would for the arm hitting a rail.
    """
    touched = contact_bodies(result)
    if touched is None:
        total = int((result.contact_summary or {}).get("forbidden_total", 0) or 0)
        if total == 0:
            lines.append("CONTACT: none recorded.")
            return True
        lines.append(
            f"CONTACT: {total} fixture contact(s), and no record of WHAT was "
            "touched — which the intended table contact cannot be told from.")
        violations.append(Violation(
            kind=ViolationKind.FORBIDDEN_CONTACT,
            description=f"{total} robot-to-fixture contact(s) with no per-body "
                        "record, so an intended contact cannot be told from an "
                        "unintended one",
            severity="hard"))
        return False

    allowed = set(spec.intended_contact_bodies or ())
    unintended = sorted(b for b, steps in touched.items()
                        if steps > 0 and b not in allowed)
    if unintended:
        lines.append("UNINTENDED CONTACT: " + ", ".join(unintended))
        for body in unintended:
            violations.append(Violation(
                kind=ViolationKind.FORBIDDEN_CONTACT,
                description=f"the arm touched {body}, which this route does "
                            "not intend to touch",
                severity="hard"))
        return False

    # SAY WHAT WAS TOUCHED, not what would have been permitted.  An earlier
    # version printed the allowed set, so a route that touched nothing at all
    # reported "CONTACT: only table_top, which this route intends" — a
    # sentence about a contact that never happened, in a verdict whose whole
    # job is to be read instead of the raw episode.
    made = sorted(b for b, steps in touched.items() if steps > 0)
    if made:
        lines.append("CONTACT: " + ", ".join(made) + ", all intended.")
    else:
        lines.append("CONTACT: none.")
        lines.append(_nothing_to_hit(result))
    return True


def _nothing_to_hit(result: EpisodeResult) -> str:
    """Whether an empty contact record is evidence, in one sentence."""
    collidable = (result.contact_summary or {}).get(COLLIDABLE_KEY)
    if collidable is None:
        return ("NOTE: nothing recorded which bodies were collidable, so an "
                "empty contact record is not evidence the route kept clear.")
    named = [str(b) for b in collidable]
    if not named:
        return ("NOTE: this scene has no collidable body the arm could have "
                "hit, so touching nothing says nothing about the corridor.")
    return f"NOTE: collidable in this scene: {', '.join(sorted(named))}."


def _verdict(result: EpisodeResult, spec: PanelRouteTaskSpec,
             policy: PanelRoutePolicy, violations: List[Violation],
             lines: List[str], accuracy: float, successful: bool,
             extra: Optional[Dict[str, float]] = None) -> EpisodeVerdict:
    """Assemble the verdict.  One place, so the tiers cannot drift per ability."""
    contact = compute_contact_metrics(result)
    effort = compute_effort_metrics(result)
    hard = [v for v in violations if v.severity == "hard"]

    is_valid = not any(v.kind in (ViolationKind.INVALID_EPISODE,
                                  ViolationKind.NAN_IN_STATE)
                       for v in violations)
    is_safe = not hard
    is_successful = bool(successful and is_valid and is_safe)

    metrics: Dict[str, float] = {
        **contact, **effort,
        "hard_violation_count": float(len(hard)),
        "soft_violation_count": float(len(violations) - len(hard)),
        "accuracy": float(accuracy),
    }
    metrics.update(extra or {})

    total_steps = max(1.0, effort.get("total_steps", 1.0))
    ranking_scores = {
        "accuracy_score": max(0.0, min(1.0, float(accuracy))),
        "effort_score": max(0.0, 1.0 - (effort.get("saturated_joint_count", 0.0)
                                        / 21.0)),
        "duration_score": 1.0 / (1.0 + total_steps / 1000.0),
    }

    lines.append(
        f"Steps: {int(total_steps)}  "
        f"ForbiddenContact: {int(contact.get('forbidden_contact_count', 0))}  "
        f"Hard: {len(hard)}  Soft: {len(violations) - len(hard)}")

    return EpisodeVerdict(
        episode_id=result.episode_id,
        trial_id=result.trial_id,
        task_type=spec.ability or spec.task_type or "panel_route",
        policy_version=policy.policy_version,
        is_valid=is_valid,
        is_safe=is_safe,
        is_successful=is_successful,
        violations=violations,
        metrics=metrics,
        ranking_scores=ranking_scores,
        explanation="\n".join(lines),
    )


def evaluate_rest_forearm(
    result: EpisodeResult,
    spec: PanelRouteTaskSpec,
    recipe: Optional[TrajectoryRecipe] = None,
    policy: Optional[PanelRoutePolicy] = None,
) -> EpisodeVerdict:
    """Rest succeeded when it reached the supported posture, touched only what
    it meant to touch, and left the board undisturbed.

    THE FOREARM-FOOTPRINT CHECK DOES NOT EXIST YET (#82).  Rest lowers the
    forearm onto the table without first asking what is underneath it, so a
    verdict of "successful" here means the route arrived and moved nothing —
    not that the space it landed in was checked before it landed.  The drift
    clause catches the object it lands ON after the fact, which is evidence and
    not prevention, and the distinction belongs in the record.
    """
    policy = policy or PanelRoutePolicy()
    lines = [f"=== Rest Verdict: {spec.task_id or spec.ability} ==="]
    violations = _common(result, policy, recipe, "PLACE_ROUTE", lines)

    arrived = _arrived(result, R.POSTURE_REST, policy, violations, lines)
    clean = _only_intended_contact(result, spec, policy, violations, lines)
    undisturbed = _undisturbed(result, spec, policy, violations, lines)
    lines.append("NOTE: rest has no forearm-footprint check (#82) — an "
                 "undisturbed board here is an outcome, not a precaution.")

    return _verdict(result, spec, policy, violations, lines,
                    accuracy=1.0 if arrived else 0.0,
                    successful=arrived and clean and undisturbed)


def evaluate_stow_arm(
    result: EpisodeResult,
    spec: PanelRouteTaskSpec,
    recipe: Optional[TrajectoryRecipe] = None,
    policy: Optional[PanelRoutePolicy] = None,
) -> EpisodeVerdict:
    """Stow succeeded when it reached the rail pocket by walking the corridor
    and touched nothing on the way.

    The corridor is the whole of the safety argument: nothing may cut across
    it, because a direct move from anywhere over the board to HOME drives the
    upper arm through the board's near edge.  So "it ended at HOME" is not
    enough on its own — `check_route_integrity` is what says it got there the
    measured way, and it is not optional for this ability.
    """
    policy = policy or PanelRoutePolicy()
    lines = [f"=== Stow Verdict: {spec.task_id or spec.ability} ==="]
    violations = _common(result, policy, recipe, "STOW_ROUTE", lines)
    if recipe is None:
        lines.append("NOTE: no recipe was supplied, so the corridor was not "
                     "checked — only where the arm ended up.")

    arrived = _arrived(result, R.POSTURE_HOME, policy, violations, lines)
    clean = _only_intended_contact(result, spec, policy, violations, lines)
    undisturbed = _undisturbed(result, spec, policy, violations, lines)

    return _verdict(result, spec, policy, violations, lines,
                    accuracy=1.0 if arrived else 0.0,
                    successful=arrived and clean and undisturbed)


def evaluate_wave(
    result: EpisodeResult,
    spec: PanelRouteTaskSpec,
    recipe: Optional[TrajectoryRecipe] = None,
    policy: Optional[PanelRoutePolicy] = None,
) -> EpisodeVerdict:
    """The wave succeeded when it completed the cycles it was asked for, inside
    the tracking tolerance, and finished recognisably at the presentation pose.

    BOUNDED IS THE POINT.  The cycle count and the 1.8 s a swing are measured,
    not chosen — at 0.9 s the three weak joints reversing together finished
    tens of degrees short and tripped the guard.  A wave that ran more cycles
    or swung wider than the measurement is not a better wave, it is an
    unmeasured one, and `check_route_integrity` refuses it before any of this.

    Finishing AT PRESENT is a success criterion rather than tidiness: one run
    completed its cycles and left the shoulder far enough out that `posture_of`
    no longer recognised the pose, and the next request refused for want of a
    posture to start from.
    """
    policy = policy or PanelRoutePolicy()
    lines = [f"=== Wave Verdict: {spec.task_id or spec.ability} ==="]
    violations = _common(result, policy, recipe, "WAVE", lines)

    asked = int(spec.expected_wave_cycles or 0)
    done = int(result.metrics.get("wave_cycles_completed", -1))
    if done < 0:
        lines.append("CYCLES: not recorded, so completion cannot be told.")
        violations.append(Violation(
            kind=ViolationKind.INVALID_EPISODE,
            description="the episode recorded no completed wave count",
            severity="hard"))
        completed = False
    elif done < asked:
        lines.append(f"CYCLES: {done} of {asked} completed.")
        violations.append(Violation(
            kind=ViolationKind.TASK_FAILURE,
            description=f"{done} of {asked} wave cycles completed",
            severity="soft"))
        completed = False
    else:
        lines.append(f"CYCLES: {done} of {asked} completed.")
        completed = True

    arrived = _arrived(result, R.POSTURE_PRESENT, policy, violations, lines)
    clean = _only_intended_contact(result, spec, policy, violations, lines)
    undisturbed = _undisturbed(result, spec, policy, violations, lines)

    accuracy = (float(done) / float(asked)) if asked > 0 and done >= 0 else 0.0
    return _verdict(result, spec, policy, violations, lines,
                    accuracy=min(1.0, max(0.0, accuracy)),
                    successful=completed and arrived and clean and undisturbed,
                    extra={"wave_cycles_completed": float(max(0, done)),
                           "wave_cycles_asked": float(asked)})


def evaluate_point(
    result: EpisodeResult,
    spec: PanelRouteTaskSpec,
    recipe: Optional[TrajectoryRecipe] = None,
    policy: Optional[PanelRoutePolicy] = None,
) -> EpisodeVerdict:
    """Point succeeded when the pad reached the hover region it selected, kept
    its clearance, and left the board undisturbed.

    IT IS A HOVER, NOT A CALIBRATED RAY, and the executor already says so out
    loud — the miss over an empty board is 2 to 9 cm.  So the tolerance is half
    a measured grid cell: pointing at a cell means being over THAT cell rather
    than over its neighbour.  Some runs the ability makes today will fail this,
    which is the intended outcome — a bound worth searching against is one the
    current behaviour does not automatically clear.

    Clearance is reported by whoever flew the route; an episode with no
    clearance record fails the clause rather than passing it, because "nobody
    measured it" and "it was clear" are the two things this ability's history
    most needs kept apart (cell_r2c1 reported +5.5 cm and still moved the can
    0.189 m).
    """
    policy = policy or PanelRoutePolicy()
    lines = [f"=== Point Verdict: {spec.task_id or spec.ability} ==="]
    violations = _common(result, policy, recipe, "POINT", lines)

    miss = result.metrics.get("pad_miss_m", None)
    if miss is None:
        lines.append("MISS: not recorded, so accuracy cannot be told.")
        violations.append(Violation(
            kind=ViolationKind.INVALID_EPISODE,
            description="the episode recorded no pad miss, so whether the "
                        "hover reached the selected region cannot be decided",
            severity="hard"))
        reached, accuracy = False, 0.0
    else:
        miss = float(miss)
        reached = miss <= policy.hover_miss_tolerance_m
        accuracy = max(0.0, 1.0 - miss / max(1e-9, policy.hover_miss_tolerance_m))
        lines.append(
            f"MISS: {miss * 100:.1f} cm against a tolerance of "
            f"{policy.hover_miss_tolerance_m * 100:.1f} cm "
            f"(half a {GRID_CELL_M * 100:.1f} cm grid cell).")
        if not reached:
            violations.append(Violation(
                kind=ViolationKind.TASK_FAILURE,
                description=f"the pad finished {miss:.4f} m from the selected "
                            "hover, which is over the next cell as readily as "
                            "this one",
                severity="soft"))

    required = float(spec.hover_clearance_required_m or R.POINT_CLEARANCE)
    worst = result.metrics.get("worst_clearance_m", None)
    if worst is None:
        lines.append("CLEARANCE: not recorded.")
        violations.append(Violation(
            kind=ViolationKind.INVALID_EPISODE,
            description="the episode recorded no clearance, and 'nobody "
                        "measured it' is not 'it was clear'",
            severity="hard"))
        clear = False
    else:
        worst = float(worst)
        clear = worst >= required
        lines.append(f"CLEARANCE: worst {worst * 100:.1f} cm against "
                     f"{required * 100:.1f} cm required.")
        if not clear:
            violations.append(Violation(
                kind=ViolationKind.FORBIDDEN_CONTACT,
                description=f"worst clearance {worst:.4f} m is under the "
                            f"{required:.4f} m this route requires",
                severity="hard"))

    clean = _only_intended_contact(result, spec, policy, violations, lines)
    undisturbed = _undisturbed(result, spec, policy, violations, lines)
    arrived = _arrived(result, R.POSTURE_PRESENT, policy, violations, lines)

    # ONLY WHAT WAS MEASURED.  Writing 0.0 for an unrecorded miss records the
    # BEST POSSIBLE one, and 0.0 for an unrecorded clearance records contact —
    # both of them a measurement, in the same verdict that just said no
    # measurement exists.  A missing key is the only honest way to say it.
    extra: Dict[str, float] = {}
    if miss is not None:
        extra["pad_miss_m"] = float(miss)
    if worst is not None:
        extra["worst_clearance_m"] = float(worst)

    return _verdict(result, spec, policy, violations, lines, accuracy=accuracy,
                    successful=reached and clear and clean and undisturbed
                    and arrived,
                    extra=extra)


#: One evaluator per ability, by the name `panel_abilities` registers it under.
#: `point_cell` and `point_object` share an evaluator because they share a
#: route; what differs between them is the target, which is in the task spec.
EVALUATORS: Dict[str, Callable[..., EpisodeVerdict]] = {
    "rest_forearm": evaluate_rest_forearm,
    "stow_arm": evaluate_stow_arm,
    "wave": evaluate_wave,
    "point_cell": evaluate_point,
    "point_object": evaluate_point,
}


def evaluate(ability: str, result: EpisodeResult, spec: PanelRouteTaskSpec,
             recipe: Optional[TrajectoryRecipe] = None,
             policy: Optional[PanelRoutePolicy] = None) -> EpisodeVerdict:
    """Evaluate one ability's episode.  Raises for an ability with no evaluator.

    Raising is deliberate: a search pointed at an ability nobody has defined
    success for would otherwise optimise a default verdict, and a default
    verdict is a number with no meaning attached to it.
    """
    try:
        evaluator = EVALUATORS[ability]
    except KeyError:
        raise KeyError(
            f"no evaluator for ability {ability!r}; success has not been "
            f"defined for it.  Known: {', '.join(sorted(EVALUATORS))}") from None
    return evaluator(result, spec, recipe, policy)
