"""Flying the rig's measured routes (issue #64).

The waypoints live in `reachy_ai.motion.rig_routes`; this is what walks them.
Extracted from `notebooks/tlh_motion-routine.ipynb` sections 3, 4.6 and 5,
which stay the source of truth and are not imported.

WHAT A ROUTE RUNNER OWES ITS CALLER
-----------------------------------
Three things the notebook's `move_to` provides and a naive loop does not:

  * A per-waypoint tracking check.  The arm lags its goal, and "the command was
    sent" is not "the arm is there".  A waypoint that did not converge is a
    stop, because the next waypoint's clearance was measured from this one.
  * A cancellation point between waypoints, and only between them.  A Stop
    pressed mid-corridor cannot stop where it is — the arm is between rails.
    It finishes the waypoint it is flying and holds, which is what the operator
    is told.
  * A phase callback, so the panel can say where in the corridor the arm is
    rather than showing a spinner for thirty-five seconds.

WHAT IT DOES NOT DO
-------------------
It does not recover.  `nearest_waypoint` reports where the arm is; easing onto
that waypoint is the one move on an unverified path, and an arm threaded under
the front rail cannot be pulled out by any commanded pose.  Recovery is
reported, not attempted, and the world is never reset to make it go away.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

from reachy_ai.motion import primitives as P
from reachy_ai.motion import rig_routes as R

log = logging.getLogger("reachy_ai.tasks.rig_motion")

Abort = Optional[Callable[[], bool]]
Phase = Optional[Callable[[str], None]]


class RouteError(RuntimeError):
    """A waypoint the arm did not reach.  The route stops here."""


class RecoveryNeeded(RouteError):
    """The arm is not at a supported start, and no validated path leads back.

    Distinct from a tracking failure because the answer is different: this one
    needs a human to look at the rig, not a retry.
    """


def sdk_move(arm, pose: Dict[str, float], seconds: float) -> None:
    """Command a pose with the SDK's own minimum-jerk trajectory generator.

    NOT `primitives.smooth_move`, and the difference is not cosmetic.  The
    first version of this module used smooth_move — a hand-rolled 25 Hz
    interpolator — and the routes could not be flown at all.  Measured live at
    CURL (elbow -125 deg) in FWDCenterLabSivaPool, re-streaming the same target:

        smooth_move   -112.6  -116.3  -110.4  -113.5  -123.4  -113.8  -119.5
        goto/JERK     -119.8  -123.2  -123.9  -124.3

    smooth_move reaches the target and sags off it again; the joint oscillates
    around 10 deg short and never converges, so a 6 deg waypoint tolerance is
    unreachable.  `goto` closes monotonically.  The notebook flies these routes
    with `goto`, and that is part of what "the routes are validated" means.
    """
    from reachy_sdk.trajectory import goto
    from reachy_sdk.trajectory.interpolation import InterpolationMode

    goto({getattr(arm, name): value for name, value in pose.items()},
         duration=seconds,
         interpolation_mode=InterpolationMode.MINIMUM_JERK)


def present_pose(arm) -> Dict[str, float]:
    return {n: getattr(arm, n).present_position for n in R.R_JOINTS}


def _worst_error(arm, targets: Dict[str, float]) -> Tuple[str, float]:
    worst = max(targets, key=lambda n: abs(getattr(arm, n).present_position
                                           - targets[n]))
    return worst, abs(getattr(arm, worst).present_position - targets[worst])


def fly_route(arm, route, *, should_abort: Abort = None, on_phase: Phase = None,
              settle_s: float = 0.3, retries: int = 5,
              settle_pass_s: float = 0.8, move=None) -> List[str]:
    """Walk a route waypoint by waypoint.  Returns the waypoints flown.

    RE-STREAMING IS NOT OPTIONAL, and the notebook says so where it defines
    `move_to`:

        A short second pass to the same target re-streams the setpoints and
        pulls out the tracking lag; without it the physics arm can finish a
        fast segment 10-20 deg short, and simply HOLDING a goal will not close
        the gap — under mujoco-remote the arm only moves while setpoints are
        streaming.

    The first version of this function dropped that, and a live run in
    FWDCenterLabSivaPool showed exactly the documented failure: HOME to CURL
    asks the elbow for -125 deg, one pass left it at -48, and a direct write to
    `goal_position` afterwards did not move it at all — the value read back
    unchanged, because nothing was streaming.  Progressively longer passes are
    what close it.

    Two thresholds, kept apart for the reason the notebook keeps them apart:
    convergence (TRACK_TOL) decides when to stop re-streaming, and the
    waypoint's own `tol` decides when to give up.  A route stops at a waypoint
    it did not reach rather than pressing on, because the next waypoint's
    clearance was measured FROM this one.
    """
    move = move or sdk_move
    flown: List[str] = []
    for wp in route:
        if should_abort is not None and should_abort():
            log.info("route aborted before %s", wp.name)
            return flown
        if on_phase is not None:
            on_phase(wp.name)
        names = wp.guard or [n for n in wp.pose if n != "r_gripper"]
        guarded = {n: wp.pose[n] for n in names if n in wp.pose}

        move(arm, wp.pose, wp.seconds)
        for k in range(retries + 1):
            if _worst_error(arm, guarded)[1] <= R.TRACK_TOL:
                break
            # Longer each time: the wrist joints are weak and one short pass
            # leaves a large offset half-closed.
            move(arm, wp.pose, settle_pass_s * (1 + k))

        worst, off = _worst_error(arm, guarded)
        if off > wp.tol:
            raise RouteError(
                f"the arm did not reach {wp.name}: {worst} is {off:.1f} deg "
                f"off, tolerance {wp.tol:.1f}, after {retries} re-streamed "
                "passes. Stopping here rather than flying the next segment "
                "from a pose it never reached."
            )
        flown.append(wp.name)
        if settle_s:
            time.sleep(settle_s)
    return flown


def check_start(arm, route, *, tol: float = 8.0) -> Tuple[bool, str]:
    """Is the arm somewhere this route may be entered from?

    A route is only as safe as its first waypoint's assumption about where the
    arm was.  Answering "no, and here is the nearest waypoint" is more use than
    a tracking failure three waypoints in.

    Judged on the GROSS joints, at a posture tolerance.  The first version used
    all seven at the waypoint's own tracking tolerance, and that is the wrong
    question twice over: the wrist angles and the gripper do not move the elbow
    or forearm through the rails, and a tracking tolerance measures whether a
    move converged, not whether the arm is standing somewhere.  An arm sitting
    correctly at REST with 8 deg of wrist_roll drift — the drift GROSS_JOINTS
    exists to ignore — was refused, with a message that contradicted itself:
    "the arm is not at REST_SHUT; the nearest waypoint is REST_SHUT".
    """
    present = present_pose(arm)
    first = route[0]
    if R.at_pose(present, first.pose, tol=tol, joints=list(R.GROSS_JOINTS)):
        return True, ""
    name, distance = R.nearest_waypoint(present, route)
    return False, (f"the arm is not at {first.name}; the nearest waypoint on "
                   f"this route is {name}, {distance:.1f} deg away")


def deploy_to_rest(arm, *, should_abort: Abort = None,
                   on_phase: Phase = None, move=None) -> List[str]:
    """Out of the rail pocket and onto the board, ending with the forearm rested.

    Resting deliberately allows forearm-to-table contact, which is why the last
    two waypoints exist as a pair: REST_SHUT arrives with the hand closed, and
    REST only opens the gripper once the forearm is already supported.  The
    caller is responsible for having checked the footprint is clear — this
    walks a corridor, it does not look at the table.
    """
    if R.at_pose(present_pose(arm), R.REST, joints=list(R.GROSS_JOINTS)):
        if on_phase is not None:
            on_phase("already at rest")
        return []
    ok, why = check_start(arm, R.PLACE_ROUTE)
    if not ok:
        raise RecoveryNeeded(why)
    return fly_route(arm, R.PLACE_ROUTE, should_abort=should_abort,
                     on_phase=on_phase, move=move)


def stow_to_home(arm, *, should_abort: Abort = None,
                 on_phase: Phase = None, move=None) -> List[str]:
    """Back into the rail pocket, the placement route run backwards.

    Nothing may cut across it: a direct move from anywhere over the board to
    HOME drives the upper arm through the board's near edge.  That is why an
    arm that is not at REST_SHUT is a recovery question rather than a starting
    position to correct on the way.
    """
    if R.at_pose(present_pose(arm), R.HOME, joints=list(R.GROSS_JOINTS)):
        if on_phase is not None:
            on_phase("already stowed")
        return []
    ok, why = check_start(arm, R.STOW_ROUTE)
    if not ok:
        raise RecoveryNeeded(why)
    return fly_route(arm, R.STOW_ROUTE, should_abort=should_abort,
                     on_phase=on_phase, move=move)


def wave(arm, *, cycles: int = R.WAVE_CYCLES, should_abort: Abort = None,
         on_phase: Phase = None, move=None) -> int:
    """A bounded wave from PRESENT, ending back at PRESENT.

    `cycles` is capped at the measured count.  The 1.8 s a swing is measured
    too: at 0.9 s the three weak joints reversing together finished tens of
    degrees short and tripped the tracking guard.  Raising either without
    re-measuring is how a bounded behaviour stops being bounded, so the cap is
    here rather than in a comment asking callers not to.
    """
    move = move or sdk_move
    cycles = max(0, min(int(cycles), R.WAVE_CYCLES))
    # Judged on the gross joints at a POSTURE tolerance, not the wave's
    # tracking tolerance.  LESSON_TOL is 90 degrees — the slack those three
    # weak joints need to be called converged mid-swing — and using it to ask
    # "is the arm at PRESENT?" answers yes from the rail pocket.
    if not R.at_pose(present_pose(arm), R.PRESENT, tol=8.0,
                     joints=list(R.GROSS_JOINTS)):
        raise RecoveryNeeded(
            "the arm is not at the presentation pose, and the approach to it "
            "from where it is has not been validated"
        )
    done = 0
    for i in range(cycles):
        if should_abort is not None and should_abort():
            break
        for label, target in (("a", R.WAVE_A), ("b", R.WAVE_B)):
            if on_phase is not None:
                on_phase(f"wave {i + 1}{label}")
            move(arm, target, R.WAVE_SECONDS)
            guarded = {n: v for n, v in target.items() if n != "r_gripper"}
            if _worst_error(arm, guarded)[1] > R.LESSON_TOL:
                # Same re-stream as the routes.  The wave is the move the
                # notebook measured the lag ON: three weak joints reversing
                # together at 0.9 s finished tens of degrees short.
                move(arm, target, R.WAVE_SECONDS)
        done += 1
    if on_phase is not None:
        on_phase("returning to the presentation pose")
    # Unguarded on purpose, and the notebook says so: WAVE_A and WAVE_B are
    # defined relative to PRESENT, so this retreat stays inside the envelope
    # the wave was measured in.  It is not a jump to rest or to the pocket —
    # those are separate validated transitions.
    #
    # It does have to ARRIVE, though, on the joints that say where the arm is
    # standing.  One run finished the cycles and left the shoulder far enough
    # out that `posture_of` no longer recognised PRESENT, and the next move
    # refused for want of a posture to start from.  The wrists and forearm yaw
    # are deliberately left wherever the last swing put them.
    move(arm, R.PRESENT, R.WAVE_SECONDS)
    for k in range(4):
        if _reached_gross(arm, R.PRESENT):
            break
        move(arm, R.PRESENT, R.WAVE_SECONDS * (0.5 + 0.25 * k))
    return done


def _reached_gross(arm, target, tol: float = 8.0) -> bool:
    return R.at_pose(present_pose(arm), target, tol=tol,
                     joints=list(R.GROSS_JOINTS))


# ---------------------------------------------------------------------------
# Pointing (notebook 4.7)
# ---------------------------------------------------------------------------

@dataclass
class PointResult:
    """What a pointing attempt actually achieved.

    Every field exists because "the trajectory finished" is not "it pointed
    accurately", and the notebook has the runs to prove it: 21 cm of pad miss
    on one cell, and a can moved 0.189 m by an arm that reported positive
    clearance throughout.  So the caller is handed the miss, the clearance and
    the drift, and decides for itself what to claim.
    """

    reached: bool
    label: str
    target: Tuple[float, float, float]
    achieved: Optional[Tuple[float, float, float]] = None
    #: Distance from where the pad ended up to where it was aimed, in metres.
    miss_m: Optional[float] = None
    #: Extra height the target had to be lifted before the whole arm fit.
    lift_m: float = 0.0
    worst_clearance_m: Optional[float] = None
    #: Objects that moved during the attempt, {id: metres}.  Non-empty means
    #: the arm disturbed the board, whatever else it achieved.
    drift: Dict[str, float] = None
    detail: str = ""

    def __post_init__(self) -> None:
        if self.drift is None:
            self.drift = {}

    @property
    def disturbed_the_board(self) -> bool:
        return any(d > POINT_DRIFT_TOLERANCE_M for d in self.drift.values())


#: How far an object may appear to move before it counts as disturbed.  Objects
#: at rest in MuJoCo jitter far below this; the same threshold the panel's
#: proposal validator uses, for the same reason.
POINT_DRIFT_TOLERANCE_M = 0.02


def point_at(planner, label: str, xy: Tuple[float, float], base_z: float, *,
             send, read, approaching: Optional[str] = None,
             object_positions: Optional[Callable[[], Dict[str, tuple]]] = None,
             refresh=None, secs: float = 2.0,
             should_abort: Abort = None, on_phase: Phase = None) -> PointResult:
    """Hover the pad over one spot, lifting until the whole arm has room.

    `maximise_clearance` is what makes this possible at all.  The arm is
    redundant — many orientations put the pad on the same point with the elbow
    somewhere completely different — and the default IK rule spends that
    redundancy on the last millimetre of pad error, a criterion that knows
    nothing about the table.  Scoring candidates by whole-arm clearance instead
    turned cell_r2c2 from +0.6 cm to +12.1 cm at the same target.

    `approaching` names the object being reached for.  It gets a margin of its
    own rather than being dropped from the check, and the difference between
    those two is a defect that took a live run to see: excluded from the guard,
    blue_cylinder was hovered to within 1.6 cm with 6.1 cm of clearance
    reported and moved 0.123 m — the loop reported a clean flight because it
    was not looking at the one object the arm was aimed at.

    The SOLVER still ignores the target, because scoring candidate poses by
    distance from the thing you are trying to reach pushes the IK away from it.
    Optimise the approach, then verify it.

    THE HAND TRAVELS SHUT.  The gripper's moving finger swings out as it opens,
    so an open hand is a 7.5 cm tube about the wrist axis and a shut one is
    5.2 cm.  Every reach across the board passes over whatever stands in the
    middle of the grid, which with an open hand left 0.6-1.9 cm and refused
    nearly every cell.  Open the hand when you arrive, not before.
    """
    from reachy_ai.motion.escort import escort
    from reachy_ai.motion.kinematics import UnreachableError

    before = dict(object_positions()) if object_positions else {}
    solve_ids = None
    if approaching is not None and planner.scene is not None:
        solve_ids = [o for o in planner.scene.obstacle_ids(include_static=True)
                     if o != approaching]

    lift = 0.0
    chosen = None
    reason = "no pose was found with room for the whole arm"
    while lift <= R.POINT_LIFT_MAX + 1e-9:
        if should_abort is not None and should_abort():
            return PointResult(False, label, (xy[0], xy[1], base_z),
                               detail="stopped before the approach")
        if refresh is not None:
            refresh()
        seed = list(read())
        target = (xy[0], xy[1], base_z + lift)
        try:
            solution = planner.solve(target, seed=seed, maximise_clearance=True,
                                     from_joints=seed, gripper_deg=R.SHUT,
                                     ids=solve_ids, include_static=True)
        except UnreachableError as exc:
            reason = f"{label} is out of reach: {exc}"
            lift += R.POINT_LIFT_STEP
            continue
        room = planner.clearance(solution, gripper_deg=R.SHUT,
                                 ids=solve_ids, include_static=True)
        if room is not None and _distance_of(room) >= R.POINT_MARGIN:
            chosen = (solution, target, _distance_of(room))
            break
        reason = (f"the whole arm does not fit over {label} at that height "
                  f"({_distance_of(room) * 100:.1f} cm, want "
                  f"{R.POINT_MARGIN * 100:.0f})")
        lift += R.POINT_LIFT_STEP

    if chosen is None:
        return PointResult(False, label, (xy[0], xy[1], base_z), detail=reason)

    solution, target, room = chosen
    if on_phase is not None:
        on_phase(f"approaching {label}")

    # The target keeps its place in the guard, with a margin derived from the
    # pose being flown to: the approach may be exactly as close as it has to be
    # and no closer.  A flight that walks THROUGH the object still trips,
    # because the far side of it is not on that path.
    margins = ({approaching: max(0.0, room - R.POINT_APPROACH_SLACK)}
               if approaching is not None else None)

    result = escort(
        planner, solution, send, read,
        margin=R.POINT_MARGIN, margins=margins, legs=R.POINT_LEGS,
        duration=secs, gripper_deg=R.SHUT, refresh=refresh,
        include_static=True,
    )

    achieved = planner.fk_world(result.reached or read())
    miss = _euclid(achieved, target)
    after = dict(object_positions()) if object_positions else {}
    drift = {oid: _euclid(after[oid], pos)
             for oid, pos in before.items() if oid in after}
    worst = result.worst_realised or result.worst_planned
    return PointResult(
        reached=result.completed,
        label=label,
        target=target,
        achieved=achieved,
        miss_m=miss,
        lift_m=lift,
        worst_clearance_m=_distance_of(worst) if worst is not None else None,
        drift={k: v for k, v in drift.items() if v > 1e-6},
        detail=result.reason or "",
    )


def _distance_of(clearance) -> float:
    return float(getattr(clearance, "distance", clearance))


def _euclid(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


# ---------------------------------------------------------------------------
# The presentation pose
# ---------------------------------------------------------------------------

def to_present(arm, *, should_abort: Abort = None, on_phase: Phase = None,
               move=None) -> List[str]:
    """Side hub -> PRESENT, the pose the wave starts and ends at.

    Entered from the side hub, because that is the one place the forward swing
    is free: below roll -72 the rail is in front of the arm, and at roll 0 the
    pocket is.  `primitives.raise_to_side` is what gets there.

    Two legs, and the order is the whole point — see PRESENT_ROUTE.
    """
    if R.at_pose(present_pose(arm), R.PRESENT, tol=8.0,
                 joints=list(R.GROSS_JOINTS)):
        if on_phase is not None:
            on_phase("already at the presentation pose")
        return []
    now = present_pose(arm)
    if abs(now["r_shoulder_roll"] - R.PRESENT_LIFT["r_shoulder_roll"]) > 20.0:
        raise RecoveryNeeded(
            "the approach to the presentation pose starts from the side hub, "
            f"and my shoulder roll is at {now['r_shoulder_roll']:.0f} rather "
            f"than {R.PRESENT_LIFT['r_shoulder_roll']:.0f}. Raising to the hub "
            "first is a separate move."
        )
    return fly_route(arm, R.PRESENT_ROUTE, should_abort=should_abort,
                     on_phase=on_phase, move=move)


def from_present(arm, *, should_abort: Abort = None, on_phase: Phase = None,
                 move=None) -> List[str]:
    """PRESENT -> side hub.  The same two legs backwards.

    Not a jump to rest or to the pocket: those are further transitions, and
    each one is measured separately or not made.
    """
    if not R.at_pose(present_pose(arm), R.PRESENT, tol=12.0,
                     joints=list(R.GROSS_JOINTS)):
        raise RecoveryNeeded(
            "the return starts at the presentation pose, and the arm is not "
            "at it"
        )
    return fly_route(arm, R.PRESENT_RETURN, should_abort=should_abort,
                     on_phase=on_phase, move=move)


# ---------------------------------------------------------------------------
# Travelling the posture graph
# ---------------------------------------------------------------------------

def _raise_to_side(arm, **kw):
    from reachy_ai.motion import primitives as P
    P.raise_to_side(arm, duration=kw.pop("duration", 3.5))
    return ["SIDE_HUB"]


def _stow_from_side(arm, **kw):
    from reachy_ai.motion import primitives as P
    # stow_from_side turns the motors off when it arrives, which is right at
    # the end of a trip and wrong in the middle of one.  The caller turns them
    # back on; this is the only route that parks.
    P.stow_from_side(kw.pop("robot"), arm, duration=kw.pop("duration", 4.0))
    return ["HOME"]


#: What actually flies each edge of the posture graph.
ROUTE_RUNNERS = {
    "PLACE_ROUTE": lambda arm, **kw: deploy_to_rest(arm, **kw),
    "STOW_ROUTE": lambda arm, **kw: stow_to_home(arm, **kw),
    "PRESENT_ROUTE": lambda arm, **kw: to_present(arm, **kw),
    "PRESENT_RETURN": lambda arm, **kw: from_present(arm, **kw),
    "RAISE_TO_SIDE": _raise_to_side,
    "STOW_FROM_SIDE": _stow_from_side,
    "WAVE": lambda arm, **kw: ["wave x%d" % wave(arm, **kw)],
}


def travel(arm, to: str, *, robot=None, should_abort: Abort = None,
           on_phase: Phase = None) -> List[str]:
    """Get the arm from wherever it is to a named posture.

    Reads the posture first, then walks the measured edges.  An arm in the rail
    pocket asked for the presentation pose leaves the pocket on its own — that
    is a fact about the rig, not something an operator should have to know and
    type.  What it will NOT do is invent an edge: no path is a refusal.
    """
    here = R.posture_of(present_pose(arm))
    if here is None:
        name, distance = R.nearest_waypoint(present_pose(arm))
        raise RecoveryNeeded(
            "my arm is not at a posture I have a measured route out of — the "
            f"nearest waypoint is {name}, {distance:.0f} degrees away. I will "
            "not guess a path from here."
        )
    steps = R.path(here, to)
    if steps is None:
        raise RecoveryNeeded(
            f"there is no measured way from {here} to {to}, so I will not "
            "invent one."
        )
    flown: List[str] = []
    for route in steps:
        if should_abort is not None and should_abort():
            break
        if on_phase is not None:
            on_phase(route)
        kw = {"should_abort": should_abort, "on_phase": on_phase}
        if route in ("RAISE_TO_SIDE", "STOW_FROM_SIDE"):
            kw = {"robot": robot} if route == "STOW_FROM_SIDE" else {}
        flown.extend(ROUTE_RUNNERS[route](arm, **kw))
        if route == "STOW_FROM_SIDE" and robot is not None:
            robot.turn_on("r_arm")
    return flown
