"""Scene-aware pick-and-place arc against a LIVE simulator (issues #51, #56).

History
-------
`run_segment` and `pick_and_place` were moved here from
`scripts/demo_pick_place.py` (issue #51) so the command panel's executor drives
the same tuned motion the demo does, rather than a second copy that would
drift.  The extraction is the point: this arc is the only pick-and-place
motion in the repository that has been driven against physics and tuned for it
-- the slow over-table transit rate, the descend-straight-down approach.

PLAN, THEN EXECUTE (issue #56)
------------------------------
The arc used to plan one segment, fly it, plan the next.  It planned with the
PAD POINT only (tabletop and static boxes), so nothing along it ever asked
whether the forearm, the upper arm or the hand would hit an object, and its
last step (`P.raise_to_side`) cannot work from where the arm is (it requires
HOME; the arm is above the place site).  Now:

  * `plan_pick_and_place` plans the COMPLETE arc -- approach, grasp, carry,
    place, retract AND the return to PRESENT -- sending NO motion command, and
    checks every segment against the whole arm (`CartesianPlanner.
    check_arm_path`: tube hand, 0 cm margin, sampled at <= 2 deg).  It raises
    before anything moves if any part is refused.  There is no safe-prefix
    execution.
  * `execute_arc` flies the plan.  The IK policy is FAST everywhere; nothing
    here selects `IKPolicy.MAX_CLEARANCE`.  This change REFUSES unsafe planned
    arcs; it does not choose clearance-optimal trajectories, and automatic
    replanning is deferred.
  * `preflight_pick_place` is the job-level preflight the worker, the demo and
    the notebook all use: the RAISE_TO_SIDE footprint, every arc, and the
    STOW_FROM_SIDE footprint, before any motion.

THE START AND END POSTURE IS `rig_routes.PRESENT`, not `P.SIDE_HIGH`.  PRESENT
and SIDE_HIGH are not interchangeable: `stow_from_side` requires PRESENT.

THE RETURN GUARANTEE.  The return is a joint-space line from the end of the
retract to PRESENT, planned and checked with the rest of the arc and flown
with no `converge` and no `smooth_move`.  Commanding a checked endpoint does
not prove the arm follows the checked path while converging, so: after the
stream the arm's arrival is tested with `P.stow_entry_ok`; if it has not
arrived, at most `RETURN_CORRECTIONS_MAX` corrective sequences are made, each
one checked with `check_arm_path` BEFORE it is commanded.  If a correction is
refused or the attempts run out, `ReturnArrivalError` is raised and nothing
else is commanded.  The caller must not stow after it.  There is no general
recovery planner.

INTERIM CONTACT EXEMPTION (D5).  The target may overlap the HAND capsule only
during descent to hover, grasp, carry, and release/retract.  The target is
checked against the upper arm and forearm throughout, and every non-target
obstacle is checked throughout.  THIS DOES NOT PROTECT AGAINST UNINTENDED
HAND-TARGET CONTACT, nor the carried object against bystanders; contact-surface
modelling and carried-object collision protection are deferred to #55.  See
docs/adr/0004-whole-arm-arc-preflight.md.

NOT the offline episode runner
------------------------------
`native_mujoco/episode_runner.py` owns its own `SimulationCore` and resets it
at the start of every run.  Driving it from the panel would advance a separate
world and report that world's result as the live one, which is precisely the
false claim the panel must never make.  This module instead commands the same
world the page displays, through the v1 SDK the compatibility core already
bridges to the native server.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from reachy_ai.motion import primitives as P
from reachy_ai.motion import rig_routes as R
from reachy_ai.motion.footprint import route_footprint_clearances
from reachy_ai.motion.kinematics import (
    ArmClearanceError,
    CartesianPlanner,
    CollisionError,
    IKPolicy,
    R_ARM_JOINTS,
    TargetContact,
    UnreachableError,
    WHOLE_ARM_HAND_MODEL,
    WHOLE_ARM_MARGIN_M,
    joint_path,
    sample_count,
)
# Moved, not re-derived (#56): see motion/transit.py.  CLEAR_Z is re-exported
# because scripts/demo_pick_place.py imports it from here.
from reachy_ai.motion.transit import CARRY_HZ, CLEAR_Z, STEP_HZ  # noqa: F401
from reachy_ai.scene.awareness import SceneModel

log = logging.getLogger("reachy12.tasks.pick_place_live")

XY = Tuple[float, float]
XYZ = Tuple[float, float, float]

#: How many checked corrective sequences the return may make after the stream
#: before it gives up and reports a failed arrival.
RETURN_CORRECTIONS_MAX = 2

# Segment labels.  They are also the phase names `on_phase` reports.
SEG_SWING_IN = "swing in over table"
SEG_TO_HOVER = "descend to hover"
SEG_TO_GRASP = "descend to grasp"
SEG_LIFT = "lift to clear height"
SEG_CARRY = "carry over table"
SEG_TO_PLACE = "descend to place"
SEG_RETRACT = "retract"
SEG_RETURN = "return to the raised pose"
PHASE_CLOSE = "close gripper"
PHASE_RELEASE = "release"

#: The raise at the start and the stow at the end are flown by the caller
#: through these routes; their footprints are what the job preflight checks.
RAISE_ROUTE = "RAISE_TO_SIDE"
STOW_ROUTE = "STOW_FROM_SIDE"


# ── The plan ──────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlannedSegment:
    name: str                      # on_phase label
    traj: List[List[float]]        # 7-joint poses to stream
    cart: Optional[List[XYZ]]      # pad points (None for the joint-space return)
    rate_hz: int
    attach: bool                   # marker follows the pad (carry phases)
    before: Optional[str] = None   # "close" | "open" gripper action before this segment


@dataclass(frozen=True)
class ArcPlan:
    object_id: str
    place_xy: XY
    place: XYZ
    start: List[float]             # == R.PRESENT arm joints
    segments: Tuple[PlannedSegment, ...]
    end: List[float]               # == R.PRESENT arm joints, exactly
    #: The scene state this arc was planned (and is re-checked) against: a
    #: private copy with the target at its pick pose.  Not part of equality.
    scene: Optional[SceneModel] = field(default=None, compare=False,
                                        repr=False)


class ReturnArrivalError(RuntimeError):
    """The arm did not arrive at PRESENT after the checked return.

    ``refusal`` is the ArmClearanceError of a corrective move that was refused,
    if that is why it stopped.  Nothing else is commanded after this error and
    the caller must NOT stow: the stow's precondition is not met.
    """

    def __init__(self, message: str, *, worst_joint: str, off_deg: float,
                 attempts: int, refusal: Optional[ArmClearanceError] = None):
        super().__init__(message)
        self.worst_joint = worst_joint
        self.off_deg = off_deg
        self.attempts = attempts
        self.refusal = refusal


def present_joints() -> List[float]:
    """PRESENT as a 7-joint list in R_ARM_JOINTS order."""
    return [float(R.PRESENT[j]) for j in R_ARM_JOINTS]


def side_hub(planner) -> Tuple[Sequence[float], XY]:
    """The transit hub as (joint seed, pad world point).

    Derived from `rig_routes.PRESENT` -- the pose the arm is actually raised
    to and the one the stow starts from.  The name is historical.
    """
    seed = present_joints()
    return seed, planner.fk_world(seed)


def _plan_error(name: str, exc: Exception) -> Exception:
    """The same error type, naming the segment it came from."""
    err = type(exc)(f"{name}: {exc}")
    err.segment = name
    return err


def plan_pick_and_place(planner, scene, object_id: str, place_xy: XY) -> ArcPlan:
    """Plan the complete pick-and-place arc, return included.  No motion.

    Allowed here: IK / FK service calls.  Not allowed, and not made: turn_on,
    goal_position writes, gripper commands, head commands.

    Chains every segment from PRESENT and raises ArmClearanceError,
    CollisionError or UnreachableError (each naming the segment) if any part of
    the arc is refused.  ``scene`` is the model the arc is planned against; it
    is copied and never mutated.
    """
    snapshot = scene.copy()
    planner = planner.with_scene(snapshot)
    opened = R.OPEN                       # every check uses the OPEN aperture

    start = present_joints()
    side_pad = planner.fk_world(start)
    hover = snapshot.hover_point(object_id)    # object top + 0.05 m
    grasp = snapshot.grasp_point(object_id)    # object centre
    place = snapshot.rest_point(place_xy, object_id)
    above_pick = (hover[0], hover[1], CLEAR_Z)
    above_place = (place_xy[0], place_xy[1], CLEAR_Z)

    segments: List[PlannedSegment] = []
    q = list(start)

    def plan(name, a, b, steps, target, rate_hz, attach=False, before=None):
        nonlocal q
        try:
            traj, cart = planner.plan_segment(
                a, b, steps, q, ik=IKPolicy.FAST, gripper_deg=opened,
                target=target, segment=name)
        except ArmClearanceError:
            raise
        except (CollisionError, UnreachableError) as exc:
            raise _plan_error(name, exc) from exc
        segments.append(PlannedSegment(name, traj, cart, rate_hz, attach, before))
        q = traj[-1]

    def target(hand_contact, **kw):
        return TargetContact(object_id, hand_contact=hand_contact, **kw)

    # hand_contact is True exactly for segments 2-7: the interim D5 window.
    plan(SEG_SWING_IN, side_pad, above_pick, 40, target(False), CARRY_HZ)
    plan(SEG_TO_HOVER, above_pick, hover, 25, target(True), STEP_HZ)
    plan(SEG_TO_GRASP, hover, grasp, 15, target(True), STEP_HZ)
    plan(SEG_LIFT, grasp, above_pick, 30, target(True, carried=True), CARRY_HZ,
         attach=True, before="close")
    plan(SEG_CARRY, above_pick, above_place, 40, target(True, carried=True),
         CARRY_HZ, attach=True)
    plan(SEG_TO_PLACE, above_place, place, 30, target(True, carried=True),
         STEP_HZ, attach=True)
    plan(SEG_RETRACT, place, above_place, 25, target(True, center=place),
         CARRY_HZ, before="open")

    # The return: a joint-space line to PRESENT, planned and checked here and
    # flown without converge/smooth_move.  Its last pose is PRESENT exactly.
    q_present = present_joints()
    n = sample_count(q, q_present)
    traj = [list(p) for p in joint_path(q, q_present, n)[1:]]
    traj[-1] = list(q_present)
    planner.check_arm_path([q] + traj, segment=SEG_RETURN, gripper_deg=opened,
                           target=target(False, center=place))
    segments.append(PlannedSegment(SEG_RETURN, traj, None, CARRY_HZ, False))

    return ArcPlan(object_id=object_id, place_xy=tuple(place_xy), place=place,
                   start=start, segments=tuple(segments), end=list(q_present),
                   scene=snapshot)


# ── Executing a plan ──────────────────────────────────────────────────────────

def _read_arm(arm) -> List[float]:
    return [float(getattr(arm, n).present_position) for n in R_ARM_JOINTS]


def _fly_return(arm, plan: ArcPlan, seg: PlannedSegment,
                planner) -> None:
    """Stream the checked return, test arrival, make checked corrections.

    No `P.converge`, no `P.smooth_move`, and no command that is not part of a
    checked sequence.  Raises ReturnArrivalError on a failed arrival.
    """
    P.execute_trajectory(arm, seg.traj, R_ARM_JOINTS, rate_hz=seg.rate_hz)
    ok, joint, off = P.stow_entry_ok(arm)
    if ok:
        return
    checker = planner.with_scene(plan.scene) if plan.scene is not None else planner
    q_present = list(plan.end)
    attempts = 0
    while attempts < RETURN_CORRECTIONS_MAX:
        attempts += 1
        q_m = _read_arm(arm)
        n = sample_count(q_m, q_present)
        path = [list(p) for p in joint_path(q_m, q_present, n)]
        path[-1] = list(q_present)
        try:
            checker.check_arm_path(
                path, segment=f"{SEG_RETURN} (correction {attempts})",
                gripper_deg=R.OPEN,
                target=TargetContact(plan.object_id, hand_contact=False,
                                     center=plan.place))
        except ArmClearanceError as refusal:
            raise ReturnArrivalError(
                f"the arm did not arrive at the raised pose ({joint} is "
                f"{off:.0f} deg off) and the corrective move was refused "
                f"before it ran: {refusal}",
                worst_joint=joint, off_deg=off, attempts=attempts,
                refusal=refusal) from refusal
        P.execute_trajectory(arm, path[1:], R_ARM_JOINTS, rate_hz=seg.rate_hz)
        ok, joint, off = P.stow_entry_ok(arm)
        if ok:
            return
    raise ReturnArrivalError(
        f"the arm did not arrive at the raised pose after {attempts} checked "
        f"corrective move(s): {joint} is {off:.0f} deg off",
        worst_joint=joint, off_deg=off, attempts=attempts)


def execute_arc(robot, plan: ArcPlan, planner, attacher=None,
                should_abort: Optional[Callable[[], bool]] = None,
                on_phase: Optional[Callable[[str], None]] = None,
                close_hold_s: float = 0.0) -> List[float]:
    """Fly a planned arc.  Returns the final arm pose (PRESENT when complete).

    ``should_abort``, when given, is checked between segments up to the grasp:
    a cancel is honoured at a phase boundary rather than mid-trajectory, so
    the arm is never left part-way through a planned move with the gripper in
    an undefined state.  Past the close an abort is ignored and the arc
    finishes (an object released mid-carry drops onto whatever is below it).
    A cancel before the close leaves the arm where it is (residual R6).

    Never calls `P.raise_to_side`: the arc ends at PRESENT by its own checked
    return.  ``close_hold_s`` is a sleep after the close, not motion.
    """
    arm = robot.r_arm
    scene = plan.scene
    oid = plan.object_id

    def phase(name: str) -> bool:
        if on_phase is not None:
            on_phase(name)
        return bool(should_abort and should_abort())

    grasp = scene.grasp_point(oid) if scene is not None else None
    table_z = scene.table_surface_z if scene is not None else plan.place[2]

    # 1. Head looks toward the object.
    if grasp is not None:
        P.look_at(robot, grasp, duration=1.0)

    last = list(plan.start)
    for seg in plan.segments:
        if seg.name in (SEG_SWING_IN, SEG_TO_HOVER, SEG_TO_GRASP):
            if phase(seg.name):
                return last
        else:
            if seg.before == "close":
                # Past this point an abort still finishes the arc.
                phase(PHASE_CLOSE)
                log.info("── %s: close gripper", oid)
                P.close_gripper(arm)
                if attacher is not None and grasp is not None:
                    attacher.follow(oid, grasp)
                if close_hold_s > 0:
                    time.sleep(close_hold_s)
            elif seg.before == "open":
                phase(PHASE_RELEASE)
                log.info("── %s: open gripper (release)", oid)
                P.open_gripper(arm)
                if attacher is not None:
                    attacher.release(oid, plan.place)
            phase(seg.name)

        if seg.name == SEG_CARRY:
            # Head tracks toward the place site while the arm swings across.
            P.look_at(robot, (plan.place_xy[0], plan.place_xy[1], table_z),
                      duration=0.8)

        log.info("── %s: %s", oid, seg.name)
        if seg.name == SEG_RETURN:
            _fly_return(arm, plan, seg, planner)
        else:
            on_step = None
            if attacher is not None and seg.attach and seg.cart is not None:
                def on_step(i, q, _c=seg.cart):
                    attacher.follow(oid, _c[i])
            P.execute_trajectory(arm, seg.traj, R_ARM_JOINTS,
                                 rate_hz=seg.rate_hz, on_step=on_step)
        last = list(seg.traj[-1])
    return last


def pick_and_place(robot, planner, scene, attacher, object_id, seed, side_pad,
                   place_xy: XY,
                   should_abort: Optional[Callable[[], bool]] = None,
                   on_phase: Optional[Callable[[str], None]] = None,
                   close_hold_s: float = 0.0):
    """One pick-and-place cycle: `plan_pick_and_place`, then `execute_arc`.

    Starts and ends at PRESENT.  ``seed`` and ``side_pad`` are accepted so
    existing callers keep working and are IGNORED: the arc always starts from
    `rig_routes.PRESENT` (not the SIDE_HIGH hub it used to start from).
    Planning raises before any motion if any part of the arc is refused.
    Returns the ending pose.  Callers that fly several arcs or a job should use
    `preflight_pick_place` first.

    Motion arc, every segment checked against the whole arm:

      PRESENT -> [over table @ CLEAR_Z] -> above pick
              -> [down] -> hover -> grasp -> close gripper
              -> [up]   -> above pick
              -> [over table @ CLEAR_Z] -> above place
              -> [down] -> place -> open gripper
              -> [up]   -> above place
              -> [joint line] -> PRESENT      (checked; see the module docstring)
    """
    plan = plan_pick_and_place(planner, scene, object_id, place_xy)
    return execute_arc(robot, plan, planner, attacher, should_abort, on_phase,
                       close_hold_s=close_hold_s)


# ── The job-level preflight ───────────────────────────────────────────────────

class PreflightRefused(Exception):
    """A job was refused BEFORE any motion.  Nothing was turned on or commanded.

    ``segment`` is a route footprint ("RAISE_TO_SIDE footprint") or an arc
    segment label; the other fields name the object being moved, the arm link,
    the obstacle, the predicted clearance, the margin and the model.  Fields
    that do not apply (an unreachable point has no link) are None.
    """

    def __init__(self, message: str, *, segment: Optional[str],
                 object_id: Optional[str], link: Optional[str],
                 obstacle: Optional[str], clearance_m: Optional[float],
                 margin_m: Optional[float], model: Optional[str]) -> None:
        super().__init__(message)
        self.segment = segment
        self.object_id = object_id
        self.link = link
        self.obstacle = obstacle
        self.clearance_m = clearance_m
        self.margin_m = margin_m
        self.model = model

    def evidence(self) -> Dict[str, object]:
        return {"segment": self.segment, "object_id": self.object_id,
                "link": self.link, "obstacle": self.obstacle,
                "clearance_m": self.clearance_m, "margin_m": self.margin_m,
                "model": self.model}


@dataclass(frozen=True)
class JobPlan:
    arcs: Tuple[ArcPlan, ...]                  # in execution order
    skipped: Tuple[Tuple[str, str], ...]       # (object_id, refusal text) -- demo only
    final_board: Dict[str, XYZ]                # predicted poses after all arcs


def _refused_from_error(object_id: str, exc: Exception) -> PreflightRefused:
    segment = getattr(exc, "segment", None)
    if isinstance(exc, ArmClearanceError):
        return PreflightRefused(
            f"{object_id}: {exc}", segment=exc.segment, object_id=object_id,
            link=exc.link, obstacle=exc.obstacle, clearance_m=exc.clearance_m,
            margin_m=exc.margin_m, model=exc.model)
    model = "pad-point rule" if isinstance(exc, CollisionError) else None
    return PreflightRefused(
        f"{object_id}: {exc}", segment=segment, object_id=object_id, link=None,
        obstacle=None, clearance_m=None, margin_m=None, model=model)


_LEGS_TEXT = {RAISE_ROUTE: "HOVER -> REST_SHUT -> REST",
              STOW_ROUTE: "PRESENT -> REST_SHUT -> HOVER"}


def _footprint_refusal(model: SceneModel, route: str) -> Optional[PreflightRefused]:
    worst = route_footprint_clearances(model, [route])
    blocking = sorted((c.distance, oid) for oid, c in worst.items()
                      if c.distance < R.FOOTPRINT_MARGIN)
    if not blocking:
        return None
    d, oid = blocking[0]
    c = worst[oid]
    legs = _LEGS_TEXT[route]
    return PreflightRefused(
        f"{route} footprint ({legs}): {c.link} would come within "
        f"{d * 100:.1f} cm of {oid} ({WHOLE_ARM_HAND_MODEL} hand model, margin "
        f"{R.FOOTPRINT_MARGIN * 100:.1f} cm)",
        segment=f"{route} footprint", object_id=None, link=c.link, obstacle=oid,
        clearance_m=d, margin_m=R.FOOTPRINT_MARGIN, model=WHOLE_ARM_HAND_MODEL)


def preflight_pick_place(planner, scene, moves: Sequence[Tuple[str, XY]], *,
                         skip_refused: bool) -> JobPlan:
    """Check a whole pick/place job before ANY of it moves.

    1. The RAISE_TO_SIDE footprint against the current board.  A refusal
       refuses the whole job.
    2. Each move's complete arc, in order, on a COPY of the model that is
       updated with each accepted arc's placed pose.  With
       ``skip_refused=False`` (the worker) a refusal refuses the job; with
       ``skip_refused=True`` (the demo, the notebook loop) a refused move is
       recorded in ``skipped`` and its object stays where it is.
    3. The STOW_FROM_SIDE footprint against the predicted final board.  A
       refusal refuses the whole job.

    The footprint checks cover only the legs in `rig_routes.FOOTPRINT_LEGS`
    (RAISE_TO_SIDE: HOVER -> REST_SHUT -> REST; STOW_FROM_SIDE: PRESENT ->
    REST_SHUT -> HOVER), manipulable objects only (no rig fixtures), at each
    leg's wider commanded aperture.  The rest of each measured route is
    covered by the route's own rig validation, not by this live-object check.

    Raises PreflightRefused; sends no motion command of any kind.  ``scene`` is
    never mutated.
    """
    work = scene.copy()

    refusal = _footprint_refusal(work, RAISE_ROUTE)
    if refusal is not None:
        raise refusal

    arcs: List[ArcPlan] = []
    skipped: List[Tuple[str, str]] = []
    for object_id, place_xy in moves:
        try:
            arc = plan_pick_and_place(planner, work, object_id, tuple(place_xy))
        except (CollisionError, UnreachableError) as exc:
            refused = _refused_from_error(object_id, exc)
            if not skip_refused:
                raise refused from exc
            skipped.append((object_id, str(refused)))
            continue
        arcs.append(arc)
        work.update_poses({object_id: arc.place})

    refusal = _footprint_refusal(work, STOW_ROUTE)
    if refusal is not None:
        raise refusal

    final = {oid: work.get(oid).center for oid in work.manipulable_ids()}
    return JobPlan(arcs=tuple(arcs), skipped=tuple(skipped), final_board=final)
