"""Arm motion, in a process of its own (issue #79).

WHY THIS IS NOT A FUNCTION CALL
-------------------------------
`reachy_sdk` is built on **grpc.aio**, which runs its own asyncio event loop
and its own completion-queue poller.  The panel is a `ThreadingHTTPServer`
that already has a WebSocket link on a loop of its own and a worker pool.  Run
the two in one process and the poller eventually dies mid-move:

    Exception in callback PollerCompletionQueue._handle_events(...)
    BlockingIOError: [Errno 11] Resource temporarily unavailable

The blocking `goto` then never returns.  Not slowly — never.  The executor's
motion lock is held for the life of the process, every later request correctly
answers "another task is already using the arm", and no SDK client can connect
to the bridge again, including a fresh one from a shell.  Two commands was
all it took, and only a container restart cleared it.

So the SDK lives here instead, where the only thing it can take down is a
child that the panel can kill and replace.  That is also the only honest
timeout available: a wedged grpc.aio call cannot be interrupted from another
thread, and a thread that will never return cannot be reclaimed.  A process
can be.

WHAT THIS PROCESS DOES AND DOES NOT DECIDE
------------------------------------------
It moves the arm and reports where the arm ended up.  It does NOT hold the
lease, read the board, or judge the outcome: the parent owns the simulator
link, and the object-drift check that decides whether a route "arrived" needs
a before and an after that only the parent has seen.  Splitting it the other
way would put the WebSocket in here too, which is the problem again with the
sockets swapped.

WHAT IT WILL NOT DO
-------------------
It re-checks `gate_check()` before every job.  The parent checks it too, and
that is not a redundancy worth removing: this is the process that actually
commands the joints, and the standing rule is about the thing that moves.

THE PROTOCOL
------------
One JSON object per line, both ways, so a job cannot be half-read.

    in   {"job": {...}}          run it
         {"cancel": true}        the job in flight should stop
         {"observe": {...}}      the parent's latest simulator observation of
                                 the job's object and the gripper block, for a
                                 job that needs live feedback (lift_object);
                                 the parent owns the link, this only reads it
         (EOF)                   the parent is gone; exit
    out  {"phase": "..."}        progress, forwarded to the page
         {"result": {...}}       exactly one per job, always

`status` in a result is `moved` when the arm work finished and the parent
should now judge it, or `failed`/`cancelled` when this process already knows
the answer.  It is never `completed` — nothing in here has seen the board.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

#: How long to wait for `turn_on` to actually take effect.
#:
#: The arm is usually parked with the motors OFF — the stow that ran before
#: turns them off deliberately — and turning them on is not instant.  Asking
#: for 40 degrees of shoulder before the controller has taken hold produced
#: exactly one symptom: the arm did not move at all, and the pocket-exit guard
#: refused, roughly one attempt in three.
MOTOR_ON_TIMEOUT_S = 3.0

#: How long to let a new connection settle before commanding through it.  The
#: SDK's background sync threads need a moment to have read the arm once.
CONNECT_SETTLE_S = 0.8

Phase = Callable[[str], None]
Abort = Callable[[], bool]


def _ensure_paths() -> None:
    """Put the repo's src/ on the path, whether running from a checkout or /opt."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "..", "src"), "/opt/src"):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


def _fail(detail: str, **evidence) -> Dict[str, Any]:
    return {"status": "failed", "detail": detail, "evidence": evidence}


# -- the SDK connection, one per process ---------------------------------

class Connection:
    """The one SDK connection this process makes, kept between jobs.

    `ReachySDK.__init__` opens a gRPC channel and starts background sync
    threads, and connecting costs the better part of a second — a poor thing
    to pay per request.  Keeping it is safe here in a way it was not in the
    server process: when this connection goes wrong the parent kills the whole
    child, and the channel and its threads go with it.
    """

    def __init__(self, host: str, port: int, sdk_class=None) -> None:
        self._host = host
        self._port = port
        self._sdk_class = sdk_class
        self._robot = None
        self.command_poll_fixed: Optional[bool] = None

    def robot(self):
        if self._robot is not None:
            try:
                arm = self._robot.r_arm
                if arm is not None:
                    # Read something across the wire, so a channel that has
                    # died is found here rather than half way through a move.
                    # A joint that is simply absent is not evidence of a dead
                    # channel, so the probe is skipped rather than failed.
                    joint = getattr(arm, "r_shoulder_pitch", None)
                    if joint is not None:
                        _ = joint.present_position
                    return self._robot
            except Exception:                     # noqa: BLE001 - stale channel
                self.close()
        cls = self._sdk_class
        if cls is None:
            from reachy_sdk import ReachySDK
            from reachy_ai.sdk_compat import fix_command_poll
            # This connection lives across jobs; the stock poll leaks per
            # command (see sdk_compat).
            self.command_poll_fixed = fix_command_poll()
            cls = ReachySDK
        self._robot = cls(host=self._host, sdk_port=self._port)
        time.sleep(CONNECT_SETTLE_S)
        return self._robot

    def close(self) -> None:
        robot, self._robot = self._robot, None
        if robot is None:
            return
        try:
            robot._stop()
        except Exception:                         # noqa: BLE001 - best effort
            pass


def wait_for_motors(arm, timeout: float = MOTOR_ON_TIMEOUT_S) -> bool:
    """Wait until the arm reports itself driven rather than compliant.

    `turn_on` is a request, not a fact.  Some joints do not expose
    `compliant` at all, and a missing attribute is not evidence of anything —
    so those are taken on trust and the ones that do report are believed.
    """
    deadline = time.time() + timeout
    watched = [n for n in ("r_shoulder_pitch", "r_elbow_pitch")
               if hasattr(getattr(arm, n, None), "compliant")]
    if not watched:
        time.sleep(0.5)
        return True
    while time.time() < deadline:
        if all(getattr(arm, n).compliant is False for n in watched):
            time.sleep(0.2)          # let the controller settle on its hold
            return True
        time.sleep(0.1)
    return False


# -- the jobs -------------------------------------------------------------

def run_ability(job: Dict[str, Any], conn: Connection, *,
                emit: Phase = None, should_abort: Abort = None) -> Dict[str, Any]:
    """Fly one ability's route and report where the arm ended up.

    CANCELLATION IS PER STAGE, and the stages are not interchangeable.  Entry
    and exit are corridors: mid-corridor the arm is between rails and cannot
    stop where it is, so a Stop finishes the waypoint it is flying and holds.
    The action itself — a wave — can stop between cycles.  None of these is an
    emergency stop, and the panel must not word them as one.

    NO UNVERIFIED AUTO-RETREAT.  A route stopped half way leaves the arm at a
    waypoint, and the transition from that waypoint to anywhere else is the
    thing that was never measured.  It is reported, not corrected.
    """
    _ensure_paths()
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.tasks import rig_motion as M

    phase = emit or (lambda _name: None)
    abort = should_abort or (lambda: False)
    scene_name = job.get("scene", "")
    wanted = job.get("expected_start_posture") or ""

    phase("connecting to the arm")
    robot = conn.robot()
    if robot is None or robot.r_arm is None:
        return _fail("the right arm is not available on the simulator")
    arm = robot.r_arm

    start = R.posture_of(M.present_pose(arm))
    recovered: List[str] = []
    if start is None and M.stranded_at(arm) is not None:
        # Left part way through the pocket entry by something that stopped
        # early.  Finishing it is two measured moves, and refusing every later
        # request until a human intervenes is correct and useless.
        phase("finishing the move that was interrupted")
        robot.turn_on("r_arm")
        wait_for_motors(arm)
        recovered = M.resume_to_home(arm, on_phase=phase)
        start = R.posture_of(M.present_pose(arm))
    if start is None:
        # Not at any named posture, and not on the pocket sequence either.
        # The nearest waypoint is reported because it is the useful fact, and
        # NOT flown to, because the segment onto it is the one nobody
        # measured.
        name, distance = R.nearest_waypoint(M.present_pose(arm))
        return _fail(
            "my arm is not at a posture I have a measured route out of — the "
            f"nearest waypoint is {name}, {distance:.0f} degrees away. I will "
            "not guess a path from here.",
            recovery_needed=True, nearest=name)

    # PLANNED BEFORE ANYTHING MOVES.  The crane lift is checked end to end
    # (whole arm, table, both pads, the carried object) from the raised pose
    # with the IK service alone; a refusal here leaves the arm untouched.
    crane: Dict[str, Any] = {}
    if job.get("task_type") == "lift_object":
        refused = _crane_preflight(job, arm, phase, crane)
        if refused is not None:
            return refused

    robot.turn_on("r_arm")
    if not wait_for_motors(arm):
        return _fail("the arm's motors did not come on, so I will not command "
                     "a move it cannot make.", motors_off=True)

    # GETTING THERE IS PART OF DOING IT.  An arm stored in the rail pocket
    # cannot wave from where it is, and that is a fact about the rig rather
    # than something the operator should have to know and type.  So the
    # approach is flown, by measured edges only — `travel` refuses rather than
    # inventing one.
    approach: List[str] = list(recovered)
    if wanted and start != wanted:
        steps = R.path(start, wanted)
        if steps is None:
            return _fail(
                f"my arm is at {start} and that needs to start at {wanted}. "
                f"There is no measured way from {start} to {wanted}, so I "
                "will not invent one.",
                posture=start, wanted=wanted)
        for route in steps:
            ok, why = R.check_route(route, scene_name)
            if not ok:
                return _fail(
                    f"getting from {start} to {wanted} means flying {route} "
                    f"first, and {why}",
                    posture=start, wanted=wanted, blocked_on=route)
        phase(f"leaving {start}")
        approach += M.travel(arm, wanted, robot=robot,
                             should_abort=abort, on_phase=phase)
        robot.turn_on("r_arm")
        time.sleep(0.2)

    time.sleep(0.3)

    def go_to(posture):
        def _run(a, **kw):
            return M.travel(a, posture, robot=robot, **kw)
        return _run

    def point(a, **kw):
        return _point_at_target(job, robot, a, phase, **kw)

    runner = {"rest_forearm": go_to(R.POSTURE_REST),
              "stow_arm": go_to(R.POSTURE_HOME),
              "wave": lambda a, **kw: ["wave x%d" % M.wave(a, **kw)],
              # THE SAME FLIGHT, AIMED DIFFERENTLY.  A cell and an object are
              # not two manoeuvres; they are one manoeuvre given a different
              # height and a different thing to be careful of.
              "point_cell": point,
              "point_object": point,
              "lift_object": lambda a, **kw: _crane_lift(robot, a, phase, crane, **kw),
              }.get(job.get("task_type"))
    if runner is None:
        return _fail(f"I have no runner wired up for {job.get('task_type')}")

    try:
        flown = runner(arm, should_abort=abort, on_phase=phase)
    except M.RecoveryNeeded as exc:
        return _fail(str(exc), recovery_needed=True)
    except M.RouteError as exc:
        return _fail(str(exc), stopped_mid_route=True)

    result_crane = {"crane": crane["outcome"]} if "outcome" in crane else {}
    return {"status": "moved",
            **result_crane,
            "flown": list(approach) + list(flown),
            # Measured, before the approach was flown.  The proposal carries
            # the posture the route EXPECTS; the two differ whenever an
            # approach was needed, and an episode record wants both (#88).
            "start_posture": start,
            "final_posture": R.posture_of(M.present_pose(arm))}


#: How long a lift waits for the parent's first observation before refusing.
FIRST_OBSERVATION_S = 3.0
#: An observation older than this is not live (the crane module's own limit).
OBSERVATION_STALE_S = 0.5


def _observer(object_id: str):
    """`tasks.crane_pick_live.Observe` over what the parent forwards.  The
    age is end to end (since SimLink ingested it, in the panel process; same
    container kernel, same CLOCK_MONOTONIC), and a stale one says why."""
    from reachy_ai.feedback import latest as L
    from reachy_ai.tasks.crane_pick_live import Observation

    def observe():
        r = OBSERVATIONS.latest()
        now = time.monotonic()
        obs = r.value if r is not None else None
        if obs is None or obs.get("object_id") != object_id or len(obs.get("quat_wxyz") or ()) != 4:
            # No feedback is reported as STALE feedback, which halts a move
            # rather than letting it continue blind.
            return Observation((0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0), 0.0, False, 1e9,
                               stale_reason=L.NO_OBSERVATION)
        g = (obs.get("grippers") or {}).get("right") or {}
        return Observation(tuple(obs["position"]), tuple(obs["quat_wxyz"]),
                           float(g.get("grip_force_n", 0.0)), bool(g.get("grasping")),
                           r.age_s(now),
                           stale_reason=L.stale_reason(r, now, OBSERVATION_STALE_S))
    return observe


#: The crane events kept in the panel's evidence (every MEASURE is not: the
#: hover and grasp-depth ones are, with the corrections, the straddle, the
#: hold, the placement and the withdrawal).  Phases are live lines; this is
#: what remains on the task afterwards.
KEPT_CRANE_EVENTS = ("CORRECTION", "CORRECTION_COMMANDED", "ALIGNED", "ALIGN_REFUSED",
                     "DESCENT_CHECK_FAILED", "STRADDLE", "CLOSED", "HELD", "SUPPORTED",
                     "RELEASED", "WITHDRAWAL_JUDGED", "WITHDRAWAL_CORRECTION",
                     "WITHDRAWN", "HALT", "REFUSED", "RETREAT_REFUSED")


def _crane_phases(phase, kept: Optional[List[Dict[str, Any]]] = None):
    """The crane module's events, as the sentences the panel shows.  The
    numbers are the measured ones -- the corrected path is the point.  With
    ``kept``, the key events are also appended there for the evidence."""
    def mm(v):
        return "(" + ", ".join(f"{x:+.1f}" for x in v) + ") mm"

    def on_event(kind, **kw):
        if kept is not None and (kind in KEPT_CRANE_EVENTS or (
                kind == "MEASURE" and kw.get("label") in ("hover after pause", "grasp depth"))):
            kept.append(json.loads(json.dumps(dict(kind=kind, t=round(time.time(), 3), **kw),
                                              default=str)))
        if kind == "phase":
            phase(f"lift: {kw.get('name')}")
        elif kind == "MEASURE" and kw.get("label") in ("hover after pause", "grasp depth"):
            phase(f"lift: {kw['label']} -- measured jaw-gap error {mm(kw['gap_error_mm'])}, "
                  f"pads {kw['pads']['thumb_side_mm']:.1f} / {kw['pads']['finger_side_mm']:.1f} mm "
                  "outside the faces")
        elif kind == "CORRECTION":
            phase(f"lift: correcting the commanded path by {mm(kw['shift_mm'])} "
                  f"({kw['because']})")
        elif kind == "ALIGNED":
            r = kw["pad_route"]
            phase(f"lift: aligned after {kw['corrections']} correction(s) -- pads clear the "
                  f"top edges by {r['route_thumb_mm']:.1f} / {r['route_finger_mm']:.1f} mm")
        elif kind == "STRADDLE":
            phase("lift: pads straddle the object -- closing" if kw.get("ok")
                  else f"lift: no straddle ({', '.join(kw.get('failed', []))}) -- not closing")
        elif kind == "HELD":
            phase(f"lift: held {kw['rise_end_mm'] / 10:.1f} cm up; slip in the hand "
                  f"{kw['in_hand_slip_mm']:.2f} mm over the hold")
        elif kind == "SUPPORTED":
            phase("lift: object resting on the table -- opening" if kw.get("reached")
                  else "lift: lowered as far as the pads allow -- opening")
        elif kind == "WITHDRAWAL_JUDGED":
            r = kw["route"]
            phase(f"lift: way out judged against the object where it now is -- pads "
                  f"{r['route_thumb_mm']:.1f} / {r['route_finger_mm']:.1f} mm "
                  f"({'clear' if kw['ok'] else 'NOT clear'})")
        elif kind == "WITHDRAWAL_CORRECTION":
            phase(f"lift: shifting the way out by {mm(kw['shift_mm'])}")
        elif kind == "HALT":
            phase(f"lift: stopped -- {kw.get('reason')}")
        elif kind in ("REFUSED", "RETREAT_REFUSED"):
            phase(f"lift: refused -- {kw.get('reason')}")
    return on_event


def _crane_preflight(job, arm, phase, crane: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Plan and check the lift (no motion).  Fills ``crane`` or returns a failure."""
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import crane_pick_live as CP

    oid = job.get("object_id") or ""
    phase(f"lift: waiting for live feedback on {oid}")
    observe = _observer(oid)
    t0 = time.monotonic()
    while observe().age_s > OBSERVATION_STALE_S:
        if time.monotonic() - t0 > FIRST_OBSERVATION_S:
            return _fail("I have no live feedback on that object (its pose, "
                         "orientation and the grip force), so I will not try "
                         "to grasp it.", refused_before_motion=True, no_feedback=True)
        time.sleep(0.05)
    model = SceneModel.from_yaml(job["scene_file"])
    model.update_poses({k: tuple(v) for k, v in job["live"].items()})
    o = observe()
    model.update_poses({oid: o.position})
    planner = CartesianPlanner(arm, scene=model)
    phase(f"lift: planning and checking the whole lift of {oid} before moving")
    try:
        plan = CP.plan_crane_pick(planner, model, oid, o, on_event=_crane_phases(phase))
    except CP.CraneRefused as exc:
        return _fail(f"I will not lift {oid}: {exc.stage}: {exc.reason}",
                     refused_before_motion=True, stage=exc.stage)
    crane.update(plan=plan, planner=planner, model=model, observe=observe)
    return None


def _arm_state(arm) -> Dict[str, Any]:
    """The arm as MEASURED when the lift returned (a read; nothing is sent):
    joints, gripper, and whether any arm motor is still on.  What the panel
    reports about a stopped lift comes from this, not from the plan."""
    from reachy_ai.motion.kinematics import R_ARM_JOINTS
    try:
        joints = [round(float(getattr(arm, n).present_position), 2) for n in R_ARM_JOINTS]
        gripper = round(float(arm.r_gripper.present_position), 1)
        on = [not bool(getattr(arm, n).compliant) for n in R_ARM_JOINTS]
        return {"joints_deg": joints, "gripper_deg": gripper, "motors_on": any(on)}
    except Exception as exc:          # a report must not turn into a failure
        return {"unreadable": str(exc)}


def _crane_lift(robot, arm, phase, crane, *, should_abort=None, on_phase=None):
    """Fly the checked lift from PRESENT and back to PRESENT (no stow: the
    ability ends raised).  The outcome rides back in the result."""
    from reachy_ai.tasks import crane_pick_live as CP

    kept: List[Dict[str, Any]] = []
    out = CP.execute_crane_pick(robot, crane["planner"], crane["model"], crane["plan"],
                                crane["observe"], on_event=_crane_phases(phase, kept),
                                should_abort=should_abort, stow=False)
    crane["outcome"] = json.loads(json.dumps(dict(out, events=kept, arm_state=_arm_state(arm)),
                                         default=lambda v: (v.tolist() if hasattr(v, "tolist") else str(v))))
    held = out.get("lifted_and_held") or {}
    return [f"lift: {out.get('halt') or out.get('refusal') or 'completed the sequence'}",
            f"lift: held {held.get('rise_end_mm', 0) / 10:.1f} cm up" if held.get("rise_end_mm")
            else "lift: no hold"]


def _point_at_target(job, robot, arm, phase, *, should_abort=None, on_phase=None):
    """Hover the pad over one spot on the board, then come back to PRESENT.

    A CELL AND AN OBJECT ARE THE SAME FLIGHT AIMED DIFFERENTLY, which is the
    notebook's own shape — one `point_at`, called twice with different
    arguments — and it is kept because the two differ in exactly three things
    and agree on everything else.  Writing them as separate routines would
    duplicate the approach, the guard and the return, and the return is where
    the last defect was.

    OVER A CELL the height comes from the whole board: `hover_height` takes the
    tallest thing actually standing on the grid, adds the air wanted under the
    pad, and floors it — so an empty board answers 12 cm and a board with a can
    on it answers 18.2 cm.  That is right for a cell, because the arm has to
    cross whatever else is up there to get to it.

    OVER AN OBJECT the height comes from THAT OBJECT: the notebook uses
    `hover_point`, which is the object's own top plus the clearance.  This is
    the number a grasp actually needs — a flat block and a tall can get the
    same air under the pad rather than the same height above the table — and
    over the tallest thing on the board the two rules agree anyway.  The object
    is also named to the guard as `approaching`, which earns it a margin
    derived from the destination hover instead of the flat 5 cm that would
    refuse every approach that could ever be made.  It is NOT dropped from the
    check; dropping it is the defect that let blue_cylinder be hovered to
    1.6 cm and moved 0.123 m while the loop reported a clean flight.

    Reading the geometry here rather than passing it in the job is deliberate:
    the scene travelled with the job, and the number should be computed from
    the same snapshot everything else is checked against.

    THE RETURN TO PRESENT IS UNGUARDED, on purpose and for the notebook's
    reason: a refusal means "do not move", which is the wrong answer for a move
    whose whole purpose is to leave a place. It is also what makes the next
    request work — the hover is not a posture anything has a measured route out
    of, and an arm parked there would refuse everything afterwards.
    """
    _ensure_paths()
    from reachy_ai.motion import primitives as P
    from reachy_ai.motion import rig_routes as R
    from reachy_ai.motion.kinematics import CartesianPlanner, R_ARM_JOINTS
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks import rig_motion as M

    model = SceneModel.from_yaml(job["scene_file"])
    model.update_poses({oid: tuple(xyz) for oid, xyz in job["live"].items()})
    planner = CartesianPlanner(arm, scene=model)

    oid = job.get("object_id") or ""
    if oid:
        label, approaching, secs = oid, oid, 2.2
        x, y, z = model.hover_point(oid, R.POINT_CLEARANCE)
        hover = z - model.table_surface_z
        phase(f"planning a hover {R.POINT_CLEARANCE * 100:.0f} cm over {oid}")
    else:
        label, approaching, secs = job["cell"], None, 2.0
        x, y, base = model.cell_center(label)
        hover = M.hover_height(model)
        z = base + hover
        phase(f"planning a hover {hover * 100:.0f} cm over {label}")

    def send(joints, secs):
        pose = dict(zip(R_ARM_JOINTS, joints), r_gripper=R.SHUT)
        M.sdk_move(arm, pose, secs)
        M.sdk_move(arm, pose, 0.4)      # one re-stream, as the notebook does

    def read():
        return [getattr(arm, n).present_position for n in R_ARM_JOINTS]

    out = M.point_at(planner, label, (x, y), z, send=send, read=read,
                     approaching=approaching, secs=secs,
                     should_abort=should_abort, on_phase=on_phase)

    # THE RETURN UNWINDS THE HAND, not just the arm.  The IK spends the arm's
    # redundancy on clearance, so a pointing pose can leave `r_forearm_yaw`
    # most of 100 degrees from home — and the gross joints alone do not wait
    # for it.  Measured: a stow started straight after a point stopped at
    # HOVER with the forearm yaw 92 degrees off, past even the loose
    # tolerance.  It is a weak joint (kp=60), so it gets passes rather than a
    # tight tolerance.
    phase("back to the raised pose")
    P.converge(arm, dict(R.PRESENT), 2.5, tol=10.0,
               joints=list(R.GROSS_JOINTS), passes=6)
    P.converge(arm, dict(R.PRESENT), 1.5, tol=20.0,
               joints=("r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll"),
               passes=8)

    if not out.reached:
        raise M.RouteError(out.detail or f"I could not point at {label}")
    if out.disturbed_the_board:
        names = ", ".join(f"{k} by {v * 100:.0f} cm"
                          for k, v in out.drift.items())
        raise M.RouteError(f"I reached {label} but I moved something: {names}")
    # Tagged, because the flown list also carries the approach out of the
    # pocket and the operator asked about the point, not the twelve waypoints
    # it took to get somewhere it could point from.  The hover is reported
    # against the TABLE either way, so "7 cm" over a can and "12 cm" over a
    # cell are not quietly the same sentence measured from different places.
    return [f"point: {label} at a {hover * 100:.0f} cm hover",
            f"point: missed by {out.miss_m * 100:.1f} cm",
            f"point: lifted {out.lift_m * 100:.0f} cm"]


def run_pick_place(job: Dict[str, Any], conn: Connection, *,
                   emit: Phase = None, should_abort: Abort = None) -> Dict[str, Any]:
    """Fly the pick-and-place arc the demo flies, against the live poses.

    The scene document supplies geometry; the parent supplies where things
    actually are.  Planning against the YAML's initial poses would aim the
    gripper at where an object started.

    THE WHOLE JOB IS PREFLIGHTED BEFORE THE ARM IS TURNED ON (#56): the
    RAISE_TO_SIDE footprint, the complete arc -- approach, grasp, carry, place,
    retract and the return to the raised pose, every segment against the whole
    arm (tube hand model, 0 cm margin) -- and the STOW_FROM_SIDE footprint
    against the predicted final board.  A refusal returns `failed` with
    `refused_before_motion` and the segment, link, obstacle, clearance, margin
    and model in the evidence; nothing has been turned on or commanded.  The
    IK policy is FAST; nothing here searches for a clearance-optimal arc.

    The arc ends at the raised pose by its own checked return, so the stow
    that follows has its precondition.  If the arm does not arrive there, the
    result is `failed_arrival` and the arm is NOT stowed.
    """
    _ensure_paths()
    from reachy_ai.motion import primitives as P
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks.pick_place_live import (
        PreflightRefused, ReturnArrivalError, execute_arc, preflight_pick_place)

    phase = emit or (lambda _name: None)
    target_id = job["target_id"]

    phase("connecting to the arm")
    robot = conn.robot()
    if robot is None or robot.r_arm is None:
        return _fail("the right arm is not available on the simulator")

    model = SceneModel.from_yaml(job["scene_file"])
    model.update_poses({oid: tuple(xyz) for oid, xyz in job["live"].items()})

    planner = CartesianPlanner(robot.r_arm, scene=model)

    phase("checking the whole arc before moving")
    try:
        plan = preflight_pick_place(
            planner, model, [(target_id, tuple(job["cell_xy"]))],
            skip_refused=False)
    except PreflightRefused as exc:
        return _fail(str(exc), refused_before_motion=True, **exc.evidence())
    arc, = plan.arcs

    robot.turn_on("r_arm")
    time.sleep(0.3)

    phase("raising to the transit hub")
    P.raise_to_side(robot.r_arm, duration=3.0)

    cancelled = {"flag": False}

    def abort() -> bool:
        if should_abort is not None and should_abort():
            cancelled["flag"] = True
            return True
        return False

    try:
        execute_arc(robot, arc, planner, None, abort, phase)
    except ReturnArrivalError as exc:
        # The stow's precondition is not met, so it is not attempted.
        return _fail(str(exc), failed_arrival=True,
                     worst_joint=exc.worst_joint, off_deg=exc.off_deg,
                     attempts=exc.attempts,
                     refusal=str(exc.refusal) if exc.refusal else None)

    phase("returning home")
    P.go_home(robot, robot.r_arm, duration=3.0)

    if cancelled["flag"]:
        # Only now, with the arm actually stopped and parked, is it true to
        # say the motion has stopped.
        return {"status": "cancelled",
                "detail": "Stopped, and the arm is back at rest.",
                "evidence": {"cancelled_at_phase": True}}
    return {"status": "moved"}


JOBS = {"ability": run_ability, "pick_place": run_pick_place}


def run_job(job: Dict[str, Any], conn: Connection, *,
            emit: Phase = None, should_abort: Abort = None) -> Dict[str, Any]:
    _ensure_paths()
    try:
        from reachy_ai.motion.safety import gate_check
    except ImportError as exc:
        return _fail(f"the motion layer is not importable: {exc}")
    if not gate_check():
        # The gate is the project's standing rule, not a formality, and this
        # is the process that would do the moving.
        return _fail("the safety gate refused motion")

    runner = JOBS.get(job.get("kind"))
    if runner is None:
        return _fail(f"I have no job called {job.get('kind')!r}")
    try:
        return runner(job, conn, emit=emit, should_abort=should_abort)
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        return _fail(f"the motion failed: {exc.__class__.__name__}: {exc}")


# -- the process ----------------------------------------------------------

class Observations:
    """The latest observation the parent forwarded (latest-wins), stamped
    with when SimLink ingested it and when it arrived here."""

    def __init__(self) -> None:
        from reachy_ai.feedback.latest import LatestSlot
        self._slot = LatestSlot()

    def put(self, obs: Dict[str, Any]) -> None:
        now = time.monotonic()
        # An older parent sends only the age at send; then the pipe is not
        # counted, as before.
        ingest = obs.get("ingest_mono", now - float(obs.get("age_s", 0.0)))
        self._slot.put(obs, float(ingest), now)

    def latest(self):
        return self._slot.latest()


#: Filled by `_reader`; read by the jobs that need live feedback.  Built at
#: import, and it needs `reachy_ai.feedback`, hence the path first.
_ensure_paths()
OBSERVATIONS = Observations()


def _reader(stream, jobs: "queue.Queue", cancel: threading.Event) -> None:
    """Jobs and cancels share one pipe, so a cancel arrives while a job runs.

    It has to: the main thread is inside a blocking move for the whole time a
    Stop is worth sending.

    DRAIN TO LATEST.  Each wake-up takes everything already in the pipe;
    cancels and jobs are kept in order, and only the newest observation is
    published.  After a pause of this whole process (a gen-2 GC), the first
    read after it is the newest the parent sent, not a walk through the
    backlog.
    """
    from reachy_ai.feedback.latest import LineDrain, newest_of
    drain = LineDrain(stream)
    while True:
        lines, eof = drain.next_batch()
        messages = []
        for line in lines:
            try:
                messages.append(json.loads(line))
            except ValueError:
                continue
        controls, newest, _skipped = newest_of(
            messages, lambda m: isinstance(m, dict) and "observe" in m)
        for message in controls:
            if not isinstance(message, dict):
                continue
            if message.get("cancel"):
                cancel.set()
            elif "job" in message:
                jobs.put(message["job"])
        if newest is not None:
            OBSERVATIONS.put(newest["observe"])
        if eof:
            break
    jobs.put(None)                                # EOF: the parent is gone


def main() -> int:
    out = sys.stdout

    def send(**message) -> None:
        out.write(json.dumps(message) + "\n")
        out.flush()

    jobs: "queue.Queue" = queue.Queue()
    cancel = threading.Event()
    threading.Thread(target=_reader, args=(sys.stdin, jobs, cancel),
                     daemon=True).start()

    conn: Optional[Connection] = None
    while True:
        job = jobs.get()
        if job is None:
            break
        cancel.clear()
        sdk = job.get("sdk") or {}
        if conn is None:
            conn = Connection(sdk.get("host", "localhost"),
                              int(sdk.get("port", 50051)))
        result = run_job(job, conn,
                         emit=lambda name: send(phase=name),
                         should_abort=cancel.is_set)
        send(result=result)
    if conn is not None:
        conn.close()
    return 0


if __name__ == "__main__":                        # pragma: no cover - entry point
    sys.exit(main())
