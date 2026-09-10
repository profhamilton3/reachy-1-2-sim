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

    runner = {"rest_forearm": go_to(R.POSTURE_REST),
              "stow_arm": go_to(R.POSTURE_HOME),
              "wave": lambda a, **kw: ["wave x%d" % M.wave(a, **kw)],
              }.get(job.get("task_type"))
    if runner is None:
        return _fail(f"I have no runner wired up for {job.get('task_type')}")

    try:
        flown = runner(arm, should_abort=abort, on_phase=phase)
    except M.RecoveryNeeded as exc:
        return _fail(str(exc), recovery_needed=True)
    except M.RouteError as exc:
        return _fail(str(exc), stopped_mid_route=True)

    return {"status": "moved",
            "flown": list(approach) + list(flown),
            "final_posture": R.posture_of(M.present_pose(arm))}


def run_pick_place(job: Dict[str, Any], conn: Connection, *,
                   emit: Phase = None, should_abort: Abort = None) -> Dict[str, Any]:
    """Fly the pick-and-place arc the demo flies, against the live poses.

    The scene document supplies geometry; the parent supplies where things
    actually are.  Planning against the YAML's initial poses would aim the
    gripper at where an object started.
    """
    _ensure_paths()
    from reachy_ai.motion import primitives as P
    from reachy_ai.motion.kinematics import CartesianPlanner
    from reachy_ai.scene.awareness import SceneModel
    from reachy_ai.tasks.pick_place_live import pick_and_place, side_hub

    phase = emit or (lambda _name: None)
    target_id = job["target_id"]

    phase("connecting to the arm")
    robot = conn.robot()
    if robot is None or robot.r_arm is None:
        return _fail("the right arm is not available on the simulator")

    model = SceneModel.from_yaml(job["scene_file"])
    model.update_poses({oid: tuple(xyz) for oid, xyz in job["live"].items()})

    planner = CartesianPlanner(robot.r_arm, scene=model)
    robot.turn_on("r_arm")
    time.sleep(0.3)

    phase("raising to the transit hub")
    P.raise_to_side(robot.r_arm, duration=3.0)
    seed, side_pad = side_hub(planner)

    cancelled = {"flag": False}

    def abort() -> bool:
        if should_abort is not None and should_abort():
            cancelled["flag"] = True
            return True
        return False

    pick_and_place(robot, planner, model, None, target_id, seed, side_pad,
                   place_xy=tuple(job["cell_xy"]),
                   should_abort=abort, on_phase=phase)

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

def _reader(stream, jobs: "queue.Queue", cancel: threading.Event) -> None:
    """Jobs and cancels share one pipe, so a cancel arrives while a job runs.

    It has to: the main thread is inside a blocking move for the whole time a
    Stop is worth sending.
    """
    for line in stream:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        if message.get("cancel"):
            cancel.set()
        elif "job" in message:
            jobs.put(message["job"])
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
