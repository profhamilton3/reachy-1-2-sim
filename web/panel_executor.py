"""Live execution adapter for the command panel (issue #51).

Turns a confirmed proposal into motion in the SAME world the page is showing,
and reports what actually happened rather than that a message was sent.

HOW COMMANDS REACH THAT WORLD
-----------------------------
Not by a new channel.  `scripts/demo_pick_place.py` already drives this exact
world: the Reachy v1 SDK on gRPC 50051 → the compatibility core's
mujoco-remote backend → the native server's WebSocket on 8765 → the physics
the panel's cameras are rendering.  This adapter uses that path, and the arc
itself is `reachy_ai.tasks.pick_place_live`, the same tuned motion the demo
runs.  Nothing here is a second implementation of pick-and-place.

WHAT IT REFUSES TO DO
---------------------
It never calls `place_object`.  That teleports scene state, and reporting it as
task success would be a lie the operator cannot see through — the board would
show the object arriving without the arm ever having grasped it.  It never
drives `native_mujoco/episode_runner.py` either: that owns its own
SimulationCore and resets it, so its result describes a different world.

ARBITRATION IS THE SERVER'S JOB
-------------------------------
Before moving, the panel takes the execution lease.  While it is held the
server refuses `place_object`, `scene_load` and `reset` from every client, and
accepts `joint_command` only from the SDK bridge.  Disabling browser buttons
would not do: the page has its own socket to the simulator.  The lease names
the bridge as the mover because the panel holds the lease but does not move.

THE LEASE COMES FIRST, THEN THE SNAPSHOT
----------------------------------------
Everything the motion is planned against is read AFTER the grant and from a
snapshot captured after it, then revalidated.  Reading first and locking second
leaves one window — the only one in the whole execution — where the plan is
being fixed while other clients can still move the board.  A `place_object`
landing there sends the arm to where the object was; `_verify` catches it, so
the panel does not lie about the outcome, but the failure reads as a slipped
grasp rather than as a race, and that is the expensive kind of wrong.

What the lease does NOT give is exclusivity against other SDK callers.  It
names `docker-core` as the mover, and `docker-core` is one bridge that can
aggregate several callers: a notebook on gRPC 50051 arrives as that same
client_id and passes the server's check.  So the honest claim is that the scene
is locked against edits, not that the arm is locked against a second caller.

THE ARM IS MOVED FROM A DIFFERENT PROCESS
-----------------------------------------
`reachy_sdk` is grpc.aio, and grpc.aio inside this `ThreadingHTTPServer` dies:
its completion-queue poller raises `BlockingIOError` mid-move, the blocking
`goto` never returns, and the motion lock is held for the life of the process.
Two commands was all it took, and only a container restart cleared it (#79).
So `motion_worker.py` owns the SDK and this owns the leash: a wedged move
kills a child instead of the panel, and a wall-clock deadline turns "the panel
is dead until someone restarts the container" into "that command failed".

A thread could not have done it.  A wedged grpc.aio call cannot be interrupted
from outside, so a deadline in this process could only ever have abandoned the
thread — releasing the lock onto an SDK that no longer works for anyone.

COMPLETION IS PROVEN, NOT ASSUMED
---------------------------------
When the arc finishes, the object's live pose is read back and tested against
the destination cell with the same predicate the simulator uses for occupancy.
A trajectory that ran to the end without the object arriving is a failure, and
this is where that gets noticed — the arm can complete every segment while the
grasp slipped.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from panel_episodes import build_recorder

log = logging.getLogger("reachy12.panel.executor")

#: client_id the Docker compatibility core uses on its simulator socket.  It is
#: the connection SDK motion actually arrives on, so it is the lease's mover.
MOTION_CLIENT_ID = os.environ.get("REACHY_PANEL_MOTION_CLIENT", "docker-core")

SDK_HOST = os.environ.get("REACHY_SIM_SDK_HOST", "localhost")
SDK_PORT = int(os.environ.get("REACHY_SIM_SDK_PORT", "50051"))

#: How long the lease is taken for.  A pick-and-place arc is tens of seconds;
#: this leaves room for a slow one without letting a wedge freeze the scene.
LEASE_TTL_S = 240.0

#: How long to wait, after the lease is granted, for a snapshot captured after
#: it.  State arrives at the simulator's push rate, so this is many frames; a
#: link that cannot produce one in that time cannot be planned against, and
#: waiting indefinitely would hold the lease open over a dead socket.
FRESH_SNAPSHOT_TIMEOUT_S = 2.0

#: How long one motion may run before the panel stops waiting for it and kills
#: the process running it.  Nothing else bounds a move: the failure this exists
#: for is a `goto` that never returns at all, so any finite number is an
#: improvement on the current one.  It sits under the lease TTL on purpose —
#: the lease outliving the motion is what lets the kill be tidied up under it.
MOTION_DEADLINE_S = float(os.environ.get("REACHY_PANEL_MOTION_DEADLINE", "180"))

#: How long a motion process is given to exit on its own before it is killed.
#: A child that has answered goes on stdin EOF; the one this bounds is the
#: child that is wedged inside grpc, which will not go at all.
WORKER_EXIT_GRACE_S = 2.0

#: How close to the destination cell the object must end up to count as placed.
#: The cell's own half-extent decides that, so this only bounds the wait for a
#: fresh snapshot after the arm has finished.
SETTLE_TIMEOUT_S = 4.0


@dataclass
class ExecutionResult:
    status: str                       # completed | failed | cancelled
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


class _PhaseLog:
    """Forwards each phase onward and remembers when it happened (#88).

    The executor already announces its phases to the browser and then forgets
    them.  An episode record wants the sequence and the timings — "the wave
    spent nine seconds leaving the pocket" is the kind of fact a later
    evaluator is built to notice — and capturing them here costs one tuple per
    phase on a path that is already doing IPC.

    Callable, because it stands exactly where the `on_phase` callback stood.
    """

    def __init__(self, on_phase: Optional[Callable[[str], None]] = None) -> None:
        self._on_phase = on_phase
        self._started = time.time()
        self._entries: List[Tuple[str, float]] = []

    def __call__(self, name: str) -> None:
        self._entries.append((name, time.time() - self._started))
        if self._on_phase is not None:
            # Last, and not guarded: a browser callback that raises is a bug
            # worth seeing, and it must not be able to lose the phase record
            # by raising before it is kept.
            self._on_phase(name)

    def entries(self) -> List[Tuple[str, float]]:
        return list(self._entries)


#: The child that owns the SDK.  Beside this file, so a checkout and /opt both
#: find it without a search path.
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "motion_worker.py")


class MotionWorker:
    """A motion process, and the leash on it (issue #79).

    ONE CHILD, KEPT.  Connecting costs the better part of a second and starts
    background sync threads, so the child outlives a single job and keeps its
    connection.  What it does not outlive is going wrong: a job that overruns
    the deadline, a child that dies, a pipe that will not take a job — each
    ends the process, and the next job gets a new one.  That is the whole
    reason the SDK is out here.  In the server process the same failure had no
    remedy short of a container restart, because the wedge was a blocking call
    in a thread and a thread that will never return cannot be reclaimed.

    THE DEADLINE IS THE POINT, not a safety net.  Nothing else in this design
    bounds a motion, and the observed failure was not a slow move — it was a
    `goto` that never returned.  When it fires the arm is left where it
    stopped and reported that way: no auto-retreat, because the path from an
    arbitrary point is exactly what was never measured.

    A CONSEQUENCE WORTH KNOWING WHILE DEVELOPING: because the child is kept,
    it holds whatever `motion_worker.py` said when it was spawned.  Editing
    that file under a running panel changes nothing until the child is
    replaced — `supervisorctl restart camera-web-server` in the container.
    Measured cost of not knowing that: a live run that answered "I have no
    runner wired up for point_object" from a file that plainly had one.
    """

    def __init__(self, *, deadline_s: Optional[float] = None,
                 argv: Optional[list] = None) -> None:
        self._deadline = MOTION_DEADLINE_S if deadline_s is None else deadline_s
        self._argv = argv or [sys.executable, "-u", WORKER_SCRIPT]
        self._proc = None
        self._lines: Optional[queue.Queue] = None

    # -- the process -------------------------------------------------------

    def _start(self) -> None:
        proc = subprocess.Popen(
            self._argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        lines: queue.Queue = queue.Queue()
        threading.Thread(target=self._pump, args=(proc.stdout, lines),
                         daemon=True).start()
        threading.Thread(target=self._log_stderr, args=(proc.stderr,),
                         daemon=True).start()
        self._proc, self._lines = proc, lines

    @staticmethod
    def _pump(stream, lines: queue.Queue) -> None:
        """Read the child's stdout into a queue, so waiting can have a timeout.

        A pipe read cannot be given one; a queue get can, and that difference
        is what makes the deadline enforceable.
        """
        try:
            for line in stream:
                lines.put(line)
        except Exception:                         # noqa: BLE001 - killed child
            pass
        lines.put(None)                           # the child's stdout closed

    @staticmethod
    def _log_stderr(stream) -> None:
        try:
            for line in stream:
                if line.strip():
                    log.info("motion worker: %s", line.rstrip())
        except Exception:                         # noqa: BLE001 - killed child
            pass

    def close(self) -> None:
        proc, self._proc, self._lines = self._proc, None, None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()                # EOF: the child exits itself
        except Exception:                         # noqa: BLE001 - best effort
            pass
        try:
            proc.wait(timeout=WORKER_EXIT_GRACE_S)
        except Exception:                         # noqa: BLE001 - it is wedged
            proc.kill()
            try:
                proc.wait(timeout=WORKER_EXIT_GRACE_S)
            except Exception:                     # noqa: BLE001 - best effort
                pass

    # -- running a job -----------------------------------------------------

    def run(self, job: Dict[str, Any], *,
            on_phase: Optional[Callable[[str], None]] = None,
            should_cancel: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
        """Send one job, forward its phases, and always return a result."""
        if self._proc is None or self._proc.poll() is not None:
            self.close()
            self._start()
        try:
            self._proc.stdin.write(json.dumps({"job": job}) + "\n")
            self._proc.stdin.flush()
        except Exception:                         # noqa: BLE001 - dead pipe
            self.close()
            return _worker_failure(
                "I could not reach the process that moves the arm.",
                worker_died=True)

        deadline = time.monotonic() + self._deadline
        cancelled = False
        while True:
            left = deadline - time.monotonic()
            if left <= 0:
                self.close()
                return _worker_failure(
                    f"the move did not finish within {self._deadline:.0f} "
                    "seconds, so I stopped it. My arm is wherever it stopped, "
                    "and I will not guess a path from there.",
                    timed_out=True, recovery_needed=True)
            if not cancelled and should_cancel is not None and should_cancel():
                cancelled = self._send_cancel()
            try:
                line = self._lines.get(timeout=min(0.2, left))
            except queue.Empty:
                continue
            if line is None:
                self.close()
                return _worker_failure(
                    "the process that moves the arm stopped before it said "
                    "what happened. My arm is wherever it stopped.",
                    worker_died=True, recovery_needed=True)
            try:
                message = json.loads(line)
            except ValueError:
                log.info("motion worker: %s", line.rstrip())
                continue
            if "phase" in message and on_phase is not None:
                on_phase(message["phase"])
            if "result" in message:
                return message["result"]

    def _send_cancel(self) -> bool:
        """A Stop is a message, not a kill.

        Killing the child mid-corridor would leave the arm between rails at
        whatever waypoint it had reached, which is the one place it must not
        be abandoned.  The route runners stop between waypoints; this asks
        them to, and the deadline is what covers a child that will not.
        """
        try:
            self._proc.stdin.write(json.dumps({"cancel": True}) + "\n")
            self._proc.stdin.flush()
            return True
        except Exception:                         # noqa: BLE001 - dead pipe
            return False


def _worker_failure(detail: str, **evidence) -> Dict[str, Any]:
    return {"status": "failed", "detail": detail, "evidence": evidence}


class NullExecutor:
    """Stands in when nothing can execute.  Says why, per proposal.

    A specific reason matters more than it looks: "execution unavailable" sends
    someone to check the simulator, while "this scene has no destination the arm
    can reach" is the actual answer.
    """

    def __init__(self, reason: str = "no execution adapter is installed") -> None:
        self._reason = reason

    def available(self, proposal=None) -> Tuple[bool, str]:
        return False, self._reason

    def execute(self, proposal, **_kw) -> ExecutionResult:
        return ExecutionResult(status="failed", detail=self._reason)


class SimulatorExecutor:
    """Drives the live simulation through the v1 SDK, under the server's lease."""

    def __init__(self, link, scene_provider, scene_file: str,
                 *, sdk_host: str = SDK_HOST, sdk_port: int = SDK_PORT,
                 worker: Optional[MotionWorker] = None,
                 recorder: Any = None) -> None:
        self._link = link
        self._scene_provider = scene_provider
        self._scene_file = scene_file
        self._sdk_host = sdk_host
        self._sdk_port = sdk_port
        # One task at a time reaches the arm, whatever the coordinator thinks.
        self._motion_lock = threading.Lock()
        # The arm is moved from a child process, not from here.  See #79 and
        # `motion_worker.py`: grpc.aio in this server's process wedges, and a
        # wedged blocking call in a thread cannot be reclaimed.
        self._worker = worker or MotionWorker()
        # Writes one durable record per ability flown (#88), or None for no
        # recording.  Supplied rather than built here, the same way live
        # execution itself is: `build_executor` is where deployment policy
        # lives, and a directly-constructed executor — every test does this —
        # must not start writing to a database nobody asked it to open.
        self._recorder = recorder

    # -- availability ------------------------------------------------------

    def available(self, proposal=None) -> Tuple[bool, str]:
        """Whether this proposal could be executed right now, and why not.

        Answered PER ABILITY.  One reason for the whole executor was fine while
        there was one ability; with six it turns "I cannot rest my forearm in
        this scene" into "the arm is unavailable", which sends whoever reads it
        to the robot instead of to the compatibility record.
        """
        # Same path setup as execute(): without it the motion layer is
        # unimportable here and every proposal is refused for the wrong reason,
        # which would read as "the arm is broken" rather than "look at sys.path".
        _ensure_paths()
        try:
            import reachy_sdk  # noqa: F401
        except ImportError:
            return False, ("the Reachy SDK is not installed here, so nothing "
                           "can command the arm (this panel is outside the "
                           "simulator container)")

        try:
            from reachy_ai.motion.safety import gate_check
        except ImportError as exc:
            return False, f"the motion layer is not importable: {exc}"
        if not gate_check():
            # The gate is the project's standing rule, not a formality.
            return False, "the safety gate refused motion"

        if self._link is None or self._link.snapshot() is None:
            return False, ("I have no live link to the simulator, so I cannot "
                           "see what I would be moving")
        if not self._link.supports_lease:
            return False, ("this simulator cannot arbitrate scene edits, so I "
                           "will not move while other clients can change the "
                           "board underneath me")

        if proposal is None:
            return True, ""

        if proposal.task_type != "pick_place":
            return self._ability_available(proposal)
        if proposal.destination_kind != "cell":
            return False, ("I can only place onto a grid cell; this plan names "
                           f"{proposal.destination_label!r}")

        scene = self._scene_provider()
        if scene.error:
            return False, f"I cannot read the scene: {scene.error}"
        obj = scene.objects.get(proposal.target_id)
        if obj is None or obj.position is None:
            return False, f"I cannot see where {proposal.target_id} is"
        if obj.on_board is not True:
            return False, (f"{proposal.target_id} is not on the board, and I "
                           "only pick from the board")
        cell = scene.cells.get(_ref_id(proposal.destination))
        if cell is None:
            return False, "that destination is not in this scene"
        if not cell.reachable:
            return False, f"{cell.name} is out of the arm's reach"
        return True, ""

    def _ability_available(self, proposal) -> Tuple[bool, str]:
        """Can this ability be flown, in this scene, on this arm?

        Three questions, and each is a different answer to give:  the ability
        may have no motion at all; the route may not be validated in the scene
        the panel is showing; or the arm may be somewhere the route cannot be
        entered from.  Collapsing them loses the one piece of information the
        operator can act on.
        """
        _ensure_paths()
        from reachy_ai.motion import rig_routes as R

        if not proposal.route:
            return False, (f"I have no motion for {proposal.task_type} — it is "
                           "recognised, planned and not yet built")
        if proposal.recipe_parameters:
            # THE LAST LINE AGAINST A SILENT SUBSTITUTION (#90).  The planner
            # only accepts a recipe whose parameters the ability declares it
            # can apply, and no ability declares any yet — so a plan reaching
            # here with parameters on it means something applied them without
            # a way to fly them.  Refuse rather than fly the default route
            # while the card claims a promoted recipe.
            varies = ", ".join(sorted(proposal.recipe_parameters))
            return False, (f"the plan names a recipe that varies {varies}, "
                           f"and I have no way to apply that to {proposal.route}")
        if proposal.arm != "right":
            return False, ("I can only do that with my right arm; the route "
                           "was measured for that arm through a rig that is "
                           "not symmetric")

        scene = self._scene_provider()
        if scene.error:
            return False, f"I cannot read the scene: {scene.error}"

        # POINTING IS MEASURED UP TO ONE OBJECT ON THE BOARD.
        #
        # Empty, and every one of the six object types alone on the centre
        # cell — which is the harshest single-object case, because everything
        # standing in the middle of the grid is crossed by every reach.  All
        # six left the board undisturbed.  Two or more has NOT been flown, and
        # section 4.7 is a catalogue of what an occupied board does to this
        # manoeuvre when it goes wrong: a can moved 0.189 m by an arm
        # reporting +5.5 cm, a cylinder hovered to 1.6 cm and moved 0.123 m.
        #
        # The limit is the number of objects rather than which ones, because
        # what has not been measured is objects INTERACTING — a reach that
        # clears the thing it is aimed at by threading past a second one.
        if proposal.task_type in ("point_cell", "point_object"):
            # THE OBJECT HAS TO BE ON THE BOARD, checked here and not only in
            # the planner.  The planner refuses a pool object when the plan is
            # made and again when it is confirmed; this is the check that runs
            # under the lease, against the scene as it is at the moment the arm
            # is about to move.  An object can be lifted off the board between
            # a confirmation and its execution, and the failure it would cause
            # is a tabletop reach aimed at the floor.
            if proposal.object_id:
                obj = scene.objects.get(proposal.object_id)
                if obj is None:
                    return False, (f"{proposal.object_id} is not in this "
                                   "scene any more")
                if obj.on_board is not True:
                    return False, (f"{proposal.object_id} is not on the board, "
                                   "so there is nothing on the table for me to "
                                   "point at")

            standing = sorted(oid for oid, o in scene.objects.items()
                              if o.on_board is True)
            if len(standing) > 1:
                return False, (
                    "I have measured pointing over an empty board and over "
                    "one object at a time, and there are "
                    f"{len(standing)} on it: {', '.join(standing)}. What I "
                    "have not flown is a reach threading past a second "
                    "object, and that is how a can was moved 0.189 m in the "
                    "runs this caution comes from.")

        # An ability is FOR a posture, not for one route.  "Store your arm"
        # means end in the pocket, and which measured routes get there depends
        # on where the arm is standing — from the presentation pose it is
        # STOW_FROM_SIDE, which is not the same route as PLACE_ROUTE's reverse
        # entered from rest.
        # So availability asks whether ANY validated route leads where the
        # ability is going; the executor checks the ACTUAL path once it can
        # see the arm, and refuses there if a leg of it is not validated.
        if proposal.end_posture:
            into = [route for (_a, b), route in R.POSTURE_TRANSITIONS.items()
                    if b == proposal.end_posture]
            usable = [r for r in into if R.check_route(r, scene.name)[0]]
            if not usable:
                blocked = [R.check_route(r, scene.name)[1] for r in into]
                return False, (blocked[0] if blocked else
                               f"nothing measured reaches {proposal.end_posture}")
            return True, ""

        ok, why = R.check_route(proposal.route, scene.name)
        if not ok:
            # Named in the scene's own terms.  This is the refusal the
            # FWDCenterLabSivaPool validation run produced, and it is the
            # correct answer there, not a placeholder for one.
            return False, why

        return True, ""

    # -- execution ---------------------------------------------------------

    def execute(self, proposal, *, should_cancel: Optional[Callable[[], bool]] = None,
                on_phase: Optional[Callable[[str], None]] = None) -> ExecutionResult:
        ok, why = self.available(proposal)
        if not ok:
            return ExecutionResult(status="failed", detail=why)

        if not self._motion_lock.acquire(blocking=False):
            return ExecutionResult(
                status="failed",
                detail="another task is already using the arm",
            )
        phases = _PhaseLog(on_phase)
        started = time.time()
        try:
            result = self._execute_locked(proposal, should_cancel, phases)
        finally:
            self._motion_lock.release()

        # AFTER BOTH LOCKS.  The lease is released in the inner `finally` and
        # the motion mutex just above, so a slow or broken disk delays nobody's
        # next movement — and a refusal taken after the lease was granted is
        # still an episode that happened, so it is recorded like any other.
        self._record_episode(proposal, result, phases, started)
        return result

    def _record_episode(self, proposal, result, phases, started) -> None:
        """Hand the episode to the recorder.  Abilities only, for now.

        Pick-and-place has its own offline task runner and its own recipe in
        `recipes/pick_place/`; what it does not have is a definition of
        success measured the way this path measures one.  Recording it here
        would put two different judgements under one task_type.  One line to
        add once #91 settles that.
        """
        if self._recorder is None or proposal.task_type == "pick_place":
            return
        try:
            # INSIDE THE WALL.  `record` swallows its own failures, but the
            # scene read that feeds it is this method's, and the scene
            # provider raises on a scene that will not load.  Recording must
            # not be able to fail a movement that already succeeded.
            scene = self._scene_provider()
            self._recorder.record(
                proposal, result,
                phases=phases.entries(),
                started_at=started,
                ended_at=time.time(),
                scene_name=getattr(scene, "name", "") or proposal.scene_name,
                # The revision moves when the board is edited; the name does
                # not.  The record wants the one that moves.
                scene_revision=getattr(scene, "scene_revision", ""),
                # THE BOARD THAT WAS OBSERVED, or None for "nobody looked".
                # `scene.objects` is the declared pool: it includes objects
                # parked off the board, it is empty on a scene that failed to
                # load, and `on_board` is None everywhere until a snapshot
                # arrives.  Recording any of those as a board would certify a
                # route against a board that was never seen — the inversion
                # the gate's "not recorded is not empty" rule exists to stop.
                obstacles=_observed_board(scene),
            )
        except Exception:
            log.exception("could not record the episode for plan %s",
                          proposal.plan_id)

    def _execute_locked(self, proposal, should_cancel, on_phase) -> ExecutionResult:
        if proposal.task_type != "pick_place":
            return self._execute_ability_locked(proposal, should_cancel, on_phase)
        return self._execute_pick_place_locked(proposal, should_cancel, on_phase)

    # -- abilities ---------------------------------------------------------

    def _execute_ability_locked(self, proposal, should_cancel, on_phase):
        """Fly one ability's route, under the lease, and prove where it ended.

        CANCELLATION IS PER STAGE, and the stages are not interchangeable.
        Entry and exit are corridors: mid-corridor the arm is between rails and
        cannot stop where it is, so a Stop finishes the waypoint it is flying
        and holds.  The action itself — a wave — can stop between cycles.  None
        of these is an emergency stop, and the panel must not word them as one.

        NO UNVERIFIED AUTO-RETREAT.  A route stopped half way leaves the arm at
        a waypoint, and the transition from that waypoint to anywhere else is
        the thing that was never measured.  It is reported, not corrected.

        The flying itself happens in the motion process; what is left here is
        everything that needs the simulator link — the lease, a fresh view of
        the board, and the before-and-after that decides whether the route
        arrived without disturbing anything.
        """
        granted, why = self._link.acquire_control(
            MOTION_CLIENT_ID, reason=f"plan {proposal.plan_id}",
            ttl_s=LEASE_TTL_S)
        if not granted:
            return ExecutionResult(
                status="failed",
                detail=f"I could not take control of the scene: {why}")

        def phase(name: str) -> None:
            log.info("ability phase: %s", name)
            if on_phase is not None:
                on_phase(name)

        try:
            scene = self._fresh_scene()
            if scene is None:
                return ExecutionResult(
                    status="failed",
                    detail=("I took control of the scene but no fresh view of "
                            "it arrived, so I will not move."))
            ok, why = self.available(proposal)
            if not ok:
                return ExecutionResult(
                    status="failed",
                    detail=f"the board changed while I was taking control: {why}",
                    evidence={"scene_changed": True})

            # Read BEFORE the job, so the approach is judged too.  An arm
            # that swept something off the table on its way to the route's
            # start has disturbed the board, and reading after the approach
            # was the one way for that to go unnoticed.
            before = {oid: o.position for oid, o in scene.objects.items()
                      if o.position is not None}

            out = self._worker.run(
                {"kind": "ability",
                 "task_type": proposal.task_type,
                 "route": proposal.route,
                 "expected_start_posture": proposal.expected_start_posture,
                 "scene": scene.name,
                 # Pointing plans against the board, so the board travels with
                 # the job: the motion process has no link to the simulator.
                 # `object_id` is the scene's own id by the time it gets here
                 # — the planner resolves "soda can" to `soda_can` and writes
                 # it back — so the worker looks it up rather than guessing.
                 "cell": _cell_id(proposal.cell),
                 "object_id": proposal.object_id or "",
                 "scene_file": self._scene_file,
                 "live": {oid: list(o.position)
                          for oid, o in scene.objects.items()
                          if o.position is not None},
                 "sdk": {"host": self._sdk_host, "port": self._sdk_port}},
                on_phase=phase, should_cancel=should_cancel)
            if out.get("status") != "moved":
                # The motion process already knows the answer: a posture it
                # has no route out of, a path it will not invent, a deadline.
                return ExecutionResult(
                    status=out.get("status", "failed"),
                    detail=out.get("detail", "the motion failed"),
                    evidence=out.get("evidence") or {})

            return self._verify_posture(proposal, out.get("flown") or [],
                                        out.get("final_posture"), before,
                                        out.get("start_posture") or "")

        except Exception as exc:
            log.exception("ability execution failed")
            return ExecutionResult(
                status="failed",
                detail=f"the motion failed: {exc.__class__.__name__}: {exc}")
        finally:
            self._link.release_control()

    def _verify_posture(self, proposal, flown, posture, before,
                        start_posture="") -> ExecutionResult:
        """Judge where the arm ended up, and check the board is where it was.

        Both, because they fail independently: the arm can arrive at HOME
        having swept an object off the table on the way, and it can leave the
        board untouched while stopping three waypoints short.

        The posture is read in the motion process, because that is where the
        arm is; the board is read here, because that is where the link is.
        """
        _ensure_paths()
        from reachy_ai.motion import rig_routes as R

        # Where the ability said it would leave the arm.  The wave ends raised,
        # on purpose and out loud: stowing afterwards is a separate request,
        # not something appended so the arm looks tidy.
        wanted = proposal.route
        end = proposal.end_posture or {
            "PLACE_ROUTE": R.POSTURE_REST,
            "STOW_ROUTE": R.POSTURE_HOME,
            "WAVE": R.POSTURE_PRESENT}.get(wanted, "")
        drift = {}
        scene = self._scene_provider()
        for oid, was in before.items():
            now = scene.objects.get(oid)
            if now is not None and now.position is not None:
                moved = _euclid(now.position, was)
                if moved > 0.02:
                    drift[oid] = round(moved, 4)

        evidence = {"route": wanted, "waypoints_flown": list(flown),
                    "start_posture": start_posture,
                    "final_posture": posture, "object_drift": drift}
        if posture != end:
            return ExecutionResult(
                status="failed",
                detail=(f"the route stopped before {end}: my arm is at "
                        f"{posture or 'no posture I recognise'}."),
                evidence=evidence)
        if drift:
            names = ", ".join(f"{k} by {v * 100:.0f} cm" for k, v in drift.items())
            return ExecutionResult(
                status="failed",
                detail=(f"I reached {end}, but I moved something on the way: "
                        f"{names}."),
                evidence=evidence)
        if proposal.task_type.startswith("point"):
            # "The trajectory finished" is not "it pointed accurately", and the
            # brief is explicit that this is a HOVER POINTER rather than a
            # calibrated ray.  The miss is 2-9 cm over an empty board, so
            # saying so is the difference between a claim and a measurement.
            said = ", ".join(str(f)[len("point: "):] for f in flown
                             if str(f).startswith("point: "))
            return ExecutionResult(
                status="completed",
                detail=(f"I hovered over it — {said}. That is a hover, not a "
                        "calibrated ray. My arm is back at "
                        f"{end}, and the board is as it was."),
                evidence=evidence)
        return ExecutionResult(
            status="completed",
            detail=f"My arm is at {end}, and the board is as it was.",
            evidence=evidence)

    def _execute_pick_place_locked(self, proposal, should_cancel, on_phase):
        target_id = proposal.target_id

        granted, why = self._link.acquire_control(
            MOTION_CLIENT_ID,
            reason=f"plan {proposal.plan_id}",
            ttl_s=LEASE_TTL_S,
        )
        if not granted:
            return ExecutionResult(
                status="failed",
                detail=f"I could not take control of the scene: {why}",
            )

        def phase(name: str) -> None:
            log.info("executor phase: %s", name)
            if on_phase is not None:
                on_phase(name)

        try:
            # Nothing about the board is read until here.  Taking the lease is
            # what makes the next few lines mean anything: they describe a
            # scene no other client can now change.
            phase("checking the board")
            scene = self._fresh_scene()
            if scene is None:
                return ExecutionResult(
                    status="failed",
                    detail=("I took control of the scene but no fresh view of "
                            "it arrived, so I will not move against a stale "
                            "picture of the board."),
                )

            ok, why = self.available(proposal)
            if not ok:
                # Distinct from a motion failure on purpose.  "The board
                # changed while I was taking control" is an answer the operator
                # can act on; "the motion finished but soda_can is not on r2c2"
                # sends them looking for a grasp problem that is not there.
                return ExecutionResult(
                    status="failed",
                    detail=f"the board changed while I was taking control: {why}",
                    evidence={"scene_changed": True, "sim_step": scene.sim_step},
                )

            cell = scene.cells[_ref_id(proposal.destination)]
            started_step = scene.sim_step

            # The scene document supplies geometry; the live snapshot supplies
            # where things actually are.  Planning against the YAML's initial
            # poses would aim the gripper at where an object started.
            live = {oid: list(o.position) for oid, o in scene.objects.items()
                    if o.position is not None}

            out = self._worker.run(
                {"kind": "pick_place",
                 "target_id": target_id,
                 "cell_xy": [cell.x, cell.y],
                 "scene_file": self._scene_file,
                 "live": live,
                 "sdk": {"host": self._sdk_host, "port": self._sdk_port}},
                on_phase=phase, should_cancel=should_cancel)
            if out.get("status") == "cancelled":
                return ExecutionResult(
                    status="cancelled",
                    detail=out.get("detail", "Stopped."),
                    evidence=out.get("evidence") or {})
            if out.get("status") != "moved":
                return ExecutionResult(
                    status=out.get("status", "failed"),
                    detail=out.get("detail", "the motion failed"),
                    evidence=out.get("evidence") or {})

            return self._verify(target_id, cell, started_step)

        except Exception as exc:
            log.exception("execution failed")
            return ExecutionResult(
                status="failed",
                detail=f"the motion failed: {exc.__class__.__name__}: {exc}",
            )
        finally:
            self._link.release_control()

    # -- reading the board under the lease ---------------------------------

    def _fresh_scene(self):
        """A scene view built from a snapshot captured after the lease.

        Moving the read below `acquire_control` is not on its own enough.  The
        link hands back whatever the simulator last pushed, and that push may
        predate the grant by most of a frame interval — so the "post-lease"
        read can still describe the board as it was while other clients could
        edit it.  Waiting for a snapshot whose `received_at` is later than the
        grant is what closes that.

        Returns None rather than a stale view if none arrives.  A link that
        cannot produce a current picture of the board is a link this must not
        plan a half-metre arm movement against.
        """
        granted_at = time.monotonic()
        deadline = granted_at + FRESH_SNAPSHOT_TIMEOUT_S
        while time.monotonic() < deadline:
            snap = self._link.snapshot()
            if snap is not None and snap.received_at >= granted_at:
                scene = self._scene_provider()
                # The provider builds from the link's current snapshot, which
                # is this one or a newer one.  Newer is fine; older is not, and
                # `live` false means it fell back to the scene file.
                if scene.live and not scene.error:
                    return scene
            time.sleep(0.05)
        return None

    # -- result verification ----------------------------------------------

    def _verify(self, target_id, cell, started_step) -> ExecutionResult:
        """Did the object actually end up on the cell?

        Read back from the simulator, not inferred from the trajectory having
        finished.  Every segment can run to completion while the grasp slipped
        somewhere in the middle, and that is a failure the panel must report as
        one.
        """
        deadline = time.time() + SETTLE_TIMEOUT_S
        position = None
        while time.time() < deadline:
            scene = self._scene_provider()
            obj = scene.objects.get(target_id)
            if (scene.live and scene.sim_step > started_step
                    and obj is not None and obj.position is not None):
                position = obj.position
                if cell.contains(position):
                    return ExecutionResult(
                        status="completed",
                        detail=f"{target_id} is on {cell.name}.",
                        evidence={
                            "target_id": target_id,
                            "cell": cell.name,
                            "position": list(position),
                            "sim_step": scene.sim_step,
                            "verified": "live_pose",
                        },
                    )
            time.sleep(0.2)

        where = (f"at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                 if position else "somewhere I cannot see")
        return ExecutionResult(
            status="failed",
            detail=(f"The motion finished but {target_id} is not on "
                    f"{cell.name} — it is {where}."),
            evidence={
                "target_id": target_id,
                "cell": cell.name,
                "position": list(position) if position else None,
                "verified": "live_pose",
            },
        )


def _observed_board(scene) -> Optional[List[str]]:
    """Which objects were on the board, or None if that was not observed.

    None is a real answer and the conservative one.  `on_board is True` and
    not merely truthy, because it is None until a snapshot has been applied
    and unknown is not absent — the same test `SceneView.on_board_tagged`
    makes for the same reason.
    """
    if scene is None or getattr(scene, "error", ""):
        return None
    objects = getattr(scene, "objects", None) or {}
    if any(o.on_board is None for o in objects.values()):
        return None
    return sorted(oid for oid, o in objects.items() if o.on_board is True)


def _euclid(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def _cell_id(cell) -> str:
    """The scene's own id for a cell the planner named, or "" for no cell.

    The panel says `r2c2`; `SceneModel` calls it `cell_r2c2`.  One place that
    conversion happens, rather than wherever it is next needed.
    """
    if not cell:
        return ""
    name = _ref_id(str(cell))
    return name if name.startswith("cell_") else f"cell_{name}"


def _ref_id(destination: str) -> str:
    return destination.split(":", 1)[1] if ":" in destination else destination


def _ensure_paths() -> None:
    """Put the repo's src/ on the path, whether running from a checkout or /opt."""
    here = os.path.dirname(os.path.abspath(__file__))
    for candidate in (os.path.join(here, "..", "src"), "/opt/src"):
        if os.path.isdir(candidate) and candidate not in sys.path:
            sys.path.insert(0, candidate)


def build_executor(link, scene_provider, scene_file: str):
    """The executor for this deployment, or a NullExecutor explaining itself.

    Enabled by REACHY_PANEL_EXECUTOR, default off.  Live motion is opt-in
    because turning it on changes what pressing Confirm does to the world, and
    that should be a deliberate act by whoever runs the container rather than a
    side effect of upgrading the page.
    """
    if os.environ.get("REACHY_PANEL_EXECUTOR", "").lower() not in ("1", "true", "yes"):
        return NullExecutor(
            "live execution is switched off on this server "
            "(set REACHY_PANEL_EXECUTOR=1 to enable it)"
        )
    return SimulatorExecutor(link, scene_provider, scene_file,
                            recorder=build_recorder(scene_file))
