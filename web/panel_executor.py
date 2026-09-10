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

COMPLETION IS PROVEN, NOT ASSUMED
---------------------------------
When the arc finishes, the object's live pose is read back and tested against
the destination cell with the same predicate the simulator uses for occupancy.
A trajectory that ran to the end without the object arriving is a failure, and
this is where that gets noticed — the arm can complete every segment while the
grasp slipped.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

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

#: How close to the destination cell the object must end up to count as placed.
#: The cell's own half-extent decides that, so this only bounds the wait for a
#: fresh snapshot after the arm has finished.
SETTLE_TIMEOUT_S = 4.0


@dataclass
class ExecutionResult:
    status: str                       # completed | failed | cancelled
    detail: str = ""
    evidence: Dict[str, Any] = field(default_factory=dict)


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
                 *, sdk_host: str = SDK_HOST, sdk_port: int = SDK_PORT) -> None:
        self._link = link
        self._scene_provider = scene_provider
        self._scene_file = scene_file
        self._sdk_host = sdk_host
        self._sdk_port = sdk_port
        # One task at a time reaches the arm, whatever the coordinator thinks.
        self._motion_lock = threading.Lock()
        # ONE SDK connection, reused.  `ReachySDK.__init__` opens a gRPC
        # channel and starts background sync threads, and nothing here ever
        # stopped them — so every task leaked a channel and its threads.  After
        # a handful the bridge stopped answering new connections at all: the
        # panel kept serving HTTP while no SDK client, including a fresh one
        # from a shell, could connect, and the task that was mid-motion never
        # returned.  Connecting also costs the better part of a second, which
        # is a poor thing to pay per request.
        self._robot = None

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
        if proposal.arm != "right":
            return False, ("I can only do that with my right arm; the route "
                           "was measured for that arm through a rig that is "
                           "not symmetric")

        scene = self._scene_provider()
        if scene.error:
            return False, f"I cannot read the scene: {scene.error}"

        # An ability is FOR a posture, not for one route.  "Store your arm"
        # means end in the pocket, and which measured routes get there depends
        # on where the arm is standing — from the presentation pose it is
        # PRESENT_RETURN then STOW_FROM_SIDE, neither of which is STOW_ROUTE.
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
        try:
            return self._execute_locked(proposal, should_cancel, on_phase)
        finally:
            self._motion_lock.release()

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
        """
        _ensure_paths()
        from reachy_sdk import ReachySDK

        from reachy_ai.motion import rig_routes as R
        from reachy_ai.tasks import rig_motion as M

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

            phase("connecting to the arm")
            robot = self._connect(ReachySDK)
            if robot is None or robot.r_arm is None:
                return ExecutionResult(
                    status="failed",
                    detail="the right arm is not available on the simulator")
            arm = robot.r_arm

            start = R.posture_of(M.present_pose(arm))
            wanted = proposal.expected_start_posture
            recovered: List[str] = []
            if start is None and M.stranded_at(arm) is not None:
                # Left part way through the pocket entry by something that
                # stopped early.  Finishing it is two measured moves, and
                # refusing every later request until a human intervenes is
                # correct and useless.
                phase("finishing the move that was interrupted")
                robot.turn_on("r_arm")
                _wait_for_motors(arm)
                recovered = M.resume_to_home(arm, on_phase=phase)
                start = R.posture_of(M.present_pose(arm))
            if start is None:
                # Not at any named posture, and not on the pocket sequence
                # either.  The nearest waypoint is reported because it is the
                # useful fact, and NOT flown to, because the segment onto it is
                # the one nobody measured.
                name, distance = R.nearest_waypoint(M.present_pose(arm))
                return ExecutionResult(
                    status="failed",
                    detail=("my arm is not at a posture I have a measured "
                            f"route out of — the nearest waypoint is {name}, "
                            f"{distance:.0f} degrees away. I will not guess a "
                            "path from here."),
                    evidence={"recovery_needed": True, "nearest": name})

            # The arm is usually parked with the motors OFF — the stow that
            # ran before this one turns them off deliberately — and turning
            # them on is not instant.  Asking for 40 degrees of shoulder
            # before the controller has taken hold produced exactly one
            # symptom: the arm did not move at all, and the pocket-exit guard
            # refused, roughly one attempt in three.  So wait for the joint to
            # report itself driven rather than sleeping a guessed interval.
            robot.turn_on("r_arm")
            if not _wait_for_motors(arm):
                return ExecutionResult(
                    status="failed",
                    detail=("the arm's motors did not come on, so I will not "
                            "command a move it cannot make."),
                    evidence={"motors_off": True})

            # GETTING THERE IS PART OF DOING IT.  An arm stored in the rail
            # pocket cannot wave from where it is, and that is a fact about
            # the rig rather than something the operator should have to know
            # and type.  So the approach is flown, by measured edges only —
            # `travel` refuses rather than inventing one.
            approach: List[str] = list(recovered)
            if wanted and start != wanted:
                steps = R.path(start, wanted)
                if steps is None:
                    return ExecutionResult(
                        status="failed",
                        detail=(f"my arm is at {start} and that needs to start "
                                f"at {wanted}. There is no measured way from "
                                f"{start} to {wanted}, so I will not invent "
                                "one."),
                        evidence={"posture": start, "wanted": wanted})
                for route in steps:
                    ok, why = R.check_route(route, scene.name)
                    if not ok:
                        return ExecutionResult(
                            status="failed",
                            detail=(f"getting from {start} to {wanted} means "
                                    f"flying {route} first, and {why}"),
                            evidence={"posture": start, "wanted": wanted,
                                      "blocked_on": route})
                phase(f"leaving {start}")
                approach += M.travel(arm, wanted, robot=robot,
                                     should_abort=(should_cancel or (lambda: False)),
                                     on_phase=phase)
                robot.turn_on("r_arm")
                time.sleep(0.2)

            time.sleep(0.3)
            before = {oid: o.position for oid, o in scene.objects.items()
                      if o.position is not None}

            def go_to(posture):
                def _run(a, **kw):
                    return M.travel(a, posture, robot=robot, **kw)
                return _run

            runner = {"rest_forearm": go_to(R.POSTURE_REST),
                      "stow_arm": go_to(R.POSTURE_HOME),
                      "wave": lambda a, **kw: ["wave x%d" % M.wave(a, **kw)],
                      }.get(proposal.task_type)
            if runner is None:
                return ExecutionResult(
                    status="failed",
                    detail=f"I have no runner wired up for {proposal.task_type}")

            try:
                flown = runner(arm, should_abort=(should_cancel or (lambda: False)),
                               on_phase=phase)
            except M.RecoveryNeeded as exc:
                return ExecutionResult(
                    status="failed", detail=str(exc),
                    evidence={"recovery_needed": True})
            except M.RouteError as exc:
                return ExecutionResult(
                    status="failed", detail=str(exc),
                    evidence={"stopped_mid_route": True})

            return self._verify_posture(arm, proposal, approach + flown, before)

        except Exception as exc:
            log.exception("ability execution failed")
            return ExecutionResult(
                status="failed",
                detail=f"the motion failed: {exc.__class__.__name__}: {exc}")
        finally:
            self._link.release_control()

    def _connect(self, sdk_class):
        """The one SDK connection, made once and kept.

        A connection that has gone stale raises on first use rather than
        returning nonsense, so the retry is a reconnect and not a guess.
        """
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
                log.info("SDK connection went stale; reconnecting")
                self._close()
        self._robot = sdk_class(host=self._sdk_host, sdk_port=self._sdk_port)
        time.sleep(0.8)
        return self._robot

    def _close(self) -> None:
        robot, self._robot = self._robot, None
        if robot is None:
            return
        try:
            robot._stop()
        except Exception:                         # noqa: BLE001 - best effort
            log.debug("could not stop the SDK connection cleanly")

    def _verify_posture(self, arm, proposal, flown, before) -> ExecutionResult:
        """Read the posture back, and check the board is where it was.

        Both, because they fail independently: the arm can arrive at HOME
        having swept an object off the table on the way, and it can leave the
        board untouched while stopping three waypoints short.
        """
        _ensure_paths()
        from reachy_ai.motion import rig_routes as R
        from reachy_ai.tasks import rig_motion as M

        # Where the ability said it would leave the arm.  The wave ends raised,
        # on purpose and out loud: stowing afterwards is a separate request,
        # not something appended so the arm looks tidy.
        wanted = proposal.route
        end = proposal.end_posture or {
            "PLACE_ROUTE": R.POSTURE_REST,
            "STOW_ROUTE": R.POSTURE_HOME,
            "WAVE": R.POSTURE_PRESENT}.get(wanted, "")
        posture = R.posture_of(M.present_pose(arm))
        drift = {}
        scene = self._scene_provider()
        for oid, was in before.items():
            now = scene.objects.get(oid)
            if now is not None and now.position is not None:
                moved = _euclid(now.position, was)
                if moved > 0.02:
                    drift[oid] = round(moved, 4)

        evidence = {"route": wanted, "waypoints_flown": list(flown),
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
        return ExecutionResult(
            status="completed",
            detail=f"My arm is at {end}, and the board is as it was.",
            evidence=evidence)

    def _execute_pick_place_locked(self, proposal, should_cancel, on_phase):
        _ensure_paths()
        from reachy_sdk import ReachySDK

        from reachy_ai.motion import primitives as P
        from reachy_ai.motion.kinematics import CartesianPlanner
        from reachy_ai.scene.awareness import SceneModel
        from reachy_ai.tasks.pick_place_live import pick_and_place, side_hub

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
            phase("connecting to the arm")
            robot = self._connect(ReachySDK)
            if robot is None or robot.r_arm is None:
                return ExecutionResult(
                    status="failed",
                    detail="the right arm is not available on the simulator",
                )

            # The scene document supplies geometry; the live snapshot supplies
            # where things actually are.  Planning against the YAML's initial
            # poses would aim the gripper at where an object started.
            model = SceneModel.from_yaml(self._scene_file)
            live = {oid: o.position for oid, o in scene.objects.items()
                    if o.position is not None}
            model.update_poses(live)

            planner = CartesianPlanner(robot.r_arm, scene=model)
            robot.turn_on("r_arm")
            time.sleep(0.3)

            phase("raising to the transit hub")
            P.raise_to_side(robot.r_arm, duration=3.0)
            seed, side_pad = side_hub(planner)

            cancelled = {"flag": False}

            def abort() -> bool:
                if should_cancel is not None and should_cancel():
                    cancelled["flag"] = True
                    return True
                return False

            pick_and_place(
                robot, planner, model, None, target_id, seed, side_pad,
                place_xy=(cell.x, cell.y),
                should_abort=abort,
                on_phase=phase,
            )

            phase("returning home")
            P.go_home(robot, robot.r_arm, duration=3.0)

            if cancelled["flag"]:
                # Only now, with the arm actually stopped and parked, is it
                # true to say the motion has stopped.
                return ExecutionResult(
                    status="cancelled",
                    detail="Stopped, and the arm is back at rest.",
                    evidence={"cancelled_at_phase": True},
                )

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


#: How long to wait for `turn_on` to actually take effect.
MOTOR_ON_TIMEOUT_S = 3.0


def _wait_for_motors(arm, timeout: float = MOTOR_ON_TIMEOUT_S) -> bool:
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


def _euclid(a, b) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


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
    return SimulatorExecutor(link, scene_provider, scene_file)
