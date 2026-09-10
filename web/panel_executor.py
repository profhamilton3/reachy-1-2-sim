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
from typing import Any, Callable, Dict, Optional, Tuple

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

    # -- availability ------------------------------------------------------

    def available(self, proposal=None) -> Tuple[bool, str]:
        """Whether this proposal could be executed right now, and why not."""
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
            return False, f"I have no motion for a {proposal.task_type} task"
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
            robot = ReachySDK(host=self._sdk_host, sdk_port=self._sdk_port)
            time.sleep(0.8)
            if robot.r_arm is None:
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
