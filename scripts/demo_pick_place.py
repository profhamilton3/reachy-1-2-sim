#!/usr/bin/env python3
"""
Scene-aware right-arm pick-and-place demo for Reachy 1.2.

Uses preloaded scene awareness (src/reachy_ai/scene) + Cartesian IK planning
(src/reachy_ai/motion/kinematics) so the gripper always approaches objects from
above and never drives through the table.  Each tracked object is grasped and
relocated across the right arm's reachable zone.

Backends (REACHY_SIM_BACKEND):
  kinematic  (default) — no grasp physics; the grasped object's RViz marker is
             animated to follow the gripper via /tmp/reachy_scene_overrides.json
             so the pick-and-place is visible in the VNC/RViz viewer.
  mujoco-remote        — real MuJoCo physics; the gripper physically grasps and
             carries the object (visible in the camera view on port 8080).
             Marker overrides are disabled (physics moves the real object).

Run inside the container:
  docker compose exec reachy-sim python3 /opt/scripts/demo_pick_place.py
Observe: VNC viewer http://localhost:6080 (RViz) — or http://localhost:8080
(camera) when running the MuJoCo backend.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

# Make src/ importable whether run from repo root or /opt inside the container.
_HERE = pathlib.Path(__file__).resolve().parent
for _c in (_HERE.parent / "src", pathlib.Path("/opt/src")):
    if _c.is_dir():
        sys.path.insert(0, str(_c))

try:
    from reachy_sdk import ReachySDK
except ImportError:
    print("ERROR: reachy-sdk not installed (run inside the simulator container).")
    sys.exit(1)

from reachy_ai.motion import primitives as P
from reachy_ai.motion.kinematics import CartesianPlanner, R_ARM_JOINTS
from reachy_ai.motion.safety import gate_check
from reachy_ai.scene.awareness import SceneModel
# The pick-and-place arc moved to reachy_ai.tasks.pick_place_live (issue #51)
# so the command panel's executor drives the same tuned motion this demo does,
# rather than a second copy of it that would drift.  Behaviour here is
# unchanged: the place site this script used to look up internally is now
# passed in from _PLACE_XY below.
from reachy_ai.tasks.pick_place_live import CLEAR_Z as _CLEAR_Z  # noqa: F401
# Issue #56: the demo preflights the COMPLETE planned job -- the raise footprint,
# every arc with its return to the raised pose, and the stow footprint -- before
# it turns the arm on, then executes the accepted arcs.
from reachy_ai.tasks.pick_place_live import (
    PreflightRefused,
    ReturnArrivalError,
    execute_arc,
    preflight_pick_place,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")

_OVERRIDES = os.environ.get("REACHY_SIM_SCENE_OVERRIDES", "/tmp/reachy_scene_overrides.json")

# Place targets (world xy) for each object — inside the right arm's reachable
# zone (x in [0.38,0.46], y in [-0.22,+0.02]).  "Other side of the table"
# interpreted within reach: each object crosses the reachable zone.
_PLACE_XY = {
    "red_cube":      (0.42, -0.02),   # front-right → toward centre
    "blue_cylinder": (0.40, -0.20),   # centre → front-right
}



class MarkerAttacher:
    """Writes /tmp/reachy_scene_overrides.json so grasped-object markers follow
    the gripper in the kinematic/RViz view.  No-op in physics mode."""

    def __init__(self, enabled: bool, path: str = _OVERRIDES) -> None:
        self.enabled = enabled
        self.path = path
        self.placed: dict = {}   # persistent positions of already-moved objects

    def _write(self, extra: dict) -> None:
        if not self.enabled:
            return
        data = {**self.placed, **extra}
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)

    def follow(self, object_id: str, xyz) -> None:
        self._write({object_id: list(xyz)})

    def release(self, object_id: str, xyz) -> None:
        self.placed[object_id] = list(xyz)
        self._write({})

    def reset(self) -> None:
        if self.enabled:
            try:
                os.remove(self.path)
            except OSError:
                pass


def run_demo(host: str, port: int, scene_path: str) -> None:
    if not gate_check():
        log.error("Safety gate failed — aborting.")
        sys.exit(1)

    backend = os.environ.get("REACHY_SIM_BACKEND", "kinematic").lower()
    physics = backend == "mujoco-remote"
    log.info("Backend: %s  (marker attachment %s)",
             backend, "OFF — real physics" if physics else "ON — kinematic RViz")

    scene = SceneModel.from_yaml(scene_path)
    log.info("Scene '%s': table surface z=%.3f, objects=%s",
             scene.frame_id, scene.table_surface_z, scene.manipulable_ids())

    robot = ReachySDK(host=host, sdk_port=port)
    time.sleep(0.8)
    arm = robot.r_arm
    if arm is None:
        log.error("r_arm not found — is the simulator running?")
        sys.exit(1)

    planner = CartesianPlanner(arm, scene=scene)
    attacher = MarkerAttacher(enabled=not physics)

    # PREFLIGHT THE WHOLE JOB BEFORE ANY MOTION (#56).  Planning sends no motion
    # command (IK service calls only).  A move whose arc is refused is skipped
    # and logged with its full refusal; its object stays where it is.  The demo
    # board and _PLACE_XY are NOT altered to avoid refusals: on a crowded board
    # this may skip objects or refuse the whole job, and that is the point.
    moves = []
    for object_id in scene.manipulable_ids():
        if object_id not in _PLACE_XY:
            log.warning("OBJECT %s has no place site in _PLACE_XY; skipping",
                        object_id)
            continue
        moves.append((object_id, _PLACE_XY[object_id]))
    log.info("── Preflight: checking the whole job before moving anything")
    try:
        job = preflight_pick_place(planner, scene, moves, skip_refused=True)
    except PreflightRefused as exc:
        log.error("Preflight REFUSED the whole job — no motion made: %s", exc)
        sys.exit(1)
    for object_id, why in job.skipped:
        log.warning("SKIPPED %s: %s", object_id, why)
    if not job.arcs:
        log.error("Preflight accepted no arcs — no motion made.")
        sys.exit(1)

    attacher.reset()
    robot.turn_on("r_arm")
    time.sleep(0.3)
    log.info("Right arm ON")

    # Raise to the PRESENT pose by the measured route out of the rail pocket.
    log.info("── Raise arm to the raised pose")
    P.raise_to_side(arm, duration=3.0)

    # Tilt the head/cameras down to the centroid of all manipulable objects on
    # the table.  Using all object centres (not just the first one) ensures
    # the gaze lands near the middle of the workspace cluster.
    ids = scene.manipulable_ids()
    gaze_x = sum(scene.get(oid).center[0] for oid in ids) / len(ids)
    gaze_y = sum(scene.get(oid).center[1] for oid in ids) / len(ids)
    log.info("── Tilting head to tabletop workspace (%.2f, %.2f, %.2f)",
             gaze_x, gaze_y, scene.table_surface_z)
    P.look_at(robot, (gaze_x, gaze_y, scene.table_surface_z), duration=1.5)

    for arc in job.arcs:
        log.info("=" * 56)
        px, py = arc.place_xy
        log.info("OBJECT: %s  → place (%.2f, %.2f)", arc.object_id, px, py)
        log.info("=" * 56)
        try:
            execute_arc(robot, arc, planner, attacher)
        except ReturnArrivalError as exc:
            # The stow's precondition (the raised pose) is not met: stop here.
            log.error("Arm did not arrive at the raised pose — NOT stowing: %s", exc)
            sys.exit(1)
        # Each arc ends at the raised pose by its own checked return, so the
        # next pick starts from there and the stow below has its precondition.

    log.info("=" * 56)
    log.info("── Stow: reverse the raise by the measured route")
    P.go_home(robot, arm, duration=3.0)
    log.info("Demo complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reachy 1.2 scene-aware pick-and-place demo")
    ap.add_argument("--host", default=os.environ.get("REACHY_IP", "localhost"))
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--scene", default=os.environ.get(
        "REACHY_SIM_SCENE", "/opt/scenes/tabletop_demo.yaml"))
    args = ap.parse_args()
    run_demo(args.host, args.port, args.scene)


if __name__ == "__main__":
    main()
