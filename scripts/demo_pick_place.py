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

_STEP_HZ = 25


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


def _run_segment(arm, planner, start, end, seed, steps, attacher=None, object_id=None):
    """Plan a collision-checked Cartesian segment and execute it.  Returns the
    final joint solution (seed for the next segment)."""
    traj, cart = planner.plan_segment(start, end, steps, seed)

    on_step = None
    if attacher is not None and object_id is not None:
        def on_step(i, q, _c=cart):
            attacher.follow(object_id, _c[i])

    P.execute_trajectory(arm, traj, R_ARM_JOINTS, rate_hz=_STEP_HZ, on_step=on_step)
    return traj[-1]


def pick_and_place(robot, planner, scene, attacher, object_id, seed):
    """Full pick-and-place cycle for one object.  Returns the ending seed."""
    arm = robot.r_arm
    obj = scene.get(object_id)
    place_xy = _PLACE_XY[object_id]

    hover  = scene.hover_point(object_id)
    grasp  = scene.grasp_point(object_id)
    lift   = (grasp[0], grasp[1], scene.carry_z())
    carry  = (place_xy[0], place_xy[1], scene.carry_z())
    place  = scene.rest_point(place_xy, object_id)
    retract = (place_xy[0], place_xy[1], scene.carry_z())

    # Head tracks the object being manipulated.
    P.look_at(robot, grasp, duration=1.0)

    # Approach in JOINT space (the arm lifts naturally): a straight Cartesian
    # line from READY to the hover point would clip the table — the collision
    # model rejects it, so we move joint-wise to the hover pose instead.
    log.info("── %s: approach above object", object_id)
    hover_q = planner.solve(hover, seed=seed)
    P.smooth_move(arm, dict(zip(R_ARM_JOINTS, hover_q)), duration=2.0)
    seed = hover_q

    log.info("── %s: descend to grasp", object_id)
    seed = _run_segment(arm, planner, hover, grasp, seed, 20)

    log.info("── %s: close gripper", object_id)
    P.close_gripper(arm)
    attacher.follow(object_id, grasp)   # marker now tracks the gripper

    log.info("── %s: lift", object_id)
    seed = _run_segment(arm, planner, grasp, lift, seed, 20, attacher, object_id)

    log.info("── %s: carry across table", object_id)
    P.look_at(robot, place, duration=0.8)   # head tracks toward the drop site
    seed = _run_segment(arm, planner, lift, carry, seed, 35, attacher, object_id)

    log.info("── %s: descend to place", object_id)
    seed = _run_segment(arm, planner, carry, place, seed, 20, attacher, object_id)

    log.info("── %s: open gripper (release)", object_id)
    P.open_gripper(arm)
    attacher.release(object_id, place)   # object stays at its new location

    log.info("── %s: retract", object_id)
    seed = _run_segment(arm, planner, place, retract, seed, 20)
    return seed


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
    attacher.reset()

    robot.turn_on("r_arm")
    time.sleep(0.3)
    log.info("Right arm ON")

    # Lift out of the rest pose in joint space (avoids sweeping up through the
    # table that a Cartesian straight-line from arms-down would cause).
    log.info("── Move to READY pose")
    P.smooth_move(arm, P.READY, duration=2.0)
    seed = [P.READY[n] for n in R_ARM_JOINTS]

    # Tilt the head/cameras down toward the tabletop workspace.
    ws = scene.get(scene.manipulable_ids()[0]).center
    P.look_at(robot, (ws[0], (ws[1] - 0.05), scene.table_surface_z), duration=1.0)

    for object_id in scene.manipulable_ids():
        log.info("=" * 56)
        px, py = _PLACE_XY.get(object_id, (None, None))
        log.info("OBJECT: %s  → place (%.2f, %.2f)", object_id, px, py)
        log.info("=" * 56)
        seed = pick_and_place(robot, planner, scene, attacher, object_id, seed)
        log.info("── Return to READY")
        P.smooth_move(arm, P.READY, duration=1.5)
        seed = [P.READY[n] for n in R_ARM_JOINTS]

    log.info("=" * 56)
    log.info("── Return HOME and turn off")
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
