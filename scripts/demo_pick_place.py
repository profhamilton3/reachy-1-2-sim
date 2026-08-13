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
# Over-table transit segments (swing-in, lift, carry, swing-out) stream at a
# SLOWER rate so the physics arm has time to actually track each setpoint at the
# commanded clear height instead of lagging and sagging below it.  ~8 Hz gives
# the arm ~125 ms per setpoint — plenty for the stiffened actuators to converge,
# so the hand holds the high arc tightly over the table.
_CARRY_HZ = 8

# The arm arcs through this height (m) when travelling between pick and place.
# Chosen 0.26 m above the table surface (0.74) so the whole hand AND forearm are
# visibly clear of the tabletop and any objects sitting on it (the pad-only
# collision model doesn't see the forearm, so we keep the transit high).
_CLEAR_Z = 1.00


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


def _run_segment(arm, planner, start, end, seed, steps, attacher=None,
                 object_id=None, rate_hz=_STEP_HZ):
    """Plan a collision-checked Cartesian segment and execute it.  Returns the
    final joint solution (seed for the next segment).  ``rate_hz`` sets the
    streaming rate — pass a slower rate for over-table transits so the physics
    arm tracks the commanded height instead of sagging."""
    traj, cart = planner.plan_segment(start, end, steps, seed)

    on_step = None
    if attacher is not None and object_id is not None:
        def on_step(i, q, _c=cart):
            attacher.follow(object_id, _c[i])

    P.execute_trajectory(arm, traj, R_ARM_JOINTS, rate_hz=rate_hz, on_step=on_step)
    return traj[-1]


def pick_and_place(robot, planner, scene, attacher, object_id, seed, side_pad):
    """Full pick-and-place cycle for one object, approaching over the table's
    right side.  Starts and ends at the SIDE_HIGH hub (``side_pad``).  Returns
    the ending seed.

    Motion arc — the arm is at the SIDE_HIGH hub (up and out to the robot's
    right, clear of the table) at entry and exit.  Every segment is a
    collision-checked Cartesian move; the hand only ever enters the table
    footprint at or above _CLEAR_Z, then descends straight down onto the object:

      SIDE_HIGH → [over table @ _CLEAR_Z] → above pick
                → [down] → hover → grasp → close gripper
                → [up]   → above pick
                → [over table @ _CLEAR_Z] → above place
                → [down] → place → open gripper
                → [up]   → above place
                → [over table @ _CLEAR_Z] → SIDE_HIGH
    """
    arm = robot.r_arm
    place_xy = _PLACE_XY[object_id]

    hover = scene.hover_point(object_id)   # object top + 0.05 m
    grasp = scene.grasp_point(object_id)   # object centre (gripper pads land here)
    place = scene.rest_point(place_xy, object_id)

    # High-clearance waypoints directly above the pick and place sites.  The arm
    # crosses the table only at _CLEAR_Z (0.21 m above the surface).
    above_pick  = (hover[0],     hover[1],     _CLEAR_Z)
    above_place = (place_xy[0],  place_xy[1],  _CLEAR_Z)

    # 1. Head looks toward the object.
    P.look_at(robot, grasp, duration=1.0)

    # 2. Swing IN from the side, over the table, to directly above the object.
    #    Slow rate so the arm holds the high arc (no sag) while crossing over.
    log.info("── %s: swing in over table to above pick", object_id)
    seed = _run_segment(arm, planner, side_pad, above_pick, seed, 40, rate_hz=_CARRY_HZ)

    # 3. Descend straight down: above → hover → grasp.
    log.info("── %s: descend to hover", object_id)
    seed = _run_segment(arm, planner, above_pick, hover, seed, 25)
    log.info("── %s: descend to grasp", object_id)
    seed = _run_segment(arm, planner, hover, grasp, seed, 15)

    # 4. Close gripper.
    log.info("── %s: close gripper", object_id)
    P.close_gripper(arm)
    attacher.follow(object_id, grasp)

    # 5. Lift straight up to clear height above the pick site (slow, so the arm
    #    actually reaches the clear height before carrying).
    log.info("── %s: lift to clear height", object_id)
    seed = _run_segment(arm, planner, grasp, above_pick, seed, 30, attacher, object_id,
                        rate_hz=_CARRY_HZ)

    # 6. Head tracks toward the place site while the arm swings across.
    P.look_at(robot, (place_xy[0], place_xy[1], scene.table_surface_z), duration=0.8)

    # 7. Carry over the table at clear height to above the place site — slow rate
    #    keeps the hand pinned to the high arc the whole way across.
    log.info("── %s: carry over table at clear height", object_id)
    seed = _run_segment(arm, planner, above_pick, above_place, seed, 40, attacher, object_id,
                        rate_hz=_CARRY_HZ)

    # 8. Descend straight down onto the place position.
    log.info("── %s: descend to place", object_id)
    seed = _run_segment(arm, planner, above_place, place, seed, 30, attacher, object_id)

    # 9. Open gripper; release marker.
    log.info("── %s: open gripper (release)", object_id)
    P.open_gripper(arm)
    attacher.release(object_id, place)

    # 10. Lift straight up to clear height, then return to the side hub by
    #     RE-ABDUCTING the shoulder (roll-first, closed-loop).  A Cartesian
    #     swing-out streams all joints together, which stalls the shoulder-roll
    #     under physics, so use the sequenced raise instead.
    log.info("── %s: retract to clear height", object_id)
    seed = _run_segment(arm, planner, place, above_place, seed, 25, rate_hz=_CARRY_HZ)
    log.info("── %s: re-abduct back out to the side hub", object_id)
    P.raise_to_side(arm)
    return [P.SIDE_HIGH[n] for n in R_ARM_JOINTS]


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

    # Lift up the robot's RIGHT SIDE to the SIDE_HIGH transit hub — the hand
    # swings out past the table's right edge before rising, so it never sweeps
    # up through the tabletop (verified clear in kinematic and physics models).
    log.info("── Raise arm up the side to the SIDE_HIGH hub")
    P.raise_to_side(arm, duration=3.0)
    side_seed = [P.SIDE_HIGH[n] for n in R_ARM_JOINTS]
    side_pad = planner.fk_world(side_seed)
    log.info("   SIDE_HIGH pad=(%.2f, %.2f, %.2f)", *side_pad)
    seed = side_seed

    # Tilt the head/cameras down toward the tabletop workspace.
    ws = scene.get(scene.manipulable_ids()[0]).center
    P.look_at(robot, (ws[0], (ws[1] - 0.05), scene.table_surface_z), duration=1.0)

    for object_id in scene.manipulable_ids():
        log.info("=" * 56)
        px, py = _PLACE_XY.get(object_id, (None, None))
        log.info("OBJECT: %s  → place (%.2f, %.2f)", object_id, px, py)
        log.info("=" * 56)
        seed = pick_and_place(robot, planner, scene, attacher, object_id, seed, side_pad)
        # pick_and_place already returned to the SIDE_HIGH hub via raise_to_side
        # (closed-loop), so the next pick starts fully abducted and table-clear.
        seed = side_seed

    log.info("=" * 56)
    log.info("── Stow: reverse the raise — lower the arm down the side by Reachy")
    # Mirror of raise_to_side: SIDE_HIGH → ABDUCT_LOW → HOME, so the arm unflexes
    # and comes down at the robot's side, never sweeping forward over the table.
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
