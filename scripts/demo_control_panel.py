#!/usr/bin/env python3
"""
Two-arm control-panel practice for Reachy 1.2.

Reachy operates the interactive console defined in scenes/control_panel.yaml:
presses the spring buttons (they latch on/off and light up) and flips the
grippable switches/levers (bistable — they snap on/off).  The right gripper
works the y<0 side of the panel and the left gripper the y>0 side; controls near
the midline go to the right arm.

Interactivity lives in the physics backend (native_mujoco/interactive.py), so run
this against the MuJoCo backend to see it work:

  REACHY_SIM_SCENE=scenes/control_panel.yaml ./scripts/start_sim.sh
  docker compose exec reachy-sim python3 /opt/scripts/demo_control_panel.py

Observe the camera (http://localhost:8080): controls light up as Reachy operates
them.  On/off states are also mirrored to /tmp/reachy_interactive_state.json,
which this script reads back to confirm each toggle.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

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
from reachy_ai.motion.kinematics import CartesianPlanner, UnreachableError
from reachy_ai.motion.safety import gate_check
from reachy_ai.scene.awareness import SceneModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("panel")

_STATE_FILE = os.environ.get("REACHY_SIM_INTERACTIVE_STATE",
                             "/tmp/reachy_interactive_state.json")
_STANDOFF = 0.09        # m in front of a control before actuating
_PRESS_DEPTH = 0.012    # m the finger drives a button in
_FLIP_PUSH = 0.06       # m the finger drags a switch/lever handle to flip it
_STEP_HZ = 10           # slow streaming so the physics arm tracks


def _add(p, d, s):
    return (p[0] + d[0] * s, p[1] + d[1] * s, p[2] + d[2] * s)


def _read_state(control_id: str):
    try:
        with open(_STATE_FILE) as f:
            return json.load(f).get(control_id)
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _pose(planner, xyz, seed):
    """IK a world pad point → {joint: deg} dict (or None if unreachable)."""
    try:
        q = planner.solve(xyz, seed=seed)
    except (UnreachableError, Exception):
        return None
    return dict(zip(planner.joints, q)), q


def operate_button(robot, arm, planner, target, side, seed):
    """Approach from a standoff, press the button, retract."""
    out = target.off_dir                       # points away from the panel
    press_to = _add(target.point, target.actuate_dir, _PRESS_DEPTH)
    standoff = _add(target.point, out, _STANDOFF)

    P.look_at(robot, target.point, duration=0.6)
    for label, xyz in (("standoff", standoff), ("press", press_to), ("release", standoff)):
        sol = _pose(planner, xyz, seed)
        if sol is None:
            log.warning("   %s: %s unreachable — skipping", target.id, label)
            return seed
        pose, seed = sol
        P.smooth_move(arm, pose, duration=1.2 if label != "press" else 0.6)
    return seed


def operate_hinge(robot, arm, planner, target, side, seed):
    """Approach the handle, close the gripper, drag it past the midpoint to flip
    it ON, then open and retract."""
    out = target.off_dir                       # away from the ON push direction
    grip_at = target.point
    standoff = _add(grip_at, out, _STANDOFF)
    flip_to = _add(grip_at, target.actuate_dir, _FLIP_PUSH)

    P.look_at(robot, grip_at, duration=0.6)
    P.open_gripper(arm, side=side)
    for label, xyz in (("approach", standoff), ("at-handle", grip_at)):
        sol = _pose(planner, xyz, seed)
        if sol is None:
            log.warning("   %s: %s unreachable — skipping", target.id, label)
            return seed
        pose, seed = sol
        P.smooth_move(arm, pose, duration=1.2)
    P.close_gripper(arm, side=side)            # pinch/push tool on the handle
    sol = _pose(planner, flip_to, seed)
    if sol is not None:
        pose, seed = sol
        P.smooth_move(arm, pose, duration=1.0)  # drag the handle past midpoint
    P.open_gripper(arm, side=side)
    sol = _pose(planner, standoff, seed)
    if sol is not None:
        pose, seed = sol
        P.smooth_move(arm, pose, duration=1.0)
    return seed


def run_demo(host: str, port: int, scene_path: str) -> None:
    if not gate_check():
        log.error("Safety gate failed — aborting.")
        sys.exit(1)

    backend = os.environ.get("REACHY_SIM_BACKEND", "kinematic").lower()
    log.info("Backend: %s", backend)
    if backend != "mujoco-remote":
        log.warning("Interactive controls only respond in the mujoco-remote "
                    "(physics) backend; run scripts/start_sim.sh first.")

    scene = SceneModel.from_yaml(scene_path)
    log.info("Scene '%s': %d controls", scene.frame_id, len(scene.controls()))

    robot = ReachySDK(host=host, sdk_port=port)
    time.sleep(0.8)
    arms = {"right": robot.r_arm, "left": robot.l_arm}
    if arms["right"] is None or arms["left"] is None:
        log.error("Both arms required — is the simulator running?")
        sys.exit(1)

    planners = {
        "right": CartesianPlanner(arms["right"], side="right"),
        "left": CartesianPlanner(arms["left"], side="left"),
    }
    seeds = {
        "right": [P.READY[n] for n in planners["right"].joints
                  if n in P.READY] or None,
        "left": None,
    }

    robot.turn_on("r_arm")
    robot.turn_on("l_arm")
    time.sleep(0.3)
    # Ready poses: right arm READY, left arm the mirror.
    P.smooth_move(arms["right"], P.READY, duration=1.5)
    P.smooth_move(arms["left"], P.mirror_pose(P.READY), duration=1.5)
    seeds["right"] = [P.READY[n] for n in planners["right"].joints]
    seeds["left"] = [P.mirror_pose(P.READY)[n] for n in planners["left"].joints]

    ok = 0
    for cid in [c.id for c in scene.controls()]:
        side = scene.preferred_arm(cid)
        if side == "either":
            side = "right"
        arm, planner = arms[side], planners[side]
        target = scene.control_target(cid)
        log.info("=" * 56)
        log.info("CONTROL: %s  (%s, %s arm)", cid, target.control_type, side)
        if target.control_type == "button":
            seeds[side] = operate_button(robot, arm, planner, target, side, seeds[side])
        else:
            seeds[side] = operate_hinge(robot, arm, planner, target, side, seeds[side])
        time.sleep(0.4)
        st = _read_state(cid)
        if st is not None:
            log.info("   %s is now %s", cid, "ON" if st.get("on") else "OFF")
            ok += 1 if st.get("on") else 0
        # Return the operating arm to its ready pose.
        ready = P.READY if side == "right" else P.mirror_pose(P.READY)
        P.smooth_move(arm, ready, duration=1.0)
        seeds[side] = [ready[n] for n in planner.joints]

    log.info("=" * 56)
    log.info("Controls turned ON: %d / %d", ok, len(scene.controls()))
    log.info("── Stow both arms")
    P.smooth_move(arms["right"], P.HOME, duration=1.5)
    P.smooth_move(arms["left"], P.mirror_pose(P.HOME), duration=1.5)
    robot.turn_off("r_arm")
    robot.turn_off("l_arm")
    log.info("Demo complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reachy 1.2 two-arm control-panel practice")
    ap.add_argument("--host", default=os.environ.get("REACHY_IP", "localhost"))
    ap.add_argument("--port", type=int, default=50051)
    ap.add_argument("--scene", default=os.environ.get(
        "REACHY_SIM_SCENE", "/opt/scenes/control_panel.yaml"))
    args = ap.parse_args()
    run_demo(args.host, args.port, args.scene)


if __name__ == "__main__":
    main()
