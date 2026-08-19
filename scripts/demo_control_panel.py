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
import math
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
from reachy_ai.motion.kinematics import (
    CartesianPlanner,
    L_ARM_JOINTS,
    R_ARM_JOINTS,
    UnreachableError,
)
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


def _pad(arm, planner):
    joints = R_ARM_JOINTS if planner.side == "right" else L_ARM_JOINTS
    return planner.fk_world([getattr(arm, n).present_position for n in joints])


def _dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def reach(arm, planner, target, iters: int = 2, dur: float = 1.4, compensate: bool = True):
    """Move the gripper pad to a world point, compensating for the physics
    tracking sag.

    The multi-joint command path under-tracks by a few cm; we re-assert the goal
    to settle, measure the residual, then aim *beyond* the target by that
    residual so the sagged arm lands on it.  Returns the final |error| (m).
    """
    aim = tuple(target)
    seed = None
    moved = False
    err = None
    for _ in range(max(1, iters)):
        q = None
        for cand in (aim, tuple(target)):        # fall back to the raw target
            try:
                q = planner.solve(cand, seed=seed)
                break
            except (UnreachableError, Exception):
                continue
        if q is None:
            return err if moved else None        # truly unreachable
        seed = q
        moved = True
        pose = dict(zip(planner.joints, q))
        P.smooth_move(arm, pose, duration=dur)
        P.wait_until(arm, pose, tol=4.0, timeout=2.2, reassert=True)
        resid = [target[i] - _pad(arm, planner)[i] for i in range(3)]
        err = math.sqrt(sum(r * r for r in resid))
        if not compensate or err < 0.02:
            break
        aim = tuple(target[i] + resid[i] for i in range(3))
    return err


def _is_on(control_id):
    st = _read_state(control_id)
    return bool(st and st.get("on"))


def operate_button(robot, arm, planner, target, side, seed):
    """Poke the button with a closed gripper: reach above (sag-compensated),
    descend through the cap to toggle it, retract.  Retries a deeper press if
    the first sweep didn't register (borderline reach)."""
    above = _add(target.point, target.off_dir, _STANDOFF)
    P.look_at(robot, target.point, duration=0.5)
    P.close_gripper(arm, duration=0.4, side=side)     # solid poker
    if reach(arm, planner, above) is None:
        log.warning("   %s: standoff unreachable — skipping", target.id)
        return seed
    for depth in (_PRESS_DEPTH, _PRESS_DEPTH + 0.02, _PRESS_DEPTH + 0.04):
        reach(arm, planner, _add(target.point, target.actuate_dir, depth))
        if _is_on(target.id):
            break
        reach(arm, planner, above, iters=1, compensate=False)   # lift & retry
    reach(arm, planner, above, iters=1, compensate=False)
    return seed


def operate_hinge(robot, arm, planner, target, side, seed):
    """Reach the handle with a closed gripper and sweep it past the midpoint to
    flip it ON.  Retries with a longer sweep if it didn't catch."""
    above = _add(target.point, target.off_dir, _STANDOFF)
    P.look_at(robot, target.point, duration=0.5)
    P.close_gripper(arm, duration=0.4, side=side)     # push tool
    if reach(arm, planner, above) is None:
        log.warning("   %s: approach unreachable — skipping", target.id)
        return seed
    for push in (_FLIP_PUSH, _FLIP_PUSH + 0.03, _FLIP_PUSH + 0.06):
        reach(arm, planner, target.point)             # to the handle
        reach(arm, planner, _add(target.point, target.actuate_dir, push))
        if _is_on(target.id):
            break
        reach(arm, planner, above, iters=1, compensate=False)
    reach(arm, planner, above, iters=1, compensate=False)
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
