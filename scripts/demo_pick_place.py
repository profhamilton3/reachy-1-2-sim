#!/usr/bin/env python3
"""
Demo: pick each tabletop object with the right arm and place it on the
opposite side of the table.

Default scene objects:
  red_cube      world [0.48, -0.12, table]  → place to [0.48, +0.18]
  blue_cylinder world [0.60, +0.12, table]  → place to [0.60, -0.18]

Run against the Docker simulator (must be running):
  python3 scripts/demo_pick_place.py

To observe: open the VNC viewer at http://localhost:6080 and watch the
right arm move through each pick-and-place trajectory.

Joint angles are approximate FK estimates — tune the poses in
src/reachy_ai/motion/primitives.py if the arm misses an object.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Make src/ importable when run from the repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

try:
    from reachy_sdk import ReachySDK
except ImportError:
    print("ERROR: reachy-sdk not installed.  Run: pip install reachy-sdk==0.7.0")
    sys.exit(1)

from reachy_ai.motion import primitives as P
from reachy_ai.motion.safety import gate_check

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_SETTLE = 0.4   # pause after each primitive (seconds) for visual clarity


def _step(label: str) -> None:
    log.info("── %s", label)
    time.sleep(_SETTLE)


def pick_and_place(
    robot: ReachySDK,
    pick_over: dict,
    pick_at: dict,
    carry: dict,
    place_over: dict,
    place_at: dict,
    label: str,
    move_s: float = 2.0,
    lower_s: float = 1.2,
) -> None:
    """Full pick-and-place cycle for one object using the right arm."""
    arm = robot.r_arm

    _step(f"{label}: move above pick site (gripper open)")
    P.smooth_move(arm, pick_over, move_s)

    _step(f"{label}: descend to grasp height")
    P.smooth_move(arm, pick_at, lower_s)

    _step(f"{label}: close gripper")
    P.close_gripper(arm)

    _step(f"{label}: lift")
    P.smooth_move(arm, carry, lower_s)

    _step(f"{label}: carry to place site")
    P.smooth_move(arm, place_over, move_s)

    _step(f"{label}: descend to place height")
    P.smooth_move(arm, place_at, lower_s)

    _step(f"{label}: open gripper (release)")
    P.open_gripper(arm)

    _step(f"{label}: retract above place site")
    P.smooth_move(arm, place_over, lower_s)


def run_demo(host: str, port: int) -> None:
    if not gate_check():
        log.error("Safety gate check failed — aborting.")
        sys.exit(1)

    log.info("Connecting to %s:%d …", host, port)
    robot = ReachySDK(host=host, sdk_port=port)
    time.sleep(0.8)

    arm = robot.r_arm
    if arm is None:
        log.error("r_arm not found on SDK object.  Is the simulator running?")
        sys.exit(1)

    robot.turn_on("r_arm")
    time.sleep(0.3)
    log.info("Right arm motors ON")

    # ── Move to READY pose ────────────────────────────────────────────────────
    _step("Moving to READY pose")
    P.smooth_move(arm, P.READY, duration=2.0)

    # ── Object 1: red cube ────────────────────────────────────────────────────
    log.info("=" * 55)
    log.info("OBJECT 1 — red cube  [0.48, -0.12] → [0.48, +0.18]")
    log.info("=" * 55)

    pick_and_place(
        robot,
        pick_over=P.OVER_RED,
        pick_at=P.GRASP_RED,
        carry=P.CARRY_RED,
        place_over=P.OVER_PLACE_RED,
        place_at=P.PLACE_RED,
        label="red_cube",
    )

    # Return to READY between objects
    _step("Return to READY")
    P.smooth_move(arm, P.READY, duration=2.0)
    time.sleep(0.5)

    # ── Object 2: blue cylinder ───────────────────────────────────────────────
    log.info("=" * 55)
    log.info("OBJECT 2 — blue cylinder  [0.60, +0.12] → [0.60, -0.18]")
    log.info("=" * 55)

    pick_and_place(
        robot,
        pick_over=P.OVER_BLUE,
        pick_at=P.GRASP_BLUE,
        carry=P.CARRY_BLUE,
        place_over=P.OVER_PLACE_BLUE,
        place_at=P.PLACE_BLUE,
        label="blue_cylinder",
    )

    # ── Return home ───────────────────────────────────────────────────────────
    log.info("=" * 55)
    _step("Return to HOME and turn off")
    P.go_home(robot, arm, duration=3.0)

    log.info("Demo complete.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reachy 1.2 pick-and-place demo")
    ap.add_argument("--host", default=os.environ.get("REACHY_IP", "localhost"))
    ap.add_argument("--port", type=int, default=50051)
    args = ap.parse_args()
    run_demo(args.host, args.port)


if __name__ == "__main__":
    main()
