"""
Motion primitives for Reachy 1.2 right-arm manipulation.

All functions that move the physical robot live here.  No raw joint angles
may be sent from an LLM response — callers must use these named primitives.

Joint angles are in DEGREES (reachy_sdk convention).
Reachy 1.2 right-arm joint sign conventions (axis = +Y for pitch joints):
  r_shoulder_pitch :  0° = arm down,  -90° = arm horizontal forward
  r_shoulder_roll  :  0° = no roll,   positive = toward +Y (left table side)
  r_arm_yaw        :  positive = counterclockwise when viewed from above
  r_elbow_pitch    :  0° = straight,  negative = bent (range 0 to -125°)
  r_forearm_yaw    :  ±100°
  r_wrist_pitch    :  positive = wrist up relative to forearm (range ±45°)
  r_wrist_roll     :  ±45°
  r_gripper        :  +20° = open,    negative = closed (down to -69°)
"""

from __future__ import annotations

import logging
import time
from typing import Dict

log = logging.getLogger(__name__)

_INTERP_HZ = 25          # interpolation update rate
_GRIPPER_OPEN_DEG = 18.0
_GRIPPER_CLOSED_DEG = -30.0  # finger gap ~30 mm — suitable for 60 mm cube / 70 mm cylinder

# ── Named joint poses (degrees) ───────────────────────────────────────────────
# All poses are for the right arm only.  Unused joints keep their current value.

HOME: Dict[str, float] = {
    "r_shoulder_pitch": 0.0,
    "r_shoulder_roll":  0.0,
    "r_arm_yaw":        0.0,
    "r_elbow_pitch":    0.0,
    "r_forearm_yaw":    0.0,
    "r_wrist_pitch":    0.0,
    "r_wrist_roll":     0.0,
    "r_gripper":        _GRIPPER_OPEN_DEG,
}

# Transitional ready pose: arm angled forward, elbow bent, hand relaxed open.
READY: Dict[str, float] = {
    "r_shoulder_pitch": -25.0,
    "r_shoulder_roll":   0.0,
    "r_arm_yaw":         0.0,
    "r_elbow_pitch":    -55.0,
    "r_forearm_yaw":     0.0,
    "r_wrist_pitch":    15.0,
    "r_wrist_roll":      0.0,
    "r_gripper":        _GRIPPER_OPEN_DEG,
}

# ── Red cube poses  (object at world [0.48, -0.12, table_surface]) ────────────
# Shoulder at world (0, -0.19, 1.0).  FK estimate puts gripper tip at:
#   shoulder_pitch=-55°, elbow_pitch=-40°, wrist_pitch=+28° →
#   gripper ~(0.54, -0.17, 0.84)
# shoulder_roll=+5° nudges arm +0.05 m in Y to align with y=-0.12.
OVER_RED: Dict[str, float] = {
    "r_shoulder_pitch": -55.0,
    "r_shoulder_roll":   5.0,
    "r_arm_yaw":         0.0,
    "r_elbow_pitch":    -40.0,
    "r_forearm_yaw":     0.0,
    "r_wrist_pitch":    28.0,
    "r_wrist_roll":      0.0,
    "r_gripper":        _GRIPPER_OPEN_DEG,
}

# Lower 3° shoulder + 5° more elbow bend to descend ~40 mm to grasp height.
GRASP_RED: Dict[str, float] = {**OVER_RED,
    "r_shoulder_pitch": -58.0,
    "r_elbow_pitch":    -45.0,
    "r_wrist_pitch":    22.0,
    "r_gripper":        _GRIPPER_OPEN_DEG,
}

CARRY_RED: Dict[str, float] = {**OVER_RED,
    "r_gripper": _GRIPPER_CLOSED_DEG,
}

# Place: swing arm to opposite (positive-Y) side of table.
# arm_yaw=-25° rotates upper arm to carry gripper to y ≈ +0.15.
OVER_PLACE_RED: Dict[str, float] = {
    "r_shoulder_pitch": -55.0,
    "r_shoulder_roll":  -5.0,
    "r_arm_yaw":       -25.0,
    "r_elbow_pitch":   -40.0,
    "r_forearm_yaw":    0.0,
    "r_wrist_pitch":   28.0,
    "r_wrist_roll":     0.0,
    "r_gripper":       _GRIPPER_CLOSED_DEG,
}

PLACE_RED: Dict[str, float] = {**OVER_PLACE_RED,
    "r_shoulder_pitch": -58.0,
    "r_elbow_pitch":    -45.0,
    "r_wrist_pitch":    22.0,
}

# ── Blue cylinder poses  (object at world [0.60, +0.12, table_surface]) ───────
# x=0.60 is ~25 mm further forward → shoulder_pitch 3° more negative.
# y=+0.12 is on the far (positive-Y) side: shoulder_roll=-8°, arm_yaw=-22°.
OVER_BLUE: Dict[str, float] = {
    "r_shoulder_pitch": -58.0,
    "r_shoulder_roll":  -8.0,
    "r_arm_yaw":       -22.0,
    "r_elbow_pitch":   -40.0,
    "r_forearm_yaw":    0.0,
    "r_wrist_pitch":   28.0,
    "r_wrist_roll":     0.0,
    "r_gripper":       _GRIPPER_OPEN_DEG,
}

GRASP_BLUE: Dict[str, float] = {**OVER_BLUE,
    "r_shoulder_pitch": -61.0,
    "r_elbow_pitch":    -45.0,
    "r_wrist_pitch":    22.0,
    "r_gripper":       _GRIPPER_OPEN_DEG,
}

CARRY_BLUE: Dict[str, float] = {**OVER_BLUE,
    "r_gripper": _GRIPPER_CLOSED_DEG,
}

# Place: swing to right (negative-Y) side — arm_yaw=+20°, roll=+10°.
OVER_PLACE_BLUE: Dict[str, float] = {
    "r_shoulder_pitch": -55.0,
    "r_shoulder_roll":  10.0,
    "r_arm_yaw":        20.0,
    "r_elbow_pitch":   -40.0,
    "r_forearm_yaw":    0.0,
    "r_wrist_pitch":   28.0,
    "r_wrist_roll":     0.0,
    "r_gripper":       _GRIPPER_CLOSED_DEG,
}

PLACE_BLUE: Dict[str, float] = {**OVER_PLACE_BLUE,
    "r_shoulder_pitch": -58.0,
    "r_elbow_pitch":    -45.0,
    "r_wrist_pitch":    22.0,
}


# ── Core motion helpers ───────────────────────────────────────────────────────

def smooth_move(arm, pose: Dict[str, float], duration: float = 2.0) -> None:
    """Linearly interpolate all joints in pose over duration seconds.

    Args:
        arm:      reachy.r_arm  (must already be turned on).
        pose:     dict of joint_name → target_degrees.
        duration: motion time in seconds.
    """
    steps = max(1, int(duration * _INTERP_HZ))
    start: Dict[str, float] = {
        name: getattr(arm, name).present_position for name in pose
    }
    dt = duration / steps
    for i in range(1, steps + 1):
        t = i / steps
        for name, goal in pose.items():
            getattr(arm, name).goal_position = start[name] + t * (goal - start[name])
        time.sleep(dt)
    log.debug("smooth_move done: %s", {k: f"{v:.1f}°" for k, v in pose.items()})


def execute_trajectory(
    arm,
    joint_traj,
    joint_names,
    rate_hz: int = 25,
    on_step=None,
) -> None:
    """Stream a pre-planned joint trajectory to the arm at rate_hz.

    Args:
        arm:         reachy.r_arm (turned on).
        joint_traj:  list of joint-angle lists (degrees), one per step.
        joint_names: names matching each joint-angle list entry.
        rate_hz:     command rate.
        on_step:     optional callback(step_index, joint_values) invoked after
                     each command is sent — used to attach a carried object's
                     marker to the gripper in the kinematic/RViz view.
    """
    dt = 1.0 / rate_hz
    for i, q in enumerate(joint_traj):
        for name, val in zip(joint_names, q):
            getattr(arm, name).goal_position = float(val)
        if on_step is not None:
            on_step(i, q)
        time.sleep(dt)


def open_gripper(arm, duration: float = 0.8) -> None:
    """Open the right gripper smoothly."""
    smooth_move(arm, {"r_gripper": _GRIPPER_OPEN_DEG}, duration)


def close_gripper(arm, duration: float = 0.8) -> None:
    """Close the right gripper to a firm grip."""
    smooth_move(arm, {"r_gripper": _GRIPPER_CLOSED_DEG}, duration)


def go_home(robot, arm, duration: float = 2.5) -> None:
    """Return the right arm to the home (zero) pose then turn off motors."""
    smooth_move(arm, READY, duration * 0.4)
    smooth_move(arm, HOME, duration * 0.6)
    robot.turn_off("r_arm")
    log.info("Right arm at HOME, motors off.")
