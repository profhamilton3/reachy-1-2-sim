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
# Gripper sign convention (measured from reachy_1_2.xml): NEGATIVE opens,
# POSITIVE closes.  Pad gap: -45° ≈ 8.0 cm (open, clears the 60 mm cube and
# 70 mm cylinder), -5° ≈ 4.6 cm (closed, squeezes both for a friction grip).
_GRIPPER_OPEN_DEG = -45.0
_GRIPPER_CLOSED_DEG = -5.0

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
# NOTE: this pose puts the gripper pad forward and BELOW the tabletop surface in
# the kinematic model — do not use it as a transit hub over the table.  Kept for
# backward-compatibility; the pick-and-place demo uses the SIDE poses below.
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

# ── Table-clearing transit poses ("jumping-jack" abduction, then over) ────────
# The demo tabletop spans x∈[0,0.90], y∈[−0.33,+0.33] with its surface at z=0.74,
# while the right shoulder sits at world (0,−0.19,1.0) — the whole forward
# workspace is above the table with only ~0.26 m of headroom.  Raising the arm by
# pitching it FORWARD (the intuitive move) sweeps the gripper straight into the
# tabletop.  Instead we raise it like a JUMPING JACK: shoulder *pitch stays ~0*
# and shoulder *roll* drives the arm laterally OUT, away from the torso, past the
# table's right edge (−Y), where there is no table to hit.  Only once the arm is
# out and up does it reach forward over the table (as a collision-checked
# Cartesian move).  Both poses were FK-verified so the gripper pad clears the
# table slab throughout every joint-space interpolation HOME → ABDUCT_LOW →
# SIDE_HIGH.
#
#   ABDUCT_LOW : arm swung straight OUT to the right, roughly horizontal, elbow
#                extended — pitch 0, so it moves away from the side of the torso
#                without going forward (pad ≈ (0.00,−0.82,0.83), well outside the
#                table's −0.33 right edge).
#   SIDE_HIGH  : from ABDUCT_LOW, lift the upper arm and FLEX THE ELBOW TIGHT —
#                a "bicep-curl" that folds the forearm UP so the whole forearm
#                (not just the gripper tip) rides high and clear of the table.
#                This is the raised transit hub (pad ≈ (0.33,−0.41,1.15), wrist ≈
#                (0.23,−0.43,1.10) — both ~0.4 m above the surface).  Verified so
#                BOTH the pad and the wrist clear the slab through every
#                interpolation; the arm only reaches forward over the table,
#                unflexing as needed, once fully raised.
ABDUCT_LOW: Dict[str, float] = {
    "r_shoulder_pitch":  0.0,
    "r_shoulder_roll":  -75.0,
    "r_arm_yaw":         0.0,
    "r_elbow_pitch":     0.0,
    "r_forearm_yaw":     0.0,
    "r_wrist_pitch":     0.0,
    "r_wrist_roll":      0.0,
    "r_gripper":        _GRIPPER_OPEN_DEG,
}

SIDE_HIGH: Dict[str, float] = {
    "r_shoulder_pitch": -25.0,
    "r_shoulder_roll":  -88.0,
    "r_arm_yaw":         0.0,
    "r_elbow_pitch":   -100.0,   # tight bicep flex → forearm folds up, rides high
    "r_forearm_yaw":     0.0,
    "r_wrist_pitch":     0.0,
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


import math

# Approx world position of the head/cameras (torso top) for look-at geometry.
_HEAD_WORLD = (0.02, 0.0, 1.18)
# Neck look-at limits (deg).  Pitch capped at 45° for a natural down-gaze at the
# tabletop (the joint allows up to 64°, which looks like staring at its chest).
_NECK_PITCH_RANGE = (-45.0, 45.0)
_NECK_YAW_RANGE = (-159.0, 159.0)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def look_at(robot, xyz, duration: float = 1.0) -> None:
    """Orient the head so the cameras point toward world point xyz.

    neck_pitch>0 looks down; neck_yaw>0 turns toward +Y (robot's left).  The
    workspace objects are down and to the right (−Y), so this yaws right and
    pitches down to face them.
    """
    head = getattr(robot, "head", None)
    if head is None:
        return
    hx, hy, hz = _HEAD_WORLD
    dx, dy, dz = xyz[0] - hx, xyz[1] - hy, xyz[2] - hz
    horiz = math.hypot(dx, dy)
    yaw = _clamp(math.degrees(math.atan2(dy, dx)), *_NECK_YAW_RANGE)
    pitch = _clamp(math.degrees(math.atan2(-dz, horiz)), *_NECK_PITCH_RANGE)
    robot.turn_on("head")
    # turn_on("head") does not reliably make the neck stiff on the sim backend;
    # stiffen the neck joints explicitly or goal_position is ignored (compliant).
    for jn in ("neck_roll", "neck_pitch", "neck_yaw"):
        try:
            getattr(head, jn).compliant = False
        except Exception:
            pass
    time.sleep(0.1)
    smooth_move(head, {"neck_roll": 0.0, "neck_pitch": pitch, "neck_yaw": yaw}, duration)


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


def _reached(arm, targets: Dict[str, float], tol: float) -> bool:
    return all(
        abs(getattr(arm, n).present_position - v) <= tol for n, v in targets.items()
    )


def wait_until(
    arm,
    targets: Dict[str, float],
    tol: float = 8.0,
    timeout: float = 3.5,
    reassert: bool = False,
) -> bool:
    """Closed-loop wait: poll present_position until every joint in ``targets``
    is within ``tol`` degrees of its goal.

    Returns True if converged, False on timeout.  Physics tracking lags the
    commanded goal, so callers that need the arm to actually *be* somewhere
    (e.g. fully abducted and clear of the table before reaching forward) gate on
    this rather than assume the pose was reached.

    ``reassert`` is OFF by default: re-sending goals every poll makes the
    mujoco-remote bridge rebuild the full target vector from lagging positions,
    which drags the abducted shoulder back down.  Set the goals once, then wait.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        if reassert:
            for n, v in targets.items():
                getattr(arm, n).goal_position = v
        if _reached(arm, targets, tol):
            return True
        time.sleep(0.1)
    return False


# Joint whose isolated abduction lifts the arm out to the side.
_ABDUCT_JOINT = "r_shoulder_roll"


def raise_to_side(arm, duration: float = 3.0) -> None:
    """Lift the right arm out to the robot's right side to the SIDE_HIGH hub.

    Sequenced to work under physics, where commanding the abduction together
    with the elbow/pitch *stalls* the shoulder-roll (a coupling quirk of the
    mujoco-remote bridge — the joint reaches full abduction only when driven
    alone from a compact arm).  So we:

      1. Abduct the shoulder-roll FIRST, isolated, and wait (closed-loop) for it
         to actually swing the arm out past the table's right edge.
      2. THEN lift/flex the rest of the arm to the hub; the roll holds out.

    The result clears the table in both the physics and kinematic models.
    """
    roll_goal = SIDE_HIGH[_ABDUCT_JOINT]
    # 1. Isolated abduction — ramp the shoulder-roll out alone and confirm the
    #    arm actually swings out to the side (it holds at its ~-85° range limit).
    smooth_move(arm, {_ABDUCT_JOINT: roll_goal}, duration * 0.5)
    wait_until(arm, {_ABDUCT_JOINT: roll_goal}, tol=8.0, timeout=3.0)
    # 2. Set the rest of the hub DIRECTLY (one shot, no ramp).  Ramping all
    #    joints together collapses the abducted roll; a single direct goal set
    #    holds it (verified: roll stays ~-83° at ~5 Nm).  Then wait to converge.
    for n, v in SIDE_HIGH.items():
        getattr(arm, n).goal_position = v
    wait_until(arm, SIDE_HIGH, tol=14.0, timeout=3.0)


def stow_from_side(robot, arm, duration: float = 3.0) -> None:
    """Reverse of raise_to_side: swing the arm back into the jumping-jack
    position, then lower it down by Reachy's side, motors off.

    The return trajectory can leave the arm partway (flexed/forward), so we first
    re-establish a clean jumping-jack pose — shoulder fully abducted OUT to the
    side, elbow/pitch straightened — and only THEN adduct the shoulder-roll to
    swing the whole arm down at the side (the downward half of a jumping jack).
    Roll is driven in isolation so the coupling that stalls it doesn't bite.
    """
    roll_out = SIDE_HIGH[_ABDUCT_JOINT]
    # 1. Move BACK into the jumping-jack direction: make sure the shoulder is
    #    fully abducted out to the side before anything else.
    smooth_move(arm, {_ABDUCT_JOINT: roll_out}, duration * 0.30)
    wait_until(arm, {_ABDUCT_JOINT: roll_out}, tol=8.0, timeout=2.5)
    # 2. Un-flex to a straight arm held OUT to the side (direct set so the ramp
    #    coupling doesn't collapse the abduction).
    jj = dict(HOME)
    jj[_ABDUCT_JOINT] = roll_out
    for n, v in jj.items():
        getattr(arm, n).goal_position = v
    wait_until(arm, {"r_elbow_pitch": 0.0, "r_shoulder_pitch": 0.0}, tol=15.0, timeout=2.5)
    # 3. Lower: adduct the shoulder-roll ALONE so the arm swings down by the side.
    smooth_move(arm, {_ABDUCT_JOINT: 0.0}, duration * 0.55)
    wait_until(arm, {_ABDUCT_JOINT: 0.0}, tol=10.0, timeout=3.0)
    # 4. Settle everything at HOME.
    for n, v in HOME.items():
        getattr(arm, n).goal_position = v
    wait_until(arm, HOME, tol=12.0, timeout=2.0)
    robot.turn_off("r_arm")
    log.info("Right arm stowed at HOME, motors off.")


def go_home(robot, arm, duration: float = 2.5) -> None:
    """Return the right arm to the home (zero) pose then turn off motors.

    Routes through the side-clearing stow path so the hand never crosses the
    tabletop on the way down."""
    stow_from_side(robot, arm, duration)
