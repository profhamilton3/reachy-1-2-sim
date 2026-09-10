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

# The measured pose set.  rig_routes imports nothing from here, so this is not
# a cycle — and the pocket waypoints belong to the route, not to this module.
from reachy_ai.motion import rig_routes as R_ROUTES  # noqa: E402

_INTERP_HZ = 25          # interpolation update rate
# Gripper sign convention (measured from reachy_1_2.xml): the RIGHT gripper
# opens NEGATIVE, closes POSITIVE.  Pad gap: -45° ≈ 8.0 cm (open), -5° ≈ 4.6 cm
# (closed).  The LEFT gripper's range is mirrored (ctrlrange -0.35..1.2 vs the
# right's -1.2..0.35), so its signs are inverted: opens POSITIVE, closes NEGATIVE.
_GRIPPER_OPEN_DEG = -45.0
_GRIPPER_CLOSED_DEG = -5.0
_L_GRIPPER_OPEN_DEG = 45.0
_L_GRIPPER_CLOSED_DEG = 5.0

# Joint-name suffixes whose sign flips when a right-arm pose is mirrored to the
# left across the sagittal (y=0) plane.  Pitch joints (about the y-axis) keep
# their sign; roll/yaw joints (about x/z) flip.
_MIRROR_FLIP_SUFFIXES = ("shoulder_roll", "arm_yaw", "forearm_yaw", "wrist_roll")


def mirror_pose(pose: Dict[str, float]) -> Dict[str, float]:
    """Map a RIGHT-arm pose dict to the mirror-image LEFT-arm pose.

    Renames ``r_*`` → ``l_*``, negates the roll/yaw joints (sagittal mirror),
    and inverts the gripper sign (the left gripper's convention is reversed).
    """
    out: Dict[str, float] = {}
    for name, val in pose.items():
        if not name.startswith("r_"):
            out[name] = val
            continue
        suffix = name[2:]
        lname = "l_" + suffix
        if suffix == "gripper":
            out[lname] = -val
        elif suffix in _MIRROR_FLIP_SUFFIXES:
            out[lname] = -val
        else:
            out[lname] = val
    return out

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


def open_gripper(arm, duration: float = 0.8, side: str = "right") -> None:
    """Open the gripper smoothly (side-aware sign)."""
    jn = f"{side[0]}_gripper"
    val = _L_GRIPPER_OPEN_DEG if side == "left" else _GRIPPER_OPEN_DEG
    smooth_move(arm, {jn: val}, duration)


def close_gripper(arm, duration: float = 0.8, side: str = "right") -> None:
    """Close the gripper to a firm grip (side-aware sign)."""
    jn = f"{side[0]}_gripper"
    val = _L_GRIPPER_CLOSED_DEG if side == "left" else _GRIPPER_CLOSED_DEG
    smooth_move(arm, {jn: val}, duration)


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


def _folded(pose: Dict[str, float]) -> Dict[str, float]:
    """The same arm position, with the elbow tucked and reaching forward.

    This is the shape that gets past `rig_rail_outer_right`.  A folded forearm
    rides high and inside the rail's arc; a straight one sweeps through it.
    """
    out = dict(pose)
    out["r_elbow_pitch"] = _FOLDED_ELBOW
    out["r_shoulder_pitch"] = _FOLDED_PITCH
    out["r_arm_yaw"] = 0.0
    out["r_forearm_yaw"] = 0.0
    out["r_wrist_pitch"] = 0.0
    out["r_wrist_roll"] = 0.0
    return out


def raise_to_side(arm, duration: float = 3.0) -> None:
    """Lift the right arm out to the robot's right side to the SIDE_HIGH hub.

    OUT OF THE POCKET FIRST, THEN UP.  This function predates the rig: with no
    rails around it, abducting the shoulder straight out of HOME was fine, and
    that is what it did — roll first, in isolation, with the arm straight.
    With the rails in place that is two different collisions at once.

    At HOME the arm is INSIDE the rail pocket, and the only way out of a pocket
    is the way you came in: extension, with the roll left alone.  The measured
    route says so in its first two waypoints — GRIP_SHUT then BACK, "back out
    of the pocket, extension only, roll stays at 0" — and the robot agrees:
    probed from HOME, a forward shoulder pitch does not move at all
    (commanded -60, -40 and -25, it held at -0.9, +4.5 and -0.5), because the
    pocket is in the way.

    Only once the arm is clear of the pocket rails is there anything to raise.
    And then the SECOND collision applies: a straight arm sweeping the roll out
    runs into `rig_rail_outer_right` between about -13 and -35 degrees, up to
    3.2 cm deep.  Measured live, commanded to -88, the roll reached -14.1 and
    held there — the old docstring claimed it "holds at its ~-85 degree range
    limit", and it was holding against a rail.  So the elbow is tucked before
    the roll moves; folded, the forearm rides inside the rail's arc.

    Four steps, and each one is a different obstacle: the pocket, then the
    rail, then the abduction, then the hub.  See #73.
    """
    # 1. Shut the hand.  It travels shut through the rig for the same reason
    #    the route does it: the moving finger swings out as the gripper opens,
    #    so an open hand is a 7.5 cm tube about the wrist axis and a shut one
    #    is 5.2 cm.
    converge(arm, dict(R_ROUTES.GRIP_SHUT), duration * 0.20, tol=12.0,
             joints=_POCKET_JOINTS)

    # 2. Back out of the pocket: extension only, roll untouched.
    #
    #    This one gets a real budget.  It is 40 degrees of shoulder against
    #    gravity, from an arm that has usually just been parked with the motors
    #    off, and it is the first move of the trip — there is no momentum in it
    #    and nothing after it can help.  Given a quarter of the default
    #    duration it failed about one attempt in three, with the shoulder
    #    exactly where it started.
    if not converge(arm, dict(R_ROUTES.BACK), duration * 0.60, tol=10.0,
                    joints=_POCKET_JOINTS, passes=8):
        joint, off = worst_joint(arm, dict(R_ROUTES.BACK), _POCKET_JOINTS)
        raise RuntimeError(
            "the arm did not back out of the rail pocket: "
            f"{joint} is {off:.0f} deg off. Raising from inside the pocket "
            "would drive it into the rails.")

    # 3. Clear of the pocket, tuck the elbow — now the rail is the obstacle,
    #    not the pocket, and a folded arm is what gets past it.
    here = {n: getattr(arm, n).present_position for n in HOME}
    if not _tuck(arm, here, duration * 0.25):
        joint, off = worst_joint(arm, _folded(here), _SHOULDER_JOINTS)
        raise RuntimeError(
            "the tuck did not take, so the arm is still straight enough to "
            f"catch rig_rail_outer_right on the way out: elbow is at "
            f"{arm.r_elbow_pitch.present_position:.0f} (needs to be past "
            f"{_TUCK_FLOOR:.0f}), {joint} is {off:.0f} deg off")

    # 4. Carry the folded arm out past the rail, then open to the hub, where
    #    there is 24 cm of room.
    out = {n: getattr(arm, n).present_position for n in HOME}
    out[_ABDUCT_JOINT] = SIDE_HIGH[_ABDUCT_JOINT]
    converge(arm, out, duration * 0.30, tol=10.0, joints=PLACING_JOINTS)
    converge(arm, dict(SIDE_HIGH), duration * 0.25, tol=14.0,
             joints=PLACING_JOINTS)


#: The tuck the arm carries through the rail band, and the two facts that
#: pin it.
#:
#: PITCH MUST BE POSITIVE.  At HOME the arm is in the rail pocket, and the only
#: way out is backwards — which is why the measured route's first waypoint is
#: BACK at +40.  Probed live from HOME: commanded -60, -40 and -25, the
#: shoulder held at -0.9, +4.5 and -0.5.  It does not move forward at all.  A
#: forward tuck reads well in the model and is unreachable on the robot.
#:
#: ELBOW MUST BE DEEP.  Rig clearance over the whole roll sweep, pitch +30:
#:
#:     elbow   -90     -100     -110     -120     -125
#:            -7.0    -9.1     -5.1     -0.4     +1.8  cm
#:
#: -125 is what the model wants; the joint will not go there from HOME at this
#: pitch — asked for it, the elbow saturates around -109 — so this asks -120
#: and gets about -110.
#:
#: AND AT -110 THE MODEL SAYS THE ARM IS INSIDE THE RAIL, by about 5 cm, while
#: the arm sweeps the full range without catching.  The old straight-armed
#: sweep models BETTER (-3.2 cm) and physically stops dead at roll -14.  So for
#: this rail the capsule model and the physics disagree about which paths are
#: passable, and they disagree in both directions.  Flown behaviour is the
#: evidence here; the model is what picked the direction to try.  That
#: disagreement is worth knowing about beyond this function — the route
#: compatibility record in rig_routes rests on the same model (#74).
_FOLDED_PITCH = 30.0
_FOLDED_ELBOW = -120.0

#: Shoulder pitch the elbow is straightened at, on the way back into the
#: pocket.  Not the same as `_FOLDED_PITCH`, and it has to be further back.
#: Probed live at roll 0: asked to straighten from -120, the elbow stalls
#: around -79 at pitch +30 and will not move further however many passes it is
#: given — the forearm is pointing into the pocket's front rail.  At +40 it
#: straightens out to within a few degrees.  This is the same fact the measured
#: route encodes as BACK: leaving and entering the pocket is EXTENSION ONLY,
#: with the arm swung behind.
_UNFOLD_PITCH = 40.0


def _stream(arm, pose: Dict[str, float], duration: float) -> None:
    """One commanded pass at `pose`, using the SDK's own trajectory generator.

    `smooth_move` is a hand-rolled 25 Hz linear interpolator and it does not
    converge under physics: it reaches the target and sags off it again.
    Measured at a folded elbow (-125 deg target), re-issuing the same command:

        smooth_move   -112.6  -116.3  -110.4  -113.5  -123.4  -113.8  -119.5
        goto/JERK     -119.8  -123.2  -123.9  -124.3

    Live, that difference is a fold that stops at -55 instead of -100, which
    leaves the arm half straight in the rail band this whole module is about.

    Falls back to `smooth_move` when the SDK is not importable, so stub arms in
    the offline tests still work.
    """
    try:
        from reachy_sdk.trajectory import goto
        from reachy_sdk.trajectory.interpolation import InterpolationMode
    except ImportError:
        smooth_move(arm, pose, duration)
        return
    goto({getattr(arm, name): value for name, value in pose.items()},
         duration=duration, interpolation_mode=InterpolationMode.MINIMUM_JERK)


#: The joints that decide where the arm SITS in the rig.  The forearm yaw and
#: the wrists spin the hand about its own axis and move the elbow nowhere, so
#: holding a safety guard to them refuses moves that are perfectly safe — and
#: on a sim that has been reset they read wherever gravity left them.
PLACING_JOINTS = ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw",
                  "r_elbow_pitch")

#: The shoulder joints alone.  The tuck holds these to a tight tolerance
#: because a shoulder still swinging while the roll crosses the rail is the
#: excursion the tuck exists to prevent — measured at -5.8 cm when the pitch
#: was allowed 15 degrees of slack.
_SHOULDER_JOINTS = ("r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw")

#: What must arrive on the way into the pocket.  The placing joints, plus the
#: WRIST PITCH — that one folds the hand and is part of the shape that fits
#: through, which is why CURL carries +45.  The forearm yaw and the wrist roll
#: only spin the hand about its own axis, and guarding on them refuses a stow
#: after a wave, which deliberately leaves both wherever the last swing put
#: them.
_POCKET_JOINTS = PLACING_JOINTS + ("r_wrist_pitch",)

#: How deep the elbow must actually BE before the roll is allowed to move.
#:
#: A floor rather than a tolerance, because the joint sags 10-30 degrees short
#: of whatever it is asked for and the sag varies run to run: observed tucks
#: were -91, -100, -108, -110 and -114 for the same command.  A symmetric
#: tolerance rejects half of those, and they all fly.
#:
#: The value separates FOLDED from STRAIGHT, and is not read off a clearance
#: curve.  It cannot be: over this range the model is not even monotonic
#: (pitch +30 gives -7.0 cm at elbow -90, -9.1 at -100, -5.1 at -110, -0.4 at
#: -120) and it is the same model that rates the straight sweep — which
#: physically stops dead — as three times safer than the tucked one, which
#: physically passes.  So this is set from what was flown: every tuck past -80
#: swept the full range, the half-fold at -55 caught, and the straight arm
#: caught every time.
_TUCK_FLOOR = -80.0


def tucked_enough(arm) -> bool:
    return arm.r_elbow_pitch.present_position <= _TUCK_FLOOR


def _tuck(arm, here: Dict[str, float], duration: float) -> bool:
    """Get the arm folded, without moving the shoulder more than it has to.

    AN ARM THAT IS ALREADY FOLDED KEEPS ITS PITCH.  The side hub is a tuck —
    elbow -100 at pitch -25 — and forcing it to `_FOLDED_PITCH` means a 55
    degree shoulder swing with the elbow folded, which does not happen: the
    shoulder stalls 44 degrees short, twice in three runs.  It is also
    pointless, because -25 models BETTER through the rail band than +30
    (+1.5 cm against -0.4).  The forward pitch exists for one case only: an
    arm that is straight and therefore still in the pocket, which has to back
    out before it can bend at all.

    Shoulder first, elbow second.  Commanding both at once, the same tuck
    landed anywhere between -72 and -114 degrees across runs; separately, it
    lands where it is asked.
    """
    target = _folded(here)
    if tucked_enough(arm):
        # Already folded: leave the shoulder where it is.
        target["r_shoulder_pitch"] = arm.r_shoulder_pitch.present_position
    hold = dict(target)
    hold["r_elbow_pitch"] = arm.r_elbow_pitch.present_position
    converge(arm, hold, duration * 0.45, tol=6.0, joints=_SHOULDER_JOINTS,
             passes=6)
    converge(arm, target, duration * 0.55, tol=8.0,
             joints=("r_elbow_pitch",), passes=8)
    shoulders = _reached(arm, {n: target[n] for n in _SHOULDER_JOINTS}, 8.0)
    return shoulders and tucked_enough(arm)


def worst_joint(arm, pose: Dict[str, float], joints=None):
    """The joint furthest from its goal, and by how much.

    Exists because the first version of the messages below named the elbow and
    the shoulder and nothing else, and then reported "the arm did not reach
    BACK (elbow -2, shoulder pitch 38)" — both of which were within tolerance.
    A refusal that names the wrong joint sends the reader to the wrong place.
    """
    names = joints if joints is not None else [n for n in pose
                                               if n != "r_gripper"]
    guarded = {n: pose[n] for n in names if n in pose}
    name = max(guarded,
               key=lambda n: abs(getattr(arm, n).present_position - guarded[n]))
    return name, abs(getattr(arm, name).present_position - guarded[name])


def converge(arm, pose: Dict[str, float], duration: float = 2.0,
             tol: float = 8.0, passes: int = 5, joints=None) -> bool:
    """Command `pose`, then re-command it until the arm actually gets there.

    Under the mujoco-remote backend the arm only moves WHILE setpoints are
    streaming.  A single pass finishes short, and HOLDING the goal does not
    close the gap — verified directly: after `goal_position = 0.0` the value
    read back unchanged for six seconds while the joint sat 107 degrees away.

    So a move is a command followed by re-commands, and `wait_until` alone is
    not enough: it waits for something that is no longer happening.

    Returns whether it arrived.  Callers that are about to move through
    somewhere narrow should look at that.
    """
    names = joints if joints is not None else [n for n in pose
                                               if n != "r_gripper"]
    guarded = {n: pose[n] for n in names if n in pose}
    _stream(arm, pose, duration)
    for k in range(passes):
        if _reached(arm, guarded, tol):
            return True
        _stream(arm, pose, 0.6 * (1 + k))
    return _reached(arm, guarded, tol)


def stow_from_side(robot, arm, duration: float = 3.0) -> None:
    """Bring the arm down from the side hub and stow it at HOME, motors off.

    LOWER FOLDED, STRAIGHTEN AT THE BOTTOM.  This used to do the opposite —
    re-establish a straight "jumping jack" out to the side, then adduct the
    roll — and a straight arm sweeping the roll home passes THROUGH
    `rig_rail_outer_right`:

        roll  -40   +3.7 cm      roll  -20   -3.2 cm
        roll  -30   -1.9 cm      roll  -10   +2.6 cm

    A band from about -35 to -13 degrees, up to 3.2 cm deep.  Neither arm_yaw
    nor the gripper changes it: it is the upper arm and forearm against the
    rail, not the hand.  Observed live in FWDCenterLabSivaPool, the arm stalled
    at roll -33.4 with the model reporting -0.03 cm and refused every further
    command — it was not disobeying, it was held by contact.

    Folded, the forearm rides high and inside the rail's arc instead of
    sweeping through it, and the descent clears by +5.2 cm at `_FOLDED_PITCH`.

    WHAT THIS STILL CANNOT SEE.  `primitives` has no scene, so the clearances
    above are the RIG only, and the rig is the part that is always there.  The
    straightening at the end sweeps the forearm forward across the near-right
    grid cell: measured against FWDCenterLabMCC's initial placements, that is
    2.2 cm INSIDE `red_cube`.  An object parked there will be hit, and nothing
    in this function can know it is.  A caller that cares must guard the move —
    `reachy_ai.motion.escort` is what does that — or clear the cell first.
    """
    here = {n: getattr(arm, n).present_position for n in HOME}

    # 1. Tuck where the arm stands.  It may be at the hub, or partway back from
    #    it, or anywhere a run left it; folding first makes the next move safe
    #    from all of them, and it is the same first step as `raise_to_side`.
    #    A half-fold is the dangerous case, not a slow one: it leaves the arm
    #    straight enough to catch the rail, so this stops rather than sweeps.
    if not _tuck(arm, here, duration * 0.30):
        joint, off = worst_joint(arm, _folded(here), _SHOULDER_JOINTS)
        raise RuntimeError(
            "the tuck did not take, so bringing the roll home would run the "
            f"arm into rig_rail_outer_right: elbow is at "
            f"{arm.r_elbow_pitch.present_position:.0f} (needs to be past "
            f"{_TUCK_FLOOR:.0f}), {joint} is {off:.0f} deg off")

    # 2. Bring the roll home, still folded.  This is the move that used to go
    #    through the rail.
    lowered = {n: getattr(arm, n).present_position for n in HOME}
    lowered[_ABDUCT_JOINT] = 0.0
    converge(arm, lowered, duration * 0.40, tol=8.0,
             joints=PLACING_JOINTS)

    # 3. Enter the pocket by the MEASURED sequence, not by an invented one.
    #    This is `raise_to_side`'s step 2 run backwards: a pocket is entered
    #    the way it is left, by extension with the roll already home.
    #    Straightening the elbow at roll 0 is the step that decides whether the
    #    arm ends in the pocket or standing in front of it, and it is fussier
    #    than it looks: probed live, an open hand with the wrist flat would not
    #    straighten past about -79 degrees at pitch +30, or at all at +40.  The
    #    route's own CURL -> BACK does it in one move — shut hand, wrist folded
    #    +45, shoulder swung back — and took the elbow from -123 to -0.9.
    #
    #    The hand travels SHUT for the same reason it does on the route: the
    #    moving finger swings out as it opens, so an open hand is a 7.5 cm tube
    #    about the wrist axis and a shut one is 5.2 cm.
    from reachy_ai.motion import rig_routes as _R

    for name, target, secs, tol in (("CURL", _R.CURL, 0.30, 10.0),
                                    ("BACK", _R.BACK, 0.30, 10.0),
                                    ("GRIP_SHUT", _R.GRIP_SHUT, 0.20, 12.0),
                                    ("HOME", HOME, 0.20, 8.0)):
        if not converge(arm, dict(target), duration * secs, tol=tol, passes=8,
                        joints=_POCKET_JOINTS):
            joint, off = worst_joint(arm, dict(target), _POCKET_JOINTS)
            raise RuntimeError(
                f"the arm did not reach {name} on the way into the pocket: "
                f"{joint} is {off:.0f} deg off, tolerance {tol:.0f}. It is "
                "being left where it is rather than driven further in.")

    robot.turn_off("r_arm")
    log.info("Right arm stowed at HOME, motors off.")


def go_home(robot, arm, duration: float = 2.5) -> None:
    """Return the right arm to the home (zero) pose then turn off motors.

    Routes through the side-clearing stow path so the hand never crosses the
    tabletop on the way down."""
    stow_from_side(robot, arm, duration)
