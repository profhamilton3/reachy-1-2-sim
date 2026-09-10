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
# tabletop.  The arm used to be raised like a JUMPING JACK instead — pitch held
# near 0 while the roll drove it laterally out past the table's right edge —
# and ABDUCT_LOW/SIDE_HIGH below are what is left of that.
#
# THAT WAS WRONG ONCE THE RIG EXISTED, and `raise_to_side` no longer does it.
# The rails put a pocket around the hanging arm and a rail beside it, so a
# lateral sweep at fixed pitch runs the hand through the board's near edge and
# the forearm through `rig_rail_outer_right`.  The notebook says so in as many
# words, as its FAILURE exercise: "Try the sideways version and watch it fail
# ... the arm jams against the outer rail."
#
# SIDE_HIGH is now the notebook's PRESENT — pitch -70, roll -25, elbow -80 —
# which is what "raised, out to the robot's right" actually means in this rig
# and is the only such pose anyone has measured (+8.9 cm worst whole-arm
# clearance across the whole of section 4, against -1.9 cm for the pose it
# replaced).  It is spelled out here rather than imported from rig_routes
# because this module is imported BY that one; `test_primitives` asserts the
# two agree.
#
#   ABDUCT_LOW : arm swung straight OUT to the right, roughly horizontal, elbow
#                extended.  Kept only for `recipe_executor`, which uses it as a
#                mid-point in its own path; nothing in the rig routes goes near
#                it, and nothing new should.
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
    "r_shoulder_pitch": -70.0,
    "r_shoulder_roll":  -25.0,
    "r_arm_yaw":         0.0,
    "r_elbow_pitch":    -80.0,
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


def fly(arm, route, *, label: str = "route", move=None) -> List[str]:
    """Walk a measured route waypoint by waypoint, and stop where it stops.

    This is the notebook's `move_to` loop, and every part of it is load-bearing:

    * ONE WAYPOINT AT A TIME.  The clearance of each waypoint was measured FROM
      the one before it, so a route is not a set of poses to reach eventually,
      it is an order.
    * THE WHOLE POSE, EVERY TIME.  Each waypoint is a complete eight-joint
      pose.  Building a target out of whatever the arm happens to be reading —
      which is what `raise_to_side` and `stow_from_side` used to do — flies a
      configuration nobody verified, and inherits, say, the wrist angles the
      wave left behind.
    * RE-STREAM UNTIL IT ARRIVES.  Under mujoco-remote the arm only moves while
      setpoints stream; `converge` does that.
    * STOP RATHER THAN CONTINUE.  A waypoint reached ten degrees short is no
      longer the pose that was verified, and the next segment's clearance no
      longer applies to it.

    Raises RuntimeError naming the waypoint.  Callers that need abort and phase
    callbacks want `tasks.rig_motion.fly_route`, which is the same walk with
    those hooks and a typed error.
    """
    from reachy_ai.motion import rig_routes as R

    flown: List[str] = []
    for wp in route:
        guard = list(wp.guard) if wp.guard else list(R.CRITICAL_JOINTS)
        loose = [n for n in wp.pose if n not in guard and n != "r_gripper"]
        if not converge(arm, dict(wp.pose), wp.seconds, tol=wp.tol,
                        joints=guard, passes=6):
            joint, off = worst_joint(arm, dict(wp.pose), guard)
            raise RuntimeError(
                f"{label} stopped at {wp.name}: {joint} is {off:.0f} deg off, "
                f"tolerance {wp.tol:.0f}. The next waypoint's clearance was "
                "measured from this one, so it does not apply from here.")
        # The joints that only spin the hand still have to be moving.  They
        # are not held to the waypoint's tolerance — see CRITICAL_JOINTS — but
        # one of them tens of degrees out is not a weak joint lagging, it is
        # nothing tracking at all, and that is worth stopping for.
        if loose:
            joint, off = worst_joint(arm, dict(wp.pose), loose)
            if off > R.LESSON_TOL:
                raise RuntimeError(
                    f"{label} stopped at {wp.name}: {joint} is {off:.0f} deg "
                    f"off its goal, past even the loose tolerance "
                    f"{R.LESSON_TOL:.0f}. Nothing is tracking.")
        flown.append(wp.name)
    return flown


def _at(arm, pose: Dict[str, float], tol: float = 8.0) -> bool:
    """Is the arm standing at `pose`, judged on the joints that place it?"""
    return _reached(arm, {n: pose[n] for n in PLACING_JOINTS}, tol)


def raise_to_side(arm, duration: float = 3.0) -> None:
    """Out of the rail pocket and up to the raised pose, by the measured route.

    THIS FLIES THE NOTEBOOK, AND IT DID NOT USED TO.  It was four hand-built
    steps: shut the hand, back out of the pocket (both of those taken from the
    route), then fold the arm from wherever it was reading and sweep the roll
    from 0 to -88 in a single command at fixed pitch.  Only the first two are
    in `tlh_motion-routine.ipynb`.  The rest was invented, and modelled against
    the scene it is 4.3 to 6.5 cm inside `table_top` for every degree of that
    sweep, clearing only in the last ten — the HAND against the board's near
    edge, with the fold holding it at the height of the slab.  An operator
    watched exactly that: the gripper going through the board.

    WHY THE INVENTED VERSION LOOKED FINE FOR SO LONG.  MuJoCo's contact pushes
    back in proportion to penetration; a position servo pushes in proportion to
    joint error.  The shoulder has kp=300 and 60 Nm to spend, so a pose a
    couple of degrees past a rail loses to the rail — which is where "commanded
    to -88, the roll reached -14.1 and held there" came from — while a pose six
    centimetres inside the board wins from the first timestep and goes straight
    through.  The physics was never off.  It was being asked for the second
    kind of pose.

    WHAT THE MEASURED ROUTE DOES INSTEAD.  `PLACE_ROUTE` crosses the rail band
    in TWO joints at once — SWING_1/2/3 step the pitch +37.5 -> +20 -> -17.5
    while the roll goes -32.5 -> -35 -> -37.5 — so it goes around the rail
    rather than through it.  There is no "-13 to -35 degree rail band" on that
    path; the band is an artefact of sweeping one joint at fixed pitch, which
    is the thing this no longer does.  Its deepest roll anywhere is -37.5.

    Then one move to PRESENT, which is the notebook's own next step (cell 18)
    and the raised pose this function is named for.

    `duration` is accepted and IGNORED.  The route carries its own measured
    per-waypoint timings and they are part of what was verified — at half the
    time the weak joints finish tens of degrees short, which is how a bounded
    route stops being one.  The parameter stays so callers keep working.
    """
    from reachy_ai.motion import rig_routes as R

    if not _at(arm, R.HOME):
        joint, off = worst_joint(arm, dict(R.HOME), PLACING_JOINTS)
        raise RuntimeError(
            "the route out of the rail pocket starts at HOME and the arm is "
            f"not there: {joint} is {off:.0f} deg off. Getting onto the route "
            "is a separate move, and guessing one is what this exists to stop.")
    fly(arm, R.PLACE_ROUTE, label="the route out of the pocket")
    fly(arm, R.LIFT_TO_PRESENT, label="the lift to the raised pose")


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
    """Back into the rail pocket from the raised pose, by the measured route.

    THE NOTEBOOK'S STOW *IS* THE RETURN FROM PRESENT.  `STOW_ROUTE` begins at
    REST_SHUT, and cell 34 flies it straight after section 4 leaves the arm at
    PRESENT — so the first move it makes is PRESENT -> REST_SHUT, and the whole
    descent and pocket entry follow from there.  Nothing else is needed, and
    this is the exact reverse of `raise_to_side`, which is what the notebook
    requires of a stow: "Retrace the placement route backwards.  Do not
    shortcut it: going straight from PRESENT to HOME drives the upper arm
    through the rig's front rail."

    WHAT THIS REPLACES.  Seven of the eleven moves it used to make were not in
    the notebook.  It tucked the arm from whatever it was reading, swept the
    roll from -88 home in one command at fixed pitch, and only then picked up
    the route's last four waypoints — entering CURL from a pose that is not the
    one the route reaches CURL from.  Modelled, that join passes the hand
    6.4 cm inside `table_top`; the notebook's own join (CURL_HIGH -> CURL, both
    behind the pocket at pitch +70 and +40) never goes below -1.4 cm, and that
    only at the endpoint.  The descent before it modelled clear — down to
    +0.7 cm, which is not much — so this one failed at the join rather than on
    the sweep, and the sweep is gone with it either way.

    It also inherited the wave's wrist and forearm angles into the pocket,
    because a target built from live readings carries whatever is in them.  A
    route waypoint is a whole pose, so the hand is placed rather than merely
    left alone — which is what an operator noticed was missing.

    `duration` is accepted and IGNORED, for the reason given on
    `raise_to_side`: the route's timings are part of what was measured.

    The motors go off at the end.  That is right for a stow — it is the one
    route that parks — and wrong in the middle of a longer trip, so callers
    that continue afterwards turn them back on.
    """
    from reachy_ai.motion import rig_routes as R

    if not _at(arm, R.PRESENT, tol=12.0):
        joint, off = worst_joint(arm, dict(R.PRESENT), PLACING_JOINTS)
        raise RuntimeError(
            "the stow starts at the raised pose and the arm is not there: "
            f"{joint} is {off:.0f} deg off. A direct move to HOME from an "
            "arbitrary pose drives the upper arm through the board's near "
            "edge, so this will not guess one.")
    fly(arm, R.STOW_ROUTE, label="the route into the pocket")

    robot.turn_off("r_arm")
    log.info("Right arm stowed at HOME, motors off.")


def go_home(robot, arm, duration: float = 2.5) -> None:
    """Return the right arm to the home (zero) pose then turn off motors.

    Routes through the side-clearing stow path so the hand never crosses the
    tabletop on the way down."""
    stow_from_side(robot, arm, duration)
