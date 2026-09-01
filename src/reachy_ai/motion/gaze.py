"""
Where to point Reachy 1.2's head so a given world point is in the stereo view.

The motion work raised the arm's working pose high and out to the robot's right
to keep the elbow off the table (see kinematics.link_capsules).  That fixed the
table and broke the view: the hand now works outside the head's default gaze, so
the stereo cameras watch an empty rig while the interesting motion happens off
frame.  This module aims the neck at whatever you want to watch.

Pure geometry — no SDK, no simulator — so it is unit-testable on the host.  The
caller applies the returned angles to reachy.head.neck_pitch / neck_yaw.

TWO THINGS ABOUT THE NECK THAT ARE NOT GUESSABLE, both measured on the sim:

  * ``turn_on("head")`` is not enough to make the Orbita neck hold a goal.  Set
    ``neck_roll/pitch/yaw.compliant = False`` explicitly first or the joint
    goals are quietly ignored.  ``head.look_at()`` exists but goes through a
    weak fake head-kinematics service; commanding neck joints directly is more
    reliable.
  * POSITIVE neck_pitch looks DOWN.  Established by aiming at red_cube and
    watching where its pixels land: invisible at pitch <= 0, and climbing the
    frame as pitch rises (row 92.5% -> 75.5% -> 52.2% at +15/+30/+45).
"""

from __future__ import annotations

import math
from typing import Tuple

XYZ = Tuple[float, float, float]

# Neck pivot in world coordinates: the torso origin (0, 0, 1.0) plus the MJCF's
# torso->head_x offset of (0.015, 0, 0.095).
NECK_PIVOT_IN_WORLD: XYZ = (0.015, 0.0, 1.095)

# The MJCF gives head_x euler="0 0.174 0" — 10 degrees of downward pitch baked
# in ahead of the joints, added because the head's neutral gaze sat 10 deg too
# high.  Joint zero is therefore already looking 10 deg down, and a commanded
# angle has to have that subtracted out.
BUILT_IN_PITCH_DEG = 10.0

# Measured correction on top of the pivot geometry.
#
# Aiming at red_cube and fitting where it lands in the left image against the
# correction applied:
#
#     extra   +0.0   +5.0   +10.0   +15.0   +20.0     (degrees)
#     row%    65.9   59.5    52.6    41.6    36.8     (0% = top of frame)
#     fit: row% = -1.52 * extra + 66.5  ->  centred at +10.8
#
# About 5 deg of that is explained: the cameras sit 0.061 m ABOVE the neck pivot
# in the head frame, and aiming an eye that is 6 cm higher at a point 0.33 m
# below needs a steeper angle than aiming the pivot does.  The remaining ~6 deg
# is not explained here.  It is a fitted constant, so treat it as valid for this
# camera mount and re-measure if the mount or the calibration profile changes —
# the fit above is cheap to repeat.
PITCH_BIAS_DEG = 10.8

# Neck joint travel, from the MJCF ranges (radians) less a degree of margin:
#   neck_pitch  range="-0.8 1.13"   = -45.8 .. +64.7 deg
#   neck_yaw    range="-2.79 2.79"  = +-159.9 deg
# Nothing this module is used for needs more than about 40 deg of either — the
# table is in front and the arm's raised pose is to the right — so the clamp is
# a guard against a bad target, not a working constraint.
PITCH_LIMITS = (-45.0, 64.0)
YAW_LIMITS = (-159.0, 159.0)


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def neck_angles_for(target: XYZ, clamp: bool = True) -> Tuple[float, float]:
    """Neck (pitch, yaw) in degrees that put ``target`` in the stereo view.

    Positive pitch looks down; positive yaw turns to the robot's left.  With
    ``clamp`` the angles are limited to the neck's travel, so a target behind
    the robot gives the closest achievable gaze rather than an impossible goal
    the joint will silently fail to reach.
    """
    dx = target[0] - NECK_PIVOT_IN_WORLD[0]
    dy = target[1] - NECK_PIVOT_IN_WORLD[1]
    dz = target[2] - NECK_PIVOT_IN_WORLD[2]
    horizontal = math.hypot(dx, dy)
    pitch = (math.degrees(math.atan2(-dz, horizontal))
             - BUILT_IN_PITCH_DEG + PITCH_BIAS_DEG)
    yaw = math.degrees(math.atan2(dy, dx))
    if clamp:
        pitch = _clamp(pitch, *PITCH_LIMITS)
        yaw = _clamp(yaw, *YAW_LIMITS)
    return pitch, yaw


def can_look_at(target: XYZ) -> bool:
    """True if the neck can actually aim at ``target`` without clamping."""
    pitch, yaw = neck_angles_for(target, clamp=False)
    return (PITCH_LIMITS[0] <= pitch <= PITCH_LIMITS[1]
            and YAW_LIMITS[0] <= yaw <= YAW_LIMITS[1])
