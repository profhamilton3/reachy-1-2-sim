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

from ..scene.awareness import ARM_BASE_IN_WORLD

XYZ = Tuple[float, float, float]

# The MJCF's torso->head_x offset (the neck_roll/pitch/yaw pivot).
TORSO_TO_NECK: XYZ = (0.015, 0.0, 0.095)

# Neck pivot in world coordinates: the torso origin plus TORSO_TO_NECK.
# Derived, never restated: it was hard-coded (0.015, 0, 1.095) for a torso at
# (0, 0, 1.0), and a mount move would have left it behind (#192).
NECK_PIVOT_IN_WORLD: XYZ = tuple(round(b + o, 6) for b, o in zip(ARM_BASE_IN_WORLD,
                                                                TORSO_TO_NECK))

# The MJCF gives head_x euler="0 0.174 0" — 10 degrees of downward pitch baked
# in ahead of the joints, added because the head's neutral gaze sat 10 deg too
# high.  Joint zero is therefore already looking 10 deg down, and a commanded
# angle has to have that subtracted out.
BUILT_IN_PITCH_DEG = 10.0

# Measured correction on top of the pivot geometry.
#
# RE-FITTED FOR THE OCTOBER LAYOUT (#192, 2026-10-17).  The August fit, with
# the torso at (0, 0, 1.0), aimed at red_cube and fitted where it lands in the
# left image against the correction applied:
#
#     extra   +0.0   +5.0   +10.0   +15.0   +20.0     (degrees)
#     row%    65.9   59.5    52.6    41.6    36.8     (0% = top of frame)
#     fit: row% = -1.52 * extra + 66.5  ->  centred at +10.8
#
# Repeated at the October mount (offline render, measured fovy, segmentation
# mask of red_cube standing on each cell), it no longer holds as one number:
#
#     cell     slope   centred at        cell     slope   centred at
#     r3c2     -1.54     +5.7            r3c3     -1.48     +8.0
#     r3c1     -1.48     +7.9            r2c1     -1.53    +11.3
#
# The slope is unchanged; the offset is not, because most of it is parallax.
# The cameras sit 0.061 m ABOVE the neck pivot in the head frame, and how much
# steeper an eye must aim than the pivot depends on how far below and away the
# target is -- in the October layout the board is 0.41 m down and starts
# almost under the head, so the spread is wide.  The nearer cells need more
# pitch than the neck's travel and cannot be centred at all (r1c3 at about
# +72).  +8.5 is the value whose worst error over the four cells above is
# smallest: +-2.8 deg, about +-4% of the frame height, with the target always in
# frame.  For a whole-board view use `board_view_pose`, which places each eye
# explicitly and needs no fitted constant.
PITCH_BIAS_DEG = 8.5

# Neck joint travel, from the MJCF ranges (radians) less a degree of margin:
#   neck_pitch  range="-0.8 1.13"   = -45.8 .. +64.7 deg
#   neck_yaw    range="-2.79 2.79"  = +-159.9 deg
# In the August layout nothing here needed more than about 40 deg of either,
# and the clamp was only a guard against a bad target.  In the October layout
# (#192) the pitch limit IS a working constraint: the board starts almost
# under the head, the near cells need more than +64 to centre, and the board
# view (below) sits on this clamp.
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


# ── The board view (#192) ────────────────────────────────────────────────────
#
# Siva sets the head by hand "to a set position to see all squares".  This is
# that position, computed from the geometry so the simulator -- and later the
# robot -- can reach it programmatically, by direct neck joint goals (not
# head.look_at(), which goes through the weak fake head kinematics).
#
# The camera model is a pinhole per eye, in the head frame the MJCF gives:
# the cameras sit at (0.0033, +-0.03625, 0.061) from the neck pivot and look
# along head +X, image up = head +Z; the head carries BUILT_IN_PITCH_DEG ahead
# of the joint.  Focal lengths are the measured profile
# (scenes/calibration_measured_2026_08_27.yaml), principal point at the image
# centre, 640 x 480.  No lens distortion: the simulator's --distortion barrel
# shows MORE of the scene, never less, so a pinhole fit is the conservative one.
# PITCH_BIAS_DEG is not used here: it is a fitted correction for aiming one
# point, and this model places every camera explicitly.

#: (name, y offset of the eye in the head frame, fx, fy) in pixels.
CAMERAS: Tuple[Tuple[str, float, float, float], ...] = (
    ("left", 0.03625, 408.4, 407.0),
    ("right", -0.03625, 400.9, 398.8),
)
CAMERA_IN_HEAD: XYZ = (0.0033, 0.0, 0.061)   # y replaced per eye
IMAGE_SIZE = (640, 480)


def project(point: XYZ, pitch_deg: float, yaw_deg: float = 0.0,
            camera: Tuple[str, float, float, float] = CAMERAS[0]
            ) -> Tuple[float, float]:
    """Pixel (u, v) of a world point in one eye at neck (pitch, yaw), roll 0.

    (nan, nan) when the point is behind the camera.
    """
    _name, cam_y, fx, fy = camera
    b = math.radians(BUILT_IN_PITCH_DEG + pitch_deg)
    yw = math.radians(yaw_deg)
    # Head rotation: yaw about world Z, then the total pitch about the head's
    # Y.  Columns are the head's X (look), Y and Z (image up) in world terms.
    cy_, sy_, cb, sb = math.cos(yw), math.sin(yw), math.cos(b), math.sin(b)
    hx = (cy_ * cb, sy_ * cb, -sb)
    hy = (-sy_, cy_, 0.0)
    hz = (cy_ * sb, sy_ * sb, cb)
    off = (CAMERA_IN_HEAD[0], cam_y, CAMERA_IN_HEAD[2])
    cam = tuple(NECK_PIVOT_IN_WORLD[i] + off[0] * hx[i] + off[1] * hy[i]
                + off[2] * hz[i] for i in range(3))
    d = tuple(point[i] - cam[i] for i in range(3))
    fwd = sum(d[i] * hx[i] for i in range(3))
    if fwd <= 1e-9:
        return float("nan"), float("nan")
    right = -sum(d[i] * hy[i] for i in range(3))
    up = sum(d[i] * hz[i] for i in range(3))
    w, h = IMAGE_SIZE
    return w / 2 + fx * right / fwd, h / 2 - fy * up / fwd


def frame_margin(points, pitch_deg: float, yaw_deg: float = 0.0) -> float:
    """Worst distance, in pixels, of any point inside either frame's edge.

    Negative when some point is outside a frame (or behind a camera).
    """
    w, h = IMAGE_SIZE
    worst = float("inf")
    for cam in CAMERAS:
        for p in points:
            u, v = project(p, pitch_deg, yaw_deg, cam)
            if math.isnan(u):
                return float("-inf")
            worst = min(worst, u, w - u, v, h - v)
    return worst


def board_view_pose(points, step_deg: float = 0.1) -> dict:
    """Neck joint goals (degrees) that put every point in both frames, with the
    largest worst-case margin, inside the neck's travel.

    Yaw aims at the points' centroid; pitch is searched over PITCH_LIMITS.
    ``margin_px`` says how much room the worst point has; negative means one
    view cannot hold them all (then view near and far rows separately).
    """
    pts = list(points)
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    yaw = math.degrees(math.atan2(cy - NECK_PIVOT_IN_WORLD[1],
                                  cx - NECK_PIVOT_IN_WORLD[0]))
    best = None
    n = int(round((PITCH_LIMITS[1] - PITCH_LIMITS[0]) / step_deg))
    for i in range(n + 1):
        pitch = PITCH_LIMITS[0] + i * step_deg
        m = frame_margin(pts, pitch, yaw)
        if best is None or m > best[1]:
            best = (pitch, m)
    return {"neck_roll": 0.0, "neck_pitch": round(best[0], 1),
            "neck_yaw": round(yaw, 1), "margin_px": round(best[1], 1)}


#: The October board's nine cells, outer corners, on the surface: the measured
#: grid of scenes/FWDCenterLabMCC.yaml (cells 5 in, pitch 6 in, centred on
#: x = 0.4318, y = 0; surface 0.7405).  tests/unit/test_robot_mount.py checks
#: these against the scene file and BOARD_VIEW against `board_view_pose`.
BOARD_CELL_CORNERS: Tuple[XYZ, ...] = tuple(
    (x, y, 0.7405) for x in (0.2159, 0.6477) for y in (0.2159, -0.2159))

#: THE BOARD VIEW: neck joint goals that show all nine squares in both sim
#: cameras, October layout.  The neck's pitch travel is what binds -- the best
#: view is at the clamp (PITCH_LIMITS, 1 deg inside the +64.7 stop), with the
#: worst cell corner still well inside both frames.  Command the three neck
#: joints directly (see the module docstring for compliance).
BOARD_VIEW = {"neck_roll": 0.0, "neck_pitch": 64.0, "neck_yaw": 0.0}
