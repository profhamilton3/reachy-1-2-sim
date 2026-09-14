"""
Scene-aware Cartesian planning for the Reachy 1.2 right arm.

Wraps reachy_sdk's inverse_kinematics with:
  * an orientation sweep (the SDK IK requires a full 6-DOF pose and fails when a
    fixed orientation is unreachable — sweeping downward-ish orientations makes
    far-side targets solvable);
  * warm-starting from the previous solution for continuity;
  * straight-line Cartesian interpolation between waypoints;
  * collision validation against a SceneModel so a path that would drive the
    gripper through the table is rejected *before* any joint command is sent.

Requires numpy and reachy_sdk (runs inside the simulator container).
The scene collision model (scene.awareness.SceneModel) is dependency-light and
unit-tested separately on the host.
"""

from __future__ import annotations

import functools
import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..scene.awareness import ARM_BASE_IN_WORLD, SceneModel

log = logging.getLogger(__name__)

XYZ = Tuple[float, float, float]

# Right-arm joint order returned by IK / consumed by FK.
R_ARM_JOINTS = [
    "r_shoulder_pitch", "r_shoulder_roll", "r_arm_yaw",
    "r_elbow_pitch", "r_forearm_yaw", "r_wrist_pitch", "r_wrist_roll",
]

# Left-arm joint order (mirror of the right).
L_ARM_JOINTS = [
    "l_shoulder_pitch", "l_shoulder_roll", "l_arm_yaw",
    "l_elbow_pitch", "l_forearm_yaw", "l_wrist_pitch", "l_wrist_roll",
]

# Baseline "point down and forward" gripper orientation (from FK of a natural
# reaching pose); the sweep rotates this about world Z (yaw) and Y (pitch).
_R0 = np.array([
    [0.39, -0.07, -0.92],
    [-0.02, 1.00, -0.09],
    [0.92, 0.05, 0.39],
])

# Reflection across the y=0 plane.  Mirroring a proper rotation R across the
# sagittal plane gives S @ R @ S (still a proper rotation), so the left arm's
# baseline orientation is the mirror image of the right's.
_MIRROR = np.diag([1.0, -1.0, 1.0])
_R0_LEFT = _MIRROR @ _R0 @ _MIRROR

_BASE = np.array(ARM_BASE_IN_WORLD)


def _joints_for(side: str):
    if side == "left":
        return L_ARM_JOINTS
    if side == "right":
        return R_ARM_JOINTS
    raise ValueError(f"side must be 'left' or 'right', got '{side!r}'")


def _r0_for(side: str) -> np.ndarray:
    if side == "left":
        return _R0_LEFT
    if side == "right":
        return _R0
    raise ValueError(f"side must be 'left' or 'right', got '{side!r}'")

# The reachy_sdk FK/IK frame is the wrist (r_wrist2hand).  The gripper contact
# pads sit ~0.12 m along the wrist's local -Z (measured from the model).  We
# plan in *pad* (contact) space: pad_world = wrist_world + R_wrist @ (0,0,-TOOL),
# so wrist_world = pad_world + R_wrist @ (0,0,+TOOL).  A true top-down grasp is
# unreachable for this arm, so we accept whatever tilt IK finds and correct for
# the pad offset at that orientation — the pads still land on the object.
_TOOL_LEN = 0.12
_TOOL = np.array([0.0, 0.0, _TOOL_LEN])


# ── Whole-arm link geometry ───────────────────────────────────────────────────
#
# The IK/FK surface above knows one point on the robot: the gripper pad.  That
# is not the robot.  The upper arm and forearm sweep a far larger volume, and
# they are what actually reach an object sitting on the near half of the board.
# Worked example, from the pose the motion notebook used to sweep from: the pad
# was 21 cm above the table and the *elbow* was 3.8 cm above it, directly over
# the near-right grid cell, with the forearm 5 mm inside the cube standing
# there.  Every check that watched the pad called that pose safe.
#
# Link lengths are the URDF/MJCF frame offsets, and the radii are the arm's
# `class="collision"` geoms, both read from native_mujoco/model/reachy_1_2.xml:
#
#     r_upper_arm_col   capsule fromto "0 0 0  0 0 -0.28"   size 0.035
#     r_forearm_col     capsule fromto "0 0 0  0 0 -0.25"   size 0.030
#     r_thumb_col       box  pos "0  0.005 -0.085"  size 0.014 0.008 0.022
#     r_finger_col      box  pos "0  0     -0.055"  size 0.014 0.008 0.014
#
# The two gripper pads are wrapped in one capsule rather than tracked
# separately: they are 4 cm apart at most, they move with the same wrist, and a
# capsule that bounds both is both simpler and conservative.  Its length
# reaches the lowest point any hand box can occupy: the finger shell's far
# corner when the finger hangs straight down (r_gripper ~ +7.5 deg), at
# 0.0325 + 0.03998 + hypot(0.076, 0.010) = 0.1492 m below the wrist frame --
# further than the thumb pad's far face (0.1395), which the first version of
# this length was sized to and which left that corner 4 mm past the end cap.
#
# ITS RADIUS DEPENDS ON THE GRIPPER ANGLE AND THE WRIST ROLL, and getting
# either wrong is not a rounding error.  The moving finger hangs off the thumb
# at y = -0.037 and swings about the thumb's local X, so opening the gripper
# throws it outward -- and it is the finger shell's FAR END (7.6 cm below the
# hinge) that swings furthest, not its centre.  The first version of this
# radius swung the centre (3.8 cm) and was 4.0 cm short with the hand open;
# the MJCF's own collision pad sat 1.8 cm outside the tube at the -45 deg the
# guarded legs fly (docs/adr/0003, "A discrepancy this ADR does not paper
# over").  Wrist roll adds a second term: the tube is drawn from the WRIST
# frame, but the hand pivots at r_gripper_thumb, 3.25 cm further down, so
# rolling the wrist moves the whole hand 0.0325 * sin(roll) off the tube's
# axis -- 1.6 cm at the 30 deg REST carries.
#
# So the radius is not a formula any more.  It is the exact farthest distance
# from the tube axis of any corner of any hand box the MJCF places under
# r_wrist2hand -- the two visual shells AND the two collision pads -- at the
# given aperture and roll (`_hand_corner_offsets`), which is the tightest
# capsule around that geometry and is verified corner-for-corner against the
# compiled model in test_arm_geometry_mjcf.py::TestTubeCoversMJCFHand:
#
#     r_gripper      roll 0      roll 30     roll 45 (limit)
#     +20 (shut)     5.2 cm      6.7 cm      7.4 cm
#       0 (neutral)  5.2 cm      6.7 cm      7.4 cm
#     -45 (open)     9.9 cm     11.5 cm     12.1 cm
#     -68.8 (wide)  11.2 cm     12.8 cm     13.5 cm     from the axis
#
# Before this (the aperture-only, centre-swung radius) the same table read
# 5.2 / 7.5 / 8.3 cm regardless of roll.
#
# A fixed 5.0 cm radius, taken from the closed hand, was wrong by 3.2 cm exactly
# when it mattered -- the notebook holds the gripper OPEN while hovering over
# the board.  Measured consequence: the guard reported 6.98 cm of clearance to
# foam_block along a hover approach, at every path sampling resolution from 13
# to 801 steps, and the physics threw the block 0.81 m.
#
# The radius defaults to the worst case over the whole supported range of
# whichever input is unknown.  A guard that has to guess should guess wide.
_SHOULDER_Y = {"right": -0.19, "left": 0.19}
_UPPER_ARM_LEN = 0.28
_FOREARM_LEN = 0.25
_HAND_LEN = 0.150
_UPPER_ARM_RADIUS = 0.035
_FOREARM_RADIUS = 0.030

# Gripper joint ranges, from the MJCF.
_GRIPPER_OPEN_LIMIT_DEG = -68.8               # MJCF range lower bound, -1.2 rad
_GRIPPER_SHUT_LIMIT_DEG = 20.05               # MJCF range upper bound, +0.35 rad
_WRIST_ROLL_LIMIT_DEG = 45.0                  # MJCF range, +/-0.785 rad

# Every box the MJCF hangs under r_wrist2hand, (geom, centre, half-extents) in
# its own body frame -- the visual shells AND the collision pads, because the
# pads are what physics contacts with and r_finger_col is 2 mm wider in X than
# the shell around it.  Thumb-frame boxes ride the wrist roll only; finger-frame
# boxes hang at _FINGER_HINGE_OFFSET below the thumb and rotate about the
# thumb's X by the gripper angle.  `hand_radius` bounds all of them.
_THUMB_FRAME_BOXES = (
    ("r_thumb_body", (0.0, -0.018, -0.022), (0.025, 0.028, 0.038)),
    ("r_thumb_col", (0.0, 0.005, -0.085), (0.014, 0.008, 0.022)),
)
_FINGER_FRAME_BOXES = (
    ("r_finger_body", (0.0, 0.0, -0.038), (0.012, 0.010, 0.038)),
    ("r_finger_col", (0.0, 0.0, -0.055), (0.014, 0.008, 0.014)),
)

# ── Per-shell hand geometry ("shells" hand mode, review 2026-09-12 / #56/#74) ─
#
# The tube above is one isotropic capsule around BOTH pads, sized from the
# aperture alone.  It is what every consumer flies against today and stays
# the default.  "shells" is a second, opt-in `link_capsules` mode: one
# capsule per MJCF visual shell (upper_arm and forearm are unchanged; the
# hand becomes three capsules instead of one), built from the same frames
# MuJoCo itself places these bodies at (native_mujoco/model/reachy_1_2.xml),
# not measured or eyeballed independently — `test_arm_geometry_mjcf.py`
# checks the two agree to 2 mm.
#
# r_wrist2hand -> r_gripper_thumb -> r_gripper_finger, exactly as the MJCF
# nests them: the thumb hangs a fixed offset below the wrist, WRIST_ROLL
# rotates the thumb (and everything under it) about the wrist's own local X,
# and r_gripper then rotates the finger about the THUMB's local X -- so the
# finger frame depends on both joints, not just the gripper angle alone.
_THUMB_ORIGIN_OFFSET = np.array([0.0, 0.0, -0.0325])
_FINGER_HINGE_OFFSET = np.array([0.0, -0.037, -0.03998])

# Visual shell boxes (geom pos/size, MJCF): capsule axis along each body's
# local Z through the box centre; radius is the box's XY half-diagonal (the
# cross-section perpendicular to that axis).
_THUMB_BOX_CENTER = np.array([0.0, -0.018, -0.022])
_THUMB_BOX_HALF_Z = 0.038
_THUMB_SHELL_RADIUS = math.hypot(0.025, 0.028)
_FINGER_BOX_CENTER = np.array([0.0, 0.0, -0.038])
_FINGER_BOX_HALF_Z = 0.038
_FINGER_SHELL_RADIUS = math.hypot(0.012, 0.010)

_WRIST_BALL_RADIUS = 0.028

Capsule = Tuple[str, XYZ, XYZ, float]


def _rotx(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def link_frames(joints: Sequence[float], side: str = "right"):
    """World positions of the shoulder, elbow and wrist, plus the hand rotation.

    Reproduces the arm's kinematic chain rather than calling the SDK, so this
    works on the host with no simulator running — and so a whole path can be
    checked without a round trip per pose.  It is the same chain the FK service
    itself evaluates (fake_reachy_server._arm_fk), which
    test_arm_clearance.py asserts by comparing the wrist frame joint-for-joint.
    """
    q = np.radians(np.asarray(list(joints)[:7], dtype=float))
    sign = 1.0 if side == "right" else -1.0
    shoulder = _BASE + np.array([0.0, _SHOULDER_Y[side], 0.0])
    R = _roty(q[0]) @ _rotx(sign * q[1]) @ _rotz(q[2])
    elbow = shoulder + R @ np.array([0.0, 0.0, -_UPPER_ARM_LEN])
    R = R @ _roty(q[3]) @ _rotz(q[4])
    wrist = elbow + R @ np.array([0.0, 0.0, -_FOREARM_LEN])
    R = R @ _roty(q[5]) @ _rotx(q[6])
    return shoulder, elbow, wrist, R


def _hand_corner_offsets(gripper_deg: float, wrist_roll_deg: float):
    """(x, y) of every corner of every hand box, in the rolled wrist frame.

    That frame is the one the tube capsule is drawn in (``link_frames``'s
    final ``R``, whose Z is the tube axis through the wrist origin), so
    ``hypot(x, y)`` of each corner is its distance from the tube axis.
    Mirrors the MJCF chain term for term: r_gripper_thumb sits at
    ``_THUMB_ORIGIN_OFFSET`` in the *pitch-only* frame, so in the rolled frame
    it is ``rotx(-roll) @ (0, 0, -0.0325)`` -- the ``-0.0325 * sin(roll)`` in
    Y below is the wrist-roll term; r_gripper_finger then hangs at
    ``_FINGER_HINGE_OFFSET`` and rotates about X by the gripper angle.
    """
    roll = math.radians(wrist_roll_deg)
    g = math.radians(gripper_deg)
    thumb_y = _THUMB_ORIGIN_OFFSET[2] * math.sin(roll)   # -0.0325 * sin(roll)
    corners = []
    for _name, c, h in _THUMB_FRAME_BOXES:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                corners.append((c[0] + sx * h[0], thumb_y + c[1] + sy * h[1]))
    finger_y = thumb_y + _FINGER_HINGE_OFFSET[1]
    cg, sg = math.cos(g), math.sin(g)
    for _name, c, h in _FINGER_FRAME_BOXES:
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                for sz in (-1.0, 1.0):
                    yl = c[1] + sy * h[1]
                    zl = c[2] + sz * h[2]
                    # rotx(g) @ (x, yl, zl) -> y' = yl*cos(g) - zl*sin(g)
                    corners.append((c[0] + sx * h[0], finger_y + yl * cg - zl * sg))
    return corners


def _hand_radius_at(gripper_deg: float, wrist_roll_deg: float) -> float:
    """Tightest tube radius that contains every hand box corner and the wrist
    ball at one exact aperture and roll."""
    worst = max(math.hypot(x, y) for x, y in
                _hand_corner_offsets(gripper_deg, wrist_roll_deg))
    return max(worst, _WRIST_BALL_RADIUS)


# Worst-case sampling grids for an unknown aperture / roll.  Every corner's
# axis distance is a sinusoid in each angle whose turning points lie outside
# the joint ranges (tan g = 0.076/0.010 -> 82 deg; sin(roll) is monotone on
# +/-45), so the range endpoints are the extremes -- but the grids include
# interior points anyway so the claim does not rest on that argument.
_GRIPPER_RANGE_SAMPLES = tuple(
    [_GRIPPER_OPEN_LIMIT_DEG] + list(range(-68, 21, 1)) + [_GRIPPER_SHUT_LIMIT_DEG])
_ROLL_RANGE_SAMPLES = tuple(
    [-_WRIST_ROLL_LIMIT_DEG] + list(range(-45, 46, 5)) + [_WRIST_ROLL_LIMIT_DEG])


@functools.lru_cache(maxsize=8192)
def hand_radius(gripper_deg: Optional[float] = None,
                wrist_roll_deg: Optional[float] = None) -> float:
    """Radius of the capsule that bounds the whole gripper -- both visual
    shells and both collision pads -- at a given aperture and wrist roll.

    Either argument may be ``None``, meaning "unknown": the result is then
    the worst case over that joint's entire supported range (fully open for
    the gripper; +/-45 deg for the roll), because a guard asked to check a
    hand whose pose it does not know must assume the widest one.  See the
    constants above for what a wrong answer here costs, and
    test_arm_geometry_mjcf.py::TestTubeCoversMJCFHand for the corner-by-corner
    verification against the compiled model.
    """
    grippers = (_GRIPPER_RANGE_SAMPLES if gripper_deg is None
                else (float(gripper_deg),))
    rolls = (_ROLL_RANGE_SAMPLES if wrist_roll_deg is None
             else (float(wrist_roll_deg),))
    return max(_hand_radius_at(g, r) for g in grippers for r in rolls)


# Joint travel, in degrees, read from the MJCF `range` attributes.
#
# THE IK DOES NOT ENFORCE THESE and the SDK exposes no `limits` on a joint, so
# nothing was checking them.  Measured cost: solve() happily returned
# r_arm_yaw=+119.7 against a +90 stop and r_forearm_yaw=-136.0 against -100.
# The arm clamps, lands 12-36 deg from the pose it was given, and the pad misses
# its target by 20 cm — while every clearance check passes, because the guard is
# asked about the pose that was COMMANDED and the arm never went there.
#
# Left/right differ only in the sign of the roll and yaw axes; mirrored below.
JOINT_LIMITS_DEG: Dict[str, Tuple[float, float]] = {
    "r_shoulder_pitch": (-150.0, 90.0),
    "r_shoulder_roll": (-180.0, 10.0),
    "r_arm_yaw": (-90.0, 90.0),
    "r_elbow_pitch": (-125.0, 0.0),
    "r_forearm_yaw": (-100.0, 100.0),
    "r_wrist_pitch": (-45.0, 45.0),
    "r_wrist_roll": (-45.0, 45.0),
}


def joint_limits(side: str = "right") -> List[Tuple[float, float]]:
    """(lo, hi) per joint in R_ARM_JOINTS order, for the given arm."""
    lims = [JOINT_LIMITS_DEG[j] for j in _joints_for("right")]
    if side == "right":
        return lims
    out = []
    for j, (lo, hi) in zip(_joints_for("right"), lims):
        # roll and yaw mirror; pitch axes are shared.
        out.append((-hi, -lo) if ("roll" in j or "yaw" in j) else (lo, hi))
    return out


def within_limits(joints: Sequence[float], side: str = "right",
                  tol: float = 0.5) -> bool:
    """True if every joint is inside its travel, allowing ``tol`` degrees.

    ``tol`` exists because a solution sitting exactly on a stop is reachable in
    principle and unreachable in practice; half a degree keeps the check from
    rejecting poses the arm can actually hold.
    """
    return all(lo - tol <= v <= hi + tol
               for v, (lo, hi) in zip(list(joints)[:7], joint_limits(side)))


def link_capsules(joints: Sequence[float], side: str = "right",
                  gripper_deg: Optional[float] = None,
                  hand: str = "tube") -> List[Capsule]:
    """The arm's collision volume as (name, end, end, radius) world capsules.

    Feed straight to ``SceneModel.clearance``.  The torso and head do not move
    here, and the shoulder ball is inside the upper-arm capsule already.

    ``gripper_deg`` sizes the hand to the actual aperture; omitting it assumes
    the hand is wide open, which is the safe assumption and costs about 6 cm
    of reported clearance against a closed hand.  The pose's own wrist roll
    sizes it too (see ``hand_radius``).

    ``hand`` selects the hand model. ``"tube"`` (default, unchanged, and every
    consumer's current behaviour) is one isotropic capsule around both pads.
    ``"shells"`` is five capsules instead of three: ``upper_arm`` and
    ``forearm`` unchanged, plus ``thumb``, ``finger`` and ``wrist_ball`` in
    place of ``hand`` -- one capsule per MJCF visual shell, at the frames
    MuJoCo itself places them at.  See the module-level comment above
    ``_THUMB_ORIGIN_OFFSET``.  No consumer passes this yet; it exists to be
    validated against the compiled MJCF (`test_arm_geometry_mjcf.py`) ahead of
    #56/#74's decision to use it.
    """
    shoulder, elbow, wrist, R = link_frames(joints, side)

    def xyz(v) -> XYZ:
        return (float(v[0]), float(v[1]), float(v[2]))

    arm = [
        ("upper_arm", xyz(shoulder), xyz(elbow), _UPPER_ARM_RADIUS),
        ("forearm", xyz(elbow), xyz(wrist), _FOREARM_RADIUS),
    ]

    if hand == "tube":
        tip = wrist + R @ np.array([0.0, 0.0, -_HAND_LEN])
        # The pose carries its own wrist roll, so the tube is sized for it
        # exactly rather than for the worst roll in range.  Only the right
        # hand's geometry is verified against the MJCF; the left falls back
        # to the worst case over roll.
        roll = float(list(joints)[6]) if side == "right" else None
        return arm + [("hand", xyz(wrist), xyz(tip),
                       hand_radius(gripper_deg, roll))]

    if hand != "shells":
        raise ValueError(f"hand must be 'tube' or 'shells', got {hand!r}")

    gripper_deg_eff = _GRIPPER_OPEN_LIMIT_DEG if gripper_deg is None else gripper_deg
    return arm + _shells_hand_capsules(wrist, gripper_deg_eff, joints, side)


def _shells_hand_capsules(wrist: np.ndarray, gripper_deg: float,
                          joints: Sequence[float], side: str) -> List[Capsule]:
    """thumb / finger / wrist_ball capsules for `link_capsules(hand="shells")`.

    Rebuilds the chain up to wrist_pitch (the frame the thumb offset is
    actually expressed in, before wrist_roll rotates the thumb -- and
    everything under it -- about the wrist's own local X) directly from
    `joints`, term for term the same way `link_frames` composes it.
    """
    q = np.radians(np.asarray(list(joints)[:7], dtype=float))
    sign = 1.0 if side == "right" else -1.0
    # Rebuild the pitch-only orientation the thumb offset is expressed in
    # (everything up to and including wrist_pitch, i.e. `R_final` with the
    # wrist_roll term removed) directly from the joint angles, matching
    # `link_frames`'s own composition order term for term.
    R = _roty(q[0]) @ _rotx(sign * q[1]) @ _rotz(q[2])
    R = R @ _roty(q[3]) @ _rotz(q[4])
    R_pitch_only = R @ _roty(q[5])
    R_thumb = R_pitch_only @ _rotx(q[6])

    def xyz(v) -> XYZ:
        return (float(v[0]), float(v[1]), float(v[2]))

    thumb_origin = wrist + R_pitch_only @ _THUMB_ORIGIN_OFFSET
    thumb_a = thumb_origin + R_thumb @ (
        _THUMB_BOX_CENTER + np.array([0.0, 0.0, -_THUMB_BOX_HALF_Z]))
    thumb_b = thumb_origin + R_thumb @ (
        _THUMB_BOX_CENTER + np.array([0.0, 0.0, _THUMB_BOX_HALF_Z]))

    R_finger = R_thumb @ _rotx(math.radians(gripper_deg))
    finger_origin = thumb_origin + R_thumb @ _FINGER_HINGE_OFFSET
    finger_a = finger_origin + R_finger @ (
        _FINGER_BOX_CENTER + np.array([0.0, 0.0, -_FINGER_BOX_HALF_Z]))
    finger_b = finger_origin + R_finger @ (
        _FINGER_BOX_CENTER + np.array([0.0, 0.0, _FINGER_BOX_HALF_Z]))

    return [
        ("thumb", xyz(thumb_a), xyz(thumb_b), _THUMB_SHELL_RADIUS),
        ("finger", xyz(finger_a), xyz(finger_b), _FINGER_SHELL_RADIUS),
        ("wrist_ball", xyz(wrist), xyz(wrist), _WRIST_BALL_RADIUS),
    ]


def joint_path(q_from: Sequence[float], q_to: Sequence[float], steps: int = 13):
    """Poses along a joint-space straight line, endpoints included.

    This is the shape a `goto` actually flies.  Minimum jerk retimes *when* each
    joint gets where it is going, but every joint still runs monotonically from
    its start to its goal, so the poses visited are these — which is why a check
    at the two endpoints alone proves nothing about the move between them.
    """
    a = np.asarray(list(q_from)[:7], dtype=float)
    b = np.asarray(list(q_to)[:7], dtype=float)
    return [list(a + (i / (steps - 1)) * (b - a)) for i in range(steps)]


class UnreachableError(RuntimeError):
    """Raised when no IK solution is found for a Cartesian target."""


class CollisionError(RuntimeError):
    """Raised when a planned path would collide with a static obstacle."""


class CartesianPlanner:
    """Plans one-arm joint trajectories for Cartesian gripper targets.

    ``side`` selects the arm: 'right' (default) or 'left'.  The left arm uses the
    mirrored joint set and baseline orientation; all other logic is shared.
    """

    def __init__(
        self,
        arm,
        scene: Optional[SceneModel] = None,
        side: str = "right",
        yaw_samples: int = 9,
        yaw_range: float = 1.0,
        pitch_samples: Sequence[float] = (-0.6, -0.3, 0.0, 0.3, 0.5, 0.8),
        tol: float = 0.012,
    ) -> None:
        self._arm = arm
        self._scene = scene
        self.side = side
        self.joints = _joints_for(side)
        self._R0 = _r0_for(side)
        self._yaws = np.linspace(-yaw_range, yaw_range, yaw_samples)
        self._pitches = pitch_samples
        self._tol = tol

    # ── IK ────────────────────────────────────────────────────────────────────

    def _world_to_armframe(self, xyz: XYZ) -> np.ndarray:
        return np.array(xyz) - _BASE

    def fk_world(self, joints: Sequence[float]) -> XYZ:
        """Forward kinematics: joint angles (deg) → gripper *pad* world point."""
        F = self._arm.forward_kinematics(list(joints))
        pad = F[:3, 3] + _BASE - F[:3, :3] @ _TOOL
        return (float(pad[0]), float(pad[1]), float(pad[2]))

    def _orientations(self, prefer: Optional[Tuple[float, float]]):
        pairs = [(y, p) for y in self._yaws for p in self._pitches]
        if prefer is not None:
            pairs = [prefer] + [c for c in pairs if c != prefer]
        return pairs

    def solve(
        self,
        xyz: XYZ,
        seed: Optional[Sequence[float]] = None,
        prefer: Optional[Tuple[float, float]] = None,
        return_orientation: bool = False,
        maximise_clearance: bool = False,
        from_joints: Optional[Sequence[float]] = None,
        gripper_deg: Optional[float] = None,
        **clearance_kw,
    ):
        """Return right-arm joint angles (deg) putting the gripper *pads* at pad
        world point xyz (tool-frame IK).

        Sweeps downward-ish orientations and keeps the lowest pad-error solution.
        ``prefer`` (a (yaw, pitch) pair) is tried first — passing the previous
        point's winning orientation makes contiguous path planning near-instant.
        Raises UnreachableError if none is within tolerance.

        ``maximise_clearance`` changes what "best" means, and on a board with
        objects on it that matters more than the last millimetre of pad error.
        The arm is redundant: many orientations put the pad on the same point
        with the elbow somewhere completely different, and the default rule
        (lowest pad error, stop at 1 mm) picks among them by a criterion that
        knows nothing about the table.  Measured over the notebook's grid, the
        difference between the solution it picked and the best one available at
        the same target:

            cell_r2c2 @ 28 cm hover    +0.6 cm  ->  +12.1 cm
            cell_r2c3 @ 28 cm hover    +0.8 cm  ->  +13.4 cm
            cell_r3c3 @ 18 cm hover    +1.9 cm  ->  +13.4 cm

        So the redundancy was there the whole time and the solver was spending
        it on nothing.  With this set, every candidate inside ``tol`` is scored
        by whole-arm clearance and the roomiest wins; pad error only has to be
        good enough, which is what "hover over the cell" actually needs.

        ``from_joints`` scores the whole PATH from that pose rather than the end
        pose alone — the same distinction that makes path_clearance worth having.
        Pass it whenever you know where the arm is starting from.  Extra keyword
        arguments go to SceneModel.clearance (``ids``, ``include_static``).

        Costs one clearance evaluation per candidate orientation, so this is for
        event-level planning, not a servo loop.
        """
        if maximise_clearance and self._scene is not None:
            return self._solve_roomiest(
                xyz, seed, return_orientation, from_joints, gripper_deg,
                **clearance_kw)
        pad = np.array(xyz)
        q0 = list(seed) if seed is not None else None
        best_q: Optional[List[float]] = None
        best_err = float("inf")
        best_pair: Optional[Tuple[float, float]] = None
        for (yaw, pit) in self._orientations(prefer):
            R = _rotz(yaw) @ _roty(pit) @ self._R0
            wrist = pad + R @ _TOOL          # wrist target so pads land at `pad`
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = wrist - _BASE
            try:
                q = self._arm.inverse_kinematics(M, q0=q0)
            except Exception:
                continue
            # A pose outside the joint travel is not a solution.  The arm
            # clamps it, ends up somewhere else, and every downstream check is
            # then answering questions about a pose that never existed.
            if not within_limits(q, self.side):
                continue
            F = self._arm.forward_kinematics(q)
            pad_fk = F[:3, 3] + _BASE - F[:3, :3] @ _TOOL
            err = float(np.linalg.norm(pad_fk - pad))
            if err < best_err:
                best_err, best_q, best_pair = err, list(q), (yaw, pit)
                if err < 1e-3:
                    break
        if best_q is None or best_err > self._tol:
            raise UnreachableError(
                f"No IK solution for pad {xyz} (best err={best_err:.4f} m)"
            )
        return (best_q, best_pair) if return_orientation else best_q

    def _solve_roomiest(
        self,
        xyz: XYZ,
        seed: Optional[Sequence[float]],
        return_orientation: bool,
        from_joints: Optional[Sequence[float]],
        gripper_deg: Optional[float],
        **clearance_kw,
    ):
        """solve() scored by whole-arm clearance instead of pad error.

        Every orientation is tried — no early exit, since the first solution to
        land on the target says nothing about where it puts the elbow.
        """
        pad = np.array(xyz)
        q0 = list(seed) if seed is not None else None
        best: Optional[Tuple[float, List[float], Tuple[float, float]]] = None
        best_err = float("inf")
        for (yaw, pit) in self._orientations(None):
            R = _rotz(yaw) @ _roty(pit) @ self._R0
            wrist = pad + R @ _TOOL
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = wrist - _BASE
            try:
                q = self._arm.inverse_kinematics(M, q0=q0)
            except Exception:
                continue
            if not within_limits(q, self.side):
                continue
            F = self._arm.forward_kinematics(q)
            pad_fk = F[:3, 3] + _BASE - F[:3, :3] @ _TOOL
            err = float(np.linalg.norm(pad_fk - pad))
            best_err = min(best_err, err)
            if err > self._tol:
                continue
            room = (self.path_clearance(from_joints, q,
                                        gripper_deg=gripper_deg, **clearance_kw)
                    if from_joints is not None
                    else self.clearance(q, gripper_deg, **clearance_kw))
            score = float("inf") if room is None else room.distance
            if best is None or score > best[0]:
                best = (score, list(q), (yaw, pit))
        if best is None:
            raise UnreachableError(
                f"No IK solution for pad {xyz} (best err={best_err:.4f} m)"
            )
        return (best[1], best[2]) if return_orientation else best[1]

    # ── Cartesian paths ─────────────────────────────────────────────────────────

    @staticmethod
    def interpolate(a: XYZ, b: XYZ, steps: int) -> List[XYZ]:
        """Straight-line Cartesian points from a to b (excluding a, including b)."""
        out: List[XYZ] = []
        for i in range(1, steps + 1):
            t = i / steps
            out.append((
                a[0] + t * (b[0] - a[0]),
                a[1] + t * (b[1] - a[1]),
                a[2] + t * (b[2] - a[2]),
            ))
        return out

    def check_collisions(
        self, points: Sequence[XYZ], ignore: Sequence[str] = ()
    ) -> None:
        """Raise CollisionError if any point violates the scene collision model."""
        if self._scene is None:
            return
        violations = self._scene.validate_path(points, ignore=ignore)
        if violations:
            first = violations[0]
            raise CollisionError(
                f"Path would collide ({len(violations)} pts): {first}"
            )

    # ── Whole-arm clearance ───────────────────────────────────────────────────

    @property
    def scene(self):
        """The SceneModel this planner guards against, or None."""
        return self._scene

    def clearances(self, joints: Sequence[float],
                   gripper_deg: Optional[float] = None, **kw):
        """Per-object clearance at one pose, as {id: Clearance}."""
        if self._scene is None:
            return {}
        return self._scene.clearances(
            link_capsules(joints, self.side, gripper_deg), **kw)

    def clearance(self, joints: Sequence[float],
                  gripper_deg: Optional[float] = None, **kw):
        """Worst approach of any arm link to any tracked object, at one pose.

        ``gripper_deg`` sizes the hand to its actual aperture; omitting it
        assumes the hand is wide open.  See ``hand_radius``.
        """
        if self._scene is None:
            return None
        return self._scene.clearance(
            link_capsules(joints, self.side, gripper_deg), **kw)

    def path_clearance(
        self, q_from: Sequence[float], q_to: Sequence[float],
        steps: int = 13, gripper_deg: Optional[float] = None, **kw,
    ):
        """Worst clearance anywhere along the joint-space move q_from → q_to.

        Checking only the endpoints is the mistake this method exists to stop:
        a move between two poses that both clear the table can still drag the
        forearm through an object halfway along, and it does.
        """
        if self._scene is None:
            return None
        worst = None
        for q in joint_path(q_from, q_to, steps):
            c = self._scene.clearance(
                link_capsules(q, self.side, gripper_deg), **kw)
            if c is not None and (worst is None or c.distance < worst.distance):
                worst = c
        return worst

    def path_clearances(
        self, q_from: Sequence[float], q_to: Sequence[float],
        steps: int = 13, gripper_deg: Optional[float] = None, **kw,
    ):
        """Per-object worst clearance along q_from → q_to, as {id: Clearance}.

        ``path_clearance`` collapses this to the single tightest object, which
        is the right answer only while every object is held to the same margin.
        It is not, once the arm is deliberately approaching one of them: hovering
        6 cm over the can puts the hand 6 cm from the can, so the can has to be
        allowed closer than everything else on the board.  Judging that needs to
        know which object each distance belongs to.
        """
        if self._scene is None:
            return {}
        worst: dict = {}
        for q in joint_path(q_from, q_to, steps):
            for oid, c in self._scene.clearances(
                    link_capsules(q, self.side, gripper_deg), **kw).items():
                if oid not in worst or c.distance < worst[oid].distance:
                    worst[oid] = c
        return worst

    def safe_fraction(
        self, q_from: Sequence[float], q_to: Sequence[float],
        margin: float = 0.03, steps: int = 13, tol: float = 1e-3,
        gripper_deg: Optional[float] = None, **kw,
    ) -> Tuple[float, object]:
        """How much of the move q_from → q_to keeps ``margin`` metres of air.

        Returns (fraction in [0, 1], the Clearance at that fraction).  1.0 means
        the whole move is clear; 0.0 means the arm is already inside the margin
        where it stands and no part of the move is safe.

        Clipping rather than skipping is deliberate: a sweep that stops at +22°
        because red_cube is under the forearm still demonstrates the joint, and
        it reports the real limit.  A skipped sweep demonstrates nothing.
        """
        full = self.path_clearance(q_from, q_to, steps, gripper_deg, **kw)
        if full is None or full.distance >= margin:
            return 1.0, full
        lo, hi = 0.0, 1.0
        a = np.asarray(list(q_from)[:7], dtype=float)
        b = np.asarray(list(q_to)[:7], dtype=float)
        while hi - lo > tol:
            mid = 0.5 * (lo + hi)
            c = self.path_clearance(a, list(a + mid * (b - a)), steps,
                                    gripper_deg, **kw)
            if c is not None and c.distance >= margin:
                lo = mid
            else:
                hi = mid
        return lo, self.path_clearance(a, list(a + lo * (b - a)), steps,
                                       gripper_deg, **kw)

    def clip(
        self, q_from: Sequence[float], q_to: Sequence[float],
        margin: float = 0.03, steps: int = 13,
        gripper_deg: Optional[float] = None, **kw,
    ) -> Tuple[List[float], float, object]:
        """``safe_fraction`` applied — the furthest pose along the move that is
        still clear, as (joints, fraction, Clearance)."""
        frac, c = self.safe_fraction(q_from, q_to, margin, steps,
                                     gripper_deg=gripper_deg, **kw)
        a = np.asarray(list(q_from)[:7], dtype=float)
        b = np.asarray(list(q_to)[:7], dtype=float)
        return list(a + frac * (b - a)), frac, c

    def plan_segment(
        self,
        start_xyz: XYZ,
        end_xyz: XYZ,
        steps: int,
        seed: Sequence[float],
        ignore: Sequence[str] = (),
    ) -> Tuple[List[List[float]], List[XYZ]]:
        """Plan a collision-checked Cartesian segment.

        Returns (joint_trajectory, cartesian_points).  Warm-starts each IK call
        from the previous solution.  Raises CollisionError/UnreachableError.
        """
        cart = self.interpolate(start_xyz, end_xyz, steps)
        self.check_collisions(cart, ignore=ignore)
        traj: List[List[float]] = []
        q = list(seed)
        prefer: Optional[Tuple[float, float]] = None
        for p in cart:
            q, prefer = self.solve(p, seed=q, prefer=prefer, return_orientation=True)
            traj.append(q)
        return traj, cart
