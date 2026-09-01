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

import logging
import math
from typing import List, Optional, Sequence, Tuple

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
# capsule that bounds both is both simpler and conservative.  Its length reaches
# the far face of the thumb pad (0.0325 + 0.085 + 0.022 = 0.1395 m below the
# wrist frame).
#
# ITS RADIUS DEPENDS ON THE GRIPPER ANGLE, and getting that wrong is not a
# rounding error.  The moving finger hangs off the thumb at y = -0.037 and
# swings about the wrist's local X, so opening the gripper throws it outward:
#
#     r_gripper   +20 (shut)   3.4 cm      0 (neutral)   4.7 cm
#                 -45 (open)   7.4 cm    -68 (wide)      8.2 cm   from the axis
#
# A fixed 5.0 cm radius, taken from the closed hand, was wrong by 3.2 cm exactly
# when it mattered — the notebook holds the gripper OPEN while hovering over the
# board.  Measured consequence: the guard reported 6.98 cm of clearance to
# foam_block along a hover approach, at every path sampling resolution from 13
# to 801 steps, and the physics threw the block 0.81 m.
#
# So the radius is computed from the aperture, and defaults to the worst case
# when the aperture is unknown.  A guard that has to guess should guess wide.
_SHOULDER_Y = {"right": -0.19, "left": 0.19}
_UPPER_ARM_LEN = 0.28
_FOREARM_LEN = 0.25
_HAND_LEN = 0.145
_UPPER_ARM_RADIUS = 0.035
_FOREARM_RADIUS = 0.030

# Gripper geometry, from the MJCF: the finger body hangs at y = -0.037 from the
# thumb frame and its shell is a box of half-extents (0.012, 0.010, 0.038)
# centred 0.038 below its own origin; the fixed thumb shell is
# (0.025, 0.028, 0.038) at y = -0.018.
_FINGER_HINGE_Y = -0.037
_FINGER_ARM = 0.038
_FINGER_HALF = (0.012, 0.010)
_THUMB_RADIUS = math.hypot(0.025, 0.046)      # fixed shell, gripper-independent
_GRIPPER_OPEN_LIMIT_DEG = -68.8               # MJCF range lower bound, -1.2 rad

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


def hand_radius(gripper_deg: Optional[float] = None) -> float:
    """Radius of the capsule that bounds the gripper at a given aperture.

    ``None`` returns the worst case — the fully open hand — because a guard
    asked to check a hand whose aperture it does not know must assume the widest
    one.  See the constants above for what a wrong answer here costs.
    """
    if gripper_deg is None:
        gripper_deg = _GRIPPER_OPEN_LIMIT_DEG
    y = _FINGER_HINGE_Y + _FINGER_ARM * math.sin(math.radians(gripper_deg))
    finger = math.hypot(_FINGER_HALF[0], abs(y) + _FINGER_HALF[1])
    return max(_THUMB_RADIUS, finger)


def link_capsules(joints: Sequence[float], side: str = "right",
                  gripper_deg: Optional[float] = None) -> List[Capsule]:
    """The arm's collision volume as (name, end, end, radius) world capsules.

    Feed straight to ``SceneModel.clearance``.  Only the three moving links are
    modelled: the torso and head do not move here, and the shoulder ball is
    inside the upper-arm capsule already.

    ``gripper_deg`` sizes the hand capsule to the actual aperture; omitting it
    assumes the hand is wide open, which is the safe assumption and costs about
    3.5 cm of reported clearance against a closed hand.
    """
    shoulder, elbow, wrist, R = link_frames(joints, side)
    tip = wrist + R @ np.array([0.0, 0.0, -_HAND_LEN])

    def xyz(v) -> XYZ:
        return (float(v[0]), float(v[1]), float(v[2]))

    return [
        ("upper_arm", xyz(shoulder), xyz(elbow), _UPPER_ARM_RADIUS),
        ("forearm", xyz(elbow), xyz(wrist), _FOREARM_RADIUS),
        ("hand", xyz(wrist), xyz(tip), hand_radius(gripper_deg)),
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
