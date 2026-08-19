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


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _roty(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


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
    ):
        """Return right-arm joint angles (deg) putting the gripper *pads* at pad
        world point xyz (tool-frame IK).

        Sweeps downward-ish orientations and keeps the lowest pad-error solution.
        ``prefer`` (a (yaw, pitch) pair) is tried first — passing the previous
        point's winning orientation makes contiguous path planning near-instant.
        Raises UnreachableError if none is within tolerance.
        """
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
