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

# Baseline "point down and forward" gripper orientation (from FK of a natural
# reaching pose); the sweep rotates this about world Z (yaw) and Y (pitch).
_R0 = np.array([
    [0.39, -0.07, -0.92],
    [-0.02, 1.00, -0.09],
    [0.92, 0.05, 0.39],
])

_BASE = np.array(ARM_BASE_IN_WORLD)


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
    """Plans right-arm joint trajectories for Cartesian gripper targets."""

    def __init__(
        self,
        arm,
        scene: Optional[SceneModel] = None,
        yaw_samples: int = 9,
        yaw_range: float = 1.0,
        pitch_samples: Sequence[float] = (-0.6, -0.3, 0.0, 0.3, 0.5, 0.8),
        tol: float = 0.012,
    ) -> None:
        self._arm = arm
        self._scene = scene
        self._yaws = np.linspace(-yaw_range, yaw_range, yaw_samples)
        self._pitches = pitch_samples
        self._tol = tol

    # ── IK ────────────────────────────────────────────────────────────────────

    def _world_to_armframe(self, xyz: XYZ) -> np.ndarray:
        return np.array(xyz) - _BASE

    def fk_world(self, joints: Sequence[float]) -> XYZ:
        """Forward kinematics: joint angles (deg) → gripper world point."""
        p = self._arm.forward_kinematics(list(joints))[:3, 3] + _BASE
        return (float(p[0]), float(p[1]), float(p[2]))

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
        """Return right-arm joint angles (deg) reaching gripper world point xyz.

        Sweeps downward orientations and keeps the lowest-error solution.
        ``prefer`` (a (yaw, pitch) pair) is tried first — passing the previous
        point's winning orientation makes contiguous path planning near-instant.
        Raises UnreachableError if none is within tolerance.
        """
        target = self._world_to_armframe(xyz)
        q0 = list(seed) if seed is not None else None
        best_q: Optional[List[float]] = None
        best_err = float("inf")
        best_pair: Optional[Tuple[float, float]] = None
        for (yaw, pit) in self._orientations(prefer):
            R = _rotz(yaw) @ _roty(pit) @ _R0
            M = np.eye(4)
            M[:3, :3] = R
            M[:3, 3] = target
            try:
                q = self._arm.inverse_kinematics(M, q0=q0)
            except Exception:
                continue
            got = np.array(self._arm.forward_kinematics(q)[:3, 3]) + _BASE
            err = float(np.linalg.norm(got - np.array(xyz)))
            if err < best_err:
                best_err, best_q, best_pair = err, list(q), (yaw, pit)
                if err < 1e-4:
                    break
        if best_q is None or best_err > self._tol:
            raise UnreachableError(
                f"No IK solution for {xyz} (best err={best_err:.4f} m)"
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
