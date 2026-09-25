"""The single degree->radian conversion boundary for the comparison tooling
(readiness review §3.2/M1; assignment T1).

``rig_routes`` poses are SDK degrees (``BACK = pose(r_shoulder_pitch=40.0)``).
``commands.jsonl``/``states.jsonl`` carry radians: ``reachy_sdk/joint.py:20,51``
converts a goto's degree setpoint with ``np.deg2rad`` and ships it as a
float32 ``FloatValue``, and the bridge copies that value unchanged
(``mujoco_remote_backend.py:701``). Mixing the two -- comparing a radian
target straight against a degree ``wp.pose`` -- is exactly M1/T1: it makes a
clean B flight a false STOP and every A flight vacuously "clean".

``route_rad()`` is the ONLY place a degree pose is converted. Every
comparison module downstream (``segments``, ``pathcheck``, ``echo``) takes a
``RadWaypoint``, whose ``pose_rad8`` is already ``float32(deg2rad(...))`` --
and never sees a degree value again. ``RadWaypoint.pose`` is a read-only
alias for ``pose_rad8`` (never a second, independently-set field) so code
written against ``rig_routes.Waypoint.pose`` (or the pre-existing local
radian test mocks in ``tests/fixtures/goalfix_cmp/make_fixtures.py``, which
are already radians and are unaffected by this module) keeps working
unchanged -- the type itself, not a renamed attribute, is what stops a
degree pose from reaching a comparison module.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from reachy_ai.motion.rig_routes import R_JOINTS, Waypoint  # noqa: E402

#: No joint on this rig approaches a full turn; a converted value at or past
#: this is a degrees/radians mix-up, not a pose (assignment T1: "assert this
#: where practical").
_PLAUSIBLE_RAD_BOUND = 2.0 * np.pi


def deg_to_rad_f32(value_deg: float) -> float:
    """``float32(deg2rad(value_deg))`` -- exactly what
    ``reachy_sdk/joint.py`` puts on the wire for one joint, and the only
    formula that may compute this conversion (fixtures mirror it via this
    same function, never re-derive it, so the two can never drift apart --
    see ``make_fixtures.FlightSim``'s ``pose_units="deg"`` mode)."""
    return float(np.float32(np.deg2rad(value_deg)))


def pose_deg8_to_rad8(pose_deg: Dict[str, float]) -> Dict[str, float]:
    """Every right-arm joint (``R_JOINTS``) in ``pose_deg``, converted."""
    out: Dict[str, float] = {}
    for j in R_JOINTS:
        rad = deg_to_rad_f32(pose_deg.get(j, 0.0))
        assert abs(rad) < _PLAUSIBLE_RAD_BOUND, (
            f"{j}: {rad} rad is not a plausible joint angle -- "
            "a degrees/radians mix-up, not a converted pose")
        out[j] = rad
    return out


@dataclass(frozen=True)
class RadWaypoint:
    """One route waypoint, converted to radians exactly once. Holds
    ``pose_rad8`` (radians, float32-quantised), ``seconds``, ``tol_deg``
    (still degrees -- it is a tolerance SIZE, read by no comparison module
    today, not a pose) and ``guard``."""

    name: str
    pose_rad8: Dict[str, float]
    seconds: float
    tol_deg: float
    guard: Optional[Tuple[str, ...]] = None

    @property
    def pose(self) -> Dict[str, float]:
        """Alias for ``pose_rad8`` -- radians, never degrees -- so this
        duck-types as a radian-pose waypoint for any existing caller
        written against ``.pose`` (``rig_routes.Waypoint`` and the local
        test-mock ``Waypoint`` both expose ``.pose`` too, and in every one
        of those cases the values are already the unit the caller expects:
        degrees for a raw ``rig_routes.Waypoint``, radians for a mock or
        for this class)."""
        return self.pose_rad8

    @property
    def tol(self) -> float:
        return self.tol_deg


def route_rad(route: Sequence[Waypoint]) -> Tuple[RadWaypoint, ...]:
    """``route`` (a real ``rig_routes`` tuple of degree ``Waypoint``s),
    converted once. The comparison modules (``segments``, ``pathcheck``,
    ``echo``) take this, never ``route`` itself."""
    return tuple(
        RadWaypoint(wp.name, pose_deg8_to_rad8(wp.pose), wp.seconds, wp.tol, wp.guard)
        for wp in route
    )


def pose8_rad(pose_deg: Dict[str, float]) -> Dict[str, float]:
    """``route_rad`` for one bare pose dict (``rig_routes.HOME``/``REST``,
    not wrapped in a ``Waypoint``) -- a leg's start pose, for example."""
    return pose_deg8_to_rad8(pose_deg)
