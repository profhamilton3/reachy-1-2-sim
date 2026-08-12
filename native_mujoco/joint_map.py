"""
R12-401: Reachy 1.2 explicit joint mapping table.

Maps between the Reachy v1 SDK domain (names, UIDs, degrees) and the MuJoCo
domain (joint indices, radians, MJCF axis conventions).

Design notes
------------
* SDK names and URDF joint names are identical for Reachy 1.2.
* MuJoCo joint ordering is depth-first in reachy_1_2.xml (verified by loading).
* SDK goal_position uses *degrees*; MuJoCo qpos/ctrl use *radians*.
* sign=+1 means SDK positive → MuJoCo positive (calibrate against hardware).
* limits_deg are the URDF joint limits converted to degrees for SDK display.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass(frozen=True)
class JointEntry:
    sdk_name: str          # reachy_sdk motor name == URDF joint name
    uid: int               # Dynamixel UID on physical hardware
    mjcf_index: int        # depth-first qpos/ctrl index in reachy_1_2.xml
    mjcf_axis: Tuple[float, float, float]  # rotation axis in MJCF body frame
    limits_rad: Tuple[float, float]        # (lower, upper) from URDF
    sign: float = 1.0      # SDK→MuJoCo sign (+1 or -1); calibrate on hardware

    @property
    def limits_deg(self) -> Tuple[float, float]:
        lo, hi = self.limits_rad
        return (math.degrees(lo), math.degrees(hi))

    def sdk_deg_to_mjcf_rad(self, deg: float) -> float:
        return self.sign * math.radians(deg)

    def mjcf_rad_to_sdk_deg(self, rad: float) -> float:
        return self.sign * math.degrees(rad)


# ---------------------------------------------------------------------------
# Canonical table — ordered by mjcf_index (depth-first in reachy_1_2.xml)
# ---------------------------------------------------------------------------
#   sdk_name             uid  mjcf_idx  axis              limits_rad
JOINT_TABLE: List[JointEntry] = [
    JointEntry("r_shoulder_pitch",  10,  0, ( 0, 1, 0), (-2.618,  1.570)),
    JointEntry("r_shoulder_roll",   11,  1, ( 1, 0, 0), (-3.140,  0.174)),
    JointEntry("r_arm_yaw",         12,  2, ( 0, 0, 1), (-1.570,  1.570)),
    JointEntry("r_elbow_pitch",     13,  3, ( 0, 1, 0), (-2.182,  0.000)),
    JointEntry("r_forearm_yaw",     14,  4, ( 0, 0, 1), (-1.745,  1.745)),
    JointEntry("r_wrist_pitch",     15,  5, ( 0, 1, 0), (-0.785,  0.785)),
    JointEntry("r_wrist_roll",      16,  6, ( 1, 0, 0), (-0.785,  0.785)),
    JointEntry("r_gripper",         17,  7, ( 1, 0, 0), (-1.200,  0.350)),

    JointEntry("l_shoulder_pitch",  20,  8, ( 0, 1, 0), (-2.618,  1.570)),
    JointEntry("l_shoulder_roll",   21,  9, ( 1, 0, 0), (-0.174,  3.140)),
    JointEntry("l_arm_yaw",         22, 10, ( 0, 0, 1), (-1.570,  1.570)),
    JointEntry("l_elbow_pitch",     23, 11, ( 0, 1, 0), (-2.182,  0.000)),
    JointEntry("l_forearm_yaw",     24, 12, ( 0, 0, 1), (-1.745,  1.745)),
    JointEntry("l_wrist_pitch",     25, 13, ( 0, 1, 0), (-0.785,  0.785)),
    JointEntry("l_wrist_roll",      26, 14, ( 1, 0, 0), (-0.785,  0.785)),
    JointEntry("l_gripper",         27, 15, ( 1, 0, 0), (-0.350,  1.200)),

    JointEntry("neck_roll",         30, 16, ( 1, 0, 0), (-0.800,  0.800)),
    JointEntry("neck_pitch",        31, 17, ( 0, 1, 0), (-0.800,  1.130)),
    JointEntry("neck_yaw",          32, 18, ( 0, 0, 1), (-2.790,  2.790)),
    JointEntry("l_antenna",         33, 19, ( 0, 0, 1), (-2.618,  2.618)),
    JointEntry("r_antenna",         34, 20, ( 0, 0, 1), (-2.618,  2.618)),
]

# ---------------------------------------------------------------------------
# Lookup helpers (build once at import time)
# ---------------------------------------------------------------------------
_BY_NAME: Dict[str, JointEntry] = {e.sdk_name: e for e in JOINT_TABLE}
_BY_UID:  Dict[int, JointEntry] = {e.uid: e for e in JOINT_TABLE}
_BY_IDX:  Dict[int, JointEntry] = {e.mjcf_index: e for e in JOINT_TABLE}

NUM_JOINTS = len(JOINT_TABLE)


def by_name(name: str) -> Optional[JointEntry]:
    return _BY_NAME.get(name)


def by_uid(uid: int) -> Optional[JointEntry]:
    return _BY_UID.get(uid)


def by_mjcf_index(idx: int) -> Optional[JointEntry]:
    return _BY_IDX.get(idx)


def all_entries() -> List[JointEntry]:
    return list(JOINT_TABLE)


def build_qpos_from_sdk_degrees(name_deg: Dict[str, float]) -> "list[float]":
    """Convert a {sdk_name: degrees} dict to a full 21-element qpos list.

    Joints not present in the dict keep their current value (0.0).
    """
    qpos = [0.0] * NUM_JOINTS
    for name, deg in name_deg.items():
        entry = _BY_NAME.get(name)
        if entry is not None:
            qpos[entry.mjcf_index] = entry.sdk_deg_to_mjcf_rad(deg)
    return qpos


def qpos_to_sdk_degrees(qpos: "list[float]") -> Dict[str, float]:
    """Convert a 21-element qpos list to a {sdk_name: degrees} dict."""
    return {
        e.sdk_name: e.mjcf_rad_to_sdk_deg(qpos[e.mjcf_index])
        for e in JOINT_TABLE
    }
