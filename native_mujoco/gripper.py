"""
R12-502: Gripper and contact model for the native MuJoCo backend.

Reachy 1.2 has a single-DOF gripper per arm: one actuated finger
(`{r,l}_gripper` joint) closing against a fixed thumb/palm.  This module reads
MuJoCo contact state to:

  * compute an approximate grip force (N) per gripper — mapped to the SDK force
    sensors `r_force_gripper` (uid 1) and `l_force_gripper` (uid 2);
  * detect a grasp — an object geom in contact with BOTH the finger and the
    thumb of the same gripper (a stable pinch);
  * expose per-gripper contact state.

Force units: newtons (sum of contact normal-force magnitudes on the gripper
collision geoms).  Limitations: this is a contact-normal approximation, not a
calibrated load cell; it ignores tangential/shear load and sensor dynamics.

Imports only `mujoco` + `numpy`; runs on the native host side.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import mujoco
import numpy as np

# side -> (finger collision geom, thumb collision geom, SDK force-sensor uid)
GRIPPER_DEFS = {
    "right": ("r_finger_col", "r_thumb_col", 1),
    "left":  ("l_finger_col", "l_thumb_col", 2),
}

# A robot collision geom has contype == 2 (see reachy_1_2.xml collision model).
_ROBOT_CONTYPE = 2
_GRASP_FORCE_MIN_N = 0.05   # ignore incidental brushing below this


@dataclass
class GripperState:
    side: str
    sensor_uid: int
    grip_force_n: float = 0.0
    grasping: bool = False
    grasped_geoms: List[str] = field(default_factory=list)
    finger_contacts: int = 0
    thumb_contacts: int = 0


class GripperModel:
    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self._finger_gid: Dict[str, int] = {}
        self._thumb_gid: Dict[str, int] = {}
        self._sensor_uid: Dict[str, int] = {}
        for side, (finger, thumb, uid) in GRIPPER_DEFS.items():
            self._finger_gid[side] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, finger)
            self._thumb_gid[side] = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, thumb)
            self._sensor_uid[side] = uid

    def _normal_force(self, data: mujoco.MjData, contact_idx: int) -> float:
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(self.model, data, contact_idx, wrench)
        # wrench[0] is the normal component in the contact frame.
        return abs(float(wrench[0]))

    def _is_object(self, gid: int) -> bool:
        """Non-robot collidable geom = a scene object (or world)."""
        return self.model.geom_contype[gid] != _ROBOT_CONTYPE

    def update(self, data: mujoco.MjData) -> Dict[str, GripperState]:
        """Scan contacts once and return per-gripper state."""
        states = {
            side: GripperState(side=side, sensor_uid=self._sensor_uid[side])
            for side in GRIPPER_DEFS
        }
        # object geoms touching the finger / thumb of each side, with force
        finger_objs: Dict[str, Set[int]] = {s: set() for s in GRIPPER_DEFS}
        thumb_objs: Dict[str, Set[int]] = {s: set() for s in GRIPPER_DEFS}

        for i in range(data.ncon):
            c = data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for side in GRIPPER_DEFS:
                fg = self._finger_gid[side]
                tg = self._thumb_gid[side]
                st = states[side]
                if fg in (g1, g2):
                    other = g2 if g1 == fg else g1
                    if self._is_object(other):
                        f = self._normal_force(data, i)
                        st.grip_force_n += f
                        st.finger_contacts += 1
                        if f >= _GRASP_FORCE_MIN_N:
                            finger_objs[side].add(other)
                if tg in (g1, g2):
                    other = g2 if g1 == tg else g1
                    if self._is_object(other):
                        f = self._normal_force(data, i)
                        st.grip_force_n += f
                        st.thumb_contacts += 1
                        if f >= _GRASP_FORCE_MIN_N:
                            thumb_objs[side].add(other)

        for side, st in states.items():
            # A pinch: same object geom held between finger AND thumb.
            pinched = finger_objs[side] & thumb_objs[side]
            if pinched:
                st.grasping = True
                st.grasped_geoms = [
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g)
                    for g in sorted(pinched)
                ]
        return states

    def force_by_sensor_uid(self, data: mujoco.MjData) -> Dict[int, float]:
        """Map SDK force-sensor uid -> grip force (N) for the sensor service."""
        states = self.update(data)
        return {st.sensor_uid: st.grip_force_n for st in states.values()}
