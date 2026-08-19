"""
R12-501: Actuator and compliance model for the native MuJoCo backend.

Maps Reachy v1 SDK joint semantics onto MuJoCo position actuators:

  * goal_position  — target angle (radians).  Stiff joints converge to it.
  * compliant      — True: motor off / back-drivable (actuator gain zeroed).
                     False (stiff): PD actuator holds goal_position.
  * speed_limit    — max |d(target)/dt| in rad/s applied to the *commanded*
                     target (0 = unlimited).  Models the SDK moving-speed cap.
  * torque_limit   — 0..100 %.  Scales the actuator forcerange; the joint
                     saturates (and is reported) when it hits the scaled limit.

Compliance is implemented by modulating the actuator's PD gains in the model:
a MuJoCo `position` actuator computes
    force = gainprm[0]*ctrl + biasprm[1]*qpos + biasprm[2]*qvel
          = kp*(ctrl - qpos) - kv*qvel
Setting gainprm[0]=biasprm[1]=biasprm[2]=0 makes the actuator produce no force,
so the joint is free save for its passive joint `damping` — i.e. compliant.

Default compliance mirrors KinematicBackend: every joint compliant EXCEPT the
two antennas (always stiff on the physical robot).

This module imports only `mujoco` + `numpy`; it runs on the native host side.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import mujoco
import numpy as np

# Actuators stiff by default: the antennas (as on the physical robot) and the
# neck, so the head HOLDS its pose and looks at the workspace instead of drooping
# to the floor under gravity (the cameras are head-mounted).
_DEFAULT_STIFF_ACTUATORS = (
    "act_l_antenna", "act_r_antenna",
    "act_neck_roll", "act_neck_pitch", "act_neck_yaw",
)
_SATURATION_EPS = 1e-3  # Nm margin for saturation detection


@dataclass
class JointControlState:
    """SDK-visible control state for one actuated joint."""
    goal_position: float = 0.0     # rad
    compliant: bool = True
    speed_limit: float = 0.0       # rad/s, 0 = unlimited
    torque_limit: float = 100.0    # percent of nominal forcerange


class ActuatorController:
    """Owns per-joint control state and drives MuJoCo actuators each step.

    One controller instance owns one (model, data) pair.  It mutates
    ``model.actuator_gainprm/biasprm/forcerange`` to realise compliance and
    torque limits, so it must be the sole writer of those arrays.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self.model = model
        self.nu = model.nu

        # actuator index -> qpos/qvel index (base model is 1:1, all hinges)
        self._qpos_adr = np.empty(self.nu, dtype=int)
        self._dof_adr = np.empty(self.nu, dtype=int)
        self._names: List[str] = []
        for a in range(self.nu):
            jid = int(model.actuator_trnid[a][0])
            self._qpos_adr[a] = model.jnt_qposadr[jid]
            self._dof_adr[a] = model.jnt_dofadr[jid]
            self._names.append(
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
            )
        self._name_to_idx = {n: i for i, n in enumerate(self._names)}

        # Snapshot nominal PD gains and force limits (restored on stiffen).
        self._nominal_kp = model.actuator_gainprm[:, 0].copy()
        self._nominal_bias_p = model.actuator_biasprm[:, 1].copy()  # = -kp
        self._nominal_bias_v = model.actuator_biasprm[:, 2].copy()  # = -kv
        self._nominal_force = model.actuator_forcerange[:, 1].copy()  # upper

        # Per-joint control state.
        self.state: List[JointControlState] = [
            JointControlState() for _ in range(self.nu)
        ]
        for name in _DEFAULT_STIFF_ACTUATORS:
            idx = self._name_to_idx.get(name)
            if idx is not None:
                self.state[idx].compliant = False

        # Rate-limited effective target fed to data.ctrl.
        self._effective_target = np.zeros(self.nu)
        self._applied = False

    # ── Command setters (accept actuator index or name) ───────────────────

    def _resolve(self, key) -> int:
        if isinstance(key, str):
            return self._name_to_idx[key]
        return int(key)

    def set_goal_position(self, key, pos_rad: float) -> None:
        self.state[self._resolve(key)].goal_position = float(pos_rad)

    def set_compliant(self, key, compliant: bool) -> None:
        self.state[self._resolve(key)].compliant = bool(compliant)

    def set_speed_limit(self, key, rad_s: float) -> None:
        self.state[self._resolve(key)].speed_limit = max(0.0, float(rad_s))

    def set_torque_limit(self, key, percent: float) -> None:
        self.state[self._resolve(key)].torque_limit = float(
            np.clip(percent, 0.0, 100.0)
        )

    # ── Per-step application ──────────────────────────────────────────────

    def sync_targets_to_current(self, data: mujoco.MjData) -> None:
        """Initialise the rate-limited target to the current pose.

        Call once after reset so speed-limited moves start from the real angle
        rather than snapping from zero.
        """
        for a in range(self.nu):
            self._effective_target[a] = data.qpos[self._qpos_adr[a]]
            self.state[a].goal_position = data.qpos[self._qpos_adr[a]]

    def apply(self, data: mujoco.MjData, dt: float) -> None:
        """Update gains, force limits and ctrl for the upcoming mj_step.

        Must be called before mujoco.mj_step each cycle.
        """
        for a in range(self.nu):
            st = self.state[a]

            # Torque limit: scale the actuator's force range.
            fmax = self._nominal_force[a] * (st.torque_limit / 100.0)
            self.model.actuator_forcerange[a, 0] = -fmax
            self.model.actuator_forcerange[a, 1] = fmax
            self.model.actuator_forcelimited[a] = 1

            if st.compliant:
                # Motor off: zero the PD gain and bias -> no actuator force.
                self.model.actuator_gainprm[a, 0] = 0.0
                self.model.actuator_biasprm[a, 1] = 0.0
                self.model.actuator_biasprm[a, 2] = 0.0
                # Track current position so re-stiffening does not jump.
                self._effective_target[a] = data.qpos[self._qpos_adr[a]]
                data.ctrl[a] = self._effective_target[a]
                continue

            # Stiff: restore nominal PD gains.
            self.model.actuator_gainprm[a, 0] = self._nominal_kp[a]
            self.model.actuator_biasprm[a, 1] = self._nominal_bias_p[a]
            self.model.actuator_biasprm[a, 2] = self._nominal_bias_v[a]

            # Speed limit: move the effective target toward goal at <= speed.
            goal = st.goal_position
            if st.speed_limit > 0.0:
                max_step = st.speed_limit * dt
                delta = goal - self._effective_target[a]
                if delta > max_step:
                    delta = max_step
                elif delta < -max_step:
                    delta = -max_step
                self._effective_target[a] += delta
            else:
                self._effective_target[a] = goal

            # Clamp to actuator ctrlrange (joint limit enforcement).
            lo, hi = self.model.actuator_ctrlrange[a]
            data.ctrl[a] = float(np.clip(self._effective_target[a], lo, hi))

        self._applied = True

    # ── Introspection ─────────────────────────────────────────────────────

    def actuator_force(self, data: mujoco.MjData, key) -> float:
        return float(data.actuator_force[self._resolve(key)])

    def is_saturated(self, data: mujoco.MjData, key) -> bool:
        """True if the joint is at its (scaled) torque limit."""
        a = self._resolve(key)
        fmax = self.model.actuator_forcerange[a, 1]
        if fmax <= 0.0:
            return bool(abs(data.actuator_force[a]) > _SATURATION_EPS)
        return bool(abs(data.actuator_force[a]) >= (fmax - _SATURATION_EPS))

    def saturation_mask(self, data: mujoco.MjData) -> List[bool]:
        return [self.is_saturated(data, a) for a in range(self.nu)]

    def present_position(self, data: mujoco.MjData, key) -> float:
        return float(data.qpos[self._qpos_adr[self._resolve(key)]])

    def present_velocity(self, data: mujoco.MjData, key) -> float:
        return float(data.qvel[self._dof_adr[self._resolve(key)]])

    def name(self, idx: int) -> str:
        return self._names[idx]
