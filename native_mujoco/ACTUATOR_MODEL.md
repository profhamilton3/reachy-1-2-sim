# Reachy 1.2 Actuator & Compliance Model (R12-501)

How the native MuJoCo backend maps Reachy v1 SDK joint semantics onto MuJoCo
`position` actuators.  Implemented in `actuator.py` (`ActuatorController`),
wired into `server.py` (`SimState`), tested in `tests/unit/test_actuator.py`.

## SDK semantics → MuJoCo

| SDK concept | Meaning | MuJoCo realisation |
|---|---|---|
| `goal_position` (rad) | target angle | `data.ctrl` of the position actuator |
| `compliant = True` | motor off, back-drivable | zero `gainprm[0]`, `biasprm[1]`, `biasprm[2]` → actuator force = 0 |
| `compliant = False` (stiff) | motor holds goal | restore nominal `kp/-kp/-kv` PD gains |
| `speed_limit` (rad/s) | max move speed, 0 = unlimited | rate-limit the effective ctrl target per step |
| `torque_limit` (0–100 %) | fraction of max torque | scale `actuator_forcerange`; report saturation |

A MuJoCo `position` actuator computes
`force = kp·(ctrl − qpos) − kv·qvel`.
Zeroing the gains makes the joint free except for its passive joint `damping`,
which is exactly the SDK "compliant" (back-drivable) behaviour.

## Default compliance

Mirrors `KinematicBackend`: **all joints start compliant except the two
antennas** (`l_antenna`, `r_antenna`), which are always stiff on the physical
robot.

## Simulated torque limits (`forcerange`, Nm)

Nominal sim values with gravity headroom — **not** exact Dynamixel stall
torques.  `torque_limit_percent` scales these at runtime.

| Joint(s) | forcerange (± Nm) |
|---|---|
| shoulder_pitch, shoulder_roll | 60 |
| elbow_pitch | 40 |
| arm_yaw | 30 |
| forearm_yaw | 15 |
| wrist_pitch, wrist_roll | 10 |
| neck_roll, neck_pitch, neck_yaw | 10 |
| gripper | 8 |
| antenna | 0.5 |

## Measured behaviour (timestep 0.002 s, `implicitfast`)

- **Stiff step response** — elbow commanded to −1.0 rad settles to −0.979 rad
  (small gravity steady-state offset from finite-gain PD), `|qvel| < 0.01` rad/s
  after ~4 s.  No oscillation, no NaN.
- **Compliance** — a compliant joint produces `|actuator_force| < 1e-6` Nm
  regardless of goal, and drifts under gravity from a lifted start.
- **Speed limit** — 0.5 rad/s cap reaches ~0.43 rad in 1.0 s vs ~0.9 rad
  unlimited.
- **Torque limit** — 3 % on a gravity-loaded shoulder saturates at exactly
  1.8 Nm (= 3 % × 60) and cannot reach the goal; `is_saturated()` returns True.
  At 100 % the same move completes and is not saturated.
- **Limit enforcement** — a goal beyond `ctrlrange` is clamped to the joint
  limit (e.g. wrist_pitch goal 5.0 → 0.781 rad, limit 0.785).

## State exposed per joint (in the `state` protocol message)

`position_rad`, `velocity_rad_s`, `effort` (= `actuator_force`, Nm),
`compliant` (bool), `saturated` (bool).

## Deliberate deviations from hardware

- Torque limits are sim-tuned for stable position control, not calibrated to
  Dynamixel datasheets.
- No motor thermal model, backlash, or friction beyond joint `damping`.
- Compliance is binary (on/off); the SDK's partial-torque compliance is
  approximated by `torque_limit_percent`.
