# SDK capability matrix

What the simulator's Reachy v1 SDK surface does, against the real Reachy 1.2. Started at #187 with the arm kinematics; rows are added as each surface is checked against the robot or the URDF, not before.

| SDK surface | Real robot | Simulator | Verified by |
|---|---|---|---|
| `r_arm.forward_kinematics(q)` / `l_arm.forward_kinematics(q)` | 4x4 pose of the gripper tip (URDF `right_tip` / `left_tip`) in the torso frame | Same frame: URDF chain to the tip, wrist roll 0.0325 m below the pitch centre, tip at (0, ∓0.012, −0.085) in the thumb frame (`fake_reachy_server._arm_fk`) | `tests/contract/test_arm_kinematics_tip.py`: the real robot's `B2_SAFE_JOINTS` give its recorded pose to < 0.001 mm / 0.003° (`tests/fixtures/kinematics/b2_safe_real_fk.json`) |
| `r_arm.inverse_kinematics(M, q0)` | Joints putting the tip at M; `ValueError` if no solution; with `q0=None` the sim seeds from the arm's present joints (#199) | Same frame. Solves the wrist at the seed's roll, then re-solves the six joints above the roll for the tip (`fake_reachy_server._arm_ik`); `ValueError` if the wrist solve misses its 0.01 cost threshold and the tip is not landed | Round trips in the contract test and `tests/unit/test_tip_frame_187.py` |
| Left-arm shoulder roll sign | URDF `l_shoulder_roll` axis +x | The sim chain applies −q (unchanged by #187) | Not checked against the robot; only the right arm is used |

## Migration note (#187)

Before #187 the simulator's arm FK/IK frame was the wrist, about 0.1175 m from the tip. For the same joints, printed `forward_kinematics()` numbers changed by about 12 cm. Code written on the real robot (taught pose → `inverse_kinematics()` → goto) now behaves the same in the sim.

The sim's own planner plans from the wrist and its pad point, as before. It converts through `kinematics.fk_wrist` / `kinematics.ik_wrist`, so pads for given joints did not move, and the planner's IK answers match the pre-#187 ones. Measured on the offline r3c2 → r3c3 placement flight: final IK answers within 0.03°, commanded goals within 0.006°, pads within 0.02 mm.

Unseeded IK (#199): the SDK sends no `q0` unless the caller passes one. The sim then seeds the solve from that arm's present joint positions (the server's joint state), not from zero. A request that carries a `q0` is solved from it, unchanged. This is the sim's chosen behaviour; the real robot's no-`q0` behaviour cannot be seen from the SDK. From a present pose near the crane ladder, the five rungs that stalled from zero (`crane_pick_rung_09`, `_10`, `_18`, `crane_dest_rung_15`, `_18`) solve; from a pose far from a target an unseeded call can still stall, as any seeded one can. `scripts/demo_control_panel.py`'s first, unseeded call may plan slightly differently (legacy demo, accepted). Every planner call is seeded.

## Perception grading tooling (#186)

Not an SDK surface: offline tools that grade the robot team's detector and stereo depth on the sim cameras. `scripts/export_truth.py` is a read-only, short-lived WebSocket client (`hello` and `heartbeat_ack` only) that writes the object truth as `truth.v1`; `scripts/perception_grade.py` grades the viewer's single-pair bundles against it in the URDF torso frame. Covered by offline tests only; no capture has been run. See `docs/reviews/2026-10-20-perception-stage1-186/README.md`.
