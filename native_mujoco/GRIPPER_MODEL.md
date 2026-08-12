# Reachy 1.2 Gripper & Contact Model (R12-502)

How the native MuJoCo backend models grasping and grip force.  Implemented in
`gripper.py` (`GripperModel`), wired into `server.py` (`SimState`), tested in
`tests/unit/test_gripper.py`.

## Mechanism

Reachy 1.2 has a **single-DOF gripper per arm**: one actuated finger
(`{r,l}_gripper`) closing against a fixed thumb/palm.  There is no mimic/coupled
second joint to model.

The **visual** gripper keeps the URDF-derived shape.  The **collision** model
(R12-502) replaces the bulky finger/thumb boxes with a clean opposed-pad pair so
the revolute finger forms a real friction pinch:

- `{r,l}_thumb_col` — fixed pad on the palm side.
- `{r,l}_finger_col` — pad on the moving finger; closes toward the thumb pad.

Pads oppose along the gripper **y** axis (horizontal at the home pose, so pad
friction opposes gravity).  Open pad gap ≈ 28 mm; closed ≈ 7 mm — a ~20 mm cube
is pinched.

## Grip force

`GripperModel.update(data)` scans MuJoCo contacts once and, for each gripper,
sums the **normal** contact-force magnitude (`mj_contactForce`) on the finger and
thumb pads against non-robot geoms.  Units: **newtons**.  This maps to the SDK
force sensors:

| SDK sensor | uid | source |
|---|---|---|
| `r_force_gripper` | 1 | right finger+thumb pad normal force |
| `l_force_gripper` | 2 | left finger+thumb pad normal force |

Exposed in the `state` message as `force_sensors: [{uid, force}]`.

## Grasp detection

A gripper is **grasping** when the same object geom is in contact with *both* its
finger pad and its thumb pad above a small force floor (0.05 N) — a stable pinch.
Reported per side in `state.grippers` as `{side, grasping, grip_force_n,
grasped_geoms}`.

## Deterministic scenario (exit gate)

`tests/unit/test_gripper.py::TestGraspScenario` runs entirely on MuJoCo contact
physics (no weld/attach shortcut):

1. Right arm stiff & straight, gripper open; a 20 mm cube rests on a support
   pillar centred in the open pad gap (0 initial penetration).
2. **Close** → `grasping=True`, `grip_force_n ≈ 2 N`, `grasped_geoms=['cube']`.
3. **Lift** shoulder_pitch to −0.7 rad → cube rises ~13 cm tracking the gripper,
   grasp held.
4. **Release** (open) → cube drops off; `grasping=False`.

The support pillar uses a dedicated collision channel (contype=8/conaffinity=4)
so objects rest on it but the robot ignores it.

## Limitations

- Grip force is a contact-normal approximation, **not** a calibrated load cell;
  shear/tangential load and sensor dynamics are ignored.
- The friction pinch reliably holds while the gripper is near-vertical
  (shoulder_pitch up to ~−0.7 rad here).  Large wrist/arm tilts can let a small
  cube slide out of the shallow pads — expected for this simplified geometry.
- Collision pads are a functional approximation of the true finger shape; the
  visual mesh is unchanged.
