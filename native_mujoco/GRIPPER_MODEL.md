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

`grip_force_n` is therefore a sum over **every contact point of both pads**: for
the reported 2026-10-06 hold it is about 250 N summed (about 125 N per side;
an earlier "~380 N" figure was wrong). Physical force realism is unvalidated.

## Contact model: no-slip (ADR-0005)

The model file runs MuJoCo's no-slip post-processing pass by default
(`noslip_iterations="10"`, tolerance 1e-6) because soft contacts otherwise let
a held cube creep within the grasp (#55). Override with
`--noslip-iterations N` (0..50; 0 = previous behaviour). Recordings made with
0 and 10 are different physics configurations (`physics_profile_id`). The
release/opening behaviour under no-slip is an open item, and none of this is
physical-robot validation. See `docs/adr/0005-mujoco-noslip-contact-model.md`.

## Pad contact: torsional friction (#201)

The four pad geoms (`r_thumb_col`, `r_finger_col` and the left mirrors) use
**`condim="4" priority="1"`**. Every other geom keeps the default `condim` 3
and priority 0, and no friction value changes.

- **`condim` 4** adds **torsional** friction about the contact normal to
  MuJoCo's sliding friction.
- **`priority` 1** makes a pad contact use the pad's own friction (sliding
  1.0, torsion 0.005 m: MuJoCo's defaults) instead of the larger of the two
  geoms'. Sliding is unchanged, since no scene object is above 1.0. Torsion
  is 0.005 m rather than the objects' 0.01 m.

**Why torsion.** Side-on, the pads close horizontally, so the contact normal
is the pinch axis. With `condim` 3 nothing resists a twist about that axis,
and a held cylinder pivots about the line between the pads. #180 found this:
26 of 56 pool cylinder placements were clean.

**Why priority.** With `condim` 4 alone, a pad contact took the object's
0.01 m torsion. The crane's release (red_cube, r3c2) then left a pad on the
cube after opening: 0.22 N, 0.04 mm, so `clean_placement` was false and
`tests/integration/test_place_offline.py` failed 3 of 3. At 0.005 m the
release is clean (0.0 N), and the cylinder still holds. At 0.002 m and
0.001 m, a cylinder lift failed (1 of 3 each).

**`multiccd`.** MuJoCo 3.11 enables multi-point convex collision by default.
It is a *disable* flag (`mjDSBL_MULTICCD`), and the model leaves it on. It
was already on in the #180 flights. A pad meets the standing cylinder at 3–5
points (thumb) and 1–3 (finger), not one, and more points did not stop the
pivot. Disabling it gives 1 point per pad, and the cylinder dropped at the
lift.

**Evidence:** `docs/reviews/2026-10-19-lab-true-pieces-201/`, with the pool
cylinder at Siva's 45.7 × 71.1 mm.

| Pads | Cylinder lifts (8 reachable cells) | Crane lift r3c2 release |
|---|---|---|
| `condim` 3 (before) | 2 of 8 | clean |
| `condim` 4, larger-of-two torsion (0.01 m) | 8 of 8 | pad left on the cube (0.22 N) |
| `condim` 6 (adds rolling) | 7 of 8 | — |
| **`condim` 4, priority 1 (0.005 m), chosen** | 8 of 8; placements 51 of 56 | clean |

That README also has the re-flown placements, the crane, and the torsion scan
(0.001–0.0075 m). In that scan, the crane release and one zero-margin red_cube pair
flip with no trend, and 0.005 m is the setting that keeps the crane clean and
every cylinder lift.

**Test fixture.** `tests/unit/test_gripper.py`'s prior fixture uses a μ 2.5
cube. It gives the cube priority 1 too, so its pad contacts keep the
larger-of-two rule.

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
- The pads' torsional friction and priority (#201) are a modelling choice
  validated only in simulation flights, not against the physical pads. With
  priority, an object whose friction is above 1.0 no longer raises the pads'.
