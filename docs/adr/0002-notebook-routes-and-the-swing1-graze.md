# ADR-0002: Rig motion flies the measured notebook routes, and the SWING_1 graze is accepted

- Status: Accepted
- Date: 2026-09-10
- Decision owners: IITG Reachy 1.2 simulation project
- Supersedes: the route-compatibility position taken in #73 and #74 for FWDCenterLabSivaPool

## Context

Reachy's right arm hangs through an opening in the 80/20 frame — the **pocket**.
`notebooks/tlh_motion-routine.ipynb` establishes the only measured way in and out
of it: `PLACE_ROUTE` (HOME → REST, eleven waypoints) and `STOW_ROUTE` (its
reverse), verified at 0.2° resolution against the board, all five rig rails, the
pedestal and the robot's own links.

`primitives.raise_to_side` and `primitives.stow_from_side` predate the rig. They
took two waypoints from the measured route and invented the rest: fold the arm
from whatever the joints happened to be **reading**, then sweep
`r_shoulder_roll` from 0 to −88 in one command at fixed pitch, onto a "side hub"
that appears nowhere in the notebook.

An operator watching RViz saw the gripper pass through the board and the elbow
through a rail, and asked whether the physics layer was on.

**It was.** The backend is `mujoco-remote`; joint commands reach `data.ctrl[]`
position actuators, not `qpos`; the right arm carries four collision geoms
including the gripper boxes; rails (contype 8 / conaffinity 2) and the board
(4 / 7) both collide with the robot (2 / 5).

The failure was the pose being asked for. MuJoCo's contact resists in proportion
to **penetration**; a position servo pushes in proportion to **joint error**, and
the shoulder has `kp=300` with 60 N·m. A pose two degrees past a rail loses to
the rail — which is where this codebase's own *"commanded to −88, the roll
reached −14.1 and held there"* came from. A pose four to six centimetres inside
the board wins from the first timestep. Modelled against the scene, the invented
sweep holds the hand **4.3–6.5 cm inside `table_top`** for every degree of its
78° travel.

The notebook names that manoeuvre as its *failure* exercise: *"Try the sideways
version and watch it fail … the arm jams against the outer rail. That jamming IS
the feedback."*

## Decision

### 1. The two functions fly the measured routes

`raise_to_side` is `PLACE_ROUTE` followed by the notebook's own lift to `PRESENT`
(cell 18). `stow_from_side` is `STOW_ROUTE` — whose first waypoint is
`REST_SHUT`, which is why cell 34 flies it straight out of `PRESENT`. Each is the
other's exact reverse.

### 2. There is no side hub

It was invented. `PRESENT` is measured, and `PRESENT` is what "raised, out to the
robot's right" means in this rig. `POSTURE_SIDE_HUB`, `SIDE_HUB`, `PRESENT_MID`,
`PRESENT_LIFT`, `PRESENT_ROUTE` and `PRESENT_RETURN` are removed. The posture
graph is `HOME ↔ REST ↔ PRESENT`, every edge a notebook move.
`primitives.SIDE_HIGH` is now `PRESENT`'s angles.

### 3. A waypoint is a whole pose

Targets are never built from live joint readings. That is the defect under the
defect: a target assembled from what the arm happens to be reading carries
whatever the last route left behind — a stow after a wave entered the pocket
still holding the wave's ±60° of forearm yaw.

Waypoints are judged on the notebook's `CRITICAL` set (the four gross joints plus
`r_wrist_pitch`), not all seven. The forearm yaw and wrists spin the hand about
its own axis, are weak (kp=60), and converge over several waypoints; holding them
to `TRACK_TOL` aborts routes in no danger, as the first rebuilt flight
demonstrated by stopping at `SWING_2` with the arm 2.3 cm from anything.

### 4. The SWING_1 graze is accepted in FWDCenterLabSivaPool

`SWING_1` passes `rig_rail_outer_right` with about **6 mm** of modelled budget,
and the physics arm's tracking error is the same size. Every flight of every
route through that crossing reads a few millimetres negative. #74 rejected
`PLACE_ROUTE` on exactly this and was right on the evidence it had.

The evidence is now different:

| route | flights | worst vs rig (cm) | board |
|---|---|---|---|
| `RAISE_TO_SIDE` | 6 | −0.33 … −1.32 | undisturbed |
| `STOW_FROM_SIDE` | 3 | −0.61 … −0.69 | undisturbed |
| `STOW_ROUTE` | 3 | −1.03 … −1.58 | undisturbed |
| `LOWER_TO_REST` | 3 | +6.07 … +6.09 | undisturbed |
| `WAVE` | 3 | +13.11 … +14.20 | undisturbed |

Eighteen legs, sampled at 20 Hz **through** each flight rather than at the
waypoints, **with objects on the board** (`soda_can` on r1c1, `foam_block` on
r2c3), and the board undisturbed in every one. A graze that never moves anything
— at the rail #73 already documents as disagreeing with the model in both
directions — is treated as a model disagreement, not a collision.

This is deliberately **not** the argument the deleted hub rows made. Those rested
on "flown six times, board undisturbed" against a board that was *empty*, for a
path 4.3–6.5 cm inside the board. Rows are not evidence; the runs behind them
are, and an empty board proves nothing about disturbance.

## Consequences

- The panel executes `wave`, `store your arm` and `rest your arm` in
  FWDCenterLabSivaPool again, on routes that are the notebook's.
- `RAISE_TO_SIDE` is now a twelve-waypoint, ~40 s move. It reaches `PRESENT` by
  way of `REST`, so **the arm touches the board on its way out of the pocket**.
  That is the measured path; there is no verified shortcut.
- `SIDE_HIGH` changing angles changes the pick-and-place transit seed, which now
  starts from `PRESENT` rather than a roll −88 abduction.
- `LIFT_TO_PRESENT`'s waypoint tolerance is 8°, matching `posture_of`, so a route
  cannot report arrival at a pose the posture check will not recognise.

### What is explicitly not covered

**An object in the rest footprint will be hit — RESOLVED by #82.** These routes
end on the board; the forearm lands on the near-right cells. Observed on a run
where `foam_block` had slid 7 cm off r2c3 into that footprint: `LOWER_TO_REST`
moved it 5.3 cm and the `STOW_ROUTE` after it pushed the total to 17 cm.
`primitives` has no scene and cannot see this, which is why the check sits
above it, in `web/panel_executor.py::_ability_available` — it sweeps
`reachy_ai.motion.rig_routes.FOOTPRINT_LEGS` (the HOVER/PRESENT → REST_SHUT →
REST tail or head of every route that lands on, or departs from, REST) against
live object poses and refuses the whole ability rather than commanding a move
that would hit something. It gates `rest_forearm`, `stow_arm`, and any `wave`
or `point_*` request flown from the pocket by way of `RAISE_TO_SIDE`, which
crosses the identical corridor. These rows now certify the tabletop as well as
the corridor — for the REST-adjacent leg specifically; guarding the rest of the
rig corridor (`SWING_1` and #74) is still open.

**Recovery from mid-corridor is still narrow.** `rig_motion.stranded_at` only
recognises the four pocket waypoints. An arm left at `SWING_2` is reported, not
recovered, and needs the notebook's `ensure_home` treatment by hand.

## Revisit conditions

Revisit this ADR when:

- any flight of these routes disturbs the board, or the realised rig clearance
  goes materially past −1.6 cm;
- the rig geometry or the scene's rail placement changes;
- a measured shortcut from `SWING_3` to `PRESENT` is flown — modelled at +3.8 cm,
  it would remove the touch-down at `REST` on the way out, and it is currently
  unmeasured and therefore not used.
