# ADR-0011: A second grasp — side-on, the hand level

- Status: Accepted. Covers #180. Simulator only.
- Date: 2026-10-19
- Relates to:
  - #195 and PR #197 (the crane at the October mount, the palm in the hand model);
  - #185 and PR #188 (Siva's taught poses: `docs/reviews/2026-10-14-taught-board-poses-185/`);
  - #187 / #199 (the SDK tip frame; unseeded IK);
  - ADR-0004 (the whole-arm preflight), ADR-0005 (no-slip contacts), ADR-0008 (the release check), ADR-0009 (placement), ADR-0010 (the October layout).
- Evidence: `docs/reviews/2026-10-19-side-on-grasp-180/` (README).

## Context

After #195 the crane picks the 6 cm cube from r3c2 only and sets it down cleanly only on r3c3. No top-down attitude straddles the cube from any other cell with the palm clear. The real robot already uses another grasp: Siva picks and places side-on, the hand level at every taught square, coming in from the front, the right or a diagonal.

## Decision

### 1. The side-on plan is a crane plan with two ladders

`tasks/side_grasp_live.py` plans a `CranePlan` whose `rungs` are the APPROACH:
- rung 0 is the grasp;
- rungs 1–8 go back along the approach, level, to the standoff;
- above it, straight up to the via, with the hover 2 rungs above the standoff.

A second ladder, `lift`, rises vertically from the grasp for the lift, the carry and the set-down (`CranePlan.up`).

The crane's executor (`execute_crane_pick`) flies it unchanged in structure. These are all the crane's: the measured hover correction, the gated slide-in re-measured at every rung, the straddle check before closing, the lift, the carry, the set-down, ADR-0008's release check and way out, and the abort path.

The changes to the crane modules are generalisations whose defaults are the crane's own behaviour:
- `CranePlan.up`, `progress`, `face_half_m` and `object_box`;
- `replan(..., on=)`;
- `centring_shift(..., target_gap_along_m, target_thumb_m)`;
- `straddle_check(..., closing=)`;
- `object_box(..., align=)`;
- the way home reads the current plan's own via.

Crane flights are bit-identical before and after (`crane_identity.txt`).

### 2. How a side-on grasp is planned

- **Directions.** A cube is tried along its four face normals; a cylinder every 15°. Only directions within 90° of the shoulder's bearing are tried, and the hand is level or pitched 5° down. Siva's taught tips are level within 5.5°. Their solved joints are the IK seeds (`TAUGHT_SEEDS`): attitudes, never positions, because the taught spacing is still open (#185).
- **Choosing.** Up to four directions are planned in full. The one passing every check with the largest margin wins.
- **Geometry.**
  - The jaw-gap centre is 34 mm above the table. The level forearm reaches 30 mm below it.
  - The object stands 1.5 mm off the thumb pad.
  - The hand backs off along the approach until the palm clears the near face by 2 mm.
- **Opening and close.** The opening leaves the object 7.8 mm of free gap (the 6 cm cube's slack at the crane's −65°). The close stops 15° past the angle at which the finger meets the object. The reason is in §4.
- **Shapes.** Cubes (class `cube`) and cylinders (class `cylinder`) only. A cylinder must also fit the open jaw. blue_cylinder (70 mm) does not: the jaw leaves 67.8 mm at −65° and 69.0 mm at −68°. The can and the foam block are refused by shape.

### 3. The checks

- **The crane's checks** are reused with the palm in the hand model:
  - joint travel;
  - arm, pads and palm against the table, commanded and predicted;
  - pad and palm route along the slide-in;
  - the predicted straddle (its jaw-face angle taken from the CLOSING direction, `HandFrames.closing`);
  - the carried object;
  - the placement's held-object, exit and way-out checks.
- **Added: neighbours.** The pads, palm and hand back are checked against every other object along every hand segment, with 10 mm required. A direction blocked by a neighbour fails, and another is chosen or the request is refused.
- **Whole-arm preflight with the box hand.** A side-on plan is checked and flown with `BoxHandPlanner`: the arm links stay capsules, and the hand is the MJCF's collision boxes instead of the tube. The tube is one capsule of radius 114 mm (at −65°) around the hand axis. A level hand 34 mm above the table pushes it through the board: at r1c1 it read −9 mm into the rail under the board, where the real boxes stand 84 mm clear. The crane keeps the tube.

### 4. Why the close is not the crane's

A level hand's finger sweeps along the gripped face as it closes, toward the fingertips. Closing to the crane's −5° carried the 6 cm cube 48 mm forward and out of the jaw in the first flight. The isolated close study (`close_study.py`) found what holds:

| Close | 6 cm cube | 40 mm objects |
|---|---|---|
| Crane rule: −5°, slack split evenly | drift up to 15 mm; squeezed out 1 in 11 | pool cylinder tipped off the table at −65° |
| Contact + 15°, 1.5 mm thumb gap | 34 of 36 held, drift ≤ 5.9 mm | box 6 of 6; cylinder held |

### 5. Planner choice

`plan_lift_choice` and `plan_place_choice` plan both grasps. The one that passes every check with the larger margin flies, and the record says which (`GRASP_CHOSEN`, `outcome.grasp`).

The margin (`plan_margin`) is the least distance to any threshold, in millimetres of hand travel. A joint stop counts as the jaw-gap centre's travel to it.

The panel's `lift_object` and `place_on_cell` use the choice.

## Consequences

- Red_cube, the pool boxes and the pool cylinders plan side-on from 8 of 9 cells. Every ordered pair among those 8 cells plans for the cube and the box.
- r3c1 is out of reach for a level hand: no grasp pose inside the joint travel at any direction, even pitched 10° down (`reach_level.txt`).
- Where both grasps plan (red_cube from r3c2), side-on has the larger margin. The crane's least margin there is 0.0: it stands on the arm-yaw stop.
- **Offline flights:**
  - red_cube: 53 of 56 placements clean, lifts from all 8 reachable origins.
  - pool_box_1: 53 of 56.
  - pool_cyl_1: 26 of 56 (see Known limitations).
- **Live:**
  - red_cube r2c2 → r1c1 placed 1.7 mm from the centre twice: the first run 5 of 6 criteria (slip 0.102 mm against 0.1 mm, with the hand itself reading 0.069 mm of motion), the repeat 6 of 6.
  - pool_cyl_1 r2c3 → r2c2: 6 of 6.

## Known limitations

- **Cylinder holds are single-point.** MuJoCo gives one contact per pad on a box–cylinder pair here (no `multiccd`, `condim` 3). A held cylinder can swing about the pinch axis. In the executor's flights it stayed within the 0.1 mm slip bound. In the isolated study about half of the holds crept more. A model change (`multiccd`, or `condim` 4 on the pads) would remove it; it is not made here.
- **The prediction tolerances are the crane's** (two recorded August descents), reused for the level slide-in.
- **The supply spot is not a target.** The taught spacing (#185) is still open.
