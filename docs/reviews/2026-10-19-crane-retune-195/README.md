# The crane re-tuned at the October mount (#195)

**Date:** 2026-10-18/19.  **Offline:** MuJoCo 3.11 on the host; host IK mirror (`tests/unit/host_ik_arm.py`);
flights through `tools/place_sweep/fly.py` (the crane module unchanged, against the offline world).  Scene
FWDCenterLabSivaPool, red_cube (6 cm).  Run every script from the repo root with
`PYTHONPATH=.:src:native_mujoco:docs/reviews/2026-10-19-crane-retune-195`.

## What was wrong

1. **The palm.**  #192 gave the hand two collision boxes (`r_palm_col`, `r_hand_back_col`) that fill the
   visual hand body.  The crane's own hand model (`grasp_alignment`) still had only the two pads, so it planned
   grasps that drive the palm into the cube.  The first October lift flight from r3c2 (caps opened to let it
   through) halted on `object moved 0.20 mm`: `r_palm_col` touched the cube a rung above grasp depth, with every
   pad check passing (`fly_probe.py`, the contact watcher).  The planned grasp put the palm 13.7 mm into the cube.
   The palm is now in the hand model: `hand_frames(...).palm`, the route check (`palm_route` stage), the table
   check, and the grasp height (raised until the palm clears the object by `CraneSpec.palm_clearance_m` = 2 mm).
2. **The correction caps** (6 / 13 mm) were the August arm's tracking error.  At the October mount the realised
   jaw-gap centre sits 5.3-5.4 mm sideways and ~9.4 mm low of the commanded one at the hover, and every lift or
   placement needs a 7.3-7.9 mm hover correction: the 6 mm cap refused all of them.

## The search

| Step | Script → output | Finding |
|---|---|---|
| Hand only | `palm_geometry.py` → `search/palm_geometry.txt` | With the palm 2 mm above the 6 cm cube, both pad centres are below its top only up to ~20-25 deg tilt. |
| Reach | `reach_attitude.py` → `search/reach.jsonl` | Bounded least squares inside the joint travel (independent of the IK service): tilts 10-20 reach r1c3, r2c3, r3c2, r3c3 at some leans; r2c2/r2c1 need >= 25; r3c1 none up to 30. The wrist's +/-45 deg pitch and roll are what bind. |
| Lift grid | `lift_grid.py` → `search/lift_grid_wide.jsonl` | 9 cells x faces x 8 tilts x 24 leans, 216-start IK set, every crane check: 2 of 3456 pass (r3c2, tilt 20, lean 90). |
| Lift, seeded | `lift_seeded.py` → `search/lift_seeded.jsonl` | Each reachable attitude with its bounded pose as an IK seed: still r3c2 only (tilt 20-25, lean 90); elsewhere the ladder runs out of travel, the straddle fails or the palm does. |
| Place grid | `place_grid.py` → `search/place_grid_r3c2.jsonl` | From r3c2, tilts 10-25: r2c1, r2c2 (22.5-25) and r3c3 (20-25) plan, all at lean 90. |
| Fine grid | `fine_grid.py` → `search/fine_grid.jsonl` | tilt 20-26.25 x lean 80-100: lift + r2c1 + r2c2 + r3c3 plan for tilt >= 22.5 at lean 85-90 (22.5 needs lean >= 87.5 for r2c2). |
| Lift flights | `fly_batch.py` → `flights/band_lift.{jsonl,md}` | The lift FLIES clean for tilt 20-22.5 (all leans) and 23.75 at lean 85-87.5; steeper, the descent's straddle check halts it (thumb pad over the face). 25 / 87.5 was tried first and halted in all 28 flights (`flights/t25_caps_open.md`). |

**Chosen: tilt 22.5, lean 87.5, for the lift and the placement** (`CraneSpec()`, `PLACE_GRASP = CraneSpec()`):
the one attitude that plans all three destinations and has clean lift flights on every side.  **Origin r3c2** is
the only one: r2c2 (the #174 origin) cannot be straddled with the palm clear at any reachable attitude.

**Caps: 8 mm lateral, 10 mm vertical** (were 6 / 13), by the August rule (largest need, rounded up to the next mm)
over the 39 October flights that grasped and finished with the caps opened to measure (lateral 7.31-7.86 mm,
vertical 8.89-9.04 mm).

**Set-down aim: the cell centre** (was 1 mm toward (0, 0)).  `aim_check.py` → `aim_check.txt`: the set-down's
least margin is now the wrist pitch, not the shoulder roll (6-54 deg spare), and aiming toward the robot only
takes wrist-pitch margin away.

**Release retreat: unchanged (9 mm).**  r3c3 passes with it; at r2c2 the result jumps between refusal, residual
grip force and a clean release as the retreat goes 0-15 mm (`flights/r2c2_release.md`): that set-down stands on
the wrist-pitch stop, and no retreat makes it robust.

## Before and after (flights, red_cube)

Before = `main` (76ce886) at the October mount: Reachy-Lab `checkpoint-2026-10-18-issue192-stopA2/` and
`fly_probe.py` on `main`'s values (`flights/before_lift_r3c2_main_values.json`).  After = this branch, caps as set (`flights/after.md`, each row 5 flights:
nominal and the cube 3 mm off in x or y).

| Pick → place | Before | After |
|---|---|---|
| lift at r2c2 | refused before motion: IK found no grasp pose | refused before motion: no reachable straddle with the palm clear |
| lift at r3c2 | needs a 7.15 mm hover correction, over the 6 mm cap (opened to measure: the palm then hit the cube) | **5/5 every criterion**, pads only |
| r2c2 → r2c1 (#174) | refused before motion: finger pad route -5.83 mm | refused before motion (no pick at r2c2) |
| r3c2 → r2c1 | refused at the hover: 7.13 mm > 6 mm | 5/5 abort cleanly: no set-down pose for the measured hold (wrist-pitch stop); cube put back on r3c2 |
| r3c2 → r2c2 | refused at the hover: 7.13 mm > 6 mm | 5/5 placed 6.2-6.5 mm from the centre, withdrawn, at PRESENT; `clean_placement` false (0.27 N grip force left after opening) |
| r3c2 → r3c3 | refused at the hover: 7.13 mm > 6 mm | **5/5 every criterion**, 7.1-7.9 mm from the centre, pads only |
| r3c2 → r3c3, stop during the carry | — | cube set back down on r3c2, withdrawn, at PRESENT |

Every after flight: the only robot geoms that touched the cube were the two pads; nothing else on the board moved.

## Live check (panel, 2026-10-19)

One run, "move the red cube to r3c3" with red_cube placed on r3c2: **completed, every criterion true**, 4.7 mm
from the centre, tilt 0.0 deg, nothing else moved, slip 0.02 mm, hover correction 7.55 / 8.99 mm; then "put your
arm away" to REST.  Image not rebuilt (src/, web/, tools/ are mounted).  Record: Reachy-Lab
`outputs/sim/working/trial-2026-10-19-issue195-live/`.

## Limitations

- **One origin, one clean destination.**  The crane takes the 6 cm cube from r3c2 only, and sets it down cleanly
  on r3c3 only.  The posture is narrow: 0.8 deg from the arm-yaw stop at the grasp, the finger pad's centre
  1.2 mm below the cube's top, the r2c1/r2c2 set-downs on the wrist-pitch stop.
- **A cube turned 5 deg either way is refused before motion** (`flights/chosen_caps_open.md`, yaw +/-5).
- **foam_block and pool_box_1 plan nowhere** (`sweep_after.md`).  The before sweep planned pool_box_1 from r3c2
  without the palm in the model.
- The prediction tolerances (`PREDICTION_TOLERANCE_M`) and the release drag study are still the August recordings'.

## Files

`fly_probe.py` (one flight, contact watcher), `fly_batch.py` (parallel flights), `flight_table.py` (tables),
`flights/` (jobs, raw JSONL, tables), `search/` (grid outputs), `sweep_after.{md,json}` (`python -m tools.place_sweep`).
