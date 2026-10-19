# Side-on grasp: evidence (#180)

**Date:** 2026-10-19. **Branch:** `feat/180-side-on-grasp` from `main` @ `b4323b9`. **Offline only** so far: host macOS arm64, Python 3.14, MuJoCo 3.11, the host IK mirror (`tests/unit/host_ik_arm.py`), the offline world (`tools/place_sweep/fly.py`). No live run yet.

**Design:** ADR-0011 (`docs/adr/0011-side-on-grasp.md`). **Code:** `src/reachy_ai/tasks/side_grasp_live.py`, plus default-preserving generalisations in `crane_pick_live.py`, `crane_place_live.py` and `grasp_alignment.py`. The panel uses the planner choice (`web/motion_worker.py`).

## 1. The hand, level (`hand_geometry.py` → `hand_geometry.txt`)

- **Free gap.** It is 67.8 mm at the −65° opening and 69.0 mm at −68°. blue_cylinder (70 mm) cannot enter the jaw.
- **Below the pad centres:** the palm and the hand back reach 25 mm, and the forearm capsule 30 mm. That makes 34 mm the lowest jaw-gap height that keeps the forearm 4 mm off the table.
- **Along the approach:**
  - the palm's front is 19.7 mm behind the jaw-gap centre;
  - the thumb pad runs 29 mm ahead of it.

## 2. Siva's taught poses as seeds (`taught_seeds.py` → `taught_seeds.txt`, `.json`)

- All 9 taught tip poses solve exactly inside the sim's joint travel. Those joints are `TAUGHT_SEEDS`.
- **Approach azimuths:** the near row points left (76–82°), B1/B2 point diagonally (44–45°), and the far right points ahead (3–18°).
- **Tilt:** the taught hands are level within 5.5°. That bounds the 5° pitch option.

## 3. Reach of a level hand (`reach_level.py` → `reach_level.txt`)

- 8 of 9 cells reach, every 15° of azimuth.
- r3c1 reaches at no azimuth, pitched up to 10° down.
- r1c3 has a 1.4° joint margin level, 7.4° at 5° pitch.

## 4. The close (`close_study.py` → `search/close_study_*.jsonl`)

The first flight (`flights/lift_v1.jsonl`, red_cube r2c2) slid in, straddled, and closed on nothing: the cube was carried 48 mm forward. The study isolates the close, from an arm set on the grasp pose with the object placed relative to the realised jaw.

| Grid | Finding |
|---|---|
| `base` | At the −65° opening a pool cylinder tipped off the table in 4 of 6 fast closes. At −30°/−35° every pool box and cylinder close held. |
| `cube_depth`, `cube_height` | Neither insertion depth (8–16 mm) nor jaw height (29–41 mm) changed the cube's hold. |
| `cube_yaw` | 1.5° of yaw with a 4 mm thumb gap reproduced the flight's squeeze-out exactly. |
| `cube_thumb` | With a 1.0–2.0 mm thumb gap, 29 of 30 held. |
| `target` | The over-travel is the cause: −5° drove the finger 19° past contact (12 mm of soft-contact penetration, ~250 N). Contact + 15° (−35° for the cube) held with ≤3.4 mm drift. Contact + 4.5° dropped a cylinder. |
| `rule` | Contact + 15°, 1.5 mm thumb gap: cube 34 of 36 (2 creep failures off-design), pool box 6 of 6, pool cylinder 4 of 6. |
| `cyl_height`, `cyl_force`, `cyl_depth`, `cyl_com` | The cylinder's creep has no trend with height, grip force, depth or pitch. |

**Why the cylinder creeps.** Each pad touches it at ONE point (`contacts` dump). With MuJoCo's default `condim` 3 there is no torsional friction, and `multiccd` is off. So a held cylinder pivots about the pinch axis.

## 5. New values and their provenance (all in `side_grasp_live.SideSpec`)

| Value | | From |
|---|---|---|
| `grasp_height_m` | 34 mm | forearm 30 mm below the gap + 4 mm (§1); Siva's taught tip height |
| `standoff_rungs` | 8 | the thumb pad leads 29 mm; insertion checkpoint ≤ 6 |
| `hover_rise_rungs` | 2 | the ~9 mm sag put the predicted hand into the table at a hover at grasp height (`flights/lift_v2.jsonl`, r3c2) |
| `via_rungs` | ≤ 15 above the standoff, at least 3 | the near row reaches 6–13 cm |
| `pitches_deg` | (0, 5) | the taught tilt envelope; r1c3's margin |
| `side_slack_total_m` | 7.8 mm | the cube's slack at the crane's flown −65° |
| `thumb_side_m` | 1.5 mm | §4 `cube_thumb` |
| `close_past_contact_deg` | 15° | §4 `target`, `rule` |
| `release_retreat_m` | 4 mm | the lifts left pads touching without it (`lift_v1`); 9 mm cut a placed cube (`place_v1`) |
| `neighbour_clearance_m` | 10 mm | the uncorrected hover error at the October mount (5.3 mm sideways, 9.4 mm low; #195 flights/) |
| `palm_clearance_m` | 2 mm | the crane's (#195) |
| `min_pad_below_top_m` | 4 mm | the 4 cm pool box leaves 6 mm |
| `cylinder_azimuth_step_deg`, `max_bearing_offset_deg`, `max_full_plans` | 15°, 90°, 4 | reach_level spans; planning cost |

## 6. Offline sweep: preflight only (`sweep/`, `sweep_table.py` → `sweep/summary.md`)

`python -m tools.place_sweep --grasp best|crane [--lift]`, FWDCenterLabSivaPool, the object alone on the board.

| Object | Lift origins (best) | Placement pairs (best) | Crane only (main's planner) |
|---|---|---|---|
| red_cube | 8 of 9, all side-on | 56 of 72, all side-on | 3 of 72 (r3c2 → r2c1, r2c2, r3c3) |
| pool_box_1 | 8 of 9 | 56 of 72 | 0 of 72 |
| pool_cyl_1 | 8 of 9 | 56 of 72 | — (the crane refuses cylinders) |
| blue_cylinder | 0: opening (70.0 mm > 67.8 mm) | — | — |
| foam_block, soda_can | 0: shape | — | — |

- **Every refusal of a cube, box or cylinder is r3c1**, as origin or destination: no level pose within the travel (§3).
- **Where both grasps plan** (red_cube from r3c2), side-on has the larger margin: about 1.0 mm. The crane's is 0.0: it stands on the arm-yaw stop.

## 7. Offline flights: physics (`fly_batch.py`, `flights/out_*.jsonl`, `flight_table.py` → `flights/summary.md`)

Judged on the #174 criteria: placed on target (D6), retained height, slip ≤ 0.1 mm, clean placement, clean withdrawal, back at PRESENT.

| | red_cube | pool_box_1 | pool_cyl_1 |
|---|---|---|---|
| Lifts, 9 origins | 8 ✓, r3c1 refused | 8 ✓, r3c1 refused | 5 ✓; r2c1 halt (object moved), r3c2 dropped, r3c3 not clean; r3c1 refused |
| Placements, 56 flown | 53 ✓ | 53 ✓ | 26 ✓ |

**red_cube, the 3 that were not clean:**
- **r2c2 → r1c3:** aborted cleanly after the hold. No level set-down pose existed for the measured hold, so the cube was put back on r2c2 and the hand withdrew to PRESENT.
- **r3c3 → r1c3 and r3c3 → r2c3:** halted at slide-in rung +1. The predicted forearm was at −0.0 mm against the table; the hand retreated to PRESENT.

**pool_box_1, the 3 that were not clean:**
- **r3c3 → r1c3 and r3c3 → r2c3:** aborted cleanly after the hold. The re-plan from the measured hold found no level set-down pose (r1c3), or one with the forearm 1.5 mm into the table (r2c3). The box was put back and the hand withdrew.
- **r2c1 → r1c1:** placed, then the withdrawal was refused before opening: no clear way out with the grip let go. The hand was left beside the box, not opened.

None of the three touched anything else.

**pool_cyl_1:** most failures are the cylinder pivoting in the grip (§4):
- 8 lost the grip carrying it back;
- 4 were dropped at the lift;
- 10 were aborted off target;
- several ended tumbled 90–172°.

The clean count depends on the origin:

| Origin | r2c3 | r1c1 | r2c2 | r1c2 | r1c3 | r3c2 | r2c1 |
|---|---|---|---|---|---|---|---|
| Clean placements | 7 of 7 | 6 of 7 | 6 of 7 | 3 of 7 | 1 of 7 | 1 of 6 | 0 of 7 |

**Before / after against PR #197's pair, red_cube r3c2 → r3c3**, nominal plus 3 mm offsets in x and y:

| | Before: crane | After: planner choice |
|---|---|---|
| Clean | 5 of 5 | 5 of 5 (side-on chosen) |
| From the centre | 7.2–7.9 mm | 2.5–2.9 mm |
| Release drag | 7.1–7.9 mm | 2.1–2.6 mm |

**Perturbations**, red_cube with ±3 mm offsets and ±5° yaw (the crane refuses 5°), over r2c2 → r1c1, r3c2 → r3c3, r1c3 → r2c1 and the r2c2 lift: 24 of 24 clean.

**Neighbours**, red_cube from r2c2:

| Neighbours | Result |
|---|---|
| pool_box on r2c3 | approach 0°, clean |
| pool_box on r1c2 | approach turns to 90°, clean |
| both | refused: "via->hover: hand would come within −1.1 cm of pool_box_2" |
| pool_cyl on r1c2, r2c2 → r1c1 | refused: forearm −2.8 cm into pool_cyl_2 on the set-down |

In every flown neighbour case, other objects moved ≤ 0.0001 mm.

## 8. The crane, unchanged (`crane_identity.py` → `crane_identity.txt`)

The crane lift at r3c2 and the crane placement r3c2 → r3c3, on `origin/main` and on this branch, hash identically: the outcome, every event and the final object pose. `tests/integration/test_place_offline.py`: 3 passed.

## 9. Tests run (focused; no full suite)

| Tests | Result |
|---|---|
| `tests/unit/test_side_grasp.py` (new) | 25 passed |
| `tests/integration/test_side_place_offline.py` (new) | 2 passed |
| `test_crane_place.py`, `test_crane_grasp_alignment.py`, with the two new files | 53 passed |
| `tests/integration/test_place_offline.py` | 3 passed |
| `test_motion_worker`, `test_panel_abilities`, `test_panel_crane_lift`, `test_place_evaluator`, `test_panel_route_evaluators` | 223 passed |
| `test_panel_episodes`, `test_panel_recipes` | 59 passed |

## 10. Live (the panel, 2026-10-19)

- **Setup:** Docker `reachy-1-2-sim:latest` plus the native server, FWDCenterLabSivaPool, launched with `SivaLabLauncher.sh` from this branch at 965f37b. The image was not rebuilt: the branch changes only src/, web/, tools/, docs/ and tests/.
- **Driving:** the panel's own HTTP routes.
- **Record:** Reachy-Lab `outputs/sim/working/trial-2026-10-19-issue180-live/`.

| Run | Grasp | From centre | Tilt | Slip in hand | Hand's own motion in the hold | Drag | #174 criteria |
|---|---|---|---|---|---|---|---|
| 1: red_cube r2c2 → r1c1 | side-on (crane: no grasp pose) | 1.7 mm | 0.0° | **0.102 mm** | 0.069 mm | 1.6 mm | 5 of 6: **slip false** (bound 0.1 mm) |
| 1 repeat (owner's one-time exception; cube reset with `place_object`) | side-on | 1.7 mm | 0.0° | 0.065 mm | 0.069 mm | 1.5 mm | 6 of 6 |
| 2: pool_cyl_1 r2c3 → r2c2 | side-on (crane: refuses cylinders) | 3.2 mm | 0.0° | 0.045 mm | 0.0 mm | 3.0 mm | 6 of 6 |
| "put your arm away" | — | — | — | — | — | — | REST; "the board is as it was" |

- **Nothing else moved** in any run.
- **Offline predictions** for the same pairs: run 1, 2.4 mm from the centre with 0.004 mm slip; run 2, 4.6 mm.
- **The slip reading.** The two live red_cube holds read the commanded-still hand moving 0.069 mm; offline it reads 0. The live slip readings (0.065 and 0.102 mm) are the same size, so the slip criterion sits at the live measurement's noise floor. The bound was not changed.
