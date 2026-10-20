# Lab-true pieces and pad contact: evidence (#201)

**Date:** 2026-10-19. **Branch:** `feat/201-lab-true-pieces` from `main` @ `a84d4b5`. **Offline:** host macOS arm64, Python 3.14, MuJoCo 3.11, the offline world (`tools/place_sweep/fly.py`). **Live:** §7.

**Owner ruling, 2026-10-19.** No pad contact setting tried met the pool cylinder target with no change at all to the cubes or the crane (§4). The owner accepted the chosen setting with red_cube −1 and pool_box_1 −2, recorded under Future refinement (§8).

## 1. Sizes (Siva, 2026-10-19)

- pool_box_1–5 are now 50.8 mm cubes.
- pool_cyl_1–5 are now 45.7 mm across and 71.1 mm tall.
- Provenance: `docs/scene-object-dimensions.md`.
- Mass and friction are unchanged. The four named objects are unchanged.

## 2. Planning sweep (`sweep/`, `step1.sh`): no change in what plans

| | #180 | #201 |
|---|---|---|
| Lift origins (best grasp), red_cube / pool_box_1 / pool_cyl_1 | 8 / 8 / 8 of 9 | 8 / 8 / 8 of 9 (all side-on, r3c1 refused) |
| Placement pairs (best grasp), each | 56 of 72 | 56 of 72 (the same pairs) |
| Crane only, pool_box_1 | 0 placements, 0 lifts | 3 placements (r3c2 → r2c1, r2c2, r3c3), 1 lift (r3c2) |

## 3. Physics flights: chosen candidate = pads `condim` 4, priority 1 (torsion 0.005 m)

These flights use the MJCF as it stands on this branch.

| | #180 (pads `condim` 3, old sizes) | #201 |
|---|---|---|
| pool_cyl_1 lifts | 5 of 8 | **8 of 8** |
| pool_cyl_1 placements | 26 of 56 | **51 of 56 (91%)** |
| pool_box_1 lifts | 8 of 8 | 8 of 8 |
| pool_box_1 placements | 53 of 56 | **51 of 56** |
| red_cube lifts | 8 of 8 | 8 of 8 |
| red_cube placements | 53 of 56 | **52 of 56** |
| Before/after (red_cube r3c2 → r3c3, ±3 mm) | 5/5 + 5/5 | 5/5 + 5/5 |
| Perturbations (red_cube) | 24 of 24 | 24 of 24 |
| Neighbours | 2 clean, 2 refused | the same |
| Crane lift r3c2 / place r3c2 → r3c3 (`crane_check.txt`) | clean, 4.05 / 7.70 mm | clean, 4.93 / 3.30 mm |
| `test_place_offline.py` | 3 passed | 3 passed |

**Size against contact.** Pool cylinder placements at the new size with the pads at `condim` 3 are 8 of 56 (`out_base_place_pool_cyl_1.jsonl`). The torsion is what brings the cylinder to 51.

## 4. The cube changes, and the candidates

**pool_box_1 (−2 net):** 4 new failures, 2 new passes.
- r1c2 → r1c1 and r2c2 → r3c3: withdrawal refused before opening.
- r2c2 → r1c3 and r3c3 → r2c2: no set-down pose after the hold, so the box was put back.
- All 4 fail the same way with the pads at `condim` 3 (`out_cube_regressions.jsonl`). The cause is the bigger box: palm route clearance drops from 9.3 to 1.6 mm, and table clearance from 11.7 to 4.0 mm. No pad setting removes that. `condim` 6 + priority 1 rescued 2 of the 4, but it fails the crane.

**red_cube (−1):** r2c1 → r1c3 now aborts safely.
- No level set-down pose was found from the measured hold, so the cube was put back on r2c1 and the hand returned to PRESENT.
- The plan margin for this pair is 0.0 mm (limited by r_wrist_pitch).
- With ±1 mm offsets: `condim` 3 places 3 of 4; the chosen setting places 0 of 4. This is a real, if narrow, contact effect.

**Candidates that cleared the cylinder lift screen** (cylinder lift cells screened; `out_screen*`, `out_torsion_scan.jsonl`, `crane_release_probe.jsonl`):

| Pads | Cylinder lifts | red_cube r2c1 → r1c3 | Crane release (r3c2) |
|---|---|---|---|
| `condim` 3 (main) | 2 of 8 | ✓ | ✓ |
| `condim` 4, larger-of-two (0.01 m) | 8 of 8 | — | ✗ 0.22 N left on |
| `condim` 6, larger-of-two | 7 of 8 | — | — |
| `condim` 6, priority 1 | — | ✓ | ✗ |
| `condim` 4, priority 1, 0.001 m | 2 of 3 | — | ✓ |
| 0.002 m | 2 of 3 | ✗ | ✓ |
| 0.003 m | 3 of 3 | ✗ | ✗ 0.18 N |
| 0.004 m | 3 of 3 | ✓ | ✗ 0.20 N |
| **0.005 m (chosen)** | **8 of 8** | ✗ | ✓ |
| 0.0075 m | 3 of 3 | ✗ | ✓ |

- The red_cube pair and the crane release flip without a trend as the torsion changes. Both sit on a margin: the pair at 0.0 mm, the release on pad contact after opening.
- Turning `multiccd` off (MuJoCo 3.11 has it on by default, and #180's flights had it on) leaves one point per pad, and the cylinder dropped at the lift.

## 5. Board margins: deferred

The open questions:
- Does the far margin end at the white board's edge (3 in) or at the table frame (4.8 in)?
- Which line does the 4 in arrow start from?
- The robot-side and right margins are not annotated.

## 6. Tests run (focused)

| Tests | Result |
|---|---|
| The size-affected tests (E1 boards, footprint, placement) | 22 passed |
| `test_mujoco_model`, `test_gripper`, `test_contact_model_noslip` | 62 passed |
| `tests/integration/test_place_offline.py` | 3 passed |
| `test_side_grasp`, `test_crane_place`, `test_crane_grasp_alignment`, `test_side_place_offline` | 53 passed |

## 7. Live (the panel)

To be filled in after the two live runs.

## 8. Future refinement

The owner accepted these on 2026-10-19 (no new issue):

- **red_cube r2c1 → r1c3 (−1).**
  - This pair's plan margin is 0.0 mm (limited by r_wrist_pitch).
  - Under the pad torsion, no level set-down pose is found from the measured hold, so the cube goes back on r2c1 and the hand returns to PRESENT. Nothing else moves.
  - A planner-side margin at r1c3 would be the place to recover it.
- **pool_box_1 (−2 net, 4 new failures).**
  - The 50.8 mm box leaves 1.6 mm of palm route clearance (9.3 mm at 40 mm) and 4.0 mm to the table (11.7 mm).
  - Two withdrawals are refused before opening, and two set-downs abort safely.
  - The same four fail with the pads at `condim` 3.
- **Board margins: deferred.**
  - Does the far margin end at the white board's edge (3 in) or at the table frame (4.8 in)?
  - Which line does the 4 in arrow start from?
  - The robot-side margin (the 0.16 m placeholder) and the right margin are not annotated.
  - The grid is unchanged.
- **ADR-0011 correction: `multiccd` was ON, not off.**
  - ADR-0011 and #180's evidence README say that a held cylinder pivots with "`condim` 3 … and `multiccd` is off".
  - In MuJoCo 3.11, `multiccd` is a disable flag (`mjDSBL_MULTICCD`) and is on by default, so #180's flights had it on.
  - Each pad touched the cylinder at 3–5 points (thumb) and 1–3 points (finger) (`contact_probe_cyl_r2c3.jsonl`). The cylinder still pivoted, because nothing resisted torsion. The fix is `condim` 4 on the pads (GRIPPER_MODEL.md).
- **Not flown:** the sweep now plans the crane for pool_box_1 at r3c2: 1 lift and 3 placements, where #180 had 0. Best grasp picks side-on for all of them.
