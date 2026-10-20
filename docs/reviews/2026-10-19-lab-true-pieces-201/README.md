# Lab-true pieces and pad contact: evidence (#201)

**Date:** 2026-10-19. **Branch:** `feat/201-lab-true-pieces` from `main` @ `a84d4b5`. **Offline:** host macOS arm64, Python 3.14, MuJoCo 3.11, the offline world (`tools/place_sweep/fly.py`). **Live:** §8.

**Owner ruling, 2026-10-19.** No pad contact setting tried met the pool cylinder target with no change at all to the cubes or the crane (§4). The owner accepted the chosen setting with red_cube −1 and pool_box_1 −2, recorded under Future refinement (§9).

**Then the release (§7).** The first live cylinder run let go one ramp step in and left the pads on the cylinder. The release is now closed-loop, and with it pool_box_1 is back to 53 of 56 and the pool cylinder is at 54 of 56.

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

These flights use the MJCF as it stands on this branch, with the release as it was before §7. The re-fly after §7 is in §7.

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
| After §7: `test_crane_grasp_alignment` (2 new `release_to` tests), `test_crane_place`, `test_side_grasp` | 53 passed |
| After §7: `test_place_offline`, `test_side_place_offline` | 5 passed |
| After §7: `test_motion_worker` (2 hover pins re-set to the new sizes), `test_panel_crane_lift`, `test_panel_episodes`, `test_place_evaluator`, `test_panel_route_evaluators`, `test_panel_abilities` | 252 passed |
| The other test files that load the pool scene (23 files) | 1073 passed; 2 failed, which also fail on `main` a84d4b5 (`test_panel_routes.py`: explicit-command confirm and duplicate confirmation, 409 ≠ 200) |

## 7. Closed-loop release (after the first live cylinder run)

**Live run 2** (pool_cyl_1 r2c3 → r2c2, before this change) let go at −18.4°. That is one 2.5° ramp step from where the hand stood (−15.8°), and 10° short of where this cylinder clears (−27.3° by the pad model; −28° offline). The pads were left on it at 29.7 N, and the withdrawal was refused before opening. The old release stopped at the first grip-force reading at 0 N.

**Reproduced offline** (`release_lag_probe.py`, `release/lag_probe_*.jsonl`): the same pair, with the server's 50 Hz state sampling, observation and command delays, and a single zero grip reading at the start of the release.

| Case | Let go | Force after | Result |
|---|---|---|---|
| No lag | −28.1° | 0 N | clean |
| Sampled 50 Hz; 100 ms obs; 100+100 ms; 200 ms cmd; 200+200 ms | −28.8° to −39.0° | 0 N | all clean: lag makes the hand open *further* |
| **One zero reading** | **−17.4°** | **37.5 N** | **withdrawal refused before opening (the live failure)** |
| One zero reading + 100+100 ms | −20.4° | 16.8 N | the same |

**The fix** (`primitives.release_to`; `crane_pick_live`, so both the side-on and crane releases use it; ADR-0008 amended):
- **Absolute target.** Open to the angle whose free gap is the object's width across the jaw plus 4 mm, never wider than the plan's opening. For this cylinder that is −33.5°.
- **Settle.** The grip counts as let go only when fresh readings stay at or under 0.5 N for 0.2 s. A lone zero does not count.
- **Open further.** If it has not let go within 1 s, open 5° further, up to the plan's opening. Only past that does it refuse, with the force left.
- **Back-off after the let-go.** A placement's thumb-side back-off (#174) runs once the grip is free.
- **Why 0.5 N and not 0.** With the exact-zero rule, two cases stopped at −39° on a 0.03 N / 0.1 N trace force (`release/fixed/`). That is the thumb resting on the cylinder: it is the jaw that does not move as the hand opens, so only the back-off takes it off. `clean_placement` still asks 0 N after the back-off and the full opening.

**With the fix,** all 8 lag/dropout cases are clean, letting go at −33.5° (`release/fixed/`).

**Re-fly** (`flights/run_release.sh`, `out_release_*.jsonl`):

| | Before §7 | After §7 | #180 |
|---|---|---|---|
| Lifts, red_cube / pool_box_1 / pool_cyl_1 | 8 / 8 / 8 of 8 | 8 / 8 / 8 of 8 | 8 / 8 / 5 |
| pool_cyl_1 placements | 51 of 56 | **54 of 56 (96%)** | 26 of 56 |
| pool_box_1 placements | 51 of 56 | **53 of 56** | 53 of 56 |
| red_cube placements | 52 of 56 | 52 of 56 | 53 of 56 |
| Before/after; perturbations; neighbours | 10/10; 24/24; 8 clean, 2 refused | the same | the same |
| Crane lift r3c2 / place r3c2 → r3c3 (`crane_check_release.txt`) | clean, 4.93 / 3.30 mm | clean, 4.99 / 6.46 mm | clean |

- The release removed every "withdrawal refused before opening" in the batches: 3 for the cylinder and 2 for the box.
- The 2 cylinder failures left are r2c2 → r2c3 (no corrected pose at the destination: put back) and r2c3 → r1c3 (slip 0.14 mm).
- The drag cost is in §9.

## 8. Live (the panel)

To be filled in after the two live runs.

## 9. Future refinement

The owner accepted these on 2026-10-19 (no new issue):

- **red_cube r2c1 → r1c3 (−1).**
  - This pair's plan margin is 0.0 mm (limited by r_wrist_pitch).
  - Under the pad torsion, no level set-down pose is found from the measured hold, so the cube goes back on r2c1 and the hand returns to PRESENT. Nothing else moves.
  - A planner-side margin at r1c3 would be the place to recover it.
- **pool_box_1 (−2 net, 4 new failures, before §7).**
  - The 50.8 mm box leaves 1.6 mm of palm route clearance (9.3 mm at 40 mm) and 4.0 mm to the table (11.7 mm).
  - Two withdrawals were refused before opening, and two set-downs abort safely.
  - The same four fail with the pads at `condim` 3.
  - **After §7:** the two withdrawals are clean, so the box is at 53 of 56, the same count as #180. The 3 that remain are safe set-down aborts: r2c2 → r1c3, r3c3 → r1c3 and r3c3 → r2c2.
- **Release drag after §7.**
  - The thumb-side back-off now follows the let-go instead of running in step with it.
  - Median drag rose: red_cube 2.5 → 4.5 mm, pool_box_1 1.3 → 2.7 mm, pool_cyl_1 1.1 → 2.8 mm.
  - 7 red_cube placements are at 5.0–5.2 mm, over the reported 5 mm target. Drag is not a criterion (#173).
  - The worst placement error is 5.4 mm, inside the 10 mm bound.
  - Starting the back-off as the force falls, rather than after it settles, would be the place to recover it.
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
