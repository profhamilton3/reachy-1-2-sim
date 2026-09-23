# Analysis — B2 s2 `finger` plan-vs-realised discrepancy: root cause (2026-09-22)

Answers `outputs/handoff-2026-09-22-b2s2-finger-clearance-investigation.md`. This was offline and
read-only. There were no code edits, services, motion or reruns, and no margin, threshold, gate or
guard changes. Reproduction scripts and the full corrected table are in
`outputs/analysis-2026-09-22-b2s2-finger-clearance/`. They import only the pinned worktree
`~/reachy-1-2-sim-stage2` @ `4ad0be7` (clean) and read the evidence dirs.

## Verdict

**The realised value is correct. The planned value is wrong.** It is wrong because of a documented
but unsafe *analysis* policy, not because of a geometry defect.

- **Worst sample:** `S2-B2-a-r4-flight` idx 136 at 3.543 cm. `a-r1` idx 132 (3.545) is effectively
  tied. This is the arrival at `REST_SHUT` with the gripper **SHUT (+19.9°)**, just before the hand
  opens.
- **Corrected LOWER_TO_REST `finger` planned:** **3.37 cm**, not 9.41.
- **Corrected Δ:** **−0.17 cm**, not +5.87. The realised path is slightly *farther* than planned.
- **Cause:** the planned side replaces each leg's aperture with whichever endpoint gives the larger
  `hand_radius`. That is OPEN (−45°) on both LOWER_TO_REST legs. For the tube model that choice is
  conservative. For the shells `finger` capsule it is not. Aperture swings the finger about the
  thumb's X axis, and the finger hangs lowest (closest to an object below) when SHUT.
  - At the `REST_SHUT` pose, `finger` clearance to `foam_block` is 9.41 cm at −45° (OPEN),
    8.06 at −20°, 5.53 at 0°, 4.66 at +7.5° and **3.37 at +20° (SHUT)**.
  - LOWER_TO_REST is the only `FOOTPRINT_LEGS` route where no leg is ever evaluated SHUT at
    `REST_SHUT`. PLACE_ROUTE, RAISE_TO_SIDE, STOW_ROUTE and STOW_FROM_SIDE each include a leg whose
    two endpoints are both SHUT, so their planned minimum is right by accident (3.37 on B2).

The same rule appears in three places:

- `scripts/measure_route_clearance.py:625` (`planned_clearance`), described by
  `PLANNED_APERTURE_POLICY`.
- `control/tools/s2_clearance_by_link.py:60` (`planned_iter`) in each of the B4/B1/B2 evidence
  copies.
- `control/tools/b2_planned_clearance_offline.py:43`.

Only `finger` is affected: `thumb`, `thumb_pad` and `wrist_ball` ride the thumb frame and do not
depend on aperture.

## 1. Worst sample, independent recomputation

`indep.py` imports nothing from the repo. It has its own rotation matrices, its own FK chain
(shoulder → elbow → wrist → pitch-only frame → thumb origin → finger hinge → rotx(gripper)) and
constants copied from `kinematics.py`. It computes its own capsule-to-box distance by sampling the
capsule axis at 20 001 points against the box SDF of `foam_block` (0.07×0.07×0.05 at (0.5842,
−0.1524, 0.765), axis-aligned).

| recording | idx | t (s) | wall_time_ns | independent | tool | planned (as reported) | planned (corrected) |
|---|---|---|---|---|---|---|---|
| **S2-B2-a-r4-flight** (`…151239Z`) | 136 | 6.805 | 573512648177958 | **3.543** | 3.54 | 9.41 | 3.37 |
| S2-B2-a-r1-flight (`…143002Z`) | 132 | 6.601 | 570955415554333 | 3.545 | 3.54 | 9.41 | 3.37 |

Full joint vector at a-r4 idx 136 (deg): shoulder_pitch −39.67, shoulder_roll −10.15, arm_yaw
−0.15, elbow_pitch −45.68, forearm_yaw 0.00, wrist_pitch −9.58, **wrist_roll 30.03**, **gripper
+19.92**. This is `REST_SHUT` (commanded −40/−10/0/−45/0/−10/+30/+20) to within 0.5°.

- The independent FK at the commanded `REST_SHUT` pose with SHUT gives 3.372 cm. That matches the
  repo's `link_capsules` + `SceneModel.clearances` (3.37), so the geometry is not in question.
- All six LOWER_TO_REST worst samples fall at idx 132–141 with the gripper at +19.1…+20.0 and
  3.54–3.64 cm, which matches the tool's range exactly.
- The a-r1 trajectory is PRESENT (idx ≤ 62, gripper −45, roll 0) → descent (idx 100: gripper
  −2.9, roll +20) → REST_SHUT reached SHUT (idx 131–133) → opening (idx 150: +8.0°) → REST OPEN
  (idx 192: −44.9). Clearance recovers to 9.16 in the parked tail.
- Wrist roll is +30 at both the worst sample and the planned REST_SHUT pose, so it is not a
  factor. The sign convention is handled correctly: SHUT = +20 is narrower (`hand_radius` 7.3 cm
  vs 12.1 cm open), and that is exactly why the max-radius rule picks OPEN.

## 2. Check order from the handoff (§5), results

1. **Comparable poses.** For LOWER_TO_REST, yes: same waypoints (PRESENT → REST_SHUT → REST), and
   the moving window [63, 192] = 3.16–9.60 s spans exactly that flight. Not comparable for
   **PLACE_ROUTE / RAISE_TO_SIDE setups**, which is a second, independent mechanism (§4).
2. **Aperture policy: root cause** (above). The planned side holds the finger OPEN along a leg
   whose commanded aperture runs −45 → +20, then OPEN again along a leg whose aperture runs
   +20 → −45. The SHUT state at REST_SHUT that the arm actually passes through is never evaluated.
3. **Timing.** `t` and `wall_time_ns` (monotonic) agree to 0.2 ms over the whole a-r1 recording at
   20 Hz. Across all 49 recordings, the moving-window restriction hides at most 0.10 cm on any link
   (`c-r6-flight` `thumb_pad` 2.07 vs 2.17). The window is not a factor.
4. **Segment selection.** The planned LOWER_TO_REST covers the same two legs the flight
   traverses. The LIFT/LOWER asymmetry is fully explained by aperture: LIFT_TO_PRESENT (REST →
   PRESENT) holds −45 at both ends and the gripper never moves, so policy and reality coincide
   (9.41 vs 9.20).
5. **Geometry.** Not implicated: the independent FK matches the repo's to <0.01 cm.

## 3. Blast radius

Scope of the effect:

- **Affected:** the *planned* column for the shells `finger` link on LOWER_TO_REST, on every board.
  It is inflated by the difference between the OPEN and SHUT finger positions at REST_SHUT.
- **Not affected:** every other link and route, and the closest-link (min-over-links) planned
  value on any board. On B2 the route's closest link is `thumb_pad` at 2.11, both before and after
  correction.
- **Not affected, live guard:** `web/panel_executor.py` gates on `CartesianPlanner.path_clearances`
  with the default **tube** model. Under tube, the max-radius endpoint rule is conservative: tube
  at that aperture ≤ tube at the interpolated aperture ≤ shells minimum, on all six
  `FOOTPRINT_LEGS` routes on B2 (verified). No gate, guard or session outcome depended on the
  inflated number.

Corrected LOWER_TO_REST `finger` (Δ = planned − worst realised, cm):

| session | object | as reported planned / Δ | corrected planned / Δ |
|---|---|---|---|
| B4 s1 | pool_box_1 | 11.40 / +0.58 | 10.75 / **−0.07** |
| B1 s1 | foam_block | 9.42 / +0.76 | 8.63 / **−0.03** |
| B1 s1 | soda_can | 44.86 / +1.52 | 43.28 / **−0.06** |
| B2 s2 | foam_block | 9.41 / +5.87 | 3.37 / **−0.17** |
| B2 s1 | — | no flights (stopped at reset 1) | — |

The effect was present in every session. It grows as the object moves closer, because the
OPEN-vs-SHUT finger swing matters more the nearer the object is. That is why it first stood out on
B2.

## 4. Correction to the handoff's §4: B1's +11.89 is a *different* artefact

The handoff groups B1 `soda_can` `finger` +11.89 with the B2 finding. **That link is wrong.** The
aperture correction leaves it unchanged (43.28 → 43.28). Its cause is **route-scope truncation**:

- `FOOTPRINT_LEGS["PLACE_ROUTE"]` and `["RAISE_TO_SIDE"]` are (HOVER, REST_SHUT, REST).
- The setup recordings capture the whole flight from HOME: GRIP_SHUT, BACK, CURL, CURL_HIGH, TUCK,
  SWING_1–3, HOVER, REST_SHUT, REST, and for RAISE_TO_SIDE also PRESENT.

B1 soda_can realised worst: `S2-B1-c-r1-setup` idx 500, t = 25.0 s, 31.38 cm, 4.0° (joint norm)
from **SWING_2**. The full-route planned value from HOME, with interpolated aperture, is **31.66**
at the leg ending at SWING_2, so Δ = **+0.28**. The same mechanism explains the `upper_arm` rows on
RAISE_TO_SIDE setup, where the realised worst lies on the REST → PRESENT leg that the footprint
omits:

| session | object | route/role | link | as reported planned / Δ | full-route planned / Δ |
|---|---|---|---|---|---|
| B1 s1 | soda_can | PLACE_ROUTE setup | finger | 43.28 / +11.89 | 31.66 / **+0.28** |
| B1 s1 | soda_can | PLACE_ROUTE setup | thumb_pad | 40.93 / +10.70 | 30.61 / **+0.38** |
| B1 s1 | foam_block | RAISE_TO_SIDE setup | upper_arm | 19.04 / +2.09 | 17.08 / **+0.12** |
| B4 s1 | pool_box_1 | RAISE_TO_SIDE setup | upper_arm | 20.89 / +1.72 | 19.26 / **+0.09** |
| B2 s2 | foam_block | RAISE_TO_SIDE setup | upper_arm | 34.05 / +3.13 | 31.00 / **+0.09** |

The handoff did not list the B2 s2 `upper_arm` +3.13 row. `REPORT.md`'s claim of "largest gap in
the campaign" is wrong in two ways: B1's +11.89 was larger, and neither number is a real
plan-vs-realised gap. `REPORT.md` was left as written.

**Campaign comparison after both corrections** (shells, non-parked, every link, every board
object; full table in `corrected.json`):

| session | max Δ as reported | max Δ corrected | min Δ corrected |
|---|---|---|---|
| B4 s1 | +1.72 (upper_arm) | **+1.10** (wrist_ball, PLACE_ROUTE flight) | −0.25 |
| B1 s1 | +11.89 (finger, soda_can) | **+0.98** (wrist_ball, PLACE_ROUTE setup) | −0.24 |
| B2 s2 | +5.87 (finger) | **+0.63** (thumb_pad, PLACE_ROUTE flight) | −0.17 |

Once both artefacts are removed, every per-link offset in the campaign is in [−0.25, +1.10] cm.
The largest remaining offsets are genuine realised-side deviations: `wrist_ball` near HOVER, where
the arm is about 3.7° off the commanded pose. They were not investigated further here.

## 5. Reconciliation 1: the "0.98 cm" in the B2 plan

`e1-stage2-b2-execution-plan-2026-09-21.md:35` and `:198` say the largest offset "on any link in
either session [B4/B1]" is 0.98 cm. Filters tested against the B1/B4 aggregates:

| filter | max Δ |
|---|---|
| any row, B1+B4 | +11.89 (B1 soda_can finger) |
| closest object only, B1+B4 | +2.09 (B1 upper_arm) |
| closest object, hand links only, B1+B4 | +1.10 (B4 wrist_ball) |
| **closest object, hand links only, B1 only** | **+0.98** (B1 wrist_ball PLACE_ROUTE setup) |

Only the last filter reproduces 0.98. It is the B1 REPORT's "next-closest shells links" line
(`clearance_by_link.md:50`). So the 1.1 cm cap was calibrated against **B1 only, `foam_block`
only, hand links only**. That excludes B4 entirely (wrist_ball +1.10, upper_arm +1.72), B1
upper_arm (+2.09) and B1 soda_can (+11.89).

With hindsight, 0.98 happens to equal B1's true corrected maximum. The campaign-wide corrected
maximum, though, is **+1.10 (B4)**, which is exactly the cap. The cap therefore had **zero
headroom** over an offset the campaign had already observed. B2 s2 came in at +0.63, so nothing
tripped. This is flagged for the owner. Per the handoff, it does not change any threshold.

## 6. Reconciliation 2: how set 1 passed a 1.1 cm cap with +5.87 "present"

Confirmed from `control/checkpoint_1.txt` (mtime 14:38Z) and the r1 recorder log and sidecar:

- The recorder's report and `server_side_clearance` give, per object and hand, the **minimum over
  links** only. They contain no per-link data: `grep -c finger` returns 0 on both. Item 8 lists
  one number per recording (e.g. LOWER_TO_REST 1.75, planned 2.11, +0.36). The "closest link" in
  8b is therefore min-over-links planned minus min-over-links realised. That is `thumb_pad` on
  both sides.
- The per-link tool that surfaces `finger` ran afterwards (`clearance_by_link_all.json` mtime
  16:30Z).
- 8b was written as closest-link and executed as closest-link, which matched its authorization.
  **It could never have caught the `finger` number.** It also *should* not have: the +5.87 is an
  artefact of the planned side, and the corrected per-link offset is −0.17. A per-link cap using
  the uncorrected planned column would have stopped the session on a false alarm. On any link at
  REST_SHUT (or any other realised pose), the realised floor in 8a is the right safety quantity,
  and it did cover `finger` (3.54 ≫ 1.0).

## 7. Remaining uncertainty

- The corrected planned side assumes the gripper moves linearly and in step with the arm along
  each leg. The realised data fit this (a-r1 idx 100: halfway through the descent, gripper −2.9 of
  −45 → +20). On these boards the minimum is at the leg endpoint (SHUT at REST_SHUT), so the result
  does not depend on that assumption.
- The genuine `wrist_ball` offsets near HOVER (+0.98, +1.10) were left unexplored. They are the
  real campaign-wide plan-vs-realised maximum.
- Board YAML hashes used: B2 `4d7b72e0…` (matches the handoff), B1 `f577fc47…`, B4 `818d8e57…`,
  all read from the worktree at `4ad0be7`. Their per-session provenance was not re-checked
  against each session's manifest.

## 8. Evidence integrity

- B2 s2: `SHA256SUMS` 675/675 OK before and after this work. B4 s1: 607/607. B2 s1: 89/89.
- **B1 s1: 634/635.** `./control/shutdown.txt` fails. This predates this work: its mtime is
  2026-09-19 04:30:01, 45 s after `SHA256SUMS` (04:29:16). It ends with the "caffeinate … killed
  … after archive verification" line, which was appended after hashing. The verified archive
  (`a0d3ff22…`) holds the hashed version. The file was left untouched. Flagged here only.
- Nothing was written inside any evidence dir.

## 9. One recommended next action

The PR below needs its own authorization. It is offline and changes analysis code only, not the
guard.

Change `planned_clearance` in `scripts/measure_route_clearance.py`, and the matching per-link
tool, so that under the **shells** model:

- (a) the gripper is interpolated along each leg, the same way `joint_path` interpolates the arm.
  Tube keeps the conservative max-radius rule.
- (b) each recorded route is compared against the **full flown waypoint list**, not only its
  `FOOTPRINT_LEGS` entry.

The existing `planned_aperture_policy` string makes this a one-policy change. Add a regression
test that pins B2 LOWER_TO_REST `finger` = 3.37 cm and B1 soda_can PLACE_ROUTE `finger` full-route
= 31.66 cm.

**Cost:** about 1–2 h to implement and test, plus one review. No simulator time, and no change to
`FOOTPRINT_LEGS`, margins, the live tube guard or any checkpoint threshold.

**Separately, for the owner (not an action here):** the 1.1 cm cap's zero headroom against B4's
+1.10 (§5).
