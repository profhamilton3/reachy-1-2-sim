# ADR-0009: Placing an object on another cell

- Status: Accepted. Covers #174. Simulator only.
- Date: 2026-10-12
- Relates to:
  - #174;
  - #172 (ADR-0007: the evaluators, the planned route type, the offline world and the contact rules);
  - #173 (ADR-0008: the release check and the withdrawal);
  - #55 (the crane lift).

## Context

**Before #174 the crane could only put an object back where it found it.** The panel's "pick and place" wording ran the older open-loop arc (`pick_place_live`). That arc:
- did not check that it had grasped anything;
- was not recorded;
- was seen on 2026-10-09 missing a cube and a cylinder, then releasing anyway.

**D6, what a place must achieve:**
- the object's centre within 1 cm (horizontally) of the target cell's centre;
- tilted at most 5°;
- nothing else moved more than 1 mm;
- no finger on it after the release.

The withdrawal is reported separately.

Everything below was established offline before any flight. The evidence is in Reachy-Lab `outputs/sim/working/`:
- `issue174-drag-study-2026-10-12.md`;
- `issue174-sweep-2026-10-12.md`;
- `issue174-exit-and-attitude-2026-10-12.md`;
- `issue174-offline-2026-10-12/` (scripts and data).

### The release drag

- **Mechanism.** At the hold, the soft contact under the gripper's force limit leaves the pads deep in the cube: 3–6 mm for the thumb, 11–13 mm for the finger. Opening unloads that into the cube, toward the finger.
  - Under the no-slip pass (ADR-0005) the pads also carry the cube while they unload.
  - With no-slip off (diagnostic only), the drag is 4.6 mm mean.
- **Strategies compared.** On 20 recorded releases, each rebuilt offline and withdrawn by #173's way out:
  - opening speed;
  - lifting or pressing while letting go;
  - a retreat along the jaw.
- **Result.** Backing the hand 9 mm away on the thumb's side while it lets go gave the least drag: 2.9 mm mean, 5.0 mm max, against 9.7 / 12.3 mm. #173's 0 refusals were kept.
- **The lift keeps #173's release.** The close pushes the cube 7–9 mm toward the thumb, and the lift's put-back relies on the drag undoing that. With the retreat, 16 of 20 put-backs would exceed 5 mm.

### Reach

**The crane's only grasp is the recorded x-face plan** (tilt 30°, lean 59.52°). It plans only at r2c2.
- A held cube stands upright only if the hand keeps its grasp attitude, turned about the vertical.
- With the lift's attitude, no arm pose sets the cube down on r2c1: a bounded global search gets no closer than 20 mm.

**The place has its own attitude: x faces, tilt 63°, lean 65°** (`crane_place_live.PLACE_GRASP`).
- Of the 1,224-attitude grid plus a refinement, the ones that pass every pick, carry, set-down and exit check for r2c2 → r2c1 lie at tilt 54–66° and lean 65–70°.
- Flown offline, lean 65° at tilt 60–66° placed every start 3.8–5.3 mm from the centre.
- 63° is the middle of that band:
  - 6.8° from a joint stop at the pick (arm yaw);
  - 0.9° at the set-down (shoulder roll).
- Search hygiene: the global search is held to the arm's forward branch. Unconstrained, every "reachable" pose was the arm thrown back over the top.

### Exits

A placement ends at PRESENT. From there the arm has two measured exits:
- LOWER_TO_REST, to REST;
- STOW_FROM_SIDE, to HOME.

Both drop over the near board. With the cube on r2c3 or r3c3, the executor's footprint gate refuses both exits (−5.3 and −8.4 cm), although the offline flight clears it by 4–5 cm. Every other reachable cell leaves both exits open.

## Decision

1. **A `place_on_cell` ability, route `CRANE_PLACE` (planned).** It is `execute_crane_pick(..., place=PlacePlan)`, with the crane module unchanged up to the hold. Then:
   - **The destination is re-planned from the MEASURED hold.** The object's pose in the thumb frame sets the destination ladder, and the whole destination is re-checked. If that is refused, the object is put back where it was.
   - **The carry runs at the hover height.** The pickup ladder is climbed to +9 cm, with a joint line to the same rung of the destination ladder.
     - It is checked with the held object in it: the held box against the table, the rails and every other object (`carried_clearance`), and against the upper arm and forearm (`carried_arm_clearance`).
     - It is also checked by the whole-arm and table checks.
     - While carrying, losing the grip, the time limit or a stop halts the carry. Where the object sits in the hand is measured at every stop (2 mm halts).
   - **The set-down is corrected toward the cell centre.** The held object's measured position is corrected at the carry height and again 2 cm up: to within 1 mm, at most 3 times, 15 mm in all. Every correction is re-solved and re-checked.
   - **The release backs the hand 9 mm off on the thumb's side while it lets go.** It is checked first and skipped if not clear. #173's check before opening and its withdrawal follow.
2. **The abort path.** Every commanded pose after the hold is on a trail. A halt during the carry or the set-down flies the trail back in reverse, to the pickup ladder, and sets the object down where it was picked up.
   - The way home then finishes whatever is asked.
   - **The hand never opens in the air.** It opens only once the object is measured to rest on the table, and stale feedback is never support. If no support is reached at the destination, the object is carried back. If none is reached at the pickup either, the arm holds it and stops.
3. **The planning rules** (`check_place`), refused before motion with the stage and the reason:
   - the destination cell must be free;
   - at least one exit from PRESENT must stay open with the object where it will stand (stage `exit`);
   - plus the checks above.
   - **Headings.** The bearing change is tried first, then every 15°. A refusal reports the bearing change's reason.
4. **`evaluate_place` (D6), in `panel_routes`.**
   - The four D6 clauses, plus the rule table from ATTEMPT_START to the release, plus plan integrity.
   - The withdrawal and the full cycle are reported.
   - `lift_replay.replay_place` measures a recorded run.
   - The crane's new phases are `carry to …`, `set-down at …` and `abort: carry back …`. They map to CARRY and SET_DOWN; the rule table already forbids the target on its support while held.
5. **Panel routing.**
   - "pick and place X at Y" and "move X to Y" are the place ability's own phrasings.
   - The grammar's put / place / drop with a grid cell also becomes the place ability.
   - A destination that is not a cell is refused.
   - The executor refuses `pick_place` proposals by name. Its arc, the job runner and the demo stay, as reference.

## The first live check, and what changed (2026-10-12)

**The first live check was stopped after 4 runs.**
- Runs 1, 2 and 4 placed the cube 2.8–3.2 mm from the centre.
- Run 3 aborted safely: the cube went back to r2c2, the hand withdrew, and the arm returned to PRESENT.
  - The live hold varies about ±4 mm in the hand, which the offline hold does not.
  - The 1.4 mm correction 2 cm up re-solved every rung of the destination ladder.
  - Rung 0 stands at the shoulder-roll stop, so the IK failed there.
- Run 2 met D6 and was still answered "failed", on the lift's 0.1 mm slip criterion.

**Owner decision: option 1, the D6 verdict, and option 3 if the margin is under 2°.**
- **The correction stops at the support point.** It re-solves the destination ladder only down to one rung past where the held object is predicted to meet the table (`replan_setdown`, `CranePlan.floor_rung`). The lowering never goes below that rung. Run 3's recorded case was replayed offline: the old correction is refused at rung 0, the new one solves and re-checks clean.
- **The carry and an abort use the pickup ladder as flown,** after the hover correction.
- **The panel's placement verdict is D6.** Slip and hold height are reported only.
- **The shoulder-roll margin at set-down is under 2°:** 0.84–1.88° offline, and 0.25° with run 3's hold. The release drag also pushes toward the robot, so aiming that way costs about 0.8 mm of D6 error per mm. The set-down is aimed 1 mm toward the robot's base (`PlaceSpec.aim_m`); offline D6 error over 10 starts was ≤ 5.7 mm. D6 is still measured from the cell centre.

**The re-fly, 10 runs:**

| run | from cell centre (mm) | tilt | others moved | shoulder-roll margin at set-down (°) | release drag (mm) | pad contact after release | withdrawal | full cycle |
|---|---|---|---|---|---|---|---|---|
| 1 | 3.81 | 0° | 0 | 2.14 | 3.23 | none | straight up | yes |
| 2 | 3.24 | 0° | 0 | 1.15 | 2.54 | none | straight up | yes |
| 3 | 3.92 | 0° | 0 | 1.61 | 3.18 | none | straight up | yes |
| 4 | 3.43 | 0° | 0 | 1.29 | 2.70 | none | straight up | yes |
| 5 | 3.78 | 0° | 0 | 2.29 | 3.22 | none | straight up | yes |
| 6 | 3.88 | 0° | 0 | 1.66 | 3.20 | none | straight up | yes |
| 7 | 3.86 | 0° | 0 | 1.05 | 3.26 | none | straight up | yes |
| 8 | 3.34 | 0° | 0 | 2.19 | 2.47 | none | straight up | yes |
| 9 | 3.26 | 0° | 0 | 1.20 | 2.50 | none | straight up | yes |
| 10 | 4.07 | 0° | 0 | 1.00 | 3.61 | none | straight up | yes |

10 of 10 placed and full cycle, with no aborts. The held cube was within 0.03 mm of its aim before the release in every run. This agrees with the offline prediction for the 1 mm aim: 4.5–5.7 mm and 1.05–2.09°.

## Consequences

- **Validated scope.** Only r2c2 → r2c1 with the 60 mm red cube is validated (the #174 live check). Other pairs are planned and checked, and refused when a check fails.
- **The attitude is a different grasp.** At the set-down it sits 1.0–2.3° from the shoulder-roll stop, live (rung 0 of the plan: 0.9°). The descent's prediction tolerances, the correction caps and the drag study were all measured at tilt 30°.
- **The release drag is still the largest error term.** At the place attitude it is 3.8–5.0 mm offline, and it is reported in every run record.
- **The −68° opening and every force figure are simulator values.**
