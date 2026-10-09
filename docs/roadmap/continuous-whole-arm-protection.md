# Proposed roadmap: continuous whole-arm clearance monitoring

Status: proposal for roadmap/issue integration; no implementation or experiment authorization.
Date: 2026-09-26

## Intended outcome

Before and during an authorized motion, evaluate the upper arm, forearm, wrist, and hand against known obstacles. Refuse a path or request a controlled stop when clearance, tracking, or evidence freshness leaves the validated operating envelope. Report the specific cause and preserve the evidence.

This is collision-risk reduction within a declared operating envelope, not a guarantee of collision avoidance or certified human protection. A stop request cannot undo contact that has already occurred.

## Relationship to existing work

- Simulator issues #56 and #74 cover the model-based clearance foundation and its integration with motion. This proposal extends that foundation into runtime monitoring; it does not redefine their existing acceptance criteria.
- PR #144 and the planned bridge A/B comparison concern trustworthy measurements and commanded-path behavior. Their completion does not, by itself, validate a runtime monitor or establish a safety margin.
- The simulator roadmap's tracked-object state, camera, and recording capabilities supply inputs and evaluation infrastructure.
- IITG's September roadmap Phase 2 supplies stereo depth and camera-to-robot grounding. Phase 4 / EPIC 9 describes visual scene memory and an avoidance precheck. Those are dependencies for perception-fed obstacles, not prerequisites for the initial simulator-ground-truth monitor.

## Milestone A — Define and validate the monitor's contract

Specify the supported routes, arm links, apertures, joint ranges, obstacle classes, and coordinate frames. Explicitly state treatment of static fixtures, table/support contact, intended grasp contacts, untracked objects, and self-collision. No object category is silently assumed protected.

Define separate inputs for commanded goals and measured joint positions, with sequence, timestamp, calibration/model identity, and freshness checks. Use all joints relevant to collision geometry, including wrist roll and gripper.

Derive the operating envelope from measured end-to-end delay, tracking error, object-state uncertainty, and distance traveled while stopping. Do not adopt the campaign's observed maximum clearance error or checkpoint thresholds as a validated margin. Document when no defensible envelope exists.

Acceptance:
- A reviewed contract identifies every monitored input, exclusion, failure response, and evidence requirement.
- Collision-model coverage is verified across the declared poses and apertures.
- Stop behavior, maximum permitted latency, and clearance allowance have a justified validation plan rather than invented constants.

## Milestone B — Simulator-ground-truth monitor

First run in observation-only mode. Consume measured arm state and authoritative simulator object poses, checking current geometry and a short forward trajectory over the response/stopping interval. Evaluate changes during holds and transitions as well as waypoint arrival.

Then connect a validated stop request to the motion owner. Ensure queued commands and re-stream passes cannot immediately override the stop. Require explicit recovery authorization; do not automatically replan or resume.

Acceptance scenarios:
- Elbow, forearm, wrist, and opening-finger hazards are each detected, including cases with a clear gripper endpoint.
- An object moves into the path after the initial precheck.
- Stale, missing, reordered, or reset-crossing state; tracking departure; monitor failure; and command-stream continuation after stop are exercised.
- Intended grasp/support contacts are handled only under explicit, bounded policy.
- For controlled scenarios inside the operating envelope, record detection time, stop-request time, realised stopping distance, minimum clearance, and physics contacts. Contact reports are evaluation evidence, not the primary means of preventing contact.
- Demonstrate that supported safe routes remain usable; report false refusals and monitoring overhead.

Offline tests precede any separately authorized simulator experiment. Physical hardware validation remains separate.

## Milestone C — Camera-fed obstacle model

Add an observation adapter that turns stereo/depth detections into obstacle geometry in the robot frame. Preserve observation time, uncertainty, calibration identity, and tracking confidence. Account for occlusion and unknown space through an explicit policy; do not treat an unseen or lost object as free space automatically.

Validate perception against simulator ground truth before using it as the monitor's input. Keep ground-truth measurements available for scoring, but do not silently substitute them when evaluating perception-driven behavior.

Acceptance:
- Quantified position/size error and latency for the declared object classes, distances, lighting, and occlusions.
- Camera movement and calibration changes invalidate incompatible observations.
- Detection loss, delayed frames, newly entering obstacles, and obstacle motion have tested responses.
- Repeat Milestone B's applicable scenarios using perception inputs, with uncertainty included in the operating envelope.

## Suggested issue sequence

1. **Contract and stop-path design:** Milestone A, design-only.
2. **Observation-only whole-arm monitor:** simulator-state inputs, logs, and offline tests.
3. **Stop enforcement and simulator validation:** reviewed stop integration and separately approved experiments.
4. **Perception-to-obstacle adapter and scoring:** coordinate alignment, uncertainty, freshness, and occlusion handling.
5. **Perception-fed monitor validation:** repeat relevant scenarios; document remaining limits.

Each issue should name its prerequisites and exclusions. Keep gripper tuning, general autonomous replanning, physical deployment, and human-safety certification outside this milestone unless explicitly added later.

## Deferred from #56 (owner scope amendment, 2026-10-06)

#56 was closed as completion of an **explicitly amended simulator scope**: PR #152 (`41fbb9e`), PR #153 (`e593f77`), and ADR-0004. These items were deferred by owner decision. They are tracked here and are **not** completed. Nothing in this entry authorizes implementation, experiments or physical motion, and none of #56's conclusions extend to the physical robot.

1. **Re-derive the transit constants.** `CLEAR_Z = 1.00` and `CARRY_HZ = 8` are in `motion/transit.py`, moved but **not re-derived**. #56's box 3 asks to "re-derive both", and that is still outstanding.
   - Whole-arm preflight now checks the planned arc. The retained height and rate remain operating choices; their effects have not been separately established.
   - Both date from `8acc2e7` (2026-09-09), when the pre-#141 bridge echo was live.
   - Unmeasured costs: the carry, which runs at `CLEAR_Z`, found no IK solution for r1c2 or r1c3, and each transit takes about 5 s.
2. **Swept checking.** `check_arm_path` samples the planned geometry at ≤ 2° joint steps. Contact between samples is not seen.
3. **Execution monitoring and replanning.**
   - Realised deviation from the planned path is not covered: in the #153 re-trial, the pad deviated a median of 0.3–2.2 cm per segment, with a maximum of 9.5 cm.
   - Nothing is re-checked during execution (ADR-0004 R4).
   - Monitoring is Milestones A–B above.
   - Automatic replanning of refused arcs, and clearance-optimal trajectories, are deferred as a separate item. They sit outside Milestones A–C, as stated under "Suggested issue sequence".
4. **Margin assessment.** The whole-arm margin stays 0 cm (`WHOLE_ARM_MARGIN_M`).
   - The evidence is one trial, in which the cube was not carried. After the grasp, the planned and realised clearances were measured against different cube positions.
   - The 2.2 cm planned-versus-realised difference is therefore not a margin.
   - Milestone A's envelope derivation applies.
5. **Atomic pose delivery.** The bridge's 20 ms batching delivered 242 of 245 planned poses whole. The blended intermediate commands are not checked.

Other documented follow-ups are unchanged (ADR-0004 and the #153 PR text):
- a guard against running a stale image (the next sim launch must recreate the container so the #153 mounts apply);
- the `"shells"` hand model, which waits on E1 and physical validation;
- the tabletop check, which remains pad-point only (R7);
- modernizing the historical demo and notebook.

Grasp-related items stay under #55: contact surfaces, carried-object protection, unintended hand–target contact, jaw orientation and grasp reliability.

### Deferred under #55: checked retreat from a refused withdrawal (2026-10-07)

**What happened.** In the 2026-10-07 repeatability series (attempt 2), the crane lift's withdrawal judgement refused after release:
- the open finger's predicted clearance at the hand's **current** pose was 0.46 mm, against a 0.494 mm uncertainty allowance;
- the permitted bounded sideways correction cannot change the pose the hand is already in.

That is a refusal at the model's boundary, not a detected collision. Recovery was a simulator reset.

**What a general "checked retreat" would need.** A principled rule for leaving a poorly cleared starting state. For example, a start below the allowance could be accepted only if the motion strictly increases the violating clearance from the first step, keeps every other clearance at or above the allowance, and keeps the contact monitor on. Re-applying the unchanged clearance rule to the start state simply rejects it again.

**Status.** Deferred; not implemented, and nothing here authorizes it. Prevention, meaning not accepting a release whose own withdrawal cannot start, is proposed separately under #55.

## Source references

- Simulator roadmap: `reachy-1-2-sim/docs/ROADMAP.md` (R12-503; camera and research instrumentation).
- Existing guard: `reachy-1-2-sim/web/panel_executor.py::_footprint_refusal`.
- Scene updates and obstacle selection: `reachy-1-2-sim/src/reachy_ai/scene/awareness.py`.
- IITG roadmap: `IITG-Reachy-Project/docs/roadmap/ROADMAP-2026-09.md` (Phases 2 and 4).
- Issues: https://github.com/profhamilton3/reachy-1-2-sim/issues/56 and https://github.com/profhamilton3/reachy-1-2-sim/issues/74.

## Next action

Review and incorporate this milestone into the simulator roadmap, cross-linking the IITG perception dependencies. Open the contract/design issue first. This proposal does not authorize new coding, services, motion, or changes to PR #144.

### #55 owner decisions (2026-10-08)

Recorded by owner decision. Simulator only.

1. **Measurement scope amendment.** For #55's measurement criterion, the delivered crane lift ("pick up the red cube and put it back", the panel `lift_object` ability) replaces the original pick-and-place arc. The original `pick_place_live` arc **remains unmeasured**.
2. **Steady-state tracking error.** On this path it is compensated through measured-state corrections to the **commanded** poses: bounded, re-solved inside the joint travel, and re-checked. The actuator model is unchanged. **This is not a general compensation claim.**
3. **Grasp geometry.** The pad and closure geometry limits the **tested** grasp set: 15 attitudes, the planner's 36 starting guesses and its current rules, at r2c2 and r2c3. Specifically:
   - the pads are face-parallel only at 0° and about 36–39° apart at contact on the 60 mm cube;
   - the finger's first contact is a corner about 2.4 mm below the top edge.

   This does **not** establish that every possible grasp of the cube is infeasible.
4. **Evaluated scope.** The 60 mm red cube at r2c2. Other objects and cells are planned and checked, and refused when a check fails. They are not a supported claim.

### Unfinished capabilities tracked here (from #55)

- **Destination transfer:** pick from one cell and place in another (e.g. r2c2 → r2c1). This needs a carry between ladders, destination-ladder planning, and a held-object return or abort route. Not implemented.
- **Robust withdrawal and recovery:** the release currently drags the cube (quasi-static, about 7.6–9.7 mm recorded), and a withdrawal can be refused at the model boundary, leaving the open hand held beside the object until a simulator reset. Prevention (release/withdrawal continuity) and general checked retreat (above) are both unfinished.
- **Observation-delivery delay on the panel → motion-worker path — RESOLVED (2026-10-09).** In the 2026-10-08 five-attempt measurement (attempt 5), the crane lift's live-feedback guard (`FEEDBACK_STALE_S` = 0.5 s) halted during the gated descent because an observation reached the motion worker 0.67 s old.
  - The simulator's own state stream showed no gap over 0.13 s in that window, so the delay arose on the panel's observation path (SimLink snapshot → feed → worker pipe).
  - The halt was safe: before any pad contact, the cube unmoved, a checked retreat to PRESENT, then the stow.
  - **Cause:** `reachy-sdk` 0.7.0 `ReachySDK._poll_waiting_commands` leaks the waits it starts for every joint on each command push. It never cancels the ones that did not fire, which is 14 per push for the joints a lift never commands.
    - They accumulate in the panel's motion worker, which is reused across lifts, so its generation-2 GC pauses keep growing: 7 → 124 ms within a single lift.
    - The attempt-5 pause itself was not recorded. The re-measure below is the evidence.
  - **Fix:**
    - #166: latest-wins observation delivery, an end-to-end age, and an explicit stale reason.
    - #167: the SDK poll fix, gated to reachy-sdk 0.7.0, plus a fresh-worker-per-lift backstop, off by default.
  - **Re-measure** (`phase1.5-remeasure-2026-10-09.md`): 10 lifts with 0 stale halts.
    - 5 lifts in one worker; 5 with the backstop on.
    - Full GC ≤ 15 ms throughout; heap and pending SDK waits flat.
