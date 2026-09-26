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

## Source references

- Simulator roadmap: `reachy-1-2-sim/docs/ROADMAP.md` (R12-503; camera and research instrumentation).
- Existing guard: `reachy-1-2-sim/web/panel_executor.py::_footprint_refusal`.
- Scene updates and obstacle selection: `reachy-1-2-sim/src/reachy_ai/scene/awareness.py`.
- IITG roadmap: `IITG-Reachy-Project/docs/roadmap/ROADMAP-2026-09.md` (Phases 2 and 4).
- Issues: https://github.com/profhamilton3/reachy-1-2-sim/issues/56 and https://github.com/profhamilton3/reachy-1-2-sim/issues/74.

## Next action

Review and incorporate this milestone into the simulator roadmap, cross-linking the IITG perception dependencies. Open the contract/design issue first. This proposal does not authorize new coding, services, motion, or changes to PR #144.
