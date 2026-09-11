# Reachy memory and ML: evidence and recommended integration

September 10, 2026. Local source and roadmap review only. Existing files were not changed, and no learning experiment, model, or simulator was run. “Implemented” below means code exists, not that a live deployment was validated.

## Answer

**Yes. Both projects discuss learning and memory, and the simulator already has substantial motion-experience storage and search code. The new conversational panel is still a deterministic parser and is not automatically learning from conversation or consuming that experience store.**

Distinguish four separate capabilities: conversational context, stored motion experience, visual scene memory, and ML model inference/training. Having one does not establish the others.

## Evidence by capability

| Capability | Evidence | Current interpretation |
| --- | --- | --- |
| Typed language understanding | Simulator `web/panel_planner.py`, `DeterministicPlanner` and `parse_intent()` | Implemented narrow rule-based parser. Module explicitly says no Coral/cloud model; an LLM adapter may be added behind `PlannerOutcome`. |
| Short-term conversation context | Simulator `web/tasks.py`: task/session dictionaries and `ConversationEvent`; `PlannerRequest.history` | Implemented in-process task history. Maximum 40 retained events per task and 200 retained tasks. Not durable personal or cross-task conversational memory. |
| Motion experience persistence | Simulator `src/reachy_ai/experience/store.py`, SQLite `ExperienceStore` | Implemented trial records, outcomes, artifacts, and successful-trial queries. Offline task runners can persist results when supplied a store. |
| Motion improvement through search | Simulator `src/reachy_ai/search/runner.py`, `samplers.py`, ranking/pruning/convergence modules | Implemented bounded recipe-parameter search, including grid, random, and adaptive sampling, resume/prior-result paths, and best-recipe export. |
| Adaptive sampling | `AdaptiveSampler` | Perturbs top-ranked prior parameter points while retaining random exploration. Its prior set is fixed within a run; it is refreshed when a new run is created. This is experience-guided optimization, not a neural policy learning continuously while the user chats. |
| Broader motion reuse | IITG `docs/roadmap/EPIC-8.md`, PR 8.6 | Planned strict-context retrieval and warm starts. A basic `query_compatible_trials()` exists; the proposed dedicated `experience/query.py` and `similarity.py` were not present in the inspected simulator source inventory. Do not call all of PR 8.6 complete. |
| Persistent visual scene memory | IITG `docs/roadmap/ROADMAP-2026-09.md`, Phase 4 / EPIC 9 / issue #13 | Explicitly planned `RigIdentity` / `SceneMemoryEntry`, dependent on camera/calibration, synthetic-data detection, and robot-frame grounding. Roadmap says to defer the store until Phase 2 lands. No corresponding implementation found in the inspected source. |
| Perception ML and reasoning | IITG updated roadmap and September roadmap; planner schema | Plans include Coral perception, synthetic-data detector work, and event-level cloud vision-language planning. The inspected IITG source has `planner/schema.py` but not the documented VLM planner implementation. |
| Reinforcement/imitation learning | IITG EPIC 8 non-goals and future directions | Camera-to-action RL, imitation learning, behavior cloning, diffusion policies, and related approaches are later proposals, not initial Epic 8 scope or demonstrated current panel capabilities. |

No ExperienceStore integration was found in the inspected `web/*.py` panel modules. Therefore do not say that a successful panel wave or point currently teaches Reachy to perform it better next time.

## Important limits before automatic reuse

`SimulatorIdentity` records model, scene, physics, calibration, and other provenance fields. However, `ExperienceStore.query_compatible_trials()` currently filters successful trials by **model hash and scene hash**, with optional task/study filters. It does not itself enforce every recorded identity field or a promoted-only policy.

Before making that method the panel's automatic recipe selector, add an explicit compatibility gate for the relevant compiled model, rig geometry, physics/actuator settings, calibration, route version, task/arm, start posture, and current obstacles. Record provenance and reject mismatches. A historical “success” is evidence for a candidate, not authorization to move in the current scene.

Also separate history from structured intent state. The current parser reconstructs the original request from the first retained user turn. Because the event list is bounded, long clarification sequences can eventually lose that original turn. Store the canonical intent and unresolved slots explicitly rather than relying on an indefinitely preserved transcript.

## Recommended ML role for the requested abilities

Use established routes as the initial motion library. Rest, stow, wave, and pointing do not need a trained model just to recognize the user's listed phrases. A deterministic ability registry provides an inspectable baseline and handles the explicitly requested aliases.

An optional language-model adapter can later interpret broader phrasing into the same typed action contract. For example, “Could you put your arm away now?” maps to `stow_arm`; it never produces joint trajectories. Validate action names and arguments, resolve live targets, enforce confirmation for movement, and run the existing deterministic executor. Keep “Hello” as a simple reply unless the user explicitly requests a gesture.

Coral's planned role is perception. It should not be a prerequisite for these typed commands or be represented as the general language reasoner. For this review, no new external model or hardware selection was made.

## Recommended memory layers

### 1. Conversation context

Keep the current task's intent, missing slots, clarification question, last explicitly selected target, and plan version. A greeting should not erase those. If “point there” has multiple plausible references or the scene changed, ask what “there” means. Never retain a human confirmation as standing authorization for future motions.

Persistent preferences, if added later, should be explicit and inspectable—for example, the user's default right arm and approved phrase aliases. Give them a separate store and reset control; conversational “reset” remains stow under the user's requirement, not erase memory.

### 2. Episode records for every ability

Connect completed simulator abilities to the existing experience infrastructure through a bounded logging adapter. Record skill/route version, scene/model identity, requested target, start and final measured posture, phases, tracking error, clearances/contact policy, object drift, cancellation/failure, and outcome evidence. Mark live interactive episodes as such; do not pretend they are deterministic offline training trials.

A record of a wave is memory. Improvement requires an explicit evaluation/search process using those records. Keep that distinction visible in project reports.

### 3. Offline learning and promotion

Extend recipe/evaluation support for `rest_forearm`, `stow_arm`, `wave`, `point_cell`, and `point_object`. Define measurable success first:

- Rest: reaches the supported posture with only intended table contact and no disturbed objects.
- Stow: reaches the measured pocket pose via the allowed corridor without unintended contacts.
- Wave: completes bounded oscillations within tracking/clearance limits.
- Point: reaches the selected hover region accurately, keeps clearance, and leaves objects undisturbed.

Use Epic 8's bounded search for parameters such as segment duration or permitted hover clearance, with task-specific bounds. Do not let search reorder a narrow rail-corridor route or remove its checks merely to improve a scalar score. Evaluate across applicable start states and object layouts. Retain failures as useful evidence.

Promote validated recipes into the panel's skill registry through a reviewed process. The live panel may retrieve compatible promoted recipes; it should not launch exploratory learning around the rig in response to an ordinary “Wave.”

### 4. Visual scene memory

Follow the September roadmap dependency order. Bind remembered rails/table geometry and obstacles to camera/rig/calibration identity and refresh against current observations. Stored positions must not override live object movement or scene-control edits. This is distinct from the current YAML/live-state scene model and from a conversation transcript.

## Recommended order

1. Implement the listed abilities using verified existing routes and clear action semantics.
2. Add structured task context and per-ability outcome recording.
3. Add stronger experience compatibility and promoted-recipe retrieval.
4. Extend offline search/evaluators to those abilities.
5. Add optional broader language interpretation and camera perception through adapters.
6. Build EPIC 9 visual memory when its grounding/calibration prerequisites are met.

## Sources

- [Panel parser and future language-model adapter contract](/Users/terrancehamilton/reachy-1-2-sim/web/panel_planner.py)
- [Task history and coordinator](/Users/terrancehamilton/reachy-1-2-sim/web/tasks.py)
- [SQLite experience store](/Users/terrancehamilton/reachy-1-2-sim/src/reachy_ai/experience/store.py)
- [Simulator identity and experience contracts](/Users/terrancehamilton/reachy-1-2-sim/src/reachy_ai/experience/models.py)
- [Adaptive sampler](/Users/terrancehamilton/reachy-1-2-sim/src/reachy_ai/search/samplers.py:92)
- [Search runner and prior-result reuse](/Users/terrancehamilton/reachy-1-2-sim/src/reachy_ai/search/runner.py)
- [EPIC 8: Adaptive Motion Search and Experience Memory](/Users/terrancehamilton/IITG-Reachy-Project/docs/roadmap/EPIC-8.md)
- [EPIC 9 discussion and dependencies](/Users/terrancehamilton/IITG-Reachy-Project/docs/roadmap/ROADMAP-2026-09.md:247)
- [IITG task/perception/voice roadmap](/Users/terrancehamilton/IITG-Reachy-Project/docs/reachy_1_2_updated_roadmap.md)
