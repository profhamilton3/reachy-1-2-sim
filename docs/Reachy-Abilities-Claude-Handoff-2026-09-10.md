# Reachy abilities: code review and Claude Code handoff

September 10, 2026. Static, read-only source review. No application code, notebooks, or configuration edited; no simulator, motion, or tests executed. This document proposes implementation work for the agent managing the repositories when assigned by the user.

Simulator baseline: `82a41c4` (PR #54 merged). IITG baseline: `3f008a9`. Recheck current status before implementation; preserve modified `notebooks/place_objects.ipynb` and untracked `notebooks/demo_pick_place.py`, plus any subsequent user work.

## What has changed since the previous design

The simulator now includes `web/tasks.py`, `panel_routes.py`, `panel_planner.py`, `panel_scene.py`, `panel_sim_link.py`, and `panel_executor.py`. These provide task/session coordination, clarification and confirmation, live scene information, and optional live simulator pick-and-place. The executor uses the SDK bridge to the same simulated world, acquires a server execution lease, and verifies object placement from live poses. The coordinator chooses the displayed execution mode per proposal. These are source-level observations, not a runtime certification.

The requested abilities are not implemented in the panel parser: `parse_intent()` recognizes object-placement verbs, `_propose()` always emits `pick_place`, and `SimulatorExecutor.available()` refuses other task types. Extend those contracts rather than adding phrase-specific shortcuts directly in browser JavaScript.

## Review findings relevant to this extension

1. **The existing panel's home route is different from the rail-pocket routine.** `panel_executor.py` calls `P.raise_to_side()` and `P.go_home()`. The latter calls `stow_from_side()`, a lateral abduction/lowering sequence. The notebook's `STOW_ROUTE` is a distinct multi-waypoint corridor through the rig. Do not map the user's “store” or “reset” request to `P.go_home()` based on its name alone. Validate scene-specific route compatibility for existing pick-and-place entry/exit as well.
2. **Conversation-only outcomes are missing.** `PlannerOutcome` documents clarification/proposal/unsupported; `_apply_locked()` treats other outcomes as failure. The planner also checks scene availability before parsing. Add a successful reply outcome and route greetings before scene-dependent planning so “Hello” can work while the simulator is disconnected.
3. **The current proposal assumes an object and destination.** `target_id` and `destination` are required strings, and live revalidation assumes object movement. Rest, stow, and wave must not invent dummy objects. Pointing must not reuse pick-and-place's requirement that a target cell be empty.
4. **Clarification slots need explicit identity.** `_apply_answers()` decides whether a destination is open largely by whether the cell exists, not whether it was rejected as occupied or unreachable. An answer giving a new cell can therefore be handled as a target answer instead. Use typed pending questions such as `which_cell`, `which_object`, and `which_arm` for the new intents, and repair the existing invalid-destination reply path.
5. **Negated recycling language is not protected by exact scene tags.** `_is_recycle_phrase()` uses substring matching, so “non-recyclable item” enters recyclable resolution. Add explicit negation handling or reject unsupported phrasing. Scene-tag correctness alone does not resolve this parser error.

Additional executor hardening to include in the same review: `_execute_locked()` reads scene state before acquiring the lease; take/revalidate a fresh snapshot after acquisition. Cancellation currently reaches the pick/place callback after the initial raise and is followed by a home motion. Define cancellation semantics for every stage of every new route; a UI Stop is not an immediate emergency stop. The lease permits the `docker-core` bridge, which can aggregate more than one SDK caller; establish command ownership at that shared ingress before claiming exclusive control against competing notebooks.

## Ability contract

Use these canonical action names as proposed names; adapt naming to the repository while preserving the distinctions.

| User request | Canonical intent | Required behavior |
| --- | --- | --- |
| “Place your forearm on the table”; typo “foream”; “rest your arm on the table” | `rest_forearm` | Follow the scene-specific deployment route to `REST`, verifying the intended supported resting posture. |
| “Wave”; “wave your hand” | `wave` | Reach the validated presentation pose, perform a bounded wave, return to the documented presentation endpoint. |
| “Hello”; “hi”; “good morning” | `greet` | Reply in text without movement or confirmation. No automatic wave or audio. |
| “Store your arm”; “put your arm away”; “return to default position”; bare “reset” | `stow_arm` | Return through the validated rail-pocket route. Preserve scene objects and simulation time. |
| “Point to a cell” | `point_cell` | Ask which cell; do not choose one silently. |
| “Point to r2c2” | `point_cell` | Hover the gripper over the named cell with validated clearance. |
| “Point to the soda_can”; “point to the soda can” | `point_object` | Resolve the object's current pose, hover above its top, and avoid moving it. |

The user explicitly defines conversational **reset = stow arm**. Keep native simulator reset distinct; never translate bare text “reset” into the WebSocket reset command, restart, world reload, or board clearing. If explicitly asked to “reset the simulation,” explain that this is a separate operation rather than matching the stow alias by substring.

Default to the right arm because the identified corridor is right-arm-specific. Ask or report unsupported behavior for left-arm requests; do not mirror the route into an asymmetric rig without validation. Distinguish `REST` (on the tabletop) from `HOME` (stored in the rail pocket), even though the user may use “rest” colloquially for both. Ask when context does not distinguish them.

## Reuse the actual motion source

Primary source: [tlh_motion-routine.ipynb](/Users/terrancehamilton/reachy-1-2-sim/notebooks/tlh_motion-routine.ipynb). Read cell **source**, not only saved outputs. Preserve the notebook. Extract reusable definitions into a reviewed module only when implementation is assigned; do not execute notebook cells by importing the notebook.

### Forearm rest and stow

Notebook `PLACE_ROUTE`:

`GRIP_SHUT → BACK → CURL → CURL_HIGH → TUCK → SWING_1 → SWING_2 → SWING_3 → HOVER → REST_SHUT → REST`

Notebook `STOW_ROUTE`:

`REST_SHUT → HOVER → SWING_3 → SWING_2 → SWING_1 → TUCK → CURL_HIGH → CURL → BACK → GRIP_SHUT → HOME`

Preserve the route's waypoint definitions, gripper states, timing, tracking checks, and scene assumptions. Its comments identify the pose set with FWDCenterLabMCC; establish compatibility with the current FWDCenterLabSivaPool rig and live object placement before enabling it. Endpoint equality does not prove the path is clear.

Do not jump directly to HOME from an arbitrary pose. `ensure_home()` contains a nearest-waypoint recovery strategy but explicitly describes the initial connecting segment as unverified. Promote recovery only with a validated connecting path; an arm threaded under a rail is not a routine stow start. Report recovery-needed if no supported transition exists. Never silently reset the world to recover.

Resting intentionally allows forearm/table support contact. Define that narrow permitted contact region and approach phase; do not globally disable collision checks. Refuse a rest request if an object occupies the forearm footprint. Already-at-rest or already-stowed requests should be successful no-ops after pose verification. If holding an object, clarify its disposition before closing/opening the gripper or entering the pocket.

### Wave

The notebook's section 4.6 defines `WAVE_A` and `WAVE_B` relative to `PRESENT`, alternates them three times with 1.8-second segments, and returns to `PRESENT`. Reuse that bounded behavior and its context, including tracking limitations for wrist/forearm joints. Validate the approach to PRESENT from the current posture and the complete swept arm geometry. Do not blindly increase amplitudes or repetitions.

The response should identify the endpoint: “I’ll wave three times and finish with my arm raised.” Returning to tabletop rest or stow is a separate validated transition, not an implicit direct jump.

### Pointing

The notebook defines `point_at(label, xy, base_z, secs=2.0, approaching=None)`, uses `scene.cell_center()`, `scene.hover_point()`, live scene refresh, path clearances, and `escort_to()`. Reuse its approach, including checking the target during flight even where the solver excludes it for optimization.

Pointing is a non-grasp hover. A cell containing an object can still be pointed at if hover height clears that object; pick/place destination occupancy rules do not apply. Reject unknown/unreachable cells. For objects, use live poses and dimensions, not initial YAML values. A soda can in the off-board pool should produce an explanation or clarification rather than an attempted tabletop reach.

The notebook explicitly documents significant pointing error and object disturbance in some runs. Do not present these routines as fully validated. Verify achieved gripper position, clearance, target identity, and object drift. “Trajectory finished” must not count as “pointed accurately.” If gripper orientation is not constrained by the current IK, label the capability a hover pointer and make the final pose visibly indicate the target; do not claim a calibrated pointing ray.

## Implementation sequence for the managing agent

1. Add a backend ability registry containing action name, accepted aliases, required slots, scene requirements, route identity, supported start/end postures, confirmation policy, and completion validator. Match full intents with negation handling; normalize the specific typo `foream`. Reject unsupported compound requests rather than executing an arbitrary portion.
2. Extend `PlannerOutcome` with a successful reply; keep greetings available without scene/SDK access. Greetings must not discard an existing clarification or arm task.
3. Extend Proposal with validated per-action arguments: arm; optional cell/object; route/version; expected start posture. Preserve old pick/place clients. Do not fill nonexistent targets/destinations with placeholders.
4. Bind clarification replies to explicit question/slot IDs. “Point to a cell” followed by “r2c2” must resolve that cell; a new stow command during clarification must be recognized as a new task or explicit replacement, not an object name.
5. Route movement through confirmed proposals and the live executor. Revalidate under the execution lease. Use scene-specific motion modules and a measured posture-transition graph. Carry cancellation checks through entry, action, exit, and settling; no unverified auto-retreat after a stop.
6. Advertise per-action availability. Show “unsupported in this posture/scene” instead of claiming the entire arm is unavailable. Keep planning-only behavior honest when no validated executor exists.
7. Add per-action completion evidence: stable rail-pocket pose for stow; supported rest pose/contact for rest; completed bounded cycles for wave; achieved hover and undisturbed objects for pointing. Record the actual result and final posture.

## Acceptance cases

- Every listed user phrase resolves to the intended action; “foream” resolves correctly.
- Bare “reset” proposes stow and emits zero world-reset messages.
- “Hello” returns a successful reply with zero motion, including when disconnected.
- “Hello” during an active task does not cancel it or occupy the execution lease.
- “Point to a cell” asks which cell; a valid response fills the correct slot.
- Pointing to an occupied cell uses object clearance, not pick/place rejection.
- “Point to soda_can” uses the current live location and does not grasp or teleport it.
- Blocked rest footprint, unsupported start posture, held object, stale confirmation, and changed scene all produce appropriate refusal/clarification.
- Rest and stow follow the rig corridor; known HOME and REST are verified no-ops when already achieved.
- Wave completes only the requested/default bounded cycles and reports its actual endpoint.
- Stop/cancel works during deployment, action, and stowing, with no claim of instantaneous stop unless implemented.
- Competing SDK callers cannot interleave arm commands during an owned action.
- Existing pick/place, heartbeat, object placement/recall, confirmation-versioning, and stereo-view behavior remain intact.

Run existing relevant panel tests plus new intent/route/result tests when implementation is assigned. Perform target-simulator validation separately and report scene identity, start posture, failures, and untested cases. Do not run physical hardware motion for this feature.

## Source entry points

- [Parser and live proposal validator](/Users/terrancehamilton/reachy-1-2-sim/web/panel_planner.py)
- [Task contracts and coordinator](/Users/terrancehamilton/reachy-1-2-sim/web/tasks.py)
- [Live executor](/Users/terrancehamilton/reachy-1-2-sim/web/panel_executor.py)
- [Current side-route home implementation](/Users/terrancehamilton/reachy-1-2-sim/src/reachy_ai/motion/primitives.py:397)
- [Closed-loop escort](/Users/terrancehamilton/reachy-1-2-sim/src/reachy_ai/motion/escort.py)
- [Scene geometry and hover helpers](/Users/terrancehamilton/reachy-1-2-sim/src/reachy_ai/scene/awareness.py)
