# Claude Code handoff: typed Reachy commands

This is a proposed implementation brief for the agent already managing these repositories. It does not authorize code changes by the author of this document. The user requested documentation only in the originating task. Implement only when the user assigns implementation to you; otherwise review and refine the design.

Read [Reachy-Command-Panel-Design.md](Reachy-Command-Panel-Design.md) first. Its baseline was inspected September 9, 2026; recheck current code because development is ongoing.

## Intended result

Extend the existing page on port 8080 with a message box and conversation area beneath the stereo view and scene controls. Users can submit commands, answer clarification questions, review a proposed task, and confirm or cancel. Preserve the existing board, placement/recall controls, stereo streams, and native simulator WebSocket connection.

## Before implementation

1. Work primarily in `/Users/terrancehamilton/reachy-1-2-sim`. Inspect `/Users/terrancehamilton/IITG-Reachy-Project` as the source of the shared task design and planner schema.
2. Read applicable repository instructions, including both `CLAUDE.md` files and any newly added `AGENTS.md`. Check current status and HEAD. Preserve all unrelated work, especially notebooks.
3. Inspect `web/camera_server.py`, `native_mujoco/protocol.py`, `native_mujoco/server.py`, `native_mujoco/placement.py`, the scene inheritance chain, scene awareness, and motion/execution modules.
4. Record which components exist versus are still planned. Do not assume the IITG README's FastAPI app or VLM planner exists. Avoid import collisions between the two `reachy_ai` packages.

## Deliver in three explicit stages

### 1. Conversation and planning-only workflow

Add the new card to `_INDEX_HTML` using the current visual style. Provide message entry, Send, bounded conversation history, working/disconnected states, and version-bound Confirm plan/Cancel controls. Treat input and output as text. Handle malformed responses and network failure without breaking scene controls.

Introduce a coordinator and thin same-origin task API. The following routes are **proposals**, not existing endpoints:

| Method and route | Behavior |
| --- | --- |
| `POST /tasks` | Validate text; establish task/session identity; queue planning; return accepted task ID. |
| `GET /tasks/{task_id}` | Return status, ordered conversation events, proposal, and mode for the owning session. |
| `POST /tasks/{task_id}/reply` | Accept a clarification answer bound to the current question/version. |
| `POST /tasks/{task_id}/confirm` | Validate plan ID/version and current state; confirm once. |
| `POST /tasks/{task_id}/cancel` | Cancel planning or an unexecuted proposal; do not claim an executing motion has stopped until acknowledged. |

Use bounded workers, timeouts, request-size limits, and one active task per session initially. Track task ownership and duplicate requests. Keep model work out of request handling and physics/render/heartbeat loops. Enforce allowed browser origins for mutating local routes; do not expose credentials to JavaScript.

Recommended states: `planning`, `needs_clarification`, `awaiting_confirmation`, `confirmed_no_motion`, `executing`, `completed`, `cancelled`, `failed`, `expired`. Advertise server capabilities so execution controls appear only when a live executor is supported.

### 2. Resolve real scene objects and destinations

Build the planner's context from the resolved scene plus live authoritative state. Do not use `/scene` object IDs or YAML initial poses alone. Keep scene metadata and live geometry coherent across placement, recall, reshape, reset, and reconnect.

Support a small documented set of typed commands first, including the user's “Put the recycle item in the bin.” Normalize that phrasing to recyclable intent; identify recyclable objects by exact metadata membership. Never classify `non-recyclable` as recyclable by substring matching.

For the current pool scene:

- No tabletop candidate: explain that the recyclable object must first be placed on the board.
- Several candidates: ask which one, using labels or current cells.
- Missing bin: report that no bin is configured and offer valid destinations only.
- Explicit grid target: support a validated cell destination without claiming it is a bin.
- Unreachable or occupied destination: reject or request another destination; do not inherit the manual placement UI's reachability override.

The current IITG destination enum does not represent bins or grid cells. Introduce a validated scene destination reference in the coordinator contract, with an explicit compatibility adapter for legacy destinations. Do not silently substitute `left_tray`. Do not modify the calibrated parent scene or restore its deliberately removed trays just to satisfy the example.

If a physical bin model is later desired, treat that as a separate scene-design change in a child scene. The first panel increment should explain missing destinations correctly.

A proposal should carry: task/plan IDs, plan version, target ID, destination reference, task type, short reason, confirmation requirement, execution mode, semantic-source label, scene identity, and state evidence. Reject unknown object IDs, unsupported action types, nonfinite numeric values, and raw joint instructions. A model's confidence or `requires_confirmation=false` must never bypass server confirmation policy.

### 3. Connect execution only when validated

Reuse deterministic simulator motion capabilities through an explicit live execution adapter. First establish how commands reach the same world displayed by the panel. Preserve one authoritative simulation loop.

Do not call `place_object` to fake task success. Do not connect the offline recipe/episode runner to the live core if it independently resets or advances state. No physical-robot connection or `REACHY_ENABLE_MOTION=true` is part of this task.

Before movement, revalidate relevant poses, geometry, destination occupancy, motion feasibility, and safety gates. Confirmation must identify the exact plan and be consumed once. Block conflicting scene edits and other motion clients during execution using backend arbitration, not merely disabled browser buttons. Handle cancellation and connection loss according to acknowledged executor state; never auto-resume after reconnection.

Only report completion from task outcome evidence. Planning-only confirmation must say “Plan confirmed; no movement performed.” Missing execution capability must remain visible rather than being replaced by an animation or success message.

## Acceptance checks

Use focused unit/contract tests and a browser acceptance check when implementation is assigned. Follow existing repository-required checks and report what actually ran.

- Empty input does not create a task; malformed or excessive input produces a readable error.
- A can still in the pool is not a tabletop candidate.
- A placed `soda_can` resolves as recyclable; `foam_block` does not.
- A missing bin produces clarification and zero movement.
- A configured reachable grid destination can produce a clearly labelled proposal.
- Ambiguous targets require an answer tied to the current question.
- Reject nonexistent destinations, unapproved aliases, occupied cells, and unreachable task paths.
- Placement, recall, reshape, reset, or relevant pose drift makes an old proposal invalid; advancing simulation time alone does not.
- Duplicate confirmation, stale versions, and another session's confirmation cannot execute a task.
- Late planner responses cannot revive cancelled or superseded tasks.
- Confirming in planning-only mode performs no simulator motion and says so.
- Cloud/model timeout and disconnect produce explicit status while camera streams and existing heartbeats remain responsive.
- Reconnect does not replay commands or confirmations.
- Existing `hello`/`heartbeat_ack`, object placement/recall, board display, both stereo feeds, and page refresh continue to work.
- User/model content cannot inject HTML; the new card supports keyboard use and a narrow viewport.
- If execution is included, prove live-world identity, command arbitration, stop/cancel behavior, and result verification. Otherwise document it as unavailable.

## Out of scope

Audio capture/playback, Coral installation or device passthrough, model purchases or downloads, new physical hardware support, physical-robot motion, wholesale web-framework migration, notebook cleanup, and calibrated-scene redesign.

Return changed paths, design decisions, validation results, screenshots if UI changes were tested, and explicit remaining limitations. Do not claim AI perception when using scene metadata or claim physical grasp success from scene teleportation.
