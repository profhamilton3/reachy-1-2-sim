# Reachy command panel: design and findings

Prepared September 9, 2026. Documentation only; no project code was edited, built, tested, or restarted.

## Outcome

Add a **Talk to Reachy** card below the existing stereo view and scene controls on port 8080. Support typed task requests, clarification replies, proposed actions, and explicit confirmation. Audio is a later input/output adapter to this same conversation flow.

Natural-language interaction was already planned. The local IITG roadmap includes text input, a proposed target/action/reason, approval buttons, and an optional voice path that transcribes into the same task pipeline. This is a planned feature with some schema support, not an existing end-to-end conversational service in the inspected source.

## Verified local baseline

| Area | Evidence and implication |
| --- | --- |
| Simulator checkout | `/Users/terrancehamilton/reachy-1-2-sim`, HEAD `4093b942afd2f3200a5055ebef43ca6f260c72f6` |
| IITG checkout | `/Users/terrancehamilton/IITG-Reachy-Project`, HEAD `c928c06ddd64252edc39656a3444e4df871e4a7b` |
| Browser page | `web/camera_server.py` embeds HTML/JavaScript in `_INDEX_HTML`; HTTP defaults to 8080. Existing routes include `/`, `/stream/left`, `/stream/right`, `/status`, and `/scene`. |
| Scene controls | Browser connects directly to the native simulator WebSocket, default 8765. It sends `hello`, answers `heartbeat` with `heartbeat_ack`, and handles `state`, `place_ack`, and `error`. Preserve these behaviors. |
| Object placement | `place_object` and `place_ack` implement scene setup, including recalling objects to their pool positions. This directly changes object state; it is not a robot grasp or pick-and-place operation. |
| Scene | `FWDCenterLabSivaPool.yaml` extends `FWDCenterLabSiva.yaml`, which extends `FWDCenterLabMCC.yaml`. Pool objects start off the board. |
| Recycling example | `soda_can` is tagged `recyclable`; `foam_block` is tagged `non-recyclable`. Match tags exactly. Do not match `recyclable` as a substring of `non-recyclable`. |
| Missing destination | `FWDCenterLabSiva.yaml` explicitly drops `left_tray` and `right_tray`. The pool scene does not restore them or add a bin. Do not silently resolve “the bin” to either absent tray. |
| Existing plan schema | IITG `src/reachy_ai/planner/schema.py` defines target, task type, destination, confidence, confirmation flag, safety notes, and brief reason. Its destination enum only allows left/right tray, handover zone, and point-only. |
| Implementation gaps | The inspected IITG source tree has the plan schema, configuration, client, and safety scaffolding, but no `app.py` or `planner/vlm_planner.py`. Those filenames in documentation are architectural intentions. |
| Simulator motion | Simulator source includes motion primitives, recipes, scene awareness, and pick/place evaluation. The recipe task runner invokes an offline episode runner; do not assume it is suitable for controlling the currently displayed live simulation. |

Unrelated simulator work observed: modified `notebooks/place_objects.ipynb` and untracked `notebooks/demo_pick_place.py`. Preserve them and recheck status before any future implementation.

## User experience

```text
[ LEFT CAMERA ]                  [ RIGHT CAMERA ]
[ Camera / simulator status ]
[ THE BOARD ]                    [ SCENE CONTROL ]

TALK TO REACHY                   Planning only · Scene data
Reachy: Tell me what you would like me to do.

You: Put the recycle item in the bin.
Reachy: This scene has no bin destination configured.
        Choose a destination before I can plan this move.

[ Type a command or answer…                         ] [Send]

When a valid plan is available:
Proposed action: move soda_can to the selected destination
[Confirm plan] [Cancel]
```

The initial example must reflect the actual scene. If the can is still parked in the pool, explain that no recyclable item is currently on the tabletop. If a destination is configured later, use its real ID and label in the proposal. A grid destination can be supported, but call it a grid cell or drop zone—not a physical bin.

Use a multiline input with Enter to send and Shift+Enter for a newline; respect input-method composition. Keep the conversation separate from the existing scene-operation log. Display connection errors and unsupported requests explicitly. Render user and model text as text, not executable HTML. Keep a bounded transcript and accessible labels/status announcements.

## Recommended architecture

Keep the native 8765 WebSocket responsible for simulator state and commands. Introduce a task coordinator outside the physics/render loop. For the first implementation, prefer same-origin HTTP task routes on the existing 8080 page, with asynchronous task processing and bounded status polling. A thin HTTP adapter can delegate to a separate process if dependency isolation requires it.

```text
Text box → task coordinator → current scene/state → planner
                   ↓                         ↓
             conversation replies ← validated proposal
                   ↓
             human confirmation
                   ↓
      deterministic simulator execution adapter
                   ↓
      authoritative simulator state → verified result
```

A dedicated conversation WebSocket is an alternative if streaming is needed; it should be owned by the coordinator. Do not put cloud calls inside the native server receive loop or per-frame processing. Reusing port 8765 for conversational message types is possible but adds coupling and is not the preferred initial design.

There are two `reachy_ai` packages across these checkouts. Do not place both source roots on `PYTHONPATH` and rely on import order. Choose explicit package ownership or process/API isolation.

## Planning, confirmation, and execution

Start with a clearly labelled **planning-only** mode. It can resolve supported commands using scene metadata and a deterministic parser without Coral, audio, credentials, or a cloud model. It must not claim unrestricted language understanding. Unsupported input should return a useful explanation. Add a language-model adapter separately, using the same validated proposal contract.

Join semantic metadata from the resolved scene with current authoritative object poses. The `/scene` route currently returns object IDs and cells, not a complete live semantic inventory. YAML initial positions are insufficient after scene-control placement. Objects in the off-table pool must not be treated as tabletop candidates. Runtime reshaping must be reflected in planning geometry; uncertain semantic labels should require clarification.

Bind every proposal to a session, task ID, plan ID/version, scene identity, and relevant state evidence. Confirmation must be checked server-side and consumed at most once. Do not trust the planner's confirmation flag to authorize motion.

Scene revision alone is insufficient for invalidation: object placement and ordinary physics motion can change relevant state without changing that value. Recheck current object poses, destination occupancy, geometry, reachability, and connection state before execution. Use a scene-mutation generation counter or equivalent server-owned evidence plus pose tolerances; do not invalidate simply because every simulation step advances.

Use these distinct outcomes:

- **Plan confirmed — no movement:** planning-only acknowledgment.
- **Execution unavailable:** no validated live execution adapter is installed.
- **Executing in simulation:** accepted by a validated simulator executor.
- **Completed / failed:** reported from execution and outcome evidence, not just message receipt.

Never implement robot task completion by sending `place_object`. Keep scene setup separate from robot execution. Do not run an offline episode runner that resets or advances a separate world and report it as the live panel's result. Physical-robot execution is outside this handoff.

## Coral TPU and future audio

Coral is appropriate for compatible image classification/detection models. It requires supported, 8-bit-quantized TensorFlow Lite models compiled for the Edge TPU. It is not the processor for the proposed general conversational language/vision planner. Use a separately configured local CPU/GPU model or cloud model for that role.

The Docker simulator does not emulate Coral hardware. Actual Coral inference requires a reachable physical accelerator and compatible runtime. For this native-macOS plus Linux-Docker architecture, verify the accelerator model and host support before promising device passthrough. A detector service on the school's compatible Coral-equipped host is a possible later integration, with image transfer and returned detections; it is not currently established by this inspection.

Initially use scene metadata, labelled **Scene data**. Later use a perception adapter labelled **Camera perception** and test on simulator frames; successful detection on real photos does not guarantee detection of rendered shapes. CPU fallback needs a suitable non-Edge-TPU model, not an assumption that an Edge-TPU-compiled file runs unchanged.

Audio remains future scope: microphone → speech-to-text → existing task submission; Reachy's response text → optional speech output. Do not add microphone permissions or audio dependencies for this change.

Primary technical reference: [Coral model compatibility and requirements](https://coral.ai/docs/edgetpu/models-intro/), consulted September 9, 2026.

## Local source references

- [Current browser panel](/Users/terrancehamilton/reachy-1-2-sim/web/camera_server.py)
- [Pool scene](/Users/terrancehamilton/reachy-1-2-sim/scenes/FWDCenterLabSivaPool.yaml)
- [Parent scene and removed trays](/Users/terrancehamilton/reachy-1-2-sim/scenes/FWDCenterLabSiva.yaml:69)
- [Simulator protocol](/Users/terrancehamilton/reachy-1-2-sim/native_mujoco/protocol.py)
- [Simulator server](/Users/terrancehamilton/reachy-1-2-sim/native_mujoco/server.py)
- [Original text UI and voice roadmap](/Users/terrancehamilton/IITG-Reachy-Project/docs/reachy_1_2_updated_roadmap.md:600)
- [Existing plan schema](/Users/terrancehamilton/IITG-Reachy-Project/src/reachy_ai/planner/schema.py)
- [Simulator architecture constraints](/Users/terrancehamilton/reachy-1-2-sim/CLAUDE.md)
- [IITG architecture constraints](/Users/terrancehamilton/IITG-Reachy-Project/CLAUDE.md)
