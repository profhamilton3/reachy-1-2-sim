# E1 Stage 1 (B4 validation set) — session report, 2026-09-15

Approved scope: Stage 0 + Stage 1 on `B4_pool_box_1_r2c3` only (three planned
flights, two recorded setup traversals, three scheduled resets), per
`outputs/e1-matrix-scope-proposal-2026-09-15.md` rev 2 with the owner's
clarifications (any failed gate or non-zero exit stops the session; no
automatic reset/recovery/retry/replacement). Code: `main` `6bca12e`. Simulator
only. **No margin is inferred, no guard was changed, no re-fly was made.**

## 1. Outcome in one line

**Stopped in cycle S1-a at the first motion leg.** The `RAISE_TO_SIDE` setup
traversal aborted at its second waypoint (`BACK`) because seven of the eight
right-arm joints were never taken out of compliant mode: the bridge's
`turn_on` compliance command (server cmd seq 2) was received but not applied.
Zero contacts, zero object disturbance, over the whole session. 0 of 3
flights flown; 1 of 2 setups attempted (aborted); 1 of 3 scheduled resets
used.

## 2. Per-cycle results

| step | gate | result |
|---|---|---|
| Stage 0 — fresh server (`6bca12e`), B4, recording | `/status` `mujoco-remote`, `frames_stale` false; manifest `contacts_tracked: True`; `scene_sha256` = board sha `818d8e57…`; scene-marker-publisher RUNNING (A1 fix effective) | **pass** |
| Stage 0 — parked recording (`…T211124Z`) | recorder 0, identity ok; linker 0, `contacts=0`, 154/154, covered/contiguous; `settled_pose_check` ok; displacement 4e-10 m; `HOME` start conditions: gross 0.0°, spread 0.3° (gripper −37.7° recorded, expected keyframe mismatch) | **pass** |
| S1-a C1 — scheduled reset #1 | server record: reset entry in `commands.jsonl`, `sim_step` 104050→550, ack gen 1 | **pass** |
| S1-a C2 — settle | quiescent (spread < 0.5°) 65.6 s after reset; pose = keyframe (`r_wrist_roll` 39.4°, `r_gripper` −36.0°) | **pass** |
| S1-a C3 — parked recording (`…T211442Z`) | recorder 0; linker 0, 154/154; settled ok (dz 4.5e-5 m); displacement 4e-10 m; `HOME` gross 0.0°, spread 0.4° | **pass** |
| S1-a notebook binding (fresh SDK connection after the reset) | identity ok, joint agreement 0.0° on all eight | **pass** |
| S1-a C4 — start check | `check_start(PLACE_ROUTE)` → `(True, "")`, `posture_of` = `home` | **pass** |
| S1-a C5 — setup `RAISE_TO_SIDE` (`…T211619Z`, 60 s) | `primitives.raise_to_side` raised `RuntimeError: the route out of the pocket stopped at BACK: r_elbow_pitch is 39 deg off, tolerance 6` after 19.2 s; recorder 0; linker 0, `contacts=0`, 3004/3004, covered/contiguous, no reset in window; displacement 8e-9 m (no disturbance); **`PARKED_AT_PRESENT=NO`** (gross residual 109.4°, spread 0.0°) | **STOP** |
| S1-a C7 — flight `LOWER_TO_REST` | not eligible; not attempted | — |
| S1-b, S1-c | not started (session stopped) | — |

End posture at stop (SDK): shoulder_pitch +39.3, shoulder_roll 0, arm_yaw 0,
elbow_pitch −39.0, forearm_yaw 0, wrist_pitch −0.1, wrist_roll +40.1,
gripper −39.8 — i.e. the arm in the rail pocket at `BACK`'s shoulder pitch
with the elbow hanging under gravity. The container and native server were
then stopped (no further motion). The state stream is complete
(29 029 states, seq contiguous); the manifest's `state_count`/`total_steps`
read 0/None because the server was stopped with SIGTERM (known deferred item).

## 3. Cause, from the server's own record

- `commands.jsonl` after reset #1: cmd seq 1 carries `compliant=False` for
  `r_shoulder_pitch`; cmd seq 2 carries `compliant=False` for the other seven
  right-arm joints; every later command carries targets only. Final targets
  were correct for `BACK` (shoulder_pitch 40, elbow 0, gripper +20, wrist_roll 0).
- `states.jsonl`: the only compliance transition in the whole run is
  `r_shoulder_pitch` → non-compliant at seq 15905. The other seven stayed
  `compliant: true`, `effort: 0.0` throughout. So seq 2 was received (it is in
  the log) and never applied.
- The pilot run (`run_20260915_173758`, `f548fc6`) shows the identical two-command
  pattern (seq 1 → 1 flag, seq 2 → 7 flags) and there seq 2 **was** applied three
  states (60 ms) later — all eight non-compliant. Same code path, different
  timing.
- Mechanism consistent with the code: `native_mujoco/server.py`
  `submit_command` holds a single `_pending_cmd` slot that each arrival
  overwrites, drained once per sim tick by `apply_pending`. The bridge sends at
  20 ms intervals. The state stream in this run has 529 push gaps > 40 ms
  (max 244 ms), i.e. the sim thread stalls across several bridge periods; a
  targets-only command arriving during a stall replaces a command that carried
  `compliant`/limit flags, which are one-shot (sent only on change) and are
  therefore lost. There was a 57 ms push gap immediately after the seq-1
  transition.
- Not the wrist-pitch droop (D-classes in the scope): the abort joint is the
  elbow, at effort 0.0, in compliant mode.

This is a tooling fault in the simulator server, not a property of the route,
the board, or the arm model. It is timing-dependent and could recur on any
`turn_on` — including the pilot's — so it must be fixed and tested before
Stage 1 is re-attempted. Filed as `reachy-1-2-sim` issue #116.

## 4. What is and is not usable

- Stage 0 and S1-a parked recordings: valid parked evidence (no motion).
- S1-a setup recording: **preserved as evidence** of the aborted traversal
  (D2-equivalent partial: `HOME` → `GRIP_SHUT` → `BACK`), with complete
  contact coverage (3004/3004) and no disturbance. It is not a setup sample
  and gates nothing.
- No flight data. No clearance number from this session is a measurement of
  any route's realised clearance.
- The three-cycle validation set is **not** validated; resets #2 and #3 were
  not used.

## 5. Preserved

Pilot evidence (`~/reachy-1-2-sim-e1`) untouched and hash-verified. Primary
checkout notebook changes untouched. Aborted first server start (container
name conflict, no recording used) kept under `control/aborted_start_1/`.
