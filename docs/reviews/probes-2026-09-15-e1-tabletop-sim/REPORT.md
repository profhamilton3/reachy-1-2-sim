# E1 single simulator pilot — report (2026-09-15)

Executed under the owner's authorization of 2026-09-15 against
`outputs/e1-pilot-execution-scope-2026-09-15.md` rev 3. **Simulator only.**
One setup traversal and one flight; no retry, no recovery motion, no reset,
no second board, no matrix, no hardware, no guard/margin change.

Artifacts (all preserved): `reachy-1-2-sim-e1/docs/reviews/probes-2026-09-15-e1-tabletop-sim/`
(README lists every file). Code: `main` `f548fc6`, worktree
`/Users/terrancehamilton/reachy-1-2-sim-e1`, tree clean at start.

## 1. Outcomes

| step | result |
|---|---|
| Services | native server `Recording to: run_20260915_173758`; bridge `Backend: mujoco-remote → ws://host.docker.internal:8765`; `/status` `backend: mujoco-remote, frames_stale: false`; manifest `scene_sha256` = board sha `818d8e57…6ab083` |
| Checker self-test | 11/11 on the checked-out `rig_routes` (REST_SHUT judged REST → NO) |
| Parked recording (3 s, no motion) | recorder exit 0, identity ok; linker exit 0, `contacts=0`, evidence 154/154, `covered`/`seq_contiguous` true, `reset_in_window` false; displacement 4.0e-10 m |
| Motion-client binding (notebook cell 2) | `verify_simulator_identity` on the notebook's own `ReachySDK("localhost", 50051)`: ok, reasons `[]`, joint agreement ≤ 0.003°, `/status` mujoco-remote |
| **Setup traversal** `primitives.raise_to_side` (cell 3) | returned in **33.2 s** (nominal 33; no re-stream passes); end pose −67.8/−24.5/0.2/−79.0, gripper −45.0; recorder 120 s exit 0; linker exit 0; **contacts `[]`**, evidence **6003/6003**, covered/contiguous true, reset false; **displacement `pool_box_1` 1.2e-8 m**; `settled_pose_check` ok (dz 4.5e-5 m); **`PARKED_AT_PRESENT=yes`** (max gross err 1.0°, spread 0.0°, gripper −45.0); motion in the recording t = 3.6 → 36.6 s, parked tail 83 s → **gate PASS** |
| **Flight** `rig_motion.from_present` (cell 4) | returned in **6.66 s**, `reached ['REST_SHUT','REST']`, no `RouteError`/`RecoveryNeeded`/`TrackingError`, no wrist droop (final wrist_pitch −9.8); recorder 60 s exit 0; linker exit 0; **contacts `[]`**, evidence **3003/3003**, covered/contiguous true, reset false; **displacement `pool_box_1` 8.0e-9 m**; settled ok; **`PARKED_AT_REST=yes`** (max gross err 0.3°, spread 0.0°, gripper −45.0 — so not stopped at REST_SHUT, whose gripper would read +20); gripper trace −45 → **+19.2** (t 6.16 s) → −45; motion t = 3.25 → 9.70 s, parked tail 50 s |
| Whole session | `states.jsonl` seq 1..45609 contiguous, **zero states with any contact anywhere in the session**; the right arm moved in exactly three intervals: a 12.9-s settle at server start before any client connected (wrist_roll/gripper settling to +40.15/−40.15, shoulder roll 0.43°), the setup cell window, and the flight cell window; nothing else |

Both motion rows and the parked row are in `ledger.csv`.

## 2. Contact and displacement evidence

- Linker (`link_e1_flight.py` at f548fc6): all three logs exit 0 with
  `contacts_recorded: true`, `contacts: []`, `contacts_window.covered: true`,
  `seq_contiguous: true`, `reset_in_window: false`; states carry the
  `contacts` key on every state in every window (N/N above). Max push gap
  over the session 107 ms (diagnostic only; coverage is proven by the
  linker's bracketing + contiguity, not by the gap).
- Displacement first→last sample, `pool_box_1`: parked 4.0e-10 m, setup
  1.2e-8 m, flight 8.0e-9 m (all ≤ 5 mm by seven orders of magnitude). All
  ten scene objects ≤ 1.5e-8 m.
- Alignment: wall offsets within ±0.08 s; per-sample joint residual max
  1.62° (setup, mid-motion) and 0.18° (flight).
- Server-side recomputed clearance for `pool_box_1` (the hand **model**, not
  physics): worst tube **−4.01 cm** (setup, t 33.6 s) and **−3.99 cm**
  (flight, t 9.55 s), both on the REST_SHUT → REST leg while the gripper
  re-opens; shells +5.56 / +6.16 cm. Recorder "realised" numbers agree
  (tube −4.0 cm vs planned −3.72; shells +5.6/+6.2 vs planned +6.3/+6.5).
  So on B4 the tube model predicts ≈ 4 cm of interpenetration while
  MuJoCo reports no contact and no displacement. That is the observation E1
  exists to collect; it is one board, one flight, in the simulator, and
  **no production margin is inferred from it.**

## 3. Tail checks (`e1_tail_check.py`, targets read from `rig_routes` at run time)

| log | target | tail samples | max gross err | max spread | gripper | result |
|---|---|---|---|---|---|---|
| parked 3 s | HOME (info only, not a gate) | 60 | 0.0° | 0.0° | −40.1 (target −45) | `PARKED_AT_HOME=NO` on the gripper criterion alone |
| setup 120 s | PRESENT | 61 | 1.0° | 0.0° | −45.0 | **yes** |
| flight 60 s | REST | 61 | 0.3° | 0.0° | −45.0 | **yes** |

The HOME "NO" is informative: the simulator's rest pose after its start-up
settle is `rig_routes.HOME` on all gross joints but **wrist_roll +40.15° and
gripper −40.15°**, not 0 / −45. `raise_to_side` judges HOME on the four
placing joints, so it accepted the start; the setup route then drove the
wrist and gripper to their commanded values. Worth a note in the sim model
(keyframe vs. `HOME`), not a pilot fault.

## 4. Identity assurance actually obtained, and its limits

- Every shell (server, recorder, notebook) had no `REACHY_*` variables other
  than the three the scope names; `REACHY_IP` and `REACHY_ENABLE_MOTION`
  unset (asserted again inside the notebook kernel).
- Recorder identity check passed before all three recordings; the notebook
  ran the same function on its own SDK object before any motion (binding
  ok). Both are correlation checks at the moment run — they do not bind the
  TCP session to the process owning the run dir, and `/status`'s backend is
  the bridge's last sidecar write. The recorded traversals show the
  simulator moved as commanded; they are not evidence that the notebook's
  commands went nowhere else.
- Topology support: `commands.jsonl` holds 1 747 `joint_command`s (all
  reaching the physics via the one bridge socket); the container's
  execution lease was never taken (no panel). `commands.jsonl` carries no
  timestamps, so command timing is inferred from state motion, not from the
  command log — a gap worth closing in the recorder (#112-class).

## 5. Deviations from the scope and other limitations

1. **Recorder/notebook Python.** The host's Python 3.14 has no `reachy_sdk`
   and `reachy-sdk==0.7.0` will not install there (`mobile-base-sdk` pins
   `protobuf<=4.25.3`, so `grpcio-tools` 1.62 with no 3.14 wheel). The
   recorder and the notebook kernel ran from a scratch venv
   (Python 3.12.14, reachy-sdk 0.7.0, grpcio 1.62.2, protobuf 4.25.3,
   numpy 1.26.4) on the host — host-native as the recorder's clock argument
   requires. The linker and tail checker ran on the host 3.14 with
   `PYTHONPATH=src`.
2. **Docker image rebuilt** (`docker compose build`, cached layers) because
   the existing `reachy-1-2-sim:latest` (built 2026-09-11) predates
   `01700ec`/`38d60aa`, which add the `/tmp/reachy_frame_meta.json` sidecar
   that `/status` reads `backend` from; without it the identity check would
   have refused. The previous image (`ae2843e2966b`, built 2026-09-11) is no
   longer listed by `docker images -a` after the rebuild; the running image is
   `e22b885423a8` (built 2026-09-15 10:35 from the f548fc6 tree).
3. **Notebook executed headlessly** with `jupyter nbconvert --execute`
   (kernel `e1venv`), cells gated by marker files in `control/` rather than
   by a person clicking; the executed notebook with outputs is retained.
   Lead-ins were 3.6 s (setup) and 3.25 s (flight) — inside the 3–5 s band.
4. **`start_sim.sh` drops the `e1_boards/` subdirectory** when it derives the
   container scene path, so the RViz `scene-marker-publisher` crash-looped
   (`Scene file not found: /opt/scenes/B4_pool_box_1_r2c3.yaml`). Physics,
   bridge, `/status` and the manifest all used the correct host path; RViz
   object markers were simply absent. Cosmetic here; a one-line script fix.
5. The native server was stopped with SIGTERM after the pilot, so
   `manifest.json` `state_count`/`command_count` stayed 0/null; counts in
   this report are from the files themselves.
6. Docker Desktop was started for the session (it was not running); the
   container is `stopped`, not removed; port 8765 is free.
7. The worktree `reachy-1-2-sim-e1` was moved from `feat/sim-e1-readiness`
   (4395a9f, identical tree) to `main` fast-forwarded to `f548fc6`. The
   primary checkout (`chore/deployment-panel-features`, dirty notebook) was
   not touched.
8. `states.jsonl` is 242 MB and sits under `docs/reviews/…` — move it out
   before committing that directory. Nothing was committed.

## 6. What this does and does not establish

Established: on `f548fc6`, one guard-absent `LOWER_TO_REST` on B4 in the
simulator produced complete contact evidence, no contact, no displacement,
and a verified parked tail at REST; the setup traversal that put the arm at
PRESENT did the same. Not established: anything about the physical hand,
any other board or route, repeatability (n = 1 each), or a production
margin. The matrix remains blocked on manifest `contacts_tracked` +
`e1_identity` enforcement, #112, N1–N9, A1, A2, A5.
