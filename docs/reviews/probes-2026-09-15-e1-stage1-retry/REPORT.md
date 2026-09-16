# E1 Stage 1 retry (B4 validation set) — session report, 2026-09-15 (UTC 2026-09-16 05:21–05:45)

Authorized scope: Stage 0 + Stage 1 on `B4_pool_box_1_r2c3` only (three
flights, two recorded setup traversals, three scheduled resets, the
specified parked recordings), under
`outputs/checklist-2026-09-15-stage1-retry-post-merge.md`; any failed
gate, non-zero exit, contact or B4 disturbance ends the session; no
automatic retry/recovery/additional reset/expanded matrix. Simulator only.
**No code, guard, margin, route, tolerance or timing limit was changed. No
production margin is inferred from anything below.**

Code: `main` merge commit **`eb34238f6aae6bb9cc14dd901abc31a677e9b0d5`**
(PR #120), checked out detached in the existing execution worktree
`~/reachy-1-2-sim-issue116` (clean before, during and after). Server
provenance stamp: `code_sha == eb34238…`, `code_sha_dirty: false`,
`started_at 2026-09-16T05:22:29Z`, `contacts_tracked: true`,
`scene_sha256 == 818d8e57…` (the same B4 YAML as 2026-09-15).

## 1. Outcome in one line

**The three-cycle B4 validation set completed: 3 of 3 flights flown to
their end posture, 2 of 2 setup traversals parked, 3 of 3 scheduled
resets verified from the server record, 4 parked recordings valid. Zero
contacts and zero object disturbance (max displacement 8.4e-9 m) on every
one of the 9 recordings. No STOP, no non-zero exit at any gate.**

## 2. Per-cycle results

| step | gate | result |
|---|---|---|
| Stage 0 — fresh server (`eb34238`), B4, recording | manifest `code_sha`/`dirty:false`/`contacts_tracked`/scene sha all as required; `/status` `mujoco-remote`, `frames_stale` false; all 10 container services RUNNING incl. scene-marker-publisher; settled 67.5 s at the keyframe (`r_wrist_roll` 39.5°, `r_gripper` −36.2°) | **pass** |
| Stage 0 — parked recording (`…T052354Z`, 3 s) | recorder 0; linker 0, `contacts=0`, 154/154, covered/contiguous, `settled_pose_check` ok, displacement 4e-10 m; tail vs `HOME`: gross 0.0°, spread 0.2° (gripper −38.1 vs −45 → `NO`, the §5 keyframe mismatch, recorded) | **pass** |
| S1-a C1 reset #1 | `{"type":"reset"}` in `commands.jsonl` (0→1), `sim_step` 67990→550, ack gen 1 | **pass** |
| S1-a C2 settle | quiescent 66.2 s after reset, keyframe pose (39.4° / −36.0°) | **pass** |
| S1-a C3 parked (`…T052946Z`) | recorder 0; linker 0, `contacts=0`, 154/154; gross 0.0°, spread 0.0°; gripper `NO` (§5) | **pass** |
| S1-a binding (cells 1–2) | kernel tree == `GENERATED_AT_SHA`; identity ok, joint agreement ≈ 0°; `provenance_ok` (code_sha == GENERATED_AT_SHA == `eb34238`, clean, descends from `3d7fc74`, started after 02:20:18Z) → `binding_ok_S1a` | **pass** |
| S1-a C4/C5 setup `RAISE_TO_SIDE` (`…T053206Z`, 59.6 s window) | `check_start(PLACE_ROUTE)` (True, ""), posture `home`; baseline `cmd_seq` int; `turn_on` → compliance applied on all 8 joints in **0.055 s** (#116 fix effective); route `returned` in **33.2 s** (nominal 33.0, no retry passes); recorder 0; linker 0, `contacts=0`, 2984/2984; displacement 7.9e-9 m; **`PARKED_AT_PRESENT=yes`** (gross ≤ 2.6°, spread 0.0°, gripper −45.0) | **pass** |
| S1-a C6/C7 flight `LOWER_TO_REST` (`…T053354Z`, 37.4 s) | start check `PRESENT` ok; compliance 0.053 s; phases REST_SHUT, REST; `returned` in **6.7 s** (nominal 6.6); recorder 0; linker 0, `contacts=0`, 1873/1873; displacement 5.0e-9 m; **`PARKED_AT_REST=yes`** (gross ≤ 0.3°) | **pass** |
| S1-b C1 reset #2 | reset entry 1→2, `sim_step` 304430→540, ack gen 2 | **pass** |
| S1-b C2 settle | quiescent 15.3 s after reset; pose all ≈ 0° incl. `r_wrist_roll` 0.05°, `r_gripper` −0.0° — see §3 | **pass** |
| S1-b C3 parked (`…T053553Z`) | recorder 0; linker 0, `contacts=0`, 154/154; gross 0.0°, spread 0.0°; gripper −0.0 vs −45 → `NO` (§5 condition 1 met: gross/spread within limits) | **pass** |
| S1-b binding | identity ok, `provenance_ok` → `binding_ok_S1b` | **pass** |
| S1-b C7 flight `PLACE_ROUTE` (`…T053749Z`, 64.1 s) | `check_start(PLACE_ROUTE)` ok, posture `home`; compliance 0.052 s; all 11 phases GRIP_SHUT…REST; `returned` in **34.4 s** (nominal 33.3 incl. settles); recorder 0; linker 0, `contacts=0`, 3208/3208; displacement 8.4e-9 m; **`PARKED_AT_REST=yes`** (gross ≤ 0.6°, gripper −44.7) | **pass** |
| S1-c C1 reset #3 | reset entry 2→3, `sim_step` 130670→560, ack gen 3 | **pass** |
| S1-c C2 settle | quiescent 16.0 s; pose all ≈ 0° (§3) | **pass** |
| S1-c C3 parked (`…T053936Z`) | recorder 0; linker 0, `contacts=0`, 153/153; gross 0.0°, spread 0.0°; gripper `NO` (§5) | **pass** |
| S1-c binding | identity ok, `provenance_ok` → `binding_ok_S1c` | **pass** |
| S1-c C5 setup `PLACE_ROUTE` (`…T054132Z`, 64.1 s) | start ok; compliance 0.055 s; 11 phases; `returned` in **33.5 s**; recorder 0; linker 0, `contacts=0`, 3208/3208; displacement 8.4e-9 m; **`PARKED_AT_REST=yes`** (gross ≤ 0.7°) | **pass** |
| S1-c C7 flight `LIFT_TO_PRESENT` (`…T054350Z`, 34.1 s) | start check `REST` ok, posture `rest`; compliance 0.051 s; phase PRESENT; `returned` in **3.4 s** (nominal 3.3); recorder 0; linker 0, `contacts=0`, 1709/1709; displacement 4.6e-9 m; **`PARKED_AT_PRESENT=yes`** (gross ≤ 3.6°, spread 0.0°, gripper −45.1) | **pass** |
| C8 ledger | `ledger.csv`, 9 rows, every gate line pasted | done |
| Shutdown | final `/status` healthy; `docker compose down` (container + network removed); native server SIGTERM; ports 8765/50051 free | done |

`torn_trailing_line`: none on any recording. Wrist residuals (info only)
≤ 0.4° everywhere; no droop class assigned — every route completed at
its own tolerance and no leg stopped short.

## 3. Observations (recorded, not acted on)

1. **Post-reset start pose differs between cycles.** After reset #1 the
   arm settled to the gravity keyframe (`r_wrist_roll` 39.4°, `r_gripper`
   −36.0°) in 66 s, as on 2026-09-15 — the motors were still compliant
   from the fresh start. After resets #2 and #3 the arm was **stiff**
   (`compliant: false` on all joints, left on by S1-a's `turn_on`; nothing
   in scope turns motors off) and held the reset's zero goals: every joint
   ≈ 0°, including `r_wrist_roll` 0.05° and `r_gripper` −0.0°, settling in
   15–16 s. Both states satisfy the approved §5 rule (gross residuals ≤ 8°
   and spread ≤ 1° vs `HOME`, `check_start` True, gripper line `NO`
   expected), and the parked recordings show zero contact in the pocket
   in either state, so no gate distinguished them. It is a real difference
   in start conditions between S1-a and S1-b/S1-c (the gripper starts 36°
   more closed and the wrist 39° different) and belongs in the keyframe
   decision (proposal plan D item), not in this session.
2. **Container-name conflict on the first `start_sim.sh`** — same as the
   2026-09-15 session's `aborted_start_1`: the *exited* container from the
   `reachy-1-2-sim-stage1` compose project still held the name. The
   native server that did start (its run dir, manifest — already stamped
   `code_sha eb34238`, `dirty: false` — and logs) is preserved under
   `control/aborted_start_1/`; that server was stopped, the stale exited
   container removed (`docker rm`; it contained no evidence), and the
   second start succeeded. No recording, reset or motion happened before
   the second start. This session's `docker compose down` removes its own
   container so the next session does not hit it.
3. **C3 parked recordings** were run with `control/tools/parked.sh`
   (the `leg.sh` chain step by step at `--duration 3`, tail vs `HOME`,
   mode `start`) because the merged `leg.sh` reads `DUR` by leg name from
   `plan_<cycle>.json` and parked recordings have no plan leg. Flown legs
   all used the merged `leg.sh` with mode `end` (7th argument omitted).
4. Every gate that was previously a false-STOP risk behaved: no `mystery
   STOP`, no `code_sha` refusal, no `cmd_seq` baseline failure, no
   compliance timeout — `require_compliance` returned in ≤ 0.055 s on all
   five legs, against the 3.0 s budget.
5. Elapsed route times (33.2 / 6.7 / 34.4 / 33.5 / 3.4 s) are the nominal
   waypoint chains plus settles with **no** convergence-retry passes;
   the one-waypoint retry allowance in each budget went unused. That is
   an observation about these five flights only — not a margin.

## 4. What this does and does not establish

- The B4 validation set is **validated** under the approved procedure:
  every gate in proposal §4 passed on every cycle, every artefact in §10
  is retained. Per §6, these samples count toward B4's core totals only if
  the procedure is not changed afterwards; otherwise they are validation
  data.
- Not established: any realised-clearance number as a route margin; any
  claim about the keyframe start (§3.1); anything about boards B1/B2,
  extension routes, or Stage 2.

## 5. Preserved / hygiene

- Historical evidence: `~/reachy-1-2-sim-e1` and `~/reachy-1-2-sim-stage1`
  clean before and after; `probes-2026-09-15-e1-stage1-b4/SHA256SUMS`
  re-verified (48/48 OK).
- Primary checkout `~/reachy-1-2-sim`: `notebooks/place_objects.ipynb`
  (modified) and `notebooks/demo_pick_place.py` (untracked) untouched.
- Execution worktree `~/reachy-1-2-sim-issue116`: detached at `eb34238`,
  `git status --porcelain` empty throughout (recorder logs went to the
  gitignored `runs/`; copies are in `recorder_logs/` here).
- No new worktree created. Services started for this session were shut
  down (§2 last row).

## 6. Artefacts (this directory: `~/e1-stage1-retry-2026-09-15/`)

- `REPORT.md` (this file), `ledger.csv` (9 rows), `SHA256SUMS` (every file
  except itself).
- `e1_server_runs/run_20260916_052229/` — `manifest.json`, `states.jsonl`
  (67 781 lines), `commands.jsonl` (4 637 lines, 3 reset entries).
- `recorder_logs/` — 9 recorder logs (schema 3) + 9 `.link.json` sidecars.
- `e1_stage1_S1{a,b,c}.ipynb` as generated and `.executed.ipynb` as run;
  `plan_S1{a,b,c}.json`.
- `control/` — every gate's stdout (`recorder_*.log`, `linker_*.txt`,
  `tailcheck_*.txt`), `reset_{1,2,3}.txt`, `settle_*.json`, `binding_ok_*`,
  `go_*`, `*_done`, `notebook_exec_*.log`, `start_sim.log`,
  `status_stage0.json`, `supervisor_stage0.txt`, `manifest_check.txt`,
  `plan_check.txt`, `host_env_check.txt`, `pre_start_check.txt`,
  `shutdown.txt`, `aborted_start_1/` + `aborted_start_1.txt`, and
  `tools/` (the exact `leg.sh`, `reset.sh`, `parked.sh`, `plan.py`,
  `make_cycle_notebook.py`, `provenance.py`, `README.md` from `eb34238`
  and `settle_wait.py` copied from the 2026-09-15 evidence).
- `board/B4_pool_box_1_r2c3.yaml` + `sha256.txt`; `native_server.log`.
