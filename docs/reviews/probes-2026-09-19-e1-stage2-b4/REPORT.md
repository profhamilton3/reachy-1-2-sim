# E1 Stage 2 — B4 only, session s1 — report (UTC 2026-09-19 04:39–07:33)

Authorized scope: the B4-only Stage 2 plan
(`IITG-Reachy-Project/outputs/e1-stage2-b4-execution-plan-2026-09-18.md`) at `main`
`e6c30b74e09c38e616dee0819a7e205d12c5853e`, policy A, strict stop rule (any failed gate,
contact or disturbance ends the session; no retry, recovery, extra reset, other board, or
hardware). Simulator only. **No code, guard, margin, route, tolerance or timing was changed.
No production margin is inferred from anything below — this is one board, one object layout,
open-loop measured routes.**

Code: execution worktree `~/reachy-1-2-sim-stage2`, detached at `e6c30b7`, `git status
--porcelain` empty before, during and after (recorder logs went to the gitignored `runs/`).
Server run `e1_server_runs/run_20260919_043924/`: `code_sha == e6c30b7…`, `code_sha_dirty:
false`, `contacts_tracked: true`, `scene_sha256 == 818d8e57…` (the same B4 YAML as Stage 1).
Container: `REACHY_PANEL_EXECUTOR=0`, backend `mujoco-remote`; the only native-server clients
were `fake_reachy_server.py` (the SDK bridge) and `camera_server.py` (read-only frames /
`/status`); no host SDK client other than the per-cycle notebook kernels.

## 1. Outcome in one line

**18 of 18 cycles completed on B4: 18/18 flights and 12/12 recorded setups flown to their end
posture, 18/18 scheduled resets verified from the server record, 18/18 parked recordings
classified stiff-zero, Stage 0 armon accepted. Zero contacts and zero object disturbance
(max displacement 8.5e-9 m, the sim's noise floor) on all 49 recordings. No STOP, no non-zero
exit at any gate. `ledger.csv`: 49 rows, 49 pass.**

## 2. Counts and timing

| item | value |
|---|---|
| Flights | LOWER_TO_REST ×6, PLACE_ROUTE ×6, LIFT_TO_PRESENT ×6 — all `returned` |
| Setups | RAISE_TO_SIDE ×6 (before LOWER_TO_REST), PLACE_ROUTE ×6 (before LIFT_TO_PRESENT) — all `returned` |
| Resets | gen 1–18, each `ack==gen`, reset count +1, `sim_step` restarted (`control/reset_<n>.txt`) |
| Settles after reset | 15.1–17.6 s, `max_spread_deg` 0.0, pose all ≈ 0° incl. `r_wrist_roll` 0.05°, `r_gripper` −0.0° (`control/settle_*.json`) |
| Start variant | `stiff-zero` on all 18 cycles (`control/start_variant_*.json`; re-verified by each binding cell) |
| Stage 0 | compliant fresh arm settled to keyframe-sag in 67 s (39.45° / −36.17°); armon `turn_on` applied on all 8 joints in 0.055 s; `STAGE0_INIT_OK=yes`; gate accepted |
| Elapsed route times | RAISE_TO_SIDE 33.21–33.24 s; LOWER_TO_REST 6.69–6.72 s; PLACE_ROUTE 33.55–33.62 s; LIFT_TO_PRESENT 3.38–3.41 s — nominal chains plus settles, **no convergence-retry pass on any of the 30 motion legs**; the one-waypoint retry allowance in every budget went unused |
| Compliance after `turn_on` | ok on all 30 motion legs against the 3.0 s budget |
| Tail checks (`end` mode) | `PARKED_AT_PRESENT=yes` ×12, `PARKED_AT_REST=yes` ×18; gross residual ≤ 3.6°, spread 0.0° |
| Experiment gate | `EXPERIMENT_ACCEPTED=yes` ×49; `contacts=[]`, N/N contact evidence, `covered`, `seq_contiguous`, `reset_in_window=False`, board coverage on every recording |
| Wall time | server up 04:39:24Z → down 07:33Z (2 h 54 min including a 30-min host-sleep gap, §4.1); Stage 0 + set r1 ≈ 25 min; sets r2–r6 ≈ 8–10 min each, growing with `states.jsonl` size |
| Data | `states.jsonl` 431 881 lines (1.81 GB), `commands.jsonl` 27 675 lines (18 reset entries); 49 recorder logs + 49 `.link.json` (44 MB) |

## 3. Clearance (observations, not margins)

`summaries/clearance_pool_box_1_by_recording.{txt,csv}` — the recorder's own report, worst
clearance to `pool_box_1` over each whole recording, both hand models, aperture measured on
every sample (`realised_aperture_assumed_samples` empty on all 49):

| role / route | n | realised shells (cm) | planned shells (cm) | tube realised (cm) |
|---|---|---|---|---|
| setup RAISE_TO_SIDE | 6 | 5.45 – 5.69 | 6.26 | −3.82 … −3.91 |
| flight LOWER_TO_REST | 6 | 6.39 – 6.44 | 6.46 | −3.74 … −3.78 |
| flight PLACE_ROUTE | 6 | 5.36 – 6.05 | 6.26 | −3.82 … −3.93 |
| setup PLACE_ROUTE | 6 | 5.50 – 5.73 | 6.26 | −3.76 … −3.98 |
| flight LIFT_TO_PRESENT | 6 | 6.20 – 6.43 | 6.46 | −3.74 … −4.00 |
| parked / armon (HOME) | 19 | 37.72 | — | +37.72 |

Same picture as Stage 1: realised tracks planned to within ~0.1–0.9 cm below on the shells
model; the tube model reads negative at REST with **zero** physics contact (the known
conservative tube). Per Stage 1 §4 and the decision note: these are observations of 30 motion
legs on one board with one layout against a capsule model; they are not a margin to design
against.

### 3.1 Planned − realised by route and link, across repetitions

`summaries/clearance_by_link.md` / `clearance_by_link_aggregate.json` /
`clearance_by_link_findings.txt` (`control/tools/s2_clearance_by_link.py`, same per-link
geometry as Stage 1's `clearance_by_link.py`): every link, both hand models, each sample at its
own measured aperture; setups and flights evaluated on their **moving** segment (joint velocity
> 2°/s) with the parked tail reported separately; parked recordings on the whole 3 s. Closest
link per group, cm; delta = planned − realised (positive = closer than planned), mean over the
repetitions:

| route | role | closest link (shells) | planned | realised min..max | delta mean, all 18 | delta mean, excl. c-r1 |
|---|---|---|---|---|---|---|
| RAISE_TO_SIDE | setup (n=6) | forearm | +6.26 | +5.45..+5.69 | +0.68 | +0.68 |
| LOWER_TO_REST | flight (n=6) | forearm | +6.46 | +6.39..+6.45 | +0.04 | +0.04 |
| PLACE_ROUTE | flight (n=6) | forearm | +6.26 | +5.36..+6.05 | +0.66 | +0.66 |
| PLACE_ROUTE | setup (n=6) | forearm | +6.26 | +5.50..+5.73 | +0.65 | +0.68 |
| LIFT_TO_PRESENT | flight (n=6) | forearm | +6.46 | +6.20..+6.43 | +0.10 | +0.07 |
| (tube hand, all motion groups) | | hand | −3.72 | −4.00..−3.69 | 0.00..+0.14 | 0.00..+0.14 |
| parked (n=18, HOME) | | upper_arm | — | +37.72 | (no corridor to compare) | — |

Next-closest shells links on every motion group: `wrist_ball` (+5.57..+6.87 realised), then
`thumb` (+6.53..+7.33) and `thumb_pad` (+7.23..+7.59); `finger` ≥ +10.4; `upper_arm` ≥ +19.2. Parked-tail minima equal the REST
posture itself for legs ending at REST (+6.26..+6.39 forearm), as in Stage 1.

Readings:
1. The two long routes through HOVER → REST_SHUT → REST (RAISE_TO_SIDE setup, PLACE_ROUTE
   setup/flight) run ~0.65 cm inside their planned corridor on the forearm; the two short legs
   (LOWER_TO_REST, LIFT_TO_PRESENT) ~0.05–0.10 cm. Consistent with Stage 1's −0.7..−1.0 vs
   −0.1..−0.3 split (tracking lag along the corridor, not a posture excursion).
2. **Sleep-affected cycle S2-B4-c-r1** (host slept after its setup recording, before its
   flight): its setup/flight deltas (+0.53 / +0.26) sit inside the other five c-repetitions'
   range (+0.52..+0.76 / +0.03..+0.20; c-r2's flight is +0.20, c-r1's +0.26 is the largest but
   by 0.06 cm). Excluding it changes every group mean by ≤ 0.03 cm and no minimum, maximum,
   ordering, or sign. The finding does not depend on it.
3. **Repetitions are not independent evidence.** The simulator is deterministic and every
   cycle starts from the same reset state (stiff-zero, verified) and flies the same open-loop
   waypoint chain; the spread across six repetitions (≤ 0.7 cm on PLACE_ROUTE's forearm,
   ≤ 0.25 cm elsewhere) measures the procedure's repeatability on this board, with the small
   variation coming from settle/tracking timing. Six near-identical runs are one sample of the
   route-on-B4 behaviour, not six samples of "the arm is reliable"; they say nothing about other
   layouts, perturbations, or the real robot.

## 4. Deviations and observations (recorded, not acted on)

1. **Host idle sleep during cycle c-r1** (~04:53–05:22Z). The `S2-B4-c-r1-setup` motion,
   recording and sidecar were complete (04:52:16/04:52:20Z) before the sleep; the recorder's
   clearance report and the linker/gate/tail chain ran after wake. Verified from
   `states.jsonl`/`commands.jsonl` (`control/sleep_gap_check_c-r1.txt`): one clock gap, pose
   identical across it (Δ 0.0°), all joints `compliant=False`, `cmd_seq` 4473 unchanged, zero
   commands in the window, no reset. The bridge websocket reconnected on wake; the container
   did not restart (supervisor uptime continuous). The c-r1 flight then passed
   `check_start(REST)`, compliance, identity, gate and tail. `caffeinate -i -s -d` ran from
   05:24Z to shutdown; no further sleep. Recorded as a deviation from the plan's environment
   assumptions, not a gate failure; the c-r1 rows are flagged in this report, not in the
   ledger's pass/fail column.
2. **Chain latency grows with the run**: the linker/recorder post-processing read the whole
   `states.jsonl` (1.8 GB by r6), so a cycle's chain took ~4 min longer at r6 than at r1.
   Harmless here; a per-board server run longer than ~3 h would want a bounded tail read.
3. **Manifest counters unfinalized on SIGTERM** (`state_count: 0`, `total_steps: None`) —
   identical to the Stage 1 retry manifest; pre-existing recorder behaviour, not a gap in the
   evidence (`states.jsonl`/`commands.jsonl` are complete and the linker verified coverage
   per recording).
4. The parked tail check vs `HOME` in `start` mode reads `PARKED_AT_HOME=NO` on the gripper
   criterion (0° vs −45°) on every stiff-zero parked recording, as it did in Stage 1 S1b/S1c —
   informational by design; gross residuals 0.0°.
5. No droop class other than D0 was observed: every route completed at its own tolerance
   (wrist residuals info-only, ≤ 0.4° in tail lines).

## 5. What this does and does not establish

- The Stage 2 procedure (policy A stiff-zero starts, repetition-unique identities, per-cycle
  start-variant + fresh-compliance gates, experiment gate with `reset_in_window`/board
  coverage) executed 18 cycles on B4 without a single refusal, and every artefact the plan
  lists is retained. These 18 flights + 12 setups are B4's Stage 2 core samples (decision
  note §5); Stage 1's samples remain validation data and are not pooled.
- Not established: any realised-clearance number as a route margin; anything about B1/B2
  (not flown, not authorized); general collision avoidance; behaviour under any change to
  routes, tolerances, budgets, or guards.

## 6. Artefacts (this directory, `~/e1-stage2-B4-s1-2026-09-18/`)

`REPORT.md`, `REPRODUCTION.md`, `README.md`, `ledger.csv` (49 rows), `SHA256SUMS`;
`e1_server_runs/run_20260919_043924/` (`manifest.json`, `states.jsonl`, `commands.jsonl`);
`recorder_logs/` (49 logs + 49 sidecars); 19 notebooks as generated + 19 `.executed.ipynb`;
19 `plan_*.json`; `control/` (every gate's stdout, `reset_1..18.txt`, `settle_*.json`,
`start_variant_*.json`, `binding_ok_*`, `go_*`, `*_done`, `notebook_exec_*.log`,
`host_env_check.txt` with the resolved e1venv dependencies, `pre_start_check.txt`,
`start_sim.log`, `manifest_check.txt`, `status_stage0.json`, `supervisor_stage0.txt`,
`checkpoint_1.txt`, `sleep_gap_check_c-r1.txt`, `final_status.txt`, `shutdown.txt`,
`e1_stage2_sessions.json`, `tools/` = the exact packaged scripts from `e6c30b7` plus
`settle_wait.py` and the three session wrappers `s2_cycle_pre.sh`, `s2_leg.sh`,
`s2_ledger.py`); `board/B4_pool_box_1_r2c3.yaml` + `sha256.txt`; `native_server.log`;
`summaries/`.
