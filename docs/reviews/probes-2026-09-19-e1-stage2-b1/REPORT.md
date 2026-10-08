# E1 Stage 2 — B1 only, session s1 — report (UTC 2026-09-19 09:03–11:21)

Code `profhamilton3/reachy-1-2-sim` `e6c30b74e09c38e616dee0819a7e205d12c5853e` (detached worktree
`~/reachy-1-2-sim-stage2`, clean, no `.env`), the same reviewed code and gates as the B4 session.
Board **B1** `scenes/e1_boards/B1_evidence.yaml` sha256
`f577fc47616ab1b4d68e0173c1adec6d1110039372f0f68bed31e7f7d6715346` — two board objects,
`soda_can` at r1c1 and `foam_block` at r2c3. Plan followed:
`IITG-Reachy-Project/outputs/e1-stage2-b1-execution-plan-2026-09-19.md` (D1–D5 as authorized).
Server run `e1_server_runs/run_20260919_090342` (manifest `code_sha` = e6c30b7…, `code_sha_dirty:
false`, `contacts_tracked: true`, `scene_sha256` = f577fc47…). Container `REACHY_PANEL_EXECUTOR=0`;
clients on :8765 were the bridge (`fake_reachy_server.py`) and `camera_server.py` only. Policy A
(Stage 0 armon before reset 1). `caffeinate -i -s -d` (pid 48760) from 09:03:27Z, **before**
`start_sim.sh`, held until the archive was verified.

## 1. Outcome in one line

**17 of 18 cycles flown, 46 of 49 planned recordings, every flown gate passed: zero contacts and
zero board disturbance on all 46 recordings** (max displacement `soda_can` 1.0e-6 m, `foam_block`
9.2e-9 m, tolerance 1e-3 m). The session **ended under the stop rules at reset 18** (cycle
`S2-B1-c-r6`): `reset.sh`'s host-side verification could not parse the last line of the growing
2.4 GB `states.jsonl` and wrote `control/stop`; the server record shows the reset itself was
executed (§4). No retry, no cycle redo, no extra reset; c-r6 was not flown.

## 2. Counts and timing

| item | planned | done |
|---|---|---|
| Stage 0 armon leg (`INIT`, end mode) | 1 | 1, `STAGE0_INIT_OK=yes`, compliance ok 8/8 joints |
| scheduled resets (gen 1–18) | 18 | 18 executed and recorded by the server; **17 verified by `reset.sh`**, #18 verification failed → stop |
| parked recordings (start variant) | 18 | 17, all `stiff-zero` |
| setups (a: RAISE_TO_SIDE, c: PLACE_ROUTE) | 12 | 11 (6 + 5) |
| flights (a: LOWER_TO_REST, b: PLACE_ROUTE, c: LIFT_TO_PRESENT) | 18 | 17 (6 + 6 + 5) |
| ledger rows / pass | 49 | 49 rows: **46 pass**, 3 `NOT FLOWN (session stopped)` (`S2-B1-c-r6-{parked,setup,flight}`) |
| contacts / disturbance > 1 mm | 0 / 0 | **0 / 0** on every flown recording |
| `binding_ok` (identity + provenance + start-variant + compliance) | 18 | 17/17 attempted; c-r6 not reached |

Motion elapsed (notebook `elapsed_s`, budgets unchanged): RAISE_TO_SIDE setup 33.23–33.85 s (n=6),
LOWER_TO_REST 6.70–6.72 s (6), PLACE_ROUTE flight 33.57–34.42 s (6), PLACE_ROUTE setup 33.59–34.46 s
(5), LIFT_TO_PRESENT 3.37–3.38 s (5) — the same as B4. Settles after reset: 15.0–15.8 s at spread
0.0° (policy A; Stage 0 keyframe-sag settle 65.6 s).

Wall clock (`summaries/cycle_wall_times.txt`): sets r1 22.3* / r2 14.7 / r3 18.5 / r4 23.5 / r5
30.0 / r6 22.6 (a+b only) min; server uptime 137 min (cap 240). *r1 includes ~10 min of operator
idle while the a-r1 pre-wrapper's output pipe was held by the notebook subshell (harness
artefact, no sim event; later wrappers wrote to files). Per-leg post-recording chain grew from
126 s (r1) to 461 s (b-r6), as predicted from B4: the linker reads the whole `states.jsonl`
(2.41 GB at the end). The plan's estimate (sets 13/14/18/23/29/38 min) held within ~1 min per set.

## 3. Clearance (observations, not margins)

Pure geometry from the recorded joint samples against the B1 YAML poses, per link, each sample at
its own measured aperture, moving segment only for setups/flights
(`summaries/clearance_by_link.md`, `clearance_by_link_aggregate.json`; recorder's own worst-case
report per recording in `summaries/clearance_by_recording_recorder.txt`). The closest object on
every flown route is `foam_block` (r2c3); `soda_can` (r1c1) is never closer than 26.5 cm.

### 3.1 Planned − realised to `foam_block`, closest shells link (`forearm`), across repetitions

| route | role | n | planned | realised min..max | Δ mean (min..max) |
|---|---|---|---|---|---|
| RAISE_TO_SIDE | setup | 6 | +4.48 | +3.79..+3.96 | +0.58 (+0.52..+0.69) |
| PLACE_ROUTE | flight | 6 | +4.48 | +3.78..+3.89 | +0.64 (+0.58..+0.70) |
| PLACE_ROUTE | setup | 5 | +4.48 | +3.74..+3.98 | +0.62 (+0.50..+0.73) |
| LOWER_TO_REST | flight | 6 | +4.79 | +4.48..+4.72 | +0.13 (+0.06..+0.31) |
| LIFT_TO_PRESENT | flight | 5 | +4.79 | +4.57..+4.78 | +0.13 (+0.01..+0.21) |
| parked (HOME), closest link `upper_arm` | parked | 17 | — | +36.18 | — |

Next-closest shells links on the PLACE_ROUTE/RAISE_TO_SIDE corridor: `wrist_ball` +4.07..+4.40
(planned +5.05, Δ ≈ +0.8), `thumb` +4.85..+5.09 (planned +5.40, Δ ≈ +0.4). Tube hand model:
−5.72..−5.97 cm to `foam_block` on every corridor and at REST (planned −5.70) with **zero physics
contact** — the conservative capsule overlapping, as on B4 (−3.7..−4.0 there).

Readings:
- The realised corridor over `foam_block` runs **~0.6 cm inside the planned corridor** on the
  three PLACE_ROUTE/RAISE_TO_SIDE legs (worst single recording +3.74 cm, `S2-B1-c-r3-setup`),
  and ~0.1 cm inside on LOWER_TO_REST/LIFT_TO_PRESENT. This is the same offset B4 showed
  (+0.65..+0.68 on the same links), on a corridor planned 1.78 cm tighter (4.48 vs 6.26).
- Repetition spread of the realised worst clearance within a route is 0.11–0.24 cm (n=5–6).
  **Deterministic simulator, same verified reset state, same open-loop route**: the repetitions
  measure repeatability of this procedure on this board, not independent trials, and do not
  establish a production margin or general collision avoidance.
- No sleep-affected cycle in this session (caffeinate from before server start), so there is
  no exclusion analysis; the B4-style "excluding …" table is not applicable.

## 4. Deviations and observations (recorded, not acted on)

1. **Session stop at reset 18 (`S2-B1-c-r6`), `control/stop` = "STOP: reset 18 not verified".**
   `reset.sh` verifies a reset from the server's own record: ack == gen, `commands.jsonl` reset
   count +1, and `sim_step` after < before, where `sim_step` is read as `tail -1 states.jsonl |
   json.loads`. For reset 18 the ack (18) and count (17→18) were correct, but the `tail -1` read
   returned two records of the growing file (JSON "Extra data: line 2 column 1 (char 5862)"), so
   `step_now` was empty, the integer test failed, and the stop marker was written
   (`control/reset_18.txt`, `control/stop_analysis.txt`). Server-side (`commands.jsonl` #18 at
   sim_step 282875, `states.jsonl` shows the 18th `sim_step` drop 282870→10 at seq 408027) the
   reset was executed normally. Under the authorized rules a stop marker ends the session
   regardless of cause; nothing was retried. Deferred cleanup (not for this session): make the
   verification read robust to a growing file (the same growing-`states.jsonl` cost that drives
   the linker latency), or read `sim_step` from the last *complete* line.
2. The a-r1 pre-wrapper call held the harness's foreground pipe open for ~10 min after
   `binding_ok` (the nbconvert subshell inherited the pipe); the robot and simulator were idle
   and unaffected (binding at 09:06:06Z, leg started 09:16:38Z, gates all passed). Subsequent
   wrapper invocations redirected to files (`control/s2pre_*.txt`, `s2leg_*.txt`).
3. `manifest.json` counters are unfinalized on SIGTERM (`state_count: 0`), as on Stage 1 and B4.
4. No host sleep, no bridge reconnect, no container restart (container uptime continuous 2 h 17 min).

## 5. What this does and does not establish

Does: on B1's layout, under the reviewed code and gates, 17 complete cycles / 46 recordings flew
with zero contact and no board disturbance on either object, with the realised corridor ~0.6 cm
inside plan on the closest link, matching B4's offset. Provenance, identity, start variant and
compliance gates all held.

Does not: a production margin (this is one deterministic simulator configuration); anything about
B2 or hardware; a complete 6-repetition set for the c-shape (5 flown); reliability of `reset.sh`'s
verification against multi-GB state files (one failure in 36 resets across B4+B1).

## 6. Artefacts (this directory, `~/e1-stage2-B1-s1-2026-09-19/`)

`README.md`, `REPORT.md`, `REPRODUCTION.md`, `ledger.csv` (49 rows), `SHA256SUMS`,
`summaries/` (`clearance_by_link.md`, `clearance_by_link_aggregate.json`, `clearance_by_link_all.json`,
`clearance_by_recording_recorder.txt`, `checkpoint1_recorder_clearance.txt`, `cycle_wall_times.txt`),
`control/` (all gate outputs, `checkpoint_1.txt`, `stop`, `stop_analysis.txt`, `post_stop_check.txt`,
`shutdown.txt`, `wrapper_validation_offline.txt`, `tools/`), `board/`, `e1_stage1_*.ipynb` (+ 18
`.executed.ipynb`), `plan_*.json` (19, incl. the unflown c-r6), `e1_server_runs/run_20260919_090342/`
(`manifest.json`, `states.jsonl` 2.41 GB, `commands.jsonl`), `recorder_logs/` (46 logs + 46 sidecars),
`native_server.log`. Archive: see `REPRODUCTION.md`.
