# E1 Stage 1 retry — B4 validation set, 2026-09-15 (completed)

Session report: `REPORT.md`. One row per recording: `ledger.csv`.
Reproduction: `REPRODUCTION.md`. Hashes of every original file:
`SHA256SUMS` (computed over the full evidence tree, including the raw
files that are **not** in Git — see "What is in Git").

- `summaries/<recording>.summary.json` — per-recording extract: recorder
  clearance report (planned vs realised, both hand models), identity,
  settled-pose check, displacement, contacts + window, linker and
  tail-check stdout, the notebook's leg record, start/end wrist and
  gripper state, per-link clearance.
- `summaries/clearance_by_link.{md,json}` — planned vs realised clearance
  by recording, hand model (`tube`, `shells`), link and object; motion
  recordings split into pre-motion / moving / parked-tail segments.
- `summaries/initial_state_by_leg.txt` — compliance, gripper and wrist
  state at each leg's start, from the server record.
- `control/` — every gate's stdout, reset verifications, settle output,
  notebook markers (`binding_ok_*`, `go_*`, `*_done`), notebook execution
  logs, start/shutdown logs, the aborted first start, and `tools/` (the
  exact scripts run).
- `e1_stage1_S1{a,b,c}.ipynb` as generated, `.executed.ipynb` as run;
  `plan_S1{a,b,c}.json` (durations used).
- `board/` — B4 YAML as flown + sha256.
- `e1_server_runs/run_20260916_052229/manifest.json` — the server's
  provenance stamp (`code_sha eb34238…`, `code_sha_dirty: false`).

## What is in Git and what is not

In Git (this directory, ~0.9 MB): everything above. **Not in Git** (raw,
large; originals preserved, see below): `e1_server_runs/…/states.jsonl`
(67 781 lines, 380 MB) and `commands.jsonl`; the 9 recorder logs and their
9 `.link.json` sidecars (`recorder_logs/`, 15 MB); `native_server.log`;
the aborted start's `states.jsonl`/`commands.jsonl`. All of them are
listed with sha256 in `SHA256SUMS`, so any copy can be verified.

Originals: `~/e1-stage1-retry-2026-09-15/` on the operator's machine
(outside every git checkout). Archive: see `REPRODUCTION.md` §Archive.

## Result

3/3 flights, 2/2 setups, 3/3 resets, 4 parked recordings; every gate
passed; zero contacts and zero disturbance on all 9 recordings. This is
**successful validation of the procedure** (gates, tooling, provenance,
timing) on one board. It is **not** proof of a production margin, and
not evidence of general collision avoidance — see `REPORT.md` §4 and
the Stage 2 decision note.
