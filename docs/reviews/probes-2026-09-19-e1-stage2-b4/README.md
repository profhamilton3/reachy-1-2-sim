# E1 Stage 2 — B4 only, session s1, 2026-09-19 (completed)

Session report: `REPORT.md`. One row per recording: `ledger.csv` (49 rows).
Reproduction: `REPRODUCTION.md`. Hashes of every file: `SHA256SUMS` (computed over the
full evidence tree, including the raw files that would not go into Git).

- `summaries/clearance_pool_box_1_by_recording.{txt,csv}` — planned vs realised worst
  clearance to the board object per recording, both hand models (recorder's own report).
- `control/` — every gate's stdout, reset verifications, settle output, start-variant
  evidence, notebook markers, notebook execution logs, start/shutdown logs, the checkpoint
  record, the host-sleep verification, and `tools/` (the exact scripts run).
- `e1_stage1_*.ipynb` as generated, `*.executed.ipynb` as run; `plan_*.json` (durations used).
- `board/` — B4 YAML as flown + sha256.
- `e1_server_runs/run_20260919_043924/` — `manifest.json` (provenance stamp
  `code_sha e6c30b7…`, `code_sha_dirty: false`), `states.jsonl` (1.81 GB), `commands.jsonl`.
- `recorder_logs/` — 49 recorder logs (schema 3) + 49 `.link.json` sidecars.

## Result

18/18 flights, 12/12 setups, 18/18 resets, 18 parked recordings (all stiff-zero), Stage 0
armon accepted; every gate passed; zero contacts and zero disturbance on all 49 recordings.
One environment deviation (host idle sleep during c-r1's post-processing, verified harmless
from the server record) — see `REPORT.md` §4. This is Stage 2's B4 core sample set under
the validated procedure. It is **not** proof of a production margin and not evidence about
B1/B2 or general collision avoidance.
