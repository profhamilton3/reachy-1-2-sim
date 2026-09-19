# E1 Stage 2 — B1 only, session s1, 2026-09-19 (ended at reset 18 under the stop rules)

Session report: `REPORT.md`. One row per planned recording: `ledger.csv` (49 rows: 46 pass,
3 not flown). Reproduction: `REPRODUCTION.md`. Hashes of every file: `SHA256SUMS` (computed over
the full evidence tree, including the raw files that would not go into Git).

- `summaries/clearance_by_link.md` / `clearance_by_link_aggregate.json` — planned vs realised
  clearance per link, per B1 object, across repetitions (pure geometry from the recordings);
  `clearance_by_link_all.json` per recording; `clearance_by_recording_recorder.txt` — the
  recorder's own worst-case report per recording; `checkpoint1_recorder_clearance.txt`;
  `cycle_wall_times.txt`.
- `control/` — every gate's stdout, reset verifications (`reset_1..18.txt`), settle output,
  start-variant evidence, notebook markers, notebook execution logs, start/shutdown logs, the
  checkpoint record (`checkpoint_1.txt`), the stop marker and its analysis (`stop`,
  `stop_analysis.txt`, `post_stop_check.txt`), the offline wrapper validation
  (`wrapper_validation_offline.txt`), and `tools/` (the exact scripts run: repo scripts at
  e6c30b7 with `REPO_TOOL_SHA256SUMS.txt`, plus the session wrappers `s2_cycle_pre.sh`,
  `s2_leg.sh`, `s2_set.sh`, `s2_ledger.py`, `s2_clearance_by_link.py`).
- `e1_stage1_*.ipynb` as generated (19), `*.executed.ipynb` as run (18); `plan_*.json` (19).
- `board/` — B1 YAML as flown + sha256.
- `e1_server_runs/run_20260919_090342/` — `manifest.json` (provenance stamp `code_sha e6c30b7…`,
  `code_sha_dirty: false`, `scene_sha256 f577fc47…`), `states.jsonl` (2.41 GB), `commands.jsonl`.
- `recorder_logs/` — 46 recorder logs (schema 3) + 46 `.link.json` sidecars.

## Result

17/18 cycles: 17 flights, 11 setups, 18 resets (17 verified), 17 parked recordings (all
stiff-zero), Stage 0 armon accepted; every flown gate passed; zero contacts and zero disturbance
of `soda_can` and `foam_block` on all 46 recordings. The session ended when `reset.sh` could not
verify reset 18 from a host-side read of the growing `states.jsonl` (the server record shows the
reset executed) — `REPORT.md` §4; `S2-B1-c-r6` was not flown and nothing was retried. This is
Stage 2's B1 sample set under the validated procedure. It is **not** proof of a production margin
and not evidence about B2 or general collision avoidance.
