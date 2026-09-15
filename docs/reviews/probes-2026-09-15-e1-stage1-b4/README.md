# E1 Stage 1 — B4 validation set, 2026-09-15 (stopped at S1-a setup)

- `REPORT.md` — per-cycle results and cause; `ledger.csv` — one row per recording.
- `summaries/*.summary.json` — per-recording extracts (identity, settled check, displacement, contacts, window, tail-check and linker stdout).
- `recorder_logs/` — the three recorder logs (schema 3) and their `.link.json` sidecars.
- `e1_server_runs/run_20260915_211004/` — `manifest.json` (committed), `states.jsonl` 160 MB and `commands.jsonl` (ignored, sole copies in this worktree; hashed in `SHA256SUMS`).
- `control/` — every gate's stdout (`recorder_*.log`, `linker_*.txt`, `tailcheck_*.txt`), reset verification, settle output, notebook markers, `stop`, and `tools/` (the exact scripts run).
- `e1_stage1_S1a.ipynb` / `.executed.ipynb` — the cycle notebook as written and as run. `S1b`/`S1c` notebooks were generated but never executed.
- `board/` — B4 YAML as flown + sha256. `native_server.log`, `start_sim.log`, `status_stage0.json`.
- Defect: compliance command lost to the server's single-slot command coalescing — see `REPORT.md` §3 and the linked issue.
