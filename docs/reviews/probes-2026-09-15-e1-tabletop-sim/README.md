# E1 single simulator pilot — 2026-09-15 (B4_pool_box_1_r2c3, LOWER_TO_REST)

Simulator only. Executed per `outputs/e1-pilot-execution-scope-2026-09-15.md`
rev 3 (IITG-Reachy-Project) with the owner's written authorization of
2026-09-15. Code: `main` `f548fc6` in the worktree
`/Users/terrancehamilton/reachy-1-2-sim-e1` (tree clean at start). Full
narrative: `REPORT.md`; what is in Git, environment, defects, reproduction,
raw-log location and worktree status: `EVIDENCE.md`; hashes: `SHA256SUMS`;
per-recording extracts: `summaries/`.

One setup traversal (`RAISE_TO_SIDE`) and one flight (`LOWER_TO_REST`), no
retry, no recovery, no reset, no second board. Both returned; no contact,
no displacement, parked tails verified.

| file | what |
|---|---|
| `board/` | B4 YAML as flown + sha256 (`818d8e57…6ab083`, = manifest `scene_sha256`) |
| `e1_server_runs/run_20260915_173758/` | the one native-server run dir: `manifest.json` (in Git), `states.jsonl` (45 609 states, 256 MB, seq 1..45609 contiguous) and `commands.jsonl` (1 747 `joint_command`s) — **not in Git**, sole copies in the worktree, hashed in `SHA256SUMS` |
| `recorder_logs/` | three recorder logs (in Git) + `.link.json` sidecars: parked at the sim keyframe (3 s, sidecar in Git), setup traversal (120 s) and flight (60 s) — those two sidecars are **not in Git** (3.8 MB), hashed |
| `summaries/` | per-recording JSON extracts of each sidecar + tail check + notebook timings (in Git) |
| `EVIDENCE.md` / `REPORT.md` / `SHA256SUMS` | handoff, narrative report, hashes of every file |
| `control/recorder_*.log` | recorder stdout (identity line, `fly the route now`, `Saved`, report JSON) |
| `control/linker_*.txt` | `link_e1_flight.py` stdout for each log (all exit 0) |
| `control/tailcheck_*.txt` | `e1_tail_check.py` outputs (HOME info-only, PRESENT, REST) |
| `e1_tail_check_selftest.txt` | 11/11 self-test on the checked-out `rig_routes` (REST_SHUT judged REST → NO) |
| `e1_pilot_2026-09-15.ipynb` / `.executed.ipynb` | the notebook as written and as run (connection literals, binding check, setup cell 3, flight cell 4, outputs) |
| `control/binding_ok`, `setup_done`, `flight_done` | markers the notebook wrote (binding result; per-cell start/end monotonic + wall ns, start/end poses, outcome) |
| `control/notebook_exec.log` | nbconvert log |
| `native_server.log` | native MuJoCo server log (scene load, `Recording to:`, bridge connections) |
| `ledger.csv` | one parked row + two motion rows |

The four large files are `.gitignore`d here and stay in the worktree until
their out-of-Git home is agreed (`EVIDENCE.md` §6). The manifest's
`state_count`/`command_count` stayed 0/null because the server was stopped
with SIGTERM after the pilot; the line counts above are from the files.
