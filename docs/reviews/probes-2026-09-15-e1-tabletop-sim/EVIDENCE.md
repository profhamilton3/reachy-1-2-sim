# E1 pilot evidence — handoff (2026-09-15)

Companion to `REPORT.md` (narrative and results) and `ledger.csv`. This file
records what is in Git and what is not, how the run was produced, the
environment it was produced in, the known limitations of the evidence, and
the state of the worktree that still holds the sole copies of the large
files.

**Conclusion scope, stated once:** one board (`B4_pool_box_1_r2c3`), two
routes (`RAISE_TO_SIDE` as setup, `LOWER_TO_REST` as the flight), one
simulator run (`run_20260915_173758`, code `f548fc6`). No margin is
selected, `"shells"` is not promoted, and nothing here transfers to the
physical robot or to any other board or route.

## 1. What is in this directory, and what is in Git

| path | in Git | notes |
|---|---|---|
| `README.md`, `EVIDENCE.md`, `REPORT.md`, `ledger.csv`, `SHA256SUMS`, `.gitignore` | yes | this handoff |
| `summaries/*.summary.json` | yes | per-recording extracts of each sidecar (identity, settled check, displacement, contacts, `contacts_window`, alignment stats, worst server-side clearance, motion timing, tail-check and linker stdout, notebook cell timings) — small enough to read without the sidecars |
| `board/` | yes | B4 YAML as flown + sha256 |
| `recorder_logs/*.json` (3 sample logs) | yes | 20 Hz joint+aperture samples, schema 3 |
| `recorder_logs/route_clearance_LOWER_TO_REST_20260915T174409Z.link.json` | yes | parked-recording sidecar (76 KB) |
| `recorder_logs/route_clearance_RAISE_TO_SIDE_20260915T174717Z.link.json` | **no** (2.5 MB) | setup sidecar; hashed in `SHA256SUMS`; see §6 |
| `recorder_logs/route_clearance_LOWER_TO_REST_20260915T175021Z.link.json` | **no** (1.3 MB) | flight sidecar; hashed; see §6 |
| `e1_server_runs/run_20260915_173758/manifest.json` | yes | see §4 for the zero counts |
| `e1_server_runs/run_20260915_173758/states.jsonl` | **no** (256 MB, 45 609 states) | **sole copy**; hashed `acf565b7…6e96c`; see §6 |
| `e1_server_runs/run_20260915_173758/commands.jsonl` | **no** (2.0 MB, 1 747 commands) | sole copy; hashed; see §6 |
| `control/` | yes | recorder stdout ×3, linker stdout ×3, tail-check outputs ×3, notebook markers (`binding_ok`, `setup_done`, `flight_done`), nbconvert log |
| `e1_pilot_2026-09-15.ipynb`, `.executed.ipynb` | yes | the notebook as written and as run, with outputs |
| `e1_tail_check.py`, `e1_tail_check_selftest.txt` | yes | the checker used, and its 11/11 self-test from this session |
| `native_server.log` | yes | scene load, `Recording to:`, bridge connections, shutdown |

`SHA256SUMS` covers every file including the four that are not committed;
verify with `shasum -a 256 -c SHA256SUMS` from this directory (the four
uncommitted files must be present for a full pass).

## 2. Environment actually used

- Host: macOS 26.6.2 (arm64). Native server: `mjpython` from the host
  Python 3.14.0 with `mujoco 3.11.0` (`manifest.json` records both).
- Container: `reachy-1-2-sim:latest` id `e22b885423a8`, **rebuilt this
  session** (`docker compose build`, cached base layers) from the `f548fc6`
  tree, because the previous image (built 2026-09-11) predated commits
  `38d60aa`/`01700ec`, which add the `/tmp/reachy_frame_meta.json` sidecar
  that `/status` reads `backend` from — without it the identity check
  refuses. Docker 29.6.2 / Compose v5.3.1. The old image is no longer
  listed after the rebuild.
- **Recorder and notebook kernel Python (the SDK environment):** the host
  Python 3.14 cannot install `reachy-sdk==0.7.0` — `reachy-sdk` requires
  `mobile-base-sdk`, whose 1.x pins `protobuf<=4.25.3`, which forces
  `grpcio-tools==1.62.2`, which has no CPython 3.14 wheel and fails to build
  (`pkg_resources` missing under 3.14). A scratch venv was made with
  `uv venv --python 3.12` and `uv pip install reachy-sdk==0.7.0 "numpy<2"
  pyyaml nbconvert nbformat ipykernel jupyter_client`. Resolved versions:
  Python 3.12.14, reachy-sdk 0.7.0, reachy-sdk-api 0.7.3, grpcio 1.62.2,
  grpcio-tools 1.62.2, protobuf 4.25.3, numpy 1.26.4, pyyaml 6.0.3,
  scipy 1.17.1, opencv-python 4.11.0.86, mobile-base-sdk 1.0.2, nbconvert
  7.17.1, ipykernel 7.3.0. **Note:** `mobile-base-sdk` pulls in
  `reachy2-sdk-api 1.0.21` as a transitive dependency; nothing in this
  repo or in the pilot imports it (the repo rule is about imports, and the
  recorder/notebook import only `reachy_sdk`). The venv lived in the
  session scratchpad and is not part of the evidence; the recipe above
  reproduces it.
- The linker and tail checker ran on the host Python 3.14 with
  `PYTHONPATH=src` (they need only numpy/yaml and the repo).
- Shell hygiene: no `REACHY_*` variables in the server, recorder or
  notebook shells beyond `REACHY_SIM_RECORD`, `REACHY_SIM_SCENE`,
  `REACHY_SIM_SERVER_LOG` (server) and `REACHY_SIM_RECORD_CLEARANCE=1`
  (recorder); `REACHY_IP` and `REACHY_ENABLE_MOTION` unset everywhere and
  asserted again inside the kernel (notebook cell 1).

## 3. Actual starting posture

The simulator does not start at `rig_routes.HOME`. After a 12.9-s settle
from the MJCF keyframe (states seq 2–646, before any SDK client connected;
shoulder roll moved 0.43°, wrist_roll 33.8°, gripper 11.9°) the arm rested at

    r_shoulder_pitch 0.0  r_shoulder_roll 0.0  r_arm_yaw 0.0  r_elbow_pitch 0.0
    r_forearm_yaw 0.0  r_wrist_pitch 0.0  r_wrist_roll +40.15  r_gripper -40.15

i.e. `HOME` on the four placing joints, but wrist_roll +40.15° (HOME: 0)
and gripper −40.15° (HOME/OPEN: −45). `primitives.raise_to_side` judges
HOME on `PLACING_JOINTS` only, so it accepted this start; the first route
waypoints then drove wrist and gripper to their commanded values. The
tail checker judged the parked recording `PARKED_AT_HOME=NO` on the
gripper criterion alone (`control/tailcheck_home_parked.txt`); that check
was informational, not a gate. Consequence for readers: "parked at HOME"
in this run means the sim's settled keyframe, not the `HOME` table row.

## 4. Known defects in the evidence

1. **RViz scene markers were absent for the whole session.**
   `scripts/start_sim.sh:69` derives the container scene path as
   `/opt/scenes/$(basename "$SCENE")`, dropping the `e1_boards/` directory,
   so `scene-marker-publisher` crash-looped on
   `FileNotFoundError: Scene file not found: /opt/scenes/B4_pool_box_1_r2c3.yaml`
   (supervisord restarted it every ~3 s). The native physics, the bridge,
   `/status`, the identity check and the recording all used the correct
   host path (manifest `scene_path`/`scene_sha256` match the board). This
   was review advisory A1, accepted for the pilot; there is no visual
   record of the board from RViz.
2. **`manifest.json` counts are zero/null** (`state_count: 0`,
   `command_count: 0`, `total_steps: null`, `total_duration_s: null`). The
   native server was stopped with SIGTERM after the pilot and did not
   finalise the manifest. Line counts from the files: 45 609 states
   (seq 1..45609, contiguous), 1 747 commands. Consumers must not treat
   the manifest counts as authoritative for this run.
3. **Timing limitations.**
   - `commands.jsonl` entries carry `seq`, `target_rad`, `compliant`,
     limits — **no timestamp and no client id**. "Commands only arrived
     during the two motion cells" is therefore inferred from *state*
     motion (the right arm moved in exactly three intervals: the start-up
     settle, the setup cell window, the flight cell window), not read from
     the command log.
   - Lead-in was produced by polling the recorder's stdout file every
     0.25 s for `fly the route now`, then `time.sleep(3.0)`; measured
     lead-ins from the sample logs are 3.60 s (setup) and 3.25 s (flight).
   - `wall_time_ns` in states and samples is `time.monotonic_ns()` (host
     monotonic), by design; wall-clock UTC exists only in the recorder
     filenames, `manifest.started_at`, the server log, and the notebook
     markers' `t_*_wall_ns`. Cross-referencing to UTC is approximate to
     the order of the SDK read lag (alignment `wall_offset_s` within
     ±0.08 s).
   - The notebook cell timings (`setup_done`, `flight_done`) are taken in
     the kernel around the motion call; they include SDK call overhead.
   - Push jitter over the whole session: max gap 107 ms, no `seq` gaps.
     Coverage is established by the linker's bracketing + contiguity, not
     by the gap statistic.
4. The notebook was executed headlessly (`jupyter nbconvert --execute`,
   kernel `e1venv`), with the motion cells gated by marker files written
   from the operator shell (`control/`), not by a person pressing
   Shift-Enter. The executed notebook with outputs is the record.

## 5. Reproduction (simulator only; needs a separate written approval to run motion)

```bash
# 0. checkout at f548fc6 in a clean worktree; no REACHY_* in the environment
git -C ~/reachy-1-2-sim worktree add ../reachy-1-2-sim-e1 main   # or reuse
cd ~/reachy-1-2-sim-e1 && git rev-parse HEAD                      # f548fc6
uv venv --python 3.12 /path/e1venv && uv pip install --python /path/e1venv/bin/python \
    reachy-sdk==0.7.0 "numpy<2" pyyaml nbconvert nbformat ipykernel jupyter_client
/path/e1venv/bin/python -m ipykernel install --prefix /path/e1venv --name e1venv
docker compose build                                              # if the image predates the tree

# 1. services (record root + server log under the evidence dir)
P=$PWD/docs/reviews/probes-<date>-e1-tabletop-sim
REACHY_SIM_RECORD=$P/e1_server_runs REACHY_SIM_SCENE=$PWD/scenes/e1_boards/B4_pool_box_1_r2c3.yaml \
REACHY_SIM_SERVER_LOG=$P/native_server.log ./scripts/start_sim.sh
curl -s localhost:8080/status            # backend mujoco-remote, frames_stale false
shasum -a 256 scenes/e1_boards/B4_pool_box_1_r2c3.yaml   # == manifest scene_sha256

# 2. checker self-test, parked recording, linker
PYTHONPATH=src python3 $P/e1_tail_check.py --selftest
REACHY_SIM_RECORD_CLEARANCE=1 /path/e1venv/bin/python -u scripts/measure_route_clearance.py \
    --route LOWER_TO_REST --duration 3 --host localhost \
    --scene $PWD/scenes/e1_boards/B4_pool_box_1_r2c3.yaml --record-root $P/e1_server_runs
PYTHONPATH=src python3 scripts/link_e1_flight.py runs/<log>.json

# 3. notebook (cells 1–2 connect + binding; cell 3 waits for control/recorder_setup.log,
#    cell 4 waits for control/go_flight then control/recorder_flight.log)
/path/e1venv/bin/jupyter nbconvert --to notebook --execute --allow-errors \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=e1venv \
    e1_pilot_<date>.ipynb --output e1_pilot_<date>.executed.ipynb &
# 4. setup recording (120 s) -> gates -> write control/go_flight -> flight recording (60 s) -> gates
REACHY_SIM_RECORD_CLEARANCE=1 /path/e1venv/bin/python -u scripts/measure_route_clearance.py \
    --route RAISE_TO_SIDE --duration 120 ... > $P/control/recorder_setup.log 2>&1
PYTHONPATH=src python3 scripts/link_e1_flight.py runs/<setup log>.json
PYTHONPATH=src python3 $P/e1_tail_check.py runs/<setup log>.json PRESENT
echo pass > $P/control/go_flight
REACHY_SIM_RECORD_CLEARANCE=1 /path/e1venv/bin/python -u scripts/measure_route_clearance.py \
    --route LOWER_TO_REST --duration 60 ... > $P/control/recorder_flight.log 2>&1
PYTHONPATH=src python3 scripts/link_e1_flight.py runs/<flight log>.json
PYTHONPATH=src python3 $P/e1_tail_check.py runs/<flight log>.json REST
# 5. stop: docker compose stop; kill the mjpython server (SIGTERM leaves manifest counts at 0)
```

Offline re-checks that need only what is in Git plus the raw bundle (§6):
`shasum -a 256 -c SHA256SUMS`; `PYTHONPATH=src python3 scripts/link_e1_flight.py
--run-dir e1_server_runs/run_20260915_173758 recorder_logs/<log>.json`
regenerates the contacts/coverage/displacement parts of each sidecar;
`e1_tail_check.py <log> <TARGET>` reproduces `control/tailcheck_*.txt`.

## 6. The 256 MB raw state log — proposed durable location (not yet acted on)

Sole copies today: this worktree, `~/reachy-1-2-sim-e1/docs/reviews/
probes-2026-09-15-e1-tabletop-sim/e1_server_runs/run_20260915_173758/`
(`states.jsonl` 256 MB, `commands.jsonl` 2.0 MB) plus the two large
sidecars in `recorder_logs/`. They are `.gitignore`d here so they cannot be
committed by accident, and they must **not be deleted or moved** until the
location below (or another) is agreed.

Proposal: one bundle `e1-pilot-2026-09-15-raw.tar.gz` containing the whole
`run_20260915_173758/` directory (manifest, states, commands) and the two
large sidecars, with `SHA256SUMS` alongside (gzip of `states.jsonl` alone
measured 42.8 MB, so the bundle is ≈ 45 MB), attached as a **GitHub Release
asset** on a tag `evidence/e1-pilot-2026-09-15` in
`profhamilton3/reachy-1-2-sim`. Reasons: it stays with the repository and
its permissions, has a 2 GB per-file limit, costs no LFS bandwidth quota,
and the sha256 in Git pins it. Secondary copy: the project Google Drive
folder the owner designates (the same tar + `SHA256SUMS`). Once uploaded and
its sha verified, the worktree copies may be removed and this section
updated with the asset URL.

## 7. Worktree status (retain until §6 is done)

`~/reachy-1-2-sim-e1` is a **Git worktree** of `~/reachy-1-2-sim`
(`git worktree list`):

    /Users/terrancehamilton/reachy-1-2-sim     0d09ce8 [chore/deployment-panel-features]   (primary; owner's notebook WIP, untouched)
    /Users/terrancehamilton/reachy-1-2-sim-e1  f548fc6 [main] -> evidence/e1-pilot-2026-09-15 (this PR's branch)

Before the pilot it was on `feat/sim-e1-readiness` at `4395a9f` (same tree
as `f548fc6`); it was moved to `main`, fast-forwarded to `f548fc6`, and the
evidence branch was created from there. Its only uncommitted content is
the four `.gitignore`d files in §6 (`runs/` holds the recorder's original
copies of the three logs and sidecars, also ignored by the repo's
`.gitignore`). **Do not `git worktree remove` or `prune` it until the raw
bundle is stored and its sha verified.**
