# Reproduction — exactly what was run

Code: `profhamilton3/reachy-1-2-sim` at merge commit
`eb34238f6aae6bb9cc14dd901abc31a677e9b0d5` (PR #120), checked out
detached in `~/reachy-1-2-sim-issue116` (`$REPO`), `git status --porcelain`
empty throughout. Evidence dir `$EV=~/e1-stage1-retry-2026-09-15`
(outside every checkout). Host: macOS, `mjpython` 3.14 for the native
server; `E1_PYTHON` = an e1venv (Python 3.12.14, `reachy-sdk` 0.7.0,
`nbconvert` 7.17.1) with an `e1venv` Jupyter kernelspec. Docker Compose
v5.3.1. Checklist followed: `outputs/checklist-2026-09-15-stage1-retry-post-merge.md`
(owner's tree), sections A–G.

```
# A/B/C — checkout, evidence dir, host env (outputs in control/host_env_check.txt, pre_start_check.txt)
git -C $REPO checkout --detach eb34238f6aae6bb9cc14dd901abc31a677e9b0d5
mkdir -p $EV/control $EV/e1_server_runs $EV/recorder_logs $EV/control/tools
export E1_PYTHON=/path/to/e1venv/bin/python3; unset REACHY_IP REACHY_ENABLE_MOTION
cp $REPO/scripts/e1_stage1/{leg.sh,reset.sh,plan.py,make_cycle_notebook.py,provenance.py,README.md} $EV/control/tools/
cp ~/reachy-1-2-sim-stage1/docs/reviews/probes-2026-09-15-e1-stage1-b4/control/tools/settle_wait.py $EV/control/tools/

# D — fresh server (first attempt hit the stale-container name conflict: control/aborted_start_1/)
cd $REPO && REACHY_SIM_SCENE=$REPO/scenes/e1_boards/B4_pool_box_1_r2c3.yaml \
  REACHY_SIM_RECORD=$EV/e1_server_runs REACHY_SIM_SERVER_LOG=$EV/native_server.log ./scripts/start_sim.sh
# manifest check (control/manifest_check.txt), /status (control/status_stage0.json), supervisorctl status
"$E1_PYTHON" $EV/control/tools/settle_wait.py $RUN 15 180          # control/settle_stage0.json
bash $EV/control/tools/parked.sh $REPO $EV parked_stage0 LOWER_TO_REST   # Stage 0 parked recording

# E — generate (control/generate.log, plan_check.txt)
for c in S1a S1b S1c; do "$E1_PYTHON" -m scripts.e1_stage1.make_cycle_notebook $c --repo $REPO --evidence-dir $EV; done

# F — per cycle (S1a shown; S1b/S1c identical with gen 2/3 and their legs)
bash $EV/control/tools/reset.sh $REPO $EV 1 $RUN                    # C1  -> control/reset_1.txt
"$E1_PYTHON" $EV/control/tools/settle_wait.py $RUN 15 180          # C2  -> control/settle_S1a.json
bash $EV/control/tools/parked.sh $REPO $EV parked_S1a RAISE_TO_SIDE  # C3  -> recorder/linker/tailcheck_parked_S1a_HOME
cd $EV && <e1venv>/bin/jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=e1venv --output e1_stage1_S1a.executed.ipynb e1_stage1_S1a.ipynb &   # cells 1-2 run; binding_ok_S1a
date -u +%FT%TZ > $EV/control/go_setup_a;  bash $EV/control/tools/leg.sh $REPO $EV S1a setup_a  RAISE_TO_SIDE PRESENT   # C5, mode end
date -u +%FT%TZ > $EV/control/go_flight_a; bash $EV/control/tools/leg.sh $REPO $EV S1a flight_a LOWER_TO_REST REST      # C7, mode end
# S1b: reset 2, settle, parked_S1b PLACE_ROUTE, notebook, go_flight_b + leg.sh S1b flight_b PLACE_ROUTE REST
# S1c: reset 3, settle, parked_S1c PLACE_ROUTE, notebook, go_setup_c + leg.sh S1c setup_c PLACE_ROUTE REST,
#      go_flight_c + leg.sh S1c flight_c LIFT_TO_PRESENT PRESENT

# G — shutdown (control/shutdown.txt)
cd $REPO && docker compose down; kill -TERM <native server pid on :8765>
```

`parked.sh` is in `control/tools/` (not in the repo): the merged `leg.sh`
reads `DUR` by leg name from `plan_<cycle>.json`, and parked recordings
have no plan leg; `parked.sh` is the same chain with `--duration 3` and a
tail check vs `HOME` in `start` mode. Flown legs always used `leg.sh`
with the 7th argument omitted (`end` mode).

## Offline analysis (no services)

```
"$E1_PYTHON" clearance_by_link.py $REPO $EV     # -> summaries/clearance_by_link.{md,json}
```
(`control/tools/clearance_by_link.py`; pure geometry from
`scripts/measure_route_clearance.py` + `reachy_ai.motion.kinematics.link_capsules`
called one capsule at a time against the B4 YAML from `$REPO/scenes/`.)
`summaries/initial_state_by_leg.txt` reads `states.jsonl` at each leg's
`t_start_mono_ns` from `control/<leg>_done`.

## Archive

Originals: `~/e1-stage1-retry-2026-09-15/` (402 MB, untouched).
Proposed durable copy: `~/e1-evidence-archive/e1-stage1-retry-2026-09-15.tar.zst`
+ `.sha256` (created, outside all checkouts), to be uploaded as a release
asset on a `evidence/e1-stage1-retry-2026-09-15` tag of the repo
(GitHub release assets, ≤ 2 GB each) and/or to the project's Google Drive
evidence folder. Verify any copy with `shasum -a 256 -c SHA256SUMS`
inside the extracted tree (124 entries; `SHA256SUMS` itself excluded). The archive's own sha256 is in its `.sha256` sidecar and in the evidence PR description (not embedded here, to keep the tree's hashes self-consistent).
