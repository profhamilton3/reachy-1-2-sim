# Reproduction — exactly what was run (Stage 2, B4, session s1)

Code: `profhamilton3/reachy-1-2-sim` at merge commit
`e6c30b74e09c38e616dee0819a7e205d12c5853e` (PR #131), checked out detached in a new worktree
`~/reachy-1-2-sim-stage2` (`$REPO`), `git status --porcelain` empty throughout; no `.env` in
the worktree (so the container's `REACHY_PANEL_EXECUTOR` defaults to `0`). Evidence dir
`$EV=~/e1-stage2-B4-s1-2026-09-18` (outside every checkout). Host: macOS 26.6.2 arm64,
`mjpython` 3.14.0 for the native server; `E1_PYTHON=~/e1venv/bin/python` — recreated with
`uv venv --python 3.12 ~/e1venv` + `uv pip install reachy-sdk==0.7.0 "numpy<2" pyyaml
nbconvert nbformat ipykernel jupyter_client` (Python 3.12.14, reachy-sdk 0.7.0, grpcio 1.62.2,
protobuf 4.25.3, numpy 1.26.4, nbconvert 7.17.1; full `uv pip freeze` in
`control/host_env_check.txt`), kernelspec `e1venv` registered. Docker Desktop 29.6.2.
Plan followed: `IITG-Reachy-Project/outputs/e1-stage2-b4-execution-plan-2026-09-18.md`.

```
# worktree, env, evidence dir, tools snapshot  (control/host_env_check.txt, pre_start_check.txt)
git -C ~/reachy-1-2-sim worktree add --detach $REPO e6c30b74e09c38e616dee0819a7e205d12c5853e
mkdir -p $EV/control/tools $EV/e1_server_runs $EV/recorder_logs $EV/board
cp $REPO/scripts/e1_stage1/{leg.sh,parked.sh,reset.sh,plan.py,make_cycle_notebook.py,provenance.py,gating.py,start_variant.py,README.md} $EV/control/tools/
cp $REPO/docs/reviews/probes-2026-09-15-e1-stage1-b4/control/tools/settle_wait.py $EV/control/tools/
cp $REPO/scripts/{experiment_gate.py,e1_tail_check.py,link_e1_flight.py,e1_identity.py} $EV/control/tools/
export E1_PYTHON=~/e1venv/bin/python; unset REACHY_IP REACHY_ENABLE_MOTION

# Stage 0 -- fresh server (control/start_sim.log, manifest_check.txt, status_stage0.json, supervisor_stage0.txt)
cd $REPO && REACHY_SIM_SCENE=$REPO/scenes/e1_boards/B4_pool_box_1_r2c3.yaml \
  REACHY_SIM_RECORD=$EV/e1_server_runs REACHY_SIM_SERVER_LOG=$EV/native_server.log ./scripts/start_sim.sh
RUN=$EV/e1_server_runs/run_20260919_043924
$E1_PYTHON $EV/control/tools/settle_wait.py $RUN 15 180              # control/settle_stage0.json (keyframe-sag, 67 s)
$E1_PYTHON -m scripts.e1_stage1.make_cycle_notebook --stage0 --board B4 --session s1 --repo $REPO --evidence-dir $EV
cd $EV && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=e1venv --output e1_stage1_stage0-B4-s1.executed.ipynb e1_stage1_stage0-B4-s1.ipynb &
date -u +%FT%TZ > $EV/control/go_stage0-B4-s1-armon
bash $EV/control/tools/leg.sh $REPO $EV stage0-B4-s1 stage0-B4-s1-armon RAISE_TO_SIDE INIT end
# (no separate compliant-arm parked recording -- plan D3)

# Per set r=1..6: generate the three cycles, then per cycle (gen = 3*(r-1)+{1,2,3}):
for s in a b c; do $E1_PYTHON -m scripts.e1_stage1.make_cycle_notebook --board B4 --shape $s --rep $r --session s1 --repo $REPO --evidence-dir $EV; done
bash $EV/control/tools/s2_cycle_pre.sh $EV $REPO $RUN S2-B4-a-r$r <gen> RAISE_TO_SIDE   # reset.sh -> settle_wait -> parked.sh (stiff-zero) -> nbconvert -> binding_ok
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B4-a-r$r S2-B4-a-r$r-setup  RAISE_TO_SIDE   PRESENT   # go_ marker + leg.sh (end mode)
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B4-a-r$r S2-B4-a-r$r-flight LOWER_TO_REST   REST
bash $EV/control/tools/s2_cycle_pre.sh $EV $REPO $RUN S2-B4-b-r$r <gen> PLACE_ROUTE
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B4-b-r$r S2-B4-b-r$r-flight PLACE_ROUTE     REST
bash $EV/control/tools/s2_cycle_pre.sh $EV $REPO $RUN S2-B4-c-r$r <gen> PLACE_ROUTE
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B4-c-r$r S2-B4-c-r$r-setup  PLACE_ROUTE     REST
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B4-c-r$r S2-B4-c-r$r-flight LIFT_TO_PRESENT PRESENT

# Checkpoint after set r1 (control/checkpoint_1.txt), caffeinate -i -s -d from 05:24Z (control/caffeinate_pid.txt)
# Ledger / summaries (offline)
python3 $EV/control/tools/s2_ledger.py $EV                            # ledger.csv
# Shutdown (control/shutdown.txt)
cd $REPO && docker compose down; kill -TERM <mjpython server pid on :8765>; kill <caffeinate pid>
```

`s2_cycle_pre.sh` / `s2_leg.sh` are thin operator wrappers (in `control/tools/`): they call
only the packaged `reset.sh`, `settle_wait.py`, `parked.sh`, `nbconvert`, and `leg.sh` in the
documented order, refuse to write a `go_` marker if `control/stop` exists or `binding_ok_<cycle>`
does not, and never run two `leg.sh` chains concurrently. `s2_ledger.py` reads only the evidence
tree.

## Archive

Originals: `~/e1-stage2-B4-s1-2026-09-18/` (2.4 GB, untouched). Local archives (both kept):
`~/e1-evidence-archive/e1-stage2-B4-s1-2026-09-18.tar.zst` (session close, before the per-link
analysis and §3.1 were added) and, final,
`~/e1-evidence-archive/e1-stage2-B4-s1-2026-09-18-final.tar.zst` + `.sha256`, each verified by extracting
to a scratch directory and running `shasum -a 256 -c SHA256SUMS` there. No GitHub Release or
Drive upload was made (not authorized in this step). Recorder-log originals also remain in
`~/reachy-1-2-sim-stage2/runs/` (gitignored; identical copies are in `recorder_logs/`).
