# Reproduction — exactly what was run (Stage 2, B1, session s1)

Code: `profhamilton3/reachy-1-2-sim` at merge commit
`e6c30b74e09c38e616dee0819a7e205d12c5853e`, the existing detached worktree
`~/reachy-1-2-sim-stage2` (`$REPO`) from the B4 session, `git status --porcelain` empty
throughout; no `.env` (container `REACHY_PANEL_EXECUTOR` defaults to `0`). Evidence dir
`$EV=~/e1-stage2-B1-s1-2026-09-19` (outside every checkout). Host: macOS 26.6.2 arm64,
`mjpython` 3.14.0 for the native server; `E1_PYTHON=~/e1venv/bin/python` (existing env from the
B4 session: Python 3.12.14, reachy-sdk 0.7.0, grpcio 1.62.2, protobuf 4.25.3, numpy 1.26.4,
nbconvert 7.17.1; `control/host_env_freeze.txt`), kernelspec `e1venv`. Docker Desktop 29.6.2.
Plan followed: `IITG-Reachy-Project/outputs/e1-stage2-b1-execution-plan-2026-09-19.md`.

```
# evidence dir, tools snapshot  (control/pre_start_check.txt, control/tools/REPO_TOOL_SHA256SUMS.txt)
mkdir -p $EV/control/tools $EV/e1_server_runs $EV/recorder_logs $EV/board $EV/summaries
cp $REPO/scripts/e1_stage1/{leg.sh,parked.sh,reset.sh,plan.py,make_cycle_notebook.py,provenance.py,gating.py,start_variant.py,README.md} $EV/control/tools/
cp $REPO/docs/reviews/probes-2026-09-15-e1-stage1-b4/control/tools/settle_wait.py $EV/control/tools/
cp $REPO/scripts/{experiment_gate.py,e1_tail_check.py,link_e1_flight.py,e1_identity.py,measure_route_clearance.py} $EV/control/tools/
# session wrappers: s2_leg.sh copied unchanged from the B4 session; s2_cycle_pre.sh takes <board> as an
# argument (B4 copy had a `B4` literal) and refuses a cycle id that does not name it; s2_ledger.py has
# BOARD_ID/BOARD/OBJECTS constants for B1; s2_set.sh sequences one repetition set through the two wrappers.
# Offline validation of the B1 naming (scratch dir, no server): control/wrapper_validation_offline.txt
export E1_PYTHON=~/e1venv/bin/python; unset REACHY_IP REACHY_ENABLE_MOTION

# caffeinate BEFORE the server (control/caffeinate_pid.txt, caffeinate_check.txt), then Stage 0
caffeinate -i -s -d &                                                  # pid 48760, 09:03:27Z
cd $REPO && REACHY_SIM_SCENE=$REPO/scenes/e1_boards/B1_evidence.yaml \
  REACHY_SIM_RECORD=$EV/e1_server_runs REACHY_SIM_SERVER_LOG=$EV/native_server.log ./scripts/start_sim.sh
RUN=$EV/e1_server_runs/run_20260919_090342                             # control/manifest_check.txt, container_env.txt,
                                                                       # status_stage0.json, supervisor_stage0.txt, clients_8765_stage0.txt
$E1_PYTHON $EV/control/tools/settle_wait.py $RUN 15 180                # control/settle_stage0.json (keyframe-sag, 65.6 s)
$E1_PYTHON -m scripts.e1_stage1.make_cycle_notebook --stage0 --board B1 --session s1 --repo $REPO --evidence-dir $EV
cd $EV && jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=e1venv --output e1_stage1_stage0-B1-s1.executed.ipynb e1_stage1_stage0-B1-s1.ipynb &
date -u +%FT%TZ > $EV/control/go_stage0-B1-s1-armon
bash $EV/control/tools/leg.sh $REPO $EV stage0-B1-s1 stage0-B1-s1-armon RAISE_TO_SIDE INIT end
# (no separate compliant-arm parked recording -- D3 of the B4 plan, carried over)

# Set r1 (checkpoint set), cycle by cycle:
for s in a b c; do $E1_PYTHON -m scripts.e1_stage1.make_cycle_notebook --board B1 --shape $s --rep 1 --session s1 --repo $REPO --evidence-dir $EV; done
bash $EV/control/tools/s2_cycle_pre.sh $EV $REPO $RUN B1 S2-B1-a-r1 1 RAISE_TO_SIDE   # reset.sh -> settle_wait -> parked.sh (stiff-zero) -> nbconvert -> binding_ok
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B1-a-r1 S2-B1-a-r1-setup  RAISE_TO_SIDE   PRESENT   # go_ marker + leg.sh (end mode)
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B1-a-r1 S2-B1-a-r1-flight LOWER_TO_REST   REST
bash $EV/control/tools/s2_cycle_pre.sh $EV $REPO $RUN B1 S2-B1-b-r1 2 PLACE_ROUTE
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B1-b-r1 S2-B1-b-r1-flight PLACE_ROUTE     REST
bash $EV/control/tools/s2_cycle_pre.sh $EV $REPO $RUN B1 S2-B1-c-r1 3 PLACE_ROUTE
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B1-c-r1 S2-B1-c-r1-setup  PLACE_ROUTE     REST
bash $EV/control/tools/s2_leg.sh $EV $REPO S2-B1-c-r1 S2-B1-c-r1-flight LIFT_TO_PRESENT PRESENT
# Checkpoint after set r1: control/checkpoint_1.txt (8/8 criteria), summaries/checkpoint1_recorder_clearance.txt

# Sets r2..r6: the same sequence per set through s2_set.sh (generate a,b,c; then pre + legs in the order above)
for r in 2 3 4 5 6; do bash $EV/control/tools/s2_set.sh $EV $REPO $RUN B1 s1 $r > $EV/control/s2set_r$r.txt 2>&1; done
# r6 halted at cycle c: reset 18 "not verified" -> control/stop (control/reset_18.txt, stop_analysis.txt); no retry.

# Shutdown (control/shutdown.txt), then offline summaries
cd $REPO && docker compose down; kill -TERM 48843   # mjpython server pid (control/server_pid.txt)
python3 $EV/control/tools/s2_ledger.py $EV                             # ledger.csv
$E1_PYTHON $EV/control/tools/s2_clearance_by_link.py $REPO $EV         # summaries/clearance_by_link.*
kill 48760                                                             # caffeinate, after the archive was verified
```

`s2_cycle_pre.sh` / `s2_leg.sh` / `s2_set.sh` are thin operator wrappers (in `control/tools/`):
they call only the packaged `reset.sh`, `settle_wait.py`, `parked.sh`, `nbconvert`, and `leg.sh`
in the documented order, refuse to write a `go_` marker if `control/stop` exists or
`binding_ok_<cycle>` does not, stop at the first non-zero exit, and never run two `leg.sh` chains
concurrently. `s2_ledger.py` and `s2_clearance_by_link.py` read only the evidence tree.

## Archive

Originals: `~/e1-stage2-B1-s1-2026-09-19/` (2.3 GB, untouched). Local archive:
`~/e1-evidence-archive/e1-stage2-B1-s1-2026-09-19.tar.zst` + `.sha256`, verified by extracting
to a scratch directory and running `shasum -a 256 -c SHA256SUMS` there. No GitHub Release or
Drive upload was made (not authorized in this step). Recorder-log originals also remain in
`~/reachy-1-2-sim-stage2/runs/` (gitignored; identical copies are in `recorder_logs/`).
