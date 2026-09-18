#!/usr/bin/env bash
# One recorded leg: recorder -> linker -> tail check. Every non-zero exit is a
# STOP (writes control/stop and exits non-zero). Usage:
#   leg.sh <name> <ROUTE> <duration_s> <TAIL_TARGET> [start]
# 'start' = judge the tail against TAIL_TARGET for start-gate purposes: the
# gross residuals / spread lines are required, the gripper line is recorded
# (scope rev 2 §5); the PARKED_AT verdict itself is informational then.
set -u
NAME=$1; ROUTE=$2; DUR=$3; TGT=$4; MODE=${5:-end}
REPO=/Users/terrancehamilton/reachy-1-2-sim-stage1
P=$REPO/docs/reviews/probes-2026-09-15-e1-stage1-b4
VENV=/private/tmp/claude-501/-Users-terrancehamilton-IITG-Reachy-Project/2ab0d1ac-fe72-429c-a2d9-192d80cab2b5/scratchpad/e1venv
SCENE=$REPO/scenes/e1_boards/B4_pool_box_1_r2c3.yaml
cd $REPO
stop() { echo "STOP: $1" | tee -a $P/control/stop; exit 9; }
[ -f $P/control/stop ] && stop "stop marker already present"
REACHY_SIM_RECORD_CLEARANCE=1 $VENV/bin/python -u scripts/measure_route_clearance.py \
  --route $ROUTE --duration $DUR --host localhost --scene $SCENE \
  --record-root $P/e1_server_runs > $P/control/recorder_$NAME.log 2>&1
rc=$?; echo "recorder exit=$rc" >> $P/control/recorder_$NAME.log
[ $rc -eq 0 ] || stop "recorder $NAME exit $rc"
LOG=$(grep -o 'Saved [0-9]* samples to .*' $P/control/recorder_$NAME.log | sed 's/.* to //')
[ -f "$LOG" ] || stop "recorder $NAME: log not found ($LOG)"
PYTHONPATH=src python3 scripts/link_e1_flight.py "$LOG" > $P/control/linker_$NAME.txt 2>&1
rc=$?; echo "linker exit=$rc" >> $P/control/linker_$NAME.txt
[ $rc -eq 0 ] || stop "linker $NAME exit $rc"
PYTHONPATH=src python3 scripts/e1_tail_check.py "$LOG" $TGT > $P/control/tailcheck_${NAME}_$TGT.txt 2>&1
rc=$?; echo "tailcheck exit=$rc (mode=$MODE)" >> $P/control/tailcheck_${NAME}_$TGT.txt
if [ "$MODE" = "end" ]; then [ $rc -eq 0 ] || stop "tailcheck $NAME $TGT exit $rc"; fi
cp "$LOG" "${LOG%.json}.link.json" $P/recorder_logs/
echo "LEG $NAME ok: $LOG"
