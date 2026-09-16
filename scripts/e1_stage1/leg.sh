#!/usr/bin/env bash
# One recorded leg: recorder -> linker -> tail check. Every non-zero exit is a
# STOP (writes control/stop and exits non-zero). DUR is read from
# plan_<CYCLE>.json (single source of truth, plan.py), never a positional
# argument -- adapted from PR #117's leg.sh (control/tools/leg.sh), which
# took DUR on the command line.
# Usage:
#   leg.sh <repo> <evidence-dir> <cycle> <name> <ROUTE> <TAIL_TARGET> [start]
set -u
REPO=$1; P=$2; CYCLE=$3; NAME=$4; ROUTE=$5; TGT=$6; MODE=${7:-end}
stop() { echo "STOP: $1" | tee -a "$P/control/stop"; exit 9; }
[ -f "$P/control/stop" ] && stop "stop marker already present"
DUR=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['legs'][sys.argv[2]]['dur_s'])" "$P/plan_${CYCLE}.json" "$NAME") \
  || stop "could not read DUR for $NAME from $P/plan_${CYCLE}.json"
cd "$REPO"
REACHY_SIM_RECORD_CLEARANCE=1 python3 -u scripts/measure_route_clearance.py \
  --route "$ROUTE" --duration "$DUR" --host localhost --scene "$REPO/scenes/e1_boards/B4_pool_box_1_r2c3.yaml" \
  --record-root "$P/e1_server_runs" > "$P/control/recorder_$NAME.log" 2>&1
rc=$?; echo "recorder exit=$rc" >> "$P/control/recorder_$NAME.log"
[ $rc -eq 0 ] || stop "recorder $NAME exit $rc"
LOG=$(grep -o 'Saved [0-9]* samples to .*' "$P/control/recorder_$NAME.log" | sed 's/.* to //')
[ -f "$LOG" ] || stop "recorder $NAME: log not found ($LOG)"
PYTHONPATH=src python3 scripts/link_e1_flight.py "$LOG" > "$P/control/linker_$NAME.txt" 2>&1
rc=$?; echo "linker exit=$rc" >> "$P/control/linker_$NAME.txt"
[ $rc -eq 0 ] || stop "linker $NAME exit $rc"
PYTHONPATH=src python3 scripts/e1_tail_check.py "$LOG" "$TGT" > "$P/control/tailcheck_${NAME}_$TGT.txt" 2>&1
rc=$?; echo "tailcheck exit=$rc (mode=$MODE)" >> "$P/control/tailcheck_${NAME}_$TGT.txt"
if [ "$MODE" = "end" ]; then [ $rc -eq 0 ] || stop "tailcheck $NAME $TGT exit $rc"; fi
mkdir -p "$P/recorder_logs"
cp "$LOG" "${LOG%.json}.link.json" "$P/recorder_logs/"
echo "LEG $NAME ok: $LOG"
