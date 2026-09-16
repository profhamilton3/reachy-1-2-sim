#!/usr/bin/env bash
# C3 parked recording (3 s, no motion): the leg.sh chain run by hand because
# parked recordings have no leg in plan_<cycle>.json (checklist F3).
# Usage: parked.sh <repo> <evidence-dir> <label> <ROUTE>   (tail vs HOME, mode=start)
set -u
REPO=$1; P=$2; NAME=$3; ROUTE=$4; PY=${E1_PYTHON:-python3}
stop() { echo "STOP: $1" | tee -a "$P/control/stop"; exit 9; }
[ -f "$P/control/stop" ] && stop "stop marker already present"
cd "$REPO"
REACHY_SIM_RECORD_CLEARANCE=1 "$PY" -u scripts/measure_route_clearance.py \
  --route "$ROUTE" --duration 3 --host localhost --scene "$REPO/scenes/e1_boards/B4_pool_box_1_r2c3.yaml" \
  --record-root "$P/e1_server_runs" > "$P/control/recorder_$NAME.log" 2>&1
rc=$?; echo "recorder exit=$rc" >> "$P/control/recorder_$NAME.log"
[ $rc -eq 0 ] || stop "recorder $NAME exit $rc"
LOG=$(grep -o 'Saved [0-9]* samples to .*' "$P/control/recorder_$NAME.log" | sed 's/.* to //')
[ -f "$LOG" ] || stop "recorder $NAME: log not found ($LOG)"
PYTHONPATH=src "$PY" scripts/link_e1_flight.py "$LOG" > "$P/control/linker_$NAME.txt" 2>&1
rc=$?; echo "linker exit=$rc" >> "$P/control/linker_$NAME.txt"
[ $rc -eq 0 ] || stop "linker $NAME exit $rc"
PYTHONPATH=src "$PY" scripts/e1_tail_check.py "$LOG" HOME > "$P/control/tailcheck_${NAME}_HOME.txt" 2>&1
rc=$?; echo "tailcheck exit=$rc (mode=start)" >> "$P/control/tailcheck_${NAME}_HOME.txt"
mkdir -p "$P/recorder_logs"; cp "$LOG" "${LOG%.json}.link.json" "$P/recorder_logs/"
echo "PARKED $NAME ok: $LOG"
