#!/usr/bin/env bash
# C3 parked recording (3 s, no motion). Promoted into the package from PR
# #117's evidence-only control/tools/parked.sh (decision note
# outputs/e1-stage2-decision-2026-09-15.md §6 item 3: "parked-recording
# tooling ... should become scripts/e1_stage1/parked.sh ... so the C3 step
# is versioned and tested like the rest") -- board/scene resolution now
# goes through plan.py's BOARDS registry instead of a hard-coded path, the
# same single source of truth leg.sh reads from plan_<CYCLE>.json.
#
# Runs the same recorder -> linker -> tailcheck chain as leg.sh, but the
# tailcheck is informational only (mode=start, matches PR #117's
# parked.sh): a parked recording proves nothing flew, it is not gated on
# reaching a route's target. After the chain, classifies the recording's
# final pose with start_variant.py (decision note §4's policy-A gate) and
# STOPs -- same fail-closed contract as every other check in this package
# -- if it matches neither known start variant ("failure stops
# progression").
#
# Usage:
#   parked.sh <repo> <evidence-dir> <name> <board> <route-label>
#     <name>         unique per parked recording in this evidence dir (the
#                    caller's responsibility -- e.g. the upcoming cycle id
#                    plus "-parked", or "stage0-<board>-<session>-parked")
#     <board>        one of plan.BOARDS's keys (B4, B1, B2)
#     <route-label>  the upcoming cycle's first leg route, for the
#                    recorder's planned-vs-realised clearance table only --
#                    nothing is flown regardless of this value
#
# Same E1_PYTHON requirement as leg.sh: measure_route_clearance.py imports
# reachy_sdk, which only the documented host SDK venv has.
set -u
REPO=$1; P=$2; NAME=$3; BOARD=$4; ROUTE=$5
PY=${E1_PYTHON:-python3}
stop() { echo "STOP: $1" | tee -a "$P/control/stop"; exit 9; }
[ -f "$P/control/stop" ] && stop "stop marker already present"
SCENE_REL=$(PYTHONPATH="$REPO/scripts" "$PY" -c "
from e1_stage1 import plan
print(plan.board_scene_rel('$BOARD'))
") || stop "unknown/unauthorized board $BOARD"
cd "$REPO"
REACHY_SIM_RECORD_CLEARANCE=1 "$PY" -u scripts/measure_route_clearance.py \
  --route "$ROUTE" --duration 3 --host localhost --scene "$REPO/$SCENE_REL" \
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
PYTHONPATH="$REPO/scripts" "$PY" -m e1_stage1.start_variant "$LOG" > "$P/control/start_variant_$NAME.json" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  stop "start_variant $NAME did not match a known policy-A variant (see $P/control/start_variant_$NAME.json) -- decision note §4: failure stops progression"
fi
mkdir -p "$P/recorder_logs"; cp "$LOG" "${LOG%.json}.link.json" "$P/recorder_logs/"
echo "PARKED $NAME ok: $LOG"
