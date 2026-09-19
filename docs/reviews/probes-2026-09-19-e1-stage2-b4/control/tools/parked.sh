#!/usr/bin/env bash
# C3 parked recording (3 s, no motion). Promoted into the package from PR
# #117's evidence-only control/tools/parked.sh (decision note
# outputs/e1-stage2-decision-2026-09-15.md §6 item 3: "parked-recording
# tooling ... should become scripts/e1_stage1/parked.sh ... so the C3 step
# is versioned and tested like the rest") -- board/scene resolution now
# goes through plan.py's BOARDS registry instead of a hard-coded path, the
# same single source of truth leg.sh reads from plan_<CYCLE>.json.
#
# Runs the same recorder -> linker -> archive -> experiment gate ->
# tailcheck chain as leg.sh, but the tailcheck is informational only
# (mode=start, matches PR #117's parked.sh): a parked recording proves
# nothing flew, it is not gated on reaching a route's target. The experiment
# gate (scripts/experiment_gate.py, PR #124 re-review R3) is NOT
# informational -- a parked recording is still a recording, and a contact or
# a disturbed board during it must STOP the same as during a flight leg;
# it runs right after the recording is archived to recorder_logs/, so a
# rejected recording's evidence is preserved for review. After the chain,
# classifies the recording's final pose with start_variant.py (decision
# note §4's policy-A gate) and STOPs -- same fail-closed contract as every
# other check in this package -- unless it matches EXACTLY "stiff-zero"
# (policy A requires stiff-zero for every Stage 2 cycle; "keyframe-sag"
# means Stage 0's turn_on did not hold and must prevent motion the same as
# no match at all -- PR #124 review, M3).
#
# `<cycle>` (M3) binds this evidence to a specific cycle id, independent of
# `<name>`: the classification is written to `start_variant_<cycle>.json`
# (not `start_variant_<name>.json`), which is the exact file
# `plan.start_variant_gate` (called from the generated notebook's binding
# cell) reads for that cycle. `<name>` remains free-form for the recorder
# log's own filename; `<cycle>` is what the notebook actually checks, so
# operator naming discipline on `<name>` can no longer be the only thing
# binding evidence to a cycle. Re-recording for a cycle that already has
# `start_variant_<cycle>.json` is refused -- no re-fly, no recovery
# (decision note §5).
#
# Usage:
#   parked.sh <repo> <evidence-dir> <name> <board> <route-label> <cycle>
#     <name>         unique per parked recording in this evidence dir (the
#                    caller's responsibility -- e.g. the upcoming cycle id
#                    plus "-parked", or "stage0-<board>-<session>-parked")
#     <board>        one of plan.BOARDS's keys (B4, B1, B2)
#     <route-label>  the upcoming cycle's first leg route, for the
#                    recorder's planned-vs-realised clearance table only --
#                    nothing is flown regardless of this value
#     <cycle>        the upcoming Stage 2 cycle id this recording will
#                    authorize (e.g. S2-B4-a-r3) -- must match the CYCLE
#                    variable baked into that cycle's generated notebook
#
# Same E1_PYTHON requirement as leg.sh: measure_route_clearance.py imports
# reachy_sdk, which only the documented host SDK venv has.
set -u
REPO=$1; P=$2; NAME=$3; BOARD=$4; ROUTE=$5; CYCLE=$6
PY=${E1_PYTHON:-python3}
stop() { echo "STOP: $1" | tee -a "$P/control/stop"; exit 9; }
[ -f "$P/control/stop" ] && stop "stop marker already present"
[ -f "$P/control/start_variant_$CYCLE.json" ] && stop "start_variant already recorded for cycle $CYCLE -- no re-recording (decision note §5: no re-fly, no recovery)"
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
mkdir -p "$P/recorder_logs"; cp "$LOG" "${LOG%.json}.link.json" "$P/recorder_logs/"
"$PY" scripts/experiment_gate.py "$LOG" > "$P/control/gate_$NAME.txt" 2>&1
rc=$?; echo "gate exit=$rc" >> "$P/control/gate_$NAME.txt"
[ $rc -eq 0 ] || stop "experiment gate $NAME rejected (see $P/control/gate_$NAME.txt): exit $rc"
PYTHONPATH=src "$PY" scripts/e1_tail_check.py "$LOG" HOME > "$P/control/tailcheck_${NAME}_HOME.txt" 2>&1
rc=$?; echo "tailcheck exit=$rc (mode=start)" >> "$P/control/tailcheck_${NAME}_HOME.txt"
PYTHONPATH="$REPO/scripts" "$PY" -m e1_stage1.start_variant "$LOG" --cycle "$CYCLE" > "$P/control/start_variant_$CYCLE.json" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  stop "start_variant for cycle $CYCLE did not match a known policy-A variant (see $P/control/start_variant_$CYCLE.json) -- decision note §4: failure stops progression"
fi
VARIANT=$(PYTHONPATH="$REPO/scripts" "$PY" -c "
import json, sys
print(json.load(open(sys.argv[1])).get('start_variant'))
" "$P/control/start_variant_$CYCLE.json")
if [ "$VARIANT" != "stiff-zero" ]; then
  stop "start_variant for cycle $CYCLE is '$VARIANT', not 'stiff-zero' -- policy A requires stiff-zero for every Stage 2 cycle (decision note §4)"
fi
echo "PARKED $NAME ok, cycle $CYCLE stiff-zero: $LOG"
