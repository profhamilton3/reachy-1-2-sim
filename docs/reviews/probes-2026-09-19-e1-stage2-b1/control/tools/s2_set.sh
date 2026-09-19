#!/usr/bin/env bash
# Operator wrapper (Stage 2 B1 s1 session): one repetition set r = cycles a, b, c
# in the documented order, each = s2_cycle_pre.sh then its s2_leg.sh legs.
# Generates the three notebooks first (make_cycle_notebook refuses reuse), stops
# at the first non-zero exit or control/stop, never removes control/stop.
# Usage: s2_set.sh <EV> <REPO> <RUN> <board> <session> <rep>
set -u
EV=$1; REPO=$2; RUN=$3; BOARD=$4; SESSION=$5; R=$6
PY=${E1_PYTHON:?set E1_PYTHON}
T="$EV/control/tools"
halt() { echo "SET r$R HALT: $*  ($(date -u +%FT%TZ))"; exit 9; }
[ -f "$EV/control/stop" ] && halt "stop marker present"
echo "=== set r$R generate ($(date -u +%FT%TZ)) ==="
for s in a b c; do
  (cd "$REPO" && "$PY" -m scripts.e1_stage1.make_cycle_notebook --board "$BOARD" --shape "$s" --rep "$R" --session "$SESSION" --repo "$REPO" --evidence-dir "$EV") || halt "generate $s-r$R"
done
G=$((3 * (R - 1)))
run_pre() { # cycle gen route
  bash "$T/s2_cycle_pre.sh" "$EV" "$REPO" "$RUN" "$BOARD" "$1" "$2" "$3" > "$EV/control/s2pre_$1.txt" 2>&1
  rc=$?; grep -E "^reset gen|settled_after|^PARKED|binding_ok|STOP|exit=" "$EV/control/s2pre_$1.txt"
  [ "$rc" -eq 0 ] || halt "pre $1 rc=$rc"
  [ -f "$EV/control/stop" ] && halt "stop after pre $1"
}
run_leg() { # cycle leg route target
  echo "--- leg $2 ($(date -u +%FT%TZ))"
  bash "$T/s2_leg.sh" "$EV" "$REPO" "$1" "$2" "$3" "$4" > "$EV/control/s2leg_$2.txt" 2>&1
  rc=$?; grep -E "EXPERIMENT_ACCEPTED|PARKED_AT|contacts=|'outcome'|start_check|compliance:|STOP" "$EV/control/s2leg_$2.txt"
  [ "$rc" -eq 0 ] || halt "leg $2 rc=$rc"
  [ -f "$EV/control/stop" ] && halt "stop after leg $2"
}
A="S2-$BOARD-a-r$R"; B="S2-$BOARD-b-r$R"; C="S2-$BOARD-c-r$R"
echo "=== $A ($(date -u +%FT%TZ)) ==="; run_pre "$A" $((G + 1)) RAISE_TO_SIDE
run_leg "$A" "$A-setup" RAISE_TO_SIDE PRESENT
run_leg "$A" "$A-flight" LOWER_TO_REST REST
echo "=== $B ($(date -u +%FT%TZ)) ==="; run_pre "$B" $((G + 2)) PLACE_ROUTE
run_leg "$B" "$B-flight" PLACE_ROUTE REST
echo "=== $C ($(date -u +%FT%TZ)) ==="; run_pre "$C" $((G + 3)) PLACE_ROUTE
run_leg "$C" "$C-setup" PLACE_ROUTE REST
run_leg "$C" "$C-flight" LIFT_TO_PRESENT PRESENT
echo "=== set r$R complete ($(date -u +%FT%TZ)) ==="
