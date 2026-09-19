#!/usr/bin/env bash
# Operator wrapper (Stage 2 B4 s1 session): C1 reset -> C2 settle -> C3 parked
# (stiff-zero gate) -> start the cycle notebook -> wait for binding marker.
# Calls only the packaged scripts in control/tools/ in the documented order;
# stops at the first failure and never removes control/stop.
# Usage: s2_cycle_pre.sh <EV> <REPO> <RUN> <cycle> <gen> <route-label>
set -u
EV=$1; REPO=$2; RUN=$3; C=$4; GEN=$5; ROUTE=$6
PY=${E1_PYTHON:?set E1_PYTHON}
[ -f "$EV/control/stop" ] && { echo "STOP marker present; refusing"; exit 9; }
echo "=== $C C1 reset gen $GEN ($(date -u +%FT%TZ)) ==="
bash "$EV/control/tools/reset.sh" "$REPO" "$EV" "$GEN" "$RUN" 2>&1 | tee "$EV/control/reset_$GEN.txt"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "reset exit=$rc"; exit "$rc"; }
[ -f "$EV/control/stop" ] && { echo "STOP after reset"; exit 9; }
echo "=== $C C2 settle ==="
"$PY" "$EV/control/tools/settle_wait.py" "$RUN" 15 180 | tee "$EV/control/settle_$C.json"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "STOP: settle exit=$rc" | tee -a "$EV/control/stop"; exit 9; }
echo "=== $C C3 parked ==="
bash "$EV/control/tools/parked.sh" "$REPO" "$EV" "$C-parked" B4 "$ROUTE" "$C" 2>&1 | tee "$EV/control/parkedsh_$C.txt"
rc=${PIPESTATUS[0]}; [ "$rc" -eq 0 ] || { echo "parked exit=$rc"; exit "$rc"; }
[ -f "$EV/control/stop" ] && { echo "STOP after parked"; exit 9; }
echo "=== $C notebook ==="
cd "$EV" && nohup "$HOME/e1venv/bin/jupyter" nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=e1venv \
  --output "e1_stage1_$C.executed.ipynb" "e1_stage1_$C.ipynb" > "$EV/control/notebook_exec_$C.log" 2>&1 &
for _ in $(seq 1 60); do
  [ -f "$EV/control/binding_ok_$C" ] || [ -f "$EV/control/binding_FAIL_$C" ] && break; sleep 1
done
if [ -f "$EV/control/binding_ok_$C" ]; then
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print('binding_ok', {k: d.get(k) for k in ['ok','provenance_ok','reasons','provenance_reasons']})" "$EV/control/binding_ok_$C"
  grep -E "start_variant gate:|pre-cycle compliance gate:" "$EV/e1_stage1_$C.executed.ipynb" 2>/dev/null | head -2
  exit 0
fi
echo "BINDING NOT OK for $C"; ls "$EV/control" | grep -E "binding_.*_$C"; tail -20 "$EV/control/notebook_exec_$C.log"; exit 9
