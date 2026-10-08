#!/usr/bin/env bash
# Operator wrapper (Stage 2 B4 s1 session): write go_<leg> only when the
# binding marker exists and no stop marker is present, then run the packaged
# leg.sh in `end` mode and print every gate's verdict. Serialized: never run
# while another leg.sh is running (plan §5 marker discipline).
# Usage: s2_leg.sh <EV> <REPO> <cycle> <leg> <ROUTE> <TAIL_TARGET>
set -u
EV=$1; REPO=$2; C=$3; L=$4; ROUTE=$5; TGT=$6
[ -f "$EV/control/stop" ] && { echo "STOP marker present; refusing go_$L"; exit 9; }
[ -f "$EV/control/binding_ok_$C" ] || { echo "no binding_ok_$C; refusing go_$L"; exit 9; }
pgrep -f "measure_route_clearance.py" >/dev/null && { echo "a recorder is still running; refusing go_$L"; exit 9; }
date -u +%FT%TZ > "$EV/control/go_$L"
bash "$EV/control/tools/leg.sh" "$REPO" "$EV" "$C" "$L" "$ROUTE" "$TGT" 2>&1 | tee "$EV/control/legsh_$L.txt"
rc=${PIPESTATUS[0]}; echo "leg.sh exit=$rc"
tail -2 "$EV/control/gate_$L.txt" 2>/dev/null
grep -E "PARKED_AT|max_gross_err|tailcheck exit" "$EV/control/tailcheck_${L}_$TGT.txt" 2>/dev/null
grep -oE "contacts=[0-9]+ \(evidence on [0-9/]+ states, reset_in_window=[A-Za-z]+\)" "$EV/control/linker_$L.txt" 2>/dev/null
python3 - "$EV/control/${L}_done" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1]))
print({k: d.get(k) for k in ['outcome', 'elapsed_s', 'returned']})
sc = d.get('start_check') or {}
print('start_check:', sc.get('ok'), sc.get('posture_of'))
print('compliance:', (d.get('compliance_check') or {}).get('ok'))
print('phases:', [p[1] for p in d.get('phases', [])])
print('end_pose:', d.get('end_pose'))
PY
echo "recorder_logs: $(ls "$EV/recorder_logs" | wc -l | tr -d ' ') files"
[ -f "$EV/control/stop" ] && { echo "STOP:"; cat "$EV/control/stop"; exit 9; } || echo "no stop marker"
exit "$rc"
