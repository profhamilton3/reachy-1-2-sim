#!/usr/bin/env bash
# Scheduled reset via the container's sentinel file (the fake_reachy_server
# reset_watcher). Verified from the SERVER's own record, not the ack file:
# commands.jsonl gains a {"type":"reset"} entry and sim_step restarts.
# Sentinel-reset semantics unchanged from PR #117's reset.sh
# (control/tools/reset.sh); repo and run dir are now arguments, not
# hardcoded to the evidence tree.
# Usage: reset.sh <repo> <evidence-dir> <gen> <run-dir>
set -u
REPO=$1; P=$2; GEN=$3; RUN=$4
[ -f "$P/control/stop" ] && { echo "STOP marker present; reset refused"; exit 9; }
cd "$REPO"
before=$(grep -c '"type":"reset"' "$RUN/commands.jsonl")
last_step_before=$(tail -1 "$RUN/states.jsonl" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['sim_step'])")
docker compose exec -T reachy-sim sh -c "echo $GEN > /tmp/reachy_reset_request"
for _ in $(seq 1 30); do
  ack=$(docker compose exec -T reachy-sim sh -c 'cat /tmp/reachy_reset_ack 2>/dev/null' | tr -d '\r\n')
  [ "$ack" = "$GEN" ] && break; sleep 0.5
done
after=$(grep -c '"type":"reset"' "$RUN/commands.jsonl")
sleep 1
step_now=$(tail -1 "$RUN/states.jsonl" | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['sim_step'])")
echo "reset gen=$GEN ack=$ack resets_recorded ${before}->${after} sim_step ${last_step_before}->${step_now}"
[ "$ack" = "$GEN" ] && [ "$after" -eq $((before+1)) ] && [ "$step_now" -lt "$last_step_before" ] || { echo "STOP: reset $GEN not verified" | tee -a "$P/control/stop"; exit 9; }
