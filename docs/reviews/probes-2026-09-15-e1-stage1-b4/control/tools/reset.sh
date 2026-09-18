#!/usr/bin/env bash
# Scheduled reset via the container's sentinel file (the fake_reachy_server
# reset_watcher). Verified from the SERVER's own record, not the ack file:
# commands.jsonl gains a {"type":"reset"} entry and sim_step restarts.
set -u
GEN=$1; RUN=$2; P=/Users/terrancehamilton/reachy-1-2-sim-stage1/docs/reviews/probes-2026-09-15-e1-stage1-b4
[ -f $P/control/stop ] && { echo "STOP marker present; reset refused"; exit 9; }
cd /Users/terrancehamilton/reachy-1-2-sim-stage1
before=$(grep -c '"type":"reset"' $RUN/commands.jsonl)
last_step_before=$(tail -1 $RUN/states.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['sim_step'])")
docker compose exec -T reachy-sim sh -c "echo $GEN > /tmp/reachy_reset_request"
for _ in $(seq 1 30); do
  ack=$(docker compose exec -T reachy-sim sh -c 'cat /tmp/reachy_reset_ack 2>/dev/null' | tr -d '\r\n')
  [ "$ack" = "$GEN" ] && break; sleep 0.5
done
after=$(grep -c '"type":"reset"' $RUN/commands.jsonl)
sleep 1
step_now=$(tail -1 $RUN/states.jsonl | python3 -c "import json,sys; print(json.loads(sys.stdin.read())['sim_step'])")
echo "reset gen=$GEN ack=$ack resets_recorded ${before}->${after} sim_step ${last_step_before}->${step_now}"
[ "$ack" = "$GEN" ] && [ "$after" -eq $((before+1)) ] && [ "$step_now" -lt "$last_step_before" ] || { echo "STOP: reset $GEN not verified" | tee -a $P/control/stop; exit 9; }
