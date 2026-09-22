#!/usr/bin/env bash
# Scheduled reset via the container's sentinel file (the fake_reachy_server
# reset_watcher). Verified from the SERVER's own record, not the ack file:
# commands.jsonl gains a {"type":"reset"} entry and sim_step restarts.
# Sentinel-reset semantics unchanged from PR #117's reset.sh
# (control/tools/reset.sh); repo and run dir are now arguments, not
# hardcoded to the evidence tree.
#
# The before/after reads go through reset_verify.py (E1 Stage 2 B1
# reset_18, docs/reviews/probes-2026-09-19-e1-stage2-b1/control/
# reset_18.txt, stop_analysis.txt): a bare `tail -1 states.jsonl` against
# a file the native server is still appending to at ~20 Hz can return two
# records in one read or a torn last line, and this script used to treat
# either as an unverified reset. reset_verify.py reads from the tail via
# the repo's existing robust reader and STOPs, with a specific reason, on
# anything short of fresh, live, unambiguous evidence of THIS reset -- see
# scripts/e1_stage1/README.md's reset step and reset_verify.py's own
# docstring for the full contract. RESET_VERIFY_TIMEOUT_S/RESET_VERIFY_POLL_S
# override reset_verify.py verify's --timeout-s/--poll-s (default 10s/0.25s)
# the same way E1_PYTHON overrides the interpreter -- an env knob, not a new
# positional argument.
#
# Request publish is atomic (E1 Stage 2 B2 s1 reset 1, 2026-09-22:
# docs/reviews/probes-2026-09-22-e1-stage2-b2/control/stop_analysis.txt).
# The old `echo $GEN > /tmp/reachy_reset_request` truncates the target
# before the shell's `echo` writes it, so a watcher polling at the same
# instant can read a file that exists but is still empty -- exactly what
# happened: fake_reachy_server.reset_watcher read gen="", triggered an
# unrequested reset, and acked "" -- ack mismatch, STOP. The fix writes to
# a same-directory temp file, then `mv`s it into place: `mv` within one
# directory on one filesystem is atomic, so a reader sees either the old
# state (no file) or the complete new content, never a partial write.
# $GEN is passed as a positional argument to the container's `sh`, not
# interpolated into the script text, so a multi-digit generation can't be
# split by a read landing mid-write either. fake_reachy_server.reset_watcher
# additionally refuses to act on anything that isn't a plain non-negative
# integer, so an empty or malformed read -- from this or any other
# publisher -- can never trigger a reset or a matching ack (see its
# docstring and tests/unit/test_reset_sentinel_protocol.py).
#
# Usage: reset.sh <repo> <evidence-dir> <gen> <run-dir>
set -u
REPO=$1; P=$2; GEN=$3; RUN=$4
PY=${E1_PYTHON:-python3}
[ -f "$P/control/stop" ] && { echo "STOP marker present; reset refused"; exit 9; }
cd "$REPO"
SNAP=$("$PY" scripts/e1_stage1/reset_verify.py snapshot "$RUN")
snap_rc=$?
if [ $snap_rc -ne 0 ]; then
  echo "STOP: reset $GEN not verified: $SNAP" | tee -a "$P/control/stop"
  exit 9
fi
docker compose exec -T reachy-sim sh -c \
  'f="/tmp/reachy_reset_request"; tmp="$f.tmp.$$"; printf "%s" "$1" > "$tmp" && mv "$tmp" "$f"' \
  sh "$GEN"
pub_rc=$?
if [ $pub_rc -ne 0 ]; then
  echo "STOP: reset $GEN not verified: request publish failed (rc=$pub_rc)" | tee -a "$P/control/stop"
  exit 9
fi
for _ in $(seq 1 30); do
  ack=$(docker compose exec -T reachy-sim sh -c 'cat /tmp/reachy_reset_ack 2>/dev/null' | tr -d '\r\n')
  [ "$ack" = "$GEN" ] && break; sleep 0.5
done
OUT=$("$PY" scripts/e1_stage1/reset_verify.py verify "$RUN" "$GEN" "$ack" "$SNAP" \
  --timeout-s "${RESET_VERIFY_TIMEOUT_S:-10}" --poll-s "${RESET_VERIFY_POLL_S:-0.25}")
rc=$?
if [ $rc -eq 0 ]; then
  echo "$OUT"
else
  echo "STOP: reset $GEN not verified: $OUT" | tee -a "$P/control/stop"
  exit 9
fi
