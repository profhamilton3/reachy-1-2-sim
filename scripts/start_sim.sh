#!/usr/bin/env bash
# Default startup for the full Reachy 1.2 sim WITH the MuJoCo physics backend.
#
# Starts the native macOS MuJoCo server (Metal, mjpython) on the host, waits for
# it to listen on :8765, then brings up the Docker container which — via
# REACHY_SIM_BACKEND=mujoco-remote in docker-compose.yml — bridges to it.
#
# Result: real physics arm dynamics + stereo camera render (port 8080), with the
# scene object poses mirrored into RViz (port 6080).  If the native server is not
# running, the container falls back to the kinematic fixture automatically.
#
# Usage:
#   ./scripts/start_sim.sh                       # demo scene
#   REACHY_SIM_SCENE=scenes/foo.yaml ./scripts/start_sim.sh
#
# Then run the pick-and-place demo:
#   docker compose exec reachy-sim python3 /opt/scripts/demo_pick_place.py
#
# Observe: RViz  → http://localhost:6080     camera → http://localhost:8080
#
# NOTE: contact-rich grasping in physics is best-effort — the gripper is tuned
# for small (~2-4 cm) objects and closes slowly over the bridge.  For a
# guaranteed visual pick-and-place, run the demo against the kinematic backend:
#   docker compose exec -e REACHY_SIM_BACKEND=kinematic reachy-sim \
#     python3 /opt/scripts/demo_pick_place.py
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${REACHY_SIM_SCENE:-$REPO/scenes/tabletop_demo.yaml}"
LOG="${REACHY_SIM_SERVER_LOG:-/tmp/reachy_mujoco_server.log}"

echo "▶ Stopping any existing native server on :8765 …"
lsof -tiTCP:8765 -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 || true
sleep 1

echo "▶ Starting native MuJoCo server (scene: $SCENE) …"
( cd "$REPO/native_mujoco" && nohup mjpython server.py \
    --scene "$SCENE" --host 0.0.0.0 --port 8765 --log-level INFO \
    >"$LOG" 2>&1 & )

echo "▶ Waiting for ws://0.0.0.0:8765 …"
for _ in $(seq 1 30); do
    if lsof -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then break; fi
    sleep 0.5
done
if ! lsof -iTCP:8765 -sTCP:LISTEN >/dev/null 2>&1; then
    echo "✖ Native server did not start — see $LOG" >&2
    exit 1
fi
echo "  server up (log: $LOG)"

echo "▶ Bringing up the Docker container (physics backend) …"
cd "$REPO"
REACHY_SIM_BACKEND=mujoco-remote docker compose up -d

echo
echo "✓ Sim running with the MuJoCo physics backend."
echo "    RViz   : http://localhost:6080"
echo "    Camera : http://localhost:8080"
echo "    Demo   : docker compose exec reachy-sim python3 /opt/scripts/demo_pick_place.py"
