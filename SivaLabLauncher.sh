#!/usr/bin/env bash
# SivaLabLauncher.sh
# Starts the Reachy 1.2 simulator with the FWDCenterLabSivaPool scene
# and MuJoCo physics backend with lens distortion enabled.
#
# Usage:
#   ./SivaLabLauncher.sh          # Start the simulator
#   ./SivaLabLauncher.sh -KILL   # Shut down the simulator and MuJoCo server
#
# Access:
#   RViz   : http://localhost:6080
#   Camera : http://localhost:8080
#   Jupyter: http://localhost:8888

set -euo pipefail

# Check for -KILL flag
if [ "${1:-}" = "-KILL" ]; then
    echo "================================================"
    echo "  Shutting down Reachy 1.2 Siva Lab Simulator"
    echo "================================================"
    
    echo "▶ Bringing down Docker container …"
    docker compose down || true
    sleep 1
    
    echo "▶ Killing MuJoCo physics server on :8765 …"
    lsof -tiTCP:8765 -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 || true
    sleep 1
    
    echo "✓ Simulator and MuJoCo server shut down."
    exit 0
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
LOG="${REACHY_SIM_SERVER_LOG:-/tmp/reachy_mujoco_server.log}"

echo "================================================"
echo "  Reachy 1.2 Siva Lab Launcher"
echo "================================================"

# Kill any existing MuJoCo server on port 8765
echo "▶ Stopping any existing native server on :8765 …"
lsof -tiTCP:8765 -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 || true
sleep 1

# Start MuJoCo physics server with FWDCenterLabSivaPool scene and lens distortion
echo "▶ Starting native MuJoCo server …"
echo "    scene       : FWDCenterLabSivaPool"
echo "    calibration : calibration_measured_2026_08_27"
echo "    distortion  : ON (lens barrel-warping)"

( cd "$REPO/native_mujoco" && nohup mjpython server.py \
    --scene "$REPO/scenes/FWDCenterLabSivaPool.yaml" \
    --calibration "$REPO/scenes/calibration_measured_2026_08_27.yaml" \
    --distortion \
    --host 0.0.0.0 --port 8765 --log-level INFO \
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

# Bring up Docker container with mujoco-remote backend and executor enabled
echo "▶ Bringing up the Docker container …"
cd "$REPO"
REACHY_SIM_BACKEND=mujoco-remote \
REACHY_SIM_SCENE_FILE=/opt/scenes/FWDCenterLabSivaPool.yaml \
REACHY_PANEL_EXECUTOR=1 \
REACHY_PANEL_LANGUAGE=1 \
    docker compose up -d

# Restart container to drop stale joint goals
echo "▶ Restarting the container …"
docker compose restart reachy-1-2-sim
sleep 2

echo
echo "✓ Sim running with MuJoCo physics backend (FWDCenterLabSivaPool + distortion)."
echo "    RViz   : http://localhost:6080"
echo "    Camera : http://localhost:8080"
echo "    Jupyter: http://localhost:8888"
echo
