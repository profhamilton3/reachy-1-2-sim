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
# Usage — pick the scene by short name (default: tabletop_demo); the SAME scene
# drives both the physics (camera :8080) and the noVNC/RViz view (:6080):
#   ./scripts/start_sim.sh                              # tabletop pick-and-place
#   REACHY_SIM_SCENE=control_panel ./scripts/start_sim.sh   # control console
#   REACHY_SIM_SCENE=/abs/path/to/scene.yaml ./scripts/start_sim.sh
#
# Then run the matching demo, e.g.:
#   docker compose exec reachy-sim python3 /opt/scripts/demo_pick_place.py
#   docker compose exec reachy-sim python3 /opt/scripts/demo_control_panel.py
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
LOG="${REACHY_SIM_SERVER_LOG:-/tmp/reachy_mujoco_server.log}"

# Scene selector: a short name (control_panel), a bare file (control_panel.yaml),
# or a full path all work.  The SAME scene drives the physics (native server,
# host path) AND the noVNC/RViz view (container path /opt/scenes/<file>), so the
# two views always match.
SCENE_IN="${REACHY_SIM_SCENE:-tabletop_demo}"
case "$SCENE_IN" in
    /*)         SCENE="$SCENE_IN" ;;                         # absolute path
    *.yaml)     SCENE="$REPO/scenes/$(basename "$SCENE_IN")" ;;
    *)          SCENE="$REPO/scenes/${SCENE_IN}.yaml" ;;     # short name
esac
if [ ! -f "$SCENE" ]; then
    echo "✖ Scene not found: $SCENE" >&2
    echo "  Available: $(ls "$REPO/scenes"/*.yaml | xargs -n1 basename | sed 's/\.yaml//' | tr '\n' ' ')" >&2
    exit 1
fi
SCENE_FILE="/opt/scenes/$(basename "$SCENE")"   # path inside the container

echo "▶ Stopping any existing native server on :8765 …"
lsof -tiTCP:8765 -sTCP:LISTEN 2>/dev/null | xargs -r kill -9 || true
sleep 1

# Camera calibration profile.  Defaults to the measured 2026-08-27 lab
# calibration; set REACHY_SIM_CALIBRATION to a short name or a path to override
# (e.g. REACHY_SIM_CALIBRATION=calibration_defaults for the synthetic pinhole).
CALIB_IN="${REACHY_SIM_CALIBRATION:-calibration_measured_2026_08_27}"
case "$CALIB_IN" in
    /*)      CALIB="$CALIB_IN" ;;
    *.yaml)  CALIB="$REPO/scenes/$(basename "$CALIB_IN")" ;;
    *)       CALIB="$REPO/scenes/${CALIB_IN}.yaml" ;;
esac
if [ ! -f "$CALIB" ]; then
    echo "✖ Calibration profile not found: $CALIB" >&2
    exit 1
fi

echo "▶ Starting native MuJoCo server (scene: $SCENE, calibration: $(basename "$CALIB")) …"
( cd "$REPO/native_mujoco" && nohup mjpython server.py \
    --scene "$SCENE" --calibration "$CALIB" \
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

echo "▶ Bringing up the Docker container (physics backend, noVNC scene: $SCENE_FILE) …"
cd "$REPO"
REACHY_SIM_BACKEND=mujoco-remote REACHY_SIM_SCENE_FILE="$SCENE_FILE" \
    docker compose up -d

# `up -d` leaves an already-running, unchanged container alone — so the bridge
# keeps the joint goals it held before the native server was restarted, and
# re-asserts them the moment it reconnects.  The arm then does NOT return to its
# home pose, which makes this script useless as the "reset the sim" escape hatch
# it is documented to be.  Restart the container so the bridge starts clean.
echo "▶ Restarting the container so the bridge drops its stale joint goals …"
docker compose restart reachy-sim

echo
echo "✓ Sim running with the MuJoCo physics backend."
echo "    RViz   : http://localhost:6080"
echo "    Camera : http://localhost:8080"
echo "    Demo   : docker compose exec reachy-sim python3 /opt/scripts/demo_pick_place.py"
