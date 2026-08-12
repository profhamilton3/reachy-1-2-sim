#!/usr/bin/env bash
# Launch the native macOS MuJoCo physics server for the demo scene.
#
# This runs on the HOST (Apple Silicon, Metal rendering) — not inside Docker.
# The Docker gRPC server bridges to it when REACHY_SIM_BACKEND=mujoco-remote
# (it connects to ws://host.docker.internal:8765).
#
# Usage:
#   ./scripts/run_physics_backend.sh                # demo scene, depth+seg off
#   ./scripts/run_physics_backend.sh --depth        # enable depth rendering
#
# Then, in another terminal, switch the container to the physics backend:
#   docker compose exec -e REACHY_SIM_BACKEND=mujoco-remote reachy-sim \
#     supervisorctl restart reachy-sdk-server
# and run the demo (marker faking auto-disables — real physics moves objects):
#   docker compose exec -e REACHY_SIM_BACKEND=mujoco-remote reachy-sim \
#     python3 /opt/scripts/demo_pick_place.py
# Watch the real grasp in the camera view: http://localhost:8080
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SCENE="${REACHY_SIM_SCENE:-$REPO/scenes/tabletop_demo.yaml}"

cd "$REPO/native_mujoco"
echo "Starting native MuJoCo server on ws://0.0.0.0:8765"
echo "  scene: $SCENE"
exec mjpython server.py --scene "$SCENE" --host 0.0.0.0 --port 8765 "$@"
