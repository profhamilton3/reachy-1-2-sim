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
# Camera/render switches (all optional, all default off unless noted):
#   REACHY_SIM_CALIBRATION=<name|path>  camera profile
#                                       (default: calibration_measured_2026_08_27)
#   REACHY_SIM_DISTORTION=1             barrel-warp frames to match the real lens
#   REACHY_SIM_DEPTH=1                  add a float16 depth map to every frame
#   REACHY_SIM_SEGMENTATION=1           add a uint16 body-ID map to every frame
#   REACHY_SIM_EFFECTS=<path.yaml>      sensor noise/blur/dropout profile
#   REACHY_SIM_RECORD=<dir>             record states+commands under <dir>
#
# The measured lab scene with the full camera model:
#   REACHY_SIM_SCENE=FWDCenterLabMCC REACHY_SIM_DISTORTION=1 ./scripts/start_sim.sh
#
# DEPTH and SEGMENTATION each add ~820 kB to every camera_frame.  Setting BOTH
# pushes the frame past the 1 MiB limit the container's websocket client uses,
# which drops the bridge into a silent reconnect loop — read the note at their
# definition below before switching them on.
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

# Lens distortion.  Off by default: renders are ground truth for the collision
# and evaluation paths.  Set REACHY_SIM_DISTORTION=1 to warp camera frames (and
# depth/segmentation with them) to match the real lens's barrel — for eyeballing
# sim against real frames, and for generating training data that matches the raw
# camera.  See native_mujoco/distortion.py.
DISTORT_ARGS=""
DISTORT_NOTE="off"
case "${REACHY_SIM_DISTORTION:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        DISTORT_ARGS="--distortion"
        DISTORT_NOTE="ON — frames are barrel-warped; corners will be black"
        ;;
esac

# Depth and segmentation channels (R12-601).  Both OFF by default, and that
# default is load-bearing: TURNING BOTH ON AT ONCE BREAKS THE DOCKER BRIDGE.
#
# Measured on the wire, one 640x480 left_camera frame:
#     jpeg_b64  54,780 B   depth_b64 819,200 B   seg_b64 819,200 B   (+246 JSON)
#     neither      55 kB   depth only  874 kB    seg only  874 kB    both 1.61 MiB
# The container's client (mujoco_remote_backend.py) calls websockets.connect()
# without max_size, so it takes the library default of 1 MiB.  With both flags
# on, every camera_frame exceeds it and the client kills the connection with
# close code 1009 "message too big", reconnects, and dies again ~6 s later —
# a silent reconnect loop in which RViz object poses and the :8080 preview
# simply stop updating.  Either flag ALONE fits, but only just: 874 kB leaves
# ~170 kB of headroom that a more detailed frame's larger JPEG eats into.
#
# Nothing in the container reads either channel — the bridge takes only
# `jpeg_b64` off a camera_frame — so over Docker they are cost with no payoff.
# Turn them on when a client talks to ws://…:8765 DIRECTLY (that client can set
# max_size itself); leave them off for RViz/preview/notebook-over-gRPC work.
# For bulk labelled data prefer native_mujoco/cli/generate_dataset.py, which
# renders segmentation in-process with no socket in the path.
#
# Raising the container client's max_size would lift the 1 MiB ceiling, but the
# extra render pass per camera per frame remains — so that is a change to make
# when something in the container actually consumes depth or segmentation.
EXTRA_ARGS=""
DEPTH_NOTE="off"
case "${REACHY_SIM_DEPTH:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        EXTRA_ARGS="$EXTRA_ARGS --depth"
        DEPTH_NOTE="ON — float16 depth map per frame"
        ;;
esac
SEG_NOTE="off"
case "${REACHY_SIM_SEGMENTATION:-0}" in
    1|true|TRUE|yes|YES|on|ON)
        EXTRA_ARGS="$EXTRA_ARGS --segmentation"
        SEG_NOTE="ON — uint16 body-ID map per frame"
        ;;
esac

# Sensor effects (R12-602): noise, blur, dropped frames, latency.  A path to a
# YAML config; there is no default profile, so this is off unless you point it
# at one.  Deliberately NOT enabled alongside the calibration work — effects
# degrade frames on purpose, which is the opposite of what you want when
# comparing sim frames against real ones.
EFFECTS_ARGS=""
EFFECTS_NOTE="off"
if [ -n "${REACHY_SIM_EFFECTS:-}" ]; then
    if [ ! -f "$REACHY_SIM_EFFECTS" ]; then
        echo "✖ Sensor effects config not found: $REACHY_SIM_EFFECTS" >&2
        exit 1
    fi
    EFFECTS_ARGS="--effects $REACHY_SIM_EFFECTS"
    EFFECTS_NOTE="$(basename "$REACHY_SIM_EFFECTS")"
fi

# Recording (R12-603): states + commands to a timestamped run dir.  Off unless
# REACHY_SIM_RECORD names a directory.
RECORD_ARGS=""
RECORD_NOTE="off"
if [ -n "${REACHY_SIM_RECORD:-}" ]; then
    mkdir -p "$REACHY_SIM_RECORD"
    RECORD_ARGS="--record $REACHY_SIM_RECORD"
    RECORD_NOTE="$REACHY_SIM_RECORD"
fi

echo "▶ Starting native MuJoCo server …"
echo "    scene        : $(basename "$SCENE")"
echo "    calibration  : $(basename "$CALIB")"
echo "    distortion   : $DISTORT_NOTE"
echo "    depth        : $DEPTH_NOTE"
echo "    segmentation : $SEG_NOTE"
echo "    effects      : $EFFECTS_NOTE"
echo "    recording    : $RECORD_NOTE"
( cd "$REPO/native_mujoco" && nohup mjpython server.py \
    --scene "$SCENE" --calibration "$CALIB" \
    $DISTORT_ARGS $EXTRA_ARGS $EFFECTS_ARGS $RECORD_ARGS \
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
