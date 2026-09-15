#!/usr/bin/env bash
# A1 (2026-09-15 matrix readiness) regression: scripts/lib/scene_path.sh's
# container-side SCENE_FILE must keep the source subdirectory (e.g.
# e1_boards/), not just the basename -- see that file's comment for the
# concrete failure this closes (RViz shows no board on any matrix run).
#
# Offline, no docker, no server: exercises resolve_scene_path() directly
# against a synthetic scenes/ tree under a temp dir standing in for $REPO.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
source "$REPO_ROOT/scripts/lib/scene_path.sh"

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/scenes/e1_boards"
touch "$TMP/scenes/tabletop_demo.yaml"
touch "$TMP/scenes/other.yaml"
touch "$TMP/scenes/e1_boards/B4.yaml"

fail=0

check() {
    local label="$1" scene_in="$2" expected="$3"
    SCENE="" SCENE_FILE=""
    if ! resolve_scene_path "$TMP" "$scene_in"; then
        echo "FAIL ($label): resolve_scene_path errored"
        fail=1
        return
    fi
    if [ "$SCENE_FILE" != "$expected" ]; then
        echo "FAIL ($label): SCENE_FILE=$SCENE_FILE, expected $expected"
        fail=1
    else
        echo "ok ($label): SCENE_FILE=$SCENE_FILE"
    fi
}

# Short name (no subdirectory).
check "short name" "tabletop_demo" "/opt/scenes/tabletop_demo.yaml"

# Bare .yaml filename (no subdirectory).
check "bare file" "other.yaml" "/opt/scenes/other.yaml"

# Absolute path INTO a subdirectory -- the A1 regression case: the old
# basename-only derivation collapsed this to /opt/scenes/B4.yaml, dropping
# e1_boards/.
check "absolute path with subdirectory" "$TMP/scenes/e1_boards/B4.yaml" \
    "/opt/scenes/e1_boards/B4.yaml"

exit "$fail"
